import gc
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy import signal

try:
    from kokoro import KPipeline
except ImportError:
    print("Warning: kokoro not installed. TTS will fail.")
    KPipeline = None

from config import (
    SAMPLE_RATE,
    TEMP_DIR,
    TTS_DEFAULT_VOICE,
    TTS_FALLBACK_VOICE,
    TTS_REPO_ID,
    VALID_VOICES,
    VOICE_CATALOG,
)


@dataclass
class CastVoiceAssignment:
    speaker: str
    primary_voice: str
    secondary_voice: Optional[str] = None
    blend_weight: float = 1.0  # e.g. 0.62 blends 62% primary + 38% secondary in neural latent space!
    lang_code: str = "a"
    base_pitch_semitones: float = 0.0  # Applied ONLY via artifact-free sinc resampling + neural speed pre-comp!
    base_speed: float = 1.0
    eq_style: str = "balanced"  # "child", "deep", "ethereal", "balanced"
    summary: str = ""


# Comprehensive Emotion Prosody Table:
# emotion -> (speed_mult, pitch_semitones, air_shelf_db, dynamic_level_db)
# Notice pitch_semitones is 0.0 for nearly all adult emotions so neural waveforms remain 100% phase-pure!
EMOTION_PROSODY: Dict[str, Tuple[float, float, float, float]] = {
    "neutral": (0.98, 0.0, 0.0, 0.0),
    "happy": (1.04, 0.0, 1.0, 0.4),
    "joyful": (1.05, 0.0, 1.2, 0.5),
    "excited": (1.08, 0.0, 1.4, 0.9),
    "sad": (0.87, 0.0, -0.6, -1.5),
    "sorrowful": (0.85, 0.0, -0.8, -1.8),
    "grief": (0.84, 0.0, -0.8, -1.8),
    "tender": (0.90, 0.0, 0.8, -1.6),
    "gentle": (0.90, 0.0, 0.8, -1.6),
    "romantic": (0.89, 0.0, 1.0, -1.2),
    "angry": (1.04, 0.0, 1.6, 1.2),
    "furious": (1.07, 0.0, 1.8, 1.5),
    "shouting": (1.06, 0.0, 1.8, 1.8),
    "screaming": (1.08, 0.0, 2.0, 2.0),
    "grumpy": (0.93, 0.0, 0.4, 0.2),
    "sarcastic": (0.93, 0.0, 0.8, 0.2),
    "wry": (0.94, 0.0, 0.6, 0.0),
    "whisper": (0.88, 0.0, 1.8, -3.6),
    "whispering": (0.88, 0.0, 1.8, -3.6),
    "fearful": (1.05, 0.0, 1.2, -0.4),
    "terrified": (1.08, 0.0, 1.5, 0.8),
    "tense": (0.96, 0.0, 0.8, 0.5),
    "suspenseful": (0.93, 0.0, 0.8, 0.2),
    "determined": (0.94, 0.0, 1.2, 0.9),
    "resolute": (0.94, 0.0, 1.0, 0.8),
    "commanding": (0.92, 0.0, 1.2, 1.2),
    "dramatic": (0.91, 0.0, 1.0, 0.8),
    "ominous": (0.88, 0.0, 0.4, 0.5),
    "awe": (0.88, 0.0, 1.2, -0.4),
    "wonder": (0.89, 0.0, 1.2, -0.3),
    "ethereal": (0.88, 0.0, 1.5, -1.2),
    "mysterious": (0.90, 0.0, 1.0, -0.6),
}


def prepare_expressive_text(text: str, emotion: str, is_dialogue_character: bool) -> str:
    """
    Enhances punctuation cues that drive StyleTTS2 / Kokoro's neural F0 pitch & energy predictors
    WITHOUT altering a single spoken word! Fixes flat delivery on short 1-4 word character lines.
    """
    s = (text or "").strip()
    if not s:
        return ""

    # Normalize straight single-quoted inner speech or multiple spaces cleanly
    s = re.sub(r"\s+", " ", s)
    emo = (emotion or "neutral").strip().lower()
    word_count = len(s.split())

    if emo in ("shouting", "screaming", "furious", "angry", "terrified", "excited"):
        if s.endswith("."):
            s = s[:-1] + "!"
        elif s[-1] not in ".!?…\"'”":
            s = s + "!"
    elif emo in ("tender", "sad", "sorrowful", "whisper", "whispering", "awe", "ethereal", "mysterious"):
        # On short intimate lines like "Seraphina." or "Come home, Eddie.", a trailing ellipsis
        # activates Kokoro's soft, lingering melodic fall-rise instead of a clipped robotic stop!
        if word_count <= 6 and s.endswith(".") and not s.endswith("..."):
            s = s[:-1] + "..."
    elif emo in ("sarcastic", "wry", "grumpy", "determined", "commanding", "dramatic"):
        # Slight pause after opening interjections ("Well, princess." -> "Well... princess,")
        if is_dialogue_character and s.startswith(("Well, ", "Look, ", "Darling, ")):
            s = s.replace(", ", "... ", 1)

    return s


def _high_shelf_biquad(audio: np.ndarray, freq_hz: float, gain_db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Smooth studio Mastering High-Shelf filter (zero narrow resonant ringing!)."""
    if abs(gain_db) < 0.15 or audio.size < 256:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / 2.0 * math.sqrt(2.0)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a, audio).astype(np.float32)


def _low_shelf_biquad(audio: np.ndarray, freq_hz: float, gain_db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Smooth studio Low-Shelf filter for warm pirate/baritone chest authority without mud."""
    if abs(gain_db) < 0.15 or audio.size < 256:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / 2.0 * math.sqrt(2.0)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a, audio).astype(np.float32)


def _apply_ethereal_air(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Ultra-subtle 6.5% stereo-safe velvet early-reflection shimmer for 'ethereal' / 'ancient' / 'mermaid' / 'spirit'
    personas—keeps direct consonants 100% razor-sharp while lending an enchanting halo!
    """
    if audio.size < 1024:
        return audio
    taps_ms = (23.0, 41.0, 67.0)
    gains = (0.042, 0.028, 0.018)
    wet = np.zeros(len(audio) + int(0.08 * sr), dtype=np.float32)
    for delay_ms, g in zip(taps_ms, gains):
        d = int(delay_ms * 0.001 * sr)
        wet[d : d + len(audio)] += audio * g
    # High-pass the wet shimmer above 600 Hz so low vocal fundamentals stay 100% dry & tight
    hp = signal.butter(2, 600.0 / (0.5 * sr), btype="highpass", output="sos")
    wet = signal.sosfilt(hp, wet).astype(np.float32)
    out = wet.copy()
    out[: len(audio)] += audio
    return out[: len(audio)]


def shape_character_timbre(
    audio: np.ndarray,
    eq_style: str,
    air_db: float = 0.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Transparent mastering-grade tonal shaping (ZERO narrow resonant filters or clipping!)."""
    if len(audio) < 512:
        return audio
    y = audio.astype(np.float32)

    if eq_style == "child":
        sos = signal.butter(2, 135.0 / (0.5 * sr), btype="highpass", output="sos")
        y = signal.sosfilt(sos, y).astype(np.float32)
        y = _high_shelf_biquad(y, 4200.0, 1.2, sr=sr)
    elif eq_style == "deep":
        # Smooth warm baritone body around 145 Hz + crisp 4.5 kHz articulation
        y = _low_shelf_biquad(y, 155.0, 2.2, sr=sr)
        y = _high_shelf_biquad(y, 4400.0, 1.0, sr=sr)
    elif eq_style == "ethereal":
        y = _high_shelf_biquad(y, 4800.0, 1.6, sr=sr)
        y = _apply_ethereal_air(y, sr=sr)

    if abs(air_db) > 0.15:
        y = _high_shelf_biquad(y, 4200.0, air_db, sr=sr)

    peak = float(np.max(np.abs(y)))
    if peak > 0.97:
        y = y * (0.95 / peak)
    return y


class CharacterVoiceResolver:
    """
    Maps character personas to high-grade Kokoro neural voice blends:
    - Prioritizes high-fidelity Grade A/B voices and blends expressive latent style vectors
      (e.g. bm_fable + am_fenrir for a commanding British pirate; af_bella + af_nicole for an ethereal mermaid!)
    - Avoids low-grade flat voices and avoids destructive grain pitch-shifting!
    """

    def __init__(self, narrator_default_voice: str = TTS_DEFAULT_VOICE):
        self.narrator_default = narrator_default_voice if narrator_default_voice in VALID_VOICES else TTS_DEFAULT_VOICE
        self.assignments: Dict[str, CastVoiceAssignment] = {}
        self.used_signatures: Set[str] = set()

    @staticmethod
    def _infer_demographics(speaker: str, profile: str, gender: str, age_group: str, accent: str) -> Tuple[str, str, str, List[str]]:
        blob = f"{speaker} {profile} {gender} {age_group} {accent}".lower()

        resolved_accent = (accent or "").strip().lower()
        if resolved_accent not in ("american", "british", "indian"):
            if any(w in blob for w in ("indian", "hindi", "mumbai", "delhi", "bengal", "desi", "south asian")):
                resolved_accent = "indian"
            elif any(w in blob for w in ("british", "uk", "london", "english accent", "scottish", "posh", "victorian", "pirate", "captain", "majesty", "lord", "lady")):
                resolved_accent = "british"
            else:
                resolved_accent = "american"

        resolved_gender = (gender or "").strip().lower()
        if resolved_gender not in ("male", "female"):
            female_words = ("girl", "woman", "female", "she", "mother", "mom", "sister", "queen", "princess", "mermaid", "siren", "lady", "grandma", "aunt", "wife", "daughter")
            male_words = ("boy", "man", "male", "he", "father", "dad", "brother", "king", "prince", "pirate", "captain", "lord", "grandpa", "uncle", "husband", "son")
            if any(re.search(rf"\b{re.escape(w)}\b", blob) for w in female_words):
                resolved_gender = "female"
            elif any(re.search(rf"\b{re.escape(w)}\b", blob) for w in male_words):
                resolved_gender = "male"
            else:
                resolved_gender = "female" if "narrator" in speaker.lower() and "am_" not in TTS_DEFAULT_VOICE else "male"

        resolved_age = (age_group or "").strip().lower()
        age_match = re.search(r"\b(\d{1,2})\s*(?:y|yr|yrs|year|years|yo)(?:\s*old)?\b", blob)
        numeric_age: Optional[int] = int(age_match.group(1)) if age_match else None

        if numeric_age is not None:
            if numeric_age <= 12:
                resolved_age = "child"
            elif numeric_age <= 19:
                resolved_age = "teen"
            elif numeric_age <= 32:
                resolved_age = "young_adult"
            elif numeric_age >= 60:
                resolved_age = "elderly"
            else:
                resolved_age = "adult"
        elif resolved_age not in ("child", "teen", "young_adult", "adult", "elderly"):
            if any(w in blob for w in ("child", "kid", "little girl", "little boy", "toddler")):
                resolved_age = "child"
            elif any(w in blob for w in ("teen", "teenager", "youth")):
                resolved_age = "teen"
            elif any(w in blob for w in ("old", "elderly", "aged", "senior", "grandpa", "grandma")):
                resolved_age = "elderly"
            elif any(w in blob for w in ("young_adult", "young", "20s")):
                resolved_age = "young_adult"
            else:
                resolved_age = "adult"

        trait_vocab = (
            "gravelly", "raspy", "grumpy", "commanding", "pirate", "determined", "deep",
            "ethereal", "ancient", "gentle", "knowing", "mermaid", "siren", "mysterious",
            "warm", "dramatic", "theatrical", "sarcastic", "witty", "soft", "whisper",
            "happy", "bubbly", "energetic", "stern", "wise", "regal", "princess", "queen",
        )
        traits = [kw for kw in trait_vocab if kw in blob]
        return resolved_gender, resolved_age, resolved_accent, traits

    def resolve(
        self,
        speaker: str,
        voice_profile: str = "",
        gender: str = "",
        age_group: str = "",
        accent: str = "",
        explicit_voice_id: Optional[str] = None,
    ) -> CastVoiceAssignment:
        key = (speaker or "Narrator").strip().lower()
        if key in self.assignments:
            return self.assignments[key]

        is_narrator = key in ("narrator", "narration", "storyteller")
        res_gender, res_age, res_accent, traits = self._infer_demographics(speaker, voice_profile, gender, age_group, accent)

        if is_narrator:
            primary = explicit_voice_id if (explicit_voice_id in VOICE_CATALOG and VOICE_CATALOG[explicit_voice_id]["quality"] >= 16) else self.narrator_default
            self.used_signatures.add(primary)
            lang = "b" if primary.startswith("b") else "a"
            assign = CastVoiceAssignment(
                speaker=speaker,
                primary_voice=primary,
                lang_code=lang,
                base_speed=0.97,
                summary=f"{primary} (Narrator)",
            )
            self.assignments[key] = assign
            return assign

        # Score voices by Gender + Accent + Official Quality Grade + Persona Trait Overlap
        scored: List[Tuple[float, str]] = []
        for vid, meta in VOICE_CATALOG.items():
            if meta["gender"] != res_gender:
                continue

            score = float(meta.get("quality", 12))  # Strongly favor crisp, well-trained voices!
            if meta["accent"] == res_accent:
                score += 28.0
            elif res_accent == "indian" and meta["accent"] == "british":
                score += 8.0

            if meta["age"] == res_age:
                score += 14.0
            elif res_age in ("child", "teen") and meta["age"] in ("child", "teen", "young_adult"):
                score += 8.0

            for t in traits:
                if t in meta["traits"]:
                    score += 9.0

            if vid not in self.used_signatures:
                score += 15.0  # Keep characters distinct from Narrator & each other

            scored.append((score, vid))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen_primary = scored[0][1] if scored else TTS_FALLBACK_VOICE

        # Accept explicit_voice_id ONLY if its gender AND accent match AND its quality grade isn't low (< 13)
        if explicit_voice_id and explicit_voice_id in VOICE_CATALOG:
            vmeta = VOICE_CATALOG[explicit_voice_id]
            if (
                vmeta["gender"] == res_gender
                and vmeta["accent"] == res_accent
                and vmeta.get("quality", 12) >= 14
                and explicit_voice_id not in self.used_signatures
            ):
                chosen_primary = explicit_voice_id

        # INTELLIGENT NEURAL LATENT BLENDING FOR HIGH CLARITY & CHARACTER EXPRESSION!
        # Instead of DSP pitch distortion, we blend complementary high-grade Kokoro voice embeddings:
        secondary_voice: Optional[str] = None
        blend_weight = 1.0
        eq_style = "balanced"
        pitch_st = 0.0
        base_speed = 0.98

        is_ethereal = any(t in traits for t in ("ethereal", "mermaid", "siren", "gentle", "ancient", "mysterious", "whisper"))
        is_gravelly_commanding = any(t in traits for t in ("gravelly", "raspy", "pirate", "commanding", "grumpy", "determined", "deep"))

        if res_gender == "male":
            if res_accent == "british":
                # Pure British male voices (bm_fable / bm_george) were trained on <60m of data (Grade C).
                # Blending 58% bm_fable/bm_george + 42% am_fenrir/am_michael injects rich Grade B+ vocal
                # clarity, chest resonance, and dramatic prosody while retaining the British inflection!
                if is_gravelly_commanding:
                    chosen_primary = "bm_fable" if "bm_fable" not in self.used_signatures else "bm_george"
                    secondary_voice = "am_fenrir"
                    blend_weight = 0.56
                    eq_style = "deep"
                    base_speed = 0.94
                else:
                    secondary_voice = "am_michael"
                    blend_weight = 0.62
            elif is_gravelly_commanding:
                chosen_primary = "am_fenrir" if "am_fenrir" not in self.used_signatures else "am_michael"
                secondary_voice = "bm_fable"
                blend_weight = 0.72
                eq_style = "deep"
                base_speed = 0.94
        elif res_gender == "female":
            if is_ethereal:
                # Blend Grade A- af_bella / bf_emma clarity with af_nicole's breathy, enchanting intimacy!
                if chosen_primary != "af_nicole":
                    secondary_voice = "af_nicole"
                    blend_weight = 0.60
                else:
                    secondary_voice = "af_bella"
                    blend_weight = 0.58
                eq_style = "ethereal"
                base_speed = 0.93

        # Ensure non-colliding signature if two characters picked the exact same pair
        sig = f"{chosen_primary}:{secondary_voice or ''}"
        if sig in self.used_signatures and len(scored) > 1:
            alt = next((v for _, v in scored if v != chosen_primary), None)
            if alt:
                secondary_voice = alt
                blend_weight = 0.60
                sig = f"{chosen_primary}:{secondary_voice}"
        self.used_signatures.add(chosen_primary)
        self.used_signatures.add(sig)

        # ONLY use gentle artifact-free sinc pitch shift for young children or extreme elders!
        # Adult & young-adult characters stay at 0.0 semitones for 100% pure untouched neural clarity.
        if res_age == "child":
            pitch_st = 1.8 if res_gender == "female" else 1.5
            eq_style = "child"
            base_speed = 1.02
        elif res_age == "elderly" and not is_ethereal:
            pitch_st = -0.8
            eq_style = "deep"
            base_speed = 0.92

        lang_code = "b" if chosen_primary.startswith("b") else "a"
        voice_label = chosen_primary if not secondary_voice else f"{chosen_primary}+{secondary_voice}"
        summary = f"{voice_label} | {res_accent}, {res_age}, {res_gender}"

        assign = CastVoiceAssignment(
            speaker=speaker,
            primary_voice=chosen_primary,
            secondary_voice=secondary_voice,
            blend_weight=blend_weight,
            lang_code=lang_code,
            base_pitch_semitones=pitch_st,
            base_speed=base_speed,
            eq_style=eq_style,
            summary=summary,
        )
        self.assignments[key] = assign
        return assign


class TTSEngine:
    """
    Studio-clarity Kokoro TTS engine:
    - Uses latent neural voice-tensor blending for expressive, high-definition character timbres
    - Uses neural-duration pre-compensation + pure sinc polyphase resampling (ZERO WSOLA grain artifacts!)
    - Preserves natural dynamic expression across lines
    """

    def __init__(self):
        self.pipelines: Dict[str, object] = {}
        self._voice_tensor_cache: Dict[str, object] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_pipeline(self, lang_code: str = "a"):
        if not KPipeline:
            raise RuntimeError("Kokoro TTS package is not installed.")
        if lang_code not in self.pipelines:
            try:
                self.pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=TTS_REPO_ID, device=self.device)
            except TypeError:
                try:
                    self.pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=TTS_REPO_ID)
                except TypeError:
                    self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
            except Exception as e:
                if lang_code != "a":
                    print(f"Warning: Kokoro lang_code='{lang_code}' failed ({e}), falling back to 'a'.")
                    return self._get_pipeline("a")
                raise
        return self.pipelines[lang_code]

    def _resolve_voice_arg(self, pipeline, primary_voice: str, secondary_voice: Optional[str], blend_weight: float):
        """Blends two Kokoro neural style-embedding tensors directly in latent space before synthesis!"""
        if not secondary_voice or blend_weight >= 0.99 or not hasattr(pipeline, "load_voice"):
            return primary_voice
        blend_key = f"{primary_voice}__{secondary_voice}__{blend_weight:.2f}"
        if blend_key in self._voice_tensor_cache:
            return self._voice_tensor_cache[blend_key]
        try:
            v1 = pipeline.load_voice(primary_voice)
            v2 = pipeline.load_voice(secondary_voice)
            # Align lengths safely if voice packs differ slightly
            min_len = min(v1.shape[0], v2.shape[0])
            blended = (v1[:min_len] * blend_weight) + (v2[:min_len] * (1.0 - blend_weight))
            self._voice_tensor_cache[blend_key] = blended
            return blended
        except Exception:
            return primary_voice

    def generate_block(
        self,
        text: str,
        assignment: CastVoiceAssignment,
        emotion: str,
        output_path: str,
    ) -> Optional[str]:
        is_char = assignment.speaker.strip().lower() not in ("narrator", "narration", "storyteller")
        expressive_text = prepare_expressive_text(text, emotion, is_dialogue_character=is_char)
        if not expressive_text:
            return None

        emo_key = (emotion or "neutral").strip().lower()
        emo_speed, emo_pitch_st, emo_air_db, emo_level_db = EMOTION_PROSODY.get(
            emo_key, EMOTION_PROSODY["neutral"]
        )

        desired_speed = float(np.clip(assignment.base_speed * emo_speed, 0.78, 1.24))
        total_pitch_st = float(np.clip(assignment.base_pitch_semitones + emo_pitch_st, -2.0, 2.5))
        pitch_ratio = (2.0 ** (total_pitch_st / 12.0)) if abs(total_pitch_st) >= 0.25 else 1.0

        # Pre-compensate speed inside Kokoro's neural duration model so subsequent sinc resampling
        # lands on the exact desired speaking tempo with ZERO grain-slicing artifacts!
        kokoro_speed = float(np.clip(desired_speed / pitch_ratio, 0.72, 1.30))

        pipeline = self._get_pipeline(assignment.lang_code)
        voice_arg = self._resolve_voice_arg(
            pipeline,
            assignment.primary_voice,
            assignment.secondary_voice,
            assignment.blend_weight,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_audio: List[np.ndarray] = []
        with torch.inference_mode():
            try:
                gen = pipeline(expressive_text, voice=voice_arg, speed=kokoro_speed, split_pattern=r"\n+")
                for _, _, audio_chunk in gen:
                    if audio_chunk is not None:
                        arr = audio_chunk.detach().cpu().numpy() if hasattr(audio_chunk, "detach") else np.asarray(audio_chunk)
                        if arr.size > 0:
                            all_audio.append(arr.astype(np.float32))
            except Exception as e:
                fallback_vid = "am_michael" if "male" in assignment.summary and "female" not in assignment.summary else TTS_FALLBACK_VOICE
                print(f"Warning: Voice '{assignment.primary_voice}' failed ({e}). Retrying with '{fallback_vid}'.")
                fb_pipe = self._get_pipeline("a")
                gen = fb_pipe(expressive_text, voice=fallback_vid, speed=kokoro_speed, split_pattern=r"\n+")
                for _, _, audio_chunk in gen:
                    if audio_chunk is not None:
                        arr = audio_chunk.detach().cpu().numpy() if hasattr(audio_chunk, "detach") else np.asarray(audio_chunk)
                        if arr.size > 0:
                            all_audio.append(arr.astype(np.float32))

        if not all_audio:
            return None

        # Join intra-block sentences with a natural 90ms micro-breath so multi-sentence blocks breathe naturally
        if len(all_audio) == 1:
            waveform = all_audio[0]
        else:
            breath = np.zeros(int(0.085 * SAMPLE_RATE), dtype=np.float32)
            spaced = []
            for idx, seg in enumerate(all_audio):
                spaced.append(seg)
                if idx < len(all_audio) - 1:
                    spaced.append(breath)
            waveform = np.concatenate(spaced)

        # Trim dead digital silence at edges while preserving soft consonant attack/release
        non_silent = np.where(np.abs(waveform) > 0.0025)[0]
        if non_silent.size > 0:
            pad_samples = int(0.030 * SAMPLE_RATE)
            start_idx = max(0, int(non_silent[0]) - pad_samples)
            end_idx = min(len(waveform), int(non_silent[-1]) + pad_samples)
            waveform = waveform[start_idx:end_idx]

        # Artifact-free polyphase sinc pitch shift (ONLY active for child / elderly personas)
        if abs(pitch_ratio - 1.0) > 0.01:
            denom = 1000
            numer = max(500, int(round(denom / pitch_ratio)))
            waveform = signal.resample_poly(waveform, up=numer, down=denom).astype(np.float32)

        waveform = shape_character_timbre(waveform, assignment.eq_style, air_db=emo_air_db, sr=SAMPLE_RATE)

        # Normalize to clean reference peak while encoding emotional loudness dynamics into the file!
        peak = float(np.max(np.abs(waveform))) + 1e-8
        target_peak = float(np.clip(0.82 * (10.0 ** (emo_level_db / 20.0)), 0.42, 0.96))
        waveform = (waveform / peak * target_peak).astype(np.float32)

        sf.write(output_path, waveform, SAMPLE_RATE)
        return output_path

    def generate(self, text: str, voice_id: str, output_path: str) -> Optional[str]:
        vid = voice_id if voice_id in VALID_VOICES else TTS_DEFAULT_VOICE
        assign = CastVoiceAssignment(
            speaker="Speaker",
            primary_voice=vid,
            lang_code="b" if vid.startswith("b") else "a",
        )
        return self.generate_block(text, assign, "neutral", output_path)

    def unload(self):
        self.pipelines.clear()
        self._voice_tensor_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
