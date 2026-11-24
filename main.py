import argparse
import gc
import os
import re
import shutil
import time
from typing import Dict, Generator, List, Optional, Set, Tuple

import torch

from config import (
    FINAL_FILENAME,
    OUTPUT_DIR,
    SCRIPTS_DIR,
    TEMP_DIR,
    TTS_DEFAULT_VOICE,
)
from core.llm_client import UnifiedLLMClient
from core.music_engine import MusicEngine, normalize_cue_slug
from core.script_engine import (
    CharacterSpec,
    ProductionScript,
    ScriptGenerator,
    build_copyable_ai_prompt,
    parse_pasted_script_json,
)
from core.studio import MixBlockCue, Studio
from core.tts_engine import CharacterVoiceResolver, TTSEngine
from utils.text_utils import chunk_text, read_pdf_file, read_text_file


def get_gpu_status_string() -> str:
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"CUDA GPU Active: {name} ({vram_gb:.1f} GB VRAM)"
        except Exception:
            return "CUDA GPU Active"
    return "CPU Mode (CUDA GPU not detected)"


def slugify_title(name: str) -> str:
    base = os.path.splitext(os.path.basename(name or "audiobook"))[0]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    return slug[:48] or "audiobook"


def build_cast_and_cues_markdown(
    script: ProductionScript,
    resolver: Optional[CharacterVoiceResolver] = None,
) -> str:
    """Builds a readable Markdown summary of characters, voice profiles, and soundscape cues."""
    res = resolver or CharacterVoiceResolver()
    for ch in script.characters:
        res.resolve(ch.name, ch.voice_profile, ch.gender, ch.age_group, ch.accent, ch.voice_id)

    speaker_counts: Dict[str, int] = {}
    speaker_profiles: Dict[str, str] = {}
    bgm_set: List[str] = []
    amb_set: List[str] = []
    sfx_list: List[str] = []

    for b in script.blocks:
        res.resolve(b.speaker, b.voice_profile, b.gender, b.age_group, b.accent, b.voice_id)
        speaker_counts[b.speaker] = speaker_counts.get(b.speaker, 0) + 1
        if b.voice_profile and b.speaker not in speaker_profiles:
            speaker_profiles[b.speaker] = b.voice_profile

        bgm = normalize_cue_slug(b.bg_music)
        if bgm != "none" and bgm not in bgm_set:
            bgm_set.append(bgm)
        amb = normalize_cue_slug(b.ambience)
        if amb != "none" and amb not in amb_set:
            amb_set.append(amb)
        sfx = normalize_cue_slug(b.sfx)
        if sfx != "none":
            sfx_list.append(f"`{sfx}` ({b.sfx_timing})")

    lines = [
        f"### 🎭 Cast & Soundscape Overview ({len(script.blocks)} Dialogue/Narration Blocks)",
        "| Speaker | Character Persona / Profile | Assigned Voice & Acoustic Profile | Lines |",
        "| :--- | :--- | :--- | :---: |",
    ]
    for spk, count in speaker_counts.items():
        assign = res.assignments.get(spk.strip().lower())
        prof = speaker_profiles.get(spk) or "Auto-detected"
        voice_info = assign.summary if assign else "Default"
        lines.append(f"| **{spk}** | {prof} | `{voice_info}` | {count} |")

    lines.append("")
    lines.append(f"- **Musical Themes ({len(bgm_set)})**: " + (", ".join(f"`{x}`" for x in bgm_set) if bgm_set else "_None_"))
    lines.append(f"- **Scene Ambiences ({len(amb_set)})**: " + (", ".join(f"`{x}`" for x in amb_set) if amb_set else "_None_"))
    lines.append(f"- **Timed SFX Cues ({len(sfx_list)})**: " + (", ".join(sfx_list[:18]) + (" ..." if len(sfx_list) > 18 else "") if sfx_list else "_None_"))
    return "\n".join(lines)


class AudiobookConverter:
    """
    4-Stage GPU-Efficient Audiobook Pipeline:
      Stage 1: Direct text into ProductionScript (Cloud API / Optional Ollama / Pasted JSON) & checkpoint JSON
      Stage 2: Deduplicate & pre-generate missing BGM / Ambience / SFX cues, then free AudioLDM 2 VRAM
      Stage 3: Render expressive multi-character speech via Kokoro TTS with full GPU headroom
      Stage 4: Master 4-track audio with scene-continuous BGM, timed SFX & sidechain voice ducking
    """

    def __init__(self, log_callback=print):
        self.log = log_callback
        self._tts_engine: Optional[TTSEngine] = None
        self._cancel_requested = False

        for d in (OUTPUT_DIR, TEMP_DIR, SCRIPTS_DIR):
            os.makedirs(d, exist_ok=True)

    def request_cancel(self):
        self._cancel_requested = True

    def _get_tts(self) -> TTSEngine:
        if self._tts_engine is None:
            self._tts_engine = TTSEngine()
        return self._tts_engine

    def free_gpu_memory(self):
        if self._tts_engine is not None:
            self._tts_engine.unload()
            self._tts_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def load_book_text(input_file: Optional[str] = None, raw_text: Optional[str] = None) -> Tuple[str, str]:
        """Returns (clean_text, project_title)."""
        if raw_text and raw_text.strip():
            return raw_text.strip(), "custom_story"
        if not input_file or not os.path.exists(input_file):
            raise FileNotFoundError("Please upload a valid .txt / .pdf file or paste story text.")
        title = slugify_title(input_file)
        if input_file.lower().endswith(".pdf"):
            text = read_pdf_file(input_file)
        else:
            text = read_text_file(input_file)
        if not text.strip():
            raise ValueError(f"No readable text could be extracted from '{os.path.basename(input_file)}'.")
        return text, title

    def stream_convert_from_text(
        self,
        input_file: Optional[str] = None,
        raw_text: Optional[str] = None,
        provider: str = "NVIDIA NIM API",
        model_name: str = "nvidia/nemotron-3.5-bolt-e4b",
        api_key: str = "",
        base_url: str = "",
        soundscape_mode: str = "neural",
        narrator_voice: str = TTS_DEFAULT_VOICE,
        bgm_volume_db: float = 0.0,
        enable_sfx: bool = True,
        chunk_chars: int = 2800,
    ) -> Generator[Tuple[float, str, Optional[str], str, str], None, None]:
        """
        Generator yielding (progress_0_to_1, log_message, audio_filepath_or_None, script_json_str, cast_md).
        """
        self._cancel_requested = False
        t0 = time.time()

        yield 0.02, f"🚀 {get_gpu_status_string()}", None, "", ""
        text, project_title = self.load_book_text(input_file=input_file, raw_text=raw_text)
        chunks = chunk_text(text, max_chars=chunk_chars)
        if not chunks:
            raise ValueError("Input text produced 0 chunks.")

        yield 0.05, f"📖 Loaded '{project_title}' ({len(text):,} chars -> {len(chunks)} scene chunks).", None, "", ""

        # === STAGE 1: AI Script Direction (with rolling Character Cast & Scene Continuity) ===
        llm_client = UnifiedLLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        script_gen = ScriptGenerator(llm_client=llm_client)

        known_cast: Dict[str, CharacterSpec] = {}
        all_blocks = []
        last_scene_state: Optional[Dict[str, str]] = None

        for idx, chunk in enumerate(chunks):
            if self._cancel_requested:
                raise InterruptedError("Conversion cancelled by user.")

            is_last_chunk = idx == len(chunks) - 1
            prog = 0.06 + 0.32 * (idx / max(1, len(chunks)))
            yield (
                prog,
                f"🧠 [Stage 1/4] Directing Script Chunk {idx + 1}/{len(chunks)} via {provider} ({llm_client.model_name})...",
                None,
                "",
                "",
            )

            chunk_script = script_gen.generate_script(
                chunk,
                known_cast=known_cast,
                last_scene_state=last_scene_state,
                unload_after=is_last_chunk,
            )

            for ch in chunk_script.characters:
                ck = ch.name.strip().lower()
                if ck not in known_cast:
                    known_cast[ck] = ch

            for blk in chunk_script.blocks:
                ck = blk.speaker.strip().lower()
                if ck not in known_cast and blk.voice_profile:
                    known_cast[ck] = CharacterSpec(
                        name=blk.speaker,
                        voice_profile=blk.voice_profile,
                        gender=blk.gender or "neutral",
                        age_group=blk.age_group or "adult",
                        accent=blk.accent or "american",
                        voice_id=blk.voice_id,
                    )
                all_blocks.append(blk)

            if chunk_script.blocks:
                tail = chunk_script.blocks[-1]
                last_scene_state = {
                    "bg_music": tail.bg_music,
                    "ambience": tail.ambience,
                    "speaker": tail.speaker,
                }

        # Ensure local Ollama releases GPU VRAM before Stage 2 loads AudioLDM 2!
        llm_client.unload_local_model()

        full_script = ProductionScript(characters=list(known_cast.values()), blocks=all_blocks)
        if not full_script.blocks:
            raise RuntimeError("Script generation returned 0 spoken dialogue blocks.")
        script_path = os.path.join(SCRIPTS_DIR, f"{project_title}_script.json")
        script_json_str = full_script.model_dump_json(indent=2)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_json_str)

        yield 0.39, f"💾 Saved reusable production script checkpoint -> {script_path}", None, script_json_str, ""

        # Run Stages 2 -> 4 via shared audio production generator
        yield from self._stream_render_script(
            script=full_script,
            project_title=project_title,
            script_json_str=script_json_str,
            soundscape_mode=soundscape_mode,
            narrator_voice=narrator_voice,
            bgm_volume_db=bgm_volume_db,
            enable_sfx=enable_sfx,
            start_time=t0,
            base_progress=0.40,
        )

    def stream_convert_from_json(
        self,
        raw_json_text: str,
        project_title: str = "custom_json_book",
        soundscape_mode: str = "neural",
        narrator_voice: str = TTS_DEFAULT_VOICE,
        bgm_volume_db: float = 0.0,
        enable_sfx: bool = True,
    ) -> Generator[Tuple[float, str, Optional[str], str, str], None, None]:
        """Renders a full multi-voice + BGM + Ambience + SFX audiobook directly from user-pasted JSON."""
        self._cancel_requested = False
        t0 = time.time()

        yield 0.04, f"🚀 {get_gpu_status_string()} | Parsing pasted JSON script...", None, "", ""
        script = parse_pasted_script_json(raw_json_text)
        clean_title = slugify_title(project_title)
        script_json_str = script.model_dump_json(indent=2)

        script_path = os.path.join(SCRIPTS_DIR, f"{clean_title}_pasted_script.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_json_str)

        yield (
            0.10,
            f"✅ Validated {len(script.blocks)} dialogue/narration blocks across {len({b.speaker for b in script.blocks})} speakers!",
            None,
            script_json_str,
            "",
        )

        yield from self._stream_render_script(
            script=script,
            project_title=clean_title,
            script_json_str=script_json_str,
            soundscape_mode=soundscape_mode,
            narrator_voice=narrator_voice,
            bgm_volume_db=bgm_volume_db,
            enable_sfx=enable_sfx,
            start_time=t0,
            base_progress=0.12,
        )

    def _stream_render_script(
        self,
        script: ProductionScript,
        project_title: str,
        script_json_str: str,
        soundscape_mode: str,
        narrator_voice: str,
        bgm_volume_db: float,
        enable_sfx: bool,
        start_time: float,
        base_progress: float = 0.40,
    ) -> Generator[Tuple[float, str, Optional[str], str, str], None, None]:
        rem = 1.0 - base_progress

        # Pre-cast all characters so cast summary appears live right away!
        voice_resolver = CharacterVoiceResolver(narrator_default_voice=narrator_voice)
        for ch in script.characters:
            voice_resolver.resolve(ch.name, ch.voice_profile, ch.gender, ch.age_group, ch.accent, ch.voice_id)
        for blk in script.blocks:
            voice_resolver.resolve(blk.speaker, blk.voice_profile, blk.gender, blk.age_group, blk.accent, blk.voice_id)

        cast_md = build_cast_and_cues_markdown(script, voice_resolver)
        yield base_progress, "🎭 Character voices & persona acoustics locked for entire book.", None, script_json_str, cast_md

        # === STAGE 2: Batch Deduplicated Soundscape Generation (BGM + Ambience + SFX) ===
        bgm_map: Dict[str, Optional[str]] = {"none": None}
        amb_map: Dict[str, Optional[str]] = {"none": None}
        sfx_map: Dict[str, Optional[str]] = {"none": None}

        if soundscape_mode != "none":
            unique_bgm: Set[str] = {b.bg_music for b in script.blocks if normalize_cue_slug(b.bg_music) != "none"}
            unique_amb: Set[str] = {b.ambience for b in script.blocks if normalize_cue_slug(b.ambience) != "none"}
            unique_sfx: Set[str] = (
                {b.sfx for b in script.blocks if normalize_cue_slug(b.sfx) != "none"} if enable_sfx else set()
            )

            total_cues = len(unique_bgm) + len(unique_amb) + len(unique_sfx)
            yield (
                base_progress + rem * 0.04,
                f"🎼 [Stage 2/4] Preparing {len(unique_bgm)} BGM themes, {len(unique_amb)} ambiences, and {len(unique_sfx)} SFX cues ({soundscape_mode} mode)...",
                None,
                script_json_str,
                cast_md,
            )

            music_engine = MusicEngine(mode=soundscape_mode)
            cue_done = 0
            try:
                for cue_raw in sorted(unique_bgm):
                    if self._cancel_requested:
                        raise InterruptedError("Conversion cancelled by user.")
                    logs_buf: List[str] = []
                    bgm_map[cue_raw] = music_engine.get_or_create_cue(cue_raw, "bgm", log_fn=logs_buf.append)
                    cue_done += 1
                    if logs_buf:
                        yield (
                            base_progress + rem * (0.04 + 0.24 * (cue_done / max(1, total_cues))),
                            "\n".join(logs_buf),
                            None,
                            script_json_str,
                            cast_md,
                        )

                for cue_raw in sorted(unique_amb):
                    if self._cancel_requested:
                        raise InterruptedError("Conversion cancelled by user.")
                    logs_buf = []
                    amb_map[cue_raw] = music_engine.get_or_create_cue(cue_raw, "ambience", log_fn=logs_buf.append)
                    cue_done += 1
                    if logs_buf:
                        yield (
                            base_progress + rem * (0.04 + 0.24 * (cue_done / max(1, total_cues))),
                            "\n".join(logs_buf),
                            None,
                            script_json_str,
                            cast_md,
                        )

                for cue_raw in sorted(unique_sfx):
                    if self._cancel_requested:
                        raise InterruptedError("Conversion cancelled by user.")
                    logs_buf = []
                    sfx_map[cue_raw] = music_engine.get_or_create_cue(cue_raw, "sfx", log_fn=logs_buf.append)
                    cue_done += 1
                    if logs_buf:
                        yield (
                            base_progress + rem * (0.04 + 0.24 * (cue_done / max(1, total_cues))),
                            "\n".join(logs_buf),
                            None,
                            script_json_str,
                            cast_md,
                        )
            finally:
                # Immediately unload AudioLDM 2 so Kokoro TTS gets 100% free GPU VRAM!
                music_engine.unload_model()

        # === STAGE 3: High-Speed Expressive Kokoro TTS Synthesis ===
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        tts = self._get_tts()
        mix_cues: List[MixBlockCue] = []
        n_blocks = len(script.blocks)

        yield (
            base_progress + rem * 0.30,
            f"🎙️ [Stage 3/4] Synthesizing {n_blocks} expressive voice blocks on {tts.device.upper()}...",
            None,
            script_json_str,
            cast_md,
        )

        for b_idx, blk in enumerate(script.blocks):
            if self._cancel_requested:
                raise InterruptedError("Conversion cancelled by user.")

            assignment = voice_resolver.resolve(
                blk.speaker,
                blk.voice_profile,
                blk.gender,
                blk.age_group,
                blk.accent,
                blk.voice_id,
            )
            voice_file = os.path.join(TEMP_DIR, f"block_{b_idx:05d}.wav")

            try:
                generated_wav = tts.generate_block(blk.text, assignment, blk.emotion, voice_file)
            except Exception as e:
                yield (
                    base_progress + rem * (0.30 + 0.55 * ((b_idx + 1) / max(1, n_blocks))),
                    f"  ⚠️ TTS warning on block {b_idx + 1} ({blk.speaker}): {e}",
                    None,
                    script_json_str,
                    cast_md,
                )
                continue

            if not generated_wav or not os.path.exists(generated_wav):
                continue

            # Guaranteed 1-to-1 synchronization: MixBlockCue binds voice + bgm + ambience + sfx atomically!
            mix_cues.append(
                MixBlockCue(
                    voice_path=generated_wav,
                    bgm_path=bgm_map.get(blk.bg_music) if soundscape_mode != "none" else None,
                    music_intensity=blk.music_intensity,
                    ambience_path=amb_map.get(blk.ambience) if soundscape_mode != "none" else None,
                    sfx_path=sfx_map.get(blk.sfx) if (soundscape_mode != "none" and enable_sfx) else None,
                    sfx_timing=blk.sfx_timing,
                    pause_after_ms=blk.pause_after_ms,
                    emotion=blk.emotion,
                )
            )

            if (b_idx + 1) % max(1, min(5, n_blocks // 6 or 1)) == 0 or (b_idx + 1) == n_blocks:
                snippet = blk.text[:48].replace("\n", " ")
                yield (
                    base_progress + rem * (0.30 + 0.55 * ((b_idx + 1) / max(1, n_blocks))),
                    f"  🗣️ Rendered [{b_idx + 1}/{n_blocks}] {blk.speaker} ({blk.emotion}): \"{snippet}...\"",
                    None,
                    script_json_str,
                    cast_md,
                )

        # === STAGE 4: 4-Track Cinema Mixing & Sidechain Ducking ===
        yield (
            base_progress + rem * 0.88,
            "🎚️ [Stage 4/4] Mastering 4-track audio (Scene-Continuous BGM + Stereo Ambience + Timed SFX + 14dB Voice Ducking)...",
            None,
            script_json_str,
            cast_md,
        )

        out_filename = f"{project_title}_audiobook.mp3" if project_title else FINAL_FILENAME
        output_path = os.path.join(OUTPUT_DIR, out_filename)
        studio = Studio(bgm_volume_offset_db=bgm_volume_db, enable_sfx=enable_sfx)
        final_file = studio.mix_cues(mix_cues, output_path)

        # Clean up temporary per-line WAVs while keeping persistent audio_library & saved scripts!
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        elapsed = time.time() - start_time
        yield (
            1.0,
            f"✨ Conversion Complete in {elapsed:.1f}s! Mastered audiobook saved to: {final_file}",
            final_file,
            script_json_str,
            cast_md,
        )

    def convert(self, input_file: str, progress_bar=None) -> Tuple[Optional[str], str]:
        """Synchronous CLI-friendly wrapper around the 4-stage streaming pipeline."""
        final_audio = None
        last_msg = ""
        for prog, msg, audio_path, _, _ in self.stream_convert_from_text(
            input_file=input_file,
            provider="Ollama (Local - Optional)",
        ):
            self.log(msg)
            last_msg = msg
            if progress_bar:
                progress_bar(prog)
            if audio_path:
                final_audio = audio_path
        return final_audio, last_msg


def main():
    parser = argparse.ArgumentParser(description="NexGen AI Cinema Audiobook Converter")
    parser.add_argument("input_file", help="Path to .txt, .pdf, or .json script file")
    parser.add_argument("--provider", default="NVIDIA NIM API", help="LLM Provider ('NVIDIA NIM API', 'Google Gemini API', 'OpenAI API', 'Ollama (Local - Optional)')")
    parser.add_argument("--model", default="", help="Model name (e.g. nvidia/nemotron-3.5-bolt-e4b, gemini-2.5-flash, gpt-5-mini)")
    parser.add_argument("--api-key", default="", help="API Key (or set NVIDIA_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--from-json", action="store_true", help="Treat input_file as a pre-generated ProductionScript JSON file")
    parser.add_argument("--export-prompt", action="store_true", help="Export copyable AI prompt(s) for input_file without calling an API")
    parser.add_argument("--soundscape", choices=["neural", "procedural", "none"], default="neural", help="Soundscape engine mode")
    args = parser.parse_args()

    converter = AudiobookConverter()

    if args.export_prompt:
        text, title = converter.load_book_text(input_file=args.input_file)
        chunks = chunk_text(text, max_chars=3800)
        prompt_out = os.path.join(SCRIPTS_DIR, f"{title}_ai_prompts.txt")
        all_prompts = [
            build_copyable_ai_prompt(ch, idx + 1, len(chunks))
            for idx, ch in enumerate(chunks)
        ]
        with open(prompt_out, "w", encoding="utf-8") as f:
            f.write(("\n\n" + "=" * 80 + "\n\n").join(all_prompts))
        print(f"Exported {len(chunks)} copyable prompt(s) to: {prompt_out}")
        return

    if args.from_json or args.input_file.lower().endswith(".json"):
        raw_json = read_text_file(args.input_file)
        for _, msg, _, _, _ in converter.stream_convert_from_json(
            raw_json_text=raw_json,
            project_title=slugify_title(args.input_file),
            soundscape_mode=args.soundscape,
        ):
            print(msg)
    else:
        from config import PROVIDER_MODELS
        chosen_model = args.model or PROVIDER_MODELS.get(args.provider, ["llama3"])[0]
        for _, msg, _, _, _ in converter.stream_convert_from_text(
            input_file=args.input_file,
            provider=args.provider,
            model_name=chosen_model,
            api_key=args.api_key,
            soundscape_mode=args.soundscape,
        ):
            print(msg)


if __name__ == "__main__":
    main()
