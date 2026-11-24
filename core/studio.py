import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from config import (
    AMBIENCE_DUCKED_DBFS,
    AMBIENCE_OPEN_DBFS,
    BGM_DUCKED_DBFS,
    BGM_OPEN_DBFS,
    MIN_VOICE_TO_BG_RATIO_DB,
    SAMPLE_RATE,
    SFX_MAX_DBFS,
    SFX_UNDER_SPEECH_DBFS,
    VOICE_TARGET_DBFS,
)


@dataclass
class MixBlockCue:
    voice_path: str
    bgm_path: Optional[str] = None
    music_intensity: float = 0.55
    ambience_path: Optional[str] = None
    sfx_path: Optional[str] = None
    sfx_timing: str = "before"  # "before", "start", "under", "after"
    pause_after_ms: int = 280
    emotion: str = "neutral"


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _load_mono_resampled(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        mono = signal.resample_poly(mono, up=target_sr // g, down=sr // g).astype(np.float32)
    return mono


def _enhance_voice_clarity(voice: np.ndarray, emotion: str = "neutral", sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Transparent Studio Vocal Chain (Preserves 100% of neural expression & dynamic contrast!):
      1. Clean 78 Hz Butterworth rumble filter (no low mud)
      2. Gentle wide +1.0 dB high-shelf air lift above 4.6 kHz (zero narrow nasal resonance, zero tanh distortion!)
      3. Preserves the relative emotional peak level encoded by TTSEngine (so whispers stay intimate,
         tender lines stay warm, and dramatic/shouting lines project with real dynamic impact!)
    """
    if voice.size < 256:
        return voice

    hp_sos = signal.butter(2, 78.0 / (0.5 * sr), btype="highpass", output="sos")
    y = signal.sosfilt(hp_sos, voice).astype(np.float32)

    # Subtle wide studio condenser air shelf (> 4.6 kHz, +1.0 dB)
    A = 10.0 ** (1.0 / 40.0)
    w0 = 2.0 * math.pi * 4600.0 / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / 2.0 * math.sqrt(2.0)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * math.sqrt(A) * alpha
    y = signal.lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], y).astype(np.float32)

    # Scale relative to TTSEngine's 0.82 neutral reference so emotional loudness differences survive intact!
    master_ref_scale = _db_to_lin(VOICE_TARGET_DBFS) / 0.82
    y = y * master_ref_scale

    peak = float(np.max(np.abs(y)))
    max_ceil = _db_to_lin(-0.8)
    if peak > max_ceil:
        y *= max_ceil / peak
    return y.astype(np.float32)


def _carve_vocal_pocket(bg_audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Smooth wide vocal-pocket dip (~3.0 dB around 2.1 kHz, Q=0.65) so music wraps warmly around speech."""
    if bg_audio.size < 512:
        return bg_audio
    b, a = signal.iirpeak(2100.0 / (0.5 * sr), Q=0.65)
    mid_band = signal.lfilter(b, a, bg_audio).astype(np.float32)
    return (bg_audio - 0.28 * mid_band).astype(np.float32)


def _compute_asymmetric_duck_envelope(voice_mono: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Broadcast-grade asymmetric sidechain envelope:
      - 55ms lookahead + fast 40ms attack (ducks BGM smoothly BEFORE the first syllable begins)
      - 380ms hold across natural inter-word breaths (zero annoying pumping)
      - Smooth 580ms release so music swells organically during 1.0s-2.5s dramatic story pauses!
    """
    n = len(voice_mono)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    win_20ms = max(1, int(0.020 * sr))
    local_peak = ndimage.maximum_filter1d(np.abs(voice_mono), size=win_20ms)
    gate = (local_peak > _db_to_lin(-41.0)).astype(np.float32)

    # Hold across short inter-clause breaths (< 380ms)
    gate = ndimage.maximum_filter1d(gate, size=int(0.38 * sr))

    # 55ms lookahead shift
    lookahead = int(0.055 * sr)
    if lookahead < n:
        gate[:-lookahead] = np.maximum(gate[:-lookahead], gate[lookahead:])

    # Downsample to 200 Hz control rate for ultra-fast asymmetric attack/release smoothing (<1ms!)
    step = max(1, sr // 200)
    ctrl = gate[::step]
    ctrl_sr = sr / step
    atk_coeff = math.exp(-1.0 / max(1.0, 0.040 * ctrl_sr))
    rel_coeff = math.exp(-1.0 / max(1.0, 0.580 * ctrl_sr))

    env_ctrl = np.zeros_like(ctrl)
    cur = 0.0
    for i in range(len(ctrl)):
        target = float(ctrl[i])
        coeff = atk_coeff if target > cur else rel_coeff
        cur = coeff * cur + (1.0 - coeff) * target
        env_ctrl[i] = cur

    # Interpolate smooth control curve back to sample rate
    x_ctrl = np.arange(len(env_ctrl), dtype=np.float32) * step
    x_full = np.arange(n, dtype=np.float32)
    return np.clip(np.interp(x_full, x_ctrl, env_ctrl), 0.0, 1.0).astype(np.float32)


def _render_seamless_loop(source: np.ndarray, target_len: int, crossfade_samples: int) -> np.ndarray:
    if target_len <= 0 or source.size == 0:
        return np.zeros(max(0, target_len), dtype=np.float32)

    src_peak = float(np.max(np.abs(source))) + 1e-8
    src = (source / src_peak).astype(np.float32)

    if len(src) >= target_len:
        return src[:target_len].copy()

    xfade = min(crossfade_samples, len(src) // 4)
    if xfade < 64:
        reps = (target_len // len(src)) + 2
        return np.tile(src, reps)[:target_len]

    step = len(src) - xfade
    out = np.zeros(target_len + len(src), dtype=np.float32)
    theta = np.linspace(0.0, math.pi / 2.0, xfade, dtype=np.float32)
    fade_in = np.sin(theta)
    fade_out = np.cos(theta)

    pos = 0
    first = True
    while pos < target_len:
        chunk = src.copy()
        if not first:
            out[pos : pos + xfade] = out[pos : pos + xfade] * fade_out + chunk[:xfade] * fade_in
            out[pos + xfade : pos + len(chunk)] = chunk[xfade:]
        else:
            out[pos : pos + len(chunk)] = chunk
            first = False
        pos += step

    return out[:target_len]


def _make_stereo_decorrelated(mono: np.ndarray, delay_ms: float = 11.0, width: float = 0.35, sr: int = SAMPLE_RATE) -> np.ndarray:
    if mono.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    d = int(delay_ms * 0.001 * sr)
    if len(mono) <= d:
        return np.stack([mono, mono], axis=1).astype(np.float32)
    side = np.roll(mono, d)
    side[:d] = 0.0
    left = mono + width * side
    right = mono - width * side
    return np.stack([left, right], axis=1).astype(np.float32)


class Studio:
    """
    4-Track Cinema Mastering Engine:
    - Preserves full character vocal dynamics & natural emotional contrast
    - Scene-continuous 24kHz BGM with emotional intensity swells & 1.8s crossfades
    - Asymmetric lookahead sidechain ducking (swells in dramatic pauses, ducks cleanly under speech)
    """

    def __init__(self, bgm_volume_offset_db: float = 0.0, enable_sfx: bool = True):
        self.bgm_volume_offset_db = float(bgm_volume_offset_db)
        self.enable_sfx = enable_sfx
        self._audio_cache: Dict[str, np.ndarray] = {}

    def _get_cached_audio(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path or not os.path.exists(path):
            return None
        if path not in self._audio_cache:
            self._audio_cache[path] = _load_mono_resampled(path, SAMPLE_RATE)
        return self._audio_cache[path]

    def mix_cues(self, cues: List[MixBlockCue], output_file: str) -> str:
        valid_cues = [c for c in cues if c.voice_path and os.path.exists(c.voice_path)]
        if not valid_cues:
            raise RuntimeError("No valid generated voice segments found to mix.")

        sr = SAMPLE_RATE
        lead_in_samples = int(0.85 * sr)
        tail_out_samples = int(1.50 * sr)

        voice_placements: List[Tuple[int, np.ndarray]] = []
        sfx_placements: List[Tuple[int, np.ndarray, bool]] = []
        block_spans: List[Tuple[int, int, Optional[str], float, Optional[str]]] = []

        cursor = lead_in_samples

        for idx, cue in enumerate(valid_cues):
            span_start = cursor
            raw_voice = _load_mono_resampled(cue.voice_path, sr)
            crisp_voice = _enhance_voice_clarity(raw_voice, emotion=cue.emotion, sr=sr)

            sfx_wav = self._get_cached_audio(cue.sfx_path) if self.enable_sfx else None
            timing = (cue.sfx_timing or "before").lower()

            if sfx_wav is not None and sfx_wav.size > 0 and timing == "before":
                max_pre_len = min(len(sfx_wav), int(2.2 * sr))
                trimmed_sfx = sfx_wav[:max_pre_len]
                beat_advance = min(int(0.85 * sr), int(len(trimmed_sfx) * 0.68))
                sfx_placements.append((cursor, trimmed_sfx, False))
                cursor += beat_advance

            voice_start = cursor
            voice_end = voice_start + len(crisp_voice)
            voice_placements.append((voice_start, crisp_voice))

            pause_samples = int((cue.pause_after_ms / 1000.0) * sr)
            if sfx_wav is not None and sfx_wav.size > 0:
                if timing == "start":
                    onset = max(0, voice_start - int(0.06 * sr))
                    sfx_placements.append((onset, sfx_wav[: int(2.5 * sr)], True))
                elif timing == "under":
                    under_bed = _render_seamless_loop(sfx_wav, len(crisp_voice), int(0.25 * sr))
                    sfx_placements.append((voice_start, under_bed, True))
                elif timing == "after":
                    after_onset = voice_end + int(0.05 * sr)
                    trimmed_after = sfx_wav[: int(2.4 * sr)]
                    sfx_placements.append((after_onset, trimmed_after, False))
                    pause_samples = max(pause_samples, min(int(0.95 * sr), int(len(trimmed_after) * 0.72)))

            cursor = voice_end + pause_samples
            span_end = cursor + (tail_out_samples if idx == len(valid_cues) - 1 else 0)
            if idx == 0:
                span_start = 0

            block_spans.append((span_start, span_end, cue.bgm_path, cue.music_intensity, cue.ambience_path))

        total_samples = max(cursor + tail_out_samples, block_spans[-1][1])

        voice_mono = np.zeros(total_samples, dtype=np.float32)
        for v_start, v_arr in voice_placements:
            end = min(total_samples, v_start + len(v_arr))
            voice_mono[v_start:end] += v_arr[: end - v_start]

        # Asymmetric Lookahead Sidechain Ducking Curve
        speech_env = _compute_asymmetric_duck_envelope(voice_mono, sr=sr)

        # Scene-Continuous BGM
        bgm_mono = self._render_scene_layer(
            block_spans,
            total_samples=total_samples,
            use_ambience=False,
            crossfade_sec=1.9,
            loop_xfade_sec=1.6,
        )
        bgm_mono = _carve_vocal_pocket(bgm_mono, sr)

        open_bgm_lin = _db_to_lin(BGM_OPEN_DBFS + self.bgm_volume_offset_db)
        duck_bgm_lin = _db_to_lin(BGM_DUCKED_DBFS + self.bgm_volume_offset_db)
        bgm_mono *= (open_bgm_lin * (1.0 - speech_env) + duck_bgm_lin * speech_env)

        # Subtle Scene Ambience
        amb_mono = self._render_scene_layer(
            block_spans,
            total_samples=total_samples,
            use_ambience=True,
            crossfade_sec=2.2,
            loop_xfade_sec=1.4,
        )
        amb_mono = _carve_vocal_pocket(amb_mono, sr)
        open_amb_lin = _db_to_lin(AMBIENCE_OPEN_DBFS + self.bgm_volume_offset_db * 0.5)
        duck_amb_lin = _db_to_lin(AMBIENCE_DUCKED_DBFS + self.bgm_volume_offset_db * 0.5)
        amb_mono *= (open_amb_lin * (1.0 - speech_env) + duck_amb_lin * speech_env)

        # Timed Spot SFX
        sfx_mono = np.zeros(total_samples, dtype=np.float32)
        fade_short = int(0.025 * sr)
        for s_start, s_arr, is_overlapping in sfx_placements:
            if s_start >= total_samples or s_arr.size == 0:
                continue
            end = min(total_samples, s_start + len(s_arr))
            seg = s_arr[: end - s_start].copy()
            peak = float(np.max(np.abs(seg))) + 1e-8
            ceil_db = SFX_UNDER_SPEECH_DBFS if is_overlapping else SFX_MAX_DBFS
            seg = seg * (_db_to_lin(ceil_db) / peak)
            if len(seg) > fade_short * 2:
                seg[:fade_short] *= np.linspace(0.0, 1.0, fade_short, dtype=np.float32)
                seg[-fade_short:] *= np.linspace(1.0, 0.0, fade_short, dtype=np.float32)
            sfx_mono[s_start:end] += seg

        sfx_mono *= (1.0 - 0.58 * speech_env)

        # Hard Voice-Dominance Guarantee during active speech
        combined_bg_mono = bgm_mono + amb_mono + sfx_mono
        active_mask = speech_env > 0.65
        if np.any(active_mask):
            v_rms = float(np.sqrt(np.mean(voice_mono[active_mask] ** 2)) + 1e-8)
            bg_rms = float(np.sqrt(np.mean(combined_bg_mono[active_mask] ** 2)) + 1e-8)
            max_allowed_bg_rms = v_rms * _db_to_lin(-MIN_VOICE_TO_BG_RATIO_DB)
            if bg_rms > max_allowed_bg_rms:
                scale_down = max_allowed_bg_rms / bg_rms
                bgm_mono *= scale_down
                amb_mono *= scale_down
                sfx_mono *= scale_down

        intro_n = min(int(0.65 * sr), total_samples // 6)
        outro_n = min(int(1.35 * sr), total_samples // 4)
        if intro_n > 0:
            ramp_in = np.linspace(0.0, 1.0, intro_n, dtype=np.float32)
            bgm_mono[:intro_n] *= ramp_in
            amb_mono[:intro_n] *= ramp_in
        if outro_n > 0:
            ramp_out = np.linspace(1.0, 0.0, outro_n, dtype=np.float32)
            bgm_mono[-outro_n:] *= ramp_out
            amb_mono[-outro_n:] *= ramp_out

        bgm_stereo = _make_stereo_decorrelated(bgm_mono, delay_ms=10.5, width=0.26, sr=sr)
        amb_stereo = _make_stereo_decorrelated(amb_mono, delay_ms=14.5, width=0.36, sr=sr)
        sfx_stereo = _make_stereo_decorrelated(sfx_mono, delay_ms=5.0, width=0.16, sr=sr)
        voice_stereo = np.stack([voice_mono, voice_mono], axis=1)

        master = voice_stereo + bgm_stereo + amb_stereo + sfx_stereo

        # Transparent peak protection only if peaks exceed -0.6 dBFS (never crushes vocal micro-dynamics!)
        ceiling = _db_to_lin(-0.6)
        peak = float(np.max(np.abs(master)))
        if peak > ceiling:
            master *= ceiling / peak

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return self._export_stereo(master, sr, output_file)

    def _render_scene_layer(
        self,
        block_spans: List[Tuple[int, int, Optional[str], float, Optional[str]]],
        total_samples: int,
        use_ambience: bool,
        crossfade_sec: float,
        loop_xfade_sec: float,
    ) -> np.ndarray:
        out = np.zeros(total_samples, dtype=np.float32)
        if not block_spans:
            return out

        intensity_curve = np.ones(total_samples, dtype=np.float32)
        if not use_ambience:
            for b_start, b_end, _, intensity, _ in block_spans:
                mult = 0.0 if intensity <= 0.02 else float(0.45 + 0.85 * intensity)
                intensity_curve[b_start : min(total_samples, b_end)] = mult
            intensity_curve = ndimage.uniform_filter1d(intensity_curve, size=max(1, int(0.75 * SAMPLE_RATE)))

        merged_scenes: List[Tuple[int, int, Optional[str]]] = []
        for b_start, b_end, bgm_p, _, amb_p in block_spans:
            path = amb_p if use_ambience else bgm_p
            if merged_scenes and merged_scenes[-1][2] == path:
                prev_s, _, prev_p = merged_scenes[-1]
                merged_scenes[-1] = (prev_s, b_end, prev_p)
            else:
                merged_scenes.append((b_start, b_end, path))

        xfade_n = int(crossfade_sec * SAMPLE_RATE)
        loop_xfade_n = int(loop_xfade_sec * SAMPLE_RATE)

        for idx, (s_start, s_end, path) in enumerate(merged_scenes):
            src = self._get_cached_audio(path)
            if src is None or src.size == 0:
                continue

            actual_start = max(0, s_start - (xfade_n // 2 if idx > 0 else 0))
            actual_end = min(total_samples, s_end + (xfade_n // 2 if idx < len(merged_scenes) - 1 else 0))
            seg_len = actual_end - actual_start
            if seg_len <= 0:
                continue

            looped = _render_seamless_loop(src, seg_len, loop_xfade_n)

            fi = min(xfade_n, seg_len // 3) if idx > 0 else min(int(0.30 * SAMPLE_RATE), seg_len // 4)
            fo = min(xfade_n, seg_len // 3) if idx < len(merged_scenes) - 1 else min(int(0.45 * SAMPLE_RATE), seg_len // 4)
            if fi > 0:
                looped[:fi] *= np.sin(np.linspace(0.0, math.pi / 2.0, fi, dtype=np.float32))
            if fo > 0:
                looped[-fo:] *= np.cos(np.linspace(0.0, math.pi / 2.0, fo, dtype=np.float32))

            out[actual_start:actual_end] += looped

        if not use_ambience:
            out *= intensity_curve
        return out

    @staticmethod
    def _resolve_ffmpeg_bin() -> Optional[str]:
        ff = shutil.which("ffmpeg")
        if ff:
            return ff
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @classmethod
    def _export_stereo(cls, stereo_float: np.ndarray, sr: int, output_file: str) -> str:
        pcm16 = np.clip(stereo_float * 32767.0, -32768, 32767).astype(np.int16)
        if output_file.lower().endswith(".wav"):
            sf.write(output_file, pcm16, sr, subtype="PCM_16")
            return output_file

        ffmpeg_bin = cls._resolve_ffmpeg_bin()
        if ffmpeg_bin:
            temp_wav = os.path.splitext(output_file)[0] + "_master_tmp.wav"
            try:
                sf.write(temp_wav, pcm16, sr, subtype="PCM_16")
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    temp_wav,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    output_file,
                ]
                subprocess.run(cmd, check=True)
                return output_file
            except Exception:
                pass
            finally:
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass

        wav_out = os.path.splitext(output_file)[0] + ".wav"
        sf.write(wav_out, pcm16, sr, subtype="PCM_16")
        return wav_out

    def mix(self, voice_files: List[str], music_files: List[str], output_file: str) -> str:
        cues = [
            MixBlockCue(
                voice_path=v,
                bgm_path=music_files[i] if i < len(music_files) else None,
            )
            for i, v in enumerate(voice_files)
        ]
        return self.mix_cues(cues, output_file)
