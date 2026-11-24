import math
import os
import numpy as np
import soundfile as sf
from scipy import signal


PROC_SR = 24000


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generates smooth natural 1/f pink noise."""
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / PROC_SR)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(np.fft.rfft(white) * scale, n=n_samples)
    peak = np.max(np.abs(pink)) + 1e-8
    return (pink / peak).astype(np.float32)


def apply_velvet_hall_reverb(mono: np.ndarray, mix: float = 0.24, sr: int = PROC_SR) -> np.ndarray:
    """Lightweight concert-hall Schroeder/allpass diffuser for warm orchestral cohesion."""
    if mono.size < 1024 or mix <= 0.01:
        return mono
    y = mono.astype(np.float32)
    wet = np.zeros_like(y)

    # 4 prime-spaced parallel comb reflections with warm high-damping
    comb_delays_ms = (29.7, 37.1, 41.1, 43.7)
    comb_gains = (0.36, 0.31, 0.27, 0.23)
    for d_ms, g in zip(comb_delays_ms, comb_gains):
        d = int(d_ms * 0.001 * sr)
        if d < len(y):
            echo1 = np.zeros_like(y)
            echo1[d:] = y[:-d] * g
            if 2 * d < len(y):
                echo1[2 * d :] += y[: -2 * d] * (g * 0.48)
            if 3 * d < len(y):
                echo1[3 * d :] += y[: -3 * d] * (g * 0.22)
            wet += echo1

    # Dampen highs above 3.6 kHz on reverb tail for warm cinematic hall sound
    lp = signal.butter(2, 3600.0 / (0.5 * sr), btype="lowpass", output="sos")
    hp = signal.butter(2, 95.0 / (0.5 * sr), btype="highpass", output="sos")
    wet = signal.sosfilt(hp, signal.sosfilt(lp, wet)).astype(np.float32)

    out = (1.0 - mix) * y + mix * wet
    peak = float(np.max(np.abs(out))) + 1e-8
    if peak > 0.96:
        out *= 0.94 / peak
    return out


def build_harmonic_score_bed(theme: str, duration: float) -> np.ndarray:
    """
    Synthesizes an evolving 2-chord cinematic string/pad progression + delicate felt-piano/harp notes
    at pure 24kHz. Used both standalone (Procedural 0-VRAM mode) and as a clean 20% harmonic foundation
    under AudioLDM 2 orchestral tracks!
    """
    key = (theme or "peaceful_calm").lower()
    n = int(duration * PROC_SR)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)

    # Select 2-chord emotional progression + arpeggio notes + pulse + filter brightness
    if any(k in key for k in ("intense", "action", "epic", "battle")):
        chord_a = [73.42, 110.0, 146.83, 220.0, 293.66]     # Dm power
        chord_b = [58.27, 116.54, 174.61, 233.08, 293.66]   # Bb maj7 dramatic lift
        arp = [293.66, 349.23, 440.0, 349.23, 523.25, 440.0]
        pulse_hz, cutoff = 2.0, 1550.0
    elif any(k in key for k in ("suspense", "tense", "thriller")):
        chord_a = [65.41, 130.81, 155.56, 196.0]            # Cm dark pedal
        chord_b = [65.41, 138.59, 164.81, 207.65]           # C pedal + Db Phrygian tension
        arp = [196.0, 207.65, 261.63, 196.0]
        pulse_hz, cutoff = 1.1, 920.0
    elif any(k in key for k in ("sad", "melanchol", "grief", "emotional")):
        chord_a = [110.0, 164.81, 220.0, 261.63, 329.63]    # Am9 poignant strings
        chord_b = [87.31, 130.81, 174.61, 220.0, 261.63]    # Fmaj7 tearful resolution
        arp = [440.0, 392.0, 329.63, 293.66, 261.63]
        pulse_hz, cutoff = 0.0, 860.0
    elif any(k in key for k in ("mystery", "eerie", "enigma")):
        chord_a = [82.41, 164.81, 246.94, 293.66, 369.99]   # E Dorian shimmer
        chord_b = [110.0, 164.81, 220.0, 277.18, 329.63]    # A luminous mystery
        arp = [493.88, 587.33, 739.99, 659.25, 493.88]
        pulse_hz, cutoff = 0.0, 1280.0
    elif any(k in key for k in ("dark", "ominous", "horror")):
        chord_a = [55.0, 82.41, 110.0, 164.81]              # Deep A1 abyss drone
        chord_b = [51.91, 77.78, 103.83, 155.56]            # Ab1 sinking shadow
        arp = [164.81, 155.56, 130.81]
        pulse_hz, cutoff = 0.4, 620.0
    elif any(k in key for k in ("roman", "warm", "heart", "tender", "love")):
        chord_a = [130.81, 196.0, 246.94, 329.63, 392.0]    # Cmaj9 tender warmth
        chord_b = [87.31, 174.61, 220.0, 261.63, 329.63]    # Fmaj9 embrace
        arp = [392.0, 329.63, 293.66, 329.63, 440.0]
        pulse_hz, cutoff = 0.0, 1120.0
    elif any(k in key for k in ("triumphant", "happy", "joy", "playful")):
        chord_a = [130.81, 196.0, 261.63, 329.63, 392.0]    # C noble major
        chord_b = [98.0, 196.0, 246.94, 293.66, 392.0]      # G soaring lift
        arp = [261.63, 329.63, 392.0, 523.25, 392.0]
        pulse_hz, cutoff = 0.65, 1450.0
    else:
        chord_a = [130.81, 196.0, 246.94, 293.66]
        chord_b = [110.0, 164.81, 220.0, 261.63]
        arp = [329.63, 293.66, 261.63]
        pulse_hz, cutoff = 0.0, 960.0

    def render_chord(freqs: list[float]) -> np.ndarray:
        layer = np.zeros(n, dtype=np.float32)
        for idx, f0 in enumerate(freqs):
            lfo = 0.14 * np.sin(2.0 * math.pi * (0.16 + 0.04 * idx) * t)
            for det in (-0.0032, 0.0, 0.0032):
                f = f0 * (1.0 + det)
                layer += (
                    0.68 * np.sin(2.0 * math.pi * f * t + lfo)
                    + 0.22 * np.sin(2.0 * math.pi * (2.0 * f) * t)
                    + 0.08 * np.sin(2.0 * math.pi * (3.0 * f) * t)
                )
        return layer

    # Smooth slow crossfade between Chord A and Chord B and back to Chord A (for seamless loop!)
    morph = (0.5 - 0.5 * np.cos(2.0 * math.pi * t / max(duration, 1.0))).astype(np.float32)
    pad = render_chord(chord_a) * (1.0 - morph) + render_chord(chord_b) * morph

    sos = signal.butter(3, min(cutoff, PROC_SR * 0.4) / (0.5 * PROC_SR), btype="lowpass", output="sos")
    pad = signal.sosfilt(sos, pad).astype(np.float32)
    pad /= float(np.max(np.abs(pad))) + 1e-8

    if pulse_hz > 0.05:
        pulse_env = 0.66 + 0.34 * (0.5 + 0.5 * np.sin(2.0 * math.pi * pulse_hz * t - math.pi / 2))
        pad *= pulse_env.astype(np.float32)

    # Add gentle felt-piano / harp harmonic droplets spaced naturally across the bar
    notes_layer = np.zeros(n, dtype=np.float32)
    note_spacing = max(1.35, duration / max(3, len(arp) + 1))
    for i, nf in enumerate(arp):
        onset_s = 0.45 + i * note_spacing
        idx0 = int(onset_s * PROC_SR)
        note_len = min(int(2.4 * PROC_SR), n - idx0)
        if note_len > 256:
            nt = np.linspace(0.0, note_len / PROC_SR, note_len, endpoint=False, dtype=np.float32)
            env = (1.0 - np.exp(-nt * 45.0)) * np.exp(-nt * 1.9)
            bell = (
                0.75 * np.sin(2.0 * math.pi * nf * nt)
                + 0.20 * np.sin(2.0 * math.pi * (2.0 * nf) * nt) * np.exp(-nt * 3.5)
            ) * env
            notes_layer[idx0 : idx0 + note_len] += bell.astype(np.float32)

    combined = 0.78 * pad + 0.24 * notes_layer
    combined = apply_velvet_hall_reverb(combined, mix=0.30, sr=PROC_SR)

    fade_n = min(int(0.75 * PROC_SR), n // 4)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        combined[:fade_n] *= ramp
        combined[-fade_n:] *= ramp[::-1]

    peak = float(np.max(np.abs(combined))) + 1e-8
    return (combined / peak * 0.84).astype(np.float32)


def synthesize_procedural_bgm(theme: str, duration: float, output_path: str) -> str:
    audio = build_harmonic_score_bed(theme, duration)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, PROC_SR)
    return output_path


def synthesize_procedural_ambience(ambience: str, duration: float, output_path: str) -> str:
    """Synthesizes clean, natural 24kHz environmental beds (ocean swells, rain, wind, hearth)."""
    key = (ambience or "gentle_rain").lower()
    n = int(duration * PROC_SR)
    rng = np.random.default_rng(abs(hash(key)) % (2**31))
    pink = _pink_noise(n, rng)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)

    if "ocean" in key or "wave" in key or "sea" in key:
        # Dual-band rolling ocean surf + deep water swell
        deep_sos = signal.butter(2, [55.0 / (0.5 * PROC_SR), 320.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        foam_sos = signal.butter(2, [350.0 / (0.5 * PROC_SR), 2200.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        swell_1 = 0.55 + 0.45 * np.sin(2.0 * math.pi * (1.0 / max(duration, 6.0)) * t)
        swell_2 = 0.40 + 0.60 * (0.5 + 0.5 * np.sin(2.0 * math.pi * (2.0 / max(duration, 6.0)) * t - 0.8)) ** 1.6
        out = 0.65 * signal.sosfilt(deep_sos, pink) * swell_1 + 0.35 * signal.sosfilt(foam_sos, pink) * swell_2
        out = out.astype(np.float32)
    elif "rain" in key or "storm" in key:
        sos = signal.butter(2, [450.0 / (0.5 * PROC_SR), 4800.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        out = signal.sosfilt(sos, pink).astype(np.float32)
        if "storm" in key or "thunder" in key:
            low_sos = signal.butter(2, 160.0 / (0.5 * PROC_SR), btype="lowpass", output="sos")
            out = 0.68 * out + 0.5 * signal.sosfilt(low_sos, pink).astype(np.float32)
    elif "wind" in key or "forest" in key:
        sos = signal.butter(2, [130.0 / (0.5 * PROC_SR), 1100.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        swell = 0.55 + 0.45 * np.sin(2.0 * math.pi * 0.20 * t)
        out = (signal.sosfilt(sos, pink) * swell).astype(np.float32)
    elif "fire" in key or "hearth" in key:
        low_sos = signal.butter(2, 240.0 / (0.5 * PROC_SR), btype="lowpass", output="sos")
        crackle = (rng.random(n) > 0.9985).astype(np.float32) * rng.uniform(0.2, 0.7, n).astype(np.float32)
        hi_sos = signal.butter(2, 1800.0 / (0.5 * PROC_SR), btype="highpass", output="sos")
        out = 0.62 * signal.sosfilt(low_sos, pink).astype(np.float32) + 0.38 * signal.sosfilt(hi_sos, crackle).astype(np.float32)
    elif "cricket" in key or "night" in key:
        chirp_env = np.maximum(0.0, np.sin(2.0 * math.pi * 3.8 * t)) ** 8
        chirps = 0.22 * np.sin(2.0 * math.pi * 4350.0 * t) * chirp_env
        low_air = signal.sosfilt(signal.butter(2, 380.0 / (0.5 * PROC_SR), btype="lowpass", output="sos"), pink)
        out = (0.45 * low_air + chirps).astype(np.float32)
    else:
        sos = signal.butter(2, [110.0 / (0.5 * PROC_SR), 900.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        out = signal.sosfilt(sos, pink).astype(np.float32)

    peak = float(np.max(np.abs(out))) + 1e-8
    out = (out / peak * 0.65).astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, out, PROC_SR)
    return output_path


def synthesize_procedural_sfx(sfx_name: str, duration: float, output_path: str) -> str:
    """Synthesizes crisp spot foley sound effects."""
    key = (sfx_name or "chime").lower()
    dur = max(1.2, min(4.0, float(duration)))
    n = int(dur * PROC_SR)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(abs(hash(key)) % (2**31))
    pink = _pink_noise(n, rng)
    out = np.zeros(n, dtype=np.float32)

    if "thunder" in key:
        sos = signal.butter(3, 185.0 / (0.5 * PROC_SR), btype="lowpass", output="sos")
        env = np.exp(-t * 1.4) * (0.68 + 0.32 * np.sin(2.0 * math.pi * 3.0 * t))
        out = signal.sosfilt(sos, pink).astype(np.float32) * env
    elif "knock" in key:
        for hit_t in (0.08, 0.30, 0.52):
            idx = int(hit_t * PROC_SR)
            ln = min(int(0.12 * PROC_SR), n - idx)
            if ln > 0:
                ht = np.linspace(0.0, 0.12, ln, endpoint=False, dtype=np.float32)
                thud = np.sin(2.0 * math.pi * (180.0 * np.exp(-ht * 18.0)) * ht) * np.exp(-ht * 36.0)
                out[idx : idx + ln] += thud
    elif "slam" in key:
        env = np.exp(-t * 10.5)
        sub = np.sin(2.0 * math.pi * 74.0 * t) * np.exp(-t * 13.0)
        sos = signal.butter(2, 560.0 / (0.5 * PROC_SR), btype="lowpass", output="sos")
        out = 0.72 * sub + 0.55 * signal.sosfilt(sos, pink).astype(np.float32) * env
    elif "creak" in key:
        creak_len = min(n, int(1.25 * PROC_SR))
        ct = t[:creak_len]
        f_mod = 270.0 + 68.0 * np.sin(2.0 * math.pi * 2.3 * ct)
        stick_slip = signal.sawtooth(2.0 * math.pi * f_mod * ct) * np.hanning(creak_len)
        sos = signal.butter(2, [220.0 / (0.5 * PROC_SR), 1250.0 / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        out[:creak_len] = signal.sosfilt(sos, stick_slip).astype(np.float32)
    elif "wind" in key or "gust" in key or "sigh" in key or "gasp" in key:
        cutoff_hi = 1400.0 if "wind" in key else 2200.0
        sos = signal.butter(2, [180.0 / (0.5 * PROC_SR), cutoff_hi / (0.5 * PROC_SR)], btype="bandpass", output="sos")
        swell = np.sin(math.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.8
        out = signal.sosfilt(sos, pink).astype(np.float32) * swell
    elif "heart" in key:
        for beat in (0.08, 0.30, 0.94, 1.16):
            if beat + 0.16 < dur:
                idx = int(beat * PROC_SR)
                ln = int(0.15 * PROC_SR)
                ht = np.linspace(0.0, 0.15, ln, endpoint=False, dtype=np.float32)
                lub = np.sin(2.0 * math.pi * 54.0 * ht) * np.exp(-ht * 22.0)
                out[idx : idx + ln] += lub
    elif "sword" in key or "clash" in key or "glass" in key:
        freqs = [1380.0, 2320.0, 3680.0, 4980.0] if "sword" in key else [2100.0, 3450.0, 4890.0, 6300.0]
        for f in freqs:
            out += 0.25 * np.sin(2.0 * math.pi * f * t) * np.exp(-t * (4.8 if "sword" in key else 12.0))
        out += 0.25 * pink * np.exp(-t * 18.0)
    else:
        # Ethereal crystal harmonic bell chord (magic_chime)
        for f, dec, amp in ((523.25, 2.0, 0.45), (659.25, 2.4, 0.32), (783.99, 2.8, 0.28), (1046.5, 3.3, 0.20)):
            out += amp * np.sin(2.0 * math.pi * f * t) * np.exp(-t * dec)
        out = apply_velvet_hall_reverb(out, mix=0.35, sr=PROC_SR)

    peak = float(np.max(np.abs(out))) + 1e-8
    out = (out / peak * 0.80).astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, out, PROC_SR)
    return output_path
