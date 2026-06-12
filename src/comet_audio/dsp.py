from __future__ import annotations

import math

import numpy as np
from scipy import signal

TAU = math.tau


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def midi_to_hz(note: int) -> float:
    return float(440.0 * (2.0 ** ((note - 69) / 12.0)))


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 1e-9:
        return audio
    if current <= peak:
        return audio
    return audio * (peak / current)


def equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * math.pi / 4.0
    left = math.cos(angle)
    right = math.sin(angle)
    return np.column_stack((mono * left, mono * right)).astype(np.float32)


def adsr_envelope(
    length: int,
    sample_rate: int,
    attack: float,
    release: float,
    decay: float = 0.03,
    sustain: float = 0.7,
) -> np.ndarray:
    env = np.ones(length, dtype=np.float32) * sustain
    attack_n = max(1, min(length, int(round(attack * sample_rate))))
    release_n = max(1, min(length, int(round(release * sample_rate))))
    decay_n = max(1, min(length, int(round(decay * sample_rate))))

    env[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False, dtype=np.float32)
    decay_end = min(length, attack_n + decay_n)
    if decay_end > attack_n:
        env[attack_n:decay_end] = np.linspace(
            1.0, sustain, decay_end - attack_n, endpoint=False, dtype=np.float32
        )
    env[-release_n:] *= np.linspace(1.0, 0.0, release_n, endpoint=True, dtype=np.float32)
    return env


def one_pole_decay(length: int, sample_rate: int, decay_seconds: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / sample_rate
    return np.exp(-t / max(decay_seconds, 1e-4)).astype(np.float32)


def butter_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    kind: str,
    order: int = 2,
) -> np.ndarray:
    nyquist = sample_rate * 0.5
    cutoff = float(np.clip(cutoff_hz / nyquist, 0.001, 0.999))
    sos = signal.butter(order, cutoff, btype=kind, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def saturate(audio: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(audio * drive).astype(np.float32) / math.tanh(drive)


def delay(
    audio: np.ndarray, sample_rate: int, delay_seconds: float, feedback: float, mix: float
) -> np.ndarray:
    delay_n = max(1, int(round(delay_seconds * sample_rate)))
    out = audio.copy().astype(np.float32)
    wet = np.zeros_like(out)
    for idx in range(delay_n, len(out)):
        wet[idx] = out[idx - delay_n] + wet[idx - delay_n] * feedback
    return (out * (1.0 - mix) + wet * mix).astype(np.float32)


def simple_reverb(audio: np.ndarray, sample_rate: int, room_size: float, mix: float) -> np.ndarray:
    delays = np.array([0.0297, 0.0371, 0.0411, 0.0437], dtype=np.float32)
    gains = np.array([0.52, 0.45, 0.38, 0.32], dtype=np.float32) * room_size
    wet = np.zeros_like(audio, dtype=np.float32)
    for delay_seconds, gain in zip(delays, gains, strict=True):
        delay_n = int(round(float(delay_seconds) * sample_rate))
        if delay_n < len(audio):
            wet[delay_n:] += audio[:-delay_n] * float(gain)
    wet = butter_filter(wet, sample_rate, 7200.0, "lowpass", order=1)
    return (audio * (1.0 - mix) + wet * mix).astype(np.float32)


def soft_limiter(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    limited = np.tanh(audio * 1.4) / np.tanh(1.4)
    return normalize_peak(limited.astype(np.float32), ceiling)


def sidechain_duck(
    audio: np.ndarray,
    kick_onsets: list[float],
    sample_rate: int,
    amount: float,
    release_seconds: float,
) -> np.ndarray:
    if not kick_onsets or amount <= 0:
        return audio
    gain = np.ones(len(audio), dtype=np.float32)
    release_n = max(1, int(round(release_seconds * sample_rate)))
    curve = 1.0 - amount * np.exp(-np.arange(release_n, dtype=np.float32) / (release_n * 0.32))
    curve = np.clip(curve, 1.0 - amount, 1.0)
    for onset in kick_onsets:
        start = int(round(onset * sample_rate))
        end = min(len(gain), start + release_n)
        if 0 <= start < len(gain):
            gain[start:end] = np.minimum(gain[start:end], curve[: end - start])
    if audio.ndim == 1:
        return (audio * gain).astype(np.float32)
    return (audio * gain[:, None]).astype(np.float32)
