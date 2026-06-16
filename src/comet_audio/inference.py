# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from comet_audio.training import (
    HOP_LENGTH,
    SAMPLE_RATE,
    SLOT_BOUNDARY_NAMES,
    SlotAttentionEventModel,
    decode_onsets,
    load_trained_model,
    load_trained_source_types,
    load_training_checkpoint,
)


def predict_song(
    audio_path: Path,
    run_dir: Path,
    out_dir: Path,
    threshold: float = 0.35,
    nms_seconds: float = 0.025,
    source_threshold: float = 0.35,
    max_waveform_points: int = 2000,
) -> Path:
    audio_path = audio_path.resolve()
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform, original_sample_rate = _load_audio_for_model(audio_path)
    duration = waveform.numel() / SAMPLE_RATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_types = load_trained_source_types(run_dir)
    model = load_trained_model(run_dir, device)
    with torch.no_grad():
        predictions = model(waveform.to(device).unsqueeze(0))

    onset_prob = torch.sigmoid(predictions["onset"][0]).cpu()
    offset_pred = predictions["onset_offset"][0].cpu()
    source_prob = torch.sigmoid(predictions["source_onset"][0]).cpu()
    decoded_onsets = decode_onsets(
        onset_prob,
        offset_pred,
        threshold=threshold,
        nms_seconds=nms_seconds,
    )
    marks = _build_marks(
        decoded_onsets,
        onset_prob,
        source_prob,
        source_types,
        source_threshold,
        duration,
    )
    viewer_audio = _write_viewer_audio(waveform, audio_path, out_dir)
    payload = {
        "audio": {
            "input_path": str(audio_path),
            "viewer_audio_path": viewer_audio.name,
            "original_sample_rate": original_sample_rate,
            "model_sample_rate": SAMPLE_RATE,
            "duration_seconds": duration,
            "samples": int(waveform.numel()),
        },
        "model": {
            "run_dir": str(run_dir),
            "threshold": threshold,
            "nms_seconds": nms_seconds,
            "source_threshold": source_threshold,
            "hop_length": HOP_LENGTH,
            "frame_seconds": HOP_LENGTH / SAMPLE_RATE,
            "source_types": source_types,
        },
        "waveform": _waveform_peaks(waveform.numpy(), max_waveform_points),
        "marks": marks,
    }
    json_path = out_dir / "predictions.json"
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json_path


def predict_anonymous_slots(
    audio_path: Path,
    run_dir: Path,
    out_dir: Path,
    max_waveform_points: int = 2000,
    min_segment_seconds: float = 0.02,
) -> Path:
    audio_path = audio_path.resolve()
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform, original_sample_rate = _load_audio_for_model(audio_path)
    duration = waveform.numel() / SAMPLE_RATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_training_checkpoint(run_dir, device)
    if checkpoint.get("target") != "anonymous_slots_v1":
        raise ValueError(f"{run_dir} is not an anonymous_slots_v1 checkpoint")
    architecture = checkpoint.get("architecture")
    if architecture != "slot_attention_event_v1":
        raise ValueError(
            f"{run_dir} uses architecture {architecture!r}; expected 'slot_attention_event_v1'"
        )
    max_tracks = int(checkpoint.get("max_tracks") or 16)
    model = SlotAttentionEventModel(max_tracks=max_tracks).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with (
        torch.no_grad(),
        torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16),
    ):
        predictions = model(waveform.to(device).unsqueeze(0))

    probabilities = {
        name: torch.sigmoid(predictions[name][0]).cpu()
        for name in ("slot_active", *SLOT_BOUNDARY_NAMES)
    }
    activity = torch.sigmoid(predictions["slot_activity"][0]).cpu()
    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    min_frames = max(0, int(round(min_segment_seconds / frame_seconds)))
    slots = [
        _decoded_slot_payload(
            slot_index,
            probabilities,
            activity,
            frame_seconds,
            duration,
            min_frames,
        )
        for slot_index in range(max_tracks)
    ]
    viewer_audio = _write_viewer_audio(waveform, audio_path, out_dir)
    payload = {
        "audio": {
            "input_path": str(audio_path),
            "viewer_audio_path": viewer_audio.name,
            "original_sample_rate": original_sample_rate,
            "model_sample_rate": SAMPLE_RATE,
            "duration_seconds": duration,
            "samples": int(waveform.numel()),
        },
        "model": {
            "run_dir": str(run_dir),
            "target": "anonymous_slots_v1",
            "architecture": "slot_attention_event_v1",
            "max_tracks": max_tracks,
            "hop_length": HOP_LENGTH,
            "frame_seconds": frame_seconds,
            "checkpoint_epoch": checkpoint.get("epoch"),
        },
        "waveform": _waveform_peaks(waveform.numpy(), max_waveform_points),
        "slots": slots,
    }
    json_path = out_dir / "predictions.json"
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json_path


def _decoded_slot_payload(
    slot_index: int,
    probabilities: dict[str, torch.Tensor],
    activity: torch.Tensor,
    frame_seconds: float,
    duration: float,
    min_frames: int,
) -> dict[str, Any]:
    slot_probabilities = {
        name: probabilities[name][slot_index] for name in ("slot_active", *SLOT_BOUNDARY_NAMES)
    }
    events = decode_slot_events(slot_probabilities, frame_seconds, duration, min_frames=min_frames)
    return {
        "slot": f"track_{slot_index:02d}",
        "activity_probability": float(activity[slot_index]),
        "raw_event_count": len(events),
        "boundary_probabilities": {
            name.replace("slot_", ""): [float(value) for value in slot_probabilities[name]]
            for name in SLOT_BOUNDARY_NAMES
        },
        "events": events,
        "segments": _events_to_segments(events),
    }


def decode_slot_events(
    probabilities: dict[str, torch.Tensor],
    frame_seconds: float,
    duration: float,
    threshold: float = 0.35,
    min_frames: int = 1,
) -> list[dict[str, Any]]:
    onset_peaks = _boundary_peaks(probabilities["slot_onset"], threshold=threshold)
    offset_peaks = _boundary_peaks(probabilities["slot_offset"], threshold=threshold)
    attack_peaks = _boundary_peaks(probabilities["slot_attack_end"], threshold=0.25)
    release_peaks = _boundary_peaks(probabilities["slot_release_start"], threshold=0.25)
    used_offsets: set[int] = set()
    events: list[dict[str, Any]] = []
    for onset in onset_peaks:
        offset = next(
            (peak for peak in offset_peaks if peak > onset and peak not in used_offsets),
            None,
        )
        if offset is None or offset - onset < min_frames:
            continue
        used_offsets.add(offset)
        attack_end = _best_inner_peak(attack_peaks, probabilities["slot_attack_end"], onset, offset)
        release_start = _best_inner_peak(
            release_peaks,
            probabilities["slot_release_start"],
            attack_end,
            offset,
        )
        attack_end = max(onset, min(attack_end, offset))
        release_start = max(attack_end, min(release_start, offset))
        start_seconds = max(0.0, min(duration, onset * frame_seconds))
        end_seconds = max(start_seconds, min(duration, offset * frame_seconds))
        events.append(
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "onset_seconds": start_seconds,
                "attack_end_seconds": max(
                    start_seconds,
                    min(duration, attack_end * frame_seconds),
                ),
                "release_start_seconds": max(
                    start_seconds,
                    min(duration, release_start * frame_seconds),
                ),
                "offset_seconds": end_seconds,
                "onset_probability": float(probabilities["slot_onset"][onset]),
                "attack_end_probability": float(probabilities["slot_attack_end"][attack_end]),
                "release_start_probability": float(
                    probabilities["slot_release_start"][release_start]
                ),
                "offset_probability": float(probabilities["slot_offset"][offset]),
                "active_probability": float(
                    probabilities["slot_active"][onset : offset + 1].mean()
                ),
            }
        )
    return events


def _boundary_peaks(
    probability: torch.Tensor, threshold: float, min_distance: int = 2
) -> list[int]:
    if probability.numel() == 0:
        return []
    candidates: list[tuple[float, int]] = []
    for index in range(int(probability.numel())):
        value = float(probability[index])
        if value < threshold:
            continue
        left = float(probability[index - 1]) if index > 0 else -math.inf
        right = float(probability[index + 1]) if index + 1 < probability.numel() else -math.inf
        if value >= left and value >= right:
            candidates.append((value, index))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _value, index in candidates:
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return sorted(selected)


def _best_inner_peak(peaks: list[int], probability: torch.Tensor, start: int, stop: int) -> int:
    inner = [peak for peak in peaks if start <= peak <= stop]
    if inner:
        return max(inner, key=lambda peak: float(probability[peak]))
    if stop <= start:
        return start
    search = probability[start : stop + 1]
    return start + int(torch.argmax(search).item())


def _events_to_segments(events: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []
    for event in events:
        spans = (
            (
                "attack",
                event["onset_seconds"],
                event["attack_end_seconds"],
                event["onset_probability"],
            ),
            (
                "held",
                event["attack_end_seconds"],
                event["release_start_seconds"],
                event["active_probability"],
            ),
            (
                "release",
                event["release_start_seconds"],
                event["offset_seconds"],
                event["offset_probability"],
            ),
        )
        for phase, start, end, probability in spans:
            if end <= start:
                continue
            segments.append(
                {
                    "phase": phase,
                    "start_seconds": float(start),
                    "end_seconds": float(end),
                    "probability": float(probability),
                }
            )
    return segments


def _load_audio_for_model(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    mono = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise RuntimeError(f"Audio contains non-finite samples: {path}")
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(int(sample_rate), SAMPLE_RATE)
        mono = signal.resample_poly(mono, SAMPLE_RATE // gcd, int(sample_rate) // gcd).astype(
            np.float32
        )
    return torch.from_numpy(mono), int(sample_rate)


def _build_marks(
    decoded_onsets: list[float],
    onset_prob: torch.Tensor,
    source_prob: torch.Tensor,
    source_types: tuple[str, ...],
    source_threshold: float,
    duration: float,
) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for index, onset_seconds in enumerate(decoded_onsets):
        frame = int(round(onset_seconds * SAMPLE_RATE / HOP_LENGTH))
        frame = max(0, min(frame, onset_prob.numel() - 1))
        source_scores = [
            {
                "source_type": source_type,
                "probability": float(source_prob[source_index, frame]),
            }
            for source_index, source_type in enumerate(source_types)
        ]
        source_scores.sort(key=lambda row: row["probability"], reverse=True)
        active_sources = [
            row for row in source_scores if row["probability"] >= source_threshold
        ] or source_scores[:1]
        marks.append(
            {
                "index": index,
                "time_seconds": max(0.0, min(float(onset_seconds), duration)),
                "onset_probability": float(onset_prob[frame]),
                "primary_source": active_sources[0]["source_type"],
                "primary_source_probability": active_sources[0]["probability"],
                "sources": active_sources[:4],
            }
        )
    return marks


def _write_viewer_audio(waveform: torch.Tensor, audio_path: Path, out_dir: Path) -> Path:
    target = out_dir / f"{audio_path.stem}.viewer.wav"
    sf.write(target, waveform.numpy(), SAMPLE_RATE, subtype="FLOAT")
    return target


def _waveform_peaks(waveform: np.ndarray, max_points: int) -> list[list[float]]:
    if waveform.size == 0:
        return []
    point_count = max(1, min(max_points, waveform.size))
    bucket_size = int(math.ceil(waveform.size / point_count))
    padded_length = bucket_size * point_count
    padded = np.pad(waveform, (0, padded_length - waveform.size))
    buckets = padded.reshape(point_count, bucket_size)
    return [[float(np.min(bucket)), float(np.max(bucket))] for bucket in buckets]
