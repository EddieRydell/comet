from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from comet_audio.generator import SOURCE_TYPES
from comet_audio.models import BatchManifestEntry, ClipMetadata

SAMPLE_RATE = 44_100
N_FFT = 1024
HOP_LENGTH = 128
N_MELS = 128
TRAIN_CROP_SECONDS = 4.0
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 16
SOURCE_TYPE_TO_INDEX = {source_type: index for index, source_type in enumerate(SOURCE_TYPES)}
TrainingTarget = Literal["source_types_v1", "anonymous_slots_v1"]
SLOT_PHASE_OFF = 0
SLOT_PHASE_ATTACK = 1
SLOT_PHASE_HELD = 2
SLOT_PHASE_RELEASE = 3
SLOT_PHASE_WEIGHTS = torch.tensor([0.08, 5.0, 0.9, 2.0], dtype=torch.float32)

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetItem:
    entry: BatchManifestEntry
    mix_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class TrainingSourceMetadata:
    source_id: str


@dataclass(frozen=True)
class TrainingEventMetadata:
    source_id: str
    onset_seconds: float
    offset_seconds: float
    attack_seconds: float
    release_seconds: float


@dataclass(frozen=True)
class TrainingClipMetadata:
    sample_rate: int
    duration_seconds: float
    sources: tuple[TrainingSourceMetadata, ...]
    events: tuple[TrainingEventMetadata, ...]


def load_manifest(data_dir: Path) -> list[DatasetItem]:
    manifest_path = data_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    items: list[DatasetItem] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = BatchManifestEntry.model_validate_json(line)
        items.append(
            DatasetItem(
                entry=entry,
                mix_path=data_dir / entry.mix_path,
                metadata_path=data_dir / entry.metadata_path,
            )
        )
    return items


def split_manifest(items: list[DatasetItem], split: SplitName) -> list[DatasetItem]:
    train_end = int(len(items) * 0.8)
    val_end = train_end + int(len(items) * 0.1)
    ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, len(items))}
    start, end = ranges[split]
    return items[start:end]


def load_metadata(path: Path) -> ClipMetadata:
    return ClipMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def load_anonymous_training_metadata(path: Path) -> TrainingClipMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainingClipMetadata(
        sample_rate=int(data["sample_rate"]),
        duration_seconds=float(data["duration_seconds"]),
        sources=tuple(
            TrainingSourceMetadata(source_id=str(source["source_id"])) for source in data["sources"]
        ),
        events=tuple(
            TrainingEventMetadata(
                source_id=str(event["source_id"]),
                onset_seconds=float(event["onset_seconds"]),
                offset_seconds=float(event["offset_seconds"]),
                attack_seconds=float(event["attack_seconds"]),
                release_seconds=float(event["release_seconds"]),
            )
            for event in data["events"]
        ),
    )


def load_mono_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> Tensor:
    audio, actual_sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if actual_sample_rate != sample_rate:
        raise ValueError(f"{path} has sample rate {actual_sample_rate}, expected {sample_rate}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def load_mono_audio_window(
    path: Path,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> Tensor:
    start = max(0, int(round(start_seconds * sample_rate)))
    length = int(round(duration_seconds * sample_rate))
    audio, actual_sample_rate = sf.read(
        path,
        start=start,
        frames=length,
        dtype="float32",
        always_2d=False,
    )
    if actual_sample_rate != sample_rate:
        raise ValueError(f"{path} has sample rate {actual_sample_rate}, expected {sample_rate}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if waveform.numel() < length:
        waveform = torch.nn.functional.pad(waveform, (0, length - waveform.numel()))
    return waveform


def crop_or_pad(waveform: Tensor, start_seconds: float, duration_seconds: float) -> Tensor:
    start = max(0, int(round(start_seconds * SAMPLE_RATE)))
    length = int(round(duration_seconds * SAMPLE_RATE))
    cropped = waveform[start : start + length]
    if cropped.numel() < length:
        cropped = torch.nn.functional.pad(cropped, (0, length - cropped.numel()))
    return cropped


def frame_count(num_samples: int, hop_length: int = HOP_LENGTH) -> int:
    return int(num_samples // hop_length + 1)


def build_targets(
    metadata: ClipMetadata,
    num_frames: int,
    crop_start_seconds: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    onset_sigma_seconds: float = 0.006,
    source_types: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Tensor]:
    source_types = tuple(source_types or SOURCE_TYPES)
    source_type_to_index = {source_type: index for index, source_type in enumerate(source_types)}
    frame_times = crop_start_seconds + torch.arange(num_frames, dtype=torch.float32) * (
        hop_length / sample_rate
    )
    onset = torch.zeros(num_frames, dtype=torch.float32)
    attack = torch.zeros(num_frames, dtype=torch.float32)
    held = torch.zeros(num_frames, dtype=torch.float32)
    release = torch.zeros(num_frames, dtype=torch.float32)
    source_onset = torch.zeros(len(source_types), num_frames, dtype=torch.float32)
    onset_offset = torch.zeros(num_frames, dtype=torch.float32)
    onset_offset_mask = torch.zeros(num_frames, dtype=torch.float32)

    event_source_types = {source.source_id: source.source_type for source in metadata.sources}
    unique_onsets = sorted({round(event.onset_seconds, 6) for event in metadata.events})
    for onset_seconds in unique_onsets:
        distance = torch.abs(frame_times - float(onset_seconds))
        gaussian = torch.exp(-0.5 * (distance / onset_sigma_seconds) ** 2)
        onset = torch.maximum(onset, gaussian)
        near = distance <= hop_length / sample_rate
        if bool(near.any()):
            onset_offset[near] = float(onset_seconds) - frame_times[near]
            onset_offset_mask[near] = 1.0

    for event in metadata.events:
        source_type = event_source_types[event.source_id]
        if source_type not in source_type_to_index:
            continue
        source_index = source_type_to_index[source_type]
        event_onset = float(event.onset_seconds)
        event_offset = float(event.offset_seconds)
        attack_end = min(event_offset, event_onset + float(event.attack_seconds))
        release_start = max(attack_end, event_offset - float(event.release_seconds))

        event_distance = torch.abs(frame_times - event_onset)
        source_onset[source_index] = torch.maximum(
            source_onset[source_index],
            torch.exp(-0.5 * (event_distance / onset_sigma_seconds) ** 2),
        )
        attack = torch.maximum(
            attack, ((frame_times >= event_onset) & (frame_times < attack_end)).float()
        )
        held = torch.maximum(
            held, ((frame_times >= attack_end) & (frame_times < release_start)).float()
        )
        release = torch.maximum(
            release, ((frame_times >= release_start) & (frame_times <= event_offset)).float()
        )

    return {
        "onset": onset.clamp(0.0, 1.0),
        "attack": attack.clamp(0.0, 1.0),
        "held": held.clamp(0.0, 1.0),
        "release": release.clamp(0.0, 1.0),
        "source_onset": source_onset.clamp(0.0, 1.0),
        "onset_offset": onset_offset,
        "onset_offset_mask": onset_offset_mask,
    }


class CometTimingDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        data_dir: Path,
        split: SplitName,
        training: bool,
        limit: int | None = None,
        crop_seconds: float = TRAIN_CROP_SECONDS,
        target: TrainingTarget = "source_types_v1",
        max_tracks: int = 16,
    ) -> None:
        items = split_manifest(load_manifest(data_dir), split)
        self.items = items[:limit] if limit is not None else items
        self.training = training
        self.crop_seconds = crop_seconds
        self.target = target
        self.max_tracks = max_tracks
        self._metadata_cache: dict[Path, ClipMetadata | TrainingClipMetadata] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.items[index]
        metadata = self._load_metadata(item.metadata_path)
        duration = float(metadata.duration_seconds)
        crop_duration = self.crop_seconds
        max_start = max(0.0, duration - crop_duration)
        crop_start = random.uniform(0.0, max_start) if self.training and max_start > 0 else 0.0
        waveform = load_mono_audio_window(
            item.mix_path,
            crop_start,
            crop_duration,
            sample_rate=metadata.sample_rate,
        )
        if self.target == "anonymous_slots_v1":
            targets = build_anonymous_slot_phase_targets(
                metadata,
                num_frames=frame_count(waveform.numel()),
                max_tracks=self.max_tracks,
                crop_start_seconds=crop_start,
                sample_rate=metadata.sample_rate,
            )
        else:
            targets = build_targets(
                metadata,
                num_frames=frame_count(waveform.numel()),
                crop_start_seconds=crop_start,
                sample_rate=metadata.sample_rate,
            )
        return {"waveform": waveform, **targets}

    def _load_metadata(self, path: Path) -> ClipMetadata | TrainingClipMetadata:
        metadata = self._metadata_cache.get(path)
        if metadata is None:
            if self.target == "anonymous_slots_v1":
                metadata = load_anonymous_training_metadata(path)
            else:
                metadata = load_metadata(path)
            self._metadata_cache[path] = metadata
        return metadata


def build_anonymous_slot_targets(
    metadata: ClipMetadata | TrainingClipMetadata,
    num_frames: int,
    max_tracks: int = 16,
    crop_start_seconds: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    mark_sigma_seconds: float = 0.006,
) -> dict[str, Tensor]:
    slot_attack = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_held = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_release = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_phase = torch.zeros(max_tracks, num_frames, dtype=torch.long)
    slot_mask = torch.zeros(max_tracks, dtype=torch.float32)
    sources = metadata.sources[:max_tracks]
    source_to_slot = {source.source_id: index for index, source in enumerate(sources)}
    for index in range(len(sources)):
        slot_mask[index] = 1.0
    for event in metadata.events:
        if event.source_id not in source_to_slot:
            continue
        slot_index = source_to_slot[event.source_id]
        onset = float(event.onset_seconds)
        offset = float(event.offset_seconds)
        attack_end = min(offset, onset + float(event.attack_seconds))
        release_start = max(attack_end, offset - float(event.release_seconds))
        held_start, held_stop = _frame_slice(
            attack_end, release_start, crop_start_seconds, num_frames, sample_rate, hop_length
        )
        release_start_frame, release_stop = _frame_slice(
            release_start,
            offset,
            crop_start_seconds,
            num_frames,
            sample_rate,
            hop_length,
            include_stop=True,
        )
        attack_start, attack_stop = _frame_slice(
            onset, attack_end, crop_start_seconds, num_frames, sample_rate, hop_length
        )
        if held_start < held_stop:
            slot_held[slot_index, held_start:held_stop] = 1.0
            slot_phase[slot_index, held_start:held_stop] = SLOT_PHASE_HELD
        if release_start_frame < release_stop:
            slot_release[slot_index, release_start_frame:release_stop] = 1.0
            slot_phase[slot_index, release_start_frame:release_stop] = SLOT_PHASE_RELEASE
        if attack_start < attack_stop:
            slot_attack[slot_index, attack_start:attack_stop] = 1.0
            slot_phase[slot_index, attack_start:attack_stop] = SLOT_PHASE_ATTACK
    slot_activity = (slot_attack + slot_held + slot_release).amax(dim=-1).clamp(0.0, 1.0)
    return {
        "slot_attack": slot_attack.clamp(0.0, 1.0),
        "slot_held": slot_held.clamp(0.0, 1.0),
        "slot_release": slot_release.clamp(0.0, 1.0),
        "slot_phase": slot_phase,
        "slot_mask": slot_mask,
        "slot_activity": slot_activity,
    }


def build_anonymous_slot_phase_targets(
    metadata: ClipMetadata | TrainingClipMetadata,
    num_frames: int,
    max_tracks: int = 16,
    crop_start_seconds: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> dict[str, Tensor]:
    slot_phase = torch.zeros(max_tracks, num_frames, dtype=torch.long)
    sources = metadata.sources[:max_tracks]
    source_to_slot = {source.source_id: index for index, source in enumerate(sources)}
    for event in metadata.events:
        if event.source_id not in source_to_slot:
            continue
        slot_index = source_to_slot[event.source_id]
        onset = float(event.onset_seconds)
        offset = float(event.offset_seconds)
        attack_end = min(offset, onset + float(event.attack_seconds))
        release_start = max(attack_end, offset - float(event.release_seconds))
        held_start, held_stop = _frame_slice(
            attack_end, release_start, crop_start_seconds, num_frames, sample_rate, hop_length
        )
        release_start_frame, release_stop = _frame_slice(
            release_start,
            offset,
            crop_start_seconds,
            num_frames,
            sample_rate,
            hop_length,
            include_stop=True,
        )
        attack_start, attack_stop = _frame_slice(
            onset, attack_end, crop_start_seconds, num_frames, sample_rate, hop_length
        )
        if held_start < held_stop:
            slot_phase[slot_index, held_start:held_stop] = SLOT_PHASE_HELD
        if release_start_frame < release_stop:
            slot_phase[slot_index, release_start_frame:release_stop] = SLOT_PHASE_RELEASE
        if attack_start < attack_stop:
            slot_phase[slot_index, attack_start:attack_stop] = SLOT_PHASE_ATTACK
    return {
        "slot_phase": slot_phase,
        "slot_activity": (slot_phase != SLOT_PHASE_OFF).any(dim=-1).float(),
    }


def _frame_slice(
    start_seconds: float,
    stop_seconds: float,
    crop_start_seconds: float,
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    include_stop: bool = False,
) -> tuple[int, int]:
    frame_seconds = hop_length / sample_rate
    start = math.ceil((start_seconds - crop_start_seconds) / frame_seconds)
    if include_stop:
        stop = math.floor((stop_seconds - crop_start_seconds) / frame_seconds) + 1
    else:
        stop = math.ceil((stop_seconds - crop_start_seconds) / frame_seconds)
    return max(0, start), min(num_frames, max(0, stop))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.05) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.net(x)
        residual = residual[..., : x.shape[-1]]
        return self.activation(x + residual)


class CNNTCNTimingModel(nn.Module):
    def __init__(self, source_type_count: int = len(SOURCE_TYPES), channels: int = 96) -> None:
        super().__init__()
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.tcn = nn.Sequential(*[ResidualTCNBlock(channels, dilation=2**idx) for idx in range(7)])
        self.global_head = nn.Conv1d(channels, 4, kernel_size=1)
        self.source_head = nn.Conv1d(channels, source_type_count, kernel_size=1)
        self.offset_head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        log_mel = torch.log1p(mel * 10.0).unsqueeze(1)
        features = self.cnn(log_mel).mean(dim=2)
        features = self.tcn(features)
        global_logits = self.global_head(features)
        return {
            "onset": global_logits[:, 0],
            "attack": global_logits[:, 1],
            "held": global_logits[:, 2],
            "release": global_logits[:, 3],
            "source_onset": self.source_head(features),
            "onset_offset": self.offset_head(features).squeeze(1),
        }


class CNNTCNSlotModel(nn.Module):
    def __init__(self, max_tracks: int = 16, channels: int = 96) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.tcn = nn.Sequential(*[ResidualTCNBlock(channels, dilation=2**idx) for idx in range(7)])
        self.slot_head = nn.Conv1d(channels, max_tracks * 4, kernel_size=1)
        self.slot_activity_head = nn.Linear(channels, max_tracks)

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        log_mel = torch.log1p(mel * 10.0).unsqueeze(1)
        features = self.cnn(log_mel).mean(dim=2)
        features = self.tcn(features)
        logits = self.slot_head(features)
        batch, _, time = logits.shape
        slots = logits.reshape(batch, self.max_tracks, 4, time)
        slot_activity = self.slot_activity_head(features.mean(dim=-1))
        return {
            "slot_phase": slots,
            "slot_activity": slot_activity,
            "slot_attack": slots[:, :, SLOT_PHASE_ATTACK],
            "slot_held": slots[:, :, SLOT_PHASE_HELD],
            "slot_release": slots[:, :, SLOT_PHASE_RELEASE],
        }


def _align_time(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    pred_time = predictions["onset"].shape[-1]
    target_time = targets["onset"].shape[-1]
    time = min(pred_time, target_time)
    aligned_predictions = {
        key: value[..., :time] if value.ndim >= 2 else value[:time]
        for key, value in predictions.items()
    }
    aligned_targets = {
        key: value[..., :time] if value.ndim >= 2 else value[:time]
        for key, value in targets.items()
    }
    return aligned_predictions, aligned_targets


def focal_bce_with_logits(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * targets + (1.0 - probability) * (1.0 - targets)
    return (bce * (1.0 - pt).pow(gamma)).mean()


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    gamma: float = 1.5,
) -> Tensor:
    ce = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=weights.to(logits.device),
        reduction="none",
    )
    probabilities = torch.softmax(logits, dim=1)
    pt = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    return (ce * (1.0 - pt).pow(gamma)).mean()


def compute_loss(
    predictions: dict[str, Tensor],
    batch: dict[str, Tensor],
    detach_metrics: bool = True,
) -> tuple[Tensor, dict[str, float] | dict[str, Tensor]]:
    predictions, targets = _align_time(predictions, batch)
    onset_loss = focal_bce_with_logits(predictions["onset"], targets["onset"])
    envelope_loss = (
        sum(
            torch.nn.functional.binary_cross_entropy_with_logits(predictions[name], targets[name])
            for name in ("attack", "held", "release")
        )
        / 3.0
    )
    source_loss = focal_bce_with_logits(predictions["source_onset"], targets["source_onset"])
    mask = targets["onset_offset_mask"] > 0
    if bool(mask.any()):
        offset_loss = torch.nn.functional.smooth_l1_loss(
            predictions["onset_offset"][mask], targets["onset_offset"][mask]
        )
    else:
        offset_loss = predictions["onset_offset"].sum() * 0.0
    total = onset_loss + 0.5 * envelope_loss + 0.5 * offset_loss + 0.15 * source_loss
    metrics = {
        "loss": total.detach(),
        "onset_loss": onset_loss.detach(),
        "envelope_loss": envelope_loss.detach(),
        "offset_loss": offset_loss.detach(),
        "source_loss": source_loss.detach(),
    }
    if detach_metrics:
        return total, {key: float(value.cpu()) for key, value in metrics.items()}
    return total, metrics


def compute_anonymous_slot_loss(
    predictions: dict[str, Tensor],
    batch: dict[str, Tensor],
    detach_metrics: bool = True,
) -> tuple[Tensor, dict[str, float] | dict[str, Tensor]]:
    time = min(predictions["slot_phase"].shape[-1], batch["slot_phase"].shape[-1])
    pred = {key: value[..., :time] for key, value in predictions.items()}
    target = {
        key: value[..., :time] if value.ndim >= 3 else value
        for key, value in batch.items()
        if key in {"slot_attack", "slot_held", "slot_release", "slot_phase", "slot_activity"}
    }
    assignments = _slot_assignments(pred, target)
    weights = SLOT_PHASE_WEIGHTS.to(pred["slot_phase"].device)
    matched_batch_indices: list[int] = []
    matched_pred_indices: list[int] = []
    matched_target_indices: list[int] = []
    unmatched_batch_indices: list[int] = []
    unmatched_pred_indices: list[int] = []
    activity_targets = torch.zeros_like(pred["slot_activity"])
    for batch_index, assignment in enumerate(assignments):
        batch_matched_pred_indices = [pair[0] for pair in assignment]
        for pred_index, target_index in assignment:
            matched_batch_indices.append(batch_index)
            matched_pred_indices.append(pred_index)
            matched_target_indices.append(target_index)
            activity_targets[batch_index, pred_index] = 1.0
        unmatched = [
            index
            for index in range(pred["slot_phase"].shape[1])
            if index not in set(batch_matched_pred_indices)
        ]
        unmatched_batch_indices.extend([batch_index] * len(unmatched))
        unmatched_pred_indices.extend(unmatched)
    zero = pred["slot_phase"].sum() * 0.0
    if matched_batch_indices:
        batch_indices = torch.as_tensor(matched_batch_indices, device=pred["slot_phase"].device)
        pred_indices = torch.as_tensor(matched_pred_indices, device=pred["slot_phase"].device)
        target_indices = torch.as_tensor(matched_target_indices, device=pred["slot_phase"].device)
        phase_loss = focal_cross_entropy(
            pred["slot_phase"][batch_indices, pred_indices],
            target["slot_phase"][batch_indices, target_indices].long(),
            weights=weights,
        )
    else:
        phase_loss = zero
    if unmatched_batch_indices:
        batch_indices = torch.as_tensor(unmatched_batch_indices, device=pred["slot_phase"].device)
        pred_indices = torch.as_tensor(unmatched_pred_indices, device=pred["slot_phase"].device)
        off_targets = torch.zeros(
            len(unmatched_pred_indices),
            time,
            dtype=torch.long,
            device=pred["slot_phase"].device,
        )
        unused_off_loss = focal_cross_entropy(
            pred["slot_phase"][batch_indices, pred_indices],
            off_targets,
            weights=weights,
        )
    else:
        unused_off_loss = zero
    activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["slot_activity"],
        activity_targets,
    )
    total = phase_loss + 0.5 * unused_off_loss + 0.2 * activity_loss
    metrics = {
        "loss": total.detach(),
        "slot_phase_loss": phase_loss.detach(),
        "slot_unused_off_loss": unused_off_loss.detach(),
        "slot_activity_loss": activity_loss.detach(),
    }
    if detach_metrics:
        return total, {key: float(value.cpu()) for key, value in metrics.items()}
    return total, metrics


def _slot_assignments(
    predictions: dict[str, Tensor], targets: dict[str, Tensor]
) -> list[list[tuple[int, int]]]:
    batch_size, slot_count, _classes, _time = predictions["slot_phase"].shape
    rows: list[list[tuple[int, int]]] = []
    with torch.no_grad():
        pred_phase_prob = torch.softmax(predictions["slot_phase"], dim=2).float().detach().cpu()
        target_phase = targets["slot_phase"].detach().cpu().long()
        target_activity = targets["slot_activity"].detach().cpu()
        weights = SLOT_PHASE_WEIGHTS.cpu()
        for batch_index in range(batch_size):
            active_targets = [
                index
                for index in range(slot_count)
                if float(target_activity[batch_index, index]) >= 0.5
            ]
            if not active_targets:
                rows.append([])
                continue
            labels = target_phase[batch_index, active_targets]
            probs = pred_phase_prob[batch_index]
            gathered = (
                probs[:, None]
                .expand(-1, len(active_targets), -1, -1)
                .gather(
                    dim=2,
                    index=labels[None, :, None, :].expand(slot_count, -1, 1, -1),
                )
            )
            phase_cost = -torch.log(gathered.squeeze(2).clamp_min(1e-6))
            phase_cost = (phase_cost * weights[labels][None]).mean(dim=-1)

            pred_active = 1.0 - probs[:, SLOT_PHASE_OFF]
            true_active = (labels != SLOT_PHASE_OFF).float()
            intersection = pred_active @ true_active.transpose(0, 1)
            dice = 1.0 - (2.0 * intersection + 1e-6) / (
                pred_active.sum(dim=-1, keepdim=True) + true_active.sum(dim=-1)[None] + 1e-6
            )
            cost = phase_cost + 0.2 * dice
            pred_indices, target_columns = linear_sum_assignment(cost.numpy())
            rows.append(
                [
                    (int(pred_index), int(active_targets[target_column]))
                    for pred_index, target_column in zip(
                        pred_indices.tolist(), target_columns.tolist(), strict=True
                    )
                ]
            )
    return rows


def train_model(
    data_dir: Path,
    run_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    loader_workers: int = 0,
    limit: int | None = None,
    learning_rate: float = 2e-4,
    target: TrainingTarget = "source_types_v1",
    max_tracks: int = 16,
    crop_seconds: float = TRAIN_CROP_SECONDS,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CometTimingDataset(
        data_dir,
        "train",
        training=True,
        limit=limit,
        crop_seconds=crop_seconds,
        target=target,
        max_tracks=max_tracks,
    )
    val_dataset = CometTimingDataset(
        data_dir,
        "val",
        training=False,
        limit=limit,
        crop_seconds=crop_seconds,
        target=target,
        max_tracks=max_tracks,
    )
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": loader_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    model: nn.Module
    if target == "anonymous_slots_v1":
        model = CNNTCNSlotModel(max_tracks=max_tracks).to(device)
    else:
        model = CNNTCNTimingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_val = math.inf
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, scaler, device, training=True, target=target
        )
        val_metrics = _run_epoch(
            model, val_loader, optimizer, scaler, device, training=False, target=target
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "target": target,
            "source_types": [] if target == "anonymous_slots_v1" else SOURCE_TYPES,
            "max_tracks": max_tracks if target == "anonymous_slots_v1" else None,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "n_mels": N_MELS,
                "crop_seconds": crop_seconds,
            },
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, run_dir / "best.pt")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    training: bool,
    target: TrainingTarget = "source_types_v1",
) -> dict[str, float]:
    model.train(training)
    totals: dict[str, Tensor] = {}
    count = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "val", mininterval=5.0)
    for batch in iterator:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with (
            torch.set_grad_enabled(training),
            torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ),
        ):
            predictions = model(batch["waveform"])
            if target == "anonymous_slots_v1":
                loss, metrics = compute_anonymous_slot_loss(
                    predictions, batch, detach_metrics=False
                )
            else:
                loss, metrics = compute_loss(predictions, batch, detach_metrics=False)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(batch["waveform"].shape[0])
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, value.new_zeros(())) + value * batch_size
        if count % (batch_size * 100) == 0:
            iterator.set_postfix(loss=float(metrics["loss"].detach().cpu()))
    if count == 0:
        return {"loss": math.inf}
    return {key: float((value / max(count, 1)).cpu()) for key, value in totals.items()}


def load_trained_model(run_dir: Path, device: torch.device) -> CNNTCNTimingModel:
    checkpoint = load_training_checkpoint(run_dir, device)
    source_types = checkpoint.get("source_types", SOURCE_TYPES)
    model = CNNTCNTimingModel(source_type_count=len(source_types)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_training_checkpoint(run_dir: Path, device: torch.device) -> dict[str, object]:
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "last.pt"
    return torch.load(checkpoint_path, map_location=device)


def load_trained_source_types(run_dir: Path) -> tuple[str, ...]:
    checkpoint = load_training_checkpoint(run_dir, torch.device("cpu"))
    source_types = checkpoint.get("source_types", SOURCE_TYPES)
    return tuple(str(source_type) for source_type in source_types)


def decode_onsets(
    onset_probability: Tensor,
    offset_seconds: Tensor | None = None,
    threshold: float = 0.35,
    nms_seconds: float = 0.025,
) -> list[float]:
    probs = onset_probability.detach().cpu()
    offsets = (
        offset_seconds.detach().cpu() if offset_seconds is not None else torch.zeros_like(probs)
    )
    min_distance = max(1, int(round(nms_seconds * SAMPLE_RATE / HOP_LENGTH)))
    candidates: list[tuple[float, int]] = []
    for index in range(1, max(1, probs.numel() - 1)):
        value = float(probs[index])
        if value < threshold:
            continue
        if value >= float(probs[index - 1]) and value >= float(probs[index + 1]):
            candidates.append((value, index))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, index in candidates:
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    selected.sort()
    return [
        max(0.0, index * HOP_LENGTH / SAMPLE_RATE + float(offsets[index])) for index in selected
    ]


def unique_onsets(metadata: ClipMetadata) -> list[float]:
    return sorted({round(float(event.onset_seconds), 6) for event in metadata.events})


def match_onsets(
    predicted: list[float],
    truth: list[float],
    tolerance: float,
) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    errors: list[float] = []
    true_positive = 0
    for pred in predicted:
        best_index = None
        best_error = math.inf
        for index, target in enumerate(truth):
            if index in used:
                continue
            error = abs(pred - target)
            if error <= tolerance and error < best_error:
                best_index = index
                best_error = error
        if best_index is None:
            continue
        used.add(best_index)
        true_positive += 1
        errors.append(best_error)
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    return true_positive, false_positive, false_negative, errors


def evaluate_model(
    data_dir: Path,
    run_dir: Path,
    split: SplitName,
    limit: int | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(run_dir, device)
    items = split_manifest(load_manifest(data_dir), split)
    if limit is not None:
        items = items[:limit]
    if threshold is None:
        threshold = (
            tune_threshold(model, data_dir, run_dir, limit=limit) if split == "test" else 0.35
        )

    all_errors: list[float] = []
    totals = {0.005: [0, 0, 0], 0.010: [0, 0, 0], 0.025: [0, 0, 0]}
    envelope_counts = {name: [0, 0, 0] for name in ("attack", "held", "release")}
    source_totals = [[0, 0, 0] for _ in SOURCE_TYPES]
    worst_rows: list[dict[str, float | int | str]] = []

    for item in tqdm(items, desc=f"eval-{split}"):
        metadata = load_metadata(item.metadata_path)
        waveform = crop_or_pad(
            load_mono_audio(item.mix_path, metadata.sample_rate),
            0.0,
            metadata.duration_seconds,
        )
        targets = build_targets(metadata, frame_count(waveform.numel()))
        with torch.no_grad():
            predictions = model(waveform.to(device).unsqueeze(0))
        onset_prob = torch.sigmoid(predictions["onset"][0]).cpu()
        offset_pred = predictions["onset_offset"][0].cpu()
        predicted_onsets = decode_onsets(onset_prob, offset_pred, threshold=threshold)
        truth = unique_onsets(metadata)
        clip_errors: list[float] = []
        for tolerance, counts in totals.items():
            tp, fp, fn, errors = match_onsets(predicted_onsets, truth, tolerance)
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
            if tolerance == 0.025:
                clip_errors = errors
                all_errors.extend(errors)
        aligned_predictions, aligned_targets = _align_time(
            {key: value[0].cpu() for key, value in predictions.items()}, targets
        )
        for name in envelope_counts:
            pred_mask = torch.sigmoid(aligned_predictions[name]) >= 0.5
            true_mask = aligned_targets[name] >= 0.5
            _accumulate_binary_counts(envelope_counts[name], pred_mask, true_mask)
        source_pred = torch.sigmoid(aligned_predictions["source_onset"]) >= threshold
        source_true = aligned_targets["source_onset"] >= 0.5
        for idx in range(len(SOURCE_TYPES)):
            _accumulate_binary_counts(source_totals[idx], source_pred[idx], source_true[idx])
        _, fp25, fn25, _ = match_onsets(predicted_onsets, truth, 0.025)
        worst_rows.append(
            {
                "clip_id": item.entry.clip_id,
                "median_abs_error_ms": (
                    float(np.median(clip_errors) * 1000.0) if clip_errors else 9999.0
                ),
                "false_positives_25ms": fp25,
                "false_negatives_25ms": fn25,
                "predicted_onsets": len(predicted_onsets),
                "true_onsets": len(truth),
            }
        )

    metrics: dict[str, float] = {"threshold": float(threshold), "clips": float(len(items))}
    for tolerance, counts in totals.items():
        metrics[f"onset_f1_{int(tolerance * 1000)}ms"] = _f1(*counts)
    metrics["median_abs_onset_error_ms"] = (
        float(np.median(all_errors) * 1000.0) if all_errors else 0.0
    )
    metrics["p90_abs_onset_error_ms"] = (
        float(np.percentile(all_errors, 90) * 1000.0) if all_errors else 0.0
    )
    for name, counts in envelope_counts.items():
        metrics[f"{name}_frame_f1"] = _f1(*counts)
    for source_type, counts in zip(SOURCE_TYPES, source_totals, strict=True):
        metrics[f"source_{source_type}_onset_f1"] = _f1(*counts)

    output_path = run_dir / f"eval_{split}.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_worst_clips(run_dir / f"eval_{split}_worst_clips.csv", worst_rows)
    return metrics


def tune_threshold(
    model: CNNTCNTimingModel,
    data_dir: Path,
    run_dir: Path,
    limit: int | None = None,
) -> float:
    candidates = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
    best_threshold = 0.35
    best_f1 = -1.0
    for threshold in candidates:
        metrics = evaluate_model(data_dir, run_dir, "val", limit=limit, threshold=threshold)
        f1 = metrics["onset_f1_25ms"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def _accumulate_binary_counts(counts: list[int], predicted: Tensor, truth: Tensor) -> None:
    counts[0] += int((predicted & truth).sum().item())
    counts[1] += int((predicted & ~truth).sum().item())
    counts[2] += int((~predicted & truth).sum().item())


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 1.0


def write_worst_clips(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            float(row["median_abs_error_ms"]),
            int(row["false_positives_25ms"]) + int(row["false_negatives_25ms"]),
        ),
        reverse=True,
    )[:50]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
