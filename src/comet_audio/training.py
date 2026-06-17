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
SLOT_PHASE_LOSS_WEIGHT = 1.0
SLOT_ACTIVE_TVERSKY_LOSS_WEIGHT = 0.5
SLOT_BOUNDARY_LOSS_WEIGHT = 0.35
SLOT_EVENT_COUNT_LOSS_WEIGHT = 0.0
SLOT_PHASE_OVERLAP_LOSS_WEIGHT = 0.10
SLOT_UNMATCHED_OFF_LOSS_WEIGHT = 0.75
SLOT_DUPLICATE_LOSS_WEIGHT = 0.75
SLOT_ACTIVITY_LOSS_WEIGHT = 0.05
SLOT_MATCHED_OFF_LOSS_WEIGHT = 1.5
SLOT_ACTIVE_DURATION_LOSS_WEIGHT = 0.75
SLOT_BOUNDARY_MASS_LOSS_WEIGHT = 0.25
SLOT_TVERSKY_FALSE_POSITIVE_WEIGHT = 0.7
SLOT_TVERSKY_FALSE_NEGATIVE_WEIGHT = 0.3
SLOT_DUPLICATE_SIMILARITY_THRESHOLD = 0.6
ANONYMOUS_SLOT_OBJECTIVE_VERSION = "phase_event_stable_v3"
SLOT_PHASE_NAMES = ("slot_attack", "slot_held", "slot_release")
SLOT_BOUNDARY_NAMES = ("slot_onset", "slot_attack_end", "slot_release_start", "slot_offset")
SLOT_BOUNDARY_WEIGHTS = {
    "slot_onset": 1.0,
    "slot_attack_end": 0.5,
    "slot_release_start": 1.25,
    "slot_offset": 1.5,
}

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
            targets = build_anonymous_slot_targets(
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
        return {
            "waveform": waveform,
            "clip_index": torch.tensor(index, dtype=torch.int64),
            **targets,
        }

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
    frame_times = crop_start_seconds + torch.arange(num_frames, dtype=torch.float32) * (
        hop_length / sample_rate
    )
    slot_attack = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_held = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_release = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_onset = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_attack_end = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_release_start = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
    slot_offset = torch.zeros(max_tracks, num_frames, dtype=torch.float32)
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
        for phase_target, phase_start, phase_stop in (
            (slot_attack, onset, attack_end),
            (slot_held, attack_end, release_start),
            (slot_release, release_start, offset),
        ):
            start_frame, stop_frame = _frame_slice(
                phase_start,
                phase_stop,
                crop_start_seconds,
                num_frames,
                sample_rate,
                hop_length,
            )
            if start_frame < stop_frame:
                phase_target[slot_index, start_frame:stop_frame] = 1.0
        for target, boundary_seconds in (
            (slot_onset, onset),
            (slot_attack_end, attack_end),
            (slot_release_start, release_start),
            (slot_offset, offset),
        ):
            distance = torch.abs(frame_times - boundary_seconds)
            gaussian = torch.exp(-0.5 * (distance / mark_sigma_seconds) ** 2)
            target[slot_index] = torch.maximum(target[slot_index], gaussian)
    slot_active = torch.stack((slot_attack, slot_held, slot_release), dim=0).amax(dim=0)
    slot_boundary_activity = torch.stack(
        (slot_onset, slot_attack_end, slot_release_start, slot_offset), dim=0
    ).amax(dim=0)
    return {
        "slot_attack": slot_attack.clamp(0.0, 1.0),
        "slot_held": slot_held.clamp(0.0, 1.0),
        "slot_release": slot_release.clamp(0.0, 1.0),
        "slot_active": slot_active.clamp(0.0, 1.0),
        "slot_onset": slot_onset.clamp(0.0, 1.0),
        "slot_attack_end": slot_attack_end.clamp(0.0, 1.0),
        "slot_release_start": slot_release_start.clamp(0.0, 1.0),
        "slot_offset": slot_offset.clamp(0.0, 1.0),
        "slot_mask": slot_mask,
        "slot_activity": torch.maximum(slot_active, slot_boundary_activity)
        .amax(dim=-1)
        .clamp(0.0, 1.0),
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


class SlotAttentionEventModel(nn.Module):
    def __init__(
        self,
        max_tracks: int = 16,
        channels: int = 96,
        slot_dim: int = 128,
        iterations: int = 3,
    ) -> None:
        super().__init__()
        self.max_tracks = max_tracks
        self.iterations = iterations
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
        self.temporal_encoder = nn.GRU(
            channels,
            slot_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.05,
        )
        self.feature_norm = nn.LayerNorm(slot_dim)
        self.slot_mu = nn.Parameter(torch.zeros(1, max_tracks, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.full((1, max_tracks, slot_dim), -1.0))
        self.query_norm = nn.LayerNorm(slot_dim)
        self.key = nn.Linear(slot_dim, slot_dim, bias=False)
        self.value = nn.Linear(slot_dim, slot_dim, bias=False)
        self.query = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.SiLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.slot_time_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 7),
        )
        self.slot_activity_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 1),
        )

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        log_mel = torch.log1p(mel * 10.0).unsqueeze(1)
        features = self.cnn(log_mel).mean(dim=2).transpose(1, 2).contiguous()
        features, _ = self.temporal_encoder(features)
        features = self.feature_norm(features)
        batch, time, dim = features.shape
        slots = self.slot_mu.expand(batch, -1, -1)
        if self.training:
            sigma = torch.exp(self.slot_log_sigma).expand(batch, -1, -1)
            slots = slots + sigma * torch.randn_like(slots)
        keys = self.key(features)
        values = self.value(features)
        scale = dim**-0.5
        for _ in range(self.iterations):
            queries = self.query(self.query_norm(slots))
            attention_logits = torch.einsum("bsd,btd->bst", queries, keys) * scale
            attention = torch.softmax(attention_logits, dim=1)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            updates = torch.einsum("bst,btd->bsd", attention, values)
            slots = self.gru(updates.reshape(-1, dim), slots.reshape(-1, dim)).reshape(
                batch,
                self.max_tracks,
                dim,
            )
            slots = slots + self.slot_mlp(slots)
        slot_time_features = self.feature_norm(features[:, None] + slots[:, :, None])
        logits = self.slot_time_head(slot_time_features).permute(0, 1, 3, 2).contiguous()
        phase_probability = torch.stack(
            [torch.sigmoid(logits[:, :, index]) for index in range(3)], dim=2
        ).amax(dim=2)
        return {
            "slot_attack": logits[:, :, 0],
            "slot_held": logits[:, :, 1],
            "slot_release": logits[:, :, 2],
            "slot_onset": logits[:, :, 3],
            "slot_attack_end": logits[:, :, 4],
            "slot_release_start": logits[:, :, 5],
            "slot_offset": logits[:, :, 6],
            "slot_active": phase_probability,
            "slot_activity": self.slot_activity_head(slots).squeeze(-1),
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
    time = min(predictions["slot_attack"].shape[-1], batch["slot_attack"].shape[-1])
    pred = {
        key: (value[..., :time] if value.ndim >= 3 else value).float()
        for key, value in predictions.items()
    }
    target = {
        key: (value[..., :time] if value.ndim >= 3 else value).float()
        for key, value in batch.items()
        if key
        in {
            "slot_attack",
            "slot_held",
            "slot_release",
            "slot_active",
            "slot_onset",
            "slot_attack_end",
            "slot_release_start",
            "slot_offset",
            "slot_activity",
        }
    }
    for name in (*SLOT_PHASE_NAMES, *SLOT_BOUNDARY_NAMES, "slot_activity"):
        _require_finite_tensor(f"predictions[{name!r}]", pred[name])
        _require_finite_tensor(f"batch[{name!r}]", target[name])
    target["slot_active"] = _target_phase_active(target)
    pred_active_probability = _phase_active_probability(pred)
    _require_finite_tensor("predicted slot phase-active probability", pred_active_probability)
    assignments = _slot_assignments(pred, target)
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
            for index in range(pred["slot_attack"].shape[1])
            if index not in set(batch_matched_pred_indices)
        ]
        unmatched_batch_indices.extend([batch_index] * len(unmatched))
        unmatched_pred_indices.extend(unmatched)
    zero = pred["slot_attack"].sum() * 0.0
    if matched_batch_indices:
        batch_indices = torch.as_tensor(matched_batch_indices, device=pred["slot_attack"].device)
        pred_indices = torch.as_tensor(matched_pred_indices, device=pred["slot_attack"].device)
        target_indices = torch.as_tensor(matched_target_indices, device=pred["slot_attack"].device)
        matched_active_targets = target["slot_active"][batch_indices, target_indices].float()
        slot_phase_loss = _slot_phase_loss(
            pred, target, batch_indices, pred_indices, target_indices
        )
        slot_active_tversky_loss = _active_tversky_loss(
            pred_active_probability[batch_indices, pred_indices],
            matched_active_targets,
        )
        slot_boundary_loss = _slot_boundary_loss(
            pred,
            target,
            batch_indices,
            pred_indices,
            target_indices,
        )
        event_count_loss = _slot_event_count_loss(
            pred,
            target,
            batch_indices,
            pred_indices,
            target_indices,
        )
        matched_off_loss = _matched_slot_off_loss(
            pred_active_probability[batch_indices, pred_indices],
            matched_active_targets,
        )
        active_duration_loss = _active_duration_ratio_loss(
            pred_active_probability[batch_indices, pred_indices],
            matched_active_targets,
        )
        boundary_mass_loss = _slot_boundary_mass_loss(
            pred,
            target,
            batch_indices,
            pred_indices,
            target_indices,
        )
    else:
        slot_phase_loss = zero
        slot_active_tversky_loss = zero
        slot_boundary_loss = zero
        event_count_loss = zero
        matched_off_loss = zero
        active_duration_loss = zero
        boundary_mass_loss = zero
    if unmatched_batch_indices:
        batch_indices = torch.as_tensor(unmatched_batch_indices, device=pred["slot_attack"].device)
        pred_indices = torch.as_tensor(unmatched_pred_indices, device=pred["slot_attack"].device)
        off_targets = torch.zeros(
            len(unmatched_pred_indices),
            time,
            dtype=pred["slot_attack"].dtype,
            device=pred["slot_attack"].device,
        )
        unmatched_phase_losses = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                pred[name][batch_indices, pred_indices],
                off_targets,
            )
            for name in SLOT_PHASE_NAMES
        ]
        unmatched_boundary_losses = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                pred[name][batch_indices, pred_indices],
                off_targets,
            )
            for name in SLOT_BOUNDARY_NAMES
        ]
        unmatched_off_loss = torch.stack(
            [*unmatched_phase_losses, *unmatched_boundary_losses]
        ).mean()
    else:
        unmatched_off_loss = zero
    slot_phase_overlap_loss = _slot_phase_overlap_loss(pred)
    duplicate_loss = _slot_duplicate_loss(pred_active_probability)
    activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["slot_activity"],
        activity_targets,
    )
    total = (
        SLOT_PHASE_LOSS_WEIGHT * slot_phase_loss
        + SLOT_ACTIVE_TVERSKY_LOSS_WEIGHT * slot_active_tversky_loss
        + SLOT_BOUNDARY_LOSS_WEIGHT * slot_boundary_loss
        + SLOT_EVENT_COUNT_LOSS_WEIGHT * event_count_loss
        + SLOT_MATCHED_OFF_LOSS_WEIGHT * matched_off_loss
        + SLOT_ACTIVE_DURATION_LOSS_WEIGHT * active_duration_loss
        + SLOT_BOUNDARY_MASS_LOSS_WEIGHT * boundary_mass_loss
        + SLOT_PHASE_OVERLAP_LOSS_WEIGHT * slot_phase_overlap_loss
        + SLOT_UNMATCHED_OFF_LOSS_WEIGHT * unmatched_off_loss
        + SLOT_DUPLICATE_LOSS_WEIGHT * duplicate_loss
        + SLOT_ACTIVITY_LOSS_WEIGHT * activity_loss
    )
    _require_finite_tensor("anonymous slot loss", total)
    metrics = {
        "loss": total.detach(),
        "slot_phase_loss": slot_phase_loss.detach(),
        "slot_active_tversky_loss": slot_active_tversky_loss.detach(),
        "slot_boundary_loss": slot_boundary_loss.detach(),
        "slot_event_count_loss": event_count_loss.detach(),
        "slot_matched_off_loss": matched_off_loss.detach(),
        "slot_active_duration_loss": active_duration_loss.detach(),
        "slot_boundary_mass_loss": boundary_mass_loss.detach(),
        "slot_phase_overlap_loss": slot_phase_overlap_loss.detach(),
        "slot_unmatched_off_loss": unmatched_off_loss.detach(),
        "slot_duplicate_loss": duplicate_loss.detach(),
        "slot_activity_loss": activity_loss.detach(),
    }
    metrics.update(_anonymous_slot_diagnostics(pred, target, assignments))
    for key, value in metrics.items():
        _require_finite_tensor(f"anonymous slot metric {key!r}", value)
    if detach_metrics:
        return total, {key: float(value.cpu()) for key, value in metrics.items()}
    return total, metrics


def _require_finite_tensor(name: str, value: Tensor) -> None:
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        bad = int((~finite).sum().detach().cpu())
        total = value.numel()
        sample = value.detach().flatten()[~finite.detach().flatten()][:3].cpu().tolist()
        raise RuntimeError(
            f"Non-finite tensor encountered in {name}: {bad}/{total} values, sample={sample}"
        )


def _target_phase_active(targets: dict[str, Tensor]) -> Tensor:
    return torch.stack([targets[name].float() for name in SLOT_PHASE_NAMES], dim=2).amax(dim=2)


def _phase_active_probability(predictions: dict[str, Tensor]) -> Tensor:
    return torch.stack([torch.sigmoid(predictions[name]) for name in SLOT_PHASE_NAMES], dim=2).amax(
        dim=2
    )


def _soft_dice_loss(probability: Tensor, target: Tensor) -> Tensor:
    intersection = (probability * target).sum(dim=-1)
    denominator = probability.sum(dim=-1) + target.sum(dim=-1) + 1e-6
    return (1.0 - (2.0 * intersection + 1e-6) / denominator).mean()


def _slot_phase_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch_indices: Tensor,
    pred_indices: Tensor,
    target_indices: Tensor,
) -> Tensor:
    losses = []
    for name in SLOT_PHASE_NAMES:
        logits = predictions[name][batch_indices, pred_indices]
        target = targets[name][batch_indices, target_indices].float()
        losses.append(
            focal_bce_with_logits(logits, target, gamma=2.0)
            + _soft_dice_loss(torch.sigmoid(logits), target)
        )
    return torch.stack(losses).mean()


def _active_tversky_loss(active_probability: Tensor, active_targets: Tensor) -> Tensor:
    true_positive = (active_probability * active_targets).sum(dim=-1)
    false_positive = (active_probability * (1.0 - active_targets)).sum(dim=-1)
    false_negative = ((1.0 - active_probability) * active_targets).sum(dim=-1)
    denominator = (
        true_positive
        + SLOT_TVERSKY_FALSE_POSITIVE_WEIGHT * false_positive
        + SLOT_TVERSKY_FALSE_NEGATIVE_WEIGHT * false_negative
        + 1e-6
    )
    return (1.0 - ((true_positive + 1e-6) / denominator)).mean()


def _matched_slot_off_loss(active_probability: Tensor, active_targets: Tensor) -> Tensor:
    inactive_mask = active_targets < 0.5
    if not bool(inactive_mask.any()):
        return active_probability.sum() * 0.0
    inactive_probability = active_probability[inactive_mask].clamp(1e-6, 1.0 - 1e-6)
    return -torch.log1p(-inactive_probability).mean()


def _active_duration_ratio_loss(active_probability: Tensor, active_targets: Tensor) -> Tensor:
    predicted_duration = active_probability.mean(dim=-1)
    target_duration = active_targets.float().mean(dim=-1)
    return torch.nn.functional.smooth_l1_loss(predicted_duration, target_duration)


def _slot_boundary_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch_indices: Tensor,
    pred_indices: Tensor,
    target_indices: Tensor,
) -> Tensor:
    losses = []
    weights = []
    for name in SLOT_BOUNDARY_NAMES:
        logits = predictions[name][batch_indices, pred_indices]
        target = targets[name][batch_indices, target_indices].float()
        focal = focal_bce_with_logits(logits, target, gamma=2.0)
        losses.append(focal + _soft_dice_loss(torch.sigmoid(logits), target))
        weights.append(
            torch.as_tensor(
                SLOT_BOUNDARY_WEIGHTS[name],
                dtype=logits.dtype,
                device=logits.device,
            )
        )
    weight_tensor = torch.stack(weights)
    return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def _slot_boundary_mass_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch_indices: Tensor,
    pred_indices: Tensor,
    target_indices: Tensor,
) -> Tensor:
    losses = []
    for name in SLOT_BOUNDARY_NAMES:
        probability = torch.sigmoid(predictions[name][batch_indices, pred_indices])
        target = targets[name][batch_indices, target_indices].float()
        pred_mass = probability.mean(dim=-1)
        target_mass = target.mean(dim=-1)
        excess_mass = torch.relu(pred_mass - target_mass * 2.0)
        losses.append(excess_mass.mean())
    return torch.stack(losses).mean()


def _slot_event_count_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch_indices: Tensor,
    pred_indices: Tensor,
    target_indices: Tensor,
) -> Tensor:
    losses = []
    for name in ("slot_onset", "slot_offset"):
        probability = torch.sigmoid(predictions[name][batch_indices, pred_indices])
        target = targets[name][batch_indices, target_indices].float()
        losses.append(
            torch.nn.functional.smooth_l1_loss(probability.sum(dim=-1), target.sum(dim=-1))
        )
    return torch.stack(losses).mean()


def _slot_phase_overlap_loss(predictions: dict[str, Tensor]) -> Tensor:
    probabilities = [torch.sigmoid(predictions[name]) for name in SLOT_PHASE_NAMES]
    return (
        probabilities[0] * probabilities[1]
        + probabilities[0] * probabilities[2]
        + probabilities[1] * probabilities[2]
    ).mean()


def _slot_duplicate_loss(slot_tracks: Tensor) -> Tensor:
    batch_size, slot_count, _time = slot_tracks.shape
    if slot_count < 2:
        return slot_tracks.sum() * 0.0
    losses: list[Tensor] = []
    for batch_index in range(batch_size):
        masks = slot_tracks[batch_index]
        intersection = torch.minimum(masks[:, None], masks[None]).sum(dim=-1)
        union = torch.maximum(masks[:, None], masks[None]).sum(dim=-1).clamp_min(1e-6)
        similarity = intersection / union
        active_mass = torch.minimum(masks.mean(dim=-1)[:, None], masks.mean(dim=-1)[None])
        gate = ((active_mass - 0.02) / 0.08).clamp(0.0, 1.0)
        pair_mask = torch.triu(
            torch.ones(slot_count, slot_count, dtype=torch.bool, device=masks.device),
            diagonal=1,
        )
        pair_penalty = (torch.relu(similarity - SLOT_DUPLICATE_SIMILARITY_THRESHOLD).pow(2) * gate)[
            pair_mask
        ]
        if pair_penalty.numel() > 0:
            losses.append(pair_penalty.mean())
    if not losses:
        return slot_tracks.sum() * 0.0
    return torch.stack(losses).mean()


def _anonymous_slot_diagnostics(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    assignments: list[list[tuple[int, int]]],
) -> dict[str, Tensor]:
    active_probability = _phase_active_probability(predictions)
    target_active = targets["slot_active"].float()
    target_activity = targets["slot_activity"].float()
    predicted_nonempty = (active_probability.mean(dim=-1) >= 0.02).float().sum(dim=-1).mean()
    diagnostics = {
        "slot_pred_active_fraction": active_probability.mean().detach(),
        "slot_target_active_fraction": target_active.mean().detach(),
        "slot_frame_zero_active_rate": active_probability[..., 0].mean().detach(),
        "slot_pred_nonempty_count": predicted_nonempty.detach(),
        "slot_onset_boundary_mass": torch.sigmoid(predictions["slot_onset"]).mean().detach(),
        "slot_offset_boundary_mass": torch.sigmoid(predictions["slot_offset"]).mean().detach(),
    }
    batch_indices: list[int] = []
    pred_indices: list[int] = []
    target_indices: list[int] = []
    for batch_index, assignment in enumerate(assignments):
        for pred_index, target_index in assignment:
            batch_indices.append(batch_index)
            pred_indices.append(pred_index)
            target_indices.append(target_index)
    if batch_indices:
        batch_tensor = torch.as_tensor(batch_indices, device=active_probability.device)
        pred_tensor = torch.as_tensor(pred_indices, device=active_probability.device)
        target_tensor = torch.as_tensor(target_indices, device=active_probability.device)
        pred_duration = active_probability[batch_tensor, pred_tensor].mean(dim=-1)
        target_duration = target_active[batch_tensor, target_tensor].mean(dim=-1)
        duration_error = (pred_duration - target_duration).abs().mean()
    else:
        duration_error = active_probability.sum() * 0.0
    diagnostics["slot_active_duration_abs_error"] = duration_error.detach()
    diagnostics["slot_target_nonempty_count"] = target_activity.sum(dim=-1).mean().detach()
    return diagnostics


def _slot_assignments(
    predictions: dict[str, Tensor], targets: dict[str, Tensor]
) -> list[list[tuple[int, int]]]:
    batch_size, slot_count, _time = predictions["slot_attack"].shape
    rows: list[list[tuple[int, int]]] = []
    with torch.no_grad():
        pred_active = _phase_active_probability(predictions).float().detach().cpu()
        target_active = targets["slot_active"].detach().cpu().float()
        target_activity = targets["slot_activity"].detach().cpu().float()
        pred_phases = {
            name: torch.sigmoid(predictions[name]).float().detach().cpu()
            for name in SLOT_PHASE_NAMES
        }
        target_phases = {name: targets[name].detach().cpu().float() for name in SLOT_PHASE_NAMES}
        pred_activity = torch.sigmoid(predictions["slot_activity"]).float().detach().cpu()
        pred_boundaries = {
            name: torch.sigmoid(predictions[name]).float().detach().cpu()
            for name in SLOT_BOUNDARY_NAMES
        }
        target_boundaries = {
            name: targets[name].detach().cpu().float() for name in SLOT_BOUNDARY_NAMES
        }
        for batch_index in range(batch_size):
            active_targets = [
                index
                for index in range(slot_count)
                if float(target_activity[batch_index, index]) >= 0.5
            ]
            if not active_targets:
                rows.append([])
                continue
            true_active = target_active[batch_index, active_targets]
            active_probability = pred_active[batch_index]
            phase_active_cost = _pairwise_dice_cost(active_probability, true_active)
            inactive_cost = _pairwise_inactive_false_positive_cost(
                active_probability,
                true_active,
            )
            duration_cost = _pairwise_active_duration_cost(active_probability, true_active)
            per_phase_cost = torch.zeros(
                slot_count, len(active_targets), dtype=active_probability.dtype
            )
            for name in SLOT_PHASE_NAMES:
                per_phase_cost += _pairwise_dice_cost(
                    pred_phases[name][batch_index],
                    target_phases[name][batch_index, active_targets],
                )
            per_phase_cost = per_phase_cost / len(SLOT_PHASE_NAMES)
            boundary_cost = torch.zeros(
                slot_count, len(active_targets), dtype=active_probability.dtype
            )
            boundary_weight_total = sum(SLOT_BOUNDARY_WEIGHTS.values())
            for name in SLOT_BOUNDARY_NAMES:
                labels = target_boundaries[name][batch_index, active_targets]
                boundary_cost += SLOT_BOUNDARY_WEIGHTS[name] * _pairwise_dice_cost(
                    pred_boundaries[name][batch_index],
                    labels,
                )
            boundary_cost = boundary_cost / boundary_weight_total
            activity_cost = (1.0 - pred_activity[batch_index])[:, None].expand(
                -1,
                len(active_targets),
            )
            cost = (
                0.25 * phase_active_cost
                + 0.25 * per_phase_cost
                + 0.15 * boundary_cost
                + 0.10 * activity_cost
                + 0.15 * inactive_cost
                + 0.10 * duration_cost
            ).clamp(0.0, 10.0)
            _require_finite_tensor(f"anonymous slot assignment cost batch {batch_index}", cost)
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


def _pairwise_dice_cost(prediction: Tensor, target: Tensor) -> Tensor:
    intersection = prediction @ target.transpose(0, 1)
    denominator = prediction.sum(dim=-1, keepdim=True) + target.sum(dim=-1)[None] + 1e-6
    return (1.0 - (2.0 * intersection + 1e-6) / denominator).clamp(0.0, 1.0)


def _pairwise_inactive_false_positive_cost(prediction: Tensor, target: Tensor) -> Tensor:
    inactive_target = 1.0 - target
    inactive_frames = inactive_target.sum(dim=-1).clamp_min(1e-6)
    return ((prediction @ inactive_target.transpose(0, 1)) / inactive_frames[None]).clamp(0.0, 1.0)


def _pairwise_active_duration_cost(prediction: Tensor, target: Tensor) -> Tensor:
    pred_duration = prediction.mean(dim=-1, keepdim=True)
    target_duration = target.mean(dim=-1)[None]
    return (pred_duration - target_duration).abs().clamp(0.0, 1.0)


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
        model = SlotAttentionEventModel(max_tracks=max_tracks).to(device)
    else:
        model = CNNTCNTimingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_val = math.inf
    global_step = 0
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        train_metrics, global_step = _run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            training=True,
            target=target,
            run_dir=run_dir,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics, global_step = _run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            training=False,
            target=target,
            run_dir=run_dir,
            epoch=epoch,
            global_step=global_step,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        checkpoint = _training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_val,
            target=target,
            max_tracks=max_tracks,
            crop_seconds=crop_seconds,
        )
        torch.save(checkpoint, run_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, run_dir / "best.pt")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    training: bool,
    target: TrainingTarget = "source_types_v1",
    run_dir: Path | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    model.train(training)
    totals: dict[str, Tensor] = {}
    count = 0
    recent_metrics: list[dict[str, float]] = []
    iterator = tqdm(loader, leave=False, desc="train" if training else "val", mininterval=5.0)
    for batch in iterator:
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        grad_norm: Tensor | None = None
        try:
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
                    _require_finite_outputs(predictions)
                    loss, metrics = compute_anonymous_slot_loss(
                        predictions, batch, detach_metrics=False
                    )
                else:
                    loss, metrics = compute_loss(predictions, batch, detach_metrics=False)
            if training:
                _require_finite_tensor("training loss", loss)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if target == "anonymous_slots_v1":
                    scaler.unscale_(optimizer)
                    _require_finite_model_gradients(model)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    _require_finite_tensor("anonymous slot gradient norm", grad_norm)
                    _require_finite_model_gradients(model)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
        except RuntimeError as error:
            if target == "anonymous_slots_v1" and run_dir is not None:
                _write_anonymous_debug_json(
                    run_dir=run_dir,
                    epoch=epoch,
                    global_step=global_step,
                    batch=batch,
                    recent_metrics=recent_metrics,
                    grad_norm=grad_norm,
                    model=model,
                    error=error,
                )
            raise
        batch_size = int(batch["waveform"].shape[0])
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, value.new_zeros(())) + value * batch_size
        recent_metrics.append({key: float(value.detach().cpu()) for key, value in metrics.items()})
        recent_metrics = recent_metrics[-5:]
        if count % (batch_size * 100) == 0:
            iterator.set_postfix(loss=float(metrics["loss"].detach().cpu()))
    if count == 0:
        return {"loss": math.inf}, global_step
    return {key: float((value / max(count, 1)).cpu()) for key, value in totals.items()}, global_step


def _objective_metadata(target: TrainingTarget) -> dict[str, object]:
    if target != "anonymous_slots_v1":
        return {"name": target}
    return {
        "name": target,
        "version": ANONYMOUS_SLOT_OBJECTIVE_VERSION,
        "weights": {
            "slot_phase_loss": SLOT_PHASE_LOSS_WEIGHT,
            "slot_active_tversky_loss": SLOT_ACTIVE_TVERSKY_LOSS_WEIGHT,
            "slot_boundary_loss": SLOT_BOUNDARY_LOSS_WEIGHT,
            "slot_event_count_loss": SLOT_EVENT_COUNT_LOSS_WEIGHT,
            "slot_matched_off_loss": SLOT_MATCHED_OFF_LOSS_WEIGHT,
            "slot_active_duration_loss": SLOT_ACTIVE_DURATION_LOSS_WEIGHT,
            "slot_boundary_mass_loss": SLOT_BOUNDARY_MASS_LOSS_WEIGHT,
            "slot_phase_overlap_loss": SLOT_PHASE_OVERLAP_LOSS_WEIGHT,
            "slot_unmatched_off_loss": SLOT_UNMATCHED_OFF_LOSS_WEIGHT,
            "slot_duplicate_loss": SLOT_DUPLICATE_LOSS_WEIGHT,
            "slot_activity_loss": SLOT_ACTIVITY_LOSS_WEIGHT,
        },
    }


def _training_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    target: TrainingTarget,
    max_tracks: int,
    crop_seconds: float,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "target": target,
        "architecture": (
            "slot_attention_phase_event_v1" if target == "anonymous_slots_v1" else "cnn_tcn_v1"
        ),
        "objective": _objective_metadata(target),
        "rng_state": _rng_state(),
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


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1],
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _require_finite_outputs(predictions: dict[str, Tensor]) -> None:
    for name, value in predictions.items():
        _require_finite_tensor(f"model output {name!r}", value)


def _nonfinite_gradient_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            names.append(name)
    return names


def _require_finite_model_gradients(model: nn.Module) -> None:
    bad_names = _nonfinite_gradient_parameter_names(model)
    if bad_names:
        joined = ", ".join(bad_names)
        raise RuntimeError(f"Non-finite gradient encountered in parameter(s): {joined}")


def _write_anonymous_debug_json(
    run_dir: Path,
    epoch: int,
    global_step: int,
    batch: dict[str, object],
    recent_metrics: list[dict[str, float]],
    grad_norm: Tensor | None,
    model: nn.Module,
    error: RuntimeError,
) -> None:
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    clip_indices = batch.get("clip_index")
    if isinstance(clip_indices, Tensor):
        clip_identifiers = [int(value) for value in clip_indices.detach().cpu().flatten().tolist()]
    else:
        clip_identifiers = []
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_clip_indices": clip_identifiers,
        "recent_metrics": recent_metrics,
        "grad_norm": None if grad_norm is None else float(grad_norm.detach().cpu()),
        "offending_parameter_names": _nonfinite_gradient_parameter_names(model),
        "error": str(error),
    }
    path = debug_dir / f"anonymous_slot_failure_epoch_{epoch:04d}_step_{global_step:08d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


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
