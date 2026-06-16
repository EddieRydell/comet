from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comet_audio.generator import SOURCE_TYPES, GeneratorConfig, generate_batch
from comet_audio.inference import decode_slot_events
from comet_audio.models import BatchManifestEntry
from comet_audio.training import (
    HOP_LENGTH,
    SAMPLE_RATE,
    SLOT_BOUNDARY_NAMES,
    CNNTCNTimingModel,
    CometTimingDataset,
    DatasetItem,
    SlotAttentionEventModel,
    build_anonymous_slot_targets,
    build_targets,
    compute_anonymous_slot_loss,
    compute_loss,
    frame_count,
    load_manifest,
    load_metadata,
    match_onsets,
    split_manifest,
)


def _dummy_items(count: int) -> list[DatasetItem]:
    return [
        DatasetItem(
            entry=BatchManifestEntry(
                clip_id=f"clip_{index:04d}",
                seed=index,
                bpm=120.0,
                time_signature="4/4",
                key="C",
                mix_path=f"audio/clip_{index:04d}.wav",
                metadata_path=f"metadata/clip_{index:04d}.json",
                source_count=1,
                event_count=1,
            ),
            mix_path=Path(f"audio/clip_{index:04d}.wav"),
            metadata_path=Path(f"metadata/clip_{index:04d}.json"),
        )
        for index in range(count)
    ]


def test_dataset_loads_manifest_audio_and_metadata(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=2,
        seed=300,
        config=config,
        write_stems=False,
        flat_layout=True,
    )

    items = load_manifest(tmp_path / "training")
    metadata = load_metadata(items[0].metadata_path)
    dataset = CometTimingDataset(tmp_path / "training", "train", training=False, crop_seconds=1.0)
    sample = dataset[0]

    assert len(items) == 2
    assert metadata.sample_rate == SAMPLE_RATE
    assert sample["waveform"].shape == (SAMPLE_RATE,)
    assert sample["onset"].shape == (frame_count(SAMPLE_RATE),)
    assert sample["source_onset"].shape == (len(SOURCE_TYPES), frame_count(SAMPLE_RATE))


def test_split_logic_returns_exact_10k_manifest_order_counts() -> None:
    items = _dummy_items(10_000)

    assert len(split_manifest(items, "train")) == 8000
    assert len(split_manifest(items, "val")) == 1000
    assert len(split_manifest(items, "test")) == 1000
    assert split_manifest(items, "train")[0].entry.clip_id == "clip_0000"
    assert split_manifest(items, "val")[0].entry.clip_id == "clip_8000"
    assert split_manifest(items, "test")[0].entry.clip_id == "clip_9000"


def test_split_logic_uses_full_large_manifest() -> None:
    items = _dummy_items(100_000)

    assert len(split_manifest(items, "train")) == 80_000
    assert len(split_manifest(items, "val")) == 10_000
    assert len(split_manifest(items, "test")) == 10_000
    assert split_manifest(items, "train")[-1].entry.clip_id == "clip_79999"
    assert split_manifest(items, "val")[0].entry.clip_id == "clip_80000"
    assert split_manifest(items, "test")[0].entry.clip_id == "clip_90000"


def test_target_builder_aligns_timestamps_to_frames(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=1,
        seed=301,
        config=config,
        write_stems=False,
        flat_layout=True,
    )
    item = load_manifest(tmp_path / "training")[0]
    metadata = load_metadata(item.metadata_path)
    targets = build_targets(metadata, num_frames=frame_count(SAMPLE_RATE))

    assert set(targets) == {
        "onset",
        "attack",
        "held",
        "release",
        "source_onset",
        "onset_offset",
        "onset_offset_mask",
    }
    for event in metadata.events:
        onset_frame = int(round(event.onset_seconds * SAMPLE_RATE / HOP_LENGTH))
        peak_frame = int(torch.argmax(targets["onset"]).item())
        if math.isclose(event.onset_seconds, metadata.events[0].onset_seconds):
            assert abs(peak_frame - onset_frame) <= 1
            break
    assert targets["attack"].shape == targets["onset"].shape
    assert targets["held"].shape == targets["onset"].shape
    assert targets["release"].shape == targets["onset"].shape


def test_model_forward_returns_all_heads_with_expected_time_dimension() -> None:
    model = CNNTCNTimingModel()
    waveform = torch.zeros(2, SAMPLE_RATE // 2)
    outputs = model(waveform)
    expected_time = frame_count(SAMPLE_RATE // 2)

    assert outputs["onset"].shape == (2, expected_time)
    assert outputs["attack"].shape == (2, expected_time)
    assert outputs["held"].shape == (2, expected_time)
    assert outputs["release"].shape == (2, expected_time)
    assert outputs["source_onset"].shape == (2, len(SOURCE_TYPES), expected_time)
    assert outputs["onset_offset"].shape == (2, expected_time)


def test_one_tiny_cpu_training_step_computes_finite_loss_and_updates(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=2,
        seed=302,
        config=config,
        write_stems=False,
        flat_layout=True,
    )
    dataset = CometTimingDataset(tmp_path / "training", "train", training=True, crop_seconds=1.0)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = CNNTCNTimingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = next(model.parameters()).detach().clone()

    outputs = model(batch["waveform"])
    loss, _ = compute_loss(outputs, batch)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, next(model.parameters()).detach())


def test_anonymous_slot_model_uses_event_boundary_heads_and_activity_loss(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=2,
        seed=303,
        config=config,
        write_stems=False,
        flat_layout=True,
    )
    item = load_manifest(tmp_path / "training")[0]
    metadata = load_metadata(item.metadata_path)
    targets = build_anonymous_slot_targets(metadata, num_frames=frame_count(SAMPLE_RATE))

    assert "slot_active" in targets
    assert "slot_onset" in targets
    assert "slot_attack_end" in targets
    assert "slot_release_start" in targets
    assert "slot_offset" in targets
    assert "slot_activity" in targets
    assert targets["slot_active"].shape == targets["slot_onset"].shape
    assert targets["slot_onset"].amax() > 0.0
    assert targets["slot_offset"].amax() > 0.0
    assert targets["slot_activity"].shape == targets["slot_mask"].shape

    dataset = CometTimingDataset(
        tmp_path / "training",
        "train",
        training=True,
        crop_seconds=1.0,
        target="anonymous_slots_v1",
        max_tracks=4,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = SlotAttentionEventModel(max_tracks=4)
    outputs = model(batch["waveform"])
    loss, metrics = compute_anonymous_slot_loss(outputs, batch)

    assert outputs["slot_active"].shape == (1, 4, frame_count(SAMPLE_RATE))
    for name in SLOT_BOUNDARY_NAMES:
        assert outputs[name].shape == outputs["slot_active"].shape
    assert outputs["slot_activity"].shape == (1, 4)
    assert torch.isfinite(loss)
    assert metrics["slot_active_loss"] >= 0.0
    assert metrics["slot_active_tversky_loss"] >= 0.0
    assert metrics["slot_boundary_loss"] >= 0.0
    assert metrics["slot_event_count_loss"] >= 0.0
    assert metrics["slot_unmatched_off_loss"] >= 0.0
    assert metrics["slot_duplicate_loss"] >= 0.0
    assert metrics["slot_activity_loss"] >= 0.0


def test_anonymous_slot_loss_penalizes_overactive_predictions() -> None:
    target_active = torch.zeros(1, 2, 20)
    target_active[0, 0, 2:8] = 1.0
    batch = {
        "slot_active": target_active,
        "slot_onset": torch.zeros(1, 2, 20),
        "slot_attack_end": torch.zeros(1, 2, 20),
        "slot_release_start": torch.zeros(1, 2, 20),
        "slot_offset": torch.zeros(1, 2, 20),
        "slot_activity": target_active.any(dim=-1).float(),
    }
    batch["slot_onset"][0, 0, 2] = 1.0
    batch["slot_attack_end"][0, 0, 4] = 1.0
    batch["slot_release_start"][0, 0, 6] = 1.0
    batch["slot_offset"][0, 0, 8] = 1.0
    mostly_off = {
        name: torch.full((1, 2, 20), -4.0) for name in ("slot_active", *SLOT_BOUNDARY_NAMES)
    }
    mostly_off["slot_active"][0, 0, 2:8] = 4.0
    mostly_off["slot_onset"][0, 0, 2] = 4.0
    mostly_off["slot_attack_end"][0, 0, 4] = 4.0
    mostly_off["slot_release_start"][0, 0, 6] = 4.0
    mostly_off["slot_offset"][0, 0, 8] = 4.0
    mostly_off["slot_activity"] = torch.tensor([[4.0, -4.0]])
    all_on = {name: torch.full((1, 2, 20), -4.0) for name in ("slot_active", *SLOT_BOUNDARY_NAMES)}
    all_on["slot_active"][:] = 4.0
    all_on["slot_onset"][:] = 4.0
    all_on["slot_offset"][:] = 4.0
    all_on["slot_activity"] = torch.tensor([[4.0, 4.0]])

    mostly_off_loss, _ = compute_anonymous_slot_loss(mostly_off, batch)
    all_on_loss, _ = compute_anonymous_slot_loss(all_on, batch)

    assert all_on_loss > mostly_off_loss


def test_anonymous_slot_loss_reports_duplicate_slot_penalty() -> None:
    target_active = torch.zeros(1, 2, 16)
    target_active[0, 0, 2:6] = 1.0
    target_active[0, 1, 10:14] = 1.0
    batch = {
        "slot_active": target_active,
        "slot_onset": torch.zeros(1, 2, 16),
        "slot_attack_end": torch.zeros(1, 2, 16),
        "slot_release_start": torch.zeros(1, 2, 16),
        "slot_offset": torch.zeros(1, 2, 16),
        "slot_activity": target_active.any(dim=-1).float(),
    }
    duplicate = {
        name: torch.full((1, 2, 16), -4.0) for name in ("slot_active", *SLOT_BOUNDARY_NAMES)
    }
    duplicate["slot_active"][:, :, 2:14] = 4.0
    duplicate["slot_activity"] = torch.ones(1, 2)

    _loss, metrics = compute_anonymous_slot_loss(duplicate, batch)

    assert metrics["slot_duplicate_loss"] > 0.0


def test_anonymous_slot_boundary_loss_penalizes_missing_targets() -> None:
    target_active = torch.zeros(1, 1, 12)
    target_active[0, 0, 2:8] = 1.0
    batch = {
        "slot_active": target_active,
        "slot_onset": torch.zeros(1, 1, 12),
        "slot_attack_end": torch.zeros(1, 1, 12),
        "slot_release_start": torch.zeros(1, 1, 12),
        "slot_offset": torch.zeros(1, 1, 12),
        "slot_activity": torch.ones(1, 1),
    }
    batch["slot_onset"][0, 0, 2] = 1.0
    batch["slot_offset"][0, 0, 8] = 1.0
    missing = {name: torch.full((1, 1, 12), -4.0) for name in ("slot_active", *SLOT_BOUNDARY_NAMES)}
    missing["slot_active"][0, 0, 2:8] = 4.0
    missing["slot_activity"] = torch.ones(1, 1) * 4.0

    _loss, metrics = compute_anonymous_slot_loss(missing, batch)

    assert metrics["slot_boundary_loss"] > 0.0


def test_slot_event_decoder_turns_boundary_peaks_into_segments() -> None:
    probabilities = {
        "slot_active": torch.zeros(20),
        "slot_onset": torch.zeros(20),
        "slot_attack_end": torch.zeros(20),
        "slot_release_start": torch.zeros(20),
        "slot_offset": torch.zeros(20),
    }
    probabilities["slot_active"][2:10] = 0.9
    probabilities["slot_onset"][2] = 0.95
    probabilities["slot_attack_end"][4] = 0.8
    probabilities["slot_release_start"][7] = 0.85
    probabilities["slot_offset"][10] = 0.96

    events = decode_slot_events(probabilities, frame_seconds=0.01, duration=0.5)

    assert len(events) == 1
    assert events[0]["onset_seconds"] == 0.02
    assert events[0]["attack_end_seconds"] == 0.04
    assert events[0]["release_start_seconds"] == 0.07
    assert events[0]["offset_seconds"] == 0.10


def test_evaluation_matching_scores_unique_duplicate_onsets_once() -> None:
    predicted = [0.100, 0.101, 0.400]
    truth = sorted({0.100, 0.400})

    true_positive, false_positive, false_negative, errors = match_onsets(
        predicted, truth, tolerance=0.005
    )

    assert true_positive == 2
    assert false_positive == 1
    assert false_negative == 0
    assert max(errors) == 0.0
