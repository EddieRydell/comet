from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comet_audio.generator import GeneratorConfig, generate_batch
from comet_audio.models import BatchManifestEntry
from comet_audio.training import (
    HOP_LENGTH,
    SAMPLE_RATE,
    CNNTCNTimingModel,
    CometTimingDataset,
    DatasetItem,
    build_targets,
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
        write_preview=False,
        write_stems=False,
        flat_layout=True,
    )

    items = load_manifest(tmp_path / "training")
    metadata = load_metadata(items[0].metadata_path)
    dataset = CometTimingDataset(tmp_path / "training", "train", training=False)
    sample = dataset[0]

    assert len(items) == 2
    assert metadata.sample_rate == SAMPLE_RATE
    assert sample["waveform"].shape == (SAMPLE_RATE,)
    assert sample["onset"].shape == (frame_count(SAMPLE_RATE),)
    assert sample["source_onset"].shape == (8, frame_count(SAMPLE_RATE))


def test_split_logic_returns_exact_10k_manifest_order_counts() -> None:
    items = _dummy_items(10_000)

    assert len(split_manifest(items, "train")) == 8000
    assert len(split_manifest(items, "val")) == 1000
    assert len(split_manifest(items, "test")) == 1000
    assert split_manifest(items, "train")[0].entry.clip_id == "clip_0000"
    assert split_manifest(items, "val")[0].entry.clip_id == "clip_8000"
    assert split_manifest(items, "test")[0].entry.clip_id == "clip_9000"


def test_target_builder_aligns_timestamps_to_frames(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=1,
        seed=301,
        config=config,
        write_preview=False,
        write_stems=False,
        flat_layout=True,
    )
    item = load_manifest(tmp_path / "training")[0]
    metadata = load_metadata(item.metadata_path)
    targets = build_targets(metadata, num_frames=frame_count(SAMPLE_RATE))

    assert set(targets) == {
        "onset",
        "attack",
        "sustain",
        "release",
        "active",
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
    assert targets["sustain"].shape == targets["onset"].shape
    assert targets["release"].shape == targets["onset"].shape
    assert targets["active"].shape == targets["onset"].shape


def test_model_forward_returns_all_heads_with_expected_time_dimension() -> None:
    model = CNNTCNTimingModel()
    waveform = torch.zeros(2, SAMPLE_RATE // 2)
    outputs = model(waveform)
    expected_time = frame_count(SAMPLE_RATE // 2)

    assert outputs["onset"].shape == (2, expected_time)
    assert outputs["attack"].shape == (2, expected_time)
    assert outputs["sustain"].shape == (2, expected_time)
    assert outputs["release"].shape == (2, expected_time)
    assert outputs["active"].shape == (2, expected_time)
    assert outputs["source_onset"].shape == (2, 8, expected_time)
    assert outputs["onset_offset"].shape == (2, expected_time)


def test_one_tiny_cpu_training_step_computes_finite_loss_and_updates(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=SAMPLE_RATE, duration_seconds=1.0)
    generate_batch(
        tmp_path / "training",
        count=2,
        seed=302,
        config=config,
        write_preview=False,
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
