from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from comet_audio.assets import choose_sfz_region, parse_sfz, validate_asset_catalog
from comet_audio.generator import GeneratorConfig, generate_batch, generate_clip


def test_fixed_seed_metadata_is_deterministic(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=2.0)
    first = generate_clip(tmp_path / "first", seed=1234, config=config)
    second = generate_clip(tmp_path / "second", seed=1234, config=config)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_event_times_and_source_references_are_valid(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=2.0, time_signatures=("7/4",))
    metadata = generate_clip(tmp_path / "clip", seed=7, config=config)
    source_ids = {source.source_id for source in metadata.sources}

    assert metadata.time_signature == "7/4"
    assert metadata.beats_per_measure == 7
    assert metadata.beat_unit == 4
    assert 70 <= metadata.bpm <= 150
    assert metadata.events
    for event in metadata.events:
        assert event.source_id in source_ids
        assert 0.0 <= event.onset_seconds < metadata.duration_seconds
        assert event.onset_seconds < event.offset_seconds <= metadata.duration_seconds
        assert 0.0 < event.velocity <= 1.0
        assert event.attack_seconds >= 0.0
        assert event.release_seconds > 0.0
        assert event.render_parameters["beat_unit"] == 4
        assert event.render_parameters["measure_seconds"] > event.render_parameters["beat_seconds"]
        assert event.render_parameters["rhythm_subdivision"] in {2, 3, 4, 6, 8}
        assert event.render_parameters["timbre_variation"]


def test_bpm_bounds_are_configurable(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=2.0,
        bpm_min=90,
        bpm_max=92,
        time_signatures=("2/2",),
    )
    metadata = generate_clip(tmp_path / "clip", seed=99, config=config)

    assert 90 <= metadata.bpm <= 92
    assert metadata.time_signature == "2/2"


def test_source_count_and_growl_basses_are_available(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=2.0,
        source_count_min=9,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config) for seed in range(20, 28)
    ]

    assert all(9 <= len(clip.sources) <= 10 for clip in clips)

    source_types = {source.source_type for clip in clips for source in clip.sources}
    assert {"synth_bass", "electric_bass", "acoustic_bass"} & source_types
    tonal_types = {"synth_pluck", "synth_lead", "pad_chord", "mallet", "string_stab", "brass_stab"}
    assert tonal_types & source_types

    bass_sources = [
        source
        for clip in clips
        for source in clip.sources
        if source.source_type in {"synth_bass", "electric_bass", "acoustic_bass"}
    ]
    pluck_sources = [
        source
        for clip in clips
        for source in clip.sources
        if source.synth_parameters["procedural_recipe"] == "pluck_stab"
    ]
    assert bass_sources
    for source in bass_sources:
        assert "wavefold_drive" in source.synth_parameters
        assert "sub_mix" in source.synth_parameters
    assert pluck_sources
    assert max(source.gain_db for source in pluck_sources) > -7.5


def test_growl_and_wub_basses_are_low_and_sustained(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config) for seed in range(70, 78)
    ]
    sustained_events = [
        event
        for clip in clips
        for event in clip.events
        if event.event_type in {"synth_bass", "electric_bass", "acoustic_bass"}
    ]

    assert sustained_events
    assert all(event.midi_note is not None and event.midi_note <= 55 for event in sustained_events)
    assert any(event.render_parameters["duration_beats"] >= 1.0 for event in sustained_events)
    assert all(event.render_parameters["duration_beats"] <= 1.750001 for event in sustained_events)
    for clip in clips:
        growl_wub_sources = [
            source
            for source in clip.sources
            if source.source_type in {"electric_bass", "acoustic_bass"}
        ]
        assert len(growl_wub_sources) <= 1


def test_plucks_use_dense_subdivisions_and_bright_parameters(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config)
        for seed in range(100, 108)
    ]
    pluck_sources = [
        source
        for clip in clips
        for source in clip.sources
        if source.synth_parameters["procedural_recipe"] == "pluck_stab"
    ]
    pluck_events = [
        event
        for clip in clips
        for event in clip.events
        if any(
            source.source_id == event.source_id
            and source.synth_parameters["procedural_recipe"] == "pluck_stab"
            for source in clip.sources
        )
    ]

    assert pluck_sources
    assert pluck_events
    assert any(event.render_parameters["rhythm_subdivision"] in {4, 6, 8} for event in pluck_events)
    assert max(source.gain_db for source in pluck_sources) > -3.0
    assert max(source.synth_parameters["filter_hz"] for source in pluck_sources) >= 6200
    for source in pluck_sources:
        assert "transient_click" in source.synth_parameters
        assert "octave_mix" in source.synth_parameters
        assert "saw_mix" in source.synth_parameters
        assert "timbre_variant" in source.synth_parameters


def test_synth_metadata_does_not_use_lfo_modulation(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config)
        for seed in range(130, 138)
    ]

    banned_fragments = ("lfo", "wub_depth", "wub_shape", "wub_division")
    for clip in clips:
        for source in clip.sources:
            for key in source.synth_parameters:
                assert not any(fragment in key for fragment in banned_fragments)


def test_effect_metadata_does_not_use_delay_or_reverb(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    metadata = generate_clip(tmp_path / "clip", seed=145, config=config)
    banned_fragments = ("delay", "reverb", "room_size")

    for source in metadata.sources:
        for key in source.effect_parameters:
            assert not any(fragment in key for fragment in banned_fragments)


def test_event_timbre_varies_within_sources(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    metadata = generate_clip(tmp_path / "clip", seed=150, config=config)

    varied_sources = 0
    for source in metadata.sources:
        source_events = [event for event in metadata.events if event.source_id == source.source_id]
        variations = [
            tuple(sorted(event.render_parameters["timbre_variation"].items()))
            for event in source_events
        ]
        if len(set(variations)) > 1:
            varied_sources += 1

    assert varied_sources >= 3


def test_duplicate_source_types_get_distinct_timbre_variants(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=8.0,
        source_count_min=10,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config)
        for seed in range(170, 180)
    ]
    found_duplicate_type = False

    for clip in clips:
        by_type: dict[str, list] = {}
        for source in clip.sources:
            by_type.setdefault(source.source_type, []).append(source)
        for sources in by_type.values():
            if len(sources) <= 1:
                continue
            found_duplicate_type = True
            variant_positions = {source.synth_parameters["variant_position"] for source in sources}
            variant_labels = {source.synth_parameters["timbre_variant"] for source in sources}
            voice_models = {source.synth_parameters.get("voice_model") for source in sources}
            assert len(variant_positions) == len(sources)
            assert len(variant_labels) >= 2
            recipe = sources[0].synth_parameters["procedural_recipe"]
            if recipe in {"fm_bass", "fm_growl", "wub_bass", "pluck_stab"}:
                assert len(voice_models) >= 2
                expected_unique = min(
                    len(sources),
                    4 if recipe in {"fm_bass", "pluck_stab"} else 3,
                )
                assert len(voice_models) == expected_unique
            assert all(
                source.synth_parameters["variant_count"] == len(sources) for source in sources
            )

    assert found_duplicate_type


def test_riser_impact_is_noise_dominant_metadata(tmp_path: Path) -> None:
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=2.0,
        source_count_min=10,
        source_count_max=10,
    )
    clips = [
        generate_clip(tmp_path / f"clip_{seed}", seed=seed, config=config) for seed in range(40, 50)
    ]
    risers = [
        source
        for clip in clips
        for source in clip.sources
        if source.synth_parameters["procedural_recipe"] == "riser_impact"
    ]

    assert risers
    for source in risers:
        assert "filter_start_hz" in source.synth_parameters
        assert "filter_end_hz" in source.synth_parameters
        assert "noise_color" in source.synth_parameters
        assert "impact_weight" in source.synth_parameters
        assert "start_hz" not in source.synth_parameters
        assert "end_hz" not in source.synth_parameters


def test_wavs_have_expected_shape_and_signal(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=2.0)
    metadata = generate_clip(tmp_path / "clip", seed=42, config=config)
    expected_frames = int(config.sample_rate * config.duration_seconds)

    audio, sample_rate = sf.read(tmp_path / "clip" / "mix.wav")
    assert sample_rate == config.sample_rate
    assert audio.shape == (expected_frames,)
    assert np.isfinite(audio).all()
    assert float(np.sqrt(np.mean(audio**2))) > 1e-4

    for source in metadata.sources:
        stem, stem_sample_rate = sf.read(tmp_path / "clip" / source.stem_path)
        assert stem_sample_rate == config.sample_rate
        assert stem.shape == (expected_frames,)
        assert np.isfinite(stem).all()
        assert float(np.sqrt(np.mean(stem**2))) > 1e-5


def test_batch_writes_manifest_and_visualizer(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=2.0)
    clips = generate_batch(tmp_path / "batch", count=2, seed=100, config=config)

    manifest_path = tmp_path / "batch" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert len(clips) == 2
    assert len(rows) == 2
    assert not (tmp_path / "batch" / "preview.html").exists()
    visualizer = tmp_path / "batch" / "visualizer.html"
    assert visualizer.exists()
    visualizer_text = visualizer.read_text(encoding="utf-8")
    assert "Comet Audio Visualizer" in visualizer_text
    assert "Attack" in visualizer_text
    assert "Held" in visualizer_text
    assert "Release" in visualizer_text
    assert "source_000" in visualizer_text
    assert "ruler-scale" in visualizer_text
    assert "trackRect.left-timelineRect.left" in visualizer_text
    for index, row in enumerate(rows):
        clip_dir = tmp_path / "batch" / f"clip_{index:04d}"
        assert row["mix_path"] == f"clip_{index:04d}/mix.wav"
        assert (clip_dir / "mix.wav").exists()
        assert (clip_dir / "metadata.json").exists()
        assert row["time_signature"] in {"2/2", "3/4", "3/2", "7/4", "5/4", "2/4", "4/4"}
        assert row["source_count"] >= 4
        assert row["event_count"] > 0


def test_training_layout_writes_mix_metadata_and_manifest_only(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=2.0)
    clips = generate_batch(
        tmp_path / "training",
        count=2,
        seed=200,
        config=config,
        write_visualizer=False,
        write_stems=False,
        flat_layout=True,
    )

    root = tmp_path / "training"
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(clips) == 2
    assert len(rows) == 2
    assert not (root / "visualizer.html").exists()
    assert not (root / "stems").exists()
    for index, row in enumerate(rows):
        clip_id = f"clip_{index:04d}"
        assert row["mix_path"] == f"audio/{clip_id}.wav"
        assert row["metadata_path"] == f"metadata/{clip_id}.json"
        assert (root / row["mix_path"]).exists()
        assert (root / row["metadata_path"]).exists()
        metadata = json.loads((root / row["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["dataset_version"] == "comet-edm-v1"
        assert metadata["paths"]["mix"] == row["mix_path"]
        assert metadata["paths"]["metadata"] == row["metadata_path"]


def test_metadata_exposes_renderer_and_asset_label_fields(tmp_path: Path) -> None:
    config = GeneratorConfig(sample_rate=22_050, duration_seconds=1.0)
    metadata = generate_clip(tmp_path / "clip", seed=500, config=config)

    for source in metadata.sources:
        assert source.family
        assert source.instrument
        assert source.articulation
        assert source.renderer == "procedural_synth"
        assert source.asset_id is None


def test_asset_catalog_validation_and_wav_renderer_are_mono(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    stereo = np.column_stack(
        [
            np.sin(np.linspace(0, np.pi * 2, 512, dtype=np.float32)),
            np.cos(np.linspace(0, np.pi * 2, 512, dtype=np.float32)),
        ]
    )
    sf.write(asset_root / "kick.wav", stereo, 22_050)
    (asset_root / "catalog.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "asset_id": "test_kick",
                        "renderer": "wav_one_shot",
                        "family": "drums",
                        "source_type": "kick",
                        "instrument": "processed_909",
                        "articulation": "hit",
                        "path": "kick.wav",
                        "default_gain_db": -6.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert validate_asset_catalog(asset_root) == []
    config = GeneratorConfig(
        sample_rate=22_050,
        duration_seconds=1.0,
        source_count_min=1,
        source_count_max=1,
    )
    metadata = generate_clip(
        tmp_path / "clip",
        seed=501,
        config=config,
        assets=asset_root,
        procedural_fallback=False,
    )
    audio, sample_rate = sf.read(tmp_path / "clip" / "mix.wav")

    assert sample_rate == 22_050
    assert audio.ndim == 1
    assert metadata.sources[0].renderer == "wav_one_shot"
    assert metadata.sources[0].asset_id == "test_kick"


def test_sfz_parser_selects_region_by_note_and_velocity(tmp_path: Path) -> None:
    wav = np.sin(np.linspace(0, np.pi * 2, 256, dtype=np.float32))
    sf.write(tmp_path / "sample.wav", wav, 22_050)
    sfz = tmp_path / "instrument.sfz"
    sfz.write_text(
        "<group> lovel=1 hivel=127\n"
        "<region> sample=sample.wav lokey=36 hikey=48 pitch_keycenter=36 volume=-3\n"
        "<region> sample=sample.wav lokey=49 hikey=72 pitch_keycenter=60\n",
        encoding="utf-8",
    )

    regions, warnings = parse_sfz(sfz)
    selected = choose_sfz_region(regions, midi_note=60, velocity=0.8)

    assert warnings == []
    assert len(regions) == 2
    assert selected.lokey == 49
