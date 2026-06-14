from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from comet_audio.assets import AssetEntry, load_asset_catalog
from comet_audio.dsp import db_to_amp
from comet_audio.models import EventMetadata, SourceMetadata

DEFAULT_BUFFER_SIZE = 128


def render_dawdreamer_source(
    source: SourceMetadata,
    events: list[EventMetadata],
    sample_rate: int,
    duration: float,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> np.ndarray:
    plugin_path = Path(str(source.synth_parameters["plugin_path"]))
    preset_path = Path(str(source.synth_parameters["preset_path"]))
    _require_existing_file(plugin_path, "DawDreamer plugin")
    _require_existing_file(preset_path, "DawDreamer state")
    daw = _import_dawdreamer()

    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor(source.source_id, plugin_path.as_posix())
    if synth.load_state(preset_path.as_posix()) is False:
        raise RuntimeError(f"Failed to load DawDreamer state for {source.source_id}: {preset_path}")
    synth.clear_midi()
    for event in events:
        _add_event_note(synth, source, event)
    engine.load_graph([(synth, [])])
    engine.render(float(duration))
    audio = _as_mono(engine.get_audio(), sample_rate, duration)
    audio *= db_to_amp(source.gain_db)
    _require_finite_signal(audio, f"DawDreamer render for {source.source_id}")
    return audio.astype(np.float32)


def render_surge_fxp_source(
    source: SourceMetadata,
    events: list[EventMetadata],
    sample_rate: int,
    duration: float,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> np.ndarray:
    plugin_path = Path(str(source.synth_parameters["plugin_path"]))
    preset_path = Path(str(source.synth_parameters["preset_path"]))
    _require_existing_file(plugin_path, "Surge XT VST3 plugin")
    _require_existing_file(preset_path, "Surge XT preset")
    if preset_path.suffix.lower() != ".fxp":
        raise ValueError(f"Surge XT preset must be an .fxp file: {preset_path}")
    daw = _import_dawdreamer()

    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor(source.source_id, plugin_path.as_posix())
    load_preset = getattr(synth, "load_preset", None)
    if load_preset is None:
        raise RuntimeError("DawDreamer plugin processor does not support load_preset()")
    if load_preset(preset_path.as_posix()) is False:
        raise RuntimeError(f"Failed to load Surge XT preset for {source.source_id}: {preset_path}")
    _configure_surge_patch(synth, source)
    synth.clear_midi()
    for event in events:
        _add_event_note(synth, source, event)
    engine.load_graph([(synth, [])])
    engine.render(float(duration))
    audio = _as_mono(engine.get_audio(), sample_rate, duration)
    audio *= db_to_amp(source.gain_db)
    _require_finite_signal(audio, f"Surge XT render for {source.source_id}")
    return audio.astype(np.float32)


def capture_plugin_preset(
    assets_root: Path,
    plugin_path: Path,
    asset_id: str,
    source_type: str,
    family: str,
    instrument: str,
    articulation: str,
    tags: list[str],
    weight: float,
    preset_path: Path | None = None,
    catalog_path: Path | None = None,
    note_min: int | None = None,
    note_max: int | None = None,
    velocity_min: float = 0.0,
    velocity_max: float = 1.0,
    root_key: int | None = None,
    default_gain_db: float = 0.0,
    sample_rate: int = 44_100,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> AssetEntry:
    assets_root = assets_root.resolve()
    plugin_path = plugin_path.resolve()
    _require_existing_file(plugin_path, "DawDreamer plugin")
    if weight <= 0:
        raise ValueError("weight must be greater than 0")
    preset_path = preset_path or assets_root / "presets" / f"{asset_id}.dawdreamer"
    if not preset_path.is_absolute():
        preset_path = assets_root / preset_path
    preset_path.parent.mkdir(parents=True, exist_ok=True)

    daw = _import_dawdreamer()
    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor(asset_id, plugin_path.as_posix())
    synth.open_editor()
    if synth.save_state(preset_path.as_posix()) is False:
        raise RuntimeError(f"Failed to save DawDreamer state: {preset_path}")

    entry = AssetEntry(
        asset_id=asset_id,
        renderer="dawdreamer_plugin",
        family=family,
        source_type=source_type,
        instrument=instrument,
        articulation=articulation,
        path=_catalog_path(assets_root, plugin_path),
        preset_path=_catalog_path(assets_root, preset_path),
        tags=sorted(set(tags)),
        weight=weight,
        note_min=note_min,
        note_max=note_max,
        velocity_min=velocity_min,
        velocity_max=velocity_max,
        root_key=root_key,
        default_gain_db=default_gain_db,
    )
    _upsert_catalog_entry(catalog_path or assets_root / "catalog.json", entry)
    return entry


def import_surge_preset(
    assets_root: Path,
    plugin_path: Path,
    preset_file: Path,
    asset_id: str,
    source_type: str,
    family: str = "bass",
    instrument: str = "surge_xt",
    articulation: str = "held",
    tags: list[str] | None = None,
    weight: float = 1.0,
    catalog_path: Path | None = None,
    note_min: int | None = 24,
    note_max: int | None = 48,
    velocity_min: float = 0.0,
    velocity_max: float = 1.0,
    root_key: int | None = 36,
    default_gain_db: float = -10.0,
    sample_rate: int = 44_100,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    audition_note: int | None = None,
    audition_duration: float = 2.0,
) -> tuple[AssetEntry, Path]:
    """Import a Surge XT .fxp preset as a DawDreamer state-backed asset."""
    assets_root = assets_root.resolve()
    plugin_path = plugin_path.resolve()
    preset_file = preset_file.resolve()
    _require_existing_file(plugin_path, "Surge XT VST3 plugin")
    _require_existing_file(preset_file, "Surge XT preset")
    if preset_file.suffix.lower() != ".fxp":
        raise ValueError(f"Surge XT preset must be an .fxp file: {preset_file}")
    if weight <= 0:
        raise ValueError("weight must be greater than 0")

    state_path = assets_root / "presets" / f"{asset_id}.dawdreamer"
    imported_preset_path = assets_root / "imports" / "surge_xt" / f"{asset_id}.fxp"
    audition_path = assets_root / "auditions" / f"{asset_id}.wav"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    imported_preset_path.parent.mkdir(parents=True, exist_ok=True)

    daw = _import_dawdreamer()
    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor(asset_id, plugin_path.as_posix())
    load_preset = getattr(synth, "load_preset", None)
    if load_preset is None:
        raise RuntimeError("DawDreamer plugin processor does not support load_preset()")
    if load_preset(preset_file.as_posix()) is False:
        raise RuntimeError(f"Failed to load Surge XT preset: {preset_file}")
    if synth.save_state(state_path.as_posix()) is False:
        raise RuntimeError(f"Failed to save DawDreamer state: {state_path}")

    shutil.copy2(preset_file, imported_preset_path)
    entry = AssetEntry(
        asset_id=asset_id,
        renderer="dawdreamer_plugin",
        family=family,
        source_type=source_type,
        instrument=instrument,
        articulation=articulation,
        path=_catalog_path(assets_root, plugin_path),
        preset_path=_catalog_path(assets_root, state_path),
        tags=sorted(set(tags or [])),
        weight=weight,
        note_min=note_min,
        note_max=note_max,
        velocity_min=velocity_min,
        velocity_max=velocity_max,
        root_key=root_key,
        default_gain_db=default_gain_db,
        metadata={
            "importer": "surge_xt_fxp",
            "source_preset_path": _catalog_path(assets_root, imported_preset_path),
        },
    )

    audio = _render_entry_audition(
        entry=entry,
        plugin_path=plugin_path,
        preset_path=state_path,
        midi_note=audition_note if audition_note is not None else root_key or 36,
        velocity=0.9,
        duration=audition_duration,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
    )
    _require_finite_signal(audio, f"Surge XT import audition for {asset_id}")
    audition_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audition_path, audio, sample_rate, subtype="PCM_16")
    _upsert_catalog_entry(catalog_path or assets_root / "catalog.json", entry)
    return entry, audition_path


def audition_plugin_preset(
    assets_root: Path,
    asset_id: str,
    output_path: Path,
    midi_note: int = 60,
    velocity: float = 0.9,
    duration: float = 2.0,
    sample_rate: int = 44_100,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> Path:
    catalog = load_asset_catalog(assets_root)
    matches = [entry for entry in catalog.entries if entry.asset_id == asset_id]
    if not matches:
        raise ValueError(f"No asset entry found for {asset_id!r}")
    entry = matches[0]
    if entry.renderer != "dawdreamer_plugin":
        raise ValueError(f"{asset_id!r} uses renderer {entry.renderer!r}, not 'dawdreamer_plugin'")
    plugin_path = catalog.resolve_path(entry)
    preset_path = catalog.resolve_preset_path(entry)
    if preset_path is None:
        raise ValueError(f"{asset_id!r} is missing preset_path")
    source = SourceMetadata(
        source_id="audition",
        source_type=entry.source_type,
        family=entry.family,
        instrument=entry.instrument,
        articulation=entry.articulation,
        renderer=entry.renderer,
        asset_id=entry.asset_id,
        synth_parameters={
            "plugin_path": str(plugin_path),
            "preset_path": str(preset_path),
            "asset_root_key": entry.root_key,
            "asset_note_min": entry.note_min,
            "asset_note_max": entry.note_max,
        },
        effect_parameters={},
        gain_db=entry.default_gain_db,
        pan=0.0,
        stem_path="audition.wav",
    )
    event = EventMetadata(
        event_id="audition_0000",
        source_id="audition",
        event_type=entry.source_type,
        onset_seconds=0.0,
        offset_seconds=duration,
        velocity=float(velocity),
        midi_note=midi_note,
        attack_seconds=0.005,
        release_seconds=0.08,
    )
    audio = render_dawdreamer_source(source, [event], sample_rate, duration, buffer_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return output_path


def _render_entry_audition(
    entry: AssetEntry,
    plugin_path: Path,
    preset_path: Path,
    midi_note: int,
    velocity: float,
    duration: float,
    sample_rate: int,
    buffer_size: int,
) -> np.ndarray:
    source = SourceMetadata(
        source_id="audition",
        source_type=entry.source_type,
        family=entry.family,
        instrument=entry.instrument,
        articulation=entry.articulation,
        renderer=entry.renderer,
        asset_id=entry.asset_id,
        synth_parameters={
            "plugin_path": str(plugin_path),
            "preset_path": str(preset_path),
            "asset_root_key": entry.root_key,
            "asset_note_min": entry.note_min,
            "asset_note_max": entry.note_max,
        },
        effect_parameters={},
        gain_db=entry.default_gain_db,
        pan=0.0,
        stem_path="audition.wav",
    )
    event = EventMetadata(
        event_id="audition_0000",
        source_id="audition",
        event_type=entry.source_type,
        onset_seconds=0.0,
        offset_seconds=duration,
        velocity=float(velocity),
        midi_note=midi_note,
        attack_seconds=0.005,
        release_seconds=0.08,
    )
    return render_dawdreamer_source(source, [event], sample_rate, duration, buffer_size)


def _add_event_note(synth: Any, source: SourceMetadata, event: EventMetadata) -> None:
    note = int(event.midi_note or source.synth_parameters.get("asset_root_key") or 60)
    velocity = int(round(np.clip(float(event.velocity), 0.0, 1.0) * 127))
    start_time = float(event.onset_seconds)
    note_off = max(start_time, float(event.offset_seconds) - float(event.release_seconds))
    synth.add_midi_note(note, velocity, start_time, max(0.001, note_off - start_time))


def _configure_surge_patch(synth: Any, source: SourceMetadata) -> None:
    _set_first_available_parameter(synth, ("FX Disable", "FX Bypass"), 1.0)
    if _has_parameter(synth, "FX Chain Bypass"):
        _set_parameter_by_name(synth, "FX Chain Bypass", 1.0)
    attack = float(source.synth_parameters["surge_amp_attack_seconds"])
    release = float(source.synth_parameters["surge_amp_release_seconds"])
    if _has_parameter(synth, "A Amp EG Attack"):
        for name in ("A Amp EG Attack", "B Amp EG Attack"):
            _set_parameter_seconds_by_name(synth, name, attack)
        for name in ("A Amp EG Release", "B Amp EG Release"):
            _set_parameter_seconds_by_name(synth, name, release)
    else:
        _set_all_parameters_seconds_by_name(synth, "AEG Attack", attack)
        _set_all_parameters_seconds_by_name(synth, "AEG Release", release)


def _set_first_available_parameter(synth: Any, names: tuple[str, ...], value: float) -> None:
    for name in names:
        if _has_parameter(synth, name):
            _set_parameter_by_name(synth, name, value)
            return
    raise RuntimeError(f"None of the Surge parameters were found: {', '.join(names)}")


def _set_parameter_by_name(synth: Any, name: str, value: float) -> None:
    index = _parameter_index(synth, name)
    synth.set_parameter(index, float(np.clip(value, 0.0, 1.0)))


def _set_parameter_seconds_by_name(synth: Any, name: str, seconds: float) -> None:
    index = _parameter_index(synth, name)
    _set_parameter_seconds(synth, index, name, seconds)


def _set_all_parameters_seconds_by_name(synth: Any, name: str, seconds: float) -> None:
    indices = _parameter_indices(synth, name)
    if not indices:
        raise RuntimeError(f"Surge parameter {name!r} was not found")
    for index in indices:
        _set_parameter_seconds(synth, index, name, seconds)


def _set_parameter_seconds(synth: Any, index: int, name: str, seconds: float) -> None:
    ranges = synth.get_parameter_range(index)
    if not isinstance(ranges, dict) or not ranges:
        raise RuntimeError(f"Surge parameter {name!r} does not expose a second-based range")
    best_value: float | None = None
    best_error: float | None = None
    for normalized_range, real_value in ranges.items():
        if not isinstance(normalized_range, tuple) or len(normalized_range) != 2:
            continue
        low, high = float(normalized_range[0]), float(normalized_range[1])
        candidate = (low + high) * 0.5
        error = abs(float(real_value) - seconds)
        if best_error is None or error < best_error:
            best_value = candidate
            best_error = error
    if best_value is None:
        raise RuntimeError(f"Surge parameter {name!r} has an unsupported range shape")
    synth.set_parameter(index, float(np.clip(best_value, 0.0, 1.0)))


def _parameter_index(synth: Any, name: str) -> int:
    indices = _parameter_indices(synth, name)
    if indices:
        return indices[0]
    raise RuntimeError(f"Surge parameter {name!r} was not found")


def _parameter_indices(synth: Any, name: str) -> list[int]:
    return [
        index
        for index in range(int(synth.get_plugin_parameter_size()))
        if synth.get_parameter_name(index) == name
    ]


def _has_parameter(synth: Any, name: str) -> bool:
    return bool(_parameter_indices(synth, name))


def _as_mono(audio: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
    rendered = np.asarray(audio, dtype=np.float32)
    if rendered.ndim == 2:
        if rendered.shape[0] <= rendered.shape[1]:
            rendered = rendered.mean(axis=0)
        else:
            rendered = rendered.mean(axis=1)
    if rendered.ndim != 1:
        raise RuntimeError(f"DawDreamer returned audio with unsupported shape {rendered.shape}")
    expected = int(round(sample_rate * duration))
    if len(rendered) < expected:
        rendered = np.pad(rendered, (0, expected - len(rendered)))
    return rendered[:expected]


def _require_finite_signal(audio: np.ndarray, context: str) -> None:
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{context} produced non-finite audio")
    if float(np.max(np.abs(audio))) <= 1e-8:
        raise RuntimeError(f"{context} produced silence")


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _import_dawdreamer() -> Any:
    try:
        import dawdreamer as daw
    except ImportError as exc:
        raise RuntimeError("DawDreamer is required for dawdreamer_plugin rendering") from exc
    return daw


def _catalog_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _upsert_catalog_entry(catalog_path: Path, entry: AssetEntry) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if catalog_path.exists():
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        payload = {"assets": []}
    rows = payload if isinstance(payload, list) else payload.get("assets", [])
    next_rows = [row for row in rows if AssetEntry.model_validate(row).asset_id != entry.asset_id]
    next_rows.append(entry.model_dump(mode="json", exclude_none=True))
    next_rows.sort(key=lambda row: row["asset_id"])
    if isinstance(payload, list):
        output = next_rows
    else:
        payload["assets"] = next_rows
        output = payload
    catalog_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
