from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer

from comet_audio.assets import (
    SURGE_PATCH_LIBRARY_ROOTS,
    import_percussion_samples,
    index_surge_patches,
    validate_asset_catalog,
)
from comet_audio.dawdreamer_renderer import (
    audition_plugin_preset,
    capture_plugin_preset,
    import_surge_preset,
)
from comet_audio.generator import DEFAULT_TIME_SIGNATURES, GeneratorConfig, generate_batch
from comet_audio.inference import predict_song
from comet_audio.training import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, evaluate_model, train_model

app = typer.Typer(help="Generate synthetic EDM clips and labels.")
assets_app = typer.Typer(help="Asset library tools.")
synth_app = typer.Typer(help="DawDreamer synth preset tools.")
app.add_typer(assets_app, name="assets")
app.add_typer(synth_app, name="synth")
DEFAULT_OUT = Path("data/generated/demo")
DEFAULT_SURGE_STANDALONE = Path(r"C:\Program Files\Surge Synth Team\Surge XT\Surge XT.exe")
DEFAULT_SURGE_PLUGIN = Path(r"C:\Program Files\Common Files\VST3\Surge Synth Team\Surge XT.vst3")
DEFAULT_SURGE_DOCUMENTS = Path.home() / "Documents" / "Surge XT"
SURGE_DEBOUNCE_SECONDS = 1.5


@app.callback()
def main() -> None:
    """Comet synthetic audio dataset tools."""


@app.command()
def generate(
    count: Annotated[int, typer.Option(min=1, help="Number of clips to generate.")] = 4,
    seed: Annotated[int, typer.Option(help="Base random seed. Clip N uses seed + N.")] = 123,
    out: Annotated[Path, typer.Option(help="Output directory.")] = DEFAULT_OUT,
    duration: Annotated[float, typer.Option(min=1.0, help="Clip duration in seconds.")] = 8.0,
    sample_rate: Annotated[int, typer.Option(min=8000, help="Sample rate in Hz.")] = 44_100,
    bpm_min: Annotated[int, typer.Option(min=1, help="Minimum generated BPM.")] = 70,
    bpm_max: Annotated[int, typer.Option(min=1, help="Maximum generated BPM.")] = 150,
    source_count_min: Annotated[int, typer.Option(min=1, help="Minimum sources per clip.")] = 5,
    source_count_max: Annotated[int, typer.Option(min=1, help="Maximum sources per clip.")] = 10,
    time_signature: Annotated[
        list[str] | None,
        typer.Option(
            "--time-signature",
            "-t",
            help="Allowed meter. Repeat to provide a pool, e.g. -t 3/4 -t 7/4.",
        ),
    ] = None,
    visualizer: Annotated[
        bool,
        typer.Option("--visualizer/--no-visualizer", help="Write visualizer.html."),
    ] = True,
    stems: Annotated[bool, typer.Option(help="Write per-source stem WAV files.")] = True,
    training_layout: Annotated[
        bool,
        typer.Option(
            help="Write one dataset root with audio/, metadata/, and manifest.jsonl paths.",
        ),
    ] = False,
    assets: Annotated[
        Path | None,
        typer.Option(help="Asset catalog root containing JSON/JSONL asset entries."),
    ] = None,
    renderer_profile: Annotated[
        str,
        typer.Option(help="Renderer profile: hybrid_v1, procedural_only, or plugin_v1."),
    ] = "hybrid_v1",
    include_tag: Annotated[
        list[str] | None,
        typer.Option("--include-tag", help="Require catalog asset tag. Repeatable."),
    ] = None,
    exclude_tag: Annotated[
        list[str] | None,
        typer.Option("--exclude-tag", help="Exclude catalog asset tag. Repeatable."),
    ] = None,
    procedural_fallback: Annotated[
        bool,
        typer.Option(
            "--procedural-fallback/--no-procedural-fallback",
            help="Allow procedural sources when no matching asset exists.",
        ),
    ] = True,
    composition_profile: Annotated[
        str,
        typer.Option(help="Composition profile: edm_v1, percussion_v1, or surge_patches_v1."),
    ] = "edm_v1",
) -> None:
    """Generate a batch of labeled procedural EDM clips."""
    if composition_profile not in {"edm_v1", "percussion_v1", "surge_patches_v1"}:
        raise typer.BadParameter(
            "composition_profile must be one of: edm_v1, percussion_v1, surge_patches_v1"
        )
    time_signatures = tuple(time_signature) if time_signature else DEFAULT_TIME_SIGNATURES
    config = GeneratorConfig(
        sample_rate=sample_rate,
        duration_seconds=duration,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        time_signatures=time_signatures,
        source_count_min=source_count_min,
        source_count_max=source_count_max,
        composition_profile=composition_profile,
    )
    clips = generate_batch(
        out,
        count=count,
        seed=seed,
        config=config,
        write_visualizer=visualizer,
        write_stems=stems,
        flat_layout=training_layout,
        assets=assets,
        renderer_profile=renderer_profile,
        procedural_fallback=procedural_fallback,
        include_tags=tuple(include_tag or ()),
        exclude_tags=tuple(exclude_tag or ()),
    )
    typer.echo(f"Generated {len(clips)} clips in {out}")


@assets_app.command("validate")
def validate_assets(
    assets: Annotated[Path, typer.Option(help="Asset catalog root to validate.")] = Path(
        "assets/library"
    ),
) -> None:
    """Validate asset catalog entries and referenced audio files."""
    warnings = validate_asset_catalog(assets)
    if warnings:
        for warning in warnings:
            typer.echo(f"warning: {warning}")
        raise typer.Exit(code=1)
    typer.echo(f"Asset catalog OK: {assets}")


@assets_app.command("import-percussion")
def import_percussion(
    source: Annotated[Path, typer.Option(help="Local unzipped sample folder to scan.")],
    assets: Annotated[Path, typer.Option(help="Asset catalog root to update.")] = Path(
        "assets/library"
    ),
    pack_id: Annotated[str, typer.Option(help="Stable pack ID, e.g. 99sounds_drums_1.")] = ...,
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="Catalog JSON to update. Defaults to assets/catalog.json."),
    ] = None,
    skip_loops: Annotated[
        bool,
        typer.Option("--skip-loops/--include-loops", help="Reject files that look like loops."),
    ] = True,
    max_duration: Annotated[
        float,
        typer.Option(min=0.05, help="Reject one-shots longer than this many seconds."),
    ] = 4.0,
    sample_rate: Annotated[int, typer.Option(min=8000)] = 44_100,
) -> None:
    """Import local WAV percussion one-shots into the asset catalog."""
    report = import_percussion_samples(
        source_dir=source,
        assets_root=assets,
        pack_id=pack_id,
        catalog_path=catalog_path,
        skip_loops=skip_loops,
        sample_rate=sample_rate,
        max_duration_seconds=max_duration,
    )
    typer.echo(
        f"Imported {report.imported_count} samples into {assets}; "
        f"duplicates={report.duplicate_count}; rejected={len(report.rejected)}"
    )
    if report.ambiguous:
        typer.echo(f"Ambiguous bucket defaults: {len(report.ambiguous)}")
    for bucket, count in report.bucket_counts.items():
        typer.echo(f"  {bucket}: {count}")
    typer.echo(f"Wrote report {report.report_path}")


@synth_app.command("capture")
def synth_capture(
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root to update."),
    ] = Path("assets/library"),
    plugin: Annotated[Path, typer.Option(help="VST/AU plugin path.")] = ...,
    asset_id: Annotated[str, typer.Option(help="Catalog asset_id to create or update.")] = ...,
    source_type: Annotated[str, typer.Option(help="Comet source_type for this preset.")] = ...,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tags", help="Catalog preset tag. Repeatable."),
    ] = None,
    weight: Annotated[float, typer.Option(min=0.0001, help="Weighted selection value.")] = 1.0,
    family: Annotated[str, typer.Option(help="Source family metadata.")] = "unknown",
    instrument: Annotated[
        str | None,
        typer.Option(help="Instrument metadata. Defaults to asset_id."),
    ] = None,
    articulation: Annotated[str, typer.Option(help="Articulation metadata.")] = "unknown",
    preset_path: Annotated[
        Path | None,
        typer.Option(help="State path relative to assets root, or absolute."),
    ] = None,
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="Catalog JSON to update. Defaults to assets/catalog.json."),
    ] = None,
    note_min: Annotated[int | None, typer.Option(min=0, max=127)] = None,
    note_max: Annotated[int | None, typer.Option(min=0, max=127)] = None,
    velocity_min: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    velocity_max: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
    root_key: Annotated[int | None, typer.Option(min=0, max=127)] = None,
    default_gain_db: Annotated[float, typer.Option(help="Source gain for this preset.")] = 0.0,
    sample_rate: Annotated[int, typer.Option(min=8000)] = 44_100,
) -> None:
    """Open a synth GUI, save a DawDreamer state file, and upsert a catalog entry."""
    entry = capture_plugin_preset(
        assets_root=assets,
        plugin_path=plugin,
        asset_id=asset_id,
        source_type=source_type,
        family=family,
        instrument=instrument or asset_id,
        articulation=articulation,
        tags=tags or [],
        weight=weight,
        preset_path=preset_path,
        catalog_path=catalog_path,
        note_min=note_min,
        note_max=note_max,
        velocity_min=velocity_min,
        velocity_max=velocity_max,
        root_key=root_key,
        default_gain_db=default_gain_db,
        sample_rate=sample_rate,
    )
    typer.echo(f"Captured {entry.asset_id} in {assets}")


@synth_app.command("audition")
def synth_audition(
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root containing the preset entry."),
    ] = Path("assets/library"),
    asset_id: Annotated[str, typer.Option(help="Catalog asset_id to render.")] = ...,
    out: Annotated[Path, typer.Option(help="Output WAV path.")] = Path(
        "data/generated/audition.wav"
    ),
    midi_note: Annotated[int, typer.Option(min=0, max=127)] = 60,
    velocity: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.9,
    duration: Annotated[float, typer.Option(min=0.1)] = 2.0,
    sample_rate: Annotated[int, typer.Option(min=8000)] = 44_100,
) -> None:
    """Render one cataloged DawDreamer preset to a mono WAV."""
    output_path = audition_plugin_preset(
        assets_root=assets,
        asset_id=asset_id,
        output_path=out,
        midi_note=midi_note,
        velocity=velocity,
        duration=duration,
        sample_rate=sample_rate,
    )
    typer.echo(f"Wrote {output_path}")


@synth_app.command("import-surge")
def synth_import_surge(
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root to update."),
    ] = Path("assets/library"),
    plugin: Annotated[
        Path,
        typer.Option(help="Surge XT VST3 path."),
    ] = DEFAULT_SURGE_PLUGIN,
    preset_file: Annotated[Path, typer.Option(help="Surge XT .fxp preset to import.")] = ...,
    asset_id: Annotated[str, typer.Option(help="Catalog asset_id to create or update.")] = ...,
    source_type: Annotated[
        str, typer.Option(help="Comet source_type for this preset.")
    ] = "synth_bass",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tags", help="Catalog preset tag. Repeatable."),
    ] = None,
    weight: Annotated[float, typer.Option(min=0.0001, help="Weighted selection value.")] = 1.0,
    family: Annotated[str, typer.Option(help="Source family metadata.")] = "bass",
    instrument: Annotated[str, typer.Option(help="Instrument metadata.")] = "surge_xt",
    articulation: Annotated[str, typer.Option(help="Articulation metadata.")] = "held",
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="Catalog JSON to update. Defaults to assets/catalog.json."),
    ] = None,
    note_min: Annotated[int | None, typer.Option(min=0, max=127)] = 24,
    note_max: Annotated[int | None, typer.Option(min=0, max=127)] = 48,
    root_key: Annotated[int | None, typer.Option(min=0, max=127)] = 36,
    default_gain_db: Annotated[float, typer.Option(help="Source gain for this preset.")] = -10.0,
    sample_rate: Annotated[int, typer.Option(min=8000)] = 44_100,
) -> None:
    """Import a saved Surge XT .fxp as a DawDreamer state-backed catalog asset."""
    entry, audition_path = import_surge_preset(
        assets_root=assets,
        plugin_path=plugin,
        preset_file=preset_file,
        asset_id=asset_id,
        source_type=source_type,
        family=family,
        instrument=instrument,
        articulation=articulation,
        tags=tags or [],
        weight=weight,
        catalog_path=catalog_path,
        note_min=note_min,
        note_max=note_max,
        root_key=root_key,
        default_gain_db=default_gain_db,
        sample_rate=sample_rate,
    )
    typer.echo(f"Imported {entry.asset_id}; wrote audition {audition_path}")


@synth_app.command("index-surge-patches")
def synth_index_surge_patches(
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root to update."),
    ] = Path("assets/library"),
    plugin: Annotated[
        Path,
        typer.Option(help="Surge XT VST3 path."),
    ] = DEFAULT_SURGE_PLUGIN,
    factory_root: Annotated[
        Path,
        typer.Option(help="Surge XT factory patch root."),
    ] = SURGE_PATCH_LIBRARY_ROOTS[0][1],
    thirdparty_root: Annotated[
        Path,
        typer.Option(help="Surge XT third-party patch root."),
    ] = SURGE_PATCH_LIBRARY_ROOTS[1][1],
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="Catalog JSON to update. Defaults to assets/catalog.json."),
    ] = None,
    asset_prefix: Annotated[str, typer.Option(help="Prefix for generated asset IDs.")] = "surge_xt",
) -> None:
    """Index installed Surge XT .fxp patches by reference without copying presets."""
    entries = index_surge_patches(
        assets_root=assets,
        plugin_path=plugin,
        patch_roots=(("factory", factory_root), ("thirdparty", thirdparty_root)),
        catalog_path=catalog_path,
        asset_prefix=asset_prefix,
    )
    counts: dict[str, int] = {}
    for entry in entries:
        category = str(entry.metadata.get("surge_category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    typer.echo(f"Indexed {len(entries)} Surge XT patches into {assets}")
    for category, count in sorted(counts.items()):
        typer.echo(f"  {category}: {count}")


@synth_app.command("studio")
def synth_studio(
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root to update."),
    ] = Path("assets/library"),
    standalone: Annotated[
        Path,
        typer.Option(help="Surge XT standalone executable path."),
    ] = DEFAULT_SURGE_STANDALONE,
    plugin: Annotated[
        Path,
        typer.Option(help="Surge XT VST3 path."),
    ] = DEFAULT_SURGE_PLUGIN,
    watch_dir: Annotated[
        list[Path] | None,
        typer.Option("--watch-dir", help="Additional folder to watch recursively for .fxp files."),
    ] = None,
    asset_prefix: Annotated[str, typer.Option(help="Prefix for generated asset IDs.")] = "surge",
    source_type: Annotated[
        str, typer.Option(help="Comet source_type for imported presets.")
    ] = "synth_bass",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tags", help="Catalog preset tag. Repeatable."),
    ] = None,
    weight: Annotated[float, typer.Option(min=0.0001, help="Weighted selection value.")] = 1.0,
    family: Annotated[str, typer.Option(help="Source family metadata.")] = "bass",
    instrument: Annotated[str, typer.Option(help="Instrument metadata.")] = "surge_xt",
    articulation: Annotated[str, typer.Option(help="Articulation metadata.")] = "held",
    note_min: Annotated[int | None, typer.Option(min=0, max=127)] = 24,
    note_max: Annotated[int | None, typer.Option(min=0, max=127)] = 48,
    root_key: Annotated[int | None, typer.Option(min=0, max=127)] = 36,
    default_gain_db: Annotated[
        float, typer.Option(help="Source gain for imported presets.")
    ] = -10.0,
    sample_rate: Annotated[int, typer.Option(min=8000)] = 44_100,
) -> None:
    """Launch Surge XT standalone and import saved .fxp patches as catalog assets."""
    if not standalone.exists():
        raise typer.BadParameter(f"Surge XT standalone does not exist: {standalone}")
    if not plugin.exists():
        raise typer.BadParameter(f"Surge XT VST3 does not exist: {plugin}")

    assets = assets.resolve()
    inbox = assets / "imports" / "surge_xt" / "inbox"
    watch_roots = _unique_paths([DEFAULT_SURGE_DOCUMENTS, inbox, *(watch_dir or [])])
    for root in watch_roots:
        root.mkdir(parents=True, exist_ok=True)

    session_start = time.time()
    process = subprocess.Popen([standalone.as_posix()])
    typer.echo(f"Launched Surge XT: {standalone}")
    typer.echo("Watching for saved .fxp files:")
    for root in watch_roots:
        typer.echo(f"  {root}")
    typer.echo("Open Surge XT's virtual keyboard with Alt+K, then save patches as .fxp.")

    observed: dict[Path, tuple[int, int, float]] = {}
    imported: dict[Path, tuple[int, int]] = {}
    try:
        while process.poll() is None:
            _import_ready_surge_files(
                assets=assets,
                plugin=plugin,
                watch_roots=watch_roots,
                observed=observed,
                imported=imported,
                session_start=session_start,
                asset_prefix=asset_prefix,
                source_type=source_type,
                family=family,
                instrument=instrument,
                articulation=articulation,
                tags=tags or [],
                weight=weight,
                note_min=note_min,
                note_max=note_max,
                root_key=root_key,
                default_gain_db=default_gain_db,
                sample_rate=sample_rate,
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        typer.echo("Stopping watcher. Surge XT was left running so unsaved work is preserved.")
        return
    typer.echo("Surge XT exited; watcher stopped.")


@app.command("synth-studio")
def synth_studio_shortcut(
    source_type: Annotated[
        str,
        typer.Option(help="Comet source_type for imported presets."),
    ] = "synth_lead",
    asset_prefix: Annotated[
        str,
        typer.Option(help="Prefix for generated asset IDs."),
    ] = "surge_synth",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tags", help="Additional catalog tag. Repeatable."),
    ] = None,
    assets: Annotated[
        Path,
        typer.Option(help="Asset catalog root to update."),
    ] = Path("assets/library"),
    standalone: Annotated[
        Path,
        typer.Option(help="Surge XT standalone executable path."),
    ] = DEFAULT_SURGE_STANDALONE,
    plugin: Annotated[
        Path,
        typer.Option(help="Surge XT VST3 path."),
    ] = DEFAULT_SURGE_PLUGIN,
) -> None:
    """Launch Surge XT with simple defaults for making synth training presets."""
    synth_studio(
        assets=assets,
        standalone=standalone,
        plugin=plugin,
        watch_dir=None,
        asset_prefix=asset_prefix,
        source_type=source_type,
        tags=["synth", "surge", *(tags or [])],
        weight=1.0,
        family="synth",
        instrument="surge_xt",
        articulation="held",
        note_min=36,
        note_max=84,
        root_key=60,
        default_gain_db=-10.0,
        sample_rate=44_100,
    )


def _import_ready_surge_files(
    assets: Path,
    plugin: Path,
    watch_roots: list[Path],
    observed: dict[Path, tuple[int, int, float]],
    imported: dict[Path, tuple[int, int]],
    session_start: float,
    asset_prefix: str,
    source_type: str,
    family: str,
    instrument: str,
    articulation: str,
    tags: list[str],
    weight: float,
    note_min: int | None,
    note_max: int | None,
    root_key: int | None,
    default_gain_db: float,
    sample_rate: int,
) -> None:
    now = time.time()
    for preset_file in _iter_fxp_files(watch_roots):
        try:
            stat = preset_file.stat()
        except OSError:
            continue
        if stat.st_mtime < session_start:
            continue
        signature = (stat.st_size, stat.st_mtime_ns)
        observed_signature = observed.get(preset_file)
        if observed_signature is None or observed_signature[:2] != signature:
            observed[preset_file] = (*signature, now)
            continue
        if now - observed_signature[2] < SURGE_DEBOUNCE_SECONDS:
            continue
        if imported.get(preset_file) == signature:
            continue

        asset_id = f"{asset_prefix}_{_slugify_asset_id(preset_file.stem)}"
        try:
            entry, audition_path = import_surge_preset(
                assets_root=assets,
                plugin_path=plugin,
                preset_file=preset_file,
                asset_id=asset_id,
                source_type=source_type,
                family=family,
                instrument=instrument,
                articulation=articulation,
                tags=tags,
                weight=weight,
                note_min=note_min,
                note_max=note_max,
                root_key=root_key,
                default_gain_db=default_gain_db,
                sample_rate=sample_rate,
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Failed to import {preset_file}: {exc}", err=True)
            imported[preset_file] = signature
            continue
        imported[preset_file] = signature
        typer.echo(f"Imported {entry.asset_id}; wrote audition {audition_path}")


def _iter_fxp_files(watch_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in watch_roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*.fxp") if path.is_file())
    return sorted(set(files))


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _slugify_asset_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "preset"


@app.command()
def train(
    data: Annotated[Path, typer.Option(help="Training dataset root.")] = Path(
        "data/generated/train_10k"
    ),
    run: Annotated[Path, typer.Option(help="Run directory.")] = Path("runs/cnn_tcn_v1"),
    epochs: Annotated[int, typer.Option(min=1, help="Training epochs.")] = DEFAULT_EPOCHS,
    batch_size: Annotated[int, typer.Option(min=1, help="Batch size.")] = DEFAULT_BATCH_SIZE,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Optional per-split item limit for smoke tests."),
    ] = None,
    learning_rate: Annotated[float, typer.Option(min=1e-7, help="AdamW learning rate.")] = 2e-4,
    target: Annotated[
        str,
        typer.Option(help="Training target: source_types_v1 or anonymous_slots_v1."),
    ] = "source_types_v1",
    max_tracks: Annotated[
        int,
        typer.Option(min=1, help="Maximum anonymous track slots for anonymous_slots_v1."),
    ] = 16,
) -> None:
    """Train the CNN+TCN V1 global event-timing detector."""
    if target not in {"source_types_v1", "anonymous_slots_v1"}:
        raise typer.BadParameter("target must be one of: source_types_v1, anonymous_slots_v1")
    train_model(
        data_dir=data,
        run_dir=run,
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        learning_rate=learning_rate,
        target=target,  # type: ignore[arg-type]
        max_tracks=max_tracks,
    )
    typer.echo(f"Training complete. Checkpoints and metrics written to {run}")


@app.command()
def evaluate(
    data: Annotated[Path, typer.Option(help="Dataset root.")] = Path("data/generated/train_10k"),
    run: Annotated[Path, typer.Option(help="Run directory.")] = Path("runs/cnn_tcn_v1"),
    split: Annotated[str, typer.Option(help="Split to evaluate: train, val, or test.")] = "test",
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Optional item limit for smoke tests."),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="Override onset threshold."),
    ] = None,
) -> None:
    """Evaluate a trained CNN+TCN V1 timing detector."""
    if split not in {"train", "val", "test"}:
        raise typer.BadParameter("split must be one of: train, val, test")
    metrics = evaluate_model(
        data_dir=data,
        run_dir=run,
        split=split,
        limit=limit,
        threshold=threshold,
    )
    typer.echo(f"Wrote evaluation for {split} to {run / f'eval_{split}.json'}")
    typer.echo(
        " ".join(
            [
                f"F1@5ms={metrics['onset_f1_5ms']:.4f}",
                f"F1@10ms={metrics['onset_f1_10ms']:.4f}",
                f"F1@25ms={metrics['onset_f1_25ms']:.4f}",
            ]
        )
    )


@app.command("predict-song")
def predict_song_command(
    audio: Annotated[Path, typer.Option(help="Song/audio file to analyze.")] = ...,
    run: Annotated[Path, typer.Option(help="Run directory containing best.pt or last.pt.")] = Path(
        "runs/cnn_tcn_v1"
    ),
    out: Annotated[Path, typer.Option(help="Output directory for predictions and HTML.")] = Path(
        "data/generated/song_predictions"
    ),
    threshold: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Global onset probability threshold."),
    ] = 0.35,
    nms_seconds: Annotated[
        float,
        typer.Option(min=0.0, help="Suppress duplicate decoded onsets within this window."),
    ] = 0.025,
    source_threshold: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Source lane probability threshold."),
    ] = 0.35,
    max_waveform_points: Annotated[
        int,
        typer.Option(min=100, help="Maximum downsampled waveform peak points embedded in HTML."),
    ] = 2000,
) -> None:
    """Run a trained timing model on a song and write an interactive prediction viewer."""
    json_path, html_path = predict_song(
        audio_path=audio,
        run_dir=run,
        out_dir=out,
        threshold=threshold,
        nms_seconds=nms_seconds,
        source_threshold=source_threshold,
        max_waveform_points=max_waveform_points,
    )
    typer.echo(f"Wrote {json_path}")
    typer.echo(f"Wrote {html_path}")
