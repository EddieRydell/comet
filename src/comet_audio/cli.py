from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from comet_audio.generator import DEFAULT_TIME_SIGNATURES, GeneratorConfig, generate_batch
from comet_audio.training import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, evaluate_model, train_model

app = typer.Typer(help="Generate synthetic EDM clips and labels.")
DEFAULT_OUT = Path("data/generated/demo")


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
    preview: Annotated[bool, typer.Option(help="Write preview.html with mix audio tags.")] = True,
    stems: Annotated[bool, typer.Option(help="Write per-source stem WAV files.")] = True,
    training_layout: Annotated[
        bool,
        typer.Option(
            help="Write one dataset root with audio/, metadata/, and manifest.jsonl paths.",
        ),
    ] = False,
) -> None:
    """Generate a batch of labeled procedural EDM clips."""
    time_signatures = tuple(time_signature) if time_signature else DEFAULT_TIME_SIGNATURES
    config = GeneratorConfig(
        sample_rate=sample_rate,
        duration_seconds=duration,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        time_signatures=time_signatures,
        source_count_min=source_count_min,
        source_count_max=source_count_max,
    )
    clips = generate_batch(
        out,
        count=count,
        seed=seed,
        config=config,
        write_preview=preview,
        write_stems=stems,
        flat_layout=training_layout,
    )
    typer.echo(f"Generated {len(clips)} clips in {out}")


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
) -> None:
    """Train the CNN+TCN V1 global event-timing detector."""
    train_model(
        data_dir=data,
        run_dir=run,
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        learning_rate=learning_rate,
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
