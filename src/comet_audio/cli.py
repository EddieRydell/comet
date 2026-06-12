from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from comet_audio.generator import DEFAULT_TIME_SIGNATURES, GeneratorConfig, generate_batch

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
    clips = generate_batch(out, count=count, seed=seed, config=config, write_preview=preview)
    typer.echo(f"Generated {len(clips)} clips in {out}")
