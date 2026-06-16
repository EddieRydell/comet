from __future__ import annotations

import argparse
from pathlib import Path

from comet_audio.generator import DEFAULT_TIME_SIGNATURES, GeneratorConfig, generate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Comet training datasets.")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("data/generated/demo"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--bpm-min", type=int, default=70)
    parser.add_argument("--bpm-max", type=int, default=150)
    parser.add_argument("--source-count-min", type=int, default=5)
    parser.add_argument("--source-count-max", type=int, default=10)
    parser.add_argument("--time-signature", action="append", default=None)
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--renderer-profile", default="hybrid_v1")
    parser.add_argument("--composition-profile", default="edm_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-procedural-fallback", action="store_true")
    parser.add_argument("--training-layout", action="store_true")
    parser.add_argument("--no-stems", action="store_true")
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.composition_profile not in {"edm_v1", "percussion_v1", "surge_patches_v1"}:
        raise ValueError("composition profile must be edm_v1, percussion_v1, or surge_patches_v1")

    config = GeneratorConfig(
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        time_signatures=tuple(args.time_signature or DEFAULT_TIME_SIGNATURES),
        source_count_min=args.source_count_min,
        source_count_max=args.source_count_max,
        composition_profile=args.composition_profile,
    )
    clips = generate_batch(
        args.out,
        count=args.count,
        seed=args.seed,
        config=config,
        write_stems=not args.no_stems,
        flat_layout=args.training_layout,
        assets=args.assets,
        renderer_profile=args.renderer_profile,
        procedural_fallback=not args.no_procedural_fallback,
        workers=args.workers,
    )
    print(f"Generated {len(clips)} clips in {args.out}")


if __name__ == "__main__":
    main()
