# Comet Audio

Comet Audio is a procedural, plugin-free synthetic EDM dataset generator. Version 0 creates short labeled clips for onset and event-oriented research: a final mono mix WAV, wet mono per-source stems, metadata JSON, and a batch manifest.

## Setup

```powershell
uv sync
```

The project is pinned to Python 3.11 through `.python-version`.

## Generate A Demo Batch

```powershell
uv run comet generate --count 4 --seed 123 --out data/generated/demo
```

By default, BPM is sampled from `70-150` and time signatures are sampled from:

- `2/2`
- `3/4`
- `3/2`
- `7/4`
- `5/4`
- `2/4`
- `4/4`

You can narrow either range:

```powershell
uv run comet generate --count 4 --bpm-min 90 --bpm-max 128 -t 3/4 -t 7/4
```

You can also make denser clips:

```powershell
uv run comet generate --count 4 --source-count-min 8 --source-count-max 12
```

## Current Instrument Parameters

The generator writes each source's randomized `synth_parameters` and `effect_parameters` into `metadata.json`.

Synth sources currently include:

- `kick`: `start_hz`, `end_hz`
- `snare_clap`: `noise_tone_hz`, `body_hz`
- `hat_noise`: `highpass_hz`
- `fm_bass`: `ratio`, `fm_index`, `index_decay_seconds`, `sub_mix`, `noise_mix`, `wavefold_drive`
- `fm_growl`: `ratio_a`, `ratio_b`, `fm_index_a`, `fm_index_b`, `sub_mix`, `noise_mix`, `wavefold_drive`, `formant_shift`
- `wub_bass`: `ratio`, `fm_index`, `sub_mix`, `harmonic_mix`, `wavefold_drive`
- `pluck_stab`: `detune_cents`, `filter_hz`, `transient_click`, `octave_mix`, `saw_mix`
- `riser_impact`: `filter_start_hz`, `filter_end_hz`, `noise_color`, `impact_weight`

Rhythmic events can use beat subdivisions of `2`, `3`, `4`, `6`, or `8`, written as `rhythm_subdivision` in each event's `render_parameters`. Hats, basses, and plucks sample from those grids; plucks favor denser subdivisions and repeating motifs for faster figures.

Training data avoids continuous LFO modulation. If a part should sound like repeated oscillations, the generator represents that as explicit repeated events on the rhythmic grid so each attack is labeled.

Sources also vary slightly over time through per-event `timbre_variation` values in `render_parameters`. These are constant within an event, not continuous modulation, so the model sees changing timbres across notes without unlabeled pseudo-attacks inside a note.

When a clip contains multiple tracks of the same `source_type`, each duplicate gets an explicit source-level `timbre_variant`, `variant_index`, `variant_count`, and `variant_position`. Plucks and basses also get a `voice_model` such as `round`, `hollow`, `bright_saw`, `metallic`, `sub_clean`, `fm_round`, `edge_distorted`, or `noisy_reese`. These change the oscillator recipe and envelope behavior, not just parameter ranges, so duplicate pluck or bass lanes are less likely to sound like the same instrument copied twice.

`fm_growl` and `wub_bass` use a lower bass note pool, are limited to at most one growl/wub layer per clip, and use short controlled event lengths so they do not blanket the recording. Plucks are bright, transient-heavy, and mixed much louder than the original v0 defaults so they remain audible in denser clips.

Common production parameters include `drive`, `duck_amount`, `duck_release_seconds`, plus delay and reverb parameters on compatible sources.

Each clip directory contains:

- `mix.wav`
- `stems/source_000.wav`, `stems/source_001.wav`, etc.
- `metadata.json`

The batch root contains:

- `manifest.jsonl`
- `preview.html`
- `visualizer.html`

Open `visualizer.html` in a browser to inspect the generated clips. It embeds batch metadata, plays the mix or any source stem, and shows per-source lanes with onset, attack, sustain, and release regions for each event.

## Scope

The v0 renderer uses NumPy/SciPy synthesis and processing with built-in effects only. It does not require external VSTs or DawDreamer. Future renderers can be added behind the same metadata model to support DawDreamer or plugin-hosted instruments while keeping the core generator usable on a clean machine.
