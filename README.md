# Comet Audio

Comet Audio is a synthetic EDM dataset generator. It creates short labeled clips for onset and event-oriented research: a final mono mix WAV, wet mono per-source stems, metadata JSON, and a batch manifest.

Version 1 keeps the same training-friendly mono layout while adding an asset-backed renderer layer for WAV one-shots, simple SFZ instruments, and DawDreamer-hosted synth presets captured from real plugin GUIs.

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

To generate from a local asset catalog, put JSON or JSONL asset entries under `assets/library` and pass the catalog root:

```powershell
uv run comet assets validate --assets assets/library
uv run comet generate --count 16 --assets assets/library --renderer-profile hybrid_v1
```

Each asset entry describes `asset_id`, `renderer`, `family`, `source_type`, `instrument`, `articulation`, `path`, optional note and velocity ranges, optional `root_key` and `round_robin_group`, and `default_gain_db`. Supported renderers are:

- `procedural_synth`
- `wav_one_shot`
- `sfz_instrument`
- `dawdreamer_plugin`

`dawdreamer_plugin` entries also use `preset_path`, `tags`, and `weight`. `path` points to the plugin file, such as a VST3, and `preset_path` points to the DawDreamer state saved from the plugin GUI. `wav_one_shot`, `sfz_instrument`, and `dawdreamer_plugin` sources are mixed down to mono on render. New datasets use `dataset_version="comet-edm-v1"` and expose `source_type`, `family`, `instrument`, `articulation`, `renderer`, and `asset_id` directly on each source.

To capture a synth preset, install a compatible plugin such as Vital or Surge XT, then open its editor through Comet:

```powershell
uv run comet synth capture --assets assets/library --plugin C:\path\to\Synth.vst3 --asset-id vital_reese_001 --source-type synth_bass --tags bass --tags reese --weight 2
```

Design the sound in the plugin window and close the editor. Comet saves a DawDreamer state file under the asset root and appends or updates the preset in `assets/library/catalog.json`.

For the lowest-friction Surge XT workflow, launch the preset studio:

```powershell
uv run comet synth studio --assets assets/library --source-type synth_bass --tags bass --tags reese
```

Comet launches the Surge XT standalone app, watches `%USERPROFILE%\Documents\Surge XT` and `assets/library/imports/surge_xt/inbox` for new or changed `.fxp` files, imports each stable save through the Surge XT VST3, writes a DawDreamer state file under `assets/library/presets`, copies the source `.fxp` for provenance, writes a mono audition WAV under `assets/library/auditions`, and updates `assets/library/catalog.json`. In Surge XT, open `Workflow > Virtual Keyboard` or press `Alt+K`, play with the computer keyboard, and save useful patches as `.fxp`.

The default Windows paths are:

- `C:\Program Files\Surge Synth Team\Surge XT\Surge XT.exe`
- `C:\Program Files\Common Files\VST3\Surge Synth Team\Surge XT.vst3`

Override them when needed:

```powershell
uv run comet synth studio --standalone C:\path\to\Surge XT.exe --plugin C:\path\to\Surge XT.vst3 --watch-dir C:\patches
```

To import an existing Surge XT `.fxp` without launching the studio:

```powershell
uv run comet synth import-surge --preset-file C:\path\MyBass.fxp --asset-id surge_my_bass --source-type synth_bass --tags bass
```

While the catalog is small, generate with `hybrid_v1` so procedural sources fill gaps. Once enough catalog assets exist, switch to `plugin_v1`.

Audition a captured preset:

```powershell
uv run comet synth audition --assets assets/library --asset-id vital_reese_001 --out data/generated/vital_reese_001.wav --midi-note 36
```

Generate from catalog-backed plugin, WAV, and SFZ assets only:

```powershell
uv run comet generate --assets assets/library --renderer-profile plugin_v1 --include-tag bass
```

`plugin_v1` fails if no valid catalog candidate exists for a selected source type. `--include-tag` and `--exclude-tag` filter catalog candidates before weighted selection.

## Percussion One-Shot Libraries

Use free one-shot packs by downloading and unzipping them locally, then importing the WAV folders into the asset catalog:

```powershell
uv run comet assets import-percussion --source C:\samples\99-drum-samples --assets assets/library --pack-id 99sounds_drums_1
uv run comet assets validate --assets assets/library
```

The importer scans WAV files, skips likely loops by default, converts to mono 44.1 kHz, trims leading and trailing silence, peak-normalizes, rejects silent or long files, deduplicates by audio hash, and writes normalized one-shots under `assets/library/samples/percussion/<pack_id>/`. It classifies filenames and folders into generator buckets such as `kick`, `snare`, `clap`, `closed_hat`, `open_hat`, `cymbal`, `tom`, and `percussion`. Ambiguous files default to `percussion` and are listed in the import report under `assets/library/imports/percussion/`.

Recommended starter packs:

- 99Sounds 99 Drum Samples I/II
- 99Sounds Percussa Toolbox
- MusicRadar Essential Drum Kit Samples
- MusicRadar Modular Percussion, hit folders only
- MusicRadar Wooden Percussion, hit samples only
- MusicRadar Processed 808/909, individual hits only

Generate percussion-only training clips from imported assets:

```powershell
uv run comet generate --count 10000 --seed 400000 --out data/generated/percussion_slots_10k --composition-profile percussion_v1 --assets assets/library --renderer-profile plugin_v1 --include-tag percussion --no-procedural-fallback --source-count-min 1 --source-count-max 16 --training-layout --no-stems --no-visualizer
```

`percussion_v1` chooses only percussion-family source buckets and varies rhythm templates across straight, half-time, breakbeat, two-step, triplet, sparse, fill, solo-hit, dense-hat, and foley-style patterns. Loops remain out of scope because repeated attacks must be represented as metadata events.

After training, run the timing model on an arbitrary song and write an interactive HTML viewer:

```powershell
uv run comet predict-song --audio C:\path\song.wav --run runs/cnn_tcn_v1 --out data/generated/song_predictions
```

The command writes `predictions.json`, copies the source audio into the output folder for browser playback, and writes `visualizer.html` with waveform display, playback controls, zoom, horizontal scrolling, decoded onset marks, and source-type lanes. The model runs at 44.1 kHz; other input sample rates are resampled for inference.

For training datasets with minimal disk use, write a single root with only mono mixes, metadata, and a manifest:

```powershell
uv run comet generate --count 10000 --seed 100000 --out data/generated/train_10k --training-layout --no-stems --no-visualizer
```

That layout is:

- `audio/clip_0000.wav`
- `metadata/clip_0000.json`
- `manifest.jsonl`

Train the anonymous slot target on percussion data:

```powershell
uv run comet train --target anonymous_slots_v1 --max-tracks 16 --data data/generated/percussion_slots_10k --run runs/percussion_slots_v1
```

This target maps generated source tracks to unordered slots such as `track_00`, `track_01`, etc. Each slot predicts attack, held, and release phase regions. `off` is implied when all three are low. Training uses Hungarian matching between predicted and generated slots and does not include a source-type classifier head.

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

Common production parameters include `drive`, `duck_amount`, and `duck_release_seconds`. Delay and reverb are intentionally excluded from default training renders so the audio does not contain unlabeled echo or tail events.

Each clip directory contains:

- `mix.wav`
- `stems/source_000.wav`, `stems/source_001.wav`, etc.
- `metadata.json`

The batch root contains:

- `manifest.jsonl`
- `visualizer.html`

Open `visualizer.html` in a browser to inspect the generated clips. It embeds batch metadata, plays the mix or any source stem, and shows per-source lanes with onset, attack, held, and release regions for each event.

## Scope

The procedural renderer uses NumPy/SciPy synthesis and processing with built-in effects only. DawDreamer is used only for `dawdreamer_plugin` rendering, capture, and audition commands. Plugin paths and preset state paths live in the asset catalog; this project does not use `.env` files or environment variables for plugin configuration.
