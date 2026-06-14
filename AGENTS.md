# Comet Agent Instructions

These instructions apply to this repository.

## Workflow

- Do not use `git` or commands associated with it unless the user specifically requests it.
- Do not jump to editing when the conversation is about diagnosing an issue, root cause analysis, architecture, or design decisions. Investigate and discuss first.
- When planning for user approval, list the files expected to be affected.
- Ask relevant questions when requirements are ambiguous or when a choice changes the generator, metadata schema, training targets, or dataset layout.
- Do not write or add tests unless specifically requested. If existing tests must be updated because the user requested a behavior change, keep those edits tightly scoped.
- After implementing a plan, run `uv run ruff format`, `uv run ruff check .`, and `uv run pytest`. Fix regressions. These checks are not necessary for docs-only changes, example-only changes, or other edits unaffected by checks.

## Codebase Principles

- Favor one clear way to do things. Minimize clutter and avoid duplicate paths for the same behavior.
- Keep a single source of truth. Source taxonomy, renderer names, metadata fields, and asset catalog schema should not be redefined in multiple places.
- Avoid strings in internal logic where structured data is practical. Prefer enums, literals, typed models, or shared constants for source types, renderer types, dataset versions, and metadata keys.
- Do not add compatibility layers, shims, hidden fallbacks, or legacy support when adding features or refactoring.
- Do not add fallbacks when something does not work. Surface clear errors instead. This repo optimizes for fast research development, not broad backward compatibility.
- Keep all generated, imported, rendered, and training audio mono. Do not introduce stereo paths or stereo metadata; stereo is out of scope for this project.
- Preserve the hard labeling rule: every audible attack must correspond to an event. Do not introduce delay, reverb, loops, or unlabeled repeated attacks unless they are explicitly represented in metadata.

## Long-Term Model Direction

- The long-term goal is not a fixed-instrument classifier. Do not assume the final model should output lanes named `kick`, `snare`, `bass`, `synth_lead`, or other source taxonomy labels.
- The desired output is an unordered bag of anonymous instrument/event tracks. Tracks should be treated as slots such as `track_00`, `track_01`, etc., where each slot represents one coherent source the model thinks it heard.
- The practical first version may use a fixed maximum number of slots, currently expected to be around 16, but conceptually the project is aiming at an undetermined/unbounded number of instruments in a song. Unused slots should be allowed.
- Training targets should move toward per-track envelope/event structure: attack, held, and release markings for each detected note/event, not only onset spikes.
- Use only `attack`, `held`, and `release` as canonical time-phase labels. `off` is implied when all three are low. Do not use `sustain`, `active`, `body`, `tail`, or other parallel names for training/viewer time phases. `sustain` is allowed only when referring to a synth sustain level/parameter, not a time region.
- Phase boundaries are `off | attack | held | release | off`: attack starts at note onset, held starts when attack ends, release starts at note-off, and release ends at event offset. Zero-length phases are allowed.
- The model should learn source separation/grouping as anonymous event streams. It should not be rewarded for naming the instrument unless the user explicitly asks for an auxiliary classifier.
- Because track slots are anonymous, training loss must handle permutation. Prefer permutation-invariant matching such as Hungarian matching between predicted slots and generated source tracks over direct slot-index supervision.
- Buckets such as drums, basses, pads, plucks, long-attack synths, vocals, and effects are useful as generation/data-diversity sources, not as the final output ontology.
- Generator realism and diversity are central. Avoid designs that let the model succeed by memorizing narrow procedural instrument fingerprints. Favor varied real samples, asset-backed synths, long attack/held/release sounds, messy mixes, masking, compression, clipping, and other domain randomization.
- CNNs can be useful as audio frontends, but avoid assuming a CNN-only template detector is the long-term architecture. Future models will likely need contextual grouping, slot/set prediction, or similar mechanisms to produce coherent anonymous tracks.

## Repository Hygiene

- Avoid unrelated edits to `uv.lock`, IDE files, generated datasets, model checkpoints, generated assets, and cache directories.
- Do not edit `uv.lock` unless dependency changes are explicitly part of the task.
- Check `pyproject.toml`, `src/comet_audio/cli.py`, and the relevant module before assuming a dependency, command, script entry point, or option belongs at the project root.
- Do not use `.env` files or environment variables for project configuration in this codebase.
- Do not commit large sample libraries, generated datasets, visualizer batches, checkpoints, or run artifacts unless the user specifically asks.

## Assets And Generation

- Treat the asset catalog as the source of truth for repo-local sample libraries.
- Don't worry too much about licensing - this is a personal project.
- Validate catalog entries with `uv run comet assets validate --assets assets/library` when asset metadata changes.
- Prefer explicit renderer errors over silent procedural fallback when the user disables fallback or expects asset-backed rendering.
- Do not reintroduce generated web bindings, desktop schema files, or unrelated frontend tooling. The visualizer is a static generated HTML output from the Python generator.
- Do not start or leave a frontend dev server running. This project should not claim frontend ports as part of normal work.
