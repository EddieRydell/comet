from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

KnownSourceType = Literal[
    "kick",
    "snare",
    "clap",
    "closed_hat",
    "open_hat",
    "cymbal",
    "tom",
    "percussion",
    "synth_bass",
    "electric_bass",
    "acoustic_bass",
    "piano",
    "electric_piano",
    "organ",
    "guitar_pluck",
    "guitar_strum",
    "mallet",
    "string_stab",
    "brass_stab",
    "synth_lead",
    "synth_pluck",
    "pad_chord",
    "riser",
    "impact",
    "noise_sweep",
]
SourceType = str
RendererType = Literal[
    "procedural_synth",
    "wav_one_shot",
    "sfz_instrument",
    "dawdreamer_plugin",
    "surge_xt_fxp",
]


class SourceMetadata(BaseModel):
    source_id: str
    source_type: SourceType
    family: str = "unknown"
    instrument: str = "unknown"
    articulation: str = "unknown"
    renderer: RendererType = "procedural_synth"
    asset_id: str | None = None
    synth_parameters: dict[str, Any] = Field(default_factory=dict)
    effect_parameters: dict[str, Any] = Field(default_factory=dict)
    gain_db: float
    pan: float
    stem_path: str


class EventMetadata(BaseModel):
    event_id: str
    source_id: str
    event_type: str
    onset_seconds: float
    offset_seconds: float
    velocity: float
    midi_note: int | None = None
    attack_seconds: float
    release_seconds: float
    render_parameters: dict[str, Any] = Field(default_factory=dict)


class ClipMetadata(BaseModel):
    dataset_version: str = "comet-edm-v0"
    seed: int
    sample_rate: int
    duration_seconds: float
    bpm: float
    time_signature: str
    beats_per_measure: int
    beat_unit: int
    key: str
    scale: str = "minor"
    paths: dict[str, str]
    sources: list[SourceMetadata]
    events: list[EventMetadata]


class BatchManifestEntry(BaseModel):
    clip_id: str
    seed: int
    bpm: float
    time_signature: str
    key: str
    mix_path: str
    metadata_path: str
    source_count: int
    event_count: int
