from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal[
    "kick",
    "snare_clap",
    "hat_noise",
    "fm_bass",
    "fm_growl",
    "wub_bass",
    "pluck_stab",
    "riser_impact",
]


class SourceMetadata(BaseModel):
    source_id: str
    source_type: SourceType
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
