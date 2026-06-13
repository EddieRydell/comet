from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field
from scipy import signal

from comet_audio.models import RendererType, SourceType

KNOWN_SOURCE_TYPES: tuple[SourceType, ...] = (
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
)


class AssetEntry(BaseModel):
    asset_id: str
    renderer: RendererType
    family: str
    source_type: SourceType
    instrument: str
    articulation: str = "unknown"
    path: str
    preset_path: str | None = None
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0
    note_min: int | None = None
    note_max: int | None = None
    velocity_min: float = 0.0
    velocity_max: float = 1.0
    root_key: int | None = None
    round_robin_group: str | None = None
    default_gain_db: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class AssetCatalog:
    root: Path
    entries: tuple[AssetEntry, ...]

    def by_renderer(self, renderer: RendererType) -> list[AssetEntry]:
        return [entry for entry in self.entries if entry.renderer == renderer]

    def candidates(
        self, source_type: SourceType, renderer: RendererType | None = None
    ) -> list[AssetEntry]:
        return [
            entry
            for entry in self.entries
            if entry.source_type == source_type and (renderer is None or entry.renderer == renderer)
        ]

    def resolve_path(self, entry: AssetEntry) -> Path:
        path = Path(entry.path)
        return path if path.is_absolute() else self.root / path

    def resolve_preset_path(self, entry: AssetEntry) -> Path | None:
        if entry.preset_path is None:
            return None
        path = Path(entry.preset_path)
        return path if path.is_absolute() else self.root / path


@dataclass(frozen=True)
class SfzRegion:
    sample: str
    lokey: int = 0
    hikey: int = 127
    pitch_keycenter: int | None = None
    lovel: float = 0.0
    hivel: float = 1.0
    volume: float = 0.0
    offset: int = 0
    end: int | None = None
    tune: float = 0.0
    ampeg_attack: float = 0.0
    ampeg_release: float = 0.05
    unsupported: dict[str, str] = field(default_factory=dict)

    def matches(self, midi_note: int, velocity: float) -> bool:
        return self.lokey <= midi_note <= self.hikey and self.lovel <= velocity <= self.hivel


SUPPORTED_SFZ_OPCODES = {
    "sample",
    "key",
    "lokey",
    "hikey",
    "pitch_keycenter",
    "lovel",
    "hivel",
    "volume",
    "offset",
    "end",
    "tune",
    "ampeg_attack",
    "ampeg_release",
}

NOTE_NAMES = {
    "c": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
}

PERCUSSION_BUCKET_KEYWORDS: tuple[tuple[SourceType, tuple[str, ...]], ...] = (
    ("closed_hat", ("closedhat", "closed_hat", "closed hat", "cl hat", "cl_hat", "chh")),
    ("open_hat", ("openhat", "open_hat", "open hat", "op hat", "op_hat", "ohh")),
    ("kick", ("kick", "bd", "bassdrum", "bass drum")),
    ("snare", ("snare", "sd")),
    ("clap", ("clap", "claps")),
    ("cymbal", ("cymbal", "crash", "ride", "splash")),
    ("tom", ("tom", "floor", "rack")),
    ("percussion", ("perc", "percussion", "foley", "wood", "rim", "clave", "shaker")),
)


@dataclass(frozen=True)
class PercussionImportReport:
    pack_id: str
    imported_count: int
    duplicate_count: int
    rejected: tuple[dict[str, str], ...]
    ambiguous: tuple[str, ...]
    bucket_counts: dict[str, int]
    duration_seconds: dict[str, float]
    rms: dict[str, float]
    report_path: Path


def load_asset_catalog(root: Path | None) -> AssetCatalog:
    if root is None:
        return AssetCatalog(Path("."), ())
    root = root.resolve()
    entries: list[AssetEntry] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif "assets" in payload:
            rows = payload["assets"]
        elif {"asset_id", "renderer", "source_type", "path"}.issubset(payload):
            rows = [payload]
        else:
            continue
        entries.extend(AssetEntry.model_validate(row) for row in rows)
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(AssetEntry.model_validate_json(line))
    return AssetCatalog(root=root, entries=tuple(entries))


def validate_asset_catalog(root: Path) -> list[str]:
    catalog = load_asset_catalog(root)
    warnings: list[str] = []
    if not catalog.entries:
        warnings.append(f"No asset entries found under {root}")
    for entry in catalog.entries:
        if entry.source_type not in KNOWN_SOURCE_TYPES:
            warnings.append(f"{entry.asset_id}: unknown source_type {entry.source_type!r}")
        path = catalog.resolve_path(entry)
        if not path.exists():
            warnings.append(f"{entry.asset_id}: missing file {path}")
            continue
        if entry.renderer == "wav_one_shot":
            try:
                load_mono_wav(path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{entry.asset_id}: unreadable WAV {path}: {exc}")
        elif entry.renderer == "sfz_instrument":
            try:
                regions, sfz_warnings = parse_sfz(path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{entry.asset_id}: unreadable SFZ {path}: {exc}")
            else:
                warnings.extend(f"{entry.asset_id}: {warning}" for warning in sfz_warnings)
                for region in regions:
                    sample_path = path.parent / region.sample
                    if not sample_path.exists():
                        warnings.append(f"{entry.asset_id}: missing SFZ sample {sample_path}")
                    else:
                        try:
                            load_mono_wav(sample_path)
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(
                                f"{entry.asset_id}: unreadable SFZ sample {sample_path}: {exc}"
                            )
        elif entry.renderer == "dawdreamer_plugin":
            preset_path = catalog.resolve_preset_path(entry)
            if preset_path is None:
                warnings.append(f"{entry.asset_id}: missing preset_path")
            elif not preset_path.exists():
                warnings.append(f"{entry.asset_id}: missing preset state {preset_path}")
        if entry.weight <= 0:
            warnings.append(f"{entry.asset_id}: weight must be greater than 0")
    return warnings


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def import_percussion_samples(
    source_dir: Path,
    assets_root: Path,
    pack_id: str,
    catalog_path: Path | None = None,
    skip_loops: bool = True,
    sample_rate: int = 44_100,
    max_duration_seconds: float = 4.0,
    trim_threshold_db: float = -60.0,
) -> PercussionImportReport:
    source_dir = source_dir.resolve()
    assets_root = assets_root.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Sample source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Sample source path is not a directory: {source_dir}")
    pack_slug = _slugify_asset_component(pack_id)
    catalog_path = catalog_path or assets_root / "catalog.json"
    sample_root = assets_root / "samples" / "percussion" / pack_slug
    report_root = assets_root / "imports" / "percussion"
    sample_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    existing_hashes = _existing_audio_hashes(assets_root)
    imported: list[AssetEntry] = []
    rejected: list[dict[str, str]] = []
    ambiguous: list[str] = []
    durations: list[float] = []
    rms_values: list[float] = []
    bucket_counts: dict[str, int] = {}
    duplicate_count = 0

    for wav_path in sorted(source_dir.rglob("*.wav")):
        label_text = " ".join(part.lower() for part in wav_path.relative_to(source_dir).parts)
        if skip_loops and _looks_like_loop(label_text):
            rejected.append({"path": str(wav_path), "reason": "loop"})
            continue
        source_type, matched = _classify_percussion_bucket(label_text)
        if not matched:
            ambiguous.append(str(wav_path))
        try:
            audio, actual_rate = load_mono_wav(wav_path)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"path": str(wav_path), "reason": f"unreadable: {exc}"})
            continue
        try:
            normalized = _prepare_one_shot_audio(
                audio,
                actual_rate=actual_rate,
                target_rate=sample_rate,
                max_duration_seconds=max_duration_seconds,
                trim_threshold_db=trim_threshold_db,
            )
        except ValueError as exc:
            rejected.append({"path": str(wav_path), "reason": str(exc)})
            continue
        audio_hash = hashlib.sha256(normalized.tobytes()).hexdigest()
        if audio_hash in existing_hashes:
            duplicate_count += 1
            rejected.append({"path": str(wav_path), "reason": "duplicate"})
            continue
        existing_hashes.add(audio_hash)

        asset_stem = _slugify_asset_component(wav_path.stem)
        asset_id = f"{pack_slug}_{source_type}_{asset_stem}_{audio_hash[:10]}"
        rel_path = Path("samples") / "percussion" / pack_slug / f"{asset_id}.wav"
        out_path = assets_root / rel_path
        sf.write(out_path, normalized, sample_rate, subtype="PCM_16")
        duration = len(normalized) / sample_rate
        rms_value = float(np.sqrt(np.mean(np.square(normalized)))) if len(normalized) else 0.0
        durations.append(duration)
        rms_values.append(rms_value)
        bucket_counts[source_type] = bucket_counts.get(source_type, 0) + 1
        imported.append(
            AssetEntry(
                asset_id=asset_id,
                renderer="wav_one_shot",
                family="drums" if source_type != "percussion" else "percussion",
                source_type=source_type,
                instrument=f"{pack_slug}_{source_type}",
                articulation="hit",
                path=rel_path.as_posix(),
                tags=["percussion", "free", f"pack:{pack_slug}", f"bucket:{source_type}"],
                default_gain_db=_default_import_gain(source_type),
                metadata={
                    "source_file": str(wav_path),
                    "audio_sha256": audio_hash,
                    "duration_seconds": round(duration, 6),
                    "rms": round(rms_value, 6),
                },
            )
        )

    _upsert_catalog_entries(catalog_path, imported)
    report_path = report_root / f"{pack_slug}_import_report.json"
    report = PercussionImportReport(
        pack_id=pack_slug,
        imported_count=len(imported),
        duplicate_count=duplicate_count,
        rejected=tuple(rejected),
        ambiguous=tuple(ambiguous),
        bucket_counts=dict(sorted(bucket_counts.items())),
        duration_seconds=_stats(durations),
        rms=_stats(rms_values),
        report_path=report_path,
    )
    report_path.write_text(
        json.dumps(
            {
                "pack_id": report.pack_id,
                "imported_count": report.imported_count,
                "duplicate_count": report.duplicate_count,
                "rejected": list(report.rejected),
                "ambiguous": list(report.ambiguous),
                "bucket_counts": report.bucket_counts,
                "duration_seconds": report.duration_seconds,
                "rms": report.rms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _prepare_one_shot_audio(
    audio: np.ndarray,
    actual_rate: int,
    target_rate: int,
    max_duration_seconds: float,
    trim_threshold_db: float,
) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.size == 0:
        raise ValueError("empty")
    if not np.isfinite(mono).all():
        raise ValueError("non-finite samples")
    if actual_rate != target_rate:
        target_len = max(1, int(round(len(mono) * target_rate / actual_rate)))
        mono = signal.resample(mono, target_len).astype(np.float32)
    threshold = 10.0 ** (trim_threshold_db / 20.0)
    non_silent = np.flatnonzero(np.abs(mono) > threshold)
    if non_silent.size == 0:
        raise ValueError("silence")
    mono = mono[non_silent[0] : non_silent[-1] + 1]
    duration = len(mono) / target_rate
    if duration > max_duration_seconds:
        raise ValueError(f"too_long:{duration:.3f}s")
    peak = float(np.max(np.abs(mono)))
    if peak <= 1e-6:
        raise ValueError("silence")
    return (mono / peak * 0.95).astype(np.float32)


def _classify_percussion_bucket(text: str) -> tuple[SourceType, bool]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    compact = normalized.replace(" ", "")
    for source_type, keywords in PERCUSSION_BUCKET_KEYWORDS:
        if any(
            keyword in normalized or keyword.replace(" ", "") in compact for keyword in keywords
        ):
            return source_type, True
    return "percussion", False


def _looks_like_loop(text: str) -> bool:
    return bool(re.search(r"\b(loop|loops|bpm|tempo|groove|break|beat)\b", text))


def _existing_audio_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    catalog = load_asset_catalog(root if root.exists() else None)
    for entry in catalog.entries:
        value = entry.metadata.get("audio_sha256")
        if isinstance(value, str):
            hashes.add(value)
    return hashes


def _upsert_catalog_entries(catalog_path: Path, entries: list[AssetEntry]) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if catalog_path.exists():
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("assets", [])
        existing.extend(dict(row) for row in rows)
    by_id = {str(row["asset_id"]): row for row in existing if "asset_id" in row}
    for entry in entries:
        by_id[entry.asset_id] = entry.model_dump(mode="json", exclude_none=True)
    catalog_path.write_text(
        json.dumps({"assets": list(by_id.values())}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


def _slugify_asset_component(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "pack"


def _default_import_gain(source_type: SourceType) -> float:
    if source_type == "kick":
        return -5.0
    if source_type in {"snare", "clap", "tom"}:
        return -8.0
    if source_type in {"closed_hat", "open_hat", "cymbal"}:
        return -12.0
    return -10.0


def parse_sfz(path: Path) -> tuple[list[SfzRegion], list[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    warnings: list[str] = []
    current_group: dict[str, str] = {}
    regions: list[SfzRegion] = []
    matches = re.findall(r"<(group|region)>([^<]*)", text, flags=re.IGNORECASE | re.DOTALL)
    for header, body in matches:
        opcodes = _parse_sfz_opcodes(body)
        unsupported = {
            key: value for key, value in opcodes.items() if key not in SUPPORTED_SFZ_OPCODES
        }
        if unsupported:
            warnings.append(f"ignored unsupported SFZ opcodes: {', '.join(sorted(unsupported))}")
        supported = {key: value for key, value in opcodes.items() if key in SUPPORTED_SFZ_OPCODES}
        if header.lower() == "group":
            current_group = supported
            continue
        merged = {**current_group, **supported}
        if "sample" not in merged:
            warnings.append("region without sample ignored")
            continue
        if "key" in merged:
            key = _parse_midi_value(merged["key"])
            merged.setdefault("lokey", str(key))
            merged.setdefault("hikey", str(key))
            merged.setdefault("pitch_keycenter", str(key))
        regions.append(
            SfzRegion(
                sample=merged["sample"].replace("\\", "/"),
                lokey=_parse_midi_value(merged.get("lokey", "0")),
                hikey=_parse_midi_value(merged.get("hikey", "127")),
                pitch_keycenter=(
                    _parse_midi_value(merged["pitch_keycenter"])
                    if "pitch_keycenter" in merged
                    else None
                ),
                lovel=_parse_velocity(merged.get("lovel", "0")),
                hivel=_parse_velocity(merged.get("hivel", "127")),
                volume=float(merged.get("volume", "0")),
                offset=max(0, int(float(merged.get("offset", "0")))),
                end=int(float(merged["end"])) if "end" in merged else None,
                tune=float(merged.get("tune", "0")),
                ampeg_attack=max(0.0, float(merged.get("ampeg_attack", "0"))),
                ampeg_release=max(0.0, float(merged.get("ampeg_release", "0.05"))),
                unsupported=unsupported,
            )
        )
    return regions, warnings


def choose_sfz_region(regions: list[SfzRegion], midi_note: int, velocity: float) -> SfzRegion:
    matches = [region for region in regions if region.matches(midi_note, velocity)]
    if not matches:
        raise ValueError(f"No SFZ region for MIDI note {midi_note}, velocity {velocity:.3f}")
    return matches[0]


def _parse_sfz_opcodes(body: str) -> dict[str, str]:
    opcodes: dict[str, str] = {}
    body = re.sub(r"//.*", "", body)
    for token in re.split(r"\s+", body.strip()):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        opcodes[key.strip().lower()] = value.strip()
    return opcodes


def _parse_midi_value(value: str) -> int:
    text = value.strip().lower()
    if text.lstrip("-").isdigit():
        return int(text)
    match = re.fullmatch(r"([a-g](?:#|b)?)(-?\d+)", text)
    if not match:
        raise ValueError(f"Invalid MIDI note value: {value!r}")
    name, octave = match.groups()
    return NOTE_NAMES[name] + (int(octave) + 1) * 12


def _parse_velocity(value: str) -> float:
    text = value.strip()
    velocity = float(text)
    if "." not in text and velocity >= 1.0:
        return velocity / 127.0
    return velocity
