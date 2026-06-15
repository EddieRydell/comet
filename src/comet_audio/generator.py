# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from comet_audio.assets import (
    BROKEN_SURGE_ASSET_ID_PREFIXES,
    BROKEN_SURGE_ASSET_IDS,
    AssetCatalog,
    AssetEntry,
    choose_sfz_region,
    load_asset_catalog,
    load_mono_wav,
    parse_sfz,
)
from comet_audio.dawdreamer_renderer import render_dawdreamer_source, render_surge_fxp_source
from comet_audio.dsp import (
    TAU,
    adsr_envelope,
    butter_filter,
    db_to_amp,
    midi_to_hz,
    normalize_peak,
    one_pole_decay,
    saturate,
    sidechain_duck,
    soft_limiter,
)
from comet_audio.models import (
    BatchManifestEntry,
    ClipMetadata,
    EventMetadata,
    SourceMetadata,
    SourceType,
)

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
DEFAULT_TIME_SIGNATURES = ("2/2", "3/4", "3/2", "7/4", "5/4", "2/4", "4/4")
RHYTHM_SUBDIVISIONS = (2, 3, 4, 6, 8)
SOURCE_TYPES: list[SourceType] = [
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
OPTIONAL_SOURCE_POOL: list[SourceType] = [
    "open_hat",
    "percussion",
    "synth_bass",
    "electric_bass",
    "acoustic_bass",
    "synth_lead",
    "synth_pluck",
    "pad_chord",
    "mallet",
    "string_stab",
    "brass_stab",
    "riser",
    "impact",
    "noise_sweep",
]
PERCUSSION_SOURCE_TYPES: tuple[SourceType, ...] = (
    "kick",
    "snare",
    "clap",
    "closed_hat",
    "open_hat",
    "cymbal",
    "tom",
    "percussion",
)
PERCUSSION_RHYTHM_TEMPLATES = (
    "straight",
    "half_time",
    "breakbeat",
    "two_step",
    "triplet",
    "sparse",
    "fills",
    "solo_hits",
    "dense_hats",
    "foley",
)
PROCEDURAL_RECIPE_BY_SOURCE_TYPE: dict[str, str] = {
    "kick": "kick",
    "snare": "snare_clap",
    "clap": "snare_clap",
    "closed_hat": "hat_noise",
    "open_hat": "hat_noise",
    "cymbal": "hat_noise",
    "tom": "kick",
    "percussion": "snare_clap",
    "synth_bass": "fm_bass",
    "electric_bass": "fm_bass",
    "acoustic_bass": "fm_bass",
    "synth_lead": "pluck_stab",
    "synth_pluck": "pluck_stab",
    "pad_chord": "pluck_stab",
    "piano": "pluck_stab",
    "electric_piano": "pluck_stab",
    "organ": "pluck_stab",
    "guitar_pluck": "pluck_stab",
    "guitar_strum": "pluck_stab",
    "mallet": "pluck_stab",
    "string_stab": "pluck_stab",
    "brass_stab": "pluck_stab",
    "riser": "riser_impact",
    "impact": "riser_impact",
    "noise_sweep": "riser_impact",
}
FAMILY_BY_SOURCE_TYPE: dict[str, str] = {
    **{
        name: "drums"
        for name in ("kick", "snare", "clap", "closed_hat", "open_hat", "cymbal", "tom")
    },
    "percussion": "percussion",
    "synth_bass": "bass",
    "electric_bass": "bass",
    "acoustic_bass": "bass",
    "riser": "fx",
    "impact": "fx",
    "noise_sweep": "fx",
}


@dataclass(frozen=True)
class GeneratorConfig:
    sample_rate: int = 44_100
    duration_seconds: float = 8.0
    bpm_min: int = 70
    bpm_max: int = 150
    time_signatures: tuple[str, ...] = DEFAULT_TIME_SIGNATURES
    source_count_min: int = 5
    source_count_max: int = 10
    renderer_profile: str = "hybrid_v1"
    procedural_fallback: bool = True
    asset_catalog: AssetCatalog | None = None
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    composition_profile: str = "edm_v1"
    duration_profile: tuple[float, ...] | None = None


def generate_batch(
    out_dir: Path,
    count: int,
    seed: int,
    config: GeneratorConfig | None = None,
    write_visualizer: bool = True,
    write_stems: bool = True,
    flat_layout: bool = False,
    assets: Path | None = None,
    renderer_profile: str = "hybrid_v1",
    procedural_fallback: bool = True,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    workers: int = 1,
) -> list[ClipMetadata]:
    config = config or GeneratorConfig()
    if assets is not None or config.asset_catalog is None:
        config = GeneratorConfig(
            sample_rate=config.sample_rate,
            duration_seconds=config.duration_seconds,
            bpm_min=config.bpm_min,
            bpm_max=config.bpm_max,
            time_signatures=config.time_signatures,
            source_count_min=config.source_count_min,
            source_count_max=config.source_count_max,
            renderer_profile=renderer_profile,
            procedural_fallback=procedural_fallback,
            asset_catalog=load_asset_catalog(assets),
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            composition_profile=config.composition_profile,
            duration_profile=config.duration_profile,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    workers = max(1, int(workers))
    if workers == 1 or count == 1:
        clips = [
            generate_clip(
                out_dir / f"clip_{index:04d}",
                int(seed + index),
                config,
                clip_id=f"clip_{index:04d}",
                dataset_root=out_dir,
                write_stems=write_stems,
                flat_layout=flat_layout,
            )
            for index in range(count)
        ]
    else:
        clips_by_index: dict[int, ClipMetadata] = {}
        chunk_size = max(1, count // (workers * 8))
        ranges = [(start, min(count, start + chunk_size)) for start in range(0, count, chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_batch_range,
                    out_dir,
                    start,
                    stop,
                    seed,
                    config,
                    write_stems,
                    flat_layout,
                )
                for start, stop in ranges
            ]
            for future in as_completed(futures):
                for index, metadata in future.result():
                    clips_by_index[index] = metadata
        clips = [clips_by_index[index] for index in range(count)]

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, metadata in enumerate(clips):
            clip_id = f"clip_{index:04d}"
            manifest_mix_path = metadata.paths["mix"] if flat_layout else f"{clip_id}/mix.wav"
            manifest_metadata_path = (
                metadata.paths["metadata"] if flat_layout else f"{clip_id}/metadata.json"
            )
            entry = BatchManifestEntry(
                clip_id=clip_id,
                seed=metadata.seed,
                bpm=metadata.bpm,
                time_signature=metadata.time_signature,
                key=metadata.key,
                mix_path=manifest_mix_path,
                metadata_path=manifest_metadata_path,
                source_count=len(metadata.sources),
                event_count=len(metadata.events),
            )
            manifest.write(entry.model_dump_json() + "\n")

    if write_visualizer:
        write_visualizer_html(out_dir, clips)
    return clips


def _generate_batch_range(
    out_dir: Path,
    start: int,
    stop: int,
    seed: int,
    config: GeneratorConfig,
    write_stems: bool,
    flat_layout: bool,
) -> list[tuple[int, ClipMetadata]]:
    clips: list[tuple[int, ClipMetadata]] = []
    for index in range(start, stop):
        clip_id = f"clip_{index:04d}"
        metadata = generate_clip(
            out_dir / clip_id,
            int(seed + index),
            config,
            clip_id=clip_id,
            dataset_root=out_dir,
            write_stems=write_stems,
            flat_layout=flat_layout,
        )
        clips.append((index, metadata))
    return clips


def generate_clip(
    clip_dir: Path,
    seed: int,
    config: GeneratorConfig | None = None,
    clip_id: str | None = None,
    dataset_root: Path | None = None,
    write_stems: bool = True,
    flat_layout: bool = False,
    assets: Path | None = None,
    renderer_profile: str | None = None,
    procedural_fallback: bool | None = None,
    include_tags: tuple[str, ...] | None = None,
    exclude_tags: tuple[str, ...] | None = None,
) -> ClipMetadata:
    config = config or GeneratorConfig()
    if (
        assets is not None
        or renderer_profile is not None
        or procedural_fallback is not None
        or include_tags is not None
        or exclude_tags is not None
    ):
        config = GeneratorConfig(
            sample_rate=config.sample_rate,
            duration_seconds=config.duration_seconds,
            bpm_min=config.bpm_min,
            bpm_max=config.bpm_max,
            time_signatures=config.time_signatures,
            source_count_min=config.source_count_min,
            source_count_max=config.source_count_max,
            renderer_profile=renderer_profile or config.renderer_profile,
            procedural_fallback=(
                config.procedural_fallback if procedural_fallback is None else procedural_fallback
            ),
            asset_catalog=load_asset_catalog(assets)
            if assets is not None
            else config.asset_catalog,
            include_tags=config.include_tags if include_tags is None else include_tags,
            exclude_tags=config.exclude_tags if exclude_tags is None else exclude_tags,
            composition_profile=config.composition_profile,
            duration_profile=config.duration_profile,
        )
    rng = np.random.default_rng(seed)
    duration_seconds = _choose_clip_duration(rng, config)
    clip_id = clip_id or clip_dir.name
    dataset_root = dataset_root or clip_dir.parent
    if flat_layout:
        audio_dir = dataset_root / "audio"
        metadata_dir = dataset_root / "metadata"
        stem_dir = dataset_root / "stems" / clip_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        mix_path = audio_dir / f"{clip_id}.wav"
        metadata_path = metadata_dir / f"{clip_id}.json"
        relative_mix_path = f"audio/{clip_id}.wav"
        relative_metadata_path = f"metadata/{clip_id}.json"
        relative_stems_path = f"stems/{clip_id}"
    else:
        clip_dir.mkdir(parents=True, exist_ok=True)
        stem_dir = clip_dir / "stems"
        mix_path = clip_dir / "mix.wav"
        metadata_path = clip_dir / "metadata.json"
        relative_mix_path = "mix.wav"
        relative_metadata_path = "metadata.json"
        relative_stems_path = "stems"
    if write_stems:
        stem_dir.mkdir(parents=True, exist_ok=True)

    if config.bpm_min > config.bpm_max:
        raise ValueError("bpm_min must be less than or equal to bpm_max")
    bpm = float(rng.integers(config.bpm_min, config.bpm_max + 1))
    beats_per_measure, beat_unit = _choose_time_signature(rng, config.time_signatures)
    key_index = int(rng.integers(0, len(KEYS)))
    key = KEYS[key_index]
    if config.source_count_min > config.source_count_max:
        raise ValueError("source_count_min must be less than or equal to source_count_max")
    source_count = int(rng.integers(config.source_count_min, config.source_count_max + 1))
    chosen_types = _choose_sources(rng, source_count, config)
    bass_note_pool = _minor_note_pool(key_index, root_octave_midi=24, octave_count=2)
    lead_note_pool = _minor_note_pool(key_index, root_octave_midi=48, octave_count=2)

    sources = _make_sources(
        rng,
        chosen_types,
        stem_prefix=relative_stems_path,
        catalog=config.asset_catalog,
        renderer_profile=config.renderer_profile,
        procedural_fallback=config.procedural_fallback,
        include_tags=config.include_tags,
        exclude_tags=config.exclude_tags,
    )
    events = _make_events(
        rng,
        sources,
        bpm,
        beats_per_measure,
        beat_unit,
        bass_note_pool,
        lead_note_pool,
        duration_seconds,
        composition_profile=config.composition_profile,
    )

    stems: dict[str, np.ndarray] = {}
    kick_onsets = [
        event.onset_seconds
        for event in events
        if _source_by_id(sources, event.source_id).source_type == "kick"
    ]

    for source in sources:
        source_events = [event for event in events if event.source_id == source.source_id]
        dry = _render_source(source, source_events, config.sample_rate, duration_seconds, rng)
        wet = _apply_source_effects(dry, source, config.sample_rate)
        if source.source_type != "kick":
            wet = sidechain_duck(
                wet,
                kick_onsets,
                config.sample_rate,
                amount=float(source.effect_parameters.get("duck_amount", 0.0)),
                release_seconds=float(source.effect_parameters.get("duck_release_seconds", 0.22)),
            )
        if source.renderer != "surge_xt_fxp":
            wet = normalize_peak(wet, peak=0.95)
        stems[source.source_id] = wet
        if write_stems:
            sf.write(
                stem_dir / Path(source.stem_path).name, wet, config.sample_rate, subtype="PCM_16"
            )

    mix = np.zeros(int(round(config.sample_rate * duration_seconds)), dtype=np.float32)
    for stem in stems.values():
        mix += stem
    mix = soft_limiter(mix, ceiling=0.98)
    sf.write(mix_path, mix, config.sample_rate, subtype="PCM_16")

    metadata = ClipMetadata(
        dataset_version="comet-edm-v1",
        seed=seed,
        sample_rate=config.sample_rate,
        duration_seconds=duration_seconds,
        bpm=bpm,
        time_signature=f"{beats_per_measure}/{beat_unit}",
        beats_per_measure=beats_per_measure,
        beat_unit=beat_unit,
        key=key,
        paths={
            "mix": relative_mix_path,
            "metadata": relative_metadata_path,
            "stems": relative_stems_path,
        },
        sources=sources,
        events=sorted(events, key=lambda event: (event.onset_seconds, event.event_id)),
    )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def write_visualizer_html(out_dir: Path, clips: list[ClipMetadata]) -> None:
    payload = [
        {
            "clip_id": f"clip_{index:04d}",
            "metadata": clip.model_dump(mode="json"),
        }
        for index, clip in enumerate(clips)
    ]
    payload_json = json.dumps(payload, sort_keys=True)
    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Comet Audio Visualizer</title>",
            "<style>",
            ":root{color-scheme:dark;--bg:#101214;--panel:#191d21;--line:#303740;"
            "--text:#f1f3f4;--muted:#a9b0b7;--attack:#f2bf5e;--held:#4dbf87;"
            "--release:#5fa8e8}",
            "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);"
            "font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}",
            "header{display:flex;align-items:center;justify-content:space-between;gap:16px;"
            "padding:18px 22px;border-bottom:1px solid var(--line);background:#15181b;position:sticky;"
            "top:0;z-index:5}h1{font-size:18px;margin:0;font-weight:650;letter-spacing:0}",
            ".controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap}.field{display:grid;gap:6px}"
            "label{font-size:12px;color:var(--muted)}select,button{height:34px;border:1px solid #3a424b;"
            "background:#20252a;color:var(--text);border-radius:6px;padding:0 10px;font:inherit}"
            "button{cursor:pointer;min-width:40px}.wrap{padding:18px 22px;display:grid;gap:16px}",
            ".stats{display:flex;gap:10px;flex-wrap:wrap}.stat{border:1px solid var(--line);"
            "border-radius:6px;background:var(--panel);padding:8px 10px;min-width:86px}.stat b{display:block;"
            "font-size:13px}.stat span{display:block;color:var(--muted);font-size:12px;margin-top:2px}",
            ".transport{display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:14px;align-items:center;"
            "border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:12px}"
            "audio{width:100%}.time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}",
            ".legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.key{display:flex;"
            "align-items:center;gap:6px}.swatch{width:12px;height:12px;border-radius:3px}.attack{background:var(--attack)}"
            ".held{background:var(--held)}.release{background:var(--release)}",
            ".timeline{position:relative;border:1px solid var(--line);border-radius:8px;background:#131619;"
            "overflow:hidden}.ruler{height:32px;border-bottom:1px solid var(--line);display:grid;"
            "grid-template-columns:190px minmax(520px,1fr);background:#171b1f}.ruler-spacer{border-right:1px solid var(--line)}"
            ".ruler-scale{position:relative}.tick{position:absolute;top:0;bottom:0;border-left:1px solid #38414a;"
            "font-size:11px;color:var(--muted);padding-left:5px;line-height:30px}",
            ".lane{display:grid;grid-template-columns:190px minmax(520px,1fr);min-height:54px;border-bottom:1px solid "
            "var(--line)}.lane:last-child{border-bottom:0}.lane-label{padding:9px 10px;border-right:1px solid "
            "var(--line);background:#171b1f}.lane-label b{display:block;font-size:13px;white-space:nowrap;"
            "overflow:hidden;text-overflow:ellipsis}.lane-label span{display:block;color:var(--muted);font-size:12px;"
            "margin-top:3px}.lane-track{position:relative;min-height:54px;background:#111417}",
            ".event{position:absolute;top:13px;height:26px;border-radius:5px;overflow:hidden;border:1px solid "
            "rgba(255,255,255,.16);display:flex;min-width:2px}.seg{height:100%;flex:0 0 auto}.seg.attack{"
            "background:var(--attack)}.seg.held{background:var(--held)}.seg.release{background:var(--release)}",
            ".playhead{position:absolute;top:0;bottom:0;width:2px;background:#ffffff;box-shadow:0 0 0 1px #000;"
            "pointer-events:none;z-index:4}.empty{padding:24px;color:var(--muted)}",
            "@media(max-width:760px){header{align-items:stretch;flex-direction:column}.transport{grid-template-columns:1fr}"
            ".lane,.ruler{grid-template-columns:130px minmax(420px,1fr)}.wrap{padding:12px}h1{font-size:16px}}",
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>Comet Audio Visualizer</h1>",
            '<div class="controls">',
            '<div class="field"><label for="clipSelect">Clip</label><select id="clipSelect"></select></div>',
            '<div class="field"><label for="audioSelect">Audio</label><select id="audioSelect"></select></div>',
            '<button id="prevButton" type="button" title="Previous clip">&lt;</button>',
            '<button id="nextButton" type="button" title="Next clip">&gt;</button>',
            "</div>",
            "</header>",
            '<main class="wrap">',
            '<section class="stats" id="stats"></section>',
            '<section class="transport"><audio id="audio" controls preload="metadata"></audio>'
            '<div class="time" id="timeReadout">0.000 / 0.000</div></section>',
            '<section class="legend"><div class="key"><span class="swatch attack"></span>Attack</div>'
            '<div class="key"><span class="swatch held"></span>Held</div>'
            '<div class="key"><span class="swatch release"></span>Release</div></section>',
            '<section class="timeline" id="timeline"></section>',
            "</main>",
            f'<script id="comet-data" type="application/json">{payload_json}</script>',
            "<script>",
            "const clips=JSON.parse(document.getElementById('comet-data').textContent);",
            "const clipSelect=document.getElementById('clipSelect');",
            "const audioSelect=document.getElementById('audioSelect');",
            "const audio=document.getElementById('audio');",
            "const timeline=document.getElementById('timeline');",
            "const stats=document.getElementById('stats');",
            "const timeReadout=document.getElementById('timeReadout');",
            "const prevButton=document.getElementById('prevButton');",
            "const nextButton=document.getElementById('nextButton');",
            "let currentIndex=0;let selectedClip=null;let playhead=null;",
            "const sourceColors={kick:'#e16f64',snare_clap:'#d89547',hat_noise:'#b9b84d',fm_bass:'#45a86f',"
            "fm_growl:'#46a6a5',wub_bass:'#5b8fd8',pluck_stab:'#a16fd1',riser_impact:'#d36fa2'};",
            "function fmt(value){return Number(value).toFixed(3)}",
            "function pct(value,duration){return Math.max(0,Math.min(100,(value/duration)*100))}",
            "function clipPath(clip,path){return `${clip.clip_id}/${path}`}",
            "function sourceLabel(source){return `${source.source_id} - ${source.source_type}`}",
            "function setOptions(){clips.forEach((clip,index)=>{const option=document.createElement('option');"
            "const m=clip.metadata;option.value=String(index);option.textContent=`${clip.clip_id} - ${m.bpm} BPM - "
            "${m.time_signature} - ${m.key} minor`;clipSelect.appendChild(option)})}",
            "function renderStats(clip){const m=clip.metadata;stats.innerHTML='';"
            "const rows=[['BPM',m.bpm],['Meter',m.time_signature],['Key',`${m.key} minor`],"
            "['Duration',`${fmt(m.duration_seconds)}s`],['Sources',m.sources.length],['Events',m.events.length],"
            "['Seed',m.seed]];rows.forEach(([label,value])=>{const el=document.createElement('div');el.className='stat';"
            "el.innerHTML=`<b>${value}</b><span>${label}</span>`;stats.appendChild(el)})}",
            "function renderAudioOptions(clip){const m=clip.metadata;audioSelect.innerHTML='';"
            "const mix=document.createElement('option');mix.value=clipPath(clip,m.paths.mix);mix.textContent='mix.wav';"
            "audioSelect.appendChild(mix);m.sources.forEach(source=>{const option=document.createElement('option');"
            "option.value=clipPath(clip,source.stem_path);option.textContent=`${source.source_id} - ${source.source_type}`;"
            "audioSelect.appendChild(option)});audio.src=audioSelect.value}",
            "function renderRuler(duration){const ruler=document.createElement('div');ruler.className='ruler';"
            "const spacer=document.createElement('div');spacer.className='ruler-spacer';const scale=document.createElement('div');"
            "scale.className='ruler-scale';ruler.append(spacer,scale);"
            "const step=duration<=10?1:2;for(let t=0;t<=duration+0.0001;t+=step){const tick=document.createElement('div');"
            "tick.className='tick';tick.style.left=`${pct(t,duration)}%`;tick.textContent=`${t.toFixed(0)}s`;"
            "scale.appendChild(tick)}timeline.appendChild(ruler)}",
            "function eventTitle(event){const note=event.midi_note===null?'':` - MIDI ${event.midi_note}`;"
            "return `${event.event_type}${note} - onset ${fmt(event.onset_seconds)} - attack "
            "${fmt(event.attack_seconds)} - release ${fmt(event.release_seconds)} - offset ${fmt(event.offset_seconds)}`}",
            "function renderEvent(track,event,duration,color){const start=event.onset_seconds;"
            "const end=event.offset_seconds;const attackEnd=Math.min(end,start+event.attack_seconds);"
            "const releaseStart=Math.max(attackEnd,end-event.release_seconds);"
            "const total=Math.max(0.001,end-start);const block=document.createElement('div');block.className='event';"
            "block.title=eventTitle(event);block.style.left=`${pct(start,duration)}%`;block.style.width="
            "`${Math.max(.35,pct(total,duration))}%`;block.style.borderColor=color;"
            "const attack=document.createElement('div');attack.className='seg attack';"
            "attack.style.width=`${((attackEnd-start)/total)*100}%`;const held=document.createElement('div');"
            "held.className='seg held';held.style.width=`${((releaseStart-attackEnd)/total)*100}%`;"
            "const release=document.createElement('div');release.className='seg release';release.style.width="
            "`${((end-releaseStart)/total)*100}%`;block.append(attack,held,release);track.appendChild(block)}",
            "function renderLane(source,events,duration){const lane=document.createElement('div');lane.className='lane';"
            "const label=document.createElement('div');label.className='lane-label';label.innerHTML="
            "`<b>${sourceLabel(source)}</b><span>${events.length} events - gain ${source.gain_db} dB - pan ${source.pan}</span>`;"
            "const track=document.createElement('div');track.className='lane-track';const color=sourceColors[source.source_type]||'#aaa';"
            "events.forEach(event=>renderEvent(track,event,duration,color));lane.append(label,track);timeline.appendChild(lane)}",
            "function renderTimeline(clip){const m=clip.metadata;timeline.innerHTML='';renderRuler(m.duration_seconds);"
            "m.sources.forEach(source=>{const events=m.events.filter(event=>event.source_id===source.source_id);"
            "renderLane(source,events,m.duration_seconds)});playhead=document.createElement('div');"
            "playhead.className='playhead';playhead.style.left='0%';timeline.appendChild(playhead)}",
            "function loadClip(index){currentIndex=(index+clips.length)%clips.length;selectedClip=clips[currentIndex];"
            "clipSelect.value=String(currentIndex);renderStats(selectedClip);renderAudioOptions(selectedClip);"
            "renderTimeline(selectedClip);updateTime()}",
            "function updateTime(){const duration=selectedClip?selectedClip.metadata.duration_seconds:0;"
            "const current=audio.currentTime||0;timeReadout.textContent=`${fmt(current)} / ${fmt(duration)}`;"
            "if(playhead&&duration>0){const track=timeline.querySelector('.lane-track');"
            "if(track){const timelineRect=timeline.getBoundingClientRect();const trackRect=track.getBoundingClientRect();"
            "const x=(trackRect.left-timelineRect.left)+(Math.max(0,Math.min(1,current/duration))*trackRect.width);"
            "playhead.style.left=`${x}px`}}}",
            "clipSelect.addEventListener('change',event=>loadClip(Number(event.target.value)));",
            "audioSelect.addEventListener('change',()=>{const wasPlaying=!audio.paused;const current=audio.currentTime;"
            "audio.src=audioSelect.value;audio.currentTime=current;if(wasPlaying){audio.play()}});",
            "audio.addEventListener('timeupdate',updateTime);audio.addEventListener('loadedmetadata',updateTime);",
            "prevButton.addEventListener('click',()=>loadClip(currentIndex-1));",
            "nextButton.addEventListener('click',()=>loadClip(currentIndex+1));",
            "setOptions();if(clips.length){loadClip(0)}else{timeline.innerHTML='<div class=\"empty\">No clips</div>'}",
            "</script>",
            "</body></html>",
        ]
    )
    (out_dir / "visualizer.html").write_text(html, encoding="utf-8")


def _choose_sources(
    rng: np.random.Generator, source_count: int, config: GeneratorConfig | None = None
) -> list[SourceType]:
    config = config or GeneratorConfig()
    if config.composition_profile == "percussion_v1":
        return _choose_percussion_sources(rng, source_count, config)
    if config.composition_profile == "surge_patches_v1":
        return _choose_surge_patch_sources(rng, source_count, config)
    if config.composition_profile != "edm_v1":
        raise ValueError(
            "composition_profile must be one of: edm_v1, percussion_v1, surge_patches_v1"
        )
    required: list[SourceType] = ["kick", "snare", "closed_hat", "synth_bass"]
    chosen = required[: min(source_count, len(required))]
    while len(chosen) < source_count:
        candidate = OPTIONAL_SOURCE_POOL[int(rng.integers(0, len(OPTIONAL_SOURCE_POOL)))]
        if candidate in {"electric_bass", "acoustic_bass"} and any(
            source in {"electric_bass", "acoustic_bass"} for source in chosen
        ):
            continue
        chosen.append(candidate)
    return chosen


def _choose_percussion_sources(
    rng: np.random.Generator, source_count: int, config: GeneratorConfig
) -> list[SourceType]:
    available = _available_percussion_source_types(config)
    if not available:
        raise ValueError("percussion_v1 requires at least one percussion source bucket")
    anchors = [
        source_type for source_type in ("kick", "snare", "closed_hat") if source_type in available
    ]
    chosen = anchors[: min(source_count, len(anchors))]
    while len(chosen) < source_count:
        chosen.append(str(rng.choice(available)))
    rng.shuffle(chosen)
    return list(chosen)


def _choose_surge_patch_sources(
    rng: np.random.Generator, source_count: int, config: GeneratorConfig
) -> list[SourceType]:
    surge_counts = _available_surge_source_type_counts(config)
    percussion_counts = _available_percussion_source_type_counts(config)
    if not surge_counts:
        raise ValueError("surge_patches_v1 requires indexed Surge XT patch assets")
    if sum(surge_counts.values()) + sum(percussion_counts.values()) < source_count:
        raise ValueError("Not enough unique assets available for requested source count")

    percussion_target = min(max(2, source_count // 4), 4, source_count)
    surge_target = source_count - percussion_target
    chosen: list[SourceType] = []
    for _ in range(surge_target):
        chosen.append(_choose_from_counts(rng, surge_counts))
    for anchor in ("kick", "snare", "closed_hat"):
        if len(chosen) >= source_count:
            break
        if percussion_counts.get(anchor, 0) > 0:
            chosen.append(anchor)
            percussion_counts[anchor] -= 1
    while len(chosen) < source_count:
        pool = percussion_counts if sum(percussion_counts.values()) > 0 else surge_counts
        chosen.append(_choose_from_counts(rng, pool))
    rng.shuffle(chosen)
    return list(chosen)


def _choose_from_counts(rng: np.random.Generator, counts: Counter[SourceType]) -> SourceType:
    available = [source_type for source_type, count in counts.items() if count > 0]
    if not available:
        raise ValueError("No available source types remain")
    chosen = str(rng.choice(available))
    counts[chosen] -= 1
    return chosen


def _available_surge_source_type_counts(config: GeneratorConfig) -> Counter[SourceType]:
    if config.asset_catalog is None:
        return Counter()
    available: Counter[SourceType] = Counter()
    for entry in config.asset_catalog.entries:
        if entry.renderer != "surge_xt_fxp":
            continue
        if entry.source_type in PERCUSSION_SOURCE_TYPES:
            continue
        if not _asset_is_selectable(entry, config):
            continue
        available[entry.source_type] += 1
    return available


def _available_surge_source_types(config: GeneratorConfig) -> list[SourceType]:
    return list(_available_surge_source_type_counts(config))


def _available_percussion_source_type_counts(config: GeneratorConfig) -> Counter[SourceType]:
    if config.procedural_fallback or config.asset_catalog is None:
        return Counter({source_type: 1_000_000 for source_type in PERCUSSION_SOURCE_TYPES})
    available: Counter[SourceType] = Counter()
    for entry in config.asset_catalog.entries:
        if entry.source_type in PERCUSSION_SOURCE_TYPES and _asset_is_selectable(entry, config):
            available[entry.source_type] += 1
    return available


def _available_percussion_source_types(config: GeneratorConfig) -> list[SourceType]:
    return list(_available_percussion_source_type_counts(config))


def _choose_clip_duration(rng: np.random.Generator, config: GeneratorConfig) -> float:
    if config.duration_profile is not None:
        profile = config.duration_profile
    elif config.composition_profile == "surge_patches_v1":
        profile = (8.0, 8.0, 8.0, 12.0, 15.0)
    else:
        profile = (config.duration_seconds,)
    if not profile:
        raise ValueError("duration_profile must contain at least one duration")
    return float(rng.choice(profile))


def _choose_time_signature(
    rng: np.random.Generator, time_signatures: tuple[str, ...]
) -> tuple[int, int]:
    if not time_signatures:
        raise ValueError("At least one time signature is required")
    selected = str(rng.choice(time_signatures))
    return _parse_time_signature(selected)


def _parse_time_signature(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid time signature: {value!r}")
    numerator = int(parts[0])
    denominator = int(parts[1])
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Invalid time signature: {value!r}")
    if denominator not in {2, 4}:
        raise ValueError("Only /2 and /4 meters are supported in v0")
    return numerator, denominator


def _minor_note_pool(key_index: int, root_octave_midi: int, octave_count: int) -> list[int]:
    root = root_octave_midi + key_index
    return [
        root + octave * 12 + interval
        for octave in range(octave_count)
        for interval in MINOR_INTERVALS
    ]


def _make_sources(
    rng: np.random.Generator,
    chosen_types: list[SourceType],
    stem_prefix: str = "stems",
    catalog: AssetCatalog | None = None,
    renderer_profile: str = "hybrid_v1",
    procedural_fallback: bool = True,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
) -> list[SourceMetadata]:
    if renderer_profile not in {"hybrid_v1", "procedural_only", "plugin_v1"}:
        raise ValueError("renderer_profile must be one of: hybrid_v1, procedural_only, plugin_v1")
    sources = []
    type_counts = {
        source_type: sum(1 for chosen_type in chosen_types if chosen_type == source_type)
        for source_type in set(chosen_types)
    }
    type_seen: dict[SourceType, int] = {}
    used_asset_keys: set[str] = set()
    for index, source_type in enumerate(chosen_types):
        instance_index = type_seen.get(source_type, 0)
        type_seen[source_type] = instance_index + 1
        asset = _choose_asset_for_source(
            rng,
            catalog,
            source_type,
            renderer_profile,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_asset_keys=used_asset_keys,
        )
        if asset is None and (renderer_profile == "plugin_v1" or not procedural_fallback):
            raise ValueError(f"No unused asset renderer available for {source_type!r}")
        if asset is not None:
            used_asset_keys.add(_asset_unique_key(asset))
        recipe = PROCEDURAL_RECIPE_BY_SOURCE_TYPE.get(source_type, "pluck_stab")
        base_gain = _default_gain(source_type, recipe)
        gain_jitter = 0.45 if asset is not None else 1.2
        gain = (asset.default_gain_db if asset is not None else base_gain) + float(
            rng.normal(0.0, gain_jitter)
        )
        pan = 0.0
        source_id = f"source_{index:03d}"
        family = asset.family if asset is not None else _family_for_source_type(source_type)
        instrument = asset.instrument if asset is not None else source_type
        articulation = (
            asset.articulation if asset is not None else _default_articulation(source_type)
        )
        sources.append(
            SourceMetadata(
                source_id=source_id,
                source_type=source_type,
                family=family,
                instrument=instrument,
                articulation=articulation,
                renderer=asset.renderer if asset is not None else "procedural_synth",
                asset_id=asset.asset_id if asset is not None else None,
                synth_parameters=_synth_parameters(
                    rng,
                    recipe,
                    instance_index=instance_index,
                    instance_count=type_counts[source_type],
                )
                | (
                    {
                        "asset_path": str(catalog.resolve_path(asset))
                        if catalog is not None
                        else asset.path,
                        "plugin_path": str(catalog.resolve_path(asset))
                        if catalog is not None
                        else asset.path,
                        "preset_path": str(catalog.resolve_preset_path(asset))
                        if catalog is not None and catalog.resolve_preset_path(asset) is not None
                        else asset.preset_path,
                        "asset_root_key": asset.root_key,
                        "asset_note_min": asset.note_min,
                        "asset_note_max": asset.note_max,
                        "asset_velocity_min": asset.velocity_min,
                        "asset_velocity_max": asset.velocity_max,
                        "asset_duration_seconds": asset.metadata.get("duration_seconds"),
                        "asset_audio_sha256": asset.metadata.get("audio_sha256"),
                        "surge_amp_attack_seconds": round(float(rng.uniform(0.004, 0.18)), 5),
                        "surge_amp_release_seconds": round(float(rng.uniform(0.08, 0.65)), 5),
                    }
                    if asset is not None
                    else {}
                ),
                effect_parameters=_effect_parameters(
                    rng,
                    recipe,
                    instance_index=instance_index,
                    instance_count=type_counts[source_type],
                ),
                gain_db=round(gain, 3),
                pan=round(pan, 3),
                stem_path=f"{stem_prefix}/{source_id}.wav",
            )
        )
    return sources


def _choose_asset_for_source(
    rng: np.random.Generator,
    catalog: AssetCatalog | None,
    source_type: SourceType,
    renderer_profile: str,
    include_tags: tuple[str, ...] = (),
    exclude_tags: tuple[str, ...] = (),
    exclude_asset_keys: set[str] | None = None,
) -> AssetEntry | None:
    if catalog is None or not catalog.entries or renderer_profile == "procedural_only":
        return None
    exclude_asset_keys = exclude_asset_keys or set()
    if renderer_profile == "plugin_v1" and source_type in PERCUSSION_SOURCE_TYPES:
        renderers = ("wav_one_shot", "surge_xt_fxp", "dawdreamer_plugin", "sfz_instrument")
    elif renderer_profile == "plugin_v1":
        renderers = ("surge_xt_fxp", "dawdreamer_plugin", "wav_one_shot", "sfz_instrument")
    else:
        renderers = ("wav_one_shot", "sfz_instrument", "surge_xt_fxp", "dawdreamer_plugin")
    candidates: list[AssetEntry] = []
    for renderer in renderers:
        candidates = [
            entry
            for entry in catalog.candidates(source_type, renderer)
            if _asset_matches_tags(entry, include_tags, exclude_tags)
            and not _asset_is_broken(entry)
            and _asset_unique_key(entry) not in exclude_asset_keys
        ]
        if candidates:
            break
    if not candidates:
        return None
    weights = np.array([max(float(entry.weight), 0.0) for entry in candidates], dtype=np.float64)
    if float(weights.sum()) <= 0:
        raise ValueError(f"Asset candidates for {source_type!r} all have non-positive weight")
    index = int(rng.choice(len(candidates), p=weights / weights.sum()))
    return candidates[index]


def _asset_unique_key(entry: AssetEntry) -> str:
    audio_hash = entry.metadata.get("audio_sha256")
    if isinstance(audio_hash, str) and audio_hash:
        return f"audio:{audio_hash}"
    if entry.preset_path:
        return f"preset:{Path(entry.preset_path).as_posix().lower()}"
    return f"asset:{entry.asset_id}"


def _asset_is_selectable(entry: AssetEntry, config: GeneratorConfig) -> bool:
    return _asset_matches_tags(
        entry, config.include_tags, config.exclude_tags
    ) and not _asset_is_broken(entry)


def _asset_is_broken(entry: AssetEntry) -> bool:
    return entry.asset_id in BROKEN_SURGE_ASSET_IDS or entry.asset_id.startswith(
        BROKEN_SURGE_ASSET_ID_PREFIXES
    )


def _asset_matches_tags(
    entry: AssetEntry, include_tags: tuple[str, ...], exclude_tags: tuple[str, ...]
) -> bool:
    tags = set(entry.tags)
    return set(include_tags).issubset(tags) and tags.isdisjoint(exclude_tags)


def _family_for_source_type(source_type: str) -> str:
    return FAMILY_BY_SOURCE_TYPE.get(source_type, "tonal")


def _default_articulation(source_type: str) -> str:
    if source_type in {"riser", "noise_sweep"}:
        return "sweep"
    if source_type == "impact":
        return "hit"
    if source_type.endswith("_bass"):
        return "held"
    if source_type in {"pad_chord", "guitar_strum"}:
        return "chord"
    return "hit"


def _default_gain(source_type: str, recipe: str) -> float:
    if source_type == "kick":
        return -3.5
    if source_type in {"snare", "clap"}:
        return -8.0
    if source_type in {"closed_hat", "open_hat", "cymbal"}:
        return -13.0
    if source_type in {"percussion", "tom"}:
        return -11.0
    if recipe == "fm_bass":
        return -10.0
    if recipe == "pluck_stab":
        return -2.0
    return -14.0


def _variant_position(instance_index: int, instance_count: int) -> float:
    if instance_count <= 1:
        return 0.5
    return instance_index / (instance_count - 1)


def _variant_label(position: float) -> str:
    if position < 0.34:
        return "dark"
    if position < 0.67:
        return "balanced"
    return "bright"


def _variant_base(
    source_type: SourceType, instance_index: int, instance_count: int
) -> dict[str, Any]:
    position = _variant_position(instance_index, instance_count)
    return {
        "procedural_recipe": source_type,
        "timbre_variant": _variant_label(position),
        "variant_index": instance_index,
        "variant_count": instance_count,
        "variant_position": round(position, 5),
    }


def _synth_parameters(
    rng: np.random.Generator,
    source_type: SourceType,
    instance_index: int,
    instance_count: int,
) -> dict[str, Any]:
    variant = _variant_base(source_type, instance_index, instance_count)
    position = float(variant["variant_position"])
    if source_type == "kick":
        variant.update(
            {
                "start_hz": float(rng.uniform(90 + 22 * position, 112 + 32 * position)),
                "end_hz": float(rng.uniform(38 + 8 * position, 48 + 14 * position)),
            }
        )
        return variant
    if source_type == "snare_clap":
        variant.update(
            {
                "noise_tone_hz": float(rng.uniform(1200 + 1600 * position, 2100 + 2400 * position)),
                "body_hz": float(rng.uniform(160 + 34 * position, 205 + 55 * position)),
            }
        )
        return variant
    if source_type == "hat_noise":
        variant.update(
            {"highpass_hz": float(rng.uniform(4200 + 3600 * position, 6200 + 5200 * position))}
        )
        return variant
    if source_type == "fm_bass":
        ratio_choices = ([0.5, 1.0], [1.0, 1.5, 2.0], [2.0, 3.0, 4.0])
        bucket = min(2, round(position * 2))
        voice_models = ("sub_clean", "fm_round", "edge_distorted", "noisy_reese")
        voice_model = voice_models[min(instance_index, len(voice_models) - 1)]
        variant.update(
            {
                "voice_model": voice_model,
                "ratio": float(rng.choice(ratio_choices[bucket])),
                "fm_index": float(rng.uniform(1.0 + 2.0 * position, 2.8 + 5.2 * position)),
                "index_decay_seconds": float(
                    rng.uniform(0.2 - 0.11 * position, 0.38 - 0.12 * position)
                ),
                "sub_mix": float(rng.uniform(0.42 - 0.24 * position, 0.62 - 0.2 * position)),
                "noise_mix": float(rng.uniform(0.0 + 0.012 * position, 0.015 + 0.04 * position)),
                "wavefold_drive": float(rng.uniform(1.0 + 0.65 * position, 1.7 + 1.4 * position)),
            }
        )
        return variant
    if source_type == "fm_growl":
        voice_models = ("sub_growl", "vowel_growl", "edge_growl")
        bucket = min(2, round(position * 2))
        voice_model = voice_models[min(instance_index, len(voice_models) - 1)]
        variant.update(
            {
                "voice_model": voice_model,
                "ratio_a": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
                "ratio_b": float(rng.choice([2.0, 3.0, 4.0, 5.0])),
                "fm_index_a": float(rng.uniform(2.2 + 2.2 * position, 5.0 + 4.0 * position)),
                "fm_index_b": float(rng.uniform(1.0 + 1.5 * position, 3.2 + 3.0 * position)),
                "sub_mix": float(rng.uniform(0.68 - 0.35 * position, 0.86 - 0.28 * position)),
                "noise_mix": float(rng.uniform(0.004 + 0.012 * position, 0.024 + 0.05 * position)),
                "wavefold_drive": float(rng.uniform(1.4 + 0.8 * position, 2.5 + 2.1 * position)),
                "formant_shift": float(rng.uniform(0.65 + 0.5 * position, 1.05 + 1.0 * position)),
            }
        )
        return variant
    if source_type == "wub_bass":
        ratio_choices = ([0.5, 1.0], [1.0, 1.5], [1.5, 2.0, 3.0])
        bucket = min(2, round(position * 2))
        voice_models = ("sub_pulse", "fm_pulse", "saw_pulse")
        voice_model = voice_models[min(instance_index, len(voice_models) - 1)]
        variant.update(
            {
                "voice_model": voice_model,
                "ratio": float(rng.choice(ratio_choices[bucket])),
                "fm_index": float(rng.uniform(1.4 + 1.3 * position, 3.2 + 3.1 * position)),
                "sub_mix": float(rng.uniform(0.78 - 0.38 * position, 0.9 - 0.3 * position)),
                "harmonic_mix": float(rng.uniform(0.04 + 0.12 * position, 0.16 + 0.34 * position)),
                "wavefold_drive": float(rng.uniform(1.2 + 0.5 * position, 2.0 + 1.6 * position)),
            }
        )
        return variant
    if source_type == "pluck_stab":
        voice_models = ("round", "hollow", "bright_saw", "metallic")
        voice_model = voice_models[min(instance_index, len(voice_models) - 1)]
        variant.update(
            {
                "voice_model": voice_model,
                "detune_cents": float(rng.uniform(2 + 8 * position, 7 + 18 * position)),
                "filter_hz": float(rng.uniform(3600 + 6200 * position, 6800 + 13200 * position)),
                "transient_click": float(
                    rng.uniform(0.12 + 0.25 * position, 0.34 + 0.42 * position)
                ),
                "octave_mix": float(rng.uniform(0.38 - 0.2 * position, 0.62 - 0.1 * position)),
                "saw_mix": float(rng.uniform(0.16 + 0.24 * position, 0.38 + 0.48 * position)),
            }
        )
        return variant
    variant.update(
        {
            "filter_start_hz": float(rng.uniform(300 + 700 * position, 620 + 1200 * position)),
            "filter_end_hz": float(rng.uniform(3800 + 3200 * position, 6400 + 5400 * position)),
            "noise_color": float(rng.uniform(0.35 + 0.25 * position, 0.65 + 0.35 * position)),
            "impact_weight": float(rng.uniform(0.28 - 0.16 * position, 0.48 - 0.1 * position)),
        }
    )
    return variant


def _effect_parameters(
    rng: np.random.Generator,
    source_type: SourceType,
    instance_index: int,
    instance_count: int,
) -> dict[str, Any]:
    position = _variant_position(instance_index, instance_count)
    params: dict[str, Any] = {
        "drive": float(rng.uniform(0.95 + 0.35 * position, 1.55 + 1.2 * position)),
        "duck_amount": 0.0 if source_type == "kick" else float(rng.uniform(0.18, 0.55)),
        "duck_release_seconds": float(rng.uniform(0.16, 0.32)),
    }
    return {
        key: round(value, 4) if isinstance(value, float) else value for key, value in params.items()
    }


def _make_events(
    rng: np.random.Generator,
    sources: list[SourceMetadata],
    bpm: float,
    beats_per_measure: int,
    beat_unit: int,
    bass_note_pool: list[int],
    lead_note_pool: list[int],
    duration: float,
    composition_profile: str = "edm_v1",
) -> list[EventMetadata]:
    events: list[EventMetadata] = []
    beat = 60.0 / bpm
    measure = beats_per_measure * beat
    backbeat_positions = _backbeat_positions(beats_per_measure)
    event_index = 0
    percussion_template = (
        str(rng.choice(PERCUSSION_RHYTHM_TEMPLATES))
        if composition_profile in {"percussion_v1", "surge_patches_v1"}
        else "edm_v1"
    )
    for source in sources:
        recipe = _procedural_recipe_for_source(source)
        rhythm_subdivision = int(rng.choice(RHYTHM_SUBDIVISIONS))
        if composition_profile == "percussion_v1" or (
            composition_profile == "surge_patches_v1"
            and source.source_type in PERCUSSION_SOURCE_TYPES
        ):
            onsets, length, rhythm_subdivision = _percussion_profile_events(
                rng,
                source.source_type,
                duration,
                beat,
                measure,
                beats_per_measure,
                percussion_template,
            )
            length = _asset_aware_percussion_length(source, length, duration)
            onsets = _thin_same_track_overlaps(
                rng,
                onsets,
                source.source_type,
                length,
                duration,
                percussion_template,
            )
        elif composition_profile == "surge_patches_v1":
            onsets, held_length, rhythm_subdivision = _surge_patch_events(
                rng,
                source,
                duration,
                beat,
            )
            attack, release = _surge_source_envelope(source)
            length = held_length + attack + release
            onsets = _thin_same_track_overlaps(
                rng,
                onsets,
                source.source_type,
                length,
                duration,
                percussion_template,
            )
        elif source.source_type == "kick":
            onsets = _metered_onsets(duration, measure, [0.0])
            extra = _metered_onsets(
                duration, measure, [position * beat for position in range(1, beats_per_measure)]
            )
            extra = extra[rng.random(len(extra)) > 0.62] if len(extra) else extra
            onsets = np.sort(np.concatenate([onsets, extra]))
            length = 0.34
        elif source.source_type in {"snare", "clap"}:
            onsets = _metered_onsets(
                duration,
                measure,
                [position * beat for position in backbeat_positions],
            )
            length = 0.28
        elif source.source_type in {"closed_hat", "open_hat", "cymbal", "percussion"}:
            rhythm_subdivision = int(rng.choice([2, 4, 6, 8]))
            onsets = _random_grid_onsets(
                rng,
                duration,
                beat,
                rhythm_subdivision,
                density=float(rng.uniform(0.42, 0.86)),
                force_downbeats=True,
            )
            length = 0.08 if source.source_type == "closed_hat" else 0.18
        elif source.source_type in {"synth_bass", "electric_bass", "acoustic_bass"}:
            rhythm_subdivision = int(rng.choice([2, 3, 4]))
            onsets = _random_grid_onsets(
                rng,
                duration,
                beat,
                rhythm_subdivision,
                density=float(rng.uniform(0.34, 0.64)),
                force_downbeats=True,
            )
            length = beat * float(rng.uniform(0.75, 1.75))
        elif recipe == "fm_growl":
            offsets = [0.0]
            if beats_per_measure > 2 and rng.random() < 0.35:
                offsets.append(measure * 0.5)
            onsets = _metered_onsets(duration, measure, offsets)
            onsets = onsets[rng.random(len(onsets)) > 0.45] if len(onsets) else onsets
            if len(onsets) == 0:
                onsets = np.array([0.0], dtype=np.float32)
            length = beat * float(rng.choice([0.75, 1.0, 1.5]))
        elif recipe == "wub_bass":
            offsets = [0.0]
            if beats_per_measure > 2 and rng.random() < 0.35:
                offsets.append(measure * 0.5)
            onsets = _metered_onsets(duration, measure, offsets)
            onsets = onsets[rng.random(len(onsets)) > 0.35] if len(onsets) else onsets
            if len(onsets) == 0:
                onsets = np.array([0.0], dtype=np.float32)
            length = beat * float(rng.choice([0.75, 1.0, 1.5]))
        elif recipe == "pluck_stab":
            rhythm_subdivision = int(rng.choice([3, 4, 6, 8]))
            motif_density = float(rng.uniform(0.36, 0.74))
            onsets = _random_grid_onsets(
                rng,
                duration,
                beat,
                rhythm_subdivision,
                density=motif_density,
                force_downbeats=False,
            )
            if len(onsets) > 0 and rng.random() < 0.72:
                onsets = _accent_repeating_motif(rng, onsets, beat, rhythm_subdivision)
            length = beat / rhythm_subdivision * float(rng.uniform(0.58, 1.25))
        else:
            onsets = np.array([max(0.0, duration - float(rng.uniform(1.8, 3.2)))])
            length = duration - float(onsets[0])

        next_available_onset = 0.0
        for onset in sorted(float(value) for value in onsets):
            if onset >= duration:
                continue
            if onset + 1e-6 < next_available_onset:
                continue
            offset = min(duration, float(onset + length))
            attack, release = _event_envelope_labels(source, recipe, offset - float(onset), rng)
            if attack + release > offset - float(onset):
                release = max(0.0, offset - float(onset) - attack)
            if offset <= onset:
                continue
            note = None
            if source.renderer == "surge_xt_fxp":
                note_pool = (
                    bass_note_pool
                    if source.source_type in {"synth_bass", "electric_bass", "acoustic_bass"}
                    else lead_note_pool
                )
                note = int(rng.choice(_notes_in_asset_range(source, note_pool)))
            elif source.source_type in {"synth_bass", "electric_bass", "acoustic_bass"}:
                note = _choose_bass_note(
                    rng,
                    _notes_in_asset_range(source, bass_note_pool),
                    prefer_low=source.source_type != "synth_bass",
                )
            elif recipe == "pluck_stab":
                note = int(rng.choice(_notes_in_asset_range(source, lead_note_pool)))
            velocity_min = max(0.0, float(source.synth_parameters.get("asset_velocity_min", 0.62)))
            velocity_max = min(1.0, float(source.synth_parameters.get("asset_velocity_max", 1.0)))
            velocity_low = max(0.62, velocity_min)
            if velocity_low > velocity_max:
                raise ValueError(f"Invalid velocity range for {source.source_id}")
            velocity = float(rng.uniform(velocity_low, velocity_max))
            timbre_variation = _event_timbre_variation(
                rng, recipe, position=float(onset) / duration
            )
            events.append(
                EventMetadata(
                    event_id=f"event_{event_index:04d}",
                    source_id=source.source_id,
                    event_type=source.source_type,
                    onset_seconds=round(float(onset), 6),
                    offset_seconds=round(float(offset), 6),
                    velocity=round(velocity, 4),
                    midi_note=note,
                    attack_seconds=round(float(attack), 5),
                    release_seconds=round(float(release), 5),
                    render_parameters={
                        "beat_seconds": round(beat, 6),
                        "measure_seconds": round(measure, 6),
                        "beat_unit": beat_unit,
                        "rhythm_subdivision": rhythm_subdivision,
                        "duration_beats": round((offset - float(onset)) / beat, 6),
                        "timbre_variation": timbre_variation,
                        "composition_profile": composition_profile,
                        "rhythm_template": percussion_template,
                    },
                )
            )
            next_available_onset = offset
            event_index += 1
    return events


def _percussion_profile_events(
    rng: np.random.Generator,
    source_type: SourceType,
    duration: float,
    beat: float,
    measure: float,
    beats_per_measure: int,
    template: str,
) -> tuple[np.ndarray, float, int]:
    if template == "solo_hits":
        subdivision = int(rng.choice([2, 4, 6, 8]))
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=float(rng.uniform(0.05, 0.16)),
            force_downbeats=False,
        )
        return (
            onsets[: max(1, int(rng.integers(1, 5)))],
            _percussion_length(source_type, beat),
            subdivision,
        )
    if template == "sparse":
        subdivision = int(rng.choice([2, 3, 4]))
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=float(rng.uniform(0.12, 0.32)),
            force_downbeats=source_type == "kick",
        )
        return onsets, _percussion_length(source_type, beat), subdivision
    if source_type == "kick":
        offsets = [0.0]
        if template in {"straight", "breakbeat", "fills"}:
            offsets.extend(position * beat for position in range(1, beats_per_measure))
        if template == "two_step" and beats_per_measure > 2:
            offsets.append((beats_per_measure - 1) * beat)
        if template == "half_time" and beats_per_measure > 2:
            offsets.append(measure * 0.5)
        onsets = _metered_onsets(duration, measure, offsets)
        if len(onsets):
            keep = 0.55 if template in {"breakbeat", "fills"} else 0.8
            onsets = onsets[rng.random(len(onsets)) < keep]
        return _ensure_onsets(onsets, duration), 0.34, 2
    if source_type in {"snare", "clap"}:
        positions = (
            [beats_per_measure - 1]
            if template == "half_time"
            else _backbeat_positions(beats_per_measure)
        )
        onsets = _metered_onsets(duration, measure, [position * beat for position in positions])
        if template in {"breakbeat", "fills"}:
            ghosts = _random_grid_onsets(
                rng,
                duration,
                beat,
                4,
                density=float(rng.uniform(0.08, 0.22)),
                force_downbeats=False,
            )
            onsets = np.sort(np.concatenate([onsets, ghosts]))
        return _ensure_onsets(onsets, duration), 0.28, 4
    if source_type in {"closed_hat", "open_hat"}:
        subdivision = 8 if template == "dense_hats" else int(rng.choice([3, 4, 6, 8]))
        density = 0.82 if template == "dense_hats" else float(rng.uniform(0.35, 0.72))
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=density,
            force_downbeats=template in {"straight", "triplet"},
        )
        return onsets, 0.08 if source_type == "closed_hat" else 0.24, subdivision
    subdivision = 3 if template == "triplet" else int(rng.choice([2, 4, 6, 8]))
    density = float(rng.uniform(0.18, 0.58))
    if template in {"foley", "fills"}:
        density = float(rng.uniform(0.32, 0.76))
    onsets = _random_grid_onsets(
        rng,
        duration,
        beat,
        subdivision,
        density=density,
        force_downbeats=source_type in {"cymbal", "tom"},
    )
    if template == "fills" and len(onsets):
        fill_start = max(0.0, duration - measure)
        onsets = onsets[onsets >= fill_start] if rng.random() < 0.5 else onsets
    return _ensure_onsets(onsets, duration), _percussion_length(source_type, beat), subdivision


def _percussion_length(source_type: SourceType, beat: float) -> float:
    if source_type == "kick":
        return 0.34
    if source_type in {"snare", "clap"}:
        return 0.28
    if source_type == "closed_hat":
        return 0.08
    if source_type in {"open_hat", "cymbal"}:
        return min(0.72, beat * 0.75)
    if source_type == "tom":
        return 0.36
    return 0.18


def _asset_aware_percussion_length(
    source: SourceMetadata, fallback_length: float, clip_duration: float
) -> float:
    if source.renderer != "wav_one_shot":
        return fallback_length
    asset_duration = source.synth_parameters.get("asset_duration_seconds")
    if asset_duration is None:
        return fallback_length
    return min(clip_duration, max(fallback_length, float(asset_duration)))


def _surge_patch_events(
    rng: np.random.Generator,
    source: SourceMetadata,
    duration: float,
    beat: float,
) -> tuple[np.ndarray, float, int]:
    if source.source_type == "synth_bass":
        subdivision = int(rng.choice([1, 2, 3, 4]))
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=float(rng.uniform(0.18, 0.42)),
            force_downbeats=True,
        )
        held_length = beat * float(rng.choice([0.75, 1.0, 1.5, 2.0]))
    elif source.source_type in {"pad_chord", "organ", "string_stab", "brass_stab"}:
        subdivision = 1
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=float(rng.uniform(0.08, 0.22)),
            force_downbeats=True,
        )
        held_length = beat * float(rng.choice([2.0, 3.0, 4.0, 6.0]))
    else:
        subdivision = int(rng.choice([2, 3, 4, 6]))
        onsets = _random_grid_onsets(
            rng,
            duration,
            beat,
            subdivision,
            density=float(rng.uniform(0.22, 0.58)),
            force_downbeats=False,
        )
        held_length = beat / subdivision * float(rng.uniform(1.0, 3.0))
    return onsets, held_length, subdivision


def _surge_source_envelope(source: SourceMetadata) -> tuple[float, float]:
    return (
        float(source.synth_parameters["surge_amp_attack_seconds"]),
        float(source.synth_parameters["surge_amp_release_seconds"]),
    )


def _event_envelope_labels(
    source: SourceMetadata, recipe: SourceType, event_duration: float, rng: np.random.Generator
) -> tuple[float, float]:
    if source.renderer == "wav_one_shot":
        return 0.0, 0.0
    if source.renderer == "surge_xt_fxp":
        return _surge_source_envelope(source)
    attack = {
        "kick": 0.002,
        "snare_clap": 0.003,
        "hat_noise": 0.001,
        "fm_bass": float(rng.uniform(0.004, 0.018)),
        "fm_growl": float(rng.uniform(0.01, 0.04)),
        "wub_bass": float(rng.uniform(0.008, 0.026)),
        "pluck_stab": float(rng.uniform(0.002, 0.012)),
        "riser_impact": 0.08,
    }[recipe]
    release = {
        "kick": 0.08,
        "snare_clap": 0.12,
        "hat_noise": 0.03,
        "fm_bass": float(rng.uniform(0.035, 0.09)),
        "fm_growl": float(rng.uniform(0.08, 0.22)),
        "wub_bass": float(rng.uniform(0.06, 0.18)),
        "pluck_stab": float(rng.uniform(0.08, 0.2)),
        "riser_impact": 0.45,
    }[recipe]
    return attack, release


def _thin_same_track_overlaps(
    rng: np.random.Generator,
    onsets: np.ndarray,
    source_type: SourceType,
    length: float,
    duration: float,
    template: str,
) -> np.ndarray:
    if len(onsets) <= 1:
        return onsets
    overlap_factor = {
        "closed_hat": 0.28,
        "open_hat": 0.7,
        "cymbal": 0.8,
        "tom": 0.9,
        "kick": 0.55,
        "snare": 0.6,
        "clap": 0.5,
        "percussion": 0.68,
    }.get(source_type, 0.65)
    overlap_chance = 0.2 if template in {"fills", "dense_hats"} else 0.08
    if source_type in {"tom", "cymbal", "open_hat"}:
        overlap_chance *= 0.5
    minimum_gap = min(max(0.02, length * overlap_factor), max(0.02, duration * 0.75))
    kept: list[float] = []
    last_strict_onset: float | None = None
    for onset in sorted(float(value) for value in onsets if value < duration):
        if last_strict_onset is None or onset - last_strict_onset >= minimum_gap:
            kept.append(onset)
            last_strict_onset = onset
            continue
        if rng.random() < overlap_chance:
            kept.append(onset)
    if not kept:
        kept.append(float(onsets[0]))
    return np.array(kept, dtype=np.float32)


def _ensure_onsets(onsets: np.ndarray, duration: float) -> np.ndarray:
    if len(onsets):
        return onsets
    return np.array(
        [0.0 if duration <= 1.0 else min(duration - 0.1, duration * 0.5)], dtype=np.float32
    )


def _event_timbre_variation(
    rng: np.random.Generator, source_type: SourceType, position: float
) -> dict[str, float]:
    drift = float(np.clip(position, 0.0, 1.0) - 0.5)

    def random_offset(spread: float) -> float:
        return float(rng.uniform(-spread, spread))

    if source_type == "kick":
        return {
            "pitch_scale": round(1.0 + drift * 0.04 + random_offset(0.025), 5),
            "decay_scale": round(1.0 + drift * 0.12 + random_offset(0.08), 5),
        }
    if source_type == "snare_clap":
        return {
            "noise_tone_scale": round(1.0 + drift * 0.16 + random_offset(0.08), 5),
            "body_tone_scale": round(1.0 + drift * 0.08 + random_offset(0.05), 5),
        }
    if source_type == "hat_noise":
        return {
            "highpass_scale": round(1.0 + drift * 0.22 + random_offset(0.12), 5),
            "decay_scale": round(1.0 + drift * 0.16 + random_offset(0.12), 5),
        }
    if source_type == "fm_bass":
        return {
            "fm_index_scale": round(1.0 + drift * 0.18 + random_offset(0.12), 5),
            "noise_scale": round(1.0 + drift * 0.24 + random_offset(0.14), 5),
        }
    if source_type == "fm_growl":
        return {
            "fm_index_a_scale": round(1.0 + drift * 0.18 + random_offset(0.1), 5),
            "fm_index_b_scale": round(1.0 - drift * 0.14 + random_offset(0.1), 5),
            "formant_scale": round(1.0 + drift * 0.16 + random_offset(0.08), 5),
        }
    if source_type == "wub_bass":
        return {
            "fm_index_scale": round(1.0 + drift * 0.16 + random_offset(0.1), 5),
            "harmonic_scale": round(1.0 + drift * 0.2 + random_offset(0.12), 5),
        }
    if source_type == "pluck_stab":
        return {
            "filter_scale": round(1.0 + drift * 0.3 + random_offset(0.18), 5),
            "detune_offset_cents": round(drift * 2.5 + random_offset(1.6), 5),
            "transient_scale": round(1.0 + drift * 0.22 + random_offset(0.14), 5),
        }
    return {
        "filter_scale": round(1.0 + drift * 0.26 + random_offset(0.14), 5),
        "impact_scale": round(1.0 + drift * 0.2 + random_offset(0.12), 5),
    }


def _choose_bass_note(rng: np.random.Generator, bass_note_pool: list[int], prefer_low: bool) -> int:
    low_notes = [note for note in bass_note_pool if note <= 43]
    if prefer_low and low_notes and rng.random() < 0.85:
        return int(rng.choice(low_notes))
    return int(rng.choice(bass_note_pool))


def _notes_in_asset_range(source: SourceMetadata, note_pool: list[int]) -> list[int]:
    note_min = source.synth_parameters.get("asset_note_min")
    note_max = source.synth_parameters.get("asset_note_max")
    candidates = [
        note
        for note in note_pool
        if (note_min is None or note >= int(note_min))
        and (note_max is None or note <= int(note_max))
    ]
    if not candidates:
        raise ValueError(
            f"No generated notes fit asset range for {source.source_id} ({source.asset_id})"
        )
    return candidates


def _backbeat_positions(beats_per_measure: int) -> list[int]:
    if beats_per_measure == 2:
        return [1]
    if beats_per_measure == 3:
        return [1]
    if beats_per_measure == 4:
        return [1, 3]
    if beats_per_measure == 5:
        return [1, 3]
    if beats_per_measure == 7:
        return [2, 4, 6]
    return [max(1, beats_per_measure // 2)]


def _metered_onsets(
    duration: float, measure_seconds: float, offsets: list[float] | np.ndarray
) -> np.ndarray:
    if len(offsets) == 0:
        return np.array([], dtype=np.float32)
    measure_starts = np.arange(0.0, duration, measure_seconds, dtype=np.float32)
    onsets = [
        float(measure_start + offset)
        for measure_start in measure_starts
        for offset in offsets
        if measure_start + offset < duration
    ]
    return np.array(onsets, dtype=np.float32)


def _random_grid_onsets(
    rng: np.random.Generator,
    duration: float,
    beat_seconds: float,
    subdivisions: int,
    density: float,
    force_downbeats: bool,
) -> np.ndarray:
    step = beat_seconds / subdivisions
    grid = np.arange(0.0, duration, step, dtype=np.float32)
    if len(grid) == 0:
        return grid
    selected = rng.random(len(grid)) < density
    if force_downbeats:
        selected[np.arange(len(grid)) % subdivisions == 0] = True
    if not bool(np.any(selected)):
        selected[int(rng.integers(0, len(selected)))] = True
    return grid[selected]


def _accent_repeating_motif(
    rng: np.random.Generator,
    onsets: np.ndarray,
    beat_seconds: float,
    subdivisions: int,
) -> np.ndarray:
    step = beat_seconds / subdivisions
    motif_length = int(rng.choice([2, 3, 4]))
    if motif_length >= subdivisions:
        motif_length = max(2, subdivisions // 2)
    phase = int(rng.integers(0, subdivisions))
    accented = [
        onset
        for onset in onsets
        if int(round(float(onset) / step)) % subdivisions
        in {(phase + i) % subdivisions for i in range(motif_length)}
    ]
    if len(accented) < max(3, len(onsets) // 3):
        return onsets
    return np.array(accented, dtype=np.float32)


def _render_source(
    source: SourceMetadata,
    events: list[EventMetadata],
    sample_rate: int,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if source.renderer == "wav_one_shot":
        return _render_wav_one_shot(source, events, sample_rate, duration)
    if source.renderer == "sfz_instrument":
        return _render_sfz_instrument(source, events, sample_rate, duration)
    if source.renderer == "dawdreamer_plugin":
        return render_dawdreamer_source(source, events, sample_rate, duration)
    if source.renderer == "surge_xt_fxp":
        return render_surge_fxp_source(source, events, sample_rate, duration)
    total_samples = int(round(sample_rate * duration))
    mono = np.zeros(total_samples, dtype=np.float32)
    render_source = source.model_copy(update={"source_type": _procedural_recipe_for_source(source)})
    for event in events:
        start = int(round(event.onset_seconds * sample_rate))
        stop = min(total_samples, int(round(event.offset_seconds * sample_rate)))
        if stop <= start:
            continue
        length = stop - start
        rendered = _render_event(render_source, event, length, sample_rate, rng)
        mono[start:stop] += rendered[: stop - start]
    mono *= db_to_amp(source.gain_db)
    return mono.astype(np.float32)


def _render_wav_one_shot(
    source: SourceMetadata,
    events: list[EventMetadata],
    sample_rate: int,
    duration: float,
) -> np.ndarray:
    asset_path = Path(str(source.synth_parameters["asset_path"]))
    audio, asset_sample_rate = load_mono_wav(asset_path)
    audio = _resample_to_rate(audio, asset_sample_rate, sample_rate)
    total_samples = int(round(sample_rate * duration))
    mono = np.zeros(total_samples, dtype=np.float32)
    for event in events:
        start = int(round(event.onset_seconds * sample_rate))
        stop = min(total_samples, start + len(audio))
        if stop <= start:
            continue
        rendered = audio[: stop - start] * float(event.velocity)
        rendered = _edge_fade(rendered, sample_rate)
        mono[start:stop] += rendered
    mono *= db_to_amp(source.gain_db)
    return mono.astype(np.float32)


def _render_sfz_instrument(
    source: SourceMetadata,
    events: list[EventMetadata],
    sample_rate: int,
    duration: float,
) -> np.ndarray:
    sfz_path = Path(str(source.synth_parameters["asset_path"]))
    regions, _warnings = parse_sfz(sfz_path)
    total_samples = int(round(sample_rate * duration))
    mono = np.zeros(total_samples, dtype=np.float32)
    for event in events:
        midi_note = int(event.midi_note or source.synth_parameters.get("asset_root_key") or 60)
        region = choose_sfz_region(regions, midi_note, float(event.velocity))
        sample_path = sfz_path.parent / region.sample
        audio, asset_sample_rate = load_mono_wav(sample_path)
        start_sample = min(region.offset, len(audio))
        end_sample = min(region.end + 1 if region.end is not None else len(audio), len(audio))
        audio = audio[start_sample:end_sample]
        audio = _resample_to_rate(audio, asset_sample_rate, sample_rate)
        root_key = region.pitch_keycenter or int(
            source.synth_parameters.get("asset_root_key") or midi_note
        )
        semitones = midi_note - root_key + region.tune / 100.0
        if abs(semitones) > 1e-6:
            audio = _pitch_shift_by_resample(audio, semitones)
        event_len = max(1, int(round((event.offset_seconds - event.onset_seconds) * sample_rate)))
        if len(audio) < event_len:
            audio = np.pad(audio, (0, event_len - len(audio)))
        else:
            audio = audio[:event_len]
        attack = max(float(region.ampeg_attack), float(event.attack_seconds))
        release = max(float(region.ampeg_release), float(event.release_seconds))
        env = adsr_envelope(len(audio), sample_rate, attack, release, sustain=0.85)
        rendered = audio * env * float(event.velocity) * db_to_amp(region.volume)
        start = int(round(event.onset_seconds * sample_rate))
        stop = min(total_samples, start + len(rendered))
        if stop > start:
            mono[start:stop] += rendered[: stop - start]
    mono *= db_to_amp(source.gain_db)
    return mono.astype(np.float32)


def _resample_to_rate(audio: np.ndarray, actual_rate: int, target_rate: int) -> np.ndarray:
    if actual_rate == target_rate:
        return audio.astype(np.float32)
    target_len = max(1, int(round(len(audio) * target_rate / actual_rate)))
    return signal.resample(audio, target_len).astype(np.float32)


def _pitch_shift_by_resample(audio: np.ndarray, semitones: float) -> np.ndarray:
    ratio = 2.0 ** (semitones / 12.0)
    pitched_len = max(1, int(round(len(audio) / ratio)))
    shifted = signal.resample(audio, pitched_len).astype(np.float32)
    if len(shifted) == len(audio):
        return shifted
    return signal.resample(shifted, len(audio)).astype(np.float32)


def _edge_fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    out = audio.astype(np.float32).copy()
    fade_n = min(len(out) // 2, max(1, int(round(sample_rate * 0.003))))
    if fade_n > 0:
        fade_in = np.linspace(0.0, 1.0, fade_n, endpoint=True, dtype=np.float32)
        out[:fade_n] *= fade_in
        out[-fade_n:] *= fade_in[::-1]
    return out


def _procedural_recipe_for_source(source: SourceMetadata) -> str:
    return str(
        source.synth_parameters.get(
            "procedural_recipe",
            PROCEDURAL_RECIPE_BY_SOURCE_TYPE.get(source.source_type, source.source_type),
        )
    )


def _render_event(
    source: SourceMetadata,
    event: EventMetadata,
    length: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / sample_rate
    velocity = event.velocity
    params = source.synth_parameters
    variation = event.render_parameters.get("timbre_variation", {})
    if source.source_type == "kick":
        pitch_scale = _variation_value(variation, "pitch_scale", 1.0, 0.85, 1.15)
        decay_scale = _variation_value(variation, "decay_scale", 1.0, 0.75, 1.3)
        start_hz = float(params["start_hz"]) * pitch_scale
        end_hz = float(params["end_hz"]) * pitch_scale
        freq = end_hz + (start_hz - end_hz) * np.exp(-t / 0.035)
        phase = np.cumsum(freq, dtype=np.float32) * (TAU / sample_rate)
        body = np.sin(phase)
        click = rng.normal(0.0, 0.12, length).astype(np.float32) * one_pole_decay(
            length, sample_rate, 0.008
        )
        body_env = one_pole_decay(length, sample_rate, 0.22 * decay_scale)
        return ((body * body_env + click) * velocity).astype(np.float32)
    if source.source_type == "snare_clap":
        noise_tone = float(params["noise_tone_hz"]) * _variation_value(
            variation, "noise_tone_scale", 1.0, 0.75, 1.35
        )
        body_tone = float(params["body_hz"]) * _variation_value(
            variation, "body_tone_scale", 1.0, 0.85, 1.2
        )
        noise = rng.normal(0.0, 1.0, length).astype(np.float32)
        noise = butter_filter(noise, sample_rate, noise_tone, "highpass", order=2)
        body = np.sin(TAU * body_tone * t) * one_pole_decay(length, sample_rate, 0.08)
        env = one_pole_decay(length, sample_rate, 0.13)
        return ((noise * env * 0.42 + body * 0.35) * velocity).astype(np.float32)
    if source.source_type == "hat_noise":
        highpass_hz = float(params["highpass_hz"]) * _variation_value(
            variation, "highpass_scale", 1.0, 0.65, 1.45
        )
        release_scale = _variation_value(variation, "decay_scale", 1.0, 0.65, 1.45)
        noise = rng.normal(0.0, 1.0, length).astype(np.float32)
        noise = butter_filter(noise, sample_rate, highpass_hz, "highpass", order=2)
        env = adsr_envelope(
            length,
            sample_rate,
            event.attack_seconds,
            event.release_seconds * release_scale,
            sustain=0.25,
        )
        return (noise * env * 0.36 * velocity).astype(np.float32)
    if source.source_type == "fm_bass":
        carrier = midi_to_hz(event.midi_note or 36)
        ratio = float(params["ratio"])
        voice_model = str(params.get("voice_model", "fm_round"))
        index = (
            float(params["fm_index"])
            * _variation_value(variation, "fm_index_scale", 1.0, 0.65, 1.45)
            * one_pole_decay(length, sample_rate, float(params["index_decay_seconds"]))
        )
        phase = TAU * carrier * t + index * np.sin(TAU * carrier * ratio * t)
        env = adsr_envelope(
            length, sample_rate, event.attack_seconds, event.release_seconds, sustain=0.82
        )
        sub = np.sin(TAU * carrier * 0.5 * t) * float(params["sub_mix"])
        noise_amount = float(params["noise_mix"]) * _variation_value(
            variation, "noise_scale", 1.0, 0.5, 1.7
        )
        noise = rng.normal(0.0, noise_amount, length).astype(np.float32)
        if voice_model == "sub_clean":
            tone = np.sin(TAU * carrier * t) * 0.55 + sub * 1.25 + noise * 0.35
        elif voice_model == "edge_distorted":
            tone = np.sin(phase) * 0.75 + signal_saw(carrier, t) * 0.34 + sub * 0.55 + noise * 1.4
        elif voice_model == "noisy_reese":
            detuned = np.sin(TAU * carrier * 1.006 * t) - np.sin(TAU * carrier * 0.994 * t)
            tone = (
                np.sin(phase) * 0.45
                + detuned * 0.55
                + signal_saw(carrier * 0.5, t) * 0.28
                + sub * 0.7
                + noise * 1.8
            )
        else:
            tone = np.sin(phase) + sub + noise
        drive = float(params["wavefold_drive"]) * (
            0.72
            if voice_model == "sub_clean"
            else 1.35
            if voice_model in {"edge_distorted", "noisy_reese"}
            else 1.0
        )
        tone = np.tanh(tone * drive)
        tone = tone * env
        return (tone * velocity).astype(np.float32)
    if source.source_type == "fm_growl":
        carrier = midi_to_hz(event.midi_note or 36)
        voice_model = str(params.get("voice_model", "vowel_growl"))
        moving_index_a = float(params["fm_index_a"]) * _variation_value(
            variation, "fm_index_a_scale", 1.0, 0.65, 1.45
        )
        moving_index_b = float(params["fm_index_b"]) * _variation_value(
            variation, "fm_index_b_scale", 1.0, 0.65, 1.45
        )
        mod_a = np.sin(TAU * carrier * float(params["ratio_a"]) * t)
        mod_b = np.sin(TAU * carrier * float(params["ratio_b"]) * t + moving_index_a * mod_a)
        phase = TAU * carrier * t + moving_index_a * mod_a + moving_index_b * mod_b
        tone = np.sin(phase)
        formant_shift = float(params["formant_shift"]) * _variation_value(
            variation, "formant_scale", 1.0, 0.75, 1.3
        )
        formant = np.sin(TAU * carrier * formant_shift * 2.0 * t)
        sub = np.sin(TAU * carrier * 0.5 * t) * float(params["sub_mix"])
        noise = rng.normal(0.0, float(params["noise_mix"]), length).astype(np.float32)
        env = adsr_envelope(
            length, sample_rate, event.attack_seconds, event.release_seconds, sustain=0.74
        )
        if voice_model == "sub_growl":
            tone = tone * 0.42 + formant * 0.06 + sub * 1.55 + noise * 0.55
            drive = float(params["wavefold_drive"]) * 0.72
        elif voice_model == "edge_growl":
            tone = (
                tone * 0.92
                + formant * 0.2
                + signal_saw(carrier, t) * 0.32
                + sub * 0.7
                + noise * 1.35
            )
            drive = float(params["wavefold_drive"]) * 1.35
        else:
            tone = tone + formant * 0.16 + sub + noise
            drive = float(params["wavefold_drive"])
        tone = np.tanh(tone * drive) * env
        return (tone * velocity * 0.48).astype(np.float32)
    if source.source_type == "wub_bass":
        carrier = midi_to_hz(event.midi_note or 36)
        voice_model = str(params.get("voice_model", "fm_pulse"))
        ratio = float(params["ratio"])
        index = float(params["fm_index"]) * _variation_value(
            variation, "fm_index_scale", 1.0, 0.7, 1.4
        )
        phase = TAU * carrier * t + index * np.sin(TAU * carrier * ratio * t)
        harmonic = (
            signal_saw(carrier, t)
            * float(params["harmonic_mix"])
            * _variation_value(variation, "harmonic_scale", 1.0, 0.65, 1.5)
        )
        sub = np.sin(TAU * carrier * 0.5 * t) * float(params["sub_mix"])
        env = adsr_envelope(
            length, sample_rate, event.attack_seconds, event.release_seconds, sustain=0.86
        )
        if voice_model == "sub_pulse":
            tone = np.sin(TAU * carrier * t) * 0.58 + sub * 1.55 + harmonic * 0.35
            drive = float(params["wavefold_drive"]) * 0.78
        elif voice_model == "saw_pulse":
            tone = np.sin(phase) * 0.55 + harmonic * 1.45 + sub * 0.65
            drive = float(params["wavefold_drive"]) * 1.28
        else:
            tone = np.sin(phase) + harmonic + sub
            drive = float(params["wavefold_drive"])
        tone = np.tanh(tone * drive) * env
        return (tone * velocity * 0.52).astype(np.float32)
    if source.source_type == "pluck_stab":
        hz = midi_to_hz(event.midi_note or 60)
        voice_model = str(params.get("voice_model", "balanced"))
        detune_cents = float(params["detune_cents"]) + _variation_value(
            variation, "detune_offset_cents", 0.0, -6.0, 6.0
        )
        detune = 2.0 ** (detune_cents / 1200.0)
        if voice_model == "round":
            tone = (
                np.sin(TAU * hz * t) * 0.9
                + np.sin(TAU * hz * detune * t) * 0.28
                + float(params["octave_mix"]) * np.sin(TAU * hz * 0.5 * t) * 0.45
            )
            sustain = 0.62
            level = 1.45
            drive = 1.25
        elif voice_model == "bright_saw":
            tone = (
                np.sin(TAU * hz * t) * 0.35
                + np.sin(TAU * hz * detune * t) * 0.5
                + float(params["octave_mix"]) * np.sin(TAU * hz * 2.0 * t)
                + float(params["saw_mix"]) * signal_saw(hz, t) * 1.25
            )
            sustain = 0.42
            level = 2.05
            drive = 2.1
        elif voice_model == "metallic":
            tone = (
                np.sin(TAU * hz * 1.5 * t) * 0.62
                + np.sin(TAU * hz * 2.01 * t) * 0.38
                + np.sin(TAU * hz * detune * 2.7 * t) * 0.34
                + signal_saw(hz * 1.5, t) * float(params["saw_mix"]) * 0.55
            )
            sustain = 0.32
            level = 1.92
            drive = 2.35
        else:
            tone = (
                np.sin(TAU * hz * t)
                - 0.5 * np.sin(TAU * hz * 2.0 * t)
                + 0.72 * np.sin(TAU * hz * detune * t)
                + float(params["saw_mix"]) * signal_saw(hz, t) * 0.55
            )
            sustain = 0.5
            level = 1.75
            drive = 1.65
        transient = rng.normal(0.0, 1.0, length).astype(np.float32) * one_pole_decay(
            length, sample_rate, 0.012
        )
        env = adsr_envelope(
            length, sample_rate, event.attack_seconds, event.release_seconds, sustain=sustain
        )
        filter_hz = float(params["filter_hz"]) * _variation_value(
            variation, "filter_scale", 1.0, 0.6, 1.6
        )
        tone = butter_filter(tone.astype(np.float32), sample_rate, filter_hz, "lowpass", order=2)
        transient_scale = _variation_value(variation, "transient_scale", 1.0, 0.55, 1.65)
        click_amount = float(params["transient_click"]) * transient_scale
        if voice_model == "round":
            click_amount *= 0.45
        elif voice_model == "bright_saw":
            click_amount *= 1.45
        elif voice_model == "metallic":
            click_amount *= 1.2
        tone = saturate(tone + transient * click_amount, drive=drive)
        return (tone * env * level * velocity).astype(np.float32)
    filter_scale = _variation_value(variation, "filter_scale", 1.0, 0.65, 1.45)
    impact_scale = _variation_value(variation, "impact_scale", 1.0, 0.65, 1.45)
    filter_start_hz = float(params["filter_start_hz"]) * filter_scale
    filter_end_hz = float(params["filter_end_hz"]) * filter_scale
    noise = rng.normal(0.0, 1.0, length).astype(np.float32)
    bright_noise = butter_filter(noise, sample_rate, filter_end_hz, "highpass", order=1)
    dark_noise = butter_filter(noise, sample_rate, filter_start_hz, "highpass", order=1)
    sweep_position = np.linspace(0.0, 1.0, length, dtype=np.float32)
    sweep_position = np.power(sweep_position, float(params["noise_color"]))
    riser_noise = dark_noise * (1.0 - sweep_position) + bright_noise * sweep_position
    impact = rng.normal(0.0, 1.0, length).astype(np.float32) * one_pole_decay(
        length, sample_rate, 0.18
    )
    env = adsr_envelope(
        length, sample_rate, event.attack_seconds, event.release_seconds, sustain=0.9
    )
    tone = riser_noise * 0.82 + impact * float(params["impact_weight"]) * impact_scale
    return (tone * env * velocity * 0.52).astype(np.float32)


def _variation_value(
    variation: dict[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    return float(np.clip(float(variation.get(key, default)), minimum, maximum))


def signal_saw(hz: float, t: np.ndarray) -> np.ndarray:
    return (2.0 * ((hz * t) % 1.0) - 1.0).astype(np.float32)


def _apply_source_effects(
    audio: np.ndarray, source: SourceMetadata, sample_rate: int
) -> np.ndarray:
    out = audio.astype(np.float32)
    drive = float(source.effect_parameters.get("drive", 1.0))
    if drive > 1.01:
        out = saturate(out, drive)
    recipe = _procedural_recipe_for_source(source)
    if recipe == "fm_bass":
        out = butter_filter(out, sample_rate, 6200.0, "lowpass", order=2)
    if recipe == "wub_bass":
        out = butter_filter(out, sample_rate, 4200.0, "lowpass", order=2)
    if recipe == "fm_growl":
        out = butter_filter(out, sample_rate, 4800.0, "lowpass", order=2)
    if recipe == "riser_impact":
        out = butter_filter(out, sample_rate, 5200.0, "highpass", order=1)
    return out.astype(np.float32)


def _source_by_id(sources: list[SourceMetadata], source_id: str) -> SourceMetadata:
    for source in sources:
        if source.source_id == source_id:
            return source
    raise KeyError(source_id)
