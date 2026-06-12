# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from comet_audio.dsp import (
    TAU,
    adsr_envelope,
    butter_filter,
    db_to_amp,
    delay,
    midi_to_hz,
    normalize_peak,
    one_pole_decay,
    saturate,
    sidechain_duck,
    simple_reverb,
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
    "snare_clap",
    "hat_noise",
    "fm_bass",
    "fm_growl",
    "wub_bass",
    "pluck_stab",
    "riser_impact",
]
OPTIONAL_SOURCE_POOL: list[SourceType] = [
    "hat_noise",
    "fm_bass",
    "fm_growl",
    "wub_bass",
    "pluck_stab",
    "pluck_stab",
    "riser_impact",
]


@dataclass(frozen=True)
class GeneratorConfig:
    sample_rate: int = 44_100
    duration_seconds: float = 8.0
    bpm_min: int = 70
    bpm_max: int = 150
    time_signatures: tuple[str, ...] = DEFAULT_TIME_SIGNATURES
    source_count_min: int = 5
    source_count_max: int = 10


def generate_batch(
    out_dir: Path,
    count: int,
    seed: int,
    config: GeneratorConfig | None = None,
    write_preview: bool = True,
) -> list[ClipMetadata]:
    config = config or GeneratorConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    clips: list[ClipMetadata] = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(count):
            clip_seed = int(seed + index)
            clip_id = f"clip_{index:04d}"
            clip_dir = out_dir / clip_id
            metadata = generate_clip(clip_dir, clip_seed, config)
            clips.append(metadata)
            entry = BatchManifestEntry(
                clip_id=clip_id,
                seed=metadata.seed,
                bpm=metadata.bpm,
                time_signature=metadata.time_signature,
                key=metadata.key,
                mix_path=f"{clip_id}/mix.wav",
                metadata_path=f"{clip_id}/metadata.json",
                source_count=len(metadata.sources),
                event_count=len(metadata.events),
            )
            manifest.write(entry.model_dump_json() + "\n")

    if write_preview:
        write_preview_html(out_dir, clips)
    return clips


def generate_clip(
    clip_dir: Path,
    seed: int,
    config: GeneratorConfig | None = None,
) -> ClipMetadata:
    config = config or GeneratorConfig()
    rng = np.random.default_rng(seed)
    clip_dir.mkdir(parents=True, exist_ok=True)
    stem_dir = clip_dir / "stems"
    stem_dir.mkdir(exist_ok=True)

    if config.bpm_min > config.bpm_max:
        raise ValueError("bpm_min must be less than or equal to bpm_max")
    bpm = float(rng.integers(config.bpm_min, config.bpm_max + 1))
    beats_per_measure, beat_unit = _choose_time_signature(rng, config.time_signatures)
    key_index = int(rng.integers(0, len(KEYS)))
    key = KEYS[key_index]
    if config.source_count_min > config.source_count_max:
        raise ValueError("source_count_min must be less than or equal to source_count_max")
    source_count = int(rng.integers(config.source_count_min, config.source_count_max + 1))
    chosen_types = _choose_sources(rng, source_count)
    bass_note_pool = _minor_note_pool(key_index, root_octave_midi=24, octave_count=2)
    lead_note_pool = _minor_note_pool(key_index, root_octave_midi=48, octave_count=2)

    sources = _make_sources(rng, chosen_types)
    events = _make_events(
        rng,
        sources,
        bpm,
        beats_per_measure,
        beat_unit,
        bass_note_pool,
        lead_note_pool,
        config.duration_seconds,
    )

    stems: dict[str, np.ndarray] = {}
    kick_onsets = [
        event.onset_seconds
        for event in events
        if _source_by_id(sources, event.source_id).source_type == "kick"
    ]

    for source in sources:
        source_events = [event for event in events if event.source_id == source.source_id]
        dry = _render_source(
            source, source_events, config.sample_rate, config.duration_seconds, rng
        )
        wet = _apply_source_effects(dry, source, config.sample_rate)
        if source.source_type != "kick":
            wet = sidechain_duck(
                wet,
                kick_onsets,
                config.sample_rate,
                amount=float(source.effect_parameters.get("duck_amount", 0.0)),
                release_seconds=float(source.effect_parameters.get("duck_release_seconds", 0.22)),
            )
        wet = normalize_peak(wet, peak=0.95)
        stems[source.source_id] = wet
        sf.write(stem_dir / Path(source.stem_path).name, wet, config.sample_rate, subtype="PCM_16")

    mix = np.zeros(int(round(config.sample_rate * config.duration_seconds)), dtype=np.float32)
    for stem in stems.values():
        mix += stem
    mix = soft_limiter(mix, ceiling=0.98)
    sf.write(clip_dir / "mix.wav", mix, config.sample_rate, subtype="PCM_16")

    metadata = ClipMetadata(
        seed=seed,
        sample_rate=config.sample_rate,
        duration_seconds=config.duration_seconds,
        bpm=bpm,
        time_signature=f"{beats_per_measure}/{beat_unit}",
        beats_per_measure=beats_per_measure,
        beat_unit=beat_unit,
        key=key,
        paths={"mix": "mix.wav", "metadata": "metadata.json", "stems": "stems"},
        sources=sources,
        events=sorted(events, key=lambda event: (event.onset_seconds, event.event_id)),
    )
    (clip_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def write_preview_html(out_dir: Path, clips: list[ClipMetadata]) -> None:
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
            "--text:#f1f3f4;--muted:#a9b0b7;--attack:#f2bf5e;--sustain:#4dbf87;"
            "--release:#5fa8e8;--onset:#f26d6d}",
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
            ".sustain{background:var(--sustain)}.release{background:var(--release)}.onset{background:var(--onset)}",
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
            "background:var(--attack)}.seg.sustain{background:var(--sustain)}.seg.release{background:var(--release)}"
            ".onset-line{position:absolute;top:8px;bottom:8px;width:2px;background:var(--onset);z-index:2}",
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
            '<section class="legend"><div class="key"><span class="swatch onset"></span>Onset</div>'
            '<div class="key"><span class="swatch attack"></span>Attack</div>'
            '<div class="key"><span class="swatch sustain"></span>Sustain</div>'
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
            "let currentIndex=0;let activeClip=null;let playhead=null;",
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
            "const onset=document.createElement('div');onset.className='onset-line';onset.style.left=`${pct(start,duration)}%`;"
            "track.appendChild(onset);const attack=document.createElement('div');attack.className='seg attack';"
            "attack.style.width=`${((attackEnd-start)/total)*100}%`;const sustain=document.createElement('div');"
            "sustain.className='seg sustain';sustain.style.width=`${((releaseStart-attackEnd)/total)*100}%`;"
            "const release=document.createElement('div');release.className='seg release';release.style.width="
            "`${((end-releaseStart)/total)*100}%`;block.append(attack,sustain,release);track.appendChild(block)}",
            "function renderLane(source,events,duration){const lane=document.createElement('div');lane.className='lane';"
            "const label=document.createElement('div');label.className='lane-label';label.innerHTML="
            "`<b>${sourceLabel(source)}</b><span>${events.length} events - gain ${source.gain_db} dB - pan ${source.pan}</span>`;"
            "const track=document.createElement('div');track.className='lane-track';const color=sourceColors[source.source_type]||'#aaa';"
            "events.forEach(event=>renderEvent(track,event,duration,color));lane.append(label,track);timeline.appendChild(lane)}",
            "function renderTimeline(clip){const m=clip.metadata;timeline.innerHTML='';renderRuler(m.duration_seconds);"
            "m.sources.forEach(source=>{const events=m.events.filter(event=>event.source_id===source.source_id);"
            "renderLane(source,events,m.duration_seconds)});playhead=document.createElement('div');"
            "playhead.className='playhead';playhead.style.left='0%';timeline.appendChild(playhead)}",
            "function loadClip(index){currentIndex=(index+clips.length)%clips.length;activeClip=clips[currentIndex];"
            "clipSelect.value=String(currentIndex);renderStats(activeClip);renderAudioOptions(activeClip);"
            "renderTimeline(activeClip);updateTime()}",
            "function updateTime(){const duration=activeClip?activeClip.metadata.duration_seconds:0;"
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
    (out_dir / "preview.html").write_text(html, encoding="utf-8")
    (out_dir / "visualizer.html").write_text(html, encoding="utf-8")


def _choose_sources(rng: np.random.Generator, source_count: int) -> list[SourceType]:
    required: list[SourceType] = ["kick", "snare_clap", "hat_noise", "fm_bass"]
    chosen = required[: min(source_count, len(required))]
    while len(chosen) < source_count:
        candidate = OPTIONAL_SOURCE_POOL[int(rng.integers(0, len(OPTIONAL_SOURCE_POOL)))]
        if candidate in {"fm_growl", "wub_bass"} and any(
            source in {"fm_growl", "wub_bass"} for source in chosen
        ):
            continue
        chosen.append(candidate)
    return chosen


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


def _make_sources(rng: np.random.Generator, chosen_types: list[SourceType]) -> list[SourceMetadata]:
    sources = []
    type_counts = {
        source_type: sum(1 for chosen_type in chosen_types if chosen_type == source_type)
        for source_type in set(chosen_types)
    }
    type_seen: dict[SourceType, int] = {}
    for index, source_type in enumerate(chosen_types):
        instance_index = type_seen.get(source_type, 0)
        type_seen[source_type] = instance_index + 1
        gain = {
            "kick": -3.5,
            "snare_clap": -8.0,
            "hat_noise": -13.0,
            "fm_bass": -9.0,
            "fm_growl": -17.0,
            "wub_bass": -16.0,
            "pluck_stab": -0.5,
            "riser_impact": -14.0,
        }[source_type] + float(rng.normal(0.0, 1.2))
        pan = 0.0
        source_id = f"source_{index:03d}"
        sources.append(
            SourceMetadata(
                source_id=source_id,
                source_type=source_type,
                synth_parameters=_synth_parameters(
                    rng,
                    source_type,
                    instance_index=instance_index,
                    instance_count=type_counts[source_type],
                ),
                effect_parameters=_effect_parameters(
                    rng,
                    source_type,
                    instance_index=instance_index,
                    instance_count=type_counts[source_type],
                ),
                gain_db=round(gain, 3),
                pan=round(pan, 3),
                stem_path=f"stems/{source_id}.wav",
            )
        )
    return sources


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
    if source_type in {"snare_clap", "pluck_stab", "riser_impact"}:
        params["reverb_mix"] = float(rng.uniform(0.08, 0.24))
        params["room_size"] = float(rng.uniform(0.45, 0.9))
    if source_type in {"pluck_stab", "fm_bass", "fm_growl", "wub_bass"}:
        params["delay_seconds"] = float(rng.choice([0.125, 0.1875, 0.25]))
        params["delay_feedback"] = float(rng.uniform(0.12, 0.28))
        params["delay_mix"] = float(rng.uniform(0.04, 0.14))
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
) -> list[EventMetadata]:
    events: list[EventMetadata] = []
    beat = 60.0 / bpm
    measure = beats_per_measure * beat
    backbeat_positions = _backbeat_positions(beats_per_measure)
    event_index = 0
    for source in sources:
        rhythm_subdivision = int(rng.choice(RHYTHM_SUBDIVISIONS))
        if source.source_type == "kick":
            onsets = _metered_onsets(duration, measure, [0.0])
            extra = _metered_onsets(
                duration, measure, [position * beat for position in range(1, beats_per_measure)]
            )
            extra = extra[rng.random(len(extra)) > 0.62] if len(extra) else extra
            onsets = np.sort(np.concatenate([onsets, extra]))
            length = 0.34
        elif source.source_type == "snare_clap":
            onsets = _metered_onsets(
                duration,
                measure,
                [position * beat for position in backbeat_positions],
            )
            length = 0.28
        elif source.source_type == "hat_noise":
            rhythm_subdivision = int(rng.choice([2, 4, 6, 8]))
            onsets = _random_grid_onsets(
                rng,
                duration,
                beat,
                rhythm_subdivision,
                density=float(rng.uniform(0.42, 0.86)),
                force_downbeats=True,
            )
            length = 0.08
        elif source.source_type == "fm_bass":
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
        elif source.source_type == "fm_growl":
            offsets = [0.0]
            if beats_per_measure > 2 and rng.random() < 0.35:
                offsets.append(measure * 0.5)
            onsets = _metered_onsets(duration, measure, offsets)
            onsets = onsets[rng.random(len(onsets)) > 0.45] if len(onsets) else onsets
            if len(onsets) == 0:
                onsets = np.array([0.0], dtype=np.float32)
            length = beat * float(rng.choice([0.75, 1.0, 1.5]))
        elif source.source_type == "wub_bass":
            offsets = [0.0]
            if beats_per_measure > 2 and rng.random() < 0.35:
                offsets.append(measure * 0.5)
            onsets = _metered_onsets(duration, measure, offsets)
            onsets = onsets[rng.random(len(onsets)) > 0.35] if len(onsets) else onsets
            if len(onsets) == 0:
                onsets = np.array([0.0], dtype=np.float32)
            length = beat * float(rng.choice([0.75, 1.0, 1.5]))
        elif source.source_type == "pluck_stab":
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

        for onset in onsets:
            if onset >= duration:
                continue
            offset = min(duration, float(onset + length))
            attack = {
                "kick": 0.002,
                "snare_clap": 0.003,
                "hat_noise": 0.001,
                "fm_bass": float(rng.uniform(0.004, 0.018)),
                "fm_growl": float(rng.uniform(0.01, 0.04)),
                "wub_bass": float(rng.uniform(0.008, 0.026)),
                "pluck_stab": float(rng.uniform(0.002, 0.012)),
                "riser_impact": 0.08,
            }[source.source_type]
            release = {
                "kick": 0.08,
                "snare_clap": 0.12,
                "hat_noise": 0.03,
                "fm_bass": float(rng.uniform(0.035, 0.09)),
                "fm_growl": float(rng.uniform(0.08, 0.22)),
                "wub_bass": float(rng.uniform(0.06, 0.18)),
                "pluck_stab": float(rng.uniform(0.08, 0.2)),
                "riser_impact": 0.45,
            }[source.source_type]
            note = None
            if source.source_type in {"fm_bass", "fm_growl", "wub_bass"}:
                note = _choose_bass_note(
                    rng, bass_note_pool, prefer_low=source.source_type != "fm_bass"
                )
            elif source.source_type == "pluck_stab":
                note = int(rng.choice(lead_note_pool))
            velocity = float(rng.uniform(0.62, 1.0))
            timbre_variation = _event_timbre_variation(
                rng, source.source_type, position=float(onset) / duration
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
                    },
                )
            )
            event_index += 1
    return events


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
    total_samples = int(round(sample_rate * duration))
    mono = np.zeros(total_samples, dtype=np.float32)
    for event in events:
        start = int(round(event.onset_seconds * sample_rate))
        stop = min(total_samples, int(round(event.offset_seconds * sample_rate)))
        if stop <= start:
            continue
        length = stop - start
        rendered = _render_event(source, event, length, sample_rate, rng)
        mono[start:stop] += rendered[: stop - start]
    mono *= db_to_amp(source.gain_db)
    return mono.astype(np.float32)


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
    if source.source_type == "fm_bass":
        out = butter_filter(out, sample_rate, 6200.0, "lowpass", order=2)
    if source.source_type == "wub_bass":
        out = butter_filter(out, sample_rate, 4200.0, "lowpass", order=2)
    if source.source_type == "fm_growl":
        out = butter_filter(out, sample_rate, 4800.0, "lowpass", order=2)
    if source.source_type == "riser_impact":
        out = butter_filter(out, sample_rate, 5200.0, "highpass", order=1)
    if "delay_seconds" in source.effect_parameters:
        out = delay(
            out,
            sample_rate,
            float(source.effect_parameters["delay_seconds"]),
            float(source.effect_parameters["delay_feedback"]),
            float(source.effect_parameters["delay_mix"]),
        )
    if "reverb_mix" in source.effect_parameters:
        out = simple_reverb(
            out,
            sample_rate,
            float(source.effect_parameters["room_size"]),
            float(source.effect_parameters["reverb_mix"]),
        )
    return out.astype(np.float32)


def _source_by_id(sources: list[SourceMetadata], source_id: str) -> SourceMetadata:
    for source in sources:
        if source.source_id == source_id:
            return source
    raise KeyError(source_id)
