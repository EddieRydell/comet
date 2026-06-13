# ruff: noqa: E501

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from comet_audio.training import (
    HOP_LENGTH,
    SAMPLE_RATE,
    decode_onsets,
    load_trained_model,
    load_trained_source_types,
)


def predict_song(
    audio_path: Path,
    run_dir: Path,
    out_dir: Path,
    threshold: float = 0.35,
    nms_seconds: float = 0.025,
    source_threshold: float = 0.35,
    max_waveform_points: int = 2000,
) -> tuple[Path, Path]:
    audio_path = audio_path.resolve()
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform, original_sample_rate = _load_audio_for_model(audio_path)
    duration = waveform.numel() / SAMPLE_RATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_types = load_trained_source_types(run_dir)
    model = load_trained_model(run_dir, device)
    with torch.no_grad():
        predictions = model(waveform.to(device).unsqueeze(0))

    onset_prob = torch.sigmoid(predictions["onset"][0]).cpu()
    offset_pred = predictions["onset_offset"][0].cpu()
    source_prob = torch.sigmoid(predictions["source_onset"][0]).cpu()
    decoded_onsets = decode_onsets(
        onset_prob,
        offset_pred,
        threshold=threshold,
        nms_seconds=nms_seconds,
    )
    marks = _build_marks(
        decoded_onsets,
        onset_prob,
        source_prob,
        source_types,
        source_threshold,
        duration,
    )
    viewer_audio = _write_viewer_audio(waveform, audio_path, out_dir)
    payload = {
        "audio": {
            "input_path": str(audio_path),
            "viewer_audio_path": viewer_audio.name,
            "original_sample_rate": original_sample_rate,
            "model_sample_rate": SAMPLE_RATE,
            "duration_seconds": duration,
            "samples": int(waveform.numel()),
        },
        "model": {
            "run_dir": str(run_dir),
            "threshold": threshold,
            "nms_seconds": nms_seconds,
            "source_threshold": source_threshold,
            "hop_length": HOP_LENGTH,
            "frame_seconds": HOP_LENGTH / SAMPLE_RATE,
            "source_types": source_types,
        },
        "waveform": _waveform_peaks(waveform.numpy(), max_waveform_points),
        "marks": marks,
    }
    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path = out_dir / "visualizer.html"
    html_path.write_text(_song_visualizer_html(payload), encoding="utf-8")
    return json_path, html_path


def _load_audio_for_model(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    mono = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise RuntimeError(f"Audio contains non-finite samples: {path}")
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(int(sample_rate), SAMPLE_RATE)
        mono = signal.resample_poly(mono, SAMPLE_RATE // gcd, int(sample_rate) // gcd).astype(
            np.float32
        )
    return torch.from_numpy(mono), int(sample_rate)


def _build_marks(
    decoded_onsets: list[float],
    onset_prob: torch.Tensor,
    source_prob: torch.Tensor,
    source_types: tuple[str, ...],
    source_threshold: float,
    duration: float,
) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for index, onset_seconds in enumerate(decoded_onsets):
        frame = int(round(onset_seconds * SAMPLE_RATE / HOP_LENGTH))
        frame = max(0, min(frame, onset_prob.numel() - 1))
        source_scores = [
            {
                "source_type": source_type,
                "probability": float(source_prob[source_index, frame]),
            }
            for source_index, source_type in enumerate(source_types)
        ]
        source_scores.sort(key=lambda row: row["probability"], reverse=True)
        active_sources = [
            row for row in source_scores if row["probability"] >= source_threshold
        ] or source_scores[:1]
        marks.append(
            {
                "index": index,
                "time_seconds": max(0.0, min(float(onset_seconds), duration)),
                "onset_probability": float(onset_prob[frame]),
                "primary_source": active_sources[0]["source_type"],
                "primary_source_probability": active_sources[0]["probability"],
                "sources": active_sources[:4],
            }
        )
    return marks


def _write_viewer_audio(waveform: torch.Tensor, audio_path: Path, out_dir: Path) -> Path:
    target = out_dir / f"{audio_path.stem}.viewer.wav"
    sf.write(target, waveform.numpy(), SAMPLE_RATE, subtype="PCM_16")
    return target


def _waveform_peaks(waveform: np.ndarray, max_points: int) -> list[list[float]]:
    if waveform.size == 0:
        return []
    point_count = max(1, min(max_points, waveform.size))
    bucket_size = int(math.ceil(waveform.size / point_count))
    padded_length = bucket_size * point_count
    padded = np.pad(waveform, (0, padded_length - waveform.size))
    buckets = padded.reshape(point_count, bucket_size)
    return [[float(np.min(bucket)), float(np.max(bucket))] for bucket in buckets]


def _song_visualizer_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, sort_keys=True)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comet Song Predictions</title>
<style>
:root{color-scheme:dark;--bg:#111315;--panel:#1a1f24;--line:#343c45;--text:#f2f4f6;--muted:#aab2ba;--accent:#f06c64;--wave:#78add0;--play:#fff;--table:#15191d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:20;background:#161a1e;border-bottom:1px solid var(--line);padding:14px 18px;display:grid;gap:12px}h1{margin:0;font-size:18px;font-weight:650}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}audio{width:min(760px,100%)}label{font-size:12px;color:var(--muted);display:flex;gap:7px;align-items:center}input[type=range]{width:190px}.time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
main{padding:16px 18px;display:grid;gap:14px}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:8px 10px;min-width:112px}.stat b{display:block;font-size:13px}.stat span{display:block;color:var(--muted);font-size:12px;margin-top:2px}
.viewport{border:1px solid var(--line);background:#0e1114;border-radius:8px;overflow:auto;overscroll-behavior:contain}.timeline{position:relative;min-width:100%;background:#101316}.ruler{position:sticky;top:0;height:30px;background:#171b20;border-bottom:1px solid var(--line);z-index:5}.tick{position:absolute;top:0;bottom:0;border-left:1px solid #3d4650;color:var(--muted);font-size:11px;line-height:28px;padding-left:5px;pointer-events:none}
.wave-wrap{position:relative;height:300px;border-bottom:1px solid var(--line);cursor:crosshair}canvas{position:sticky;left:0;top:0}.playhead{position:absolute;left:0;top:0;width:2px;background:var(--play);box-shadow:0 0 0 1px #000;z-index:8;pointer-events:none;will-change:transform}.wave-playhead{height:300px}.lane-playhead{height:100%}
.tooltip{display:none;position:fixed;z-index:30;background:#20262c;border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-size:12px;color:var(--text);pointer-events:none;box-shadow:0 8px 28px rgba(0,0,0,.35);white-space:nowrap}
.lanes{position:relative;background:#111519}.lane-label{position:absolute;left:0;width:150px;height:26px;background:#171b20;border-right:1px solid var(--line);border-bottom:1px solid #252c33;padding:6px 8px;font-size:12px;color:var(--muted);z-index:3;pointer-events:none}
.table-wrap{border:1px solid var(--line);border-radius:8px;overflow:auto;background:var(--table);max-height:360px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #283039;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#1a1f24;color:var(--muted);z-index:2}tr:hover{background:#20262c}.empty{padding:18px;color:var(--muted)}
@media(max-width:760px){main{padding:12px}header{padding:12px}.wave-wrap{height:240px}}
</style>
</head>
<body>
<header>
<h1>Comet Song Predictions</h1>
<div class="bar"><audio id="audio" controls preload="metadata"></audio><span class="time" id="time">0.000 / 0.000</span><label>Zoom <input id="zoom" type="range" min="4" max="420" value="24"></label><label><input id="follow" type="checkbox" checked> Follow</label></div>
</header>
<main>
<section class="stats" id="stats"></section>
<section class="viewport" id="viewport"><div class="timeline" id="timeline"><div class="ruler" id="ruler"></div><div class="wave-wrap" id="waveWrap"><canvas id="wave"></canvas><div class="playhead wave-playhead" id="playhead"></div></div><div class="lanes" id="lanes"><canvas id="lanesCanvas"></canvas><div class="playhead lane-playhead" id="lanePlayhead"></div></div></div></section>
<section class="table-wrap"><table><thead><tr><th>#</th><th>Time</th><th>Onset</th><th>Primary Source</th><th>Source Prob</th><th>Other Sources</th></tr></thead><tbody id="marks"></tbody></table></section>
</main>
<div class="tooltip" id="tooltip"></div>
<script id="comet-data" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('comet-data').textContent);
const audio=document.getElementById('audio'),time=document.getElementById('time'),zoom=document.getElementById('zoom');
const follow=document.getElementById('follow'),viewport=document.getElementById('viewport'),timeline=document.getElementById('timeline');
const ruler=document.getElementById('ruler'),waveWrap=document.getElementById('waveWrap'),wave=document.getElementById('wave');
const lanes=document.getElementById('lanes'),lanesCanvas=document.getElementById('lanesCanvas'),playhead=document.getElementById('playhead'),lanePlayhead=document.getElementById('lanePlayhead');
const stats=document.getElementById('stats'),marksBody=document.getElementById('marks'),tooltip=document.getElementById('tooltip');
const duration=data.audio.duration_seconds;const dpr=Math.max(1,window.devicePixelRatio||1);audio.src=data.audio.viewer_audio_path;
let pxPerSec=Number(zoom.value);let layoutWidth=0;let renderQueued=false;let rafId=0;
function fmt(v,d=3){return Number(v).toFixed(d)}function xFor(t){return t*pxPerSec}function tFor(x){return Math.max(0,Math.min(duration,x/pxPerSec))}
function renderStats(){const rows=[['Duration',`${fmt(duration)}s`],['Marks',data.marks.length],['Threshold',data.model.threshold],['NMS',`${fmt(data.model.nms_seconds*1000,1)}ms`],['Frame',`${fmt(data.model.frame_seconds*1000,2)}ms`],['Sample Rate',data.audio.model_sample_rate]];stats.innerHTML='';rows.forEach(([label,value])=>{const el=document.createElement('div');el.className='stat';el.innerHTML=`<b>${value}</b><span>${label}</span>`;stats.appendChild(el)})}
function setCanvasSize(canvas,width,height){canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;canvas.width=Math.max(1,Math.round(width*dpr));canvas.height=Math.max(1,Math.round(height*dpr));const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return ctx}
function queueRender(){if(renderQueued)return;renderQueued=true;requestAnimationFrame(()=>{renderQueued=false;renderAll()})}
function renderAll(){pxPerSec=Number(zoom.value);layoutWidth=Math.max(viewport.clientWidth,Math.ceil(duration*pxPerSec));const visibleWidth=Math.max(1,viewport.clientWidth);const waveHeight=waveWrap.clientHeight;const laneHeight=data.model.source_types.length*26;timeline.style.width=`${layoutWidth}px`;waveWrap.style.width=`${layoutWidth}px`;lanes.style.width=`${layoutWidth}px`;lanes.style.height=`${laneHeight}px`;renderRuler();drawWave(setCanvasSize(wave,visibleWidth,waveHeight),visibleWidth,waveHeight,viewport.scrollLeft);drawLanes(setCanvasSize(lanesCanvas,visibleWidth,laneHeight),visibleWidth,laneHeight,viewport.scrollLeft);updatePlayhead()}
function renderRuler(){ruler.innerHTML='';const candidates=[.1,.25,.5,1,2,5,10,15,30,60];let step=candidates[candidates.length-1];for(const c of candidates){if(c*pxPerSec>=72){step=c;break}}for(let t=0;t<=duration+.001;t+=step){const tick=document.createElement('div');tick.className='tick';tick.style.left=`${xFor(t)}px`;tick.textContent=step<1?`${fmt(t,2)}s`:`${Math.round(t)}s`;ruler.appendChild(tick)}}
function drawWave(ctx,width,height,scrollX){ctx.clearRect(0,0,width,height);ctx.fillStyle='#101316';ctx.fillRect(0,0,width,height);ctx.strokeStyle='rgba(120,173,208,.95)';ctx.lineWidth=1;ctx.beginPath();const peaks=data.waveform;const mid=height/2;const usable=height*.9;if(peaks.length<2)return;const step=width>900?2:1;for(let x=0;x<width;x+=step){const time=tFor(scrollX+x);const idx=Math.max(0,Math.min(peaks.length-1,Math.floor((time/duration)*peaks.length)));const p=peaks[idx];ctx.moveTo(x,mid+p[0]*usable*.5);ctx.lineTo(x,mid+p[1]*usable*.5)}ctx.stroke();ctx.strokeStyle='rgba(255,255,255,.12)';ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(width,mid);ctx.stroke()}
function drawLanes(ctx,width,height,scrollX){ctx.clearRect(0,0,width,height);ctx.fillStyle='#111519';ctx.fillRect(0,0,width,height);ctx.font='12px Inter, system-ui, sans-serif';const start=tFor(scrollX)-.05;const end=tFor(scrollX+width)+.05;data.model.source_types.forEach((source,i)=>{const y=i*26;ctx.fillStyle='#171b20';ctx.fillRect(0,y,150,26);ctx.strokeStyle='#252c33';ctx.beginPath();ctx.moveTo(0,y+25.5);ctx.lineTo(width,y+25.5);ctx.stroke();ctx.fillStyle='#aab2ba';ctx.fillText(source,8,y+17);for(const m of data.marks){if(m.time_seconds<start||m.time_seconds>end)continue;const hit=m.sources.find(s=>s.source_type===source);if(!hit)continue;ctx.fillStyle=`rgba(139,207,155,${Math.max(.25,hit.probability)})`;ctx.fillRect(Math.round(xFor(m.time_seconds)-scrollX),y+8,2,11)}})}
function renderTable(){marksBody.innerHTML='';if(!data.marks.length){marksBody.innerHTML='<tr><td class="empty" colspan="6">No marks</td></tr>';return}const frag=document.createDocumentFragment();data.marks.forEach(m=>{const tr=document.createElement('tr');const other=m.sources.slice(1).map(s=>`${s.source_type} ${fmt(s.probability,2)}`).join(', ');tr.innerHTML=`<td>${m.index}</td><td>${fmt(m.time_seconds)}</td><td>${fmt(m.onset_probability,3)}</td><td>${m.primary_source}</td><td>${fmt(m.primary_source_probability,3)}</td><td>${other}</td>`;tr.addEventListener('click',()=>seek(m.time_seconds,true));frag.appendChild(tr)});marksBody.appendChild(frag)}
function nearestMark(time,maxSeconds){let best=null,bestDist=maxSeconds;for(const m of data.marks){const dist=Math.abs(m.time_seconds-time);if(dist<bestDist){best=m;bestDist=dist}}return best}
function describeMark(m){return `#${m.index} ${fmt(m.time_seconds)}s - ${m.primary_source} ${fmt(m.primary_source_probability,2)} - onset ${fmt(m.onset_probability,2)}`}
function timelineXFromWaveEvent(event){const rect=waveWrap.getBoundingClientRect();return event.clientX-rect.left}
function seek(seconds,play=false){audio.currentTime=Math.max(0,Math.min(duration,seconds));if(play)audio.play();updatePlayhead()}
function updatePlayhead(){const absoluteX=xFor(audio.currentTime||0);const visibleX=absoluteX-viewport.scrollLeft;const display=visibleX>=0&&visibleX<=viewport.clientWidth?'block':'none';playhead.style.transform=`translateX(${absoluteX}px)`;lanePlayhead.style.transform=`translateX(${absoluteX}px)`;playhead.style.display=display;lanePlayhead.style.display=display;time.textContent=`${fmt(audio.currentTime||0)} / ${fmt(duration)}`;if(follow.checked&&!audio.paused){viewport.scrollLeft=Math.max(0,absoluteX-viewport.clientWidth*.35)}}
function playLoop(){updatePlayhead();rafId=requestAnimationFrame(playLoop)}
audio.addEventListener('play',()=>{cancelAnimationFrame(rafId);playLoop()});audio.addEventListener('pause',()=>{cancelAnimationFrame(rafId);updatePlayhead()});audio.addEventListener('seeked',updatePlayhead);audio.addEventListener('loadedmetadata',updatePlayhead);
zoom.addEventListener('input',()=>{const centerTime=tFor(viewport.scrollLeft+viewport.clientWidth*.5);queueRender();requestAnimationFrame(()=>{viewport.scrollLeft=Math.max(0,xFor(centerTime)-viewport.clientWidth*.5)})});
viewport.addEventListener('wheel',event=>{if(!event.ctrlKey)return;event.preventDefault();const rect=viewport.getBoundingClientRect();const anchorX=viewport.scrollLeft+event.clientX-rect.left;const anchorTime=tFor(anchorX);const factor=event.deltaY<0?1.16:1/1.16;zoom.value=String(Math.max(Number(zoom.min),Math.min(Number(zoom.max),Number(zoom.value)*factor)));queueRender();requestAnimationFrame(()=>{viewport.scrollLeft=Math.max(0,xFor(anchorTime)-(event.clientX-rect.left))})},{passive:false});
waveWrap.addEventListener('click',event=>{seek(tFor(timelineXFromWaveEvent(event)),false)});
waveWrap.addEventListener('mousemove',event=>{const x=timelineXFromWaveEvent(event);const m=nearestMark(tFor(x),Math.max(.025,8/pxPerSec));if(!m){tooltip.style.display='none';return}tooltip.textContent=describeMark(m);tooltip.style.left=`${event.clientX+12}px`;tooltip.style.top=`${event.clientY+12}px`;tooltip.style.display='block'});
waveWrap.addEventListener('mouseleave',()=>{tooltip.style.display='none'});
window.addEventListener('keydown',event=>{if(event.code!=='Space')return;const tag=(event.target&&event.target.tagName)||'';if(['INPUT','BUTTON','SELECT','TEXTAREA'].includes(tag))return;event.preventDefault();if(audio.paused)audio.play();else audio.pause()});
window.addEventListener('resize',queueRender);
viewport.addEventListener('scroll',queueRender,{passive:true});
renderStats();renderTable();renderAll();updatePlayhead();
</script>
</body></html>"""
    return html.replace("__PAYLOAD__", payload_json)
