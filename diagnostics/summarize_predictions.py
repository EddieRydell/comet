import json
from collections import defaultdict
from pathlib import Path

PRED_ROOTS = {
    "epoch13_best": Path(r"C:\dev\comet\runs\offphase_100k_b16w4_lr2e4_e40_best_predictions"),
    "lr5e5_best": Path(r"C:\dev\comet\runs\offphase_100k_best_finetune_lr5e5_e5_predictions"),
}
META_ROOT = Path(r"C:\dev\comet\data\generated\surge_train_100k_shards\shard_000\metadata")
CLIPS = ["clip_0000", "clip_0003", "clip_0004"]


def pred_summary(path):
    data = json.loads(path.read_text())
    duration = float(data.get("duration_seconds") or 8.0)
    lanes = data.get("lanes") or data.get("tracks") or data.get("slots") or []
    slots = events = fullish = 0
    total = maxdur = 0.0
    for lane in lanes:
        evs = lane.get("events") or lane.get("segments") or []
        if evs:
            slots += 1
        lane_total = lane_max = 0.0
        for ev in evs:
            onset = float(ev.get("onset_seconds", ev.get("start_seconds", ev.get("start", 0))))
            offset = float(ev.get("offset_seconds", ev.get("end_seconds", ev.get("end", onset))))
            dur = max(0, offset - onset)
            events += 1
            total += dur
            lane_total += dur
            lane_max = max(lane_max, dur)
            maxdur = max(maxdur, dur)
        if lane_max >= duration * 0.80 or lane_total >= duration * 0.90:
            fullish += 1
    return slots, fullish, events, total, maxdur


def truth_summary(path):
    data = json.loads(path.read_text())
    duration = float(data.get("duration_seconds") or 8.0)
    by = defaultdict(list)
    for ev in data.get("events") or []:
        by[ev.get("source_id", "?")].append(ev)
    sources = events = fullish = 0
    total = maxdur = 0.0
    for evs in by.values():
        if evs:
            sources += 1
        lane_total = lane_max = 0.0
        for ev in evs:
            onset = float(ev.get("onset_seconds", 0))
            offset = float(ev.get("offset_seconds", onset))
            dur = max(0, offset - onset)
            events += 1
            total += dur
            lane_total += dur
            lane_max = max(lane_max, dur)
            maxdur = max(maxdur, dur)
        if lane_max >= duration * 0.80 or lane_total >= duration * 0.90:
            fullish += 1
    return sources, fullish, events, total, maxdur


for clip in CLIPS:
    t = truth_summary(META_ROOT / f"{clip}.json")
    print(
        f"{clip} truth sources={t[0]} fullish={t[1]} events={t[2]} active={t[3]:.2f}s max={t[4]:.2f}s"
    )
    for name, root in PRED_ROOTS.items():
        s = pred_summary(root / clip / "predictions.json")
        print(
            f"  {name} slots={s[0]} fullish={s[1]} events={s[2]} active={s[3]:.2f}s max={s[4]:.2f}s"
        )
