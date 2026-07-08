from collections import Counter
from comet_audio.training import CometTimingDataset
from pathlib import Path
import torch

d = CometTimingDataset(
    Path(r"C:\dev\comet\data\generated\surge_train_100k"),
    "val",
    training=False,
    target="anonymous_slots_v1",
    max_tracks=16,
    crop_seconds=8.0,
)
counts = Counter()
active_slots = 0
slot_count = 0
for i in range(len(d)):
    b = d[i]
    phases = torch.stack([b["slot_attack"], b["slot_held"], b["slot_release"]], dim=1)
    active = phases.amax(dim=1)
    cls = torch.where(
        active >= 0.5, phases.argmax(dim=1) + 1, torch.zeros_like(active, dtype=torch.long)
    )
    vals, nums = torch.unique(cls, return_counts=True)
    for v, n in zip(vals.tolist(), nums.tolist()):
        counts[int(v)] += int(n)
    active_slots += int(b["slot_activity"].sum().item())
    slot_count += int(b["slot_activity"].numel())
labels = ["off", "attack", "held", "release"]
total = sum(counts.values())
print("clips", len(d), "active_slots", active_slots, "slots", slot_count)
for i, l in enumerate(labels):
    print(l, counts[i], counts[i] / total)
