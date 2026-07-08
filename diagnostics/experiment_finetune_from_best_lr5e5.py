from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comet_audio.training import (
    CometTimingDataset,
    SlotAttentionEventModel,
    _run_epoch,
    _training_checkpoint_payload,
)

DATA_DIR = Path(r"C:\dev\comet\data\generated\surge_train_100k")
SOURCE_RUN = Path(r"C:\dev\comet\runs\offphase_100k_b16w4_lr2e4_e40")
RUN_DIR = Path(r"C:\dev\comet\runs\offphase_100k_best_finetune_lr5e5_e5")
EPOCHS = 5
BATCH_SIZE = 16
LOADER_WORKERS = 4
LR = 5e-5
MAX_TRACKS = 16
CROP_SECONDS = 8.0
TARGET = "anonymous_slots_v1"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(SOURCE_RUN / "best.pt", map_location=device, weights_only=False)
    if checkpoint.get("target") != TARGET:
        raise RuntimeError(f"unexpected checkpoint target: {checkpoint.get('target')!r}")

    train_dataset = CometTimingDataset(
        DATA_DIR,
        "train",
        training=True,
        target=TARGET,
        max_tracks=MAX_TRACKS,
        crop_seconds=CROP_SECONDS,
    )
    val_dataset = CometTimingDataset(
        DATA_DIR,
        "val",
        training=False,
        target=TARGET,
        max_tracks=MAX_TRACKS,
        crop_seconds=CROP_SECONDS,
    )
    loader_kwargs = {
        "num_workers": LOADER_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": LOADER_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = SlotAttentionEventModel(max_tracks=MAX_TRACKS).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = LR
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_val = float(checkpoint.get("best_validation_loss", math.inf))
    global_step = int(checkpoint.get("global_step", 0))
    start_epoch = int(checkpoint.get("epoch", 0))
    metrics_path = RUN_DIR / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    config = {
        "source_run": str(SOURCE_RUN),
        "source_checkpoint": "best.pt",
        "source_epoch": start_epoch,
        "starting_best_val": best_val,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "loader_workers": LOADER_WORKERS,
        "crop_seconds": CROP_SECONDS,
        "optimizer_state": "loaded_from_best_with_lr_override",
    }
    (RUN_DIR / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for local_epoch in range(1, EPOCHS + 1):
        epoch = start_epoch + local_epoch
        train_metrics, global_step = _run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            training=True,
            target=TARGET,
            run_dir=RUN_DIR,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics, global_step = _run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            training=False,
            target=TARGET,
            run_dir=RUN_DIR,
            epoch=epoch,
            global_step=global_step,
        )
        row = {
            "epoch": epoch,
            "local_epoch": local_epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        payload = _training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_val,
            target=TARGET,
            max_tracks=MAX_TRACKS,
            crop_seconds=CROP_SECONDS,
        )
        payload["source_checkpoint"] = str(SOURCE_RUN / "best.pt")
        torch.save(payload, RUN_DIR / "last.pt")
        if is_best:
            torch.save(payload, RUN_DIR / "best.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "local_epoch": local_epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "best_val": best_val,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
