"""
train_split.py

Train FLAIR on a pre-computed train/val split (from create_splits_retrain.py).
Bypasses the internal split logic in train_flair.py by slicing arrays directly.

Usage:
    python scripts/train_split.py --split 80_10_10
    python scripts/train_split.py --split 75_15_15
    python scripts/train_split.py --split 60_20_20

Outputs:
    experiments/results/flair_{split}.pt

Run from project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FLAIRDataset, DatasetConfig
from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig
from src.training.train_flair import (
    TrainConfig,
    _resolve_device,
    set_seed,
    train_one_epoch,
    eval_one_epoch,
)


def train_on_split(split_name: str, config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _t = cfg.get("training", {})
    _p = cfg.get("paths", {})
    _m = cfg.get("model", {})

    train_cfg = TrainConfig(
        batch_size=int(_t.get("batch_size", 512)),
        learning_rate=float(_t.get("learning_rate", 1e-3)),
        epochs=int(_t.get("epochs", 32)),
        seed=int(_t.get("seed", 42)),
        device=str(_t.get("device", "auto")),
        checkpoint_path=f"experiments/results/flair_{split_name}.pt",
        val_split=float(_t.get("val_split", 0.1)),   # unused — we use pre-split indices
        patience=_t.get("patience", 10),
        num_workers=int(_t.get("num_workers", 4)),
        amp=bool(_t.get("amp", True)),
    )

    set_seed(train_cfg.seed)
    device = _resolve_device(train_cfg.device)

    # ---- Load data ----
    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))
    splits_path = f"data/processed/splits_{split_name}.npz"

    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    sport_vocab = bundle["sport_vocab"][0]
    dport_vocab = bundle["dport_vocab"][0]
    proto_vocab = bundle["proto_vocab"][0]

    splits = np.load(splits_path, allow_pickle=True)
    train_idx = splits["train_idx"]
    val_idx   = splits["val_idx"]

    Xn_tr = X_num[train_idx]
    Xc_tr = X_cat[train_idx]
    Xn_val = X_num[val_idx]
    Xc_val = X_cat[val_idx]

    print(f"[train_split] Split: {split_name}")
    print(f"[train_split] Train windows: {len(Xn_tr):,}  Val windows: {len(Xn_val):,}")

    # ---- Build model ----
    model_cfg = FLAIRConfig(
        numeric_dim=int(X_num.shape[-1]),
        sport_vocab_size=len(sport_vocab) + 1,
        dport_vocab_size=len(dport_vocab) + 1,
        proto_vocab_size=len(proto_vocab) + 1,
        embed_dim=int(_m.get("embed_dim", 8)),
        hidden_dim=int(_m.get("hidden_dim", 128)),
        num_layers=int(_m.get("num_layers", 1)),
        dropout=float(_m.get("dropout", 0.1)),
        bidirectional=bool(_m.get("bidirectional", False)),
        cat_loss_weight=float(_m.get("cat_loss_weight", 0.1)),
    )

    # ---- DataLoaders ----
    pin = device.type == "cuda"
    persistent = train_cfg.num_workers > 0

    train_loader = DataLoader(
        FLAIRDataset(Xn_tr, Xc_tr, config=DatasetConfig(return_targets=True)),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        FLAIRDataset(Xn_val, Xc_val, config=DatasetConfig(return_targets=True)),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    model = FLAIRAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[train_split] AMP: {'enabled' if use_amp else 'disabled'}  num_workers: {train_cfg.num_workers}")

    # ---- Training loop (same logic as train_flair.py) ----
    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_left = train_cfg.patience if train_cfg.patience is not None else None

    for epoch in range(1, train_cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va = eval_one_epoch(model, val_loader, device)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"Epoch {epoch}/{train_cfg.epochs} - train: {tr:.6f}  val: {va:.6f}")

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if patience_left is not None:
                patience_left = train_cfg.patience
        else:
            if patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train_split] Early stopping at epoch {epoch} (best epoch {best_epoch}, best val {best_val:.6f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Save checkpoint ----
    ckpt_path = Path(train_cfg.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "split_name": split_name,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    print(f"\n[train_split] Saved checkpoint: {ckpt_path}")
    print(f"[train_split] Best val loss: {best_val:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FLAIR on a pre-computed split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["80_10_10", "70_15_15", "60_20_20"],
        help="Split name (must match splits_{name}.npz in data/processed/)",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train_on_split(args.split, config_path=args.config)
