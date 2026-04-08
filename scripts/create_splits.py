"""
create_splits.py

Creates a temporal 80/10/10 holdout split from the existing preprocessed.npz
WITHOUT retraining. Indices are purely temporal (chronological window order).

Split definition:
  [0 : 80%)   → train candidates → filter to normal only
  [80% : 90%) → val candidates   → filter to normal only
  [90% : 100%] → test set        → all windows (normal + attack)

Threshold for holdout evaluation = 99th percentile of normal scores in [0:90%).

Output:
    data/processed/splits.npz  — contains: train_idx, val_idx, test_idx

Run from project root:
    python scripts/create_splits.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    npz_path = "data/processed/preprocessed.npz"
    out_path = "data/processed/splits.npz"

    bundle = np.load(npz_path, allow_pickle=True)
    y_seq = bundle["y_seq"].astype(np.int64)
    N = len(y_seq)
    print(f"[create_splits] Total windows N={N:,}")

    split_80 = int(N * 0.80)
    split_90 = int(N * 0.90)
    print(f"[create_splits] Split boundaries: train/val at {split_80:,}, val/test at {split_90:,}")

    # Normal indices in each zone
    all_idx = np.arange(N)
    train_idx = all_idx[(all_idx < split_80) & (y_seq == 0)]
    val_idx   = all_idx[(all_idx >= split_80) & (all_idx < split_90) & (y_seq == 0)]
    test_idx  = all_idx[all_idx >= split_90]   # all windows (normal + attack)

    n_test_attack = int((y_seq[test_idx] == 1).sum())
    n_test_normal = int((y_seq[test_idx] == 0).sum())
    attack_rate = n_test_attack / len(test_idx) * 100

    print(f"\n[create_splits] Split summary:")
    print(f"  Train (normal only): {len(train_idx):>8,}")
    print(f"  Val   (normal only): {len(val_idx):>8,}")
    print(f"  Test  (all):         {len(test_idx):>8,}")
    print(f"    Test normal:       {n_test_normal:>8,}")
    print(f"    Test attack:       {n_test_attack:>8,}  ({attack_rate:.2f}% of test)")

    # Sanity checks
    assert (y_seq[train_idx] == 0).all(), "train_idx contains attack windows!"
    assert (y_seq[val_idx] == 0).all(),   "val_idx contains attack windows!"
    assert test_idx.min() >= split_90,    "test_idx contains pre-test windows!"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"\n[create_splits] Saved: {out_path}")


if __name__ == "__main__":
    main()
