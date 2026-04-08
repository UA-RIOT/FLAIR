"""
create_splits_retrain.py

Creates train/val/test index files for three split ratios:
  80/10/10, 75/15/15, 60/20/20

Split design:
  Normal windows  → temporal split (chronological, matching sort order from preprocessing)
  Attack windows  → proportionally sampled to match dataset's natural 7.28% attack rate
                    and per-type distribution (DoS/Recon/CmdInj/Backdoor)

Outputs:
  data/processed/splits_80_10_10.npz
  data/processed/splits_75_15_15.npz
  data/processed/splits_60_20_20.npz

Each npz contains: train_idx, val_idx, test_idx

Run from project root:
    python scripts/extract_window_types.py   # must run first
    python scripts/create_splits_retrain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Attack type fractions matching dataset's natural distribution (Table 2)
# Names must match the actual Traffic column values in the CSV
TYPE_FRACS = {
    "DoS":      0.8998,
    "Reconn":   0.0946,
    "CommInj":  0.0031,
    "Backdoor": 0.0025,
}

# Target attack rate in test set (matching dataset natural rate)
ATTACK_RATE = 7.28 / 100.0   # 7.28% of test windows are attack

SPLITS = {
    "80_10_10": (0.80, 0.10, 0.10),
    "70_15_15": (0.70, 0.15, 0.15),
    "60_20_20": (0.60, 0.20, 0.20),
}

SEED = 42


def main() -> None:
    npz_path = "data/processed/preprocessed.npz"
    types_path = "data/processed/window_types.npy"
    out_dir = Path("data/processed")

    bundle = np.load(npz_path, allow_pickle=True)
    y_seq = bundle["y_seq"].astype(np.int64)
    N = len(y_seq)
    print(f"[create_splits] Loaded preprocessed.npz: N={N:,}")

    window_types = np.load(types_path, allow_pickle=True)
    if len(window_types) != N:
        raise ValueError(
            f"window_types.npy has {len(window_types)} entries but y_seq has {N}. "
            "Re-run extract_window_types.py."
        )
    print(f"[create_splits] Loaded window_types.npy: {len(window_types):,}")

    rng = np.random.default_rng(SEED)
    normal_idx = np.where(y_seq == 0)[0]   # chronological order preserved
    n_normal = len(normal_idx)

    # Pre-compute attack index pools by type
    attack_idx_by_type = {
        t: np.where(window_types == t)[0]
        for t in TYPE_FRACS
    }
    print("\n[create_splits] Attack window pool sizes:")
    for t, idx in attack_idx_by_type.items():
        print(f"  {t:25s}: {len(idx):>8,}")

    for name, (train_frac, val_frac, _test_frac) in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"[create_splits] Building split: {name}")

        # ---- Normal: temporal ----
        train_end = int(n_normal * train_frac)
        val_end   = int(n_normal * (train_frac + val_frac))

        train_idx       = normal_idx[:train_end]
        val_idx         = normal_idx[train_end:val_end]
        test_normal_idx = normal_idx[val_end:]

        n_test_normal = len(test_normal_idx)

        # ---- Attack: proportional sampling ----
        n_attack_needed = int(round(n_test_normal * (ATTACK_RATE / (1.0 - ATTACK_RATE))))

        test_attack_parts = []
        for t, frac in TYPE_FRACS.items():
            n_sample = int(round(n_attack_needed * frac))
            pool = attack_idx_by_type[t]
            if n_sample > len(pool):
                print(
                    f"  [WARNING] {t}: need {n_sample} but only {len(pool)} available. "
                    "Using all."
                )
                n_sample = len(pool)
            sampled = rng.choice(pool, size=n_sample, replace=False)
            test_attack_parts.append(sampled)

        test_attack_idx = np.concatenate(test_attack_parts)
        test_idx = np.sort(np.concatenate([test_normal_idx, test_attack_idx]))

        # ---- Save ----
        out_path = out_dir / f"splits_{name}.npz"
        np.savez(
            out_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        # ---- Verification ----
        n_test_attack = int((y_seq[test_idx] == 1).sum())
        actual_attack_rate = n_test_attack / len(test_idx) * 100
        print(f"  Train:        {len(train_idx):>8,} normal windows")
        print(f"  Val:          {len(val_idx):>8,} normal windows")
        print(f"  Test normal:  {n_test_normal:>8,}")
        print(f"  Test attack:  {n_test_attack:>8,}  ({actual_attack_rate:.2f}% of test set)")
        print(f"  Test total:   {len(test_idx):>8,}")

        # Per-type in test
        print(f"  Attack type breakdown in test set:")
        for t in TYPE_FRACS:
            type_mask = window_types[test_idx] == t
            n_type = int(type_mask.sum())
            pct = n_type / max(n_test_attack, 1) * 100
            print(f"    {t:25s}: {n_type:>6,}  ({pct:.1f}% of attacks)")

        # Verify train/val are all normal
        assert (y_seq[train_idx] == 0).all(), "train_idx contains attack windows!"
        assert (y_seq[val_idx] == 0).all(),   "val_idx contains attack windows!"

        print(f"  Saved: {out_path}")

    print("\n[create_splits] Done.")


if __name__ == "__main__":
    main()
