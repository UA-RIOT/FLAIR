"""
evaluate_split.py

Evaluate a FLAIR model trained on a pre-computed split.
Threshold is derived from val_idx normal windows only (never from test).
Reports full metrics on the test set with per-attack-type breakdown.

Usage:
    python scripts/evaluate_split.py --split 80_10_10
    python scripts/evaluate_split.py --split 70_15_15
    python scripts/evaluate_split.py --split 60_20_20

Outputs (per split):
    experiments/results/scores_{split}.csv   — window_idx, anomaly_score, y_true, y_type
    experiments/results/metrics_{split}.txt  — full text report

Run from project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.training.evaluate_flair import (
    compute_scores,
    compute_threshold,
    confusion_from_threshold,
    metrics_from_confusion,
    roc_pr_curves,
    auc_trapz,
    best_f1_threshold,
    load_checkpoint,
)

ATTACK_TYPES = ["DoS", "Reconn", "CommInj", "Backdoor"]


def evaluate_split(split_name: str, config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _p = cfg.get("paths", {})
    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))
    splits_path = f"data/processed/splits_{split_name}.npz"
    ckpt_path = f"experiments/results/flair_{split_name}.pt"
    types_path = "data/processed/window_types.npy"
    out_csv = f"experiments/results/scores_{split_name}.csv"
    out_txt = f"experiments/results/metrics_{split_name}.txt"

    # ---- Resolve device ----
    device_str = cfg.get("training", {}).get("device", "auto")
    if device_str in ("auto", "cuda"):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[evaluate_split] Device: {device}")

    # ---- Load model ----
    model, _ = load_checkpoint(ckpt_path, device)
    print(f"[evaluate_split] Loaded checkpoint: {ckpt_path}")

    # ---- Load data ----
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64)

    splits = np.load(splits_path, allow_pickle=True)
    val_idx  = splits["val_idx"]
    test_idx = splits["test_idx"]

    window_types = np.load(types_path, allow_pickle=True) if Path(types_path).exists() else None

    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 2048))
    num_workers = int(cfg.get("training", {}).get("num_workers", 4))

    # ---- Compute scores on val set (for threshold) ----
    print(f"[evaluate_split] Scoring val set ({len(val_idx):,} windows)...")
    val_scores = compute_scores(model, X_num[val_idx], X_cat[val_idx], batch_size, device, num_workers)
    threshold = compute_threshold(val_scores, percentile=99.0)
    print(f"[evaluate_split] Threshold (p99 val normal): {threshold:.6f}")

    # ---- Compute scores on test set ----
    print(f"[evaluate_split] Scoring test set ({len(test_idx):,} windows)...")
    test_scores = compute_scores(model, X_num[test_idx], X_cat[test_idx], batch_size, device, num_workers)
    y_test = y_seq[test_idx]

    # ---- Metrics ----
    cm = confusion_from_threshold(y_test, test_scores, threshold)
    m = metrics_from_confusion(**cm)
    curves = roc_pr_curves(y_test, test_scores)
    roc_auc = auc_trapz(curves["fpr"], curves["tpr"])
    pr_auc  = auc_trapz(curves["recall"], curves["precision"])
    best_thr, best_m = best_f1_threshold(y_test, test_scores)
    best_cm = confusion_from_threshold(y_test, test_scores, best_thr)

    # ---- Per-attack-type breakdown ----
    type_breakdown = {}
    if window_types is not None:
        test_types = window_types[test_idx]
        for t in ATTACK_TYPES:
            t_mask = test_types == t
            n_total = int(t_mask.sum())
            if n_total == 0:
                type_breakdown[t] = (0, 0, 0.0)
                continue
            t_scores = test_scores[t_mask]
            n_detected = int((t_scores > threshold).sum())
            detection_rate = n_detected / n_total * 100
            type_breakdown[t] = (n_total, n_detected, detection_rate)

    # ---- Build report ----
    lines = [
        f"FLAIR Evaluation — Split {split_name}",
        "=" * 60,
        f"Checkpoint:   {ckpt_path}",
        f"Val windows:  {len(val_idx):,} (normal only, used for threshold)",
        f"Test windows: {len(test_idx):,}  (normal + attack)",
        f"Threshold (p99 val normal): {threshold:.6f}",
        "",
        "=== Metrics @ operational threshold ===",
        f"Confusion: TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}",
        f"Accuracy:  {m['accuracy']:.6f}",
        f"Precision: {m['precision']:.6f}",
        f"Recall:    {m['recall']:.6f}  (TPR)",
        f"F1:        {m['f1']:.6f}",
        f"FPR:       {m['fpr']:.6f}",
        "",
        "=== Threshold-independent metrics ===",
        f"ROC AUC: {roc_auc:.6f}",
        f"PR  AUC: {pr_auc:.6f}",
        "",
        "=== Best-F1 threshold (label-informed upper bound) ===",
        f"Best threshold: {best_thr:.6f}",
        f"Confusion: TP={best_cm['tp']}  FP={best_cm['fp']}  TN={best_cm['tn']}  FN={best_cm['fn']}",
        f"Precision: {best_m['precision']:.6f}  Recall: {best_m['recall']:.6f}  F1: {best_m['f1']:.6f}",
    ]

    if type_breakdown:
        lines += ["", "=== Per-attack-type detection rate @ operational threshold ==="]
        for t, (n_total, n_det, rate) in type_breakdown.items():
            lines.append(f"  {t:25s}: {n_det:>6,} / {n_total:>6,}  ({rate:.1f}%)")

    report = "\n".join(lines)
    print("\n" + report)

    # ---- Save outputs ----
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\n[evaluate_split] Saved metrics: {out_txt}")

    y_type_col = window_types[test_idx] if window_types is not None else np.full(len(test_idx), "unknown", dtype=object)
    pd.DataFrame({
        "window_idx": test_idx,
        "anomaly_score": test_scores.astype(np.float32),
        "y_true": y_test.astype(int),
        "y_type": y_type_col,
    }).to_csv(out_csv, index=False)
    print(f"[evaluate_split] Saved scores:  {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FLAIR on a pre-computed split's test set.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["80_10_10", "70_15_15", "60_20_20"],
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate_split(args.split, config_path=args.config)
