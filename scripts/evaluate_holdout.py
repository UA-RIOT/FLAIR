"""
evaluate_holdout.py

Re-reports FLAIR metrics on a formally held-out test set (last 10% of windows)
using pre-computed scores from anomaly_scores_full.csv.
No model inference needed — all scores already computed.

Threshold = 99th percentile of normal scores in train+val portion (first 90%).

Outputs:
    experiments/results/holdout_metrics.txt

Run from project root:
    python scripts/create_splits.py       # must run first
    python scripts/evaluate_holdout.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.evaluate_flair import (
    compute_threshold,
    confusion_from_threshold,
    metrics_from_confusion,
    roc_pr_curves,
    auc_trapz,
    best_f1_threshold,
)


def main() -> None:
    scores_csv = "experiments/results/anomaly_scores_full.csv"
    splits_path = "data/processed/splits.npz"
    out_txt = "experiments/results/holdout_metrics.txt"

    # ---- Load scores ----
    df = pd.read_csv(scores_csv)
    required = {"window_idx", "anomaly_score", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"anomaly_scores_full.csv is missing columns: {missing}")

    # Sort by window_idx to be safe
    df = df.sort_values("window_idx").reset_index(drop=True)
    scores_all = df["anomaly_score"].to_numpy(dtype=np.float32)
    y_all = df["y_true"].to_numpy(dtype=np.int64)
    N = len(df)
    print(f"[evaluate_holdout] Loaded scores: N={N:,}")

    # ---- Load split indices ----
    splits = np.load(splits_path, allow_pickle=True)
    train_idx = splits["train_idx"]
    val_idx   = splits["val_idx"]
    test_idx  = splits["test_idx"]

    # Threshold: 99th percentile of normal windows in train+val (indices 0–90%)
    pre_test_normal_mask = np.zeros(N, dtype=bool)
    pre_test_normal_mask[train_idx] = True
    pre_test_normal_mask[val_idx]   = True
    normal_pre_test_scores = scores_all[pre_test_normal_mask]
    threshold = compute_threshold(normal_pre_test_scores, percentile=99.0)
    print(f"[evaluate_holdout] Threshold (p99, train+val normal): {threshold:.6f}")

    # ---- Test set ----
    test_scores = scores_all[test_idx]
    y_test = y_all[test_idx]

    n_test_attack = int((y_test == 1).sum())
    n_test_normal = int((y_test == 0).sum())
    print(f"[evaluate_holdout] Test set: {len(test_idx):,} windows  "
          f"({n_test_normal:,} normal, {n_test_attack:,} attack, "
          f"{n_test_attack/len(test_idx)*100:.2f}% attack rate)")

    # ---- Metrics ----
    cm = confusion_from_threshold(y_test, test_scores, threshold)
    m = metrics_from_confusion(**cm)
    curves = roc_pr_curves(y_test, test_scores)
    roc_auc = auc_trapz(curves["fpr"], curves["tpr"])
    pr_auc  = auc_trapz(curves["recall"], curves["precision"])
    best_thr, best_m = best_f1_threshold(y_test, test_scores)
    best_cm = confusion_from_threshold(y_test, test_scores, best_thr)

    # ---- Full dataset metrics (for comparison) ----
    cm_full = confusion_from_threshold(y_all, scores_all, threshold)
    m_full = metrics_from_confusion(**cm_full)

    # ---- Build report ----
    lines = [
        "FLAIR Holdout Evaluation (Temporal 80/10/10 Split)",
        "=" * 60,
        f"Scores source:   {scores_csv}",
        f"Splits source:   {splits_path}",
        f"Total windows:   {N:,}",
        f"Train (normal):  {len(train_idx):,}",
        f"Val (normal):    {len(val_idx):,}",
        f"Test (all):      {len(test_idx):,}  ({n_test_attack/len(test_idx)*100:.2f}% attack)",
        f"Threshold (p99 train+val normal): {threshold:.6f}",
        "",
        "=== Metrics on HELD-OUT TEST SET (last 10% of windows) ===",
        f"Confusion: TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}",
        f"Accuracy:  {m['accuracy']:.6f}",
        f"Precision: {m['precision']:.6f}",
        f"Recall:    {m['recall']:.6f}  (TPR)",
        f"F1:        {m['f1']:.6f}",
        f"FPR:       {m['fpr']:.6f}",
        "",
        "=== Threshold-independent metrics (test set) ===",
        f"ROC AUC: {roc_auc:.6f}",
        f"PR  AUC: {pr_auc:.6f}",
        "",
        "=== Best-F1 threshold (label-informed upper bound, test set) ===",
        f"Best threshold: {best_thr:.6f}",
        f"Confusion: TP={best_cm['tp']}  FP={best_cm['fp']}  TN={best_cm['tn']}  FN={best_cm['fn']}",
        f"Precision: {best_m['precision']:.6f}  Recall: {best_m['recall']:.6f}  F1: {best_m['f1']:.6f}",
        "",
        "=== Full-dataset metrics (for reference) ===",
        f"Confusion: TP={cm_full['tp']}  FP={cm_full['fp']}  TN={cm_full['tn']}  FN={cm_full['fn']}",
        f"Accuracy:  {m_full['accuracy']:.6f}",
        f"Precision: {m_full['precision']:.6f}",
        f"Recall:    {m_full['recall']:.6f}",
        f"F1:        {m_full['f1']:.6f}",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\n[evaluate_holdout] Saved: {out_txt}")


if __name__ == "__main__":
    main()
