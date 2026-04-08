"""
extract_window_types.py

Replays the same preprocessing pipeline as preprocess_data.py to extract
per-window traffic type labels (DoS, Reconnaissance, Command_Injection,
Backdoor, Normal) and saves them as data/processed/window_types.npy.

Must produce the same N windows in the same order as preprocessed.npz.
Run from the project root:
    python scripts/extract_window_types.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_datetime_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    NUMERIC_FEATURES = cfg["features"]["numeric"]
    time_col = cfg["data"]["time_column"]       # "StartTime"
    label_col = cfg["data"]["label_column"]     # "Target"
    traffic_col = "Traffic"                     # WUSTL-IIoT traffic type column

    window_size = int(cfg["preprocess"]["window_size"])
    stride = int(cfg["preprocess"].get("stride", 1))
    sort_time = bool(cfg["preprocess"].get("sort_time", True))
    dropna = bool(cfg["preprocess"].get("dropna", True))

    paths_cfg = cfg.get("paths", {})
    full_csv = paths_cfg.get("full_csv")
    sample_xlsx = paths_cfg.get("sample_xlsx")
    input_path = full_csv or sample_xlsx
    if not input_path:
        raise ValueError("No input dataset path in config (paths.full_csv or paths.sample_xlsx).")

    out_npy = "data/processed/window_types.npy"
    npz_path = paths_cfg.get("processed_npz", "data/processed/preprocessed.npz")

    print(f"[extract_window_types] Reading: {input_path}")
    if input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.lower().endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path}")
    print(f"[extract_window_types] Loaded shape: {df.shape}")

    # Verify Traffic column exists
    if traffic_col not in df.columns:
        raise KeyError(
            f"Column '{traffic_col}' not found. Available: {list(df.columns)}"
        )

    # ---- Apply EXACT same pipeline as preprocess_data.py ----
    required = [time_col, label_col, traffic_col] + NUMERIC_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    work = df[required].copy()
    work[time_col] = to_datetime_safe(work[time_col])

    if dropna:
        # Same as preprocess_data.py: drop on time+label+numeric only (not categorical/Traffic)
        work = work.dropna(subset=[time_col, label_col] + NUMERIC_FEATURES).copy()

    if sort_time:
        work = work.sort_values(by=time_col).reset_index(drop=True)

    y_row = work[label_col].astype(int).to_numpy(dtype=np.int64)
    traffic_arr = work[traffic_col].to_numpy(dtype=object)

    M = len(y_row)
    if M < window_size:
        raise ValueError(f"Only {M} rows after filtering, need at least {window_size}.")

    num_windows = 1 + (M - window_size) // stride
    print(f"[extract_window_types] Rows after filtering: {M}, windows: {num_windows}")

    # ---- Build per-window type using same loop as build_sliding_windows ----
    window_types = np.empty(num_windows, dtype=object)

    w = 0
    for start in range(0, M - window_size + 1, stride):
        end = start + window_size
        labels_in_window = y_row[start:end]
        if labels_in_window.max() > 0:
            # Majority type among attack rows in this window
            attack_mask = labels_in_window == 1
            attack_types = traffic_arr[start:end][attack_mask]
            # Get majority (most common)
            values, counts = np.unique(attack_types, return_counts=True)
            window_types[w] = values[counts.argmax()]
        else:
            window_types[w] = "Normal"
        w += 1

    # ---- Verify N matches preprocessed.npz ----
    if Path(npz_path).exists():
        bundle = np.load(npz_path, allow_pickle=True)
        n_npz = len(bundle["y_seq"])
        if n_npz != num_windows:
            print(
                f"[WARNING] window_types N={num_windows} does NOT match preprocessed.npz N={n_npz}. "
                "Check that the same CSV and dropna/sort settings are used."
            )
        else:
            print(f"[extract_window_types] N matches preprocessed.npz: {num_windows}")
    else:
        print(f"[WARNING] {npz_path} not found, skipping N verification.")

    Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, window_types)
    print(f"[extract_window_types] Saved: {out_npy}  shape={window_types.shape}")

    # Print value counts
    unique, counts = np.unique(window_types, return_counts=True)
    print("\n[extract_window_types] Value counts:")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {u:25s}: {c:>8,}")


if __name__ == "__main__":
    main()
