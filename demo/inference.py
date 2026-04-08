"""
demo/inference.py

Loads model, data, and pre-computes all scores once on startup.
Provides run_inference(window_idx) and get_indices(filter_mode) for the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
import yaml

from src.training.evaluate_flair import load_checkpoint, compute_scores


# ---------------------------------------------------------------------------
# Startup loading (called once, cached by Streamlit)
# ---------------------------------------------------------------------------

_state: Dict = {}


def _load_all(config_path: str = "config.yaml") -> None:
    """Load model, data, and compute all scores. Stored in module-level _state."""
    if _state:
        return  # already loaded

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _p = cfg.get("paths", {})
    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))
    ckpt_path = str(cfg.get("training", {}).get("checkpoint_path", "experiments/results/flair_minimal.pt"))

    # Device (CPU for demo — laptop)
    device = torch.device("cpu")

    # Load model
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    # Load data
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)    # (N, 10, 21)
    X_cat = bundle["X_cat"].astype(np.int64)      # (N, 10, 3)
    y_seq = bundle["y_seq"].astype(np.int64)      # (N,)

    # Vocab dicts for decoding categorical IDs → original values
    sport_vocab: dict = bundle["sport_vocab"][0]
    dport_vocab: dict = bundle["dport_vocab"][0]
    proto_vocab: dict = bundle["proto_vocab"][0]

    # Invert vocabs: id → raw value
    inv_sport = {v: k for k, v in sport_vocab.items()}
    inv_dport = {v: k for k, v in dport_vocab.items()}
    inv_proto = {v: k for k, v in proto_vocab.items()}

    # Compute all anomaly scores once
    print("[demo] Computing anomaly scores for all windows...")
    scores = compute_scores(model, X_num, X_cat, batch_size=2048, device=device, num_workers=0)

    # Threshold: 99th percentile of normal-only scores
    threshold = float(np.percentile(scores[y_seq == 0], 99))
    print(f"[demo] Threshold (p99 normal): {threshold:.6f}")

    _state.update({
        "model": model,
        "device": device,
        "X_num": X_num,
        "X_cat": X_cat,
        "y_seq": y_seq,
        "scores": scores,
        "threshold": threshold,
        "inv_sport": inv_sport,
        "inv_dport": inv_dport,
        "inv_proto": inv_proto,
        "num_features": list(bundle["num_features"]),
    })


def ensure_loaded(config_path: str = "config.yaml") -> None:
    _load_all(config_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_threshold() -> float:
    ensure_loaded()
    return _state["threshold"]


def get_scores() -> np.ndarray:
    ensure_loaded()
    return _state["scores"]


def get_labels() -> np.ndarray:
    ensure_loaded()
    return _state["y_seq"]


def get_indices(filter_mode: Literal["All", "Normal", "Attack"] = "All") -> np.ndarray:
    """Return window indices matching the filter."""
    ensure_loaded()
    y = _state["y_seq"]
    N = len(y)
    if filter_mode == "Normal":
        return np.where(y == 0)[0]
    elif filter_mode == "Attack":
        return np.where(y == 1)[0]
    return np.arange(N)


def run_inference(window_idx: int) -> Dict:
    """
    Run model forward pass on a single window.

    Returns dict with:
        x_num_raw:     (10, 21) numpy  — normalized numeric input
        x_cat_raw:     (10, 3)  numpy  — categorical IDs
        latent:        (128,)   numpy  — encoder hidden state
        x_hat_num:     (10, 21) numpy  — decoder reconstruction
        per_feat_err:  (21,)    numpy  — MSE per feature (mean over T)
        anomaly_score: float
        threshold:     float
        is_attack:     bool
        ground_truth:  int  (0=normal, 1=attack)
        cat_decoded:   list of dicts [{Sport, Dport, Proto}, ...]  for each of 10 timesteps
        num_features:  list of 21 feature names
    """
    ensure_loaded()

    model: torch.nn.Module = _state["model"]
    device: torch.device = _state["device"]
    X_num: np.ndarray = _state["X_num"]
    X_cat: np.ndarray = _state["X_cat"]

    x_num_np = X_num[window_idx]   # (10, 21)
    x_cat_np = X_cat[window_idx]   # (10, 3)

    x_num_t = torch.from_numpy(x_num_np).unsqueeze(0).to(device)  # (1, 10, 21)
    x_cat_t = torch.from_numpy(x_cat_np).unsqueeze(0).to(device)  # (1, 10, 3)

    with torch.no_grad():
        out = model(x_num_t, x_cat_t)

    x_hat_np  = out["x_hat_num"].squeeze(0).cpu().numpy()   # (10, 21)
    latent_np = out["latent"].squeeze(0).cpu().numpy()       # (128,)

    per_feat_err = np.mean((x_hat_np - x_num_np) ** 2, axis=0)  # (21,)

    score = float(_state["scores"][window_idx])
    threshold = _state["threshold"]

    # Decode categorical IDs back to original values
    inv_sport = _state["inv_sport"]
    inv_dport = _state["inv_dport"]
    inv_proto = _state["inv_proto"]

    cat_decoded: List[Dict] = []
    for t in range(x_cat_np.shape[0]):
        sport_id, dport_id, proto_id = int(x_cat_np[t, 0]), int(x_cat_np[t, 1]), int(x_cat_np[t, 2])
        cat_decoded.append({
            "Sport": inv_sport.get(sport_id, f"UNK({sport_id})"),
            "Dport": inv_dport.get(dport_id, f"UNK({dport_id})"),
            "Proto": inv_proto.get(proto_id, f"UNK({proto_id})"),
        })

    return {
        "x_num_raw": x_num_np,
        "x_cat_raw": x_cat_np,
        "latent": latent_np,
        "x_hat_num": x_hat_np,
        "per_feat_err": per_feat_err,
        "anomaly_score": score,
        "threshold": threshold,
        "is_attack": score > threshold,
        "ground_truth": int(_state["y_seq"][window_idx]),
        "cat_decoded": cat_decoded,
        "num_features": _state["num_features"],
    }
