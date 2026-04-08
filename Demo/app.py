"""
demo/app.py

FLAIR Live Demo — Streamlit GUI
Interactive visualization of the FLAIR anomaly detection pipeline.

Run from the project root:
    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import demo.inference as inference
from demo.visualizations import (
    input_heatmap,
    latent_bar,
    reconstruction_comparison,
    anomaly_gauge,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FLAIR — Anomaly Detection Demo",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model and data (cached — runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FLAIR model and computing scores...")
def load_everything():
    inference.ensure_loaded()
    return True


load_everything()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("F.L.A.I.R")
    st.caption("Flow-Level Autoencoder for Intrusion Recognition")
    st.divider()

    filter_mode = st.radio(
        "Window filter",
        options=["All", "Normal", "Attack"],
        index=0,
        help="Filter the slider to show only normal, only attack, or all windows.",
    )

    available_idx = inference.get_indices(filter_mode)  # type: ignore[arg-type]
    n_available = len(available_idx)

    if n_available == 0:
        st.error("No windows match the selected filter.")
        st.stop()

    slider_pos = st.slider(
        f"Window ({filter_mode}, {n_available:,} total)",
        min_value=0,
        max_value=n_available - 1,
        value=0,
        step=1,
    )
    window_idx = int(available_idx[slider_pos])

    st.divider()

    # Run inference for sidebar stats
    result = inference.run_inference(window_idx)
    score = result["anomaly_score"]
    threshold = result["threshold"]
    is_attack = result["is_attack"]
    ground_truth = result["ground_truth"]

    # Ground truth label
    gt_label = "ATTACK" if ground_truth == 1 else "NORMAL"
    gt_color = "red" if ground_truth == 1 else "green"
    st.markdown(f"**Ground Truth:** :{gt_color}[{gt_label}]")

    # Prediction
    pred_label = "ATTACK" if is_attack else "NORMAL"
    pred_color = "red" if is_attack else "green"
    st.markdown(f"**Prediction:**   :{pred_color}[{pred_label}]")

    st.metric("Anomaly Score", f"{score:.6f}")
    st.metric("Threshold (p99)", f"{threshold:.6f}")
    st.metric("Window Index", f"{window_idx:,}")

    st.divider()

    # Stats summary
    all_scores = inference.get_scores()
    all_labels = inference.get_labels()
    n_total = len(all_scores)
    n_attack_total = int((all_labels == 1).sum())
    st.caption(f"Dataset: {n_total:,} windows | {n_attack_total:,} attack ({n_attack_total/n_total*100:.1f}%)")

# ---------------------------------------------------------------------------
# Main area — Decision banner
# ---------------------------------------------------------------------------
if is_attack:
    st.error(f"🚨 ATTACK DETECTED   |   Score: {score:.6f}   |   Threshold: {threshold:.6f}")
else:
    st.success(f"✅ NORMAL TRAFFIC   |   Score: {score:.6f}   |   Threshold: {threshold:.6f}")

st.caption(
    f"Window {window_idx:,} · Ground truth: **{gt_label}** · "
    f"{'Correct' if (is_attack == bool(ground_truth)) else 'Incorrect'} prediction"
)

# ---------------------------------------------------------------------------
# Row 1: Input heatmap + categorical table
# ---------------------------------------------------------------------------
col_heat, col_cat = st.columns([3, 1])

with col_heat:
    fig_input = input_heatmap(result["x_num_raw"], result["num_features"])
    st.plotly_chart(fig_input, use_container_width=True)

with col_cat:
    st.markdown("**Categorical Features**")
    cat_df = pd.DataFrame(result["cat_decoded"])
    cat_df.index = [f"Flow {i+1}" for i in range(len(cat_df))]
    st.dataframe(cat_df, use_container_width=True, height=260)

# ---------------------------------------------------------------------------
# Row 2: Latent vector + Reconstruction comparison
# ---------------------------------------------------------------------------
col_latent, col_recon = st.columns([1, 2])

with col_latent:
    fig_latent = latent_bar(result["latent"])
    st.plotly_chart(fig_latent, use_container_width=True)

with col_recon:
    fig_recon = reconstruction_comparison(
        result["x_num_raw"],
        result["x_hat_num"],
        result["per_feat_err"],
        result["num_features"],
    )
    st.plotly_chart(fig_recon, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Anomaly gauge + Score distribution context
# ---------------------------------------------------------------------------
col_gauge, col_dist = st.columns([1, 2])

with col_gauge:
    fig_gauge = anomaly_gauge(score, threshold)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_dist:
    st.markdown("**Top-10 Most Anomalous Features**")
    feat_err = result["per_feat_err"]
    feat_names = result["num_features"]
    top10_idx = np.argsort(feat_err)[::-1][:10]
    top10_df = pd.DataFrame({
        "Feature": [feat_names[i] for i in top10_idx],
        "MSE": [f"{feat_err[i]:.6f}" for i in top10_idx],
    })
    top10_df.index = range(1, 11)
    st.dataframe(top10_df, use_container_width=True, height=310)
