"""
Demo/pages/1_Architecture_Explainer.py

FLAIR Pipeline Walkthrough — Architecture Explainer
Walks an audience through each stage of the FLAIR anomaly detection pipeline
using live inference on the 80/10/10 test split.

Run from project root:
    streamlit run Demo/app.py
Then click "1 Architecture Explainer" in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root (FLAIR/) so src.* imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add Demo/ so inference and visualizations resolve without package prefix
sys.path.insert(0, str(Path(__file__).parent.parent))

import inference
from visualizations import (
    input_heatmap,
    latent_bar,
    reconstruction_comparison,
    anomaly_gauge,
    embedding_fusion_diagram,
    per_attack_bar,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FLAIR — Architecture Explainer",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load everything once (shared cache with app.py)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FLAIR model and data...")
def load_everything():
    inference.ensure_loaded()
    return True

load_everything()

# ---------------------------------------------------------------------------
# Sidebar — window selector scoped to 80/10/10 test split
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("F.L.A.I.R.")
    st.caption("Architecture Explainer · 80/10/10 Eval")
    st.divider()

    filter_mode = st.radio(
        "Filter by class",
        options=["All", "Normal", "Attack"],
        index=0,
        help="Restrict the slider to normal-only, attack-only, or all test windows.",
    )

    available_idx = inference.get_indices(filter_mode)
    n_available   = len(available_idx)

    if n_available == 0:
        st.error("No windows match the selected filter.")
        st.stop()

    slider_pos = st.slider(
        f"Window position ({filter_mode}, {n_available:,} total)",
        min_value=0,
        max_value=n_available - 1,
        value=0,
        step=1,
    )
    window_idx = int(available_idx[slider_pos])

    st.divider()

    result     = inference.run_inference(window_idx)
    score      = result["anomaly_score"]
    threshold  = result["threshold"]
    is_attack  = result["is_attack"]
    gt         = result["ground_truth"]
    wtype      = result["window_type"]

    gt_label    = "ATTACK" if gt == 1 else "NORMAL"
    gt_color    = "red"    if gt == 1 else "green"
    pred_label  = "ATTACK" if is_attack else "NORMAL"
    pred_color  = "red"    if is_attack else "green"
    correct     = is_attack == bool(gt)

    st.markdown(f"**Ground Truth:** :{gt_color}[{gt_label}]")
    if gt == 1:
        st.markdown(f"**Attack Type:** `{wtype}`")
    st.markdown(f"**Prediction:**   :{pred_color}[{pred_label}]")
    st.markdown(f"**Correct:** {'✓' if correct else '✗'}")
    st.metric("Anomaly Score",   f"{score:.6f}")
    st.metric("Threshold (p99)", f"{threshold:.6f}")
    st.metric("Window Index",    f"{window_idx:,}")

    st.divider()

    all_labels = inference.get_labels()
    test_idx   = inference.get_test_indices()
    n_test     = len(test_idx)
    n_atk_test = int((all_labels == 1).sum())
    st.caption(
        f"Test set: {n_test:,} windows · "
        f"{n_atk_test:,} attack ({n_atk_test/n_test*100:.1f}%)"
    )

# ---------------------------------------------------------------------------
# Stage 0 — Decision banner
# ---------------------------------------------------------------------------
if is_attack:
    st.error(f"🚨  ATTACK DETECTED   ·   Score: {score:.6f}   ·   Threshold: {threshold:.6f}")
else:
    st.success(f"✅  NORMAL TRAFFIC   ·   Score: {score:.6f}   ·   Threshold: {threshold:.6f}")

attack_note = f"  ·  Attack type: **{wtype}**" if gt == 1 else ""
st.caption(
    f"Window {window_idx:,}  ·  Ground truth: **{gt_label}**{attack_note}  ·  "
    f"Prediction: **{'correct ✓' if correct else 'incorrect ✗'}**"
)

st.divider()

# ===========================================================================
# STAGE 1 — INPUT WINDOW
# ===========================================================================
st.subheader("Stage 1 — Input Window")
st.markdown(
    "FLAIR receives a **sliding window of 10 consecutive network flows** sorted by "
    "timestamp. Each flow contributes **21 numeric features** (traffic statistics such "
    "as packet counts, byte counts, load, rate, jitter, and inter-packet timing) and "
    "**3 categorical fields** (source port, destination port, protocol). The window "
    "size of 10 was chosen to capture short-range temporal patterns typical of ICS "
    "polling cycles while keeping inference latency low."
)

col_heat, col_cat = st.columns([3, 1])
with col_heat:
    st.plotly_chart(
        input_heatmap(result["x_num_raw"], result["num_features"]),
        width='stretch',
    )
with col_cat:
    st.markdown("**Categorical Features**")
    cat_df = pd.DataFrame(result["cat_decoded"])
    cat_df.index = [f"Flow {i+1}" for i in range(len(cat_df))]
    st.dataframe(cat_df, width='stretch', height=270)

col_exp1, col_exp2 = st.columns(2)
with col_exp1:
    with st.expander("How to read the input heatmap"):
        st.markdown(
            "- **Rows** = individual network flows, ordered from oldest (Flow 1) to "
            "most recent (Flow 10).\n"
            "- **Columns** = numeric features.\n"
            "- **Color** = z-score distance from the normal-traffic mean: "
            "blue → below average, white → near average, red → above average.\n"
            "- A cluster of deep **red** in byte/packet columns (SrcBytes, TotBytes, "
            "Load, Rate…) signals a volume spike — a classic signature of DoS attacks.\n"
            "- Z-score normalization uses statistics computed from **normal traffic only**, "
            "so even moderate anomalies show up visually."
        )
with col_exp2:
    with st.expander("How to read the categorical table"):
        st.markdown(
            "- **Sport / Dport** = source and destination port numbers.\n"
            "- **Proto** = transport-layer protocol (tcp, udp, icmp, …).\n"
            "- In normal ICS traffic these fields are highly repetitive — devices "
            "communicate on fixed, well-known ports with predictable protocols.\n"
            "- During **reconnaissance** attacks you may see many different destination "
            "ports appearing across flows as the attacker scans the network.\n"
            "- During **command injection**, unusual protocols or unexpected port "
            "combinations may appear."
        )

st.divider()

# ===========================================================================
# STAGE 2 — FEATURE EMBEDDING & FUSION
# ===========================================================================
st.subheader("Stage 2 — Feature Embedding & Fusion")
st.markdown(
    "The 3 categorical fields cannot be fed directly to the GRU as raw integers — "
    "port 80 is not numerically *similar* to port 443 in any meaningful sense. "
    "Instead, each field is passed through a **learned embedding layer** that maps it "
    "to an **8-dimensional dense vector**. During training the model learns that "
    "semantically related ports cluster together in embedding space. "
    "The embedded vectors are then **concatenated with the 21 numeric features** to "
    "form a **45-dimensional input vector per flow** (21 + 3 × 8 = 45), which is what "
    "the GRU encoder actually processes."
)

st.plotly_chart(
    embedding_fusion_diagram(result["fused_input"], result["num_features"]),
    width='stretch',
)

with st.expander("How to read the fusion diagram"):
    st.markdown(
        "- The chart shows the **average activation across the 10 flows** for all 45 "
        "input dimensions fed to the GRU.\n"
        "- **Gray bars (dims 0–20)** = z-score normalized numeric features. Values near "
        "zero indicate the feature is close to its normal-traffic average.\n"
        "- **Blue bars (dims 21–28)** = 8 learned dimensions for the source port "
        "(Sport) embedding.\n"
        "- **Green bars (dims 29–36)** = 8 learned dimensions for the destination port "
        "(Dport) embedding.\n"
        "- **Orange bars (dims 37–44)** = 8 learned dimensions for the protocol "
        "(Proto) embedding.\n"
        "- The embedding values have no fixed units — their absolute magnitude matters "
        "less than *how they change* between normal and attack windows."
    )

st.divider()

# ===========================================================================
# STAGE 3 — GRU ENCODER → LATENT VECTOR
# ===========================================================================
st.subheader("Stage 3 — GRU Encoder → Latent Vector")
st.markdown(
    "The 45-dimensional per-flow vectors for all 10 timesteps are processed "
    "**sequentially** by a single-layer **Gated Recurrent Unit (GRU)**. The GRU "
    "maintains a hidden state that is updated at each timestep, allowing it to capture "
    "temporal dependencies across flows. After the final flow, its hidden state — a "
    "**128-dimensional latent vector** — is the encoder output. "
    "This vector is a compressed representation of the entire 10-flow window. Because "
    "the autoencoder was trained exclusively on normal traffic, the encoder has learned "
    "to efficiently represent the space of normal ICS communication patterns."
)

st.plotly_chart(
    latent_bar(result["latent"]),
    width='stretch',
)

with st.expander("How to read the latent vector"):
    st.markdown(
        "- Each of the **128 bars** represents one dimension of the compressed "
        "representation learned by the encoder.\n"
        "- **Red bars** = positive activation, **blue bars** = negative activation.\n"
        "- The dimensions have no explicit semantic meaning — they are jointly "
        "optimized during training to minimize reconstruction error.\n"
        "- Try switching between a **Normal** and an **Attack** window: you will notice "
        "the overall activation pattern and magnitude change noticeably. The encoder "
        "has no learned basis for representing anomalous traffic, so unusual windows "
        "produce atypical latent patterns.\n"
        "- Large magnitude activations (tall bars, either direction) are not inherently "
        "good or bad — what matters is whether the *decoder* can use them to produce "
        "a low-error reconstruction."
    )

st.divider()

# ===========================================================================
# STAGE 4 — GRU DECODER → RECONSTRUCTION
# ===========================================================================
st.subheader("Stage 4 — GRU Decoder → Reconstruction")
st.markdown(
    "The decoder GRU receives the **128-dimensional latent vector** and expands it "
    "back into a 10-flow sequence, attempting to reconstruct the original input. "
    "The decoder has **two output heads**: a *numeric head* that reproduces the 21 "
    "feature values per flow, and *categorical heads* that predict the most likely "
    "source port, destination port, and protocol at each timestep. "
    "Because the model was trained only on normal traffic, it can only reconstruct "
    "what **normal looks like** — traffic that deviates from learned patterns produces "
    "a high-error reconstruction."
)

st.plotly_chart(
    reconstruction_comparison(
        result["x_num_raw"],
        result["x_hat_num"],
        result["per_feat_err"],
        result["num_features"],
    ),
    width='stretch',
)

with st.expander("How to read the reconstruction comparison"):
    st.markdown(
        "- **Top-left heatmap** = the original normalized input (same as Stage 1).\n"
        "- **Top-right heatmap** = the model's reconstruction of that input. The two "
        "heatmaps share the same color scale, so differences are directly visible as "
        "color mismatches.\n"
        "- **Bottom bar chart** = per-feature reconstruction error (MSE averaged over "
        "the 10 flows):\n"
        "  - 🟢 **Green** = low error — the model reconstructed this feature accurately.\n"
        "  - 🟡 **Yellow** = moderate error.\n"
        "  - 🔴 **Red** = high error — the model struggled to reconstruct this feature, "
        "indicating it deviated significantly from normal patterns.\n"
        "- During a **DoS attack**, expect red bars on byte/packet features (TotBytes, "
        "SrcBytes, Load, Rate) because the traffic volume far exceeds anything seen "
        "during training.\n"
        "- During **reconnaissance**, you may see elevated error on port-adjacent "
        "features and rate metrics."
    )

st.divider()

# ===========================================================================
# STAGE 5 — ANOMALY SCORING & DECISION
# ===========================================================================
st.subheader("Stage 5 — Anomaly Scoring & Decision")
st.markdown(
    "The **anomaly score** is a weighted combination of two reconstruction error terms — "
    "a single scalar per window:\n\n"
    "1. **Numeric MSE** — mean squared reconstruction error across all 10 flows and all "
    "21 numeric features (packet counts, byte volumes, load, rate, jitter, etc.).\n"
    "2. **Categorical cross-entropy** — average cross-entropy loss for the three categorical "
    "heads (source port, destination port, protocol), measuring how well the decoder "
    "predicted the correct port/protocol at each timestep. This term is weighted at "
    "0.1 × avg(CE\\_sport + CE\\_dport + CE\\_proto).\n\n"
    "The final score is: **MSE\\_numeric + 0.1 × avg\\_categorical\\_CE**. "
    "Unusual port patterns (e.g., a port scanner hitting many distinct destination ports) "
    "raise the categorical term even when numeric traffic statistics appear normal.\n\n"
    "The detection **threshold** was set at the **99th percentile** of anomaly scores "
    "computed on the validation set, which contains only normal traffic. "
    "No attack labels were used to set this threshold — it is entirely unsupervised. "
    "Any window whose score exceeds the threshold is flagged as an attack."
)

col_gauge, col_top10 = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(
        anomaly_gauge(score, threshold),
        width='stretch',
    )

with col_top10:
    st.markdown("**Top-10 Most Anomalous Features**")
    feat_err   = result["per_feat_err"]
    feat_names = result["num_features"]
    top10_idx  = np.argsort(feat_err)[::-1][:10]
    top10_df   = pd.DataFrame({
        "Feature": [feat_names[i] for i in top10_idx],
        "MSE":     [f"{feat_err[i]:.6f}" for i in top10_idx],
    })
    top10_df.index = range(1, 11)
    st.dataframe(top10_df, width='stretch', height=320)

col_exp_g, col_exp_t = st.columns(2)
with col_exp_g:
    with st.expander("How to read the anomaly gauge"):
        st.markdown(
            f"- The **needle** shows the current window's anomaly score ({score:.4f}).\n"
            f"- The **red threshold line** marks the decision boundary "
            f"({threshold:.4f}).\n"
            "- **Green zone** (below threshold) = scores consistent with normal "
            "traffic.\n"
            "- **Red zone** (above threshold) = the model considers this window an "
            "attack.\n"
            "- The **delta** (±) shows how far the score is from the threshold. A "
            "delta of +0.5 means the score is 0.5 units above the threshold.\n"
            "- The gauge's upper limit scales dynamically so the green zone is always "
            "clearly visible."
        )
with col_exp_t:
    with st.expander("How to read the top-10 features table"):
        st.markdown(
            "- Features are ranked by their individual MSE (highest error first).\n"
            "- The feature at rank 1 is the one where the model's reconstruction was "
            "furthest from the actual input — i.e., the most anomalous aspect of this "
            "window.\n"
            "- This table provides **interpretability**: instead of just knowing that "
            "a window is anomalous, you can see *what* was anomalous.\n"
            "- Example: if TotBytes, SrcLoad, and Rate dominate the top-10, the "
            "anomaly is driven by abnormally high traffic volume — a strong indicator "
            "of a DoS attack."
        )

st.divider()

# ===========================================================================
# STAGE 6 — MODEL PERFORMANCE (80/10/10 SPLIT) — static section
# ===========================================================================
st.subheader("Stage 6 — Model Performance (80/10/10 Temporal Split)")
st.caption(
    "Training set: 80% of dataset (normal windows only) · "
    "Validation set: 10% (normal only, used to set threshold) · "
    "Test set: 114,957 windows (normal + all attack types) · "
    "Threshold: 99th percentile of validation anomaly scores"
)

col_metrics, col_atk_bar = st.columns([1, 1])

with col_metrics:
    st.markdown("**Operational Threshold Metrics**")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy",  "98.96%")
    m2.metric("F1 Score",  "93.20%")
    m3, m4 = st.columns(2)
    m3.metric("Recall (TPR)", "98.08%",  help="Fraction of attacks correctly detected")
    m4.metric("Precision",    "88.79%",  help="Fraction of flagged windows that are true attacks")
    m5, m6 = st.columns(2)
    m5.metric("ROC AUC", "0.9994", help="Threshold-independent discrimination ability (1.0 = perfect)")
    m6.metric("FPR",     "0.97%",  help="Fraction of normal windows falsely flagged as attacks")

    st.markdown("**Confusion Matrix (Test Set)**")
    cm_df = pd.DataFrame(
        [["TP = 8,208", "FN = 161"], ["FP = 1,036", "TN = 105,552"]],
        index=["Predicted: Attack", "Predicted: Normal"],
        columns=["Actual: Attack", "Actual: Normal"],
    )
    st.dataframe(cm_df, width='stretch')

with col_atk_bar:
    detection_rates = {
        "DoS":      99.5,
        "Reconn":   86.2,
        "CommInj":  80.8,
        "Backdoor": 61.9,
    }
    st.plotly_chart(
        per_attack_bar(detection_rates),
        width='stretch',
    )

with st.expander("How to read these metrics"):
    st.markdown(
        "**Operational threshold metrics** are computed at the fixed 99th-percentile "
        "threshold — this is what the system would achieve in deployment.\n\n"
        "- **Recall (98.08%)**: Of all true attack windows in the test set, the model "
        "correctly flagged 98.08% of them. The remaining 1.92% are *false negatives* "
        "(missed attacks).\n"
        "- **FPR (0.97%)**: Of all truly normal windows, only 0.97% were incorrectly "
        "flagged as attacks. In a network carrying thousands of flows per second, this "
        "is a very low false-alarm rate.\n"
        "- **ROC AUC (0.9994)**: Threshold-independent — measures the model's ability "
        "to *rank* attack windows above normal ones regardless of where the threshold "
        "is set. A value of 0.9994 is near-perfect.\n\n"
        "**Per-attack-type breakdown:**\n"
        "- 🟢 **DoS (99.5%)**: Denial-of-service attacks flood the network with "
        "traffic, creating extreme feature values that are far outside the normal "
        "distribution — easy for the autoencoder to detect.\n"
        "- 🟡 **Reconn (86.2%)**: Reconnaissance (port scanning) generates unusual "
        "port patterns and elevated rates, but at lower volume than DoS.\n"
        "- 🟡 **CommInj (80.8%)**: Command injection attacks may subtly alter packet "
        "content while keeping traffic volume roughly normal.\n"
        "- 🔴 **Backdoor (61.9%)**: The hardest class — backdoor communication often "
        "mimics normal traffic patterns closely, producing reconstruction errors that "
        "are elevated but may fall below the threshold."
    )
