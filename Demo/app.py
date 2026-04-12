"""
Demo/app.py

FLAIR — Introduction & Project Overview
Static landing page. No inference is run here.

Run from the project root:
    streamlit run Demo/app.py
Then click "1 Architecture Explainer" in the sidebar to interact with the model.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FLAIR — Introduction",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("F.L.A.I.R.")
st.subheader("Flow-Level Autoencoder for Intrusion Recognition")
st.caption("University of Arkansas · RIOT Lab (Secure and Trustworthy Robotics and Embedded Systems)")
st.divider()

# ---------------------------------------------------------------------------
# Section 1 — The Problem
# ---------------------------------------------------------------------------
st.header("The Problem")

col_prob, col_stat = st.columns([2, 1])

with col_prob:
    st.markdown(
        "Industrial Control Systems (ICS) and Industrial Internet-of-Things (IIoT) "
        "devices — sensors, PLCs, RTUs, and actuators — form the backbone of critical "
        "infrastructure: power grids, water treatment, manufacturing, and nuclear "
        "facilities. Unlike enterprise IT networks, these environments prioritize "
        "**availability and determinism** over confidentiality, leaving them "
        "increasingly exposed as they connect to broader networks.\n\n"
        "Cyberattacks targeting ICS environments have grown in both frequency and "
        "sophistication. Detecting these attacks is uniquely challenging because:\n\n"
        "- **Normal ICS traffic is highly repetitive** — devices communicate in fixed "
        "polling cycles with predictable patterns.\n"
        "- **Labeled attack data is scarce** — supervised methods require large "
        "labeled datasets, which are difficult to obtain in operational environments.\n"
        "- **Attack patterns are diverse** — from high-volume DoS floods to subtle "
        "backdoor communication that mimics normal traffic.\n"
        "- **False alarms are costly** — unnecessary shutdowns of industrial processes "
        "can be as damaging as the attacks themselves."
    )

with col_stat:
    st.info(
        "**WUSTL-IIoT-2021 Dataset**\n\n"
        "University of Washington St. Louis\n\n"
        "~1.19 million network flow windows\n\n"
        "4 attack types:\n"
        "- Denial of Service (DoS)\n"
        "- Reconnaissance\n"
        "- Command Injection\n"
        "- Backdoor\n\n"
        "Natural attack rate: **7.28%**"
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — The Approach
# ---------------------------------------------------------------------------
st.header("Our Approach")

st.markdown(
    "FLAIR uses an **unsupervised anomaly detection** strategy based on a "
    "GRU autoencoder. The key insight is that ICS traffic, when operating normally, "
    "occupies a compact, learnable region of feature space. An autoencoder trained "
    "exclusively on normal traffic learns to reconstruct that region accurately — "
    "but fails to reconstruct anomalous traffic it has never seen.\n\n"
    "**No attack labels are needed for training or threshold selection.** "
    "This makes FLAIR deployable in environments where labeled attack data is "
    "unavailable or unrepresentative of future threats."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 1 · Capture")
    st.markdown(
        "Raw network traffic is parsed into **flow-level records** — one row per "
        "connection. Flows capture packet counts, byte volumes, timing, and "
        "protocol information without needing packet-level content."
    )

with col2:
    st.markdown("### 2 · Window")
    st.markdown(
        "Flows are sorted by time and grouped into **sliding windows of 10 consecutive "
        "flows**. Each window is the model's unit of analysis, capturing short-range "
        "temporal context."
    )

with col3:
    st.markdown("### 3 · Encode")
    st.markdown(
        "A **GRU encoder** compresses the 10-flow window into a 128-dimensional "
        "latent vector. Categorical features (ports, protocol) pass through learned "
        "embeddings before fusion with numeric features."
    )

with col4:
    st.markdown("### 4 · Score")
    st.markdown(
        "A **GRU decoder** reconstructs the original window from the latent vector. "
        "The mean squared reconstruction error is the **anomaly score** — high error "
        "means the window looks unlike anything seen in normal training traffic."
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Features
# ---------------------------------------------------------------------------
st.header("Input Features")

col_cat, col_num = st.columns([1, 2])

with col_cat:
    st.markdown("**Categorical Features** (embedded)")
    st.markdown(
        "| Feature | Description |\n"
        "|---------|-------------|\n"
        "| Sport   | Source port number |\n"
        "| Dport   | Destination port number |\n"
        "| Proto   | Transport protocol (TCP/UDP/ICMP/…) |"
    )
    st.caption(
        "Mapped to learned 8-dimensional embedding vectors rather than raw integers, "
        "allowing the model to capture semantic similarity between related ports and protocols."
    )

with col_num:
    st.markdown("**Numeric Features** (z-score normalized)")
    feat_cols = st.columns(3)
    numeric_features = [
        ("Mean", "Mean packet size"),
        ("SrcPkts", "Source packet count"),
        ("DstPkts", "Destination packet count"),
        ("TotPkts", "Total packet count"),
        ("SrcBytes", "Source byte count"),
        ("DstBytes", "Destination byte count"),
        ("TotBytes", "Total byte count"),
        ("SrcLoad", "Source load (bps)"),
        ("DstLoad", "Destination load (bps)"),
        ("Load", "Total load (bps)"),
        ("SrcRate", "Source packet rate (pps)"),
        ("DstRate", "Destination packet rate (pps)"),
        ("Rate", "Total packet rate (pps)"),
        ("SrcLoss", "Source packet loss count"),
        ("DstLoss", "Destination packet loss count"),
        ("Loss", "Total packet loss count"),
        ("pLoss", "Packet loss rate (%)"),
        ("SrcJitter", "Source inter-arrival jitter"),
        ("DstJitter", "Destination inter-arrival jitter"),
        ("SIntPkt", "Source inter-packet time"),
        ("DIntPkt", "Destination inter-packet time"),
    ]
    for i, (name, desc) in enumerate(numeric_features):
        feat_cols[i % 3].markdown(f"**{name}** — {desc}")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Results
# ---------------------------------------------------------------------------
st.header("Results — 80/10/10 Temporal Split")
st.caption(
    "Trained on 80% of the dataset (normal windows only). "
    "Threshold set on 10% validation set (p99 of normal scores). "
    "Evaluated on 114,957 held-out test windows."
)

col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
col_m1.metric("Accuracy",     "98.96%")
col_m2.metric("F1 Score",     "93.20%")
col_m3.metric("Recall",       "98.08%",  help="Fraction of attacks correctly detected")
col_m4.metric("Precision",    "88.79%")
col_m5.metric("ROC AUC",      "0.9994",  help="Threshold-independent discrimination (1.0 = perfect)")
col_m6.metric("False Alarm",  "0.97%",   help="Fraction of normal windows incorrectly flagged")

st.markdown("")

col_dos, col_rec, col_ci, col_bd = st.columns(4)
with col_dos:
    st.success("**DoS Detection**\n\n# 99.5%\n7,491 / 7,530 attacks detected")
with col_rec:
    st.warning("**Reconn Detection**\n\n# 86.2%\n683 / 792 attacks detected")
with col_ci:
    st.warning("**Comm. Injection**\n\n# 80.8%\n21 / 26 attacks detected")
with col_bd:
    st.error("**Backdoor Detection**\n\n# 61.9%\n13 / 21 attacks detected")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — How This Demo Works
# ---------------------------------------------------------------------------
st.header("How This Demo Works")

st.markdown(
    "👈 **Click \"1 Architecture Explainer\" in the sidebar** to begin the interactive walkthrough."
)

st.info(
    "**This demo is an interactive replay system — it is not processing live network traffic.**\n\n"
    "The FLAIR model was trained and evaluated on the WUSTL-IIoT-2021 dataset on a "
    "high-performance computing cluster. Running the full evaluation (scoring all ~1.19M "
    "windows) on a laptop CPU would take 30–60 minutes. To make the demo fast and "
    "interactive, the anomaly scores were pre-computed during the evaluation run and "
    "saved to `scores_80_10_10.csv`. The demo loads those scores at startup in seconds."
)

col_how1, col_how2, col_how3 = st.columns(3)

with col_how1:
    st.markdown("#### What is Pre-computed")
    st.markdown(
        "The **anomaly score** displayed for each window comes directly from "
        "`scores_80_10_10.csv`, generated by running `flair_80_10_10.pt` over the "
        "entire test set during the offline evaluation. The **threshold (0.3226)** "
        "was set at the 99th percentile of validation-set scores during that same "
        "run — no attack labels were used."
    )

with col_how2:
    st.markdown("#### What is Computed Live")
    st.markdown(
        "Every time you move the slider, the selected window is fed through a **real "
        "forward pass** of the model. The latent vector, decoder reconstruction, "
        "per-feature error, and feature fusion diagram are all generated live. "
        "You are seeing genuine model inference — just with the final anomaly score "
        "sourced from the pre-computed evaluation for consistency with published metrics."
    )

with col_how3:
    st.markdown("#### How to Explore")
    st.markdown(
        "Use the **Filter by class** control to browse normal-only or attack-only "
        "windows. When browsing attacks, the sidebar shows the **attack type** "
        "(DoS, Reconn, CommInj, Backdoor). Expand the **'How to read this'** "
        "sections on the Architecture Explainer page for guidance on interpreting "
        "each visualization."
    )
