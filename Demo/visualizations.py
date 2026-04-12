"""
Demo/visualizations.py

Pure Plotly chart builders — no Streamlit imports.
All functions return a plotly.graph_objects.Figure.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def input_heatmap(x_num: np.ndarray, feature_names: List[str]) -> go.Figure:
    """
    Heatmap of a single input window.
    x_num: (10, 21) — normalized numeric features
    Rows = timesteps (flows), cols = features.
    """
    fig = go.Figure(
        go.Heatmap(
            z=x_num,
            x=feature_names,
            y=[f"Flow {i+1}" for i in range(x_num.shape[0])],
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="z-score"),
        )
    )
    fig.update_layout(
        title="Input Window — Normalized Numeric Features",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=60, r=20, t=40, b=80),
    )
    return fig


def latent_bar(latent: np.ndarray) -> go.Figure:
    """
    Bar chart of the 128-dim encoder latent vector.
    latent: (128,)
    """
    dims = [f"{i}" for i in range(len(latent))]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in latent]

    fig = go.Figure(
        go.Bar(
            x=dims,
            y=latent,
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Encoder Output — Latent Vector (128-dim)",
        xaxis=dict(title="Dimension", tickvals=list(range(0, 128, 16)),
                   ticktext=[str(i) for i in range(0, 128, 16)]),
        yaxis=dict(title="Activation"),
        height=220,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False,
    )
    return fig


def reconstruction_comparison(
    x_num: np.ndarray,
    x_hat_num: np.ndarray,
    per_feat_err: np.ndarray,
    feature_names: List[str],
) -> go.Figure:
    """
    Two side-by-side heatmaps (Original | Reconstructed) plus a per-feature error bar chart.
    x_num, x_hat_num: (10, 21)
    per_feat_err: (21,)
    """
    flow_labels = [f"Flow {i+1}" for i in range(x_num.shape[0])]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original", "Reconstructed", "", "Per-Feature Reconstruction Error"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "bar", "colspan": 2}, None]],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    shared_zmin = float(min(x_num.min(), x_hat_num.min()))
    shared_zmax = float(max(x_num.max(), x_hat_num.max()))

    fig.add_trace(
        go.Heatmap(
            z=x_num,
            x=feature_names,
            y=flow_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=shared_zmin,
            zmax=shared_zmax,
            showscale=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=x_hat_num,
            x=feature_names,
            y=flow_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=shared_zmin,
            zmax=shared_zmax,
            colorbar=dict(title="z-score", len=0.5, y=0.75),
        ),
        row=1, col=2,
    )

    err_colors = [
        "#e74c3c" if e > np.percentile(per_feat_err, 75) else "#f39c12"
        if e > np.percentile(per_feat_err, 50) else "#2ecc71"
        for e in per_feat_err
    ]
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=per_feat_err,
            marker_color=err_colors,
            name="MSE per feature",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Decoder Reconstruction vs Original",
        height=520,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="MSE", row=2, col=1)

    return fig


def anomaly_gauge(score: float, threshold: float) -> go.Figure:
    """
    Gauge chart with needle at score.
    Green zone = below threshold, red zone = above.
    """
    max_val = max(score * 1.5, threshold * 2.0, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": threshold, "valueformat": ".4f"},
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, max_val], "tickformat": ".3f"},
                "bar": {"color": "#e74c3c" if score > threshold else "#2ecc71", "thickness": 0.3},
                "steps": [
                    {"range": [0, threshold], "color": "#d5f5e3"},
                    {"range": [threshold, max_val], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#c0392b", "width": 3},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
            title={"text": f"Anomaly Score<br><span style='font-size:0.8em'>Threshold: {threshold:.4f}</span>"},
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def embedding_fusion_diagram(fused_input: np.ndarray, num_feature_names: List[str]) -> go.Figure:
    """
    Bar chart of the 45-dim fused GRU input (averaged over 10 timesteps).

    fused_input: (10, 45) — output of model._combine_inputs()
    Layout of the 45 dims: [0..20] numeric | [21..28] Sport emb | [29..36] Dport emb | [37..44] Proto emb
    """
    avg = fused_input.mean(axis=0)  # (45,)

    labels = num_feature_names[:21] + [f"Sport-{i}" for i in range(8)] + \
             [f"Dport-{i}" for i in range(8)] + [f"Proto-{i}" for i in range(8)]

    colors = (
        ["#7f8c8d"] * 21 +   # numeric — gray
        ["#2980b9"] * 8  +   # Sport emb — blue
        ["#27ae60"] * 8  +   # Dport emb — green
        ["#e67e22"] * 8      # Proto emb — orange
    )

    fig = go.Figure(
        go.Bar(
            x=list(range(45)),
            y=avg,
            marker_color=colors,
            hovertext=labels,
            hovertemplate="%{hovertext}: %{y:.4f}<extra></extra>",
        )
    )

    for x_pos in [20.5, 28.5, 36.5]:
        fig.add_vline(x=x_pos, line_width=1.5, line_dash="dash", line_color="#bdc3c7")

    for x_center, txt, color in [
        (10,   "Numeric (21)", "#7f8c8d"),
        (24.5, "Sport (8)",    "#2980b9"),
        (32.5, "Dport (8)",    "#27ae60"),
        (40.5, "Proto (8)",    "#e67e22"),
    ]:
        fig.add_annotation(
            x=x_center, y=1.08, xref="x", yref="paper",
            text=f"<b>{txt}</b>", showarrow=False,
            font=dict(size=10, color=color),
        )

    fig.update_layout(
        title="Feature Fusion — 45-Dim GRU Input (averaged over 10 flows)",
        xaxis=dict(title="Dimension index", tickvals=list(range(0, 45, 5)),
                   ticktext=[str(i) for i in range(0, 45, 5)]),
        yaxis=dict(title="Activation (z-score / embedding units)"),
        height=260,
        margin=dict(l=50, r=20, t=70, b=50),
        showlegend=False,
    )
    return fig


def per_attack_bar(detection_rates: Dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart of per-attack-type detection rates (80/10/10 split).

    detection_rates: {'DoS': 99.5, 'Reconn': 86.2, 'CommInj': 80.8, 'Backdoor': 61.9}
    """
    attack_types = list(detection_rates.keys())
    rates        = list(detection_rates.values())

    bar_colors = []
    for r in rates:
        if r >= 95:
            bar_colors.append("#2ecc71")
        elif r >= 80:
            bar_colors.append("#f39c12")
        else:
            bar_colors.append("#e74c3c")

    fig = go.Figure(
        go.Bar(
            x=rates,
            y=attack_types,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{r:.1f}%" for r in rates],
            textposition="outside",
        )
    )

    for x_val, label, color in [(80, "80%", "#f39c12"), (95, "95%", "#2ecc71")]:
        fig.add_vline(x=x_val, line_width=1.5, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(color=color, size=11))

    fig.update_layout(
        title="Per-Attack-Type Detection Rate (80/10/10 Test Set)",
        xaxis=dict(title="Detection Rate (%)", range=[0, 110]),
        yaxis=dict(title="Attack Type", autorange="reversed"),
        height=260,
        margin=dict(l=90, r=60, t=50, b=50),
        showlegend=False,
    )
    return fig
