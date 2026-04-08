# F.L.A.I.R — Flow-Level Autoencoder for Intrusion Recognition

FLAIR is an unsupervised anomaly detection system for Industrial IoT (IIoT) network traffic. It uses a GRU-based autoencoder trained exclusively on normal traffic to detect intrusions by measuring how poorly it reconstructs a sliding window of network flows. No attack labels are used during training.

This project was developed as an undergraduate honors thesis at the University of Arkansas, evaluated on the WUSTL-IIoT-2021 dataset.

---

## Results (Primary Split: 80/10/10)

| Metric | Operational Threshold | Best-F1 Threshold (Upper Bound) |
|--------|-----------------------|----------------------------------|
| Accuracy | 98.96% | — |
| Precision | 88.79% | 97.02% |
| Recall (TPR) | 98.08% | 95.01% |
| F1 Score | 93.20% | 96.00% |
| FPR | 0.97% | — |
| ROC AUC | 0.9994 | — |
| PR AUC | 0.9932 | — |

**Per-attack-type detection @ operational threshold:**

| Attack Type | Detected | Total | Rate |
|-------------|----------|-------|------|
| DoS | 7,491 | 7,530 | 99.5% |
| Reconnaissance | 683 | 792 | 86.2% |
| Command Injection | 21 | 26 | 80.8% |
| Backdoor | 13 | 21 | 61.9% |

Threshold set at the 99th percentile of normal-window anomaly scores from the validation set (no attack labels used).

### Cross-Split Robustness

| Split | F1 | ROC AUC | PR AUC |
|-------|----|---------|--------|
| 80/10/10 | 93.20% | 0.9994 | 0.9932 |
| 70/15/15 | 92.68% | 0.9991 | 0.9877 |
| 60/20/20 | 92.53% | 0.9992 | 0.9904 |

F1 never drops below 92.5% and ROC AUC never below 0.999 across all splits, demonstrating that FLAIR is robust to the choice of train/test split ratio.

---

## Dataset

**WUSTL-IIoT-2021** — a network flow dataset collected from an industrial IoT testbed.

| Stat | Value |
|------|-------|
| Total flows | 1,194,464 |
| Total windows (T=10, stride=1) | 1,194,455 |
| Normal windows | 1,065,874 (89.2%) |
| Attack windows | 128,581 (10.8%) |
| Attack rate (flow-level) | 7.28% |

Attack type distribution:

| Type | Share of attacks |
|------|-----------------|
| DoS | 89.98% |
| Reconnaissance | 9.46% |
| Command Injection | 0.31% |
| Backdoor | 0.25% |

The dataset is not included in this repository. Download it separately and set the path in `config.yaml`.

---

## Repository Structure

```
FLAIR/
├── config.yaml                          # All settings — paths, hyperparameters, features
├── requirements.txt                     # Python dependencies
│
├── scripts/
│   ├── preprocess_data.py               # Step 1: raw CSV → preprocessed.npz
│   ├── extract_window_types.py          # Extract per-window attack type labels
│   ├── create_splits_retrain.py         # Build 80/10/10, 70/15/15, 60/20/20 split indices
│   ├── train_split.py                   # Train on a pre-computed split
│   ├── evaluate_split.py                # Evaluate on a split's held-out test set
│   ├── create_splits.py                 # Temporal holdout split (no retraining)
│   ├── evaluate_holdout.py              # Re-report metrics on held-out 10% test set
│   ├── export_onnx.py                   # (experimental) Export trained model to ONNX
│   └── infer_realtime.py                # (experimental) Real-time inference — incomplete
│
├── src/
│   ├── data/
│   │   ├── feature_definitions.py       # Feature name lists (3 categorical + 21 numeric)
│   │   ├── dataset.py                   # PyTorch FLAIRDataset
│   │   └── flow_window_builder.py       # Sliding-window construction helpers
│   │
│   ├── models/
│   │   ├── flair_model.py               # FLAIRAutoencoder + FLAIRConfig
│   │   ├── encoder.py                   # GRUEncoder
│   │   └── decoder.py                   # GRUDecoder with categorical output heads
│   │
│   └── training/
│       ├── train_flair.py               # Training loop with early stopping
│       └── evaluate_flair.py            # Metrics: F1, ROC AUC, PR AUC, confusion matrix
│
├── demo/
│   ├── app.py                           # Streamlit live demo GUI
│   ├── inference.py                     # Model loader + per-window inference
│   ├── visualizations.py                # Plotly chart builders
│   └── requirements_demo.txt            # Demo-specific dependencies
│
├── data/
│   └── processed/                       # Generated files (not committed)
│       ├── preprocessed.npz
│       ├── window_types.npy
│       └── splits_*.npz
│
└── experiments/
    └── results/                         # Generated files (not committed)
        ├── flair_minimal.pt             # Original model checkpoint
        ├── flair_80_10_10.pt            # Split-specific checkpoints
        ├── flair_70_15_15.pt
        ├── flair_60_20_20.pt
        ├── anomaly_scores_full.csv      # All-window scores for demo
        ├── scores_*.csv                 # Per-split test set scores
        └── metrics_*.txt                # Per-split evaluation reports
```

---

## Model Architecture

```
Input: x_num (B, T, 21)  +  x_cat (B, T, 3)
            │                       │
            │         ┌─────────────┴──────────────┐
            │    Embed(Sport, 8D)  Embed(Dport, 8D)  Embed(Proto, 8D)
            │         └─────────────┬──────────────┘
            └──────────────────────►│
                            concat → x_in (B, T, 45)
                                     │
                              GRU Encoder
                                     │
                              latent (B, 128)
                                     │
                              GRU Decoder
                                     │
                    ┌────────────────┼────────────────┐
               x_hat_num         sport_logits    dport_logits
               (B, T, 21)        (B, T, 51058)  (B, T, 7782)   proto_logits (B, T, 9)

Anomaly score = MSE(x_num, x_hat_num) + 0.1 × mean(CE_sport + CE_dport + CE_proto) / log(vocab_size)
```

| Component | Detail |
|-----------|--------|
| Embeddings | Sport (51,058×8), Dport (7,782×8), Proto (9×8) |
| Input dim | 21 + 8+8+8 = 45 |
| GRU hidden | 128 |
| Layers | 1 |
| Dropout | 0.1 |
| Loss | MSE + 0.1 × normalized cross-entropy (categorical heads) |
| Threshold | 99th percentile of normal-window scores (val set) |

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU training requires PyTorch with CUDA support. Set `device: "cuda"` in `config.yaml`.

---

## Pipeline

All commands run from the repository root.

### 1 — Preprocess

```bash
python -m scripts.preprocess_data
```

Reads the raw CSV, builds categorical vocabularies, z-score normalizes numeric features on normal rows only, and produces sliding-window sequences.

Output: `data/processed/preprocessed.npz`

### 2 — Train (original)

```bash
python -m src.training.train_flair
```

Trains on normal-only windows with 90/10 internal train/val split and early stopping.

Output: `experiments/results/flair_minimal.pt`

### 3 — Evaluate (original)

```bash
python -m src.training.evaluate_flair
```

Output: `experiments/results/anomaly_scores.csv`, `anomaly_scores_full.csv`, and printed metrics.

### 4 — Train/Evaluate on held-out splits

```bash
# Extract attack type labels (run once)
python scripts/extract_window_types.py

# Build split index files (run once)
python scripts/create_splits_retrain.py

# Train and evaluate each split
python scripts/train_split.py --split 80_10_10
python scripts/evaluate_split.py --split 80_10_10

python scripts/train_split.py --split 70_15_15
python scripts/evaluate_split.py --split 70_15_15

python scripts/train_split.py --split 60_20_20
python scripts/evaluate_split.py --split 60_20_20
```

### 5 — Holdout evaluation without retraining

```bash
python scripts/create_splits.py
python scripts/evaluate_holdout.py
```

Re-reports metrics on the held-out last 10% of windows using pre-computed scores from `anomaly_scores_full.csv`.

### 6 — Export to ONNX *(experimental, incomplete)*

> **Note:** ONNX export and real-time inference are purely experimental and have not been completed. The scripts below exist as a starting point for future work but are not part of the evaluated pipeline.

```bash
python scripts/export_onnx.py
python scripts/infer_realtime.py --onnx flair_minimal.onnx --meta deploy_meta.npz --mode batch --npz preprocessed.npz
python scripts/infer_realtime.py --onnx flair_minimal.onnx --meta deploy_meta.npz --mode stream
```

### 8 — Live demo

```bash
pip install streamlit plotly
streamlit run demo/app.py
```

Opens a browser-based interactive demo with heatmaps, latent vector visualization, reconstruction comparison, and an anomaly gauge.

---

## Configuration Reference

All settings in [`config.yaml`](config.yaml):

| Section | Key settings |
|---------|-------------|
| `features` | `categorical`, `numeric` — feature column names |
| `data` | `time_column`, `label_column` |
| `preprocess` | `window_size` (10), `stride` (1), `sort_time`, `dropna` |
| `model` | `hidden_dim` (128), `embed_dim` (8), `num_layers`, `dropout`, `cat_loss_weight` |
| `training` | `batch_size` (512), `learning_rate` (0.001), `epochs` (32), `patience` (10), `amp`, `device` |
| `evaluation` | `threshold_percentile` (99), `output_csv` |
| `paths` | `full_csv`, `processed_npz`, `checkpoint_path` |
