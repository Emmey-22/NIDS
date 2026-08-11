# NIDS — Network Intrusion Detection System
A binary hybrid **AdaBoost + true C4.5** classifier for network intrusion detection,
structured according to the five-stage system architecture:

```
Input Dataset
    |
Data Preprocessing          nids_core.py
    |
Model Development           nids_core.py
  |-- C4.5 (Gain Ratio)
  +-- AdaBoost + C4.5 decision stumps
    |
Hybrid Ensemble             nids_core.py
    |
Prediction Module           nids_core.py
    |
Result Evaluation           nids_core.py
```

---

## Project Structure

```
code/
├── nids_core.py                   # All shared logic (preprocessing, models, evaluation)
├── train.py                       # Entry point: train a new model
├── predict.py                     # Entry point: run batch inference
├── evaluate.py                    # Entry point: evaluate + export metrics
├── requirements.txt
└── artifacts_baseline/            # Frozen baseline reference (do not overwrite)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1 — Train

```bash
# Full dataset, binary mode (Normal vs Attack)
python train.py --out-dir artifacts

# Quick experiment with 5 % of data
python train.py --sample-frac 0.05 --out-dir artifacts_test

# Custom hyperparameters
python train.py \
  --ada-estimators 200 \
  --ada-learning-rate 0.3 \
  --tree-max-depth 10 \
  --out-dir artifacts
```

Outputs saved to `--out-dir`:
- `nids_model.joblib` — serialised model bundle
- `metrics.json` — training-time metrics for all three models

### 2 — Predict

```bash
python predict.py \
  --model  artifacts/nids_model.joblib \
  --input  Bruteforce-Tuesday-no-metadata.parquet \
  --output artifacts/predictions.csv \
  --drop-label
```

### 3 — Evaluate

```bash
python evaluate.py \
  --model   artifacts/nids_model.joblib \
  --data    Bruteforce-Tuesday-no-metadata.parquet \
  --out-dir artifacts_eval
```

Outputs written to `--out-dir`:
- `evaluation_report.json`
- `predictions_detailed.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `roc_curve_points.csv` (if probabilities available)

---

## Key Design Choices

### True C4.5 (not ID3)
`nids_core.py` implements a genuine C4.5 tree using **Gain Ratio**:

    GainRatio(A) = InformationGain(A) / SplitInformation(A)

sklearn's `DecisionTreeClassifier(criterion='entropy')` is ID3 (plain Information
Gain), **not** C4.5. This implementation corrects that distinction.

### Sample-weight support
`C45Classifier.fit(X, y, sample_weight=...)` is fully supported, enabling
`AdaBoostClassifier` to reweight misclassified samples across boosting rounds.
Every AdaBoost weak learner is fixed at depth 1, making it a gain-ratio C4.5
decision stump as specified by the study methodology.

### Baseline preservation
`artifacts_baseline/` contains the pre-restructure frozen reference model.
New training runs always default to `artifacts/` and never touch the baseline.

---

## CLI Reference

```
python train.py    --help
python predict.py  --help
python evaluate.py --help
```
