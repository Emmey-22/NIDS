# Binary Network Intrusion Detection System

This project implements and compares three supervised binary classifiers for
network intrusion detection:

1. A standalone gain-ratio C4.5 decision tree.
2. AdaBoost using depth-1 gain-ratio C4.5 decision stumps.
3. A hybrid vote model combining the standalone C4.5 and AdaBoost models.

All source labels named `Benign` or `Normal` are encoded as `Normal = 0`.
Every other source label is encoded as `Attack = 1`. Training is always binary.

The candidate with the highest macro-F1 score on the held-out test split is
saved as the selected model. Candidate order is preserved, so an exact
model-selection tie favours the standalone C4.5 model.

## Hybrid decision rule

The hybrid model applies the following rule to each record:

- If C4.5 and AdaBoost predict the same class, that class is returned.
- If they disagree, the confidence each model assigns to its own prediction is
  compared.
- The prediction with the higher confidence is returned.
- An exact confidence tie favours C4.5.

For `predict_proba()`, the two probability vectors are averaged when the models
agree. When they disagree, the probability vector of the winning classifier is
returned.

## Project structure

```text
code/
|-- nids_core.py              # Preprocessing, models, evaluation, and inference
|-- train.py                  # Train and select the best candidate model
|-- predict.py                # Run batch prediction with a saved model
|-- evaluate.py               # Evaluate a saved model and export reports
|-- requirements.txt          # Python dependencies
|-- artifacts_final/
|   `-- metrics.json          # Metrics from the recorded full-data run
`-- README.md
```

The dataset and generated `.joblib` model files are intentionally excluded from
Git because they are local/generated artifacts.

## Installation

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

The default input is:

```text
Bruteforce-Tuesday-no-metadata.parquet
```

Both Parquet and CSV files are supported. The label column is auto-detected, or
it can be supplied with `--label-col`.

## Training

Run training with the default configuration:

```powershell
python train.py
```

The defaults are:

- Test fraction: `0.20`
- Random state: `42`
- AdaBoost estimators: `120`
- AdaBoost learning rate: `0.5`
- Standalone C4.5 maximum depth: `8`
- Output directory: `artifacts/`

Run a quick experiment on 10% of the dataset:

```powershell
python train.py --sample-frac 0.1 --ada-estimators 30 --out-dir artifacts_test
```

Supply custom model settings:

```powershell
python train.py --data Bruteforce-Tuesday-no-metadata.parquet `
  --test-size 0.2 `
  --ada-estimators 200 `
  --ada-learning-rate 0.3 `
  --tree-max-depth 10 `
  --out-dir artifacts
```

Training writes:

- `nids_model.joblib`: the selected model bundle and preprocessing metadata.
- `metrics.json`: separate C4.5, AdaBoost-C4.5, hybrid, and selected-model
  metrics.

Valid selected-model names are `c45_tree`, `adaboost_c45`, and `hybrid_vote`.

## Prediction

```powershell
python predict.py `
  --model artifacts/nids_model.joblib `
  --input Bruteforce-Tuesday-no-metadata.parquet `
  --output artifacts/predictions.csv `
  --drop-label
```

The prediction file contains the encoded class, class label, and class
probabilities when the selected model supports `predict_proba()`.

## Evaluation

```powershell
python evaluate.py `
  --model artifacts/nids_model.joblib `
  --data Bruteforce-Tuesday-no-metadata.parquet `
  --out-dir artifacts_eval
```

The evaluation directory can contain:

- `evaluation_report.json`
- `predictions_detailed.csv`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `roc_curve_points.csv`

## Recorded held-out results

The tracked [`artifacts_final/metrics.json`](artifacts_final/metrics.json)
contains the results from a full-data run using a stratified 80/20 split,
random state 42, 120 AdaBoost estimators, learning rate 0.5, and standalone C4.5
maximum depth 8.

| Candidate | Accuracy | Macro-F1 | Weighted-F1 | ROC AUC |
|---|---:|---:|---:|---:|
| C4.5 | 0.999705 | 0.996769 | 0.999704 | 0.998839 |
| AdaBoost-C4.5 | 0.993444 | 0.919735 | 0.993040 | 0.998352 |
| Hybrid vote | 0.999705 | 0.996769 | 0.999704 | 0.998121 |

The selected model for that run was `c45_tree`. C4.5 and the hybrid produced
the same held-out class predictions, so the model-selection tie was resolved in
favour of C4.5 by candidate order. Their probability outputs, and therefore ROC
AUC values, were not identical.

These values describe this particular dataset split and configuration; they
should not be interpreted as guaranteed performance on unseen network traffic.

## Command reference

```powershell
python train.py --help
python predict.py --help
python evaluate.py --help
```

Only load `.joblib` model files produced by a trusted source.
