# Binary NIDS Hybrid (AdaBoost + C4.5)

This project trains a Network Intrusion Detection System with:
- a standalone gain-ratio `C4.5 Decision Tree`
- `AdaBoost` with depth-1 C4.5 decision stumps
- a confidence-tiebroken voting hybrid of both models

Dataset expected: `Bruteforce-Tuesday-no-metadata.parquet`

## 1) Local setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Train

```powershell
python train.py
```

Outputs:
- `artifacts/nids_model.joblib`
- `artifacts/metrics.json`

## 3) Binary mode (Normal vs Attack)

Training is always binary. Source labels named `Benign` or `Normal` are stored as
`Normal` (class ID 0); every other label is stored as `Attack` (class ID 1).

```powershell
python train.py
```

The included `artifacts_baseline/` model is a frozen legacy binary reference and
is not overwritten by current training commands.

Baseline model metrics from the full-dataset binary retrain:
- Accuracy: `0.9999358506`
- F1 macro: `0.9992999569`
- F1 weighted: `0.9999358249`
- ROC AUC: `0.9999999103`

The baseline labels are:
- `Benign`
- `Attack` (merged from `FTP-Patator` and `SSH-Patator`)

## 4) Quick smoke run on smaller sample

```powershell
python train.py --sample-frac 0.1 --ada-estimators 30
```

## 5) Predict on a parquet/csv file

```powershell
python predict.py --model artifacts/nids_model.joblib --input Bruteforce-Tuesday-no-metadata.parquet --drop-label
```

Output:
- `artifacts/predictions.csv`

## 6) Tune Hybrid Model (GridSearchCV)

```powershell
python tune_nids_hybrid.py --sample-frac 0.2 --cv-folds 5 --out-dir artifacts_tuning
```

Outputs:
- `artifacts_tuning/nids_model_tuned.joblib`
- `artifacts_tuning/tuning_summary.json`
- `artifacts_tuning/cv_results.csv`
- `artifacts_tuning/cv_top_results.csv`

## 7) Detailed Evaluation Artifacts

```powershell
python evaluate.py --model artifacts/nids_model.joblib --data Bruteforce-Tuesday-no-metadata.parquet --out-dir artifacts_eval
```

Outputs:
- `artifacts_eval/evaluation_report.json`
- `artifacts_eval/predictions_detailed.csv`
- `artifacts_eval/confusion_matrix.csv`
- `artifacts_eval/per_class_metrics.csv`
- `artifacts_eval/roc_curve_points.csv` (if probabilities are available)
