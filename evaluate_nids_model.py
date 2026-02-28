#!/usr/bin/env python
"""
Stage-2 evaluation script for trained NIDS models.
Exports detailed artifacts for analysis/reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NIDS model and export report artifacts."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/nids_model.joblib"),
        help="Path to model bundle.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Bruteforce-Tuesday-no-metadata.parquet"),
        help="Path to parquet/csv evaluation data.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name. Falls back to model metadata or auto-detection.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional row sampling fraction in (0,1].",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts_eval"),
        help="Directory to write evaluation files.",
    )
    parser.add_argument(
        "--drop-label-for-predict",
        action="store_true",
        help="Force drop label column before prediction if present.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Supported input formats: .parquet, .csv")


def detect_label_column(
    df: pd.DataFrame,
    user_label_col: str | None,
    metadata_label_col: str | None,
) -> str | None:
    if user_label_col:
        return user_label_col
    if metadata_label_col and metadata_label_col in df.columns:
        return metadata_label_col
    for col in ["Label", "label", "Class", "class", "Target", "target"]:
        if col in df.columns:
            return col
    return None


def align_labels_for_model(
    y_raw: pd.Series,
    class_names: np.ndarray,
    binary_mode: bool,
) -> np.ndarray:
    y_norm = y_raw.astype(str)
    if binary_mode:
        y_norm = np.where(y_norm.str.lower() == "benign", "Benign", "Attack")
        y_norm = pd.Series(y_norm)
    unknown = sorted(set(y_norm.unique()) - set(class_names.tolist()))
    if unknown:
        preview = ", ".join(map(str, unknown[:5]))
        raise ValueError(f"Found labels not recognized by the model: {preview}")
    return y_norm.to_numpy()


def create_roc_dataframe(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
    rows = []
    if probs.shape[1] == 2:
        fpr, tpr, thr = roc_curve(y_true, probs[:, 1])
        for i in range(len(fpr)):
            rows.append(
                {
                    "class": class_names[1],
                    "fpr": float(fpr[i]),
                    "tpr": float(tpr[i]),
                    "threshold": float(thr[i]),
                }
            )
    elif probs.shape[1] > 2:
        y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        for class_idx, class_name in enumerate(class_names):
            fpr, tpr, thr = roc_curve(y_bin[:, class_idx], probs[:, class_idx])
            for i in range(len(fpr)):
                rows.append(
                    {
                        "class": str(class_name),
                        "fpr": float(fpr[i]),
                        "tpr": float(tpr[i]),
                        "threshold": float(thr[i]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    metadata: Dict[str, Any] = bundle.get("metadata", {})
    class_names = np.asarray(encoder.classes_)
    binary_mode = bool(metadata.get("binary_mode", False))

    df = load_table(args.data)
    if not 0 < args.sample_frac <= 1:
        raise ValueError("--sample-frac must be in (0,1].")
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.random_state).reset_index(drop=True)

    label_col = detect_label_column(
        df=df,
        user_label_col=args.label_col,
        metadata_label_col=metadata.get("label_column"),
    )

    has_truth = bool(label_col and label_col in df.columns)
    x = df.copy()
    y_true_ids = None
    y_true_labels = None

    if has_truth:
        y_true_labels = align_labels_for_model(
            y_raw=df[label_col],
            class_names=class_names,
            binary_mode=binary_mode,
        )
        y_true_ids = encoder.transform(y_true_labels)
        if args.drop_label_for_predict or label_col in x.columns:
            x = x.drop(columns=[label_col])

    x = x.replace([np.inf, -np.inf], np.nan)

    y_pred_ids = model.predict(x)
    y_pred_labels = encoder.inverse_transform(y_pred_ids)

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)
        except Exception:
            probs = None

    pred_df = pd.DataFrame(
        {
            "prediction_id": y_pred_ids,
            "prediction_label": y_pred_labels,
        }
    )
    if has_truth and y_true_ids is not None and y_true_labels is not None:
        pred_df.insert(0, "true_label", y_true_labels)
        pred_df.insert(0, "true_id", y_true_ids)

    if probs is not None:
        for class_idx, class_name in enumerate(class_names):
            pred_df[f"prob_{class_name}"] = probs[:, class_idx]

    pred_path = args.out_dir / "predictions_detailed.csv"
    pred_df.to_csv(pred_path, index=False)

    report: Dict[str, Any] = {
        "model_path": str(args.model),
        "data_path": str(args.data),
        "sample_frac": args.sample_frac,
        "n_rows": int(df.shape[0]),
        "n_features_used": int(x.shape[1]),
        "classes": class_names.tolist(),
        "has_ground_truth": has_truth,
    }

    if has_truth and y_true_ids is not None:
        report.update(
            {
                "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
                "f1_macro": float(f1_score(y_true_ids, y_pred_ids, average="macro")),
                "f1_weighted": float(f1_score(y_true_ids, y_pred_ids, average="weighted")),
                "classification_report": classification_report(
                    y_true_ids,
                    y_pred_ids,
                    target_names=class_names.tolist(),
                    digits=4,
                    output_dict=True,
                ),
                "confusion_matrix": confusion_matrix(y_true_ids, y_pred_ids).tolist(),
            }
        )

        cm_df = pd.DataFrame(
            report["confusion_matrix"],
            index=[f"true_{c}" for c in class_names],
            columns=[f"pred_{c}" for c in class_names],
        )
        cm_path = args.out_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)

        per_class_rows = []
        class_report = report["classification_report"]
        for class_name in class_names:
            row = class_report.get(str(class_name), {})
            per_class_rows.append(
                {
                    "class": str(class_name),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1_score": row.get("f1-score"),
                    "support": row.get("support"),
                }
            )
        per_class_path = args.out_dir / "per_class_metrics.csv"
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

        if probs is not None:
            try:
                if probs.shape[1] == 2:
                    report["roc_auc"] = float(roc_auc_score(y_true_ids, probs[:, 1]))
                elif probs.shape[1] > 2:
                    report["roc_auc_ovr_weighted"] = float(
                        roc_auc_score(
                            y_true_ids, probs, multi_class="ovr", average="weighted"
                        )
                    )
                roc_df = create_roc_dataframe(y_true_ids, probs, class_names)
                roc_path = args.out_dir / "roc_curve_points.csv"
                roc_df.to_csv(roc_path, index=False)
            except Exception:
                pass

    report_path = args.out_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved evaluation artifacts:")
    print(f"- Report      : {report_path.as_posix()}")
    print(f"- Predictions : {pred_path.as_posix()}")
    if has_truth:
        print(f"- Confusion   : {(args.out_dir / 'confusion_matrix.csv').as_posix()}")
        print(f"- Per-class   : {(args.out_dir / 'per_class_metrics.csv').as_posix()}")
        if probs is not None:
            print(f"- ROC points  : {(args.out_dir / 'roc_curve_points.csv').as_posix()}")

    if has_truth:
        print("\nQuick summary:")
        print(f"Accuracy     : {report.get('accuracy', float('nan')):.6f}")
        print(f"F1 (Macro)   : {report.get('f1_macro', float('nan')):.6f}")
        print(f"F1 (Weighted): {report.get('f1_weighted', float('nan')):.6f}")
        if "roc_auc" in report:
            print(f"ROC AUC      : {report['roc_auc']:.6f}")
        if "roc_auc_ovr_weighted" in report:
            print(f"ROC AUC (OVR): {report['roc_auc_ovr_weighted']:.6f}")
    else:
        print("\nGround-truth labels were not found; only predictions were exported.")


if __name__ == "__main__":
    main()
