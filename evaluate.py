#!/usr/bin/env python
"""Evaluate a saved NIDS model and export its metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles may not support the symbols used in CLI output by default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Any, Dict

import numpy as np
import pandas as pd

from nids_core import detect_label_column, load_dataset, Evaluator, Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained NIDS model and export detailed report artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/nids_model.joblib"),
        help="Path to the .joblib model bundle.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Bruteforce-Tuesday-no-metadata.parquet"),
        help="Path to the labelled .parquet or .csv evaluation dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name. Falls back to bundle metadata or auto-detection.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Row sampling fraction in (0, 1]. Useful for large datasets.",
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
        help="Directory to write evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n[Stage 4 / Prediction Module]")
    print(f"Loading model bundle: {args.model}")
    predictor = Predictor.from_bundle(args.model)
    print(
        f"Model        : {predictor.selected_model}\n"
        f"Classes      : {predictor.class_names}\n"
        f"Binary mode  : {predictor.binary_mode}"
    )

    print(f"\nLoading evaluation data: {args.data}")
    df = load_dataset(args.data)

    if not 0 < args.sample_frac <= 1:
        raise ValueError("--sample-frac must be in (0, 1].")
    if args.sample_frac < 1.0:
        df = df.sample(
            frac=args.sample_frac, random_state=args.random_state
        ).reset_index(drop=True)
        print(f"Sampled {args.sample_frac:.1%} → {len(df):,} rows")

    label_col = detect_label_column(
        df,
        args.label_col or predictor.label_column,
    )
    has_labels = label_col in df.columns
    print(f"Label column : '{label_col}' (found={has_labels})")

    X = df.copy()
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    if has_labels and label_col in X.columns:
        X = X.drop(columns=[label_col])

    pred_ids, pred_labels = predictor.predict(X)
    probs = predictor.predict_proba(X)

    pred_df = pd.DataFrame(
        {
            "prediction_id": pred_ids,
            "prediction_label": pred_labels,
        }
    )

    print("\n[Stage 5 / Result Evaluation]")

    if not has_labels:
        print(
            "No ground-truth labels found. "
            "Only predictions will be exported (no metrics computed)."
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.out_dir / "predictions_detailed.csv", index=False)
        print(f"Saved: {(args.out_dir / 'predictions_detailed.csv').as_posix()}")
        return

    y_raw = df[label_col].astype(str)
    if predictor.binary_mode:
        is_normal = y_raw.str.strip().str.lower().isin(("benign", "normal"))
        y_raw = pd.Series(np.where(is_normal, "Normal", "Attack"))

    known = set(predictor.encoder.classes_.tolist())
    unknown = sorted(set(y_raw.unique()) - known)
    if unknown:
        preview = ", ".join(map(str, unknown[:5]))
        raise ValueError(
            f"Evaluation data contains labels not seen during training: {preview}"
        )
    y_true_ids: np.ndarray = predictor.encoder.transform(y_raw)
    y_true_labels: np.ndarray = predictor.encoder.inverse_transform(y_true_ids)

    pred_df.insert(0, "true_label", y_true_labels)
    pred_df.insert(0, "true_id", y_true_ids)
    if probs is not None:
        for cls_idx, cls_name in enumerate(predictor.class_names):
            pred_df[f"prob_{cls_name}"] = probs[:, cls_idx]

    evaluator = Evaluator()
    evaluator.compute(
        y_true_ids=y_true_ids,
        y_pred_ids=pred_ids,
        probs=probs,
        class_names=predictor.class_names,
    )
    evaluator.print_summary(predictor.selected_model)

    report_meta: Dict[str, Any] = {
        "model_path": str(args.model),
        "data_path": str(args.data),
        "sample_frac": args.sample_frac,
        "n_rows_evaluated": int(len(df)),
        "n_features_used": int(X.shape[1]),
        "classes": predictor.class_names,
        "selected_model": predictor.selected_model,
    }

    evaluator.export(
        out_dir=args.out_dir,
        pred_df=pred_df,
        report_meta=report_meta,
    )

    print("\n-- Quick summary --------------------------------------------")
    print(f"  Accuracy     : {evaluator.metrics_['accuracy']:.6f}")
    print(f"  F1 (Macro)   : {evaluator.metrics_['f1_macro']:.6f}")
    print(f"  F1 (Weighted): {evaluator.metrics_['f1_weighted']:.6f}")
    if "roc_auc" in evaluator.metrics_:
        print(f"  ROC AUC      : {evaluator.metrics_['roc_auc']:.6f}")
    if "roc_auc_ovr_weighted" in evaluator.metrics_:
        print(f"  ROC AUC (OVR): {evaluator.metrics_['roc_auc_ovr_weighted']:.6f}")


if __name__ == "__main__":
    main()
