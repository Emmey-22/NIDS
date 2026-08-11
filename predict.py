#!/usr/bin/env python
# Run batch inference with a saved NIDS model bundle.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from nids_core import load_dataset, Predictor


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NIDS inference using a trained model bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/nids_model.joblib"),
        help="Path to the .joblib model bundle produced by train.py.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input .parquet or .csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/predictions.csv"),
        help="Output path for the predictions CSV.",
    )
    parser.add_argument(
        "--drop-label",
        action="store_true",
        help=(
            "Drop the label column from the input before prediction "
            "(uses the label column name stored in the model bundle)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n[Stage 4 / Prediction Module]")
    print(f"Loading model bundle: {args.model}")
    predictor = Predictor.from_bundle(args.model)
    print(f"Model: {predictor.selected_model} | Classes: {predictor.class_names}")

    print(f"Loading input data : {args.input}")
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")

    label_col = predictor.label_column
    X = df.copy()
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    if args.drop_label and label_col and label_col in X.columns:
        X = X.drop(columns=[label_col])
        print(f"Dropped label column: '{label_col}'")
    elif label_col and label_col in X.columns:
        X = X.drop(columns=[label_col])

    pred_ids, pred_labels = predictor.predict(X)

    probs = predictor.predict_proba(X)

    output_df = pd.DataFrame(
        {
            "prediction_id": pred_ids,
            "prediction_label": pred_labels,
        }
    )

    if probs is not None:
        for cls_idx, cls_name in enumerate(predictor.class_names):
            output_df[f"prob_{cls_name}"] = probs[:, cls_idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"\nSaved {len(output_df):,} predictions → {args.output.as_posix()}")
    print("\nSample (first 10 rows):")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
