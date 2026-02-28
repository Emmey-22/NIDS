#!/usr/bin/env python
"""
Run predictions with a trained NIDS model bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict network traffic labels.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/nids_model.joblib"),
        help="Path to model bundle.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to parquet/csv input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/predictions.csv"),
        help="Path for output predictions CSV.",
    )
    parser.add_argument(
        "--drop-label",
        action="store_true",
        help="Drop label column if present in input.",
    )
    return parser.parse_args()


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    metadata = bundle.get("metadata", {})
    label_col = metadata.get("label_column")

    df = load_input(args.input)
    if args.drop_label and label_col and label_col in df.columns:
        df = df.drop(columns=[label_col])

    x = df.replace([np.inf, -np.inf], np.nan)
    pred_ids = model.predict(x)
    pred_labels = label_encoder.inverse_transform(pred_ids)

    output_df = pd.DataFrame({"prediction_id": pred_ids, "prediction_label": pred_labels})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output.as_posix()}")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
