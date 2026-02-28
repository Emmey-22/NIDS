#!/usr/bin/env python
"""
Network Intrusion Detection System (NIDS) training pipeline
using a hybrid AdaBoost + C4.5-style Decision Tree approach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a NIDS model using AdaBoost + C4.5-style Decision Tree."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Bruteforce-Tuesday-no-metadata.parquet"),
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label/target column name. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Convert labels to binary: Benign=0, Attack=1.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional row sampling fraction in (0, 1]. Useful for quick experiments.",
    )
    parser.add_argument(
        "--ada-estimators",
        type=int,
        default=120,
        help="Number of AdaBoost estimators (default: 120).",
    )
    parser.add_argument(
        "--ada-learning-rate",
        type=float,
        default=0.5,
        help="AdaBoost learning rate (default: 0.5).",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=8,
        help="Max depth for standalone C4.5-style tree (default: 8).",
    )
    parser.add_argument(
        "--ada-tree-depth",
        type=int,
        default=3,
        help="Max depth for AdaBoost base C4.5-style tree (default: 3).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save model artifacts (default: ./artifacts).",
    )
    return parser.parse_args()


def detect_label_column(df: pd.DataFrame, user_label_col: str | None) -> str:
    if user_label_col:
        if user_label_col not in df.columns:
            raise ValueError(f"Label column '{user_label_col}' not found in dataset.")
        return user_label_col

    candidates = ["Label", "label", "Class", "class", "Target", "target"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not auto-detect label column. Use --label-col to specify one."
    )


def prepare_data(
    df: pd.DataFrame,
    label_col: str,
    make_binary: bool,
    sample_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, Dict[str, Any]]:
    if not 0 < sample_frac <= 1:
        raise ValueError("--sample-frac must be in (0, 1].")

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if make_binary:
        y_raw = np.where(y_raw.str.lower() == "benign", "Benign", "Attack")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    metadata = {
        "label_column": label_col,
        "binary_mode": make_binary,
        "class_names": encoder.classes_.tolist(),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
    }
    return X, y, encoder, metadata


def build_tree_pipeline(max_depth: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=max_depth,
                    min_samples_leaf=1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hybrid_pipeline(
    weak_tree_depth: int,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> Pipeline:
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=weak_tree_depth,
        random_state=random_state,
    )
    hybrid = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", hybrid),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
) -> Dict[str, Any]:
    y_pred = model.predict(x_test)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
            elif probs.ndim == 2 and probs.shape[1] > 2:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass
    return metrics


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"Accuracy     : {metrics['accuracy']:.6f}")
    print(f"F1 (Macro)   : {metrics['f1_macro']:.6f}")
    print(f"F1 (Weighted): {metrics['f1_weighted']:.6f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC      : {metrics['roc_auc']:.6f}")
    if "roc_auc_ovr_weighted" in metrics:
        print(f"ROC AUC (OVR): {metrics['roc_auc_ovr_weighted']:.6f}")
    print("\nClassification report:")
    print(metrics["classification_report"])


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    df = pd.read_parquet(args.data)
    label_col = detect_label_column(df, args.label_col)
    print(f"Detected label column: {label_col}")

    X, y, encoder, metadata = prepare_data(
        df=df,
        label_col=label_col,
        make_binary=args.binary,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
    )
    print(
        f"Prepared data with {metadata['n_samples']} samples, "
        f"{metadata['n_features']} features, classes={metadata['class_names']}"
    )

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    c45_model = build_tree_pipeline(
        max_depth=args.tree_max_depth, random_state=args.random_state
    )
    hybrid_model = build_hybrid_pipeline(
        weak_tree_depth=args.ada_tree_depth,
        n_estimators=args.ada_estimators,
        learning_rate=args.ada_learning_rate,
        random_state=args.random_state,
    )

    print("\nTraining C4.5-style Decision Tree...")
    c45_model.fit(x_train, y_train)
    c45_metrics = evaluate_model(c45_model, x_test, y_test, metadata["class_names"])
    print_metrics("C4.5-style Tree", c45_metrics)

    print("\nTraining Hybrid AdaBoost + C4.5-style Tree...")
    hybrid_model.fit(x_train, y_train)
    hybrid_metrics = evaluate_model(
        hybrid_model, x_test, y_test, metadata["class_names"]
    )
    print_metrics("Hybrid AdaBoost + C4.5-style Tree", hybrid_metrics)

    winner_name = (
        "hybrid_adaboost_c45"
        if hybrid_metrics["f1_macro"] >= c45_metrics["f1_macro"]
        else "c45_tree"
    )
    best_model = hybrid_model if winner_name == "hybrid_adaboost_c45" else c45_model
    best_metrics = hybrid_metrics if winner_name == "hybrid_adaboost_c45" else c45_metrics

    model_bundle = {
        "model": best_model,
        "label_encoder": encoder,
        "metadata": metadata,
        "selected_model": winner_name,
    }
    model_path = out_dir / "nids_model.joblib"
    joblib.dump(model_bundle, model_path)

    metrics_bundle = {
        "selected_model": winner_name,
        "selected_metrics": best_metrics,
        "c45_metrics": c45_metrics,
        "hybrid_metrics": hybrid_metrics,
        "config": {
            "data": str(args.data),
            "label_col": label_col,
            "binary": args.binary,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "sample_frac": args.sample_frac,
            "ada_estimators": args.ada_estimators,
            "ada_learning_rate": args.ada_learning_rate,
            "tree_max_depth": args.tree_max_depth,
            "ada_tree_depth": args.ada_tree_depth,
        },
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=2)

    print("\nSaved:")
    print(f"- Model   : {model_path.as_posix()}")
    print(f"- Metrics : {metrics_path.as_posix()}")
    print(f"\nSelected best model: {winner_name}")


if __name__ == "__main__":
    main()
