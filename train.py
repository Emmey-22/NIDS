#!/usr/bin/env python
# Train and save a NIDS model bundle.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
from sklearn.model_selection import train_test_split

from nids_core import DataPreprocessor, detect_label_column, load_dataset, Evaluator, HybridEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary NIDS model using C4.5, AdaBoost-C4.5 stumps, "
            "and hybrid voting.\n"
            "Outputs are written to --out-dir (default: artifacts/)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Bruteforce-Tuesday-no-metadata.parquet"),
        help="Path to the input .parquet or .csv dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label/target column name. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Row sampling fraction in (0, 1]. Useful for quick experiments.",
    )
    parser.add_argument(
        "--ada-estimators",
        type=int,
        default=120,
        help="Number of AdaBoost boosting rounds.",
    )
    parser.add_argument(
        "--ada-learning-rate",
        type=float,
        default=0.5,
        help="AdaBoost learning rate (shrinkage).",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=8,
        help="Max depth for the standalone C4.5 tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help=(
            "Directory to save model artifacts. "
            "Use a different path to avoid overwriting artifacts_baseline/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Stage 1 / Data Preprocessing]")
    print(f"Loading dataset: {args.data}")
    df = load_dataset(args.data)
    label_col = detect_label_column(df, args.label_col)
    print(f"Detected label column: '{label_col}'")

    preprocessor = DataPreprocessor()
    X, y, metadata = preprocessor.fit_transform(
        df=df,
        label_col=label_col,
        binary=True,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
    )
    print(
        f"Prepared {metadata['n_samples']:,} samples, "
        f"{metadata['n_features']} features, "
        f"classes={metadata['class_names']}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(
        f"Train / test split: "
        f"{len(X_train):,} / {len(X_test):,} samples "
        f"(test_size={args.test_size})"
    )

    ensemble = HybridEnsemble(
        ada_estimators=args.ada_estimators,
        ada_learning_rate=args.ada_learning_rate,
        c45_max_depth=args.tree_max_depth,
        random_state=args.random_state,
    )
    ensemble.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=metadata["class_names"],
    )

    print("\n[Stage 5 / Result Evaluation]")
    evaluator = Evaluator()
    y_pred = ensemble.best_model_.predict(X_test)
    probs = None
    if hasattr(ensemble.best_model_, "predict_proba"):
        try:
            probs = ensemble.best_model_.predict_proba(X_test)
        except Exception:
            pass
    evaluator.compute(y_test, y_pred, probs, metadata["class_names"])
    evaluator.print_summary(f"Best Model ({ensemble.best_model_name_})")

    bundle = ensemble.to_bundle(preprocessor.encoder, metadata)
    model_path = out_dir / "nids_model.joblib"
    joblib.dump(bundle, model_path)

    metrics_bundle = {
        "selected_model": ensemble.best_model_name_,
        "selected_metrics": ensemble.best_metrics_,
        "c45_metrics": ensemble.c45_metrics_,
        "adaboost_metrics": ensemble.adaboost_metrics_,
        "hybrid_metrics": ensemble.hybrid_metrics_,
        "config": {
            "data": str(args.data),
            "label_col": label_col,
            "binary": True,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "sample_frac": args.sample_frac,
            "ada_estimators": args.ada_estimators,
            "ada_learning_rate": args.ada_learning_rate,
            "tree_max_depth": args.tree_max_depth,
        },
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_bundle, fh, indent=2)

    print("\n-- Saved artifacts ------------------------------------------")
    print(f"  Model  : {model_path.as_posix()}")
    print(f"  Metrics: {metrics_path.as_posix()}")
    print(f"\nSelected best model : {ensemble.best_model_name_}")
    print(f"F1 (Macro)          : {ensemble.best_metrics_['f1_macro']:.6f}")
    print(f"Accuracy            : {ensemble.best_metrics_['accuracy']:.6f}")


if __name__ == "__main__":
    main()
