#!/usr/bin/env python
"""
Stage-2 tuning for NIDS hybrid model:
AdaBoost with C4.5-style (entropy) Decision Tree weak learners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune hybrid AdaBoost + C4.5-style tree for NIDS."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Bruteforce-Tuesday-no-metadata.parquet"),
        help="Path to parquet dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use Benign vs Attack labels.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional row sampling fraction in (0,1].",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test split fraction.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits for CV.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="f1_macro",
        help="GridSearch scoring metric (default: f1_macro).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV.",
    )
    parser.add_argument(
        "--estimators-grid",
        type=str,
        default="60,120,180",
        help="Comma-separated n_estimators values.",
    )
    parser.add_argument(
        "--learning-rate-grid",
        type=str,
        default="0.1,0.5,1.0",
        help="Comma-separated learning_rate values.",
    )
    parser.add_argument(
        "--weak-depth-grid",
        type=str,
        default="1,2,3,4",
        help="Comma-separated max_depth values for weak trees.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts_tuning"),
        help="Output directory for tuning artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top CV rows to export/show.",
    )
    return parser.parse_args()


def parse_grid_values(raw: str, cast_type: type) -> list:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Empty grid string: '{raw}'")
    return [cast_type(v) for v in vals]


def detect_label_column(df: pd.DataFrame, user_label_col: str | None) -> str:
    if user_label_col:
        if user_label_col not in df.columns:
            raise ValueError(f"Label column '{user_label_col}' not found.")
        return user_label_col
    for col in ["Label", "label", "Class", "class", "Target", "target"]:
        if col in df.columns:
            return col
    raise ValueError("Unable to detect label column. Provide --label-col.")


def prepare_data(
    df: pd.DataFrame,
    label_col: str,
    make_binary: bool,
    sample_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, Dict[str, Any]]:
    if not 0 < sample_frac <= 1:
        raise ValueError("--sample-frac must be in (0,1].")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    y_raw = df[label_col].astype(str)
    if make_binary:
        y_raw = np.where(y_raw.str.lower() == "benign", "Benign", "Attack")

    x = df.drop(columns=[label_col]).copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    metadata = {
        "label_column": label_col,
        "binary_mode": make_binary,
        "class_names": encoder.classes_.tolist(),
        "n_samples": int(df.shape[0]),
        "n_features": int(x.shape[1]),
    }
    return x, y, encoder, metadata


def build_pipeline(random_state: int) -> Pipeline:
    weak_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=1,
        random_state=random_state,
    )
    adaboost = AdaBoostClassifier(
        estimator=weak_tree,
        n_estimators=60,
        learning_rate=0.5,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", adaboost),
        ]
    )


def evaluate(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: Iterable[str],
) -> Dict[str, Any]:
    y_pred = model.predict(x_test)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(class_names), digits=4
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x_test)
            if probs.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
            elif probs.shape[1] > 2:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass
    return metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    df = pd.read_parquet(args.data)
    label_col = detect_label_column(df, args.label_col)
    print(f"Detected label column: {label_col}")

    x, y, encoder, metadata = prepare_data(
        df=df,
        label_col=label_col,
        make_binary=args.binary,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
    )
    print(
        f"Prepared {metadata['n_samples']} rows, {metadata['n_features']} features, "
        f"classes={metadata['class_names']}"
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(args.random_state)
    param_grid = {
        "classifier__n_estimators": parse_grid_values(args.estimators_grid, int),
        "classifier__learning_rate": parse_grid_values(args.learning_rate_grid, float),
        "classifier__estimator__max_depth": parse_grid_values(args.weak_depth_grid, int),
    }

    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    print("\nRunning GridSearchCV...")
    try:
        search.fit(x_train, y_train)
    except PermissionError:
        # Some Windows/sandboxed environments block loky worker creation.
        if args.n_jobs != 1:
            print(
                "Parallel CV workers unavailable in this environment. "
                "Retrying with --n-jobs 1."
            )
            search.set_params(n_jobs=1)
            search.fit(x_train, y_train)
        else:
            raise
    best_model: Pipeline = search.best_estimator_
    print(f"Best CV score ({args.scoring}): {search.best_score_:.6f}")
    print(f"Best params: {search.best_params_}")

    holdout_metrics = evaluate(best_model, x_test, y_test, metadata["class_names"])
    print("\n=== Holdout Metrics (Best Tuned Model) ===")
    print(f"Accuracy     : {holdout_metrics['accuracy']:.6f}")
    print(f"F1 (Macro)   : {holdout_metrics['f1_macro']:.6f}")
    print(f"F1 (Weighted): {holdout_metrics['f1_weighted']:.6f}")
    if "roc_auc" in holdout_metrics:
        print(f"ROC AUC      : {holdout_metrics['roc_auc']:.6f}")
    if "roc_auc_ovr_weighted" in holdout_metrics:
        print(f"ROC AUC (OVR): {holdout_metrics['roc_auc_ovr_weighted']:.6f}")

    model_bundle = {
        "model": best_model,
        "label_encoder": encoder,
        "metadata": metadata,
        "selected_model": "hybrid_adaboost_c45_tuned",
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
    }
    model_path = args.out_dir / "nids_model_tuned.joblib"
    joblib.dump(model_bundle, model_path)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score", ascending=True)
    cv_results_path = args.out_dir / "cv_results.csv"
    results_df.to_csv(cv_results_path, index=False)

    top_df = results_df.head(args.top_k)
    top_path = args.out_dir / "cv_top_results.csv"
    top_df.to_csv(top_path, index=False)

    summary = {
        "selected_model": "hybrid_adaboost_c45_tuned",
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "cv_scoring": args.scoring,
        "cv_folds": args.cv_folds,
        "holdout_metrics": holdout_metrics,
        "config": {
            "data": str(args.data),
            "label_col": label_col,
            "binary": args.binary,
            "sample_frac": args.sample_frac,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
            "estimators_grid": param_grid["classifier__n_estimators"],
            "learning_rate_grid": param_grid["classifier__learning_rate"],
            "weak_depth_grid": param_grid["classifier__estimator__max_depth"],
        },
    }
    summary_path = args.out_dir / "tuning_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved tuning artifacts:")
    print(f"- Tuned model : {model_path.as_posix()}")
    print(f"- Summary     : {summary_path.as_posix()}")
    print(f"- CV results  : {cv_results_path.as_posix()}")
    print(f"- Top CV rows : {top_path.as_posix()}")


if __name__ == "__main__":
    main()
