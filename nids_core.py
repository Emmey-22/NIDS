#Shared preprocessing, model, prediction, and evaluation logic for NIDS.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.validation import check_is_fitted


# Stage 1 - Data Preprocessing
def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported file format '{suffix}'. Accepted formats: .parquet, .csv"
    )


def detect_label_column(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(
                f"Label column '{user_col}' was not found in the dataset. "
                f"Available columns: {list(df.columns)[:10]}"
            )
        return user_col

    candidates = ["Label", "label", "Class", "class", "Target", "target"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Cannot auto-detect the label column. "
        "Use --label-col to specify one explicitly."
    )


class DataPreprocessor:
    """Data cleaning and label encoding for the NIDS pipeline."""

    def __init__(self) -> None:
        self.encoder: LabelEncoder = LabelEncoder()

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str,
        binary: bool = False,
        sample_frac: float = 1.0,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")

        if sample_frac < 1.0:
            df = (
                df.sample(frac=sample_frac, random_state=random_state)
                .reset_index(drop=True)
            )

        y_raw: pd.Series = df[label_col].astype(str)
        X: pd.DataFrame = df.drop(columns=[label_col]).copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        if binary:  # Treat Benign/Normal as Normal and every other label as Attack.
            is_normal = y_raw.str.strip().str.lower().isin(("benign", "normal"))
            y_raw = pd.Series(np.where(is_normal, "Normal", "Attack"))
            # LabelEncoder sorts labels alphabetically by default, which would assign
            # Attack=0. Set the dissertation's required order explicitly instead.
            self.encoder.classes_ = np.asarray(["Normal", "Attack"], dtype=object)
            y = np.where(y_raw.to_numpy() == "Normal", 0, 1).astype(np.int64)
        else:
            y = self.encoder.fit_transform(y_raw)

        metadata: Dict[str, Any] = {
            "label_column": label_col,
            "binary_mode": binary,
            "class_names": self.encoder.classes_.tolist(),
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
        }
        return X, y, metadata

    def transform(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
    ) -> pd.DataFrame:
        X = df.copy()
        if label_col and label_col in X.columns:
            X = X.drop(columns=[label_col])
        return X.replace([np.inf, -np.inf], np.nan)

    def align_labels(
        self,
        y_raw: pd.Series,
        binary: bool,
    ) -> np.ndarray:
        y_norm = y_raw.astype(str)
        if binary:
            is_normal = y_norm.str.strip().str.lower().isin(("benign", "normal"))
            y_norm = pd.Series(np.where(is_normal, "Normal", "Attack"))
        known = set(self.encoder.classes_.tolist())
        unknown = sorted(set(y_norm.unique()) - known)
        if unknown:
            preview = ", ".join(map(str, unknown[:5]))
            raise ValueError(
                f"Evaluation data contains labels not seen during training: {preview}"
            )
        return self.encoder.transform(y_norm)


# Stage 2 - Model Development: C4.5 Decision Tree
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "proba", "is_leaf")

    def __init__(self) -> None:
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None
        self.value: Optional[int] = None
        self.proba: Optional[np.ndarray] = None
        self.is_leaf: bool = False


def _entropy_batch(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    safe_t = np.where(totals > 1e-12, totals, 1.0)[:, None]
    probs = counts / safe_t
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0.0, np.log2(probs), 0.0)
    h = -np.einsum("ij,ij->i", probs, log_p)
    return np.where(totals > 1e-12, h, 0.0)


class C45Classifier(BaseEstimator, ClassifierMixin):  # C4.5 classifier using gain-ratio splits and batch prediction.

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_leaf: int = 1,
        min_gain_ratio: float = 1e-4,
        max_thresholds: int = 64,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "C45Classifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.classes_: np.ndarray = np.unique(y)
        self.n_classes_: int = int(len(self.classes_))
        self.n_features_in_: int = X.shape[1]

        y_enc = np.searchsorted(self.classes_, y).astype(np.int64)

        if sample_weight is None:
            weights = np.ones(len(y), dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape[0] != len(y):
                raise ValueError(
                    f"sample_weight length {weights.shape[0]} != y {len(y)}."
                )
            total = weights.sum()
            if total < 1e-12:
                raise ValueError("sample_weight sums to zero.")
            weights = weights * (len(y) / total)

        root = self._build_tree(X, y_enc, weights, depth=0)
        self._compile_tree(root)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_node_feature")
        X = np.asarray(X, dtype=np.float64)
        leaf_node_ids = self._route_samples(X)
        class_indices = self._node_value[leaf_node_ids]
        return self.classes_[class_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_node_feature")
        X = np.asarray(X, dtype=np.float64)
        leaf_node_ids = self._route_samples(X)
        return self._node_proba[leaf_node_ids]

    def _compile_tree(self, root: _Node) -> None:
        id_to_idx: dict = {}
        queue = [root]
        ordered: list = []
        while queue:
            node = queue.pop(0)
            id_to_idx[id(node)] = len(ordered)
            ordered.append(node)
            if not node.is_leaf:
                queue.append(node.left)   # type: ignore[arg-type]
                queue.append(node.right)  # type: ignore[arg-type]

        n = len(ordered)
        feature   = np.full(n, -1, dtype=np.int32)
        threshold = np.full(n, np.nan, dtype=np.float64)
        left      = np.full(n, -1, dtype=np.int32)
        right     = np.full(n, -1, dtype=np.int32)
        value     = np.zeros(n, dtype=np.int64)
        proba     = np.zeros((n, self.n_classes_), dtype=np.float64)
        is_leaf   = np.ones(n, dtype=bool)

        for node in ordered:
            i = id_to_idx[id(node)]
            if node.is_leaf:
                value[i] = node.value  # type: ignore[assignment]
                proba[i] = node.proba  # type: ignore[assignment]
            else:
                is_leaf[i]   = False
                feature[i]   = node.feature    # type: ignore[assignment]
                threshold[i] = node.threshold  # type: ignore[assignment]
                left[i]      = id_to_idx[id(node.left)]
                right[i]     = id_to_idx[id(node.right)]

        self._node_feature   = feature
        self._node_threshold = threshold
        self._node_left      = left
        self._node_right     = right
        self._node_value     = value
        self._node_proba     = proba
        self._node_is_leaf   = is_leaf

    def _route_samples(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        node_idx = np.zeros(n, dtype=np.int32)

        depth_limit = (self.max_depth or 64) + 2
        for _ in range(depth_limit):
            active = ~self._node_is_leaf[node_idx]
            if not active.any():
                break

            feat_idx = self._node_feature[node_idx[active]]
            thresh   = self._node_threshold[node_idx[active]]
            col_vals = X[active, feat_idx]

            # Missing values follow the same left branch used during split scoring.
            go_left = np.isnan(col_vals) | (col_vals <= thresh)

            cur     = node_idx[active]
            new_idx = np.where(go_left,
                               self._node_left[cur],
                               self._node_right[cur])
            node_idx[active] = new_idx

        return node_idx

    def _best_split(
        self,
        X: np.ndarray,
        y_enc: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float]]:
        best_gr:        float         = self.min_gain_ratio
        best_feature:   Optional[int] = None
        best_threshold: Optional[float] = None

        w_total: float = float(weights.sum())
        if w_total < 1e-12:
            return None, None

        parent_counts = np.bincount(
            y_enc, weights=weights, minlength=self.n_classes_
        )
        h_parent: float = float(
            _entropy_batch(parent_counts[np.newaxis, :], np.array([w_total]))[0]
        )

        for feat_idx in range(X.shape[1]):
            col      = X[:, feat_idx]
            not_nan  = ~np.isnan(col)
            nan_mask = ~not_nan
            n_valid  = int(not_nan.sum())

            if n_valid < 2:
                continue

            w_nan      = float(weights[nan_mask].sum()) if nan_mask.any() else 0.0
            nan_counts = (
                np.bincount(y_enc[nan_mask], weights=weights[nan_mask],
                            minlength=self.n_classes_)
                if nan_mask.any()
                else np.zeros(self.n_classes_, dtype=np.float64)
            )

            valid_idx = np.where(not_nan)[0]
            order     = np.argsort(col[valid_idx], kind="stable")
            sorted_y  = y_enc[valid_idx[order]]
            sorted_w  = weights[valid_idx[order]]
            sorted_v  = col[valid_idx[order]]

            # Vectorized counts keep split scoring practical for large flow datasets.
            one_hot             = np.zeros((n_valid, self.n_classes_), dtype=np.float64)
            one_hot[np.arange(n_valid), sorted_y] = sorted_w
            cum_counts          = np.empty((n_valid + 1, self.n_classes_), dtype=np.float64)
            cum_counts[0]       = 0.0
            np.cumsum(one_hot, axis=0, out=cum_counts[1:])
            cum_w               = np.empty(n_valid + 1, dtype=np.float64)
            cum_w[0]            = 0.0
            np.cumsum(sorted_w, out=cum_w[1:])
            total_valid_w: float = float(cum_w[n_valid])

            unique_v = np.unique(sorted_v)
            if len(unique_v) < 2:
                continue
            thresholds = (unique_v[:-1] + unique_v[1:]) / 2.0

            if len(thresholds) > self.max_thresholds:
                idx        = np.round(
                    np.linspace(0, len(thresholds) - 1, self.max_thresholds)
                ).astype(int)
                thresholds = thresholds[idx]

            split_idx    = np.searchsorted(sorted_v, thresholds, side="right")
            w_left_valid = cum_w[split_idx]
            w_left       = w_left_valid + w_nan
            w_right      = total_valid_w - w_left_valid

            valid_mask = (
                (w_left  >= self.min_samples_leaf) &
                (w_right >= self.min_samples_leaf)
            )
            if not valid_mask.any():
                continue

            si = split_idx[valid_mask]
            wl = w_left[valid_mask]
            wr = w_right[valid_mask]

            lc_valid = cum_counts[si]
            rc       = cum_counts[n_valid] - lc_valid
            lc       = lc_valid + nan_counts[np.newaxis, :]

            h_left  = _entropy_batch(lc, wl)
            h_right = _entropy_batch(rc, wr)

            ig = h_parent - (wl / w_total) * h_left - (wr / w_total) * h_right

            p_l = wl / w_total
            p_r = wr / w_total
            with np.errstate(divide="ignore", invalid="ignore"):
                split_info = -(p_l * np.log2(p_l) + p_r * np.log2(p_r))

            good = (ig > 0.0) & (split_info > 1e-10)
            if not good.any():
                continue
            gr = np.where(good, ig / split_info, 0.0)

            best_i = int(np.argmax(gr))
            if gr[best_i] > best_gr:
                best_gr        = float(gr[best_i])
                best_feature   = feat_idx
                valid_thresh   = thresholds[valid_mask]
                best_threshold = float(valid_thresh[best_i])

        return best_feature, best_threshold

    def _make_leaf(self, y_enc: np.ndarray, weights: np.ndarray) -> _Node:
        node         = _Node()
        node.is_leaf = True
        w_counts     = np.bincount(y_enc, weights=weights, minlength=self.n_classes_)
        node.value   = int(np.argmax(w_counts))
        total        = float(w_counts.sum())
        node.proba   = (
            w_counts / total if total > 0
            else np.full(self.n_classes_, 1.0 / self.n_classes_)
        )
        return node

    def _build_tree(
        self,
        X: np.ndarray,
        y_enc: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> _Node:
        if len(y_enc) == 0:
            node         = _Node()
            node.is_leaf = True
            node.value   = 0
            node.proba   = np.full(self.n_classes_, 1.0 / self.n_classes_)
            return node

        if (
            len(np.unique(y_enc)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or weights.sum() < 2.0 * self.min_samples_leaf
        ):
            return self._make_leaf(y_enc, weights)

        feat, thresh = self._best_split(X, y_enc, weights)
        if feat is None:
            return self._make_leaf(y_enc, weights)

        col        = X[:, feat]
        left_mask  = np.isnan(col) | (col <= thresh)  # type: ignore[operator]
        right_mask = ~left_mask

        node           = _Node()
        node.is_leaf   = False
        node.feature   = feat
        node.threshold = thresh
        node.left  = self._build_tree(X[left_mask],  y_enc[left_mask],  weights[left_mask],  depth + 1)
        node.right = self._build_tree(X[right_mask], y_enc[right_mask], weights[right_mask], depth + 1)
        return node


def build_c45_pipeline(max_depth: int = 8, random_state: int = 42) -> Pipeline:
    return Pipeline(steps=[
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("classifier", C45Classifier(max_depth=max_depth, random_state=random_state)),
    ])


# Stage 2 - Model Development: AdaBoost + C4.5
def build_adaboost_pipeline(
    n_estimators: int = 120,
    learning_rate: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    weak_learner = C45Classifier(
        max_depth=1,
        random_state=random_state,
    )

    adaboost = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", adaboost),
        ]
    )


# Stage 3 - Hybrid Ensemble
class HybridVoteModel:
    """Combine fitted C4.5 and AdaBoost pipelines by voting and confidence."""

    def __init__(self, c45_pipeline: Pipeline, adaboost_pipeline: Pipeline) -> None:
        self.c45_pipeline = c45_pipeline
        self.adaboost_pipeline = adaboost_pipeline

        classes_c45 = getattr(c45_pipeline, "classes_", None)
        classes_adaboost = getattr(adaboost_pipeline, "classes_", None)
        if (
            classes_c45 is None
            or classes_adaboost is None
            or not np.array_equal(classes_c45, classes_adaboost)
        ):
            raise ValueError(
                "HybridVoteModel requires both pipelines to expose matching classes_."
            )
        self.classes_ = np.asarray(classes_c45)

    def _combined_outputs(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        proba_c45 = np.asarray(self.c45_pipeline.predict_proba(X))
        proba_adaboost = np.asarray(self.adaboost_pipeline.predict_proba(X))
        if proba_c45.shape != proba_adaboost.shape:
            raise ValueError("Hybrid component probability outputs must have equal shapes.")

        idx_c45 = np.argmax(proba_c45, axis=1)
        idx_adaboost = np.argmax(proba_adaboost, axis=1)
        agreement = idx_c45 == idx_adaboost
        confidence_c45 = proba_c45[np.arange(len(proba_c45)), idx_c45]
        confidence_adaboost = proba_adaboost[
            np.arange(len(proba_adaboost)), idx_adaboost
        ]

        # Exact confidence ties favor C4.5.
        choose_c45 = agreement | (confidence_c45 >= confidence_adaboost)
        combined_proba = np.where(
            agreement[:, None],
            (proba_c45 + proba_adaboost) / 2.0,
            np.where(choose_c45[:, None], proba_c45, proba_adaboost),
        )
        prediction_idx = np.where(choose_c45, idx_c45, idx_adaboost)
        return self.classes_[prediction_idx], combined_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions, _ = self._combined_outputs(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _, probabilities = self._combined_outputs(X)
        return probabilities


def _evaluate_pipeline(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_names, digits=4
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
            elif probs.ndim == 2 and probs.shape[1] > 2:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y_test, probs, multi_class="ovr", average="weighted"
                    )
                )
        except Exception:
            pass

    return metrics


def _print_metrics(title: str, metrics: Dict[str, Any]) -> None:
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


class HybridEnsemble:  # Trains C4.5, AdaBoost-C4.5, and their voting hybrid.

    def __init__(
        self,
        ada_estimators: int = 120,
        ada_learning_rate: float = 0.5,
        c45_max_depth: int = 8,
        random_state: int = 42,
    ) -> None:
        self.ada_estimators = ada_estimators
        self.ada_learning_rate = ada_learning_rate
        self.c45_max_depth = c45_max_depth
        self.random_state = random_state

        self.best_model_: Optional[Any] = None
        self.best_model_name_: Optional[str] = None
        self.best_metrics_: Optional[Dict[str, Any]] = None
        self.c45_metrics_: Optional[Dict[str, Any]] = None
        self.adaboost_metrics_: Optional[Dict[str, Any]] = None
        self.hybrid_metrics_: Optional[Dict[str, Any]] = None
        self.c45_pipeline_: Optional[Pipeline] = None
        self.adaboost_pipeline_: Optional[Pipeline] = None
        self.hybrid_pipeline_: Optional[HybridVoteModel] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> "HybridEnsemble":
        print("\n[Stage 2 / Model Development]")
        print("Training C4.5 Decision Tree (Gain Ratio splitting)...")
        self.c45_pipeline_ = build_c45_pipeline(
            max_depth=self.c45_max_depth,
            random_state=self.random_state,
        )
        self.c45_pipeline_.fit(X_train, y_train)
        self.c45_metrics_ = _evaluate_pipeline(
            self.c45_pipeline_, X_test, y_test, class_names
        )
        _print_metrics("C4.5 Decision Tree", self.c45_metrics_)

        print("\n[Stage 3 / AdaBoost + C4.5]")
        print(
            f"Training AdaBoost + C4.5 "
            f"(n_estimators={self.ada_estimators}, "
            f"lr={self.ada_learning_rate}, "
            f"weak_depth=1)..."
        )
        self.adaboost_pipeline_ = build_adaboost_pipeline(
            n_estimators=self.ada_estimators,
            learning_rate=self.ada_learning_rate,
            random_state=self.random_state,
        )
        self.adaboost_pipeline_.fit(X_train, y_train)
        self.adaboost_metrics_ = _evaluate_pipeline(
            self.adaboost_pipeline_, X_test, y_test, class_names
        )
        _print_metrics("AdaBoost + C4.5", self.adaboost_metrics_)

        print("\n[Stage 4 / Hybrid Ensemble]")
        print("Combining C4.5 and AdaBoost+C4.5 using confidence-tiebroken voting...")
        self.hybrid_pipeline_ = HybridVoteModel(
            self.c45_pipeline_, self.adaboost_pipeline_
        )
        self.hybrid_metrics_ = _evaluate_pipeline(
            self.hybrid_pipeline_, X_test, y_test, class_names
        )
        _print_metrics("Hybrid Ensemble (Majority Vote)", self.hybrid_metrics_)

        candidates: Dict[str, Tuple[Any, Dict[str, Any]]] = {
            "c45_tree": (self.c45_pipeline_, self.c45_metrics_),
            "adaboost_c45": (self.adaboost_pipeline_, self.adaboost_metrics_),
            "hybrid_vote": (self.hybrid_pipeline_, self.hybrid_metrics_),
        }
        self.best_model_name_ = max(
            candidates, key=lambda name: candidates[name][1]["f1_macro"]
        )
        self.best_model_, self.best_metrics_ = candidates[self.best_model_name_]

        print(f"\n-> Selected best model: {self.best_model_name_} "
              f"(F1-macro = {self.best_metrics_['f1_macro']:.6f})")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("HybridEnsemble.fit() must be called before predict().")
        return self.best_model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model_ is None:
            raise RuntimeError("HybridEnsemble.fit() must be called before predict_proba().")
        return self.best_model_.predict_proba(X)

    def to_bundle(
        self,
        encoder: LabelEncoder,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "model": self.best_model_,
            "label_encoder": encoder,
            "metadata": metadata,
            "selected_model": self.best_model_name_,
            "c45_metrics": self.c45_metrics_,
            "adaboost_metrics": self.adaboost_metrics_,
            "hybrid_metrics": self.hybrid_metrics_,
        }


# Stage 5 - Result Evaluation
class Evaluator:
    #Computes and exports all evaluation metrics for a NIDS model.

    def __init__(self) -> None:
        self.metrics_: Dict[str, Any] = {}
        self.class_names_: List[str] = []
        self._roc_df: Optional[pd.DataFrame] = None

    def compute(
        self,
        y_true_ids: np.ndarray,
        y_pred_ids: np.ndarray,
        probs: Optional[np.ndarray],
        class_names: List[str],
    ) -> "Evaluator":
        self.class_names_ = list(class_names)

        self.metrics_ = {
            "accuracy": float(accuracy_score(y_true_ids, y_pred_ids)),
            "f1_macro": float(f1_score(y_true_ids, y_pred_ids, average="macro")),
            "f1_weighted": float(
                f1_score(y_true_ids, y_pred_ids, average="weighted")
            ),
            "classification_report": classification_report(
                y_true_ids,
                y_pred_ids,
                target_names=self.class_names_,
                digits=4,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_true_ids, y_pred_ids).tolist(),
        }

        if probs is not None:
            try:
                if probs.shape[1] == 2:
                    self.metrics_["roc_auc"] = float(
                        roc_auc_score(y_true_ids, probs[:, 1])
                    )
                    fpr, tpr, thr = roc_curve(y_true_ids, probs[:, 1])
                    self._roc_df = pd.DataFrame(
                        {
                            "class": self.class_names_[1],
                            "fpr": fpr.astype(float),
                            "tpr": tpr.astype(float),
                            "threshold": thr.astype(float),
                        }
                    )
                elif probs.shape[1] > 2:
                    self.metrics_["roc_auc_ovr_weighted"] = float(
                        roc_auc_score(
                            y_true_ids,
                            probs,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
                    n_cls = len(self.class_names_)
                    y_bin = label_binarize(
                        y_true_ids, classes=np.arange(n_cls)
                    )
                    rows: List[Dict[str, Any]] = []
                    for cls_idx, cls_name in enumerate(self.class_names_):
                        fpr, tpr, thr = roc_curve(
                            y_bin[:, cls_idx], probs[:, cls_idx]
                        )
                        for f, t, h in zip(fpr, tpr, thr):
                            rows.append(
                                {
                                    "class": str(cls_name),
                                    "fpr": float(f),
                                    "tpr": float(t),
                                    "threshold": float(h),
                                }
                            )
                    self._roc_df = pd.DataFrame(rows)
            except Exception:
                pass

        return self

    def print_summary(self, title: str = "Model") -> None:
        print(f"\n=== {title} ===")
        print(f"Accuracy     : {self.metrics_.get('accuracy', float('nan')):.6f}")
        print(f"F1 (Macro)   : {self.metrics_.get('f1_macro', float('nan')):.6f}")
        print(f"F1 (Weighted): {self.metrics_.get('f1_weighted', float('nan')):.6f}")
        if "roc_auc" in self.metrics_:
            print(f"ROC AUC      : {self.metrics_['roc_auc']:.6f}")
        if "roc_auc_ovr_weighted" in self.metrics_:
            print(f"ROC AUC (OVR): {self.metrics_['roc_auc_ovr_weighted']:.6f}")
        print("\nClassification report:")
        cr = self.metrics_.get("classification_report", {})
        if isinstance(cr, dict):
            for cls in self.class_names_:
                row = cr.get(str(cls), {})
                print(
                    f"  {str(cls):<20} "
                    f"P={row.get('precision', 0):.4f}  "
                    f"R={row.get('recall', 0):.4f}  "
                    f"F1={row.get('f1-score', 0):.4f}  "
                    f"n={int(row.get('support', 0))}"
                )

    def export(
        self,
        out_dir: Path,
        pred_df: pd.DataFrame,
        report_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exportable: Dict[str, Any] = dict(report_meta or {})
        for k, v in self.metrics_.items():
            if k == "classification_report" and isinstance(v, dict):
                exportable[k] = v
            else:
                exportable[k] = v

        report_path = out_dir / "evaluation_report.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(exportable, fh, indent=2)

        pred_path = out_dir / "predictions_detailed.csv"
        pred_df.to_csv(pred_path, index=False)

        if "confusion_matrix" in self.metrics_:
            cm_df = pd.DataFrame(
                self.metrics_["confusion_matrix"],
                index=[f"true_{c}" for c in self.class_names_],
                columns=[f"pred_{c}" for c in self.class_names_],
            )
            cm_df.to_csv(out_dir / "confusion_matrix.csv")

        if (
            "classification_report" in self.metrics_
            and isinstance(self.metrics_["classification_report"], dict)
        ):
            cr = self.metrics_["classification_report"]
            per_class_rows = [
                {
                    "class": str(cls),
                    "precision": cr.get(str(cls), {}).get("precision"),
                    "recall": cr.get(str(cls), {}).get("recall"),
                    "f1_score": cr.get(str(cls), {}).get("f1-score"),
                    "support": cr.get(str(cls), {}).get("support"),
                }
                for cls in self.class_names_
            ]
            pd.DataFrame(per_class_rows).to_csv(
                out_dir / "per_class_metrics.csv", index=False
            )

        if self._roc_df is not None:
            self._roc_df.to_csv(out_dir / "roc_curve_points.csv", index=False)

        print("\nSaved evaluation artifacts:")
        print(f"  - {report_path.as_posix()}")
        print(f"  - {pred_path.as_posix()}")
        if "confusion_matrix" in self.metrics_:
            print(f"  - {(out_dir / 'confusion_matrix.csv').as_posix()}")
            print(f"  - {(out_dir / 'per_class_metrics.csv').as_posix()}")
        if self._roc_df is not None:
            print(f"  - {(out_dir / 'roc_curve_points.csv').as_posix()}")


# Stage 4 - Prediction Module
class Predictor:
    #Loads a saved NIDS model bundle and exposes prediction methods.

    def __init__(self) -> None:
        self.model: Optional[Pipeline] = None
        self.encoder: Optional[LabelEncoder] = None
        self.metadata: Dict[str, Any] = {}
        self.selected_model: str = ""

    @classmethod
    def from_bundle(cls, path: Path) -> "Predictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")

        bundle: Dict[str, Any] = joblib.load(path)

        required = ("model", "label_encoder")
        missing = [k for k in required if k not in bundle]
        if missing:
            raise ValueError(
                f"Bundle at '{path}' is missing required keys: {missing}"
            )

        pred = cls()
        pred.model = bundle["model"]
        pred.encoder = bundle["label_encoder"]
        pred.metadata = bundle.get("metadata", {})
        pred.selected_model = bundle.get("selected_model", "unknown")
        return pred

    def predict(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.encoder is None:
            raise RuntimeError(
                "Predictor is not initialised. Call Predictor.from_bundle() first."
            )
        pred_ids: np.ndarray = self.model.predict(X)
        pred_labels: np.ndarray = self.encoder.inverse_transform(pred_ids)
        return pred_ids, pred_labels

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.model is None:
            raise RuntimeError(
                "Predictor is not initialised. Call Predictor.from_bundle() first."
            )
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                pass
        return None

    @property
    def label_column(self) -> Optional[str]:
        return self.metadata.get("label_column")

    @property
    def class_names(self) -> List[str]:
        return self.metadata.get("class_names", [])

    @property
    def binary_mode(self) -> bool:
        return bool(self.metadata.get("binary_mode", False))
