#!/usr/bin/env python
"""
Real-time NIDS inference API (FastAPI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SinglePredictRequest(BaseModel):
    record: Dict[str, Any] = Field(
        ..., description="Single network-flow feature dictionary."
    )
    include_probabilities: bool = Field(
        default=False, description="Include class probability scores."
    )


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ..., description="List of network-flow feature dictionaries."
    )
    include_probabilities: bool = Field(
        default=False, description="Include class probability scores."
    )


class NIDSModelService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.metadata: Dict[str, Any] = bundle.get("metadata", {})
        self.selected_model = bundle.get("selected_model", "unknown")
        self.model_path = model_path

        self.label_col = self.metadata.get("label_column")
        self.class_names = [str(c) for c in self.label_encoder.classes_]
        self.feature_names = self._infer_feature_names()

    def _infer_feature_names(self) -> List[str]:
        if hasattr(self.model, "feature_names_in_"):
            return [str(c) for c in self.model.feature_names_in_]

        if hasattr(self.model, "named_steps"):
            for step_name in ["imputer", "scaler", "classifier"]:
                step = self.model.named_steps.get(step_name)
                if step is not None and hasattr(step, "feature_names_in_"):
                    return [str(c) for c in step.feature_names_in_]

        raise ValueError(
            "Could not infer training feature names from model. "
            "Retrain model with dataframe input."
        )

    def _prepare_frame(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ValueError("No records provided.")

        df = pd.DataFrame(records)
        if self.label_col and self.label_col in df.columns:
            df = df.drop(columns=[self.label_col])

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self.feature_names].replace([np.inf, -np.inf], np.nan)
        return df

    def predict(
        self, records: List[Dict[str, Any]], include_probabilities: bool
    ) -> List[Dict[str, Any]]:
        x = self._prepare_frame(records)
        pred_ids = self.model.predict(x)
        pred_labels = self.label_encoder.inverse_transform(pred_ids)

        probs = None
        if include_probabilities and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)

        outputs: List[Dict[str, Any]] = []
        for i in range(len(pred_ids)):
            item: Dict[str, Any] = {
                "prediction_id": int(pred_ids[i]),
                "prediction_label": str(pred_labels[i]),
            }
            if probs is not None:
                item["probabilities"] = {
                    self.class_names[j]: float(probs[i, j])
                    for j in range(len(self.class_names))
                }
            outputs.append(item)
        return outputs


MODEL_PATH = Path(os.getenv("NIDS_MODEL_PATH", "artifacts_baseline/nids_model.joblib"))
service = NIDSModelService(MODEL_PATH)

app = FastAPI(
    title="NIDS Real-time API",
    version="1.0.0",
    description="Inference API for AdaBoost + C4.5-style NIDS model.",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": service.model_path.as_posix(),
        "selected_model": service.selected_model,
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "model_path": service.model_path.as_posix(),
        "selected_model": service.selected_model,
        "label_column": service.label_col,
        "classes": service.class_names,
        "feature_count": len(service.feature_names),
        "metadata": service.metadata,
    }


@app.post("/predict")
def predict_single(payload: SinglePredictRequest) -> Dict[str, Any]:
    try:
        result = service.predict(
            records=[payload.record],
            include_probabilities=payload.include_probabilities,
        )[0]
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictRequest) -> Dict[str, Any]:
    try:
        results = service.predict(
            records=payload.records,
            include_probabilities=payload.include_probabilities,
        )
        return {"count": len(results), "predictions": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
