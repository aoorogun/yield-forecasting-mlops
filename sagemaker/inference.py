"""SageMaker inference-toolkit compatible entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def model_fn(model_dir: str) -> dict[str, Any]:
    root = Path(model_dir)
    return {
        "model": joblib.load(root / "model.joblib"),
        "metadata": json.loads((root / "metadata.json").read_text(encoding="utf-8")),
    }


def input_fn(request_body: str, content_type: str) -> pd.DataFrame:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("JSON payload requires a non-empty records list")
    return pd.DataFrame(records)


def predict_fn(data: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = bundle["metadata"]
    missing = sorted(set(metadata["features"]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    predictions = bundle["model"].predict(data[metadata["features"]])
    return {
        "model_version": metadata["model_version"],
        "targets": metadata["targets"],
        "predictions": predictions.tolist(),
    }


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), accept
