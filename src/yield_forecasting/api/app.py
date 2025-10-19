from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import load_config
from ..modeling import load_artifacts


class ForecastRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment: str
    product: str
    base_rate: float
    inflation: float
    unemployment: float
    market_volatility: float = Field(ge=0)
    assets_mn: float = Field(gt=0)
    net_flows_mn: float
    fee_bps: float = Field(ge=0)
    duration_years: float = Field(ge=0)
    credit_spread_bps: float = Field(ge=0)
    digital_engagement: float = Field(ge=0, le=1)
    month_sin: float
    month_cos: float
    # Serving expects engineered historical features supplied by the feature store.
    historical_features: dict[str, float]


class ForecastRequest(BaseModel):
    records: list[ForecastRecord] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_categories(self) -> ForecastRequest:
        if any(not record.segment.strip() or not record.product.strip() for record in self.records):
            raise ValueError("segment and product must not be blank")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(os.getenv("YF_CONFIG"))
    artifact_dir = os.getenv("YF_MODEL_DIR", config["paths"]["artifact_dir"])
    try:
        app.state.model, app.state.metadata = load_artifacts(artifact_dir)
        app.state.ready = True
    except FileNotFoundError:
        app.state.ready = False
    yield


app = FastAPI(title="Yield Forecasting API", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    return {"status": "ready", "model_version": request.app.state.metadata["model_version"]}


@app.post("/v1/forecast")
def forecast(payload: ForecastRequest, request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    metadata = request.app.state.metadata
    rows = []
    for record in payload.records:
        row = record.model_dump(exclude={"historical_features"})
        row.update(record.historical_features)
        rows.append(row)
    frame = pd.DataFrame(rows)
    missing = sorted(set(metadata["features"]) - set(frame.columns))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing engineered features: {missing}")
    predictions = request.app.state.model.predict(frame[metadata["features"]])
    response = []
    for row in predictions:
        item: dict[str, Any] = {}
        for idx, target in enumerate(metadata["targets"]):
            lower, upper = metadata["prediction_interval_offsets"][target]
            item[target] = float(row[idx])
            item[f"{target}_lower"] = float(row[idx] + lower)
            item[f"{target}_upper"] = float(row[idx] + upper)
        response.append(item)
    return {"model_version": metadata["model_version"], "predictions": response}
