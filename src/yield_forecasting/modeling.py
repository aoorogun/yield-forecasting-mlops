from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import file_sha256, load_data
from .features import build_features, chronological_split


@dataclass
class TrainResult:
    model: Pipeline
    metadata: dict[str, Any]
    predictions: pd.DataFrame


def _estimator(name: str, seed: int) -> Any:
    if name == "elastic_net":
        # Target standardisation prevents the differently-scaled KPIs from receiving
        # materially different effective regularisation.
        return TransformedTargetRegressor(
            regressor=ElasticNet(alpha=0.003, l1_ratio=0.15, max_iter=5000, random_state=seed),
            transformer=StandardScaler(),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=240,
            max_leaf_nodes=25,
            l2_regularization=0.1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model candidate: {name}")


def make_pipeline(name: str, categorical: list[str], numeric: list[str], seed: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("model", MultiOutputRegressor(_estimator(name, seed), n_jobs=-1)),
        ]
    )


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, targets: list[str]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    maes = []
    for idx, target in enumerate(targets):
        mae = mean_absolute_error(actual[:, idx], predicted[:, idx])
        metrics[f"{target}_mae"] = float(mae)
        metrics[f"{target}_rmse"] = float(
            np.sqrt(mean_squared_error(actual[:, idx], predicted[:, idx]))
        )
        metrics[f"{target}_r2"] = float(r2_score(actual[:, idx], predicted[:, idx]))
        # Normalised MAE makes different target scales comparable during selection.
        scale = np.ptp(actual[:, idx]) or 1.0
        maes.append(mae / scale)
    metrics["mean_mae"] = float(np.mean(maes))
    return metrics


def train(config: dict[str, Any]) -> TrainResult:
    frame = load_data(config["paths"]["raw_data"])
    engineered = build_features(frame, config)
    splits = chronological_split(
        engineered.frame, config["split"]["validation_periods"], config["split"]["test_periods"]
    )
    features = engineered.feature_columns
    targets = config["features"]["targets"]
    categorical = config["features"]["categorical"]
    numeric = [c for c in features if c not in categorical]
    seed = config["project"]["random_seed"]

    candidates: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name in config["training"]["candidates"]:
        pipeline = make_pipeline(name, categorical, numeric, seed)
        pipeline.fit(splits["train"][features], splits["train"][targets])
        prediction = pipeline.predict(splits["validation"][features])
        candidates[name] = regression_metrics(
            splits["validation"][targets].to_numpy(), prediction, targets
        )
        fitted[name] = pipeline
    best_name = min(
        candidates, key=lambda name: candidates[name][config["training"]["selection_metric"]]
    )

    train_validation = pd.concat([splits["train"], splits["validation"]]).sort_values("date")
    best = make_pipeline(best_name, categorical, numeric, seed)
    best.fit(train_validation[features], train_validation[targets])
    test_pred = best.predict(splits["test"][features])
    test_metrics = regression_metrics(splits["test"][targets].to_numpy(), test_pred, targets)
    naive = splits["test"][[f"{t}__lag_1" for t in targets]].to_numpy()
    naive_metrics = regression_metrics(splits["test"][targets].to_numpy(), naive, targets)

    # Empirical central interval based on validation residuals from the selected pre-refit model.
    val_pred = fitted[best_name].predict(splits["validation"][features])
    residuals = splits["validation"][targets].to_numpy() - val_pred
    alpha = config["training"]["prediction_interval_alpha"]
    interval_offsets = {
        target: [
            float(np.quantile(residuals[:, i], alpha / 2)),
            float(np.quantile(residuals[:, i], 1 - alpha / 2)),
        ]
        for i, target in enumerate(targets)
    }
    prediction_frame = splits["test"][["date", "segment", "product", *targets]].copy()
    for idx, target in enumerate(targets):
        prediction_frame[f"predicted_{target}"] = test_pred[:, idx]

    metadata = {
        "model_version": config["project"]["model_version"],
        "model_type": best_name,
        "features": features,
        "targets": targets,
        "categorical_features": categorical,
        "candidate_validation_metrics": candidates,
        "test_metrics": test_metrics,
        "naive_test_metrics": naive_metrics,
        "prediction_interval_offsets": interval_offsets,
        "train_end": str(train_validation["date"].max().date()),
        "test_start": str(splits["test"]["date"].min().date()),
        "test_end": str(splits["test"]["date"].max().date()),
        "training_rows": len(train_validation),
        "test_rows": len(splits["test"]),
        "data_sha256": file_sha256(config["paths"]["raw_data"]),
    }
    artifact_dir = Path(config["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    prediction_frame.to_csv(artifact_dir / "test_predictions.csv", index=False)
    train_validation[[*features]].to_csv(artifact_dir / "reference.csv", index=False)
    return TrainResult(best, metadata, prediction_frame)


def load_artifacts(artifact_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    path = Path(artifact_dir)
    model = joblib.load(path / "model.joblib")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata
