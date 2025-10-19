from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_data
from .features import build_features
from .modeling import load_artifacts

ALLOWED_OPERATIONS = {"add", "multiply", "set"}


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    if scenario.get("status") not in {"proposed", "committed"}:
        raise ValueError("Scenario status must be proposed or committed")
    for column, change in scenario.get("changes", {}).items():
        if change.get("operation") not in ALLOWED_OPERATIONS:
            raise ValueError(f"Unsupported operation for {column}: {change.get('operation')}")
        if not isinstance(change.get("value"), (int, float)):
            raise ValueError(f"Scenario value for {column} must be numeric")
    return scenario


def apply_scenario(frame: pd.DataFrame, scenario: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    result = frame.copy()
    mask = pd.to_datetime(result["date"]) >= pd.Timestamp(scenario["effective_from"])
    for column, values in scenario.get("filters", {}).items():
        if column not in result:
            raise ValueError(f"Unknown filter column: {column}")
        mask &= result[column].isin(values)
    for column, change in scenario["changes"].items():
        if column not in result:
            raise ValueError(f"Unknown change column: {column}")
        operation, value = change["operation"], change["value"]
        if operation == "add":
            result.loc[mask, column] = result.loc[mask, column] + value
        elif operation == "multiply":
            result.loc[mask, column] = result.loc[mask, column] * value
        else:
            result.loc[mask, column] = value
    return result, mask


def reforecast(config: dict[str, Any], scenario_path: str | Path) -> pd.DataFrame:
    raw = load_data(config["paths"]["raw_data"])
    scenario = load_scenario(scenario_path)
    changed, affected_raw = apply_scenario(raw, scenario)
    baseline_features = build_features(raw, config)
    scenario_features = build_features(changed, config)
    model, metadata = load_artifacts(config["paths"]["artifact_dir"])
    features, targets = metadata["features"], metadata["targets"]
    base_pred = model.predict(baseline_features.frame[features])
    scenario_pred = model.predict(scenario_features.frame[features])

    output = baseline_features.frame[["date", "segment", "product"]].copy()
    # Feature construction drops initial history; realign affected status by business key.
    affected_keys = changed.loc[affected_raw, ["date", "segment", "product"]].assign(affected=True)
    output = output.merge(affected_keys, on=["date", "segment", "product"], how="left")
    output["affected"] = output["affected"].fillna(False).astype(bool)
    output["scenario_id"] = scenario["scenario_id"]
    for idx, target in enumerate(targets):
        output[f"baseline_{target}"] = base_pred[:, idx]
        output[f"scenario_{target}"] = scenario_pred[:, idx]
        output[f"delta_{target}"] = scenario_pred[:, idx] - base_pred[:, idx]
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(report_dir / "scenario_impact.csv", index=False)
    return output
