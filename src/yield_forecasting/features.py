from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    assumption_flags: list[str]


def apply_assumptions(frame: pd.DataFrame, drivers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Impute non-reported drivers using known history, then segment medians."""
    result = frame.sort_values(["segment", "product", "date"]).copy()
    flags: list[str] = []
    global_medians = result[drivers].median(numeric_only=True)
    for column in drivers:
        flag = f"{column}__assumed"
        flags.append(flag)
        result[flag] = result[column].isna().astype("int8")
        result[column] = result.groupby(["segment", "product"], sort=False)[column].ffill()
        segment_median = result.groupby("segment")[column].transform("median")
        result[column] = result[column].fillna(segment_median).fillna(global_medians[column])
    return result, flags


def build_features(
    frame: pd.DataFrame, config: dict[str, Any], drop_incomplete: bool = True
) -> FeatureResult:
    fcfg = config["features"]
    result, flags = apply_assumptions(frame, fcfg["contemporaneous"])
    groups = result.groupby(["segment", "product"], sort=False)
    generated: list[str] = []
    for source in fcfg["lag_sources"]:
        for lag in fcfg["lags"]:
            name = f"{source}__lag_{lag}"
            result[name] = groups[source].shift(lag)
            generated.append(name)
        shifted = groups[source].shift(1)
        for window in fcfg["rolling_windows"]:
            name = f"{source}__rolling_mean_{window}"
            result[name] = (
                shifted.groupby([result["segment"], result["product"]], sort=False)
                .rolling(window, min_periods=window)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            generated.append(name)
    result["month_sin"] = pd.Series(
        __import__("numpy").sin(2 * __import__("numpy").pi * result["date"].dt.month / 12),
        index=result.index,
    )
    result["month_cos"] = pd.Series(
        __import__("numpy").cos(2 * __import__("numpy").pi * result["date"].dt.month / 12),
        index=result.index,
    )
    generated.extend(["month_sin", "month_cos"])
    features = fcfg["categorical"] + fcfg["contemporaneous"] + flags + generated
    if drop_incomplete:
        result = result.dropna(subset=generated).reset_index(drop=True)
    return FeatureResult(result, features, flags)


def chronological_split(
    frame: pd.DataFrame, validation_periods: int, test_periods: int
) -> dict[str, pd.DataFrame]:
    dates = sorted(frame["date"].unique())
    if len(dates) <= validation_periods + test_periods:
        raise ValueError("Not enough periods for requested train/validation/test split")
    val_start = dates[-(validation_periods + test_periods)]
    test_start = dates[-test_periods]
    return {
        "train": frame[frame["date"] < val_start].copy(),
        "validation": frame[(frame["date"] >= val_start) & (frame["date"] < test_start)].copy(),
        "test": frame[frame["date"] >= test_start].copy(),
    }
