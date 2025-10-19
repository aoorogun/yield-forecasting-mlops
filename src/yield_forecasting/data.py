from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "segment",
    "product",
    "base_rate",
    "inflation",
    "unemployment",
    "market_volatility",
    "assets_mn",
    "net_flows_mn",
    "fee_bps",
    "duration_years",
    "credit_spread_bps",
    "digital_engagement",
    "gross_yield",
    "strategic_score",
}


def generate_synthetic_data(
    config: dict[str, Any], output_path: str | Path | None = None
) -> pd.DataFrame:
    """Generate reproducible monthly panel data with realistic temporal relationships."""
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed)
    dcfg = config["data"]
    dates = pd.date_range(dcfg["start_date"], periods=dcfg["periods"], freq="MS")
    rows: list[dict[str, Any]] = []

    segment_effect = {"consumer": 0.004, "commercial": 0.008, "institutional": -0.001}
    product_effect = {"fixed_income": 0.006, "multi_asset": 0.011}
    for s_idx, segment in enumerate(dcfg["segments"]):
        for p_idx, product in enumerate(dcfg["products"]):
            prior_yield = 0.035 + segment_effect[segment] + product_effect[product]
            prior_score = 68.0 + 2 * s_idx - p_idx
            for t, date in enumerate(dates):
                cycle = np.sin(2 * np.pi * t / 24)
                base_rate = max(0.001, 0.012 + 0.00045 * t + 0.006 * cycle + rng.normal(0, 0.0015))
                inflation = max(
                    0.005, 0.021 + 0.007 * np.sin(2 * np.pi * (t + 5) / 30) + rng.normal(0, 0.002)
                )
                unemployment = max(
                    0.025, 0.052 - 0.00015 * t + 0.004 * cycle + rng.normal(0, 0.0015)
                )
                volatility = max(7, 17 + 5 * abs(np.sin(t / 7)) + rng.normal(0, 2))
                assets = 450 + 90 * s_idx + 60 * p_idx + 3.2 * t + rng.normal(0, 18)
                flows = 5 + 0.018 * assets - 0.7 * volatility + 3 * cycle + rng.normal(0, 5)
                fee_bps = 62 + 8 * s_idx + 5 * p_idx + rng.normal(0, 2)
                duration = 3.2 + 1.4 * (product == "fixed_income") + rng.normal(0, 0.25)
                spread = 95 + 13 * s_idx + 16 * cycle + 1.2 * volatility + rng.normal(0, 6)
                engagement = np.clip(
                    0.48 + 0.0045 * t - 0.04 * s_idx + rng.normal(0, 0.025), 0.2, 0.95
                )
                gross_yield = (
                    0.40 * prior_yield
                    + 0.45 * base_rate
                    + 0.000025 * spread
                    - 0.00002 * fee_bps
                    + segment_effect[segment]
                    + product_effect[product]
                    - 0.00012 * volatility
                    + rng.normal(0, 0.0018)
                )
                strategic_score = (
                    0.48 * prior_score
                    + 28 * engagement
                    + 0.10 * flows
                    - 0.20 * volatility
                    - 25 * unemployment
                    + 26
                    + rng.normal(0, 1.4)
                )
                rows.append(
                    {
                        "date": date,
                        "segment": segment,
                        "product": product,
                        "base_rate": base_rate,
                        "inflation": inflation,
                        "unemployment": unemployment,
                        "market_volatility": volatility,
                        "assets_mn": assets,
                        "net_flows_mn": flows,
                        "fee_bps": fee_bps,
                        "duration_years": duration,
                        "credit_spread_bps": spread,
                        "digital_engagement": engagement,
                        "gross_yield": gross_yield,
                        "strategic_score": strategic_score,
                    }
                )
                prior_yield, prior_score = gross_yield, strategic_score

    frame = pd.DataFrame(rows).sort_values(["segment", "product", "date"]).reset_index(drop=True)
    driver_columns = config["features"]["contemporaneous"]
    missing_mask = rng.random((len(frame), len(driver_columns))) < dcfg["missing_rate"]
    # Keep the first observation per series complete so assumptions always have a starting point.
    first_rows = frame.groupby(["segment", "product"], sort=False).head(1).index
    missing_mask[first_rows, :] = False
    frame.loc[:, driver_columns] = frame[driver_columns].mask(missing_mask)

    path = Path(output_path or config["paths"]["raw_data"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def validate_data(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input data is empty")
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("date contains invalid values")
    keys = ["date", "segment", "product"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate business keys found: {keys}")
    for target in ["gross_yield", "strategic_score"]:
        if frame[target].isna().any() or not np.isfinite(frame[target]).all():
            raise ValueError(f"Target {target} contains missing or non-finite values")


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    validate_data(frame)
    return frame.sort_values(["segment", "product", "date"]).reset_index(drop=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
