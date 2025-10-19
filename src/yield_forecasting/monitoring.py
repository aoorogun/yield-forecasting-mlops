from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_data
from .features import build_features


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI with bins learned only from the reference population."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def monitor(config: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(config["paths"]["artifact_dir"])
    reference = pd.read_csv(artifact_dir / "reference.csv")
    current = build_features(load_data(config["paths"]["raw_data"]), config).frame
    numeric = [c for c in config["features"]["contemporaneous"] if c in reference]
    scores = {
        c: population_stability_index(reference[c], current[c], config["monitoring"]["bins"])
        for c in numeric
    }
    maximum = max(scores.values(), default=0.0)
    warning, critical = config["monitoring"]["psi_warning"], config["monitoring"]["psi_critical"]
    status = "critical" if maximum >= critical else "warning" if maximum >= warning else "healthy"
    report = {
        "status": status,
        "maximum_psi": maximum,
        "feature_psi": scores,
        "thresholds": {"warning": warning, "critical": critical},
    }
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
