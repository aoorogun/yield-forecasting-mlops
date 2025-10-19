from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from yield_forecasting.monitoring import population_stability_index
from yield_forecasting.scenario import apply_scenario


class ScenarioMonitoringTests(unittest.TestCase):
    def test_scenario_only_changes_filtered_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "segment": ["consumer", "institutional"],
                "fee_bps": [60.0, 70.0],
            }
        )
        scenario = {
            "effective_from": "2024-01-01",
            "filters": {"segment": ["consumer"]},
            "changes": {"fee_bps": {"operation": "add", "value": -5}},
        }
        changed, mask = apply_scenario(frame, scenario)
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(changed.loc[0, "fee_bps"], 55.0)
        self.assertEqual(changed.loc[1, "fee_bps"], 70.0)

    def test_psi_detects_shift(self) -> None:
        rng = np.random.default_rng(42)
        reference = pd.Series(rng.normal(0, 1, 5000))
        stable = pd.Series(rng.normal(0, 1, 5000))
        shifted = pd.Series(rng.normal(2, 1, 5000))
        self.assertLess(population_stability_index(reference, stable), 0.1)
        self.assertGreater(population_stability_index(reference, shifted), 0.25)


if __name__ == "__main__":
    unittest.main()
