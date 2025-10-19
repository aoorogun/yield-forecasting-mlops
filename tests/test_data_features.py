from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from yield_forecasting.config import load_config
from yield_forecasting.data import generate_synthetic_data, load_data
from yield_forecasting.features import build_features, chronological_split


class DataFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_generation_is_reproducible_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = generate_synthetic_data(self.config, Path(directory) / "a.csv")
            second = generate_synthetic_data(self.config, Path(directory) / "b.csv")
            self.assertEqual(len(first), 432)
            self.assertTrue(first.equals(second))
            loaded = load_data(Path(directory) / "a.csv")
            self.assertEqual(len(loaded), len(first))

    def test_lag_features_do_not_use_current_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = generate_synthetic_data(self.config, Path(directory) / "data.csv")
            featured = build_features(frame, self.config).frame
            series = featured[
                (featured.segment == "consumer") & (featured.product == "fixed_income")
            ]
            original = frame[(frame.segment == "consumer") & (frame.product == "fixed_income")]
            by_date = original.set_index("date")["gross_yield"]
            expected = series["date"].map(
                lambda date: by_date.loc[date - __import__("pandas").DateOffset(months=1)]
            )
            np.testing.assert_allclose(
                series["gross_yield__lag_1"].to_numpy(dtype=float),
                expected.to_numpy(dtype=float),
            )

    def test_chronological_split_has_no_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = generate_synthetic_data(self.config, Path(directory) / "data.csv")
            featured = build_features(frame, self.config).frame
            split = chronological_split(featured, 12, 12)
            self.assertLess(split["train"].date.max(), split["validation"].date.min())
            self.assertLess(split["validation"].date.max(), split["test"].date.min())


if __name__ == "__main__":
    unittest.main()
