from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yield_forecasting.config import load_config
from yield_forecasting.data import generate_synthetic_data
from yield_forecasting.modeling import load_artifacts, train


class IntegrationTests(unittest.TestCase):
    def test_training_persists_loadable_artifacts(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config["paths"]["raw_data"] = str(root / "data.csv")
            config["paths"]["artifact_dir"] = str(root / "artifacts")
            config["training"]["candidates"] = ["elastic_net"]
            generate_synthetic_data(config)
            result = train(config)
            model, metadata = load_artifacts(config["paths"]["artifact_dir"])
            self.assertIsNotNone(model)
            self.assertEqual(metadata["model_type"], "elastic_net")
            self.assertEqual(len(result.predictions), 72)
            self.assertIn("gross_yield_mae", metadata["test_metrics"])


if __name__ == "__main__":
    unittest.main()
