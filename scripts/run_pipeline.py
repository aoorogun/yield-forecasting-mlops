"""Local and SageMaker Processing-compatible pipeline entry point."""

import sys
from pathlib import Path

# SageMaker uploads the repository as source rather than installing it as a wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yield_forecasting.config import load_config  # noqa: E402
from yield_forecasting.data import generate_synthetic_data  # noqa: E402
from yield_forecasting.modeling import train  # noqa: E402
from yield_forecasting.monitoring import monitor  # noqa: E402
from yield_forecasting.scenario import reforecast  # noqa: E402


def main() -> None:
    config = load_config()
    generate_synthetic_data(config)
    result = train(config)
    reforecast(config, "configs/scenario_example.json")
    drift = monitor(config)
    print(
        {
            "model": result.metadata["model_type"],
            "metrics": result.metadata["test_metrics"],
            "drift_status": drift["status"],
        }
    )


if __name__ == "__main__":
    main()
