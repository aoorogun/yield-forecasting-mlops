from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import generate_synthetic_data
from .modeling import train
from .monitoring import monitor
from .scenario import reforecast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yield forecasting platform")
    parser.add_argument("--config", default=None, help="Path to YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate-data")
    commands.add_parser("train")
    commands.add_parser("evaluate")
    scenario = commands.add_parser("reforecast")
    scenario.add_argument("--scenario", required=True)
    commands.add_parser("monitor")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "generate-data":
        frame = generate_synthetic_data(config)
        print(f"Generated {len(frame):,} rows at {config['paths']['raw_data']}")
    elif args.command == "train":
        result = train(config)
        print(
            json.dumps(
                {
                    "selected_model": result.metadata["model_type"],
                    "test_metrics": result.metadata["test_metrics"],
                },
                indent=2,
            )
        )
    elif args.command == "evaluate":
        metadata = json.loads((Path(config["paths"]["artifact_dir"]) / "metadata.json").read_text())
        print(
            json.dumps(
                {
                    "model": metadata["model_type"],
                    "test": metadata["test_metrics"],
                    "naive": metadata["naive_test_metrics"],
                },
                indent=2,
            )
        )
    elif args.command == "reforecast":
        output = reforecast(config, args.scenario)
        deltas = [c for c in output if c.startswith("delta_")]
        print(output.loc[output["affected"], deltas].mean().to_json(indent=2))
    elif args.command == "monitor":
        print(json.dumps(monitor(config), indent=2))


if __name__ == "__main__":
    main()
