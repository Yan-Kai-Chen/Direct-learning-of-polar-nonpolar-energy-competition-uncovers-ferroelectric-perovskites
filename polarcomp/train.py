from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


from polarcomp.training import TrainingConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AE3GNN on the fixed split stored in the input CSV."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional bounded override for testing a configuration.",
    )
    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    summary = run_training(
        data_path=args.data,
        structure_dir=args.structures,
        output_dir=args.output,
        config=config,
    )
    print(
        json.dumps(
            {
                "best_epoch": summary["best_epoch"],
                "metrics": summary["metrics"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
