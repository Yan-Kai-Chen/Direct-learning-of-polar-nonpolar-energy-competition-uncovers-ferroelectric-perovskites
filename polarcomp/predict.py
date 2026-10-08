from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


from polarcomp.graph import GraphStore
from polarcomp.inference import EnergyPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict polar-nonpolar energy differences from a GNN bundle."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path)
    parser.add_argument("--graph-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = EnergyPredictor(args.bundle, device=args.device)
    store = GraphStore(
        args.structures,
        args.graph_cache,
        predictor.graph_config,
    )
    frame = pd.read_csv(args.data, low_memory=False)
    result = predictor.predict(frame, store, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
