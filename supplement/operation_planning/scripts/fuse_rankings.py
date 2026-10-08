"""Fuse saved language and graph operation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from polargen.fusion import fuse_operation_scores, select_top_operations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    with np.load(args.scores, allow_pickle=False) as payload:
        operation_ids = [str(value) for value in payload["operation_ids"]]
        branches = {
            name: np.asarray(payload[name], dtype=np.float32)
            for name in config["branches"]
        }
    fused = fuse_operation_scores(branches, config["branches"])
    selected = select_top_operations(fused, operation_ids, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        operation_ids=np.asarray(operation_ids, dtype=np.str_),
        fused_scores=fused,
        selected=np.asarray(selected, dtype=np.str_),
    )
    print(f"PASS rows={len(fused)} output={args.output}")


if __name__ == "__main__":
    main()
