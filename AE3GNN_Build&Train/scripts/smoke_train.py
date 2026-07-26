from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
SRC = MODULE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ae3gnn.training import one_step_smoke


def toy_graph(scale: float) -> dict[str, np.ndarray]:
    return {
        "z": np.asarray([8, 22, 8], dtype=np.int64),
        "edge_index": np.asarray(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=np.int64,
        ),
        "edge_vec": scale
        * np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "edge_dist": scale
        * np.asarray(
            [1.0, 1.0, 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            dtype=np.float32,
        ),
        "triplet_center": np.asarray([0, 1, 2], dtype=np.int64),
        "triplet_e1": np.asarray([0, 2, 4], dtype=np.int64),
        "triplet_e2": np.asarray([1, 3, 5], dtype=np.int64),
        "triplet_cos": np.asarray(
            [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            dtype=np.float32,
        ),
    }


def main() -> None:
    report = one_step_smoke(
        [toy_graph(1.00), toy_graph(1.05)],
        [toy_graph(1.10), toy_graph(0.95)],
        np.asarray([25.0, -40.0], dtype=np.float32),
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
