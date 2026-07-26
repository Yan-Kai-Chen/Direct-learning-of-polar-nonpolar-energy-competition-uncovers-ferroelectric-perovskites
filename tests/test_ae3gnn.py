from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms


ROOT = Path(__file__).resolve().parents[1]
AE3GNN_SRC = ROOT / "AE3GNN_Build&Train" / "src"
sys.path.insert(0, str(AE3GNN_SRC))

from ae3gnn.graph import (
    GraphBuildConfig,
    build_periodic_angle_graph,
    validate_graph,
)
from ae3gnn.model import ModelConfig, PairEnergyModel


class AE3GNNTests(unittest.TestCase):
    def test_periodic_graph_and_forward(self) -> None:
        atoms = Atoms(
            symbols=["Ba", "Ti", "O"],
            positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [1.5, 1.5, 0.0]],
            cell=[3.0, 3.0, 3.0],
            pbc=True,
        )
        graph = build_periodic_angle_graph(
            atoms,
            GraphBuildConfig(r_max=2.7, max_neighbors=12, k_angle=6),
        )
        validate_graph(graph)
        self.assertEqual(graph["edge_index"].shape[0], 2)
        self.assertGreater(graph["edge_index"].shape[1], 0)
        self.assertTrue(np.isfinite(graph["edge_dist"]).all())

        model = PairEnergyModel(
            ModelConfig(
                dim=16,
                n_layers=1,
                rbf_dim=8,
                dropout=0.0,
                angle_hidden=16,
            )
        )
        prediction = model([graph], [graph])
        self.assertEqual(tuple(prediction.shape), (1,))
        self.assertTrue(bool(torch.isfinite(prediction).all()))


if __name__ == "__main__":
    unittest.main()
