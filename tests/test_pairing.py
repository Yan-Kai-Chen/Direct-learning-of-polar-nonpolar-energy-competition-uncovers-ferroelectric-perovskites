from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIR_SRC = ROOT / "Polar-Nonpolar_pair_model" / "src"
sys.path.insert(0, str(PAIR_SRC))

from pair_retriever.metrics import evaluate_topk_predictions
from pair_retriever.protocol import (
    SymmetryBiasConfig,
    apply_symmetry_bias,
    audit_polar_holdout,
    load_pair_table,
)


class PairingTests(unittest.TestCase):
    def test_public_polar_holdout(self) -> None:
        split_root = (
            ROOT
            / "Polar-Nonpolar_pair_model"
            / "pairing_ai_out_group_sym"
            / "splits"
        )
        audit = audit_polar_holdout(
            load_pair_table(split_root / "train_pairs_pos.csv"),
            load_pair_table(split_root / "test_pairs_pos.csv"),
        )
        self.assertTrue(audit.polar_holdout_passed)
        self.assertEqual(audit.n_polars_train, 810)
        self.assertEqual(audit.n_polars_test, 203)
        self.assertEqual(audit.polar_overlap_train_test, 0)
        self.assertEqual(audit.nonpolar_overlap_train_test, 321)

    def test_symmetry_bias_and_ranking(self) -> None:
        adjusted = apply_symmetry_bias(
            [0.2, 0.2],
            polar_operation_count=24,
            candidate_operation_counts=[24, 216],
            config=SymmetryBiasConfig(),
        )
        np.testing.assert_allclose(adjusted, [0.2, 0.35], atol=1e-7)

        truth = pd.DataFrame(
            {
                "Polar_mpid": ["p1", "p2"],
                "NPolar_mpid": ["n2", "n3"],
            }
        )
        ranked = pd.DataFrame(
            {
                "Polar_mpid": ["p1", "p1", "p2", "p2"],
                "Rank": [1, 2, 1, 2],
                "Pred_NPolar_mpid": ["n1", "n2", "n3", "n4"],
                "Has_GT_in_pool": [1, 1, 1, 1],
            }
        )
        audit = evaluate_topk_predictions(truth, ranked, ks=(1, 2))
        self.assertEqual(audit.hit_at[1], 0.5)
        self.assertEqual(audit.hit_at[2], 1.0)
        self.assertEqual(audit.mrr, 0.75)


if __name__ == "__main__":
    unittest.main()
