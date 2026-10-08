from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
from polarcomp.data import audit_training_table, select_numeric_features


class PublicDataTests(unittest.TestCase):
    def test_fixed_training_split(self) -> None:
        frame, audit = audit_training_table(ROOT / "Train_EXAMPLE.csv")
        self.assertTrue(audit.passed)
        self.assertEqual(audit.rows, 3238)
        self.assertEqual(
            audit.split_counts,
            {"train": 2531, "val": 398, "test": 309},
        )
        self.assertEqual(audit.polar_overlap_train_test, 0)
        self.assertEqual(audit.duplicate_pairs, 0)

        features = select_numeric_features(frame)
        for forbidden in (
            "row_idx",
            "split",
            "Energy_diff_meV",
            "Polar_mpid",
            "NPolar_mpid",
        ):
            self.assertNotIn(forbidden, features.columns)


if __name__ == "__main__":
    unittest.main()
