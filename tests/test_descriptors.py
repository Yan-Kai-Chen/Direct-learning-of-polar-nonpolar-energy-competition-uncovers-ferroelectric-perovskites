from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_ROOT = ROOT / "Descriptor_engineering"
sys.path.insert(0, str(DESCRIPTOR_ROOT))

from default_rules import (
    A_GEOM_RULES,
    B_GEOM_RULES,
    DERIVED_RULES,
    ELEMENT_RULES,
    EWALD_RULES,
    EXPORT_RULES,
    SITE_RULES,
)
from derived_features import run_derived_features_stage
from public_api import DescriptorPipeline, PipelinePaths
from site_assignment import run_site_assignment_stage


class DescriptorTests(unittest.TestCase):
    def test_public_site_rule_and_derived_operation(self) -> None:
        assigned = run_site_assignment_stage(
            pd.DataFrame({"Polar_pretty_formula": ["BaTiO3"]}),
            SITE_RULES,
        )
        self.assertEqual(assigned.loc[0, "$A_{site}$"], "Ba")
        self.assertEqual(assigned.loc[0, "$B_{site}$"], "Ti")
        self.assertEqual(assigned.loc[0, "$X_{site}$"], "O")

        derived = run_derived_features_stage(
            pd.DataFrame({"A": [1.0], "X": [3.5]}),
            {
                "operations": [
                    {
                        "output": "mismatch",
                        "left": "A",
                        "right": "X",
                        "operation": "absolute_difference",
                    }
                ]
            },
        )
        self.assertEqual(derived.loc[0, "mismatch"], 2.5)

    def test_small_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            structures = root / "structures"
            structures.mkdir()
            cubic = Structure(
                Lattice.cubic(4.0),
                ["Ba", "Ti", "O", "O", "O"],
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
            )
            polar = cubic.copy()
            polar.translate_sites([1], [0.01, 0.0, 0.0], frac_coords=True)
            polar.to(filename=structures / "p1.cif")
            cubic.to(filename=structures / "n1.cif")

            pairs = root / "pairs.csv"
            pd.DataFrame(
                {
                    "Polar_mpid": ["p1"],
                    "NPolar_mpid": ["n1"],
                    "Polar_pretty_formula": ["BaTiO3"],
                }
            ).to_csv(pairs, index=False)
            properties = root / "elements.csv"
            pd.DataFrame(
                {
                    "symbol": ["Ba", "Ti", "O"],
                    "electronegativity": [0.89, 1.54, 3.44],
                }
            ).to_csv(properties, index=False)

            output = DescriptorPipeline(
                paths=PipelinePaths(
                    input_pair_csv=pairs,
                    element_property_csv=properties,
                    structure_dir=structures,
                    work_dir=root / "work",
                    output_dir=root / "output",
                ),
                site_rules=SITE_RULES,
                element_rules=ELEMENT_RULES,
                a_geom_rules=A_GEOM_RULES,
                b_geom_rules=B_GEOM_RULES,
                ewald_rules=EWALD_RULES,
                derived_rules=DERIVED_RULES,
                export_rules=EXPORT_RULES,
            ).run_all()
            final = pd.read_csv(output.final_feature_csv)
            self.assertEqual(len(final), 1)
            self.assertIn("d_B_offcenter_mean", final.columns)
            self.assertIn("AX_chi_mismatch", final.columns)
            self.assertFalse(pd.isna(final.loc[0, "AX_chi_mismatch"]))


if __name__ == "__main__":
    unittest.main()
