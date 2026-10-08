from __future__ import annotations

import importlib
import pkgutil
import unittest

import numpy as np

import polarevolve
from polarevolve.crystal.asu import expand_orbit
from polarevolve.crystal.program import OrbitSpec, compile_hard_condition
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase
from polarevolve.diffusion.metric import MEMBER_MEAN_CARTESIAN_V1, metric_contract_metadata


class PolarEvolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = polarevolve.asset_root()
        cls.groups = GroupDatabase(root / "group_database_v1")
        cls.wyckoff = WyckoffDatabase(root / "wyckoff_gauge_v1")

    def test_all_core_modules_import(self) -> None:
        for module in pkgutil.walk_packages(polarevolve.__path__, "polarevolve."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_fixed_cubic_program(self) -> None:
        compiled = compile_hard_condition(
            condition_id="synthetic_batio3", hall_number=517,
            orbit_specs=(OrbitSpec("Ba", "a"), OrbitSpec("Ti", "b"), OrbitSpec("O", "c")),
            base_cell_representation="conventional",
            group_database=self.groups, wyckoff_database=self.wyckoff,
        )
        self.assertEqual(compiled.hall.space_group_number, 221)
        self.assertEqual(compiled.hard.group_num_atoms, 5)
        self.assertEqual(sum(o.free_dimension for o in compiled.hard.wyckoff_orbits), 0)
        gauge = self.wyckoff.entry(517, "c")
        self.assertEqual(expand_orbit(gauge, []).shape, (3, 3))

    def test_free_orbit_periodicity(self) -> None:
        gauge = self.wyckoff.entry(1, "a")
        np.testing.assert_allclose(expand_orbit(gauge, [1.2, -0.1, 0.3]), [[0.2, 0.9, 0.3]])

    def test_metric_contract(self) -> None:
        metadata = metric_contract_metadata(MEMBER_MEAN_CARTESIAN_V1)
        self.assertEqual(metadata["diffusion_metric"], MEMBER_MEAN_CARTESIAN_V1)
        with self.assertRaises(ValueError):
            metric_contract_metadata("not_a_metric")

    def test_cli_parsers_use_installed_assets(self) -> None:
        from polarevolve.runtime.sample import _parser as sample_parser
        from polarevolve.runtime.train import _parser as train_parser

        args = sample_parser().parse_args(["--run-id", "interface_test"])
        self.assertEqual(args.asset_root, str(polarevolve.asset_root()))
        args = train_parser().parse_args(["--run-id", "interface_test"])
        self.assertEqual(args.asset_root, str(polarevolve.asset_root()))


if __name__ == "__main__":
    unittest.main()
