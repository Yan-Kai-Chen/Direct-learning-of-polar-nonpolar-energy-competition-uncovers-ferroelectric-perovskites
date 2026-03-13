# -*- coding: utf-8 -*-
"""
Public API for descriptor engineering.

This file keeps the workflow and interfaces public,
while sensitive descriptor rules are injected externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


# =========================================================
# Data classes
# =========================================================
@dataclass
class PipelinePaths:
    input_pair_csv: Path
    element_property_csv: Path
    structure_dir: Path
    work_dir: Path
    output_dir: Path


@dataclass
class PipelineOutputs:
    site_csv: Path
    elemental_csv: Path
    a_geom_csv: Path
    b_geom_csv: Path
    ewald_csv: Path
    derived_csv: Path
    final_feature_csv: Path


# =========================================================
# Main pipeline
# =========================================================
class DescriptorPipeline:
    def __init__(
        self,
        paths: PipelinePaths,
        site_rules: Dict[str, Any],
        element_rules: Dict[str, Any],
        a_geom_rules: Dict[str, Any],
        b_geom_rules: Dict[str, Any],
        ewald_rules: Dict[str, Any],
        derived_rules: Dict[str, Any],
        export_rules: Dict[str, Any],
    ) -> None:
        self.paths = paths
        self.site_rules = site_rules
        self.element_rules = element_rules
        self.a_geom_rules = a_geom_rules
        self.b_geom_rules = b_geom_rules
        self.ewald_rules = ewald_rules
        self.derived_rules = derived_rules
        self.export_rules = export_rules

    def run_all(self) -> PipelineOutputs:
        self._validate_inputs()

        site_csv = self.run_site_assignment()
        elemental_csv = self.run_elemental_mapping(site_csv)
        a_geom_csv = self.run_a_geometry(elemental_csv)
        b_geom_csv = self.run_b_geometry(a_geom_csv)
        ewald_csv = self.run_ewald(b_geom_csv)
        derived_csv = self.run_derived_features(ewald_csv)
        final_feature_csv = self.run_export(derived_csv)

        return PipelineOutputs(
            site_csv=site_csv,
            elemental_csv=elemental_csv,
            a_geom_csv=a_geom_csv,
            b_geom_csv=b_geom_csv,
            ewald_csv=ewald_csv,
            derived_csv=derived_csv,
            final_feature_csv=final_feature_csv,
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    def _validate_inputs(self) -> None:
        if not self.paths.input_pair_csv.is_file():
            raise FileNotFoundError(f"Missing input pair CSV: {self.paths.input_pair_csv}")
        if not self.paths.element_property_csv.is_file():
            raise FileNotFoundError(f"Missing element property CSV: {self.paths.element_property_csv}")
        if not self.paths.structure_dir.exists():
            raise FileNotFoundError(f"Missing structure directory: {self.paths.structure_dir}")

    # -----------------------------------------------------
    # Stage 1: A/B/X site assignment
    # -----------------------------------------------------
    def run_site_assignment(self) -> Path:
        df = pd.read_csv(self.paths.input_pair_csv)
        df.columns = [str(c).strip() for c in df.columns]

        # Public-level logic only:
        # - parse formula
        # - assign A/B/X by a private site rule
        # - write site-level composition metadata
        out = self._apply_site_assignment(df, self.site_rules)

        out_path = self.paths.work_dir / "01_site_assignment.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] site assignment -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 2: elemental property mapping
    # -----------------------------------------------------
    def run_elemental_mapping(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)
        prop = pd.read_csv(self.paths.element_property_csv)

        out = self._apply_elemental_mapping(df, prop, self.element_rules)

        out_path = self.paths.work_dir / "02_elemental_mapping.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] elemental mapping -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 3: A-site geometry features
    # -----------------------------------------------------
    def run_a_geometry(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)

        out = self._apply_a_geometry(df, self.paths.structure_dir, self.a_geom_rules)

        out_path = self.paths.work_dir / "03_A_geometry.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] A geometry -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 4: B-site geometry / off-centering
    # -----------------------------------------------------
    def run_b_geometry(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)

        out = self._apply_b_geometry(df, self.paths.structure_dir, self.b_geom_rules)

        out_path = self.paths.work_dir / "04_B_geometry.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] B geometry -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 5: Ewald / electrostatic features
    # -----------------------------------------------------
    def run_ewald(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)

        out = self._apply_ewald(df, self.paths.structure_dir, self.ewald_rules)

        out_path = self.paths.work_dir / "05_ewald.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] Ewald features -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 6: derived features / interactions
    # -----------------------------------------------------
    def run_derived_features(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)

        out = self._apply_derived_features(df, self.derived_rules)

        out_path = self.paths.work_dir / "06_derived_features.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] derived features -> {out_path}")
        return out_path

    # -----------------------------------------------------
    # Stage 7: export final selected feature table
    # -----------------------------------------------------
    def run_export(self, input_csv: Path) -> Path:
        df = pd.read_csv(input_csv)

        out = self._apply_export_rules(df, self.export_rules)

        out_path = self.paths.output_dir / "descriptor_table_public_ready.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] final export -> {out_path}")
        return out_path

    # =====================================================
    # Internal public-safe wrappers
    # =====================================================
    def _apply_site_assignment(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        out = df.copy()

        # Placeholder public interface:
        # real ranking criteria / tie-break rules stay private
        required_cols = rules.get("required_input_cols", ["Polar_pretty_formula"])
        for c in required_cols:
            if c not in out.columns:
                raise ValueError(f"Missing required column for site assignment: {c}")

        # Public placeholders for output schema
        public_site_cols = rules.get(
            "public_output_cols",
            [
                "A_site_symbol",
                "B_site_symbol",
                "X_site_symbol",
                "A_site_fraction",
                "B_site_fraction",
                "X_site_fraction",
            ],
        )
        for c in public_site_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_elemental_mapping(
        self,
        df: pd.DataFrame,
        prop: pd.DataFrame,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()

        # Public-only contract:
        # private rules define which elemental properties are actually mapped
        property_cols = rules.get("public_property_cols", [])
        for c in property_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_a_geometry(
        self,
        df: pd.DataFrame,
        structure_dir: Path,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()

        public_cols = rules.get(
            "public_output_cols",
            [
                "polar_A_geom_1",
                "npolar_A_geom_1",
                "d_A_geom_1",
            ],
        )
        for c in public_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_b_geometry(
        self,
        df: pd.DataFrame,
        structure_dir: Path,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()

        public_cols = rules.get(
            "public_output_cols",
            [
                "polar_B_geom_1",
                "npolar_B_geom_1",
                "d_B_geom_1",
            ],
        )
        for c in public_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_ewald(
        self,
        df: pd.DataFrame,
        structure_dir: Path,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()

        public_cols = rules.get(
            "public_output_cols",
            [
                "polar_Ewald_1",
                "npolar_Ewald_1",
                "d_Ewald_1",
            ],
        )
        for c in public_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_derived_features(
        self,
        df: pd.DataFrame,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()

        public_cols = rules.get(
            "public_output_cols",
            [
                "derived_feature_1",
                "derived_feature_2",
            ],
        )
        for c in public_cols:
            if c not in out.columns:
                out[c] = pd.NA

        return out

    def _apply_export_rules(
        self,
        df: pd.DataFrame,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        keep_cols = rules.get("public_keep_cols")
        if keep_cols is None:
            return df.copy()

        keep_cols = [c for c in keep_cols if c in df.columns]
        return df.loc[:, keep_cols].copy()