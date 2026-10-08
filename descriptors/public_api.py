from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .a_site_geometry import run_a_site_geometry_stage
from .b_site_geometry import run_b_site_geometry_stage
from .derived_features import run_derived_features_stage
from .elemental_mapping import run_elemental_mapping_stage
from .ewald_features import run_ewald_stage
from .site_assignment import run_site_assignment_stage


@dataclass(frozen=True)
class PipelinePaths:
    input_pair_csv: Path
    element_property_csv: Path
    structure_dir: Path
    work_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class PipelineOutputs:
    site_csv: Path
    elemental_csv: Path
    a_geom_csv: Path
    b_geom_csv: Path
    ewald_csv: Path
    derived_csv: Path
    final_feature_csv: Path


class DescriptorPipeline:
    """Seven-stage descriptor workflow with externally supplied rule mappings."""

    def __init__(
        self,
        paths: PipelinePaths,
        site_rules: dict[str, Any],
        element_rules: dict[str, Any],
        a_geom_rules: dict[str, Any],
        b_geom_rules: dict[str, Any],
        ewald_rules: dict[str, Any],
        derived_rules: dict[str, Any],
        export_rules: dict[str, Any],
    ) -> None:
        self.paths = paths
        self.site_rules = site_rules
        self.element_rules = element_rules
        self.a_geom_rules = a_geom_rules
        self.b_geom_rules = b_geom_rules
        self.ewald_rules = ewald_rules
        self.derived_rules = derived_rules
        self.export_rules = export_rules

    def _validate_inputs(self) -> None:
        if not self.paths.input_pair_csv.is_file():
            raise FileNotFoundError(
                f"Missing input pair CSV: {self.paths.input_pair_csv}"
            )
        if not self.paths.element_property_csv.is_file():
            raise FileNotFoundError(
                "Missing element-property CSV: "
                f"{self.paths.element_property_csv}"
            )
        if not self.paths.structure_dir.is_dir():
            raise FileNotFoundError(
                f"Missing structure directory: {self.paths.structure_dir}"
            )

    @staticmethod
    def _write(frame: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"[OK] {path}")
        return path

    def run_site_assignment(self) -> Path:
        frame = pd.read_csv(self.paths.input_pair_csv, low_memory=False)
        output = run_site_assignment_stage(frame, self.site_rules)
        return self._write(output, self.paths.work_dir / "01_site_assignment.csv")

    def run_elemental_mapping(self, input_csv: Path) -> Path:
        frame = pd.read_csv(input_csv, low_memory=False)
        properties = pd.read_csv(
            self.paths.element_property_csv,
            low_memory=False,
        )
        output = run_elemental_mapping_stage(
            frame,
            properties,
            self.element_rules,
        )
        return self._write(
            output,
            self.paths.work_dir / "02_elemental_mapping.csv",
        )
    def run_a_geometry(self, input_csv: Path) -> Path:
        output = run_a_site_geometry_stage(
            pd.read_csv(input_csv, low_memory=False),
            self.paths.structure_dir,
            self.a_geom_rules,
        )
        return self._write(output, self.paths.work_dir / "03_A_geometry.csv")

    def run_b_geometry(self, input_csv: Path) -> Path:
        output = run_b_site_geometry_stage(
            pd.read_csv(input_csv, low_memory=False),
            self.paths.structure_dir,
            self.b_geom_rules,
        )
        return self._write(output, self.paths.work_dir / "04_B_geometry.csv")

    def run_ewald(self, input_csv: Path) -> Path:
        output = run_ewald_stage(
            pd.read_csv(input_csv, low_memory=False),
            self.paths.structure_dir,
            self.ewald_rules,
        )
        return self._write(output, self.paths.work_dir / "05_ewald.csv")

    def run_derived_features(self, input_csv: Path) -> Path:
        output = run_derived_features_stage(
            pd.read_csv(input_csv, low_memory=False),
            self.derived_rules,
        )
        return self._write(
            output,
            self.paths.work_dir / "06_derived_features.csv",
        )

    def run_export(self, input_csv: Path) -> Path:
        frame = pd.read_csv(input_csv, low_memory=False)
        keep = self.export_rules.get("public_keep_cols")
        if keep:
            missing = sorted(set(keep) - set(frame.columns))
            if missing and self.export_rules.get("strict_mode", True):
                raise KeyError(f"Export columns are missing: {missing}")
            frame = frame.loc[:, [column for column in keep if column in frame]]
        return self._write(
            frame,
            self.paths.output_dir / "descriptor_table_public_ready.csv",
        )

    def run_all(self) -> PipelineOutputs:
        self._validate_inputs()
        self.paths.work_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        site = self.run_site_assignment()
        elemental = self.run_elemental_mapping(site)
        a_geometry = self.run_a_geometry(elemental)
        b_geometry = self.run_b_geometry(a_geometry)
        ewald = self.run_ewald(b_geometry)
        derived = self.run_derived_features(ewald)
        final = self.run_export(derived)
        return PipelineOutputs(
            site_csv=site,
            elemental_csv=elemental,
            a_geom_csv=a_geometry,
            b_geom_csv=b_geometry,
            ewald_csv=ewald,
            derived_csv=derived,
            final_feature_csv=final,
        )
