# -*- coding: utf-8 -*-
"""
Public descriptor-engineering pipeline controller.

This script exposes the high-level workflow only.
Sensitive thresholds, feature-selection lists, and detailed local rules
must be stored in `private_rules_local.py`, which should NOT be uploaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from public_api import (
    DescriptorPipeline,
    PipelinePaths,
    PipelineOutputs,
)

try:
    # Real local rules (private, NOT tracked on GitHub)
    from private_rules_local import (
        SITE_RULES,
        ELEMENT_RULES,
        A_GEOM_RULES,
        B_GEOM_RULES,
        EWALD_RULES,
        DERIVED_RULES,
        EXPORT_RULES,
    )
except Exception:
    # Public fallback template for open-source skeleton
    from private_rules_local.example import (
        SITE_RULES,
        ELEMENT_RULES,
        A_GEOM_RULES,
        B_GEOM_RULES,
        EWALD_RULES,
        DERIVED_RULES,
        EXPORT_RULES,
    )


def build_default_paths(base_dir: str | os.PathLike) -> PipelinePaths:
    base = Path(base_dir).resolve()
    return PipelinePaths(
        input_pair_csv=base / "data" / "input_pairs.csv",
        element_property_csv=base / "data" / "element_properties.csv",
        structure_dir=base / "data" / "structures",
        work_dir=base / "work_descriptor",
        output_dir=base / "outputs_descriptor",
    )


def main() -> None:
    paths = build_default_paths(".")
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DescriptorPipeline(
        paths=paths,
        site_rules=SITE_RULES,
        element_rules=ELEMENT_RULES,
        a_geom_rules=A_GEOM_RULES,
        b_geom_rules=B_GEOM_RULES,
        ewald_rules=EWALD_RULES,
        derived_rules=DERIVED_RULES,
        export_rules=EXPORT_RULES,
    )

    outputs: PipelineOutputs = pipe.run_all()

    print("\n============================================================")
    print("[DONE] Descriptor pipeline finished.")
    print(f"[FINAL] {outputs.final_feature_csv}")
    print("============================================================")


if __name__ == "__main__":
    main()