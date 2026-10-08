from __future__ import annotations

import argparse
from pathlib import Path

from .public_api import DescriptorPipeline, PipelinePaths

try:
    from .private_rules_local import (
        A_GEOM_RULES,
        B_GEOM_RULES,
        DERIVED_RULES,
        ELEMENT_RULES,
        EWALD_RULES,
        EXPORT_RULES,
        SITE_RULES,
    )
except ModuleNotFoundError:
    from .default_rules import (
        A_GEOM_RULES,
        B_GEOM_RULES,
        DERIVED_RULES,
        ELEMENT_RULES,
        EWALD_RULES,
        EXPORT_RULES,
        SITE_RULES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build public polar-nonpolar descriptor tables."
    )
    parser.add_argument("--input-pairs", required=True, type=Path)
    parser.add_argument("--element-properties", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pipeline = DescriptorPipeline(
        paths=PipelinePaths(
            input_pair_csv=args.input_pairs,
            element_property_csv=args.element_properties,
            structure_dir=args.structures,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
        ),
        site_rules=SITE_RULES,
        element_rules=ELEMENT_RULES,
        a_geom_rules=A_GEOM_RULES,
        b_geom_rules=B_GEOM_RULES,
        ewald_rules=EWALD_RULES,
        derived_rules=DERIVED_RULES,
        export_rules=EXPORT_RULES,
    )
    outputs = pipeline.run_all()
    print(f"Final descriptor table: {outputs.final_feature_csv}")


if __name__ == "__main__":
    main()
