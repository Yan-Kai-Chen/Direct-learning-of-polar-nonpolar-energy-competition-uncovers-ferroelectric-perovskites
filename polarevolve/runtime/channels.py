"""Propose nonpolar parent channels for a user-supplied polar CIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.core import Structure

from polarevolve import asset_root
from polarevolve.crystal.parent import discover_parent_channels
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path, default=asset_root())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    structure = Structure.from_file(args.cif)
    proposals = discover_parent_channels(
        lattice=structure.lattice.matrix,
        fractional=structure.frac_coords,
        atomic_numbers=structure.atomic_numbers,
        group_database=GroupDatabase(args.asset_root / "group_database_v1"),
        wyckoff_database=WyckoffDatabase(args.asset_root / "wyckoff_gauge_v1"),
    )
    channels = [
        {
            "proposal_id": proposal.proposal_id,
            "child_space_group": proposal.child_space_group,
            "child_hall_number": proposal.child_hall_number,
            "parent_space_group": proposal.parent_space_group,
            "parent_hall_number": proposal.parent_hall_number,
            "parent_lattice": proposal.parent_lattice,
            "parent_fractional": proposal.parent_fractional,
            "parent_atomic_numbers": proposal.parent_atomic_numbers,
            "transformation_matrix": proposal.transformation_matrix,
            "origin_shift": proposal.origin_shift,
            "discovery_method": proposal.discovery_method,
        }
        for proposal in proposals
    ]
    payload = {
        "schema_version": "polarevolve_parent_channels_v1",
        "input_structure": args.cif.name,
        "channels": channels,
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
