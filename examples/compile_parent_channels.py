"""Checkpoint-free parent proposals for a synthetic BaTiO3-like polar cell."""

from __future__ import annotations

from polarevolve import asset_root

ASSET_ROOT = asset_root()

# Tetragonal P4mm BaTiO3 (polar child): Ba 1a, Ti 1b, O 1b + 2c.
BTO_LATTICE = [[3.99, 0.0, 0.0], [0.0, 3.99, 0.0], [0.0, 0.0, 4.04]]
BTO_FRACTIONAL = [
    (0.0, 0.0, 0.0),    # Ba
    (0.5, 0.5, 0.52),   # Ti (displaced along +c)
    (0.5, 0.5, -0.02),  # O (apical)
    (0.0, 0.5, 0.48),   # O (equatorial)
    (0.5, 0.0, 0.48),   # O (equatorial)
]
BTO_NUMBERS = [56, 22, 8, 8, 8]


def compile_parent_channels() -> None:
    """Compile the acceptable nonpolar parent channels of a polar child."""
    from polarevolve.crystal.parent import discover_parent_channels
    from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase

    groups = GroupDatabase(ASSET_ROOT / "group_database_v1")
    wyckoff = WyckoffDatabase(ASSET_ROOT / "wyckoff_gauge_v1")
    proposals = discover_parent_channels(
        lattice=BTO_LATTICE,
        fractional=BTO_FRACTIONAL,
        atomic_numbers=BTO_NUMBERS,
        group_database=groups,
        wyckoff_database=wyckoff,
    )
    for proposal in proposals:
        print(
            f"Child SG {proposal.child_space_group} -> parent SG "
            f"{proposal.parent_space_group}, Hall {proposal.parent_hall_number}; "
            f"method={proposal.discovery_method}"
        )


if __name__ == "__main__":
    compile_parent_channels()
