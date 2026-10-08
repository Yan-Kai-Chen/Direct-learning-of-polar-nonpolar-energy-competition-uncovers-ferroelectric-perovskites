"""Bundled crystallographic tables and their original licenses/provenance."""

from pathlib import Path


def asset_root() -> Path:
    """Return the installed directory containing the three symmetry databases."""
    return Path(__file__).resolve().parent


__all__ = ["asset_root"]
