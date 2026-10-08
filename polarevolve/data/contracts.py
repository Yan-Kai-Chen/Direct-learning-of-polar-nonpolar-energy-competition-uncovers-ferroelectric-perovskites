"""Immutable contracts for externally stored datasets, assets, and outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from polarevolve.crystal.contracts import ContractError

INDEPENDENT_LATTICE_MODE = "independent_bounded_lattice_coordinate_generation"
DEFAULT_CACHE_RELATIVE = "mp_20_asu_cache_v2"
DEFAULT_GROUP_ASSETS_RELATIVE = "group_database_v1"
DEFAULT_WYCKOFF_ASSETS_RELATIVE = "wyckoff_gauge_v1"

# Ordered provenance shared by replay and independent evaluation.
LATTICE_PROVENANCE_FIELDS = (
    "lattice_seed",
    "lattice_checkpoint_sha256",
    "lattice_mixture_component",
    "lattice_coordinates",
    "lattice_volume_per_atom",
    "lattice_aspect_ratio",
    "lattice_metric_invariance_error",
    "lattice_volume_was_clipped",
    "lattice_shape_scale",
)
LATTICE_REPLAY_FIELDS = ("lattice", *LATTICE_PROVENANCE_FIELDS)
JOINT_LATTICE_FIELDS = (*LATTICE_PROVENANCE_FIELDS, "lattice")
PAIRED_LATTICE_IDENTITY_FIELDS = ("lattice", *LATTICE_PROVENANCE_FIELDS[:4])


def _absolute_path(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty absolute path")
    normalized = value.strip()
    if not (
        Path(normalized).is_absolute()
        or PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
    ):
        raise ContractError(f"{field_name} must be an absolute path")
    trimmed = normalized.rstrip("/\\")
    if not trimmed or re.fullmatch(r"[A-Za-z]:", trimmed):
        return normalized
    return trimmed


@dataclass(frozen=True)
class ExternalRoots:
    data_root: str
    asset_root: str
    output_root: str
    schema_version: str = "gt_sge_external_roots_v1"

    def __post_init__(self) -> None:
        roots = {
            "data_root": _absolute_path(self.data_root, field_name="data_root"),
            "asset_root": _absolute_path(self.asset_root, field_name="asset_root"),
            "output_root": _absolute_path(self.output_root, field_name="output_root"),
        }
        if len({value.casefold() for value in roots.values()}) != 3:
            raise ContractError("data, asset, and output roots must be distinct")
        for name, value in roots.items():
            object.__setattr__(self, name, value)


__all__ = [
    "ExternalRoots", "INDEPENDENT_LATTICE_MODE",
    "DEFAULT_CACHE_RELATIVE", "DEFAULT_GROUP_ASSETS_RELATIVE", "DEFAULT_WYCKOFF_ASSETS_RELATIVE",
    "LATTICE_REPLAY_FIELDS", "JOINT_LATTICE_FIELDS", "PAIRED_LATTICE_IDENTITY_FIELDS",
]
