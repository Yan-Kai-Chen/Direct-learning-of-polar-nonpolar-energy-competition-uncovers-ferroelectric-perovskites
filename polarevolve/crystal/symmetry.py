"""Licensed, hash-verified Hall and Wyckoff asset readers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from polarevolve.crystal.contracts import ContractError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ContractError("boolean values are not rational coordinates")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"invalid rational coordinate: {value!r}") from exc


def _fraction_vector(raw: Any, *, size: int, name: str) -> tuple[Fraction, ...]:
    if not isinstance(raw, list) or len(raw) != size:
        raise ContractError(f"{name} must contain {size} rational values")
    return tuple(_fraction(value) for value in raw)


def _fraction_matrix(
    raw: Any, *, rows: int, columns: int, name: str
) -> tuple[tuple[Fraction, ...], ...]:
    if not isinstance(raw, list) or len(raw) != rows:
        raise ContractError(f"{name} must contain {rows} rows")
    return tuple(
        _fraction_vector(row, size=columns, name=f"{name}[{index}]")
        for index, row in enumerate(raw)
    )


def _determinant_3x3(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _payload(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON asset: {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"JSON asset root must be an object: {path}")
    return value


@dataclass(frozen=True)
class AssetProvenance:
    root: Path
    database_version: str
    schema_version: str
    artifacts: tuple[tuple[str, str], ...]
    license_spdx: str | None
    source_version: str | None

    @classmethod
    def load(cls, root: str | Path) -> "AssetProvenance":
        resolved = Path(root).resolve()
        payload = _payload(resolved / "PROVENANCE.json")
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, Mapping) or not raw_artifacts:
            raise ContractError("asset provenance must declare artifact hashes")
        artifacts: list[tuple[str, str]] = []
        for relative, expected in sorted(raw_artifacts.items()):
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ContractError("asset provenance contains an unsafe relative path")
            digest = str(expected).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ContractError(f"invalid SHA-256 for asset {relative}")
            absolute = resolved / relative_path
            if not absolute.is_file():
                raise ContractError(f"declared asset is missing: {absolute}")
            actual = sha256_file(absolute)
            if actual != digest:
                raise ContractError(
                    f"asset hash mismatch for {relative}: expected {digest}, got {actual}"
                )
            artifacts.append((relative_path.as_posix(), digest))
        version = str(payload.get("database_version", "")).strip()
        schema = str(payload.get("schema_version", "")).strip()
        if not version or not schema:
            raise ContractError("asset provenance must declare database and schema versions")
        license_value = payload.get("license_spdx")
        if license_value is None:
            sources = payload.get("sources")
            licenses = {
                str(item.get("license", "")).strip()
                for item in sources or ()
                if isinstance(item, Mapping) and item.get("license")
            }
            license_value = ",".join(sorted(licenses)) or None
        source_version = payload.get("source_version")
        return cls(
            root=resolved,
            database_version=version,
            schema_version=schema,
            artifacts=tuple(artifacts),
            license_spdx=str(license_value) if license_value else None,
            source_version=str(source_version) if source_version else None,
        )

    def artifact_sha256(self, relative_path: str) -> str:
        table = dict(self.artifacts)
        try:
            return table[relative_path]
        except KeyError as exc:
            raise ContractError(f"artifact is not declared in provenance: {relative_path}") from exc


@dataclass(frozen=True)
class HallSetting:
    hall_number: int
    setting_id: str
    space_group_number: int
    crystal_system: str
    centering_symbol: str
    centering_index: int
    international_short: str
    point_group: str


@dataclass(frozen=True)
class CellTransform:
    hall_number: int
    setting_id: str
    centering_index: int
    group_to_primitive: tuple[tuple[Fraction, ...], ...]


class GroupDatabase:
    """Read-only Hall metadata loaded from an explicit verified asset root."""

    def __init__(self, root: str | Path) -> None:
        self.provenance = AssetProvenance.load(root)
        payload = _payload(self.provenance.root / "spacegroup_settings.json")
        if payload.get("schema_version") != "spacegroup_settings_v1":
            raise ContractError("unsupported space-group settings schema")
        settings: dict[int, HallSetting] = {}
        for raw in payload.get("settings", ()):
            if not isinstance(raw, Mapping):
                raise ContractError("space-group setting entries must be objects")
            setting = HallSetting(
                hall_number=int(raw["hall_number"]),
                setting_id=str(raw["setting_id"]),
                space_group_number=int(raw["space_group_number"]),
                crystal_system=str(raw["crystal_system"]),
                centering_symbol=str(raw["centering_symbol"]),
                centering_index=int(raw["centering_index"]),
                international_short=str(raw["international_short"]),
                point_group=str(raw["point_group"]),
            )
            if not 1 <= setting.hall_number <= 530 or setting.hall_number in settings:
                raise ContractError("Hall settings must uniquely cover values in 1..530")
            if setting.centering_index <= 0:
                raise ContractError("centering index must be positive")
            settings[setting.hall_number] = setting
        if set(settings) != set(range(1, 531)):
            raise ContractError("group database must contain all 530 Hall settings")
        self._settings = settings
        transforms_payload = _payload(self.provenance.root / "cell_transforms.json")
        if transforms_payload.get("schema_version") != "cell_transforms_v1":
            raise ContractError("unsupported cell-transform schema")
        transforms: dict[int, CellTransform] = {}
        for raw in transforms_payload.get("transforms", ()):
            hall_number = int(raw["hall_number"])
            transform = CellTransform(
                hall_number=hall_number,
                setting_id=str(raw["setting_id"]),
                centering_index=int(raw["centering_index"]),
                group_to_primitive=_fraction_matrix(
                    raw["group_to_primitive"],
                    rows=3,
                    columns=3,
                    name="group_to_primitive",
                ),
            )
            setting = settings.get(hall_number)
            if setting is None or hall_number in transforms:
                raise ContractError("cell transforms must uniquely cover Hall settings")
            if (
                transform.setting_id != setting.setting_id
                or transform.centering_index != setting.centering_index
            ):
                raise ContractError("cell transform and Hall setting disagree")
            if abs(_determinant_3x3(transform.group_to_primitive)) != Fraction(
                1, setting.centering_index
            ):
                raise ContractError("cell transform determinant contradicts centering")
            transforms[hall_number] = transform
        if set(transforms) != set(settings):
            raise ContractError("cell transforms must cover all 530 Hall settings")
        self._cell_transforms = transforms

    def setting(self, hall_number: int) -> HallSetting:
        try:
            return self._settings[int(hall_number)]
        except (KeyError, ValueError) as exc:
            raise ContractError(f"unknown Hall setting: {hall_number}") from exc

    def cell_transform(self, hall_number: int) -> CellTransform:
        try:
            return self._cell_transforms[int(hall_number)]
        except (KeyError, ValueError) as exc:
            raise ContractError(f"unknown Hall cell transform: {hall_number}") from exc


@dataclass(frozen=True)
class AffineMemberMap:
    origin: tuple[Fraction, Fraction, Fraction]
    basis: tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]


@dataclass(frozen=True)
class WyckoffGauge:
    hall_number: int
    setting_id: str
    letter: str
    multiplicity: int
    free_dimension: int
    site_symmetry: str
    affine_origin: tuple[Fraction, ...]
    basis: tuple[tuple[Fraction, ...], ...]
    parameter_period_basis: tuple[tuple[Fraction, ...], ...]
    member_maps: tuple[AffineMemberMap, ...]
    table_entry_hash: str


class WyckoffDatabase:
    """Exact Wyckoff gauges loaded from an explicit verified asset root."""

    def __init__(self, root: str | Path) -> None:
        self.provenance = AssetProvenance.load(root)
        payload = _payload(self.provenance.root / "wyckoff_gauges.json")
        if payload.get("schema_version") != "wyckoff_gauge_database_v1":
            raise ContractError("unsupported Wyckoff gauge database schema")
        entries: dict[tuple[int, str], WyckoffGauge] = {}
        for raw in payload.get("entries", ()):
            if not isinstance(raw, Mapping):
                raise ContractError("Wyckoff gauge entries must be objects")
            dimension = int(raw["free_dimension"])
            multiplicity = int(raw["multiplicity"])
            gauge_raw = raw["gauge"]
            if not isinstance(gauge_raw, Mapping):
                raise ContractError("Wyckoff gauge payload must be an object")
            members = tuple(
                AffineMemberMap(
                    origin=_fraction_vector(
                        member["origin"], size=3, name="member.origin"
                    ),
                    basis=_fraction_matrix(
                        member["basis"],
                        rows=3,
                        columns=dimension,
                        name="member.basis",
                    ),
                )
                for member in raw["member_maps"]
            )
            entry = WyckoffGauge(
                hall_number=int(raw["hall_number"]),
                setting_id=str(raw["setting_id"]),
                letter=str(raw["wyckoff_letter"]),
                multiplicity=multiplicity,
                free_dimension=dimension,
                site_symmetry=str(raw["site_symmetry"]),
                affine_origin=_fraction_vector(
                    gauge_raw["affine_origin"],
                    size=3,
                    name="gauge.affine_origin",
                ),
                basis=_fraction_matrix(
                    gauge_raw["basis"],
                    rows=3,
                    columns=dimension,
                    name="gauge.basis",
                ),
                parameter_period_basis=_fraction_matrix(
                    gauge_raw["parameter_period_basis"],
                    rows=dimension,
                    columns=dimension,
                    name="gauge.parameter_period_basis",
                ),
                member_maps=members,
                table_entry_hash=str(raw["table_entry_hash"]),
            )
            if len(members) != multiplicity:
                raise ContractError("Wyckoff multiplicity must equal member-map count")
            if not 0 <= dimension <= 3:
                raise ContractError("Wyckoff free dimension must lie in 0..3")
            key = (entry.hall_number, entry.letter)
            if key in entries:
                raise ContractError(f"duplicate Wyckoff gauge key: {key}")
            entries[key] = entry
        if not entries:
            raise ContractError("Wyckoff database contains no entries")
        self._entries = entries
        by_hall: dict[int, list[WyckoffGauge]] = {}
        for (hall_number, _), entry in entries.items():
            by_hall.setdefault(hall_number, []).append(entry)
        if set(by_hall) != set(range(1, 531)):
            raise ContractError("Wyckoff gauges must cover all 530 Hall settings")
        self._entries_by_hall = {
            hall_number: tuple(sorted(values, key=lambda item: item.letter))
            for hall_number, values in by_hall.items()
        }

    def entry(self, hall_number: int, letter: str) -> WyckoffGauge:
        key = (int(hall_number), str(letter).lower())
        try:
            return self._entries[key]
        except KeyError as exc:
            raise ContractError(f"unknown Wyckoff gauge: Hall {key[0]} letter {key[1]}") from exc

    def entries_for_hall(self, hall_number: int) -> tuple[WyckoffGauge, ...]:
        try:
            return self._entries_by_hall[int(hall_number)]
        except (KeyError, ValueError) as exc:
            raise ContractError(f"unknown Hall setting: {hall_number}") from exc

    @property
    def entry_count(self) -> int:
        return len(self._entries)


__all__ = [
    "AffineMemberMap",
    "AssetProvenance",
    "GroupDatabase",
    "HallSetting",
    "WyckoffDatabase",
    "WyckoffGauge",
    "sha256_file",
]
