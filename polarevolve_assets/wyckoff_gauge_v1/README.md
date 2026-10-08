# Wyckoff gauge database v1

This generated asset contains exact-rational Wyckoff equations and canonical
affine gauges for every Hall setting from 1 through 530.

## Contents

- `wyckoff_gauges.json`: 3467 Wyckoff entries and 24295 exact orbit-member maps.
- `BUILD_AUDIT.json`: coverage, gauge, multiplicity, and exact round-trip checks.
- `PROVENANCE.json`: builder, source commit, license, capability, and artifact hashes.
- `source_snapshot/`: pinned unmodified spglib `v2.7.0` source table and manifest.
- `LICENSES/`: the required BSD-3-Clause notice.

Build with:

```bash
python -m scripts.data.build_wyckoff_gauge_database
```

Runtime code must use `WyckoffDatabase`; it verifies every artifact hash before
making entries available. Maximal-subgroup and Wyckoff-splitting capabilities
are deliberately marked unavailable because they are not supplied by this
source.
