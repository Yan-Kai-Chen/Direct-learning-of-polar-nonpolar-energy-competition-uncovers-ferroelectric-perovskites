# Symmetry Assets

These versioned assets are required for offline group-theory and Wyckoff
contracts. They are source data, not runtime output.

- `group_database_v1/`: Hall settings, point-group hierarchy, and cell maps.
- `wyckoff_source_spglib_v2_7_0/`: pinned upstream source and license.
- `wyckoff_gauge_v1/`: compiled exact affine gauges and build provenance.

Verify the complete asset snapshot from this directory:

```bash
sha256sum --check SHA256SUMS.txt
```

Every generated database also carries its own `PROVENANCE.json`. Rebuild
commands and capability limits are documented in the versioned subdirectory.
