# Descriptor engineering

This directory provides a seven-stage, public descriptor pipeline:

1. A/B/X site assignment;
2. elemental-property mapping;
3. A-site local geometry;
4. B-site local geometry and off-centering;
5. Ewald electrostatic summaries;
6. declared derived-feature operations;
7. final table export.

Unlike the earlier placeholder controller, `public_api.py` now calls the
implemented stage modules directly.

## Inputs

- a pair CSV containing `Polar_mpid`, `NPolar_mpid`, and a composition column
  such as `Polar_pretty_formula`;
- an element-property CSV with an element-symbol column and the properties
  referenced by the rule mapping;
- a local CIF directory whose filenames match the material identifiers.

The public default rule expects exactly three unique elements, chooses the most
electronegative element as X, and assigns the larger remaining element to A.
This rule is intended as a transparent baseline, not a universal crystallographic
site classifier.

## Run

From the repository root:

```bash
python Descriptor_engineering/run_descriptor_pipeline.py \
  --input-pairs data/input_pairs.csv \
  --element-properties data/element_properties.csv \
  --structures /path/to/cif_directory \
  --work-dir work_descriptor \
  --output-dir outputs_descriptor
```

Intermediate files are numbered `01_...csv` through `06_...csv`. The final
file is `descriptor_table_public_ready.csv`.

## Rule overrides

`default_rules.py` is tracked and contains the executable public baseline. To
test a different site-assignment or oxidation-state policy, copy
`private_rules_local.example.py` to `private_rules_local.py` and define the
same seven rule dictionaries. The local override is ignored by Git.

Keep proprietary property tables, structure archives, and rule overrides
outside the repository.
