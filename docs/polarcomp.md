# PolarComp: Learning Phase Competition

This module implements the public angular equivariant graph-neural-network
workflow for polar–nonpolar energy regression.

## Model and graph contract

- periodic radius graph: 6.0 Å cutoff;
- at most 64 outgoing neighbors per center;
- angle construction from the nearest 12 primary edges;
- at most 200 angle triplets per center;
- two PaiNN-style scalar/vector message-passing layers;
- hidden dimension 128, 48 radial basis functions, dropout 0.2;
- pair representation: polar, nonpolar, signed difference, and absolute
  difference;
- target: `Energy_diff_meV`.

The defaults are recorded in `configs/polarcomp/train_example.yaml`.

## Data contract

`Train_EXAMPLE.csv` supplies the fixed split through its `split` column. The
training code never regenerates this split. `row_idx`, structure identifiers,
the split label, and the target are excluded from optional numeric residual
features.

The repository does not include CIF files or graph caches. Supply a local
structure directory containing files named by `Polar_mpid` and `NPolar_mpid`.

## Commands

From the repository root:

```bash
polarcomp-check-data \
  --data Train_EXAMPLE.csv

python -m polarcomp.smoke_train

polarcomp-train \
  --data Train_EXAMPLE.csv \
  --structures /path/to/cif_directory \
  --output outputs/polarcomp_run \
  --config configs/polarcomp/train_example.yaml
```

The smoke command creates synthetic graphs in memory, performs exactly one
optimizer step, and writes no checkpoint.

Prediction from a user-created bundle:

```bash
polarcomp-predict \
  --bundle outputs/polarcomp_run/gnn_bundle.pt \
  --data External_Validation_1.csv \
  --structures /path/to/cif_directory \
  --graph-cache /path/to/local_graph_cache \
  --output outputs/external_validation_1_predictions.csv
```

Full training is intentionally not part of the test suite.
