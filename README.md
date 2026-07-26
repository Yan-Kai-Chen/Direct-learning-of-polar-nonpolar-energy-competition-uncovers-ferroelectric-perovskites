# Direct learning of polar–nonpolar energy competition uncovers ferroelectric perovskites

This repository contains the public, reproducible code and example tables for a
workflow that learns the energy competition between paired polar and nonpolar
perovskite structures.

The release follows the original three-part project layout:

```text
.
├── AE3GNN_Build&Train/          angular equivariant pair-energy GNN
├── Descriptor_engineering/      interpretable pair descriptors
├── Polar-Nonpolar_pair_model/   polar-holdout pair retrieval
├── Train_EXAMPLE.csv            fixed public train/val/test table
├── External_Validation_1.csv    public external-validation table
└── External_Validation_2.csv    public external-validation table
```

## Reproducibility boundary

- `Train_EXAMPLE.csv` contains the fixed public `train`, `val`, and `test`
  labels. Code reads this column directly and does not regenerate a split.
- `row_idx` is an identifier only. It is explicitly excluded from model
  features.
- The three public CSV files remain byte-for-byte unchanged. The pair-split
  JSON contains only relative public metadata.
- CIF structure libraries, graph caches, trained weights, prediction exports,
  API credentials, private rules, and machine-specific paths are not included.
- Different user-created splits can produce different results. They should be
  reported as separate experiments, not compared as exact reproductions of the
  fixed public split.

See [DATA_AND_PRIVACY.md](DATA_AND_PRIVACY.md) and
[DATA_MANIFEST.json](DATA_MANIFEST.json) for the public-data contract.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
conda env create -f environment.yml
conda activate polar-nonpolar-learning
```

The graph models use PyTorch and PyTorch Geometric. If your CUDA setup requires
a platform-specific PyTorch wheel, install PyTorch first using the official
selector and then run:

```bash
python -m pip install -r requirements.txt
```

## Quick verification

These commands audit the fixed public data and split without training a model:

```bash
python "AE3GNN_Build&Train/scripts/audit_training_data.py" \
  --data Train_EXAMPLE.csv

python "Polar-Nonpolar_pair_model/scripts/audit_pairing.py" \
  --train-pairs "Polar-Nonpolar_pair_model/pairing_ai_out_group_sym/splits/train_pairs_pos.csv" \
  --test-pairs "Polar-Nonpolar_pair_model/pairing_ai_out_group_sym/splits/test_pairs_pos.csv"

python -m unittest discover -s tests -v
```

The unit tests use synthetic graphs and small in-memory tables; they do not run
full training.

## Workflow

1. **Pair retrieval.** Use a polar-structure holdout and a two-tower graph
   encoder to retrieve candidate nonpolar partners. The optional symmetry score
   adjustment is implemented explicitly and audited independently.
2. **Descriptor engineering.** Map composition, local geometry, and
   electrostatic descriptors from user-supplied structures and element-property
   data.
3. **Energy learning.** Build periodic graphs with angular triplets and train
   the pair-energy model on the fixed split in `Train_EXAMPLE.csv`.
4. **External evaluation.** Apply the trained model to the two public
   validation tables with matching CIF structures supplied locally.

Each module has its own README with exact inputs and commands.

## Public data summary

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `Train_EXAMPLE.csv` | 3,238 | 126 | Fixed public train/val/test table |
| `External_Validation_1.csv` | 1,130 | 124 | External validation |
| `External_Validation_2.csv` | 413 | 124 | External validation |

The target for energy regression is `Energy_diff_meV`. A derived binary label,
when needed for analysis, is `+1` for `abs(Energy_diff_meV) < 70` and `-1`
otherwise; the label is derived in memory and is not required as an input
column.

## Citation

If this repository supports your work, cite the associated manuscript and the
software metadata in [CITATION.cff](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). Users are responsible for
checking the terms attached to any structures or elemental data they supply.
