# Direct Learning of Polar-Nonpolar Energy Competition

**PolarMatch establishes phase correspondence. PolarComp learns phase
competition. PolarEvolve generates target crystal configurations.**

This repository is the unified source release for *Direct learning of
polar-nonpolar energy competition uncovers ferroelectric perovskites*.
The three core stages share complete structures and explicit pair records,
not language-model hidden states.

| Stage | Question | Source |
|---|---|---|
| **PolarMatch** | Which polar and nonpolar structures should be paired? | [`polarmatch/`](polarmatch/), [guide](docs/polarmatch.md) |
| **PolarComp** | What is the energy competition within a structure pair? | [`polarcomp/`](polarcomp/), [guide](docs/polarcomp.md) |
| **PolarEvolve** | Which crystal configurations can be generated within a chosen symmetry channel? | [`polarevolve/`](polarevolve/), [guide](docs/polarevolve.md) |

PolarMatch supplies candidate correspondences; PolarComp evaluates ordered
polar/nonpolar pairs; PolarEvolve constructs and refines candidates that can
be evaluated by PolarComp. This is a modular research workflow, not a claim
that a pretrained, one-command discovery pipeline is bundled here.

## Repository Layout

```text
polarmatch/                     two-tower pair retrieval and split protocol
polarcomp/                      angular equivariant pair-energy model
polarevolve/                    symmetry compilation and ASU score diffusion
polarevolve_assets/             licensed Hall/Wyckoff/supergroup tables
descriptors/                    shared interpretable descriptor pipeline
configs/                       core training configuration examples
data/polarmatch/splits/         preserved public retrieval split
examples/                      checkpoint-free crystallographic example
supplement/operation_planning/  optional language/graph operation planning
docs/                          architecture, usage and source provenance
tests/                         focused public-code tests
```

## Installation

Python 3.10 or newer is supported; CI uses Python 3.11. In an environment with
an appropriate PyTorch build:

```bash
python -m pip install -e ".[test]"
```

For a CPU-only environment, install CPU PyTorch first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

A Conda environment is also described in `environment.yml`. All commands below
run from the repository root after installation. Windows PowerShell supports
the same single-line commands.

## Start Here

These commands require no model checkpoint and do not train a model:

```bash
polarcomp-check-data --data Train_EXAMPLE.csv
polarmatch-check-split --train-pairs data/polarmatch/splits/train_pairs_pos.csv --test-pairs data/polarmatch/splits/test_pairs_pos.csv
python examples/compile_parent_channels.py
python -m unittest discover -s tests -v
python tools/verify_public_data.py
python tools/check_release.py
```

The crystallographic example illustrates parent proposals from a synthetic
BaTiO3-like polar structure. It is not a generated-material or stability result.

Training and inference entry points:

```bash
polarmatch-train --help
polarcomp-train --help
polarcomp-predict --help
polar-descriptors --help
polarevolve-channels --help
polarevolve-train --help
polarevolve-sample --help
```

Full workflows require user-supplied structure collections, data caches and
trained checkpoints. See the stage guides for the exact boundary.

## Public Data

The existing public tables and their fixed splits are preserved byte-for-byte.

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `Train_EXAMPLE.csv` | 3,238 | 126 | Fixed public train/validation/test table |
| `External_Validation_1.csv` | 1,130 | 124 | External validation |
| `External_Validation_2.csv` | 413 | 124 | External validation |

The regression target is `Energy_diff_meV`; `row_idx` is an identifier, not a
feature. Normalization is fitted on training rows only. See
[`DATA_AND_PRIVACY.md`](DATA_AND_PRIVACY.md) and
[`DATA_MANIFEST.json`](DATA_MANIFEST.json).

## Supplementary Planning

Useful language/graph operation-ranking components from the earlier PolarGen
repository are isolated in [`supplement/operation_planning/`](supplement/operation_planning/).
They are optional and are not imported by the three core packages. The old
compact reference diffusion and its historical generation metrics are not
presented as results of the new PolarEvolve backend.

## Reproducibility and Attribution

This is a source release: it includes training/sampling code and licensed
symmetry assets, but no trained weights, private structure corpus, graph cache,
full candidate ledger or private experiment harness. Structural validity,
energy prediction and physical validation are distinct stages. See
[`docs/reproducibility.md`](docs/reproducibility.md).

The old-to-new path map and supplied-code fingerprint are documented in
[`docs/migration.md`](docs/migration.md) and
[`docs/source_manifest.json`](docs/source_manifest.json).

Code is released under [MIT](LICENSE); bundled crystallographic data retain
their own licenses. See [third-party notices](THIRD_PARTY_NOTICES.md).
Please cite the associated manuscript and [software metadata](CITATION.cff).
