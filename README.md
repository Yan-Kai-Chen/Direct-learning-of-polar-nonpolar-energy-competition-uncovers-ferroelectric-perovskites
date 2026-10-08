# PolarGen

**Learning polar-nonpolar phase competition for ferroelectric crystal discovery.**

**PolarMatch** establishes phase correspondence. **PolarComp** learns phase
competition. **PolarEvolve** generates target crystal configurations.

PolarGen brings these three tasks into one modular research codebase for
ferroelectric perovskites. Rather than evaluating a polar structure in
isolation, it connects that structure to a nonpolar reference, learns their
relative energetics, and provides symmetry-conditioned generation of new
configurations.

This repository accompanies the manuscript *Direct learning of polar-nonpolar
energy competition uncovers ferroelectric perovskites*.

[Architecture](docs/architecture.md) | [Quick Start](#quick-start) |
[Stage Guides](#using-the-three-stages) | [Data](#public-data) |
[Reproducibility](docs/reproducibility.md)

## Scientific Workflow

| Stage | Scientific task | Main input | Main output |
|---|---|---|---|
| **PolarMatch** | Establish phase correspondence | A polar query and a nonpolar candidate pool | Ranked candidate pairs |
| **PolarComp** | Learn phase competition | An ordered polar/nonpolar structure pair | Predicted pair-energy difference |
| **PolarEvolve** | Generate target crystal configurations | Symmetry/orbit conditions and compatible trained models | Candidate crystal configurations |

The conceptual interfaces are:

```text
PolarMatch   -> candidate polar/nonpolar pairs -> PolarComp
PolarEvolve  -> generated configurations       -> pair construction -> PolarComp
```

Complete crystal structures and explicit pair records connect the stages.
The modules can be used independently; this release does not bundle an
automatic end-to-end search driver or pretrained discovery pipeline.

### PolarMatch: Establish Phase Correspondence

A two-tower periodic graph model learns to retrieve nonpolar partners for a
polar query. Symmetry-aware score adjustment supports candidate ranking.
Retrieval proposes a correspondence; crystallographic group relations are
handled separately, rather than inferred as a proof from a retrieval score.

### PolarComp: Learn Phase Competition

An angle-aware equivariant graph network encodes both members of a pair with a
shared backbone. Their embeddings, signed difference and absolute difference
form an explicit pair representation for energy regression.

The manuscript defines the competition as

$$
\Delta E = E_{\mathrm{P}} - E_{\mathrm{NP}},
$$

where both energies are per atom. The public target column `Energy_diff_meV`
uses **meV/atom**: a negative value favors the polar member of that pair, and a
positive value favors the nonpolar member. A favorable energy difference is a
screening signal, not by itself evidence of switchable ferroelectricity.

### PolarEvolve: Generate Target Configurations

Hall settings, Wyckoff orbits and group relations define symmetry-compatible
coordinate spaces. Score-based diffusion operates on independent
asymmetric-unit (ASU) parameters; equivalent atoms are reconstructed through
the selected symmetry operations instead of being diffused independently.
Fixed zero-dimensional orbits have no free coordinate variables.

Group relations determine which degrees of freedom are allowed, not a unique
final structure. The learned generative prior samples concrete configurations
within those conditions. Optional OD readouts provide interpretable structural
constraints for refinement when compatible targets and atomic roles are
supplied. See the [PolarEvolve guide](docs/polarevolve.md) for the distinction
between the OD implementation and the default sampling CLI.

## Installation

Use a dedicated environment with Python **3.11 or 3.12**. Package metadata
declares Python >=3.10; CI runs on Python 3.11, and local release checks used
Python 3.12.

Clone the current main repository into a local folder named `PolarGen`:

```bash
git clone https://github.com/Yan-Kai-Chen/Direct-learning-of-polar-nonpolar-energy-competition-uncovers-ferroelectric-perovskites.git PolarGen
cd PolarGen
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a CPU installation:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

For GPU use, install a PyTorch build appropriate for your CUDA environment,
then run the final editable-install command above. Core dependencies are
declared in [`pyproject.toml`](pyproject.toml); a Conda environment is also
provided in [`environment.yml`](environment.yml).

## Quick Start

Run these commands from the repository root after installation. They require
**no trained checkpoint and perform no model training**.

Check the public energy table and the fixed retrieval split:

```bash
polarcomp-check-data --data Train_EXAMPLE.csv
polarmatch-check-split --train-pairs data/polarmatch/splits/train_pairs_pos.csv --test-pairs data/polarmatch/splits/test_pairs_pos.csv
```

Explore parent-channel proposals for the bundled synthetic BaTiO3-like polar
structure:

```bash
python examples/compile_parent_channels.py
```

Expected output:

```text
Child SG 99 -> parent SG 221, Hall 517; method=symmetry_tolerance_ascent_v1
```

To write the proposals to JSON:

```bash
polarevolve-channels --cif examples/batio3_polar.cif --output outputs/parent_channels.json
```

This example proposes parent structures by a symmetry-tolerance ladder. It
does not run diffusion, enumerate every group embedding, or establish material
stability. Explicit group-relation compilation is described in the stage guide.

## Using the Three Stages

| Task | Entry point | Additional assets required |
|---|---|---|
| Train pair retrieval | `polarmatch-train` | CIF collection and configured structure paths |
| Train pair-energy regression | `polarcomp-train` | Labeled pair table and corresponding CIFs |
| Predict pair energies | `polarcomp-predict` | Compatible trained bundle and structure inputs |
| Compute pair descriptors | `polar-descriptors` | Pair table, CIFs and element-property table |
| Train ASU diffusion | `polarevolve-train` | Prepared compatible ASU cache |
| Sample configurations | `polarevolve-sample` | Compatible checkpoint and task/query assets |

Inspect the command interfaces:

```bash
polarmatch-train --help
polarcomp-train --help
polarcomp-predict --help
polar-descriptors --help
polarevolve-train --help
polarevolve-sample --help
```

Detailed input contracts, configurations and command examples:

- [PolarMatch](docs/polarmatch.md): retrieval training and polar-identity holdout.
- [PolarComp](docs/polarcomp.md): graph construction, fixed-split training and prediction.
- [PolarEvolve](docs/polarevolve.md): symmetry compilation, ASU caches, training and sampling.
- [Descriptors](docs/descriptors.md): shared chemistry, geometry and electrostatic features.

The diffusion trainer consumes a prepared ASU cache; it is not a raw-CIF-folder
training command. The generic sampler currently connects the physics-guidance
path, not automatic OD target planning or activation.

## Public Data

The three public tables retain their original bytes and splits. File hashes
and table dimensions are recorded in [`DATA_MANIFEST.json`](DATA_MANIFEST.json).

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `Train_EXAMPLE.csv` | 3,238 | 126 | Public table with fixed train/validation/test labels |
| `External_Validation_1.csv` | 1,130 | 124 | Pair-disjoint external validation |
| `External_Validation_2.csv` | 413 | 124 | Structure-disjoint subset of external validation |

`Polar_mpid` and `NPolar_mpid` identify the ordered structure pair. `row_idx`
is a row identifier, not a feature. For energy learning, use the existing
`split` column and fit preprocessing only on training rows. The retrieval
split under `data/polarmatch/splits/` holds out polar identities; reuse of
nonpolar candidates is permitted and reported by the audit.

Source CIF libraries and trained weights are not included. See
[`DATA_AND_PRIVACY.md`](DATA_AND_PRIVACY.md) for data handling and naming rules.

## Supplementary Operation Planning

The optional [operation-planning package](supplement/operation_planning/)
contains the useful language/graph planning components from the earlier
PolarGen repository: operation ranking, graph-derived numerical predictions,
language priors, fusion and Top-3 plan schemas.

```bash
python -m pip install -e "supplement/operation_planning[test]"
python -m pytest -q supplement/operation_planning/tests
```

Only this operation-planning GNN/LLM branch is supplementary. **The graph
models in PolarMatch and PolarComp are core components.** The three core
packages do not depend on the planner or a language model.

Legacy operation IDs, signs and magnitudes are not automatically converted
into PolarEvolve OD targets or displacement modes. Historical planner and
reference-diffusion results are not evaluations of the current ASU backend.

## Repository Layout

```text
polarmatch/                     phase-correspondence model and split protocol
polarcomp/                      angular equivariant pair-energy model
polarevolve/                    symmetry compilation and ASU score diffusion
polarevolve_assets/             Hall/Wyckoff/supergroup tables and licenses
descriptors/                    shared interpretable descriptor pipeline
configs/                       core configuration examples
data/polarmatch/splits/         fixed public retrieval split
examples/                      checkpoint-free crystallographic examples
supplement/operation_planning/  optional operation-planning GNN/LLM
docs/                          stage guides, architecture and provenance
tests/                         focused software tests
tools/                         public data and release-content checks
```

## Reproducibility

This is a **source release** with public tables, configurations and licensed
symmetry assets. It does not distribute trained checkpoints, real training
CIF libraries, prepared ASU caches, full candidate ledgers or private
experiment harnesses. Training and sampling require the compatible external
inputs listed above. The precise scope is documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

Run the software checks:

```bash
python -m unittest discover -s tests -v
python tools/verify_public_data.py
python tools/check_release.py
```

GitHub Actions also checks the optional planner interfaces and installs a
built wheel outside the checkout to verify packaged symmetry assets. These
are software checks, not reproduction of manuscript accuracy, generation
quality or DFT validation. See the [release check record](docs/validation.md).

The [migration map](docs/migration.md) and
[source manifest](docs/source_manifest.json) document code origins and the
supplied diffusion-source fingerprint.

## Citation and License

Please cite the associated manuscript and the software version or commit used
in your work. Machine-readable software citation metadata is provided in
[`CITATION.cff`](CITATION.cff).

Project code is released under the [MIT License](LICENSE). Bundled
crystallographic databases retain their own licenses and attribution; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
