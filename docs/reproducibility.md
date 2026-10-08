# Reproducibility Boundary

## Available

- Public PolarMatch and PolarComp implementations, configs and fixed tables.
- PolarEvolve source for symmetry compilation, ASU score training and sampling.
- Hash-verified crystallographic tables with original third-party licenses.
- Small synthetic examples, focused tests and independently installable core.
- Optional graph/language operation-planning source and schema tests.

## Not Bundled

- Real CIF training libraries, external bulk-download corpora and graph caches.
- Prepared MP20 ASU caches, private pair assets and evaluation panels.
- Trained energy, retrieval, score, lattice or language-adapter checkpoints.
- Full generated-candidate ledgers, private experiment harnesses or DFT runs.
- An automatic bridge from legacy Top-3 plans to new OD targets.

The bundled training code requires the documented external inputs. An
installation test is not reproduction of a paper's numerical results.

## What the Tests Establish

Tests cover preserved public splits, descriptor execution on synthetic
structures, a small pair-energy forward pass, symmetry-asset loading, ASU
orbit expansion, versioned metric metadata and command-line parsers.
Optional tests cover planning schemas/fusion rather than trained-model quality.
CI also builds a wheel and loads its assets outside the checkout.

No training campaign, candidate-generation benchmark, energy improvement or
physical validation is claimed by these checks. The older PolarGen screening
numbers are not used as measurements of this reorganized backend.

## Fixed Data and Compatibility

The three public CSV hashes remain in `DATA_MANIFEST.json`. Retrieval split
files were relocated, not resplit. Fit preprocessing only on training rows;
do not substitute a new split and call it exact reproduction.

Public Python namespaces changed to `polarmatch`, `polarcomp`, `polarevolve`
and `descriptors`. Serialized checkpoint schema identifiers are intentionally
retained where they are part of the supplied data contract. Existing full-object
pickles or external scripts may need their import paths adapted; compatibility
with undistributed checkpoints has not been established by this release.
