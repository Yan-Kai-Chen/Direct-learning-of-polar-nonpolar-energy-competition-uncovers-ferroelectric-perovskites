# Supplementary Operation Planning

This optional module retains the useful operation-planning components of the
earlier PolarGen repository. It is **not a required stage of PolarEvolve**.
The GNNs used for PolarMatch and PolarComp remain core models; only this older
operation-ranking GNN and its language branch are supplementary.

## Included

- Periodic graph ranking, magnitude quantiles, direction-label prediction,
  calibration and inference.
- Language hidden-state extraction, operation heads and sequential adapters.
- Language/graph score fusion and Top-3 operation-plan schemas.
- Training/inference scripts and small interface tests.

The `polargen` import namespace is retained for these optional interfaces.
The package distribution is named `polar-operation-planning` to avoid a name
collision with the unified core.

## Installation

From the main repository root:

```bash
python -m pip install -e "supplement/operation_planning[test]"
# Only when using the language branch:
python -m pip install -e "supplement/operation_planning[language]"
python -m pytest -q supplement/operation_planning/tests
```

Scripts and configuration files are under `scripts/` and `configs/` in this
directory. They require externally supplied NP graphs, descriptors, model
assets and training records. Their `--help` interfaces document required
arguments. No checkpoint or language base model is included.

## Boundary With the Core

The operation IDs, predicted signs, magnitudes and intervals are legacy
planning outputs. They are not automatically converted into PolarEvolve OD
targets or symmetry-breaking modes. In particular, a predicted scalar sign
is not a uniquely specified crystallographic displacement vector or domain.
An explicit definition-aware adapter would be needed to connect these outputs
to the new backend; no such connection is asserted by this release.

The earlier compact residual-diffusion reference, generic reference guidance,
generation figures and historical screening metrics are intentionally omitted.
They must not be interpreted as evaluations of the supplied ASU backend.

Source: `Yan-Kai-Chen/PolarGen` at
`cf613b495938dc25053c5c1477125e5aafa04e76`. Original attribution is retained in
`LICENSE` and `THIRD_PARTY_NOTICES.md`.
