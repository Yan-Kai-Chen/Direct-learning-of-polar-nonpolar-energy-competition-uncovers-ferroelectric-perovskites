# Three Core Stages

## PolarMatch

`polarmatch` retains the two-tower periodic graph retrieval model and its
polar-identity holdout protocol. Its output is a ranked set of possible
nonpolar partners for a polar query. Candidate identity, ranking and symmetry
score adjustment are distinct from proof of a crystallographic relation.

## PolarComp

`polarcomp` retains the angular equivariant pair-energy network. The input is
an ordered polar/nonpolar structure pair; the regression target is
`Energy_diff_meV`. Complete structures and their identifiers are the boundary
between this model and the other stages. Generated structures must be mapped
to the same input and energy-normalization convention before evaluation.

`descriptors` contains shared composition, local-geometry and electrostatic
features. Descriptor engineering is a supporting utility, not a fourth core
stage or an automatic substitute for an energy model.

## PolarEvolve

`polarevolve.crystal` supplies Hall/Wyckoff programs, parent proposals,
supergroup relations, common-cell mappings and symmetry-breaking mode spaces.
`polarevolve.diffusion`, `models`, `training` and `sampling` implement score
learning and sampling on independent ASU parameters. Equivalent atoms are
rebuilt by the selected symmetry operations; fixed 0D orbits do not acquire
coordinate diffusion variables.

The group relation determines a legal mode space, not a unique structure.
Sampling determines concrete candidates. Scalar OD readouts may constrain
finite-distortion amplitudes and local geometry through optional bounded
low-noise guidance, but do not uniquely choose equivalent polar domains.
Energy assessment remains external to this generator.

## Optional Planning

`supplement/operation_planning` retains the older graph/language operation
selector, numerical heads and fusion. It is a separate installable package
with no import dependency from the core. Its plan schema is not automatically
wired to the new OD definitions or relation compiler. No language hidden state
is passed to the core denoiser.

## Integration Boundary

The repository exposes real stage implementations, not a fabricated combined
checkpoint. It does not bundle an automatic end-to-end search orchestrator,
private training data, trained checkpoints or the original experimental
harness. Those boundaries are listed in `reproducibility.md`.
