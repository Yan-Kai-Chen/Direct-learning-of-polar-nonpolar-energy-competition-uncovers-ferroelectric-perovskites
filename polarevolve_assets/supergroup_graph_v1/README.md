# supergroup_graph_v1

Verified maximal-supergroup graph over the 530 Hall settings (spglib 2.7.0).

## Derivation

For each Hall setting g the finite group F_g = G/T_conv (T_conv = conventional
cell translations) is enumerated completely; maximal proper subgroups up to
F_g conjugacy lift to maximal space subgroups with the same conventional cell:
translationengleiche (t) edges plus same-conventional-cell klassengleiche edges
(centering resolution). Each subgroup is re-identified to its standard Hall
setting and the child->parent basis transform is solved on a dummy generic-orbit
structure, then validated by exact bidirectional conjugation against the
spglib database operations. Subgroup enumeration retries every representative
of a conjugacy class before declaring a class unsolved.

## Contents

- `supergroup_graph.json`: 3052 directed edges
  (2091 t, 961 k_same_cell),
  keyed by (child_hall, parent_hall, index, transform, origin); each edge stores
  the exact rational basis map x_parent = M x_child + o as Fraction strings.
- `build_summary` inside the payload records anchors (all true), the empty
  polar-SG-without-nonpolar-ancestor audit, Smidt 134 SG-pair reachability,
  and unsolved classes.

## Scope and known gaps

- Prime-index primitive-sublattice klassengleiche edges (cell multiplication)
  are excluded: they preserve the point group, so they cannot connect a polar
  child to a non-polar parent; supercell cell relations are resolved by the
  detection layer. Consequence: 11 of the 134 Smidt SG pairs are structurally
  unreachable in this same-cell graph (10->4, 12->9, 14->9, 191->185, 194->185,
  216->9, 221->9, 221->33, 221->161, 62->9, 65->33) and must be handled by
  detection-layer admission, not graph reachability.
- 9 conjugacy classes failed transform validation and are missing as edges
  (all involve rhombohedral R-axis second settings re-identified under cubic
  or Pn-3 parents): parent->child halls 461->444, 495->436, 504->444,
  506->444, 518->460, 521->458, 524->460, 525->458, 527->460. None of them
  affects Smidt pair reachability; the hexagonal-setting equivalents of the
  same SG relations are present through other classes.
- Wyckoff splitting tables are NOT part of this asset; orbit-merge legality
  is checked at runtime from detected symmetry datasets.
