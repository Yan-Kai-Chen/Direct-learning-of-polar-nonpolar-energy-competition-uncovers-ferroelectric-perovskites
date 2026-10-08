# Unified Source Layout

The publication repository remains
`Yan-Kai-Chen/Direct-learning-of-polar-nonpolar-energy-competition-uncovers-ferroelectric-perovskites`.
Its Git history is retained; the separate PolarGen repository is not modified.

| Source | New home |
|---|---|
| `AE3GNN_Build&Train/src/ae3gnn` | `polarcomp/` |
| `AE3GNN_Build&Train/scripts` | PolarComp command modules |
| `Polar-Nonpolar_pair_model/src/pair_retriever` | `polarmatch/` |
| Original public retrieval split | `data/polarmatch/splits/` |
| `Descriptor_engineering` | `descriptors/` |
| Supplied `PolarGen/src/diffcsp` | `polarevolve/` |
| Supplied `PolarGen/assets/symmetry` | `polarevolve_assets/` |
| Useful graph/language planning from online PolarGen | `supplement/operation_planning/` |

The core rebuild changes package names, import paths, CLI entry points,
installation metadata, source-text line endings and documentation. It does not redesign the model,
training loss, group-theoretic kernels or guidance algorithm. The existing
public CSVs and bundled crystallographic asset bytes are preserved.

## Deliberately Not Carried Forward

- A superseded notebook launcher and per-module packaging replaced by the
  single core package.
- Python bytecode from the supplied archive.
- The online PolarGen compact reference diffusion and its generic reference
  guidance; the supplied ASU backend is the core generator instead.
- Old generation figures, screening summaries and unrelated historical
  release documents that could conflate two generation implementations.
- Private training assets, model weights, machine paths and full results.

Source revisions and per-file diffusion mappings are recorded in
`source_manifest.json`. Source hashes refer to the received bytes; published
hashes include the documented namespace, packaging and entry-point edits.
