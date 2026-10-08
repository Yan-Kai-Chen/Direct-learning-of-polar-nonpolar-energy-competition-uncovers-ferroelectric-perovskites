# Third-Party Notices

Project source is distributed under the root MIT license. This does not
replace the licenses of bundled third-party data or optional model assets.

## Bundled crystallographic assets

`polarevolve_assets/` contains Hall settings, Wyckoff gauges, source snapshots,
and a supergroup graph supplied with the PolarEvolve source. The three
database directories retain their original provenance, artifact hashes and
`LICENSES/spglib_COPYING.txt` files. The underlying spglib data are licensed
under BSD-3-Clause. The original asset builders are not part of this release.

- Source: <https://github.com/spglib/spglib>
- Recorded source version: 2.7.0
- Included records: `polarevolve_assets/*/PROVENANCE.json`

## Supplementary operation planning

Selected planning code from <https://github.com/Yan-Kai-Chen/PolarGen> is
retained under `supplement/operation_planning/`, with its MIT license and
third-party notices. Its independently written residual-diffusion reference,
figures and historical generation evaluation are not imported into the core.

Qwen3 base weights and LoRA adapters are not redistributed. Users of the
optional language branch must obtain those assets separately and comply with
their terms. See the supplementary notices.

## Scientific predecessors

DiffCSP and SGEquiDiff are relevant scientific predecessors of crystal
diffusion and symmetry-aware generation. A shared package naming convention
does not establish source identity. The supplied implementation's source
mapping is recorded in `docs/source_manifest.json`.

- DiffCSP: <https://github.com/jiaor17/DiffCSP>
- SGEquiDiff: <https://github.com/rees-c/sgequidiff>

Runtime dependencies retain their respective licenses. Public tables remain
under the repository's existing data contract; user-supplied structures and
elemental-property tables require their own redistribution permissions.
