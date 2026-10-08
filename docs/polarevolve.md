# PolarEvolve: Generating Target Crystal Configurations

## Included Implementation

| Module | Responsibility |
|---|---|
| `crystal/` | Hall/Wyckoff programs, parent channels, group relations, common cells and mode spaces |
| `data/` | External ASU cache contracts and packed training records |
| `diffusion/` | Orbit metrics, periodic corruption, score targets and reverse updates |
| `models/` | Equivariant score network and lattice distribution model |
| `tasks/ferroelectric/` | OD definitions, selection rules, optional guidance and relation records |
| `guidance/` | Chemistry/physics guidance covectors |
| `training/` | Training, validation and checkpoint handling |
| `sampling/` | Conditional queries, checkpoint loading and output records |
| `runtime/` | Train, sample and parent-channel entry points |

Bundled databases are in the installed `polarevolve_assets` package. Use
`polarevolve.asset_root()` instead of assuming a source-checkout path.

## Without a Checkpoint

```bash
python examples/compile_parent_channels.py
polarevolve-channels --cif examples/batio3_polar.cif --output outputs/parent_channels.json
```

The simple example/CLI uses the supplied tolerance-ladder parent proposal
function. It is not an exhaustive group-graph search or a physical generation
result. Explicit relation compilation is separately available through
`crystal.supergroup` and `crystal.parent.compile_relation_parent_channel`.

## Training and Sampling

The shared score-training path expects an externally prepared ASU cache with
the schemas read by `polarevolve.data.cache`. It does not build that cache
from an arbitrary CIF folder. A trained checkpoint is required for sampling.

```bash
polarevolve-train --help
polarevolve-sample --help
```

For users with compatible assets, the command shape is:

```bash
polarevolve-train --data-root /absolute/path/to/data --output-root /absolute/path/to/runs --run-id score_run
polarevolve-sample --data-root /absolute/path/to/data --output-root /absolute/path/to/samples --checkpoint /absolute/path/to/checkpoint.pt --run-id sample_run
```

Additional panel/query/lattice options depend on the task and are listed in
`--help`. The packaged symmetry root is the default and can be overridden
with `--asset-root`. Public environment variables use `POLAREVOLVE_` prefixes.

## OD Boundary

`tasks.ferroelectric.od_guidance.OrbitODGuidance` is the optional orbit-space
guidance implementation. Callers must supply meaningful, compatible targets
and role assignments. No deployed target planner is asserted here. The
generic sampling CLI currently wires the separate physics-guidance path;
the existence of the OD class does not imply that the CLI activates it.

OD readouts are scalar structural constraints, not unique Cartesian
displacement commands. Fixed or inapplicable readouts cannot be independently
controlled by coordinates. Low-noise refinement is damped and bounded; it is
not a guarantee of lower energy or better recovery.

The experimental bidirectional evaluation harness, FE-specific production
orchestration, model checkpoints and pair-energy assessor are not contained
in the supplied backend. PolarComp is provided separately in this repository.
