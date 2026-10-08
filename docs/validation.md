# Release Checks

Local checks on 2026-10-09 used Windows, Python 3.12.14 and CPU PyTorch
2.14.1. This records software-release checks, not new scientific experiments.

| Check | Result |
|---|---|
| Core unit/interface tests | 11 passed |
| Supplementary planner tests | 4 passed |
| Installed CLI `--help` entry points | 9 passed |
| Existing three public CSVs | Original row counts, column counts and SHA-256 hashes preserved |
| Checkpoint-free parent example | SG 99 to SG 221 proposal returned |
| Core wheel | Built and imported outside the source checkout |
| Wheel symmetry assets | Group and Wyckoff readers loaded successfully |
| Release content check | No unintended artifacts/credentials/profile paths detected by the bounded checker |

The public source manifest additionally verifies the published hashes of the
94 supplied source/example/asset files. Crystallographic table bytes and
public split files are exempted from Git newline conversion.

GitHub Actions independently reruns the core tests, optional interface tests,
data/content checks and wheel-asset installation check. Its live run status
is the authority for CI success.

No full training, checkpoint-based sampling, new candidate benchmark or DFT
calculation was performed for this reorganization.
