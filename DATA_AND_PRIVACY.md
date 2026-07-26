# Data and privacy boundary

## Included

This repository tracks only the three public tabular datasets and the public
pair-split manifests already associated with the project. Their sizes and
SHA-256 digests are recorded in `DATA_MANIFEST.json`.

The tables contain material identifiers, compositions, energy differences, and
derived scientific descriptors. They do not contain local filesystem paths,
user names, access tokens, API keys, model checkpoints, or private structure
archives.

## Excluded

The following assets must remain local:

- source CIF collections and any provider-specific bulk downloads;
- graph caches and intermediate descriptor tables;
- trained model weights, optimizer states, and complete prediction exports;
- API keys, access tokens, cookies, credentials, and `.env` files;
- private rule overrides;
- machine-specific paths, host names, scheduler files, and run logs.

The repository `.gitignore` covers these classes. Before contributing, inspect
the staged file list and run the tests from a clean clone.

## Fixed split

`Train_EXAMPLE.csv` is the public split authority. Its `split` column partitions
all rows into `train`, `val`, and `test`. Training and preprocessing must:

1. fit normalization or imputation only on `train`;
2. use `val` for early stopping or model selection;
3. reserve `test` for the final evaluation;
4. exclude `row_idx`, identifiers, the target, and the split label from
   tabular model features.

The public pair-retrieval split is a polar-identity holdout: polar structures
must not overlap between train and test. Reuse of nonpolar candidates is
allowed and is reported by the split audit.

## User-supplied structures

Structure files should be named with the identifier used in `Polar_mpid` or
`NPolar_mpid`, for example `mp-1234.cif`. They are read from a local directory
passed on the command line and are never copied into the repository by the
provided code.
