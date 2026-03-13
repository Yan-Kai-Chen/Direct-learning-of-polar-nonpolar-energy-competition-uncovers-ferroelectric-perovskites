# Polar–Nonpolar Pair Model

This folder contains the code and example metadata for the polar–nonpolar pair retrieval model.

## Contents

This subdirectory includes:

- core source code for the pair-retriever model
- training entry scripts
- example configuration files
- **a sample training/test split file set provided only as an example**

## Example Split Files

The split-related files included in this folder are provided only to demonstrate the expected file format for training and evaluation.

These example files show how:

- positive polar–nonpolar pairs are stored
- train/test splits are organized
- split metadata is recorded for model training

They are intended as **example metadata only**, not as the complete dataset used in large-scale experiments.

## Important Note

This folder does **not** include:

- the full CIF structure library
- the full training dataset
- graph cache files
- trained model weights
- large intermediate outputs

Users should prepare their own structure files and full datasets following the same format.

## Expected Pair File Format

The positive pair CSV files are expected to contain at least the following columns:

- `Polar_mpid`
- `NPolar_mpid`

Additional split or grouping information may be stored in separate CSV, TXT, or JSON files.

## Configuration

An example configuration file is provided to illustrate the expected path and parameter structure.

Users should modify the config file according to their own local environment and data locations before running training.

## Training

The training entry script can be executed with:

```bash
python scripts/train_retriever.py --config configs/train_pair_retriever.example.yaml