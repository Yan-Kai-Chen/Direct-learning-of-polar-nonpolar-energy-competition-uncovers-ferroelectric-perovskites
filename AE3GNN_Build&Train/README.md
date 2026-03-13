# Angular Equivariant Pair GNN

This module contains the public-release training pipeline for an **angle-augmented PaiNN-style equivariant graph neural network** designed for polar–nonpolar pair learning.

## Overview

The workflow is organized into two main stages:

1. **Graph construction and caching**  
   Build periodic structure graphs with distance-based edges and explicit angle triplets.

2. **Model training and evaluation**  
   Train an angular PaiNN pair model for predicting the energy difference between polar and nonpolar structures, with an optional one-shot XGBoost residual fusion stage.

This public version is intended to document the workflow and provide a reproducible training framework. It does **not** include the full private dataset, complete structure library, or all internal experimental variants.

## Method Summary

### Stage 1 — Graph construction

For each structure, the graph-building stage:

- reads CIF files
- constructs periodic neighbor graphs using a radius cutoff
- applies per-center top-K neighbor selection
- forces bidirectional edges
- builds angle triplets on primary edges
- saves one cached graph per structure

Each cached graph contains at least:

- atomic numbers
- Cartesian positions
- edge indices
- edge vectors
- edge distances
- periodic shift vectors
- angle-triplet indices
- angle-triplet cosine values

### Stage 2 — Pair model training

The training stage:

- loads cached polar and nonpolar graphs
- builds pair-level mini-batches
- encodes each structure using an angular PaiNN-style equivariant encoder
- forms a pair representation from polar/nonpolar graph embeddings
- predicts the energy difference between the paired structures

The main model is the **GNN itself**.

An optional **XGBoost residual fusion** stage can be enabled to learn residual errors from tabular numeric features after the GNN prediction.

## Repository Structure

Typical files in this module include:

- `CELL 1` or graph-build notebook section  
  Build graph cache with angle triplets.

- `CELL 2` or training notebook section  
  Train the angular PaiNN pair model and evaluate performance.

If refactored into scripts, the structure may look like:

```text
graph_build/          graph construction and cache generation
training/             model training and evaluation
artifacts/            saved graph caches and model outputs
README.md             module description