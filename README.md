# Direct Learning of Polar–Nonpolar Energy Competition for Ferroelectric Perovskites

This repository contains the public-release code and example data accompanying our workflow for **direct learning of polar–nonpolar energy competition** in perovskite materials. The repository integrates three major components:

1. **Polar–nonpolar pair modeling**
2. **Descriptor engineering**
3. **Angular equivariant graph neural network training**

The overall objective is to learn structure-aware and pair-aware representations that can predict or rank the energetic competition between matched polar and nonpolar structures, enabling data-driven ferroelectric screening and external validation.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Scientific Scope](#scientific-scope)
- [Included Data Files](#included-data-files)
- [Module 1: Polar–Nonpolar Pair Model](#module-1-polarnonpolar-pair-model)
- [Module 2: Descriptor Engineering](#module-2-descriptor-engineering)
- [Module 3: Angular Equivariant GNN](#module-3-angular-equivariant-gnn)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Expected Input Format](#expected-input-format)
- [Outputs](#outputs)
- [Reproducibility Notes](#reproducibility-notes)
- [Scope of This Public Release](#scope-of-this-public-release)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

Ferroelectric discovery depends on understanding the energetic competition between **polar** and **nonpolar** structural states. This repository provides a workflow-level public release for building, training, and evaluating models that operate directly on **polar–nonpolar pairs**, including:

- pair retrieval / pair modeling logic,
- descriptor-based feature engineering,
- angular graph construction with equivariant message passing,
- repeated grouped evaluation,
- optional post-hoc residual fusion with gradient-boosted trees,
- example training and external validation tables.

The code is organized to make the main workflow transparent while keeping certain research-specific engineering details abstracted in the public release where appropriate.

---
## Authors

- **Yankai Chen**, Tsinghua University  
  Email: chenyankai25@mails.tsinghua.edu.cn

- **Yuxuan Wang**, Northwestern Polytechnical University  
  Email: wyx2025201020@mail.nwpu.edu.cn

## Citation

If you use this code, please cite this repository and the associated paper.  
Citation metadata is provided in `CITATION.cff`.
## Repository Structure

Current top-level structure:

```text
.
├── AE3GNN_Build&Train/
├── Descriptor_engineering/
├── Polar-Nonpolar_pair_model/
├── Train_EXAMPLE.csv
├── External_Validation_1.csv
├── External_Validation_2.csv
└── LICENSE

