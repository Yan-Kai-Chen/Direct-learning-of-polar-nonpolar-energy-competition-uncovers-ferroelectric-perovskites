# Descriptor Engineering (Public Release)

This directory contains the **public-release version** of the descriptor-engineering pipeline used for polar–nonpolar pair analysis.

## Purpose

The goal of this module is to provide a **clean, modular, and reproducible public framework** for descriptor generation, while keeping sensitive implementation details abstracted from the open-source release.

This public version preserves:

- the overall workflow structure
- module boundaries
- input/output interfaces
- representative feature categories
- final table generation logic

This public version does **not** expose:

- private feature-selection rules
- detailed local environment thresholds
- proprietary oxidation-state assignment logic
- full internal derived-feature formulas
- the complete private descriptor set used in production studies

## Pipeline Overview

The descriptor pipeline is organized into the following stages:

1. **Site assignment**  
   Parse formula information and assign A/B/X sites.

2. **Elemental mapping**  
   Map selected elemental properties to the assigned A/B/X sites.

3. **A-site geometry**  
   Compute public-safe local geometric descriptors around the A site.

4. **B-site geometry**  
   Compute public-safe local geometric descriptors around the B site.

5. **Ewald features**  
   Compute public-safe electrostatic descriptors using a callback-based oxidation-state interface.

6. **Derived features**  
   Generate a small set of transparent public postprocessed features.

7. **Final export**  
   Select a public-facing subset of columns and write the final descriptor table.

## File Structure

Typical files in this directory include:

- `run_descriptor_pipeline.py`  
  Main entry script for the full descriptor workflow.

- `public_api.py`  
  Public orchestration layer that connects all stages.

- `site_assignment.py`  
  Public-safe A/B/X site assignment stage.

- `elemental_mapping.py`  
  Public-safe elemental-property mapping stage.

- `a_site_geometry.py`  
  Public-safe A-site local geometry stage.

- `b_site_geometry.py`  
  Public-safe B-site local geometry stage.

- `ewald_features.py`  
  Public-safe electrostatic feature stage.

- `derived_features.py`  
  Public-safe derived-feature stage.

- `export_features.py`  
  Final export and public column selection stage.

- `private_rules_local.example.py`  
  Public template showing the expected rule interface.

## Important Note on Private Rules

This repository includes only a **public template** for rule configuration:

- `private_rules_local.example.py`

The actual private rule file:

- `private_rules_local.py`

is intentionally **not included** in the public release.

Users who want to adapt this framework should create their own local rule file based on the example template.

## Input Requirements

The pipeline expects the following inputs:

### 1. Pair table
A CSV file containing at least:

- `Polar_mpid`
- `NPolar_mpid`
- `Polar_pretty_formula`

Additional columns may be used depending on the stage.

### 2. Element property table
A CSV file containing an element-symbol key column and the elemental properties required by the mapping stage.

### 3. Structure directory
A directory containing CIF files for the materials referenced in the pair table.

For the public workflow, CIF filenames are expected to be compatible with normalized material IDs.

## Running the Pipeline

Run the full public descriptor workflow with:

```bash
python run_descriptor_pipeline.py