# Third-Party Notices

PolarGen's independently written public reference is informed by the scientific
literature and interfaces surrounding the following open-source projects. This
repository does not redistribute their production source trees.

## DiffCSP

- Project: Crystal Structure Prediction by Joint Equivariant Diffusion
- Repository: https://github.com/jiaor17/DiffCSP
- License: MIT
- Relevance to PolarGen: crystal graph denoising foundations and joint
  coordinate/lattice diffusion conventions.

## SGEquiDiff

- Project: Space Group Equivariant Crystal Diffusion
- Repository: https://github.com/rees-c/sgequidiff
- License: MIT
- Relevance to PolarGen: conceptual basis for space-group/Wyckoff tangent fields
  and symmetry-equivariant extensions. Those production projectors are not
  included in this public repository.

## Qwen3-32B

- Model: https://huggingface.co/Qwen/Qwen3-32B
- License: Apache-2.0
- Use in PolarGen: base model for the sequential crystallography and
  operation-ranking LoRA adapters.

The base model is not redistributed by this repository. Users must obtain it
from the official model repository and comply with its license.

## Hugging Face libraries

Transformers, PEFT, Accelerate, and Safetensors are used through their public
Python APIs and retain their respective licenses.
