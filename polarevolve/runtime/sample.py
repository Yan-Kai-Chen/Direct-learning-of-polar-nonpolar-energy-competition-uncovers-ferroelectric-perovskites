"""CLI for cache recovery or chemistry/space-group search."""

from __future__ import annotations

import argparse
import os
from dataclasses import fields
from pathlib import Path

from polarevolve import asset_root
from polarevolve.data.contracts import ExternalRoots
from polarevolve.diffusion.sampler import SamplerConfig
from polarevolve.diffusion.update import SAMPLER_INTEGRATORS
from polarevolve.sampling.config import SamplingRunConfig
from polarevolve.sampling.pipeline import run_sampling

# Keep the public CLI spellings while the dataclasses own values and validation.
_SAMPLER_FLAGS = {
    "integrator": "integrator", "maximum_step_rms_angstrom": "maximum-step-rms",
    "terminal_mixing_tolerance": "terminal-mixing-tolerance",
    "guidance_step_scale": "guidance-step-scale",
    "maximum_guidance_step_rms_angstrom": "maximum-guidance-step-rms",
}
_RUN_FLAGS = {"physics_guidance_enabled": "physics-guidance"}
_PATH_FIELDS = {"checkpoint", "lattice_checkpoint", "lattice_replay_samples", "evaluation_panel",
                "queries", "program_selection", "physics_prior"}
_ENV_FIELDS = {"checkpoint", "lattice_checkpoint", "lattice_replay_samples"}
_CHOICES = {"split": ("train", "val", "test"),
            "query_cif_policy": ("all", "geometry_eligible"),
            "integrator": SAMPLER_INTEGRATORS}


def _parser() -> argparse.ArgumentParser:
    run = SamplingRunConfig(checkpoint=Path("checkpoint.pt"), run_id="defaults")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "asset_root", "output_root"):
        default = str(asset_root()) if name == "asset_root" else None
        parser.add_argument("--" + name.replace("_", "-"), default=os.environ.get("POLAREVOLVE_" + name.upper(), default))
    for obj, names, aliases in (
        (run, [f.name for f in fields(run) if f.name != "sampler"], _RUN_FLAGS),
        (run.sampler, _SAMPLER_FLAGS, _SAMPLER_FLAGS),
    ):
        for name in names:
            flag = "--" + aliases.get(name, name.replace("_", "-"))
            default = getattr(obj, name)
            kwargs = {"dest": name, "default": default}
            if name in _ENV_FIELDS:
                kwargs["default"] = os.environ.get("POLAREVOLVE_" + name.upper())
            if name == "run_id":
                kwargs.update(required=True, default=None)
            if isinstance(default, bool):
                kwargs["action"] = "store_true"
            else:
                kwargs["type"] = Path if name in _PATH_FIELDS else type(default)
            if name in _CHOICES:
                kwargs["choices"] = _CHOICES[name]
            parser.add_argument(flag, **kwargs)
    return parser


def main() -> None:
    values = vars(_parser().parse_args())
    if not values["checkpoint"]:
        raise ValueError("--checkpoint or POLAREVOLVE_CHECKPOINT is required")
    roots = ExternalRoots(**{key: values.pop(key) for key in ("data_root", "asset_root", "output_root")})
    sampler = SamplerConfig(**{key: values.pop(key) for key in _SAMPLER_FLAGS})
    for key in _PATH_FIELDS:
        if values.get(key) is not None:
            values[key] = Path(values[key]).expanduser()
    run_sampling(roots=roots, run=SamplingRunConfig(**values, sampler=sampler))


if __name__ == "__main__":
    main()
