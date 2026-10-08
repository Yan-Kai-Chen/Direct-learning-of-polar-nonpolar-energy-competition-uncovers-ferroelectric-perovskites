"""Training-only gradient calibration and immutable objective admission."""

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from polarevolve.crystal.symmetry import sha256_file
from polarevolve.data.packing import pack_decoded_records
from polarevolve.diffusion.schedule import FIXED_SIGMA_WINDOW_V1
from polarevolve.guidance.physics import PhysicsGuidanceConfig
from polarevolve.tasks.mp20.objective import MP20ScoreObjective
from polarevolve.training.checkpoint import load_payload, model_from_payload, diffusion_metric_from_payload, model_state_sha256

CALIBRATION_SCHEMA = "mp20_chemistry_gradient_calibration_v1"


def calibrate_physics(records, checkpoint, prior_path, cache_sha256, device, *,
                      sigma_full_strength=0.10, sigma_cutoff=0.50):
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    payload = load_payload(checkpoint)
    model, _, schedule = model_from_payload(payload, cache_manifest_sha256=cache_sha256)
    objective = MP20ScoreObjective(
        model=model, schedule=schedule, diffusion_metric=diffusion_metric_from_payload(payload),
        sigma_sampling_mode=FIXED_SIGMA_WINDOW_V1, physics_auxiliary_weight=1.0,
        physics_config=PhysicsGuidanceConfig(
            prior_path=str(prior_path),
            sigma_full_strength=sigma_full_strength,
            sigma_cutoff=sigma_cutoff,
        ),
    ).to(device)
    parameters = tuple(model.parameters())
    rows, material_ids = [], []
    torch.manual_seed(20260906)
    for record in records:
        if record.split != "train":
            raise ValueError("physics calibration may use training records only")
        batch = pack_decoded_records((record,)).to(device)
        if not batch.num_parameters:
            continue
        material_ids.append(record.material_id)
        for sigma in (0.05, 0.15, 0.35):
            if sigma > schedule.sigma_max:
                continue
            output = objective(batch, sigma_by_structure=torch.tensor([sigma], device=device))
            score = torch.autograd.grad(output.score_loss, parameters, retain_graph=True, allow_unused=True)
            score_norm = sum(g.square().sum() for g in score if g is not None).sqrt()
            for term in ("physics_loss", "physics_overlap_excess", "physics_bond_radius_excess",
                         "physics_bond_valence_excess", "physics_coordination_excess"):
                gradients = torch.autograd.grad(getattr(output, term), parameters,
                                                retain_graph=True, allow_unused=True)
                norm = sum(g.square().sum() for g in gradients if g is not None).sqrt()
                dot = sum((g * h).sum() for g, h in zip(score, gradients) if g is not None and h is not None)
                rows.append({"material_id": record.material_id, "sigma": sigma, "term": term,
                             "score_gradient_norm": float(score_norm.detach()),
                             "physics_gradient_norm": float(norm.detach()),
                             "gradient_cosine": float((dot / (score_norm * norm).clamp_min(1e-12)).detach())})
        if len(material_ids) >= 32:
            break
    ratios = [r["score_gradient_norm"] / r["physics_gradient_norm"] for r in rows
              if r["term"] == "physics_loss" and r["physics_gradient_norm"] > 1e-10]
    finite = all(math.isfinite(v) for r in rows for v in (
        r["score_gradient_norm"], r["physics_gradient_norm"], r["gradient_cosine"]))
    weight = min(1.0, 0.1 * float(np.median(ratios))) if ratios and finite else None
    return {"schema_version": CALIBRATION_SCHEMA, "status": "passed" if weight else "failed",
            "seed": 20260906, "sigma_grid": [0.05, 0.15, 0.35], "precision": "fp32",
            "sigma_full_strength": sigma_full_strength, "sigma_cutoff": sigma_cutoff,
            "diffusion_metric": diffusion_metric_from_payload(payload),
            "model_parameter_count": sum(p.numel() for p in parameters),
            "elapsed_seconds": time.perf_counter() - started, "device": str(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "split": "train", "material_ids": material_ids, "rows": rows,
            "checkpoint_sha256": sha256_file(checkpoint), "prior_sha256": sha256_file(prior_path),
            "model_state_sha256": model_state_sha256(model.state_dict()),
            "model_config": payload["model_config"], "sigma_schedule": payload["sigma_schedule"],
            "cache_manifest_sha256": cache_sha256, "auxiliary_weight": weight,
            "policy": "10_percent_median_gradient_ratio; absolute_weight_cap_1; frozen_after_calibration",
            "scope": "initial_gradient_scale_not_proof_of_learning_or_compatible_objectives"}


def admit_physics_training(config, source_checkpoint, cache_sha256):
    if config.physics_prior is None:
        return config, None
    if config.physics_calibration is None:
        raise ValueError("v2 physics training requires --physics-calibration before a full run")
    report = json.loads(Path(config.physics_calibration).read_text(encoding="utf-8"))
    if (report.get("schema_version") != CALIBRATION_SCHEMA or report.get("status") != "passed"
            or report.get("split") != "train" or report.get("cache_manifest_sha256") != cache_sha256
            or report.get("prior_sha256") != sha256_file(config.physics_prior)):
        raise ValueError("physics calibration status or prior/cache identity mismatch")
    if source_checkpoint is not None and report["checkpoint_sha256"] != sha256_file(source_checkpoint):
        raise ValueError("physics calibration was not measured on the initialization checkpoint")
    if (float(report.get("sigma_full_strength", 0.10)) != float(config.physics_sigma_full_strength)
            or float(report.get("sigma_cutoff", 0.50)) != float(config.physics_sigma_cutoff)):
        raise ValueError("physics calibration sigma gate does not match the training config")
    weight = report["auxiliary_weight"]
    if not isinstance(weight, (int, float)) or not 0 < weight <= 1:
        raise ValueError("invalid calibrated physics weight")
    return replace(config, physics_auxiliary_weight=weight), sha256_file(config.physics_calibration)
