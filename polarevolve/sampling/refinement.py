"""Bounded, opt-in overlap refinement; raw diffusion outputs remain authoritative."""
from __future__ import annotations

import time

import torch

from polarevolve.crystal.geometry import GeometryThresholds, device_geometry_screen
from polarevolve.diffusion.metric import apply_state_quotient_tangent, limit_cartesian_rms, metric_inverse_multiply


def refine_overlap(batch, u, physics, *, steps, state_quotient, diffusion_metric):
    if len(batch.hard_conditions) != 1 or not 1 <= steps <= 8:
        raise ValueError("overlap refinement requires one structure and 1--8 steps")
    started = time.monotonic()
    state = u.detach().clone()
    thresholds = GeometryThresholds()
    def screen(x):
        return device_geometry_screen(batch.lattice[0], batch.expand(x), thresholds)
    before = screen(state)
    after, accepted, reason = before, 0, "step_budget"
    if before["eligible"]:
        reason = "already_eligible"
    elif state.numel() == 0:
        reason = "no_active_ASU_parameters"
    elif (not before["finite"] or before["lattice_aspect_ratio"] > thresholds.maximum_aspect_ratio
          or not thresholds.minimum_volume_per_atom <= before["volume_per_atom"] <= thresholds.maximum_volume_per_atom
          or before["minimum_pair_distance"] >= thresholds.minimum_distance):
        reason = "not_a_coordinate_overlap"
    else:
        with torch.enable_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            for _ in range(steps):
                point = state.detach().float().requires_grad_(True)
                energy = physics.energy(batch, point).overlap_by_structure.sum()
                gradient, = torch.autograd.grad(energy, point)
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError("nonfinite_overlap_gradient")
                delta = -metric_inverse_multiply(batch, gradient, metric_convention=diffusion_metric)
                delta = apply_state_quotient_tangent(batch, delta, state_quotient, metric_convention=diffusion_metric)
                delta, _, _ = limit_cartesian_rms(batch, delta, .05)
                for backtrack in range(4):
                    trial = torch.remainder(point.detach() + delta.detach() * (0.5 ** backtrack), 1.)
                    with torch.no_grad():
                        value = physics.energy(batch, trial).overlap_by_structure.sum()
                        candidate_screen = screen(trial)
                    if (bool(value < energy.detach()) and candidate_screen["finite"]
                            and candidate_screen["minimum_pair_distance"] >= after["minimum_pair_distance"]):
                        state, after = trial, candidate_screen
                        accepted += 1
                        break
                else:
                    reason = "no_monotone_step"
                    break
                if after["eligible"]:
                    reason = "geometry_eligible"
                    break
    return state.detach(), {
        "schema_version": "gt_sge_overlap_refinement_v1", "raw_geometry": before, "geometry": after,
        "steps_requested": steps, "steps_accepted": accepted, "stop_reason": reason,
        "maximum_step_rms_angstrom": .05, "backtracking_attempts": 4,
        "objective": "existing_physics_overlap_only", "resampling": False,
        "elapsed_seconds": time.monotonic() - started, "stability": "not_evaluated",
    }
