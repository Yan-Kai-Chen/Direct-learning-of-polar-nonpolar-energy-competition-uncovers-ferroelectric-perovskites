"""Sampling artifacts and runtime-guidance telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from polarevolve.crystal.symmetry import sha256_file
from polarevolve.diffusion.metric import FULL_ASU_STATE_V1, TRANSLATION_QUOTIENT_V1
from polarevolve.diffusion.sampler import GUIDANCE_TELEMETRY_CONTRACT
from polarevolve.runtime.distributed import run_on_primary_and_broadcast
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.sampling import queries


@dataclass(frozen=True)
class SamplingOutputContext:
    """Immutable identities needed to merge rank outputs and write evidence."""

    output_dir: Path
    run: Any
    source_fingerprint: str
    generation_mode: str
    world_size: int
    checkpoint_payload: dict
    model_config: Any
    sampler_config: Any
    physics_config: Any
    physics_guidance: Any
    panel: Any
    query_plan: dict | None
    lattice: Any


def prepare_sampling_output(path: Path, rank: int) -> None:
    def prepare() -> None:
        if path.exists():
            raise FileExistsError(f"sampling output already exists; choose a new run-id: {path}")
        (path / "rank_rows").mkdir(parents=True)
        (path / "cifs").mkdir()

    run_on_primary_and_broadcast(
        prepare,
        rank=rank,
        operation="sampling output preparation",
    )


def append_sample_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def aggregate_guidance_telemetry(rows: list[dict]) -> dict:
    guided = [row for row in rows if bool(row.get("soft_guidance_enabled", False))]
    if not guided:
        return {
            "contract": GUIDANCE_TELEMETRY_CONTRACT, "enabled": False,
            "completed_samples": len(rows), "structure_steps": 0,
            "applied_structure_step_fraction": 0.0,
            "cap_hit_structure_step_fraction": 0.0,
            "mean_proposed_rms_angstrom": 0.0, "mean_applied_rms_angstrom": 0.0,
            "maximum_proposed_rms_angstrom": 0.0, "sigma_bins": [],
            "runtime_target_fields_observed": 0,
            "nonfinite_failure_policy": "fail_fast_no_completed_artifact"}
    aggregate: dict[int, dict] = {}
    pullback_by_rank: dict[int, dict] = {}
    for row in guided:
        if row.get("guidance_telemetry_contract") != GUIDANCE_TELEMETRY_CONTRACT:
            raise ValueError("guided sample is missing the guidance telemetry contract")
        bins = row.get("guidance_sigma_bins")
        if not isinstance(bins, list) or not bins:
            raise ValueError("guided sample is missing sigma-binned guidance telemetry")
        diagnostics = row.get("physics_guidance_runtime_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("guided sample is missing pullback runtime diagnostics")
        pullback_by_rank[int(row["sampling_rank"])] = diagnostics
        for item in bins:
            index = int(item["bin_index"])
            target = aggregate.setdefault(
                index, {"bin_index": index, "sigma_lower": float(item["sigma_lower"]),
                        "sigma_upper": float(item["sigma_upper"]), "structure_steps": 0,
                        "applied_structure_steps": 0, "clipped_structure_steps": 0,
                        "proposed_sum": 0.0, "applied_sum": 0.0,
                        "maximum_proposed_rms_angstrom": 0.0},
            )
            if (
                target["sigma_lower"] != float(item["sigma_lower"])
                or target["sigma_upper"] != float(item["sigma_upper"])
            ):
                raise ValueError("guidance sigma-bin boundaries differ between samples")
            count = int(item["structure_steps"])
            target["structure_steps"] += count
            target["applied_structure_steps"] += int(item["applied_structure_steps"])
            target["clipped_structure_steps"] += int(item["clipped_structure_steps"])
            target["proposed_sum"] += float(item["mean_proposed_rms_angstrom"]) * count
            target["applied_sum"] += float(item["mean_applied_rms_angstrom"]) * count
            target["maximum_proposed_rms_angstrom"] = max(
                target["maximum_proposed_rms_angstrom"],
                float(item["maximum_proposed_rms_angstrom"]),
            )
    sigma_bins = []
    for index in sorted(aggregate):
        item = aggregate[index]
        count = item.pop("structure_steps")
        proposed_sum = item.pop("proposed_sum")
        applied_sum = item.pop("applied_sum")
        sigma_bins.append({
            **item, "structure_steps": count,
            "mean_proposed_rms_angstrom": proposed_sum / count if count else 0.0,
            "mean_applied_rms_angstrom": applied_sum / count if count else 0.0})
    structure_steps = sum(item["structure_steps"] for item in sigma_bins)
    proposed_sum = sum(item["mean_proposed_rms_angstrom"] * item["structure_steps"]
                       for item in sigma_bins)
    applied_sum = sum(item["mean_applied_rms_angstrom"] * item["structure_steps"]
                      for item in sigma_bins)
    pullback_residuals = [
        float(item["maximum_pullback_residual"])
        for item in pullback_by_rank.values()
        if int(item["pullback_audit_observations"]) > 0
    ]
    return {
        "contract": GUIDANCE_TELEMETRY_CONTRACT,
        "enabled": True,
        "completed_samples": len(guided),
        "structure_steps": structure_steps,
        "applied_structure_step_fraction": (
            sum(item["applied_structure_steps"] for item in sigma_bins) / structure_steps
            if structure_steps else 0.0),
        "cap_hit_structure_step_fraction": (
            sum(item["clipped_structure_steps"] for item in sigma_bins) / structure_steps
            if structure_steps else 0.0),
        "mean_proposed_rms_angstrom": proposed_sum / structure_steps,
        "mean_applied_rms_angstrom": applied_sum / structure_steps,
        "maximum_proposed_rms_angstrom": max(
            item["maximum_proposed_rms_angstrom"] for item in sigma_bins),
        "sigma_bins": sigma_bins,
        "runtime_target_fields_observed": 0,
        "pullback_audited_ranks": len(pullback_residuals),
        "maximum_pullback_residual": max(pullback_residuals, default=None),
        "nonfinite_failure_policy": "fail_fast_no_completed_artifact",
        "nonfinite_guidance_failures_in_completed_run": 0,
    }


def finalize_sampling_output(context: SamplingOutputContext) -> None:
    """Merge rank rows and write the one authoritative sampling manifest."""

    output_dir, run = context.output_dir, context.run
    rows = []
    for path in sorted((output_dir / "rank_rows").glob("rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: (row["material_id"], row["candidate"]))
    if context.panel is not None:
        counts: dict[str, int] = {}
        for row in rows:
            material_id = str(row["material_id"])
            counts[material_id] = counts.get(material_id, 0) + 1
        expected = context.panel.material_ids
        observed = frozenset(counts)
        wrong_counts = {key: count for key, count in counts.items()
                        if count != run.candidates}
        if observed != expected or wrong_counts:
            missing = sorted(expected.difference(observed))[:10]
            unexpected = sorted(observed.difference(expected))[:10]
            raise RuntimeError(
                "distributed panel sampling did not preserve exact coverage: "
                f"missing={missing}, unexpected={unexpected}, "
                f"wrong_candidate_counts={dict(list(wrong_counts.items())[:10])}")
    if context.query_plan is not None:
        expected = queries.planned_candidate_count(context.query_plan, run.candidates)
        if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
            raise RuntimeError("query sampling did not preserve exact candidate coverage")
    merged = output_dir / "samples.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    diagnostics = {}
    if context.query_plan is not None:
        diagnostics["candidate_delivery"] = {
            "schema_version": "gt_sge_query_delivery_v1", "cif_policy": run.query_cif_policy,
            "raw_attempts": len(rows), "cifs_written": sum(bool(row.get("cif_relative")) for row in rows),
            "geometry_rejected": sum(
                row.get("delivery_geometry_screen", {}).get("eligible") is False for row in rows),
            "raw_coordinates_retained": True, "resampling": False,
            "screened_yield_is_not_raw_sun": True,
        }
    sampler_payload = asdict(context.sampler_config)
    if context.sampler_config.state_quotient == FULL_ASU_STATE_V1:
        sampler_payload.pop("state_quotient")
    independent_lattice = context.lattice is not None
    physics_guidance = context.physics_guidance
    replay_block = ({"lattice_replay": {
        "contract": "c2l3_shared_lattice_replay_v1",
        "source_samples": str(run.lattice_replay_samples.resolve()),
        "source_sha256": context.lattice.replay_sha256}}
        if run.lattice_replay_samples is not None else {})
    lattice_block = ({"independent_lattice": {
        "contract": "c2l3_one_lattice_per_hard_condition_v1",
        "checkpoint": str(run.lattice_checkpoint.resolve()),
        "checkpoint_sha256": context.lattice.checkpoint_sha256,
        "checkpoint_step": int(context.lattice.payload["global_step"]),
        "target_contract": context.lattice.payload["target_contract"],
        "selection_split": context.lattice.payload.get("selection_split"),
        "candidates_per_condition": 1, "shared_across_coordinate_candidates": True,
        "runtime_inputs": ["hall_number", "space_group_number", "atom_count",
                           "wyckoff_letter", "wyckoff_multiplicity",
                           "wyckoff_free_dimension", "orbit_atomic_number"],
        "bounds": {"minimum_volume_per_atom": run.lattice_minimum_volume_per_atom,
                   "maximum_volume_per_atom": run.lattice_maximum_volume_per_atom,
                   "maximum_aspect_ratio": run.lattice_maximum_aspect_ratio}}}
        if independent_lattice else {})
    panel_block = ({"evaluation_panel": {
        "path": str(run.evaluation_panel.resolve()),
        "selection_sha256": context.panel.selection_sha256,
        "records": context.panel.requested_records, "split": context.panel.split,
        "seed": context.panel.seed}} if context.panel is not None else {})
    atomic_json(
        output_dir / "sampling_manifest.json",
        {
            "schema_version": (
                queries.QUERY_MANIFEST_SCHEMA if context.query_plan is not None
                else "gt_sge_joint_crystal_sampling_manifest_v1"
                if independent_lattice
                else "gt_sge_sampling_manifest_v3"
                if context.sampler_config.state_quotient == TRANSLATION_QUOTIENT_V1
                else "gt_sge_sampling_manifest_v1"
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": context.source_fingerprint,
            "status": "passed" if rows and all(row["status"] == "ok" for row in rows)
            else "failed",
            "generation_mode": context.generation_mode,
            "production_claim": (
                "chemistry_spacegroup_search_experimental" if context.query_plan is not None
                else "conditional_independent_lattice_coordinate_generation_experimental"
                if independent_lattice
                else "coordinate_recovery_only"
            ),
            "diagnostic_only": False,
            "soft_guidance_enabled": run.physics_guidance_enabled,
            "physics_guidance": {
                "contract": physics_guidance.contract if physics_guidance is not None else None,
                "enabled": run.physics_guidance_enabled,
                "mode": "cartesian_energy_gradient_asu_pullback",
                "config": asdict(context.physics_config),
                "provenance": physics_guidance.provenance if physics_guidance is not None else None,
            },
            "state_quotient": context.sampler_config.state_quotient,
            "diffusion_metric": context.sampler_config.diffusion_metric,
            "records_requested": run.records,
            "candidates_per_record": run.candidates,
            "candidate_budget_mode": (
                "per_program" if context.query_plan is not None
                and any(program.get("candidate_budget") is not None
                        for report in context.query_plan["queries"]
                        for program in report["programs"])
                else "uniform"
            ),
            "samples_written": len(rows),
            "world_size": context.world_size,
            "checkpoint": str(run.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(run.checkpoint),
            "checkpoint_step": int(context.checkpoint_payload["global_step"]),
            "cache_manifest_sha256": context.checkpoint_payload["cache_manifest_sha256"],
            "cache_accessed_at_inference": context.query_plan is None,
            "query_plan": "query_plan.json" if context.query_plan is not None else None,
            "structure_quality_status": "not_evaluated_by_sampler",
            **replay_block, **lattice_block, **panel_block,
            "model": context.model_config.to_dict(),
            "sampler": sampler_payload,
            **diagnostics,
            "sampler_diagnostics": {
                **{
                    output: reduce([float(row.get(source, 0.0)) for row in rows])
                    if rows
                    else None
                    for output, source, reduce in (
                        ("mean_clipped_reverse_step_fraction", "clipped_reverse_step_fraction", np.mean),
                        ("maximum_proposed_step_rms_angstrom", "maximum_proposed_step_rms_angstrom", max),
                        ("mean_final_denoise_clipped_fraction", "final_denoise_clipped_fraction", np.mean),
                    )
                },
                "runtime_guidance": aggregate_guidance_telemetry(rows),
            },
        },
    )
    print(f"[OK] samples={len(rows)} output={merged}", flush=True)


__all__ = [
    "SamplingOutputContext",
    "aggregate_guidance_telemetry",
    "append_sample_row",
    "finalize_sampling_output",
    "prepare_sampling_output",
]
