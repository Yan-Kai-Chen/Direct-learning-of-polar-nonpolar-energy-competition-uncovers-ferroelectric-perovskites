"""Distributed coordinate-pilot sampling pipeline for MP20."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.distributed import is_initialized

from polarevolve.crystal.io import safe_stem, write_p1_cif
from polarevolve.crystal.geometry import GeometryThresholds, device_geometry_screen
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase
from polarevolve.data.cache import decode_cache_record, iter_cache_records, load_cache_manifest
from polarevolve.data.contracts import ExternalRoots
from polarevolve.data.batch import PackedASUBatch
from polarevolve.data.packing import pack_decoded_records
from polarevolve.data.panel import load_evaluation_panel
from polarevolve.data.provenance import source_fingerprint
from polarevolve.diffusion.metric import TRANSLATION_QUOTIENT_V1
from polarevolve.diffusion.sampler import ASUSamplingResult, GUIDANCE_TELEMETRY_CONTRACT, sample
from polarevolve.guidance.physics import PhysicsGuidance, PhysicsGuidanceConfig
from polarevolve.runtime.distributed import (
    distributed_device, raise_if_any_rank_failed, run_on_primary_and_broadcast,
    seed_everything, shutdown_distributed)
from polarevolve.runtime.filesystem import atomic_json
from polarevolve.sampling.config import INDEPENDENT_LATTICE_MODE, SamplingRunConfig
from polarevolve.sampling.telemetry import (
    SamplingOutputContext, append_sample_row, finalize_sampling_output,
    prepare_sampling_output)
from polarevolve.sampling import queries
from polarevolve.sampling.lattice import prepare_lattice_sampling
from polarevolve.sampling.refinement import refine_overlap
from polarevolve.training.checkpoint import (
    diffusion_metric_from_payload, load_sampling_payload, model_from_payload,
    state_quotient_from_payload)


def _candidate_seed(seed: int, material_id: str, candidate: int) -> int:
    payload = f"{seed}:{material_id}:{candidate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _candidate_validation_error(
    result: ASUSamplingResult,
    batch: PackedASUBatch,
) -> str | None:
    if not result.finite:
        return "sampler reported a non-finite state"
    expected_atoms = int(batch.atom_types.numel())
    if result.normalized_u.shape != batch.parameter_shape:
        return "normalized ASU state shape does not match the packed batch"
    if result.full_fractional.shape != (expected_atoms, 3):
        return "expanded coordinate shape does not match the atom count"
    tensors = {
        "normalized_u": result.normalized_u,
        "full_fractional": result.full_fractional,
        "lattice": batch.lattice,
    }
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all().item()):
            return f"{name} contains non-finite values"
    return None


def run_sampling(*, roots: ExternalRoots, run: SamplingRunConfig) -> None:
    owns_process_group = not is_initialized()
    run_source_fingerprint = source_fingerprint()
    device, rank, world_size, _ = distributed_device()
    seed_everything(run.seed, rank)
    cache_root = Path(roots.data_root) / run.cache_relative
    group_root = Path(roots.asset_root) / run.group_assets_relative
    wyckoff_root = Path(roots.asset_root) / run.wyckoff_assets_relative
    output_dir = Path(roots.output_root) / "mp20" / run.run_id / "sampling"

    wyckoff_database = WyckoffDatabase(wyckoff_root)
    group_database = GroupDatabase(group_root)
    manifest, panel, query_plan = None, None, None
    if run.queries is not None:
        query_plan = run_on_primary_and_broadcast(
            lambda: queries.plan_queries(
                run.queries, groups=group_database, wyckoff=wyckoff_database,
                seed=run.seed, records=run.records, lattices=run.lattices_per_program,
                program_selection=run.program_selection),
            rank=rank, operation="chemistry/space-group query planning",
        )
    else:
        manifest = run_on_primary_and_broadcast(
            lambda: load_cache_manifest(cache_root, wyckoff_database=wyckoff_database),
            rank=rank,
            operation="sampling cache manifest load",
        )
        panel = None
        if run.evaluation_panel is not None:
            panel = run_on_primary_and_broadcast(
                lambda: load_evaluation_panel(
                    run.evaluation_panel,
                    expected_cache_manifest_sha256=manifest.manifest_sha256,
                    expected_split=run.split,
                ),
                rank=rank,
                operation="evaluation panel load",
            )
            if run.records != panel.requested_records:
                raise ValueError("sampling records must equal the frozen evaluation panel size")
    payload = run_on_primary_and_broadcast(
        lambda: load_sampling_payload(run.checkpoint),
        rank=rank,
        operation="sampling checkpoint load",
    )
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast a valid checkpoint payload")
    # Query inference validates checkpoint lineage without accessing the training cache.
    cache_digest = payload["cache_manifest_sha256"] if query_plan is not None else manifest.manifest_sha256
    model, model_config, schedule = model_from_payload(
        payload, cache_manifest_sha256=cache_digest
    )
    state_quotient = state_quotient_from_payload(payload)
    diffusion_metric = diffusion_metric_from_payload(payload)
    independent_lattice = run.lattice_checkpoint is not None
    if independent_lattice and state_quotient != TRANSLATION_QUOTIENT_V1:
        raise ValueError(
            "C2L-3 independent-lattice generation requires translation_quotient_v1"
        )
    lattice_context = prepare_lattice_sampling(
        run=run,
        cache_manifest_sha256=cache_digest,
        panel=panel,
        device=device,
        rank=rank,
    )
    model = model.to(device).eval()
    generation_mode = (
        INDEPENDENT_LATTICE_MODE
        if independent_lattice
        else "verified_cache_lattice_coordinate_pilot"
    )
    if query_plan is not None:
        generation_mode = queries.QUERY_GENERATION_MODE
    prepare_sampling_output(output_dir, rank)
    if query_plan is not None:
        run_on_primary_and_broadcast(
            lambda: atomic_json(output_dir / "query_plan.json", query_plan),
            rank=rank, operation="query plan evidence",
        )
    sampler_config = replace(
        run.sampler,
        schedule=schedule,
        state_quotient=state_quotient,
        diffusion_metric=diffusion_metric,
    )
    physics_config = PhysicsGuidanceConfig(prior_path=str(run.physics_prior) if run.physics_prior else None)
    physics_guidance = (
        PhysicsGuidance(physics_config).to(device)
        if run.physics_guidance_enabled or run.physics_prior is not None or run.overlap_repair_steps
        else None
    )
    trained_prior = payload.get("physics_identity", {}).get("provenance", {}).get("prior")
    if trained_prior and run.physics_guidance_enabled and run.physics_prior is None:
        raise ValueError("v2-trained checkpoint guidance requires its explicit physics prior")
    if physics_guidance is not None and physics_guidance.chemistry_prior is not None:
        if physics_guidance.chemistry_prior.identity["cache_manifest_sha256"] != payload["cache_manifest_sha256"]:
            raise ValueError("runtime prior and coordinate checkpoint training cache differ")
        if trained_prior and trained_prior != physics_guidance.chemistry_prior.identity:
            raise ValueError("runtime physics prior differs from the checkpoint training prior")
    guidance_callback = (
        physics_guidance.cartesian_pullback_covector
        if run.physics_guidance_enabled
        else None
    )
    rank_path = output_dir / "rank_rows" / f"rank_{rank:03d}.jsonl"
    local_error = None
    try:
        if query_plan is not None:
            inputs = queries.iter_query_records(
                query_plan, groups=group_database, wyckoff=wyckoff_database,
                rank=rank, world_size=world_size)
        else:
            inputs = (
                (decode_cache_record(raw, group_database=group_database,
                                     wyckoff_database=wyckoff_database), {})
                for raw in iter_cache_records(
                    manifest, split=run.split, rank=rank, world_size=world_size,
                    global_limit=None if panel is not None else run.records,
                    selected_material_ids=panel.material_ids if panel is not None else None,
                )
            )
        for record, query_metadata in inputs:
            request = query_metadata.get("soft_condition_request")
            preferences = request["conditions"] if request else []
            seed_identity = query_metadata.get("hard_sample_identity", record.material_id)
            if preferences and (not run.physics_guidance_enabled or physics_guidance.chemistry_prior is None):
                raise ValueError("local coordination preferences require enabled v2 physics and its prior")
            if physics_guidance is not None:
                physics_guidance.local_preferences = preferences
            lattice_metadata = {}
            if lattice_context is not None:
                lattice_seed = _candidate_seed(run.seed, seed_identity, -1)
                sampled_lattice, _ = lattice_context.draw(
                    record, seed=lattice_seed, wyckoff_database=wyckoff_database
                )
                batch = pack_decoded_records(
                    (record,),
                    lattice_override=(sampled_lattice["lattice"],),
                    include_clean_target=False,
                ).to(device)
                lattice_metadata = {
                    "lattice_seed": lattice_seed,
                    "lattice_checkpoint_sha256": lattice_context.checkpoint_sha256,
                    **{"lattice_" + name: sampled_lattice[name] for name in (
                        "mixture_component", "coordinates", "volume_per_atom", "aspect_ratio",
                        "metric_invariance_error", "volume_was_clipped", "shape_scale",
                    )},
                }
            else:
                batch = pack_decoded_records((record,)).to(device)
            lattice = (
                np.asarray(sampled_lattice["lattice"], dtype=np.float64).tolist()
                if independent_lattice
                else batch.lattice[0].detach().float().cpu().tolist()
            )
            atomic_numbers = batch.atom_types.detach().cpu().tolist()
            screen_lattice = torch.as_tensor(lattice, device=device, dtype=torch.float64)
            candidate_count = int(query_metadata.get("candidate_budget") or run.candidates)
            for candidate in range(candidate_count):
                candidate_seed = _candidate_seed(run.seed, seed_identity, candidate)
                generator = torch.Generator(device=device).manual_seed(candidate_seed)
                result = sample(
                    model=model,
                    batch=batch,
                    config=sampler_config,
                    generator=generator,
                    guidance=guidance_callback,
                )
                sample_id = f"{safe_stem(record.material_id)}_c{candidate:03d}"
                validation_error = _candidate_validation_error(result, batch)
                common_row = {
                    "atomic_numbers": atomic_numbers,
                    "schema_version": (
                        queries.QUERY_SAMPLE_SCHEMA if query_plan is not None else
                        "gt_sge_joint_crystal_sample_v1"
                        if independent_lattice
                        else (
                            "gt_sge_coordinate_pilot_sample_v3"
                            if state_quotient == TRANSLATION_QUOTIENT_V1
                            else "gt_sge_coordinate_pilot_sample_v1"
                        )
                    ),
                    "sample_id": sample_id,
                    "material_id": record.material_id,
                    "split": record.split,
                    "candidate": candidate,
                    "sampling_rank": rank,
                    "seed": candidate_seed,
                    "generation_mode": generation_mode,
                    "soft_guidance_enabled": run.physics_guidance_enabled,
                    "physics_guidance_contract": (
                        physics_guidance.contract
                        if run.physics_guidance_enabled
                        else None
                    ),
                    "state_quotient": state_quotient,
                    "diffusion_metric": diffusion_metric,
                    "sampler_integrator": sampler_config.integrator,
                    **{
                        field.name: getattr(result, field.name)
                        for field in fields(result)
                        if field.name not in {
                            "normalized_u", "full_fractional", "finite",
                            "state_quotient", "guidance_sigma_bins",
                        }
                    },
                    "guidance_telemetry_contract": GUIDANCE_TELEMETRY_CONTRACT,
                    "guidance_sigma_bins": [
                        asdict(item) for item in result.guidance_sigma_bins
                    ],
                    "physics_guidance_runtime_diagnostics": (
                        physics_guidance.runtime_diagnostics
                        if physics_guidance is not None
                        else None
                    ),
                    "target_hall_number": record.hard_condition.hall_number,
                    "physics_terms": (physics_guidance.summarize(batch, result.normalized_u)
                                      if physics_guidance is not None and not validation_error else None),
                    "target_space_group_number": record.space_group_number,
                    "base_num_atoms": record.hard_condition.base_num_atoms,
                    "lattice": lattice,
                    **(
                        {"evaluation_panel_sha256": panel.selection_sha256}
                        if panel is not None
                        else {}
                    ),
                    "wyckoff_orbits": [
                        asdict(item) for item in record.hard_condition.wyckoff_orbits
                    ],
                    **lattice_metadata,
                    **query_metadata,
                }
                if validation_error is not None:
                    append_sample_row(
                        rank_path,
                        {
                            **common_row,
                            "status": "nonfinite",
                            "error": validation_error,
                            "cif_relative": None,
                        },
                    )
                    continue
                screen = None
                if query_plan is not None:
                    screen = device_geometry_screen(
                        screen_lattice,
                        result.full_fractional.float(), GeometryThresholds(),
                    )
                    common_row["delivery_geometry_screen"] = screen
                fractional = result.full_fractional.detach().float().cpu().tolist()
                if run.overlap_repair_steps:
                    repaired, repair = refine_overlap(
                        batch, result.normalized_u, physics_guidance, steps=run.overlap_repair_steps,
                        state_quotient=state_quotient, diffusion_metric=diffusion_metric,
                    )
                    repaired_fractional = batch.expand(repaired).detach().float().cpu().tolist()
                    repaired_cif = f"overlap_repair/{sample_id}.cif"
                    write_p1_cif(
                        output_dir / repaired_cif, identifier=sample_id, lattice=lattice,
                        fractional=repaired_fractional, atomic_numbers=atomic_numbers,
                        target_hall_number=record.hard_condition.hall_number,
                        target_space_group_number=record.space_group_number,
                    )
                    common_row["overlap_repair"] = {
                        **repair, "cif_relative": repaired_cif,
                        "normalized_u": repaired.detach().float().cpu().tolist(),
                        "fractional_coordinates": repaired_fractional,
                    }
                deliver = screen is None or run.query_cif_policy == "all" or screen["eligible"]
                cif_relative = f"cifs/{sample_id}.cif" if deliver else None
                if deliver:
                    write_p1_cif(
                        output_dir / cif_relative,
                        identifier=sample_id,
                        lattice=lattice,
                        fractional=fractional,
                        atomic_numbers=atomic_numbers,
                        target_hall_number=record.hard_condition.hall_number,
                        target_space_group_number=record.space_group_number,
                    )
                append_sample_row(
                    rank_path,
                    {
                        **common_row,
                        "status": "ok",
                        "normalized_u": result.normalized_u.detach().float().cpu().tolist(),
                        "fractional_coordinates": fractional,
                        "cif_relative": cif_relative,
                    },
                )
            print(
                f"[sample] rank={rank} material={record.material_id} candidates={candidate_count}",
                flush=True,
            )
    except Exception as exc:
        local_error = f"rank={rank} {type(exc).__name__}: {exc}"
    raise_if_any_rank_failed(local_error, operation="rank-local sampling")

    output_context = SamplingOutputContext(
        output_dir=output_dir, run=run, source_fingerprint=run_source_fingerprint,
        generation_mode=generation_mode,
        world_size=world_size, checkpoint_payload=payload, model_config=model_config,
        sampler_config=sampler_config,
        physics_config=physics_config, physics_guidance=physics_guidance,
        panel=panel, query_plan=query_plan, lattice=lattice_context,
    )
    run_on_primary_and_broadcast(
        lambda: finalize_sampling_output(output_context),
        rank=rank,
        operation="sampling result merge",
    )
    if owns_process_group:
        shutdown_distributed()

__all__ = [
    "_candidate_validation_error",
    "run_sampling",
]
