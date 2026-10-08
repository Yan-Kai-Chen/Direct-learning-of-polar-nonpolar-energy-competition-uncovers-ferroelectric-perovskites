"""Bounded one-shot sampling from the independent C2L lattice model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Mapping

import numpy as np
import torch

from polarevolve.runtime.filesystem import iter_jsonl
from polarevolve.crystal.lattice import (
    HallMetricFrame,
    LatticeBounds,
    LatticeCoordinates,
    build_hall_metric_frame,
    decode_lattice_coordinates,
)
from polarevolve.crystal.symmetry import sha256_file
from polarevolve.data.contracts import INDEPENDENT_LATTICE_MODE, LATTICE_REPLAY_FIELDS
from polarevolve.data.panel import EvaluationPanel
from polarevolve.data.packing import (
    LatticeConditionBatch,
    LatticeNormalizer,
    pack_lattice_conditions,
)
from polarevolve.runtime.distributed import run_on_primary_and_broadcast
from polarevolve.sampling.config import SamplingRunConfig
from polarevolve.training.lattice import lattice_model_from_payload, load_lattice_payload


@dataclass
class LatticeSamplingContext:
    """Resolved C2L sampling state and its per-Hall immutable frames."""

    payload: dict
    checkpoint_sha256: str
    bounds: LatticeBounds
    model: torch.nn.Module | None = None
    normalizer: LatticeNormalizer | None = None
    replay: dict[str, dict] | None = None
    replay_sha256: str | None = None
    frames: dict[int, HallMetricFrame] = field(default_factory=dict)

    def draw(self, record, *, seed: int, wyckoff_database) -> tuple[dict, HallMetricFrame]:
        hall = record.hard_condition.hall_number
        frame = self.frames.setdefault(hall, build_hall_metric_frame(wyckoff_database, hall))
        if self.replay is None:
            if self.model is None or self.normalizer is None:
                raise RuntimeError("lattice sampling context has no model")
            sampled = sample_lattice_candidates(
                model=self.model,
                condition=pack_lattice_conditions((record,), frames={hall: frame}),
                normalizer=self.normalizer,
                frames={hall: frame},
                bounds=self.bounds,
                candidates=1,
                seed=seed,
            )[0]
        else:
            sampled = self.replay[record.material_id]
            observed = (sampled["hall_number"], sampled["space_group_number"],
                        sampled["atom_count"], sampled["lattice_seed"])
            expected = (hall, record.space_group_number,
                        record.hard_condition.base_num_atoms, seed)
            if observed != expected:
                raise ValueError(f"replayed lattice hard identity mismatch for {record.material_id!r}")
        return sampled, frame


def prepare_lattice_sampling(
    *,
    run: SamplingRunConfig,
    cache_manifest_sha256: str,
    panel: EvaluationPanel | None,
    device: torch.device,
    rank: int,
) -> LatticeSamplingContext | None:
    """Resolve one independent-lattice model or a verified replay on rank zero."""
    if run.lattice_checkpoint is None:
        return None
    payload, checkpoint_sha256 = run_on_primary_and_broadcast(
        lambda: load_lattice_payload(run.lattice_checkpoint,
                                     cache_manifest_sha256=cache_manifest_sha256),
        rank=rank,
        operation="independent lattice checkpoint load",
    )
    context = LatticeSamplingContext(
        payload=payload,
        checkpoint_sha256=checkpoint_sha256,
        bounds=LatticeBounds(
            minimum_volume_per_atom=run.lattice_minimum_volume_per_atom,
            maximum_volume_per_atom=run.lattice_maximum_volume_per_atom,
            maximum_aspect_ratio=run.lattice_maximum_aspect_ratio),
    )
    if run.lattice_replay_samples is None:
        context.model, context.normalizer = lattice_model_from_payload(payload)
        context.model = context.model.to(device).eval()
    else:
        if panel is None:
            raise ValueError("lattice replay requires a frozen evaluation panel")
        context.replay, context.replay_sha256 = run_on_primary_and_broadcast(
            lambda: load_lattice_replay_samples(
                run.lattice_replay_samples,
                expected_material_ids=panel.material_ids,
                expected_checkpoint_sha256=checkpoint_sha256,
            ),
            rank=rank,
            operation="shared lattice replay load",
        )
    return context


@torch.no_grad()
def sample_lattice_candidates(
    *,
    model: torch.nn.Module,
    condition: LatticeConditionBatch,
    normalizer: LatticeNormalizer,
    frames: Mapping[int, HallMetricFrame],
    bounds: LatticeBounds,
    candidates: int,
    seed: int,
) -> list[dict]:
    """Sample bounded Hall-invariant lattices without target-derived inputs."""

    if candidates <= 0:
        raise ValueError("lattice candidate count must be positive")
    device = next(model.parameters()).device
    condition = condition.to(device)
    normalizer = normalizer.to(device)
    model.eval()
    output = model(condition)
    probability = torch.softmax(output.logits, dim=-1)
    generator = torch.Generator(device=device).manual_seed(seed)
    rows: list[dict] = []
    mask = condition.active_coordinate_mask()
    structure = torch.arange(condition.batch_size, device=device)
    for candidate in range(candidates):
        component = torch.multinomial(
            probability,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
        noise = torch.randn(
            (condition.batch_size, output.means.shape[-1]),
            device=device,
            dtype=output.means.dtype,
            generator=generator,
        )
        normalized = (
            output.means[structure, component]
            + output.log_scales[structure, component].exp() * noise
        )
        coordinates = normalizer.denormalize(normalized)
        coordinates = torch.where(mask, coordinates, torch.zeros_like(coordinates))
        for index in range(condition.batch_size):
            hall = int(condition.hall_numbers[index])
            frame = frames[hall]
            values = coordinates[index].detach().float().cpu().numpy()
            lattice, diagnostics = decode_lattice_coordinates(
                LatticeCoordinates(
                    log_volume_per_atom=float(values[0]),
                    shape_coefficients=tuple(
                        float(value) for value in values[1 : 1 + frame.shape_dimension]
                    ),
                ),
                num_atoms=int(condition.atom_counts[index]),
                frame=frame,
                bounds=bounds,
            )
            rows.append(
                {
                    "condition_index": index,
                    "candidate_index": candidate,
                    "hall_number": hall,
                    "space_group_number": int(condition.space_group_numbers[index]),
                    "atom_count": int(condition.atom_counts[index]),
                    "mixture_component": int(component[index]),
                    "coordinates": values.tolist(),
                    "lattice": np.asarray(lattice).tolist(),
                    "volume_per_atom": diagnostics.volume_per_atom,
                    "aspect_ratio": diagnostics.aspect_ratio,
                    "metric_invariance_error": diagnostics.metric_invariance_error,
                    "volume_was_clipped": diagnostics.volume_was_clipped,
                    "shape_scale": diagnostics.shape_scale,
                }
            )
    return rows


def load_lattice_replay_samples(
    path: Path,
    *,
    expected_material_ids: Collection[str],
    expected_checkpoint_sha256: str,
) -> tuple[dict[str, dict], str]:
    """Load only target-free lattice fields from one completed baseline lane."""

    draws: dict[str, dict] = {}
    for line_number, row in iter_jsonl(path):
        if (
            row.get("schema_version") != "gt_sge_joint_crystal_sample_v1"
            or row.get("generation_mode") != INDEPENDENT_LATTICE_MODE
        ):
            raise ValueError(f"invalid C2L-3 lattice replay row at line {line_number}")
        missing = [field for field in LATTICE_REPLAY_FIELDS if field not in row]
        if missing:
            raise ValueError(f"lattice replay row is missing fields: {missing}")
        if row["lattice_checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("lattice replay checkpoint identity does not match")
        material_id = str(row["material_id"])
        draw = {
            "hall_number": int(row["target_hall_number"]),
            "space_group_number": int(row["target_space_group_number"]),
            "atom_count": int(row["base_num_atoms"]),
            "mixture_component": int(row["lattice_mixture_component"]),
            "coordinates": row["lattice_coordinates"],
            "lattice": row["lattice"],
            "volume_per_atom": float(row["lattice_volume_per_atom"]),
            "aspect_ratio": float(row["lattice_aspect_ratio"]),
            "metric_invariance_error": float(row["lattice_metric_invariance_error"]),
            "volume_was_clipped": bool(row["lattice_volume_was_clipped"]),
            "shape_scale": float(row["lattice_shape_scale"]),
            "lattice_seed": int(row["lattice_seed"]),
            "lattice_checkpoint_sha256": str(row["lattice_checkpoint_sha256"]),
        }
        existing = draws.setdefault(material_id, draw)
        if existing != draw:
            raise ValueError(
                f"lattice replay contains multiple draws for condition {material_id!r}"
            )
    expected = {str(value) for value in expected_material_ids}
    observed = set(draws)
    if observed != expected:
        raise ValueError(
            "lattice replay does not match the frozen panel: "
            f"missing={sorted(expected - observed)[:10]}, "
            f"unexpected={sorted(observed - expected)[:10]}"
        )
    return draws, sha256_file(path)


__all__ = [
    "LatticeSamplingContext",
    "load_lattice_replay_samples",
    "prepare_lattice_sampling",
    "sample_lattice_candidates",
]
