"""Distributed process ownership and collective error propagation."""

from __future__ import annotations

import hashlib
import os
import random
import traceback
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, Callable, Iterable, Mapping, TypeVar

import numpy as np
import torch
import torch.distributed as dist

T = TypeVar("T")
FIXED_PARAMETER_MEAN_V1 = "fixed_parameter_order_mean_v1"


def _process_group_timeout() -> timedelta:
    raw_seconds = os.environ.get("POLAREVOLVE_DISTRIBUTED_TIMEOUT_SECONDS", "1800")
    try:
        seconds = int(raw_seconds)
    except ValueError as exc:
        raise ValueError("POLAREVOLVE_DISTRIBUTED_TIMEOUT_SECONDS must be an integer") from exc
    if seconds <= 0:
        raise ValueError("POLAREVOLVE_DISTRIBUTED_TIMEOUT_SECONDS must be positive")
    return timedelta(seconds=seconds)


def configure_determinism_from_environment() -> bool:
    raw = os.environ.get("POLAREVOLVE_DETERMINISTIC_ALGORITHMS")
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError("POLAREVOLVE_DETERMINISTIC_ALGORITHMS must be true or false")
    if normalized == "false":
        return False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return True


def distributed_device() -> tuple[torch.device, int, int, int]:
    configure_determinism_from_environment()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("multi-process training requires CUDA/NCCL")
        return torch.device("cpu"), rank, world_size, local_rank
    ranks_per_device = int(os.environ.get("POLAREVOLVE_CUDA_RANKS_PER_DEVICE", "1"))
    device_count = torch.cuda.device_count()
    if ranks_per_device < 1 or local_rank >= device_count * ranks_per_device:
        raise RuntimeError("local CUDA rank exceeds the declared device topology")
    device_index = local_rank % device_count
    backend = os.environ.get(
        "POLAREVOLVE_DISTRIBUTED_BACKEND", "gloo" if ranks_per_device > 1 else "nccl"
    )
    if ranks_per_device > 1 and backend != "gloo":
        raise RuntimeError("shared-GPU sampling ranks require the gloo control plane")
    torch.cuda.set_device(device_index)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=_process_group_timeout(),
        )
    return torch.device("cuda", device_index), rank, world_size, local_rank


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + 10_007 * int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)


def is_primary(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def mean_named_scalars_across_ranks(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Mean a fixed ordered scalar mapping with one tensor collective."""

    names = tuple(values)
    if not names:
        return {}
    scalars = []
    for name in names:
        value = values[name].detach().float()
        if value.numel() != 1:
            raise ValueError(f"distributed mean value {name!r} must be scalar")
        scalars.append(value.reshape(()))
    packed = torch.stack(scalars)
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed /= dist.get_world_size()
    return dict(zip(names, packed.unbind()))


def mean_parameter_gradients_across_ranks(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Fixed layout avoids DDP bucket rebuilding changing resumed reductions."""
    if not dist.is_initialized():
        return
    gradients = [p.grad for p in parameters if p.requires_grad]
    if not gradients or any(g is None for g in gradients):
        raise RuntimeError("fixed gradient mean requires every trainable parameter gradient")
    if any(g.dtype != gradients[0].dtype or g.device != gradients[0].device for g in gradients):
        raise RuntimeError("fixed gradient mean requires one parameter dtype and device")
    sizes = [g.numel() for g in gradients]
    packed = torch.cat([g.reshape(-1) for g in gradients])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed.div_(dist.get_world_size())
    with torch.no_grad():
        for gradient, reduced in zip(gradients, packed.split(sizes)):
            gradient.copy_(reduced.reshape_as(gradient))


def gather_named_1d_across_ranks(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Gather fixed named variable-length diagnostics without object collectives."""

    names = tuple(values)
    if not names:
        return {}
    if not dist.is_initialized():
        return {
            name: values[name].detach().float().reshape(-1) for name in names
        }
    world_size = dist.get_world_size()
    device = values[names[0]].device
    name_digest = int.from_bytes(
        hashlib.sha256("\0".join(names).encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)
    identity = torch.tensor([len(names), name_digest], dtype=torch.int64, device=device)
    identities = [torch.empty_like(identity) for _ in range(world_size)]
    dist.all_gather(identities, identity)
    if any(not torch.equal(item, identity) for item in identities):
        raise RuntimeError("distributed diagnostic names differ across ranks")

    lengths = torch.tensor(
        [values[name].numel() for name in names], dtype=torch.int64, device=device
    )
    gathered_lengths = [torch.empty_like(lengths) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, lengths)
    length_rows = [row.cpu().tolist() for row in gathered_lengths]
    maximum_size = max(sum(row) for row in length_rows)
    if maximum_size == 0:
        return {
            name: values[name].detach().float().reshape(-1) for name in names
        }
    local = torch.cat(
        [values[name].detach().float().reshape(-1).to(device) for name in names]
    )
    padded = torch.zeros(maximum_size, dtype=torch.float32, device=device)
    padded[: local.numel()] = local
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for payload, row in zip(gathered, length_rows):
        offset = 0
        for name, length in zip(names, row):
            pieces[name].append(payload[offset : offset + length])
            offset += length
    return {
        name: torch.cat(pieces[name]).to(device=values[name].device)
        for name in names
    }


def broadcast_object(value: Any, rank: int) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def run_on_primary_and_broadcast(
    action: Callable[[], T],
    *,
    rank: int,
    operation: str,
) -> T:
    envelope: dict[str, Any] | None = None
    if is_primary(rank):
        try:
            envelope = {"ok": True, "value": action()}
        except Exception as exc:
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
    envelope = broadcast_object(envelope, rank)
    if not isinstance(envelope, dict) or "ok" not in envelope:
        raise RuntimeError(f"{operation} returned an invalid rank-0 transaction envelope")
    if not envelope["ok"]:
        detail = f"{envelope.get('error_type', 'Error')}: {envelope.get('error_message', '')}"
        primary_traceback = str(envelope.get("traceback", "")).strip()
        suffix = f"\nRank 0 traceback:\n{primary_traceback}" if primary_traceback else ""
        raise RuntimeError(f"{operation} failed on rank 0: {detail}{suffix}")
    return envelope.get("value")


def raise_if_any_rank_failed(local_error: str | None, *, operation: str) -> None:
    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
    else:
        errors = [local_error]
    failures = [error for error in errors if error]
    if failures:
        detail = "\n".join(failures)
        raise RuntimeError(f"{operation} failed on one or more ranks:\n{detail}")


def maybe_no_sync(module: torch.nn.Module, enabled: bool):
    return module.no_sync() if enabled and hasattr(module, "no_sync") else nullcontext()


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "FIXED_PARAMETER_MEAN_V1",
    "barrier",
    "broadcast_object",
    "configure_determinism_from_environment",
    "distributed_device",
    "gather_named_1d_across_ranks",
    "is_primary",
    "mean_named_scalars_across_ranks",
    "mean_parameter_gradients_across_ranks",
    "maybe_no_sync",
    "raise_if_any_rank_failed",
    "run_on_primary_and_broadcast",
    "seed_everything",
    "shutdown_distributed",
]
