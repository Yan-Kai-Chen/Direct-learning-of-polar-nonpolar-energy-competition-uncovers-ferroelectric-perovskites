#!/usr/bin/env python3
"""Weighted assistant-only Qwen3-32B LoRA trainer for four-GPU model parallelism."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


@dataclass
class PackedBlock:
    input_ids: list[int]
    labels: list[int]
    loss_weights: list[float]
    valid_length: int
    valid_assistant_tokens: int
    weighted_assistant_tokens: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def disk_free_gb(path: Path) -> int:
    output = subprocess.check_output(
        ["df", "-BG", "--output=avail", str(path)],
        text=True,
    )
    return int(
        "".join(char for char in output.splitlines()[-1] if char.isdigit())
    )


def gpu_snapshot() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return output.strip().splitlines()
    except Exception as exc:
        return [f"nvidia-smi failed: {exc!r}"]


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-step-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def prune_checkpoints(
    output_dir: Path,
    recent_limit: int,
    milestone_every: int,
) -> list[str]:
    checkpoints = sorted(
        [
            path
            for path in output_dir.glob("checkpoint-step-*")
            if path.is_dir()
        ],
        key=checkpoint_step,
    )
    protected = (
        set(checkpoints[-recent_limit:]) if recent_limit > 0 else set()
    )
    if milestone_every > 0:
        protected.update(
            path
            for path in checkpoints
            if checkpoint_step(path) % milestone_every == 0
        )
    removed: list[str] = []
    for path in checkpoints:
        if path in protected:
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def lr_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(
        1,
        total_steps - warmup_steps,
    )
    return max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))),
    )


def fixed_device_map() -> dict[str, int]:
    """Keep every Qwen3 decoder layer intact on exactly one GPU."""
    mapping: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 3,
        "lm_head": 3,
    }
    ranges = {
        0: range(0, 14),
        1: range(14, 31),
        2: range(31, 49),
        3: range(49, 64),
    }
    for device, layer_range in ranges.items():
        for layer_index in layer_range:
            mapping[f"model.layers.{layer_index}"] = device
    return mapping


def validate_device_map(model: Any) -> dict[str, Any]:
    actual = getattr(model, "hf_device_map", None)
    if not isinstance(actual, dict):
        raise RuntimeError("model does not expose hf_device_map")
    expected = fixed_device_map()
    mismatches = {
        name: {"expected": device, "actual": actual.get(name)}
        for name, device in expected.items()
        if actual.get(name) != device
    }
    invalid_devices = {
        name: device
        for name, device in actual.items()
        if str(device) in {"cpu", "disk", "meta"}
    }
    if mismatches or invalid_devices:
        raise RuntimeError(
            "device-map validation failed: "
            + json.dumps(
                {
                    "mismatches": mismatches,
                    "invalid_devices": invalid_devices,
                },
                sort_keys=True,
            )
        )
    return {
        "layers_per_gpu": {
            "0": 14,
            "1": 17,
            "2": 18,
            "3": 15,
        },
        "embedding_device": actual["model.embed_tokens"],
        "lm_head_device": actual["lm_head"],
        "map_entries": len(actual),
    }


def make_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    min_prefix_tokens: int,
    stats: dict[str, Any],
) -> tuple[list[int], list[int], list[float]] | None:
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        stats["bad_messages"] += 1
        return None
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    stats["template_enable_thinking_false"] += 1
    if "<think>\n\n</think>" not in prefix_text:
        raise RuntimeError("non-thinking Qwen3 scaffold was not rendered")

    eos = tokenizer.eos_token
    if not eos:
        raise RuntimeError("tokenizer.eos_token is required")
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(
        str(messages[-1].get("content") or "") + eos,
        add_special_tokens=False,
    )["input_ids"]
    if not answer_ids:
        stats["zero_answer_tokens"] += 1
        return None
    sample_weight = float(row.get("sample_weight", 1.0))
    if not math.isfinite(sample_weight) or sample_weight <= 0.0:
        raise RuntimeError(f"invalid sample_weight={sample_weight!r}")

    original_prefix = len(prefix_ids)
    original_answer = len(answer_ids)
    max_answer = max_length - min_prefix_tokens
    if len(answer_ids) > max_answer:
        answer_ids = answer_ids[:max_answer]
        stats["answer_truncated"] += 1
    prefix_capacity = max_length - len(answer_ids)
    if len(prefix_ids) > prefix_capacity:
        prefix_ids = prefix_ids[-prefix_capacity:]
        stats["prefix_left_truncated"] += 1

    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    loss_weights = [0.0] * len(prefix_ids) + [
        sample_weight
    ] * len(answer_ids)
    if not any(label != -100 for label in labels):
        raise RuntimeError("assistant-only label construction failed")
    stats["examples"] += 1
    stats["original_prefix_tokens"] += original_prefix
    stats["original_answer_tokens"] += original_answer
    stats["kept_assistant_tokens"] += len(answer_ids)
    stats["weighted_assistant_tokens"] += (
        len(answer_ids) * sample_weight
    )
    stats["sample_weight_sum"] += sample_weight
    return input_ids, labels, loss_weights


def build_packed_blocks(
    tokenizer: Any,
    train_file: Path,
    max_length: int,
    min_prefix_tokens: int,
    seed: int,
    expected_rows: int,
) -> tuple[list[PackedBlock], dict[str, Any]]:
    rows: list[tuple[list[int], list[int], list[float]]] = []
    stats = {
        key: 0
        for key in [
            "examples",
            "bad_messages",
            "zero_answer_tokens",
            "answer_truncated",
            "prefix_left_truncated",
            "template_enable_thinking_false",
            "original_prefix_tokens",
            "original_answer_tokens",
            "kept_assistant_tokens",
            "weighted_assistant_tokens",
            "sample_weight_sum",
        ]
    }
    with train_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = make_example(
                tokenizer,
                json.loads(line),
                max_length,
                min_prefix_tokens,
                stats,
            )
            if example is not None:
                rows.append(example)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} usable rows, got {len(rows)}"
        )
    random.Random(seed).shuffle(rows)

    blocks: list[PackedBlock] = []
    buffer_ids: list[int] = []
    buffer_labels: list[int] = []
    buffer_weights: list[float] = []
    for input_ids, labels, loss_weights in rows:
        if buffer_ids and len(buffer_ids) + len(input_ids) > max_length:
            blocks.append(
                PackedBlock(
                    input_ids=buffer_ids,
                    labels=buffer_labels,
                    loss_weights=buffer_weights,
                    valid_length=len(buffer_ids),
                    valid_assistant_tokens=sum(
                        label != -100 for label in buffer_labels
                    ),
                    weighted_assistant_tokens=sum(buffer_weights),
                )
            )
            buffer_ids, buffer_labels, buffer_weights = [], [], []
        buffer_ids.extend(input_ids)
        buffer_labels.extend(labels)
        buffer_weights.extend(loss_weights)
        if len(buffer_ids) == max_length:
            blocks.append(
                PackedBlock(
                    input_ids=buffer_ids,
                    labels=buffer_labels,
                    loss_weights=buffer_weights,
                    valid_length=max_length,
                    valid_assistant_tokens=sum(
                        label != -100 for label in buffer_labels
                    ),
                    weighted_assistant_tokens=sum(buffer_weights),
                )
            )
            buffer_ids, buffer_labels, buffer_weights = [], [], []

    if buffer_ids:
        if tokenizer.pad_token_id is None:
            raise RuntimeError("tokenizer.pad_token_id is required")
        valid_length = len(buffer_ids)
        buffer_ids.extend(
            [tokenizer.pad_token_id] * (max_length - valid_length)
        )
        buffer_labels.extend([-100] * (max_length - valid_length))
        buffer_weights.extend([0.0] * (max_length - valid_length))
        blocks.append(
            PackedBlock(
                input_ids=buffer_ids,
                labels=buffer_labels,
                loss_weights=buffer_weights,
                valid_length=valid_length,
                valid_assistant_tokens=sum(
                    label != -100 for label in buffer_labels
                ),
                weighted_assistant_tokens=sum(buffer_weights),
            )
        )

    if not blocks or any(
        block.valid_assistant_tokens <= 0
        or block.weighted_assistant_tokens <= 0
        for block in blocks
    ):
        raise RuntimeError("packed blocks contain zero-assistant-label block")
    stats.update(
        {
            "packed_blocks": len(blocks),
            "packed_token_utilization": sum(
                block.valid_length for block in blocks
            )
            / (len(blocks) * max_length),
            "mean_assistant_tokens_per_block": sum(
                block.valid_assistant_tokens for block in blocks
            )
            / len(blocks),
            "weighted_assistant_tokens_packed": sum(
                block.weighted_assistant_tokens for block in blocks
            ),
            "mean_weighted_assistant_tokens_per_block": sum(
                block.weighted_assistant_tokens for block in blocks
            )
            / len(blocks),
        }
    )
    return blocks, stats


def tensor_batch(
    block: PackedBlock,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    input_ids = torch.tensor(
        [block.input_ids],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        [block.labels],
        dtype=torch.long,
        device=device,
    )
    loss_weights = torch.tensor(
        [block.loss_weights],
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, : block.valid_length] = 1
    return input_ids, attention_mask, labels, loss_weights


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    optimizer_step: int,
    micro_step: int,
    learning_rate: float,
    recent_limit: int,
    milestone_every: int,
) -> tuple[Path, list[str]]:
    checkpoint_dir = output_dir / f"checkpoint-step-{optimizer_step}"
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    state = {
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "learning_rate": learning_rate,
        "route": "polargen_crystallography_lora",
    }
    (checkpoint_dir / "TRAINING_STATE.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        checkpoint_dir,
        prune_checkpoints(
            output_dir,
            recent_limit,
            milestone_every,
        ),
    )


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config.resolve())
    if cfg.get("loss_weight_field") != "sample_weight":
        raise RuntimeError(
            "weighted training requires loss_weight_field='sample_weight'"
        )
    seed = int(cfg["seed"])
    set_seed(seed)

    model_dir = Path(cfg["model_dir"])
    train_file = Path(cfg["train_file"])
    output_dir = Path(cfg["output_dir"])
    report_path = Path(cfg["report_path"])
    metrics_path = Path(cfg["metrics_jsonl"])
    status_path = Path(cfg["status_path"])
    packing_manifest_path = Path(cfg["packing_manifest_path"])
    storage_guard_path = Path(cfg["storage_guard_path"])
    lora_cfg = load_json(Path(cfg["lora_config"]))
    max_length = int(cfg["max_length"])
    min_prefix_tokens = int(cfg["min_prefix_tokens"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    learning_rate = float(cfg["learning_rate"])
    warmup_ratio = float(cfg["warmup_ratio"])
    max_grad_norm = float(cfg["max_grad_norm"])
    weight_decay = float(cfg["weight_decay"])
    save_every = int(cfg["save_every_optimizer_steps"])
    recent_limit = int(cfg["save_recent_limit"])
    milestone_every = int(cfg["milestone_every_optimizer_steps"])
    expected_rows = int(cfg["expected_train_rows"])
    min_free_start = int(cfg["min_free_gb_start"])
    min_free_runtime = int(cfg["min_free_gb_runtime"])

    if disk_free_gb(storage_guard_path) < min_free_start:
        raise RuntimeError("storage free space is below the start guard")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    packing_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"output_dir is not empty: {output_dir}")
    status_path.write_text("PREPARING\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks, packing = build_packed_blocks(
        tokenizer,
        train_file,
        max_length,
        min_prefix_tokens,
        seed,
        expected_rows,
    )
    total_steps = math.ceil(len(blocks) / grad_accum)
    packing.update(
        {
            "run_name": cfg["run_name"],
            "train_file": str(train_file),
            "max_length": max_length,
            "gradient_accumulation_steps": grad_accum,
            "optimizer_steps": total_steps,
            "enable_thinking": False,
        }
    )
    packing_manifest_path.write_text(
        json.dumps(packing, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"event": "packing_complete", **packing},
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=fixed_device_map(),
        low_cpu_mem_usage=True,
    )
    device_map_report = validate_device_map(model)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**lora_cfg))
    model.train()
    model.print_trainable_parameters()
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, warmup_steps, total_steps),
    )
    optimizer.zero_grad(set_to_none=True)
    first_device = model.get_input_embeddings().weight.device
    status_path.write_text("RUNNING\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "first_device": str(first_device),
                "device_map": device_map_report,
                "gpu": gpu_snapshot(),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )

    start = time.time()
    optimizer_step = 0
    raw_losses: list[float] = []
    status = "FAILED"
    error: str | None = None
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for index, block in enumerate(blocks):
                group_start = (index // grad_accum) * grad_accum
                group_end = min(
                    group_start + grad_accum,
                    len(blocks),
                )
                group_valid_tokens = sum(
                    candidate.valid_assistant_tokens
                    for candidate in blocks[group_start:group_end]
                )
                group_weighted_tokens = sum(
                    candidate.weighted_assistant_tokens
                    for candidate in blocks[group_start:group_end]
                )
                (
                    input_ids,
                    attention_mask,
                    labels,
                    loss_weights,
                ) = tensor_batch(
                    block,
                    first_device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].to(logits.device).contiguous()
                shift_weights = (
                    loss_weights[:, 1:].to(logits.device).contiguous()
                )
                valid_tokens = int(
                    (shift_labels != -100).sum().item()
                )
                if valid_tokens <= 0:
                    raise RuntimeError("zero assistant labels after shift")
                weighted_tokens = float(shift_weights.sum().item())
                if weighted_tokens <= 0.0:
                    raise RuntimeError(
                        "zero weighted assistant tokens after shift"
                    )
                token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                raw_loss = (
                    token_loss
                    * shift_weights.view(-1).to(token_loss.dtype)
                ).sum() / weighted_tokens
                (
                    raw_loss
                    * weighted_tokens
                    / group_weighted_tokens
                ).backward()
                raw_losses.append(float(raw_loss.detach().cpu()))
                micro_step = index + 1
                if (
                    micro_step % grad_accum
                    and micro_step != len(blocks)
                ):
                    continue

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        max_grad_norm,
                    )
                    .detach()
                    .cpu()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                elapsed = time.time() - start
                free_runtime = disk_free_gb(storage_guard_path)
                record = {
                    "optimizer_step": optimizer_step,
                    "target_optimizer_steps": total_steps,
                    "micro_step": micro_step,
                    "target_micro_steps": len(blocks),
                    "raw_loss": raw_losses[-1],
                    "mean_raw_loss_recent": sum(
                        raw_losses[-grad_accum:]
                    )
                    / min(grad_accum, len(raw_losses)),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "valid_assistant_tokens": valid_tokens,
                    "group_valid_assistant_tokens": group_valid_tokens,
                    "weighted_assistant_tokens": weighted_tokens,
                    "group_weighted_assistant_tokens": (
                        group_weighted_tokens
                    ),
                    "elapsed_sec": round(elapsed, 2),
                    "optimizer_steps_per_sec": round(
                        optimizer_step / elapsed,
                        6,
                    )
                    if elapsed
                    else None,
                    "storage_free_gb": free_runtime,
                    "gpu": gpu_snapshot(),
                }
                print(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                metrics_handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_handle.flush()
                if free_runtime < min_free_runtime:
                    raise RuntimeError(
                        "storage free space is below the runtime guard"
                    )
                if (
                    save_every > 0
                    and optimizer_step % save_every == 0
                ):
                    checkpoint_dir, removed = save_checkpoint(
                        model,
                        tokenizer,
                        output_dir,
                        optimizer_step,
                        micro_step,
                        scheduler.get_last_lr()[0],
                        recent_limit,
                        milestone_every,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint_saved",
                                "path": str(checkpoint_dir),
                                "pruned": removed,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        status = "DONE"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        error = "KeyboardInterrupt"
        raise
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        report = {
            "status": status,
            "error": error,
            "run_name": cfg["run_name"],
            "route": "polargen_crystallography_lora",
            "packing": packing,
            "device_map": device_map_report,
            "optimizer_steps": optimizer_step,
            "target_optimizer_steps": total_steps,
            "elapsed_sec": round(time.time() - start, 2),
            "mean_raw_loss": (
                sum(raw_losses) / len(raw_losses)
                if raw_losses
                else None
            ),
            "last_raw_loss": raw_losses[-1] if raw_losses else None,
            "storage_free_gb_final": disk_free_gb(
                storage_guard_path
            ),
        }
        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        status_path.write_text(status + "\n", encoding="utf-8")
        print(
            json.dumps(
                report,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
