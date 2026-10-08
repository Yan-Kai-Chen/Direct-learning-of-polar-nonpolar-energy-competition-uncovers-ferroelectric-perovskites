#!/usr/bin/env python3
"""Train the PolarGen 18-operation language-ranking adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


CORE_IDS = (
    "OD01",
    "OD02",
    "OD03",
    "OD04",
    "OD05",
    "OD06",
    "OD07",
    "OD08",
    "OD09",
    "OD10",
    "OD11",
    "OD12",
    "OD13",
    "OD17",
    "OD18",
    "OD19",
    "OD20",
    "OD22",
)


@dataclass
class RankExample:
    row_id: str
    input_ids: list[int]
    scores: np.ndarray
    selected: np.ndarray
    sample_weight: float


class PriorHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, len(CORE_IDS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(hidden))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


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


def fixed_device_map() -> dict[str, int]:
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
    invalid = {
        name: device
        for name, device in actual.items()
        if str(device) in {"cpu", "disk", "meta"}
    }
    if mismatches or invalid:
        raise RuntimeError(
            "device-map validation failed: "
            + json.dumps(
                {"mismatches": mismatches, "invalid": invalid},
                sort_keys=True,
            )
        )
    return {
        "layers_per_gpu": {"0": 14, "1": 17, "2": 18, "3": 15},
        "map_entries": len(actual),
    }


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


def load_examples(
    tokenizer: Any,
    input_path: Path,
    label_path: Path,
    max_length: int,
    expected_rows: int,
) -> tuple[list[RankExample], dict[str, Any]]:
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        label_rows = {
            str(row["id"]): row for row in csv.DictReader(handle)
        }
    public_rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(public_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} public rows, got {len(public_rows)}"
        )
    examples: list[RankExample] = []
    token_lengths: list[int] = []
    truncated = 0
    for row in public_rows:
        row_id = str(row["id"])
        label = label_rows.get(row_id)
        if label is None:
            raise RuntimeError(f"missing labels for {row_id}")
        messages = row.get("input_messages")
        if (
            not isinstance(messages, list)
            or [message.get("role") for message in messages]
            != ["system", "user"]
        ):
            raise RuntimeError(f"invalid answer-free messages for {row_id}")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        original_length = len(input_ids)
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            truncated += 1
        scores = np.asarray(
            [float(label[f"score_{op_id}"]) for op_id in CORE_IDS],
            dtype=np.float32,
        )
        selected = np.asarray(
            [int(float(label[f"selected_{op_id}"])) for op_id in CORE_IDS],
            dtype=np.float32,
        )
        sample_weight = float(label["sample_weight"])
        if (
            not np.isfinite(scores).all()
            or int(selected.sum()) != 3
            or not math.isfinite(sample_weight)
            or sample_weight <= 0
        ):
            raise RuntimeError(f"invalid labels for {row_id}")
        examples.append(
            RankExample(
                row_id=row_id,
                input_ids=input_ids,
                scores=scores,
                selected=selected,
                sample_weight=sample_weight,
            )
        )
        token_lengths.append(original_length)
    if len(label_rows) != expected_rows or len(examples) != expected_rows:
        raise RuntimeError("public and private row counts differ")
    return examples, {
        "rows": len(examples),
        "unique_ids": len({example.row_id for example in examples}),
        "min_original_tokens": min(token_lengths),
        "max_original_tokens": max(token_lengths),
        "mean_original_tokens": float(np.mean(token_lengths)),
        "truncated_rows": truncated,
        "max_length": max_length,
        "public_input_sha256": sha256(input_path),
        "private_labels_sha256": sha256(label_path),
    }


def row_losses(
    logits: torch.Tensor,
    scores: torch.Tensor,
    selected: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    operation_count = logits.shape[1]
    scale = math.log(operation_count)
    target_distribution = torch.softmax(scores / temperature, dim=1)
    listwise = -(
        target_distribution * torch.log_softmax(logits, dim=1)
    ).sum(dim=1) / scale
    top1 = F.cross_entropy(
        logits,
        scores.argmax(dim=1),
        reduction="none",
    ) / scale
    positive_weight = torch.tensor(
        5.0,
        device=logits.device,
        dtype=logits.dtype,
    )
    top3 = F.binary_cross_entropy_with_logits(
        logits,
        selected,
        pos_weight=positive_weight,
        reduction="none",
    ).mean(dim=1)
    positive_count = selected.sum(dim=1)
    negative_count = operation_count - positive_count
    top3 = top3 / (
        (negative_count + positive_weight * positive_count) / operation_count
    ).clamp_min(1.0)
    pair_mask = selected[:, :, None] * (1.0 - selected[:, None, :])
    pairwise = (
        F.softplus(-(logits[:, :, None] - logits[:, None, :])) * pair_mask
    ).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    components = {
        "top1": top1,
        "listwise": listwise,
        "pairwise": pairwise,
        "top3": top3,
    }
    total = (
        0.45 * top1
        + 0.25 * listwise
        + 0.20 * pairwise
        + 0.10 * top3
    )
    return total, components


def batch_tensors(
    examples: list[RankExample],
    tokenizer: Any,
    first_device: torch.device,
    output_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(example.input_ids) for example in examples)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for example in examples:
        pad_count = max_length - len(example.input_ids)
        padded.append(
            [tokenizer.pad_token_id] * pad_count + example.input_ids
        )
        masks.append([0] * pad_count + [1] * len(example.input_ids))
    input_ids = torch.tensor(
        padded,
        dtype=torch.long,
        device=first_device,
    )
    attention_mask = torch.tensor(
        masks,
        dtype=torch.long,
        device=first_device,
    )
    scores = torch.tensor(
        np.stack([example.scores for example in examples]),
        dtype=torch.float32,
        device=output_device,
    )
    selected = torch.tensor(
        np.stack([example.selected for example in examples]),
        dtype=torch.float32,
        device=output_device,
    )
    return input_ids, attention_mask, scores, selected


def rank_metrics(
    scores: np.ndarray,
    logits: np.ndarray,
) -> dict[str, float | int]:
    truth = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    prediction = np.argsort(-logits, axis=1, kind="stable")[:, :3]
    top1 = float(np.mean(truth[:, 0] == prediction[:, 0]))
    recalls: list[float] = []
    exact: list[float] = []
    ndcgs: list[float] = []
    discounts = 1.0 / np.log2(np.arange(3) + 2.0)
    for row_index in range(len(scores)):
        truth_set = set(truth[row_index].tolist())
        prediction_set = set(prediction[row_index].tolist())
        recalls.append(len(truth_set & prediction_set) / 3.0)
        exact.append(float(truth_set == prediction_set))
        relevance = np.maximum(scores[row_index], 0.0)
        dcg = float(np.sum(relevance[prediction[row_index]] * discounts))
        ideal = float(np.sum(relevance[truth[row_index]] * discounts))
        ndcgs.append(dcg / ideal if ideal > 0 else 1.0)
    return {
        "rows": len(scores),
        "top1_accuracy": top1,
        "top3_recall": float(np.mean(recalls)),
        "top3_exact_set": float(np.mean(exact)),
        "ndcg_at_3": float(np.mean(ndcgs)),
    }


@torch.inference_mode()
def evaluate(
    model: Any,
    backbone: Any,
    rank_head: PriorHead,
    tokenizer: Any,
    examples: list[RankExample],
    first_device: torch.device,
    output_device: torch.device,
    batch_size: int,
    temperature: float,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    rank_head.eval()
    all_logits: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    losses: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, attention_mask, scores, selected = batch_tensors(
            batch,
            tokenizer,
            first_device,
            output_device,
        )
        output = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, -1, :].float()
        logits = rank_head(hidden)
        row_loss, _components = row_losses(
            logits,
            scores,
            selected,
            temperature,
        )
        losses.extend(row_loss.float().cpu().tolist())
        all_logits.append(logits.float().cpu().numpy())
        all_scores.append(scores.float().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    scores_np = np.concatenate(all_scores, axis=0)
    result = {
        "mean_rank_loss": float(np.mean(losses)),
        **rank_metrics(scores_np, logits_np),
    }
    model.train()
    rank_head.train()
    return result, logits_np


def save_checkpoint(
    model: Any,
    rank_head: PriorHead,
    tokenizer: Any,
    output_dir: Path,
    optimizer_step: int,
    metrics: dict[str, Any] | None,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-step-{optimizer_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    torch.save(
        {
            "schema": "polargen_language_rank_head_v1",
            "core_ids": CORE_IDS,
            "hidden_size": rank_head.norm.normalized_shape[0],
            "state_dict": {
                key: value.detach().cpu()
                for key, value in rank_head.state_dict().items()
            },
            "optimizer_step": optimizer_step,
            "validation_metrics": metrics,
        },
        checkpoint_dir / "rank_head.pt",
    )
    write_json(
        checkpoint_dir / "RANK_HEAD_CONFIG.json",
        {
            "schema": "polargen_language_rank_head_v1",
            "core_ids": list(CORE_IDS),
            "hidden_size": rank_head.norm.normalized_shape[0],
            "representation": "final prompt token at decoder layer 63",
            "optimizer_step": optimizer_step,
            "validation_metrics": metrics,
        },
    )
    return checkpoint_dir


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config.resolve())
    seed = int(cfg["seed"])
    set_seed(seed)
    random.seed(seed)

    run_name = str(cfg["run_name"])
    model_dir = Path(cfg["model_dir"])
    train_input = Path(cfg["train_input"])
    train_labels = Path(cfg["train_labels"])
    val_input = Path(cfg["val_input"])
    val_labels = Path(cfg["val_labels"])
    init_head_path = Path(cfg["init_rank_head"])
    output_dir = Path(cfg["output_dir"])
    run_dir = Path(cfg["run_dir"])
    metrics_path = Path(cfg["metrics_jsonl"])
    report_path = Path(cfg["report_path"])
    status_path = Path(cfg["status_path"])
    selected_path = Path(cfg["selected_checkpoint_path"])
    storage_guard = Path(cfg["storage_guard_path"])
    max_length = int(cfg["max_length"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    lora_lr = float(cfg["lora_learning_rate"])
    head_lr = float(cfg["head_learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    warmup_ratio = float(cfg["warmup_ratio"])
    max_grad_norm = float(cfg["max_grad_norm"])
    eval_every = int(cfg["eval_every_optimizer_steps"])
    eval_batch_size = int(cfg["eval_batch_size"])
    temperature = float(cfg["rank_temperature"])
    min_free_start = int(cfg["min_free_gb_start"])
    min_free_runtime = int(cfg["min_free_gb_runtime"])

    if disk_free_gb(storage_guard) < min_free_start:
        raise RuntimeError("storage free space is below the start guard")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("PREPARING\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_examples, train_stats = load_examples(
        tokenizer,
        train_input,
        train_labels,
        max_length,
        int(cfg["expected_train_rows"]),
    )
    val_examples, val_stats = load_examples(
        tokenizer,
        val_input,
        val_labels,
        max_length,
        int(cfg["expected_val_rows"]),
    )
    data_manifest = {
        "run_name": run_name,
        "train": train_stats,
        "validation": val_stats,
        "core_ids": list(CORE_IDS),
        "answer_free_inputs": True,
        "loss": {
            "top1": 0.45,
            "listwise": 0.25,
            "pairwise": 0.20,
            "top3": 0.10,
            "temperature": temperature,
        },
    }
    write_json(run_dir / "data_manifest.json", data_manifest)
    print(
        json.dumps(
            {"event": "data_ready", **data_manifest},
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=fixed_device_map(),
        low_cpu_mem_usage=True,
    )
    device_map_report = validate_device_map(base_model)
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.train()
    base_causal_model = model.get_base_model()
    backbone = base_causal_model.model
    first_device = model.get_input_embeddings().weight.device
    output_device = backbone.norm.weight.device
    rank_head = PriorHead(int(model.config.hidden_size)).to(
        device=output_device,
        dtype=torch.float32,
    )
    init_head = torch.load(
        init_head_path,
        map_location="cpu",
        weights_only=False,
    )
    rank_head.load_state_dict(init_head["state_dict"], strict=True)
    rank_head.train()

    lora_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    head_parameters = list(rank_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": lora_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": head_parameters,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ]
    )
    total_steps = math.ceil(len(train_examples) / grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, warmup_steps, total_steps),
    )
    optimizer.zero_grad(set_to_none=True)
    status_path.write_text("RUNNING\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "device_map": device_map_report,
                "first_device": str(first_device),
                "output_device": str(output_device),
                "lora_trainable_parameters": sum(
                    parameter.numel() for parameter in lora_parameters
                ),
                "head_trainable_parameters": sum(
                    parameter.numel() for parameter in head_parameters
                ),
                "target_optimizer_steps": total_steps,
                "gpu": gpu_snapshot(),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )

    order = list(range(len(train_examples)))
    random.Random(seed).shuffle(order)
    started = time.time()
    optimizer_step = 0
    raw_losses: list[float] = []
    validation_history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_checkpoint: Path | None = None
    status = "FAILED"
    error: str | None = None
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for micro_index, example_index in enumerate(order):
                group_start = (micro_index // grad_accum) * grad_accum
                group_end = min(group_start + grad_accum, len(order))
                group_weight = sum(
                    train_examples[order[index]].sample_weight
                    for index in range(group_start, group_end)
                )
                example = train_examples[example_index]
                input_ids, attention_mask, scores, selected = batch_tensors(
                    [example],
                    tokenizer,
                    first_device,
                    output_device,
                )
                output = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state[:, -1, :].float()
                logits = rank_head(hidden)
                row_loss, components = row_losses(
                    logits,
                    scores,
                    selected,
                    temperature,
                )
                raw_loss = row_loss.mean()
                (
                    raw_loss
                    * example.sample_weight
                    / max(group_weight, 1.0e-12)
                ).backward()
                raw_losses.append(float(raw_loss.detach().cpu()))
                micro_step = micro_index + 1
                if micro_step % grad_accum and micro_step != len(order):
                    continue

                all_parameters = lora_parameters + head_parameters
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        all_parameters,
                        max_grad_norm,
                    )
                    .detach()
                    .cpu()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                elapsed = time.time() - started
                record = {
                    "optimizer_step": optimizer_step,
                    "target_optimizer_steps": total_steps,
                    "micro_step": micro_step,
                    "target_micro_steps": len(order),
                    "raw_loss": raw_losses[-1],
                    "mean_raw_loss_recent": float(
                        np.mean(raw_losses[-grad_accum:])
                    ),
                    "top1_component": float(
                        components["top1"].mean().detach().cpu()
                    ),
                    "listwise_component": float(
                        components["listwise"].mean().detach().cpu()
                    ),
                    "pairwise_component": float(
                        components["pairwise"].mean().detach().cpu()
                    ),
                    "top3_component": float(
                        components["top3"].mean().detach().cpu()
                    ),
                    "lora_learning_rate": scheduler.get_last_lr()[0],
                    "head_learning_rate": scheduler.get_last_lr()[1],
                    "grad_norm": grad_norm,
                    "elapsed_sec": round(elapsed, 2),
                    "optimizer_steps_per_sec": (
                        round(optimizer_step / elapsed, 6)
                        if elapsed
                        else None
                    ),
                    "storage_free_gb": disk_free_gb(storage_guard),
                    "gpu": gpu_snapshot(),
                }
                metrics_handle.write(
                    json.dumps(record, ensure_ascii=True, sort_keys=True)
                    + "\n"
                )
                metrics_handle.flush()
                print(
                    json.dumps(record, ensure_ascii=True, sort_keys=True),
                    flush=True,
                )
                if record["storage_free_gb"] < min_free_runtime:
                    raise RuntimeError(
                        "storage free space is below the runtime guard"
                    )

                should_evaluate = (
                    optimizer_step % eval_every == 0
                    or optimizer_step == total_steps
                )
                if should_evaluate:
                    validation, val_logits = evaluate(
                        model,
                        backbone,
                        rank_head,
                        tokenizer,
                        val_examples,
                        first_device,
                        output_device,
                        eval_batch_size,
                        temperature,
                    )
                    validation["optimizer_step"] = optimizer_step
                    validation_history.append(validation)
                    checkpoint_dir = save_checkpoint(
                        model,
                        rank_head,
                        tokenizer,
                        output_dir,
                        optimizer_step,
                        validation,
                    )
                    np.savez(
                        checkpoint_dir / "val_logits.npz",
                        ids=np.asarray(
                            [example.row_id for example in val_examples],
                            dtype=np.str_,
                        ),
                        core_ids=np.asarray(CORE_IDS, dtype=np.str_),
                        logits=val_logits.astype(np.float32),
                    )
                    selection_key = (
                        float(validation["top1_accuracy"]),
                        float(validation["top3_recall"]),
                        float(validation["ndcg_at_3"]),
                    )
                    if best_key is None or selection_key > best_key:
                        best_key = selection_key
                        best_checkpoint = checkpoint_dir
                        selected_path.write_text(
                            str(checkpoint_dir) + "\n",
                            encoding="utf-8",
                        )
                    print(
                        json.dumps(
                            {
                                "event": "validation_checkpoint",
                                "path": str(checkpoint_dir),
                                "selected": checkpoint_dir
                                == best_checkpoint,
                                **validation,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        torch.save(
            {
                "schema": "polargen_language_rank_head_v1",
                "core_ids": CORE_IDS,
                "hidden_size": rank_head.norm.normalized_shape[0],
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in rank_head.state_dict().items()
                },
                "optimizer_step": optimizer_step,
            },
            final_dir / "rank_head.pt",
        )
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
            "run_name": run_name,
            "route": "polargen_operation_rank_lora",
            "model_dir": str(model_dir),
            "train": train_stats,
            "validation": val_stats,
            "optimizer_steps": optimizer_step,
            "target_optimizer_steps": total_steps,
            "mean_raw_loss": (
                float(np.mean(raw_losses)) if raw_losses else None
            ),
            "last_raw_loss": raw_losses[-1] if raw_losses else None,
            "validation_history": validation_history,
            "selected_checkpoint": (
                str(best_checkpoint) if best_checkpoint else None
            ),
            "elapsed_sec": round(time.time() - started, 2),
            "storage_free_gb_final": disk_free_gb(storage_guard),
            "init_rank_head": str(init_head_path),
            "init_rank_head_sha256": sha256(init_head_path),
        }
        write_json(report_path, report)
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
