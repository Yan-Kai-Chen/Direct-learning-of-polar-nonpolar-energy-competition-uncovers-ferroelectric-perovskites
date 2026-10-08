from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml

from .data import (
    NONPOLAR_COLUMN,
    POLAR_COLUMN,
    TARGET_COLUMN,
    audit_training_table,
    split_indices,
)
from .graph import GraphBuildConfig, GraphStore
from .model import ModelConfig, PairEnergyModel


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    epochs: int = 120
    patience: int = 15
    batch_size_train: int = 8
    batch_size_inference: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 2.0
    use_amp: bool = True
    huber_beta_normalized: float = 0.25
    tail_lambda: float = 0.8
    tail_scale_mev: float = 60.0
    tail_clip: float = 2.5
    max_sample_weight: float = 3.5
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphBuildConfig = field(default_factory=GraphBuildConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        data = dict(values)
        data["model"] = ModelConfig.from_dict(data.get("model", {}))
        data["graph"] = GraphBuildConfig.from_dict(data.get("graph", {}))
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise TypeError("Training configuration must be a mapping")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _load_graph_pairs(
    frame: pd.DataFrame,
    indices: np.ndarray,
    store: GraphStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = frame.iloc[indices]
    polar = [store.get(value) for value in rows[POLAR_COLUMN]]
    nonpolar = [store.get(value) for value in rows[NONPOLAR_COLUMN]]
    return polar, nonpolar


def tail_weights(target_mev: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    weights = 1.0 + config.tail_lambda * torch.clamp(
        target_mev.abs() / config.tail_scale_mev,
        0.0,
        config.tail_clip,
    )
    return torch.clamp(weights, max=config.max_sample_weight)


@torch.no_grad()
def predict_indices(
    model: PairEnergyModel,
    frame: pd.DataFrame,
    indices: np.ndarray,
    store: GraphStore,
    *,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for batch_index in _batches(
        indices,
        batch_size,
        shuffle=False,
        rng=rng,
    ):
        polar, nonpolar = _load_graph_pairs(frame, batch_index, store)
        normalized = model(polar, nonpolar)
        prediction = normalized * target_std + target_mean
        values.append(prediction.detach().float().cpu().numpy())
    return (
        np.concatenate(values).astype(np.float32)
        if values
        else np.zeros(0, dtype=np.float32)
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    return {
        "mae_mev": float(np.mean(np.abs(error))),
        "rmse_mev": float(np.sqrt(np.mean(error**2))),
        "bias_mev": float(np.mean(error)),
    }


def run_training(
    *,
    data_path: str | Path,
    structure_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Train on the fixed table split and save one reproducible GNN bundle."""

    if config.epochs <= 0 or config.batch_size_train <= 0:
        raise ValueError("epochs and batch_size_train must be positive")
    set_seed(config.seed)
    frame, audit = audit_training_table(data_path)
    if not audit.passed:
        raise ValueError("Training-table audit failed")
    indices = split_indices(frame)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=np.float32
    )
    train_target = target[indices["train"]]
    target_mean = float(train_target.mean())
    target_std = float(train_target.std(ddof=0) + 1e-6)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    store = GraphStore(
        structure_dir,
        output / "graph_cache",
        config.graph,
    )
    device = resolve_device(config.device)
    model = PairEnergyModel(config.model).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index in _batches(
            indices["train"],
            config.batch_size_train,
            shuffle=True,
            rng=rng,
        ):
            polar, nonpolar = _load_graph_pairs(frame, batch_index, store)
            batch_target = torch.as_tensor(
                target[batch_index],
                dtype=torch.float32,
                device=device,
            )
            normalized_target = (batch_target - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(polar, nonpolar)
                per_sample = functional.smooth_l1_loss(
                    prediction,
                    normalized_target,
                    beta=config.huber_beta_normalized,
                    reduction="none",
                )
                loss = (tail_weights(batch_target, config) * per_sample).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.clip_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_prediction = predict_indices(
            model,
            frame,
            indices["val"],
            store,
            batch_size=config.batch_size_inference,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_metrics = regression_metrics(target[indices["val"]], val_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **val_metrics,
            }
        )
        if val_metrics["mae_mev"] < best_val_mae:
            best_val_mae = val_metrics["mae_mev"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no model state")
    model.load_state_dict(best_state, strict=True)
    split_metrics: dict[str, dict[str, float]] = {}
    for name in ("val", "test"):
        prediction = predict_indices(
            model,
            frame,
            indices[name],
            store,
            batch_size=config.batch_size_inference,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        split_metrics[name] = regression_metrics(target[indices[name]], prediction)

    bundle = {
        "format_version": 1,
        "state_dict": best_state,
        "model_config": config.model.to_dict(),
        "graph_config": config.graph.to_dict(),
        "target_column": TARGET_COLUMN,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "seed": config.seed,
    }
    torch.save(bundle, output / "gnn_bundle.pt")
    summary = {
        "data_audit": audit.to_dict(),
        "training_config": config.to_dict(),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_mae_mev": best_val_mae,
        "metrics": split_metrics,
        "history": history,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def one_step_smoke(
    polar_graphs: list[dict[str, Any]],
    nonpolar_graphs: list[dict[str, Any]],
    target_mev: np.ndarray,
) -> dict[str, Any]:
    """Run one CPU optimizer step on caller-provided graphs without saving."""

    config = TrainingConfig(
        model=ModelConfig(dim=16, n_layers=1, rbf_dim=8, angle_hidden=16),
        use_amp=False,
    )
    set_seed(config.seed)
    model = PairEnergyModel(config.model).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    target = torch.as_tensor(target_mev, dtype=torch.float32)
    mean = float(target.mean())
    std = float(target.std(unbiased=False) + 1e-6)
    before = copy.deepcopy(model.state_dict())
    optimizer.zero_grad(set_to_none=True)
    prediction = model(polar_graphs, nonpolar_graphs)
    loss = functional.smooth_l1_loss(
        prediction,
        (target - mean) / std,
        beta=config.huber_beta_normalized,
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.clip_grad_norm,
    )
    optimizer.step()
    delta = sum(
        float((value.detach() - before[key]).abs().sum())
        for key, value in model.state_dict().items()
        if value.is_floating_point()
    )
    return {
        "rows": len(polar_graphs),
        "loss": float(loss.detach()),
        "gradient_norm": float(grad_norm),
        "parameter_l1_delta": delta,
        "checkpoint_saved": False,
        "passed": bool(
            torch.isfinite(loss)
            and np.isfinite(float(grad_norm))
            and delta > 0.0
        ),
    }
