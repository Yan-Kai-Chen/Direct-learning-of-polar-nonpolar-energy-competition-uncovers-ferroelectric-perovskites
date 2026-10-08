from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import NONPOLAR_COLUMN, POLAR_COLUMN
from .graph import GraphBuildConfig, GraphStore
from .model import ModelConfig, PairEnergyModel


def configure_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, RuntimeError):
        pass


def load_bundle(path: str | Path) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise TypeError("Expected a GNN bundle containing state_dict")
    return bundle


class EnergyPredictor:
    def __init__(
        self,
        bundle_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        configure_fp32()
        self.bundle = load_bundle(bundle_path)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.model_config = ModelConfig.from_dict(
            self.bundle.get("model_config", self.bundle.get("arch", {}))
        )
        self.graph_config = GraphBuildConfig.from_dict(
            self.bundle.get("graph_config", {})
        )
        self.target_mean = float(
            self.bundle.get("target_mean", self.bundle.get("y_mu", 0.0))
        )
        self.target_std = float(
            self.bundle.get("target_std", self.bundle.get("y_sd", 1.0))
        )
        self.model = PairEnergyModel(self.model_config).float().to(self.device)
        self.model.load_state_dict(self.bundle["state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        frame: pd.DataFrame,
        store: GraphStore,
        *,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        missing = [
            column
            for column in (POLAR_COLUMN, NONPOLAR_COLUMN)
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"Input table is missing columns: {missing}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        result = frame.copy()
        parts: list[np.ndarray] = []
        for start in range(0, len(result), batch_size):
            batch = result.iloc[start : start + batch_size]
            polar = [store.get(value) for value in batch[POLAR_COLUMN]]
            nonpolar = [store.get(value) for value in batch[NONPOLAR_COLUMN]]
            normalized = self.model(polar, nonpolar)
            prediction = (
                normalized * self.target_std + self.target_mean
            ).detach().float().cpu().numpy()
            parts.append(prediction)
        result["dE_pred_gnn_meV"] = (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.zeros(0, dtype=np.float32)
        )
        return result
