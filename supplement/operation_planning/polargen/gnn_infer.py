from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gnn_model import OperationGraphConfig, OperationGraphNetwork


CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode_template(
    np_hard: dict[str, Any],
    target_hard: dict[str, Any],
    roles: dict[str, str],
) -> dict[str, float]:
    output: dict[str, float] = {}
    np_global = np_hard["global"]
    target_global = target_hard["global"]
    np_sg = float(np_global["space_group_number"])
    target_sg = float(target_global["space_group_number"])
    output["tpl_np_sg"] = np_sg / 230.0
    output["tpl_target_sg"] = target_sg / 230.0
    output["tpl_sg_delta"] = (target_sg - np_sg) / 230.0
    output["tpl_np_log_atoms"] = math.log1p(
        float(np_global["num_atoms"])
    ) / 5.0
    output["tpl_target_log_atoms"] = math.log1p(
        float(target_global["num_atoms"])
    ) / 5.0
    for prefix, hard in (("np", np_hard), ("target", target_hard)):
        sites = hard["site"]
        output[f"tpl_{prefix}_orbit_count"] = min(len(sites), 20) / 20.0
        free = [float(site["free_dimension"]) for site in sites]
        output[f"tpl_{prefix}_free_mean"] = (
            float(np.mean(free) / 3.0) if free else 0.0
        )
    crystal = str(target_global.get("crystal_system", "")).lower()
    for system in CRYSTAL_SYSTEMS:
        output[f"tpl_target_crystal_{system}"] = float(crystal == system)
    target_free = [
        int(site["free_dimension"]) for site in target_hard["site"]
    ]
    for value in range(4):
        output[f"tpl_target_free_{value}_frac"] = (
            target_free.count(value) / max(len(target_free), 1)
        )
    for role in ("A", "B", "X"):
        element = roles[role]
        for prefix, hard in (("np", np_hard), ("target", target_hard)):
            sites = [
                site for site in hard["site"] if site["element"] == element
            ]
            free = [float(site["free_dimension"]) for site in sites]
            output[f"tpl_{prefix}_{role}_orbits"] = (
                min(len(sites), 10) / 10.0
            )
            output[f"tpl_{prefix}_{role}_free_mean"] = (
                float(np.mean(free) / 3.0) if free else 0.0
            )
        multiplicity = sum(
            int(site["multiplicity"])
            for site in target_hard["site"]
            if site["element"] == element
        )
        output[f"tpl_target_{role}_atom_frac"] = multiplicity / max(
            int(target_global["num_atoms"]), 1
        )
    return output


def load_graph(path: Path) -> dict[str, np.ndarray]:
    required = (
        "z",
        "edge_index",
        "edge_vec",
        "edge_dist",
        "triplet_center",
        "triplet_e1",
        "triplet_e2",
        "triplet_cos",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Graph is missing arrays: {missing}")
        graph = {key: np.asarray(data[key]) for key in required}
    if graph["edge_index"].ndim != 2 or graph["edge_index"].shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if graph["edge_vec"].shape != (graph["edge_index"].shape[1], 3):
        raise ValueError("edge_vec shape does not match edge_index")
    return graph


def _raw_vector(
    values: dict[str, Any],
    columns: list[str],
) -> np.ndarray:
    missing = [name for name in columns if name not in values]
    if missing:
        raise ValueError(
            f"Request is missing {len(missing)} features; first={missing[:5]}"
        )
    return np.asarray(
        [
            np.nan if values[name] is None else float(values[name])
            for name in columns
        ],
        dtype=np.float32,
    )[None, :]


def _normalized(
    raw: np.ndarray,
    median: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, median[None, :], raw)
    scaled = (imputed - mean[None, :]) / std[None, :]
    return np.concatenate(
        [scaled, missing.astype(np.float32)],
        axis=1,
    ).astype(np.float32)


class PolarGenGraphBranch:
    def __init__(
        self,
        checkpoint_path: Path,
        registry_path: Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        self.registry_path = registry_path.resolve()
        self.device = torch.device(
            "cuda"
            if device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = OperationGraphConfig(**self.checkpoint["model_config"])
        self.model = OperationGraphNetwork(config).to(self.device)
        self.model.load_state_dict(
            self.checkpoint["model_state"],
            strict=True,
        )
        self.model.eval()
        registry = json.loads(
            self.registry_path.read_text(encoding="utf-8")
        )
        self.registry = {item["od_id"]: item for item in registry["operations"]}
        self.model_sha256 = sha256(self.checkpoint_path)

    def predict(
        self,
        request: dict[str, Any],
        request_dir: Path,
    ) -> dict[str, Any]:
        graph = load_graph((request_dir / request["graph_file"]).resolve())
        template_map = request.get("template_features")
        if template_map is None:
            template_map = encode_template(
                request["np_hard"],
                request["target_hard"],
                request["formula_roles"],
            )
        template_columns = list(self.checkpoint["template_columns"])
        descriptor_values = request["descriptors"]
        rank_columns = list(self.checkpoint["rank_descriptor_columns"])
        numeric_columns = list(self.checkpoint["numeric_descriptor_columns"])
        template = _raw_vector(template_map, template_columns)
        rank_raw = _raw_vector(descriptor_values, rank_columns)
        numeric_raw = _raw_vector(descriptor_values, numeric_columns)
        rank = _normalized(
            rank_raw,
            np.asarray(self.checkpoint["rank_descriptor_median"]),
            np.asarray(self.checkpoint["rank_descriptor_mean"]),
            np.asarray(self.checkpoint["rank_descriptor_std"]),
        )
        numeric = _normalized(
            numeric_raw,
            np.asarray(self.checkpoint["numeric_descriptor_median"]),
            np.asarray(self.checkpoint["numeric_descriptor_mean"]),
            np.asarray(self.checkpoint["numeric_descriptor_std"]),
        )
        core_ids = list(self.checkpoint["core_ids"])
        operation_indices = torch.arange(
            len(core_ids),
            dtype=torch.long,
            device=self.device,
        )[None, :]
        with torch.inference_mode():
            logits, quantiles, direction_logits = self.model(
                [graph],
                torch.from_numpy(template).to(self.device),
                torch.from_numpy(rank).to(self.device),
                torch.from_numpy(numeric).to(self.device),
                operation_indices,
            )
        logits_np = logits[0].cpu().numpy()
        weights = torch.softmax(logits[0], dim=0).cpu().numpy()
        q = quantiles[0].cpu().numpy()
        direction_probability = torch.sigmoid(
            direction_logits[0]
        ).cpu().numpy()
        target_mean = np.asarray(self.checkpoint["target_mean"])
        target_std = np.asarray(self.checkpoint["target_std"])
        expansion = np.asarray(
            self.checkpoint["conformal_expansion_normalized"]
        )
        median = q[:, 1] * target_std + target_mean
        lower = (q[:, 0] - expansion) * target_std + target_mean
        upper = (q[:, 2] + expansion) * target_std + target_mean
        rows: list[dict[str, Any]] = []
        for index, od_id in enumerate(core_ids):
            metadata = self.registry[od_id]
            if metadata["value_semantics"] == "signed_change":
                direction = (
                    "increase"
                    if direction_probability[index] >= 0.5
                    else "decrease"
                )
            else:
                direction = "target"
            rows.append(
                {
                    "operation_id": od_id,
                    "name": metadata["name"],
                    "direction": direction,
                    "confidence": float(direction_probability[index]),
                    "magnitude": float(median[index]),
                    "target_interval": [
                        float(lower[index]),
                        float(upper[index]),
                    ],
                    "_rank_logit": float(logits_np[index]),
                    "_ranking_weight": float(weights[index]),
                }
            )
        rows.sort(key=lambda item: (-item["_rank_logit"], item["operation_id"]))
        for rank_index, row in enumerate(rows, start=1):
            row["rank"] = rank_index
        top_k = max(1, min(int(request.get("top_k", 3)), len(rows)))
        selected = []
        for row in rows[:top_k]:
            selected.append(
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
            )
        parent = request["np_reference"]
        return {
            "schema_version": "polargen_operation_plan_v1",
            "request_id": request["request_id"],
            "parent": {
                "structure_id": parent["structure_id"],
                "formula": parent["formula"],
            },
            "selected_operations": selected,
        }

    def predict_file(
        self,
        request_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        request_path = request_path.resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = self.predict(request, request_path.parent)
        output_path.write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    predictor = PolarGenGraphBranch(
        args.checkpoint,
        args.registry,
        args.device,
    )
    output = predictor.predict_file(args.request, args.output)
    selected = ", ".join(
        item["operation_id"] for item in output["selected_operations"]
    )
    print(f"PASS selected={selected} output={args.output}")


if __name__ == "__main__":
    main()
