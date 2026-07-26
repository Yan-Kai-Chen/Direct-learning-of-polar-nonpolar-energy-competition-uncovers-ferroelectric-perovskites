"""Public AE3GNN graph, model, training, and inference interfaces."""

from .data import audit_training_table, split_indices

__all__ = [
    "GraphBuildConfig",
    "GraphStore",
    "ModelConfig",
    "PairEnergyModel",
    "audit_training_table",
    "build_periodic_angle_graph",
    "split_indices",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name in {"GraphBuildConfig", "GraphStore", "build_periodic_angle_graph"}:
        from .graph import (
            GraphBuildConfig,
            GraphStore,
            build_periodic_angle_graph,
        )

        return {
            "GraphBuildConfig": GraphBuildConfig,
            "GraphStore": GraphStore,
            "build_periodic_angle_graph": build_periodic_angle_graph,
        }[name]
    if name in {"ModelConfig", "PairEnergyModel"}:
        from .model import ModelConfig, PairEnergyModel

        return {
            "ModelConfig": ModelConfig,
            "PairEnergyModel": PairEnergyModel,
        }[name]
    raise AttributeError(name)
