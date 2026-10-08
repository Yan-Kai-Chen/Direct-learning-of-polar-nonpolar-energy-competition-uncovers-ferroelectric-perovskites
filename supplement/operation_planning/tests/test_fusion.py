import numpy as np

from polargen.fusion import (
    fuse_operation_scores,
    preserve_graph_numerics,
    select_top_operations,
)


def test_language_branch_can_change_order_without_changing_numerics() -> None:
    branches = {
        "primary_graph": np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32),
        "language_prior": np.asarray([[0.0, 4.0, 1.0]], dtype=np.float32),
    }
    fused = fuse_operation_scores(
        branches,
        {"primary_graph": 1.0, "language_prior": 2.0},
    )
    assert select_top_operations(fused, ["a", "b", "c"], 1) == [["b"]]
    graph_outputs = [
        {"operation_id": "a", "predicted_median": 0.1},
        {"operation_id": "b", "predicted_median": 0.2},
        {"operation_id": "c", "predicted_median": 0.3},
    ]
    selected = preserve_graph_numerics(fused, graph_outputs, top_k=1)
    assert selected[0]["operation_id"] == "b"
    assert selected[0]["predicted_median"] == 0.2
