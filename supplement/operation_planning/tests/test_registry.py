from polargen.registry import (
    CORE_OPERATION_IDS,
    STRUCTURAL_OPERATIONS,
    operation_spec,
)


def test_selector_uses_eighteen_structural_operations() -> None:
    assert len(CORE_OPERATION_IDS) == 18
    assert len(set(CORE_OPERATION_IDS)) == 18
    assert all(key in STRUCTURAL_OPERATIONS for key in CORE_OPERATION_IDS)


def test_public_operation_metadata() -> None:
    assert operation_spec("OD01").name == "B-X distortion contrast"
    assert operation_spec("OD06").value_semantics == "absolute_target"
    assert operation_spec("OD19").site_role is None
