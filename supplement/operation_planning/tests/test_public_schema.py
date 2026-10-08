import json
from pathlib import Path

import jsonschema

from polargen.public_api import load_operation_plan


ROOT = Path(__file__).resolve().parents[1]


def test_operation_plan_example_matches_schema() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "operation_plan.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (ROOT / "examples" / "operation_plan.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(example, schema)
    plan = load_operation_plan(ROOT / "examples" / "operation_plan.json")
    assert len(plan.operations) == 3
