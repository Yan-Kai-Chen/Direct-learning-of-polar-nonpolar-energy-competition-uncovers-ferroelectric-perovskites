"""Query planning and hard-input iteration; never reads crystal targets."""

import json
import hashlib
from dataclasses import replace
from pathlib import Path

from polarevolve.crystal.program import OrbitSpec, compile_hard_condition
from polarevolve.crystal.search import (
    PROGRAM_SELECTION_SCHEMA,
    PROGRAM_SELECTION_SCHEMA_V2,
    SearchQuery,
    planned_candidate_count as planned_candidate_count,
    propose_programs,
)
from polarevolve.data.packing import ConditionRecord
from polarevolve.guidance.chemistry import validate_preferences

QUERY_GENERATION_MODE = "chemistry_spacegroup_program_search_v5"
QUERY_PLAN_SCHEMA = "gt_sge_query_plan_v5"
QUERY_SAMPLE_SCHEMA = "gt_sge_query_crystal_sample_v5"
QUERY_MANIFEST_SCHEMA = "gt_sge_query_sampling_manifest_v5"


def canonical_condition_request(values, elements=None):
    conditions = validate_preferences(values, elements)
    mode = "hard_only" if not conditions else "independent" if len(conditions) == 1 else "joint"
    return {"schema_version": "gt_sge_soft_condition_request_v1", "mode": mode,
            "aggregation": "normalized_weighted_mean_v1", "conditions": conditions}


def _search_identity(query):
    return {"reduced_composition": dict(query.composition()),
            "space_group": query.space_group,
            "maximum_primitive_atoms": query.maximum_primitive_atoms}


def _load_selection(path):
    raw = path.read_bytes()
    selection = json.loads(raw)
    if not isinstance(selection, dict):
        raise ValueError("program selection must be a supported nonempty selection")
    ids = selection.get("selected_program_ids")
    if (selection.get("schema_version") not in {
                PROGRAM_SELECTION_SCHEMA, PROGRAM_SELECTION_SCHEMA_V2
            }
            or not isinstance(ids, list) or not ids
            or len(ids) != len(set(ids)) or not all(isinstance(value, str) for value in ids)):
        raise ValueError("program selection must be a supported nonempty selection")
    budgets = selection.get("candidate_budgets")
    if selection.get("schema_version") == PROGRAM_SELECTION_SCHEMA_V2 and (
        not isinstance(budgets, dict) or set(budgets) != set(ids)
        or any(type(value) is not int or value <= 0 for value in budgets.values())
    ):
        raise ValueError("v2 program selection requires one positive candidate budget per program")
    return selection, hashlib.sha256(raw).hexdigest()


def plan_queries(path: Path, *, groups, wyckoff, seed: int, records: int, lattices: int,
                 program_selection: Path | None = None):
    source = path.read_bytes()
    queries, requests = [], []
    for line in source.decode("utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        legacy = raw.pop("local_preferences", None)
        conditions = raw.pop("soft_conditions", None)
        if legacy is not None and conditions is not None:
            raise ValueError("use soft_conditions or legacy local_preferences, not both")
        query = SearchQuery(**raw)
        queries.append(query)
        values = conditions if conditions is not None else legacy or []
        requests.append(canonical_condition_request(values, dict(query.composition())))
    if len(queries) != records or len({q.query_id.casefold() for q in queries}) != len(queries):
        raise ValueError("query file must contain exactly --records distinct query IDs")
    reports = [propose_programs(q, groups=groups, wyckoff=wyckoff, seed=seed) for q in queries]
    selection_metadata = None
    if program_selection is not None:
        selection, selection_sha256 = _load_selection(program_selection)
        selected = set(selection["selected_program_ids"])
        for query, report in zip(queries, reports):
            if selection.get("hard_search_identity") != _search_identity(query):
                raise ValueError("program selection does not match the query hard-search identity")
            available = {program["program_id"] for program in report["programs"]}
            if selected - available:
                raise ValueError("program selection references programs absent from exhaustive search")
            report["programs_before_selection"] = len(report["programs"])
            report["programs"] = [program for program in report["programs"]
                                  if program["program_id"] in selected]
            for program in report["programs"]:
                if selection.get("candidate_budgets") is not None:
                    program["candidate_budget"] = selection["candidate_budgets"][
                        program["program_id"]
                    ]
            report["program_selection_sha256"] = selection_sha256
        selection_metadata = {"path": str(program_selection.resolve()),
                              "sha256": selection_sha256,
                              "selected_programs": len(selected),
                              "source_samples_sha256": selection.get("source_samples_sha256")}
    for report, request in zip(reports, requests):
        report["soft_condition_request"] = request
    return {"schema_version": QUERY_PLAN_SCHEMA, "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_path": str(path.resolve()), "lattices_per_program": lattices,
            "condition_records": sum(len(r["programs"]) for r in reports) * lattices,
            "program_selection": selection_metadata,
            "queries": reports}


def iter_query_records(plan, *, groups, wyckoff, rank=0, world_size=1):
    index = 0
    for report in plan["queries"]:
        for program in report["programs"]:
            hard = program["hard_condition"]
            compiled = compile_hard_condition(
                condition_id=hard["condition_id"], hall_number=hard["hall_number"],
                orbit_specs=tuple(OrbitSpec(o["element"], o["letter"], o["occurrence"])
                                  for o in hard["wyckoff_orbits"]),
                base_cell_representation="primitive", group_database=groups, wyckoff_database=wyckoff,
            )
            for draw in range(plan["lattices_per_program"]):
                assigned = index % world_size == rank
                index += 1
                if not assigned:
                    continue
                condition = replace(compiled.hard, condition_id=f"{hard['condition_id']}_l{draw:03d}")
                yield ConditionRecord.build(condition, compiled.hall.space_group_number, wyckoff), {
                    "query_id": report["query"]["query_id"], "program_id": program["program_id"],
                    "hard_sample_identity": f"{program['program_id']}:l{draw:03d}",
                    "soft_condition_request": report["soft_condition_request"],
                    "candidate_budget": program.get("candidate_budget"),
                    "lattice_draw": draw, "z_primitive": condition.formula_unit_scale_base,
                    "primitive_num_atoms": condition.base_num_atoms,
                    "group_num_atoms": condition.group_num_atoms,
                    "cif_cell_representation": "canonical_group_cell",
                    "hard_condition": condition.to_dict(),
                }
