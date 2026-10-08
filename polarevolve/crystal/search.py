"""Bounded, target-free Wyckoff program proposals, not a learned selector."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import zip_longest

from pymatgen.core import Composition, Element

from polarevolve.crystal.contracts import ContractError
from polarevolve.crystal.program import OrbitSpec, compile_hard_condition
from polarevolve.crystal.symmetry import GroupDatabase, WyckoffDatabase

PROGRAM_SELECTION_SCHEMA = "gt_sge_query_program_selection_v1"
PROGRAM_SELECTION_SCHEMA_V2 = "gt_sge_query_program_selection_v2"


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    formula: str
    space_group: int
    maximum_primitive_atoms: int = 20
    maximum_programs: int | None = 16
    maximum_nodes_per_z: int = 20000

    def __post_init__(self):
        if not self.query_id or not self.query_id.isascii() or any(
            not (char.isalnum() or char in "_-") for char in self.query_id
        ):
            raise ContractError("query_id must use ASCII letters, digits, '-' or '_'")
        for name, upper in (
            ("space_group", 230), ("maximum_primitive_atoms", 20),
            ("maximum_nodes_per_z", 1000000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ContractError(f"{name} must be an integer in [1,{upper}]")
        if self.maximum_programs is not None and (
            type(self.maximum_programs) is not int or self.maximum_programs < 1
        ):
            raise ContractError("maximum_programs must be null or a positive integer")
        self.composition()

    def composition(self) -> tuple[tuple[str, int], ...]:
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ContractError("formula must be a non-empty string")
        counts = Composition(self.formula).get_el_amt_dict()
        if not counts or any(
            not Element.is_valid_symbol(e) or n <= 0 or not math.isfinite(n)
            or abs(n - round(n)) > 1e-8 for e, n in counts.items()
        ):
            raise ContractError("formula must contain positive integer element counts")
        divisor = math.gcd(*(round(n) for n in counts.values()))
        return tuple(sorted((e, round(n) // divisor) for e, n in counts.items()))


def propose_programs(
    query: SearchQuery, *, groups: GroupDatabase, wyckoff: WyckoffDatabase, seed: int,
) -> dict:
    """Visit a bounded search tree per primitive Z; interleave Z strata fairly.

    Lexicographic orbit indices remove permutations, not crystallographic
    equivalences. Randomized branch order is reproducible, but a truncated DFS
    is NOT uniform sampling over all legal programs or a normalized prior.
    """
    composition = query.composition()
    hall = next(
        groups.setting(h) for h in range(1, 531)
        if groups.setting(h).space_group_number == query.space_group
    )
    gauges = list(wyckoff.entries_for_hall(hall.hall_number))
    # Program order belongs to the hard search domain, not to a caller's budget.
    search_identity = composition, query.space_group, query.maximum_primitive_atoms
    digest = hashlib.sha256(f"{seed}:{search_identity}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(gauges)
    strata, audits = [], []
    max_z = query.maximum_primitive_atoms // sum(n for _, n in composition)
    for z in range(1, max_z + 1):
        found, visited = [], 0
        program_limit_hit = False
        node_limit_hit = False

        def visit(element, remaining, start, chosen, occupied):
            nonlocal visited, program_limit_hit, node_limit_hit
            if visited >= query.maximum_nodes_per_z:
                node_limit_hit = True
                return
            if query.maximum_programs is not None and len(found) >= query.maximum_programs:
                program_limit_hit = True
                return
            visited += 1
            if remaining == 0:
                if element + 1 == len(composition):
                    found.append(tuple(chosen))
                else:
                    visit(element + 1, composition[element + 1][1] * z * hall.centering_index,
                          0, chosen, occupied)
                return
            symbol = composition[element][0]
            for index in range(start, len(gauges)):
                gauge = gauges[index]
                if gauge.multiplicity > remaining or gauge.letter in occupied:
                    continue
                fixed = gauge.free_dimension == 0
                visit(element, remaining - gauge.multiplicity, index + int(fixed),
                      chosen + [(symbol, gauge.letter)],
                      occupied | {gauge.letter} if fixed else occupied)
                if node_limit_hit or program_limit_hit:
                    return

        visit(0, composition[0][1] * z * hall.centering_index, 0, [], set())
        strata.append(found)
        reasons = []
        if node_limit_hit:
            reasons.append("node_budget")
        if program_limit_hit:
            reasons.append("program_budget")
        audits.append({"z_primitive": z, "visited_nodes": visited,
                       "programs_found": len(found),
                       "search_exhausted": not reasons,
                       "truncation_reasons": reasons})

    proposals = []
    for group in zip_longest(*strata):
        for program in group:
            if program is None or (
                query.maximum_programs is not None
                and len(proposals) >= query.maximum_programs
            ):
                continue
            occurrences = Counter()
            specs = []
            # The ordering is an explicit inference convention, not cache-derived.
            for element, letter in sorted(program, key=lambda item: (Element(item[0]).Z, item[1])):
                occurrences[element, letter] += 1
                specs.append(OrbitSpec(element, letter, occurrences[element, letter]))
            identity = json.dumps([asdict(spec) for spec in specs], sort_keys=True)
            program_id = hashlib.sha256(f"{hall.hall_number}:{identity}".encode()).hexdigest()[:16]
            compiled = compile_hard_condition(
                condition_id=f"{query.query_id}_{program_id}", hall_number=hall.hall_number,
                orbit_specs=tuple(specs), base_cell_representation="primitive",
                group_database=groups, wyckoff_database=wyckoff,
            )
            hard = compiled.hard
            if hard.reduced_composition != composition:
                raise ContractError("compiled program changed the requested composition")
            proposals.append({"program_id": program_id, "hard_condition": asdict(hard)})
    enumeration_complete = all(audit["search_exhausted"] for audit in audits)
    enumerated_count = sum(len(stratum) for stratum in strata)
    candidate_cap_applied = len(proposals) < enumerated_count or any(
        "program_budget" in audit["truncation_reasons"] for audit in audits
    )
    return {
        "schema_version": "gt_sge_program_search_v2", "query": asdict(query), "seed": seed,
        "hall_number": hall.hall_number, "centering_index": hall.centering_index,
        "reduced_composition": dict(composition), "programs": proposals, "strata": audits,
        "enumerated_programs": enumerated_count,
        "raw_program_count": enumerated_count if enumeration_complete else None,
        "enumeration_complete": enumeration_complete,
        "status": "proposed" if proposals else (
            "infeasible_within_bounds" if all(a["search_exhausted"] for a in audits)
            else "search_budget_exhausted_without_solution"
        ),
        "proposal_policy": "hard_input_seeded_bounded_dfs_z_round_robin_v3",
        "orbit_order": "atomic_number_then_letter_then_occurrence_v1",
        "program_equivalence": "orbit_order_only; origin/basis equivalents may remain",
        "candidate_cap_applied": candidate_cap_applied,
        "gauge_sha256": wyckoff.provenance.artifact_sha256("wyckoff_gauges.json"),
        "group_sha256": groups.provenance.artifact_sha256("spacegroup_settings.json"),
    }


def planned_candidate_count(plan, default_candidates: int) -> int:
    """Return the exact query sample count under optional per-program budgets."""

    if default_candidates <= 0:
        raise ValueError("default candidate count must be positive")
    return int(plan.get("lattices_per_program", 1)) * sum(
        int(program.get("candidate_budget", default_candidates))
        for report in plan["queries"] for program in report["programs"]
    )
