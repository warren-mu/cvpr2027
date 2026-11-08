"""The human-authorized E2E twenty-state exception, never formal T02 approval.

All per-state numerical checks are the unchanged formal checks. Only the sample
size/scope is different. No function in this module sets table.cache_verified.
"""
import copy
import hashlib
import json
import random

from src.envs.pair_table import TableEnvironment, TableObservationBackend
from src.envs.pair_table_cache import CACHE_POLICY, check_cached_state
from src.envs.pair_table_checks import verify_cached_binding

VERSION = "smoke_e2e_twenty_paired_states_v1"
POLICY = {"sample_size": 20, "seed": 0, "depth": 4, "episode_turns": 5,
          "scope": "approved_five_pair_E2E_smoke_only_not_paper_or_formal_T02",
          "formal_T02_minimum_distinct_pair_states": 200,
          "per_state_policy": copy.deepcopy(CACHE_POLICY),
          "failure_scope": "all_selected_pairs_and_all_twenty_states_no_filtering"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sample_plan(population):
    keys = sorted(tuple(key) for key in population)
    if (any(len(key) != 2 or any(not isinstance(v, str) or not v for v in key) for key in keys)
            or len(keys) != len(set(keys)) or len(keys) < 20):
        raise ValueError("Smoke sample requires a complete population with >=20 distinct pair/state keys")
    return {"version": VERSION, "algorithm": "python_random_sample_without_replacement",
            "seed": 0, "sample_size": 20, "population_size": len(keys),
            "population_sha256": digest(keys),
            "selected_keys": [list(key) for key in random.Random(0).sample(keys, 20)]}


def validate_sample(samples, *, population, plan):
    expected = sample_plan(population)
    if plan != expected:
        raise ValueError("Smoke sample does not reproduce its complete population and seed")
    records = list(samples)
    keys = [(item.pair_id, item.state_id) for item in records]
    selected = {tuple(key) for key in plan["selected_keys"]}
    if len(keys) != len(set(keys)) or set(keys) - selected:
        raise ValueError("Duplicate or unrequested smoke sample")
    cached_processes = {view.process_id for item in records for view in item.cached}
    identities = [(view.process_id, view.render_id) for item in records for view in item.fresh]
    if len(identities) != len(set(identities)):
        raise ValueError("Fresh render reused between sampled states")
    rows = []
    for item in records:
        if any(view.process_id in cached_processes for view in item.fresh):
            raise ValueError("Fresh process overlaps an export process")
        rows.append(check_cached_state(item))
    missing = sorted(selected - set(keys))
    complete = len(keys) == 20 and not missing
    checks = complete and all(row["checks_passed"] is True for row in rows)
    real = bool(rows) and all(row["synthetic"] is False for row in rows)
    return {"version": VERSION, "policy": copy.deepcopy(POLICY), "sampling_plan": expected,
            "states": rows, "observed_distinct_states": len(keys), "sample_complete": complete,
            "missing_state_keys": [list(key) for key in missing], "checks_passed": checks,
            "synthetic_fixture": not real, "smoke_cache_matches_live": checks and real,
            "formal_cache_matches_live": None, "full_T02_acceptance": None,
            "paper_result": False}


def authorize_smoke(tables, *, pair_checks, report, authenticated_ids):
    """Return an explicit smoke certificate without changing formal flags."""
    indexed = {table.pair_id: table for table in tables}
    if len(indexed) != len(tables) or not indexed:
        raise ValueError("Explicit unique smoke table inventory required")
    expected = sample_plan([(key, state) for key, table in indexed.items() for state in table.states])
    if (report.get("version") != VERSION or report.get("policy") != POLICY
            or report.get("sampling_plan") != expected or report.get("observed_distinct_states") != 20
            or report.get("sample_complete") is not True or report.get("checks_passed") is not True
            or report.get("smoke_cache_matches_live") is not True or report.get("synthetic_fixture") is not False
            or report.get("formal_cache_matches_live") is not None or report.get("full_T02_acceptance") is not None):
        raise ValueError("Twenty-state smoke validation incomplete; stop downstream smoke")
    rows = report.get("states", [])
    if (len(rows) != 20 or {(row["pair_id"], row["state_id"]) for row in rows}
            != {tuple(key) for key in expected["selected_keys"]}):
        raise ValueError("Smoke report state inventory mismatch")
    required = set()
    for row in rows:
        if row.get("cache_matches_live") is not True or row.get("checks_passed") is not True:
            raise ValueError("A failed state blocks the whole smoke")
        verify_cached_binding(indexed[row["pair_id"]], row)
        for world in ("w0", "w1"):
            required.add((row["pair_id"], row["state_id"], world,
                          row["worlds"][world]["fresh_source"]["render_id"]))
    if set(authenticated_ids) != required:
        raise ValueError("Smoke fresh native proofs are missing or cover another sample")
    for key, table in indexed.items():
        check = pair_checks.get(key, {})
        if (table.synthetic or table.export_verified is not True or table.cache_verified is not False
                or not isinstance(table.manifest_reference, dict)
                or check.get("passed") is not True or check.get("synthetic_only") is not False
                or check.get("prefix_verified_identical") is not True
                or any(row.get("passed") is not True for row in check.get("witness_replay", []))
                or len(check.get("witness_replay", [])) != 2):
            raise ValueError("Every selected pair needs native export and T02 checks 1–4")
    return {"version": VERSION, "policy": copy.deepcopy(POLICY), "cache_report_sha256": digest(report),
            "tables": {key: copy.deepcopy(value.manifest_reference) for key, value in indexed.items()},
            "smoke_only": True, "formal_cache_matches_live": None, "full_T02_acceptance": None,
            "paper_result": False}


class SmokeTableBackend(TableObservationBackend):
    """Dedicated twenty-state authorization; never use allow_synthetic for it."""
    def __init__(self, tables, *, certificate, pair_checks, report, authenticated_ids):
        actual = authorize_smoke(list(tables.values()), pair_checks=pair_checks, report=report,
                                 authenticated_ids=authenticated_ids)
        if certificate != actual:
            raise ValueError("Smoke certificate belongs to different tables or evidence")
        self.tables, self.certificate = dict(tables), copy.deepcopy(certificate)
        self.world_orders, self.serial, self.public_cache = {}, 0, {}

    def verify_capture(self, frame):
        state = self.table.states[self.state_id]
        capture = state.captures[0 if state.shared else self.world]
        marked, _ = self.public_cache[(self.table.pair_id, self.state_id, 0 if state.shared else self.world)]
        return (self.table.export_verified is True and self.table.cache_verified is False
                and self.certificate["tables"].get(self.table.pair_id) == self.table.manifest_reference
                and frame.raw_rgb == capture.rgb and frame.marked_rgb == marked
                and frame.capture.state_id == self.state_id and frame.capture.source_id == capture.render_id)


class SmokeTableEnvironment(TableEnvironment):
    @property
    def info(self):
        return {**super().info, "cache_matches_live": None, "smoke_cache_matches_live": True,
                "authorization_scope": VERSION, "formal_T02_acceptance": None, "paper_result": False}


def coverage(tables):
    tau0 = all(not table.states[table.initial_state].shared for table in tables)
    return {"all_selected_pairs_tau0": tau0,
            "pre_evidence_credit_coverage": "unmeasured" if tau0 else "not_tested_in_steps_1_2",
            "paired_generation_dedup_coverage": "unmeasured",
            "T3_correctness_claim": None,
            "reason": "Direct/tau0-only smoke cannot exercise pre-evidence generation or credit" if tau0
                      else "No model generation or paired-credit test is performed by steps 1–2"}
