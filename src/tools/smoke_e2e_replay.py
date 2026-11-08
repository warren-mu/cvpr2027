"""E2E step 2, CPU only: reauthenticate step 1 and replay both-world witnesses.

No THOR Controller/model is started. A successful, audited T02 contract-test
run is mandatory. This does not certify pre-evidence credit or generation dedup.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import traceback

from src.envs.thor_common import ROOT
from src.envs.pair_table_checks import replay_witness
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json
from src.tools.smoke_e2e_scope import VERSION, SmokeTableBackend, SmokeTableEnvironment, coverage
from src.tools.smoke_e2e_views import load_success


def python_args(command):
    if not isinstance(command, list) or not command or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name):
        raise ValueError("Expected actual Python command")
    result = command[1:]
    return result[1:] if result[:1] == ["-u"] else result


def contract_tests(path):
    data, reference = audited_result(path)
    meta, config, metrics = data["_meta"], data["config"], data["metrics"]
    command = config.get("command")
    expected = ["-m", "unittest", "discover", "-s", "tests", "-p", "test_pair_table*.py"]
    wrapper = python_args(meta.get("command"))
    if (python_args(command) != expected or wrapper != ["-m", "src.tools.record_command", "--name", meta["name"], "--", *command]
            or meta.get("exit_code") != 0 or metrics.get("command_exit_code") != 0 or data.get("error") is not None):
        raise ValueError("Require successful audited canonical T02 §6 unittest command")
    log = ROOT / "logs/runs" / (meta["name"] + ".log")
    text = log.read_text(errors="strict")
    if not re.search(r"(?m)^Ran [1-9][0-9]* tests? in .+$", text) or not re.search(r"(?m)^OK\s*$", text):
        raise ValueError("Contract-test log does not show a nonempty successful suite")
    return reference


def replay(tables, packages, checks, report, identities, certificate):
    backend = SmokeTableBackend(tables, certificate=certificate, pair_checks=checks,
                                report=report, authenticated_ids=identities)
    rows = []
    try:
        for pair_id, table in tables.items():
            for world in (0, 1):
                witness = replay_witness(table, world, packages[pair_id]["t01"]["witnesses"][world])
                if witness.get("passed") is not True or witness.get("applicable") is not True:
                    raise ValueError("An original <=5-turn witness did not replay")
                env = SmokeTableEnvironment(backend)
                observation = env.reset(pair_id, f"w{world}")
                trace = [{"decision_index": 0, "state_id": env.info["state_id"],
                          "tau": env.info["tau"], "x_seen_step": env.info["x_seen_step"],
                          "marked_rgb_sha256": observation.image.sha256,
                          "public_text": observation.text}]
                for item in witness["trace"]:
                    action = item["action"]
                    observation, reward, done, info = env.step(action)
                    trace.append({"action": action, "reward": reward, "done": done,
                        "state_id": info["state_id"], "tau": info["tau"], "x_seen_step": info["x_seen_step"],
                        "success": info["success"], "marked_rgb_sha256": observation.image.sha256,
                        "public_text": observation.text, "table_temporal_proof": info["table_temporal_proof"]})
                if not env.done or env.info["success"] is not True or env.steps > 5:
                    raise ValueError("Environment replay did not terminate successfully within D+1")
                rows.append({"pair_id": pair_id, "world_id": f"w{world}", "kind": table.kind,
                             "steps": env.steps, "tau": env.info["tau"], "x_seen_step": env.info["x_seen_step"],
                             "selected_target_name": env.info["selected_name"], "success": True, "trace": trace})
    finally:
        backend.close()
    return {"witness_replay_passed": True, "world_replays": rows, "coverage": coverage(list(tables.values())),
            "max_steps": 5, "max_model_calls": 5, "model_calls_made": 0,
            "formal_T02_acceptance": None, "paper_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract-tests", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"smoke_e2e_[A-Za-z0-9_]+", args.name):
        parser.error("Dedicated smoke_e2e_ name required")
    output = ROOT / "results" / f"{args.name}.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh CPU replay name")
    config = {"name": args.name, "source": str(args.source), "contract_tests": str(args.contract_tests),
              "version": VERSION, "scope": "E2E step2 CPU witness replay only", "starts_Controllers_or_models": False}
    metrics = {"kind": "smoke_e2e_replay", "step2_passed": False, "errors": [],
               "formal_T02_acceptance": None, "paper_result": False, "steps_3_to_5_started": False}
    sources, code = [], 1
    try:
        tests = contract_tests(args.contract_tests)
        sources.append(tests)
        tables, packages, checks, report, identities, certificate, evidence = load_success(args.source)
        sources.extend(ref for ref in evidence if ref not in sources)
        metrics.update(replay(tables, packages, checks, report, identities, certificate))
        metrics.update(step2_passed=True, T02_section6_tests=tests)
        code = 0
    except BaseException as error:
        traceback.print_exc()
        metrics["errors"].append(f"{type(error).__name__}: {error}")
    metrics["producer_exit_code"] = code
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {"metrics": metrics, "config": config, "sources": sources})
    print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
