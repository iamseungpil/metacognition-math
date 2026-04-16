#!/usr/bin/env python3
"""Verify that the current mainline code/data/docs still match the active contract."""

from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STRICT_PAIR_SCRIPT = ROOT / "scripts/build_v8_strict_paired_data.py"
STRICT_PAIR_SPEC = importlib.util.spec_from_file_location("build_v8_strict_paired_data", STRICT_PAIR_SCRIPT)
if STRICT_PAIR_SPEC is None or STRICT_PAIR_SPEC.loader is None:
    raise ImportError(f"Unable to load strict-pair helpers from {STRICT_PAIR_SCRIPT}")
STRICT_PAIR_MODULE = importlib.util.module_from_spec(STRICT_PAIR_SPEC)
STRICT_PAIR_SPEC.loader.exec_module(STRICT_PAIR_MODULE)

extract_last_boxed = STRICT_PAIR_MODULE.extract_last_boxed
load_messages = STRICT_PAIR_MODULE.load_messages
parse_assistant = STRICT_PAIR_MODULE.parse_assistant


CONTRACT_PATH = ROOT / "configs/mainline_contract.yaml"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def emit(ok: bool, message: str, errors: list[str]) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {message}")
    if not ok:
        errors.append(message)


def verify_file_exists(rel_path: str, errors: list[str]) -> Path:
    path = ROOT / rel_path
    emit(path.exists(), f"{rel_path} exists", errors)
    return path


def check_sft_contract(contract: dict, errors: list[str]) -> None:
    print("## SFT")
    raw_base = contract["sft"]["raw_base_model"]
    shared = contract["sft"]["shared"]

    for lane in ["meta", "base"]:
        lane_contract = contract["sft"][lane]
        cfg_path = verify_file_exists(lane_contract["config"], errors)
        verify_file_exists(lane_contract["dataset"], errors)
        if not cfg_path.exists():
            continue
        cfg = load_yaml(cfg_path)

        emit(cfg.get("model_name_or_path") == raw_base, f"{cfg_path.name} starts from raw base", errors)
        emit(cfg.get("dataset_path") == lane_contract["dataset"], f"{cfg_path.name} dataset path matches contract", errors)
        emit(cfg.get("output_dir") == lane_contract["output_dir"], f"{cfg_path.name} output dir matches contract", errors)
        emit(cfg.get("run_name") == lane_contract["run_name"], f"{cfg_path.name} run name matches contract", errors)

        for key, expected in shared.items():
            emit(cfg.get(key) == expected, f"{cfg_path.name} {key}={expected}", errors)


def check_strict_data(contract: dict, errors: list[str]) -> None:
    print("\n## Strict Paired Data")
    pair = contract["data"]["strict_pair"]
    meta_path = verify_file_exists(pair["meta"], errors)
    base_path = verify_file_exists(pair["base"], errors)
    summary_path = verify_file_exists(pair["validation_summary"], errors)

    if not meta_path.exists() or not base_path.exists():
        return

    meta_df = pd.read_parquet(meta_path)
    base_df = pd.read_parquet(base_path)
    emit(len(meta_df) == pair["rows"], f"strict meta rows = {pair['rows']}", errors)
    emit(len(base_df) == pair["rows"], f"strict base rows = {pair['rows']}", errors)
    emit(len(meta_df) == len(base_df), "strict row parity", errors)

    prompt_match = 0
    boxed_match = 0
    scenario_counts: dict[str, int] = {}
    for idx in range(len(meta_df)):
        meta_row = meta_df.iloc[idx]
        base_row = base_df.iloc[idx]

        meta_msgs = load_messages(meta_row["messages"])
        base_msgs = load_messages(base_row["messages"])
        if meta_msgs[0]["content"] == base_msgs[0]["content"]:
            prompt_match += 1

        meta_assistant = str(meta_msgs[1]["content"])
        base_assistant = str(base_msgs[1]["content"])
        meta_boxed = extract_last_boxed(parse_assistant(meta_assistant)[1])
        base_boxed = extract_last_boxed(base_assistant)
        if meta_boxed == base_boxed:
            boxed_match += 1

        scenario = str(meta_row.get("scenario", ""))
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

    emit(prompt_match == len(meta_df), f"strict prompt parity {prompt_match}/{len(meta_df)}", errors)
    emit(boxed_match == len(meta_df), f"strict boxed parity {boxed_match}/{len(meta_df)}", errors)
    print(f"[INFO] strict scenario counts: {scenario_counts}")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        emit(bool(summary.get("passed")), "strict validation summary passed=true", errors)


def _check_rl_pair(meta_df: pd.DataFrame, base_df: pd.DataFrame, expected_rows: int, split_name: str, errors: list[str]) -> None:
    emit(len(meta_df) == expected_rows, f"{split_name} meta rows = {expected_rows}", errors)
    emit(len(base_df) == expected_rows, f"{split_name} base rows = {expected_rows}", errors)
    emit(len(meta_df) == len(base_df), f"{split_name} RL row parity", errors)

    if len(meta_df) != len(base_df):
        return

    emit((meta_df["prompt"].astype(str) == base_df["prompt"].astype(str)).all(), f"{split_name} prompt parity", errors)
    emit((meta_df["reward_model"].astype(str) == base_df["reward_model"].astype(str)).all(), f"{split_name} reward parity", errors)
    if "split_tags" in meta_df.columns and "split_tags" in base_df.columns:
        emit((meta_df["split_tags"].astype(str) == base_df["split_tags"].astype(str)).all(), f"{split_name} split_tags parity", errors)


def check_rl_data(contract: dict, errors: list[str]) -> None:
    print("\n## RL Paired Data")
    rl_pair = contract["data"]["rl_pair"]
    meta_train = pd.read_parquet(verify_file_exists(rl_pair["meta_train"], errors))
    meta_val = pd.read_parquet(verify_file_exists(rl_pair["meta_val"], errors))
    base_train = pd.read_parquet(verify_file_exists(rl_pair["base_train"], errors))
    base_val = pd.read_parquet(verify_file_exists(rl_pair["base_val"], errors))

    _check_rl_pair(meta_train, base_train, rl_pair["train_rows"], "train", errors)
    _check_rl_pair(meta_val, base_val, rl_pair["val_rows"], "val", errors)


def assert_contains(text: str, needle: str, message: str, errors: list[str]) -> None:
    emit(needle in text, message, errors)


def assert_numeric_assignment(text: str, key: str, expected: object, message: str, errors: list[str]) -> None:
    candidates = {str(expected)}
    if isinstance(expected, float):
        candidates.add(f"{expected:g}")
        candidates.add(f"{expected:.0e}".replace("e-0", "e-").replace("e+0", "e+"))
    emit(any(f"{key}={cand}" in text for cand in candidates), message, errors)


def check_rl_launcher(contract: dict, errors: list[str]) -> None:
    print("\n## RL Launcher")
    launcher_rel = contract["rl"]["canonical_launcher"]
    launcher_path = verify_file_exists(launcher_rel, errors)
    if not launcher_path.exists():
        return

    text = launcher_path.read_text(encoding="utf-8")
    shared = contract["rl"]["shared"]

    for key, expected in [
        ("PROMPT_LENGTH", shared["prompt_length"]),
        ("RESPONSE_LENGTH", shared["response_length"]),
        ("BATCH_SIZE", shared["train_batch_size"]),
        ("ROLLOUT_N", shared["rollout_n"]),
        ("LR", shared["learning_rate"]),
        ("KL_COEF", shared["kl_coef"]),
        ("SAVE_FREQ", shared["save_freq"]),
        ("TEST_FREQ", shared["test_freq"]),
        ("TOTAL_STEPS", shared["total_training_steps"]),
    ]:
        assert_numeric_assignment(text, key, expected, f"launcher {key}={expected}", errors)

    meta = contract["rl"]["meta"]
    base = contract["rl"]["base"]
    for lane in [meta, base]:
        assert_contains(text, lane["model_path"], f"launcher contains {lane['mode']} model path", errors)
        assert_contains(text, lane["train_data"], f"launcher contains {lane['mode']} train data", errors)
        assert_contains(text, lane["val_data"], f"launcher contains {lane['mode']} val data", errors)
        assert_contains(text, lane["reward_name"], f"launcher contains {lane['mode']} reward name", errors)
        assert_contains(text, lane["experiment_name"], f"launcher contains {lane['mode']} experiment name", errors)

    for reward_key in meta["reward_keys"]:
        assert_contains(text, reward_key, f"launcher contains meta reward key {reward_key}", errors)


def check_docs(contract: dict, errors: list[str]) -> None:
    print("\n## Docs / Registry")
    verify_file_exists(contract["plan_doc"], errors)
    verify_file_exists(contract["node_policy"], errors)
    verify_file_exists("docs/artifact_policy.md", errors)
    verify_file_exists("docs/pipeline_stages.md", errors)
    verify_file_exists("docs/mainline_registry_2026_04_13.md", errors)
    verify_file_exists("data/README.md", errors)

    plan_text = (ROOT / contract["plan_doc"]).read_text(encoding="utf-8")
    policy_text = (ROOT / contract["node_policy"]).read_text(encoding="utf-8")

    for needle in [
        "prompt_length=2048",
        "response_length=4096",
        "train_batch_size=64",
        "rollout.n=4",
        "learning_rate=1e-6",
        "kl_coef=0.001",
    ]:
        assert_contains(plan_text, needle, f"plan mentions {needle}", errors)
        assert_contains(policy_text, needle, f"policy mentions {needle}", errors)

    assert_contains(policy_text, contract["rl"]["canonical_launcher"], "policy names canonical RL launcher", errors)


def check_analysis_artifacts(contract: dict, errors: list[str]) -> None:
    print("\n## Analysis Artifacts")
    verify_file_exists(contract["analysis"]["paired_eval"]["meta"], errors)
    verify_file_exists(contract["analysis"]["paired_eval"]["base"], errors)
    pair_json = verify_file_exists(contract["analysis"]["paired_behavior_report"], errors)
    if pair_json.exists():
        payload = json.loads(pair_json.read_text(encoding="utf-8"))
        emit("artifact_sources" in payload, "paired behavior report records artifact_sources", errors)
        emit(payload.get("rows") == 1560, "paired behavior report rows = 1560", errors)


def check_self_distill_contract(contract: dict, errors: list[str]) -> None:
    print("\n## Self-Distill Next")
    sd = contract.get("self_distill_next")
    if not sd:
        emit(False, "self_distill_next is missing from contract", errors)
        return

    shared = sd.get("shared", {})
    verify_file_exists(shared.get("launcher_roundtrip", ""), errors)
    verify_file_exists(shared.get("launcher_sft_h200", ""), errors)
    verify_file_exists(shared.get("teacher_topk_builder", ""), errors)

    emit(sd.get("artifact_mode") == "question_only_best_of_n", "self-distill artifact_mode=question_only_best_of_n", errors)
    emit(
        sd.get("selector_ladder") == ["correctness_only", "correct_then_meta", "meta_only_kl"],
        "self-distill selector ladder matches D0/D1/D2",
        errors,
    )
    emit(shared.get("repair_candidates") == 4, "self-distill repair_candidates=4", errors)
    emit(shared.get("max_length") == 4096, "self-distill max_length=4096", errors)
    emit(shared.get("synthetic_meta_injected_rate_must_equal") == 0, "self-distill synthetic_meta_injected_rate_must_equal=0", errors)

    for lane in ["base", "meta"]:
        lane_cfg = sd.get(lane, {})
        if lane == "base":
            verify_file_exists(lane_cfg.get("sft_config_h200", ""), errors)
        else:
            verify_file_exists(lane_cfg.get("sft_config_h200", ""), errors)
            verify_file_exists(lane_cfg.get("scored_config_h200", ""), errors)
            verify_file_exists(lane_cfg.get("meta_kl_config_h200", ""), errors)

    plan_text = (ROOT / contract["plan_doc"]).read_text(encoding="utf-8")
    registry_text = (ROOT / "docs/mainline_registry_2026_04_13.md").read_text(encoding="utf-8")
    for needle in [
        "question_only_best_of_n",
        "correct_then_meta",
        "meta_only KL",
        "compute_score_e21r_v4_smoke",
    ]:
        assert_contains(plan_text, needle, f"plan mentions {needle}", errors)
        assert_contains(registry_text, needle, f"registry mentions {needle}", errors)


def check_side_evidence_smoke(contract: dict, errors: list[str]) -> None:
    print("\n## Side-Evidence RL Smoke")
    smoke = contract.get("side_evidence_rl_smoke")
    if not smoke:
        emit(False, "side_evidence_rl_smoke is missing from contract", errors)
        return

    launcher_path = verify_file_exists(smoke.get("launcher", ""), errors)
    if launcher_path.exists():
        text = launcher_path.read_text(encoding="utf-8")
        assert_contains(text, smoke.get("reward_name", ""), "side-evidence launcher uses configured reward name", errors)
    emit(smoke.get("evidence_class") == "side_evidence", "side-evidence RL smoke is labeled side_evidence", errors)
    emit(bool(smoke.get("must_not_overwrite_historical")), "side-evidence RL smoke preserves historical checkpoint path", errors)


def check_side_evidence_sdpo_regen(contract: dict, errors: list[str]) -> None:
    print("\n## Side-Evidence SDPO Regen")
    lane = contract.get("side_evidence_sdpo_regen")
    if not lane:
        emit(False, "side_evidence_sdpo_regen is missing from contract", errors)
        return

    verify_file_exists(lane.get("launcher_roundtrip", ""), errors)
    generate_path = verify_file_exists(lane.get("launcher_generate", ""), errors)
    verify_file_exists(lane.get("teacher_topk_builder", ""), errors)
    verify_file_exists(lane.get("config_generator", ""), errors)
    emit(lane.get("dataset_mode") == "sdpo_regen", "sdpo_regen contract dataset_mode=sdpo_regen", errors)
    emit(lane.get("evidence_class") == "side_evidence", "sdpo_regen lane is labeled side_evidence", errors)
    emit(bool(lane.get("must_not_be_claim_bearing")), "sdpo_regen lane must_not_be_claim_bearing=true", errors)
    if generate_path.exists():
        text = generate_path.read_text(encoding="utf-8")
        assert_contains(text, "--claim-bearing is only allowed for question_only_best_of_n", "sdpo_regen launcher rejects claim-bearing side-evidence modes", errors)


def main() -> int:
    contract = load_yaml(CONTRACT_PATH)
    errors: list[str] = []

    print(f"Contract: {CONTRACT_PATH.relative_to(ROOT)}")
    check_sft_contract(contract, errors)
    check_strict_data(contract, errors)
    check_rl_data(contract, errors)
    check_rl_launcher(contract, errors)
    check_docs(contract, errors)
    check_analysis_artifacts(contract, errors)
    check_self_distill_contract(contract, errors)
    check_side_evidence_smoke(contract, errors)
    check_side_evidence_sdpo_regen(contract, errors)

    print("\n## Verdict")
    if errors:
        print(f"FAILED with {len(errors)} issue(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
