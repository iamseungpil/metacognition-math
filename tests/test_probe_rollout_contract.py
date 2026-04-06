"""Smoke checks for prefix-conditioned probe rollout contract."""
import sys

sys.path.insert(0, ".")

from scripts.build_probe_rollouts_hf import (
    _empirical_prefix_success,
    _iter_meta_completion_prefixes,
    _select_shard,
    _shard_output_path,
    merge_rollout_shards,
)
from src.training.rewards import _prefix_payloads_for_probe
import json
import tempfile
from pathlib import Path
import pandas as pd


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


completion = (
    "<|meta|>confidence: 0.40 something feels off<|/meta|>\n"
    "work\n"
    "<|meta|>confidence: 0.75 verify before commit<|/meta|>\n"
    "\\boxed{4}"
)
prefixes = _iter_meta_completion_prefixes(completion)
check("two completion prefixes extracted", len(prefixes) == 2)
check("first completion prefix ends after first meta", prefixes[0][1].endswith("<|/meta|>"))

prompt = [{"role": "user", "content": "What is 2+2?"}]
payloads = _prefix_payloads_for_probe(completion, prompt=prompt, tokenizer=None)
check("prompt-conditioned payload count matches meta blocks", len(payloads) == 2)
check("prompt-conditioned payload contains question context", "What is 2+2?" in payloads[0])
check("prompt-conditioned payload ends with completion prefix", payloads[1].endswith(prefixes[1][1]))


def fake_sampler(payload):
    if "verify before commit" in payload:
        return ["\\boxed{4}", "\\boxed{4}", "\\boxed{4}", "\\boxed{5}"]
    if "something feels off" in payload:
        return ["\\boxed{4}", "\\boxed{5}", "\\boxed{4}", "\\boxed{7}"]
    return ["\\boxed{5}"]


target_probs = _empirical_prefix_success(
    payloads,
    gold_answer="4",
    continuation_sampler=fake_sampler,
)
check("first empirical prefix target computed", abs(target_probs[0] - 0.5) < 1e-6)
check("second empirical prefix target computed", abs(target_probs[1] - 0.75) < 1e-6)

items = [{"problem_id": f"p{i}"} for i in range(7)]
shards = [_select_shard(items, i, 4) for i in range(4)]
check("all items assigned across shards", sum(len(s) for s in shards) == len(items))
check("first shard gets p0 and p4", [x["problem_id"] for x in shards[0]] == ["p0", "p4"])
check("shard path suffix includes shard id", str(_shard_output_path(Path("tmp.parquet"), 2, 4)).endswith(".shard2of4.parquet"))

tmpdir = Path(tempfile.mkdtemp(prefix="probe_rollout_merge_"))
merged_path = tmpdir / "rollouts.parquet"
for shard_idx in range(2):
    shard_path = _shard_output_path(merged_path, shard_idx, 2)
    pd.DataFrame([
        {
            "problem_id": f"p{shard_idx}",
            "question": "q",
            "gold_answer": "4",
            "prompt_text": "prompt",
            "completion": "\\boxed{4}",
            "is_correct": True,
            "meta_prefix_count": 1,
            "meta_prefix_target_probs": [0.5],
            "continuations_per_prefix": 4,
        }
    ]).to_parquet(shard_path, index=False)
summary = merge_rollout_shards(merged_path, 2)
merged_df = pd.read_parquet(merged_path)
check("merge produces combined file", len(merged_df) == 2)
check("merge summary row count", summary["n_rows"] == 2)
summary_json = json.loads(merged_path.with_suffix(".summary.json").read_text())
check("merge writes summary json", summary_json["n_rows"] == 2)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        raise SystemExit(1)
