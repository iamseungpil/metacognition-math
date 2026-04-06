"""Lightweight checks for probe cache building helpers."""
import sys

sys.path.insert(0, ".")

import numpy as np

from src.probes.retrain import (
    _group_train_val_split,
    _iter_meta_prefix_texts,
    _lookup_prefix_target,
    _resolve_probe_target,
)


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


sample = "Q\n<|meta|>confidence: 0.4<|/meta|>\nwork\n<|meta|>confidence: 0.8<|/meta|>\n\\boxed{4}"
prefixes = _iter_meta_prefix_texts(sample)
check("two meta prefixes extracted", len(prefixes) == 2)
check("prefix ends at first closing tag", prefixes[0][1].endswith("<|/meta|>"))
check("second prefix includes first and second meta blocks", prefixes[1][1].count("<|meta|>") == 2)

manifest = [
    {"problem_id": "p1", "idx": 0},
    {"problem_id": "p1", "idx": 1},
    {"problem_id": "p2", "idx": 2},
    {"problem_id": "p3", "idx": 3},
]
train_manifest, val_manifest = _group_train_val_split(manifest, val_fraction=0.34, seed=42)
train_ids = {row["problem_id"] for row in train_manifest}
val_ids = {row["problem_id"] for row in val_manifest}

check("group split keeps problem ids disjoint", train_ids.isdisjoint(val_ids))
check("all rows preserved after split", len(train_manifest) + len(val_manifest) == len(manifest))
check("at least one validation group exists", len(val_ids) >= 1)

row_with_targets = {
    "is_correct": True,
    "meta_prefix_target_probs": [0.25, 0.8],
}
target0, source0 = _lookup_prefix_target(row_with_targets, 0)
check("prefix target lookup finds first entry", abs(target0 - 0.25) < 1e-6 and source0 == "meta_prefix_target_probs")
target1, source1 = _resolve_probe_target(row_with_targets, "meta_prefix", 1)
check("resolve probe target uses prefix-conditioned probability", abs(target1 - 0.8) < 1e-6 and source1 == "meta_prefix_target_probs")

row_with_numpy_targets = {
    "is_correct": True,
    "meta_prefix_target_probs": np.array([0.2, 0.9], dtype=float),
}
target_numpy, source_numpy = _lookup_prefix_target(row_with_numpy_targets, 1)
check(
    "prefix target lookup accepts numpy arrays from parquet",
    abs(target_numpy - 0.9) < 1e-6 and source_numpy == "meta_prefix_target_probs",
)

row_missing_prefix = {"is_correct": False}
missing_target, missing_source = _resolve_probe_target(row_missing_prefix, "meta_prefix", 0)
check("missing prefix target is explicit", missing_target is None and missing_source == "missing_prefix_target")

full_target, full_source = _resolve_probe_target({"is_correct": True}, "full", None)
check("full sample falls back to final correctness only", abs(full_target - 1.0) < 1e-6 and full_source == "trajectory_final_correctness")

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
