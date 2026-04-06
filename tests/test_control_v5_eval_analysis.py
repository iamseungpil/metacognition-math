"""Smoke test for control-v5 eval summary analysis."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = load_module(
    Path("scripts/analyze_control_v5_eval.py").resolve(),
    "analyze_control_v5_eval",
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


payload = {
    "model": "dummy_model",
    "results": [
        {
            "benchmark": "gsm8k",
            "full_question": "Q1",
            "is_correct": True,
            "num_meta_blocks": 1,
            "avg_confidence": 0.8,
            "completion": "<|meta|>\nconfidence: 0.8\nI should verify once before committing.\n<|/meta|>\nLet me substitute back. \\boxed{4}",
        },
        {
            "benchmark": "aime2024",
            "full_question": "Q2",
            "is_correct": False,
            "num_meta_blocks": 2,
            "avg_confidence": 0.92,
            "completion": "<|meta|>\nconfidence: 0.92\nI may be overcommitting and should check.\n<|/meta|>\nWrong route.\n<|meta|>\nconfidence: 0.41\nSomething feels off. The current route is weak because I am forcing the wrong invariant.\nstudy_need: parity invariant\nI should switch methods.\n<|/meta|>\n\\boxed{7}",
        },
    ],
}

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "eval_dummy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = module.summarize_file(path)
    markdown = module.build_markdown({"dummy_model": summary})

check("summary keeps model name", summary["overall"]["model"] == "dummy_model")
check("overall accuracy computed", abs(summary["overall"]["accuracy"] - 0.5) < 1e-6)
check("study_need detected", summary["overall"]["study_need_rate"] > 0.0)
check("redirect sample extracted", len(summary["overall"]["redirect_samples"]) >= 1)
check("markdown contains model header", "## dummy_model" in markdown)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
