#!/usr/bin/env python3
"""Smoke check for behavior-uncertainty analysis inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/v-seungplee/metacognition/analysis/behavior_uncertainty_lab")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(ROOT / "configs" / "targets.json"))
    args = parser.parse_args()

    targets_path = Path(args.targets)
    with targets_path.open() as f:
        targets = json.load(f)["models"]
    assert targets, "No targets configured"
    for target in targets:
        eval_path = Path(target["eval_json"])
        assert eval_path.exists(), f"Missing eval file: {eval_path}"
        with eval_path.open() as f:
            payload = json.load(f)
        assert "results" in payload and payload["results"], f"Empty results: {eval_path}"
        sample = payload["results"][0]
        required = {"benchmark", "question", "completion", "is_correct", "avg_confidence", "num_meta_blocks"}
        missing = required - set(sample.keys())
        assert not missing, f"Missing keys {missing} in {eval_path}"
    print("smoke_ok")


if __name__ == "__main__":
    main()
