r"""`bash -n` syntax lint for `scripts/local/*.sh` launchers (e.g. `run_sft.sh`,
`run_arm.sh`).

WHY THIS EXISTS (0907, added alongside `scripts/local/run_sft.sh`).
`tests/test_launcher_yaml_lint.py` catches shell defects in the amlt
launcher yamls' embedded `command:` blocks, but those checks only look at
`*.yaml` under the repo root with a `jobs:` list — the local-box bash
launchers under `scripts/local/` are not covered by anything. A syntactically
broken launcher (stray quote, comment eating a `\`-continuation, unbalanced
quoting) parses fine as a file and fails only when actually queued, which is
exactly the failure mode `bash -n` catches for free.

This is a much smaller check than the yaml lint (no comment-after-backslash
heuristic — these scripts are plain bash files, not YAML-embedded here-strings
assembled from `command:` lists), but the same principle applies: run `bash -n`
on every `scripts/local/*.sh` file so a broken launcher is caught by the test
suite instead of by a queued job burning a GPU slot before crashing.

실행:  python -m pytest tests/test_local_shell_lint.py -q
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SCRIPTS_DIR = ROOT / "scripts" / "local"


def _local_shell_scripts():
    return sorted(LOCAL_SCRIPTS_DIR.glob("*.sh"))


SCRIPTS = _local_shell_scripts()


def test_there_are_scripts_to_lint():
    assert SCRIPTS, f"no *.sh files found under {LOCAL_SCRIPTS_DIR}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_is_syntactically_valid_bash(path):
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{path.name} is not valid bash:\n{proc.stderr.strip()}"
