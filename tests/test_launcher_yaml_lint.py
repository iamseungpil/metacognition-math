"""Lint the amlt launcher YAMLs for shell defects that `bash -n` accepts.

WHY THIS EXISTS (0727). Two commits on 0727 (E-117, E-118) inserted explanatory
comment blocks in the MIDDLE of a backslash-continuation chain:

    ++algorithm.norm_adv_by_std_in_grpo=false \\
    # OOM FIX 0727 (E-118): ...
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\

A trailing backslash splices the next line on, and everything after the `#` on
the spliced line is a comment -- so the command ENDS at the comment and every
argument below it becomes a separate command that dies with "command not found".
Six launchers were truncated this way, including the arguments the commits
existed to add, and one of them was submitted before it was caught.

`bash -n` does not catch this: the result is syntactically valid, just wrong.
Nor does yaml validation, nor reading the diff -- the comment looks like it
belongs to the line it documents. Only a rule about adjacency catches it, so
that rule lives here.

The second check is the same idea for the other direction: a continuation line
that is entirely blank also ends the command silently.
"""
from __future__ import annotations

from pathlib import Path

import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _launcher_yamls():
    """Every root yaml that parses AND declares jobs.

    Parse failures are collected separately rather than skipped. Skipping them is
    how this file first failed: five freshly written launchers had a stray quote
    that terminated the description scalar early, they raised YAMLError, this
    function dropped them, and the suite reported all green over a set that no
    longer contained the broken files. A lint that quietly shrinks its own input
    is worse than no lint.
    """
    out, broken = [], []
    for p in sorted(ROOT.glob("*.yaml")):
        try:
            doc = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            broken.append((p.name, str(exc).splitlines()[0]))
            continue
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), list):
            out.append((p, doc))
    return out, broken


LAUNCHERS, UNPARSEABLE = _launcher_yamls()


def test_every_root_yaml_parses():
    assert not UNPARSEABLE, "root yaml that does not parse (and would be skipped by every" \
        " other check here):\n" + "\n".join(f"  {n}: {e}" for n, e in UNPARSEABLE)


def _command_blocks(doc):
    """Yield (job_name, command_index, text) for every command string."""
    for job in doc["jobs"]:
        for idx, cmd in enumerate(job.get("command") or []):
            if isinstance(cmd, str):
                yield job.get("name", "<unnamed>"), idx, cmd


def test_there_are_launchers_to_lint():
    """Guard against the glob silently matching nothing and the suite passing."""
    assert LAUNCHERS, f"no launcher yamls with a jobs: list found under {ROOT}"


@pytest.mark.parametrize("path,doc", LAUNCHERS, ids=lambda v: getattr(v, "name", ""))
def test_no_comment_inside_a_continuation_chain(path, doc):
    offenders = []
    for job, idx, cmd in _command_blocks(doc):
        lines = cmd.split("\n")
        for i in range(len(lines) - 1):
            if lines[i].rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("#"):
                offenders.append(
                    f"{path.name}:{job}:cmd[{idx}] line {i + 1}\n"
                    f"    {lines[i].strip()}\n"
                    f"    {lines[i + 1].strip()}"
                )
    assert not offenders, (
        "a comment directly after a line ending in `\\` truncates the command:\n"
        + "\n".join(offenders)
        + "\n  Move the comment ABOVE the command instead."
    )


@pytest.mark.parametrize("path,doc", LAUNCHERS, ids=lambda v: getattr(v, "name", ""))
def test_no_blank_line_inside_a_continuation_chain(path, doc):
    offenders = []
    for job, idx, cmd in _command_blocks(doc):
        lines = cmd.split("\n")
        for i in range(len(lines) - 1):
            if lines[i].rstrip().endswith("\\") and not lines[i + 1].strip():
                offenders.append(f"{path.name}:{job}:cmd[{idx}] line {i + 1}: {lines[i].strip()}")
    assert not offenders, (
        "a blank line directly after a line ending in `\\` ends the command:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path,doc", LAUNCHERS, ids=[p.name for p, _ in LAUNCHERS])
def test_every_command_is_syntactically_valid_bash(path, doc):
    """The outer command is `bash -c '<whole script>'`, so a single apostrophe
    anywhere inside - including inside a comment - closes that quote and the
    rest of the job becomes garbage. yaml parses it, a diff reads fine, and the
    node fails at runtime. `bash -n` is what actually sees it.
    """
    for i, job in enumerate(doc["jobs"]):
        command = job.get("command")
        if command is None:
            continue
        script = command if isinstance(command, str) else "\n".join(command)
        proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert proc.returncode == 0, (
            f"{path.name} jobs[{i}] is not valid bash:\n{proc.stderr.strip()}"
        )


def test_the_bash_check_would_catch_a_stray_apostrophe():
    """Negative control: without this, the test above could pass by never running."""
    proc = subprocess.run(
        ["bash", "-n"], text=True, capture_output=True,
        input="bash -c '\necho hi\n# a typo'd word\necho bye\n'\n",
    )
    assert proc.returncode != 0
