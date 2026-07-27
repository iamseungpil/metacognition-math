"""Preemption survival wiring for SFT: resume-from-newest-checkpoint + save_total_limit.

WHY THIS EXISTS. `src/training/sft.py` called `trainer.train()` with no
`resume_from_checkpoint`, so every checkpoint a preempted node left behind was
dead weight: a resubmitted job restarted from step 0. And `save_total_limit` was
hardcoded to 3, so a config could not trade disk for save frequency. Both are now
yaml passthroughs, and both MUST stay backward compatible — a config that sets
neither key has to behave exactly as before (fresh start, limit 3).

GPU-free and network-free: only the two pure resolver helpers are exercised,
against temp dirs. The single call-site assertion is done on the AST, so the wiring
is pinned without instantiating a model.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from src.training import sft as sft_mod
from src.training.sft import _resolve_resume_checkpoint, _resolve_save_total_limit


def _make_ckpt(root: Path, step: int, complete: bool = True) -> Path:
    """Fabricate a checkpoint-N dir. `complete` controls trainer_state.json, which
    Trainer writes strictly LAST (transformers 4.57.6 trainer.py:3361) and which is
    therefore our completeness marker."""
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    if complete:
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return d


# --------------------------------------------------------------------------
# resume resolution
# --------------------------------------------------------------------------

def test_config_without_key_never_resumes_even_when_checkpoints_exist(tmp_path):
    """The backward-compat guarantee. Every pre-existing config omits the key;
    None is what `trainer.train()` already defaulted to, so behaviour is identical."""
    _make_ckpt(tmp_path, 100)
    assert _resolve_resume_checkpoint({}, str(tmp_path)) is None


@pytest.mark.parametrize("setting", [None, False])
def test_explicit_null_or_false_does_not_resume(tmp_path, setting):
    _make_ckpt(tmp_path, 100)
    cfg = {"resume_from_checkpoint": setting}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) is None


def test_auto_picks_newest_checkpoint_by_step_number(tmp_path):
    """Lexicographic order would pick checkpoint-9; step order must win."""
    _make_ckpt(tmp_path, 9)
    _make_ckpt(tmp_path, 25)
    newest = _make_ckpt(tmp_path, 100)
    cfg = {"resume_from_checkpoint": "auto"}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) == str(newest)


@pytest.mark.parametrize("setting", [True, "auto", "AUTO", "true", "yes", "1"])
def test_truthy_spellings_all_mean_auto(tmp_path, setting):
    newest = _make_ckpt(tmp_path, 50)
    cfg = {"resume_from_checkpoint": setting}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) == str(newest)


def test_auto_skips_torn_checkpoint_from_a_node_death(tmp_path):
    """A node dying mid-save leaves a checkpoint-N with no trainer_state.json.
    Trainer reads that file before any deepspeed branch (trainer.py:2299), so
    handing it back would crash the resume instead of rescuing it."""
    good = _make_ckpt(tmp_path, 100)
    _make_ckpt(tmp_path, 120, complete=False)
    cfg = {"resume_from_checkpoint": "auto"}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) == str(good)


def test_auto_on_empty_output_dir_starts_fresh(tmp_path):
    """transformers' own get_last_checkpoint would make Trainer raise
    ValueError('No valid checkpoint found') here; auto must degrade to a fresh run."""
    cfg = {"resume_from_checkpoint": "auto"}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) is None


def test_auto_on_missing_output_dir_starts_fresh(tmp_path):
    """First submission: output_dir does not exist yet. os.listdir would raise
    FileNotFoundError, so the isdir guard is load-bearing."""
    cfg = {"resume_from_checkpoint": "auto"}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path / "nope")) is None


def test_auto_ignores_non_checkpoint_entries(tmp_path):
    (tmp_path / "checkpoint-notanumber").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoint-77").write_text("a file, not a dir")
    _make_ckpt(tmp_path, 25)
    cfg = {"resume_from_checkpoint": "auto"}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) == str(tmp_path / "checkpoint-25")


def test_explicit_path_is_honoured(tmp_path):
    _make_ckpt(tmp_path, 200)
    target = _make_ckpt(tmp_path, 50)
    cfg = {"resume_from_checkpoint": str(target)}
    assert _resolve_resume_checkpoint(cfg, str(tmp_path)) == str(target)


def test_explicit_path_that_does_not_exist_fails_fast(tmp_path):
    """Better a ValueError at t=0 than 12h of training silently started from scratch."""
    cfg = {"resume_from_checkpoint": str(tmp_path / "ghost")}
    with pytest.raises(ValueError):
        _resolve_resume_checkpoint(cfg, str(tmp_path))


# --------------------------------------------------------------------------
# save_total_limit
# --------------------------------------------------------------------------

def test_save_total_limit_defaults_to_the_old_hardcoded_three():
    assert _resolve_save_total_limit({}) == 3


def test_save_total_limit_reads_the_yaml_value():
    assert _resolve_save_total_limit({"save_total_limit": 2}) == 2
    assert _resolve_save_total_limit({"save_total_limit": "1"}) == 1


# --------------------------------------------------------------------------
# call-site wiring (AST — no model, no GPU)
# --------------------------------------------------------------------------

def _run_sft_ast() -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(sft_mod))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_sft":
            return node
    raise AssertionError("run_sft not found in src/training/sft.py")


def test_trainer_train_is_called_with_resolved_resume_checkpoint():
    """The helper is worthless if run_sft still calls a bare trainer.train()."""
    calls = [
        n for n in ast.walk(_run_sft_ast())
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "train"
    ]
    assert calls, "no trainer.train() call in run_sft"
    kwargs = {k.arg for c in calls for k in c.keywords}
    assert "resume_from_checkpoint" in kwargs


def test_training_arguments_gets_save_total_limit_from_the_resolver():
    calls = [
        n for n in ast.walk(_run_sft_ast())
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "TrainingArguments"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "save_total_limit" in kw
    assert isinstance(kw["save_total_limit"], ast.Call), "save_total_limit is hardcoded again"
    assert kw["save_total_limit"].func.id == "_resolve_save_total_limit"
