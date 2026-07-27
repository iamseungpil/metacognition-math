"""push_sft_ckpts_to_hf — completeness rule + prune selection, no network.

The daemon's whole value is that it never uploads garbage and never deletes the
last good copy, so those two pure predicates are what is tested here. Uploads
themselves are exercised against a fake HfApi (recorded calls only); nothing
here touches HF, the network, or /scratch.
"""

import json
import os
import time

import pytest

from scripts.push_sft_ckpts_to_hf import (
    UPLOAD_IGNORE_PATTERNS,
    ckpt_step,
    is_complete,
    main,
    remote_ckpt_names,
    repo_path,
    select_prune_targets,
)


# ---------------------------------------------------------------------------
# fixtures: a fake HF-Trainer checkpoint tree
# ---------------------------------------------------------------------------
def make_ckpt(root, step, *, shards=2, state=True, index=True, config=True,
              deepspeed=False, truncate_shard=False, torn_state=False,
              state_step=None):
    """Build a checkpoint-N/ dir mimicking transformers 4.57.6 _save_checkpoint."""
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    for i, n in enumerate(names):
        (d / n).write_bytes(b"" if (truncate_shard and i == 0) else b"\x00" * 16)
    if index:
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"layer.{i}": n for i, n in enumerate(names)}})
        )
    if config:
        (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (d / "tokenizer.json").write_text("{}")
    (d / "training_args.bin").write_bytes(b"\x00")
    if deepspeed:
        gs = d / f"global_step{step}"
        gs.mkdir()
        (gs / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"\x00" * 64)
        (d / "latest").write_text(f"global_step{step}")
    if state:
        # written LAST, exactly as the Trainer does
        body = "{not json" if torn_state else json.dumps(
            {"global_step": step if state_step is None else state_step, "epoch": 1.0}
        )
        (d / "trainer_state.json").write_text(body)
    return d


def age(d, seconds=3600):
    """Backdate every mtime so the quiescence backstop is satisfied."""
    t = time.time() - seconds
    for p in d.rglob("*"):
        os.utime(p, (t, t))
    os.utime(d, (t, t))
    return d


# ---------------------------------------------------------------------------
# completeness rule
# ---------------------------------------------------------------------------
def test_complete_checkpoint_accepted(tmp_path):
    assert is_complete(age(make_ckpt(tmp_path, 25))) is True


def test_missing_trainer_state_rejected(tmp_path):
    """The exact node-death shape: shards landed, trainer_state.json did not.

    trainer_state.json is written strictly last (trainer.py:3361), so its
    absence means the save was cut off — uploading here ships a checkpoint whose
    weights may be a torn mix.
    """
    d = age(make_ckpt(tmp_path, 25, state=False))
    assert is_complete(d) is False


def test_torn_trainer_state_rejected(tmp_path):
    """save_to_json is a plain f.write with no atomic rename — a half-flushed
    JSON is possible and must not be trusted just because the file exists."""
    assert is_complete(age(make_ckpt(tmp_path, 25, torn_state=True))) is False


def test_state_step_mismatch_rejected(tmp_path):
    """A trainer_state.json left over from an earlier step (stale/renamed dir)
    means the dir's identity is not what its name claims."""
    assert is_complete(age(make_ckpt(tmp_path, 50, state_step=25))) is False


def test_missing_shard_rejected(tmp_path):
    d = age(make_ckpt(tmp_path, 25))
    (d / "model-00001-of-00002.safetensors").unlink()
    assert is_complete(d) is False


def test_zero_byte_shard_rejected(tmp_path):
    """An allocated-but-unwritten shard is the classic mid-write artifact."""
    assert is_complete(age(make_ckpt(tmp_path, 25, truncate_shard=True))) is False


def test_missing_config_rejected(tmp_path):
    assert is_complete(age(make_ckpt(tmp_path, 25, config=False))) is False


def test_no_weights_at_all_rejected(tmp_path):
    d = age(make_ckpt(tmp_path, 25, index=False))
    for p in d.glob("*.safetensors"):
        p.unlink()
    assert is_complete(d) is False


def test_single_file_weights_accepted(tmp_path):
    """A small model saves one model.safetensors with no index; still complete."""
    d = make_ckpt(tmp_path, 25, index=False)
    for p in d.glob("model-*.safetensors"):
        p.unlink()
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    assert is_complete(age(d)) is True


def test_freshly_written_rejected_by_quiescence_backstop(tmp_path):
    """Backstop only — with min_quiet_s=0 the structural rule alone decides."""
    d = make_ckpt(tmp_path, 25)  # mtimes = now
    assert is_complete(d, min_quiet_s=300) is False
    assert is_complete(d, min_quiet_s=0) is True


def test_deepspeed_optimizer_dir_does_not_break_completeness(tmp_path):
    """A full (not save_only_model) checkpoint carries global_stepN/; it is
    still complete — the subdir is excluded at UPLOAD time, not here."""
    assert is_complete(age(make_ckpt(tmp_path, 25, deepspeed=True))) is True


def test_non_checkpoint_dir_rejected(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    assert is_complete(d) is False
    assert is_complete(tmp_path / "checkpoint-nope") is False
    assert is_complete(tmp_path / "does-not-exist") is False


def test_ckpt_step_parsing():
    assert ckpt_step("checkpoint-1212") == 1212
    # -1 sorts oldest, so junk can never be picked as "newest".
    for junk in ("checkpoint-", "checkpoint-abc", "global_step_25", "checkpoint-25.tmp"):
        assert ckpt_step(junk) == -1


# ---------------------------------------------------------------------------
# prune selection
# ---------------------------------------------------------------------------
def test_prune_keeps_newest_k_numerically(tmp_path):
    """Lexical sort would rank checkpoint-100 below checkpoint-25 and delete the
    newest state — the selection must be numeric."""
    names = ["checkpoint-25", "checkpoint-50", "checkpoint-100", "checkpoint-1000"]
    assert select_prune_targets(names, keep=2, latest="checkpoint-1000") == [
        "checkpoint-25",
        "checkpoint-50",
    ]


def test_prune_never_deletes_the_just_uploaded_latest():
    """Even if `latest` falls outside the newest K (clock/rename anomaly), the
    checkpoint we just made durable is retained."""
    names = ["checkpoint-25", "checkpoint-50", "checkpoint-100"]
    assert "checkpoint-25" not in select_prune_targets(names, keep=1, latest="checkpoint-25")
    assert select_prune_targets(names, keep=1, latest="checkpoint-25") == ["checkpoint-50"]


def test_prune_disabled_when_keep_non_positive():
    names = ["checkpoint-25", "checkpoint-50", "checkpoint-100"]
    assert select_prune_targets(names, keep=0, latest="checkpoint-100") == []
    assert select_prune_targets(names, keep=-1, latest="checkpoint-100") == []


def test_prune_ignores_foreign_names():
    """Only what this daemon created is ever deleted."""
    names = ["checkpoint-25", "checkpoint-50", "global_step_50", "README.md", "wandb"]
    assert select_prune_targets(names, keep=1, latest="checkpoint-50") == ["checkpoint-25"]


def test_prune_returns_oldest_first():
    names = [f"checkpoint-{n}" for n in (25, 50, 75, 100)]
    assert select_prune_targets(names, keep=1, latest="checkpoint-100") == [
        "checkpoint-25",
        "checkpoint-50",
        "checkpoint-75",
    ]


def test_prune_noop_when_fewer_than_keep():
    assert select_prune_targets(["checkpoint-25"], keep=2, latest="checkpoint-25") == []


# ---------------------------------------------------------------------------
# remote listing / path layout
# ---------------------------------------------------------------------------
def test_remote_ckpt_names_root_and_prefix():
    files = [
        "checkpoint-25/config.json",
        "checkpoint-25/model.safetensors",
        "checkpoint-100/config.json",
        "README.md",
        "wandb/sft/run.log",
    ]
    assert remote_ckpt_names(files) == {"checkpoint-25", "checkpoint-100"}
    # A prefixed layout must not be harvested by the root pattern, and vice versa.
    pfx = ["sft2/b0p2/checkpoint-25/config.json", "checkpoint-9/config.json"]
    assert remote_ckpt_names(pfx, "sft2/b0p2") == {"checkpoint-25"}
    assert remote_ckpt_names(pfx) == {"checkpoint-9"}
    assert remote_ckpt_names(None) == set()


def test_repo_path_layout():
    assert repo_path("", "checkpoint-25") == "checkpoint-25"
    assert repo_path("/sft2/b0p2/", "checkpoint-25") == "sft2/b0p2/checkpoint-25"


def test_ignore_patterns_exclude_deepspeed_optimizer_dump():
    """~98GB of ZeRO-3 CPU-offloaded Adam state must never be pushed."""
    for pat in ("global_step*", "global_step*/*"):
        assert pat in UPLOAD_IGNORE_PATTERNS


# ---------------------------------------------------------------------------
# loop behaviour against a fake HfApi (no network)
# ---------------------------------------------------------------------------
class FakeApi:
    def __init__(self, files=(), fail_upload=False, fail_probe=False):
        self.files = list(files)
        self.uploads = []
        self.deletes = []
        self.squashes = 0
        self.fail_upload = fail_upload
        self.fail_probe = fail_probe
        self.probes = []
        self.probe_deletes = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type,
                    commit_message, token=None):
        if self.fail_probe:
            raise RuntimeError("403 Forbidden: token has no write scope")
        self.probes.append(path_in_repo)

    def delete_file(self, *, path_in_repo, repo_id, repo_type, token=None):
        self.probe_deletes.append(path_in_repo)

    def list_repo_files(self, repo_id, repo_type="model"):
        return list(self.files)

    def upload_folder(self, *, folder_path, repo_id, repo_type, path_in_repo,
                      commit_message, ignore_patterns):
        if self.fail_upload:
            raise RuntimeError("simulated 500 from the Hub")
        self.uploads.append(path_in_repo)
        self.files.append(f"{path_in_repo}/config.json")

    def delete_folder(self, *, path_in_repo, repo_id, repo_type, commit_message):
        self.deletes.append(path_in_repo)
        self.files = [f for f in self.files if not f.startswith(path_in_repo + "/")]

    def super_squash_history(self, *, repo_id, repo_type):
        self.squashes += 1


@pytest.fixture
def fake_hub(monkeypatch):
    """Install a fake huggingface_hub module so main() never touches the network."""
    import sys
    import types

    holder = {}

    def _factory(*a, **k):
        return holder["api"]

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = _factory
    mod.create_repo = lambda **k: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return holder


def _run(tmp_path, api, extra=()):
    return main([
        "--ckpt_dir", str(tmp_path / "out"),
        "--repo_id", "iamseungpil/fake-run",
        "--interval", "0", "--once",
        "--marker_dir", str(tmp_path / "marker"),
        "--min_quiet_s", "0",
        *extra,
    ])


def test_uploads_newest_complete_and_prunes(tmp_path, fake_hub):
    out = tmp_path / "out"
    for step in (25, 50, 75):
        make_ckpt(out, step)
    fake_hub["api"] = api = FakeApi(files=[
        "checkpoint-25/config.json", "checkpoint-50/config.json",
    ])
    _run(tmp_path, api, extra=["--keep", "2"])
    # Only the newest is worth uploading; the backlog is not re-pushed.
    assert api.uploads == ["checkpoint-75"]
    assert api.deletes == ["checkpoint-25"]


def test_incomplete_newest_falls_back_to_older_complete(tmp_path, fake_hub):
    """A checkpoint being written right now must not block the previous one from
    reaching HF — that is the difference between losing 25 steps and losing all
    of them."""
    out = tmp_path / "out"
    make_ckpt(out, 50)
    make_ckpt(out, 75, state=False)  # mid-save
    fake_hub["api"] = api = FakeApi()
    _run(tmp_path, api, extra=["--keep", "0"])
    assert api.uploads == ["checkpoint-50"]


def test_incomplete_checkpoint_is_not_marked_done(tmp_path, fake_hub):
    """done means 'verified on HF'. Marking a skipped dir done is exactly the
    bug that froze HF at gs245 while training ran to gs295."""
    out = tmp_path / "out"
    make_ckpt(out, 75, state=False)
    fake_hub["api"] = api = FakeApi()
    _run(tmp_path, api)
    assert api.uploads == []
    marker = tmp_path / "marker" / ".pushed_sft_sft.json"
    assert not marker.exists() or json.loads(marker.read_text()) == []

    # Now the save finishes; the next scan must pick it up.
    (out / "checkpoint-75" / "trainer_state.json").write_text(json.dumps({"global_step": 75}))
    _run(tmp_path, api)
    assert api.uploads == ["checkpoint-75"]


def test_remote_seed_prevents_reupload_after_restage(tmp_path, fake_hub):
    """A node that pulled checkpoint-75 for a warm restart must not re-push
    those 16GB just because its local marker died with the old node."""
    out = tmp_path / "out"
    make_ckpt(out, 75)
    fake_hub["api"] = api = FakeApi(files=["checkpoint-75/config.json"])
    _run(tmp_path, api)
    assert api.uploads == []


def test_upload_failure_does_not_abort_or_mark_done(tmp_path, fake_hub):
    out = tmp_path / "out"
    make_ckpt(out, 75)
    fake_hub["api"] = api = FakeApi(fail_upload=True)
    _run(tmp_path, api)  # must return normally, not raise
    assert api.uploads == []
    api.fail_upload = False
    _run(tmp_path, api)
    assert api.uploads == ["checkpoint-75"]


def test_missing_ckpt_dir_is_survivable(tmp_path, fake_hub):
    """Training has not saved anything yet — the daemon just waits."""
    fake_hub["api"] = api = FakeApi()
    _run(tmp_path, api)
    assert api.uploads == []


def test_no_token_flag_exists():
    """SECURITY 0716: a --token argument shows up in `ps` for the daemon's whole
    lifetime and in std_log under `set -x`. The token must come from HF_TOKEN."""
    from scripts.push_sft_ckpts_to_hf import build_parser

    opts = {o for a in build_parser()._actions for o in a.option_strings}
    assert "--token" not in opts


def test_boot_probe_writes_and_cleans_up(tmp_path, fake_hub):
    """A healthy destination leaves nothing behind - the probe file is deleted."""
    make_ckpt(tmp_path / "out", 25)
    fake_hub["api"] = api = FakeApi()
    _run(tmp_path, api)
    assert api.probes == [".push_probe"]
    assert api.probe_deletes == [".push_probe"]


def test_boot_probe_aborts_when_the_destination_is_unwritable(tmp_path, fake_hub):
    """The failure this guards against: a typo'd repo_id or a read-only token
    let the daemon look healthy forever while the run stayed undurable. It must
    die at boot instead, before the compute window is spent."""
    make_ckpt(tmp_path / "out", 25)
    fake_hub["api"] = api = FakeApi(fail_probe=True)
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, api)
    assert exc.value.code == 1
    assert api.uploads == []  # never got as far as pretending to work
