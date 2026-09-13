"""★감사 9: gpu_queue._run_one_job 의 «부활 경쟁» — gate_judgment 가 running/→aborted/ 로
옮긴 뒤 워커가 job_path.write_text 하면 running/ 에 파일이 되살아나 failed/ 로도 기록된다.
exists() 검사 뒤에는 job_path 에 절대 쓰지 않고, os.rename 이 FileNotFoundError 를 내면
aborted/ 쪽 기록만 갱신한다는 것을 실제 프로세스로 확인한다."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
import gpu_queue as Q  # noqa: E402


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "queue"
    for s in ("pending", "running", "done", "failed", "aborted", "pids"):
        (root / s).mkdir(parents=True)
    monkeypatch.setattr(Q, "QUEUE_ROOT", root)
    monkeypatch.setattr(Q, "PIDS_DIR", root / "pids")
    monkeypatch.setattr(Q, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()
    return root


def _job(root, cmd, name="j1"):
    p = root / "running" / "1_abc.json"
    p.write_text(json.dumps({"id": "1_abc", "cmd": cmd, "name": name}))
    return p


def test_normal_finish_moves_to_done_without_tmp_leftover(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    p = _job(root, "true")
    Q._run_one_job(7, p, None)
    assert not p.exists() and not list((root / "running").iterdir())
    j = json.loads((root / "done" / "1_abc.json").read_text())
    assert j["exit_code"] == 0 and j["assigned_gpu"] == 7 and "finished_at" in j


def test_failed_finish_moves_to_failed(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    p = _job(root, "exit 3")
    Q._run_one_job(7, p, None)
    assert json.loads((root / "failed" / "1_abc.json").read_text())["exit_code"] == 3
    assert not list((root / "running").iterdir())


def test_rc75_goes_to_aborted(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    p = _job(root, "exit 75")
    Q._run_one_job(7, p, None)
    j = json.loads((root / "aborted" / "1_abc.json").read_text())
    assert j["exit_code"] == 75 and "aborted_reason" in j
    assert not list((root / "running").iterdir())


def test_gate_move_between_exists_check_and_rename_does_not_resurrect(tmp_path, monkeypatch):
    """exists() 는 True 였는데 rename 직전에 gate 가 파일을 aborted/ 로 옮긴 경우(경쟁의 창).
    옛 코드는 job_path.write_text 로 running/ 에 되살린 뒤 failed/ 로 옮겼다(이중 기록)."""
    root = _setup(tmp_path, monkeypatch)
    p = _job(root, "exit 1")
    real_rename = os.rename
    state = {"raced": False}

    def racing_rename(src, dst, *a, **kw):
        # 워커가 running/<job> 을 옮기려는 첫 순간에 gate 가 끼어든다.
        if not state["raced"] and Path(src) == p:
            state["raced"] = True
            j = json.loads(p.read_text()); j["aborted_reason"] = "gate_judgment: 판정 스텝 지표 미달"
            (root / "aborted" / p.name).write_text(json.dumps(j))
            os.unlink(p)
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", racing_rename)
    Q._run_one_job(7, p, None)
    assert state["raced"]
    assert not list((root / "running").iterdir()), "running/ 에 부활하면 안 된다"
    assert not list((root / "failed").iterdir()) and not list((root / "done").iterdir())
    j = json.loads((root / "aborted" / "1_abc.json").read_text())
    assert j["exit_code"] == 1 and j["aborted_reason"].startswith("gate_judgment")
    assert "finished_at" in j
