"""★감사 10: 사다리 자동화 — ladder_next.sh 는 <ckpt>/ABORTED.txt 가 생기거나 잡이 running/
을 떠나면 다음 팔 제출 커맨드를 **한 번만** 실행한다(사전등록 §4 «중단되면 다음 팔 자동 제출»).
가짜 ckpt 디렉토리·가짜 큐로 실제 스크립트를 돌린다."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "local" / "ladder_next.sh"


def _env(tmp_path, **kw):
    work = tmp_path / "work"; q = tmp_path / "queue"
    for d in ("pending", "running", "done", "failed", "aborted"):
        (q / d).mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WORK=str(work), QUEUE_ROOT=str(q), LADDER_POLL_S="0.2",
               LADDER_APPEAR_TIMEOUT_S="2", LADDER_LOG=str(work / "logs" / "ladder.log"))
    env.update(kw)
    return env, work, q


def _run(env, lineage, submit_cmd, timeout=30):
    return subprocess.run(["bash", str(SCRIPT), lineage, submit_cmd], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_aborted_marker_triggers_submit_once(tmp_path):
    env, work, q = _env(tmp_path)
    ck = work / "checkpoints" / "cd9_M_JUDGE_s1"; ck.mkdir(parents=True)
    (q / "running" / "1.json").write_text(json.dumps({"name": "cd9_M_JUDGE_s1", "cmd": "x"}))
    out = tmp_path / "submitted.txt"
    proc = subprocess.Popen(["bash", str(SCRIPT), "cd9_M_JUDGE_s1", f"echo hi >> {out}"], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)
    assert proc.poll() is None and not out.exists()          # 아직 기다린다
    (ck / "ABORTED.txt").write_text("[MATH][ABORT] ...")
    proc.wait(timeout=20)
    assert proc.returncode == 0, proc.stdout.read()
    assert out.read_text().count("hi") == 1
    assert (ck / ".ladder_next_submitted").exists()
    # 다시 불러도 두 번 제출하지 않는다
    r = _run(env, "cd9_M_JUDGE_s1", f"echo hi >> {out}")
    assert r.returncode == 0 and out.read_text().count("hi") == 1


def test_job_leaving_running_triggers_submit(tmp_path):
    env, work, q = _env(tmp_path)
    (work / "checkpoints" / "cd9_M_G1_s2").mkdir(parents=True)
    jp = q / "running" / "2.json"
    jp.write_text(json.dumps({"name": "w_something", "cmd": "bash scripts/local/run_math_arm.sh M_G1 2"}))
    out = tmp_path / "submitted.txt"
    env["LADDER_MATCH"] = "run_math_arm.sh M_G1 2"
    proc = subprocess.Popen(["bash", str(SCRIPT), "cd9_M_G1_s2", f"echo go >> {out}"], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)
    assert proc.poll() is None and not out.exists()
    jp.rename(q / "done" / "2.json")
    proc.wait(timeout=20)
    assert proc.returncode == 0 and out.read_text().strip() == "go"


def test_pending_job_is_waited_for_then_running_then_left(tmp_path):
    env, work, q = _env(tmp_path)
    (work / "checkpoints" / "cd9_M_RAND_s1").mkdir(parents=True)
    jp = q / "pending" / "3.json"
    jp.write_text(json.dumps({"name": "cd9_M_RAND_s1", "cmd": "x"}))
    out = tmp_path / "submitted.txt"
    proc = subprocess.Popen(["bash", str(SCRIPT), "cd9_M_RAND_s1", f"echo go >> {out}"], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(3.0)                                            # APPEAR_TIMEOUT(2s) 를 넘겨도 pending 이면 기다린다
    assert proc.poll() is None and not out.exists()
    jp.rename(q / "running" / "3.json")
    time.sleep(1.0)
    assert proc.poll() is None
    (q / "running" / "3.json").rename(q / "failed" / "3.json")
    proc.wait(timeout=20)
    assert proc.returncode == 0 and out.read_text().strip() == "go"


def test_never_seen_job_submits_after_grace_and_logs(tmp_path):
    env, work, q = _env(tmp_path)
    out = tmp_path / "submitted.txt"
    r = _run(env, "cd9_M_PROBE_s9", f"echo go >> {out}")
    assert r.returncode == 0 and out.read_text().strip() == "go"
    assert "never seen" in (work / "logs" / "ladder.log").read_text()


def test_submit_failure_is_nonzero_and_logged(tmp_path):
    env, work, q = _env(tmp_path)
    ck = work / "checkpoints" / "cd9_M_G0_s1"; ck.mkdir(parents=True)
    (ck / "ABORTED.txt").write_text("x")
    r = _run(env, "cd9_M_G0_s1", "exit 4")
    assert r.returncode != 0 and "rc=4" in (work / "logs" / "ladder.log").read_text()
