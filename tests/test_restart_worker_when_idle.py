"""★감사 8: restart_worker_when_idle.sh 의 경쟁 수리 — STOP 을 먼저 만들고, 그 GPU 에 running
잡이 없을 때까지 기다린 뒤, 옛 워커에 TERM → 종료 대기(최대 N초, 그 뒤 KILL), 잠금 파일은
절대 지우지 않고, 새 워커 pid 를 queue/pids/gpu_<G>_slot_0.pid 에 쓴다. 가짜 큐·가짜
워커(가짜 gpu_queue.py)로 실제 스크립트를 돌린다. GPU 번호는 실제 워커와 겹치지 않게 77."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "local" / "restart_worker_when_idle.sh"
PY = "/hdd_data/seungpil/envs/simplerl/bin/python"
G = "77"

FAKE_WORKER = r'''
import os, sys, time, signal
mode = os.environ.get("FAKE_WORKER_MODE", "graceful")
if mode == "graceful":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
else:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("fake worker up", flush=True)
while True:
    time.sleep(0.2)
'''


def _setup(tmp_path):
    q = tmp_path / "queue"
    for d in ("pending", "running", "done", "failed", "aborted", "pids"):
        (q / d).mkdir(parents=True)
    logs = tmp_path / "logs"; logs.mkdir()
    fake = tmp_path / "fake" / "gpu_queue.py"
    fake.parent.mkdir()
    fake.write_text(FAKE_WORKER)
    env = dict(os.environ, QUEUE_ROOT=str(q), PY=PY, GPU_QUEUE_PY=str(fake), RESTART_LOG=str(logs / "dec.log"),
               WORKER_LOG_DIR=str(logs), RESTART_POLL_S="0.3", RESTART_TERM_WAIT_S="3", RESTART_STOP_SETTLE_S="1")
    return q, logs, fake, env


def _spawn_old(fake, mode="graceful"):
    env = dict(os.environ, FAKE_WORKER_MODE=mode)
    return subprocess.Popen([PY, str(fake), "worker", "--gpu", G, "--jobs-per-gpu", "1"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/status").read_text().split("State:")[1].split()[0] != "Z"
    except OSError:
        return False


def _kill_new(q):
    pf = q / "pids" / f"gpu_{G}_slot_0.pid"
    if pf.exists():
        try:
            os.kill(int(pf.read_text().strip()), 9)
        except OSError:
            pass


def _cmdline(pid: int) -> str:
    return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()


def test_waits_for_running_job_and_creates_stop_first(tmp_path):
    q, logs, fake, env = _setup(tmp_path)
    lock = q / f"gpu_{G}.lock"; lock.write_text("")
    (q / "running" / "r.json").write_text(json.dumps({"name": "busy", "assigned_gpu": int(G)}))
    old = _spawn_old(fake)
    proc = subprocess.Popen(["bash", str(SCRIPT), G], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(1.5)
        # 긴 대기 동안은 전역 STOP 을 들지 않는다(GPU 0/1 을 막지 않기 위해); STOP 은 비어진 뒤 재확인 창에서만.
        assert not (q / "STOP").exists()
        assert proc.poll() is None and old.poll() is None       # 잡이 도는 동안 옛 워커를 건드리지 않는다
        (q / "running" / "r.json").rename(q / "done" / "r.json")
        proc.wait(timeout=30)
        assert proc.returncode == 0, proc.stdout.read()
        old.wait(timeout=5)                                        # TERM 으로 곱게 내려갔다
        assert lock.exists(), "잠금 파일은 지우지 않는다"
        assert not (q / "STOP").exists()
        pf = q / "pids" / f"gpu_{G}_slot_0.pid"
        new_pid = int(pf.read_text().strip())
        assert _alive(new_pid) and f"gpu_queue.py worker --gpu {G} " in _cmdline(new_pid)
        assert new_pid != old.pid
        dec = (logs / "dec.log").read_text()
        assert "gpu77" in dec and "STOP 아래에서 비었음 확정" in dec
    finally:
        _kill_new(q)
        if old.poll() is None:
            old.kill(); old.wait()


def test_stubborn_worker_gets_killed_after_wait(tmp_path):
    q, logs, fake, env = _setup(tmp_path)
    old = _spawn_old(fake, mode="ignore_term")
    t0 = time.time()
    r = subprocess.run(["bash", str(SCRIPT), G], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert time.time() - t0 >= 3.0                             # TERM 대기 창을 다 썼다
        old.wait(timeout=5)
        assert old.returncode == -9                                # KILL 로 끝났다
        assert "KILL" in (logs / "dec.log").read_text()
        assert _alive(int((q / "pids" / f"gpu_{G}_slot_0.pid").read_text()))
    finally:
        _kill_new(q)
        if old.poll() is None:
            old.kill(); old.wait()


def test_no_old_worker_still_starts_new_one(tmp_path):
    q, logs, fake, env = _setup(tmp_path)
    r = subprocess.run(["bash", str(SCRIPT), G], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert _alive(int((q / "pids" / f"gpu_{G}_slot_0.pid").read_text()))
        assert not (q / "STOP").exists()
    finally:
        _kill_new(q)


def test_job_slipping_in_during_settle_releases_stop_and_waits(tmp_path):
    """비어진 것을 본 직후 STOP 을 만들기 전에 잡이 붙은 경우: STOP 을 풀고 다시 기다린다."""
    q, logs, fake, env = _setup(tmp_path)
    env["RESTART_STOP_SETTLE_S"] = "2"
    old = _spawn_old(fake)
    proc = subprocess.Popen(["bash", str(SCRIPT), G], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(0.8)                                            # busy 통과 → STOP 생성 → 정착 대기 중
        assert (q / "STOP").exists()
        (q / "running" / "late.json").write_text(json.dumps({"name": "late", "assigned_gpu": int(G)}))
        time.sleep(2.5)
        assert proc.poll() is None and old.poll() is None and not (q / "STOP").exists()
        assert "풀고 다시 기다린다" in (logs / "dec.log").read_text()
        (q / "running" / "late.json").unlink()
        proc.wait(timeout=30)
        assert proc.returncode == 0
        old.wait(timeout=5)
    finally:
        _kill_new(q)
        if old.poll() is None:
            old.kill(); old.wait()
