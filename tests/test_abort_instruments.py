"""0913 계기 수리 회귀 시험 — 실제 프로세스/파일로 확인한다(문자열만 보는 시험은 못 잡는다).

1. 재시도 래퍼 둘은 rc 75(사전등록 중단)를 재시도하지 않고, 다른 rc 는 재시도한다.
2. parse_meta(form="math") 는 decision 줄이 없어도 신뢰도만 있으면 발화로 센다(new 는 불변).
3. gate_judgment 는 --arm-token 없이도 잡 cmd 문자열로 pid 를 재확인해 죽인다.
4. gpu_queue 워커는 rc 75 를 failed 가 아니라 aborted 로 기록한다.
5. math_sites 의 판단 심기 모드는 앞부분 뒤에 결정이 다른 메타를 잇는다.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))


def _run_wrapper(wrapper: str, rc: int, tmp_path: Path) -> tuple[int, str]:
    marker = tmp_path / "count"
    script = tmp_path / "job.sh"
    script.write_text(f"#!/bin/bash\necho x >> {marker}\nexit {rc}\n")
    script.chmod(0o755)
    if wrapper == "retry_cmd":
        cmd = ["bash", "scripts/local/retry_cmd.sh", "3", "0", "--", str(script)]
    else:
        # run_arm_retry 는 run_arm.sh 를 부른다 — 시험용으로 가짜 run_arm.sh 를 PATH 대신
        # 작업 디렉터리 복제본으로 준다.
        d = tmp_path / "repo" / "scripts" / "local"
        d.mkdir(parents=True)
        (d / "run_arm_retry.sh").write_text((REPO / "scripts/local/run_arm_retry.sh").read_text())
        (d / "run_arm.sh").write_text(f"#!/bin/bash\nexec {script}\n")
        cmd = ["bash", "scripts/local/run_arm_retry.sh", "3", "0", "--", "ARM"]
        r = subprocess.run(cmd, cwd=tmp_path / "repo", capture_output=True, text=True)
        return r.returncode, marker.read_text() if marker.exists() else ""
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return r.returncode, marker.read_text() if marker.exists() else ""


def test_retry_cmd_does_not_retry_rc75(tmp_path):
    rc, runs = _run_wrapper("retry_cmd", 75, tmp_path)
    assert rc == 75 and runs.count("x") == 1


def test_retry_cmd_retries_other_rc(tmp_path):
    rc, runs = _run_wrapper("retry_cmd", 1, tmp_path)
    assert rc == 1 and runs.count("x") == 3


def test_run_arm_retry_does_not_retry_rc75(tmp_path):
    rc, runs = _run_wrapper("run_arm_retry", 75, tmp_path)
    assert rc == 75 and runs.count("x") == 1


def test_run_arm_retry_retries_other_rc(tmp_path):
    rc, runs = _run_wrapper("run_arm_retry", 1, tmp_path)
    assert rc == 1 and runs.count("x") == 3


def test_parse_meta_math_form_decision_optional():
    from src.training import countdown_rewards as cdr
    txt = "work\n<meta>\nconfidence: 0.7\nThe substitution approach is fine.\n</meta>\nmore"
    assert cdr.parse_meta(txt, "new")["emitted"] == 0        # Countdown 계약 불변
    m = cdr.parse_meta(txt, "math")
    assert m["emitted"] == 1 and m["confidence"] == 0.7 and m["decision"] is None
    assert cdr.parse_meta("no block here", "math")["emitted"] == 0
    assert cdr.parse_meta("<meta>\njust words\n</meta>", "math")["emitted"] == 0   # 신뢰도 없음


def test_gate_kills_by_job_cmd_when_default_token_absent(tmp_path, monkeypatch):
    import gate_judgment as G
    monkeypatch.setattr(G, "QUEUE", tmp_path / "queue")
    (tmp_path / "queue" / "running").mkdir(parents=True)
    cmd = "sleep 30; : GATE_CMD_TOKEN_4242"
    proc = subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    try:
        time.sleep(0.3)
        jp = tmp_path / "queue" / "running" / "job.json"
        job = {"name": "some_job_name", "cmd": cmd, "pid": proc.pid}
        jp.write_text(json.dumps(job))
        status = G.kill_job_exact_pid(jp, job, "some_job_name")   # 기본 토큰 = 잡 이름
        assert status.startswith("KILLED"), status
        time.sleep(0.5)
        assert proc.poll() is not None
        assert (tmp_path / "queue" / "aborted" / "job.json").exists()
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


def test_gpu_queue_records_rc75_as_aborted(tmp_path, monkeypatch):
    import gpu_queue as Q
    monkeypatch.setattr(Q, "QUEUE_ROOT", tmp_path)
    for d in ("running", "done", "failed", "aborted", "logs"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(Q, "LOG_DIR", tmp_path / "logs", raising=False)
    jp = tmp_path / "running" / "j.json"
    jp.write_text(json.dumps({"name": "j", "cmd": "exit 75", "priority": 1, "need_mb": 1}))
    Q._run_one_job(0, jp, None)
    assert (tmp_path / "aborted" / "j.json").exists()
    assert not (tmp_path / "failed" / "j.json").exists()


def test_math_sites_seeded_modes():
    import math_sites as M
    pre = "Let x be the root.\n"
    v, r = M.build_fed("verify", pre, None), M.build_fed("redirect", pre, None)
    assert v.startswith(pre.rstrip()) and "decision: verify" in v
    assert r.startswith(pre.rstrip()) and "decision: redirect" in r
    assert v.replace(M.SEED_VERIFY, "") == r.replace(M.SEED_REDIRECT, "")
    assert set(M.ALL_MODES) == {"nometa", "meta", "donor", "verify", "redirect"}
