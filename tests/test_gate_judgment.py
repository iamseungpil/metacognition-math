"""gate_judgment.py 안전장치 회귀 시험 — 정확한 pid만 죽이고, 이름/문자열 grep으로
아무 pid나 잡지 않는다는 것을 실제 프로세스로 확인한다(문자열만 보는 시험은 이 프로젝트
과거 오폭 사고 계열 버그를 못 잡는다)."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
import gate_judgment as G  # noqa: E402


def _spawn(marker: str, seconds: int = 30) -> subprocess.Popen:
    """bash -c 로 띄운다 — /proc/<pid>/cmdline에 marker 문자열 자체가 그대로 들어가고
    (진짜 잡 커맨드처럼), 자기 세션(새 프로세스그룹)이라 killpg가 안전하다.
    `;`로 두 명령을 이어 bash의 exec-최적화(마지막 단일 명령이면 자기 프로세스 이미지를
    그 명령으로 바꿔치기해 argv에서 marker가 사라짐)를 피한다."""
    return subprocess.Popen(["bash", "-c", f"sleep {seconds}; : {marker}"], start_new_session=True)


def test_pid_cmdline_contains_matches_real_running_process():
    proc = _spawn("OUR_UNIQUE_TOKEN_XYZ_777")
    try:
        time.sleep(0.2)
        assert G.pid_cmdline_contains(proc.pid, "OUR_UNIQUE_TOKEN_XYZ_777") is True
        assert G.pid_cmdline_contains(proc.pid, "SOME_OTHER_TOKEN") is False
    finally:
        proc.kill()
        proc.wait()


def test_pid_cmdline_contains_false_for_dead_pid():
    """죽은/존재하지 않는 pid는 무조건 False — 재활용된 pid를 잘못 죽이는 것을 막는
    1차 방어선(존재하지 않으면 아예 매치 시도조차 안 함)."""
    assert G.pid_cmdline_contains(999999999, "anything") is False


def test_kill_job_exact_pid_skips_when_token_not_in_cmdline():
    """arm_token이 실제 pid의 cmdline과 안 맞으면 killpg를 호출하지 않고 SKIP만 반환한다
    — 이것이 '이름으로 grep해서 죽이기'와 다른 지점이다."""
    proc = _spawn("SOME_REAL_MARKER")
    try:
        time.sleep(0.2)
        job = {"pid": proc.pid, "name": "irrelevant"}
        status = G.kill_job_exact_pid(Path("/tmp/does_not_matter.json"), job, "TOKEN_NOT_PRESENT")
        assert status.startswith("SKIP")
        assert proc.poll() is None  # 여전히 살아있다 — 죽이지 않았다
    finally:
        proc.kill()
        proc.wait()


def test_kill_job_exact_pid_kills_only_when_token_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "QUEUE", tmp_path / "queue")
    proc = _spawn("MATCH_TOKEN_ABC")
    try:
        time.sleep(0.2)
        job_path = tmp_path / "queue" / "running" / "job.json"
        job_path.parent.mkdir(parents=True)
        job = {"pid": proc.pid, "name": "w_test"}
        job_path.write_text(json.dumps(job))
        status = G.kill_job_exact_pid(job_path, job, "MATCH_TOKEN_ABC")
        assert status.startswith("KILLED")
        proc.wait(timeout=10)
        assert proc.returncode is not None  # 실제로 죽었다
        assert (tmp_path / "queue" / "aborted" / "job.json").exists()
        assert not job_path.exists()  # running/ 에서 옮겨졌다
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_kill_job_exact_pid_skip_when_no_pid_recorded():
    job = {"name": "old_style_job_without_pid"}
    status = G.kill_job_exact_pid(Path("/tmp/x.json"), job, "anything")
    assert status.startswith("SKIP(no pid")


def test_read_acc_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "WORK", tmp_path)
    assert G.read_acc("cd7_NOPE_chk_s1", 30) is None


def test_read_acc_reads_top_level_acc_field(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "WORK", tmp_path)
    d = tmp_path / "eval" / "cd7_X_chk_s1" / "step_30"
    d.mkdir(parents=True)
    (d / "telemetry.json").write_text(json.dumps({"acc": 0.671, "n_rows": 4000}))
    assert G.read_acc("cd7_X_chk_s1", 30) == 0.671


def test_abort_file_triggers_on_abort_cmd(tmp_path, monkeypatch):
    """★감사 10: --abort-file 이 있으면 telemetry 를 기다리는 동안 ABORTED.txt 를 같이 보고,
    나타나면 --on-abort-cmd 를 실행한 뒤 0 으로 끝난다(잡은 트레이너가 rc 75 로 스스로 죽었다)."""
    monkeypatch.setattr(G, "WORK", tmp_path)
    monkeypatch.setattr(G, "QUEUE", tmp_path / "queue")
    monkeypatch.setattr(G, "LOG", tmp_path / "logs" / "dec.log")
    for d in ("pending", "running", "failed"):
        (tmp_path / "queue" / d).mkdir(parents=True)
    ab = tmp_path / "ck" / "ABORTED.txt"; ab.parent.mkdir()
    ab.write_text("[MATH][ABORT] arm=M_JUDGE step=12: ...")
    out = tmp_path / "next.txt"
    monkeypatch.setattr(sys, "argv", ["gate_judgment.py", "--lineage", "cd9_M_JUDGE_s1", "--job-name", "w_x",
                                      "--step", "30", "--min-acc", "0.5", "--poll-s", "1", "--timeout-s", "20",
                                      "--abort-file", str(ab), "--on-abort-cmd", f"echo next > {out}"])
    assert G.main() == 0
    assert out.read_text().strip() == "next"
    assert "ABORTED.txt" in (tmp_path / "logs" / "dec.log").read_text()
