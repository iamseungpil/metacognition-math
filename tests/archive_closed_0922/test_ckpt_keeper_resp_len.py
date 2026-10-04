"""ckpt_keeper _LIN / running_lineages RESP_LEN 접미사 회귀 테스트 (0906, OPT_MT-L).

run_arm.sh 가 응답 예산을 키운 계보에 `_r{N}` 접미사를 붙이게 되면서(_mixed 뒤,
기본값 2048 이 아닐 때만) ckpt_keeper.py 쪽도 그 접미사를 알아야 한다 —
1. `_LIN` 정규식이 그 접미사를 파싱해 group "resp" 로 내놓아야 `submit_eval`이
   eval/이어쓰기 예산을 학습 예산에 맞출 수 있다.
2. `running_lineages()`는 큐의 cmd 문자열에서 `RESP_LEN=NNNN`을 읽어 같은
   접미사를 재현해야 한다 — 못 하면 keeper 가 "실행 중" 계보를 못 알아보고
   run_arm.sh 와 step 100 병합을 경합한다(모듈 docstring §step 100 참조).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "local"))

import ckpt_keeper as ck  # noqa: E402


def test_lin_regex_no_suffix():
    m = ck._LIN.match("cd7_N0_plain_s1")
    assert m is not None
    assert m.group("arm") == "N0"
    assert m.group("variant") == "plain"
    assert m.group("seed") == "1"
    assert m.group("mixed") is None
    assert m.group("resp") is None


def test_lin_regex_mixed_only():
    m = ck._LIN.match("cd7_OPT_MT_opt_s1_mixed")
    assert m is not None
    assert m.group("mixed") == "_mixed"
    assert m.group("resp") is None


def test_lin_regex_mixed_and_resp_suffix():
    m = ck._LIN.match("cd7_OPT_MT_opt_s1_mixed_r3072")
    assert m is not None
    assert m.group("arm") == "OPT_MT"
    assert m.group("variant") == "opt"
    assert m.group("seed") == "1"
    assert m.group("mixed") == "_mixed"
    assert m.group("resp") == "3072"


def test_running_lineages_reproduces_resp_len_suffix(tmp_path, monkeypatch):
    running_dir = tmp_path / "running"
    running_dir.mkdir()
    cmd = (
        "MIXED_DATA=mixed_train_v3c RESP_LEN=3072 bash scripts/local/run_arm.sh "
        "OPT_MT 1 100 opt"
    )
    (running_dir / "job1.json").write_text(json.dumps({"cmd": cmd}))

    monkeypatch.setattr(ck, "QUEUE", tmp_path)
    lineages = ck.running_lineages()
    assert lineages == {"cd7_OPT_MT_opt_s1_mixed_r3072"}


def test_running_lineages_default_no_resp_suffix(tmp_path, monkeypatch):
    running_dir = tmp_path / "running"
    running_dir.mkdir()
    cmd = "MIXED_DATA=mixed_train_v3c bash scripts/local/run_arm.sh OPT_MT 1 100 opt"
    (running_dir / "job1.json").write_text(json.dumps({"cmd": cmd}))

    monkeypatch.setattr(ck, "QUEUE", tmp_path)
    lineages = ck.running_lineages()
    assert lineages == {"cd7_OPT_MT_opt_s1_mixed"}


def test_submit_eval_wraps_with_retry_and_uses_safer_gpu_util(tmp_path, monkeypatch):
    """0911: 09-10 새벽·저녁 두 번의 OOM 사고(gpu_util 0.6 이 공유 카드 여유보다 큼,
    단발 제출이라 재시도 없이 조용히 죽음)를 막는 수리. submit_eval 이 만드는 명령이
    retry_cmd.sh 로 감싸져 있고 gpu_util 0.6 을 더는 안 쓰는지 확인한다."""
    monkeypatch.setattr(ck, "WORK", tmp_path)
    (tmp_path / "merged" / "cd7_N0_plain_s1" / "step_30").mkdir(parents=True)
    captured = []

    def fake_run(args, cwd=None, capture_output=None, text=None):
        captured.append(args[args.index("--cmd") + 1])
        class R: returncode = 0; stderr = ""
        return R()

    monkeypatch.setattr(ck.subprocess, "run", fake_run)
    ck.submit_eval("cd7_N0_plain_s1", 30, dry=False)
    # submit_eval 은 eval 잡과 judge-site 이어쓰기 잡을 둘 다 제출한다 — 둘 다 감싸져야 한다.
    eval_cmd = next(c for c in captured if "countdown_gs0_eval.py" in c)
    jsite_cmd = next(c for c in captured if "gen_continuations.py" in c)
    for cmd in (eval_cmd, jsite_cmd):
        assert "retry_cmd.sh" in cmd
        assert "gpu_util 0.6" not in cmd
        assert "gpu_util 0.45" not in cmd
    assert "gpu_util 0.4" in eval_cmd
    assert "gpu_util 0.4" in jsite_cmd


def test_submit_eval_cmd_expands_work_after_env_sh_not_before(tmp_path, monkeypatch):
    """0911 2차 수리: 큰따옴표로 감싼 최초 수정이 FIXED_CHK step100 평가를 4연속
    FileNotFoundError(`/data/...`, `$WORK` 없이)로 또 죽였다 — gpu_queue 워커가
    --cmd 문자열을 셸에 넘기는 **바깥** 시점에 큰따옴표 안의 `$WORK` 가 (env.sh 를
    소싱하기도 전에) 먼저 빈 문자열로 확장된 것. 작은따옴표로 재수리했고, 문자열
    내용이 아니라 **실제로 셸을 통과시켜** `$WORK` 가 안쪽 `bash -c` 에서(= env.sh
    소싱 뒤) 확장되는지를 실행으로 검증한다(CPU, GPU/네트워크 없음)."""
    import subprocess
    real_run = subprocess.run  # ck.subprocess.run 을 가짜로 바꾸기 **전에** 진짜를 저장해 둔다
    #   (subprocess 는 모듈 싱글턴이라 ck.subprocess.run 을 바꾸면 이 테스트 파일이 부른
    #   subprocess.run 도 같이 바뀐다 — 프로브 실행엔 진짜가 필요하다).
    monkeypatch.setattr(ck, "WORK", tmp_path)
    (tmp_path / "merged" / "cd7_N0_plain_s1" / "step_30").mkdir(parents=True)
    captured = []

    def fake_run(args, cwd=None, capture_output=None, text=None):
        captured.append(args[args.index("--cmd") + 1])
        class R: returncode = 0; stderr = ""
        return R()

    monkeypatch.setattr(ck.subprocess, "run", fake_run)
    ck.submit_eval("cd7_N0_plain_s1", 30, dry=False)
    eval_cmd = next(c for c in captured if "countdown_gs0_eval.py" in c)
    # 실제 python 호출 전, "$WORK 가 살아 있는가"만 떼어내 같은 감싸기로 실행해 본다.
    probe_cmd = eval_cmd.split("python scripts/countdown_gs0_eval.py")[0] + "echo WORK_IS:$WORK'"
    repo_root = Path(__file__).resolve().parents[1]
    r = real_run(probe_cmd, shell=True, cwd=str(repo_root), capture_output=True, text=True)
    assert "WORK_IS:/data" not in r.stdout, f"$WORK 이 바깥 셸에서 먼저 비워졌다: {r.stdout!r} {r.stderr[-200:]!r}"
    assert "WORK_IS:" in r.stdout and "WORK_IS:\n" not in r.stdout
