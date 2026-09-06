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
