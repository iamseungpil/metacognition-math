"""RESP_LEN knob (0906, OPT_MT-L) 회귀 테스트 — scripts/local/run_arm.sh --dry-run.

OPT_MT 계열이 메모 뒤 재시도로 응답의 31~38% 를 잘리는 것(docs/RESULTS_cd7.md
OPT_MT-L 절)이 held-out 손실의 원인인지 직접 검사하려면 응답 예산만 키운 짝
실험이 필요하다. RESP_LEN 은 그 예산 knob 이다 — 기본값(2048)에서는 mixed 분기의
기존 하드코딩(max_response_length=2048, max_model_len/max_num_batched_tokens=4352)과
바이트 동일해야 하고, 값을 바꾸면 그 세 숫자가 같이 커지며 LINEAGE 에 `_r{N}`
접미사가 붙어 체크포인트/merged/eval/logs 가 2048 계보와 충돌하지 않아야 한다.

GPU 를 쓰지 않는다 — `--dry-run` 은 학습 커맨드를 조립해 stdout 에 찍고
종료한다(run_arm.sh:usage 및 DRY_RUN 분기 참조).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ARM = ROOT / "scripts" / "local" / "run_arm.sh"


def _dry_run(env_extra: dict, args: list[str]) -> str:
    import os

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env_extra)
    r = subprocess.run(
        ["bash", str(RUN_ARM), *args, "--dry-run"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"run_arm.sh --dry-run failed: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def test_resp_len_explicit_appends_suffix_and_scales_budget():
    out = _dry_run(
        {"MIXED_DATA": "mixed_train_v3c", "RESP_LEN": "3072"},
        ["OPT_MT", "1", "100", "opt"],
    )
    assert "LINEAGE=cd7_OPT_MT_opt_s1_mixed_r3072" in out
    assert "data.max_response_length=3072" in out
    assert "actor_rollout_ref.rollout.max_model_len=5376" in out
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=5376" in out


def test_resp_len_default_matches_old_hardcoded_behaviour():
    out = _dry_run(
        {"MIXED_DATA": "mixed_train_v3c"},
        ["OPT_MT", "1", "100", "opt"],
    )
    assert "LINEAGE=cd7_OPT_MT_opt_s1_mixed ARM=" in out
    # LINEAGE 는 접미사 없이 그대로여야 한다 (즉 "_mixed_r" 가 나타나면 실패).
    assert "_mixed_r" not in out
    assert "data.max_response_length=2048" in out
    assert "actor_rollout_ref.rollout.max_model_len=4352" in out
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=4352" in out
