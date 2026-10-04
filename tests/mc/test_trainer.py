"""mc.trainer — 접두 공유 트리 채점(= naive forward) · 워커 env 목록."""
from __future__ import annotations

import os

import pytest
import torch

from mc import trainer as T


def tiny_lm(vocab: int = 128, seed: int = 0):
    """CPU 작은 Qwen3(sdpa) — 트리 채점 대조용."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).eval()


def naive_score(model, ctx, tgt) -> float:
    """문맥+대상 한 시퀀스 forward — Σ log p(대상)."""
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.as_tensor([list(ctx) + list(tgt)])).logits[0].float(), -1)
    return sum(lp[len(ctx) + j - 1, t].item() for j, t in enumerate(tgt))


def test_tree_score_equals_naive_per_sequence():
    """접두 공유 트리 forward 한 번 = 자리·후보마다 따로 넣은 naive forward(자리 3 × 후보 3, 대상 길이 제각각)."""
    m, g = tiny_lm(), torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 128, (57,), generator=g).tolist()
    heads = [5, 6, 7]
    blocks = [(t, heads + tg, len(tg)) for t in (9, 30, 57) for tg in ([11], [12, 13, 14], [15, 16])]
    with torch.no_grad():
        got = T.tree_score(m, prefix, blocks)
    want = [naive_score(m, prefix[:t] + heads + b[:len(b) - n - len(heads)], b[len(b) - n:])
            for t, b, n in blocks]
    assert got == pytest.approx(want, abs=1e-4)


def test_tree_inputs_mask_and_padding():
    ids, pos, ok, keep, lab, own = T.tree_inputs([1, 2, 3, 4], [(2, [7, 8, 9], 2)], align=16)
    assert ids.shape[0] == 16 and ids[:7].tolist() == [1, 2, 3, 4, 7, 8, 9]
    assert pos[:7].tolist() == [0, 1, 2, 3, 2, 3, 4]                 # 블록 위치는 자리 t 부터
    assert ok[4].tolist()[:7] == [True, True, False, False, True, False, False]   # 접두[:2] + 자기
    assert not ok[6, 2:4].any() and ok[6, 4:7].all()                  # 접두[2:] 는 못 본다
    assert keep == [4, 5] and lab == [8, 9] and own == [0, 0]
    assert ok[7:, :7].sum() == 0 and ok[7:, 7:].equal(torch.eye(9, dtype=torch.bool))   # 채움은 자기만


def test_worker_env_keys_cover_every_live_knob():
    """Ray 워커는 드라이버 env 를 상속하지 않는다 — 손잡이를 빼먹으면 팔이 조용히 대조군이
    된다(이 저장소에 두 번 있던 사고). SPONT_PFX 만 남은 뒤(0925)의 live 손잡이만 남는다."""
    for k in ("LABEL", "MC_CKPT_DIR", "PROMPT_VARIANT", "MC_DUMP_ADV", "OUTCOME_MODE", "PFX_WEIGHT_KEY", "PFX_TRUNC",
              "PFX_FORK", "PFX_REP", "PFX_BREAK_W"):
        assert k in T.WORKER_ENV_KEYS
    for k in ("W_PMI", "W_PMI2", "SCORE_ALPHA_POS", "SCORE_ALPHA_NEG", "DECISION_TOKENS", "ZONE_SCALE",
              "BETA_FIRST", "OPT_GRADE", "W_DISTILL", "MC_TEACHER_PATH", "T2_LEN", "PFX_CHECK_COST", "PFX_RIGHT_SHARE"):
        assert k not in T.WORKER_ENV_KEYS


def test_ray_env_only_carries_set_vars(monkeypatch):
    monkeypatch.delenv("PFX_WEIGHT_KEY", raising=False)
    assert "PFX_WEIGHT_KEY" not in T._ray_env()          # 미설정 ≠ 빈 문자열
    monkeypatch.setenv("PFX_WEIGHT_KEY", "sample_weight")
    assert T._ray_env()["PFX_WEIGHT_KEY"] == "sample_weight"
    monkeypatch.delenv("MC_ZMQ_TAG", raising=False)                  # 0928: 잡마다 다른 vLLM 가중치 소켓 꼬리표
    assert T._ray_env()["MC_ZMQ_TAG"] == f"-mc{__import__('os').getpid()}"


def test_compute_reward_colocate_fallback_returns_rm_scores_shape():
    """실제 경로에서는 이 오버라이드가 호출되지 않는다(rm_scores 가 agent-loop 에서 먼저
    채워진다) — 여기서는 zero-fallback 안전망 자체의 모양만 고정한다."""
    import pytest
    pytest.importorskip("verl")
    MCTrainer = T.build_trainer_cls()

    class _B:
        batch = {"responses": torch.zeros(3, 5, dtype=torch.long)}
    out = MCTrainer._compute_reward_colocate(object.__new__(MCTrainer), _B())
    assert out.batch["rm_scores"].shape == (3, 5)


def test_abort_exit_code_is_75():
    assert T.ABORT_EXIT_CODE == 75


def test_ray_init_kwargs_pin_num_cpus_and_timeouts():
    """2026-09-21 사고: num_cpus 미지정 → Ray 가 124코어만큼 워커를 선기동 → sitecustomize 가
    워커마다 NFS 에서 torch/verl 을 import → 116개가 등록 타임아웃(60s)에 걸려 죽고 main_task
    가 영원히 스케줄되지 않았다. 세 겹의 방어를 코드로 고정한다."""
    kw = T.ray_init_kwargs()
    assert kw["num_cpus"] == 16                       # 기본 agent.num_workers=8 + 여유
    sc = kw["_system_config"]
    assert sc["worker_register_timeout_seconds"] == 600
    assert sc["agent_register_timeout_ms"] == 600000
    assert kw["include_dashboard"] is False
    assert T.PRESTART_OFF == {"RAY_enable_worker_prestart": "0",
                              "RAY_prestart_worker_first_driver": "0"}


def test_ray_init_num_cpus_follows_agent_workers():
    class _C:
        class actor_rollout_ref:
            class rollout:
                class agent:
                    num_workers = 32
    assert T.ray_init_kwargs(_C())["num_cpus"] == 40


def test_run_sh_exports_prestart_off():
    from pathlib import Path
    txt = (Path(__file__).resolve().parents[2] / "mc" / "run.sh").read_text()
    assert "export RAY_enable_worker_prestart=0" in txt
    assert "export RAY_prestart_worker_first_driver=0" in txt


def test_run_sh_never_selects_bitsandbytes_even_with_slim_paged(tmp_path):
    """2026-09-21 연기시험 실패: 잡 cmd 의 `PAGED=1 SLIM=1` 이 옛 런처 관례대로 8비트
    옵티마이저를 골라 `build_optimizer` 에서 `ImportError: bitsandbytes.optim` 으로 죽었다
    (verl09 env 에 그 패키지가 없고 env 는 바꿀 수 없다). 손잡이를 없앴음을 고정한다."""
    import os
    import re
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, SLIM="1", PAGED="1", CKPT_DIR=str(tmp_path / "ckpt"))   # 실제 체크포인트 자리에 mkdir 금지
    out = subprocess.run(["bash", "mc/run.sh", "SPONT_PFX", "2", "3", "--dry-run"],
                        cwd=root, env=env, capture_output=True, text=True).stdout
    cmd = out.split("exact train command:")[1]
    assert "bitsandbytes" not in cmd and "8bit" not in cmd.lower()
    assert re.search(r"optim\.optimizer_impl=torch\.optim", cmd)
    assert re.search(r"optim\.optimizer=AdamW\b", cmd)
    # 8비트를 못 쓰므로 메모리는 이 오프로드가 맡는다 — 항상 켜져 있어야 한다.
    assert "actor.fsdp_config.optimizer_offload=true" in cmd


def test_check_prompt_lengths_dies_on_overlong(tmp_path):
    """verl 은 `max_prompt_length` 를 넘는 행을 **조용히 버린다**(overlong 필터) — 학습 풀의
    프롬프트가 max 1,891 · p99 1,079 인데 판을 1024 로 두면 가장 어려운 문제만 표본에서
    사라진다(0921 측정 오류). 조용한 필터 대신 발사 자리에서 죽어야 한다."""
    import pandas as pd
    import pytest

    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return [0] * len(s)

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False):
            return "".join(m["content"] for m in msgs)

    from mc.context import SOLVE_MATH_OPT
    f = tmp_path / "train.parquet"
    probs = ["q", "qq", "q" * 5000]
    # 프롬프트는 parquet 의 `prompt` 컬럼에 구워져 있다 — check_prompt_lengths 가
    # PROMPT_VARIANT 와 대조한다(어긋나면 팔이 조용히 대조군이 된다).
    pr = [[{"role": "system", "content": SOLVE_MATH_OPT}, {"role": "user", "content": q}]
          for q in probs]
    pd.DataFrame({"problem": probs, "prompt": pr}).to_parquet(f, index=False)
    with pytest.raises(ValueError) as ei:
        T.check_prompt_lengths(_Tok(), str(f), 2048)
    assert "조용히 버리" in str(ei.value) and "MAX_PROMPT" in str(ei.value)
    # 충분히 넓으면 통과하고 통계를 돌려준다
    stat = T.check_prompt_lengths(_Tok(), str(f), 99999)
    assert stat["n"] == 3 and stat["n_over"] == 0 and stat["max"] > 5000
    assert stat["p99"] >= stat["max"] or stat["p99"] > 0
    # 손잡이와 데이터가 어긋나면 fail-closed
    os.environ["PROMPT_VARIANT"] = "plain"
    try:
        with pytest.raises(ValueError) as ei2:
            T.check_prompt_lengths(_Tok(), str(f), 99999)
        assert "PROMPT_VARIANT=plain" in str(ei2.value)
    finally:
        os.environ.pop("PROMPT_VARIANT")


def test_fill_verl_defaults_fills_missing_rollout_keys():
    """verl 0.9 는 rollout.disaggregation 같은 키를 raw DictConfig 에서 직접 읽는데
    우리 config 는 `defaults:` 없는 단독 yaml 이라 이 키들이 없다(ConfigAttributeError,
    LLMServerManager.__init__). fill_verl_defaults 가 verl 완전판 기본값으로 채우되
    우리 값은 그대로 살아남아야 한다."""
    import pytest
    pytest.importorskip("verl")
    from pathlib import Path
    from omegaconf import OmegaConf
    from mc import trainer as T

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "configs" / "countdown_6arm.yaml")
    merged = T.fill_verl_defaults(cfg)

    # verl 이 raw 로 읽는, 우리 yaml 에 없던 키가 채워져야 한다
    assert merged.actor_rollout_ref.rollout.disaggregation.enabled is False
    # 우리 값은 병합 뒤에도 살아남아야 한다(interpolation 은 resolve 하지 않는다)
    assert merged.actor_rollout_ref.rollout.mode == "async"
    assert merged.actor_rollout_ref.rollout.name == "vllm"
    assert merged.algorithm.adv_estimator == cfg.algorithm.adv_estimator


def test_ref_worker_pickles_by_reference():
    """0924 사고: 함수 안에서 만든 워커 클래스는 cloudpickle 이 값으로 실어 verl 전역(transfer_queue)에서 즉사했다."""
    import ray
    from ray import cloudpickle

    from mc.ref_worker import MCRefWorker
    cloudpickle.dumps(ray.remote(MCRefWorker).__ray_metadata__.modified_class)
    assert "mc.ref_worker" in str(cloudpickle.dumps(MCRefWorker))[:200]      # 참조(모듈·이름)로 실린다


def test_no_gpu_visible_in_tests():
    assert not torch.cuda.is_available()                              # conftest 가 GPU 를 가린다


def test_tree_score_per_token_sums_to_block_score():
    """수정 28: per_token 이면 블록마다 대상 토큰별 log p — 합 = 기본(블록 합) 점수, 길이 = 대상 수."""
    m, g = tiny_lm(), torch.Generator().manual_seed(2)
    prefix = torch.randint(1, 128, (40,), generator=g).tolist()
    blocks = [(39, [prefix[38]] + [11, 12, 13], 3), (20, [5, 6, 7, 8, 9], 4)]
    with torch.no_grad():
        tot, per = T.tree_score(m, prefix, blocks), T.tree_score(m, prefix, blocks, per_token=True)
    assert [len(v) for v in per] == [3, 4] and [sum(v) for v in per] == pytest.approx(tot, abs=1e-6)


def test_run_sh_pfx_lineage_suffixes(tmp_path):
    """resume 안전: 변형마다 계보(체크포인트 자리)가 다르다 — PFX-B · R2 검산 비용 · CH-Fork/HSD."""
    import os
    import re
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = dict(os.environ, PROMPT_VARIANT="plain", LR_SCHED="constant", LABEL="gold", TAG="p1333sw",
                CKPT_DIR=str(tmp_path / "ckpt"))
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC"):
        base.pop(k, None)

    def lineage(**kw):
        out = subprocess.run(["bash", "mc/run.sh", "SPONT_PFX", "2", "20", "--dry-run"], cwd=root,
                             env={**base, **kw}, capture_output=True, text=True).stdout
        return re.search(r"LINEAGE=(\S+)", out).group(1)
    assert lineage() == "mc_SPONT_PFX_gold_s2_r4096_p1333sw"
    assert lineage(PFX_FORK="ch") == "mc_SPONT_PFX_gold_s2_r4096_forkch_p1333sw"
    assert lineage(PFX_FORK="hsd") == "mc_SPONT_PFX_gold_s2_r4096_forkhsd_p1333sw"
    assert lineage(PFX_TRUNC="mask", PFX_FORK="ch", PFX_BREAK_W="2", PFX_REP="0.25") == "mc_SPONT_PFX_gold_s2_r4096_forkch_tm_bw2_rp0.25_p1333sw"
    assert lineage(PFX_TRUNC="mask", PFX_FORK="chdir", PFX_BREAK_W="2", PFX_REP="0.25", PFX_REP_HARD="1") == \
        "mc_SPONT_PFX_gold_s2_r4096_forkchdir_tm_bw2_rp0.25h_p1333sw"
    assert lineage(PFX_FORK="ch", PFX_REP="0.25", PFX_ADV_CAP="4") == "mc_SPONT_PFX_gold_s2_r4096_forkch_rp0.25_ac4_p1333sw"
    assert lineage(PFX_REP="0.25", PFX_REP_HARD="1", PFX_DISTILL="0.25") == "mc_SPONT_PFX_gold_s2_r4096_rp0.25h_ds0.25_p1333sw"
    assert lineage(PFX_TRUNC="mask", PFX_FORK="ch") == "mc_SPONT_PFX_gold_s2_r4096_forkch_tm_p1333sw"   # 수정 32
    assert lineage(PFX_TRUNC="keep") == "mc_SPONT_PFX_gold_s2_r4096_p1333sw"                # 기본값 = 옛 계보


def test_run_sh_refuses_resume_with_changed_env(tmp_path):
    """0926: 계보 이름에 없는 손잡이(LR 등)가 바뀌면 같은 체크포인트를 잇지 않는다(RUN_ENV.txt 대조, 드라이런도)."""
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ck = tmp_path / "ckpt"
    env = dict(os.environ, PROMPT_VARIANT="plain", LR_SCHED="constant", LABEL="gold", LR="5e-6", CKPT_DIR=str(ck))
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W"):
        env.pop(k, None)
    run = lambda e: subprocess.run(["bash", "mc/run.sh", "SPONT_PFX", "2", "3", "--dry-run"], cwd=root, env=e,  # noqa: E731
                                   capture_output=True, text=True)
    assert run(env).returncode == 0 and not (ck / "RUN_ENV.txt").exists()            # 드라이런은 쓰지 않는다
    (ck / "RUN_ENV.txt").write_text("LR=1e-6 DATA_TRAIN=x")
    bad = run(env)
    assert bad.returncode == 1 and "FATAL" in bad.stderr
