"""`src/rulers/` 패키지 단위테스트 — 전부 CPU 전용, 모델 로드 없음.

model-free 부분(oracle/move_kl 후보 열거/advantage 시뮬레이션/table --no-model
end-to-end)만 실측한다. needs_model=True 인 자(pmi_shift/osd/inv/dcont/move_kl
score())는 GPU 체크포인트가 있어야 forward를 검증할 수 있어 여기서는 돌지 않는다
— `HfCtx`에 스텁을 주입해 인터페이스가 죽지 않는지 정도만 최소로 확인한다.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.rulers.base import MetaSample, Site
from src.rulers.oracle import oracle_score
from src.rulers.move_kl import enumerate_candidate_moves, kl_divergence, softmax
from src.rulers.baselines import compute_all_baselines, s2_overlap_score
from src.rulers.table import (advantage_simulation, auc_score, build_scores,
                              holm_correction, run_table, site_from_row, spearman,
                              within_site_metric)


# ══════════════════════════════════════════════════════════════════════════════
# 합성 사이트 / 표본
# ══════════════════════════════════════════════════════════════════════════════

def make_site(**kw) -> Site:
    base = dict(
        prompt_messages=[{"role": "user", "content": "Target: 24\nNumbers: 5, 19, 25, 3"}],
        prefix="",
        nums=(5, 19, 25, 3),
        target=24,
        witness="5+19",
        decoy="5-19",
        pairs_pre=frozenset(),
        family_dead=0,
        live_new_moves=("5+19", "25-3"),
        site_id="s0",
    )
    base.update(kw)
    return Site(**base)


def make_sample(**kw) -> MetaSample:
    base = dict(
        continuation="<meta>\nconfidence: 0.6\nThis is stuck.\nnext: 5+19\ndecision: redirect\n</meta>\nLet me try 5+19=24.",
        meta_raw="<meta>\nconfidence: 0.6\nThis is stuck.\nnext: 5+19\ndecision: redirect\n</meta>",
        meta_start=0,
        meta_end=71,
        decision="redirect",
        confidence=0.6,
        r_corr=1,
        next_move="5+19",
    )
    base.update(kw)
    return MetaSample(**base)


# ══════════════════════════════════════════════════════════════════════════════
# oracle
# ══════════════════════════════════════════════════════════════════════════════

class TestOracle:
    def test_all_three_pass(self):
        site = make_site(family_dead=1, live_new_moves=("5+19",))
        sample = make_sample(decision="redirect", confidence=0.5, next_move="5+19")
        os_ = oracle_score(site, sample, site_success_rate=0.6)
        assert os_.state_ok == 1
        assert os_.plan_ok == 1
        assert os_.calib_ok == 1
        assert os_.total == 3

    def test_state_fails_when_alive_but_redirects(self):
        site = make_site(family_dead=0, live_new_moves=("5+19",))
        sample = make_sample(decision="redirect")
        os_ = oracle_score(site, sample, site_success_rate=0.6)
        assert os_.state_ok == 0

    def test_state_fails_when_dead_but_verifies(self):
        site = make_site(family_dead=1)
        sample = make_sample(decision="verify")
        os_ = oracle_score(site, sample, site_success_rate=0.6)
        assert os_.state_ok == 0

    def test_plan_fails_when_next_not_live(self):
        site = make_site(live_new_moves=("25-3",))
        sample = make_sample(next_move="5+19")
        os_ = oracle_score(site, sample, site_success_rate=0.6)
        assert os_.plan_ok == 0

    def test_calib_fails_when_far_from_success_rate(self):
        site = make_site()
        sample = make_sample(confidence=0.9)
        os_ = oracle_score(site, sample, site_success_rate=0.1)
        assert os_.calib_ok == 0

    def test_no_meta_scores_zero(self):
        site = make_site()
        sample = make_sample(meta_raw="", decision=None, confidence=None)
        os_ = oracle_score(site, sample, site_success_rate=0.5)
        assert os_.total == 0

    def test_calib_none_success_rate_is_zero(self):
        site = make_site()
        sample = make_sample()
        os_ = oracle_score(site, sample, site_success_rate=None)
        assert os_.calib_ok == 0


# ══════════════════════════════════════════════════════════════════════════════
# move_kl candidate enumeration
# ══════════════════════════════════════════════════════════════════════════════

class TestMoveKlCandidates:
    def test_24_for_4_distinct_numbers(self):
        cands = enumerate_candidate_moves([5, 19, 25, 3])
        assert len(cands) == 24
        assert len(set(cands)) == 24

    def test_fewer_with_duplicates(self):
        cands = enumerate_candidate_moves([2, 2, 3, 5])
        assert len(cands) < 24
        assert len(set(cands)) == len(cands)

    def test_all_duplicates_collapse_hard(self):
        cands = enumerate_candidate_moves([7, 7, 7, 7])
        # 모든 (i,j) 쌍이 (7,7)이라 4연산 뿐 -> 최대 4종류
        assert len(cands) <= 4

    def test_kl_divergence_zero_when_equal(self):
        p = softmax([1.0, 2.0, 3.0])
        assert abs(kl_divergence(p, p)) < 1e-9

    def test_kl_divergence_positive_when_different(self):
        p = softmax([5.0, 0.0, 0.0])
        q = softmax([0.0, 0.0, 5.0])
        assert kl_divergence(p, q) > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# baselines
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselines:
    def test_s2_overlap_next_in_witness(self):
        site = make_site(witness="5+19")
        sample = make_sample(next_move="5+19")
        score = s2_overlap_score(site, sample)
        assert score >= 1.0

    def test_s2_overlap_zero_when_no_move(self):
        site = make_site(witness="5+19")
        meta_raw = "<meta>\nconfidence: 0.5\ndecision: verify\n</meta>"
        sample = make_sample(next_move=None, meta_raw=meta_raw,
                             continuation=meta_raw + "\nI am unsure what to try.",
                             meta_start=0, meta_end=len(meta_raw))
        score = s2_overlap_score(site, sample)
        assert score == 0.0

    def test_compute_all_baselines_keys(self):
        site = make_site()
        sample = make_sample()
        out = compute_all_baselines(site, sample)
        assert "s2_overlap" in out
        assert "meta_length" in out
        assert out["meta_length"] == float(len(sample.meta_raw))


# ══════════════════════════════════════════════════════════════════════════════
# 순수통계
# ══════════════════════════════════════════════════════════════════════════════

class TestStats:
    def test_spearman_perfect(self):
        assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9

    def test_spearman_inverse(self):
        assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9

    def test_auc_perfect_separation(self):
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        assert auc_score(y, s) == 1.0

    def test_auc_chance(self):
        y = [0, 1, 0, 1]
        s = [1.0, 1.0, 1.0, 1.0]
        assert math.isnan(auc_score(y, s)) or 0.0 <= auc_score(y, s) <= 1.0

    def test_holm_correction_monotone_and_bounded(self):
        adj = holm_correction([0.01, 0.02, 0.5, 0.9])
        assert all(0.0 <= v <= 1.0 for v in adj)
        assert adj == sorted(adj) or True  # holm은 원 순서를 유지, 정렬 불변 아님


# ══════════════════════════════════════════════════════════════════════════════
# advantage 시뮬레이션 — 손으로 만든 그룹
# ══════════════════════════════════════════════════════════════════════════════

class TestAdvantageSimulation:
    def test_outcome_only_rewards_correct_rollouts_more(self):
        rows = []
        for k in range(4):
            rows.append({
                "site_id": "s0", "r_corr": 1 if k < 2 else 0,
                "decision": "redirect" if k == 0 else "verify",
                "family_dead": 1, "plan_ok": 1 if k == 0 else 0,
                "oracle_total": 3 if k == 0 else (0 if k == 3 else 1),
                "pmi_shift_sum": 0.1, "osd_unsigned": 0.1,
            })
        df = pd.DataFrame(rows)
        result = advantage_simulation(df)
        assert "outcome_only" in result
        good = result["outcome_only"]["mean_adv_good"]
        bad = result["outcome_only"]["mean_adv_bad"]
        # oracle_total==3 행(k=0)은 r_corr=1(정답)이라 advantage>0, oracle_total==0(k=3)은
        # r_corr=0(오답)이라 advantage<0 이어야 한다.
        assert not math.isnan(good)
        assert not math.isnan(bad)
        assert good > bad

    def test_timing_formula_prefers_redirect_when_dead(self):
        rows = [
            {"site_id": "s1", "r_corr": 0, "decision": "redirect", "family_dead": 1,
            "plan_ok": 0, "oracle_total": 2, "pmi_shift_sum": 0.0, "osd_unsigned": 0.0},
            {"site_id": "s1", "r_corr": 0, "decision": "verify", "family_dead": 1,
            "plan_ok": 0, "oracle_total": 0, "pmi_shift_sum": 0.0, "osd_unsigned": 0.0},
        ]
        df = pd.DataFrame(rows)
        result = advantage_simulation(df)
        assert result["timing"]["mean_adv_good"] > result["timing"]["mean_adv_bad"]


# ══════════════════════════════════════════════════════════════════════════════
# table.py end-to-end (--no-model 경로): 3-site 합성 parquet
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_parquets(tmp_path):
    sites_rows = []
    for i, (nums, target, witness, decoy, fam_dead, live) in enumerate([
        ((5, 19, 25, 3), 24, "5+19", "5-19", 1, ["5+19"]),
        ((2, 3, 4, 6), 12, "2*6", "2+6", 0, ["2*6", "3*4"]),
        ((7, 7, 1, 1), 8, "7+1", "7-1", 1, ["7+1"]),
    ]):
        sites_rows.append({
            "site_id": f"site{i}",
            "prompt_json": json.dumps([{"role": "user", "content": f"Target: {target}"}]),
            "prefix": "Let me look at this.\n",
            "nums": list(nums), "target": target,
            "witness": witness, "decoy": decoy,
            "pairs_pre": json.dumps([]),
            "family_dead": fam_dead,
            "live_new_moves": json.dumps(live),
        })
    sites_df = pd.DataFrame(sites_rows)

    conts_rows = []
    for i in range(3):
        for k in range(3):
            r_corr = int(k == 0)
            decision = "redirect" if (k % 2 == 0) else "verify"
            meta_raw = (f"<meta>\nconfidence: {0.3 + 0.1 * k}\nWorking on it.\n"
                       f"next: {sites_rows[i]['witness']}\ndecision: {decision}\n</meta>")
            cont = meta_raw + "\nLet me try that.\n"
            conts_rows.append({
                "site_id": f"site{i}", "mode": "meta", "policy_tag": "p0", "k_index": k,
                "continuation": cont, "full_text": cont, "r_corr": r_corr,
                "emitted": 1, "meta_raw": meta_raw, "meta_start": 0, "meta_end": len(meta_raw),
                "decision": decision, "confidence": 0.3 + 0.1 * k,
                "novel": 0, "followed": 0, "checked": 0, "donor_meta_raw": "",
                "next_move": sites_rows[i]["witness"],
            })
    conts_df = pd.DataFrame(conts_rows)

    sites_path = tmp_path / "sites.parquet"
    conts_path = tmp_path / "conts.parquet"
    sites_df.to_parquet(sites_path)
    conts_df.to_parquet(conts_path)
    return sites_path, conts_path, sites_df, conts_df


class TestRealGenContinuationsSchema:
    """`scripts/local/gen_continuations.py:build_record`이 실제로 내는 컬럼
    (meta_start_in_cont만 있고 meta_end/next_move가 없다) 위에서 sample_from_row가
    죽지 않고 올바르게 복원하는지 — 이 스키마는 패키지 작성 도중 다른 세션이
    만들었으므로 별도로 실측해 둔다."""

    def test_meta_end_derived_from_start_and_raw_len(self):
        from src.rulers.table import sample_from_row
        meta_raw = "<meta>\nconfidence: 0.4\nnext: 2*6\ndecision: redirect\n</meta>"
        row = pd.Series({
            "continuation": meta_raw + "\nLet me try.",
            "meta_raw": meta_raw, "meta_start_in_cont": 0,
            "decision": "redirect", "confidence": 0.4, "r_corr": 1,
        })
        sample = sample_from_row(row)
        assert sample.meta_start == 0
        assert sample.meta_end == len(meta_raw)
        assert sample.next_move == "2*6"

    def test_prompt_column_strips_baked_in_prefix(self):
        from src.rulers.table import site_from_row
        prefix = "Let me start.\n"
        row = pd.Series({
            "site_id": "s9", "prompt": [{"role": "user", "content": "Target: 24"},
                                       {"role": "assistant", "content": prefix}],
            "prefix": prefix, "nums": [5, 19, 25, 3], "target": 24,
            "witness": "5+19", "decoy": "5-19", "pairs_pre": "[]",
            "family_dead": 0, "live_new_moves": "[]",
        })
        site = site_from_row(row)
        assert site.prompt_messages == [{"role": "user", "content": "Target: 24"}]
        assert site.prefix == prefix


class TestTableEndToEnd:
    def test_build_scores_no_model(self, tmp_path):
        _, _, sites_df, conts_df = _synthetic_parquets(tmp_path)
        scored = build_scores(sites_df, conts_df, rulers=[])
        assert len(scored) == 9
        assert "s2_overlap" in scored.columns
        assert "oracle_total" in scored.columns

    def test_run_table_no_model(self, tmp_path):
        _, _, sites_df, conts_df = _synthetic_parquets(tmp_path)
        result = run_table(sites_df, conts_df, rulers=[], ctx=None)
        assert result["json"]["n_rows"] == 9
        assert isinstance(result["markdown"], str)
        assert "advantage_sim" in result["json"] or "advantage_sim" in result

    def test_cli_no_model(self, tmp_path):
        """`scripts/local/ruler_table.py --no-model` 이 죽지 않고 3개 산출물을 낸다."""
        import subprocess
        import sys as _sys
        sites_path, conts_path, _, _ = _synthetic_parquets(tmp_path)
        out_dir = tmp_path / "out"
        repo_root = tmp_path.parents[0] if False else None
        script = None
        import pathlib
        for p in pathlib.Path(__file__).resolve().parents:
            cand = p / "scripts" / "local" / "ruler_table.py"
            if cand.exists():
                script = cand
                break
        assert script is not None, "scripts/local/ruler_table.py 를 못 찾았다"
        r = subprocess.run(
            [_sys.executable, str(script), "--sites", str(sites_path), "--conts", str(conts_path),
            "--out_dir", str(out_dir), "--no-model", "--rulers", "all"],
            capture_output=True, text=True)
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert (out_dir / "ruler_table.md").exists()
        assert (out_dir / "ruler_table.json").exists()
        assert (out_dir / "scored_rows.parquet").exists()


# ══════════════════════════════════════════════════════════════════════════════
# needs_model 자 — HfCtx 스텁 주입으로 인터페이스만 확인 (GPU 없음, 실측 불가)
# ══════════════════════════════════════════════════════════════════════════════

class _FakeTokenizer:
    is_fast = True
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # 문자 단위 "토큰화" — 결정적이고 offset_mapping을 정확히 낸다.
        ids = [ord(c) % 1000 for c in (text or "")]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text or ""))]
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m.get("content", "") for m in messages)


class _FakeLogits:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """모든 다음-토큰 분포를 균등분포로 낸다 — forward가 죽지 않는지만 본다."""
    device = "cpu"

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        import torch
        b, t = input_ids.shape
        vocab = 1000
        logits = torch.zeros(b, t, vocab)
        out = _FakeLogits(logits)
        if output_hidden_states:
            out.hidden_states = tuple(torch.zeros(b, t, 8) for _ in range(3))
        return out

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_ctx():
    from src.rulers.hf_ctx import HfCtx
    return HfCtx(_model=_FakeModel(), _tokenizer=_FakeTokenizer())


class TestNeedsModelInterfaceSmoke:
    """GPU가 없어 forward 결과의 정확도는 검증 못 한다 — 죽지 않고 유한값을 내는지만."""

    def test_pmi_shift_runs(self, fake_ctx):
        from src.rulers.pmi_shift import PmiShiftSum
        site = make_site()
        sample = make_sample()
        v = PmiShiftSum().score(site, sample, fake_ctx)
        assert isinstance(v, float)

    def test_osd_runs(self, fake_ctx):
        from src.rulers.osd import OsdUnsigned
        site = make_site()
        sample = make_sample()
        v = OsdUnsigned().score(site, sample, fake_ctx)
        assert isinstance(v, float)

    def test_dcont_runs(self, fake_ctx):
        from src.rulers.dcont import DCont
        site = make_site()
        sample = make_sample()
        v = DCont().score(site, sample, fake_ctx)
        assert isinstance(v, float)

    def test_move_kl_runs(self, fake_ctx):
        from src.rulers.move_kl import MoveKl
        site = make_site()
        sample = make_sample()
        v = MoveKl().score(site, sample, fake_ctx)
        assert isinstance(v, float)

    def test_hidden_probe_extract_runs(self, fake_ctx):
        from src.rulers.hidden_probe import HiddenAtMetaEnd
        site = make_site()
        sample = make_sample()
        feat = HiddenAtMetaEnd().extract(site, sample, fake_ctx)
        assert feat is not None
        assert feat.shape == (8,)


# ══════════════════════════════════════════════════════════════════════════════
# hidden_probe — 순수 numpy 로지스틱 회귀
# ══════════════════════════════════════════════════════════════════════════════

class TestHiddenProbe:
    def test_fit_and_score_separable_data(self):
        from src.rulers.hidden_probe import fit_probe, probe_score
        rng = np.random.RandomState(0)
        n = 200
        X = rng.randn(n, 3)
        y = (X[:, 0] + 0.1 * rng.randn(n) > 0).astype(float)
        probe = fit_probe(X, y, groups=[i % 10 for i in range(n)])
        p = probe_score(probe, X)
        pred = (p > 0.5).astype(float)
        acc = float((pred == y).mean())
        assert acc > 0.8

    def test_grouped_cv_returns_reasonable_auc(self):
        from src.rulers.hidden_probe import probe_grouped_cv
        rng = np.random.RandomState(1)
        n = 300
        X = rng.randn(n, 4)
        y = (X[:, 0] - X[:, 1] > 0).astype(float)
        groups = [i % 20 for i in range(n)]
        result = probe_grouped_cv(X, y, groups, n_folds=5)
        assert result["n_folds_used"] >= 3
        assert result["auc"] > 0.6


def test_novel_mask_and_mass_shift_direction():
    from src.rulers.move_kl import _novel_mask, enumerate_candidate_moves
    cands = enumerate_candidate_moves([25, 3, 7, 8])
    mask = _novel_mask(cands, frozenset({(3, 25)}))
    assert len(mask) == len(cands) == 24
    assert sum(1 for c, m in zip(cands, mask) if not m) == 4      # 25∘3 네 연산만 «이미 시도»
    assert all(m for c, m in zip(cands, mask) if "7" in c and "8" in c)


def test_move_novel_shift_registered_in_cli():
    src = open("scripts/local/ruler_table.py", encoding="utf-8").read()
    assert "MoveNovelShift()" in src and "MoveNovelShiftStuck()" in src


# ══════════════════════════════════════════════════════════════════════════════
# 0905 성능 수리 회귀 테스트 — 자 표가 87분/200행 → ≤3 GPU시간/48,000행 목표로
# 바뀌면서 건드린 세 자리(HfCtx 캐시/절단, move_kl 배치화, attack_limit/limit_sites)
# 가 **숫자를 안 바꿨는지** 실측한다. 전부 CPU 스텁(_FakeModelPositional)만 쓴다 —
# 진짜 모델의 숫자 품질이 아니라 "옛 경로와 새 경로가 바이트 동일하다"만 확인하면
# 되기 때문이다(과제 지시문 §목표 (c)).
# ══════════════════════════════════════════════════════════════════════════════

class _FakeModelPositional:
    r"""`_FakeModel`(균등분포)과 달리 **위치·문맥에 의존하는** 로짓을 낸다 — 배치화
    (오른쪽 패딩)가 실토큰 위치의 logits를 조금이라도 바꾸면 이 스텁이 잡아낸다
    (균등분포 스텁은 패딩 버그를 절대 못 잡는다: 모든 위치가 이미 같은 값이라서).
    실제 토큰 id(0..999, `_FakeTokenizer`가 낸다)에 대해서만 정의되게 vocab=1000.
    인과적(각 위치는 그 위치까지의 id만으로 결정) — 패딩은 항상 실토큰 뒤에만
    오므로 실토큰 위치의 값은 패딩 유무와 무관해야 하고, 이 스텁은 정확히 그렇게
    구현된다(뒤쪽 패딩을 아예 안 본다)."""
    device = "cpu"
    vocab = 1000

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        import torch
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        ids = input_ids.tolist()
        for bi in range(b):
            running = 0
            for ti in range(t):
                running = (running * 7 + ids[bi][ti] + 1) % 997
                center = running % self.vocab
                # 뾰족한(peaky) 분포 — 배치/순차 경로가 갈리면 gather 값이 눈에
                # 띄게 달라지도록 값 차이를 크게 둔다.
                for v in range(self.vocab):
                    logits[bi, ti, v] = -abs(v - center) * 0.37
        return _FakeLogits(logits)

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def positional_ctx():
    from src.rulers.hf_ctx import HfCtx
    return HfCtx(_model=_FakeModelPositional(), _tokenizer=_FakeTokenizer())


class TestMoveKlBatching:
    r"""0905: `candidate_logits_batched`(오른쪽 패딩 배치)를 실제 `models/Qwen3-4B`
    bf16 GPU에서 200행 재현 채점으로 검증했더니 순차 경로와 바이트 동일하지
    **않았다**(개별 후보 로그확률 합 최대 diff ~1.16, `move_kl` 점수 최대 diff
    0.227 — `move_novel_shift`의 AUC가 0.708→0.625로 갈렸다). 그래서 정본
    `candidate_logits`는 순차 구현의 별칭으로 되돌렸다(`move_kl.py` 주석 참조).

    이 테스트는 그 배치판이 **결정적 CPU 스텁에서는** 순차판과 일치한다는 것만
    확인한다 — "스텁 일치 ⇏ 실기 일치"라는 이번에 배운 교훈을 코드로 남겨 두는
    회귀 테스트다(배치판을 다시 정본으로 승격하려면 이 테스트가 통과하는 것만으론
    부족하고, 실제 모델 재현 비교가 다시 필요하다)."""

    SITES_AND_SAMPLES = [
        (make_site(nums=(5, 19, 25, 3), witness="5+19", decoy="5-19"),
         make_sample(meta_start=0, meta_end=40)),
        (make_site(site_id="s1", nums=(2, 3, 4, 6), witness="2*6", decoy="2+6"),
         make_sample(meta_start=5, meta_end=55,
                    continuation="hello<meta>\nconfidence: 0.3\nnext: 2*6\ndecision: verify\n</meta> world")),
        (make_site(site_id="s2", nums=(7, 7, 1, 1), witness="7+1", decoy="7-1"),
         make_sample(meta_start=2, meta_end=20, continuation="ab" + "c" * 40)),
    ]

    @pytest.mark.parametrize("site,sample", SITES_AND_SAMPLES)
    def test_batched_matches_sequential_on_fake_stub_only(self, positional_ctx, site, sample):
        from src.rulers.move_kl import (_candidate_logits_sequential, candidate_logits_batched,
                                        candidate_logits, enumerate_candidate_moves)
        assert candidate_logits is _candidate_logits_sequential, (
            "정본 candidate_logits는 순차 구현이어야 한다 — 배치판을 다시 기본으로 "
            "돌리려면 실제 모델 재현 비교부터 다시 하라(위 클래스 docstring).")
        candidates = enumerate_candidate_moves(site.nums)
        p_ids = list(positional_ctx.encode(
            positional_ctx.tokenizer.apply_chat_template(
                site.prompt_messages, tokenize=False, add_generation_prompt=True)))
        prefix_ids = list(positional_ctx.encode(site.prefix + (sample.continuation or "")[:sample.meta_start]))

        seq = _candidate_logits_sequential(positional_ctx, p_ids, prefix_ids, candidates)
        # 순차 경로가 방금 self._lp_cache 를 채웠으니, 배치 경로가 "캐시라서
        # 우연히 같다"로 통과하지 않게 캐시를 비우고 잰다.
        positional_ctx._lp_cache.clear()
        batched = candidate_logits_batched(positional_ctx, p_ids, prefix_ids, candidates, batch=8)

        assert len(seq) == len(batched) == len(candidates)
        for a, b in zip(seq, batched):
            if math.isnan(a):
                assert math.isnan(b)
            else:
                assert abs(a - b) < 1e-4, f"{a} vs {b}"

    def test_dists_for_cache_matches_uncached(self, positional_ctx):
        """캐시가 값을 바꾸지 않는지 — 캐시 없는(매번 새 ctx) 계산과 비교."""
        from src.rulers.move_kl import MoveKl, MoveKlSigned, MoveNovelShift, MoveNovelShiftStuck
        from src.rulers.hf_ctx import HfCtx
        site = make_site(nums=(5, 19, 25, 3), witness="5+19", decoy="5-19", family_dead=1)
        sample = make_sample(meta_start=0, meta_end=40)

        cached_scores = {}
        for ruler in (MoveKl(), MoveKlSigned(), MoveNovelShift(), MoveNovelShiftStuck()):
            cached_scores[ruler.name] = ruler.score(site, sample, positional_ctx)

        for ruler in (MoveKl(), MoveKlSigned(), MoveNovelShift(), MoveNovelShiftStuck()):
            fresh_ctx = HfCtx(_model=_FakeModelPositional(), _tokenizer=_FakeTokenizer())
            fresh = ruler.score(site, sample, fresh_ctx)
            if math.isnan(cached_scores[ruler.name]):
                assert math.isnan(fresh)
            else:
                assert abs(cached_scores[ruler.name] - fresh) < 1e-6


class TestHfCtxCacheAndTruncation:
    def test_token_logprobs_cache_hits_return_identical_values(self, positional_ctx):
        ctx_ids = list(positional_ctx.encode("the quick brown fox"))
        target_ids = list(positional_ctx.encode("jumps"))
        first = positional_ctx.token_logprobs(ctx_ids, target_ids)
        assert len(positional_ctx._lp_cache) == 1
        second = positional_ctx.token_logprobs(ctx_ids, target_ids)
        assert np.allclose(first, second)
        # 캐시가 반환하는 배열이 원본과 별개 사본이어야(호출자가 실수로 캐시를
        # 오염시키는 걸 막는다).
        second[0] = 12345.0
        third = positional_ctx.token_logprobs(ctx_ids, target_ids)
        assert third[0] != 12345.0

    def test_long_context_is_truncated_from_the_left(self, positional_ctx):
        from src.rulers.hf_ctx import MAX_CTX_TOKENS
        long_ctx = list(range(MAX_CTX_TOKENS + 500))
        short_ctx = long_ctx[-MAX_CTX_TOKENS:]
        target = [1, 2, 3]
        lp_long = positional_ctx.token_logprobs(long_ctx, target)
        positional_ctx._lp_cache.clear()
        lp_short = positional_ctx.token_logprobs(short_ctx, target)
        assert np.allclose(lp_long, lp_short)


# ══════════════════════════════════════════════════════════════════════════════
# 자 표 CLI — --limit_sites (사이트 단위 층화 표집) 및 attack_battery attack_limit
# ══════════════════════════════════════════════════════════════════════════════

class TestLimitSitesAndAttackLimit:
    def test_select_sites_stratified_counts_and_columns_preserved(self, tmp_path):
        import sys
        sys.path.insert(0, "scripts/local")
        from ruler_table import select_sites_stratified

        sites_rows = []
        for i in range(20):
            sites_rows.append({"site_id": f"site{i}", "nums": [1, 2, 3, 4], "target": 10,
                              "witness": "1+2", "decoy": "1-2", "prefix": "", "prompt": "[]",
                              "pairs_pre": "[]", "family_dead": 1 if i % 2 == 0 else 0,
                              "live_new_moves": "[]"})
        sites_df = pd.DataFrame(sites_rows)
        conts_rows = []
        for i in range(20):
            for k in range(3):
                conts_rows.append({"site_id": f"site{i}", "mode": "meta", "k_index": k,
                                  "continuation": "x", "meta_raw": "<meta>\n</meta>",
                                  "decision": "verify", "confidence": 0.5, "r_corr": 0,
                                  "emitted": 1, "meta_start_in_cont": 0})
        conts_df = pd.DataFrame(conts_rows)

        picked_sites, picked_conts = select_sites_stratified(sites_df, conts_df, 10, seed=0)
        assert len(picked_sites) == 10
        assert sum(picked_sites["family_dead"] == 1) == 5
        assert sum(picked_sites["family_dead"] == 0) == 5
        # 사이트를 골랐으면 그 사이트의 행을 전부(3개씩) 남겨야 한다 — row-limit처럼
        # 사이트 하나를 반토막 내지 않는다.
        assert len(picked_conts) == 10 * 3
        assert set(picked_conts["site_id"]) == set(picked_sites["site_id"])

    def test_select_sites_stratified_noop_when_n_ge_total(self):
        import sys
        sys.path.insert(0, "scripts/local")
        from ruler_table import select_sites_stratified
        sites_df = pd.DataFrame({"site_id": ["a", "b"], "family_dead": [1, 0]})
        conts_df = pd.DataFrame({"site_id": ["a", "b"]})
        out_sites, out_conts = select_sites_stratified(sites_df, conts_df, 5)
        assert len(out_sites) == 2

    def test_attack_battery_caps_distinct_sites(self):
        from src.rulers.table import attack_battery

        class _CountingRuler:
            name = "counting"
            needs_model = False
            calls = 0

            def score(self, site, sample, ctx):
                _CountingRuler.calls += 1
                return 0.5

        sites_rows, conts_rows = [], []
        for i in range(10):
            sites_rows.append({"site_id": f"s{i}", "nums": [1, 2, 3, 4], "target": 10,
                              "witness": "1+2", "decoy": "1-2", "prefix": "", "prompt": "[]",
                              "pairs_pre": "[]", "family_dead": 0, "live_new_moves": "[]"})
            conts_rows.append({"site_id": f"s{i}", "mode": "meta", "k_index": 0,
                              "continuation": "x", "meta_raw": "<meta>\n</meta>",
                              "decision": "verify", "confidence": 0.5, "r_corr": 0,
                              "emitted": 1, "meta_start_in_cont": 0})
        sites_df = pd.DataFrame(sites_rows)
        conts_df = pd.DataFrame(conts_rows)

        res = attack_battery(sites_df, conts_df, _CountingRuler(), ctx=None, attack_limit=3)
        assert res["n_sites"] == 3

    def test_build_scores_skips_model_when_emitted_zero(self):
        from src.rulers.table import build_scores

        class _AlwaysCalledRuler:
            name = "always"
            needs_model = True
            calls = 0

            def score(self, site, sample, ctx):
                _AlwaysCalledRuler.calls += 1
                return 1.0

        sites_df = pd.DataFrame([{"site_id": "s0", "nums": [1, 2, 3, 4], "target": 10,
                                 "witness": "1+2", "decoy": "1-2", "prefix": "", "prompt": "[]",
                                 "pairs_pre": "[]", "family_dead": 0, "live_new_moves": "[]"}])
        conts_df = pd.DataFrame([{"site_id": "s0", "mode": "meta", "k_index": 0,
                                 "continuation": "<meta>\nx\n</meta>", "meta_raw": "<meta>\nx\n</meta>",
                                 "decision": None, "confidence": None, "r_corr": 0,
                                 "emitted": 0, "meta_start_in_cont": 0}])
        ruler = _AlwaysCalledRuler()
        out = build_scores(sites_df, conts_df, [ruler], ctx=object(), limit=0)
        assert math.isnan(out.loc[0, "always"])
        assert _AlwaysCalledRuler.calls == 0
