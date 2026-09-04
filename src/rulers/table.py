r"""자 채점표 — 모든 자를 오라클·결과·기준선·공격·advantage 시뮬레이션에 대해
한 번에 채점한다. `docs/VERDICT_cd6_pair_rulers.md`의 판정 방법론(within-site
Spearman/AUC, outcome-fixed 재계산, 공격 배터리, partial correlation)을
`Ruler` 프로토콜 위에서 재사용 가능하게 일반화한 것 — VERDICT는 손으로 짠
일회성 스크립트(`scripts/pair_rulers.py`)였고, 이 파일은 그 방법론을 임의의
새 자에 적용할 수 있는 함수로 만든다.

관문 상수(§Pass rule, 과제 지시문 그대로):
  AUC ≥ 0.60, 모든 기준선을 ≥0.05 차로 이겨야, outcome-fixed(=r_corr별 층)에서
  생존, 공격이 정직한 메타를 이기는 사이트 비율이 40% 초과면 불합격.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
from dataclasses import asdict
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.rulers.base import (MetaSample, Site, live_new_moves_from_json,
                             pairs_pre_from_json)
from src.rulers.oracle import oracle_score

# ── 관문 상수 ────────────────────────────────────────────────────────────────
AUC_MIN = 0.60
BASELINE_MARGIN_MIN = 0.05
ATTACK_OUTRANK_MAX_FRAC = 0.40

__all__ = [
    "AUC_MIN", "BASELINE_MARGIN_MIN", "ATTACK_OUTRANK_MAX_FRAC",
    "load_sites", "site_from_row", "sample_from_row", "spearman", "auc_score",
    "holm_correction", "partial_correlation", "regress_out",
    "within_site_metric", "build_scores", "attack_battery", "placebo_diff",
    "advantage_simulation", "run_table", "SYNTH_ATTACKS",
]


# ══════════════════════════════════════════════════════════════════════════════
# 0. 로드 / 변환
# ══════════════════════════════════════════════════════════════════════════════

def load_sites(sites_path: str) -> pd.DataFrame:
    return pd.read_parquet(sites_path)



def _int_or_none(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return int(f)
    except (TypeError, ValueError):
        return None

def site_from_row(row) -> Site:
    r"""`scripts/local/build_sites.py`가 실제로 내는 site parquet 스키마(확인함,
    2026-09-04): 컬럼명은 `prompt_json`이 아니라 **`prompt`**이고,
    `countdown_sites.build_site_row`(:390-406)가 그 안에 **prefix를 마지막
    assistant 메시지로 이미 구운다**(`render_prefix_prompt`가
    `msgs.append({"role":"assistant","content":prefix})`). `Site.prompt_messages`는
    "prefix 없는 프롬프트"를 기대하므로(각 자가 `add_generation_prompt=True`로
    렌더링한 뒤 `site.prefix`를 별도 텍스트로 이어붙인다), 여기서 그 마지막
    assistant 메시지를 벗겨낸다 — 벗기지 않으면 prefix가 두 번(프롬프트 안 + 별도
    텍스트) 들어간다.

    구 스키마(`prompt_json`, 문자열 JSON, prefix 미포함)도 과제 지시문이 원래
    명시한 대안 스키마이므로 폴백으로 계속 받는다.
    """
    raw = row["prompt"] if "prompt" in row and row.get("prompt") is not None else row.get("prompt_json")
    if isinstance(raw, str):
        prompt_messages = json.loads(raw) if raw else []
    else:
        prompt_messages = list(raw) if raw is not None else []
    prefix = str(row.get("prefix", "") or "")
    if (prompt_messages and isinstance(prompt_messages[-1], dict)
            and prompt_messages[-1].get("role") == "assistant"
            and prompt_messages[-1].get("content", "") == prefix and prefix):
        prompt_messages = prompt_messages[:-1]
    nums = tuple(int(x) for x in row["nums"])
    return Site(
        prompt_messages=prompt_messages,
        prefix=str(row.get("prefix", "") or ""),
        nums=nums,
        target=int(row["target"]),
        witness=str(row.get("witness", "") or ""),
        decoy=str(row.get("decoy", "") or ""),
        pairs_pre=pairs_pre_from_json(row.get("pairs_pre")),
        # family_dead 는 «메타 앞 시도 없음» 자리에서 NaN/None 이다(정답표 없음). 0 으로 뭉개면
        # «살아 있는 계열»로 잘못 읽히므로 None 을 유지한다(오라클·타이밍 자는 None 을 0 점으로 본다).
        family_dead=_int_or_none(row.get("family_dead")),
        live_new_moves=live_new_moves_from_json(row.get("live_new_moves")),
        site_id=str(row.get("site_id")),
    )


_NEXT_FIELD_RE = re.compile(r"\bnext\s*:\s*(.*)", re.I)


def sample_from_row(row) -> MetaSample:
    r"""`scripts/local/gen_continuations.py:build_record`(확인함, 2026-09-04)가
    실제로 내는 컬럼과 과제 지시문이 가정한 컬럼 사이에 두 군데 틈이 있다 —
    둘 다 여기서 메운다(그 파일을 고치지 않는다):

      meta_end       그 파일은 `meta_start_in_cont`만 낸다(끝 오프셋이 없다).
                     `meta_start_in_cont + len(meta_raw)`로 복원한다 —
                     `parse_meta`의 `raw`가 여는~닫는 태그를 전부 포함하는
                     블록 원문이므로 길이를 더하면 끝 오프셋과 같다(태그
                     내부에 멀티바이트 유니코드가 없는 한 문자 오프셋 산수로
                     충분하다 — `find_meta_token_span`처럼 재-스캔하지 않는
                     이유는 이 패키지가 그 파일의 재-토큰화 없이 문자 오프셋만
                     복원하면 되기 때문).
      next_move      그 파일에는 컬럼 자체가 없다. `meta_raw`의 `next:` 필드에서
                     `countdown_rewards._NEXT_RE`와 같은 정규식으로 다시 뽑는다.
    """
    meta_raw = str(row.get("meta_raw", "") or "")
    ms_raw = row.get("meta_start", row.get("meta_start_in_cont", None))
    meta_start = int(ms_raw) if pd.notna(ms_raw) else -1
    me_raw = row.get("meta_end", None)
    if pd.notna(me_raw):
        meta_end = int(me_raw)
    elif meta_start >= 0 and meta_raw:
        meta_end = meta_start + len(meta_raw)
    else:
        meta_end = -1

    next_move = None
    if "next_move" in row and pd.notna(row.get("next_move")):
        next_move = str(row["next_move"])
    elif meta_raw:
        m = _NEXT_FIELD_RE.search(meta_raw)
        if m:
            next_move = m.group(1).strip()

    return MetaSample(
        continuation=str(row.get("continuation", "") or ""),
        meta_raw=meta_raw,
        meta_start=meta_start,
        meta_end=meta_end,
        decision=(str(row["decision"]) if pd.notna(row.get("decision", None)) else None),
        confidence=(float(row["confidence"]) if pd.notna(row.get("confidence", None)) else None),
        r_corr=(int(row["r_corr"]) if pd.notna(row.get("r_corr", None)) else None),
        next_move=next_move,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. 순수통계 (sklearn/scipy 없음 — CPU 전용 관례)
# ══════════════════════════════════════════════════════════════════════════════

def _rank(xs: np.ndarray) -> np.ndarray:
    order = np.argsort(xs, kind="mergesort")
    ranks = np.empty(len(xs), dtype=float)
    ranks[order] = np.arange(1, len(xs) + 1, dtype=float)
    sorted_x = xs[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_score(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y, s = y[mask], s[mask]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rank(s)
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def holm_correction(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down. NaN p값은 1.0으로 취급(보수적)."""
    p = [1.0 if (v is None or not math.isfinite(v)) else float(v) for v in pvals]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def regress_out(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """`y`에서 `X`(baseline 특징들, bias 열 자동 추가)로 설명되는 부분을 최소자승으로
    빼고 잔차를 돌려준다 — partial correlation의 표준 구현."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < X.shape[1] + 2:
        return resid
    Xb = np.hstack([X[mask], np.ones((mask.sum(), 1))])
    beta, *_ = np.linalg.lstsq(Xb, y[mask], rcond=None)
    resid[mask] = y[mask] - Xb @ beta
    return resid


def partial_correlation(y: Sequence[float], z: Sequence[float], baselines: np.ndarray) -> float:
    """corr(y,z) baselines를 통제한 뒤. y=ruler 점수, z=오라클 점수(또는 r_corr)."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ry = regress_out(y, baselines)
    rz = regress_out(z, baselines)
    return spearman(ry, rz)


# ══════════════════════════════════════════════════════════════════════════════
# 2. within-site 지표
# ══════════════════════════════════════════════════════════════════════════════

def within_site_metric(df: pd.DataFrame, score_col: str, label_col: str,
                       site_col: str = "site_id", metric: str = "spearman") -> dict:
    r"""사이트별로 지표를 계산해 평균낸다(그룹 크기 가중 없음 — 사이트가 단위).
    Returns {"mean", "n_sites", "per_site": {site_id: value}}."""
    vals = {}
    for sid, g in df.groupby(site_col):
        if metric == "spearman":
            v = spearman(g[score_col].values, g[label_col].values)
        elif metric == "auc":
            v = auc_score(g[label_col].values, g[score_col].values)
        else:
            raise ValueError(f"within_site_metric: metric={metric!r} 모른다")
        if math.isfinite(v):
            vals[sid] = v
    mean = float(np.mean(list(vals.values()))) if vals else float("nan")
    return {"mean": mean, "n_sites": len(vals), "per_site": vals}


# ══════════════════════════════════════════════════════════════════════════════
# 3. 자 채점 (모델 필요/불필요 공통 진입점)
# ══════════════════════════════════════════════════════════════════════════════

def build_scores(sites_df: pd.DataFrame, conts_df: pd.DataFrame, rulers: Sequence,
                 ctx: Optional[Any] = None, limit: int = 0) -> pd.DataFrame:
    r"""각 (site, sample) 행에 대해 자·오라클·기준선 점수를 붙인 DataFrame.

    `rulers`에 `needs_model=True`인 자가 있는데 `ctx is None`이면 그 자는 NaN으로
    채워지고 스킵된다(예외를 던지지 않는다 — `--no-model` 경로가 그것을 정상 동작
    으로 기대한다).
    """
    from src.rulers.baselines import compute_all_baselines

    sites_by_id = {str(r["site_id"]): site_from_row(r) for _, r in sites_df.iterrows()}
    rows = []
    it = conts_df.iterrows()
    for i, (_, row) in enumerate(it):
        if limit and i >= limit:
            break
        sid = str(row["site_id"])
        site = sites_by_id.get(sid)
        if site is None:
            continue
        sample = sample_from_row(row)
        out = {"site_id": sid, "mode": row.get("mode"), "policy_tag": row.get("policy_tag"),
               "k_index": row.get("k_index"), "r_corr": sample.r_corr,
               "emitted": row.get("emitted"), "novel": row.get("novel"),
               "followed": row.get("followed"), "checked": row.get("checked")}
        out.update(compute_all_baselines(site, sample))
        for ruler in rulers:
            if ruler.needs_model and ctx is None:
                out[ruler.name] = float("nan")
                continue
            try:
                out[ruler.name] = float(ruler.score(site, sample, ctx))
            except Exception:
                out[ruler.name] = float("nan")
        rows.append(out)
    scored = pd.DataFrame(rows)

    # 오라클: 사이트별 성공률(=이 사이트에서 관측된 r_corr 평균)을 계산한 뒤 사용.
    if "r_corr" in scored.columns:
        succ = scored.groupby("site_id")["r_corr"].mean().to_dict()
    else:
        succ = {}
    oracle_rows = []
    for i, (_, row) in enumerate(conts_df.iterrows()):
        if limit and i >= limit:
            break
        sid = str(row["site_id"])
        site = sites_by_id.get(sid)
        if site is None:
            continue
        sample = sample_from_row(row)
        os_ = oracle_score(site, sample, site_success_rate=succ.get(sid))
        oracle_rows.append({"state_ok": os_.state_ok, "plan_ok": os_.plan_ok,
                            "calib_ok": os_.calib_ok, "oracle_total": os_.total})
    oracle_df = pd.DataFrame(oracle_rows)
    scored = pd.concat([scored.reset_index(drop=True), oracle_df.reset_index(drop=True)], axis=1)
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# 4. 공격 배터리
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_meta(kind: str, nums) -> str:
    if kind == "empty":
        return "<meta>\n</meta>"
    if kind == "random_numbers":
        rng = random.Random(0)
        vals = [rng.randint(1, 99) for _ in range(3)]
        return f"<meta>\nconfidence: 0.5\n{' '.join(map(str, vals))}\ndecision: verify\n</meta>"
    if kind == "gibberish":
        return "<meta>\nasdf qwer zxcv blah blah nonsense\n</meta>"
    if kind == "verify_one_liner":
        return "<meta>\nconfidence: 0.5\ndecision: verify\n</meta>"
    raise ValueError(f"_synthetic_meta: kind={kind!r} 모른다")


SYNTH_ATTACKS = ("empty", "random_numbers", "gibberish", "verify_one_liner")


def attack_battery(sites_df: pd.DataFrame, conts_df: pd.DataFrame, ruler,
                   ctx: Optional[Any] = None, limit: int = 0) -> dict:
    r"""정직한 메타 vs 4종 공격 메타. 사이트별로 공격이 정직한 메타를 이긴(점수가
    더 높은) 비율을 낸다. `docs/POSTMORTEM_cd6_rulers_2026-09-03.md` §1.2의
    공격 배터리(빈 메타 0.34, 횡설수설 0.34, 숫자 나열 0.74/0.77)를 재현하는지가
    새 자의 첫 관문이다."""
    if ruler.needs_model and ctx is None:
        return {"outrank_frac": float("nan"), "n_sites": 0, "per_attack": {}}
    sites_by_id = {str(r["site_id"]): site_from_row(r) for _, r in sites_df.iterrows()}
    per_attack_wins: dict[str, list[int]] = {k: [] for k in SYNTH_ATTACKS}
    n_sites = 0
    for i, (_, row) in enumerate(conts_df.iterrows()):
        if limit and i >= limit:
            break
        if not row.get("meta_raw"):
            continue
        sid = str(row["site_id"])
        site = sites_by_id.get(sid)
        if site is None:
            continue
        sample = sample_from_row(row)
        try:
            honest = float(ruler.score(site, sample, ctx))
        except Exception:
            continue
        if not math.isfinite(honest):
            continue
        n_sites += 1
        cont = sample.continuation or ""
        for kind in SYNTH_ATTACKS:
            fake_meta = _synthetic_meta(kind, site.nums)
            fake_sample = MetaSample(
                continuation=cont[:max(0, sample.meta_start)] + fake_meta + cont[max(0, sample.meta_end):],
                meta_raw=fake_meta, meta_start=sample.meta_start,
                meta_end=(sample.meta_start if sample.meta_start >= 0 else 0) + len(fake_meta),
                decision=sample.decision, confidence=sample.confidence, r_corr=sample.r_corr,
            )
            try:
                fake_score = float(ruler.score(site, fake_sample, ctx))
            except Exception:
                fake_score = float("nan")
            if math.isfinite(fake_score):
                per_attack_wins[kind].append(int(fake_score > honest))
    per_attack = {k: (float(np.mean(v)) if v else float("nan")) for k, v in per_attack_wins.items()}
    overall = [v for vs in per_attack_wins.values() for v in vs]
    return {"outrank_frac": float(np.mean(overall)) if overall else float("nan"),
            "n_sites": n_sites, "per_attack": per_attack}


# ══════════════════════════════════════════════════════════════════════════════
# 5. 위약(placebo) 차분
# ══════════════════════════════════════════════════════════════════════════════

def placebo_diff(scored: pd.DataFrame, score_col: str) -> dict:
    r"""같은 site_id에서 mode=="meta"(자기 메타) 대 mode=="donor"(다른 사이트에서
    가져온 메타, `donor_meta_raw` 사용)의 점수 차 vs 성공률 차. `scored`에 "mode"
    컬럼이 없으면(또는 두 모드가 없으면) NaN."""
    if "mode" not in scored.columns:
        return {"corr": float("nan"), "n_sites": 0}
    meta_rows = scored[scored["mode"] == "meta"].groupby("site_id").agg(
        score=(score_col, "mean"), succ=("r_corr", "mean"))
    donor_rows = scored[scored["mode"] == "donor"].groupby("site_id").agg(
        score=(score_col, "mean"), succ=("r_corr", "mean"))
    joined = meta_rows.join(donor_rows, lsuffix="_meta", rsuffix="_donor", how="inner")
    if joined.empty:
        return {"corr": float("nan"), "n_sites": 0}
    score_diff = joined["score_meta"] - joined["score_donor"]
    succ_diff = joined["succ_meta"] - joined["succ_donor"]
    return {"corr": spearman(score_diff.values, succ_diff.values), "n_sites": len(joined)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. advantage 시뮬레이션 (GRPO)
# ══════════════════════════════════════════════════════════════════════════════

def _grpo_advantage(rewards: np.ndarray) -> np.ndarray:
    mu, sigma = rewards.mean(), rewards.std()
    if sigma < 1e-8:
        return np.zeros_like(rewards)
    return (rewards - mu) / sigma


def _reward_formula(name: str, row) -> float:
    y = float(row.get("r_corr", 0.0) or 0.0)
    pmi = row.get("pmi_shift_sum", row.get("pmi_shift_mean", float("nan")))
    osd = row.get("osd_unsigned", float("nan"))
    decision = (row.get("decision") or "").strip().lower()
    family_dead = int(row.get("family_dead", 0) or 0)
    live_hit = int(row.get("plan_ok", 0) or 0)

    def _finite(v, default=0.0):
        v = float(v) if v is not None else default
        return v if math.isfinite(v) else default

    if name == "outcome_only":
        return y
    if name == "outcome_x_pmi":
        return y * max(-1.0, min(1.0, _finite(pmi)))
    if name == "outcome_x_osd":
        return y * max(-1.0, min(1.0, _finite(osd)))
    if name == "timing":
        if decision == "redirect" and family_dead == 1:
            return 1.0
        if decision == "redirect" and family_dead == 0:
            return -1.0
        if decision != "redirect" and family_dead == 1:
            return -1.0
        return 0.0
    if name == "live_new":
        return float(live_hit)
    if name == "timing_live_new":
        return _reward_formula("timing", row) + float(live_hit)
    raise ValueError(f"_reward_formula: name={name!r} 모른다")


REWARD_FORMULAS = ("outcome_only", "outcome_x_pmi", "outcome_x_osd", "timing",
                   "live_new", "timing_live_new")


def advantage_simulation(scored: pd.DataFrame, site_col: str = "site_id") -> dict:
    r"""각 후보 보상식에 대해, site 내부 GRPO advantage를 계산하고
    oracle-good(oracle_total>=2) vs oracle-bad(oracle_total==0) 메타가 받는 평균
    advantage를 보고한다."""
    out = {}
    for formula in REWARD_FORMULAS:
        adv_good, adv_bad = [], []
        for sid, g in scored.groupby(site_col):
            if len(g) < 2:
                continue
            rewards = np.array([_reward_formula(formula, r) for _, r in g.iterrows()])
            adv = _grpo_advantage(rewards)
            oracle_total = g["oracle_total"].values if "oracle_total" in g.columns else np.zeros(len(g))
            for a, o in zip(adv, oracle_total):
                if o >= 2:
                    adv_good.append(a)
                elif o == 0:
                    adv_bad.append(a)
        out[formula] = {
            "mean_adv_good": float(np.mean(adv_good)) if adv_good else float("nan"),
            "mean_adv_bad": float(np.mean(adv_bad)) if adv_bad else float("nan"),
            "n_good": len(adv_good), "n_bad": len(adv_bad),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. 진입점 — 전체 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def run_table(sites_df: pd.DataFrame, conts_df: pd.DataFrame, rulers: Sequence,
             ctx: Optional[Any] = None, limit: int = 0) -> dict:
    r"""전체 채점표. Returns dict with keys "scored"(DataFrame, JSON엔 안 실림),
    "rulers"(ruler별 지표 dict), "advantage_sim", "markdown", "json"."""
    scored = build_scores(sites_df, conts_df, rulers, ctx=ctx, limit=limit)

    baseline_cols = [c for c in ("s2_overlap", "meta_length", "stated_confidence") if c in scored.columns]
    baselines_mat = scored[baseline_cols].values if baseline_cols else np.zeros((len(scored), 0))

    ruler_results = {}
    pvals_for_holm = []
    metric_keys = []
    for ruler in rulers:
        col = ruler.name
        if col not in scored.columns:
            continue
        res: dict[str, Any] = {}
        res["auc_all"] = within_site_metric(scored, col, "r_corr", metric="auc")
        for corr_val, tag in ((1, "r_corr1"), (0, "r_corr0")):
            sub = scored[scored["r_corr"] == corr_val]
            res[f"auc_{tag}"] = within_site_metric(sub, col, "oracle_total", metric="spearman") \
                if len(sub) else {"mean": float("nan"), "n_sites": 0}
        res["spearman_oracle"] = within_site_metric(scored, col, "oracle_total", metric="spearman")
        res["partial_corr_oracle"] = partial_correlation(
            scored[col].values, scored["oracle_total"].values, baselines_mat)
        for b in baseline_cols:
            res[f"beats_{b}"] = bool(
                math.isfinite(res["auc_all"]["mean"]) and math.isfinite(
                    within_site_metric(scored, b, "r_corr", metric="auc")["mean"])
                and res["auc_all"]["mean"] - within_site_metric(scored, b, "r_corr", metric="auc")["mean"]
                >= BASELINE_MARGIN_MIN)
        res["placebo"] = placebo_diff(scored, col)
        res["attack"] = attack_battery(sites_df, conts_df, ruler, ctx=ctx, limit=limit) \
            if not (ruler.needs_model and ctx is None) else {"outrank_frac": float("nan"), "n_sites": 0, "per_attack": {}}
        auc = res["auc_all"]["mean"]
        res["pass_auc"] = bool(math.isfinite(auc) and auc >= AUC_MIN)
        res["pass_attack"] = bool(math.isfinite(res["attack"]["outrank_frac"])
                                  and res["attack"]["outrank_frac"] <= ATTACK_OUTRANK_MAX_FRAC)
        ruler_results[col] = res
        # p-value proxy for Holm: use 1 - |spearman| mapped roughly (no scipy). We
        # report the correlation magnitude's rank-based two-sided p via permutation
        # is too slow here; instead we record NaN p and let callers use raw effect
        # sizes — Holm is applied on a normal-approx p from spearman rho, n.
        rho = res["spearman_oracle"]["mean"]
        n = res["spearman_oracle"]["n_sites"]
        pvals_for_holm.append(_spearman_p_approx(rho, n))
        metric_keys.append(col)

    holm_adj = holm_correction(pvals_for_holm)
    for col, p_adj in zip(metric_keys, holm_adj):
        ruler_results[col]["holm_adj_p"] = p_adj

    adv_sim = advantage_simulation(scored)

    md = _to_markdown(ruler_results, adv_sim)
    out = {"rulers": ruler_results, "advantage_sim": adv_sim,
          "n_rows": len(scored), "gates": {"auc_min": AUC_MIN,
          "baseline_margin_min": BASELINE_MARGIN_MIN,
          "attack_outrank_max_frac": ATTACK_OUTRANK_MAX_FRAC}}
    return {"scored": scored, "rulers": ruler_results, "advantage_sim": adv_sim,
           "markdown": md, "json": out}


def _spearman_p_approx(rho: float, n: int) -> float:
    """정규근사 p값(양측). scipy 없이 Fisher z 근사. n<4 또는 rho가 NaN이면 1.0."""
    if not math.isfinite(rho) or n < 4:
        return 1.0
    rho = max(-0.999999, min(0.999999, rho))
    z = math.sqrt((n - 3) / 1.06) * 0.5 * math.log((1 + rho) / (1 - rho))
    # 표준정규 양측 p (erf 기반, scipy 없이)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return max(0.0, min(1.0, p))


def _to_markdown(ruler_results: dict, adv_sim: dict) -> str:
    lines = ["| ruler | AUC(r_corr) | spearman(oracle) | holm_adj_p | attack outrank | pass_auc | pass_attack |",
            "|---|---|---|---|---|---|---|"]
    for name, res in ruler_results.items():
        lines.append(
            f"| {name} | {res['auc_all']['mean']:.3f} | {res['spearman_oracle']['mean']:.3f} | "
            f"{res.get('holm_adj_p', float('nan')):.3f} | {res['attack']['outrank_frac']:.3f} | "
            f"{res['pass_auc']} | {res['pass_attack']} |")
    lines.append("")
    lines.append("| reward formula | mean adv (oracle-good) | mean adv (oracle-bad) | n_good | n_bad |")
    lines.append("|---|---|---|---|---|")
    for name, res in adv_sim.items():
        lines.append(f"| {name} | {res['mean_adv_good']:.3f} | {res['mean_adv_bad']:.3f} | "
                     f"{res['n_good']} | {res['n_bad']} |")
    return "\n".join(lines)
