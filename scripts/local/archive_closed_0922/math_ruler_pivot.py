#!/usr/bin/env python
r"""math_ruler_pivot — cd9 §2 «자(ruler) 기각 시험»: math_sites 가 낸 라벨(SAVE/DERAIL/Δ̂/
정답 판단) 위에서 후보 자들이 살아남는지 한 표로 판정한다.

입력(math_sites.py 산출): <sites_dir>/sites.jsonl, continuations.jsonl, 선택 --decoys(math_rollout).
출력: <out_dir>/ruler_table.json · ruler_table.md (+ 표를 stdout 에 마크다운으로).

자 종류
  per-site (자리 하나에 스칼라 하나) — 앞부분(prefix)만 본다:
    probe_cut@L        자리 마지막 토큰 은닉벡터 → SAVE(1)/DERAIL(0) 문제-그룹 5-fold 프로브
    probe_cut_dec@L    같은 벡터 → 정답 판단 redirect(1)/verify(0) 프로브
    fork_entropy_mean  자리 직전 64 토큰 평균 엔트로피(학습 없음)
    fork_entropy_max   같은 창의 최대 엔트로피
  per-meta (모델이 스스로 쓴 메타 하나에 스칼라 하나):
    probe_metaend@L    </meta> 토큰 은닉벡터 → SAVE/DERAIL 프로브 (+ _dec 판단 프로브)
    entropy_drop       메타 앞 32 토큰 평균 엔트로피 − 메타 뒤 32 토큰 평균 엔트로피
    stated_conf        메타가 쓴 confidence 값(parse_meta form="math")
    judgment_match     메타의 decision == 자리의 best_decision (1/0/NaN)
    pmi_shift          [log p(정답|앞+메타) − log p(디코이|앞+메타)] − [같은 것, 메타 없이]

★결과 고정(outcome-fixed) AUC 가 핵심 열인 이유: cd7 에서 25개 자가 전부 «정답 여부»를 되
  읽는 자였다(결과를 고정하면 AUC 가 .5 이하로 무너짐). 그래서 per-meta 자는 r_corr(그 이어쓰기
  의 정답 여부) 층 안에서, per-site 자는 p̂(nometa) 3분위 층 안에서 AUC 를 따로 재고 평균한다.
  r_corr 그 자체와 같은 자는 정의상 층 안에서 .5 가 된다(테스트가 이를 고정한다).
★donor 대조: 다른 문제의 완성 메타를 같은 자리에 심은 이어쓰기에서 같은 자를 재면, «메타 내용»
  을 보는 자는 donor 에서 점수가 낮아야 한다. «멈춤 효과»만 보는 자는 구별 못 한다.
★프로브 점수는 전부 out-of-fold — 같은 fold 배정(hidden_probe.probe_grouped_cv 와 동일 규칙:
  정렬·셔플·i%k)으로 학습 fold 밖 행에만 점수를 매기고, donor 행도 그 문제가 속한 fold 의
  프로브로 채점한다. in-sample 점수로 결과 고정/donor 비교를 하면 프로브가 유리하게 부풀려진다.
★하나의 자리(site_id="<group_id>@<cut>")는 문제 하나(group_id)에서 왔다. fold 경계·층은 전부
  group_id 단위 — 같은 문제의 다른 자리가 train/val 에 갈리면 문제 정체성 누출이다.

통과 규칙(docs/PREREGISTRATION_cd9_math_judgment.md §2, 셋 다):
    AUC(SAVE vs DERAIL) ≥ .65  ∧  결과 고정 AUC ≥ .55  ∧  donor 평균 < own 평균  → PASS

사용:
  python scripts/local/math_ruler_pivot.py --sites_dir <dir> --model_path <hf> --out_dir <dir> \
      [--decoys <rollout>/decoys.json] [--layer -1,mid] [--limit 20] [--max_conts_per_site 4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import render_generation_prompt as _gen_prompt  # noqa: E402
from src.rulers.hidden_probe import _auc, _standardize, fit_logreg, probe_grouped_cv  # noqa: E402
from src.training.countdown_rewards import parse_meta  # noqa: E402

_NAN = float("nan")
META_SEED = "\n<meta>\n"          # math_sites.build_fed("meta") 가 심는 문자열
META_CLOSE = "</meta>"
FORK_WINDOW = 64
DROP_WINDOW = 32
PASS_AUC, PASS_OFA = 0.65, 0.55


# ── 순수 헬퍼(테스트 대상) ────────────────────────────────────────────────────────
def group_of(site_id: str) -> str:
    """site_id "<group_id>@<cut>" → group_id. group_id 자체에 '@' 가 있어도 마지막 '@' 로 가른다."""
    return site_id.rsplit("@", 1)[0]


def extract_meta_block(cont: str) -> Optional[str]:
    """meta 모드 cont 는 "\\n<meta>\\n" 바로 뒤에서 시작한다. 첫 </meta> 까지(포함) 잘라 완성된
    블록 "<meta>\\n…</meta>" 로 돌려준다. 닫히지 않았으면 None(그 메타는 자에서 제외)."""
    j = cont.find(META_CLOSE)
    if j < 0:
        return None
    return "<meta>\n" + cont[:j + len(META_CLOSE)]


def auc(y: Sequence, s: Sequence) -> float:
    """NaN 점수 행은 버린 뒤 Mann-Whitney AUC. 한 쪽 클래스가 비면 NaN."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() == 0:
        return _NAN
    return _auc(y[ok], s[ok])


def outcome_fixed_auc(scores: Sequence, labels: Sequence, strata: Sequence) -> dict:
    """층(strata) 안에서 따로 잰 AUC 의 평균. 두 클래스가 다 있는 층만 센다.
    Returns {"auc", "per_stratum": {stratum: auc}, "n_strata"}."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    strata = np.asarray(list(strata))
    per = {}
    for st in sorted(set(strata.tolist()), key=str):
        m = strata == st
        a = auc(labels[m], scores[m])
        if math.isfinite(a):
            per[str(st)] = a
    return {"auc": float(np.mean(list(per.values()))) if per else _NAN,
            "per_stratum": per, "n_strata": len(per)}


def tertile_strata(values: Sequence) -> list[int]:
    """p̂(nometa) 3분위 층 번호(0/1/2). 동률이 많으면 층이 비어도 된다(빈 층은 위에서 건너뜀)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return []
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    return [0 if x <= q1 else (1 if x <= q2 else 2) for x in v]


def pass_rule(auc_sd: float, ofa: float, donor_lower: Optional[bool]) -> str:
    """사전등록 §2: 셋 다 만족해야 PASS. donor 비교가 정의되지 않는 자(None)는 그 절을
    «해당 없음»으로 두고 표에 n/a 로 드러낸다 — 통과를 주되 열에서 보이게."""
    if not (math.isfinite(auc_sd) and auc_sd >= PASS_AUC):
        return "FAIL"
    if not (math.isfinite(ofa) and ofa >= PASS_OFA):
        return "FAIL"
    if donor_lower is False:
        return "FAIL"
    return "PASS" if donor_lower else "PASS(donor n/a)"


def _group_fold_ids(groups: Sequence, n_folds: int, seed: int) -> tuple[np.ndarray, dict, int]:
    """★단일 스플리터 — grouped_oof_probe 와 grouped_oof_probe_mlp 가 **똑같이** 이 함수만
    부른다. 같은 seed 면 두 자가 같은 행을 같은 fold 에 넣어야 자 비교(선형 vs MLP)가 fold
    배정 차이가 아니라 머리(head) 차이만 반영한다. probe_grouped_cv 와 같은 규칙(정렬·셔플·i%k).
    Returns (fold_ids[len(groups)] — 배정 안 됐으면 -1, fold_of: group→fold, k)."""
    groups = list(groups)
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    k = min(n_folds, len(uniq)) if uniq else 0
    fold_of = {g: i % k for i, g in enumerate(uniq)} if k >= 2 else {}
    fold_ids = np.array([fold_of.get(g, -1) for g in groups]) if fold_of else np.full(len(groups), -1)
    return fold_ids, fold_of, k


def grouped_oof_probe(X: np.ndarray, y: np.ndarray, groups: Sequence, *,
                      X_other: Optional[np.ndarray] = None, groups_other: Sequence = (),
                      n_folds: int = 5, l2: float = 1e-2, seed: int = 0) -> dict:
    """probe_grouped_cv 와 같은 fold 배정으로 out-of-fold 확률을 낸다. X_other(donor 행)는
    그 group 이 test 에 든 fold 의 프로브로 채점한다(그 group 이 학습 group 에 없으면 NaN)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    fold_ids, fold_of, k = _group_fold_ids(groups, n_folds, seed)
    oof = np.full(len(y), _NAN)
    other = np.full(0 if X_other is None else len(X_other), _NAN)
    if k < 2:
        return {"oof": oof, "other": other, "n_folds_used": 0}
    fold_other = np.array([fold_of.get(g, -1) for g in groups_other]) if X_other is not None else None
    used = 0
    for f in range(k):
        te, tr = fold_ids == f, fold_ids != f
        if te.sum() == 0 or tr.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        Xtr, mu, sigma = _standardize(X[tr])
        w, b = fit_logreg(Xtr, y[tr], l2=l2)
        sig = np.where(sigma == 0, 1.0, sigma)

        def _p(Z):
            z = ((Z - mu) / sig) @ w + b
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        oof[te] = _p(X[te])
        if X_other is not None and (fold_other == f).any():
            other[fold_other == f] = _p(np.asarray(X_other, dtype=float)[fold_other == f])
        used += 1
    return {"oof": oof, "other": other, "n_folds_used": used}


# ── MLP 머리(2610.xxxxx HSRM 대조용) ──────────────────────────────────────────────
# ★왜 MLP 를 추가하는가: math_meta_probe.py 의 천장(.577)이 «신호가 없다» 가 아니라 «선형
#   256차원 사영 머리가 약하다» 일 수 있다 — HSRM 은 ~2M 파라미터 인코더로 .519→.724 를 낸다.
#   선형 머리 하나로 null 을 내면 그 둘을 못 가른다. 그래서 같은 grouped-OOF 규율 위에 조금
#   더 강한 머리를 하나 더 얹어, 천장이 머리 탓인지 신호 탓인지를 가른다.
# ★torch 를 끌어오지 않는다 — CPU analyze 단계는 torch 없는 env 에서도 돌아야 한다
#   (math_ruler_pivot.py 는 지금 numpy 뿐이고, hidden_probe.py 의 관례(순수 numpy IRLS)와 같다).


def _mlp_init(d: int, hidden: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    # ★He 초기화(scaled by fan-in) — Xavier(1/sqrt(fan_in))보다 tanh 은닉층 초기 그라디언트가
    #   덜 죽는다. 평문 GD 세대엔 최적화 스텝 수 자체가 병목이라 초기화 차이가 안 보였지만,
    #   Adam(아래)으로 스텝이 늘면 초기화가 처음 수렴 속도에 다시 영향을 준다.
    return {"W1": rng.normal(0.0, math.sqrt(2.0 / d), size=(d, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, math.sqrt(2.0 / hidden), size=(hidden,)),
            "b2": 0.0}


def _mlp_forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a1 = np.tanh(X @ params["W1"] + params["b1"])
    z2 = a1 @ params["W2"] + params["b2"]
    p = 1.0 / (1.0 + np.exp(-np.clip(z2, -60, 60)))
    return a1, p


def _mlp_fit(X: np.ndarray, y: np.ndarray, *, hidden: int, epochs: int, lr: float, l2: float,
            seed: int, beta1: float = 0.9, beta2: float = 0.999, adam_eps: float = 1e-8) -> dict:
    """1-은닉층(tanh) · 이진 교차엔트로피 · Adam(전량배치), 순수 numpy.
    ★왜 Adam: 평문 GD(lr=1e-2, epochs=200)는 XOR 처럼 곡률이 뒤틀린 손실면에서 200 스텝 안에
    거의 안 움직인다 — 독립 점검에서 grouped 5-fold OOF AUC .562(선형 .486 과 거의 같음)로
    실측됐다. 1차 모멘텀(진동 상쇄)과 2차 모멘트(파라미터별 스텝 크기 적응)를 더하면 같은
    은닉폭에서도 수백 스텝 안에 XOR 을 사실상 완전히 푼다(튜닝 기록은 이 함수를 부르는
    grouped_oof_probe_mlp 의 기본값 주석 참조)."""
    n, d = X.shape
    params = _mlp_init(d, max(1, hidden), seed)
    m = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0) for k, v in params.items()}
    v = {k: (np.zeros_like(vv) if isinstance(vv, np.ndarray) else 0.0) for k, vv in params.items()}
    for t in range(1, epochs + 1):
        a1, p = _mlp_forward(params, X)
        dz2 = (p - y) / n                                   # 평균 BCE 의 그라디언트
        gW2 = a1.T @ dz2 + l2 * params["W2"]
        gb2 = float(dz2.sum())
        da1 = np.outer(dz2, params["W2"])
        dz1 = da1 * (1.0 - a1 ** 2)                          # tanh'
        gW1 = X.T @ dz1 + l2 * params["W1"]
        gb1 = dz1.sum(axis=0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        for k in params:
            g = grads[k]
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
            mhat = m[k] / (1 - beta1 ** t)
            vhat = v[k] / (1 - beta2 ** t)
            params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + adam_eps)
    return params


def grouped_oof_probe_mlp(X: np.ndarray, y: np.ndarray, groups: Sequence, *, n_folds: int = 5,
                          hidden: int = 16, epochs: int = 800, lr: float = 1e-2, l2: float = 1e-4,
                          seed: int = 0) -> dict:
    """grouped_oof_probe 와 **같은 fold 배정**(_group_fold_ids, 같은 seed → 같은 행이 같은
    fold) · 같은 표준화 규율(학습 fold 통계만)을 쓰지만 머리는 1-은닉층 MLP(numpy, 전량배치
    Adam). Returns {"oof": ..., "n_folds_used": ...} — grouped_oof_probe 의 "oof" 와 같은 모양
    이라 math_meta_probe.py 가 한 분기로 갈아 낀다.
    ★기본값 튜닝 근거(독립 점검, n=1200·groups of 4·2-D features·grouped 5-fold·seed=0):
    hidden=16·epochs=800·lr=1e-2·l2=1e-4(Adam) → XOR OOF AUC ≈.999(선형은 같은 데이터에서
    ≈.51, 우연 수준), 선형 분리 표적 ≈1.000, 순수 잡음(라벨 셔플) OOF AUC ≈.48(≈.5 우연,
    신호 조작 없음). hidden 8~64 범위에서 결과가 안정적이라 가장 작은 16 을 골랐다(런타임·
    과적합 여유 둘 다 유리). epochs 800 은 XOR 수렴에 필요한 최소보다 여유 있게 잡은 값 —
    200(구 기본값)은 OOF AUC .562 로 미수렴이었다."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    fold_ids, _fold_of, k = _group_fold_ids(groups, n_folds, seed)
    oof = np.full(len(y), _NAN)
    if k < 2:
        return {"oof": oof, "n_folds_used": 0}
    used = 0
    for f in range(k):
        te, tr = fold_ids == f, fold_ids != f
        if te.sum() == 0 or tr.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        Xtr, mu, sigma = _standardize(X[tr])   # ★표준화 통계는 train fold 에서만(누출 금지)
        sig = np.where(sigma == 0, 1.0, sigma)
        params = _mlp_fit(Xtr, y[tr], hidden=hidden, epochs=epochs, lr=lr, l2=l2,
                          seed=seed * 1_000_003 + f)
        _a1, p_te = _mlp_forward(params, (X[te] - mu) / sig)
        oof[te] = p_te
        used += 1
    return {"oof": oof, "n_folds_used": used}


# ── 문맥 조립(math_sites 와 바이트 단위로 같은 프롬프트) ───────────────────────────────

class Job:
    """forward 한 번의 요청. ids 조각을 이어 붙이고 «어느 위치에서 무엇을 읽을지»를 지정한다.
    hidden_at: 은닉벡터를 읽을 토큰 인덱스 / ent_positions: 엔트로피를 읽을 위치들 /
    lp_spans: (start, end) — ids[start:end] 를 teacher-forcing 한 log p 합(정답·디코이 채점) /
    hidden_span: (start, end) — 그 구간 토큰들의 은닉벡터 **평균**(mean-pool)을 같은 forward 에서
      함께 읽는다(math_dist_effect 가 이어쓰기 전체의 평균 벡터를 쓰려고 추가). None 이면
      결과 dict 에 "hidden_mean" 키가 없다 — 기존 호출자는 영향받지 않는다.
    hidden_ats: 여러 토큰 자리의 은닉벡터를 **한 forward 로** 읽는다(math_meta_probe 가
      한 응답에서 meta_end/answer_start/last/post_answer 넷을 동시에 읽으려고 추가).
      인과 어텐션이라 자리 i 의 은닉은 i 이하 토큰에만 의존하므로, 가장 긴 시퀀스 하나를
      돌려 네 자리를 함께 읽는 것과 자리마다 따로 자른 시퀀스를 돌리는 것이 **같다**
      (자리당 따로 돌리면 forward 가 4배). 비면 결과 dict 에 "hidden_multi" 키가 없다.
    tok_lp_spans: (start, end) — lp_spans 와 같은 구간이되 **합이 아니라 토큰별** log p 를
      돌려준다(결과 dict 의 "tok_lp": [[float, ...] per span]). math_anti_teacher_ruler 가
      두 문맥의 토큰별 차 Δ_t 를 만들려고 추가했다. 비면 그 키가 없다 — 기존 호출자는 영향
      없다. 왼쪽 잘림으로 사라진 앞부분 토큰은 NaN 으로 채워 **길이를 보존**한다(두 문맥의
      Δ_t 를 자리 맞춰 빼려면 길이가 같아야 한다)."""

    def __init__(self, ids: list[int], hidden_at: Optional[int] = None,
                 ent_positions: Sequence[int] = (), lp_spans: Sequence[tuple[int, int]] = (),
                 hidden_span: Optional[tuple[int, int]] = None,
                 hidden_ats: Sequence[int] = (),
                 tok_lp_spans: Sequence[tuple[int, int]] = ()):
        self.ids = ids
        self.hidden_at = hidden_at
        self.ent_positions = [p for p in ent_positions if 0 <= p < len(ids)]
        self.lp_spans = list(lp_spans)
        self.hidden_span = hidden_span
        self.hidden_ats = [int(p) for p in hidden_ats if 0 <= int(p) < len(ids)]
        self.tok_lp_spans = list(tok_lp_spans)


ForwardFn = Callable[[list[Job], list[int]], list[dict]]
# ForwardFn(jobs, layers) -> per job {"hidden": {layer: np.ndarray}, "entropy": [float,...],
#                                    "lp": [float per span]}
# (hidden_span 을 준 job 에 한해 "hidden_mean": {layer: np.ndarray},
#  hidden_ats 를 준 job 에 한해 "hidden_multi": {pos: {layer: np.ndarray}},
#  tok_lp_spans 를 준 job 에 한해 "tok_lp": [[float per token] per span] 이 더 붙는다.)


def hf_forward_factory(model_path: str, device: str = "cuda", batch_size: int = 4,
                       max_len: int = 8192) -> tuple[ForwardFn, object, int]:
    """transformers bf16 forward. 길이순 정렬 배치, 필요한 위치의 로짓만 fp32 로 뽑는다
    (전체 vocab×T 로짓을 fp32 로 펼치면 시퀀스 하나에 GB 단위라 위치 선택이 필수)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    n_layers = int(model.config.num_hidden_layers)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    @torch.no_grad()
    def forward(jobs: list[Job], layers: list[int]) -> list[dict]:
        out: list[Optional[dict]] = [None] * len(jobs)
        order = sorted(range(len(jobs)), key=lambda i: len(jobs[i].ids))
        for b0 in range(0, len(order), batch_size):
            idx = order[b0:b0 + batch_size]
            seqs = [jobs[i].ids[-max_len:] for i in idx]
            shift = [len(jobs[i].ids) - len(s) for i, s in zip(idx, seqs)]  # 왼쪽 잘림 보정
            T = max(len(s) for s in seqs)
            inp = torch.full((len(seqs), T), pad_id, dtype=torch.long)
            att = torch.zeros((len(seqs), T), dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            res = model(input_ids=inp.to(device), attention_mask=att.to(device),
                        output_hidden_states=True)
            hs = res.hidden_states  # (n_layers+1) x [B,T,D]; hs[-1]=마지막, hs[L] = L번째 층 출력
            for r, i in enumerate(idx):
                j, sh = jobs[i], shift[r]
                rec: dict = {"hidden": {}, "entropy": [], "lp": []}
                if j.hidden_at is not None:
                    for L in layers:
                        rec["hidden"][L] = hs[L][r, j.hidden_at - sh].float().cpu().numpy()
                if j.hidden_ats:
                    # ★여러 자리를 한 forward 에서 읽는다. 왼쪽 잘림(sh)으로 사라진 자리는
                    #   **넣지 않는다** — 호출자가 «없음» 으로 세게 두지, 0 으로 메우지 않는다.
                    hm: dict[int, dict] = {}
                    for p in j.hidden_ats:
                        q = p - sh
                        if 0 <= q < len(seqs[r]):
                            hm[p] = {L: hs[L][r, q].float().cpu().numpy() for L in layers}
                    rec["hidden_multi"] = hm
                if j.hidden_span is not None:
                    # ★같은 forward 에서 구간 평균 벡터도 읽는다(왼쪽 잘림 sh 보정 후, 잘려
                    #   나간 앞부분은 빠진다 — 남은 구간이 없으면 키를 안 만든다).
                    s0 = max(j.hidden_span[0] - sh, 0)
                    s1 = min(j.hidden_span[1] - sh, len(seqs[r]))
                    if s1 > s0:
                        rec["hidden_mean"] = {
                            L: hs[L][r, s0:s1].float().mean(0).cpu().numpy() for L in layers}
                # ★엔트로피: 위치 t 의 로짓은 토큰 t+1 의 분포. «토큰 t 를 쓸 때의 불확실성»은
                #   위치 t-1 의 로짓이다.
                pos = [p - 1 - sh for p in j.ent_positions if p - 1 - sh >= 0]
                if pos:
                    lg = res.logits[r, pos].float()
                    lp_ = torch.log_softmax(lg, dim=-1)
                    rec["entropy"] = (-(lp_.exp() * lp_).sum(-1)).cpu().tolist()
                for (s0, s1) in j.lp_spans:
                    ps = list(range(max(s0, sh + 1), s1))
                    if not ps:
                        rec["lp"].append(_NAN)
                        continue
                    lg = torch.log_softmax(res.logits[r, [p - 1 - sh for p in ps]].float(), dim=-1)
                    tgt = torch.tensor([jobs[i].ids[p] for p in ps], device=lg.device)
                    rec["lp"].append(float(lg.gather(1, tgt[:, None]).sum()))
                for (s0, s1) in j.tok_lp_spans:
                    # ★토큰별 log p. 왼쪽 잘림으로 사라진 앞부분은 NaN 으로 채워 길이를 보존한다.
                    vals = [_NAN] * max(0, s1 - s0)
                    ps = list(range(max(s0, sh + 1), s1))
                    # ★fp32 log_softmax 를 구간 전체에 한 번에 걸면 [T, vocab] (3.5k×151k×4B
                    #   ≈ 2.1 GB)가 순간적으로 잡힌다 — 512 위치씩 끊어 ~300 MB 로 묶는다
                    #   (critique_scorer._CHUNK 와 같은 관례).
                    for c0 in range(0, len(ps), 512):
                        chunk = ps[c0:c0 + 512]
                        lg = torch.log_softmax(
                            res.logits[r, [p - 1 - sh for p in chunk]].float(), dim=-1)
                        tgt = torch.tensor([jobs[i].ids[p] for p in chunk], device=lg.device)
                        got = lg.gather(1, tgt[:, None]).squeeze(1).cpu().tolist()
                        for p, v in zip(chunk, got):
                            vals[p - s0] = float(v)
                        del lg
                    rec.setdefault("tok_lp", []).append(vals)
                out[i] = rec
            del res
        return out  # type: ignore[return-value]

    return forward, tok, n_layers


def mock_forward_factory(dim: int = 8, seed: int = 0) -> ForwardFn:
    """테스트용: 모델 없이 난수 특징. 파이프라인 배선만 검사한다."""
    rng = np.random.RandomState(seed)

    def forward(jobs: list[Job], layers: list[int]) -> list[dict]:
        res = []
        for j in jobs:
            rec = {"hidden": {L: rng.randn(dim) for L in layers} if j.hidden_at is not None else {},
                   "entropy": rng.rand(len(j.ent_positions)).tolist(),
                   "lp": [float(-rng.rand() * 5) for _ in j.lp_spans]}
            if j.hidden_span is not None:
                rec["hidden_mean"] = {L: rng.randn(dim) for L in layers}
            if j.hidden_ats:
                rec["hidden_multi"] = {p: {L: rng.randn(dim) for L in layers}
                                       for p in j.hidden_ats}
            if j.tok_lp_spans:
                rec["tok_lp"] = [[float(-rng.rand() * 5) for _ in range(max(0, s1 - s0))]
                                 for (s0, s1) in j.tok_lp_spans]
            res.append(rec)
        return res
    return forward


class MockTok:
    """테스트용 토크나이저: 공백 단위 해시. apply_chat_template 은 단순 접합."""

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 50000 for w in text.split()]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}> {m['content']} " for m in msgs) + "<assistant> "


# ── 특징 추출 ───────────────────────────────────────────────────────────────────
def _enc(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def build_jobs(tok, sites: list[dict], conts: list[dict], decoys: Optional[dict],
               max_conts_per_site: int) -> dict:
    """모든 forward 요청을 한 번에 만든다. Returns {"jobs": [...], "index": {...}}.
    index: site_jobs[site_id] = job#, meta_rows = [ {site_id, mode, r_corr, block, job_end, job_pmi_meta,
    job_pmi_nometa, conf, decision} ], 여기서 job_pmi_* 는 없으면 None."""
    jobs: list[Job] = []
    site_jobs: dict[str, int] = {}
    meta_rows: list[dict] = []
    by_site: dict[str, dict] = {s["site_id"]: s for s in sites}

    # (1) 자리 특징 — 롤아웃이 만들어진 문맥(math_plain) + 앞부분.
    for s in sites:
        ids = _enc(tok, _gen_prompt(tok, "math_plain", s["problem"]) + s["prefix"])
        n = len(ids)
        site_jobs[s["site_id"]] = len(jobs)
        jobs.append(Job(ids, hidden_at=n - 1, ent_positions=range(max(0, n - FORK_WINDOW), n)))

    # (2) 메타 특징 — meta 모드는 math_new 프롬프트로 생성됐다. donor 도 같은 프롬프트로 채점해
    #     «프롬프트 차이»가 아니라 «블록 내용 차이»만 자에 들어가게 한다.
    per_site_count: dict[tuple, int] = {}
    for c in conts:
        if c["mode"] not in ("meta", "donor"):
            continue
        s = by_site.get(c["site_id"])
        if s is None:
            continue
        key = (c["site_id"], c["mode"])
        if per_site_count.get(key, 0) >= max_conts_per_site:
            continue
        if c["mode"] == "meta":
            block = extract_meta_block(c["cont"])
            if block is None:
                continue
            after_text = c["cont"][len(block) - len("<meta>\n"):]   # </meta> 뒤 본문
        else:
            block = s.get("donor_meta")
            if not block:
                continue
            after_text = c["cont"]
        per_site_count[key] = per_site_count.get(key, 0) + 1
        pm = parse_meta(block, form="math")

        head = _enc(tok, _gen_prompt(tok, "math_new", s["problem"]) + s["prefix"].rstrip())
        blk = _enc(tok, "\n" + block)
        aft = _enc(tok, after_text)[:DROP_WINDOW]
        ids = head + blk + aft
        n_h, n_hb = len(head), len(head) + len(blk)
        ent_pos = list(range(max(0, n_h - DROP_WINDOW), n_h)) + list(range(n_hb, n_hb + len(aft)))
        row = {"site_id": c["site_id"], "mode": c["mode"], "r_corr": int(c["r_corr"]),
               "block": block, "conf": pm["confidence"], "decision": pm["decision"],
               "n_before": min(DROP_WINDOW, n_h), "job_end": len(jobs),
               "job_pmi_meta": None, "job_pmi_nometa": None}
        jobs.append(Job(ids, hidden_at=n_hb - 1, ent_positions=ent_pos))

        # (3) PMI-shift: "\boxed{ans}" 를 메타 뒤 / 메타 없이 각각 teacher-forcing.
        gid = group_of(c["site_id"])
        decoy = (decoys or {}).get(gid)
        if decoy is not None and str(decoy) != str(s["gold"]):
            g_ids = _enc(tok, "\\boxed{" + str(s["gold"]) + "}")
            d_ids = _enc(tok, "\\boxed{" + str(decoy) + "}")
            for tag, ctx in (("job_pmi_meta", head + blk + _enc(tok, "\n")),
                             ("job_pmi_nometa", head + _enc(tok, "\n"))):
                # 정답·디코이를 각각 이어붙인 두 시퀀스 → Job 둘. lp_spans 로 접미사만 채점.
                row[tag] = len(jobs)
                jobs.append(Job(ctx + g_ids, lp_spans=[(len(ctx), len(ctx) + len(g_ids))]))
                jobs.append(Job(ctx + d_ids, lp_spans=[(len(ctx), len(ctx) + len(d_ids))]))
        meta_rows.append(row)
    return {"jobs": jobs, "site_jobs": site_jobs, "meta_rows": meta_rows}


def collect_features(res: list[dict], built: dict, layers: list[int]) -> dict:
    """forward 결과를 자 입력으로 정리한다."""
    site_feat = {}
    for sid, j in built["site_jobs"].items():
        r = res[j]
        e = np.asarray(r["entropy"], dtype=float)
        site_feat[sid] = {"hidden": r["hidden"],
                          "fork_entropy_mean": float(e.mean()) if len(e) else _NAN,
                          "fork_entropy_max": float(e.max()) if len(e) else _NAN}
    meta_feat = []
    for m in built["meta_rows"]:
        r = res[m["job_end"]]
        e = np.asarray(r["entropy"], dtype=float)
        nb = m["n_before"]
        before, after = e[:nb], e[nb:]
        drop = (float(before.mean()) - float(after.mean())) if len(before) and len(after) else _NAN
        pmi = _NAN
        if m["job_pmi_meta"] is not None:
            jm, jn = m["job_pmi_meta"], m["job_pmi_nometa"]
            with_meta = res[jm]["lp"][0] - res[jm + 1]["lp"][0]
            without = res[jn]["lp"][0] - res[jn + 1]["lp"][0]
            pmi = float(with_meta - without)
        meta_feat.append({**{k: m[k] for k in ("site_id", "mode", "r_corr", "conf", "decision")},
                          "hidden": r["hidden"], "entropy_drop": drop, "pmi_shift": pmi})
    return {"site": site_feat, "meta": meta_feat}


# ── 자 평가 ─────────────────────────────────────────────────────────────────────
def _row(name, kind, n, auc_sd, ofa, own_mean, donor_mean, extra=None):
    donor_lower = None
    if own_mean is not None and donor_mean is not None and math.isfinite(own_mean) and math.isfinite(donor_mean):
        donor_lower = bool(donor_mean < own_mean)
    r = {"ruler": name, "kind": kind, "n": int(n), "auc_save_derail": auc_sd,
         "outcome_fixed_auc": ofa, "own_mean": own_mean, "donor_mean": donor_mean,
         "donor_lower": donor_lower, "verdict": pass_rule(auc_sd, ofa, donor_lower)}
    r.update(extra or {})
    return r


def evaluate_rulers(sites: list[dict], feats: dict, *, layers: list[int], seed: int = 0) -> list[dict]:
    rows: list[dict] = []
    by_site = {s["site_id"]: s for s in sites}
    sf = feats["site"]

    # ── per-site ─────────────────────────────────────────────────────────
    S = [s for s in sites if s["site_id"] in sf]
    sd = [s for s in S if s["label"] in ("SAVE", "DERAIL")]
    y_sd = np.array([1 if s["label"] == "SAVE" else 0 for s in sd])
    g_sd = [group_of(s["site_id"]) for s in sd]
    mov = [s for s in S if s.get("movable", 1)]
    y_dpos = np.array([1 if s["delta"] > 0 else 0 for s in mov])
    st_mov = tertile_strata([s["p_nometa"] for s in mov])
    g_mov = [group_of(s["site_id"]) for s in mov]
    dec = [s for s in S if s.get("best_decision") in ("verify", "redirect")]
    y_dec = np.array([1 if s["best_decision"] == "redirect" else 0 for s in dec])
    g_dec = [group_of(s["site_id"]) for s in dec]
    st_dec = tertile_strata([s["p_nometa"] for s in dec])

    for L in layers:
        X_sd = np.array([sf[s["site_id"]]["hidden"][L] for s in sd]) if sd else np.zeros((0, 1))
        a = probe_grouped_cv(X_sd, y_sd, g_sd, seed=seed)["auc"] if len(sd) else _NAN
        # ★결과 고정(per-site): p̂₀ 3분위 층 안에서 «Δ̂>0» 프로브의 OOF AUC. SAVE/DERAIL 라벨은
        #   p̂₀ 로 정의돼 층 안에 두 라벨이 공존할 수 없으므로 Δ̂>0 을 표적으로 쓴다.
        X_mov = np.array([sf[s["site_id"]]["hidden"][L] for s in mov]) if mov else np.zeros((0, 1))
        oof = grouped_oof_probe(X_mov, y_dpos, g_mov, seed=seed)["oof"] if len(mov) else np.zeros(0)
        ofa = outcome_fixed_auc(oof, y_dpos, st_mov)["auc"] if len(mov) else _NAN
        rows.append(_row(f"probe_cut@L{L}", "site", len(sd), a, ofa, None, None,
                         {"auc_delta_pos_oof": auc(y_dpos, oof) if len(mov) else _NAN}))
        if len(dec):
            X_dec = np.array([sf[s["site_id"]]["hidden"][L] for s in dec])
            a_dec = probe_grouped_cv(X_dec, y_dec, g_dec, seed=seed)["auc"]
            oof_d = grouped_oof_probe(X_dec, y_dec, g_dec, seed=seed)["oof"]
            ofa_d = outcome_fixed_auc(oof_d, y_dec, st_dec)["auc"]
            rows.append(_row(f"probe_cut_dec@L{L}", "site", len(dec), a_dec, ofa_d, None, None,
                             {"target": "best_decision redirect=1"}))

    for name in ("fork_entropy_mean", "fork_entropy_max"):
        sc_sd = [sf[s["site_id"]][name] for s in sd]
        sc_all = [sf[s["site_id"]][name] for s in S]
        y_save_rest = [1 if s["label"] == "SAVE" else 0 for s in S]
        sc_mov = [sf[s["site_id"]][name] for s in mov]
        rows.append(_row(name, "site", len(S), auc(y_sd, sc_sd),
                         outcome_fixed_auc(sc_mov, y_dpos, st_mov)["auc"], None, None,
                         {"auc_save_vs_rest": auc(y_save_rest, sc_all)}))

    # ── per-meta ─────────────────────────────────────────────────────────
    own = [m for m in feats["meta"] if m["mode"] == "meta" and m["site_id"] in by_site]
    don = [m for m in feats["meta"] if m["mode"] == "donor" and m["site_id"] in by_site]
    lab = [by_site[m["site_id"]]["label"] for m in own]
    sd_ix = [i for i, l in enumerate(lab) if l in ("SAVE", "DERAIL")]
    y_sd_m = np.array([1 if lab[i] == "SAVE" else 0 for i in sd_ix])
    g_own = [group_of(m["site_id"]) for m in own]
    y_dpos_m = np.array([1 if by_site[m["site_id"]]["delta"] > 0 else 0 for m in own])
    y_rc = np.array([m["r_corr"] for m in own])
    dec_ix = [i for i, m in enumerate(own) if by_site[m["site_id"]].get("best_decision") in ("verify", "redirect")]
    y_dec_m = np.array([1 if by_site[own[i]["site_id"]]["best_decision"] == "redirect" else 0 for i in dec_ix])

    def _judgment(m):
        bd = by_site[m["site_id"]].get("best_decision")
        if m["decision"] is None or bd not in ("verify", "redirect"):
            return _NAN
        return 1.0 if m["decision"] == bd else 0.0

    scalar = {
        "entropy_drop": lambda m: m["entropy_drop"],
        "stated_conf": lambda m: _NAN if m["conf"] is None else float(m["conf"]),
        "judgment_match": _judgment,
        "pmi_shift": lambda m: m["pmi_shift"],
    }
    for name, fn in scalar.items():
        s_own = np.array([fn(m) for m in own], dtype=float)
        s_don = np.array([fn(m) for m in don], dtype=float)
        a_sd = auc(y_sd_m, s_own[sd_ix]) if sd_ix else _NAN
        ofa = outcome_fixed_auc(s_own, y_dpos_m, y_rc)["auc"] if len(own) else _NAN
        extra = {"auc_delta_pos": auc(y_dpos_m, s_own) if len(own) else _NAN,
                 "n_donor": int(np.isfinite(s_don).sum())}
        if name == "pmi_shift":
            extra["auc_rcorr"] = auc(y_rc, s_own) if len(own) else _NAN
        rows.append(_row(name, "meta", int(np.isfinite(s_own).sum()), a_sd, ofa,
                         float(np.nanmean(s_own)) if np.isfinite(s_own).any() else _NAN,
                         float(np.nanmean(s_don)) if np.isfinite(s_don).any() else _NAN, extra))

    for L in layers:
        if not own:
            break
        X_own = np.array([m["hidden"][L] for m in own])
        X_don = np.array([m["hidden"][L] for m in don]) if don else None
        g_don = [group_of(m["site_id"]) for m in don]
        a_sd = (probe_grouped_cv(X_own[sd_ix], y_sd_m, [g_own[i] for i in sd_ix], seed=seed)["auc"]
                if sd_ix else _NAN)
        # OOF 점수는 Δ̂>0 표적으로 전 own 행에 매긴다(SAVE/DERAIL 만으로는 r_corr 층이 성기다).
        pr = grouped_oof_probe(X_own, y_dpos_m, g_own, X_other=X_don, groups_other=g_don, seed=seed)
        ofa = outcome_fixed_auc(pr["oof"], y_dpos_m, y_rc)["auc"]
        rows.append(_row(f"probe_metaend@L{L}", "meta", len(own), a_sd, ofa,
                         float(np.nanmean(pr["oof"])) if np.isfinite(pr["oof"]).any() else _NAN,
                         float(np.nanmean(pr["other"])) if np.isfinite(pr["other"]).any() else _NAN,
                         {"auc_delta_pos_oof": auc(y_dpos_m, pr["oof"]), "n_donor": len(don)}))
        if dec_ix:
            Xd = X_own[dec_ix]
            gd = [g_own[i] for i in dec_ix]
            a_dec = probe_grouped_cv(Xd, y_dec_m, gd, seed=seed)["auc"]
            oof_d = grouped_oof_probe(Xd, y_dec_m, gd, seed=seed)["oof"]
            ofa_d = outcome_fixed_auc(oof_d, y_dec_m, y_rc[dec_ix])["auc"]
            rows.append(_row(f"probe_metaend_dec@L{L}", "meta", len(dec_ix), a_dec, ofa_d, None, None,
                             {"target": "best_decision redirect=1"}))
    return rows


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.3f}"
    return str(v)


def to_markdown(rows: list[dict]) -> str:
    cols = ["ruler", "kind", "n", "auc_save_derail", "outcome_fixed_auc", "own_mean", "donor_mean",
            "donor_lower", "verdict"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    lines.append("")
    lines.append(f"pass rule: auc_save_derail ≥ {PASS_AUC} ∧ outcome_fixed_auc ≥ {PASS_OFA} ∧ donor_mean < own_mean")
    return "\n".join(lines)


def run_pivot(sites: list[dict], conts: list[dict], decoys: Optional[dict], forward: ForwardFn,
              tok, layers: list[int], *, max_conts_per_site: int = 4, seed: int = 0) -> dict:
    built = build_jobs(tok, sites, conts, decoys, max_conts_per_site)
    res = forward(built["jobs"], layers)
    feats = collect_features(res, built, layers)
    rows = evaluate_rulers(sites, feats, layers=layers, seed=seed)
    return {"rows": rows, "n_jobs": len(built["jobs"]), "n_meta_rows": len(built["meta_rows"]),
            "n_sites": len(sites)}


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """'-1,mid' → [n_layers, n_layers//2] (hidden_states 인덱스: 0=임베딩, n_layers=마지막)."""
    out = []
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        if t == "mid":
            out.append(n_layers // 2)
        else:
            v = int(t)
            out.append(v if v >= 0 else n_layers + 1 + v)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--decoys", default=None, help="math_rollout decoys.json {group_id: wrong_answer}")
    ap.add_argument("--layer", default="-1,mid", help="쉼표 구분. -1=마지막, mid=중간, 정수=hidden_states 인덱스")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="smoke: 앞 N 자리만")
    ap.add_argument("--max_conts_per_site", type=int, default=4, help="자리당 meta/donor 이어쓰기 상한")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sd = Path(a.sites_dir)
    sites = [json.loads(l) for l in open(sd / "sites.jsonl")]
    if a.limit:
        sites = sites[:a.limit]
    keep = {s["site_id"] for s in sites}
    conts = [r for r in (json.loads(l) for l in open(sd / "continuations.jsonl")) if r["site_id"] in keep]
    decoys = json.loads(Path(a.decoys).read_text()) if a.decoys else None
    print(f"[pivot] 자리 {len(sites)} · 이어쓰기 {len(conts)} · 디코이 {len(decoys) if decoys else 0}", flush=True)

    forward, tok, n_layers = hf_forward_factory(a.model_path, a.device, a.batch_size, a.max_len)
    layers = parse_layers(a.layer, n_layers)
    print(f"[pivot] layers(hidden_states idx) {layers} / n_layers {n_layers}", flush=True)
    out = run_pivot(sites, conts, decoys, forward, tok, layers,
                    max_conts_per_site=a.max_conts_per_site, seed=a.seed)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "ruler_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    md = to_markdown(out["rows"])
    (od / "ruler_table.md").write_text(md)
    print(md)
    print(f"[out] {od}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
