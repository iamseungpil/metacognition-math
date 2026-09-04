r"""hidden_probe 자 (신규) — 메타 끝의 은닉상태에서 "좋은 메타였는가"를 읽는 선형
프로브. 기존 25개 자는 전부 **디코딩된 텍스트**(logprob/KL) 위에서 움직였다;
이 자는 대신 은닉벡터 자체를 특징으로 쓴다. 검증되지 않은 신규 자다.

★"문제 단위(problem-grouped) k-fold" 인 이유: 같은 Countdown 문제(nums,target)가
  train/val 양쪽에 섞이면 프로브가 "이 문제는 쉽다/어렵다"는 문제 정체성을 외워
  일반화 성능을 부풀린다(`docs/POSTMORTEM_cd6_rulers_2026-09-03.md`가 반복 지적하는
  "정답 경로 누출"과 같은 종류의 사고). 그룹(=문제)이 fold 경계를 넘지 않게 한다.

★sklearn 없이 numpy로 로지스틱 회귀를 직접 푸는 이유: 이 저장소는 CPU 전용
  파이프라인에 무거운 의존을 추가하지 않는 관례다(`countdown_pmi.py`가 verl/ray를
  지연 import 하는 것과 같은 절제). 뉴턴-랩슨(IRLS) 몇 스텝이면 특징 수십 차원
  로지스틱 회귀는 sklearn과 사실상 같은 해에 수렴한다.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from src.rulers.base import MetaSample, Site

__all__ = ["HiddenAtMetaEnd", "fit_logreg", "probe_grouped_cv", "fit_probe", "probe_score"]

_NAN = float("nan")


class HiddenAtMetaEnd:
    r"""특징 추출기(자가 아니라 `hidden_probe`용 부속물). `score()`는 프로브가 이미
    학습돼 있을 때만 의미 있는 스칼라를 낸다 — 프로브 없이 부르면 NaN.

    `mean_last_n`을 주면 마지막 n개 레이어의 평균, 아니면 마지막 레이어.
    """
    name = "hidden_probe"
    needs_model = True

    def __init__(self, mean_last_n: Optional[int] = None, weights: Optional[np.ndarray] = None,
                 bias: float = 0.0, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None):
        self.mean_last_n = mean_last_n
        self.weights = weights
        self.bias = bias
        self.mu = mu
        self.sigma = sigma

    def extract(self, site: Site, sample: MetaSample, ctx) -> Optional[np.ndarray]:
        if ctx is None or not sample.meta_raw:
            return None
        cont = sample.continuation or ""
        me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else len(cont)
        prompt_text = ""
        if hasattr(ctx.tokenizer, "apply_chat_template"):
            prompt_text = ctx.tokenizer.apply_chat_template(
                site.prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + site.prefix + cont[:me]
        ids = list(ctx.encode(full_text))
        if not ids:
            return None
        return ctx.hidden_at(ids, token_index=len(ids) - 1, mean_last_n=self.mean_last_n)

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if self.weights is None:
            return _NAN
        feat = self.extract(site, sample, ctx)
        if feat is None:
            return _NAN
        x = feat
        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / np.where(self.sigma == 0, 1.0, self.sigma)
        z = float(np.dot(x, self.weights) + self.bias)
        return _sigmoid(z)


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, n_iter: int = 50,
              tol: float = 1e-8) -> tuple[np.ndarray, float]:
    r"""표준화된 특징 위 L2-정칙 로지스틱 회귀, IRLS(뉴턴-랩슨). 순수 numpy.

    Returns (weights[d], bias). `X`는 이미 표준화됐다고 가정(호출자가 mu/sigma 적용).
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    for _ in range(n_iter):
        z = Xb @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = Xb.T @ (y - p) - l2 * np.concatenate([beta[:-1], [0.0]])
        H = -(Xb * w[:, None]).T @ Xb - l2 * np.eye(d + 1) * np.array([1.0] * d + [0.0])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta[:-1], float(beta[-1])


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma_safe, mu, sigma


def fit_probe(features: np.ndarray, labels: Sequence[int], groups: Sequence,
              *, l2: float = 1e-2) -> dict:
    """전체 데이터로 최종 프로브를 학습(배포용). 5-fold CV 성능은 `probe_grouped_cv`."""
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    Xs, mu, sigma = _standardize(X)
    w, b = fit_logreg(Xs, y, l2=l2)
    return {"weights": w, "bias": b, "mu": mu, "sigma": sigma}


def probe_grouped_cv(features: np.ndarray, labels: Sequence[int], groups: Sequence,
                     *, n_folds: int = 5, l2: float = 1e-2, seed: int = 0) -> dict:
    r"""문제-그룹 k-fold CV. 그룹(=문제, 보통 (nums,target) 또는 site의 problem_id)이
    fold 경계를 넘지 않는다. Returns {"auc", "fold_aucs", "n_folds_used"}.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    groups = list(groups)
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    k = min(n_folds, len(uniq)) if uniq else 0
    if k < 2:
        return {"auc": _NAN, "fold_aucs": [], "n_folds_used": 0}
    fold_of = {}
    for i, g in enumerate(uniq):
        fold_of[g] = i % k
    fold_ids = np.array([fold_of[g] for g in groups])

    fold_aucs = []
    for f in range(k):
        te = fold_ids == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr.tolist())) < 2:
            continue
        Xtr, mu, sigma = _standardize(X[tr])
        w, b = fit_logreg(Xtr, y_tr, l2=l2)
        Xte = (X[te] - mu) / np.where(sigma == 0, 1.0, sigma)
        z = Xte @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        auc = _auc(y_te, p)
        if math.isfinite(auc):
            fold_aucs.append(auc)
    return {"auc": float(np.mean(fold_aucs)) if fold_aucs else _NAN,
            "fold_aucs": fold_aucs, "n_folds_used": len(fold_aucs)}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUC. 순수 numpy(sklearn 없음)."""
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return _NAN
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # 동점 처리: 평균 순위
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def probe_score(probe: dict, features: np.ndarray) -> np.ndarray:
    """학습된 프로브(`fit_probe`가 낸 dict)로 새 특징에 확률을 매긴다."""
    X = np.asarray(features, dtype=float)
    mu, sigma, w, b = probe["mu"], probe["sigma"], probe["weights"], probe["bias"]
    Xs = (X - mu) / np.where(sigma == 0, 1.0, sigma)
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
