r"""retry_metrics — M_RETRY 의 **within-problem 선택성** 지표(0914 사전등록 수정 후속).

★왜 별도 모듈인가. `math_meta.retry_telemetry` 의 redirect_rate_given_wrong/right 는 **문제를
가리지 않고** 전체 행 위에서 잰다 — 그래서 «어려운 문제일수록 redirect 가 많다»(난이도 신호)와
«같은 문제 안에서 틀렸을 때만 redirect 한다»(진짜 선택성) 를 구별하지 못한다. 이 모듈은 K 샘플이
있는 문제 단위로 판단력을 잰다:
  1. mixed_selectivity   — 결과가 흔들리는 문제(0<first_pass_rate<1)로만 제한한 redirect_rate
                           given wrong/right 와 그 차(선택성). 문제 난이도가 섞여도 각 문제
                           안에서 wrong/right 를 비교하므로 난이도 자체는 통제된다.
  2. per_problem_auc     — 각 mixed 문제 안에서 decision==redirect 를 wrong(양성)/right(음성)로
                           가르는 Mann-Whitney AUC. 문제 평균 AUC 와 AUC>0.5 문제 비율.
  3. difficulty_bucket_lift — 문제를 first_pass_rate 로 {0, (0,.5], (.5,1), 1} 버킷화해
                           버킷별 final_acc − first_acc(재시도가 실제로 정확도를 올렸는가).
  4. judgment_acc_mixed  — judgment_acc(=«틀리면 redirect, 맞으면 verify» 를 맞힌 비율)를
                           mixed 문제로만 제한한 값.

행/그룹 키는 호출자가 정한다(`group_key`, 기본 "group_id") — held-out 평가(`math_retry_eval.py`)는
문제당 K 샘플이 실제로 있는 group_id 를 쓰고, 학습 텔레메트리(`math_meta.retry_telemetry`)는
배치 안 같은 프롬프트의 롤아웃을 묶는 uid 를 쓴다. `pass_rate` 를 안 주면 행 자체의
`first_correct` 로 배치-로컬 pass_rate 를 계산하고, 주면(예: build_math_parquet.py 가 붙인
`group_pass_rate` — 오프라인 롤아웃에서 미리 잰 값) 그것을 우선한다(배치에 K 샘플이 다 없어도
mixed 여부를 알 수 있다).

순수 함수 — 생성기·GPU 불필요, 단위 테스트 가능.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

_NAN = float("nan")


def _bool01(x) -> int:
    return int(bool(x)) if x is not None else 0


def group_first_pass_rate(rows: Sequence[Mapping], *, group_key: str = "group_id") -> dict[str, float]:
    """group_key → 배치-로컬 first_correct 평균(«이 배치/이 평가에서 관측된» pass rate)."""
    g: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        g[str(r[group_key])].append(_bool01(r.get("first_correct", 0)))
    return {k: sum(v) / len(v) for k, v in g.items()}


def _resolve_mixed_keys(rows: Sequence[Mapping], *, group_key: str,
                        pass_rate: Mapping[str, float] | None) -> tuple[dict[str, float], set[str]]:
    """(key→rate, mixed 키 집합). pass_rate 가 주어지면 그것을 쓰고(None 값은 무시), 아니면
    배치-로컬 first_correct 로 계산한다."""
    if pass_rate is not None:
        rates: dict[str, float] = {}
        for r in rows:
            k = str(r[group_key])
            v = pass_rate.get(k)
            if v is None:
                v = r.get("group_pass_rate")
            if v is not None:
                rates[k] = float(v)
    else:
        rates = group_first_pass_rate(rows, group_key=group_key)
    mixed = {k for k, v in rates.items() if 0.0 < v < 1.0}
    return rates, mixed


def _auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney U 기반 AUC. labels: 1=양성(wrong), 0=음성(right). scores 는 낮을수록 음성으로
    예측(여기선 decision==redirect indicator, 0/1). 동점(대부분 0/1 두 값)은 평균 순위로 처리한다.
    한쪽 클래스가 없으면 정의 불가 → NaN."""
    n = len(labels)
    if n == 0:
        return _NAN
    n_pos = sum(1 for x in labels if x)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return _NAN
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0        # 1-indexed 평균 순위(동점 구간)
        for t in range(i, j):
            ranks[order[t]] = avg_rank
        i = j
    sum_ranks_pos = sum(ranks[i] for i in range(n) if labels[i])
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def mixed_problem_selectivity(rows: Sequence[Mapping], *, group_key: str = "group_id",
                              pass_rate: Mapping[str, float] | None = None) -> dict:
    """mixed 문제(0<pass_rate<1)로만 제한한 redirect_rate given wrong/right 와 그 차."""
    rates, mixed = _resolve_mixed_keys(rows, group_key=group_key, pass_rate=pass_rate)
    mixed_rows = [r for r in rows if str(r[group_key]) in mixed]
    is_red = lambda r: r.get("decision") == "redirect"
    wrong = [r for r in mixed_rows if not _bool01(r.get("first_correct", 0))]
    right = [r for r in mixed_rows if _bool01(r.get("first_correct", 0))]
    _rate = lambda xs: (sum(1 for r in xs if is_red(r)) / len(xs)) if xs else _NAN
    rr_wrong, rr_right = _rate(wrong), _rate(right)
    sel = (rr_wrong - rr_right) if (rr_wrong == rr_wrong and rr_right == rr_right) else _NAN
    return {
        "n_mixed_problems": len(mixed), "n_mixed_rows": len(mixed_rows),
        "redirect_rate_given_wrong_mixed": rr_wrong,
        "redirect_rate_given_right_mixed": rr_right,
        "selectivity_mixed": sel,
    }


def per_problem_auc(rows: Sequence[Mapping], *, group_key: str = "group_id",
                    pass_rate: Mapping[str, float] | None = None) -> dict:
    """mixed 문제마다 decision==redirect indicator 로 wrong(양성)/right(음성) 를 가르는 AUC.
    문제 평균 AUC 와 AUC>0.5 인 문제 비율(둘 다 클래스가 있는 문제만 — 없으면 그 문제는 뺀다)."""
    rates, mixed = _resolve_mixed_keys(rows, group_key=group_key, pass_rate=pass_rate)
    by_group: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        by_group[str(r[group_key])].append(r)
    aucs = []
    for k in mixed:
        grp = by_group.get(k, [])
        labels = [0 if _bool01(r.get("first_correct", 0)) else 1 for r in grp]   # wrong=1(양성)
        scores = [1.0 if r.get("decision") == "redirect" else 0.0 for r in grp]
        a = _auc_binary(labels, scores)
        if a == a:
            aucs.append(a)
    mean_auc = (sum(aucs) / len(aucs)) if aucs else _NAN
    frac_gt_half = (sum(1 for a in aucs if a > 0.5) / len(aucs)) if aucs else _NAN
    return {"n_problems_auc": len(aucs), "mean_within_problem_auc": mean_auc,
            "frac_problems_auc_gt_half": frac_gt_half}


_BUCKETS = ("p0", "p0_50", "p50_100", "p100")


def _bucket_of(rate: float) -> str:
    if rate <= 0.0:
        return "p0"
    if rate <= 0.5:
        return "p0_50"
    if rate < 1.0:
        return "p50_100"
    return "p100"


def difficulty_bucket_lift(rows: Sequence[Mapping], *, group_key: str = "group_id",
                           pass_rate: Mapping[str, float] | None = None) -> dict:
    """문제를 first_pass_rate 로 {0, (0,.5], (.5,1), 1} 버킷화해 버킷별 final_acc − first_acc
    (행 평균; «재시도가 실제로 정확도를 끌어올렸는가»)."""
    rates, _ = _resolve_mixed_keys(rows, group_key=group_key, pass_rate=pass_rate)
    out = {}
    buckets: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        k = str(r[group_key])
        rate = rates.get(k)
        if rate is None:
            continue
        buckets[_bucket_of(rate)].append(r)
    for b in _BUCKETS:
        xs = buckets.get(b, [])
        if not xs:
            out[f"retry_lift_{b}"] = _NAN
            out[f"n_rows_{b}"] = 0
            continue
        first_acc = sum(_bool01(r.get("first_correct", 0)) for r in xs) / len(xs)
        final_acc = sum(_bool01(r.get("final_correct", 0)) for r in xs) / len(xs)
        out[f"retry_lift_{b}"] = final_acc - first_acc
        out[f"n_rows_{b}"] = len(xs)
    return out


def judgment_acc_mixed(rows: Sequence[Mapping], *, group_key: str = "group_id",
                       pass_rate: Mapping[str, float] | None = None) -> float:
    """judgment_acc(=«틀리면 redirect, 맞으면 verify» 를 맞힌 비율, 결정이 파싱된 행만)를
    mixed 문제로만 제한한 값."""
    rates, mixed = _resolve_mixed_keys(rows, group_key=group_key, pass_rate=pass_rate)
    dec = [r for r in rows if str(r[group_key]) in mixed and r.get("decision") in ("verify", "redirect")]
    if not dec:
        return _NAN
    correct = sum(1 for r in dec
                 if (r.get("decision") == "redirect") != bool(_bool01(r.get("first_correct", 0))))
    return correct / len(dec)


def all_metrics(rows: Sequence[Mapping], *, group_key: str = "group_id",
               pass_rate: Mapping[str, float] | None = None) -> dict:
    """위 네 지표를 한 번에(호출자가 하나로 합쳐 텔레메트리/요약에 넣기 편하도록)."""
    out = {}
    out.update(mixed_problem_selectivity(rows, group_key=group_key, pass_rate=pass_rate))
    out.update(per_problem_auc(rows, group_key=group_key, pass_rate=pass_rate))
    out.update(difficulty_bucket_lift(rows, group_key=group_key, pass_rate=pass_rate))
    out["judgment_acc_mixed"] = judgment_acc_mixed(rows, group_key=group_key, pass_rate=pass_rate)
    return out
