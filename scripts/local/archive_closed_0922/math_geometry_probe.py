#!/usr/bin/env python
r"""math_geometry_probe — 저장된 math_end_probe 특징(features.npz) 위에서 «그룹 기하»가
정오(r_corr)를 읽는가를 잰다. 새 forward pass 없음 — math_end_probe.py 가 이미 뽑아 둔
hidden_metaend/hidden_last 벡터만 쓴다(CPU 전용).

왜 기하인가. math_agree_probe 는 «다수 답과 다른가»(disagree, 답 문자열)를 표적으로 삼았다.
여기선 **은닉공간에서 형제(같은 문제의 다른 롤아웃)들과 얼마나 떨어져 있는가** 자체를 자로
쓴다 — 학습 없는 자다. 표적은 wrongness(=1−r_corr).

재는 것(전부 «흔들리는» 그룹, 0<group pass rate<1 안에서):
  (a) cos_lso   자기 제외(leave-self-out) 그룹 중심과의 코사인 거리 — 멀수록 틀렸다(가설).
  (b) l2_lso    같은 것의 L2 거리.
  (c) resid_probe  난이도 잔차 프로브. 은닉상태를 **그룹 정답률 p 한 방향**에 회귀시켜
                (차원마다 OLS: x ≈ a + b·p) 잔차를 만들고, 그 잔차 위에서 r_corr 을
                그룹-OOF 프로브(math_ruler_pivot.grouped_oof_probe)로 읽는다. 「문제가
                쉬운가」를 제거하고도 남는 신호가 있는지를 본다.
                ★회귀는 npz 의 **모든** 행으로 적합한다(라벨 r_corr 은 안 쓰고 그룹 수준
                난이도 p 만 쓴다). 평가는 mixed 행에서만.
  (d) eigenscore  EigenScore 풍 그룹 퍼짐: 그룹의 형제 벡터 k개를 중심화해 만든 k×k
                공분산(+αI)의 **고유값 로그 평균**(= log-det/k). 그룹 정답률과의
                Spearman·Pearson 상관을 낸다(퍼진 그룹이 어려운 그룹인가).
  (e) n 행 / n 문제 (a~c 는 k≥2 인 mixed 그룹의 행만 쓴다).

AUC 는 두 가지로 보고한다:
  auc_per_problem  문제마다 따로 잰 Mann-Whitney AUC 의 평균(두 클래스가 다 있는 문제만) —
                   math_ruler_pivot.outcome_fixed_auc(strata=group_id) 재사용.
  auc_pooled       mixed 행을 한 덩어리로 본 AUC.

그룹 정답률(p̂) 출처: --rollouts 를 주면 math_end_probe.group_pass_rates(원본 전체 행) —
math_end_probe/math_agree_probe 와 같은 정의다. 안 주면 npz 의 target 으로 그룹 평균을
계산한다(메타를 낸 행만의 정답률이라 정의가 다르다 — 표/JSON 에 출처를 찍는다).

사용:
  python scripts/local/math_geometry_probe.py \
      --features <out>/features.npz [--rollouts <dir>/texts.jsonl] --out_dir <out>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_ruler_pivot as P  # noqa: E402  (auc/outcome_fixed_auc/grouped_oof_probe 재사용)
import math_end_probe as E  # noqa: E402  (group_pass_rates 재사용)
import math_agree_probe as A  # noqa: E402  (per_problem_auc 래퍼 재사용 — 정의를 한 곳에 둔다)

_NAN = float("nan")
EIG_ALPHA = 1e-3      # EigenScore 정칙화(INSIDE 식 log det(Σ+αI))


# ── 순수 헬퍼(테스트 대상) ────────────────────────────────────────────────────────
def leave_self_out_distances(X: np.ndarray, groups: Sequence) -> dict:
    """행마다 «자기를 뺀 같은 그룹 중심»과의 (코사인 거리, L2 거리).

    그룹 크기가 1 이면 중심이 정의되지 않으므로 NaN(호출자가 그 행을 뺀다).
    코사인 거리 = 1 − cos(x_i, centroid_{-i}); 둘 중 하나가 영벡터면 NaN.
    """
    X = np.asarray(X, dtype=float)
    groups = list(groups)
    n = len(groups)
    cos = np.full(n, _NAN)
    l2 = np.full(n, _NAN)
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    for g, ii in idx.items():
        k = len(ii)
        if k < 2:
            continue
        sub = X[ii]
        tot = sub.sum(axis=0)
        for j, i in enumerate(ii):
            c = (tot - sub[j]) / (k - 1)
            d = sub[j] - c
            l2[i] = float(np.sqrt((d * d).sum()))
            na, nb = float(np.linalg.norm(sub[j])), float(np.linalg.norm(c))
            if na > 0 and nb > 0:
                cos[i] = 1.0 - float(sub[j] @ c) / (na * nb)
    return {"cos": cos, "l2": l2}


def residualize_on_scalar(X: np.ndarray, s: Sequence) -> np.ndarray:
    """차원마다 x ≈ a + b·s 로 OLS 적합한 뒤의 잔차(한 방향 = 난이도 방향 제거).

    s 의 분산이 0 이면(전부 같은 난이도) 평균만 빼고 돌려준다.
    """
    X = np.asarray(X, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    mu_s = float(s[ok].mean()) if ok.any() else 0.0
    sc = np.where(ok, s, mu_s) - mu_s
    var = float((sc * sc).sum())
    Xm = X - X.mean(axis=0, keepdims=True)
    if var <= 0:
        return Xm
    b = (sc[:, None] * Xm).sum(axis=0) / var          # 차원별 기울기
    return Xm - np.outer(sc, b)


def eigenscore(V: np.ndarray, alpha: float = EIG_ALPHA) -> float:
    """EigenScore 풍 그룹 퍼짐: 중심화한 k×D 행렬의 k×k 공분산(+αI) 고유값 **로그 평균**
    (= log det / k). k<2 면 NaN. 큰 값 = 형제들이 서로 멀다(퍼져 있다)."""
    V = np.asarray(V, dtype=float)
    k = V.shape[0]
    if k < 2:
        return _NAN
    Z = V - V.mean(axis=0, keepdims=True)
    C = (Z @ Z.T) / (k - 1)
    w = np.linalg.eigvalsh(C + alpha * np.eye(k))
    w = np.clip(w, 1e-300, None)
    return float(np.log(w).mean())


def group_eigenscores(X: np.ndarray, groups: Sequence, alpha: float = EIG_ALPHA) -> dict:
    """group_id → eigenscore(형제 벡터들). k<2 인 그룹은 뺀다."""
    X = np.asarray(X, dtype=float)
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    return {g: eigenscore(X[ii], alpha) for g, ii in idx.items() if len(ii) >= 2}


def _rank(v: np.ndarray) -> np.ndarray:
    """동점은 평균 순위."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and v[order[j]] == v[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return r


def pearson(a: Sequence, b: Sequence) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return _NAN
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float(a @ b) / d if d > 0 else _NAN


def spearman(a: Sequence, b: Sequence) -> float:
    """순위 Pearson(scipy 없이 — qwen35 env 에 scipy 가 없다)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return _NAN
    return pearson(_rank(a[ok]), _rank(b[ok]))


# ── 평가 ────────────────────────────────────────────────────────────────────────
def _auc_pair(scores: np.ndarray, wrong: np.ndarray, groups: np.ndarray) -> dict:
    """NaN 점수 행을 뺀 뒤 (문제별 평균 AUC, pooled AUC, n, n_problems)."""
    ok = np.isfinite(scores)
    s, y, g = scores[ok], wrong[ok], groups[ok]
    ppa = A.per_problem_auc(s, y, g) if len(s) else {"auc": _NAN, "n_strata": 0}
    return {"auc_per_problem": ppa["auc"], "n_problems": ppa["n_strata"],
            "auc_pooled": P.auc(y, s) if len(s) else _NAN, "n": int(len(s))}


def evaluate_geometry(X: np.ndarray, y_corr: np.ndarray, groups: np.ndarray,
                      pass_rates: dict, *, layer: int, position: str,
                      seed: int = 0, alpha: float = EIG_ALPHA) -> tuple[list[dict], dict]:
    """한 (layer, position) 특징 행렬에 대한 표 rows 와 eigenscore 상관 dict."""
    wrong_all = 1.0 - np.asarray(y_corr, dtype=float)
    p_all = np.array([pass_rates.get(g, _NAN) for g in groups], dtype=float)
    mixed = np.isfinite(p_all) & (p_all > 0.0) & (p_all < 1.0)

    tag = {"layer": layer, "position": position}
    rows: list[dict] = []

    # (a)(b) leave-self-out 거리 — mixed 행 안에서만 중심을 잡는다(그 밖 행은 안 쓴다).
    Xm, ym, gm = X[mixed], wrong_all[mixed], groups[mixed]
    d = leave_self_out_distances(Xm, gm.tolist())
    for name, sc in (("cos_lso", d["cos"]), ("l2_lso", d["l2"])):
        rows.append({**tag, "ruler": name, **_auc_pair(sc, ym, gm)})

    # (c) 난이도 잔차 프로브 — 회귀는 전체 행(라벨 안 씀), 평가는 mixed 행.
    R = residualize_on_scalar(X, p_all)
    pr = P.grouped_oof_probe(R[mixed], np.asarray(y_corr, dtype=float)[mixed], gm.tolist(),
                             seed=seed)
    oof = pr["oof"]
    # 표적은 r_corr(높을수록 정답). 위 두 자와 방향을 맞추려 wrongness 기준으로도 읽히게
    # 점수를 뒤집지 않고 그대로 r_corr 표적으로 잰다 — 열 이름(target)으로 구분한다.
    ok = np.isfinite(oof)
    ppa = A.per_problem_auc(oof[ok], np.asarray(y_corr, dtype=float)[mixed][ok], gm[ok]) \
        if ok.any() else {"auc": _NAN, "n_strata": 0}
    rows.append({**tag, "ruler": "resid_probe(target=r_corr)",
                 "auc_per_problem": ppa["auc"], "n_problems": ppa["n_strata"],
                 "auc_pooled": P.auc(np.asarray(y_corr, dtype=float)[mixed][ok], oof[ok])
                 if ok.any() else _NAN,
                 "n": int(ok.sum()), "n_folds_used": pr["n_folds_used"]})

    # (d) EigenScore 그룹 퍼짐 ↔ 그룹 정답률
    es_all = group_eigenscores(X, groups.tolist(), alpha)
    gs_all = sorted(es_all)
    v_all = [es_all[g] for g in gs_all]
    p_of = [pass_rates.get(g, _NAN) for g in gs_all]
    mixed_g = [i for i, g in enumerate(gs_all)
               if math.isfinite(p_of[i]) and 0.0 < p_of[i] < 1.0]
    eig = {
        **tag,
        "n_groups_all(k>=2)": len(gs_all),
        "spearman_eig_vs_pass_all": spearman(v_all, p_of),
        "pearson_eig_vs_pass_all": pearson(v_all, p_of),
        "n_groups_mixed(k>=2)": len(mixed_g),
        "spearman_eig_vs_pass_mixed": spearman([v_all[i] for i in mixed_g],
                                               [p_of[i] for i in mixed_g]),
        "pearson_eig_vs_pass_mixed": pearson([v_all[i] for i in mixed_g],
                                             [p_of[i] for i in mixed_g]),
        "mean_eigenscore_all": float(np.mean(v_all)) if v_all else _NAN,
    }
    return rows, eig


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.3f}"
    return str(v)


def to_markdown(rows: list[dict], eig_rows: list[dict], base: dict) -> str:
    cols = ["layer", "position", "ruler", "n", "n_problems", "auc_per_problem", "auc_pooled"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    lines.append("")
    lines.append("### eigenscore(group spread) vs group pass rate")
    ecols = ["layer", "position", "n_groups_all(k>=2)", "spearman_eig_vs_pass_all",
             "pearson_eig_vs_pass_all", "n_groups_mixed(k>=2)", "spearman_eig_vs_pass_mixed",
             "pearson_eig_vs_pass_mixed", "mean_eigenscore_all"]
    lines.append("| " + " | ".join(ecols) + " |")
    lines.append("|" + "---|" * len(ecols))
    for r in eig_rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in ecols) + " |")
    lines.append("")
    lines.append("### base")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in base.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")
    lines.append("자 방향: cos_lso/l2_lso 는 wrongness(1−r_corr) 표적 — AUC>.5 면 «형제 중심에서 "
                 "멀수록 틀렸다». resid_probe 는 r_corr 표적(난이도 한 방향 제거 후).")
    return "\n".join(lines)


def run(npz, pass_rates: dict, *, seed: int = 0, alpha: float = EIG_ALPHA) -> dict:
    y_corr = np.asarray(npz["target"], dtype=float)
    groups = np.asarray(npz["group_id"])
    layers = [int(x) for x in npz["layers"]]
    p_all = np.array([pass_rates.get(g, _NAN) for g in groups], dtype=float)
    mixed = np.isfinite(p_all) & (p_all > 0.0) & (p_all < 1.0)

    rows: list[dict] = []
    eig_rows: list[dict] = []
    for L in layers:
        for pos, key in (("metaend", f"hidden_metaend_L{L}"), ("last", f"hidden_last_L{L}")):
            if key not in npz:
                continue
            X = np.asarray(npz[key], dtype=float)
            r, e = evaluate_geometry(X, y_corr, groups, pass_rates, layer=L, position=pos,
                                     seed=seed, alpha=alpha)
            rows.extend(r)
            eig_rows.append(e)

    gm = groups[mixed]
    sizes = {}
    for g in gm.tolist():
        sizes[g] = sizes.get(g, 0) + 1
    base = {
        "n_rows_npz": int(len(y_corr)),
        "n_groups_npz": int(len(set(groups.tolist()))),
        "n_rows_mixed": int(mixed.sum()),
        "n_groups_mixed": int(len(sizes)),
        "n_groups_mixed_k>=2": int(sum(1 for v in sizes.values() if v >= 2)),
        "n_rows_mixed_k>=2": int(sum(v for v in sizes.values() if v >= 2)),
        "wrongness_rate_mixed": float(1.0 - y_corr[mixed].mean()) if mixed.any() else _NAN,
        "eig_alpha": alpha,
    }
    return {"rows": rows, "eig_rows": eig_rows, "base": base}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="math_end_probe --save_features 산출 npz")
    ap.add_argument("--rollouts", default=None,
                    help="선택: 같은 실행의 texts.jsonl — 그룹 정답률을 원본 전체 행으로 잰다"
                         "(math_end_probe/math_agree_probe 와 같은 정의). 없으면 npz target 으로 계산")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=EIG_ALPHA)
    a = ap.parse_args()

    npz = np.load(a.features, allow_pickle=True)
    groups = np.asarray(npz["group_id"])
    if a.rollouts:
        rows_raw = [json.loads(l) for l in open(a.rollouts)]
        pass_rates = E.group_pass_rates(rows_raw)
        src = "rollouts(all rows)"
    else:
        y = np.asarray(npz["target"], dtype=float)
        acc: dict = {}
        for g, v in zip(groups.tolist(), y.tolist()):
            acc.setdefault(g, []).append(v)
        pass_rates = {g: float(np.mean(v)) for g, v in acc.items()}
        src = "npz target(meta-emitting rows only)"

    out = run(npz, pass_rates, seed=a.seed, alpha=a.alpha)
    out["base"]["pass_rate_source"] = src
    md = to_markdown(out["rows"], out["eig_rows"], out["base"])
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "geometry_table.md").write_text(md)
    (od / "geometry_table.json").write_text(json.dumps(
        {"rows": out["rows"], "eigenscore": out["eig_rows"], "base": out["base"]},
        ensure_ascii=False, indent=2, default=float))
    print(md)
    print(f"[out] {od}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
