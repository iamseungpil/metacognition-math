#!/usr/bin/env python
r"""math_dist_effect — «메타를 심으면 이어쓰기들의 분포가 **어디로** 움직이는가»를 같은 자리에서
잰다. math_sites.py 가 낸 자리(sites.jsonl)와 이어쓰기(continuations.jsonl) 위에서, 모드마다
이어쓰기 은닉표현의 **중심·퍼짐·답 엔트로피**가 nometa 대비 어떻게 달라지는지를 본다.

왜. 지금까지의 자는 «이 메타가 좋은가»를 스칼라 하나로 읽으려 했다(전부 기각). 여기선 효과의
**방향**을 본다 — 메타가 분포를 정답 무리 쪽으로 밀면 그것이 «쓸모 있는 메타»의 기하학적 정의다.

한 이어쓰기당 HF forward 한 번:
    render_generation_prompt(tok, variant, problem) + fed + cont
  fed = math_sites.build_fed(mode, prefix, donor_meta, own_meta) — **생성 때와 바이트 동일**하게
  하려고 그 함수를 그대로 import 한다. --variant 는 그 이어쓰기를 만든 변형과 같아야 한다
  (math_sites.SOURCE_VARIANT: cut→math_plain, own_meta→math_opt). 특징 둘: cont 의 **마지막
  토큰** 은닉벡터와 cont 토큰 전체의 **평균** 벡터(같은 forward 에서 함께 읽는다).

자리마다, nometa 가 아닌 모드 m 마다(레이어 × 특징별):
  (1) centroid_shift    cos 거리( centroid(m), centroid(nometa) )
  (2) toward_correct    [centroid(m) − centroid(nometa)] · û,  û = unit( centroid(nometa|r_corr=1)
                        − centroid(nometa) ). 양수 = 정답 무리 쪽으로 움직였다.
                        ★nometa 에 정답이 없거나 오답이 없는 자리는 건너뛴다(방향이 정의 안 됨).
  (3) spread_delta      EigenScore(m) − EigenScore(nometa)  (k×k 공분산 고유값 로그 평균 —
                        math_geometry_probe.eigenscore 재사용)
  (4) ans_entropy_delta 답 클러스터 엔트로피(nats) 차. 클러스터는 **정오와 무관**하게 마지막
                        \boxed 를 수학 동치(math_meta.answers_equivalent)로 묶는다.
  (5) success_delta     p(m) − p(nometa)  (그 자리 그 모드의 r_corr 평균)

집계: 모드별 평균 + 자리 부트스트랩 95% CI, own(없으면 meta) vs donor 의 **짝지은** 비교(같은
자리에서 둘 다 정의된 것만), 그리고 (2) 와 success_delta 의 자리 간 Spearman.

사용:
  python scripts/local/math_dist_effect.py --sites_dir <dir> --model_path <hf> --out_dir <dir> \
      --variant math_plain [--layer -1,mid] [--k_per_mode 8] [--max_len 8192] [--limit 20]
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

import math_ruler_pivot as P  # noqa: E402  (Job/_enc/hf_forward_factory/parse_layers 재사용)
import math_geometry_probe as G  # noqa: E402  (eigenscore/spearman 재사용 — 정의를 한 곳에)
from math_sites import build_fed  # noqa: E402  (★생성 때와 같은 fed 조립)
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

_NAN = float("nan")
BASE_MODE = "nometa"
METRICS_GEOM = ("centroid_shift", "toward_correct", "spread_delta")
METRICS_TEXT = ("ans_entropy_delta", "success_delta")


# ── 이어쓰기 선별 ────────────────────────────────────────────────────────────────
def select_continuations(conts: Sequence[dict], k_per_mode: int,
                         site_ids: Optional[set] = None) -> list[dict]:
    """자리×모드마다 앞 k 개만(파일 순서 — 결정적). site_ids 를 주면 그 자리만."""
    seen: dict = {}
    out = []
    for c in conts:
        if site_ids is not None and c["site_id"] not in site_ids:
            continue
        key = (c["site_id"], c["mode"])
        if seen.get(key, 0) >= k_per_mode:
            continue
        if not (c.get("cont") or "").strip():
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(c)
    return out


def build_jobs(tok, sites: Sequence[dict], sel: Sequence[dict], variant: str) -> dict:
    """이어쓰기 하나당 Job 하나. hidden_at = cont 마지막 토큰, hidden_span = cont 전체(평균)."""
    by_site = {s["site_id"]: s for s in sites}
    jobs: list[P.Job] = []
    index: list[dict] = []
    for c in sel:
        s = by_site.get(c["site_id"])
        if s is None:
            continue
        fed = build_fed(c["mode"], s["prefix"], s.get("donor_meta"), s.get("own_meta"))
        head = render_generation_prompt(tok, variant, s["problem"]) + fed
        head_ids = P._enc(tok, head)
        cont_ids = P._enc(tok, c["cont"])
        if not cont_ids:
            continue
        ids = head_ids + cont_ids
        index.append({"site_id": c["site_id"], "mode": c["mode"], "r_corr": int(c["r_corr"]),
                      "cont": c["cont"], "job": len(jobs), "n_head": len(head_ids),
                      "n_cont": len(cont_ids)})
        jobs.append(P.Job(ids, hidden_at=len(ids) - 1,
                          hidden_span=(len(head_ids), len(ids))))
    return {"jobs": jobs, "index": index}


def collect_features(res: Sequence[dict], built: dict, layers: Sequence[int]) -> list[dict]:
    """forward 결과를 행(index)에 붙인다. hidden_mean 이 없으면(잘림) NaN 벡터."""
    rows = []
    for it in built["index"]:
        r = res[it["job"]]
        rec = dict(it)
        rec["vec"] = {("last", L): np.asarray(r["hidden"][L], dtype=float) for L in layers}
        hm = r.get("hidden_mean") or {}
        for L in layers:
            v = hm.get(L)
            rec["vec"][("mean", L)] = (np.asarray(v, dtype=float) if v is not None
                                       else np.full_like(rec["vec"][("last", L)], _NAN))
        rows.append(rec)
    return rows


# ── 답 클러스터 / 엔트로피 ────────────────────────────────────────────────────────
def answer_clusters(answers: Sequence[str]) -> list[int]:
    """마지막 \\boxed 문자열들을 **수학 동치**로 묶는다(정오와 무관). 빈 답("")은 자기들끼리
    한 클러스터. 먼저 문자열 동일로 묶고, 서로 다른 문자열끼리만 동치 검사(호출 수 절약)."""
    from src.training.math_meta import answers_equivalent  # noqa: PLC0415
    reps: list[str] = []
    of_str: dict[str, int] = {}
    for a in answers:
        a = (a or "").strip()
        if a in of_str:
            continue
        cid = None
        if a:
            for i, rep in enumerate(reps):
                if rep and answers_equivalent(rep, a):
                    cid = i
                    break
        else:
            for i, rep in enumerate(reps):
                if rep == "":
                    cid = i
                    break
        if cid is None:
            reps.append(a)
            cid = len(reps) - 1
        of_str[a] = cid
    return [of_str[(a or "").strip()] for a in answers]


def entropy_of(labels: Sequence[int]) -> float:
    """클러스터 라벨 분포의 Shannon 엔트로피(nats). 비면 NaN."""
    if not len(labels):
        return _NAN
    _, cnt = np.unique(np.asarray(labels), return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def boxed_answers(conts: Sequence[str]) -> list[str]:
    from src.training.math_meta import last_boxed  # noqa: PLC0415
    return [last_boxed(c or "") for c in conts]


# ── 자리 하나의 지표 ──────────────────────────────────────────────────────────────
def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (na > 0 and nb > 0):
        return _NAN
    return 1.0 - float(a @ b) / (na * nb)


def site_geometry(vecs: dict, rcorr: dict, feature_keys: Sequence) -> dict:
    """vecs[(mode, fkey)] = (k,D) 배열, rcorr[mode] = 0/1 리스트.
    Returns {(mode, fkey): {centroid_shift, toward_correct, spread_delta}}."""
    out: dict = {}
    for fk in feature_keys:
        base = vecs.get((BASE_MODE, fk))
        if base is None or len(base) == 0 or not np.isfinite(base).all():
            continue
        c0 = base.mean(axis=0)
        e0 = G.eigenscore(base)
        yb = np.asarray(rcorr.get(BASE_MODE, []), dtype=float)
        u = None
        if len(yb) == len(base) and 0 < yb.sum() < len(yb):
            cc = base[yb == 1].mean(axis=0)
            d = cc - c0
            nd = float(np.linalg.norm(d))
            if nd > 0:
                u = d / nd
        for (mode, fk2), V in vecs.items():
            if fk2 != fk or mode == BASE_MODE or len(V) == 0 or not np.isfinite(V).all():
                continue
            cm = V.mean(axis=0)
            out[(mode, fk)] = {
                "centroid_shift": _cos_dist(cm, c0),
                "toward_correct": float((cm - c0) @ u) if u is not None else _NAN,
                "spread_delta": G.eigenscore(V) - e0,
            }
    return out


def site_text_metrics(conts: dict, rcorr: dict) -> dict:
    """conts[mode] = 이어쓰기 텍스트 리스트 → {mode: {ans_entropy_delta, success_delta}}."""
    if BASE_MODE not in conts or not conts[BASE_MODE]:
        return {}
    modes = [m for m in conts if m != BASE_MODE and conts[m]]
    # 자리 안 **모든** 모드의 답을 한 번에 클러스터링한다 — 모드마다 따로 묶으면 클러스터
    # 정체성이 달라져 엔트로피가 비교 불가가 된다.
    flat, owner = [], []
    for m in [BASE_MODE] + modes:
        for a in boxed_answers(conts[m]):
            flat.append(a)
            owner.append(m)
    lab = answer_clusters(flat)
    by_mode: dict = {}
    for m, c in zip(owner, lab):
        by_mode.setdefault(m, []).append(c)
    h0 = entropy_of(by_mode[BASE_MODE])
    p0 = float(np.mean(rcorr[BASE_MODE]))
    return {m: {"ans_entropy_delta": entropy_of(by_mode[m]) - h0,
                "success_delta": float(np.mean(rcorr[m])) - p0}
            for m in modes}


def per_site_records(feats: Sequence[dict], layers: Sequence[int]) -> list[dict]:
    """행(이어쓰기)들을 자리별로 접어 지표 레코드를 만든다."""
    fkeys = [(pos, L) for pos in ("last", "mean") for L in layers]
    by_site: dict = {}
    for f in feats:
        by_site.setdefault(f["site_id"], []).append(f)
    recs = []
    for sid, rows in by_site.items():
        vecs: dict = {}
        rcorr: dict = {}
        conts: dict = {}
        for r in rows:
            rcorr.setdefault(r["mode"], []).append(r["r_corr"])
            conts.setdefault(r["mode"], []).append(r["cont"])
        for fk in fkeys:
            for m in rcorr:
                vecs[(m, fk)] = np.array([r["vec"][fk] for r in rows if r["mode"] == m])
        geom = site_geometry(vecs, rcorr, fkeys)
        text = site_text_metrics(conts, rcorr)
        for (mode, fk), g in geom.items():
            recs.append({"site_id": sid, "mode": mode, "position": fk[0], "layer": fk[1],
                         **g, **text.get(mode, {"ans_entropy_delta": _NAN,
                                                "success_delta": _NAN})})
    return recs


# ── 집계 ────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """자리 부트스트랩 백분위 CI. 유한한 값만 쓴다."""
    v = np.asarray([x for x in values if x == x and math.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": _NAN, "lo": _NAN, "hi": _NAN, "n": 0}
    rng = np.random.RandomState(seed)
    bs = v[rng.randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, alpha / 2)),
            "hi": float(np.quantile(bs, 1 - alpha / 2)), "n": int(len(v))}


def aggregate(recs: Sequence[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """모드×레이어×위치별 평균/CI, own(or meta) vs donor 짝 비교, (2)↔success Spearman."""
    metrics = list(METRICS_GEOM) + list(METRICS_TEXT)
    cells: dict = {}
    for r in recs:
        cells.setdefault((r["layer"], r["position"], r["mode"]), []).append(r)
    rows = []
    for (L, pos, mode), xs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        row = {"layer": L, "position": pos, "mode": mode, "n_sites": len(xs)}
        for m in metrics:
            ci = bootstrap_ci([x[m] for x in xs], n_boot=n_boot, seed=seed)
            row[m] = ci["mean"]
            row[f"{m}_lo"] = ci["lo"]
            row[f"{m}_hi"] = ci["hi"]
            row[f"{m}_n"] = ci["n"]
        row["spearman_toward_vs_success"] = G.spearman([x["toward_correct"] for x in xs],
                                                       [x["success_delta"] for x in xs])
        rows.append(row)

    # 짝지은 비교: 같은 자리에서 두 모드가 다 있는 것만.
    modes = {r["mode"] for r in recs}
    tgt = "own" if "own" in modes else ("meta" if "meta" in modes else None)
    pairs = []
    if tgt and "donor" in modes:
        for (L, pos) in sorted({(r["layer"], r["position"]) for r in recs}):
            a = {r["site_id"]: r for r in recs
                 if r["layer"] == L and r["position"] == pos and r["mode"] == tgt}
            b = {r["site_id"]: r for r in recs
                 if r["layer"] == L and r["position"] == pos and r["mode"] == "donor"}
            common = sorted(set(a) & set(b))
            prow = {"layer": L, "position": pos, "pair": f"{tgt}-donor", "n_sites": len(common)}
            for m in metrics:
                ci = bootstrap_ci([a[s][m] - b[s][m] for s in common], n_boot=n_boot, seed=seed)
                prow[m] = ci["mean"]
                prow[f"{m}_lo"] = ci["lo"]
                prow[f"{m}_hi"] = ci["hi"]
                prow[f"{m}_n"] = ci["n"]
            pairs.append(prow)
    return {"rows": rows, "pairs": pairs,
            "pair_target": tgt, "pair_available": bool(tgt and "donor" in modes)}


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def _ci(r: dict, m: str) -> str:
    return f"{_fmt(r.get(m))} [{_fmt(r.get(m + '_lo'))}, {_fmt(r.get(m + '_hi'))}] (n={r.get(m + '_n')})"


def to_markdown(agg: dict, base: dict) -> str:
    metrics = list(METRICS_GEOM) + list(METRICS_TEXT)
    lines = ["### per-mode (mean [bootstrap 95% CI] over sites)"]
    cols = ["layer", "position", "mode", "n_sites"] + metrics + ["spearman_toward_vs_success"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in agg["rows"]:
        cells = [str(r["layer"]), r["position"], r["mode"], str(r["n_sites"])]
        cells += [_ci(r, m) for m in metrics]
        cells.append(_fmt(r.get("spearman_toward_vs_success")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### paired (same site, target − donor)")
    if agg["pairs"]:
        pcols = ["layer", "position", "pair", "n_sites"] + metrics
        lines.append("| " + " | ".join(pcols) + " |")
        lines.append("|" + "---|" * len(pcols))
        for r in agg["pairs"]:
            cells = [str(r["layer"]), r["position"], r["pair"], str(r["n_sites"])]
            cells += [_ci(r, m) for m in metrics]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append(f"(짝 비교 없음 — 이 sites dir 에 donor 모드가 없다. 대상 모드="
                     f"{agg.get('pair_target')})")
    lines.append("")
    lines.append("### base")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in base.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")
    lines.append("toward_correct > 0 = 그 모드의 중심이 nometa 의 **정답** 무리 쪽으로 움직였다. "
                 "ans_entropy_delta < 0 = 답이 더 모였다. 부호 해석은 success_delta 와의 "
                 "Spearman 과 함께 읽어야 한다.")
    return "\n".join(lines)


def run(sites: Sequence[dict], conts: Sequence[dict], forward, tok, layers: Sequence[int],
        variant: str, *, k_per_mode: int = 8, n_boot: int = 2000, seed: int = 0) -> dict:
    sel = select_continuations(conts, k_per_mode, {s["site_id"] for s in sites})
    built = build_jobs(tok, sites, sel, variant)
    res = forward(built["jobs"], list(layers))
    feats = collect_features(res, built, layers)
    recs = per_site_records(feats, layers)
    agg = aggregate(recs, n_boot=n_boot, seed=seed)
    modes: dict = {}
    for f in feats:
        modes[f["mode"]] = modes.get(f["mode"], 0) + 1
    agg["base"] = {"n_sites": len({f["site_id"] for f in feats}), "n_forwards": len(built["jobs"]),
                   "k_per_mode": k_per_mode, "variant": variant,
                   "modes": json.dumps(modes, sort_keys=True),
                   "n_site_mode_records": len(recs)}
    agg["records"] = recs
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_dir", required=True, help="math_sites.py 산출(sites.jsonl + continuations.jsonl)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", required=True,
                    help="★그 이어쓰기를 만든 변형과 같아야 한다(cut→math_plain, own_meta→math_opt)")
    ap.add_argument("--layer", default="-1,mid")
    ap.add_argument("--k_per_mode", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="smoke: 앞 N 자리만")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sd = Path(a.sites_dir)
    sites = [json.loads(l) for l in open(sd / "sites.jsonl")]
    if a.limit:
        sites = sites[:a.limit]
    conts = [json.loads(l) for l in open(sd / "continuations.jsonl")]
    src = {s.get("site_source") for s in sites}
    want = {"cut": "math_plain", "own_meta": "math_opt"}
    for s in src:
        if s in want and want[s] != a.variant:
            print(f"[dist_effect] ⚠️ site_source={s} 의 기대 variant 는 {want[s]} 인데 "
                  f"--variant {a.variant} 를 받았다 — 문맥이 생성 때와 달라진다.", flush=True)
    modes = sorted({c["mode"] for c in conts})
    if BASE_MODE not in modes:
        raise SystemExit(f"[dist_effect] nometa 모드가 없다(있는 모드: {modes}) — 기준선 없이는 못 잰다.")
    print(f"[dist_effect] 자리 {len(sites)} · 이어쓰기 {len(conts)} · 모드 {modes}", flush=True)

    forward, tok, n_layers = P.hf_forward_factory(a.model_path, a.device, a.batch_size, a.max_len)
    layers = P.parse_layers(a.layer, n_layers)
    print(f"[dist_effect] layers {layers} / n_layers {n_layers}", flush=True)
    out = run(sites, conts, forward, tok, layers, a.variant,
              k_per_mode=a.k_per_mode, n_boot=a.n_boot, seed=a.seed)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    md = to_markdown(out, out["base"])
    (od / "dist_effect_table.md").write_text(md)
    (od / "dist_effect_table.json").write_text(json.dumps(
        {"rows": out["rows"], "pairs": out["pairs"], "base": out["base"],
         "records": out["records"]}, ensure_ascii=False, indent=2, default=float))
    print(md)
    print(f"[out] {od}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
