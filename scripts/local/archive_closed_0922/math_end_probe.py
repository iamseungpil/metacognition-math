#!/usr/bin/env python
r"""math_end_probe — cd9 «메타 끝 은닉상태가 다음 정오를 아는가»를 math_rollout 산출물
위에서 직접 잰다(자리 개입 없음 — 모델이 실제로 쓴 메타 그 자체).

입력: math_rollout.py 가 낸 <rollout_dir>/texts.jsonl 한 줄 = {group_id, problem_id,
problem, gold, text, r_corr, final_answer, truncated, n_tok}.
  - --variant math_opt(또는 math_plain/math_new): text 안 첫 <meta> 블록(parse_meta
    form="math", emitted 필수) 을 쓴다. 표적 = r_corr(그 이어쓰기 전체의 정답 여부).
  - --variant math_retry: `src.training.math_meta.split_attempts` 로 첫 \\boxed 뒤의
    메타 블록을 잡는다. 표적 = 첫 답 정오(grade_math(first_answer, gold)) — 메타가
    "이 첫 답이 맞았는가"를 알고 있는지를 재는 것이 M_RETRY 판단 항의 전제이기 때문.

특징(HF forward 한 번, bf16, math_ruler_pivot.hf_forward_factory 재사용):
  probe_metaend@L   </meta> 마지막 토큰의 은닉벡터 (layer L, --layer "-1,mid" 형식)
  probe_last@L      전체 응답 마지막 토큰의 은닉벡터
  stated_conf       메타가 쓴 confidence 값
  entropy_mean      메타 블록 시작 앞 32 토큰의 평균 엔트로피(math_ruler_pivot DROP_WINDOW 와 같음)
프로브(probe_metaend/probe_last)는 문제-그룹 5-fold out-of-fold(math_ruler_pivot.
grouped_oof_probe) — 같은 문제(group_id)가 train/val 에 갈리지 않는다.

★결과 고정(outcome-fixed) 노출을 이 판에선 난이도 라벨(extra_info.level)이 아니라
  **문제(그룹)의 관측 정답률 3분위**로 한다 — math_rollout 산출물엔 level 이 없다.
  ruler 가 "문제가 쉬운가"만 되읽으면, 0<p<1 인(움직일 여지가 있는) 그룹만 남긴
  auc_0p1 과, p̂ 3분위 층 안에서 잰 auc_tertile 이 둘 다 .5 쪽으로 무너진다 —
  그룹 관측 정답률 자체를 자로 넣으면 이 두 열이 그 붕괴를 확인해 준다
  (tests/test_math_end_probe.py::test_group_pass_rate_ruler_is_near_chance_stratified).

통과 규칙: auc_overall ≥ .70  ∧  auc_0p1(0<p<1 그룹만) ≥ .60.

사용:
  python scripts/local/math_end_probe.py --rollouts <dir>/texts.jsonl \
      --model_path <hf> --variant math_opt --out_dir <out> [--layer -1,mid] \
      [--limit 200] [--save_features]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_ruler_pivot as P  # noqa: E402  (Job/_enc/hf_forward_factory/probe helpers 재사용)
from src.training.countdown_rewards import parse_meta  # noqa: E402
from src.rulers.hidden_probe import fit_probe, probe_score  # noqa: E402

_NAN = float("nan")
DROP_WINDOW = 32          # ★math_ruler_pivot 과 같은 창(메타 시작 앞 엔트로피)
PASS_AUC, PASS_AUC_0P1 = 0.70, 0.60


# ── 순수 헬퍼(테스트 대상) ────────────────────────────────────────────────────────
def select_rows(rows: list[dict], variant: str,
                grade_fn: Optional[Callable[[str, str], int]] = None) -> list[dict]:
    """math_rollout 행에서 «파싱된 메타가 있는» 행만 남기고 표적을 붙인다.

    variant == "math_retry": src.training.math_meta.split_attempts 로 첫 \\boxed 뒤의
      메타를 잡는다 — 첫 답이 없거나 그 뒤 메타가 미완결(confidence 없음)이면 버린다.
      표적 = grade_fn(first_answer, gold) (기본은 math_meta.grade_math, 지연 임포트 —
      math_verify 없이도 math_opt 경로만 쓰는 호출자를 깨지 않기 위해서다).
    그 외: text 전체에서 parse_meta(form="math"). emitted=0(신뢰도 없음)이면 버린다.
      표적 = r_corr.
    """
    out = []
    for i, r in enumerate(rows):
        text = r.get("text") or ""
        if variant == "math_retry":
            from src.training.math_meta import split_attempts  # noqa: PLC0415
            sp = split_attempts(text)
            m = sp["meta"]
            if not m["emitted"] or m["start"] is None or sp["first_answer"] is None:
                continue
            gf = grade_fn
            if gf is None:
                from src.training.math_meta import grade_math as gf  # noqa: PLC0415
            target = int(gf(sp["first_answer"], r.get("gold")))
        else:
            m = parse_meta(text, form="math")
            if not m["emitted"] or m["start"] is None:
                continue
            target = int(r["r_corr"])
        out.append({"idx": i, "group_id": r["group_id"], "problem": r["problem"],
                    "gold": r.get("gold"), "text": text, "meta_start": m["start"],
                    "meta_end": m["end"], "confidence": m["confidence"], "target": target})
    return out


def group_pass_rates(rows: list[dict]) -> dict:
    """group_id → 관측 정답률(r_corr 평균), math_rollout 원본 전체 행 기준(선별 전)."""
    acc: dict[str, list[int]] = {}
    for r in rows:
        acc.setdefault(r["group_id"], []).append(int(r["r_corr"]))
    return {g: float(np.mean(v)) for g, v in acc.items()}


def nontrivial_auc(scores: Sequence, labels: Sequence, groups: Sequence, pass_rates: dict) -> float:
    """0 < p̂(그룹) < 1 인 행만 남긴 AUC — 「문제가 쉬운가」만 되읽는 자를 노출한다."""
    mask = np.array([0.0 < pass_rates.get(g, -1.0) < 1.0 for g in groups])
    if mask.sum() == 0:
        return _NAN
    s = np.asarray(scores, dtype=float)[mask]
    y = np.asarray(labels, dtype=float)[mask]
    return P.auc(y, s)


def tertile_auc(scores: Sequence, labels: Sequence, groups: Sequence, pass_rates: dict,
                *, verbose: bool = True) -> dict:
    """p̂(그룹) 3분위 층 안에서 잰 outcome-fixed AUC(math_ruler_pivot.outcome_fixed_auc).

    ★수리(0914, RESULTS_cd9.md §3 결함): math500 은 그룹 관측 정답률이 극단으로 쏠려 있다
    (52% 가 8/8, 24.8% 가 0/8 — 모듈 docstring 참조). 이 분포에서 np.quantile([1/3, 2/3])
    은 종종 **q1 == q2 == max(v)** 로 무너진다(정답 그룹이 전체의 절반을 넘으면 66.7 백분위
    도 이미 "전부 1.0" 구간 안이다). `tertile_strata` 는 x<=q1 을 층 0 으로 묶으므로 이 경우
    **모든 행이 층 0 하나**에 들어간다 — 층화가 사실상 일어나지 않은 채 `outcome_fixed_auc`
    가 «층 하나짜리 평균» = 전체 집합 AUC 를 그대로 돌려준다(실측: probe_table.json 의
    auc_tertile 이 auc_overall 과 소수점까지 동일). 이걸 "층화된 진짜 auc_tertile"로 오독하면
    ruler 가 난이도를 되읽는지 못 본다.

    수리: 경계·층별 n 을 항상 찍고(진단), **두 개 미만의 층에 행이 실제로 나뉘면**(즉 층화가
    일어나지 않았으면) NaN 을 돌린다 — 「값은 있는데 사실은 안 잰 것」을 조용히 auc_overall 로
    위장시키지 않는다."""
    vals = [pass_rates.get(g, 0.5) for g in groups]
    strata = P.tertile_strata(vals)
    v = np.asarray(vals, dtype=float)
    q1, q2 = (np.quantile(v, [1 / 3, 2 / 3]) if len(v) else (_NAN, _NAN))
    counts = {st: int(sum(1 for s in strata if s == st)) for st in (0, 1, 2)}
    n_nonempty = sum(1 for c in counts.values() if c > 0)
    if verbose:
        print(f"[math_end_probe][tertile] q1={q1:.4f} q2={q2:.4f} "
              f"n0={counts[0]} n1={counts[1]} n2={counts[2]} n_nonempty_strata={n_nonempty}",
              flush=True)
    if n_nonempty < 2:
        # ★층화가 실제로 일어나지 않았다(전부 한 층) — outcome_fixed_auc 가 auc_overall 과
        #   바이트 동일한 값을 «층화된 것처럼» 돌려주는 것을 막는다.
        return {"auc": _NAN, "per_stratum": {}, "n_strata": 0,
                "degenerate": True, "boundaries": (float(q1), float(q2)), "counts": counts}
    out = P.outcome_fixed_auc(scores, labels, strata)
    out["degenerate"] = False
    out["boundaries"] = (float(q1), float(q2))
    out["counts"] = counts
    return out


def pass_rule(auc_overall: float, auc_0p1: float) -> str:
    """사전등록: auc_overall ≥ .70 ∧ auc_0p1 ≥ .60 (둘 다 유한해야 함)."""
    ok = (math.isfinite(auc_overall) and auc_overall >= PASS_AUC
          and math.isfinite(auc_0p1) and auc_0p1 >= PASS_AUC_0P1)
    return "PASS" if ok else "FAIL"


# ── 문맥 조립(math_meta_prompt.render_generation_prompt, 생성과 바이트 단위로 같은 프롬프트) ──
# ★중복 제거(0914): math_sites.py 는 더 이상 자체 `_gen_prompt` 를 갖지 않는다(단일 진실
#   원천은 src.metacot.math_meta_prompt.render_generation_prompt) — 여기서도 직접 쓴다.
def _row_gen_prompt(tok, variant: str, problem: str) -> str:
    from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: PLC0415
    return render_generation_prompt(tok, variant, problem)


def build_jobs(tok, variant: str, selected: list[dict]) -> dict:
    """행마다 job 둘: (1) </meta> 끝 은닉 + 그 앞 DROP_WINDOW 엔트로피, (2) 응답 마지막
    토큰 은닉. head/blk/tail 인코딩은 math_ruler_pivot.build_jobs 의 meta 자리 조립과 같은
    방식(문자열 경계에서 나눠 따로 인코딩 후 이어붙임)."""
    jobs: list = []
    index: list[dict] = []
    for row in selected:
        gen = _row_gen_prompt(tok, variant, row["problem"])
        s, e = row["meta_start"], row["meta_end"]
        head_ids = P._enc(tok, gen + row["text"][:s])
        blk_ids = P._enc(tok, row["text"][s:e])
        tail_ids = P._enc(tok, row["text"][e:])
        ids_meta = head_ids + blk_ids
        n_h = len(head_ids)
        ent_pos = list(range(max(0, n_h - DROP_WINDOW), n_h))
        j_meta = P.Job(ids_meta, hidden_at=len(ids_meta) - 1, ent_positions=ent_pos)
        ids_full = ids_meta + tail_ids
        j_last = P.Job(ids_full, hidden_at=len(ids_full) - 1)
        idx_meta, idx_last = len(jobs), len(jobs) + 1
        jobs.extend([j_meta, j_last])
        index.append({"row": row, "job_meta": idx_meta, "job_last": idx_last})
    return {"jobs": jobs, "index": index}


def collect_features(res: list[dict], built: dict) -> list[dict]:
    feats = []
    for it in built["index"]:
        rm, rl = res[it["job_meta"]], res[it["job_last"]]
        e = np.asarray(rm["entropy"], dtype=float)
        feats.append({"row": it["row"], "hidden_metaend": rm["hidden"], "hidden_last": rl["hidden"],
                     "entropy_before": float(e.mean()) if len(e) else _NAN})
    return feats


# ── 자 평가 ─────────────────────────────────────────────────────────────────────
def _row(name: str, n: int, auc_overall: float, auc_0p1: float, auc_tert: float) -> dict:
    return {"ruler": name, "n": int(n), "auc_overall": auc_overall, "auc_0p1": auc_0p1,
            "auc_tertile": auc_tert, "verdict": pass_rule(auc_overall, auc_0p1)}


def evaluate_probes(feats: list[dict], layers: list[int], pass_rates: dict, *,
                    seed: int = 0) -> tuple[list[dict], dict]:
    """Returns (표 rows, {ruler_name: fit_probe dict-or-None}) — 후자는 --save_features 용."""
    rows: list[dict] = []
    fitted: dict[str, Optional[dict]] = {}
    if not feats:
        return rows, fitted
    groups = [f["row"]["group_id"] for f in feats]
    y = np.array([f["row"]["target"] for f in feats], dtype=float)

    for L in layers:
        for kind, key in (("probe_metaend", "hidden_metaend"), ("probe_last", "hidden_last")):
            name = f"{kind}@L{L}"
            X = np.array([f[key][L] for f in feats])
            pr = P.grouped_oof_probe(X, y, groups, seed=seed)
            oof = pr["oof"]
            auc_o = P.auc(y, oof)
            auc_0p1 = nontrivial_auc(oof, y, groups, pass_rates)
            auc_t = tertile_auc(oof, y, groups, pass_rates)["auc"]
            rows.append(_row(name, int(np.isfinite(oof).sum()), auc_o, auc_0p1, auc_t))
            fitted[name] = fit_probe(X, y, groups) if pr["n_folds_used"] >= 2 else None

    conf = np.array([f["row"]["confidence"] if f["row"]["confidence"] is not None else _NAN
                     for f in feats], dtype=float)
    ok_conf = np.isfinite(conf)
    rows.append(_row("stated_conf", int(ok_conf.sum()),
                     P.auc(y[ok_conf], conf[ok_conf]) if ok_conf.any() else _NAN,
                     nontrivial_auc(conf, y, groups, pass_rates),
                     tertile_auc(conf, y, groups, pass_rates)["auc"]))

    ent = np.array([f["entropy_before"] for f in feats], dtype=float)
    ok_ent = np.isfinite(ent)
    rows.append(_row("entropy_mean", int(ok_ent.sum()),
                     P.auc(y[ok_ent], ent[ok_ent]) if ok_ent.any() else _NAN,
                     nontrivial_auc(ent, y, groups, pass_rates),
                     tertile_auc(ent, y, groups, pass_rates)["auc"]))

    # ★진단용(패스 대상 아님): 자가 아니라 「그룹 관측 정답률 자체」 — 이게 auc_overall 은
    #   높게 나와도 auc_0p1/auc_tertile 이 .5 쪽으로 무너지는 것이 «난이도 되읽기 노출»이다.
    gpr = np.array([pass_rates.get(g, 0.5) for g in groups], dtype=float)
    rows.append(_row("group_pass_rate(diagnostic)", len(gpr), P.auc(y, gpr),
                     nontrivial_auc(gpr, y, groups, pass_rates),
                     tertile_auc(gpr, y, groups, pass_rates)["auc"]))
    return rows, fitted


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.3f}"
    return str(v)


def to_markdown(rows: list[dict]) -> str:
    cols = ["ruler", "n", "auc_overall", "auc_0p1", "auc_tertile", "verdict"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    lines.append("")
    lines.append(f"pass rule: auc_overall ≥ {PASS_AUC} ∧ auc_0p1(0<p<1 그룹) ≥ {PASS_AUC_0P1}")
    return "\n".join(lines)


def probe_to_json(probe: dict) -> dict:
    return {"weights": np.asarray(probe["weights"]).tolist(), "bias": float(probe["bias"]),
            "mu": np.asarray(probe["mu"]).tolist(), "sigma": np.asarray(probe["sigma"]).tolist()}


def probe_from_json(d: dict) -> dict:
    return {"weights": np.asarray(d["weights"], dtype=float), "bias": float(d["bias"]),
            "mu": np.asarray(d["mu"], dtype=float), "sigma": np.asarray(d["sigma"], dtype=float)}


def run(selected: list[dict], forward: "P.ForwardFn", tok, layers: list[int], variant: str,
       pass_rates: dict, *, seed: int = 0) -> dict:
    built = build_jobs(tok, variant, selected)
    res = forward(built["jobs"], layers)
    feats = collect_features(res, built)
    rows, fitted = evaluate_probes(feats, layers, pass_rates, seed=seed)
    return {"rows": rows, "fitted": fitted, "feats": feats, "n_selected": len(selected)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout texts.jsonl 경로")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="math_opt",
                    help="math_opt/math_plain/math_new(=r_corr 표적) 또는 math_retry(=첫 답 정오 표적)")
    ap.add_argument("--layer", default="-1,mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="smoke: 앞 N 행만")
    ap.add_argument("--save_features", action="store_true",
                    help="특징(npz) + 프로브 가중치/표준화(json) 저장 — 온라인 스코어러 재사용용")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rollouts)]
    if a.limit:
        rows = rows[:a.limit]
    pass_rates = group_pass_rates(rows)
    selected = select_rows(rows, a.variant)
    print(f"[probe] 롤아웃 {len(rows)} · 메타 파싱됨 {len(selected)} · variant={a.variant}", flush=True)
    if not selected:
        raise SystemExit("[probe] 파싱된 <meta> 블록이 하나도 없다 — variant/rollout 확인.")

    forward, tok, n_layers = P.hf_forward_factory(a.model_path, a.device, a.batch_size, a.max_len)
    layers = P.parse_layers(a.layer, n_layers)
    print(f"[probe] layers(hidden_states idx) {layers} / n_layers {n_layers}", flush=True)
    out = run(selected, forward, tok, layers, a.variant, pass_rates, seed=a.seed)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    md = to_markdown(out["rows"])
    (od / "probe_table.md").write_text(md)
    (od / "probe_table.json").write_text(json.dumps(
        {"rows": out["rows"], "n_selected": out["n_selected"]}, ensure_ascii=False, indent=2,
        default=float))
    print(md)

    if a.save_features:
        feats = out["feats"]
        groups = np.array([f["row"]["group_id"] for f in feats])
        y = np.array([f["row"]["target"] for f in feats], dtype=float)
        conf = np.array([f["row"]["confidence"] if f["row"]["confidence"] is not None else _NAN
                         for f in feats], dtype=float)
        ent = np.array([f["entropy_before"] for f in feats], dtype=float)
        npz_kw = {"target": y, "group_id": groups, "confidence": conf, "entropy_before": ent,
                  "layers": np.array(layers)}
        for L in layers:
            npz_kw[f"hidden_metaend_L{L}"] = np.array([f["hidden_metaend"][L] for f in feats])
            npz_kw[f"hidden_last_L{L}"] = np.array([f["hidden_last"][L] for f in feats])
        np.savez(od / "features.npz", **npz_kw)
        weights = {name: probe_to_json(p) for name, p in out["fitted"].items() if p is not None}
        (od / "probe_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2))
        print(f"[probe] 특징 {od / 'features.npz'} · 프로브 가중치 {od / 'probe_weights.json'}",
              flush=True)

    print(f"[out] {od}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
