r"""도치 자의 **witness 민감도** — 4수 Countdown 은 정답 식이 여럿인데 힌트는 하나를 고정한다.

설계검토가 발사 전 조건으로 요구한 검사다:
  «다른 유효 해를 탐색한 메타가 체계적으로 벌받는 편향이 생긴다.
    발사 전 witness 를 바꿔가며 점수 분산을 재는 검사가 필요하다.»

같은 사이트를 서로 다른 유효 해 3개로 각각 채점하고,
  · 사이트 **안** 흩어짐(witness 를 바꿨을 때) 과
  · 사이트 **사이** 흩어짐(자가 실제로 재려는 신호)
를 비교한다. 앞이 뒤와 비슷하면 이 자의 값은 사실상 witness 추첨 결과다.
또한 witness 를 바꿨을 때 harm/help 분리력이 살아남는지 본다.

자는 `src/training/countdown_inv.py` 의 함수 그대로 — `reencode/a2d/min`
(= `reverse_ruler.V1_prose_min`, ρ=0.974 로 재현 확인).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training import countdown_inv as CI                    # noqa: E402
from src.training.countdown_task import eval_countdown, extract_expr  # noqa: E402
# `scripts/` 는 패키지가 아니므로 파일 경로로 싣는다(그 파일을 복제하지 않기 위해서다 —
# 프롬프트 조립·채점 백엔드가 두 벌이 되면 두 프로브가 서로 다른 자를 잰다).
import importlib.util as _ilu                                  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_inv_ruler_unified", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "inv_ruler_unified.py"))
_iru = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_iru)
chat_text, cohen_d, make_hf_scorer = _iru.chat_text, _iru.cohen_d, _iru.make_hf_scorer

W = "/home/jovyan/beomi/splee"
MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

SCOPE, FORM, AGG = "reencode", "a2d", "min"


def all_solutions(nums, target, cap=6):
    """`scripts/goldset_test.all_solutions` 와 **같은 열거**(중복 정의 금지 원칙상 복사가
    아니라 같은 규칙임을 명시한다 — 그 파일은 진입점이 main 뿐이라 import 가 불가)."""
    t = int(target)
    found, seen = [], set()
    for perm in set(itertools.permutations([int(v) for v in nums])):
        a, b, c, d = perm
        for o1 in "+-*/":
            for o2 in "+-*/":
                for o3 in "+-*/":
                    for tmpl in (f"((({a}{o1}{b}){o2}{c}){o3}{d})",
                                 f"(({a}{o1}{b}){o2}({c}{o3}{d}))",
                                 f"({a}{o1}(({b}{o2}{c}){o3}{d}))"):
                        if tmpl in seen:
                            continue
                        seen.add(tmpl)
                        if eval_countdown(tmpl) == t:
                            found.append(tmpl)
                            if len(found) >= cap * 4:
                                return sorted(found, key=len)[:cap]
    return sorted(found, key=len)[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"{W}/cd6_work/probe/labels_extreme_all.parquet")
    ap.add_argument("--sites_glob", default=f"{W}/cd6_work/probe*/sites_shard*.parquet")
    ap.add_argument("--out", default=f"{W}/cd6_work/probe/inv_unified/inv_witness.parquet")
    ap.add_argument("--k", type=int, default=3, help="사이트당 witness 개수")
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    lab = pd.read_parquet(args.labels)
    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    df = lab.merge(sites, on="site_id", how="inner", suffixes=("", "_s")).reset_index(drop=True)

    sols = {}
    for _, r in df.iterrows():
        s = all_solutions(list(r["nums"]), int(r["target"]), cap=args.k)
        w0 = str(r["witness"])
        if w0 not in s:
            s = [w0] + s
        sols[r["site_id"]] = s[:args.k]
    n_sol = np.array([len(v) for v in sols.values()])
    print(f"[wit] 사이트 {len(df)} · 해 개수 median={int(np.median(n_sol))} "
          f"min={n_sol.min()} (k={args.k} 까지만 열거)", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    scorer = make_hf_scorer(model, batch_size=args.batch_size)

    P = [chat_text(tok, r["prompt_json"]) for _, r in df.iterrows()]
    R = [str(r["response_text"]) for _, r in df.iterrows()]
    T = [int(r["target"]) for _, r in df.iterrows()]
    E = [extract_expr(t) or "" for t in R]

    recs = []
    for slot in range(args.k):
        Wt = [sols[r["site_id"]][min(slot, len(sols[r["site_id"]]) - 1)]
              for _, r in df.iterrows()]
        arm_p, arm_r, attempts, per_row, diag = CI.build_inv_arms(
            tok, P, R, Wt, T, E, scope=SCOPE)
        print(f"[wit] slot={slot} 팔 {len(arm_p)} attempts {len(attempts)}", flush=True)
        ref_lp = scorer(arm_p, arm_r)
        vals = CI.read_inv_from_ref_logprobs(ref_lp, attempts, AGG, FORM)
        by_row = {at.row: k for k, at in enumerate(attempts)}
        for i, r in df.iterrows():
            k = by_row.get(i)
            recs.append({"site_id": r["site_id"], "label": r["label"], "slot": slot,
                         "witness": Wt[i], "n_sol": len(sols[r["site_id"]]),
                         "value": float(vals[k]) if k is not None else np.nan})
    d = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(args.out)

    w = d.pivot_table(index="site_id", columns="slot", values="value")
    multi = d[d.n_sol >= 2].site_id.unique()
    w2 = w.loc[[s for s in w.index if s in set(multi)]].dropna()
    within = w2.std(axis=1, ddof=1)
    between = w2.mean(axis=1).std(ddof=1)
    print(f"\n## witness 민감도 (해가 2개 이상인 {len(w2)} 사이트)\n")
    print(f"- 사이트 **안** sd (witness 바꿈): mean {within.mean():.3f} · median "
          f"{within.median():.3f} · p90 {within.quantile(0.9):.3f}")
    print(f"- 사이트 **사이** sd (자가 재려는 신호): {between:.3f}")
    print(f"- **분산비 within/between = {within.mean() / between:.3f}** "
          f"(1 에 가까우면 값이 사실상 witness 추첨)")
    cc = np.corrcoef(w2.to_numpy().T)
    print(f"- slot 간 피어슨 상관 (평균 비대각): {cc[np.triu_indices(len(cc), 1)].mean():.3f}")

    print("\n## witness 를 바꿔도 harm/help 분리력이 남는가\n")
    print("| slot | harm mean | help mean | d(harm−help) | n |")
    print("|---|---|---|---|---|")
    for slot in range(args.k):
        s = d[d.slot == slot]
        h = s.loc[s.label == "null", "value"].to_numpy(float)
        g = s.loc[s.label == "good", "value"].to_numpy(float)
        print(f"| {slot} | {np.nanmean(h):+.3f} | {np.nanmean(g):+.3f} | "
              f"{cohen_d(h, g):+.3f} | {int(np.isfinite(s.value).sum())} |")
    # 여러 witness 평균으로 쓰면 나아지는가 (편향 완화 후보)
    mv = w2.mean(axis=1)
    lm = dict(zip(d.site_id, d.label))
    hh = np.array([mv[s] for s in mv.index if lm[s] == "null"])
    gg = np.array([mv[s] for s in mv.index if lm[s] == "good"])
    print(f"\n- **{args.k} witness 평균**: harm {hh.mean():+.3f} help {gg.mean():+.3f} "
          f"d={cohen_d(hh, gg):+.3f} (n={len(hh)}/{len(gg)})")
    print(f"\n[wit] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
