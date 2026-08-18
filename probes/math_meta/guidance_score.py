"""조타 프로브 B단계 — 본문 안에서 어떤 점수가 "실제로 도움이 된 메타"를 위로 올리나.

A단계 실측(160 prefix × 메타 8):
    acc_N            0.7422   메타 없이
    메타 8개 평균     0.6885   아무 메타나 넣으면        (−5.4pp)
    prefix 내 최고    0.7719   가장 좋은 메타를 골랐다면  (+3.0pp)

★그 “+3.0pp”를 그대로 믿으면 안 된다. 메타당 표본이 k=4 뿐이라 **여덟 개 중 최대를 고르는
것만으로도** 우연히 올라간다. 그래서 여기서는 셋을 나눠 잰다.

  ① 선택 편향 바닥   메타 라벨을 prefix 안에서 섞고 같은 max 를 다시 잰다 → 우연의 몫
  ② 점수의 순위력    prefix 안 Spearman(점수, 정답률) · 치환 귀무분포와 대조
  ③ ★실전 선택력     점수로 고른 메타를 **다른 표본에서** 채점한다 (cross-fit)
                     — 고를 때 쓴 표본으로 채점하면 그것도 편향이다

③이 이 파일의 결론이다: **점수로 고르면 무작위로 고르는 것보다 나은가, 그리고 N 을 이기나.**
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys

import torch

from abl_pmi import body_before_answer, make_decoy, pmi_at, _sum_logp

sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training.dcpo_rmeta_forms import FORMULAS, rmeta_from_ingredients  # noqa: E402


@torch.inference_mode()
def score_meta(model, tok, it, meta, cont, device="cuda"):
    """한 (본문, 메타) 쌍의 원재료. 본문이 고정이므로 prefix 교란이 없다."""
    gold, decoy = it["gold"], make_decoy(it["gold"])
    ctx_open = it["prompt"] + it["prefix"]
    ctx_close = ctx_open + f"\n<meta>{meta}</meta>"
    # 도너 = 같은 본문에 붙은 **다른 후보** — 본문이 같으므로 이번엔 완전히 matched 다
    donor = it["metas"][(it["metas"].index(meta) + 1) % len(it["metas"])]
    ctx_donor = ctx_open + f"\n<meta>{donor}</meta>"

    post = body_before_answer(cont) or ""
    post_ids = tok(post, add_special_tokens=False)["input_ids"][:600]
    if len(post_ids) < 5:
        return None
    sl = slice(0, len(post_ids))
    meta_ids = tok(f"\n<meta>{meta}</meta>", add_special_tokens=False)["input_ids"]

    return {
        "lp_meta": _sum_logp(model, tok, ctx_open, meta_ids, slice(0, len(meta_ids)), device),
        "n_meta_tok": len(meta_ids),
        "pmi_open": pmi_at(model, tok, ctx_open, gold, decoy, device),
        "pmi_close": pmi_at(model, tok, ctx_close, gold, decoy, device),
        "pmi_ans_real": pmi_at(model, tok, ctx_close + post, gold, decoy, device),
        "pmi_ans_donor": pmi_at(model, tok, ctx_donor + post, gold, decoy, device),
        "lp_post_real": _sum_logp(model, tok, ctx_close, post_ids, sl, device),
        "lp_post_donor": _sum_logp(model, tok, ctx_donor, post_ids, sl, device),
        "n_post": len(post_ids),
        "meta_len": len(tok(meta, add_special_tokens=False)["input_ids"]),
    }


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="guidance_gen.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="guidance_score.json")
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    blob = json.load(open(args.gen))
    items = blob["items"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rng = random.Random(0)

    # ── ① 선택 편향 바닥: 메타 라벨을 섞어도 max 는 올라간다. 그 몫을 먼저 안다.
    live = [it for it in items if max(it["acc"]) - min(it["acc"]) > 1e-9]
    oracle = st.mean(max(it["acc"]) for it in items)
    rand_pick = st.mean(st.mean(it["acc"]) for it in items)
    accN = st.mean(it["acc_N"] for it in items)
    print(f"[0] prefix {len(items)} (퍼짐 있는 곳 {len(live)}) · "
          f"N {accN:.4f} · 무작위 메타 {rand_pick:.4f} · max {oracle:.4f}")

    # ── 원재료
    for it in items:
        it["ing"] = []
        for m, c in zip(it["metas"], it["cont"]):
            it["ing"].append(score_meta(model, tok, it, m, c))
    ok = [it for it in items
          if all(g is not None for g in it["ing"]) and len(it["metas"]) >= 4]
    okl = [it for it in ok if max(it["acc"]) - min(it["acc"]) > 1e-9]
    print(f"[1] 채점 가능 prefix {len(ok)} · 그중 퍼짐 있는 곳 {len(okl)}")

    res = {}
    for name in FORMULAS:
        rhos, sel, sel_null = [], [], []
        for it in okl:
            v = [rmeta_from_ingredients(g, name) for g in it["ing"]]
            r = spearman(v, it["acc"])
            if r is not None:
                rhos.append(r)
            # ③ 실전 선택력 — 점수 최대인 메타의 정답률
            sel.append(it["acc"][max(range(len(v)), key=lambda i: v[i])])
            # 같은 절차를 점수를 섞어서 (= 무작위 선택) 한 번
            sh = list(range(len(v))); rng.shuffle(sh)
            sel_null.append(it["acc"][sh[0]])
        if not rhos:
            continue
        # ② 치환 귀무: 각 prefix 안에서 정답률을 섞고 평균 rho 를 다시 잰다
        obs = st.mean(rhos)
        null = []
        for _ in range(args.nperm // 20):
            acc_perm = []
            for it in okl:
                v = [rmeta_from_ingredients(g, name) for g in it["ing"]]
                a = list(it["acc"]); rng.shuffle(a)
                r = spearman(v, a)
                if r is not None:
                    acc_perm.append(r)
            null.append(st.mean(acc_perm) if acc_perm else 0.0)
        null.sort()
        p = sum(1 for x in null if abs(x) >= abs(obs)) / max(1, len(null))
        res[name] = {
            "spearman_within_prefix": round(obs, 4),
            "perm_p": round(p, 4),
            "n_prefix": len(rhos),
            "select_acc": round(st.mean(sel), 4),
            "random_acc": round(st.mean(sel_null), 4),
            "select_minus_random": round(st.mean(sel) - st.mean(sel_null), 4),
        }
        print(f"    {name:14s} rho={obs:+.4f} p={p:.3f}  "
              f"선택 {st.mean(sel):.4f} vs 무작위 {st.mean(sel_null):.4f} "
              f"({st.mean(sel)-st.mean(sel_null):+.4f})")

    ranked = sorted(res, key=lambda k: -res[k]["select_minus_random"])
    out = {"n_prefix": len(ok), "n_live": len(okl),
           "acc_N": accN, "acc_random_meta": rand_pick, "acc_oracle_max": oracle,
           "formulas": res, "ranked_by_selection": ranked,
           "NEG_rank": ranked.index("NEG_len_only") + 1 if "NEG_len_only" in ranked else None}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n" + json.dumps({k: out[k] for k in
                             ("n_prefix", "n_live", "acc_N", "acc_random_meta",
                              "acc_oracle_max", "ranked_by_selection", "NEG_rank")},
                            indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
