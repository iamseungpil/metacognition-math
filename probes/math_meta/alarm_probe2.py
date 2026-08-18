"""경보 프로브 v3 — 위치차 계열을 **정본 decoy** 로 다시 재고, "정답만" 축을 추가한다.

v2 의 결함 둘.

  ⚠**decoy 불충실.** `abl_pmi.make_decoy` 는 숫자가 아니면 따옴표 하나를 덧붙인다.
    MATH-500 golds 의 **34.7%(104/300)** 가 그 폴백으로 갔고, 그런 행의 PMI 는
    "따옴표 한 토큰의 logp 차"였다 -- 믿음을 재는 자가 아니다. 정본은
    `src/training/_decoy_utils._rule_based_decoy` 의 6전략 생성기를 쓴다.
    ⇒ 여기서는 **정본 생성기를 그대로 import** 한다. 프로브가 학습과 같은 자를 쓴다.

  ⚠**정답만 축이 없었다.** PMI 는 gold−decoy 대조인데, "오답을 안 보고 gold 확률만
    보면 어떤가"는 한 번도 안 쟀다. 대조가 도움이 되는지 자체가 질문이므로 넣는다.

네 축을 같은 300행에서 낸다 (행당 5 forward):

    pmi_shift      (logp_g@close − logp_d@close) − (logp_g@open − logp_d@open)   현행
    gold_shift      logp_g@close − logp_g@open                                   ★정답만
    decoy_shift     logp_d@close − logp_d@open                                   (부수)
    surprise        −lp_meta / n_meta_tok                                        v2 에서 ★

판정: **진짜 경보(주장O·실제 틀림, n=41)를 헛경보(주장O·실제 맞음, n=115)보다 높게 매기나.**
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys

import torch

from abl_pmi import divergent_positions, _sum_logp
from alarm_probe import claims_error, get_conf, auc, boot_auc_ci, mean

sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training._decoy_utils import _rule_based_decoy  # noqa: E402


@torch.inference_mode()
def gd_logp(model, tok, ctx, gold, decoy, device="cuda"):
    """(logp_gold, logp_decoy) — 발산 토큰만 합산. PMI 의 두 항을 **따로** 돌려준다."""
    g_ids, d_ids, g_sl, d_sl = divergent_positions(tok, gold, decoy)
    return (_sum_logp(model, tok, ctx, g_ids, g_sl, device),
            _sum_logp(model, tok, ctx, d_ids, d_sl, device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_4b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="alarm_probe2_4b.json")
    args = ap.parse_args()

    blob = json.load(open(args.splice))
    items, accN = blob["items"], blob["per_problem"]["N"]

    # ── 0. decoy 배선검사: 정본 생성기가 gold 와 실제로 다른가 · 폴백률
    same = sum(1 for it in items if _rule_based_decoy(it["gold"]).strip() == it["gold"].strip())
    print(f"[0] 정본 decoy 배선검사 — gold 와 같은 행 {same}/{len(items)} (0 이어야 한다)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for n, it in enumerate(items):
        gold = it["gold"]
        decoy = _rule_based_decoy(gold)
        ctx_open = it["prompt"] + it["prefix"]
        ctx_close = ctx_open + f"\n<meta>{it['R']}</meta>"
        meta_ids = tok(f"\n<meta>{it['R']}</meta>", add_special_tokens=False)["input_ids"]

        go, do = gd_logp(model, tok, ctx_open, gold, decoy)
        gc, dc = gd_logp(model, tok, ctx_close, gold, decoy)
        rows.append({
            "i": n, "claim": claims_error(it["R"]), "accN": accN[n],
            "conf": get_conf(it["R"]),
            "g_open": go, "g_close": gc, "d_open": do, "d_close": dc,
            "lp_meta": _sum_logp(model, tok, ctx_open, meta_ids,
                                 slice(0, len(meta_ids))),
            "n_meta_tok": len(meta_ids),
            "meta_len": len(tok(it["R"], add_special_tokens=False)["input_ids"]),
        })
        if (n + 1) % 75 == 0:
            print(f"    ... {n+1}/{len(items)}")

    T = [r for r in rows if r["claim"] == 1 and r["accN"] <= 0.001]
    F = [r for r in rows if r["claim"] == 1 and r["accN"] >= 0.999]
    print(f"[1] 참 경보 {len(T)} · 헛경보 {len(F)}\n")

    def clip(x, c=2.0):
        return max(-c, min(c, x))

    def rev(o, c, save=1.0, derail=2.0):
        return save if (o < 0 and c > 0) else (-derail if (o > 0 and c < 0) else 0.0)

    PMI_O = lambda r: r["g_open"] - r["d_open"]
    PMI_C = lambda r: r["g_close"] - r["d_close"]

    AXES = {
        "pmi_shift (현행 pos)":   lambda r: PMI_C(r) - PMI_O(r),
        "pmi_shift_full (라이브)": lambda r: clip(PMI_C(r) - PMI_O(r)) + rev(PMI_O(r), PMI_C(r)),
        "★gold_shift (정답만)":    lambda r: r["g_close"] - r["g_open"],
        "gold_shift/len":         lambda r: (r["g_close"] - r["g_open"]),
        "decoy_shift (부수)":      lambda r: r["d_close"] - r["d_open"],
        "-surprise":              lambda r: r["lp_meta"] / max(1, r["n_meta_tok"]),
        "-conf":                  lambda r: -(r["conf"] if r["conf"] is not None else 0.5),
        "meta_len (음성대조)":     lambda r: float(r["meta_len"]),
        "pmi_open (여유)":         PMI_O,
    }

    res = {}
    for name, fn in AXES.items():
        p, q = [fn(r) for r in T], [fn(r) for r in F]
        a = auc(p, q)
        lo, hi = boot_auc_ci(p, q)
        star = "★" if (lo > 0.5 or hi < 0.5) else " "
        res[name] = {"auc": round(a, 4), "ci": [round(lo, 4), round(hi, 4)],
                     "mean_true": round(mean(p), 4), "mean_false": round(mean(q), 4)}
        print(f" {star}{name:24s} AUC {a:.4f} [{lo:.3f},{hi:.3f}]   "
              f"참 {mean(p):+9.3f} vs 헛 {mean(q):+9.3f}")

    n_fire = sum(1 for r in F if PMI_O(r) > 0 and PMI_C(r) < 0)
    print(f"\n[Q3] derail 발화 — 헛경보 {n_fire}/{len(F)} · "
          f"헛경보 pmi_open 중앙값 {st.median([PMI_O(r) for r in F]):+.3f} "
          f"(참경보 {st.median([PMI_O(r) for r in T]):+.3f})")

    json.dump({"model": args.model, "n_true": len(T), "n_false": len(F),
               "axes": res, "derail_fire_false": n_fire, "rows": rows},
              open(args.out, "w"), indent=1)
    print(f"[2] wrote {args.out}")


if __name__ == "__main__":
    main()
