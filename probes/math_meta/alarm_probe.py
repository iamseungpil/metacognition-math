"""경보 프로브 — R_meta 후보 신호들이 **헛경보를 아는가**.

왜 이것인가. 0817 실측(splice_4b, n=300)에서 자기점검의 손익이 **경보의 참/거짓으로**
정확히 갈렸다:

    [잘 푸는 문제 acc_N=1]  오류주장 n=121  R-N  -0.0537      <- 헛경보
                           주장없음 n= 87  R-N  -0.0192
    [틀리는 문제 acc_N=0]   오류주장 n= 38  R-N  +0.0658      <- 진짜 경보
                           주장없음 n= 17  R-N  +0.0196

같은 행동인데 **부호가 뒤집힌다.** 그러면 좋은 R_meta 의 조건은 하나로 좁혀진다 --
***진짜 경보를 헛경보보다 높게 매기는가.*** 이 파일은 그것을 직접 잰다.

이 프로젝트는 R_meta 를 열 달 넘게 썼지만 **이 검정을 한 번도 하지 않았다.** 지금까지
잰 것은 "정형문이 상을 받나"(예), "redirect 가 벌을 받나"(예) 같은 *증상*이었고,
"신호가 옳은 것을 고르나"라는 *타당성*은 안 봤다.

세 가지를 한 판에서 낸다 -- GPU 한 번에 세 질문.

  Q1  판별력    14개 공식 각각에 대해 AUC(진짜경보 > 헛경보). 0.5 = 눈이 멀었다.
  Q2  중심화    학습은 그룹 평균을 뺀다. 같은 prefix 의 메타 넷(R/B/D/S)으로 그룹을
                흉내내어 **중심화 후에도 판별력이 남는지** 본다. 벌이 지워지는지 여부.
  Q3  부호역전  pos_full 의 derail(-2.0)이 발화하려면 pmi_open>0 & pmi_close<0 이어야
                한다. 잘 푸는 문제에서 pmi_open 이 이미 크게 양수면 **영원히 안 터진다.**
                그 발화율을 실측한다.

⛔주의: 이것은 EXP(탐색) 트랙이다. 헤드라인 금지. 판정문은 codex-sol 게이트 후.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys

import torch

from abl_pmi import body_before_answer, make_decoy, pmi_at, _sum_logp

sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training.dcpo_rmeta_forms import FORMULAS, rmeta_from_ingredients  # noqa: E402


# ────────────────────────────────────────────────────── 경보 탐지기 (문자열)

# "오류를 주장한다" = 작업물의 어딘가가 **틀렸다고 단언**한다.
# "확인을 권한다" 와 구별해야 한다 -- 0817 실측에서 둘의 손해가 2.8배 차이났다.
_CLAIM = re.compile(
    r"\b("
    r"incorrect\w*|wrong\w*|erroneous\w*|error\w*|mistak\w*|flaw\w*|invalid\w*|"
    r"fails? to|failed to|does not (?:account|consider|include|hold|follow|match)|"
    r"did not (?:account|consider|include)|"
    r"missing|omits?|omitted|overlook\w*|ignor(?:es|ed)|neglect\w*|"
    r"misinterpret\w*|misappl\w*|misus\w*|miscalculat\w*|misread\w*|"
    r"not (?:valid|correct|justified|true|accurate)|"
    r"instead of|should (?:be|have been) \$"
    r")\b", re.I)

# 이것들은 **주장이 아니다** -- 확인 권고다. 먼저 지운 뒤 _CLAIM 을 건다.
_HEDGE = re.compile(
    r"should (?:be )?(?:verified|checked|confirmed|double[- ]?checked|re-?checked|"
    r"validated|re-?examined)|"
    r"without an independent check|needs? (?:to be )?(?:verified|checked)",
    re.I)


def meta_body(m: str) -> str:
    """confidence/decision 껍데기를 벗긴 한 문장."""
    parts = [p.strip() for p in m.split("|")]
    mid = [p for p in parts if not p.lower().startswith(("confidence:", "decision:"))]
    return " ".join(mid) if mid else m


def claims_error(m: str) -> int:
    body = _HEDGE.sub(" ", meta_body(m))
    return 1 if _CLAIM.search(body) else 0


def get_conf(m: str):
    g = re.search(r"confidence:\s*([0-9.]+)", m)
    return float(g.group(1)) if g else None


# ────────────────────────────────────────────────────── 통계 도구 (순수 python)

def auc(pos, neg):
    """P(무작위 pos > 무작위 neg). 동점은 0.5. n 이 작으니 전수 비교."""
    if not pos or not neg:
        return None
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def boot_auc_ci(pos, neg, n=2000, seed=0):
    import random
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        p = [rng.choice(pos) for _ in pos]
        q = [rng.choice(neg) for _ in neg]
        a = auc(p, q)
        if a is not None:
            vals.append(a)
    vals.sort()
    if not vals:
        return (None, None)
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])


def mean(xs):
    return st.mean(xs) if xs else float("nan")


# ────────────────────────────────────────────────────── 원재료 한 항목

@torch.inference_mode()
def ingredients(model, tok, it, meta_key, donor_key, device="cuda"):
    """한 (문제, 메타) 쌍의 원재료 10종. 본문·도너가 고정이라 matched 다."""
    gold = it["gold"]
    decoy = make_decoy(gold)
    meta = it[meta_key]
    donor = it[donor_key]

    ctx_open = it["prompt"] + it["prefix"]
    ctx_close = ctx_open + f"\n<meta>{meta}</meta>"
    ctx_donor = ctx_open + f"\n<meta>{donor}</meta>"

    meta_ids = tok(f"\n<meta>{meta}</meta>", add_special_tokens=False)["input_ids"]

    # ★후속 본문(post)은 **문맥차 계열에만** 필요하다. 첫 판(0817 01:0x)에서 post 가
    #   없는 행을 통째로 버렸더니 **참 경보 41 -> 5** 로 무너졌다 -- 틀리는 문제일수록
    #   답을 못 맺어 \boxed 가 없고, 그래서 양성 계급만 집중적으로 잘려나갔다.
    #   위치차 계열(현행 PMI-shift)은 post 가 필요 없으므로 여기서 갈라 받는다.
    g = {
        "lp_meta": _sum_logp(model, tok, ctx_open, meta_ids,
                             slice(0, len(meta_ids)), device),
        "n_meta_tok": len(meta_ids),
        "pmi_open": pmi_at(model, tok, ctx_open, gold, decoy, device),
        "pmi_close": pmi_at(model, tok, ctx_close, gold, decoy, device),
        "meta_len": len(tok(meta, add_special_tokens=False)["input_ids"]),
        # 아래 넷은 post 가 있을 때만 진짜 값. 없으면 0 을 채우되 post_ok=False 로
        # 표시하고, 채점 단계에서 해당 공식은 **그 행을 아예 안 쓴다**(가짜 0 금지).
        "pmi_ans_real": 0.0, "pmi_ans_donor": 0.0,
        "lp_post_real": 0.0, "lp_post_donor": 0.0, "n_post": 1,
        "post_ok": False,
    }

    post = body_before_answer(it.get("cont_R") or "") or ""
    post_ids = tok(post, add_special_tokens=False)["input_ids"][:600]
    if len(post_ids) >= 5:
        sl = slice(0, len(post_ids))
        g.update({
            "pmi_ans_real": pmi_at(model, tok, ctx_close + post, gold, decoy, device),
            "pmi_ans_donor": pmi_at(model, tok, ctx_donor + post, gold, decoy, device),
            "lp_post_real": _sum_logp(model, tok, ctx_close, post_ids, sl, device),
            "lp_post_donor": _sum_logp(model, tok, ctx_donor, post_ids, sl, device),
            "n_post": len(post_ids), "post_ok": True,
        })
    return g


@torch.inference_mode()
def pmi_close_only(model, tok, it, meta, device="cuda"):
    """중심화 검사용 -- 그룹 형제의 pos 만 싸게 얻는다 (pmi_open 은 공유)."""
    gold, decoy = it["gold"], make_decoy(it["gold"])
    ctx = it["prompt"] + it["prefix"] + f"\n<meta>{meta}</meta>"
    return pmi_at(model, tok, ctx, gold, decoy, device)


# ────────────────────────────────────────────────────── 본체

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_4b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="alarm_probe.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blob = json.load(open(args.splice))
    items = blob["items"]
    accN = blob["per_problem"]["N"]
    accR = blob["per_problem"]["R"]
    if args.limit:
        items, accN, accR = items[:args.limit], accN[:args.limit], accR[:args.limit]

    # ── 0. 배선검사: 경보 탐지기가 0817 분해와 같은 수를 내는가
    #    (이것을 먼저 안 하면 아래 전부가 다른 라벨 위에 서게 된다)
    for it, a in zip(items, accN):
        it["_claim"] = claims_error(it["R"])
        it["_accN"] = a
    n_easy = [it for it in items if it["_accN"] >= 0.999]
    n_hard = [it for it in items if it["_accN"] <= 0.001]
    print(f"[0] 배선검사 — 층 크기  acc_N=1 {len(n_easy)}  acc_N=0 {len(n_hard)}  "
          f"중간 {len(items)-len(n_easy)-len(n_hard)}")
    print(f"    잘 푸는 문제 · 오류주장 {sum(i['_claim'] for i in n_easy)}"
          f" / 주장없음 {sum(1-i['_claim'] for i in n_easy)}   (0817 기대: 121 / 87)")
    print(f"    틀리는 문제 · 오류주장 {sum(i['_claim'] for i in n_hard)}"
          f" / 주장없음 {sum(1-i['_claim'] for i in n_hard)}   (0817 기대:  38 / 17)")

    # 라벨이 실제 손익과 같은 부호를 내는지도 여기서 확인한다 (GPU 0)
    for name, pool in (("잘 푸는", n_easy), ("틀리는", n_hard)):
        for c in (1, 0):
            sub = [(accR[items.index(it)] - it["_accN"]) for it in pool if it["_claim"] == c]
            if sub:
                print(f"    {name} · {'주장O' if c else '주장X'} n={len(sub):3d} "
                      f"R-N {mean(sub):+.4f}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    # ── 1. 원재료. 도너는 D(남의 문제 메타) -- 길이·문체가 맞춰진 유일한 통제군.
    rows = []
    for n, it in enumerate(items):
        g = ingredients(model, tok, it, "R", "D")
        if g is None:
            continue
        sib = {}
        for k in ("B", "D", "S"):
            if it.get(k):
                sib[k] = {**g, "pmi_close": pmi_close_only(model, tok, it, it[k]),
                          "meta_len": len(tok(it[k], add_special_tokens=False)["input_ids"])}
        rows.append({"i": n, "g": g, "sib": sib, "claim": it["_claim"],
                     "accN": it["_accN"], "accR": accR[n], "conf": get_conf(it["R"])})
        if (n + 1) % 50 == 0:
            print(f"    ... {n+1}/{len(items)}  (유효 {len(rows)})")
    print(f"[1] 원재료 완료 {len(rows)}/{len(items)}")

    # ── 2. 라벨: 참 경보 / 헛경보
    TRUE_A = [r for r in rows if r["claim"] == 1 and r["accN"] <= 0.001]
    FALSE_A = [r for r in rows if r["claim"] == 1 and r["accN"] >= 0.999]
    print(f"[2] 참 경보 {len(TRUE_A)} · 헛경보 {len(FALSE_A)}")

    # ── Q1 판별력 · Q2 중심화 후 판별력
    #
    # ⚠공식마다 **쓸 수 있는 행이 다르다.** 문맥차 계열은 post 가 필요하고, 그 행들은
    #   양성 계급에서 훨씬 자주 없다. 그래서 한 모집단으로 몰아 비교하면 공식 비교가
    #   아니라 **모집단 비교**가 된다. 각 공식의 n 을 같이 찍어 그 차이를 눈에 보이게 둔다.
    NEEDS_POST = {"ans", "ans_clip", "ans_cond", "drive", "drive_raw",
                  "ans+drive", "surprise+ans"}
    # 형제(B/D/S)는 pmi_close 만 다르게 채웠으므로 문맥차 계열의 중심화는 정의상 0 이
    # 된다 -- 그건 발견이 아니라 인공물이다. 위치차 계열에서만 중심화를 판정한다.
    CENTER_OK = {"pos", "pos_clip", "pos_rev_only", "pos_full", "pos_cond",
                 "NEG_len_only"}

    res = {}
    for name in FORMULAS:
        need = name in NEEDS_POST

        def val(r, centered=False):
            v = rmeta_from_ingredients(r["g"], name)
            if not centered:
                return v
            sibs = [rmeta_from_ingredients(s, name) for s in r["sib"].values()]
            return v - st.mean([v] + sibs)   # 그룹 중심화 (학습이 하는 그 일)

        T = [r for r in TRUE_A if r["g"]["post_ok"] or not need]
        F = [r for r in FALSE_A if r["g"]["post_ok"] or not need]
        pos, neg = [val(r) for r in T], [val(r) for r in F]
        a = auc(pos, neg)
        lo, hi = boot_auc_ci(pos, neg)
        ac = auc([val(r, True) for r in T], [val(r, True) for r in F]) \
            if name in CENTER_OK else None
        res[name] = {
            "auc_raw": round(a, 4) if a is not None else None,
            "auc_ci": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "auc_centered": round(ac, 4) if ac is not None else None,
            "mean_true": round(mean(pos), 4), "mean_false": round(mean(neg), 4),
            "n_true": len(pos), "n_false": len(neg),
        }
        star = "★" if a is not None and (lo > 0.5 or hi < 0.5) else " "
        acs = f"{ac:.4f}" if ac is not None else "  n/a "
        print(f"  {star}{name:14s} AUC {a:.4f} [{lo:.3f},{hi:.3f}] n={len(pos):3d}/{len(neg):3d}"
              f"  중심화후 {acs}   참 {mean(pos):+8.3f} vs 헛 {mean(neg):+8.3f}")

    # ── Q3 derail 발화율. 안 터지면 "벌이 없다"가 정의상 참이 된다.
    def rev_fires(r):
        o, c = r["g"]["pmi_open"], r["g"]["pmi_close"]
        return (o > 0 and c < 0)
    fire_false = sum(rev_fires(r) for r in FALSE_A)
    fire_true = sum(rev_fires(r) for r in TRUE_A)
    open_pos_false = sum(1 for r in FALSE_A if r["g"]["pmi_open"] > 0)
    print(f"\n[Q3] derail(-2.0) 발화 — 헛경보 {fire_false}/{len(FALSE_A)} · "
          f"참경보 {fire_true}/{len(TRUE_A)}")
    print(f"     헛경보의 pmi_open>0 비율 {open_pos_false}/{len(FALSE_A)}"
          f"  ·  pmi_open 중앙값 {st.median([r['g']['pmi_open'] for r in FALSE_A]):.3f}")

    # ── 부수: confidence 가 헛경보 탐지기로 쓸 만한가 (지금 안 쓰는 정보)
    cf = [r["conf"] for r in FALSE_A if r["conf"] is not None]
    ct = [r["conf"] for r in TRUE_A if r["conf"] is not None]
    ac_conf = auc(ct, cf)
    print(f"[부수] confidence AUC(참>헛) {ac_conf if ac_conf is None else round(ac_conf,4)}"
          f"   참 {mean(ct):.3f} vs 헛 {mean(cf):.3f}")

    ranked = sorted((k for k in res if res[k]["auc_raw"] is not None),
                    key=lambda k: -abs(res[k]["auc_raw"] - 0.5))
    out = {"model": args.model, "n_rows": len(rows),
           "n_true_alarm": len(TRUE_A), "n_false_alarm": len(FALSE_A),
           "formulas": res, "ranked_by_separation": ranked,
           "derail_fire_false": fire_false, "derail_fire_true": fire_true,
           "open_pos_false": open_pos_false,
           "conf_auc": ac_conf,
           "rows": [{k: r[k] for k in ("i", "claim", "accN", "accR", "conf")}
                    | {"g": r["g"]} for r in rows]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n[3] wrote {args.out}   상위 판별력: {ranked[:4]}")


if __name__ == "__main__":
    main()
