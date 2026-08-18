"""R_meta 공식 sweep -- 원재료는 한 번 재고, 공식은 공짜로 쓸어본다.

핵심 구조. PMI-shift 계열의 모든 변형은 **같은 원재료**에서 나온다.

    pmi_open        메타 앞 위치의 gold−decoy logp
    pmi_close       메타 뒤 위치의 gold−decoy logp
    pmi_ans_real    답 직전 위치, 메타 = 진짜
    pmi_ans_donor   답 직전 위치, 메타 = 다른 문제 것
    lp_post_real    메타 뒤 본문의 logp, 문맥 = 진짜 메타
    lp_post_donor   같은 본문의 logp, 문맥 = 도너 메타
    n_post          그 본문의 토큰 수
    acc_R, acc_B    재생성 채점 (splice_probe 산출) -> **인과 기준선**

원재료를 롤아웃당 한 번 로그해 두면, 공식 수십 개가 **GPU 없이 산술로** 나온다.
그래서 "sweep 처럼 다양한 pmi shift"가 가능하다 -- 변형마다 새로 생성할 필요가 없다.

그리고 공식을 무엇으로 채점하나. 여기가 이 파일의 요점이다:

    ★각 공식이 **d_true = acc(진짜 메타) − acc(정형문)** 를 얼마나 예측하는가

d_true 는 같은 접합점에서 메타만 갈아 끼우고 **실제로 다시 풀린** 결과다. 대리가 아니다.
싼 공식 중 d_true 를 잘 따라가는 것이 있으면, 매 스텝 재생성하지 않고 그 공식을 써도
된다는 **자격**이 생긴다. 하나도 못 따라가면 싼 대리는 없는 것이고, 부분 재생성으로 간다.

⚠ 이 파일은 공식을 고를 뿐, "그 공식으로 학습하면 좋아진다"를 증명하지 않는다.
   그건 학습을 돌려야 안다. 여기서 하는 일은 **돌릴 가치가 있는 공식을 고르는 것**이다.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st

import sys

import torch

from abl_pmi import (body_before_answer, make_decoy, pmi_at, _sum_logp,
                     with_meta)

# 공식은 **정본 모듈 한 곳**에만 있다. 여기서 다시 정의하면 오프라인에서 고른 식과
# 학습에서 도는 식이 갈릴 수 있고, 이 저장소는 정확히 그 사고 전례가 있다(원장 0726).
sys.path.insert(0, "/home/v-seungplee/metacognition-math")
from src.training.dcpo_rmeta_forms import FORMULAS, rmeta_from_ingredients  # noqa: E402


# ─────────────────────────────────────────────── 1. 원재료 (GPU, 한 번)

@torch.inference_mode()
def ingredients(model, tok, it, cont_text, device="cuda"):
    """한 롤아웃의 원재료 전부. 여기서만 GPU 를 쓴다."""
    gold, decoy = it["gold"], make_decoy(it["gold"])
    prompt, prefix = it["prompt"], it["prefix"]
    real, donor = it["R"], it["D"]

    ctx_open = prompt + prefix
    ctx_close = prompt + prefix + f"\n<meta>{real}</meta>"
    ctx_close_d = prompt + prefix + f"\n<meta>{donor}</meta>"

    # 메타 뒤 본문 = R 조건에서 실제로 생성된 이어쓰기의, 답 앞부분
    post = body_before_answer(cont_text)
    if post is None or len(post.strip()) < 20:
        return None
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    if not (5 <= len(post_ids) <= 900):
        return None
    full = slice(0, len(post_ids))

    # 메타 자신의 로그확률 -- RLRT (arXiv 2605.10781, "Rebellious Student") 계열.
    # 그 논문의 원리는 "학생이 교사가 예측하지 않았을 경로로 성공했을 때 그 토큰을 강화한다"
    # 이고, 우리 자리로 옮기면 "사전확률이 낮은 메타가 성공을 앞설 때 상을 준다"가 된다.
    # 정형문은 정의상 **최빈 메타**라 이 축에서 자동으로 0 에 붙는다 -- 문맥 차분과는
    # 완전히 다른 이유로 같은 결론에 도달하므로, 둘이 함께 오르면 서로의 확인이 된다.
    meta_ids = tok(f"\n<meta>{real}</meta>", add_special_tokens=False)["input_ids"]
    lp_meta = _sum_logp(model, tok, ctx_open, meta_ids, slice(0, len(meta_ids)), device)

    out = {
        "lp_meta": lp_meta,
        "n_meta_tok": len(meta_ids),
        "pmi_open":      pmi_at(model, tok, ctx_open,    gold, decoy, device),
        "pmi_close":     pmi_at(model, tok, ctx_close,   gold, decoy, device),
        "pmi_ans_real":  pmi_at(model, tok, ctx_close + post,   gold, decoy, device),
        "pmi_ans_donor": pmi_at(model, tok, ctx_close_d + post, gold, decoy, device),
        # 같은 본문을 두 문맥에서 채점 -> "메타가 그 다음 사고를 이끌었나"
        "lp_post_real":  _sum_logp(model, tok, ctx_close,   post_ids, full, device),
        "lp_post_donor": _sum_logp(model, tok, ctx_close_d, post_ids, full, device),
        "n_post": len(post_ids),
        "meta_len": len(tok(real, add_special_tokens=False)["input_ids"]),
    }
    return out


# ─────────────────────────────────────────────── 2. 공식 (CPU, 공짜)

# ─────────────────────────────────────────────── 3. 채점

def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", default="splice_8b.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--out", default="sweep.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    blob = json.load(open(args.splice))
    items = blob["items"][:args.limit]
    accR, accB = blob["per_problem"]["R"], blob["per_problem"]["B"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for i, it in enumerate(items):
        if "prompt" not in it or "cont_R" not in it:
            continue
        g = ingredients(model, tok, it, it["cont_R"])
        if g is None:
            continue
        g["d_true"] = accR[i] - accB[i]          # ★인과 기준선
        g["is_boiler"] = False
        rows.append(g)
    print(f"원재료 {len(rows)} 행")

    res = {}
    dt = [r["d_true"] for r in rows]
    for name in FORMULAS:
        try:
            v = [rmeta_from_ingredients(r, name) for r in rows]
        except Exception as e:
            res[name] = {"error": repr(e)[:80]}
            continue
        sd = st.pstdev(v)
        res[name] = {
            "spearman_vs_d_true": round(spearman(v, dt), 4),
            "mean": round(st.mean(v), 4),
            "sd": round(sd, 4),
            # 퍼짐이 없으면 그룹 중심화 뒤 gradient 가 0 이다 -- 상관이 높아도 못 쓴다
            "degenerate": sd < 1e-6,
        }
    ranked = sorted((k for k in res if "error" not in res[k]),
                    key=lambda k: -abs(res[k]["spearman_vs_d_true"]))
    out = {"n": len(rows), "formulas": res, "ranked": ranked,
           "NEG_control_rank": ranked.index("NEG_len_only") + 1 if "NEG_len_only" in ranked else None}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
