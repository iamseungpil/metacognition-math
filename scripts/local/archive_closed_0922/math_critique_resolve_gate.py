#!/usr/bin/env python
r"""math_critique_resolve_gate — «문맥 안에서 판단·비평하고, 답 없이 다시 푼다» 게이트.

측정된 사실(cd9): **완성된 오답 풀이를 같은 문맥에서 이어 쓰면 절대 안 살아난다** —
seeded 모드 전부에서 0/1,248, 86%는 첫 답을 글자 그대로 되풀이하고, 심어 준 `redirect` 는
95.6% 가 새 `decision: verify` 로 덮인다. 반면 **독립 재표본**은 49% 를 살린다. 그러니까
문제는 «판단» 이 아니라 **첫 답이 문맥에 남아 있다는 것**이라는 가설이 선다.

이 게이트가 그 가설을 판정한다. 설계: 문맥 안에서 판단·비평하되(1단계), 다시 풀 때는
**첫 답을 지우고 비평만 들고** 푼다(2단계).

    A. critique   원 풀이를 assistant 턴으로 그대로 담고 "틀렸다 — 답·숫자 말고 무엇이
                  잘못됐는지 2~3문장" 을 묻는다(그리디 1회). 답 누출은 거른다.
    B. re-solve   같은 variant system 프롬프트 아래 K 개씩 (temperature 1.0):
        (i)   blind        문제만 — **재표본 기준선**(49% 자리)
        (ii)  critique     문제 + "Note from a previous attempt: <비평>"
        (iii) donor        (ii) 와 같되 **다른 문제**의 비평 — 내용 대조
        (iv)  in-context   원 풀이 + 메타 씨앗 + "Second attempt:" 이어쓰기 —
                           **우리가 버리는 설계**(기대값 ≈ 0)

★(iii) donor 대조가 핵심이다. 이게 없으면 «아무 말이나 덧붙이면 분포가 흔들려 오른다»
  (재표본·온도 효과)를 «비평 내용이 유용했다»로 오독한다 — cd9 EVC 판정과 같은 함정.
★(i) blind 기준선도 핵심이다. 비평 조건이 blind 보다 높지 않으면 비평은 «다시 풀기» 이상의
  값이 없고, 그러면 비평을 보상하는 RL 은 형식만 학습한다.
★고정점(anchoring): 비평이 답을 흘리면(누출 가드를 통과해도) 다시 푼 답이 원래 오답으로
  되돌아온다. `anchor_rate` = 비평 재풀이의 최종 답이 **원래 오답과 동치**인 비율 — 이게
  blind 보다 크게 높으면 Δ가 양수여도 «비평이 오답을 붙잡고 있다»는 반대 증거다.

통과 규칙: (crit − blind) 의 95% CI 가 0 을 제외 ∧ 평균 > +0.03
       ∧ (crit − donor) 의 95% CI 가 0 을 제외 ∧ 평균 > 0.

사용(예):
  python scripts/local/math_critique_resolve_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path <hf> --variant math_opt --k 8 --max_sites 200 --seed 11 \
      --out_dir /hdd_data/seungpil/scratch/eval/critique_resolve_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import (  # noqa: E402  (선별·통계 정의를 한 곳에 둔다)
    bootstrap_ci, select_wrong_rollouts, sign_test_p,
)
from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_meta import (  # noqa: E402
    NOTE_PREFIX, answers_equivalent, assign_donors, critique_leaks, critique_len,
    grade_math, last_boxed, selftest_math_verify,
)

# ★0914c 단일 진실 원천: 누출 가드(critique_leaks)·비평 길이(critique_len)·donor 배정
#   (assign_donors)·NOTE_PREFIX 는 이제 `src/training/math_meta.py` 에 산다. M_CRIT 팔이
#   학습 중에 **같은 판정**으로 IG 항을 정의하기 때문이다 — 정의가 두 벌이면 «게이트가 통과시킨
#   비평»과 «RL 이 보상한 비평»이 조용히 갈린다. 이름은 여기서 그대로 재수출한다(기존 호출자·
#   테스트·math_critique_ig_ruler 의 import 경로 불변).
__all__ = ["NOTE_PREFIX", "assign_donors", "critique_leaks", "critique_len"]

_NAN = float("nan")
PASS_DELTA_BLIND = 0.03          # crit − blind 의 하한(평균)
CONDS = ("blind", "crit", "donor", "incontext")

CRITIQUE_ASK = (
    "Your answer above is wrong. Without stating any final answer or numbers from your "
    "solution, write a 2-3 sentence critique: which step or assumption is most likely "
    "mistaken and what a correct approach must do differently."
)
# ★(iv) 씨앗 — cd9 가 실측한 그 설계 그대로(문맥에 첫 답이 남은 채 redirect 를 심는다).
INCONTEXT_SEED = ("\n<meta>\nconfidence: 0.2\n{critique}\ndecision: redirect\n</meta>\n"
                  "Second attempt:")

_SENTINEL = "<<<CRITIQUE_GATE_SOLUTION>>>"


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────────
def build_critique_prompt(tok, variant: str, problem: str, text: str,
                          ask: str = CRITIQUE_ASK) -> str:
    """원 풀이를 assistant 턴으로 **바이트 동일**하게 담고 새 user 턴(ask)을 붙인 생성 프롬프트.

    ★sentinel 되꽂기: Qwen3.5 템플릿은 assistant 본문의 끝 공백을 지운다 — 본문을 그냥 넣으면
      프롬프트가 원 롤아웃 문맥과 조용히 달라진다(math_cited_site_gate 와 같은 규약).
    """
    msgs = build_math_prompt(problem, variant) + [
        {"role": "assistant", "content": _SENTINEL},
        {"role": "user", "content": ask},
    ]
    rendered = render_chat_messages(tok, msgs)
    if rendered.count(_SENTINEL) != 1:
        raise RuntimeError(f"[CRIT] chat 템플릿에서 sentinel 을 {rendered.count(_SENTINEL)}번 "
                           "찾았다 — 1번이어야 한다(템플릿이 assistant 본문을 변형한다).")
    head, tail = rendered.split(_SENTINEL)
    return head + (text or "") + tail


def resolve_prompt(tok, variant: str, problem: str, note: str | None = None) -> str:
    """재풀이 프롬프트. note 가 None 이면 render_generation_prompt 와 **바이트 동일**(blind),
    있으면 user 턴 끝에 NOTE_PREFIX + note 만 붙는다 — system 프롬프트는 네 조건이 모두 같다.
    ★user 턴에 붙이는 이유: 새 system 문구를 만들면 «비평의 효과»와 «프롬프트가 달라진 효과»가
      섞인다. 조건 사이의 유일한 차이는 note 문자열이어야 한다."""
    if note is None:
        return render_generation_prompt(tok, variant, problem)
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + NOTE_PREFIX + note.strip()}
    return render_chat_messages(tok, msgs)


def incontext_prompt(tok, variant: str, problem: str, text: str, critique: str) -> str:
    """(iv) 이어쓰기 문맥 = 롤아웃이 생성된 문맥 + 원 풀이 전체 + 메타 씨앗.
    ★생성 문맥과 바이트 동일해야 «같은 자리»다(math_cited_site_gate.continuation_prompt 규약).
    반환값의 접두를 그대로 채점 앞부분으로 쓰지 않는다 — 채점은 이어쓴 부분만 본다."""
    return (render_generation_prompt(tok, variant, problem) + (text or "")
            + INCONTEXT_SEED.format(critique=critique.strip()))


# ── 비평 누출 가드·길이 (정의는 src/training/math_meta.py — 위 import 참조) ─────────
def clean_critique(raw: str) -> str:
    """생성물에서 비평 본문만 남긴다(앞뒤 공백·따옴표 제거). 문장 자르기는 하지 않는다 —
    max_tokens=200 이 상한이고, 잘린 비평도 그대로 조건에 들어가야 실제 설계와 같다."""
    return (raw or "").strip().strip('"').strip()


# ── donor 배정 (정의는 src/training/math_meta.assign_donors — 위 import 참조) ────────


# ── 요약 ────────────────────────────────────────────────────────────────────────
def _p(vals: Sequence[int]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN


def summarize(recs: Sequence[dict], *, k: int = 0, seed: int = 0, n_boot: int = 2000,
              n_leaked: int = 0, n_no_critique: int = 0) -> dict:
    """per-rollout 기록 → 게이트 요약. recs 는 네 조건이 모두 성립한 롤아웃만."""
    n = len(recs)
    d_cb = [r["p_crit"] - r["p_blind"] for r in recs]
    d_cd = [r["p_crit"] - r["p_donor"] for r in recs]
    d_ib = [r["p_incontext"] - r["p_blind"] for r in recs]
    n_att = n + n_leaked + n_no_critique
    out = {
        "n_rollouts": n, "k": k,
        "n_leaked": int(n_leaked), "n_no_critique": int(n_no_critique),
        "leak_rate": (n_leaked / n_att) if n_att else _NAN,
        "critique_words": bootstrap_ci([r["critique_words"] for r in recs], seed=seed + 9,
                                       n_boot=n_boot),
        "p_blind": bootstrap_ci([r["p_blind"] for r in recs], seed=seed, n_boot=n_boot),
        "p_crit": bootstrap_ci([r["p_crit"] for r in recs], seed=seed + 1, n_boot=n_boot),
        "p_donor": bootstrap_ci([r["p_donor"] for r in recs], seed=seed + 2, n_boot=n_boot),
        "p_incontext": bootstrap_ci([r["p_incontext"] for r in recs], seed=seed + 3, n_boot=n_boot),
        "paired_crit_minus_blind": bootstrap_ci(d_cb, seed=seed + 4, n_boot=n_boot),
        "paired_crit_minus_donor": bootstrap_ci(d_cd, seed=seed + 5, n_boot=n_boot),
        "paired_incontext_minus_blind": bootstrap_ci(d_ib, seed=seed + 6, n_boot=n_boot),
        "sign_p_crit_minus_blind": sign_test_p(d_cb),
        "sign_p_crit_minus_donor": sign_test_p(d_cd),
        "sign_p_incontext_minus_blind": sign_test_p(d_ib),
        "frac_crit_gt_blind": (sum(1 for d in d_cb if d > 0) / n) if n else _NAN,
        "frac_crit_gt_donor": (sum(1 for d in d_cd if d > 0) / n) if n else _NAN,
        # ★고정점: 비평 재풀이가 원래 오답으로 되돌아온 비율(blind 와 나란히 읽는다).
        "anchor_rate_crit": bootstrap_ci([r["anchor_crit"] for r in recs], seed=seed + 7,
                                         n_boot=n_boot),
        "anchor_rate_blind": bootstrap_ci([r["anchor_blind"] for r in recs], seed=seed + 8,
                                          n_boot=n_boot),
        "trunc_rate": (sum(r.get("trunc_rate", 0.0) for r in recs) / n) if n else _NAN,
    }
    out["pass"] = int(gate_pass(out))
    return out


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def gate_pass(summ: dict) -> bool:
    """crit−blind: CI 가 0 제외 ∧ 평균 > +.03;  crit−donor: CI 가 0 제외 ∧ 평균 > 0."""
    cb = summ.get("paired_crit_minus_blind") or {}
    cd = summ.get("paired_crit_minus_donor") or {}
    if not (_finite(cb, "lo", "hi", "mean") and _finite(cd, "lo", "hi", "mean")):
        return False
    ok_cb = (cb["lo"] > 0 or cb["hi"] < 0) and cb["mean"] > PASS_DELTA_BLIND
    ok_cd = (cd["lo"] > 0 or cd["hi"] < 0) and cd["mean"] > 0
    return bool(ok_cb and ok_cd)


def _f(v) -> str:
    if isinstance(v, dict):
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(summ: dict) -> str:
    keys = ["n_rollouts", "k", "n_leaked", "n_no_critique", "leak_rate", "critique_words",
            "p_blind", "p_crit", "p_donor", "p_incontext",
            "paired_crit_minus_blind", "paired_crit_minus_donor",
            "paired_incontext_minus_blind", "sign_p_crit_minus_blind",
            "sign_p_crit_minus_donor", "sign_p_incontext_minus_blind",
            "frac_crit_gt_blind", "frac_crit_gt_donor",
            "anchor_rate_crit", "anchor_rate_blind", "trunc_rate"]
    lines = ["## math_critique_resolve_gate — 답 없이, 비평만 들고 다시 풀기", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(summ.get(k))} |" for k in keys]
    lines += ["", f"**{'PASS' if summ.get('pass') else 'FAIL'}** — 규칙: (crit−blind) CI 가 0 제외 "
                  f"∧ 평균 > +{PASS_DELTA_BLIND} ∧ (crit−donor) CI 가 0 제외 ∧ 평균 > 0", ""]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_sites", type=int, default=200)
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--critique_tokens", type=int, default=200)
    ap.add_argument("--max_prefix_tokens", type=int, default=8192,
                    help="★비평·(iv) 프롬프트는 풀이 전체를 담는다 — mathL5 실측 최대 8,489 토큰")
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    cands = select_wrong_rollouts(rolls, per_problem=a.per_problem)
    rng.shuffle(cands)
    cands = cands[:a.max_sites]
    print(f"[crit] MIXED·오답·미잘림 롤아웃 {len(cands)}개", flush=True)
    if not cands:
        raise SystemExit("[crit] 후보가 없다 — 입력 롤아웃에 MIXED 오답이 있는지 확인하라.")

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()
    lim = a.max_prefix_tokens + 512

    # ── A) 비평 한 번(그리디) ────────────────────────────────────────────────
    reqs, ix, n_long = [], [], 0
    for i, r in enumerate(cands):
        q = build_critique_prompt(tok, a.variant, r["problem"], r["text"])
        if len(tok.encode(q)) > lim:
            n_long += 1
            continue
        reqs.append(q)
        ix.append(i)
    print(f"[crit] 비평 요청 {len(reqs)}개 (너무 긴 것 {n_long}개 제외, 한도 {lim} 토큰)", flush=True)
    couts = llm.generate(reqs, SamplingParams(n=1, temperature=0.0,
                                              max_tokens=a.critique_tokens, seed=a.seed))

    kept: list[dict] = []
    n_leaked = n_empty = 0
    crit_rows = []
    for i, o in zip(ix, couts):
        r = cands[i]
        crit = clean_critique(o.outputs[0].text)
        wrong_ans = last_boxed(r["text"])
        leak = critique_leaks(crit, wrong_ans)
        crit_rows.append({"roll_id": r["roll_id"], "group_id": r["group_id"],
                          "wrong_answer": wrong_ans, "critique": crit,
                          "critique_words": critique_len(crit), "leaked": int(leak)})
        if not crit:
            n_empty += 1
            continue
        if leak:
            n_leaked += 1
            continue
        kept.append({**r, "critique": crit, "wrong_answer": wrong_ans,
                     "critique_words": critique_len(crit)})
    with (out / "critiques.jsonl").open("w") as fh:
        for c in crit_rows:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[crit] 비평 성립 {len(kept)} / 누출 {n_leaked} / 빈 것 {n_empty}", flush=True)
    if not kept:
        raise SystemExit("[crit] 유효한 비평이 하나도 없다 — 프롬프트/누출 가드를 확인하라.")

    donors = assign_donors(kept, rng)
    for r, d in zip(kept, donors):
        r["donor_idx"] = d
        r["donor_critique"] = kept[d]["critique"] if d is not None else None

    # ── B) 네 조건 재풀이 ────────────────────────────────────────────────────
    reqs, ix, n_drop = [], [], 0
    for si, s in enumerate(kept):
        prompts = {
            "blind": resolve_prompt(tok, a.variant, s["problem"], None),
            "crit": resolve_prompt(tok, a.variant, s["problem"], s["critique"]),
            "donor": (resolve_prompt(tok, a.variant, s["problem"], s["donor_critique"])
                      if s["donor_critique"] else None),
            "incontext": incontext_prompt(tok, a.variant, s["problem"], s["text"], s["critique"]),
        }
        for c in CONDS:
            q = prompts[c]
            if q is None or len(tok.encode(q)) > lim:
                n_drop += 1
                continue
            reqs.append(q)
            ix.append((si, c))
    print(f"[crit] 재풀이 요청 {len(reqs)}개 x K={a.k} (버린 것 {n_drop}개)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    rows = []
    for (si, c), o in zip(ix, outs):
        s = kept[si]
        for x in o.outputs:
            # ★(iv) 는 이어쓴 부분만 채점한다 — 앞부분에 이미 (틀린) \boxed 가 있으므로
            #   prefix+cont 로 채점하면 «이어쓰기가 무엇을 했는가»가 아니라 앞부분이 점수를 정한다.
            corr = grade_math(x.text, s["gold"])
            ans = last_boxed(x.text)
            anchored = int(bool(ans) and answers_equivalent(s["wrong_answer"], ans))
            agg.setdefault((si, c), []).append(
                {"r_corr": corr, "anchor": anchored, "trunc": int(x.finish_reason == "length")})
            rows.append({"roll_id": s["roll_id"], "cond": c, "r_corr": corr, "anchor": anchored,
                         "truncated": int(x.finish_reason == "length"), "text": x.text})

    # ── per-rollout ─────────────────────────────────────────────────────────
    recs = []
    for si, s in enumerate(kept):
        got = {c: agg.get((si, c)) for c in CONDS}
        if not all(got.values()):
            continue                        # 네 조건이 다 있어야 짝 비교가 성립한다
        rec = {
            "roll_id": s["roll_id"], "group_id": s["group_id"], "problem": s["problem"],
            "gold": s["gold"], "wrong_answer": s["wrong_answer"],
            "critique": s["critique"], "critique_words": s["critique_words"],
            "donor_roll_id": (kept[s["donor_idx"]]["roll_id"] if s["donor_idx"] is not None
                              else None),
            "p_blind": _p([x["r_corr"] for x in got["blind"]]),
            "p_crit": _p([x["r_corr"] for x in got["crit"]]),
            "p_donor": _p([x["r_corr"] for x in got["donor"]]),
            "p_incontext": _p([x["r_corr"] for x in got["incontext"]]),
            "anchor_crit": _p([x["anchor"] for x in got["crit"]]),
            "anchor_blind": _p([x["anchor"] for x in got["blind"]]),
            "trunc_rate": _p([x["trunc"] for c in CONDS for x in got[c]]),
        }
        recs.append(rec)

    summ = summarize(recs, k=a.k, seed=a.seed, n_boot=a.n_boot, n_leaked=n_leaked,
                     n_no_critique=n_empty)
    summ.update({"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
                 "seed": a.seed, "n_candidates": len(cands), "n_dropped_long_prompt": n_long,
                 "n_dropped_long_resolve": n_drop, "n_generations": len(rows)})

    with (out / "per_rollout.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "resolves.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(to_markdown(summ))
    print(to_markdown(summ))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
