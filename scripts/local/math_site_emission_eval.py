#!/usr/bin/env python
"""math_site_emission_eval — 사전등록 §4 의 **첫 번째 판정 지표**: 라벨된 held-out 자리에서
학습된 정책이 «언제 말하고 무엇을 판단하는가».

입력은 math_sites.py 가 만든 sites.jsonl(자리마다 label ∈ {SAVE, DERAIL, NEUTRAL} 과
best_decision ∈ {verify, redirect, tie, None}). 각 자리에서 정책에게 **앞부분만**(아무것도
심지 않고) 먹이고 K 개를 이어 쓰게 한 뒤:

    발화 «그 자리에서» — 이어 쓴 텍스트의 앞 400자 안에 <meta> 블록이 있다
    발화 «어디서든»   — 이어 쓴 텍스트 어딘가에 <meta> 블록이 있다
    판단 일치        — 블록의 decision == 그 자리의 best_decision (verify/redirect 자리만)
    정답             — 앞부분 + 이어 쓴 텍스트를 math_verify 로 채점

§4 의 세 곡선이 여기서 나온다: SAVE 발화 ↑ · DERAIL 발화 ↓ · NEUTRAL 불변. 셋 다 오르면
«자주 말하기»의 재현 = 실패. 그래서 발화율을 **라벨별로** 낸다(전체 발화율은 진단 불가).

★프롬프트는 반드시 허용판(math_opt)이다. 강제판(math_new, «At least once while solving»)은
발화를 정의상 1로 만들어 «언제»를 잴 수 없으므로 실행 자체를 거부한다.
★생성 문맥은 math_sites.gen_request 와 같은 함수로 만든다 — 라벨을 캔 «같은 자리»여야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_sites as MS  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS  # noqa: E402
from src.training.countdown_rewards import parse_meta  # noqa: E402

try:
    from src.training.math_meta import grade_math  # noqa: E402  (RL 워커와 같은 채점기)
except ImportError:  # pragma: no cover
    grade_math = MS._grade

MANDATE_SENTENCE = "At least once while solving"
AT_SITE_CHARS = 400     # «그 자리에서» = 이어 쓴 텍스트의 앞 400자 안에 블록 시작
LABELS = ("SAVE", "DERAIL", "NEUTRAL")

# 생성기 시그니처: prompts -> 프롬프트마다 [(text, truncated), ...] K 개. 테스트에서 모의한다.
Generator = Callable[[Sequence[str]], Sequence[Sequence[tuple[str, int]]]]


def check_variant(variant: str) -> str:
    """강제판 거부 — 지표가 무의미해지므로 조용히 돌지 않고 즉사한다."""
    if variant not in MATH_PROMPT_VARIANTS:
        raise SystemExit(f"unknown variant {variant!r} (choose from {sorted(MATH_PROMPT_VARIANTS)})")
    if MANDATE_SENTENCE in MATH_PROMPT_VARIANTS[variant]:
        raise SystemExit(
            f"variant {variant!r} 는 강제 프롬프트(«{MANDATE_SENTENCE}…»)다 — 발화가 정의상 1 이라 "
            "«언제 말하는가» 지표가 무의미하다. 허용판(math_opt)을 써라.")
    return variant


def classify_continuation(cont: str) -> dict:
    """이어 쓴 텍스트 하나의 메타 발화 판정(form="math": 신뢰도 필수, decision 선택)."""
    got = parse_meta(cont, form="math")
    emitted = int(got["emitted"])
    at_site = int(emitted and got["start"] is not None and got["start"] < AT_SITE_CHARS)
    return {"emitted": emitted, "at_site": at_site, "decision": got["decision"],
            "confidence": got["confidence"], "meta_start": got["start"]}


def _rate(num: float, den: int):
    return (num / den) if den else None


def run_eval(sites: list[dict], generate: Generator, variant: str, k: int, tok=None) -> dict:
    """라벨된 자리 위에서 정책을 K 회 이어 쓰게 하고 라벨별로 모은다. 모델 무관(생성기 주입).

    tok 이 None 이면 generate 는 «(variant, problem, prefix)» 를 알아서 처리하는 모의 생성기다
    — 실제 경로에선 gen_request 로 바이트 동일 문맥을 만든다.
    """
    check_variant(variant)
    if tok is not None:
        prompts = [MS.gen_request(tok, variant, s["problem"], MS.build_fed("nometa", s["prefix"], None))
                   for s in sites]
    else:
        prompts = [s["prefix"] for s in sites]
    outs = generate(prompts)
    if len(outs) != len(sites):
        raise RuntimeError(f"생성기가 {len(outs)}개를 돌려줬다(자리 {len(sites)}개).")

    per_site, conts = [], []
    for s, o in zip(sites, outs):
        if len(o) != k:
            raise RuntimeError(f"site {s['site_id']}: K={k} 인데 {len(o)}개.")
        rows = []
        for text, trunc in o:
            c = classify_continuation(text)
            c["r_corr"] = grade_math(s["prefix"] + text, s["gold"])
            c["truncated"] = int(trunc)
            c["match"] = (int(c["decision"] == s["best_decision"])
                          if (c["emitted"] and s.get("best_decision") in ("verify", "redirect")) else None)
            rows.append(c)
            conts.append({"site_id": s["site_id"], "label": s["label"], "cont": text, **c})
        n = len(rows)
        judged = [r for r in rows if r["match"] is not None]
        decided = [r for r in judged if r["decision"] is not None]
        per_site.append({
            "site_id": s["site_id"], "label": s["label"], "best_decision": s.get("best_decision"),
            "p_nometa_label": s.get("p_nometa"), "k": n,
            "emit_at_site": sum(r["at_site"] for r in rows) / n,
            "emit_anywhere": sum(r["emitted"] for r in rows) / n,
            "acc": sum(r["r_corr"] for r in rows) / n,
            "truncated": sum(r["truncated"] for r in rows) / n,
            "n_verify": sum(1 for r in rows if r["decision"] == "verify"),
            "n_redirect": sum(1 for r in rows if r["decision"] == "redirect"),
            "judgment_match": _rate(sum(r["match"] for r in judged), len(judged)),
            "judgment_match_decided": _rate(sum(r["match"] for r in decided), len(decided)),
        })

    summ: dict = {"variant": variant, "k": k, "n_sites": len(sites), "at_site_chars": AT_SITE_CHARS,
                  "n_by_label": {}, "emit_at_site_by_label": {}, "emit_anywhere_by_label": {},
                  "acc_by_label": {}}
    for lab in LABELS:
        xs = [r for r in conts if r["label"] == lab]
        summ["n_by_label"][lab] = sum(1 for p in per_site if p["label"] == lab)
        summ["emit_at_site_by_label"][lab] = _rate(sum(r["at_site"] for r in xs), len(xs))
        summ["emit_anywhere_by_label"][lab] = _rate(sum(r["emitted"] for r in xs), len(xs))
        summ["acc_by_label"][lab] = _rate(sum(r["r_corr"] for r in xs), len(xs))
    judged = [r for r in conts if r["match"] is not None]
    decided = [r for r in judged if r["decision"] is not None]
    summ.update({
        "overall_acc": _rate(sum(r["r_corr"] for r in conts), len(conts)),
        "emit_at_site": _rate(sum(r["at_site"] for r in conts), len(conts)),
        "emit_anywhere": _rate(sum(r["emitted"] for r in conts), len(conts)),
        "truncated_rate": _rate(sum(r["truncated"] for r in conts), len(conts)),
        # 판단 일치율: 발화한 이어쓰기 중(verify/redirect 자리만) decision == best_decision.
        # ★decision 줄이 없는 발화는 «불일치»로 센다(주 지표) — 결정을 안 쓰면 판단이 없는 것.
        #   _decided 는 결정을 쓴 것만 분모로 둔 보조 지표(수학 메타는 대개 결정이 없어 분모 경고).
        "n_judged_conts": len(judged), "n_decided_conts": len(decided),
        "judgment_match_rate": _rate(sum(r["match"] for r in judged), len(judged)),
        "judgment_match_rate_decided": _rate(sum(r["match"] for r in decided), len(decided)),
    })
    return {"summary": summ, "per_site": per_site, "continuations": conts}


def markdown_table(summ: dict) -> str:
    def f(x):
        return "-" if x is None else f"{x:.3f}"
    lines = ["| label | n_sites | emit@site | emit_any | acc |", "|---|---|---|---|---|"]
    for lab in LABELS:
        lines.append(f"| {lab} | {summ['n_by_label'][lab]} | {f(summ['emit_at_site_by_label'][lab])} "
                     f"| {f(summ['emit_anywhere_by_label'][lab])} | {f(summ['acc_by_label'][lab])} |")
    lines.append(f"| ALL | {summ['n_sites']} | {f(summ['emit_at_site'])} | {f(summ['emit_anywhere'])} "
                 f"| {f(summ['overall_acc'])} |")
    lines.append(f"\njudgment_match {f(summ['judgment_match_rate'])} (n={summ['n_judged_conts']}) · "
                 f"decided-only {f(summ['judgment_match_rate_decided'])} (n={summ['n_decided_conts']}) · "
                 f"truncated {f(summ['truncated_rate'])}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, help="math_sites.py 의 sites.jsonl (라벨된 held-out 자리)")
    ap.add_argument("--model_path", required=True, help="정책 ckpt 또는 base")
    ap.add_argument("--variant", default="math_opt", help="허용판만. 강제판은 거부한다.")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_sites", type=int, default=0, help="0=전부")
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    check_variant(a.variant)          # ★vllm 을 올리기 전에 거부한다.

    sites = [json.loads(l) for l in open(a.sites)]
    if a.max_sites:
        sites = sites[:a.max_sites]
    if not sites:
        raise SystemExit("자리가 0개.")
    if any("label" not in s or "prefix" not in s for s in sites):
        raise SystemExit("sites.jsonl 에 label/prefix 가 없다 — math_sites.py 출력이 맞나?")
    print(f"[emit] 자리 {len(sites)}개 x K={a.k}, variant={a.variant}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 4096, enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)

    def generate(prompts):
        outs = llm.generate(list(prompts), sp)
        return [[(x.text, int(x.finish_reason == "length")) for x in o.outputs] for o in outs]

    res = run_eval(sites, generate, a.variant, a.k, tok=tok)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["summary"]["model_path"] = a.model_path
    res["summary"]["sites"] = a.sites
    (out / "summary.json").write_text(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    with (out / "per_site.jsonl").open("w") as fh:
        for r in res["per_site"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "continuations.jsonl").open("w") as fh:
        for r in res["continuations"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(markdown_table(res["summary"]))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
