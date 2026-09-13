#!/usr/bin/env python
"""math_sites — 수학판 «같은 자리» 인과 측정: 메타 하나의 기여 Δ̂ 를 재고 라벨한다.

배경(0912). Countdown 에서 «좋은 메타»의 정답 라벨은 오라클(지목한 수가 해를
살리는가)이었다. 수학엔 그 오라클이 없으므로 **행동 결과 라벨**을 쓴다 — 이 메타를
끼워 넣으면 정답률이 실제로 오르는가.

    Δ̂ = p̂(정답 | 앞부분 + 메타) − p̂(정답 | 앞부분)

세 조건을 같은 자리에서 각각 K 개 이어 쓴다:
    nometa — 앞부분 그대로 이어 씀(기준선)
    meta   — 앞부분 + "\\n<meta>\\n" 를 심어 **그 자리에서** 메타를 쓰게 만든다
    donor  — 앞부분 + **다른 문제**에서 뽑은 완성된 메타 블록
★donor 가 핵심 대조다. 이게 없으면 «메타 텍스트가 끼어든 효과»를 «이 메타의 내용
효과»로 오독한다(Countdown gen_continuations 의 donor 모드와 같은 역할).

자리 선별: `p̂(nometa)` 가 0 이나 1 인 자리는 Δ̂ 가 구조적으로 0 이라 버린다
(math500 실측: 문제의 52% 가 8/8 정답, 24.8% 가 8/8 오답 — 위치 단위로 잘라야
움직일 여지가 있는 자리가 나온다).

라벨:
    SAVE   p̂(nometa) <= 0.25  ∧  Δ̂ >= +0.25
    DERAIL p̂(nometa) >= 0.75  ∧  Δ̂ <= -0.25
    NEUTRAL 나머지
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS  # noqa: E402

_META_BLOCK = re.compile(r"<meta>.*?</meta>", re.S)


def _grade(text: str, gold: str) -> int:
    from math_verify import parse, verify
    try:
        return int(verify(parse(str(gold)), parse(text)))
    except Exception:
        return 0


def _selftest() -> None:
    if [_grade("\\boxed{42}", "42"), _grade("\\boxed{7}", "42")] != [1, 0]:
        raise RuntimeError("math_verify 자가검사 실패 — 조용한 오채점 방지를 위해 즉사한다.")


def cut_points(text: str, n_cuts: int, rng: random.Random) -> list[int]:
    """줄 경계에서 자리 후보를 뽑는다(수학엔 Countdown 의 «시도 경계»가 없다).

    응답 앞 10%·뒤 20% 는 버린다 — 너무 이르면 아무 진전이 없고, 너무 늦으면
    결과가 이미 정해져 Δ̂ 가 구조적으로 0 이 된다.
    """
    bounds = [m.end() for m in re.finditer(r"\n", text)]
    lo, hi = int(len(text) * 0.10), int(len(text) * 0.80)
    cand = [b for b in bounds if lo <= b <= hi]
    if not cand:
        return []
    rng.shuffle(cand)
    return sorted(cand[:n_cuts])


def build_fed(mode: str, prefix: str, donor_meta: str | None) -> str:
    """모델에게 «이미 이렇게 썼다»고 먹일 텍스트."""
    if mode == "nometa":
        return prefix
    if mode == "meta":
        return prefix.rstrip() + "\n<meta>\n"
    if mode == "donor":
        if not donor_meta:
            raise ValueError("donor 모드인데 기증 메타가 비었다.")
        return prefix.rstrip() + "\n" + donor_meta + "\n"
    raise ValueError(f"unknown mode: {mode!r}")


def _gen_prompt(tok, msgs) -> str:
    """math_rollout.chat 과 같은 생성 프롬프트 — 롤아웃이 만들어진 문맥과 바이트 단위로 같아야
    «같은 자리» 측정이 된다."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_plain 롤아웃 texts.jsonl")
    ap.add_argument("--donor_rollouts", default=None,
                    help="기증 메타를 캘 롤아웃(math_new/opt). 없으면 donor 모드 생략")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cuts_per_rollout", type=int, default=2)
    ap.add_argument("--max_sites", type=int, default=400)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    _selftest()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    # 한 문제당 롤아웃 하나만 자리 원천으로 쓴다(같은 문제가 자리를 독점하지 않게).
    seen, srcs = set(), []
    for r in rolls:
        if r["group_id"] in seen or r.get("truncated"):
            continue
        seen.add(r["group_id"])
        srcs.append(r)
    rng.shuffle(srcs)

    sites = []
    for r in srcs:
        for c in cut_points(r["text"], a.cuts_per_rollout, rng):
            sites.append({"site_id": f"{r['group_id']}@{c}", "problem": r["problem"],
                          "gold": r["gold"], "prefix": r["text"][:c]})
            if len(sites) >= a.max_sites:
                break
        if len(sites) >= a.max_sites:
            break
    print(f"[sites] 자리 {len(sites)}개 (문제 {len(seen)}개에서)", flush=True)

    donors: list[str] = []
    if a.donor_rollouts:
        for l in open(a.donor_rollouts):
            m = _META_BLOCK.search(json.loads(l)["text"])
            if m:
                donors.append(m.group(0))
        print(f"[sites] 기증 메타 {len(donors)}개", flush=True)
    modes = ["nometa", "meta"] + (["donor"] if donors else [])

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 4096, enforce_eager=True)
    tok = llm.get_tokenizer()

    reqs, meta_ix = [], []
    for si, s in enumerate(sites):
        # ★기증 메타는 **다른 문제**에서만 — 같은 문제면 내용 누출이다.
        dm = None
        if donors:
            for _ in range(10):
                cand = donors[rng.randrange(len(donors))]
                if cand not in s["prefix"]:
                    dm = cand
                    break
        s["donor_meta"] = dm
        for mode in modes:
            if mode == "donor" and not dm:
                continue
            # meta 모드만 강제 프롬프트를 쓴다(그 자리에서 메타를 쓰게).
            variant = "math_new" if mode == "meta" else "math_plain"
            msgs = [{"role": "system", "content": MATH_PROMPT_VARIANTS[variant]},
                    {"role": "user", "content": s["problem"]},
                    {"role": "assistant", "content": build_fed(mode, s["prefix"], dm)}]
            # ★0913 수리: continue_final_message 를 쓰지 않는다. Qwen3.5 템플릿은 assistant
            #   본문의 끝 공백을 지운다 — 자리는 줄 경계라 앞부분이 늘 개행으로 끝나므로
            #   (a) vLLM 토크나이저 래퍼는 «마지막 메시지가 안 보인다»며 거부해 3회 rc 1,
            #   (b) transformers 는 통과시키되 개행을 **조용히 삭제**해 자리가 한 줄 옮겨진다
            #   (실측: 렌더 522 vs 523 바이트, 차이는 끝 '\n' 하나). 롤아웃이 생성된 문맥은
            #   «생성 프롬프트(enable_thinking=False) + 텍스트» 그 자체이므로 그걸 그대로 잇는다.
            reqs.append(_gen_prompt(tok, msgs[:2]) + msgs[2]["content"])
            meta_ix.append((si, mode))

    print(f"[sites] 요청 {len(reqs)}개 x K={a.k}", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict[tuple, list] = {}
    rows = []
    for (si, mode), o in zip(meta_ix, outs):
        s = sites[si]
        fed = build_fed(mode, s["prefix"], s.get("donor_meta"))
        for x in o.outputs:
            full = fed + x.text
            c = _grade(full, s["gold"])
            agg.setdefault((si, mode), []).append(c)
            rows.append({"site_id": s["site_id"], "mode": mode, "r_corr": c,
                         "cont": x.text, "truncated": int(x.finish_reason == "length")})

    recs = []
    for si, s in enumerate(sites):
        p = {m: (sum(v) / len(v)) for (i, m), v in agg.items() if i == si}
        if "nometa" not in p or "meta" not in p:
            continue
        d = p["meta"] - p["nometa"]
        dd = (p.get("donor") - p["nometa"]) if "donor" in p else None
        if p["nometa"] <= 0.25 and d >= 0.25:
            lab = "SAVE"
        elif p["nometa"] >= 0.75 and d <= -0.25:
            lab = "DERAIL"
        else:
            lab = "NEUTRAL"
        recs.append({"site_id": s["site_id"], "problem": s["problem"], "gold": s["gold"],
                     "prefix": s["prefix"], "donor_meta": s.get("donor_meta"),
                     "p_nometa": p["nometa"], "p_meta": p["meta"],
                     "p_donor": p.get("donor"), "delta": d, "delta_donor": dd,
                     "label": lab,
                     "movable": int(0.0 < p["nometa"] < 1.0)})

    with (out / "sites.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "continuations.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    lab = {k: sum(1 for r in recs if r["label"] == k) for k in ("SAVE", "DERAIL", "NEUTRAL")}
    summ = {
        "n_sites": n, "k": a.k, "modes": modes,
        "movable_rate": sum(r["movable"] for r in recs) / max(1, n),
        "labels": lab,
        "save_rate": lab["SAVE"] / max(1, n), "derail_rate": lab["DERAIL"] / max(1, n),
        "mean_delta": sum(r["delta"] for r in recs) / max(1, n),
        "mean_delta_donor": (sum(r["delta_donor"] for r in recs if r["delta_donor"] is not None)
                             / max(1, sum(1 for r in recs if r["delta_donor"] is not None))),
        "mean_p_nometa": sum(r["p_nometa"] for r in recs) / max(1, n),
    }
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    for k, v in summ.items():
        print(f"  {k:18s} {v}")
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
