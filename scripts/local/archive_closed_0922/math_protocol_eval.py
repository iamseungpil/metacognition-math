#!/usr/bin/env python
r"""math_protocol_eval — T0 «규칙 기반 추론 프로토콜 기준선»(훈련 없음).

물음: **훈련 없이**, 합의 상태로 게이트한 «문맥 폐기 재시도 + 답 배제»가 **같은 총
토큰 예산**에서 정확도를 올리는가 — 그리고 심사자가 요구할 대조들(표본을 더 뽑기,
더 길게 생각하기, 내용 없는 패드, 사실 고지만)보다 나은가.

시도 1 은 **이미 있는 롤아웃**을 재사용한다(재생성 금지):
  `/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl`
  (난이도-5 800문제 × K=8; group_id/problem_id/problem/gold/text/r_corr/final_answer/
   truncated/n_tok)

프로토콜(각각 800문제 정확도 + 총 생성 토큰 = 시도-1 토큰 + 추가 토큰):
  maj8                  기존 8개 답의 다수결(동치 군집, 동률 → 먼저 나온 것)
  maj16                 평범한 표본 8개를 **더** 뽑아 16개로 다수결 (= «예산을 표본에»)
  gated_notx            상태 ∈ GATE_STATES 이면 행마다 리셋 재시도(fact + notx, X = 그 행의
                        자기 답) → vote16(재시도 8 ∪ 시도-1 8) 와 vote_a2(재시도만) 둘 다
  gated_habit_by_state  위와 같되 습관을 상태별로 고른다(`--habit_by_state`)
  always_notx           ALL_SAME 포함 **모든** 상태에 notx 재시도
  gated_fact            습관 줄 없이 fact 만(배제 절 제거)
  gated_pad             fact_pad — 같은 프롬프트 길이에서 내용만 뺀 대조
  gated_blind           fact 줄조차 없는 리셋(= F1 `blind`) — «틀렸다는 사실»의 대조
  long8                 16384 예산으로 새 표본 8개(= SEVRA 식 «그냥 더 길게») — `--include_long8`

★프롬프트는 **F1 팔과 바이트 동일**해야 비교가 성립한다. 그래서 문자열을 복사하지 않고
  `src.training.trial2`(fact_prompt/attempt2_prompt/NOTX_TMPL/SWITCH_TMPL) 와
  `math_activation_gate`(pad_prompt/blind_prompt/label_prompt/clean_label) 를 **그대로
  import** 한다(`tests/test_math_protocol_eval.py` 가 문자열 동일을 고정한다).

★재개 가능: (protocol, problem_id, roll_id, mode) 가 이미 `gens.jsonl` 에 있으면 다시
  생성하지 않는다.

사용(예):
  python scripts/local/math_protocol_eval.py --out_name proto_t0_s1 \
      --protocols maj8,maj16,gated_notx,gated_fact,gated_pad --seed 11 --gpu_util 0.45
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TMPDIR", "/hdd_data/seungpil/tmp")
os.environ.setdefault("HF_HOME", "/hdd_data/seungpil/scratch/hf_home")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_activation_gate as A  # noqa: E402
from math_cited_site_gate import bootstrap_ci  # noqa: E402
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, grade_math, last_boxed,
)
from src.training.trial2 import attempt2_prompt, fact_prompt  # noqa: E402

ATTEMPT1_PATH = "/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl"
OUT_ROOT = "/hdd_data/seungpil/scratch/eval"
DEFAULT_MODEL = "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507"
VARIANT = "math_opt"
LONG_MAX_TOKENS = 16384
LABEL_MAX_TOKENS = 64

#: 재시도를 켜는 합의 상태(기본) — ALL_SAME 은 절대 켜지 않는다.
GATE_STATES = ("DOMINANT", "SPLIT", "SCATTER", "NOANS")
#: 상태별 습관(기본) — `--habit_by_state` 의 기본값과 같다.
DEFAULT_HABIT_BY_STATE = {"DOMINANT": "notx", "SPLIT": "notx",
                          "SCATTER": "switch", "NOANS": "fact"}
HABIT_MODES = ("fact", "notx", "switch", "pad")
#: 재시도 프로토콜 → (게이트를 쓰는가, 고정 습관(None 이면 상태별))
RETRY_PROTOCOLS = {
    "gated_notx": (True, "notx"),
    "gated_habit_by_state": (True, None),
    "always_notx": (False, "notx"),
    "gated_fact": (True, "fact"),
    "gated_pad": (True, "pad"),
    "gated_blind": (True, "blind"),
}
#: 새 표본 프로토콜 → (모드, 개수, 예산(None = --retry_max_tokens))
SAMPLE_PROTOCOLS = {"maj16": ("plain", 8, None), "long8": ("plain", 8, LONG_MAX_TOKENS)}
#: 시도-1 롤아웃을 **쓰지 않는** 프로토콜 — 토큰 회계에서 시도-1 토큰을 세지 않는다.
#: long8 은 16k 새 표본 8개가 시도 1 을 **대체**한다(상태도 X 도 쓰지 않는다).
NO_A1_PROTOCOLS = ("long8",)
ALL_PROTOCOLS = ("maj8", "maj16", *RETRY_PROTOCOLS, "long8")
DEFAULT_PROTOCOLS = tuple(p for p in ALL_PROTOCOLS if p != "long8")
BASE = "maj8"
A2_SUFFIX = "__a2"


# ── 시도-1 읽기 ────────────────────────────────────────────────────────────────
def load_attempt1(path: str, limit: int = 0) -> list[dict]:
    """texts.jsonl → 문제별 레코드 [{problem_id, problem, gold, rows, state, ...}].

    `rows` 는 그 문제의 시도-1 롤아웃들 [{roll_id, answer, r_corr, n_tok, text, truncated}]
    이고 `roll_id` 는 그 문제 **안에서의 순번**(0…K−1)이다. 상태는
    `math_activation_gate.agreement_state`(gold 를 보지 않는다)로 붙인다."""
    groups: dict = {}
    order: list = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gid = r.get("group_id", r.get("problem_id"))
            if gid not in groups:
                groups[gid] = {"problem_id": gid, "problem": r["problem"], "gold": r["gold"],
                               "rows": []}
                order.append(gid)
            ans = r.get("final_answer")
            if ans is None:
                ans = last_boxed(r.get("text", ""))
            groups[gid]["rows"].append({
                "roll_id": len(groups[gid]["rows"]), "answer": (ans or "").strip(),
                "r_corr": int(r.get("r_corr", 0)), "n_tok": int(r.get("n_tok", 0) or 0),
                "text": r.get("text", ""), "truncated": int(r.get("truncated", 0) or 0)})
    probs = [groups[g] for g in order]
    if limit and limit > 0:
        probs = probs[:limit]
    for p in probs:
        st = A.agreement_state([x["answer"] for x in p["rows"]], k=len(p["rows"]))
        p["state"] = st["state"]
        p["a1_tokens"] = sum(x["n_tok"] for x in p["rows"])
    return probs


# ── 투표 ───────────────────────────────────────────────────────────────────────
def vote(pairs: Sequence[tuple]) -> tuple:
    """(답, 정오) 쌍들의 다수결. 군집은 `answers_equivalent`, **동률이면 먼저 나온 군집**.
    빈 답은 세지 않는다. 군집의 정오는 그 군집 구성원의 정오(동치 답이므로 같아야 한다 —
    채점기가 엇갈리면 하나라도 1 이면 1 로 본다). 후보가 없으면 ("", 0)."""
    clusters: list[list] = []          # [대표, 개수, 첫 등장, 정오]
    for i, (a, c) in enumerate(pairs):
        s = str(a or "").strip()
        if not s:
            continue
        for cl in clusters:
            if answers_equivalent(cl[0], s):
                cl[1] += 1
                cl[3] = max(cl[3], int(c))
                break
        else:
            clusters.append([s, 1, i, int(c)])
    if not clusters:
        return "", 0
    best = min(clusters, key=lambda cl: (-cl[1], cl[2]))
    return best[0], int(best[3])


def resolve_states(spec: str | None) -> list[str]:
    """`--k_states` 파싱 → 재시도를 켜는 상태 목록. ALL_SAME 은 넣어도 **버린다**
    (게이트의 정의: 전원 합의면 재시도하지 않는다)."""
    want = [x.strip().upper() for x in (spec or "").split(",") if x.strip()] or list(GATE_STATES)
    bad = [x for x in want if x not in A.AGREE_STATES]
    if bad:
        raise SystemExit(f"[proto] 모르는 합의 상태 {bad} — 가능: {', '.join(A.AGREE_STATES)}")
    return [s for s in A.AGREE_STATES if s in set(want) and s != "ALL_SAME"]


def resolve_protocols(spec: str | None, include_long8: bool = False) -> list[str]:
    want = [x.strip() for x in (spec or "").split(",") if x.strip()] or list(DEFAULT_PROTOCOLS)
    bad = [x for x in want if x not in ALL_PROTOCOLS]
    if bad:
        raise SystemExit(f"[proto] 모르는 프로토콜 {bad} — 가능: {', '.join(ALL_PROTOCOLS)}")
    out = [p for p in ALL_PROTOCOLS if p in set(want)]
    if include_long8 and "long8" not in out:
        out.append("long8")
    if "long8" in out and not include_long8:
        out = [p for p in out if p != "long8"]
    if BASE not in out:
        out.insert(0, BASE)
    return out


def resolve_habit_by_state(spec: str | None) -> dict:
    d = dict(DEFAULT_HABIT_BY_STATE) if not spec else json.loads(spec)
    bad_s = [s for s in d if s not in A.AGREE_STATES]
    bad_m = [m for m in d.values() if m not in HABIT_MODES]
    if bad_s or bad_m:
        raise SystemExit(f"[proto] --habit_by_state 가 이상하다: 상태 {bad_s} / 모드 {bad_m}")
    return d


def mode_for(protocol: str, state: str, gate: Sequence[str], habit_by_state: dict) -> str | None:
    """이 프로토콜이 이 상태의 문제에 쓸 재시도 모드(None 이면 재시도하지 않는다)."""
    gated, fixed = RETRY_PROTOCOLS[protocol]
    if gated and state not in gate:
        return None
    if fixed is not None:
        return fixed
    return habit_by_state.get(state)


# ── 프롬프트 ───────────────────────────────────────────────────────────────────
def build_prompt(tok, problem: str, mode: str, *, notx: str = "", label: str = "",
                 text: str = "") -> str:
    """모드 → 프롬프트. **F1 팔과 바이트 동일**(문자열을 여기서 만들지 않는다):
      plain/blind → `math_activation_gate.blind_prompt`(= 시도-1 프롬프트)
      fact        → `trial2.fact_prompt(..., "")`   (F1 `fact` = blind_external)
      notx        → `trial2.attempt2_prompt(mode="notx")`  (F1 `fact_notx`)
      switch      → `trial2.attempt2_prompt(mode="switch")`(F1 `fact_switch`)
      pad         → `math_activation_gate.pad_prompt`       (F1 `fact_pad`)
      label       → `math_activation_gate.label_prompt`(switch 의 사전 패스)"""
    if mode in ("plain", "blind"):
        return A.blind_prompt(tok, VARIANT, problem)
    if mode == "fact":
        return fact_prompt(tok, VARIANT, problem, "")
    if mode == "notx":
        return attempt2_prompt(tok, VARIANT, problem, mode="notx", notx=notx)
    if mode == "switch":
        return attempt2_prompt(tok, VARIANT, problem, mode="switch", label=label)
    if mode == "pad":
        return A.pad_prompt(tok, VARIANT, problem)
    if mode == "label":
        return A.label_prompt(tok, VARIANT, problem, text)
    raise ValueError(f"[proto] 모르는 모드 {mode!r}")


def plan_requests(problems: Sequence[dict], protocols: Sequence[str], *,
                  gate: Sequence[str], habit_by_state: dict, n_extra: int = 8) -> list[dict]:
    """생성해야 할 행 목록(프롬프트 조립 전). 각 항목은
    {protocol, problem_id, roll_id, mode, state, notx, budget_kind}.
    `budget_kind` ∈ {retry, long} — 호출자가 예산별로 묶어 돌린다.
    switch 모드 행은 사전 패스(label)가 먼저 필요하다(`needs_label`)."""
    out: list[dict] = []
    for p in problems:
        for proto in protocols:
            if proto in SAMPLE_PROTOCOLS:
                mode, n, budget = SAMPLE_PROTOCOLS[proto]
                for j in range(n if proto != "maj16" else n_extra):
                    out.append({"protocol": proto, "problem_id": p["problem_id"], "roll_id": j,
                                "mode": mode, "state": p["state"], "notx": "",
                                "needs_label": False, "degraded": False,
                                "budget_kind": "long" if budget else "retry"})
            elif proto in RETRY_PROTOCOLS:
                mode = mode_for(proto, p["state"], gate, habit_by_state)
                if mode is None:
                    continue
                for r in p["rows"]:
                    x = r["answer"] if mode == "notx" else ""
                    out.append({"protocol": proto, "problem_id": p["problem_id"],
                                "roll_id": r["roll_id"], "mode": mode, "state": p["state"],
                                "notx": x, "needs_label": mode == "switch",
                                # ★notx 인데 자기 답이 없으면(`\boxed` 없음·절단) 배제 절이
                                #   빠져 프롬프트가 `fact` 와 **바이트 동일**해진다. 조용히
                                #   넘어가면 «배제의 효과»가 그만큼 희석된 채 보고되므로
                                #   행에 표식을 남기고 요약에 개수를 낸다.
                                "degraded": bool(mode == "notx" and not x),
                                "budget_kind": "retry"})
    return out


# ── 채점·집계 ──────────────────────────────────────────────────────────────────
def key_of(rec: dict) -> tuple:
    return (rec["protocol"], rec["problem_id"], int(rec["roll_id"]), rec["mode"])


def label_key(rec: dict) -> tuple:
    """switch 사전 패스 라벨의 키 — **프로토콜까지** 포함한다. 캐시 키(`key_of`)가
    프로토콜을 포함하므로 라벨 사전도 같은 폭이어야 한다: 두 프로토콜이 같은 행에서
    switch 로 풀리면(예: `gated_habit_by_state` 와 훗날의 다른 습관 표) 한쪽의 라벨이
    다른 쪽에 조용히 새어 «각자 자기 캐시를 쓴다»는 재개 규약이 깨진다."""
    return (rec["protocol"], rec["problem_id"], int(rec["roll_id"]))


def row_seed(rec: dict, seed: int) -> int:
    """행마다 **다른·결정적인** 샘플링 시드.

    ⚠️왜 필요한가: vLLM 의 `SamplingParams.seed` 는 **요청 단위**라 같은 프롬프트 + 같은
    시드는 **바이트 동일한 출력**을 낸다. maj16 의 8개 새 표본(같은 프롬프트)·long8 의
    8개·그리고 프롬프트가 행에 의존하지 않는 재시도(fact/pad/blind, 또는 X 가 같은 notx)는
    전부 한 표본의 복제가 되어 «다수결»이 표본 1개의 투표가 된다.
    `zlib.crc32`(파이썬 `hash()` 와 달리 프로세스 간 **소금이 없다**)로 키를 섞어 재개
    실행에서도 같은 시드가 나오게 한다."""
    k = f"{rec['protocol']}|{rec['problem_id']}|{int(rec['roll_id'])}|{rec['mode']}"
    return (int(seed) * 100003 + zlib.crc32(k.encode())) % (2 ** 31 - 1)


def chunk_spans(n: int, size: int) -> list[tuple]:
    """[0, n) 를 `size` 크기 덩어리로 자른 (lo, hi) 목록. size ≤ 0 이면 1 로 본다."""
    size = max(1, int(size))
    return [(i, min(i + size, int(n))) for i in range(0, int(n), size)]


def generate_chunked(llm, prompts: Sequence[str], sps: Sequence, chunk: int, handle) -> None:
    """`llm.generate` 를 덩어리로 나눠 부르고 덩어리마다 `handle(lo, outs)` 를 부른다.

    ★왜: 한 판에 수만 프롬프트를 넣으면 잡이 죽을 때 그 판 전체가 사라진다. 덩어리마다
    방출·flush 하면 다음 실행이 `gens.jsonl` 캐시에서 이어간다. `sps` 는 **행마다 미리**
    만들어 넘기므로(행 시드는 `row_seed`) 덩어리를 나눠도 결과는 한 판과 동일하다."""
    for lo, hi in chunk_spans(len(prompts), chunk):
        outs = llm.generate(list(prompts[lo:hi]), list(sps[lo:hi]))
        handle(lo, outs)


def load_cache(path: Path) -> dict:
    """gens.jsonl → {(protocol, problem_id, roll_id, mode): row}. 없으면 빈 dict."""
    cache: dict = {}
    if not path.exists():
        return cache
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cache[key_of(r)] = r
    return cache


def score_protocol(problems: Sequence[dict], cache: dict, protocol: str, *,
                   gate: Sequence[str], habit_by_state: dict,
                   n_extra: int = 8) -> list[dict]:
    """문제별 레코드 [{problem_id, state, correct, correct_a2, extra_tokens, retried}].
    `correct_a2` 는 재시도만으로 투표한 결과(재시도가 없으면 `correct` 와 같다)."""
    by_pid: dict = {}
    for k, row in cache.items():
        if k[0] == protocol:
            by_pid.setdefault(k[1], []).append(row)
    recs: list[dict] = []
    for p in problems:
        a1 = [(r["answer"], r["r_corr"]) for r in p["rows"]]
        rows = by_pid.get(p["problem_id"], [])
        got = sorted((r for r in rows if r["mode"] != "label"),
                     key=lambda r: int(r["roll_id"]))
        extra = [(r.get("final_answer", ""), int(r.get("r_corr", 0))) for r in got]
        # ★토큰 회계: 이 프로토콜이 **추가로** 쓴 토큰(사전 패스 라벨 포함) 전부
        toks = sum(int(r.get("n_tok", 0) or 0) for r in rows)
        if protocol == BASE:
            pool, pool_a2 = a1, a1
        elif protocol in SAMPLE_PROTOCOLS:
            pool = (a1 + extra) if protocol == "maj16" else extra
            pool_a2 = pool
        else:
            pool = a1 + extra
            pool_a2 = extra or a1
        ans, corr = vote(pool)
        ans2, corr2 = vote(pool_a2)
        recs.append({"problem_id": p["problem_id"], "state": p["state"],
                     "correct": corr, "answer": ans, "correct_a2": corr2, "answer_a2": ans2,
                     "extra_tokens": toks,
                     # ★long8 은 시도 1 을 **대체**하므로 그 예산을 총계에 넣지 않는다.
                     "a1_tokens": 0 if protocol in NO_A1_PROTOCOLS else p["a1_tokens"],
                     "n_extra": len(got), "retried": int(bool(got) and protocol
                                                         in RETRY_PROTOCOLS),
                     # ★notx 가 fact 로 퇴화한 행(자기 답이 없어 배제 절이 빠진 행)
                     "n_degraded": sum(int(r.get("degraded", 0)) for r in got)})
    return recs


def summarize(recs_by_proto: dict, protocols: Sequence[str], *, seed: int = 0,
              n_boot: int = 2000) -> dict:
    """프로토콜 × {acc, Δ vs maj8 (부트스트랩 CI), 토큰, flip} + 상태별 정확도."""
    base = recs_by_proto[BASE]
    base_by_id = {r["problem_id"]: r["correct"] for r in base}
    rows: dict = {}
    states: dict = {}
    names: list[str] = []
    for proto in protocols:
        recs = recs_by_proto[proto]
        variants = [(proto, "correct")]
        if proto in RETRY_PROTOCOLS:
            variants.append((proto + A2_SUFFIX, "correct_a2"))
        for i, (name, key) in enumerate(variants):
            names.append(name)
            acc = [r[key] for r in recs]
            diffs = [r[key] - base_by_id[r["problem_id"]] for r in recs]
            tot = sum(r["a1_tokens"] + r["extra_tokens"] for r in recs)
            rows[name] = {
                "acc": (sum(acc) / len(acc)) if acc else float("nan"),
                "n_problems": len(acc),
                "delta_vs_maj8": bootstrap_ci(diffs, seed=seed + 7 * len(names) + i,
                                              n_boot=n_boot),
                "total_tokens": tot,
                "tokens_per_problem": (tot / len(recs)) if recs else float("nan"),
                "extra_tokens": sum(r["extra_tokens"] for r in recs),
                "n_retried_problems": sum(r["retried"] for r in recs),
                "n_degraded_rows": sum(r.get("n_degraded", 0) for r in recs),
                "n_degraded_problems": sum(1 for r in recs if r.get("n_degraded", 0)),
                "flip_wrong": sum(1 for r in recs
                                  if base_by_id[r["problem_id"]] and not r[key]),
                "flip_right": sum(1 for r in recs
                                  if not base_by_id[r["problem_id"]] and r[key]),
            }
            states[name] = {}
            for st in A.AGREE_STATES:
                sub = [r[key] for r in recs if r["state"] == st]
                states[name][st] = {"acc": (sum(sub) / len(sub)) if sub else float("nan"),
                                    "n": len(sub)}
    return {"protocols": names, "table": rows, "by_state": states}


def to_markdown(summ: dict) -> str:
    def f(x, nd=4):
        try:
            return "nan" if x != x else f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return "nan"

    L = ["# math_protocol_eval — T0 규칙 기반 추론 프로토콜", "",
         "| protocol | acc | Δ vs maj8 [95% CI] | total tok | tok/prob | n_retried | "
         "flip_wrong | flip_right | n_degraded (rows/probs) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in summ["protocols"]:
        r = summ["table"][name]
        d = r["delta_vs_maj8"]
        L.append(f"| {name} | {f(r['acc'])} | {f(d['mean'])} [{f(d['lo'])}, {f(d['hi'])}] | "
                 f"{r['total_tokens']} | {f(r['tokens_per_problem'], 1)} | "
                 f"{r['n_retried_problems']} | {r['flip_wrong']} | {r['flip_right']} | "
                 f"{r['n_degraded_rows']}/{r['n_degraded_problems']} |")
    L += ["", "## 상태별 정확도", "",
          "| protocol | " + " | ".join(A.AGREE_STATES) + " |",
          "|---|" + "---|" * len(A.AGREE_STATES)]
    for name in summ["protocols"]:
        cells = [f"{f(summ['by_state'][name][s]['acc'])} (n={summ['by_state'][name][s]['n']})"
                 for s in A.AGREE_STATES]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--protocols", default="")
    ap.add_argument("--k_states", default="", help=f"재시도를 켜는 상태(기본 {','.join(GATE_STATES)})")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--habit_by_state", default="")
    ap.add_argument("--retry_max_tokens", type=int, default=8192)
    ap.add_argument("--include_long8", action="store_true")
    ap.add_argument("--attempt1", default=ATTEMPT1_PATH,
                    help="시도-1 texts.jsonl (기본 = 평가 풀; 훈련 풀을 주면 그 위에서 돈다)")
    ap.add_argument("--gen_chunk", type=int, default=2048,
                    help="llm.generate 한 판의 프롬프트 수 — 덩어리마다 방출·flush 한다")
    a = ap.parse_args()

    gate = resolve_states(a.k_states)
    protocols = resolve_protocols(a.protocols, a.include_long8)
    habit = resolve_habit_by_state(a.habit_by_state)
    problems = load_attempt1(a.attempt1, a.limit_problems)
    out = Path(OUT_ROOT) / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    gens_path = out / "gens.jsonl"
    cache = load_cache(gens_path)
    print(f"[proto] 문제 {len(problems)}개 · 프로토콜 {protocols} · 게이트 {gate} · "
          f"캐시 {len(cache)}행", flush=True)

    reqs = [r for r in plan_requests(problems, protocols, gate=gate, habit_by_state=habit)
            if key_of(r) not in cache]
    print(f"[proto] 생성할 행 {len(reqs)}개", flush=True)

    if reqs:
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        max_tok = max(a.retry_max_tokens, LONG_MAX_TOKENS if "long8" in protocols else 0)
        llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
                  gpu_memory_utilization=a.gpu_util,
                  max_model_len=max_tok + 9216, enforce_eager=True)
        tok = llm.get_tokenizer()
        pmap = {p["problem_id"]: p for p in problems}
        fh = gens_path.open("a")

        def emit(rec: dict, text: str, gold: str, finish: str, ntok: int) -> None:
            row = {"protocol": rec["protocol"], "problem_id": rec["problem_id"],
                   "roll_id": int(rec["roll_id"]), "state": rec["state"], "mode": rec["mode"],
                   "text": text, "final_answer": last_boxed(text),
                   # ★라벨 행은 답이 아니다 — 채점하지 않고 -1 로 표시한다(투표에 안 들어간다).
                   "r_corr": -1 if rec["mode"] == "label" else grade_math(text, gold),
                   "truncated": int(finish == "length"),
                   "n_tok": int(ntok), "degraded": int(bool(rec.get("degraded")))}
            cache[key_of(row)] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

        # ── switch 사전 패스(방법 라벨) — 필요한 행만, 캐시 키 mode="label" ──────
        labels: dict = {}
        need = [r for r in reqs if r["needs_label"]]
        for r in need:
            k = (r["protocol"], r["problem_id"], int(r["roll_id"]), "label")
            if k in cache:
                labels[label_key(r)] = A.clean_label(
                    cache[k]["text"], pmap[r["problem_id"]]["rows"][r["roll_id"]]["answer"])[0]
        todo = [r for r in need if label_key(r) not in labels]
        if todo:
            prompts = [build_prompt(tok, pmap[r["problem_id"]]["problem"], "label",
                                    text=pmap[r["problem_id"]]["rows"][r["roll_id"]]["text"])
                       for r in todo]
            print(f"[proto] 라벨 사전 패스 {len(prompts)}개", flush=True)
            # ★프롬프트마다 SamplingParams 하나 — 시드가 행마다 다르다(row_seed 참조).
            sp = [SamplingParams(n=1, temperature=0.0, top_p=1.0,
                                 max_tokens=LABEL_MAX_TOKENS,
                                 seed=row_seed({**r, "mode": "label"}, a.seed)) for r in todo]
            def take_labels(lo: int, outs) -> None:
                for r, o in zip(todo[lo:lo + len(outs)], outs):
                    row = pmap[r["problem_id"]]["rows"][r["roll_id"]]
                    raw = o.outputs[0].text
                    emit({**r, "mode": "label"}, raw, pmap[r["problem_id"]]["gold"],
                         o.outputs[0].finish_reason,
                         len(getattr(o.outputs[0], "token_ids", []) or []))
                    labels[label_key(r)] = A.clean_label(raw, row["answer"])[0]

            generate_chunked(llm, prompts, sp, a.gen_chunk, take_labels)

        # ── 본 생성 — 예산별로 한 판씩 ─────────────────────────────────────────
        for kind, budget in (("retry", a.retry_max_tokens), ("long", LONG_MAX_TOKENS)):
            batch = [r for r in reqs if r["budget_kind"] == kind]
            if not batch:
                continue
            prompts = [build_prompt(tok, pmap[r["problem_id"]]["problem"], r["mode"],
                                    notx=r["notx"],
                                    label=labels.get(label_key(r), ""))
                       for r in batch]
            print(f"[proto] {kind} 생성 {len(prompts)}개 (max_tokens={budget})", flush=True)
            # ★행마다 다른 시드 — 같은 프롬프트(maj16/long8 의 8개, X 가 같은 재시도)가
            #   한 표본의 복제가 되지 않게 한다.
            sp = [SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=budget,
                                 seed=row_seed(r, a.seed)) for r in batch]
            def take_gens(lo: int, outs, batch=batch) -> None:
                for r, o in zip(batch[lo:lo + len(outs)], outs):
                    x = o.outputs[0]
                    emit(r, x.text, pmap[r["problem_id"]]["gold"], x.finish_reason,
                         len(getattr(x, "token_ids", []) or []))

            generate_chunked(llm, prompts, sp, a.gen_chunk, take_gens)
        fh.close()

    recs_by_proto = {p: score_protocol(problems, cache, p, gate=gate, habit_by_state=habit)
                     for p in protocols}
    summ = summarize(recs_by_proto, protocols, seed=a.seed)
    summ["meta"] = {"attempt1": a.attempt1, "model_path": a.model_path, "variant": VARIANT,
                    "gate_states": gate, "habit_by_state": habit, "seed": a.seed,
                    "retry_max_tokens": a.retry_max_tokens, "n_problems": len(problems)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(summ)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
