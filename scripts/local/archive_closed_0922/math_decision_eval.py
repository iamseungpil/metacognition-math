#!/usr/bin/env python
r"""math_decision_eval — 설계 C 의 **추론 프로토콜**(훈련된 모델 또는 기준선).

물음 둘:
  ① 훈련이 **문제 안에서 판별하는 발화**를 만들었나 — P(restart) 가 그 행의 시도-1
     오답을 예측하는 within-problem AUC ≥ .60 인가. (훈련 전에는 S1/B1/G1 에서 우연
     수준이었다 — 그게 이 실험의 출발점이다.)
  ② 그 발화로 **게이트한 재시도**가 같은 토큰에서 always-retry / never-retry 를 이기나.

절차
  (a) 문제마다 **표준 수학 프롬프트**로 한 패스 K개 — 발화를 요구하지 않는다. 모델이 스스로
      끝에 붙이면 그것이 발화다(붙이지 않으면 emission=0 으로 보고된다).
  (b) 재시도 = 문맥 리셋 + 배제(`trial2.attempt2_prompt(mode="notx")`, X = 발화가 말한 x,
      없으면 그 행의 `last_boxed`) — F1 `fact_notx` 와 바이트 동일.
  (c) 채점은 `grade_math` — **gold 는 여기(평가)에서만** 쓴다.
  (d) 보고: 발화율 · 상태별 restart 율 · AUC(문제 안/전체) · 네 프로토콜의 정확도와 토큰.

프로토콜(행 단위, 토큰은 그 프로토콜이 **실제로 쓴** 것만 센다)
  never        시도 1 만
  utter_gated  발화가 restart 인 행만 재시도로 대체 (= 이 실험의 주장)
  always       모든 행 재시도
  agree_gated  만장일치가 아닌 문제의 모든 행 재시도 (= T0 의 규칙 기반 게이트)

★`--retry_rows all`(기본)이면 재시도를 **전 행**에 굴린다 — always/agree_gated 가
  계산 가능하려면 필요하다. `restart` 로 두면 발화 행만 굴리고 그 두 프로토콜은 NaN 이다.

사용(예):
  python scripts/local/math_decision_eval.py --model_path <ckpt> \
      --dataset /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --out_name dec_eval_s1 --k 8 --seed 11 --gpu_util 0.45
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TMPDIR", "/hdd_data/seungpil/tmp")
os.environ.setdefault("HF_HOME", "/hdd_data/seungpil/scratch/hf_home")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_activation_gate as A  # noqa: E402
from math_protocol_eval import (  # noqa: E402
    DEFAULT_MODEL, OUT_ROOT, VARIANT, generate_chunked, row_seed, vote,
)
from src.training.decision import has_meta_block, parse_decision  # noqa: E402
from src.training.math_meta import grade_math, last_boxed  # noqa: E402
from src.training.trial2 import agreement_state, attempt2_prompt  # noqa: E402

PROTOCOLS = ("never", "utter_gated", "always", "agree_gated")
RETRY_ROWS = ("all", "restart")


# ── 문제 읽기 ─────────────────────────────────────────────────────────────────
def load_problems(spec: str, limit: int = 0) -> list[dict]:
    """texts.jsonl(문제별로 접는다) 또는 `parquet:<path>[:<level>]` → [{problem_id,
    problem, gold}]. texts.jsonl 의 시도-1 롤아웃 본문은 **쓰지 않는다** — 이 스크립트는
    자기 모델로 다시 뽑는다(훈련된 정책이 다른 모델이기 때문)."""
    out: list[dict] = []
    if spec.startswith("parquet:"):
        import pandas as pd  # noqa: PLC0415

        parts = spec.split(":", 2)
        df = pd.read_parquet(parts[1])
        lvl = parts[2] if len(parts) > 2 else None
        for i, (_, r) in enumerate(df.iterrows()):
            if lvl and (r.get("extra_info") or {}).get("level") != lvl:
                continue
            out.append({"problem_id": f"p{i}", "problem": r["problem"], "gold": str(r["gold"])})
    else:
        seen: set = set()
        with open(spec) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                gid = r.get("group_id", r.get("problem_id"))
                if gid in seen:
                    continue
                seen.add(gid)
                out.append({"problem_id": gid, "problem": r["problem"],
                            "gold": str(r["gold"])})
    return out[:limit] if limit and limit > 0 else out


# ── AUC ───────────────────────────────────────────────────────────────────────
def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """이진 라벨에 대한 ROC AUC(동점은 0.5 로 나눠 갖는 순위 통계량).
    한 쪽 클래스만 있으면 nan — 발화가 전부 같으면 .5 가 나온다(«판별 없음»)."""
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else (0.5 if p == q else 0.0)
    return tot / (len(pos) * len(neg))


def restart_score(row: dict) -> float:
    """P(restart) 의 관측 가능한 대리값 — 발화가 restart 면 1, commit 이면 0,
    발화가 없으면 0.5(중립: 판단을 안 한 행은 어느 쪽으로도 세지 않는다)."""
    d = row.get("decision")
    return 1.0 if d == "restart" else (0.0 if d == "commit" else 0.5)


def auc_report(problems: Sequence[dict]) -> dict:
    """within-problem AUC(두 클래스가 다 있는 문제만, 문제별 평균) + pooled AUC."""
    per: list[float] = []
    all_s: list[float] = []
    all_y: list[int] = []
    for p in problems:
        s = [restart_score(r) for r in p["rows"]]
        y = [1 - int(r["a1_correct"]) for r in p["rows"]]
        all_s.extend(s)
        all_y.extend(y)
        if 0 < sum(y) < len(y):
            per.append(auc(s, y))
    return {"within_problem_auc": (sum(per) / len(per)) if per else float("nan"),
            "n_problems_scored": len(per),
            "pooled_auc": auc(all_s, all_y), "n_rows": len(all_s)}


def notx_for_row(row: dict) -> tuple[str, int]:
    r"""재시도의 X 와 degraded 표식(D4) — 발화가 말한 `x`, 없으면 그 행의 `last_boxed`
    (`a1_answer`), **둘 다 비면 ("", 1)**. X 가 비면 `attempt2_prompt(mode="notx")` 이
    배제 절을 빼고 plain fact 프롬프트와 바이트 동일해지므로(프롬프트 동작은 그대로 둔다)
    그 행을 degraded 로 표시해 보고에서 센다."""
    x = str(row.get("x") or row.get("a1_answer") or "").strip()
    return x, int(not x)


def parse_report(problems: Sequence[dict]) -> dict:
    """D3 버킷 — 블록이 있었나 / 결정이 파싱됐나 / **블록은 있는데 결정이 안 나왔나**.
    세 번째가 위험 버킷이다: 그 행들은 `restart_score` 0.5(중립)로 AUC 에 동점으로 들어가고
    `utter_gated` 를 조용히 `never` 로 퇴화시킨다. 크기를 보고해 눈에 보이게 한다."""
    n = n_block = n_parsed = 0
    for p in problems:
        for r in p["rows"]:
            n += 1
            parsed = r.get("decision") is not None
            # 손으로 만든 행에는 has_block 이 없다 — 결정이 있으면 블록도 있었다.
            blk = bool(r.get("has_block", parsed))
            n_block += int(blk)
            n_parsed += int(parsed)
    n_bad = n_block - n_parsed
    return {"n_rows": n, "n_emitted_block": n_block, "n_decision_parsed": n_parsed,
            "n_block_without_decision": n_bad,
            "emitted_block_rate": (n_block / n) if n else float("nan"),
            "decision_parsed_rate": (n_parsed / n) if n else float("nan"),
            "block_without_decision_rate": (n_bad / n) if n else float("nan")}


def degraded_report(problems: Sequence[dict]) -> dict:
    """D4 버킷 — 배제 재시도인데 X 가 비어(발화의 x 도, 그 행의 `last_boxed` 도 없음)
    프롬프트가 **plain fact 와 바이트 동일**로 내려간 재시도. 전체·상태별."""
    n_retry = n_deg = 0
    by_state: dict = {}
    for p in problems:
        for r in p["rows"]:
            if r.get("a2_answer") is None:
                continue
            c = by_state.setdefault(p["state"], {"n_retry": 0, "n_degraded": 0})
            n_retry += 1
            c["n_retry"] += 1
            d = int(r.get("a2_degraded", 0))
            n_deg += d
            c["n_degraded"] += d
    for c in by_state.values():
        c["degraded_rate"] = (c["n_degraded"] / c["n_retry"]) if c["n_retry"] else float("nan")
    return {"n_retries": n_retry, "n_degraded_retries": n_deg,
            "degraded_rate": (n_deg / n_retry) if n_retry else float("nan"),
            "by_state": by_state}


# ── 프로토콜 회계 ─────────────────────────────────────────────────────────────
def uses_retry(protocol: str, row: dict, state: str) -> bool:
    """이 프로토콜이 이 행에서 재시도 결과를 **쓰는가**(재시도가 실제로 있어야 한다)."""
    if row.get("a2_answer") is None:
        return False
    if protocol == "never":
        return False
    if protocol == "utter_gated":
        return row.get("decision") == "restart"
    if protocol == "always":
        return True
    if protocol == "agree_gated":
        return state != "ALL_SAME"
    raise ValueError(f"[DECISION] 모르는 프로토콜 {protocol!r}")


def protocol_rows(protocol: str, p: dict, strict: bool = False) -> list[dict]:
    """문제 하나에서 이 프로토콜의 행별 최종 (답, 정오, 토큰, 재시도 여부).

    D5: 재시도가 `\\boxed` 답을 **아예 못 낸** 행은 기본(`strict=False`)에서 시도-1 답으로
    **되돌린다** — 빈/절단 재시도가 맞은 시도-1 을 덮어쓰는 것을 막는다. `strict=True` 는
    옛 동작(빈 재시도 = 오답)이며 대조로만 보고한다. **토큰 회계는 두 변종이 동일**하다 —
    재시도는 어느 쪽이든 실제로 굴렸으므로 값을 치른다."""
    out = []
    for r in p["rows"]:
        on = uses_retry(protocol, r, p["state"])
        a2 = str(r.get("a2_answer") or "").strip()
        fell_back = bool(on and not a2 and not strict)
        use_a2 = on and not fell_back
        out.append({
            "answer": (r["a2_answer"] if use_a2 else r["a1_answer"]),
            "correct": int(r["a2_correct"] if use_a2 else r["a1_correct"]),
            "tokens": int(r["a1_tokens"]) + (int(r["a2_tokens"]) if on else 0),
            "retried": int(on), "a1_correct": int(r["a1_correct"]),
            "a2_no_answer": int(on and not a2),
            "a2_truncated": int(on and int(r.get("a2_truncated", 0))),
            "a2_degraded": int(on and int(r.get("a2_degraded", 0))),
            "fell_back": int(fell_back),
        })
    return out


def score_protocol(problems: Sequence[dict], protocol: str, strict: bool = False) -> dict:
    """행 단위 pass@1 · 최종답 maj@K · 문제당 평균 생성 토큰 · flip_wrong ·
    재시도 결함 계수(D4/D5: 답 없음 · 절단 · 배제 절 소실 · 시도-1 로 되돌림)."""
    n_rows = n_ok = n_retried = n_flip = 0
    n_noans = n_trunc = n_deg = n_fb = 0
    tokens = 0
    maj_ok = 0
    for p in problems:
        rows = protocol_rows(protocol, p, strict=strict)
        n_rows += len(rows)
        n_ok += sum(r["correct"] for r in rows)
        n_retried += sum(r["retried"] for r in rows)
        tokens += sum(r["tokens"] for r in rows)
        n_flip += sum(1 for r in rows if r["retried"] and r["a1_correct"] and not r["correct"])
        n_noans += sum(r["a2_no_answer"] for r in rows)
        n_trunc += sum(r["a2_truncated"] for r in rows)
        n_deg += sum(r["a2_degraded"] for r in rows)
        n_fb += sum(r["fell_back"] for r in rows)
        maj_ok += vote([(r["answer"], r["correct"]) for r in rows])[1]
    n_p = len(problems)
    return {"protocol": protocol + ("__strict" if strict else ""), "strict": int(strict),
            "pass1": (n_ok / n_rows) if n_rows else float("nan"),
            "maj_k": (maj_ok / n_p) if n_p else float("nan"),
            "tokens_per_problem": (tokens / n_p) if n_p else float("nan"),
            "total_tokens": tokens, "n_rows": n_rows, "n_retried_rows": n_retried,
            "flip_wrong": n_flip, "n_a2_no_answer": n_noans, "n_a2_truncated": n_trunc,
            "n_degraded_retries": n_deg, "n_fallback_to_a1": n_fb,
            "degraded_rate": (n_deg / n_retried) if n_retried else float("nan")}


def emission_report(problems: Sequence[dict]) -> dict:
    """발화율(전체·상태별)과 상태별 restart 율(발화한 행 중)."""
    n = n_utt = 0
    by_state: dict = {}
    for p in problems:
        for r in p["rows"]:
            n += 1
            has = r.get("decision") is not None
            n_utt += int(has)
            c = by_state.setdefault(p["state"], {"n": 0, "n_utter": 0, "n_restart": 0})
            c["n"] += 1
            c["n_utter"] += int(has)
            c["n_restart"] += int(r.get("decision") == "restart")
    for c in by_state.values():
        c["emission_rate"] = (c["n_utter"] / c["n"]) if c["n"] else float("nan")
        c["restart_rate_of_uttered"] = ((c["n_restart"] / c["n_utter"]) if c["n_utter"]
                                        else float("nan"))
        c["restart_rate_of_all"] = (c["n_restart"] / c["n"]) if c["n"] else float("nan")
    return {"emission_rate": (n_utt / n) if n else float("nan"), "n_rows": n,
            "by_state": by_state}


def summarize(problems: Sequence[dict], protocols: Sequence[str] = PROTOCOLS) -> dict:
    """★재시도를 쓰는 프로토콜은 **두 변종**을 다 보고한다(D5): `X`(빈 재시도는 시도-1 로
    되돌림 = 주된 값) 와 `X__strict`(옛 동작, 빈 재시도 = 오답). 토큰은 동일하다."""
    scored: dict = {}
    for p in protocols:
        scored[p] = score_protocol(problems, p)
        if p != "never":
            scored[f"{p}__strict"] = score_protocol(problems, p, strict=True)
    return {"n_problems": len(problems),
            "emission": emission_report(problems),
            "parse": parse_report(problems),
            "degraded": degraded_report(problems),
            "auc": auc_report(problems),
            "protocols": scored}


def to_markdown(summ: dict) -> str:
    def f(x, nd=4):
        try:
            return "nan" if x != x else f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return "nan"

    e, au = summ["emission"], summ["auc"]
    pa = summ.get("parse", {})
    dg = summ.get("degraded", {})
    L = ["# math_decision_eval — 설계 C «발화 = 결정, 실행 = 리셋»", "",
         f"문제 {summ['n_problems']} · 행 {e['n_rows']} · 발화율 {f(e['emission_rate'])}", "",
         f"파싱 버킷 — 블록 있음 {pa.get('n_emitted_block')} ({f(pa.get('emitted_block_rate'))}) · "
         f"결정 파싱 {pa.get('n_decision_parsed')} ({f(pa.get('decision_parsed_rate'))}) · "
         f"**블록 있는데 결정 없음 {pa.get('n_block_without_decision')}** "
         f"({f(pa.get('block_without_decision_rate'))}) ← 중립 0.5 로 새는 위험 버킷", "",
         f"배제 절 소실(degraded) 재시도 {dg.get('n_degraded_retries')} / "
         f"{dg.get('n_retries')} ({f(dg.get('degraded_rate'))})", "",
         f"within-problem AUC **{f(au['within_problem_auc'])}** "
         f"(n={au['n_problems_scored']} 문제) · pooled AUC {f(au['pooled_auc'])}", "",
         "| protocol | pass@1 | maj@K | tok/prob | n_retried | flip_wrong | "
         "a2_no_answer | a2_trunc | degraded | →a1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in summ["protocols"].items():
        L.append(f"| {name} | {f(r['pass1'])} | {f(r['maj_k'])} | "
                 f"{f(r['tokens_per_problem'], 1)} | {r['n_retried_rows']} | "
                 f"{r['flip_wrong']} | {r.get('n_a2_no_answer')} | "
                 f"{r.get('n_a2_truncated')} | {r.get('n_degraded_retries')} | "
                 f"{r.get('n_fallback_to_a1')} |")
    L += ["", "★`X` = 빈 재시도를 시도-1 로 되돌린 값(주된 값) · `X__strict` = 옛 동작"
          "(빈 재시도 = 오답). 토큰은 두 변종이 같다.", "",
          "## 상태별 발화·restart·degraded", "",
          "| state | n | emission | restart(발화 중) | restart(전체) | degraded/retry |",
          "|---|---|---|---|---|---|"]
    for st, c in e["by_state"].items():
        d = dg.get("by_state", {}).get(st, {})
        L.append(f"| {st} | {c['n']} | {f(c['emission_rate'])} | "
                 f"{f(c['restart_rate_of_uttered'])} | {f(c['restart_rate_of_all'])} | "
                 f"{d.get('n_degraded', 0)}/{d.get('n_retry', 0)} |")
    return "\n".join(L) + "\n"


# ── 캐시 ──────────────────────────────────────────────────────────────────────
def key_of(rec: dict) -> tuple:
    return (rec["stage"], rec["problem_id"], int(rec["roll_id"]))


def load_cache(path: Path) -> dict:
    cache: dict = {}
    if not path.exists():
        return cache
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[key_of(r)] = r
    return cache


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", required=True,
                    help="texts.jsonl(문제만 읽는다) 또는 parquet:<path>[:<level>]")
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--retry_max_tokens", type=int, default=8192)
    ap.add_argument("--retry_rows", default="all", choices=list(RETRY_ROWS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--gen_chunk", type=int, default=2048)
    a = ap.parse_args()

    probs = load_problems(a.dataset, a.limit_problems)
    out = Path(OUT_ROOT) / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    gens_path = out / "gens.jsonl"
    cache = load_cache(gens_path)
    print(f"[dec] 문제 {len(probs)} · K={a.k} · 캐시 {len(cache)}행", flush=True)

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = None
    tok = None
    fh = gens_path.open("a")

    def ensure_llm():
        nonlocal llm, tok
        if llm is None:
            llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
                      gpu_memory_utilization=a.gpu_util,
                      max_model_len=max(a.max_tokens, a.retry_max_tokens) + 9216,
                      enforce_eager=True)
            tok = llm.get_tokenizer()
        return llm, tok

    def emit(stage: str, pid: str, roll: int, text: str, gold: str, finish: str,
             ntok: int, extra: dict | None = None) -> None:
        row = {"stage": stage, "problem_id": pid, "roll_id": int(roll), "text": text,
               "final_answer": last_boxed(text), "r_corr": grade_math(text, gold),
               "truncated": int(finish == "length"), "n_tok": int(ntok), **(extra or {})}
        cache[key_of(row)] = row
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()

    def run(reqs: list[dict], prompts: list[str], budget: int, stage: str) -> None:
        if not reqs:
            return
        _llm, _ = ensure_llm()
        print(f"[dec] {stage} 생성 {len(reqs)}개 (max_tokens={budget})", flush=True)
        sp = [SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=budget,
                             seed=row_seed({"protocol": stage, "problem_id": r["problem_id"],
                                            "roll_id": r["roll_id"], "mode": stage}, a.seed))
              for r in reqs]

        def take(lo: int, outs) -> None:
            for r, o in zip(reqs[lo:lo + len(outs)], outs):
                x = o.outputs[0]
                emit(stage, r["problem_id"], r["roll_id"], x.text, r["gold"],
                     x.finish_reason, len(getattr(x, "token_ids", []) or []),
                     r.get("extra"))

        generate_chunked(_llm, prompts, sp, a.gen_chunk, take)

    # ── (a) 시도 1 — 표준 프롬프트, 발화를 **요구하지 않는다** ──────────────────
    todo = [{"problem_id": p["problem_id"], "roll_id": j, "gold": p["gold"], "problem":
             p["problem"]}
            for p in probs for j in range(a.k) if ("a1", p["problem_id"], j) not in cache]
    if todo:
        _, t = ensure_llm()
        run(todo, [A.blind_prompt(t, VARIANT, r["problem"]) for r in todo], a.max_tokens, "a1")

    # ── 행 조립 + 결정 파싱 ────────────────────────────────────────────────────
    problems: list[dict] = []
    for p in probs:
        rows = []
        for j in range(a.k):
            g = cache.get(("a1", p["problem_id"], j))
            if g is None:
                continue
            d = parse_decision(g["text"])
            rows.append({"roll_id": j, "a1_answer": g.get("final_answer", ""),
                         "a1_correct": int(g.get("r_corr", 0)), "a1_tokens": int(g.get("n_tok", 0)),
                         "decision": d["decision"], "x": d["x"],
                         "has_block": int(has_meta_block(g["text"])),
                         "a2_answer": None, "a2_correct": 0, "a2_tokens": 0,
                         "a2_truncated": 0, "a2_degraded": 0})
        st = agreement_state([r["a1_answer"] for r in rows], k=len(rows))["state"]
        problems.append({"problem_id": p["problem_id"], "problem": p["problem"],
                         "gold": p["gold"], "state": st, "rows": rows})

    # ── (b) 리셋 + 배제 재시도 ────────────────────────────────────────────────
    todo = []
    for p in problems:
        for r in p["rows"]:
            if a.retry_rows == "restart" and r["decision"] != "restart":
                continue
            if ("a2", p["problem_id"], r["roll_id"]) in cache:
                continue
            # X = 발화가 말한 답, 없으면 그 행의 last_boxed. ★D4: 둘 다 비면
            # `attempt2_prompt(mode="notx", notx="")` 이 배제 절을 **통째로 빼** plain
            # fact 프롬프트와 바이트 동일해진다 — 프롬프트는 그대로 두고 그 행을
            # degraded 로 표시해 보고에서 세운다.
            x, deg = notx_for_row(r)
            todo.append({"problem_id": p["problem_id"], "roll_id": r["roll_id"],
                         "gold": p["gold"], "problem": p["problem"], "notx": x,
                         "extra": {"degraded": deg}})
    if todo:
        _, t = ensure_llm()
        run(todo, [attempt2_prompt(t, VARIANT, r["problem"], mode="notx", notx=r["notx"])
                   for r in todo], a.retry_max_tokens, "a2")
    fh.close()

    for p in problems:
        for r in p["rows"]:
            g = cache.get(("a2", p["problem_id"], r["roll_id"]))
            if g is not None:
                r["a2_answer"] = g.get("final_answer", "")
                r["a2_correct"] = int(g.get("r_corr", 0))
                r["a2_tokens"] = int(g.get("n_tok", 0))
                r["a2_truncated"] = int(g.get("truncated", 0))
                # ★`degraded` 필드가 없는 **옛 캐시**는 그 행의 x/last_boxed 로 되돌려
                #   센다 — 기본 0 으로 두면 옛 캐시가 «degraded 0건»으로 조용히 보고된다.
                r["a2_degraded"] = int(g.get("degraded", notx_for_row(r)[1]))

    summ = summarize(problems)
    summ["meta"] = {"model_path": a.model_path, "dataset": a.dataset, "k": a.k,
                    "max_tokens": a.max_tokens, "retry_max_tokens": a.retry_max_tokens,
                    "retry_rows": a.retry_rows, "seed": a.seed, "variant": VARIANT}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(summ)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
