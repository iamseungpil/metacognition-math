#!/usr/bin/env python
r"""math_cited_site_gate — «자기 지목 자리»(self-cited site) 반사실 게이트.

물음 하나. 기저 정책에게 **자기 자신의 틀린 풀이**를 보여 주고 "가장 의심스러운 단계
하나를 대라"고 하면, 그 단계가 **무작위 단계보다 다시 풀기 좋은 자리**인가?

    Δ = p̂(정답 | 지목한 단계 시작부터 다시) − p̂(정답 | 무작위 다른 단계부터 다시)

이게 0 이면 «자기 지목»은 자리 정보가 없다 — 그러면 지목을 보상하는 RL 은 내용이 아니라
형식을 학습한다(cd9 의 EVC 판정과 같은 함정). 이 게이트는 그 전제를 **RL 발사 전에**
오프라인에서 판정한다. 자리를 우리가 자르지 않는다는 점에서 math_sites 의 `own_meta`
자리와 같은 계열이되, 자리를 고르는 것은 «모델이 멈춘 곳»이 아니라 «모델이 지목한 곳»이다.

세 조건(같은 롤아웃, 같은 문맥, 각 K 개 이어쓰기, temperature 1.0):
    cited   앞부분 = 지목 단계 **시작 직전**까지
    random  앞부분 = 지목이 아닌 **무작위 다른** 인용 가능 단계 시작 직전까지 (seeded 대조)
    early   앞부분 = 2번째 단계 시작 직전까지 (고정된 이른 닻 — 두 번째 대조)
★random 대조가 핵심이다. 이게 없으면 «어디서든 다시 풀면 오른다»(재시도 효과)를 «지목이
  옳았다»(자리 정보)로 오독한다.

앞부분 문맥은 **롤아웃이 생성된 그 문맥**과 바이트 동일하다 —
`render_generation_prompt(tok, variant, problem) + text[:start]` (math_sites.gen_request 와
같은 규약). 채점은 `grade_math(prefix + cont, gold)`.

편승 가능성 점검 둘:
  cite_last_rate   지목이 «마지막 인용 가능 단계»인 비율 — 늘 끝을 대는 자명한 전략.
  adherence        지목 조건에서 다시 쓴 단계가 원래 단계와 **다른가**(단어 3-gram Jaccard
                   < 0.5, 잘린 앞 300자). 낮으면 모델이 지목해 놓고 같은 걸 다시 쓴 것이라
                   Δ 가 «지목의 가치»가 아니라 «샘플링 잡음»이다.

통과 규칙: 짝지은 (cited − random) 의 부트스트랩 95% CI 가 0 을 제외하고 평균 > +0.05.

사용(예):
  python scripts/local/math_cited_site_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path <hf> --variant math_opt --k 8 --max_sites 200 --seed 11 \
      --out_dir /hdd_data/seungpil/scratch/eval/cited_site_gate_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_meta import boxed_spans, grade_math, selftest_math_verify  # noqa: E402

_NAN = float("nan")
MIN_STEP_CHARS = 40
MIN_CITABLE = 3
ADHERE_CHARS = 300
ADHERE_JACCARD = 0.5
PASS_DELTA = 0.05

CITATION_ASK = ("Your answer above is wrong. Name the single step most likely to contain the "
                "error. Reply with exactly: suspect: <step number>")
_SENTINEL = "<<<CITED_SITE_GATE_SOLUTION>>>"

# 단락 경계 / 단계 머리말 — 결정적이고 단순하게(정규식 둘).
_BLANK_RE = re.compile(r"\n[ \t]*\n+")
_HEAD_RE = re.compile(r"^[ \t]*(?:#{1,6}\s*)?(?:\*\*\s*)?(?:Step|Case|Therefore|Thus|So)\b", re.M)
# ★«몇 번째 단계인가»의 단일 기준(아래 declared_step_map 주석 참조).
_STEP_NUM_RE = re.compile(r"^[ \t]*(?:#{1,6}\s*)?(?:\*\*\s*)?Step\s*#?\s*(\d+)", re.M)
_CITE_RE = re.compile(r"suspect\s*:\s*\**\s*#?\s*(\d+)", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")


# ── 단계 분절 ───────────────────────────────────────────────────────────────────
def segment_steps(text: str, min_chars: int = MIN_STEP_CHARS) -> list[tuple[int, int]]:
    """응답 텍스트를 번호 매긴 단계 구간 [(start, end), ...] 로 자른다(전체를 덮는다).

    경계: (a) 빈 줄(단락 구분) 뒤, (b) "Step"/"**Step"/"Case"/"Therefore"/"Thus"/"So" 로
    시작하는 줄의 **줄머리**. `min_chars` 보다 짧은 구간은 앞 구간에 합친다(첫 구간이 짧으면
    다음 구간을 자기 쪽으로 당긴다). 공백뿐인 꼬리도 앞에 붙는다.
    """
    t = text or ""
    if not t.strip():
        return []
    bounds = {0, len(t)}
    for m in _BLANK_RE.finditer(t):
        bounds.add(m.end())
    for m in _HEAD_RE.finditer(t):
        bounds.add(m.start())
    bs = sorted(b for b in bounds if 0 <= b <= len(t))
    raw = [(a, b) for a, b in zip(bs, bs[1:]) if b > a]
    segs: list[tuple[int, int]] = []
    for a, b in raw:
        if segs and (b - a) < min_chars:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    # 첫 구간이 짧으면(앞에 합칠 데가 없다) 다음 구간을 당겨 온다.
    if len(segs) > 1 and (segs[0][1] - segs[0][0]) < min_chars:
        segs = [(segs[0][0], segs[1][1])] + segs[2:]
    return segs


def citable_indices(text: str, segs: Sequence[tuple[int, int]]) -> list[int]:
    r"""인용 가능한 단계의 0-기반 인덱스. 마지막 \boxed 가 든 단계 **와 그 뒤**는 뺀다 —
    그 자리의 앞부분은 이미 답을 담고 있어 «다시 풀기»가 성립하지 않는다(채점은
    prefix+cont 위에서 하므로 답이 앞부분에 있으면 이어쓰기와 무관하게 점수가 정해진다).
    """
    sp = boxed_spans(text or "")
    if not sp:
        return list(range(len(segs)))
    pos = sp[-1][1] - 1                      # 마지막 \boxed 의 닫는 괄호
    cut = len(segs)
    for i, (a, b) in enumerate(segs):
        if a <= pos < b:
            cut = i
            break
    return list(range(cut))


def declared_step_map(text: str, segs: Sequence[tuple[int, int]]) -> dict[int, int]:
    r"""모델이 **스스로 붙인** "Step k" 머리말 → 그 머리말이 든 구간 인덱스.

    ★왜 필요한가(실측, mathL5 Qwen3-4B-Instruct opt): 응답 169개 중 167개가 "### Step k"
    머리말을 달고 있고, 단락 분절은 한 응답을 중앙값 90 구간으로 쪼갠다. 지목 프롬프트는
    번호 목록을 **보여 주지 않으므로** 모델이 말하는 «step 3» 은 우리 90 구간의 3번이 아니라
    자기가 쓴 "Step 3" 이다. 머리말이 있으면 그 번호를 기준으로 삼고, 없을 때만 구간 순번
    (1-기반)으로 읽는다 — 그러지 않으면 «지목 자리»가 사실상 무작위 자리가 되어 게이트가
    자기 가설을 죽인다.
    """
    starts = {a: i for i, (a, _) in enumerate(segs)}
    out: dict[int, int] = {}
    for m in _STEP_NUM_RE.finditer(text or ""):
        k = int(m.group(1))
        if k in out:
            continue                      # 같은 번호가 여러 번이면 첫 번째
        i = starts.get(m.start())
        if i is None:                     # 머리말이 구간 머리가 아니면(짧아서 합쳐짐) 포함 구간
            i = next((j for j, (a, b) in enumerate(segs) if a <= m.start() < b), None)
        if i is not None:
            out[k] = i
    return out


# ── 인용(지목) 프롬프트 ──────────────────────────────────────────────────────────
def build_citation_prompt(tok, variant: str, problem: str, text: str,
                          ask: str = CITATION_ASK) -> str:
    """원 풀이를 assistant 턴으로 **그대로** 담고, 새 user 턴(ask)을 붙인 생성 프롬프트.

    ★바이트 동일 보장: 템플릿에 sentinel 을 넣어 렌더한 뒤 sentinel 자리에 원문을 되꽂는다.
      (Qwen3.5 템플릿은 assistant 본문의 끝 공백을 지운다 — 본문을 그대로 넣으면 프롬프트가
      원 롤아웃 문맥과 조용히 달라진다. math_sites.gen_request 의 교훈과 같은 함정.)
    """
    msgs = build_math_prompt(problem, variant) + [
        {"role": "assistant", "content": _SENTINEL},
        {"role": "user", "content": ask},
    ]
    rendered = render_chat_messages(tok, msgs)
    if rendered.count(_SENTINEL) != 1:
        raise RuntimeError(f"[CITE] chat 템플릿에서 sentinel 을 {rendered.count(_SENTINEL)}번 찾았다 "
                           "— 1번이어야 한다(템플릿이 assistant 본문을 변형한다).")
    head, tail = rendered.split(_SENTINEL)
    return head + (text or "") + tail


def continuation_prompt(tok, variant: str, problem: str, text: str, start: int) -> str:
    """반사실 이어쓰기 문맥 = 롤아웃이 생성된 문맥 + 앞부분 (math_sites.gen_request 와 같은 규약).
    ★바이트 동일이 전제다 — 여기서 한 글자라도 달라지면 «같은 자리»가 아니다."""
    return render_generation_prompt(tok, variant, problem) + (text or "")[:start]


def parse_citation(out_text: str, n_segs: int, citable: Sequence[int],
                   step_map: dict | None = None) -> int | None:
    """`suspect: k` → 0-기반 구간 인덱스. 파싱 실패·범위 밖·인용 불가 단계면 None.

    step_map 이 있으면(모델이 "Step k" 머리말을 달았으면) 그 번호로 읽고, 없으면 구간 순번
    (1-기반)으로 읽는다 — declared_step_map 주석 참조.
    """
    m = _CITE_RE.search(out_text or "")
    if not m:
        return None
    k = int(m.group(1))
    idx = step_map.get(k) if step_map else (k - 1)
    if idx is None or not (0 <= idx < n_segs) or idx not in set(citable):
        return None
    return idx


def random_other(cited: int, citable: Sequence[int], rng: random.Random) -> int | None:
    """지목이 아닌 인용 가능 단계 하나(균등). 후보가 없으면 None."""
    cands = [i for i in citable if i != cited]
    return rng.choice(cands) if cands else None


# ── 준수(adherence) 프록시 ──────────────────────────────────────────────────────
def _grams(s: str, n: int = 3) -> set[tuple[str, ...]]:
    w = [x.lower() for x in _WORD_RE.findall(s or "")]
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def jaccard3(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def differs_from_original(orig_step: str, cont: str, *, chars: int = ADHERE_CHARS,
                          thresh: float = ADHERE_JACCARD) -> int:
    """다시 쓴 단계가 원래 단계와 **다른가**(앞 chars 자, 단어 3-gram Jaccard < thresh)."""
    return int(jaccard3((orig_step or "")[:chars], (cont or "")[:chars]) < thresh)


# ── 통계 ────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """백분위 부트스트랩 CI(롤아웃 단위 재표본). 유한한 값만 쓴다."""
    v = [float(x) for x in values if x is not None and isinstance(x, (int, float))
         and math.isfinite(float(x))]
    if not v:
        return {"mean": _NAN, "lo": _NAN, "hi": _NAN, "n": 0}
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(v) / n, "lo": bs[int(alpha / 2 * n_boot)],
            "hi": bs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))], "n": n}


def sign_test_p(diffs: Sequence[float]) -> float:
    """양측 부호검정 p(0 인 차는 버린다). n=0 이면 1.0."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def summarize(recs: Sequence[dict], *, n_no_citation: int = 0, k: int = 0,
              seed: int = 0, n_boot: int = 2000) -> dict:
    """per-rollout 기록 → 게이트 요약. `recs` 는 지목이 성립한 롤아웃만(no_citation 제외)."""
    n = len(recs)
    diffs = [r["p_cited"] - r["p_random"] for r in recs]
    adh = [r for r in recs if r["adherent"]]
    last = [r for r in recs if r["cite_last"]]
    notlast = [r for r in recs if not r["cite_last"]]
    out = {
        "n_rollouts": n,
        "n_no_citation": int(n_no_citation),
        "no_citation_rate": (n_no_citation / (n + n_no_citation)) if (n + n_no_citation) else _NAN,
        "cite_last_rate": (len(last) / n) if n else _NAN,
        "mean_rel_pos": (sum(r["rel_pos"] for r in recs) / n) if n else _NAN,
        "k": k,
        "p_cited": bootstrap_ci([r["p_cited"] for r in recs], seed=seed, n_boot=n_boot),
        "p_random": bootstrap_ci([r["p_random"] for r in recs], seed=seed + 1, n_boot=n_boot),
        "p_early": bootstrap_ci([r["p_early"] for r in recs], seed=seed + 2, n_boot=n_boot),
        "paired_cited_minus_random": bootstrap_ci(diffs, seed=seed + 3, n_boot=n_boot),
        "sign_test_p": sign_test_p(diffs),
        "frac_cited_gt_random": (sum(1 for d in diffs if d > 0) / n) if n else _NAN,
        "adherence_rate": (sum(r["adherence"] for r in recs) / n) if n else _NAN,
        "n_adherent": len(adh),
        "paired_cited_minus_random_adherent": bootstrap_ci(
            [r["p_cited"] - r["p_random"] for r in adh], seed=seed + 4, n_boot=n_boot),
        "p_cited_when_last": bootstrap_ci([r["p_cited"] for r in last], seed=seed + 5, n_boot=n_boot),
        "p_cited_when_not_last": bootstrap_ci([r["p_cited"] for r in notlast], seed=seed + 6,
                                              n_boot=n_boot),
        "trunc_rate": (sum(r.get("trunc_rate", 0.0) for r in recs) / n) if n else _NAN,
    }
    out["pass"] = int(gate_pass(out))
    return out


def gate_pass(summ: dict) -> bool:
    """짝지은 Δ(cited − random) 의 95% CI 가 0 을 제외하고 평균 > +.05 → PASS."""
    ci = summ.get("paired_cited_minus_random") or {}
    lo, hi, mean = ci.get("lo"), ci.get("hi"), ci.get("mean")
    if any(x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x))
           for x in (lo, hi, mean)):
        return False
    return (lo > 0 or hi < 0) and mean > PASS_DELTA


def _f(v) -> str:
    if isinstance(v, dict):
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(summ: dict) -> str:
    keys = ["n_rollouts", "n_no_citation", "no_citation_rate", "cite_last_rate", "mean_rel_pos",
            "k", "p_cited", "p_random", "p_early", "paired_cited_minus_random", "sign_test_p",
            "frac_cited_gt_random", "adherence_rate", "n_adherent",
            "paired_cited_minus_random_adherent", "p_cited_when_last", "p_cited_when_not_last",
            "trunc_rate"]
    lines = ["## math_cited_site_gate — 자기 지목 자리 반사실", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(summ.get(k))} |" for k in keys]
    lines += ["", f"**{'PASS' if summ.get('pass') else 'FAIL'}** — 규칙: paired(cited − random) "
                  f"95% CI 가 0 을 제외 ∧ 평균 > +{PASS_DELTA}", ""]
    return "\n".join(lines)


# ── 롤아웃 선별 ─────────────────────────────────────────────────────────────────
def select_wrong_rollouts(rolls: Sequence[dict], *, per_problem: int = 2) -> list[dict]:
    """MIXED 그룹(0 < 그룹 정답률 < 1)의 **오답·미잘림** 롤아웃, 문제당 최대 per_problem 개."""
    acc: dict = {}
    for r in rolls:
        acc.setdefault(r["group_id"], []).append(int(r["r_corr"]))
    mixed = {g for g, v in acc.items() if 0 < sum(v) < len(v)}
    seen: dict = {}
    out = []
    for i, r in enumerate(rolls):
        if r["group_id"] not in mixed or int(r["r_corr"]) or r.get("truncated"):
            continue
        if seen.get(r["group_id"], 0) >= per_problem:
            continue
        seen[r["group_id"]] = seen.get(r["group_id"], 0) + 1
        out.append({"roll_id": f"{r['group_id']}#{i}", "group_id": r["group_id"],
                    "problem": r["problem"], "gold": r["gold"], "text": r["text"]})
    return out


def cite_options(segs: Sequence[tuple[int, int]], citable: Sequence[int], step_map: dict,
                 *, min_citable: int = MIN_CITABLE) -> list[int]:
    """지목·무작위 대조가 **같은 후보 집합**에서 나오게 한다.

    ★모델이 "Step k" 머리말을 달았고 그중 인용 가능한 것이 min_citable 개 이상이면 그
    단계들만 후보다 — 그러지 않으면 cited 는 «단계 머리»이고 random 은 «단락 아무 데»라서
    자리의 종류 자체가 달라진다(Δ 가 지목의 가치가 아니라 «단계 경계에서 끊은 효과»를 잰다).
    머리말이 없으면(실측 169개 중 2개) 단락 순번으로 되돌아간다.
    """
    cs = set(citable)
    declared = sorted({i for i in step_map.values() if i in cs})
    return declared if len(declared) >= min_citable else list(citable)


def prepare_rollouts(rolls: Sequence[dict], *, per_problem: int = 2,
                     min_citable: int = MIN_CITABLE,
                     require_declared: bool = True) -> tuple[list[dict], int]:
    """선별 + 분절. 인용 후보가 min_citable 미만인 롤아웃은 버린다 → (남은 것, 버린 수).

    ★require_declared(기본 True): "Step k" 머리말 기반 후보가 min_citable 개 이상인 롤아웃만
    남긴다. 단락 순번으로 되돌아간 롤아웃에선 모델이 말하는 «step 3» 과 우리가 세는 «단락 3»
    이 다를 게 거의 확실하고(실측: 단락 중앙값 90개), 그러면 그 행의 «지목»은 사실상 이른
    무작위 자리가 되어 Δ 를 0 쪽으로 희석한다. False 로 두면 단락 순번 판도 포함한다.
    """
    kept, dropped = [], 0
    for r in select_wrong_rollouts(rolls, per_problem=per_problem):
        segs = segment_steps(r["text"])
        cit = citable_indices(r["text"], segs)
        sm = declared_step_map(r["text"], segs)
        opts = cite_options(segs, cit, sm, min_citable=min_citable)
        declared = bool(sm) and opts != list(cit)
        if len(opts) < min_citable or (require_declared and not declared):
            dropped += 1
            continue
        kept.append({**r, "segs": segs, "citable": cit, "step_map": sm, "options": opts,
                     "declared": int(declared)})
    return kept, dropped


def _p(vals: Sequence[int]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_sites", type=int, default=200, help="지목을 물을 롤아웃 수 상한")
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--allow_paragraph_numbering", action="store_true",
                    help="'Step k' 머리말이 없는(→ 단락 순번으로 번호를 읽는) 롤아웃도 포함한다. "
                         "기본은 제외 — prepare_rollouts 주석 참조.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_prefix_tokens", type=int, default=8192,
                    help="★지목 프롬프트는 풀이 전체를 담는다 — mathL5 실측 최대 8,489 토큰이라 "
                         "6144 로 두면 후보의 25%%가 조용히 빠진다")
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    cands, n_short = prepare_rollouts(rolls, per_problem=a.per_problem,
                                      require_declared=not a.allow_paragraph_numbering)
    rng.shuffle(cands)
    cands = cands[:a.max_sites]
    print(f"[gate] 오답·미잘림·MIXED 롤아웃 {len(cands)}개 (인용 후보 부족/번호 불명으로 버린 것 {n_short}개)",
          flush=True)
    if not cands:
        raise SystemExit("[gate] 후보가 없다 — 입력 롤아웃에 MIXED 오답이 있는지 확인하라.")

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()

    # ── 1) 지목 한 번(그리디) ────────────────────────────────────────────────
    lim = a.max_prefix_tokens + 512
    cite_reqs, cite_ix = [], []
    n_long = 0
    for i, r in enumerate(cands):
        req = build_citation_prompt(tok, a.variant, r["problem"], r["text"])
        if len(tok.encode(req)) > lim:
            n_long += 1
            continue
        cite_reqs.append(req)
        cite_ix.append(i)
    print(f"[gate] 지목 요청 {len(cite_reqs)}개 (너무 긴 것 {n_long}개 제외, 한도 {lim} 토큰)", flush=True)
    cite_outs = llm.generate(cite_reqs, SamplingParams(n=1, temperature=0.0, max_tokens=64,
                                                       seed=a.seed))

    cited: list[dict] = []
    n_no_citation = 0
    for i, o in zip(cite_ix, cite_outs):
        r = cands[i]
        raw = o.outputs[0].text
        k = parse_citation(raw, len(r["segs"]), r["options"], r["step_map"])
        if k is None:
            n_no_citation += 1
            continue
        other = random_other(k, r["options"], rng)
        if other is None:
            n_no_citation += 1
            continue
        # ★early = «두 번째 단계» 고정 닻(후보 집합의 두 번째 — 머리말이 있으면 Step 2).
        early = r["options"][1] if len(r["options"]) > 1 else r["options"][0]
        cited.append({**r, "cite_raw": raw, "cited_idx": k, "random_idx": other,
                      "early_idx": early})
    print(f"[gate] 지목 성립 {len(cited)} / 무효 {n_no_citation}", flush=True)
    if not cited:
        raise SystemExit("[gate] 유효한 지목이 하나도 없다 — 지목 프롬프트/파서를 확인하라.")

    # ── 2) 세 조건 반사실 이어쓰기 ────────────────────────────────────────────
    conds = ("cited", "random", "early")
    reqs, ix = [], []
    for si, s in enumerate(cited):
        for c in conds:
            start = s["segs"][s[f"{c}_idx"]][0]
            reqs.append(continuation_prompt(tok, a.variant, s["problem"], s["text"], start))
            ix.append((si, c, start))
    keep = [j for j, q in enumerate(reqs) if len(tok.encode(q)) <= lim]
    n_drop = len(reqs) - len(keep)
    reqs = [reqs[j] for j in keep]
    ix = [ix[j] for j in keep]
    print(f"[gate] 이어쓰기 요청 {len(reqs)}개 x K={a.k} (긴 앞부분으로 버린 것 {n_drop}개)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    rows = []
    for (si, c, start), o in zip(ix, outs):
        s = cited[si]
        prefix = s["text"][:start]
        for x in o.outputs:
            corr = grade_math(prefix + x.text, s["gold"])
            agg.setdefault((si, c), []).append(
                {"r_corr": corr, "cont": x.text, "trunc": int(x.finish_reason == "length")})
            rows.append({"roll_id": s["roll_id"], "cond": c, "r_corr": corr,
                         "truncated": int(x.finish_reason == "length"), "cont": x.text})

    # ── 3) per-rollout 기록 ──────────────────────────────────────────────────
    recs = []
    for si, s in enumerate(cited):
        got = {c: agg.get((si, c)) for c in conds}
        if not all(got.values()):
            continue                      # 세 조건이 다 있어야 짝 비교가 성립한다
        ca, cb = s["segs"][s["cited_idx"]]
        orig_step = s["text"][ca:cb]
        adh = [differs_from_original(orig_step, x["cont"]) for x in got["cited"]]
        rec = {
            "roll_id": s["roll_id"], "group_id": s["group_id"], "problem": s["problem"],
            "gold": s["gold"], "n_segs": len(s["segs"]), "n_citable": len(s["citable"]),
            "n_options": len(s["options"]), "declared_numbering": s["declared"],
            "cited_idx": s["cited_idx"], "random_idx": s["random_idx"],
            "n_declared_steps": len(s["step_map"]),
            "early_idx": s["early_idx"], "cite_raw": s["cite_raw"],
            "cite_last": int(s["cited_idx"] == s["options"][-1]),
            "rel_pos": ca / max(1, len(s["text"])),
            "p_cited": _p([x["r_corr"] for x in got["cited"]]),
            "p_random": _p([x["r_corr"] for x in got["random"]]),
            "p_early": _p([x["r_corr"] for x in got["early"]]),
            "adherence": _p(adh),
            "trunc_rate": _p([x["trunc"] for c in conds for x in got[c]]),
        }
        rec["adherent"] = int(rec["adherence"] >= 0.5)
        recs.append(rec)

    summ = summarize(recs, n_no_citation=n_no_citation, k=a.k, seed=a.seed, n_boot=a.n_boot)
    summ.update({"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
                 "seed": a.seed, "n_candidates": len(cands), "n_dropped_short": n_short,
                 "n_dropped_long_prompt": n_long, "n_dropped_long_prefix": n_drop,
                 "n_continuations": len(rows)})

    with (out / "per_rollout.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "continuations.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(to_markdown(summ))
    print(to_markdown(summ))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
