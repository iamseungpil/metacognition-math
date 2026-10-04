r"""MATH_DIFF — «난이도 판단»(verbalized difficulty → sample allocation) 메타인지의 단일 진실 원천.

★왜 이 파일인가 (2026-09-14, cd9). 이 정책의 **내부 상태**는 문제 난이도를 읽는다 —
문제 홀드아웃 은닉 프로브의 풀링 AUC .825, 자기 pass rate 와 Spearman +.51
(docs/HYPOTHESIS_LEDGER_cd9.md §C1). 그런데 **말로 낸** confidence 는 그 축을 거의 담지
못하고(+.13), 더 나쁘게도 **GRPO 그룹 중심화가 문제 간 축을 통째로 지운다** — 같은 문제의
형제끼리만 비교하므로 «이 문제가 어렵다»는 성분은 정확히 상수로 빠진다.

그래서 M_DIFF 는 셋을 한꺼번에 바꾼다:
  ① **말하게 한다** — 응답은 `<meta>…</meta>` 한 블록으로 **시작**하고, 그 안에
     `difficulty: easy|medium|hard` 한 줄(그리고 `why: …` 한 문장)을 **풀기 전에** 쓴다.
  ② **중심화하지 않고 보상한다** — 판단 크레딧은 그룹 평균을 빼지 않고 메타 스팬에 그대로
     얹는다(verl_sdc._math_add_diff_meta_advantage). 같은 문제의 형제는 **같은 라벨**을
     공유하므로 중심화하면 신호가 0 이 된다 — 그것이 바로 지우려는 축이다.
  ③ **gold 없이 라벨을 만든다** — 라벨은 그 GRPO 그룹 형제들의 **LEAVE-ONE-OUT 답 동의도**
     (자기 자신 제외)다. 형제 동의도는 참 pass rate 와 ρ≈.55 로 상관한다(ledger C1) — 이
     팔이 쓰는 **gold-free 대리자**다.

  diff_row_credit = +1  말한 버킷 == 동의도 버킷
                    −1  그 외(이웃도 반대도 −1 — «이웃=0»은 세이프 하버였다: 항상 medium 이라고만
                        말하면 기댓값이 절대 음수가 될 수 없어 RL이 배우지 않고 거기 주저앉을 수
                        있다. 방향만 남긴다.)
                    미정의(0, False)  블록 없음 · 블록이 맨 앞이 아님 · 버킷 미파싱 · why 없음 ·
                                      메타 안 \boxed · 블록 다수 · LOO 라벨 미정의

★«신호가 작으면 방향만 정규화해서 보상»(사용자 규칙) — 크레딧은 연속 오차가 아니라 ±1/0 의
  **방향**이다. 동의도 숫자 자체를 맞히라고 하면(회귀) 작은 신호 위에서 잡음만 학습한다.
★gold 는 크레딧에 **한 번도** 들어가지 않는다. `diff_row_metrics`(형제 r_corr 로 잰 실제
  LOO pass rate)는 **지표 전용**이고 보상 경로의 어떤 함수도 읽지 않는다(테스트가 gold 와
  r_corr 를 지운 행으로 확인한다).

통제(control)는 추론에서 따라온다: «hard 라고 말한 문제에 표본을 더 준다» — 평균 예산을
고정한 채 배분하면 균일 self-consistency 를 이기는가(scripts/local/math_diff_eval.py 의
allocation 표). 이 파일을 데이터 빌더(build_math_parquet.py --variant math_diff)·팔
(src/training/math_meta.py)·평가(scripts/local/math_diff_eval.py)가 **함께** 쓴다 —
프롬프트·파싱·버킷·크레딧 정의가 갈리면 학습과 판정을 나란히 읽을 수 없다(math_dis 와 같은 이유).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from src.training import countdown_rewards as _cdr
from src.training import math_dis as _md
from src.training import math_meta as _mm

DIFF_SPEC_VERSION = "math-diff-0914"

# 세 버킷. 순서가 곧 «난이도 축»이다(인덱스 차 1 = 이웃, 2 = 반대) — 크레딧이 이 순서에 묶인다.
BUCKETS = ("easy", "medium", "hard")

# ★동의도 → 버킷 문턱. 그룹 n=8 의 LOO 형제는 7 개이므로 3/7·6/7 은 «7 중 3 이하» / «7 중 6 이상»
#   이라는 **개수 문턱**이다(연속 값에 임의로 그은 선이 아니다). 형제 동의도는 참 pass rate 와
#   ρ≈.55 (docs/HYPOTHESIS_LEDGER_cd9.md §C1) — gold 를 안 쓰는 대신 이 만큼의 잡음을 안고 간다.
AGREE_HARD_MAX = 3.0 / 7.0      # a ≤ 3/7  → hard
AGREE_EASY_MIN = 6.0 / 7.0      # a ≥ 6/7  → easy  (그 사이는 medium)


# ── 프롬프트(사용자 턴 접미) ────────────────────────────────────────────────────
# ★시스템 프롬프트는 math_opt 와 **바이트 동일**하다(src/metacot/math_meta_prompt.py 의
#   `math_diff` 변형) — 갈리면 «허가된 메타»의 세금이 두 팔에서 달라진다. 갈리는 것은 이
#   사용자 턴 접미뿐이고, 이 접미는 **문제 하나만 있으면** 붙는다(M_DIS 와 달리 행마다 다른
#   재료가 없다 — build_math_prompt(problem, "math_diff") 로 조립된다).
# ★`\boxed{}` 가 들어 있으므로 str.format 을 쓰지 않는다(중괄호가 포맷 자리로 읽힌다).
DIFF_ASK = (
    "\n\nBefore solving, inside one <meta>...</meta> block write "
    "`difficulty: easy|medium|hard` (how likely YOU are to solve this correctly on one "
    "attempt: easy = almost surely, medium = uncertain, hard = probably not) and one "
    "sentence `why: ...`. Then solve and give the final answer in \\boxed{}."
)

# 변형 어서션·빌더가 공유하는 표식 — 이 문자열이 사용자 턴에 없으면 math_diff 프롬프트가 아니다.
DIFF_MARKER = "difficulty: easy|medium|hard"
assert DIFF_MARKER in DIFF_ASK


def build_diff_user_turn(problem: str) -> str:
    """문제 + DIFF_ASK. ★M_DIS 와 달리 행마다 다른 재료가 없다 — 그래서 parquet 빌더가
    `build_math_prompt(problem, "math_diff")` 로 바로 조립할 수 있다."""
    return str(problem).strip() + DIFF_ASK


# ── 응답 파싱 ──────────────────────────────────────────────────────────────────
# ★`difficulty:` 는 **줄 하나**로 쓰라고 지시했지만 파서는 줄 시작을 요구하지 않는다 —
#   요구하면 «> difficulty: hard» 같은 사소한 장식이 버킷 미파싱(항 미정의)이 되어 크레딧이
#   형식 사고로 사라진다. 대신 **첫** 매치만 본다(둘 이상 쓰면 첫 것이 구속력을 갖는다 —
#   나중 것으로 바꿔 쓰면 «말 바꾸기»가 공짜가 된다. math_dis.parse_commit 과 같은 규약).
_DIFF_RE = re.compile(r"difficulty\s*:\s*(easy|medium|hard)", re.IGNORECASE)
_WHY_RE = re.compile(r"why\s*:\s*(\S.*)", re.IGNORECASE)
_META_TAG_RE = re.compile(r"</?meta>", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")


def parse_difficulty(meta_text: str) -> str | None:
    """메타 블록 원문 → `easy|medium|hard` 중 하나(소문자). 없거나 다른 낱말이면 None."""
    m = _DIFF_RE.search(meta_text or "")
    return m.group(1).lower() if m else None


def parse_why(meta_text: str) -> str:
    """`why:` 뒤 한 줄(없으면 ""). 이유 문장이 없으면 «버킷 한 낱말만 찍은 것»이다 —
    항이 정의되지 않는다(diff_row_flags.has_why)."""
    m = _WHY_RE.search(meta_text or "")
    return m.group(1).strip() if m else ""


def diff_text(meta_text: str) -> str:
    """메타 본문(태그 제거) — 정형문 자(countdown_rewards.boilerplate_rate)가 읽는 `body`."""
    return " ".join(_META_TAG_RE.sub(" ", meta_text or "").split())


def diff_words(meta_text: str) -> int:
    """메타 본문 단어 수(관찰용 — 길면 판단 자리에서 문제를 풀고 있는 것이다)."""
    return len(_WORD_RE.findall(diff_text(meta_text)))


def parse_diff_row(text: str, gold: str, problem: str, *, truncated: int = 0) -> dict:
    r"""M_DIFF 한 롤아웃의 원재료 — 학습(math_meta._compute_diff_rows)과 held-out
    (scripts/local/math_diff_eval.py)이 **같은 파서**를 쓴다.

    ★최종 답은 <meta> **밖의** 마지막 \boxed 다(M_DIS/M_RETRY 와 같은 규약) — 메타 안 박스가
      답으로 읽히면 boxed_in_meta 가드가 무력해진다.
    ★`meta_first`: 첫 블록이 응답 **맨 앞**에서 시작하는가(앞의 공백만 허용). 이 팔의 주장은
      «풀기 **전에** 난이도를 읽는다»이므로, 다 풀고 나서 쓴 난이도는 판단이 아니라 사후 보고다.
    """
    t = text or ""
    blocks = _mm._meta_block_spans(t)
    raw = t[blocks[0][0]:blocks[0][1]] if blocks else ""
    spans = _mm.boxed_spans(t, exclude=blocks)
    final_ans = spans[-1][0] if spans else None
    bucket = parse_difficulty(raw)
    why = parse_why(raw)
    body = diff_text(raw)
    return {
        "text": t, "gold": str(gold), "problem": problem,
        # ★`meta` 를 직접 싣는 이유(math_dis 와 같다): countdown_rewards 의 자는 행에 `meta` 가
        #   있으면 그것을 읽고 없으면 parse_meta(form="math")로 다시 판다 — M_DIFF 블록엔
        #   `confidence:` 줄이 없어 그 파싱은 emitted=0 을 내므로 발화율·정형문이 «못 쟀다»로 빠진다.
        "meta": {"emitted": int(bool(blocks)), "form": "diff", "body": body, "raw": raw,
                 "start": blocks[0][0] if blocks else None,
                 "end": blocks[0][1] if blocks else None,
                 "confidence": None, "decision": None, "n_blocks": len(blocks)},
        "bucket": bucket,
        "why": why,
        "diff_text": body,
        "diff_words": diff_words(raw),
        "final_answer": final_ans,
        "r_corr": _mm.grade_math(f"\\boxed{{{final_ans}}}", gold) if final_ans else 0,
        "emitted": int(bool(blocks)),      # ★M_DIFF 의 «발화» = 블록이 있다(confidence 줄은 안 쓴다)
        "has_meta": int(bool(blocks)),
        "meta_first": int(bool(blocks) and not t[:blocks[0][0]].strip()),
        "n_blocks": len(blocks),
        "multi_block": int(len(blocks) > 1),
        "meta_start": blocks[0][0] if blocks else None,
        "meta_end": blocks[0][1] if blocks else None,
        "meta_raw": raw,
        "boxed_in_meta": int(any(_mm._BOXED_RE.search(t[a:b]) for a, b in blocks)),
        "n_chars": len(t),
        "truncated": int(_cdr._bool01(truncated)),
    }


# ── 형식 플래그 ────────────────────────────────────────────────────────────────
def diff_row_flags(row: Mapping) -> dict:
    r"""중단 규칙·텔레메트리가 읽는 형식 불리언. 행이 파싱 결과(parse_diff_row)든 원문만
    들고 있든 같은 답을 내도록, 없는 키는 `text` 에서 다시 뽑는다.

    has_meta       <meta>…</meta> 가 하나라도 있다
    meta_first     첫 블록이 응답 맨 앞에서 시작한다(앞 공백 허용) — «풀기 전에 판단했다»
    bucket_parsed  `difficulty: easy|medium|hard` 가 파싱됐다
    has_why        `why: …` 문장이 있다(버킷 한 낱말만 찍은 것이 아니다)
    boxed_in_meta  메타 블록 안에 \boxed 가 있다(답 누출 — 항 미정의)
    multi_block    블록이 둘 이상(형식 위반 — 첫 블록만 채점되므로 나머지는 공짜 토큰)
    """
    if "bucket" in row and "meta_first" in row:
        r = row
    else:
        r = parse_diff_row(row.get("text", ""), row.get("gold", ""), row.get("problem", ""))
    return {
        "has_meta": int(bool(r.get("has_meta", r.get("emitted", 0)))),
        "meta_first": int(bool(r.get("meta_first", 0))),
        "bucket_parsed": int(r.get("bucket") in BUCKETS),
        "has_why": int(bool(str(r.get("why") or "").strip())),
        "boxed_in_meta": int(bool(r.get("boxed_in_meta", 0))),
        "multi_block": int(bool(r.get("multi_block", int(r.get("n_blocks", 0)) > 1))),
    }


# ── gold-free 라벨: LOO 형제 동의도 ────────────────────────────────────────────
def loo_agreement(row_idx: int, group_rows: Sequence[Mapping]) -> float | None:
    r"""**이 행을 뺀** 형제들의 답 동의도 ∈ [0,1] — 나머지 롤아웃 중 그들의 다수답
    (`math_dis.plurality_answer`, 수학 동치로 군집)과 같은 답을 낸 비율.

    ★왜 leave-one-out 인가: 자기 답을 넣으면 «내가 쓴 답이 다수에 표를 보태» 자기 난이도를
      스스로 쉬워 보이게 만들 수 있다(math_dis.dis_row_credit 의 D1 과 같은 순응 유인).
      이 프로젝트의 다른 자기증류 라벨(ledger c1.py 형제 답 점유율, M_RETRY_SL 의 p_retry,
      M_DIS 의 판단 크레딧)도 전부 LOO 다.
    ★답 있는 형제가 **2 미만**이거나 다수답이 **동률**이면 None(= 라벨 미정의) — 하나로는
      «동의»를 논할 수 없고, 동률에서 하나를 고르면 라벨이 사실은 «배치 순서»가 된다.
    ★gold 를 읽지 않는다 — 형제들의 \boxed 답끼리만 비교한다.
    """
    rest = [r for i, r in enumerate(group_rows) if i != int(row_idx)]
    answers = [str(r.get("final_answer") or "").strip() for r in rest]
    answers = [a for a in answers if a]
    if len(answers) < 2:
        return None
    plur = _md.plurality_answer(answers)
    if plur is None:
        return None
    return sum(1 for a in answers if _mm.answers_equivalent(a, plur)) / len(answers)


def agree_bucket(a: float | None) -> str | None:
    """동의도 → 버킷(문턱은 AGREE_HARD_MAX/AGREE_EASY_MIN 상수). None 이면 None."""
    if a is None or not _cdr._finite(a):
        return None
    x = float(a)
    if x <= AGREE_HARD_MAX:
        return "hard"
    if x >= AGREE_EASY_MIN:
        return "easy"
    return "medium"


# ── 방향 크레딧 ────────────────────────────────────────────────────────────────
def _bucket_idx(b) -> int | None:
    return BUCKETS.index(b) if b in BUCKETS else None


def diff_row_credit(row: Mapping, group_rows: Sequence[Mapping],
                    self_idx: int | None = None) -> tuple[float, bool]:
    r"""난이도 판단 크레딧 → (크레딧, 정의됨). **방향만** 준다(사용자 규칙 «신호가 작으면
    방향만 정규화해서 보상»):

        +1  말한 버킷 == 동의도 버킷
        −1  그 외(이웃도 반대도 구분 없이)

    ★«이웃 = 0»은 세이프 하버였다: 항상 `medium`이라고만 말하는 정책은 절대 음수를 받을 수
      없다(기댓값 = P(라벨=medium) ≥ 0) — RL이 아무것도 배우지 않고 그 자리에 주저앉을 수
      있었고, 버킷-붕괴 중단(최빈 점유율 > .90)은 medium 60~80% 정체를 못 잡는다. 그래서
      방향만 남긴다: 맞으면 +1, **아니면 전부 −1**(이웃도 반대와 같은 −1) — «가깝게 틀렸다»는
      보상을 만들지 않는다. `diff_row_metrics`는 여전히 서수 거리(ordinal distance)를 로깅용
      으로 낸다 — 크레딧 함수만 방향-only 로 바뀐다.
    ★라벨은 `loo_agreement`(gold-free, 자기 제외) 를 `agree_bucket` 으로 이산화한 값이다 —
      **gold 도 r_corr 도 읽지 않는다**(tests/test_math_diff.py 가 두 키를 지운 행으로 부른다).
    ★행 식별은 self_idx 우선, 없으면 object identity(`is`) — `==`(값 비교)로 찾으면 값이
      우연히 같은 verl 행(«가짜 쌍둥이»)에서 엉뚱한 형제를 지운다(math_dis 와 같은 규약).
    ★미정의(0, False)는 «보상 0»이 아니라 «항이 없다»다 — 어드밴티지 주입에서 아예 빠진다
      (math_meta._compute_diff_rows → verl_sdc._math_add_diff_meta_advantage).
    """
    f = diff_row_flags(row)
    if (not f["has_meta"] or not f["meta_first"] or f["multi_block"] or f["boxed_in_meta"]
            or not f["bucket_parsed"] or not f["has_why"]):
        return 0.0, False
    stated = row.get("bucket") or parse_difficulty(row.get("meta_raw", ""))
    si = _bucket_idx(stated)
    if si is None:
        return 0.0, False
    if self_idx is None:
        self_idx = next((i for i, r in enumerate(group_rows) if r is row), None)
    if self_idx is None:
        # 호출자가 이미 자기 자신을 뺀 group_rows 를 준 경우 — 뺄 것이 없으므로 전부 형제다.
        # (loo_agreement 는 인덱스를 하나 빼므로, 범위 밖 인덱스로 «아무도 안 빼기»를 만든다.)
        self_idx = len(group_rows)
    li = _bucket_idx(agree_bucket(loo_agreement(self_idx, group_rows)))
    if li is None:
        return 0.0, False
    d = abs(si - li)
    return (1.0 if d == 0 else -1.0), True


def diff_row_metrics(row: Mapping, group_rows: Sequence[Mapping],
                     self_idx: int | None = None) -> dict:
    r"""★**지표 전용**(METRICS ONLY). 보상 경로의 어떤 함수도 이것을 읽지 않는다 — 여기서만
    gold 파생값(`r_corr`)을 본다: 말한 버킷 인덱스 vs **실제** LOO pass rate(형제들의 정오
    평균). «동의도 대리자가 참 난이도를 얼마나 따라가는가»(ledger C1 의 ρ≈.55)를 학습 중에도
    확인하기 위한 창이다. 보상에 들어가면 메타 스팬이 gold 를 받아 «판단»이 아니라 «정답
    복사»를 보상하게 된다.
    """
    if self_idx is None:
        self_idx = next((i for i, r in enumerate(group_rows) if r is row), len(group_rows))
    rest = [r for i, r in enumerate(group_rows) if i != int(self_idx)]
    pass_rate = (sum(int(_cdr._bool01(r.get("r_corr", 0))) for r in rest) / len(rest)) if rest else None
    stated = row.get("bucket") or parse_difficulty(row.get("meta_raw", ""))
    return {
        "stated_idx": _bucket_idx(stated),
        "loo_agreement": loo_agreement(self_idx, group_rows),
        "loo_pass_rate": pass_rate,          # ★gold 파생 — 지표 전용
    }


# ── 텔레메트리 헬퍼 ───────────────────────────────────────────────────────────
def bucket_histogram(rows: Sequence[Mapping]) -> dict:
    """버킷별 행 수(분모 = 버킷이 파싱된 행). 세 키는 항상 존재한다(0 이라도)."""
    out = {b: 0 for b in BUCKETS}
    for r in rows:
        b = r.get("bucket")
        if b in out:
            out[b] += 1
    return out


def bucket_entropy(rows: Sequence[Mapping]) -> float:
    """배치 안 버킷 분포의 엔트로피(nats, 최대 ln3≈1.099). 0 에 붙으면 «전부 한 버킷»
    (붕괴 — ABORT_RULES 의 bucket_max_share 가 그것을 죽인다). 버킷 행이 없으면 NaN."""
    h = bucket_histogram(rows)
    n = sum(h.values())
    if n <= 0:
        return float("nan")
    return -sum((c / n) * math.log(c / n) for c in h.values() if c > 0)


def label_bucket_histogram(rows: Sequence[Mapping]) -> dict:
    """★STATED 버킷이 아니라 **LABEL**(`row["label_bucket"]`, 즉 LOO 동의도를 이산화한 값)
    쪽 히스토그램. 2604.24070 이 보여준 대로 4B 정책의 자기 롤아웃 분포는 이봉(bimodal)일 수
    있고, 라벨이 90% 이상 한 버킷에 몰려 있으면 그 크레딧은 «맞혀도 배울 것이 없는» 상태다 —
    STATED 히스토그램만 보고는 이 붕괴가 보이지 않는다(정책이 다양하게 말해도 라벨 자체가
    한쪽으로 몰려 있을 수 있다). 세 키는 항상 존재한다(0 이라도)."""
    out = {b: 0 for b in BUCKETS}
    for r in rows:
        b = r.get("label_bucket")
        if b in out:
            out[b] += 1
    return out


def label_bucket_entropy(rows: Sequence[Mapping]) -> float:
    """라벨 버킷 분포의 엔트로피(nats, 최대 ln3≈1.099) — bucket_entropy 의 라벨 쪽 짝.
    라벨 행이 없으면 NaN."""
    h = label_bucket_histogram(rows)
    n = sum(h.values())
    if n <= 0:
        return float("nan")
    return -sum((c / n) * math.log(c / n) for c in h.values() if c > 0)


def _rank(xs: Sequence[float]) -> list[float]:
    """평균 순위(동률 처리) — 아래 spearman 전용."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """순위 상관(동률 평균 순위). n<3 이거나 한쪽이 상수면 NaN(«못 쟀다»).
    ★`src/rulers/table.spearman`(numpy 판)과 같은 정의다 — 여기서 다시 쓰는 이유는 학습
      워커가 numpy/pandas 스택(src.rulers)을 끌어오지 않게 하기 위해서다."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and _cdr._finite(a) and _cdr._finite(b)]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den else float("nan")


def stated_vs_agree_spearman(rows: Sequence[Mapping]) -> float:
    """말한 버킷 인덱스(easy0/medium1/hard2) vs **1 − LOO 동의도**의 Spearman —
    부호가 양수면 «어렵다고 말한 행일수록 형제와 덜 일치한다»(판단이 신호를 담는다)."""
    xs, ys = [], []
    for r in rows:
        si = _bucket_idx(r.get("bucket"))
        a = r.get("loo_agreement")
        if si is None or a is None:
            continue
        xs.append(float(si))
        ys.append(1.0 - float(a))
    return spearman(xs, ys)


def stated_vs_truepass_spearman(rows: Sequence[Mapping]) -> float:
    """★지표 전용(gold 파생): 말한 버킷 인덱스 vs **1 − 실제 LOO pass rate**. 보상은 이 값을
    읽지 않는다 — 동의도 대리자가 참 난이도를 얼마나 따라가는지 보는 창이다."""
    xs, ys = [], []
    for r in rows:
        si = _bucket_idx(r.get("bucket"))
        p = r.get("loo_pass_rate")
        if si is None or p is None:
            continue
        xs.append(float(si))
        ys.append(1.0 - float(p))
    return spearman(xs, ys)
