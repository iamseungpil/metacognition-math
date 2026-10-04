r"""trial2 — S3 «2-시도 GRPO + 반성문 크레딧» 의 순수 함수 층 (2026-09-15).

설계 원문은 `docs/DESIGN_S3_trial2_0915.md`. 요약:

문제 x 마다 **문맥을 버리는** 두 시도를 굴린다.
  시도 1  평소 프롬프트 → a1, R1 = grade(a1)
  반성문  R1=0 인 행만 — a1 을 보여 주고 «무엇이 잘못됐나, 숫자·답 금지» 한 문장(≤64 토큰).
          누출 가드(`clean_note`)가 숫자/`\boxed`/a1 의 최종 답이 든 노트를 버린다.
  시도 2  **문맥 폐기** — `fact_prompt(tok, variant, x, note)` (= BLIND_EXTERNAL_NOTE 뒤에
          반성문 한 문장). note="" 면 `.617` 참조 프롬프트와 **바이트 동일**.

크레딧(LaMer arXiv:2512.16848, γ_traj=.6):
  시도-1 스팬   CREDIT: R1 + γ·R2      OUTCOME: R1
  반성문 스팬   CREDIT: γ·R2           OUTCOME: 0
  시도-2 스팬   R2                      R2
중심화는 **단계별·문제별**(Dr.GRPO 평균 빼기, /std 없음) — `row_group_keys` 가 uid 를
`uid#a1`/`uid#note`/`uid#a2` 로 갈라 두면 verl 의 기본 GRPO 중심화가 그대로 그 일을 한다.
비활성 슬롯은 싱글턴 그룹(`uid#dead{i}`)이라 활성 행의 평균을 오염시키지 않고, 보상이 0 이며,
응답 마스크가 전부 0 인 **불활성 더미**다(`verl_sdc._s3_blank_dead_rows` — 손실·KL·엔트로피·
토큰 수 어디에도 닿지 않는다).

★사행(sandbagging) 불가: 시도-1 행이 받는 것은 R1 + γ·R2 뿐이고 γ<1 이다 —
  R1=1 → 1.0, R1=0 → 최대 γ(=0.6). 「틀리는 편이 낫다」가 되는 조합이 없고, R1=0 에
  붙는 보너스는 **오직 γ·R2**(실패 자체에 대한 보너스는 0)다. `assert_no_sandbag` 가
  이 부등식을 코드로 고정한다.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

# ── 팔 명세 쪽 상수 ───────────────────────────────────────────────────────────
# ★재시도 계열 팔의 «항 이름» 단일 진실 원천 — verl_sdc 의 S3 게이트(슬롯 기하 3K·
#   `select_retry_rows`·recredit 훅)가 이 집합으로 팔을 알아본다. 새 2-시도 팔은 여기에만
#   더하면 두 시도 슬롯·ROLLOUT_N=3K 규약을 그대로 물려받는다.
TRIAL2_TERMS = frozenset({"trial2_credit", "trial2_outcome", "trial2_score"})
CREDIT_TERM = "trial2_credit"
OUTCOME_TERM = "trial2_outcome"
SCORE_TERM = "trial2_score"
STAGES = ("a1", "note", "a2")

# ── 프롬프트 재료 ────────────────────────────────────────────────────────────
# ★`math_activation_gate.BLIND_EXTERNAL_NOTE` 의 단일 진실 원천(그 스크립트가 여기서
#   import 한다) — G8 의 `.617` 참조와 바이트 동일해야 하므로 절대 편집 금지.
BLIND_EXTERNAL_NOTE = ("\n\nNote: a previous attempt at this problem was wrong. Solve it "
                       "carefully and give the final answer in \\boxed{}.")

# 반성문을 **얻는** 사전 패스(오답 본문을 보여 주는 유일한 자리 — 시도-2 문맥에는 안 들어간다)
NOTE_ASK = ("In at most 20 words, say what went wrong in this attempt and what to do "
            "differently. Do not state any number, formula value, or final answer.")
GENERIC_NOTE = "the previous attempt made a reasoning mistake; take a different route"
NOTE_MAX_WORDS = 20
# 반성문을 시도-2 프롬프트의 fact 줄 뒤에 붙이는 틀(SWITCH_TMPL 과 같은 «한 문장» 규약)
NOTE_TMPL = "Hint from a previous attempt: {note}."

# ── F1 승자 팔의 재료(0916) ─────────────────────────────────────────────────
# F1(`/hdd_data/seungpil/scratch/eval/reset_content_s1_resum`)에서 리셋 자리의 내용 셋이
# 모두 사실만 고지(fact)보다 구제율이 높았다: effort +.041 · switch +.040 · notx +.069
# (Holm p<.03). 그중 **switch**(자기 방법 라벨을 옮기고 «다른 길로 가라»)는 기증자 라벨
# 대조보다도 +.037(CI 0 제외)로 이겨 «자기 라벨이어야 한다»가 확인됐다. 그래서 S3 의
# NOTE_MODE 에 `switch`/`notx`/`switch_notx` 를 들인다.
# ★아래 네 상수는 **`math_activation_gate` 가 쓰던 바이트 그대로** 이 파일로 옮긴 것이며
#   (그 스크립트가 지금 여기서 import 한다) S3 시도-2 프롬프트가 F1 의 `fact_switch` /
#   `fact_notx` 프롬프트와 **바이트 동일**해야 F1 수치가 그대로 참조가 된다 — 편집 금지.
SWITCH_TMPL = "The previous attempt used {label}. Use a different approach this time."
NOTX_TMPL = "In particular, the answer is not {answer}."
LABEL_ASK = ("In at most 12 words, name the mathematical approach used in this attempt. "
             "Do not state any number or final answer.")
GENERIC_LABEL = "the previous approach"
LABEL_MAX_WORDS = 12

#: NOTE_MODE 의 허용값. 노트 단계(사전 패스)가 **있는** 모드와 없는 모드로 갈린다.
NOTE_MODES = ("self", "none", "switch", "notx", "switch_notx")
#: 사전 패스가 있는 모드 — 그 슬롯이 실재 생성으로 채워지고 CREDIT 의 크레딧 스팬이 된다.
NOTE_STAGE_MODES = ("self", "switch", "switch_notx")


# ── 누출 가드(공유) ──────────────────────────────────────────────────────────
def clean_short_text(raw: str, banned_answer: str = "", max_words: int = 12,
                     fallback: str = "") -> tuple[str, str]:
    r"""짧은 자유 텍스트의 **누출 방지 정제**. 숫자·`\boxed`·금지 답 문자열이 들어간
    텍스트는 **버리고** `fallback` 으로 되돌린다. reason ∈ {ok, empty, digit, boxed, answer}.

    ★`math_activation_gate.clean_label` 의 본체다(그 함수는 이 함수의 얇은 래퍼) — 같은
    가드를 두 벌 두면 한쪽만 고쳐져 «몰래 답을 알려 주는 팔»이 생긴다.
    """
    s = (raw or "").strip()
    if "\\boxed" in s:
        return fallback, "boxed"
    # 첫 비어 있지 않은 줄만 쓴다(모델이 설명을 덧붙이는 경우가 있다)
    s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    s = s.strip().strip('"“”‘’\'').strip()
    s = s.rstrip(".").strip()
    if not s:
        return fallback, "empty"
    s = " ".join(s.split()[:max_words])
    if any(ch.isdigit() for ch in s):
        return fallback, "digit"
    wa = (banned_answer or "").strip()
    if wa and wa.lower() in s.lower():
        return fallback, "answer"
    return s, "ok"


#: 자기 배제(«그 답은 아니다») 표현을 허용할 때 인정하는 부정 표지.
NEGATIONS = ("not", "isn't", "is not", "rule out", "exclude", "wrong", "incorrect")
NEG_WINDOW = 6                  # 리터럴과 부정 표지 사이 허용 거리(단어)
#: LaTeX 수식 표식 — 하나라도 있으면 «습관»이 아니라 풀이 조각이다.
MATH_MARKS = ("\\boxed", "\\frac", "\\sqrt", "\\dfrac", "\\sum", "\\int", "\\pi",
              "$", "\\(", "\\)", "\\[", "\\]", "^", "_{", "=")


def _literals(words: Sequence[str]) -> list[tuple[int, str]]:
    """(위치, 리터럴) — 숫자를 품은 토큰(구두점 제거). «수식/수 리터럴»의 근사."""
    out = []
    for i, w in enumerate(words):
        t = w.strip(".,;:!?()[]{}'\"").strip()
        if any(ch.isdigit() for ch in t):
            out.append((i, t))
    return out


def _norm_ans(s: str) -> str:
    """답 문자열 정규화 — 공백·`$`·후행 마침표·중괄호를 떼고 소문자."""
    t = (s or "").strip().strip("$").strip()
    t = t.replace(" ", "").replace("\\left", "").replace("\\right", "")
    return t.strip("{}").rstrip(".").lower()


def clean_habit_note(raw: str, wrong_answer: str = "") -> tuple[str, str]:
    r"""**습관 노트** 전용 가드(자기 배제 1회 허용판). 규칙:

      ① `\boxed` 나 어떤 LaTeX 수식 표식이 있으면 거절 → `GENERIC_NOTE`
      ② 첫 비어 있지 않은 줄만, ≤ `NOTE_MAX_WORDS` 단어
      ③ 숫자를 품은 리터럴은 **최대 하나**, 그리고 그것이 자기 오답(`wrong_answer`)과
         정규화 후 **정확히 같고** 부정 표지(`NEGATIONS`)가 `NEG_WINDOW` 단어 안에 있을
         때만 허용한다. 그 밖의 숫자는 전부 거절.
    reason ∈ {ok, ok_exclusion, empty, boxed, math, digit, two_numbers, answer}.

    ★왜 «하나만, 자기 오답만»인가: 이 노트는 **습관**(무엇을 하라)만 옮기게 하려는 것이다.
      형제 분포를 보여 주는 팔(habit_sib)에서 모델이 다른 후보 Y 를 적으면 그건 습관이
      아니라 **답 힌트**이므로 반드시 거절돼야 한다(그 팔의 해석이 무너진다). 자기 오답 X 를
      «그건 아니다»로 적는 것만 허용하는 이유는 검증기가 이미 R1=0 이라 말했기 때문이다.
    """
    s = (raw or "").strip()
    if "\\boxed" in s:
        return GENERIC_NOTE, "boxed"
    s = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    s = s.strip().strip('"“”‘’\'').strip().rstrip(".").strip()
    if not s:
        return GENERIC_NOTE, "empty"
    if any(m in s for m in MATH_MARKS):
        return GENERIC_NOTE, "math"
    s = " ".join(s.split()[:NOTE_MAX_WORDS])
    words = s.split()
    lits = _literals(words)
    if len(lits) > 1:
        return GENERIC_NOTE, "two_numbers"
    if lits:
        pos, lit = lits[0]
        if _norm_ans(lit) != _norm_ans(wrong_answer) or not _norm_ans(wrong_answer):
            return GENERIC_NOTE, "digit"
        low = s.lower()
        near = " ".join(words[max(0, pos - NEG_WINDOW):pos + NEG_WINDOW + 1]).lower()
        if not any(n in near for n in NEGATIONS) and not any(n in low for n in NEGATIONS):
            return GENERIC_NOTE, "digit"
        return s, "ok_exclusion"
    return s, "ok"


def clean_note(raw: str, wrong_answer: str = "",
               allow_exclusion: bool = False) -> tuple[str, str]:
    """반성문용 누출 가드 — `clean_short_text` 에 S3 상수를 물린 것.
    `allow_exclusion=True` 면 **습관 노트** 규칙(`clean_habit_note`)을 쓴다 — 자기 오답을
    «그 답은 아니다»로 한 번 적는 것만 허용한다."""
    if allow_exclusion:
        return clean_habit_note(raw, wrong_answer)
    return clean_short_text(raw, wrong_answer, max_words=NOTE_MAX_WORDS,
                           fallback=GENERIC_NOTE)


def clean_label(raw: str, wrong_answer: str = "") -> tuple[str, str]:
    r"""방법 라벨용 누출 가드 — `math_activation_gate.clean_label` 의 본체와 같은 호출
    (≤12 단어, 숫자/`\boxed`/오답 문자열이 들어가면 `GENERIC_LABEL` 로 되돌린다)."""
    return clean_short_text(raw, wrong_answer, max_words=LABEL_MAX_WORDS,
                           fallback=GENERIC_LABEL)


def clean_stage_text(mode: str, raw: str, wrong_answer: str = "") -> tuple[str, str]:
    """노트 단계 출력의 정제 — 모드에 맞는 가드를 고른다(self=반성문, switch*=방법 라벨).
    사전 패스가 없는 모드(none/notx)는 정제할 것이 없다 → ("", "no_note_stage")."""
    mode = check_note_mode(mode)
    if mode == "self":
        return clean_note(raw, wrong_answer)
    if mode in ("switch", "switch_notx"):
        return clean_label(raw, wrong_answer)
    return "", "no_note_stage"


# ── 프롬프트 조립 ────────────────────────────────────────────────────────────
def fact_prompt(tok, variant: str, problem: str, extra: str = "") -> str:
    """★공통 조립기 — user 턴 끝에 `BLIND_EXTERNAL_NOTE`(fact 줄) + (있으면) 공백 하나 +
    `extra`. `extra=""` 이면 G8 의 `blind_external` 프롬프트와 **바이트 동일**이다."""
    from src.metacot.math_meta_prompt import (  # noqa: PLC0415
        build_math_prompt, render_chat_messages,
    )
    msgs = build_math_prompt(problem, variant)
    tail = BLIND_EXTERNAL_NOTE + ((" " + extra) if extra else "")
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + tail}
    return render_chat_messages(tok, msgs)


def attempt2_extra(mode: str = "self", *, note: str = "", label: str = "",
                   notx: str = "") -> str:
    r"""NOTE_MODE → fact 줄 뒤에 붙는 `extra` 문자열. F1 프롬프트와의 대응:

      none         ""                                   → F1 `fact`(=blind_external)
      self         NOTE_TMPL(반성문)                      → S3 원래 팔
      switch       SWITCH_TMPL(자기 방법 라벨)             → F1 `fact_switch`
      notx         NOTX_TMPL(X = a1 의 `\boxed` 값)        → F1 `fact_notx`
      switch_notx  위 둘을 **그 순서로 공백 하나**로 이은 것 → F1 에 없던 합성 팔

    ★`notx` 의 정당성: 시도-2 는 채점기가 이미 **R1=0 이라고 말한 뒤**에만 돈다 —
      즉 «a1 의 답은 틀렸다»는 정책이 자기 롤아웃과 검증기만으로 아는 사실이고, gold 를
      노출하지 않는다(F1 의 `fact_notx` 가 훈련 컴포넌트 후보가 아니었던 이유는 그 게이트가
      정답 행에서도 X 를 고르려고 **gold 를 봐야** 했기 때문이다 — 여기서는 오답 행만이다).
    ★a1 에 `\boxed` 가 없으면 X 가 없으므로 notx 절을 **건너뛴다**(notx → fact 전용,
      switch_notx → switch 만). 조용히 빈 문자열을 넣지 않고 절 자체를 뺀다.
    """
    mode = check_note_mode(mode)
    if mode == "none":
        return ""
    if mode == "self":
        n = (note or "").strip()
        return NOTE_TMPL.format(note=n) if n else ""
    parts: list[str] = []
    if mode in ("switch", "switch_notx"):
        parts.append(SWITCH_TMPL.format(label=(label or "").strip() or GENERIC_LABEL))
    if mode in ("notx", "switch_notx"):
        x = (notx or "").strip()
        if x:
            parts.append(NOTX_TMPL.format(answer=x))
    return " ".join(parts)


def attempt2_prompt(tok, variant: str, problem: str, note: str = "", *,
                    mode: str = "self", label: str = "", notx: str = "") -> str:
    """시도-2 프롬프트. extra 가 비면 `.617` 참조와 바이트 동일(NOTE_MODE=none 팔).
    ★오답 **본문**은 어떤 경우에도 여기에 들어가지 않는다(G8: 본문 제시 −10.4pp).
    ★`mode` 가 switch/notx 계열이면 `fact_prompt(..., extra)` 로 F1 의 `fact_switch` /
    `fact_notx` 프롬프트와 **바이트 동일**한 문자열이 나온다(테스트가 고정한다)."""
    return fact_prompt(tok, variant, problem,
                       attempt2_extra(mode, note=note, label=label, notx=notx))


NOTE_CTX_ELLIPSIS = "… "


def note_ctx_max_tokens() -> int:
    """`NOTE_CTX_MAX_TOKENS` (기본 2048) — 반성문 요청 프롬프트에 **끼워 넣는** 시도-1 본문의
    토큰 상한. 시도-1 응답은 `rollout.response_length`(=4096)까지 길 수 있는데, 그 본문이
    통째로 들어가면 반성문 호출의 프롬프트가 `rollout.prompt_length`(=1024)를 훨씬 넘고,
    verl 의 agent-loop 은 **cf 경로의 프롬프트를 자르지 않고 그 길이로 패딩**하므로
    (agent_loop.py:444-460 의 cap 은 chat-템플릿 경로 전용) 그 호출의 판 폭이 시도-1 호출과
    어긋난다. 폭은 `_s3_harmonize` 가 맞춰 주지만 **폭 = 학습 배치 메모리**이므로 여기서 상한을
    둔다. **뒤쪽을 남긴다** — 틀린 최종 답과 마무리 논증이 꼬리에 있고, 반성문은 그것을 본다.

    ★기본 1536 의 산수(0916): E-131 가드는 배치 프롬프트 폭 == `data.max_prompt_length` 를
    요구하고 `run_math_arm.sh` 가 M_TRIAL2_* 에 주는 값은 **3072** 이다. 반성문 프롬프트 =
    chat 템플릿·시스템 ~40 + 문제 ≤ ~900 + 이 꼬리 1536 + NOTE_ASK ~40 ≈ 2,520 ≤ 3072
    (여유 ~550). 이 상한을 올리려면 MAX_PROMPT 도 같이 올려야 한다 — 넘치면
    `_s3_harmonize` 가 «반성문 프롬프트가 config 폭을 넘었다»고 즉사한다."""
    return int(os.environ.get("NOTE_CTX_MAX_TOKENS", "1536"))


def cap_attempt1_text(tok, attempt1_text: str, max_tokens: int | None = None) -> str:
    """시도-1 본문의 **꼬리** `max_tokens` 토큰만 남긴다(자르면 앞에 `…` 를 붙인다).
    토크나이저에 `encode` 가 없으면(테스트용 최소 토크나이저) 토큰≈4자 근사로 자른다."""
    txt = attempt1_text or ""
    n = int(note_ctx_max_tokens() if max_tokens is None else max_tokens)
    if n <= 0 or not txt:
        return txt
    enc = getattr(tok, "encode", None)
    if enc is None:
        return txt if len(txt) <= 4 * n else NOTE_CTX_ELLIPSIS + txt[-4 * n:]
    ids = enc(txt, add_special_tokens=False)
    if len(ids) <= n:
        return txt
    return NOTE_CTX_ELLIPSIS + tok.decode(ids[-n:])


def note_ask_prompt(tok, variant: str, problem: str, attempt1_text: str,
                    ask: str = NOTE_ASK) -> str:
    """반성문을 얻는 사전 패스 — 오답 본문을 assistant 턴으로 담고 새 user 턴(`ask`).
    `ask` 기본값은 NOTE_ASK(반성문) — switch 계열은 LABEL_ASK 를 넣어 F1 사전 패스와
    같은 질문을 쓴다(`note_stage_prompt` 가 모드에 따라 고른다).
    `math_activation_gate.external_prompt` 와 같은 sentinel 규약(본문 끝 공백 보존).
    본문은 `cap_attempt1_text` 로 꼬리 `NOTE_CTX_MAX_TOKENS` 토큰만 싣는다."""
    from src.metacot.math_meta_prompt import (  # noqa: PLC0415
        build_math_prompt, render_chat_messages,
    )
    sentinel = "<<<TRIAL2_ATTEMPT1>>>"
    msgs = build_math_prompt(problem, variant) + [
        {"role": "assistant", "content": sentinel},
        {"role": "user", "content": ask},
    ]
    rendered = render_chat_messages(tok, msgs)
    if rendered.count(sentinel) != 1:
        raise RuntimeError(f"[TRIAL2] chat 템플릿에서 sentinel 을 {rendered.count(sentinel)}번 "
                           "찾았다 — 1번이어야 한다(템플릿이 assistant 본문을 변형한다).")
    head, tail = rendered.split(sentinel)
    return head + cap_attempt1_text(tok, attempt1_text) + tail


def note_stage_on(mode: str) -> bool:
    """이 모드가 노트 단계(사전 패스)를 쓰는가. none/notx 는 **쓰지 않는다** — 그 슬롯은
    죽은 자리(dead slot)로 남고 CREDIT 팔의 크레딧 스팬도 없어진다."""
    return check_note_mode(mode) in NOTE_STAGE_MODES


def note_stage_prompt(tok, variant: str, problem: str, attempt1_text: str,
                      mode: str = "self") -> str | None:
    """노트 단계의 프롬프트 — self 는 NOTE_ASK, switch/switch_notx 는 **LABEL_ASK**
    (F1 사전 패스와 같은 질문). 사전 패스가 없는 모드면 None.
    ★사전 패스는 시도-2 문맥이 아니므로 F1 의 `label_prompt` 와 바이트 동일할 필요가 없다
      (여기서는 본문을 `cap_attempt1_text` 로 꼬리만 싣는다 — 판 폭 상한 때문)."""
    mode = check_note_mode(mode)
    if not note_stage_on(mode):
        return None
    ask = NOTE_ASK if mode == "self" else LABEL_ASK
    return note_ask_prompt(tok, variant, problem, attempt1_text, ask)


# ── 손잡이(환경변수) ─────────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


def gamma_traj() -> float:
    """GAMMA_TRAJ (기본 0.6, LaMer γ_traj). 0 ≤ γ < 1 이어야 한다 — γ≥1 이면 «시도 1 을
    틀리는 편이 낫다»가 될 수 있어 사행 방지 부등식이 깨진다."""
    g = _env_float("GAMMA_TRAJ", 0.6)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"[TRIAL2] GAMMA_TRAJ={g} 는 [0,1) 밖이다 — 사행 방지 부등식이 깨진다.")
    return g


def score_alpha_pos() -> float:
    """SCORE_ALPHA_POS (기본 1.0) — `trial2_score` 의 **구제** 보너스 계수 α⁺.
    음수는 즉사(구제에 벌을 주는 항이 되어 팔의 뜻이 뒤집힌다)."""
    a = _env_float("SCORE_ALPHA_POS", 1.0)
    if a < 0.0:
        raise ValueError(f"[TRIAL2] SCORE_ALPHA_POS={a} < 0 — 구제에 벌을 주게 된다.")
    return a


def score_alpha_neg() -> float:
    """SCORE_ALPHA_NEG (기본 2.0 = 2α⁺) — `trial2_score` 의 **파괴** 벌 계수 α⁻.
    음수는 즉사(파괴에 상을 주게 된다)."""
    a = _env_float("SCORE_ALPHA_NEG", 2.0)
    if a < 0.0:
        raise ValueError(f"[TRIAL2] SCORE_ALPHA_NEG={a} < 0 — 파괴에 상을 주게 된다.")
    return a


def note_max_tokens() -> int:
    return int(os.environ.get("NOTE_MAX_TOKENS", "64"))


def a1_max_tokens(resp_len: int) -> int:
    """시도-1 의 **생성** 상한. `A1_RESP_LEN`(미설정이면 `RESP_LEN`, 그것도 없으면 판 폭
    `resp_len`)을 쓴다.

    ★왜 필요한가(0916). A2_RESP_LEN 을 6144 로 올리면 `data.max_response_length` 도 같이
    올라간다 — verl 의 agent-loop 은 **판 폭 = `rollout.response_length`** 로 패딩·절단하므로
    (agent_loop.py:775-789, single_turn_agent_loop.py 의 `[: self.response_length]`) 그 폭을
    올리지 않으면 시도 2 는 절대 4096 을 넘을 수 없다. 그런데 폭만 올리면 **시도 1 도** 덩달아
    6144 를 쓴다(M_G0 기준선과 예산이 어긋난다). 그래서 폭은 A2 로 올리고, 시도 1 은 이 값으로
    **호출 단위 max_tokens** 를 걸어 4096 에서 멈춘다(판은 넓고 생성만 좁다 — 남는 자리는
    오른쪽 패딩이라 채점·마스크에 영향이 없다)."""
    v = os.environ.get("A1_RESP_LEN") or os.environ.get("RESP_LEN")
    return int(v) if v else int(resp_len)


def a2_resp_len(resp_len: int) -> int:
    """시도-2 의 생성 상한. `A2_RESP_LEN`(미설정이면 판 폭 `resp_len`).

    ★0916 관측: 오프라인 재시도 중앙값 4,675 토큰 — 4096 에서 57% 가 잘리고 `\\boxed` 가
    없어져 R2 가 «판단» 이 아니라 «예산» 으로 0 이 된다. 판 폭(`data.max_response_length`)이
    이 값 이상이어야 실제로 늘어난다 — `run_math_arm.sh` 가 MAX_RESP 를 그렇게 잡는다."""
    v = os.environ.get("A2_RESP_LEN")
    n = int(v) if v else int(resp_len)
    if n > int(resp_len):
        raise ValueError(
            f"[TRIAL2] A2_RESP_LEN={n} 이 판 폭 data.max_response_length={resp_len} 보다 크다 — "
            "agent-loop 이 응답을 판 폭으로 자르므로 조용히 무시된다. run_math_arm.sh 의 "
            "MAX_RESP(= max(RESP_LEN, A2_RESP_LEN))를 올려라.")
    return n


def check_note_mode(mode: str) -> str:
    """NOTE_MODE 값 검증(소문자 정규화). 모르는 값은 즉사 — 조용히 self 로 흐르면 팔이
    바뀐 줄 모르고 결과를 본다."""
    m = (mode or "self").strip().lower()
    if m not in NOTE_MODES:
        raise ValueError(f"[TRIAL2] NOTE_MODE={mode!r} 은 {'|'.join(NOTE_MODES)} 중 "
                         "하나여야 한다.")
    return m


def note_mode() -> str:
    """NOTE_MODE ∈ {self, none, switch, notx, switch_notx}.
      self        자기 반성문 한 문장(원래 S3 팔)
      none        반성문 없는 사실만 재시도(= `.617` 참조 팔)
      switch      자기 방법 라벨 + «다른 길로» (F1 `fact_switch` 와 바이트 동일)
      notx        «답은 X 가 아니다»(X = a1 의 `\\boxed`; F1 `fact_notx` 와 바이트 동일)
      switch_notx 위 둘을 그 순서로 이은 합성 팔

    ★크레딧 스팬: CREDIT 팔이 크레딧을 얹는 곳은 여전히 **노트 토큰**이다 — switch 계열
      에서는 그것이 «방법 라벨» 토큰이다. `notx`/`none` 은 노트 단계가 아예 없어(죽은 자리)
      CREDIT 과 OUTCOME 이 시도-1 의 γ·R2 항만 빼고 같아진다(`row_reward` 참조)."""
    return check_note_mode(os.environ.get("NOTE_MODE") or "self")


def retry_only_wrong() -> bool:
    """RETRY_ONLY_WRONG (기본 1) — 시도 2 는 오답 행에만. 0 이면 전 행에 시도 2 를 굴린다
    (정답 행의 시도 2 는 «거짓 경보» 진단용이며 기본값이 아니다)."""
    return (os.environ.get("RETRY_ONLY_WRONG", "1") or "1").strip() not in ("0", "false", "False")


# ── 재시도 게이트(0916) ──────────────────────────────────────────────────────
#: RETRY_GATE 의 허용값.
RETRY_GATES = ("wrong", "agree", "all")
#: `agree` 게이트가 기본으로 재시도하는 합의 상태 — ALL_SAME 을 뺀 전부.
DEFAULT_RETRY_GATE_STATES = ("DOMINANT", "SPLIT", "SCATTER", "NOANS")
#: `agreement_state` 가 내는 상태 이름(math_activation_gate.AGREE_STATES 와 같은 집합).
AGREE_STATES = ("ALL_SAME", "DOMINANT", "SPLIT", "SCATTER", "NOANS")


def agreement_state(answers: Sequence[str], *, k: int | None = None) -> dict:
    """★gold 없는 **합의 상태** — 형제 boxed 답들의 수학적 동치 군집만 본다.

    `agree_credit.py:state_of` 와 같은 정의(정오는 쓰지 않는다): 최대 군집 크기 `top` 에
    대해 top == K → ALL_SAME, ≥5 → DOMINANT, ≥3 → SPLIT, 그 밖 → SCATTER, 답이 하나도
    없으면 NOANS. **분모 K 는 그룹의 롤아웃 수**이고 무응답도 K 에 센다(그래서 8개 중
    하나가 무응답이고 일곱이 같으면 ALL_SAME 이 아니라 DOMINANT 다).
    반환 {state, top, n_clusters, dom_frac, dominant_answer, n_answered, k}.

    ★`math_activation_gate.agreement_state` 의 본체다(그 스크립트가 여기서 import 한다) —
      S3 의 `agree` 재시도 게이트가 **추론 프로토콜과 같은 상태 정의**를 써야 하므로 단일
      진실 원천을 학습 쪽(Ray 워커가 import 할 수 있는 곳)에 둔다. 임계값을 복사하지 않는다.
    """
    from src.training.math_meta import answers_equivalent  # noqa: PLC0415

    xs = [(a or "").strip() for a in answers]
    kk = int(k if k is not None else len(xs))
    ans = [a for a in xs if a]
    if not ans or kk <= 0:
        return {"state": "NOANS", "top": 0, "n_clusters": 0, "dom_frac": 0.0,
                "dominant_answer": "", "n_answered": len(ans), "k": kk}
    clusters: list[list] = []              # [대표, 개수, 첫 등장]
    for i, a in enumerate(ans):
        for cl in clusters:
            if answers_equivalent(cl[0], a):
                cl[1] += 1
                break
        else:
            clusters.append([a, 1, i])
    clusters.sort(key=lambda c: (-c[1], c[2]))
    top = clusters[0][1]
    state = ("ALL_SAME" if top >= kk else "DOMINANT" if top >= 5
             else "SPLIT" if top >= 3 else "SCATTER")
    return {"state": state, "top": top, "n_clusters": len(clusters), "dom_frac": top / kk,
            "dominant_answer": clusters[0][0], "n_answered": len(ans), "k": kk}


def check_retry_gate(gate: str) -> str:
    """RETRY_GATE 값 검증(소문자 정규화). 모르는 값은 즉사."""
    g = (gate or "wrong").strip().lower()
    if g not in RETRY_GATES:
        raise ValueError(f"[TRIAL2] RETRY_GATE={gate!r} 은 {'|'.join(RETRY_GATES)} 중 "
                         "하나여야 한다.")
    return g


def retry_gate() -> str:
    """RETRY_GATE ∈ {wrong, agree, all}. **미설정이면 현행 유지** — `RETRY_ONLY_WRONG` 의
    값에 따라 `wrong`(기본) 또는 `all` 이다(그 손잡이의 옛 의미를 한 글자도 안 바꾼다).

      wrong  시도-1 이 **gold 로 채점해** 틀린 행만 재시도(현행 기본).
      agree  같은 문제(uid)의 K 개 시도-1 답으로 **gold 없이** 합의 상태를 매기고,
             그 상태가 `RETRY_GATE_STATES`(기본 ALL_SAME 을 뺀 전부) 안이면 그 문제의
             **모든** a1 행을 정오와 무관하게 재시도 — 추론 때 켜지는 규칙과 같은 규칙이다.
      all    전 행 재시도.
    """
    v = os.environ.get("RETRY_GATE")
    if v is None or not str(v).strip():
        return "wrong" if retry_only_wrong() else "all"
    return check_retry_gate(v)


def retry_gate_states() -> tuple[str, ...]:
    """RETRY_GATE_STATES — `agree` 게이트가 재시도로 여는 상태 목록(쉼표 구분).
    기본은 `DEFAULT_RETRY_GATE_STATES`(= ALL_SAME 을 뺀 전부). 모르는 이름은 즉사."""
    v = os.environ.get("RETRY_GATE_STATES")
    if v is None or not str(v).strip():
        return DEFAULT_RETRY_GATE_STATES
    out = tuple(s.strip().upper() for s in str(v).split(",") if s.strip())
    bad = [s for s in out if s not in AGREE_STATES]
    if bad:
        raise ValueError(f"[TRIAL2] RETRY_GATE_STATES 의 {bad} 는 "
                         f"{'|'.join(AGREE_STATES)} 중에 없다.")
    return out


def group_states(answers: Sequence[str], k: int) -> list[str]:
    """시도-1 행별 **그룹 합의 상태** — 행 p·K+j 는 문제 p 의 K 개 답으로 매긴 상태를 받는다
    (`agreement_state(..., k=K)["state"]`, gold 를 보지 않는다)."""
    k = int(k)
    n = len(answers)
    if k < 1 or n % k != 0:
        raise ValueError(f"[TRIAL2] 답 {n}개가 K={k} 의 배수가 아니다 — 슬롯 규약이 깨졌다.")
    out: list[str] = []
    for p in range(n // k):
        st = agreement_state(answers[p * k:(p + 1) * k], k=k)["state"]
        out.extend([st] * k)
    return out


def select_retry_rows(r1: Sequence[float], k: int, *, gate: str = "wrong",
                      states: Sequence[str] | None = None,
                      answers: Sequence[str] | None = None,
                      ) -> tuple[list[int], dict]:
    """시도 2 를 **실제로 굴릴** 시도-1 행 인덱스(오름차순) + 단계 텔레메트리.

    `gate="wrong"` 은 gold 채점(`r1`)을, `"agree"` 는 `answers`(a1 의 boxed 답, gold 없음)만
    본다. `"all"` 은 전 행. 텔레메트리의 `retried_a1_correct_frac` 는 **로그 전용**으로
    gold 를 쓴다(«정답인데도 다시 풀었다»의 비율 — 예산과 거짓 경보를 읽는 자).
    """
    gate = check_retry_gate(gate)
    n = len(r1)
    row_states = ["" for _ in range(n)]
    if answers is not None:
        if len(answers) != n:
            raise ValueError("[TRIAL2] 길이 불일치: r1/answers")
        row_states = group_states(answers, k)
    elif gate == "agree":
        raise ValueError("[TRIAL2] RETRY_GATE=agree 는 시도-1 답(answers)이 있어야 한다.")
    allowed = tuple(states) if states is not None else retry_gate_states()
    rows: list[int] = []
    for i in range(n):
        if gate == "all":
            on = True
        elif gate == "wrong":
            on = float(r1[i]) <= 0.0
        else:
            on = row_states[i] in allowed
        if on:
            rows.append(i)
    by_state: dict[str, float] = {}
    for i in rows:
        key = row_states[i] or "NA"
        by_state[key] = by_state.get(key, 0.0) + 1.0
    n_corr = sum(1 for i in rows if float(r1[i]) > 0.0)
    tel = {
        "gate": gate,
        "states": ",".join(allowed) if gate == "agree" else "",
        "n_rows": float(n),
        "n_retried": float(len(rows)),
        "retried_by_state": by_state,
        "retried_a1_correct_frac": (n_corr / len(rows)) if rows else 0.0,
    }
    return rows, tel


def attempt1_kl_coef(trainer_default: float) -> float:
    """ATTEMPT1_KL_COEF — 시도-1 토큰에만 거는 참조 KL 계수. 미설정이면 트레이너의 기존
    KL 설정을 그대로 쓴다(«기본은 현행 유지» 규약)."""
    v = os.environ.get("ATTEMPT1_KL_COEF")
    return float(trainer_default) if v is None or v == "" else float(v)


# ── 슬롯 기하 ────────────────────────────────────────────────────────────────
def rollout_n(k: int) -> int:
    """엔진에 요구할 rollout.n — 한 문제당 3K 행(시도1/반성문/시도2 슬롯)."""
    if int(k) < 1:
        raise ValueError(f"[TRIAL2] k={k} < 1")
    return 3 * int(k)


def slot_layout(n_problems: int, k: int) -> dict[str, list[int]]:
    """전역 행 인덱스 배치. 문제 p 의 블록은 [p·3K, (p+1)·3K) 이고 그 안에서
    0…K−1 = 시도 1, K…2K−1 = 반성문, 2K…3K−1 = 시도 2 (슬롯 K+j / 2K+j 는 샘플 j 의 것)."""
    k = int(k)
    out: dict[str, list[int]] = {"a1": [], "note": [], "a2": []}
    for p in range(int(n_problems)):
        base = p * 3 * k
        out["a1"].extend(range(base, base + k))
        out["note"].extend(range(base + k, base + 2 * k))
        out["a2"].extend(range(base + 2 * k, base + 3 * k))
    return out


def stage_of(row: int, k: int) -> str:
    """행 인덱스 → 단계 이름."""
    return STAGES[(int(row) % (3 * int(k))) // int(k)]


def trial_index(row: int, k: int) -> int:
    """행 인덱스 → 그 문제 안에서 몇 번째 샘플(trial_id 의 지역 부분)인가."""
    return int(row) % int(k)


def trial_id(row: int, k: int, uid: str) -> str:
    """세 행을 잇는 연결 키 — `{uid}#t{j}`."""
    return f"{uid}#t{trial_index(row, k)}"


def reassembly_plan(n_problems: int, k: int, retried: Sequence[int],
                    note_on: bool = True) -> tuple[list[int], list[str], list[int]]:
    """세 단계 출력을 이어 붙인 텐서(블록 순서 a1|note|a2, 각 K·P 행)를 **슬롯 순서**로
    되돌리는 순열 + 행별 단계·활성 표식.

    `retried` 는 시도 2 가 **실제로 생성된** 시도-1 행 인덱스(p·K+j). `note_on=False`
    (NOTE_MODE=none)면 반성문 슬롯은 아무것도 생성되지 않았으므로 **항상 비활성**이다 —
    그 자리에 든 복제를 활성으로 두면 크레딧이 유령 행에 실린다.
    """
    k, p_n = int(k), int(n_problems)
    kp = k * p_n
    got = {int(i) for i in retried}
    perm: list[int] = []
    stages: list[str] = []
    active: list[int] = []
    for p in range(p_n):
        for si, st in enumerate(STAGES):
            for j in range(k):
                perm.append(si * kp + p * k + j)
                stages.append(st)
                on = (p * k + j) in got
                active.append(1 if st == "a1"
                              else int(on and (note_on if st == "note" else True)))
    return perm, stages, active


def sibling_rows(row: int, k: int) -> dict[str, int]:
    """같은 trial 의 세 행 인덱스."""
    k = int(k)
    base = (int(row) // (3 * k)) * 3 * k + trial_index(row, k)
    return {"a1": base, "note": base + k, "a2": base + 2 * k}


# ── 엔진 배선 ────────────────────────────────────────────────────────────────
#: `cf_prefix_agent` 를 Ray 롤아웃 워커에 등록하는 yaml (레포 루트 기준 상대 경로).
CF_AGENT_CONFIG_REL = "configs/cf_prefix_agent.yaml"


def cf_agent_config_path() -> str:
    """`configs/cf_prefix_agent.yaml` 의 절대 경로. 없으면 즉사."""
    import os.path as _p

    path = _p.normpath(_p.join(_p.dirname(_p.dirname(_p.dirname(
        _p.abspath(__file__)))), CF_AGENT_CONFIG_REL))
    if not _p.isfile(path):
        raise FileNotFoundError(
            f"[TRIAL2] {CF_AGENT_CONFIG_REL} 가 없다 ({path}) — 이 파일 없이는 Ray "
            "롤아웃 워커가 cf_prefix_agent 를 모르고 «Agent loop cf_prefix_agent not "
            "registered» AssertionError 로 죽는다.")
    return path


def ensure_cf_agent_registered(rollout_cfg) -> str:
    """시도-2·반성문 생성이 타는 `cf_prefix_agent` 를 **워커 프로세스**에 등록시킨다.

    `@register` 데코레이터는 드라이버의 레지스트리만 채운다 — 실제 인스턴스화는 Ray actor
    `AgentLoopWorker` 안에서 일어나므로 `rollout.agent.agent_loop_config_path` 가
    비어 있으면 agent_loop.py:692 의 assert 로 잡 전체가 죽는다(0915 S3 스모크 실패).
    이미 값이 있으면 **건드리지 않는다**(다른 팔/런처 설정 우선).
    반환값은 최종 경로."""
    agent_cfg = getattr(rollout_cfg, "agent", None)
    if agent_cfg is None:
        raise RuntimeError("[TRIAL2] rollout.agent 블록이 없다 — verl 버전 확인 필요.")
    cur = getattr(agent_cfg, "agent_loop_config_path", None)
    if cur:
        return str(cur)
    path = cf_agent_config_path()
    agent_cfg.agent_loop_config_path = path
    return path


# ── 보상·그룹 ────────────────────────────────────────────────────────────────
def assert_no_sandbag(gamma: float) -> None:
    """시도-1 행의 보상 R1 + γ·R2 가 «틀리는 편이 낫다»를 만들 수 없음을 고정한다.
    최댓값(R1=0) = γ < 1 = 최솟값(R1=1, R2=0). γ≥1 이면 즉사."""
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError(f"[TRIAL2] γ_traj={gamma} — 사행 방지 부등식(γ<1)이 깨진다.")


def row_reward(term: str, stage: str, active: int, r1: float, r2: float,
               gamma: float) -> float:
    """행 하나의 **중심화 전** 스칼라 보상.

    CREDIT:  a1 = R1 + γ·R2 · note = γ·R2 · a2 = R2
    OUTCOME: a1 = R1        · note = 0     · a2 = R2
    SCORE:   a1 = R1        · note = 0     · a2 = R2 + α⁺·max(R2−R1,0) − α⁻·max(R1−R2,0)
    비활성 행(active=0)은 어느 팔에서도 0 이다.

    ★SCORE 의 설계 근거(docs/PLAN_h2_twoturn_0921.md §1·§2): 정책은 문제 안에서 자기 답의
      정오를 구분하지 못한다(문제 내 AUC .51~.55; 결정 발화 SFT 는 발화율 .834 를 얻고 풀이가
      .046 으로 붕괴). 그래서 X 와 Y 중 «고르는» 커밋 분류기를 학습시키지 않고 시도 2 를
      무조건 취한다. 비대칭 벌(α⁻=2α⁺)이 가르치는 것은 선택이 아니라 «맞던 것을 깨뜨리지
      않기»다. 실측 표적: 무조건 재시도의 구제 p_fix .377 · 파괴 .078 · 순 −1.56pp 이고,
      같은 p_fix 에서 파괴만 .04 로 내리면 +1.75pp 다.
    """
    if term not in TRIAL2_TERMS:
        raise ValueError(f"[TRIAL2] 미지의 항 {term!r}")
    if not int(active):
        return 0.0
    g = float(gamma)
    if stage == "a1":
        return float(r1) + (g * float(r2) if term == CREDIT_TERM else 0.0)
    if stage == "note":
        return g * float(r2) if term == CREDIT_TERM else 0.0
    if stage == "a2":
        if term == SCORE_TERM:
            d = float(r2) - float(r1)
            return float(r2) + score_alpha_pos() * max(d, 0.0) - score_alpha_neg() * max(-d, 0.0)
        return float(r2)
    raise ValueError(f"[TRIAL2] 미지의 단계 {stage!r}")


def build_row_rewards(term: str, stages: Sequence[str], active: Sequence[int],
                      r1: Sequence[float], r2: Sequence[float],
                      gamma: float) -> list[float]:
    """행별 보상 벡터. r1/r2 는 **그 행이 속한 trial** 의 값이어야 한다(세 행 모두 같은 값)."""
    assert_no_sandbag(gamma)
    n = len(stages)
    if not (len(active) == len(r1) == len(r2) == n):
        raise ValueError("[TRIAL2] 길이 불일치: stages/active/r1/r2")
    return [row_reward(term, stages[i], active[i], r1[i], r2[i], gamma) for i in range(n)]


def row_group_keys(uids: Sequence[str], stages: Sequence[str],
                   active: Sequence[int]) -> list[str]:
    """중심화 그룹 키 — `uid#a1` / `uid#note` / `uid#a2`. 비활성 행은 `uid#dead{i}`
    싱글턴이라 **활성 행의 그룹 평균을 오염시키지 않는다**.

    ★비활성 행의 어드밴티지가 0 인 이유는 «싱글턴이라 중심화된다»가 아니다 — verl 0.9 의
      `compute_grpo_outcome_advantage`(core_algos.py:315)는 싱글턴 그룹에 평균 0·std 1 을
      쓴다(자기 자신을 빼지 않는다). 0 인 진짜 이유는 두 겹이다: ①`row_reward` 가
      active=0 에 0 을 돌려주고 ②그 행의 `response_mask` 가 전부 0 이라 어떤 값이 실려도
      손실·KL·엔트로피에 닿지 않는다(`_s3_blank_dead_rows`)."""
    n = len(uids)
    if not (len(stages) == len(active) == n):
        raise ValueError("[TRIAL2] 길이 불일치: uids/stages/active")
    return [f"{uids[i]}#{stages[i]}" if int(active[i]) else f"{uids[i]}#dead{i}"
            for i in range(n)]


def trial2_advantages(term: str, uids: Sequence[str], stages: Sequence[str],
                      active: Sequence[int], r1: Sequence[float], r2: Sequence[float],
                      gamma: float) -> list[float]:
    """행별 **중심화된** 스칼라 어드밴티지(= 그 행의 응답 토큰 전체에 얹히는 값).

    실제 학습 경로에서는 verl 의 기본 GRPO 가 `row_group_keys` 로 갈린 uid 위에서 같은
    계산을 한다 — 이 함수는 그것과 **같은 정의**를 CPU 에서 검증하기 위한 것이다
    (`dcpo_region.group_mean_subtract`: 평균만 빼고 /std 하지 않는다 = Dr.GRPO)."""
    rew = build_row_rewards(term, stages, active, r1, r2, gamma)
    keys = row_group_keys(uids, stages, active)
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}
    for k, v in zip(keys, rew):
        sums[k] = sums.get(k, 0.0) + v
        cnts[k] = cnts.get(k, 0) + 1
    return [rew[i] - sums[keys[i]] / cnts[keys[i]] for i in range(len(rew))]


def recredit(term: str, own_correct: Sequence[float], stages: Sequence[str],
             active: Sequence[int], uids: Sequence[str], trial_keys: Sequence[str],
             gamma: float) -> tuple[list[float], list[str], dict[str, float]]:
    """학습 경로의 **유일한 훅**. 각 행의 «자기 응답의 gold 정오»(own_correct — 기존
    `_compute_math_arm_stash` 가 이미 내는 값)를 받아

      ① trial 별 R1(시도-1 행의 정오)·R2(활성 시도-2 행의 정오, 없으면 0)을 모으고
      ② `row_reward` 로 **중심화 전** 시퀀스 보상을 다시 쓰고
      ③ uid 를 단계별로 갈라(`row_group_keys`) verl 기본 GRPO 중심화가 단계별·문제별
         평균 빼기가 되게 한다.

    반환 (rewards, group_uids, telemetry). 반성문 행의 own_correct 는 **쓰이지 않는다**
    (반성문은 답이 아니다 — 채점기가 거의 항상 0 을 내지만 그 값에 의존하지 않는다)."""
    assert_no_sandbag(gamma)
    n = len(own_correct)
    if not (len(stages) == len(active) == len(uids) == len(trial_keys) == n):
        raise ValueError("[TRIAL2] recredit 길이 불일치")
    r1: dict[str, float] = {}
    r2: dict[str, float] = {}
    for i in range(n):
        if stages[i] == "a1":
            r1[trial_keys[i]] = float(own_correct[i])
        elif stages[i] == "a2" and int(active[i]):
            r2[trial_keys[i]] = float(own_correct[i])
    rew = [row_reward(term, stages[i], active[i],
                      r1.get(trial_keys[i], 0.0), r2.get(trial_keys[i], 0.0), gamma)
           for i in range(n)]
    keys = row_group_keys(uids, stages, active)
    n_a1 = sum(1 for s in stages if s == "a1")
    n_a2 = sum(1 for i in range(n) if stages[i] == "a2" and int(active[i]))
    tel = {
        "n_rows": float(n), "n_trials": float(n_a1), "n_retried": float(n_a2),
        "r1_mean": (sum(r1.values()) / len(r1)) if r1 else 0.0,
        "r2_mean": (sum(r2.values()) / len(r2)) if r2 else 0.0,
        # 2-시도 정확도 = R1 ∨ R2 (판정 게이트의 «.617 대비» 와 같은 정의)
        "two_trial_acc": (sum(1.0 for t in r1 if r1[t] > 0 or r2.get(t, 0.0) > 0) / len(r1))
        if r1 else 0.0,
        "gamma": float(gamma),
    }
    # ★중단 규칙(설계 §2·§4)을 스텝마다 읽을 수 있게 구제·파괴를 분모와 함께 찍는다.
    #   p_fix      = R1=0 인 trial 중 R2=1 이 된 몫(표적 ≥ .35)
    #   break_rate = R1=1 인 trial 중 R2=0 이 된 몫(표적 ≤ .04 — 지금 .078)
    #   재시도를 안 받은 trial(R2 없음)은 R2=0 이 아니라 **분모에서 뺀다** — 안 굴린 시도를
    #   파괴로 세면 RETRY_GATE=wrong 에서 break_rate 가 항상 1 이 된다.
    wrong = [t for t in r1 if r1[t] <= 0 and t in r2]
    right = [t for t in r1 if r1[t] > 0 and t in r2]
    tel["n_r1_wrong"] = float(len(wrong))
    tel["n_r1_right"] = float(len(right))
    tel["p_fix"] = (sum(1.0 for t in wrong if r2[t] > 0) / len(wrong)) if wrong else 0.0
    tel["break_rate"] = (sum(1.0 for t in right if r2[t] <= 0) / len(right)) if right else 0.0
    return rew, keys, tel


# ── 중단 규칙(설계 §4) ───────────────────────────────────────────────────────
STOP_WARMUP = 5
STOP_DROP = 0.01        # 1pp
STOP_PATIENCE = 3


def stop_reason(steps: Sequence[dict], *, warmup: int = STOP_WARMUP,
                drop: float = STOP_DROP, patience: int = STOP_PATIENCE) -> str | None:
    """설계 §4 의 두 중단 조건. `steps` 는 스텝 순서의 텔레메트리
    (`{"step":int, "r1_mean":float, "note_generic":float|None}`). 위반이면 사유 문자열,
    아니면 None — 호출자(verl_sdc)가 사유를 받아 `_CountdownAbort`(rc 75)를 던진다.

    ① **사행(sandbagging) 감시** — 첫 `warmup` 스텝 평균 대비 `drop` 이상 낮은 `r1_mean`
       이 `patience` 스텝 **연속**이면 중단. 시도-1 정확도를 깎아 가며 구제율을 올리는
       퇴화를 잡는다(§3 의 부등식이 막는 것은 «의도적 사행»이고, 이건 실측 감시다).
    ② **팔 퇴화 감시** — 반성문이 전부 일반 문구(`note_generic == 1.0`)면 그 팔은
       NOTE_MODE=none 과 바이트 동일해진다. `patience` 스텝 연속이면 중단(§4).
       워밍업을 기다리지 않는다 — 처음부터 100% 면 그 팔은 이미 대조 팔이다.
    """
    steps = list(steps)
    g = [s.get("note_generic") for s in steps]
    if len(g) >= patience and all(x is not None and float(x) >= 1.0 for x in g[-patience:]):
        return (f"note_generic=1.0 이 {patience} 스텝 연속 — 반성문이 전부 일반 문구라 "
                "이 팔은 NOTE_MODE=none 과 바이트 동일하다(설계 §4).")
    if len(steps) <= warmup:
        return None
    base = sum(float(s["r1_mean"]) for s in steps[:warmup]) / warmup
    tail = steps[warmup:]
    if len(tail) >= patience and all(float(s["r1_mean"]) <= base - drop for s in tail[-patience:]):
        last = ", ".join(f"{float(s['r1_mean']):.4f}" for s in tail[-patience:])
        return (f"r1_mean 이 첫 {warmup} 스텝 평균 {base:.4f} 보다 {drop:.3f} 이상 낮은 상태로 "
                f"{patience} 스텝 연속({last}) — 시도-1 이 깎이고 있다(설계 §4).")
    return None


def leak_stats(reasons: Sequence[str]) -> dict[str, float]:
    """`clean_note` reason 벡터 → 스텝 텔레메트리(ok/empty/digit/boxed/answer + generic 비율)."""
    out: dict[str, float] = {"n": float(len(reasons))}
    for r in ("ok", "empty", "digit", "boxed", "answer"):
        out[r] = float(sum(1 for x in reasons if x == r))
    out["generic_rate"] = (out["n"] - out["ok"]) / out["n"] if out["n"] else 0.0
    return out
