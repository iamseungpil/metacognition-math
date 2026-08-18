r"""Countdown 전용 PMI 스코어러 — 평문 `<meta>…</meta>` 토큰 스팬 + gold/decoy 대조.

왜 새 파일인가 (추측 아님 — 읽고 실측해서 확인한 것)
──────────────────────────────────────────────────────────────────────────────
기존 `verl_sdc.py:1583 _compute_dcpo_v4_pmi_shift_rmeta` 는 Countdown 에서 **전 행
None** 이 된다. 세 자리가 각각 막는다:

  ① 태그 어휘.   그 함수는 `split_first_meta` 로 `src/metacot/prompt.META_START`
     = 특수토큰 `<|meta|>` 를 찾는다. Countdown 프롬프트가 요구하는 것은 평문
     `<meta>…</meta>` 다(`countdown_task.META_OPEN_TAG`). → 매칭 0건 → attempted=0 →
     로그는 `skipped` 한 줄, R_meta 전원 0. **선언은 있고 배선은 없는** 그 형태다.
  ② 정답 문자열.  그 함수는 `ground_truths[i]` 를 gold 로 쓰고 `_rule_based_decoy`
     로 오답을 만든다. Countdown 의 `ground_truth` 는 `{"target":…, "nums":[…]}`
     JSON 이라(`countdown_task.make_ground_truth`) gold 식이 아니다. 증인식·연산자교체
     오답은 `build_records` 가 **최상위 컬럼**(`witness` / `decoy`)으로 싣는다.
  ③ 길이 동수 가정.  `_pmi_position_scalar`(:1531)는 `len(decoy_ids) != len(gold_ids)`
     이면 **NaN 으로 fail-closed** 한다. Countdown 의 연산자 교체 오답은 문자 하나만
     다른데도 토큰 길이가 갈린다 — **300개 중 11개(3.7%)** 실측(Qwen3-4B 토크나이저,
     `gen_instances(300, seed=0)`; 예: `((11+(6*21))*(19-18))` 19토큰 vs
     `((11+(6*21))-(19-18))` 20토큰). 그 3.7% 가 통째로 버려진다.

그래서 ①②③ 을 고친 스코어러를 여기 둔다. **기존 함수는 한 바이트도 건드리지 않는다.**

기존 인프라 재사용 판정 (항목 3 — 읽고 내린 판정)
──────────────────────────────────────────────────────────────────────────────
  `_build_pmi_score_batches` (verl_sdc.py:1445)   ★재사용한다.
      토큰 id 리스트 두 벌만 받는 순수 배치 조립기다. 태그·정답·길이에 대한 가정이
      전혀 없고, 프롬프트 좌측패딩 + 응답이 정확히 p_max 열에서 시작하는 verl 표준
      레이아웃(그 자리에 latent C3급 정렬 버그가 있었다)을 이미 고쳐 담고 있다.
      직접 다시 짜면 그 수리를 잃는다.

  `_dcpo_v4_ref_logprobs` (verl_sdc.py:1502)      ★재사용한다.
      T=1.0 하드코딩 + `use_legacy_worker_impl` fail-closed assert 가 들어 있다.
      우리가 원하는 성질이 바로 그 둘이다. 복제하면 assert 를 잃는다.

  `divergent_token_mask` (dcpo_directional.py:29) ⛔재사용하지 않는다. 이유:
      gold 토큰 스팬 위의 **위치 정렬** 마스크이고, 짝인 `_pmi_position_scalar` 가
      길이 불일치를 NaN 으로 죽인다(위 ③, 실측 3.7%). Countdown 의 gold/decoy 는
      "같은 구조·연산자 하나"라 길이가 갈릴 수 있으므로 위치 정렬 자체가 성립하지
      않는다. 대신 여기서는 **접두·접미를 잘라낸 두 슬라이스를 각각 합산**한다
      (`divergent_spans`). 길이가 달라도 정의된다.
      ⇒ 같은 이유로 `_pmi_position_scalar` 도 쓰지 않는다.

config 요구사항 (항목 4 — `_dcpo_v4_ref_logprobs` 의 assert 를 읽고 확인)
──────────────────────────────────────────────────────────────────────────────
  `trainer.use_legacy_worker_impl: disable`  ★**필수**. 없으면 학습이 step 1 에서
  죽는다. `verl_sdc.py:1516` 이 `trainer.config.trainer.use_legacy_worker_impl` 을
  읽는데, 키가 아예 없으면 except 로 `"<unreadable>"` 이 되고 :1519 의 assert 가
  터진다(fail-closed 설계). 즉 "키를 안 쓰면 조용히 넘어간다"가 아니다.
  ⚠`countdown_rl_8arm.yaml` 이 이미 `actor_rollout_ref.rollout.temperature=1.0` 을
  주므로 레거시 워커라도 압축 배율은 1.0 이겠지만, **assert 는 무조건 켜져 있으므로**
  config 에 이 한 줄이 없으면 그 사실과 무관하게 크래시한다.
  현행 선례: `configs/verl_sdc_e21r_shared.yaml:308`.
  ⚠`configs/countdown_grpo_qwen3_4b.yaml`(런처 `CONFIG_NAME`)은 **이 파일을 쓰는 시점에
  존재하지 않았다** — 미확인. 그 config 를 만드는 쪽이 이 줄을 넣어야 한다.

토큰 스팬을 문자 오프셋으로 안 잡는 이유 (실측)
──────────────────────────────────────────────────────────────────────────────
  Qwen3-4B 토크나이저에서 `<meta>` 는 **단일 토큰이 아니고**, 앞뒤 문맥에 따라
  토큰 경계가 바뀐다(실측):
      '<meta>'        → ['<meta', '>']
      '</meta>'       → ['</', 'meta', '>']
      '\n<meta>\n'    → ['\n', '<meta', '>\n']      ← 닫는 '>' 와 다음 줄바꿈이 한 토큰
      'x <meta>'      → ['x', ' <', 'meta', '>']    ← 앞 공백과 '<' 가 한 토큰
      convert_tokens_to_ids('<meta>') → None        ← 어휘에 없다
  따라서 고정 id 열을 찾는 방식은 성립하지 않는다. 여기서는 **fast 토크나이저의
  offset_mapping** 으로 문자 스팬 → 토큰 인덱스를 옮기고, 경계에 걸친 토큰
  (straddle)을 **보수적으로** 처리한다:
      OPEN  컨텍스트: 태그에 조금이라도 걸친 토큰을 **뺀다** → 메타가 절대 새지 않는다.
      CLOSE 컨텍스트: `</meta>` 끝에 걸친 토큰을 **넣는다** → 메타 블록이 절대 잘리지 않는다.
  대가는 OPEN 이 공백 한두 글자를 잃고 CLOSE 가 줄바꿈 한 글자를 더 먹는 것뿐이며,
  두 경우 모두 `MetaSpan.straddle_*` 로 **기록**된다(조용히 넘어가지 않는다).

⛔이 파일은 GPU 를 잡지 않는다. verl/ray 도 **지연 import** 한다(`score_pmi_shift`
  안에서만). 그래서 스팬·발산토큰 로직 전체가 CPU 단위테스트로 검증된다.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

# 태그 정의는 복제하지 않는다 — 프롬프트를 만드는 파일이 정의처다.
from src.training.countdown_task import META_CLOSE_TAG, META_OPEN_TAG

__all__ = [
    "MetaSpan",
    "find_meta_token_span",
    "boxed",
    "divergent_spans",
    "build_pmi_arms",
    "read_pmi_from_ref_logprobs",
    "score_pmi_shift",
    "assert_pmi_config",
    "CONFIG_REQUIREMENTS",
    "REUSED_FROM_VERL_SDC",
    "NOT_REUSED",
]

# 항목 3·4 의 판정을 **기계가 읽을 수 있게** 남긴다(문서만 남기면 갈린다).
REUSED_FROM_VERL_SDC = (
    "_build_pmi_score_batches",   # 배치 조립 (좌측패딩 정렬 수리 포함)
    "_dcpo_v4_ref_logprobs",      # T=1.0 + use_legacy_worker_impl fail-closed assert
)
NOT_REUSED = {
    "divergent_token_mask": (
        "gold 토큰 스팬 위 위치정렬 마스크. 짝인 _pmi_position_scalar 가 "
        "len(decoy)!=len(gold) 를 NaN 으로 죽이는데, Countdown 연산자교체 오답은 "
        "실측 3.7%(11/300, Qwen3-4B) 가 길이 불일치다."),
    "_pmi_position_scalar": (
        "위와 같은 이유. 여기서는 두 슬라이스를 각각 합산해 길이 불일치를 허용한다."),
    "_compute_dcpo_v4_pmi_shift_rmeta": (
        "특수토큰 <|meta|> 를 찾고 ground_truth 를 gold 로 쓴다. Countdown 은 평문 "
        "<meta> 이고 ground_truth 는 {target,nums} JSON 이라 전 행 None 이 된다."),
}

CONFIG_REQUIREMENTS = {
    "trainer.use_legacy_worker_impl": "disable",
}

_NAN = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 메타 블록의 **토큰** 스팬
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MetaSpan:
    r"""응답 토큰열 위에서 본 첫 번째 닫힌 `<meta>…</meta>` 블록의 위치.

    전부 **토큰 인덱스**다(문자 오프셋 아님). 반열린 구간 [start, end).

      open_start   OPEN 컨텍스트의 길이. `ids[:open_start]` 에는 태그가 한 글자도 없다.
      inner_start  태그 안 내용의 첫 토큰 (완전히 안에 든 토큰만)
      inner_end    태그 안 내용의 끝(배타)
      close_end    CLOSE 컨텍스트의 길이. `ids[:close_end]` 는 `</meta>` 를 반드시 포함한다.
      n_inner_tok  inner_end - inner_start  ← G 팔(가짜 대조군)의 `meta_n_tok` 원재료
                   ⚠**하한**이다. 태그 경계 토큰에 빨려 들어간 내용 글자는 안 센다.
                   실측(Qwen3-4B): 여러 줄짜리 실제 메타는 13·5·5 토큰으로 정상이지만
                   `<meta>c</meta>` 처럼 내용이 한 글자면 **0** 이 나온다(그 'c' 가
                   `>c</` 경계 토큰에 흡수된다). G 팔은 길이의 대리를 재는 음성 대조군이라
                   이 하한 편향이 G 를 **불리하게** 만들 뿐 유리하게 만들지 않는다.
      straddle_open   True 면 `<` 앞 토큰이 태그에 걸쳐 OPEN 에서 **빠졌다**(글자 몇 개 손실)
      straddle_close  True 면 `>` 뒤 글자가 CLOSE 에 **딸려 들어왔다**
      meta_first   open_start == 0 (본문 없이 메타부터 냈다). PMI_open 은 여전히 정의되지만
                   "메타 앞 믿음"이 프롬프트만의 믿음이 된다 — 판정에서 갈라 봐야 한다.
      ids          이 스팬이 서 있는 **바로 그 토큰화**. 컨텍스트를 만들 때 반드시 이걸 써라
                   (텍스트를 다시 토큰화하면 표류한다).
    """

    open_start: int
    inner_start: int
    inner_end: int
    close_end: int
    ids: tuple = field(repr=False)
    char_open: int = -1
    char_close_end: int = -1
    straddle_open: bool = False
    straddle_close: bool = False

    @property
    def n_inner_tok(self) -> int:
        return max(0, self.inner_end - self.inner_start)

    @property
    def meta_first(self) -> bool:
        return self.open_start == 0

    @property
    def n_tok(self) -> int:
        return len(self.ids)


def find_meta_token_span(
    tokenizer,
    response_text: str,
    *,
    open_tag: str = META_OPEN_TAG,
    close_tag: str = META_CLOSE_TAG,
    response_ids: Optional[Sequence[int]] = None,
) -> Optional[MetaSpan]:
    r"""첫 번째 **닫힌** `<meta>…</meta>` 블록의 토큰 스팬. 없으면 None.

    ⚠`response_text` 는 **응답만** 넘겨라. 프롬프트를 붙여 넘기면 안 된다 —
      Countdown 시스템 프롬프트(`countdown_task.SOLVE_SYS_NEW`)가 형식을 글자 그대로
      보여주려고 `<meta>…</meta>` 예시 블록을 **본문에 박고 있다**. 프롬프트를 포함해
      찾으면 모델이 낸 메타가 아니라 프롬프트의 예시를 잡는다.

    fast 토크나이저(offset_mapping)가 필요하다. 없으면 조용히 None 을 돌려주지 않고
    **예외를 던진다** — 여기서 None 을 흘리면 "메타를 아무도 안 냈다"로 오독되고,
    그것이 정확히 우리가 여러 번 당한 무효 레버다.

    `response_ids` 를 주면 이 함수가 쓴 토큰화와 **같은지 확인**만 하고(진단),
    다르면 예외를 던진다. 스팬은 언제나 이 함수의 토큰화 위에 정의된다.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "find_meta_token_span 은 fast 토크나이저의 offset_mapping 이 필요하다 "
            f"(받은 것: {type(tokenizer).__name__}, is_fast="
            f"{getattr(tokenizer, 'is_fast', None)!r}). 평문 <meta> 는 Qwen3 어휘에 "
            "단일 토큰으로 없고 앞뒤 문맥에 따라 경계가 바뀌므로, 고정 id 열 탐색으로는 "
            "대체할 수 없다.")

    text = response_text or ""
    i = text.find(open_tag)
    if i < 0:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j < 0:
        return None                      # 열렸는데 안 닫힌 블록 = 스팬 없음
    c_open = i
    c_inner_start = i + len(open_tag)
    c_inner_end = j
    c_close_end = j + len(close_tag)

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if not ids:
        return None
    if response_ids is not None and list(response_ids) != ids:
        raise ValueError(
            "find_meta_token_span: 넘겨받은 response_ids 가 이 함수의 토큰화와 다르다 "
            f"({len(list(response_ids))} vs {len(ids)} 토큰). 컨텍스트를 두 토큰화로 "
            "섞어 만들면 PMI 가 조용히 어긋난다.")

    n = len(ids)

    def _first_start_ge(c: int) -> int:
        for k, (a, b) in enumerate(offsets):
            if b <= a:                   # 길이 0 오프셋(특수토큰)은 건너뛴다
                continue
            if a >= c:
                return k
        return n

    def _count_end_le(c: int) -> int:
        k = 0
        for a, b in offsets:
            if b <= a:
                k += 1
                continue
            if b <= c:
                k += 1
            else:
                break
        return k

    # OPEN: 태그에 **조금이라도** 걸친 토큰은 뺀다 → 메타가 새지 않는다.
    open_start = _count_end_le(c_open)
    straddle_open = bool(open_start < n and offsets[open_start][0] < c_open)

    # CLOSE: `</meta>` 끝에 걸친 토큰은 넣는다 → 메타가 잘리지 않는다.
    close_end = _first_start_ge(c_close_end)
    straddle_close = bool(close_end > 0 and offsets[close_end - 1][1] > c_close_end)

    # 안쪽: 완전히 태그 사이에 든 토큰만 센다(경계 토큰을 세면 길이가 부풀어 G 팔이 오염된다).
    inner_start = _first_start_ge(c_inner_start)
    inner_end = max(inner_start, _count_end_le(c_inner_end))

    if not (0 <= open_start <= inner_start <= inner_end <= close_end <= n):
        raise AssertionError(
            "find_meta_token_span: 스팬 순서가 깨졌다 "
            f"(open={open_start} inner=[{inner_start},{inner_end}) close={close_end} n={n}). "
            "offset_mapping 이 단조가 아니거나 태그 탐색이 어긋났다.")
    if close_end <= open_start:
        return None

    return MetaSpan(
        open_start=open_start, inner_start=inner_start, inner_end=inner_end,
        close_end=close_end, ids=tuple(ids),
        char_open=c_open, char_close_end=c_close_end,
        straddle_open=straddle_open, straddle_close=straddle_close,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. gold / decoy 발산 토큰
# ══════════════════════════════════════════════════════════════════════════════

def boxed(expr) -> str:
    r"""PMI 가 재는 답 문자열: `\boxed{expr}`.

    `dcpo_directional.boxed_answer_string` 과 **같은 식**이지만 그 함수를 부르지 않는다
    — 그쪽은 numpy 를 끌고 오는 모듈이고, 이 한 줄을 위해 의존을 늘릴 이유가 없다.
    두 곳이 갈릴 위험은 `tests/test_countdown_pmi.py` 의 동치 검사가 막는다.
    """
    return r"\boxed{" + str(expr).strip() + "}"


@dataclass(frozen=True)
class DivergentPair:
    gold_ids: tuple
    decoy_ids: tuple
    gold_slice: slice
    decoy_slice: slice
    path: str          # "div" (발산 토큰만) 또는 "full" (전체 문자열 폴백)


def divergent_spans(tokenizer, gold_expr: str, decoy_expr: str) -> Optional[DivergentPair]:
    r"""`\boxed{gold}` / `\boxed{decoy}` 토큰과 **비교할 슬라이스**. 못 만들면 None.

    두 경로 (`mini_abl/signal_probe.py:pair_logp` 의 수리를 그대로 가져왔다):

      div   공통 접두·접미를 잘라낸 **발산 토큰만** 비교. 원래 방식.
      full  ★폴백. 한쪽이 다른 쪽의 부분열이면(`'2'` vs `'12'`) 잘라낸 뒤 슬라이스가
            **빈다**. 0817 스모크에서 후보 쌍의 다수가 그 형태였고 전부 버려졌다(n=0).
            그때는 `\boxed{...}` 전체 문자열을 비교한다.

    ⚠`full` 은 길이 편향이 있다(긴 문자열의 logp 가 낮다). 그래서 경로를 **행마다 기록**해
      판정에서 갈라 볼 수 있게 한다. 길이 음성 대조군(G 팔)이 이 편향의 최종 방어선이다.

    ★길이 불일치는 여기서 **버리지 않는다**. 두 슬라이스를 각각 합산하므로 정의된다
      (`verl_sdc._pmi_position_scalar` 는 여기서 NaN 으로 죽는다 — 실측 3.7%).
    """
    if gold_expr is None or decoy_expr is None:
        return None
    g_txt, d_txt = boxed(gold_expr), boxed(decoy_expr)
    if not str(gold_expr).strip() or not str(decoy_expr).strip():
        return None
    if g_txt == d_txt:
        return None                       # gold==decoy → 구성상 신호 0
    g = list(tokenizer(g_txt, add_special_tokens=False)["input_ids"])
    d = list(tokenizer(d_txt, add_special_tokens=False)["input_ids"])
    if not g or not d:
        return None

    p = 0
    while p < min(len(g), len(d)) and g[p] == d[p]:
        p += 1
    s = 0
    while (s < min(len(g), len(d)) - p) and g[len(g) - 1 - s] == d[len(d) - 1 - s]:
        s += 1
    gs, ds = slice(p, len(g) - s), slice(p, len(d) - s)
    path = "div"
    if gs.stop <= gs.start or ds.stop <= ds.start:
        gs, ds, path = slice(0, len(g)), slice(0, len(d)), "full"
    return DivergentPair(tuple(g), tuple(d), gs, ds, path)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 배치 조립 → ref 스코어링 → PMI 읽기
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Attempt:
    row: int
    span: MetaSpan
    pair: DivergentPair


def build_pmi_arms(
    tokenizer,
    prompt_texts: Sequence[str],
    response_texts: Sequence[str],
    witnesses: Sequence[Optional[str]],
    decoys: Sequence[Optional[str]],
):
    r"""점수를 매길 4n 개 (컨텍스트, 답) 팔을 만든다. GPU 를 잡지 않는다.

    행 하나당 팔 넷, **고정 순서**:
        gold@open, decoy@open, gold@close, decoy@close

    돌려주는 것: (arm_prompts, arm_resps, attempts, diag)
      arm_prompts/arm_resps  `_build_pmi_score_batches` 에 그대로 넘길 토큰 id 리스트
      attempts               읽기용 부기 (_Attempt 리스트, 길이 = 점수 매기는 행 수)
      diag                   건너뛴 사유별 개수 — **0 인 팔을 조용히 만들지 않기 위한 것**
    """
    B = len(response_texts)
    if not (len(prompt_texts) == len(witnesses) == len(decoys) == B):
        raise ValueError(
            f"build_pmi_arms: 길이 불일치 prompt={len(prompt_texts)} "
            f"resp={B} witness={len(witnesses)} decoy={len(decoys)}")

    arm_prompts: list = []
    arm_resps: list = []
    attempts: list = []
    diag = {"no_meta": 0, "no_witness": 0, "no_decoy": 0, "bad_pair": 0,
            "meta_first": 0, "straddle_open": 0, "straddle_close": 0,
            "path_div": 0, "path_full": 0}

    for i in range(B):
        span = find_meta_token_span(tokenizer, response_texts[i] or "")
        if span is None:
            diag["no_meta"] += 1
            continue
        w = (witnesses[i] or "").strip() if witnesses[i] is not None else ""
        dc = (decoys[i] or "").strip() if decoys[i] is not None else ""
        if not w:
            diag["no_witness"] += 1
            continue
        if not dc:
            # `gen_instances(require_decoy=True)` 면 여기 안 온다. 오면 데이터가 다른 것이다.
            diag["no_decoy"] += 1
            continue
        pair = divergent_spans(tokenizer, w, dc)
        if pair is None:
            diag["bad_pair"] += 1
            continue

        if span.meta_first:
            diag["meta_first"] += 1
        if span.straddle_open:
            diag["straddle_open"] += 1
        if span.straddle_close:
            diag["straddle_close"] += 1
        diag["path_div" if pair.path == "div" else "path_full"] += 1

        p_ids = list(tokenizer(prompt_texts[i] or "",
                               add_special_tokens=False)["input_ids"])
        ctx_open = p_ids + list(span.ids[:span.open_start])
        ctx_close = p_ids + list(span.ids[:span.close_end])
        g, d = list(pair.gold_ids), list(pair.decoy_ids)
        arm_prompts.append(ctx_open);  arm_resps.append(g)
        arm_prompts.append(ctx_open);  arm_resps.append(d)
        arm_prompts.append(ctx_close); arm_resps.append(g)
        arm_prompts.append(ctx_close); arm_resps.append(d)
        attempts.append(_Attempt(row=i, span=span, pair=pair))

    return arm_prompts, arm_resps, attempts, diag


def read_pmi_from_ref_logprobs(ref_lp, attempts: Sequence[_Attempt]) -> list:
    r"""ref 토큰별 logp → 행별 (pmi_open, pmi_close).

    `ref_lp[i, t]` = logP_ref(arm_resps[i][t] | arm_prompts[i] + arm_resps[i][:t]).
    PMI(ctx) = Σ_{gold_slice} logp(gold|ctx) − Σ_{decoy_slice} logp(decoy|ctx).

    ★두 슬라이스를 **각각** 합산한다 — 길이가 달라도 정의된다(재사용 판정 참조).
    유한하지 않은 값이 하나라도 있으면 그 행은 NaN 으로 fail-closed 한다(포이즌 행이
    그룹 형제의 센터링을 망치지 못하게 — `countdown_rewards` 의 규약과 같다).
    """
    out = []
    for k, at in enumerate(attempts):
        base = 4 * k
        gsl, dsl = at.pair.gold_slice, at.pair.decoy_slice
        Lg, Ld = len(at.pair.gold_ids), len(at.pair.decoy_ids)
        try:
            g_open = _row_sum(ref_lp, base + 0, Lg, gsl)
            d_open = _row_sum(ref_lp, base + 1, Ld, dsl)
            g_close = _row_sum(ref_lp, base + 2, Lg, gsl)
            d_close = _row_sum(ref_lp, base + 3, Ld, dsl)
        except Exception:
            out.append((_NAN, _NAN))
            continue
        pmi_open = g_open - d_open
        pmi_close = g_close - d_close
        if not (math.isfinite(pmi_open) and math.isfinite(pmi_close)):
            out.append((_NAN, _NAN))
            continue
        out.append((float(pmi_open), float(pmi_close)))
    return out


def _row_sum(ref_lp, r: int, length: int, sl: slice) -> float:
    """`ref_lp[r, :length][sl]` 의 합. torch/numpy/리스트 전부 받는다."""
    row = ref_lp[r]
    try:
        row = row[:length]
        seg = row[sl]
        tot = seg.sum()
    except (AttributeError, TypeError):
        seq = list(row)[:length][sl]
        return float(sum(float(v) for v in seq))
    try:
        return float(tot.item())
    except AttributeError:
        return float(tot)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 진입점 — verl 은 여기서만 지연 import 한다
# ══════════════════════════════════════════════════════════════════════════════

def assert_pmi_config(trainer) -> str:
    r"""`trainer.use_legacy_worker_impl=disable` 인지 **먼저** 확인한다.

    `_dcpo_v4_ref_logprobs` 안에도 같은 assert 가 있지만(그게 최종 방어선),
    여기서 먼저 보면 배치 조립을 다 한 뒤가 아니라 진입 즉시 깨져서 로그가 읽힌다.
    읽을 수 없는 config 는 여기서 통과시키고(단위테스트의 mock trainer 를 위해)
    ref 스코어러의 하드 assert 에 맡긴다 — `verl_sdc.py:1640-1648` 과 같은 규약.
    """
    try:
        legacy = str(trainer.config.trainer.use_legacy_worker_impl)
    except Exception:
        return "<unreadable>"
    assert legacy == "disable", (
        "Countdown PMI 는 engine 워커 경로가 필요하다 "
        f"(trainer.use_legacy_worker_impl=disable, 받은 값 {legacy!r}). "
        "레거시 fsdp 워커는 호출자가 세운 meta_info['temperature']=1.0 을 "
        "rollout.temperature 로 덮어써 PMI 를 압축한다 (verl_sdc.py:1502-1523).")
    return legacy


def _pad_unit(trainer) -> int:
    """verl dispatch 나눗셈 단위. `_compute_dcpo_v4_pmi_shift_rmeta` 와 같은 계산."""
    def _get(path, default):
        cur = trainer
        try:
            for p in path.split("."):
                cur = getattr(cur, p)
            return int(cur)
        except Exception:
            return int(default)
    nnodes = _get("config.trainer.nnodes", 1)
    n_gpus = _get("config.trainer.n_gpus_per_node", 4)
    micro = _get("config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 4)
    return max(1, nnodes * n_gpus * micro)


def score_pmi_shift(
    *,
    tokenizer,
    trainer,
    prompt_texts: Sequence[str],
    response_texts: Sequence[str],
    witnesses: Sequence[Optional[str]],
    decoys: Sequence[Optional[str]],
    step: int = 0,
    debug: Optional[bool] = None,
    _ref_scorer=None,
) -> tuple[list, dict]:
    r"""행별 `{pmi_open, pmi_close, …}` 와 진단. **여기서만 GPU 를 쓴다.**

    Returns (rows, diag).
      rows[i]  = {"pmi_open", "pmi_close", "meta_n_tok", "emitted", "path",
                  "meta_first", "scored"}   ← `countdown_rewards.arm_reward` 의 행 규약
                 점수를 못 매긴 행은 pmi_* 가 NaN, scored=False, emitted 는 스팬 유무.
      diag     건너뛴 사유별 개수 + attempted/scored 비율.

    `witnesses` / `decoys` 는 **`non_tensor_batch` 의 최상위 컬럼**에서 온다
    (`countdown_task.build_records(include_flat_cols=True)` 가 싣는 `witness`/`decoy`).
    `ground_truth` 에서 꺼내려 하지 마라 — 거기엔 `{target, nums}` 뿐이다.

    `_ref_scorer` 는 테스트용 주입구다(주면 verl 을 import 하지 않는다).
    """
    if debug is None:
        debug = os.environ.get("DCPO_DEBUG", "1") == "1"
    B = len(response_texts)
    rows = [{"pmi_open": _NAN, "pmi_close": _NAN, "meta_n_tok": 0,
             "emitted": 0, "path": None, "meta_first": False, "scored": False}
            for _ in range(B)]

    arm_prompts, arm_resps, attempts, diag = build_pmi_arms(
        tokenizer, prompt_texts, response_texts, witnesses, decoys)
    diag = dict(diag)
    diag["B"] = B
    diag["attempted"] = len(attempts)
    diag["scored"] = 0
    diag["attempted_rate"] = (len(attempts) / B) if B else 0.0
    diag["ref_error"] = None

    for at in attempts:
        r = rows[at.row]
        r["emitted"] = 1
        r["meta_n_tok"] = at.span.n_inner_tok
        r["path"] = at.pair.path
        r["meta_first"] = at.span.meta_first

    if not attempts:
        if debug:
            print(f"[CD-PMI] step={step}: B={B} attempted=0 — {diag}", flush=True)
        return rows, diag

    if _ref_scorer is None:
        assert_pmi_config(trainer)
        # ★지연 import: verl/ray 는 학습 노드에만 있다. 이 모듈을 CPU 에서 테스트할 수
        #   있어야 하므로 모듈 최상단으로 올리지 않는다.
        from src.training.verl_sdc import (          # noqa: PLC0415
            _build_pmi_score_batches, _dcpo_v4_ref_logprobs,
        )
        tensors, real_n = _build_pmi_score_batches(
            arm_prompts, arm_resps, _pad_unit(trainer))
        if real_n != 4 * len(attempts):
            raise AssertionError(
                f"CD-PMI 팔 부기가 깨졌다: {real_n} != 4*{len(attempts)}")
        try:
            ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
        except AssertionError:
            raise                       # config 위반은 절대 삼키지 않는다
        except Exception as e:
            diag["ref_error"] = f"{type(e).__name__}: {e}"
            print(f"[CD-PMI] step={step}: ref 스코어링 실패 ({diag['ref_error']}) — "
                  f"이 배치의 PMI 는 전부 NaN.", flush=True)
            return rows, diag
    else:
        ref_lp = _ref_scorer(arm_prompts, arm_resps)

    pmis = read_pmi_from_ref_logprobs(ref_lp, attempts)
    n_scored = 0
    for at, (po, pc) in zip(attempts, pmis):
        rows[at.row]["pmi_open"] = po
        rows[at.row]["pmi_close"] = pc
        ok = math.isfinite(po) and math.isfinite(pc)
        rows[at.row]["scored"] = bool(ok)
        n_scored += int(ok)
    diag["scored"] = n_scored
    diag["scored_rate"] = n_scored / B if B else 0.0

    if debug:
        print(f"[CD-PMI] step={step}: B={B} attempted={len(attempts)} "
              f"scored={n_scored} no_meta={diag['no_meta']} "
              f"path(div/full)={diag['path_div']}/{diag['path_full']} "
              f"meta_first={diag['meta_first']}", flush=True)
    return rows, diag
