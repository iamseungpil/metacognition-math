#!/usr/bin/env python
r"""math_meta_content_gate — «메타인지 **행위의 내용**이 조향하는가» 게이트.

왜: 이 정책에서 **판단을 요구하는 메타인지는 전부 실패했다**. (a) 자기 롤아웃의 정오를 문제
  안에서 가르는 내성 판별력은 토큰 불확실도 12종 + 은닉 프로브를 써도 AUC .57 이 천장, (b) 자기
  표본 4개를 보여 줘도 89% 는 다수결을 따라 읽는다(판단이 아니라 모방), (c) 완성된 오답을 같은
  문맥에서 이어 쓰며 다시 푸는 것은 1,248 자리 중 **0개**를 살리고(눈감고 새로 푸는 것 대비
  −.348) **독립 재풀이**는 .600 을 살린다. 즉 «판단해라»와 «다시 유도해라»는 둘 다 값이 없다 —
  그런데 그 실패들은 전부 *같은 종류의 행위*를 자리만 바꿔 잰 것이다. 안 재 본 축은 **행위의
  내용**이다: 답을 원 문제에 **대입**하기, **크기·부호·단위** 보기, **특수한 경우**로 검사하기,
  **제약 다시 읽기** — 재유도보다 싸고, 틀렸다는 신호가 자기 판단이 아니라 밖에서 온다
  (비용 비대칭). 이 게이트는 그 문장들을 쓸어 어느 것이 정답으로 조향하는지 가른다.

★내용 없는 대조(`filler`/`filler_continue`)가 중심이다. 어떤 지시든 붙이면 응답은 길어지고
  분포는 흔들린다 — 그걸 «이 행위가 도움이 됐다»로 오독한 것이 cd9 EVC·plan-ceiling 함정이다.
  그래서 **내용 주장은 (act − plain) 이 아니라 (act − filler)** 다.
★donor 팔을 뺀 이유(스펙의 조건부 지시): `substitute` 의 donor 형은 «남의 문제의 양을 이름
  부르는 대입 지시»여야 하는데 이 코퍼스는 LaTeX 서술형이라 «양»을 기계적으로 뽑을 방법이 없고,
  뽑아 붙여도 *내용 없는* 대조가 아니라 *내용이 틀린* 대조가 된다(혼란해서 나빠진 것과 내용이
  없어 안 오른 것을 못 가른다). 그 자리에 **내용 없는 이어쓰기 대조** `filler_continue` 를 둔다.
★예산은 **max_tokens 8192**(하드 요구). cd9 G4 는 4,096 에서 Level-5 재풀이의 54% 가 잘렸고
  그때 정확도 순서는 절단률 순서의 **정확히 역순**이었다 — 4k 의 숫자는 해석 불가능하다.
  조건별 `trunc_rate` 를 같이 읽고, 어느 조건이든 .20 을 넘으면 `[BUDGET?]` 를 크게 찍는다.

★0915 갱신(문헌 조사로 행위 목록을 다시 짰다): `units`(단위·꼴 확인)는 뺐다 — 이 코퍼스는
  압도적으로 무차원 숫자 답이라 단위 검사가 공허하고, 다중비교 예산만 갉아먹는다. 대신 셋을
  더한다 — `verification_first`(arXiv:2511.21734, Qwen2.5-1.5B/3B·Llama3.1-8B-Instruct 에서
  실측된 **유일하게 문헌에 근거가 있는** 행위: 문제와 함께 **그럴듯한 오답 후보**를 주고
  "먼저 그게 맞는지 검증한 뒤 스스로 풀라"), `verification_first_random`(같은 문장, 후보만
  문제와 무관한 무작위 정수 — 그 논문이 보고하는 **공개된 대조**다: 후보의 내용이 아니라
  «후보가 주어졌다는 사실» 자체가 행동을 바꾼다는 결과. `verification_first` 가 `filler` 는
  이겨도 `verification_first_random` 을 못 이기면, 정직한 해석은 «내용이 조향했다» 가 아니라
  «후보가 주어지면 뭐든 검증한다» 다), `backward_mask`(역방향 조건-마스킹, arXiv:2212.09561 /
  2312.06867 — 실제 수학 이득을 보고하지만 ~7B 미만에서는 일관되지 않는다고도 보고한다).
★`verification_first` 후보 선정(정답 누출 방지, 아래 `pick_non_plurality_wrong` 참조): 같은
  문제의 **형제 롤아웃**(같은 K 개 중 group_id 로 모은 것)의 \boxed 답을 math_verify 동치로
  군집화하고, **다수결이 아닌 군집 중 gold 와 동치가 아닌 것**만 후보로 남겨 그중 가장 큰
  군집을 고른다. 다수결이 아닌 군집이 전부 gold 와 같거나(소수만 정답인 MIXED 문제) 전원이
  한 값에 합의했으면 그 문제는 이 act 를 **스킵**한다 — 정답을 몰래 후보로 건네지 않기
  위해서다(`n_skipped_no_cand` 로 센다).
★PAL 경고(arXiv:2211.10435): **말로 검산한다고 쓰는 것은 실제로 검산을 수행한 것이 아니다** —
  그 논문은 모델이 자기 프로그램을 **머릿속으로 시뮬레이션**하면 23.2%, **진짜 인터프리터로
  실행**하면 72.0% 라고 보고한다. `substitute`/`backward_mask` 의 null 결과는 «검산을 **말한**
  것의 값»에 상한을 긋는 것이지 «검산을 **수행**한» 것의 값을 재는 게 아니다.
★0915 저녁 갱신(길이-매치 대조 `pad_length`, Gandhi 외 arXiv:2503.01307): 지금까지의 유일한
  내용-없는 대조 `filler`는 한 문장짜리라 **토큰 수**가 행위 접미보다 훨씬 짧다 — Δ가 올라도
  «내용이 조향했다»와 «접미가 길어서 조향했다»를 못 가른다. `pad_length`는 «신중함이 왜
  중요한지 세 문장으로 반추한 뒤 풀라»는, 그 논문이 **null**(효과 없음)로 보고한 길이-매치
  대조다 — 어떤 행위도 이름 부르지 않고 이 문제에 대한 정보도 안 주지만, 토큰 수와 메타인지적
  **어조**는 늘린다. 통과 규칙은 그대로 filler(`CONTROL`) 기준이다 — `pad_length`는 진단용
  둘째 잣대일 뿐이다: 어떤 행위든 (act − filler) 는 양수인데 (act − pad_length) 는 0 이하면
  `[TOKEN-CONFOUND?]` 로 찍는다(그 이득이 내용이 아니라 토큰·어조로 산 것이라는 뜻).

두 모드(같은 롤아웃 파일·지표·부트스트랩을 공유한다):
  --mode prompt    (먼저 돈다) 두 코퍼스(mathL5 / math500)의 MIXED 문제에 **푸는 처음부터**
                   행위를 지시한다(출처별 표도 찍는다). 기준 = `plain`(render_generation_
                   prompt 과 바이트 동일), 대조 = `filler`.
  --mode continue  MIXED 문제의 **오답 롤아웃**을 열린 assistant 턴에서 이어 쓰며 행위를 심는다
                   (math_activation_gate 가 검증한 규약: render_generation_prompt(...) + text
                   + cue = continue_final_message). 기준 = `blind`(눈감고 새로 풀기, 매치드
                   앵커), 대조 = `filler_continue`. 채점은 **이어쓴 부분의 마지막 \boxed 만**
                   본다 — 없으면 답은 접두 그대로다(`no_new_boxed_rate`).

설계: 조건은 전부 `META_ACTS` **한 dict** 에서 나온다 — 그 dict 만 고치면 조건 목록·요약·통과
  판정·마크다운이 따라온다(테스트가 전파를 고정). `suffix`=None 이면 prompt 모드에, `cue`=None
  이면 continue 모드에 안 들어가고 ""은 기준선(아무것도 안 붙인다), `rx`=None 은 대조군이다.
  `suffix`/`rx` 에 리터럴 "{CAND}" 가 있으면 문제마다 다른 후보값을 실행 시점에 채운다
  (`act_prompt`/`complied` 의 `cand=` 인자 — verification_first/verification_first_random).

통과 규칙(찍는다): CONTENT-EFFECT PASS ⇔ 어떤 행위가 (act − filler) 의 95% CI 가 0 을 제외
  ∧ 평균 ≥ +0.03 ∧ `complied_rate` ≥ .5 ∧ `complied_rate − complied_rate(REF[mode])` ≥ .20
  ∧ `trunc_rate` ≤ .20.
  ★compliance 를 규칙에 넣는 이유: 준수율이 0 에 가까우면 그 조건은 **아무것도 시험하지 않은
    것**이고, 그때의 Δ≈0 은 «행위가 무용하다»가 아니라 «지시가 안 닿았다»다.
  ★0915 갱신(compliance 를 절대값이 아니라 **차분**으로 읽는다): 지시 없는 기준(REF[mode]:
    plain/blind)의 생성물도 그 행위의 정규식을 이미 자연발생적으로 맞힌다 — 실측 substitute
    .353 · stepcheck .323 · recompute .278 · constraint .190 · magnitude .142 · special_case
    .107 · plan .007. 절대 `complied_rate ≥ .5` 만으로는 «지시가 행동을 거의 안 바꿨는데도
    통과»할 수 있다. `complied_rate_ref` 는 REF[mode] 의 생성물에 **그 행위의 정규식을 그대로**
    적용해 잰다(같은 잣대, 다른 조건의 텍스트) — 그래서 기준 대비 **+.20 이상**을 추가로 요구한다.
  ★TOKEN-COST 도 찍는다 — 기준 대비 평균 응답 토큰이 50% 넘게 늘어난 승자에는 `[COMPUTE-
    BOUGHT]`. 토큰으로만 산 이득을 «내용의 효과»로 읽지 않기 위해서다.

두 단계 — **다중비교를 통제한다(이게 핵심이다)**: 행위 ~12개를 전부 filler 와 α=.05 로 비교하면
  우연히 이기는 것이 절반 가까이 나온다. `--stage {screen,confirm}` 이 문제를 `--split_seed`
  (기본 11)로 group_id 를 해시해 **겹치지 않는 절반**으로 가른다(테스트가 겹침 없음·재현성을
  고정한다).
  · `--stage screen` — 그 절반에서 **모든** 행위를 `K=4` 로 돌려 (act − control) 순위만 낸다.
    **PASS/FAIL 을 찍지 않는다** — 마크다운은 `SCREEN — 순위만, 판정 없음`, json 에는 통과
    판정 키 자체가 없다(False 로도 안 남긴다).
  · `--stage confirm` — screen 순위를 보고 고른 `--acts a,b,c`(+ 기준·제어)만 나머지 절반에서
    `K=8` 로 돌린다. 여기서만 판정하고, 확정한 acts 수만큼 **Holm step-down**(짝지은 부호검정
    p)을 보정해 「CI 가 0 제외 ∧ 평균 ≥ +0.03 ∧ complied ≥ .5 ∧ trunc ≤ .20 ∧ Holm-adjusted
    p < .05」인 것만 승자로 찍는다. 보정 전엔 통과했을 행위가 보정 후 떨어질 수 있다(그게
    요점이다).

사용(예) — screen 먼저, 그 순위를 보고 confirm 의 --acts 를 채운다:
  python scripts/local/math_meta_content_gate.py --mode prompt --stage screen \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --rollouts /hdd_data/seungpil/scratch/eval/math500_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 --variant math_opt \
      --max_problems 120 --seed 11 --split_seed 11 --max_tokens 8192 --gpu_util 0.45 \
      --out_dir /hdd_data/seungpil/scratch/eval/meta_content_prompt_screen

  python scripts/local/math_meta_content_gate.py --mode prompt --stage confirm \
      --acts verification_first,substitute,magnitude \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --rollouts /hdd_data/seungpil/scratch/eval/math500_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 --variant math_opt \
      --max_problems 120 --seed 11 --split_seed 11 --max_tokens 8192 --gpu_util 0.45 \
      --out_dir /hdd_data/seungpil/scratch/eval/meta_content_prompt_confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import (  # noqa: E402  (선별·통계 정의를 한 곳에 둔다)
    bootstrap_ci, select_wrong_rollouts, sign_test_p,
)
from math_plan_gate import select_mixed_problems  # noqa: E402  (MIXED 문제 선별 규약 재사용)
from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, grade_math, last_boxed, selftest_math_verify,
)

_NAN = float("nan")
PASS_DELTA = 0.03          # (act − filler) 평균의 하한
MIN_COMPLIED = 0.50        # 준수율 하한 — 이 아래면 그 조건은 아무것도 시험하지 않았다
MIN_COMPLIED_DELTA = 0.20  # complied_rate − complied_rate(REF[mode]) 하한(0915, 절대값만으론 약하다)
MAX_TRUNC = 0.20           # 조건별 절단율 상한(넘으면 [BUDGET?])
COMPUTE_RATIO = 1.50       # 기준 대비 응답 토큰이 이 배를 넘으면 [COMPUTE-BOUGHT]
PLAN_RX_HEAD_CHARS = 600   # `plan` 준수는 응답 서두에서만 본다(아래 주석 참조)
CAND_LO, CAND_HI = 0, 999  # verification_first_random 의 "그럴듯한 범위"(무차원 정수 답 코퍼스)
STAGE_K = {"screen": 4, "confirm": 8}   # --k 를 안 주면 --stage 가 정한다


class Act(NamedTuple):
    """한 조건. suffix/cue 가 None 이면 그 모드에 안 들어가고 ""이면 아무것도 안 붙인다
    (기준선). rx=None 은 «행위를 이름 부르지 않는» 기준·대조군이라는 뜻이다. suffix/rx 에
    리터럴 "{CAND}" 가 있으면 문제마다 다른 후보값이 필요하다(act_prompt/complied 의 cand=)."""
    suffix: str | None      # prompt 모드: 사용자 턴 끝에 붙는 접미
    cue: str | None         # continue 모드: 열린 assistant 턴 끝에 붙는 씨앗 문장
    rx: str | None          # 준수 프록시 정규식(re.I)


# ★★ 조건의 단일 진실 원천 — 문장을 바꾸려면 여기만 고친다. 준수 정규식은 «그 행위의 어휘가
#   나타났는가»만 본다(쓰고 안 해도 1) — 낮을 때만 «지시가 안 닿았다»는 증거로 읽는다.
#   ★모든 항목에 출처 주석을 단다(arXiv id, 없으면 "no prior") — 0915 문헌 조사 갱신. ★★
META_ACTS: dict[str, Act] = {
    # ── 기준·대조군(rx=None): 어조는 메타인지적이되 어떤 행위도 이름 부르지 않는다.
    #   넷 다 — no prior(행위를 이름 부르지 않는 설계 그 자체이므로 문헌 근거가 성립하지 않는다).
    "plain": Act(suffix="", cue=None, rx=None),     # prompt 기준선(바이트 동일)
    "blind": Act(suffix=None, cue="", rx=None),     # continue 기준선(눈감고 새로 풀기)
    "filler": Act(suffix="\n\nBe careful and think it through before answering.", cue=None,
                  rx=None),
    "filler_continue": Act(suffix=None, cue="\n\nLet me look at this once more.", rx=None),
    # ── 행위(rx 있음) ───────────────────────────────────────────────────────────
    # 대입: 답을 원 문제에 넣어 조건이 성립하는지 본다(비용 비대칭이 가장 큰 행위). — no prior
    #   rx: substitut* / plug (it|back|in) / "check|verify … condition|constraint"
    "substitute": Act(
        suffix=("\n\nAfter you reach an answer, substitute it back into the original problem "
                "statement and check that every stated condition holds. If it fails, solve "
                "again."),
        cue=("\n\nLet me substitute this answer back into the original problem statement and "
             "check that every stated condition holds."),
        rx=r"(substitut\w*|plug\w*\s+(?:it|this|that|them|back|in\b)|"
           r"(?:check|verif\w+|confirm\w*)[^.\n]{0,60}(?:condition|constraint)s?)"),
    # 크기·부호: 계산 전에 기대 부호·규모를 말하고 계산 후 대조한다. — no prior
    #   rx: sign / magnitude / ballpark / rough size|estimate / "expect … to be|around|between"
    "magnitude": Act(
        suffix=("\n\nBefore computing, state the expected sign and rough size of the answer. "
                "After computing, check the answer against that expectation."),
        cue="\n\nLet me check this answer against the sign and rough size I should expect.",
        rx=r"(\bsigns?\b|magnitude|ballpark|rough(?:ly)?\s+(?:size|estimate|value)|"
           r"order of magnitude|expect\w*[^.\n]{0,40}(?:to be|around|between|positive|negative))"),
    # 특수한 경우: 작은 값·경계·퇴화 경우로 답을 검사한다. — no prior
    #   rx: special|edge case / boundary case / degenerate / sanity-check / "try|test … small|simple"
    "special_case": Act(
        suffix=("\n\nAfter you reach an answer, test it on a simple special case of the "
                "problem (a small value, a boundary, or a degenerate case) and check "
                "consistency."),
        cue=("\n\nLet me test this answer on a simple special case of the problem and check "
             "consistency."),
        rx=r"(special case|edge case|degenerate|boundary case|sanity[- ]check|"
           r"(?:try|test)\w*[^.\n]{0,40}(?:small|simple|specific)\s+(?:value|case|example|number))"),
    # 제약 재진술: 문제가 무엇을 요구하고 무엇을 제약하는지 먼저 다시 읽는다. — no prior
    #   rx: restat* / "the problem asks|requires" / "we are asked" / "constraints:" / "conditions given"
    "constraint": Act(
        suffix=("\n\nFirst restate exactly what the problem asks for and every constraint it "
                "imposes, in one or two lines, then solve."),
        cue=("\n\nLet me restate exactly what the problem asks for and every constraint it "
             "imposes, and check my answer against that."),
        rx=r"(restat\w+|the problem (?:asks|requires|states|wants)|we are asked|"
           r"what (?:is|we are) (?:asked|being asked|looking for)|constraints?\s*[:\-]|"
           r"conditions?\s+(?:given|imposed|stated))"),
    # 검증-먼저: 그럴듯한 오답 후보 {CAND} 를 주고, 먼저 그게 맞는지 검증한 뒤 스스로 풀게
    #   한다. provenance: **arXiv:2511.21734** — Qwen2.5-1.5B/3B·Llama3.1-8B-Instruct 에서
    #   실측된, 이 목록에서 유일하게 우리 모델급에 문헌 근거가 있는 행위. {CAND} 는 형제
    #   롤아웃에서 뽑은 **다수결이 아닌 오답**(`pick_non_plurality_wrong`) — act_prompt/
    #   complied 가 실행 시점에 문제별로 채운다(문제마다 값이 다르므로 rx 도 그때그때 컴파일).
    #   rx: "candidate/proposed/possible/given answer ... is/was (not) correct/wrong" 또는
    #   "verify/check ... candidate answer" 또는 후보값 자체를 이름 불러 "<CAND> is wrong" 식
    #   재계산 — "말했다"(언급)가 아니라 "그게 맞는지 아닌지에 대한 진술"을 요구한다.
    "verification_first": Act(
        suffix=("\n\nA possible answer to this problem is {CAND}. First verify whether that "
                "answer is correct, then solve the problem and give your own final answer in "
                "\\boxed{}."),
        cue=None,
        rx=(r"((?:the\s+)?(?:candidate|proposed|possible|given)\s+answer[^.\n]{0,40}"
            r"(?:is|was|turns out to be)\s+(?:not\s+)?(?:correct|right|wrong|incorrect)|"
            r"(?:verify|verifying|check(?:ing)?|confirm\w*)[^.\n]{0,50}"
            r"(?:candidate|proposed|possible|given)\s+answer|"
            r"\b{CAND}\b[^.\n]{0,40}(?:is|was|turns out to be)\s+(?:not\s+)?"
            r"(?:correct|right|wrong|incorrect))")),
    # 검증-먼저(무작위 대조): 문장은 위와 **글자 그대로 동일**, 후보만 문제와 무관한 무작위
    #   정수([0,999], `pick_random_wrong_cand`). provenance: **arXiv:2511.21734** — 같은
    #   논문이 보고하는 **공개된 대조**: trivial/무작위 후보도 진짜(그럴듯한) 후보만큼 효과가
    #   있었다는 결과다. `verification_first` 가 `filler` 는 이기고 이 팔은 못 이기면, 정직한
    #   해석은 «내용이 조향했다» 가 아니라 «후보가 주어지면 검증한다»(행위 자체의 효과)다.
    #   rx: verification_first 와 **같다**(같은 종류의 "검증했다는 진술"을 요구한다).
    "verification_first_random": Act(
        suffix=("\n\nA possible answer to this problem is {CAND}. First verify whether that "
                "answer is correct, then solve the problem and give your own final answer in "
                "\\boxed{}."),
        cue=None,
        rx=(r"((?:the\s+)?(?:candidate|proposed|possible|given)\s+answer[^.\n]{0,40}"
            r"(?:is|was|turns out to be)\s+(?:not\s+)?(?:correct|right|wrong|incorrect)|"
            r"(?:verify|verifying|check(?:ing)?|confirm\w*)[^.\n]{0,50}"
            r"(?:candidate|proposed|possible|given)\s+answer|"
            r"\b{CAND}\b[^.\n]{0,40}(?:is|was|turns out to be)\s+(?:not\s+)?"
            r"(?:correct|right|wrong|incorrect))")),
    # 역방향 조건-마스킹: 답을 낸 뒤, 그 답이 맞다고 가정하고 문제에 주어진 수량 하나를
    #   거꾸로 복원해 원 문장의 값과 맞는지 본다. provenance: **arXiv:2212.09561**(Weng et
    #   al., self-verification, backward reasoning) / **arXiv:2312.06867** — 둘 다 실제 수학
    #   이득을 보고하지만, 둘 다 ~7B 미만 모델에서는 일관되지 않는다고도 보고한다(우리 4B 는
    #   그 구간 — PAL 경고와 별개로 null 이 나올 수 있다는 사전 기대치를 낮춰 둔다).
    #   rx: "work(ing) backwards" / "recover … quantity|value|number" / "matches … stated value"
    "backward_mask": Act(
        suffix=("\n\nSolve the problem. Then, assuming your answer is correct, work backwards "
                "to recover one numeric quantity given in the problem statement, and check it "
                "matches the stated value."),
        cue=None,
        rx=(r"(work(?:ing)?\s+backwards?|recover\w*[^.\n]{0,40}(?:quantity|value|number)|"
            r"(?:matches?|matched|consistent with|agrees?\s+with)[^.\n]{0,40}"
            r"(?:the\s+)?(?:stated|given)\s+(?:value|quantity)|back[- ]?substitut\w*)")),
    # 계획 한 줄: 풀기 전에 접근을 한 줄로 말한다(math_plan_gate 의 값싼 판). — no prior
    #   rx: approach|plan|strategy|idea|method — **서두 600자 안에서만**(끝에서 "another
    #   approach"라 쓴 것을 «풀기 전 계획»으로 세지 않기 위해)
    "plan": Act(suffix="\n\nBriefly state your approach in one line before solving.", cue=None,
                rx=r"(\bapproach\b|\bplan\b|\bstrategy\b|\bidea\b|\bmethod\b)"),
    # 단계 검산: 큰 단계마다 그 단계의 산술을 확인한다. — no prior
    #   rx: double-check / "check|verify … arithmetic|calculation|computation|this step|each step"
    "stepcheck": Act(
        suffix="\n\nAfter each major step, verify that step's arithmetic before continuing.",
        cue=None,
        rx=r"(double[- ]?check|(?:check|verif\w+|confirm\w*)[^.\n]{0,40}"
           r"(?:arithmetic|calculation|computation|this step|each step|the math))"),
    # 재유도(비싼 대조 케이스): 다른 방법으로 다시 풀고 두 답을 비교한다. — no prior
    #   rx: "second|another|different|alternative method|way|approach" / re-solve / cross-check
    "recompute": Act(
        suffix=("\n\nAfter you reach an answer, solve the problem a second time by a "
                "different method and compare the two answers."),
        cue=("\n\nLet me solve this problem a second time by a different method and compare "
             "the two answers."),
        rx=r"((?:second|another|different|alternative)\s+(?:method|way|approach|solution)|"
           r"re-?solv\w+|cross[- ]?check|solve (?:it|this) again|alternatively)"),
    # ── 길이-매치 대조(rx 있음 — 준수율은 잰다 — 이지만 승자 후보에선 뺀다, 아래 NOT_WINNERS):
    #   Gandhi 외(arXiv:2503.01307) 의 «length-matched placeholder» — 신중함이 왜 중요한지 세
    #   문장으로 반추하라고만 시킨다. 어떤 행위도 이름 부르지 않고 이 문제에 대한 정보도 안
    #   주지만, 토큰 수와 메타인지적 **어조**는 늘린다. 그 논문은 이 대조를 **null**(효과 없음)
    #   로 보고한다. ★목적: 어떤 행위든 (act − filler) 는 양수인데 (act − pad_length) 는 0
    #   이하면, 그 이득은 «내용»이 아니라 «토큰 분량 + 메타 어조»로 산 것이다(요약이
    #   `[TOKEN-CONFOUND?]` 로 찍는다). 통과 규칙은 그대로 filler(`CONTROL`) 기준이다 —
    #   pad_length 는 진단용 둘째 잣대일 뿐, 그 자체가 이름 부르는 행위가 아니므로(CONTROLS 와
    #   같은 이유) `NOT_WINNERS` 로 승자 후보에서 뺀다 — 그런데 filler/blind 와 달리 준수(길이를
    #   실제로 채웠는가) 는 재야 하므로 rx 는 둔다(그래서 `CONTROLS`(rx=None) 자체엔 안 든다).
    #   rx: «수식·숫자가 나오기 전 비수학 산문 문장 3개 이상» 근사 — **단순한 상한 추정**이다.
    #   그 세 문장이 실제로 «신중함에 대한 반추»인지는 못 잰다. 문서 맨 앞부터 숫자나 역슬래시
    #   (수식 마커)가 나오기 전에 문장부호(. ! ?)가 3번 이상 나오는지만 본다.
    "pad_length": Act(
        suffix=("\n\nBefore solving, write three sentences of general reflection on why "
                "careful work matters in mathematics. Then solve the problem."),
        cue=None,
        rx=r"^[^\d\\]*[.!?][^\d\\]*[.!?][^\d\\]*[.!?]"),
}

CONTROLS = frozenset(n for n, a in META_ACTS.items() if a.rx is None)  # 준수율이 nan(행위가 없다)
# ★승자 후보에서 빼는 이름들 — CONTROLS(행위를 이름 부르지 않는 rx=None 기준·대조군) 에
#   `pad_length` 를 더한다: rx 가 있어 준수율은 재지만, 그 자체도 «내용 없는 진단용 대조»이지
#   시험 대상 행위가 아니다(F1). `acts_ranked`/통과 규칙 양쪽에서 이 집합을 뺀다.
NOT_WINNERS = CONTROLS | {"pad_length"}
REF = {"prompt": "plain", "continue": "blind"}                   # 매치드 앵커
CONTROL = {"prompt": "filler", "continue": "filler_continue"}    # 내용 없는 대조
# ★0915b 수정(F3): complied_rate_ref(«지시 없이도 이미 자연발생하는 준수율»)를 잴 때 쓰는
#   기준 조건 — **REF[mode] 가 아니다**. continue 모드에서 REF["continue"] = "blind" 는 눈감고
#   **처음부터 새로 푸는 4k 토큰짜리 완결된 풀이**인 반면, `complied` 자체는 **이어쓴 부분만**
#   (수십~수백 토큰)을 본다 — 그러니 "blind" 를 잣대로 쓰면 comp_ref 가 훨씬 긴 텍스트에서 잰
#   값이라 구조적으로 comp(짧은 이어쓰기) 보다 높게 나오고, `complied − complied_ref`
#   (MIN_COMPLIED_DELTA 규칙)는 **원리상 음수로 시작**해 continue 모드에서 절대 통과할 수
#   없었다. 수리: continue 모드는 CONTROL[mode](`filler_continue`, 같은 길이의 이어쓰기 대조)
#   를 잣대로 쓴다 — 텍스트 길이·문맥이 `complied` 를 재는 대상과 매치된다. prompt 모드는
#   그대로 REF[mode](`plain`, 매치드 앵커) — 거긴 모든 조건이 처음부터 푸는 완결된 응답이라
#   길이 비대칭이 없다.
COMPLIANCE_REF = {"prompt": REF["prompt"], "continue": CONTROL["continue"]}


def prompt_conds() -> tuple[str, ...]:
    """prompt 모드 조건 — META_ACTS 에서 suffix 가 있는 항목(정의 순서)."""
    return tuple(n for n, a in META_ACTS.items() if a.suffix is not None)


def continue_conds() -> tuple[str, ...]:
    """continue 모드 조건 — META_ACTS 에서 cue 가 있는 항목(정의 순서)."""
    return tuple(n for n, a in META_ACTS.items() if a.cue is not None)


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────────
def act_prompt(tok, variant: str, problem: str, name: str, *, cand=None) -> str:
    """prompt 모드 한 조건. suffix 가 "" 이면 `render_generation_prompt` 와 **바이트 동일**,
    아니면 **사용자 턴 끝에 접미만** 붙는다 — 조건 사이의 유일한 차이가 그 문장이어야 한다
    (system 문구를 새로 만들면 «행위의 효과»와 «프롬프트가 달라진 효과»가 섞인다).
    ★suffix 에 리터럴 "{CAND}" 가 있으면(verification_first/verification_first_random) `cand`
      를 **문자열 치환**한다 — `str.format` 이 아니라 `str.replace` 인 이유는 접미 자체에
      "\\boxed{}" 라는 빈 중괄호가 들어 있어서다(포맷 문자열로 읽으면 그 자체가 플레이스홀더로
      해석돼 깨진다). cand 가 없으면 fail-loud(조용히 «{CAND}» 문자열을 프롬프트에 흘리지 않는다)."""
    suf = META_ACTS[name].suffix
    if suf is None:
        raise KeyError(f"[CONTENT] {name!r} 은 prompt 모드 조건이 아니다(suffix=None)")
    if not suf:
        return render_generation_prompt(tok, variant, problem)
    if "{CAND}" in suf:
        if cand is None:
            raise KeyError(f"[CONTENT] {name!r} 은 candidate 값이 필요하다(cand=None)")
        suf = suf.replace("{CAND}", str(cand))
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + suf}
    return render_chat_messages(tok, msgs)


def act_continuation(tok, variant: str, problem: str, text: str, name: str) -> str:
    r"""continue 모드 한 조건. cue 가 "" 이면 `blind`(= render_generation_prompt 바이트 동일,
    눈감고 새로 푸는 매치드 앵커), 아니면 **열린 assistant 턴**을 원 풀이 + cue 로 이어 쓴다.
    ★규약은 `math_activation_gate.wait_prompt` 가 실 토크나이저로 검증한 것과 같다:
    `render_generation_prompt` 이 `add_generation_prompt=True` 로 assistant 턴을 연 채 끝나므로
    뒤에 문자열을 그냥 이어 붙이면 `continue_final_message=True` 렌더와 바이트 동일하다.
    cue 뒤에 턴 종료 표식이 붙지 않아야 한다(테스트가 고정). 후보값이 필요한 행위는 continue
    모드에 없다(META_ACTS 의 cue=None) — 그래서 이 함수에 cand 인자가 없다."""
    cue = META_ACTS[name].cue
    if cue is None:
        raise KeyError(f"[CONTENT] {name!r} 은 continue 모드 조건이 아니다(cue=None)")
    if not cue:
        return render_generation_prompt(tok, variant, problem)
    return render_generation_prompt(tok, variant, problem) + (text or "") + cue


_RX_CACHE: dict[str, re.Pattern] = {}


def complied(name: str, gen_text: str, *, cand=None) -> float:
    """생성물에 그 행위의 흔적이 있는가 → 1/0, 대조군(rx=None)은 nan.
    ★프록시다: «어휘가 나타났는가»만 본다. 통과 규칙이 이 양을 «준수율이 낮으면 그 조건은
      아무것도 시험하지 않았다»는 **한 방향으로만** 쓰는 이유가 그것이다.
    ★rx 에 리터럴 "{CAND}" 가 있으면(verification_first/verification_first_random) 그 단위의
      후보값을 `re.escape` 해 넣고 **그때그때 컴파일**한다(단위마다 후보가 달라 이름으로 캐시할
      수 없다) — cand 가 없으면 fail-loud(act_prompt 와 같은 계약)."""
    pat = META_ACTS[name].rx
    if pat is None:
        return _NAN
    if "{CAND}" in pat:
        if cand is None:
            raise KeyError(f"[CONTENT] {name!r} 의 준수 판정은 candidate 값이 필요하다(cand=None)")
        rx = re.compile(pat.replace("{CAND}", re.escape(str(cand))), re.I)
    else:
        rx = _RX_CACHE.get(name)
        if rx is None or rx.pattern != pat:
            rx = _RX_CACHE[name] = re.compile(pat, re.I)
    body = (gen_text or "")
    return 1.0 if rx.search(body[:PLAN_RX_HEAD_CHARS] if name == "plan" else body) else 0.0


def grade_continuation(gen_text: str, prefix_text: str, gold: str) -> tuple[int, int]:
    r"""continue 모드 채점 → (정답 여부, 새 \boxed 가 없었는가). ★**이어쓴 부분만** 본다 —
    접두의 \boxed 를 같이 보면 원래 오답이 계속 채점에 들어와 조건 사이 차이가 묽어진다.
    새 \boxed 가 없으면 답은 접두 그대로이므로 접두로 채점한다(이 모집단에선 정의상 오답)."""
    if last_boxed(gen_text or ""):
        return grade_math(gen_text, gold), 0
    return grade_math(prefix_text or "", gold), 1


# ── verification_first 후보 선정 ────────────────────────────────────────────────
def cluster_answers(answers: Sequence[str]) -> list[list[int]]:
    r"""답 문자열 목록 → math_verify 동치로 묶은 인덱스 군집(등장 순서 보존, 결정적).
    빈 문자열(\boxed 없음)은 통째로 뺀다 — «값이 없다»는 어떤 값과도 같지 않다."""
    clusters: list[list[int]] = []
    reps: list[str] = []
    for i, a in enumerate(answers):
        if not (a or "").strip():
            continue
        for ci, rep in enumerate(reps):
            if answers_equivalent(rep, a):
                clusters[ci].append(i)
                break
        else:
            reps.append(a)
            clusters.append([i])
    return clusters


def pick_non_plurality_wrong(answers: Sequence[str], gold: str) -> str | None:
    r"""형제 롤아웃(같은 문제, K 개)의 답에서 **다수결이 아닌 오답** 하나를 고른다
    (verification_first 의 "그럴듯한 오답 후보" — arXiv:2511.21734).

    ★정답 누출 방지(0915 doubt): «다수결이 아니다»만으로는 정답이 아니라는 보장이 안 된다 —
      MIXED 문제는 다수가 틀리고 소수가 맞힐 수도 있다. 그래서 다수결이 아닌 군집 중 **gold 와
      동치가 아닌** 것만 후보로 남기고, 그중 가장 큰 군집(가장 «그럴듯한» 오답, 동률이면 먼저
      나온 군집)을 고른다. 남는 후보가 없으면(= 전원이 한 값에 합의했거나, 다수결이 아닌 군집이
      전부 gold — 즉 다수가 틀리고 유일한 소수가 정답인 경우) **None** 을 돌려 그 문제에서 이
      act 를 스킵한다(호출부가 `n_skipped_no_cand` 로 센다). 다수결 자체가 오답이어도(더 흔한
      MIXED 모양 — 다수가 틀리고 소수가 맞힌다) 문제없다: 다수결은 애초에 후보에서 빠진다."""
    clusters = cluster_answers(list(answers))
    if len(clusters) < 2:
        return None                          # 전원 합의(또는 답이 하나도 없다) — 스킵
    sizes = [len(c) for c in clusters]
    plurality_ci = max(range(len(clusters)), key=lambda i: sizes[i])
    wrong_non_plurality = [ci for ci in range(len(clusters))
                           if ci != plurality_ci
                           and not answers_equivalent(gold, answers[clusters[ci][0]])]
    if not wrong_non_plurality:
        return None
    best_ci = max(wrong_non_plurality, key=lambda ci: (sizes[ci], -ci))
    return answers[clusters[best_ci][0]]


def pick_random_wrong_cand(gold: str, rng: random.Random, *, lo: int = CAND_LO,
                           hi: int = CAND_HI, tries: int = 10) -> str | None:
    """무작위 정수 오답 후보 — 문제와 무관(verification_first_random 의 **공개된 대조**,
    arXiv:2511.21734). gold 와 우연히 같으면(정답 누출) 다시 뽑는다 — tries 번 다 걸리면
    (사실상 없다) None 을 돌려 스킵한다."""
    for _ in range(tries):
        c = rng.randint(lo, hi)
        if not answers_equivalent(gold, str(c)):
            return str(c)
    return None


# ── screen/confirm 문제 분할 ────────────────────────────────────────────────────
def split_bucket(group_id: str, split_seed: int) -> str:
    """결정적 50/50 분할 — (split_seed, group_id) 를 md5 해시해 "screen"/"confirm" 을 정한다.
    ★같은 입력이면 항상 같은 절반(재현성 — screen 과 confirm 을 서로 다른 프로세스·다른
      시각에 돌려도 겹치지 않는다). md5 라 group_id 접두 상관에 안전하다(테스트가 겹침 없음·
      안정성을 고정한다)."""
    h = hashlib.md5(f"{split_seed}:{group_id}".encode()).hexdigest()
    return "screen" if int(h[:8], 16) % 2 == 0 else "confirm"


def _p(vals: Sequence[float]) -> float:
    v = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return (sum(v) / len(v)) if v else _NAN


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def summarize(recs: Sequence[dict], conds: Sequence[str], *, ref: str, control: str,
              k: int = 0, seed: int = 0, n_boot: int = 2000, stage: str | None = None,
              confirm_acts: Sequence[str] | None = None) -> dict:
    """단위(문제 또는 오답 롤아웃)별 기록 → 조건별 지표 + 짝지은 Δ + 순위 (+ 통과 판정).
    recs[i]["cond"][c] = {"acc","trunc","tokens","complied","no_new_boxed"} — **`ref`·`control`
    이 있는 단위만** 넘기면 된다(0915 수정 — 예전엔 «모든 조건이 다 있는 단위만»을 요구해
    verification_first 처럼 후보가 없어 스킵된 조건 하나 때문에 83→55개 단위가 통째로
    버려졌다). `recs[i]["cond"]`에 없는 조건 `c` 는 그 단위가 `c` 를 갖지 않는다는 뜻이고,
    각 조건별 통계·각 짝지은 Δ 는 **그 특정 항목이 있는 단위들만**으로 계산한다(`n_units_cond`/
    `n_paired_ref`/`n_paired_control` 로 몇 개인지 보고 — 아래 markdown 의 `n` 열).
    ★"complied_ref"(옵션) 가 있으면(0915 갱신) — 준수 기준 조건(0915b: prompt 는 REF[mode],
      continue 는 CONTROL[mode] — `COMPLIANCE_REF` 참조) 생성물에 그 행위의 정규식을 그대로
      적용한 준수율. 없으면(구 버전 recs) nan 으로 떨어져 comp-vs-ref 조건이 항상 FAIL 이다.

    `stage`: None(레거시, 전체 판정) · "screen"(순위만, 통과 판정 키 자체를 안 만든다) ·
    "confirm"(`confirm_acts` 만 Holm 보정으로 판정) — 위 두 단계 설명은 모듈 docstring 참조."""
    fin = math.isfinite

    def _has(r: dict, c: str) -> bool:
        return c in r["cond"]

    acc, trunc, tokens, comp, comp_ref, nnb, n_units_cond = {}, {}, {}, {}, {}, {}, {}
    d_ref, d_ctl, s_ref, s_ctl, n_paired_ref, n_paired_ctl = {}, {}, {}, {}, {}, {}
    for i, c in enumerate(conds):
        rc = [r for r in recs if _has(r, c)]
        n_units_cond[c] = len(rc)
        acc[c] = bootstrap_ci([r["cond"][c]["acc"] for r in rc], seed=seed + i, n_boot=n_boot)
        trunc[c] = _p([r["cond"][c]["trunc"] for r in rc])
        tokens[c] = _p([r["cond"][c]["tokens"] for r in rc])
        comp[c] = _p([r["cond"][c]["complied"] for r in rc])
        comp_ref[c] = _p([r["cond"][c].get("complied_ref", _NAN) for r in rc])
        nnb[c] = _p([r["cond"][c].get("no_new_boxed", _NAN) for r in rc])
        rc_ref = [r for r in rc if _has(r, ref)]
        rc_ctl = [r for r in rc if _has(r, control)]
        n_paired_ref[c] = len(rc_ref)
        n_paired_ctl[c] = len(rc_ctl)
        dr = [r["cond"][c]["acc"] - r["cond"][ref]["acc"] for r in rc_ref]
        dc = [r["cond"][c]["acc"] - r["cond"][control]["acc"] for r in rc_ctl]
        d_ref[c] = bootstrap_ci(dr, seed=seed + 100 + i, n_boot=n_boot)
        d_ctl[c] = bootstrap_ci(dc, seed=seed + 200 + i, n_boot=n_boot)
        s_ref[c], s_ctl[c] = sign_test_p(dr), sign_test_p(dc)
    comp_delta = {c: (comp[c] - comp_ref[c] if fin(comp[c]) and fin(comp_ref[c]) else _NAN)
                 for c in conds}
    tok_ref = tokens.get(ref, _NAN)
    # ★pad_length(F1, 진단용 둘째 잣대) — 있을 때만 (act − pad_length) 를 (act − filler) 옆에
    #   낸다. 통과 규칙에는 안 넣는다(그대로 filler=`CONTROL` 기준) — 아래 token_confound 만 찍는다.
    pad_present = "pad_length" in conds
    d_pad: dict[str, dict] = {}
    s_pad: dict[str, float] = {}
    if pad_present:
        for i, c in enumerate(conds):
            dp = [r["cond"][c]["acc"] - r["cond"]["pad_length"]["acc"]
                  for r in recs if _has(r, c) and _has(r, "pad_length")]
            d_pad[c] = bootstrap_ci(dp, seed=seed + 300 + i, n_boot=n_boot)
            s_pad[c] = sign_test_p(dp)
    out = {
        # ★n_units = ref∧control 둘 다 있는 단위 수(호출부가 그 최소 조건만으로 recs 를 넘긴다).
        #   조건별 실제 표본수는 n_units_cond/n_paired_ref/n_paired_control 를 본다(0915 수정).
        "n_units": sum(1 for r in recs if _has(r, ref) and _has(r, control)),
        "n_units_cond": n_units_cond, "n_paired_ref": n_paired_ref,
        "n_paired_control": n_paired_ctl,
        "k": k, "ref": ref, "control": control, "conds": list(conds),
        # (act − control) 내림차순, nan 은 맨 뒤. NOT_WINNERS(대조군 + pad_length)는 승자 후보가
        # 아니다.
        "acts_ranked": sorted(
            [c for c in conds if c not in NOT_WINNERS],
            key=lambda c: -d_ctl[c]["mean"] if _finite(d_ctl[c], "mean") else math.inf),
        "acc": acc, "trunc_rate": trunc, "resp_tokens": tokens, "complied_rate": comp,
        "complied_rate_ref": comp_ref, "complied_rate_delta": comp_delta,
        "no_new_boxed_rate": nnb, "delta_vs_ref": d_ref, "delta_vs_control": d_ctl,
        "sign_p_vs_ref": s_ref, "sign_p_vs_control": s_ctl,
        "token_cost_vs_ref": {c: (tokens[c] - tok_ref if fin(tokens[c]) and fin(tok_ref)
                                  else _NAN) for c in conds},
        "token_ratio_vs_ref": {c: (tokens[c] / tok_ref if fin(tokens[c]) and fin(tok_ref)
                                   and tok_ref > 0 else _NAN) for c in conds},
        "budget_flag": [c for c in conds if fin(trunc[c]) and trunc[c] > MAX_TRUNC],
    }
    if pad_present:
        out["delta_vs_pad_length"], out["sign_p_vs_pad_length"] = d_pad, s_pad
        # ★[TOKEN-CONFOUND?]: (act − filler) 는 양수인데 (act − pad_length) 는 0 이하인 행위 —
        #   그 이득이 내용이 아니라 토큰·어조로 산 것일 수 있다는 뜻. acts_ranked 가 이미
        #   pad_length 자신을 뺀 목록이라 자기 자신과 비교되는 일은 없다.
        out["token_confound"] = [
            c for c in out["acts_ranked"]
            if _finite(d_ctl[c], "mean") and d_ctl[c]["mean"] > 0
            and _finite(d_pad.get(c), "mean") and d_pad[c]["mean"] <= 0]
    if stage == "screen":
        # ★판정 키를 아예 안 만든다(False 로도 안 남긴다) — screen 은 순위만 내는 단계다.
        out["stage"] = "screen"
        return out
    if stage == "confirm":
        acts_c = [c for c in (confirm_acts or []) if c in conds and c not in NOT_WINNERS]
        ok, winners, holm_p = content_effect_pass_confirm(out, acts_c)
        out["stage"] = "confirm"
        out["confirm_acts"] = list(acts_c)
        out["holm_adjusted_p"] = holm_p
        out["pass_content_effect"] = int(ok)
        out["winners"] = winners
        out["compute_bought"] = [c for c in winners
                                 if fin(out["token_ratio_vs_ref"][c])
                                 and out["token_ratio_vs_ref"][c] > COMPUTE_RATIO]
        return out
    ok, winners = content_effect_pass(out)
    out["pass_content_effect"] = int(ok)
    out["winners"] = winners
    out["compute_bought"] = [c for c in winners
                             if fin(out["token_ratio_vs_ref"][c])
                             and out["token_ratio_vs_ref"][c] > COMPUTE_RATIO]
    return out


def content_effect_pass(summ: dict) -> tuple[bool, list[str]]:
    """CONTENT-EFFECT(레거시/무단계) — (act − control) CI 가 0 제외 ∧ 평균 ≥ +0.03 ∧
    complied ≥ .5 ∧ complied − complied_ref(REF[mode]) ≥ .20 ∧ trunc ≤ .20 인 행위가 하나라도
    있으면 통과. nan 은 전부 FAIL 이다.
    ★0915 갱신: 절대 준수율만으로는 «지시 없이도 이미 자연발생하는 행동»이 통과할 수 있다
      (실측 substitute .353 등) — 그래서 REF[mode](plain/blind) 대비 **+.20 이상**도 요구한다."""
    def _num(x) -> bool:      # nan·None·비수치는 전부 FAIL 쪽으로 떨어뜨린다
        return isinstance(x, (int, float)) and math.isfinite(x)

    winners: list[str] = []
    for c in summ.get("acts_ranked") or []:
        d = (summ.get("delta_vs_control") or {}).get(c) or {}
        comp = (summ.get("complied_rate") or {}).get(c, _NAN)
        comp_d = (summ.get("complied_rate_delta") or {}).get(c, _NAN)
        tr = (summ.get("trunc_rate") or {}).get(c, _NAN)
        if (_finite(d, "lo", "hi", "mean") and (d["lo"] > 0 or d["hi"] < 0)
                and d["mean"] >= PASS_DELTA and _num(comp) and comp >= MIN_COMPLIED
                and _num(comp_d) and comp_d >= MIN_COMPLIED_DELTA
                and _num(tr) and tr <= MAX_TRUNC):
            winners.append(c)
    return bool(winners), winners


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down 보정: 오름차순 정렬한 p_(1)≤…≤p_(m) 에 (m−i) 를 곱하고(0-기반 i), 단조가
    되도록 누적 최댓값을 취한다(표준 Holm-Bonferroni). 이름 → 보정된 p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def content_effect_pass_confirm(summ: dict, acts: Sequence[str]) -> tuple[bool, list[str], dict]:
    """CONFIRM 단계 통과 규칙 — `content_effect_pass` 와 같은 CI·효과크기·준수·절단 조건에
    **Holm 보정 짝지은 부호검정 p < .05** 를 더한다. `acts` 는 `--acts` 로 확정한 후보만
    (screen 순위를 보고 사람이 고른 것) — 그래서 보정 m 이 작게 유지된다. screen 에서 ~12개
    전부에 α=.05 를 그대로 쓰면 우연히 이기는 팔이 절반 가까이 나온다는 것이 이 함수가 따로
    있는 이유(다중비교 통제) — 보정 전엔 통과할 act 가 보정 후 FAIL 로 뒤집힐 수 있다."""
    def _num(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    raw_p = {c: (summ.get("sign_p_vs_control") or {}).get(c, 1.0) for c in acts}
    adj_p = holm_adjust(raw_p)
    winners: list[str] = []
    for c in acts:
        d = (summ.get("delta_vs_control") or {}).get(c) or {}
        comp = (summ.get("complied_rate") or {}).get(c, _NAN)
        comp_d = (summ.get("complied_rate_delta") or {}).get(c, _NAN)
        tr = (summ.get("trunc_rate") or {}).get(c, _NAN)
        p = adj_p.get(c, 1.0)
        if (_finite(d, "lo", "hi", "mean") and (d["lo"] > 0 or d["hi"] < 0)
                and d["mean"] >= PASS_DELTA and _num(comp) and comp >= MIN_COMPLIED
                and _num(comp_d) and comp_d >= MIN_COMPLIED_DELTA
                and _num(tr) and tr <= MAX_TRUNC and _num(p) and p < 0.05):
            winners.append(c)
    return bool(winners), winners, adj_p


def _f(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}]"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def to_markdown(summ: dict, *, mode: str, title: str = "pooled") -> str:
    ref, ctl = summ["ref"], summ["control"]
    stage_tag = f" · stage={summ['stage']}" if summ.get("stage") else ""
    pad = "pad_length" in summ["conds"]
    header = (f"| cond | n | acc | Δ vs {ref} | Δ vs {ctl} | sign_p(vs {ctl}) | complied | "
              "complied_ref | Δcomplied | trunc | tokens | Δtok |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    if pad:
        header += " Δ vs pad_length | sign_p(vs pad_length) |"
        sep += "---|---|"
    lines = [f"### {title} — n_units={summ['n_units']} · K={summ['k']} · mode={mode}{stage_tag}",
             "", "n = 그 행이 참여한 단위 수(짝 비교는 delta_vs_control 열 기준 — "
             "verification_first 처럼 후보가 없어 스킵된 단위가 있으면 다른 행보다 작을 수 있다).",
             "", header, sep]
    for c in summ["conds"]:
        n_c = (summ.get("n_paired_control") or {}).get(c, summ.get("n_units_cond", {}).get(c))
        row = (f"| {c} | {n_c} | {_f(summ['acc'][c])} | {_f(summ['delta_vs_ref'][c])} | "
               f"{_f(summ['delta_vs_control'][c])} | {_f(summ['sign_p_vs_control'][c])} | "
               f"{_f(summ['complied_rate'][c])} | {_f(summ['complied_rate_ref'][c])} | "
               f"{_f(summ['complied_rate_delta'][c])} | {_f(summ['trunc_rate'][c])} | "
               f"{_f(summ['resp_tokens'][c])} | {_f(summ['token_cost_vs_ref'][c])} |")
        if pad:
            row += (f" {_f(summ['delta_vs_pad_length'][c])} | "
                    f"{_f(summ['sign_p_vs_pad_length'][c])} |")
        lines.append(row)
    lines += ["", f"**RANK (act − {ctl})**: " + " > ".join(
        f"{c}({_f(summ['delta_vs_control'][c].get('mean'))})" for c in summ["acts_ranked"]), ""]
    if mode == "continue":
        lines += [f"no_new_boxed_rate: {_f(summ['no_new_boxed_rate'])}", "",
                  "**[NOTE]** continue 모드의 `complied` 는 그 행위의 어휘가 이어쓴 부분에 "
                  "**반복됐는가**만 본다 — cue 자체가 이미 프롬프트(열린 assistant 턴)에 있으므로, "
                  "이건 «그 행위를 수행했다»의 약한 프록시일 뿐 지시가 있었다는 사실과 크게 "
                  "다르지 않을 수 있다.", ""]
    if summ.get("stage") == "screen":
        lines.append(
            "**SCREEN — 순위만, 판정 없음** — 행위 전부를 α=.05 로 filler 와 비교하면 우연으로도 "
            "절반 가까이 «이긴다»(다중비교). 이 절반은 순위만 보고하고, PASS/FAIL 은 "
            "`--stage confirm`(나머지 절반 · Holm 보정)에서만 찍는다.")
    else:
        holm = (f" ∧ Holm-adjusted sign_p < .05(m={len(summ.get('confirm_acts') or [])})"
                if summ.get("stage") == "confirm" else "")
        lines.append(
            f"**CONTENT-EFFECT {'PASS' if summ.get('pass_content_effect') else 'FAIL'}** — "
            f"(act − {ctl}) CI 가 0 제외 ∧ 평균 ≥ +{PASS_DELTA} ∧ complied_rate ≥ {MIN_COMPLIED} "
            f"∧ complied_rate − complied_rate_ref ≥ {MIN_COMPLIED_DELTA} ∧ trunc_rate ≤ "
            f"{MAX_TRUNC}{holm}; winners={summ.get('winners')}")
        if summ.get("stage") == "confirm":
            lines.append(f"보정: Holm step-down(짝지은 부호검정 p, 확정 행위 "
                         f"{summ.get('confirm_acts')} 에서만) — adjusted p: "
                         f"{json.dumps(summ.get('holm_adjusted_p'), ensure_ascii=False)}")
    if summ.get("compute_bought"):
        lines.append(f"**[COMPUTE-BOUGHT]** {summ['compute_bought']} — 기준({ref}) 대비 응답 "
                     f"토큰이 {COMPUTE_RATIO}배를 넘는다(내용이 아니라 계산으로 산 이득일 수 있다)")
    if summ.get("budget_flag"):
        lines.append(f"**[BUDGET?]** trunc_rate > {MAX_TRUNC} 인 조건: {summ['budget_flag']} — "
                     "cd9 G4 처럼 절단률이 정확도 순서를 뒤집을 수 있다. 이 표를 그대로 읽지 말 것.")
    for c in summ.get("token_confound") or []:
        lines.append(f"[TOKEN-CONFOUND?] {c}")
    if summ.get("token_confound"):
        lines.append(f"— (act − {ctl}) 는 양수인데 (act − pad_length) 는 0 이하다: 그 이득은 "
                     "내용이 아니라 토큰·메타 어조로 산 것일 수 있다(Gandhi 외 arXiv:2503.01307).")
    lines.append("**[PAL] 말한 검산 ≠ 수행한 검산** (arXiv:2211.10435: 머릿속 시뮬레이션 23.2% vs "
                 "실제 인터프리터 실행 72.0%) — 이 표의 null 결과는 «검산을 말한» 값의 상한이지 "
                 "«검산을 수행한» 값을 재는 게 아니다.")
    return "\n".join(lines + [""])


def render_report(pooled: dict, per_source: dict, *, mode: str, meta: dict) -> str:
    lines = ["## math_meta_content_gate — 메타인지 «행위의 내용»이 조향하는가", "",
             f"mode={mode} · stage={meta.get('stage')} · model={meta.get('model_path')} · "
             f"variant={meta.get('variant')} · seed={meta.get('seed')} · "
             f"max_tokens={meta.get('max_tokens')} · sources={meta.get('sources')} · "
             f"n_generations={meta.get('n_generations')}", "",
             to_markdown(pooled, mode=mode, title="pooled")]
    lines += [to_markdown(s, mode=mode, title=f"source={t}") for t, s in per_source.items()]
    return "\n".join(lines)


def source_tag(path: str) -> str:
    """출처 표식 = 롤아웃 파일의 상위 디렉터리 이름(eval 산출물 규약). ★두 코퍼스가 group_id
    를 «g0»부터 각자 붙이므로(실측), 이 표식으로 네임스페이스를 나누지 않으면 두 파일의 서로
    다른 문제가 같은 단위로 뭉개진다."""
    return Path(path).resolve().parent.name or Path(path).name


def load_sources(paths: Sequence[str]) -> list[tuple[str, list[dict]]]:
    """(출처 표식, 롤아웃) 쌍 — 같은 이름이 두 번 오면 #2, #3 을 붙여 가른다."""
    out, seen = [], {}
    for p in paths:
        t = source_tag(p)
        seen[t] = seen.get(t, 0) + 1
        out.append((t if seen[t] == 1 else f"{t}#{seen[t]}", [json.loads(l) for l in open(p)]))
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("prompt", "continue"), required=True)
    ap.add_argument("--stage", choices=("screen", "confirm"), required=True,
                    help="screen: 모든 행위를 K=4 로 돌려 순위만 낸다(판정 없음). confirm: "
                         "--acts 로 지정한 행위만 K=8 로 돌려 Holm 보정 판정을 낸다.")
    ap.add_argument("--split_seed", type=int, default=11,
                    help="screen/confirm 문제 분할 시드(group_id 해시) — --seed 와 별개")
    ap.add_argument("--acts", default=None,
                    help="--stage confirm 전용(필수) — 확정할 행위 이름(쉼표 구분). 기준·제어는 "
                         "자동으로 더해진다.")
    ap.add_argument("--rollouts", action="append", required=True,
                    help="math_rollout 산출물 texts.jsonl (여러 번 줄 수 있다 — 출처별 표를 찍는다)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=None, help="비우면 --stage 기본값(screen=4, confirm=8)")
    ap.add_argument("--max_problems", type=int, default=120, help="prompt 모드(출처에 고르게 배분)")
    ap.add_argument("--max_sites", type=int, default=120, help="continue 모드(출처에 고르게 배분)")
    ap.add_argument("--per_problem", type=int, default=2, help="continue 모드: 문제당 오답 상한")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=8192,
                    help="★4k 는 절단률이 정확도 순서를 뒤집는다(cd9 G4). 내리지 말 것.")
    ap.add_argument("--max_prefix_tokens", type=int, default=8192)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.stage == "confirm" and not a.acts:
        raise SystemExit("[content] --stage confirm 은 --acts 가 필요하다(확정할 행위, 쉼표 구분).")
    k = a.k if a.k is not None else STAGE_K[a.stage]

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs = load_sources(a.rollouts)
    conds_all = prompt_conds() if a.mode == "prompt" else continue_conds()
    ref, ctl = REF[a.mode], CONTROL[a.mode]

    confirm_acts: list[str] | None = None
    if a.stage == "confirm":
        confirm_acts = [x.strip() for x in a.acts.split(",") if x.strip()]
        bad = [x for x in confirm_acts if x not in conds_all]
        if bad:
            raise SystemExit(f"[content] --acts 에 mode={a.mode} 의 조건이 아닌 이름이 있다: {bad}")
        # ★pad_length(F1) 는 진단용 둘째 잣대라 --acts 로 고르지 않아도 항상 같이 돈다(prompt
        #   모드에서만 존재한다 — continue 모드엔 cue=None 이라 conds_all 에 없다).
        pad = ["pad_length"] if a.mode == "prompt" and "pad_length" in conds_all else []
        conds = tuple(dict.fromkeys([ref, ctl, *pad, *confirm_acts]))  # 기준+제어+pad+확정 acts
    else:
        conds = conds_all

    # ── 모집단: screen/confirm 절반으로 먼저 가르고, 출처별 상한으로 고르게 담는다 ─────
    units: list[dict] = []
    n_skip_cand = 0
    cap = max(1, (a.max_problems if a.mode == "prompt" else a.max_sites) // max(1, len(srcs)))
    for tag, rolls in srcs:
        if a.mode == "prompt":
            all_mixed = select_mixed_problems(rolls, max_problems=0)   # 0 = 상한 없음, 전부
            half = [p for p in all_mixed if split_bucket(p["group_id"], a.split_seed) == a.stage]
            got = [{"text": "", **p} for p in half[:cap]]
            key = "group_id"
            by_group: dict[str, list[str]] = {}
            for r in rolls:
                by_group.setdefault(r["group_id"], []).append(last_boxed(r.get("text", "")))
            for g in got:
                vf = pick_non_plurality_wrong(by_group.get(g["group_id"], []), g["gold"])
                vr = pick_random_wrong_cand(g["gold"], rng)
                if vf is None:
                    n_skip_cand += 1
                g["cand"] = {"verification_first": vf, "verification_first_random": vr}
        else:
            all_wrong = select_wrong_rollouts(rolls, per_problem=a.per_problem)
            half = [s for s in all_wrong if split_bucket(s["group_id"], a.split_seed) == a.stage]
            rng.shuffle(half)
            got = half[:cap]
            key = "roll_id"
        units += [{"unit_id": f"{tag}::{g[key]}", "tag": tag, **g} for g in got]
        print(f"[content] {tag}: 단위 {len(got)}개 ({a.stage} 절반, 출처 상한 {cap})", flush=True)
    if not units:
        raise SystemExit("[content] 후보가 없다 — 입력 롤아웃에 MIXED 그룹이 있는지 확인하라.")
    if a.mode == "prompt":
        print(f"[content] verification_first 후보 없어 문제 단위로 스킵: {n_skip_cand}개", flush=True)
    print(f"[content] 단위 {len(units)} x 조건 {len(conds)} x K={k} = "
          f"{len(units) * len(conds) * k}개 생성 예정", flush=True)

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()
    lim = a.max_prefix_tokens + 512

    reqs, ix, n_drop, n_no_cand = [], [], 0, 0
    for ui, u in enumerate(units):
        for c in conds:
            if a.mode == "prompt" and "{CAND}" in (META_ACTS[c].suffix or ""):
                cand = (u.get("cand") or {}).get(c)
                if cand is None:
                    n_no_cand += 1
                    continue
                q = act_prompt(tok, a.variant, u["problem"], c, cand=cand)
            else:
                q = (act_prompt(tok, a.variant, u["problem"], c) if a.mode == "prompt"
                     else act_continuation(tok, a.variant, u["problem"], u.get("text", ""), c))
            if len(tok.encode(q)) > lim:
                n_drop += 1
                continue
            reqs.append(q)
            ix.append((ui, c))
    print(f"[content] 요청 {len(reqs)}개 (버린 것 {n_drop}개, 후보 없어 뺀 것 {n_no_cand}개, "
          f"한도 {lim} 토큰)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    gens = []
    texts: dict[tuple[int, str], list[str]] = {}   # (ui, cond) → 생성 텍스트(기준 교차-준수용)
    for (ui, c), o in zip(ix, outs):
        u = units[ui]
        cand = (u.get("cand") or {}).get(c) if a.mode == "prompt" else None
        for x in o.outputs:
            corr, nnb = ((grade_math(x.text, u["gold"]), 0) if a.mode == "prompt"
                         else grade_continuation(x.text, u.get("text", ""), u["gold"]))
            rec = {"r_corr": corr, "trunc": int(x.finish_reason == "length"), "no_new_boxed":
                   nnb, "tokens": len(x.token_ids), "complied": complied(c, x.text, cand=cand)}
            agg.setdefault((ui, c), []).append(rec)
            texts.setdefault((ui, c), []).append(x.text)
            gens.append({"unit_id": u["unit_id"], "tag": u["tag"], "cond": c,
                         "gen_r_corr": corr, "n_tok": rec["tokens"], "text": x.text,
                         "truncated": rec["trunc"], "complied": rec["complied"],
                         # ★후보값을 같이 남긴다 — 이걸 안 적으면 나중에 이 생성물의 프롬프트를
                         #   바이트 동일하게 되살릴 수 없다(무작위 팔은 재계산이 불가능하다).
                         "cand": cand})
    with (out / "gens.jsonl").open("w") as fh:
        for r in gens:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ★F2(0915)+F3(0915b): 준수 기준 조건(`COMPLIANCE_REF[mode]` — prompt 는 REF[mode]="plain",
    #   continue 는 CONTROL[mode]="filler_continue", 위 정의부 주석 참조) 생성물에 **다른 행위의
    #   정규식을 그대로** 적용해 «지시 없이도 이미 자연발생하는 준수율»(complied_rate_ref)을
    #   잰다 — 통과 규칙이 절대 준수율만이 아니라 이 기준 대비 차분도 요구한다(모듈 docstring
    #   참조). {CAND} 행위는 그 단위에 실제로 쓰인 후보값으로 컴파일해야 rx 가 맞다 — 없으면
    #   (정답 누출 방지로 그 단위가 스킵됐으면) 그 (단위, 행위) 조합은 뺀다(만들어 내지 않는다).
    comp_ref_cond = COMPLIANCE_REF[a.mode]
    ref_complied: dict[int, dict[str, float]] = {}
    for ui, u in enumerate(units):
        ref_texts = texts.get((ui, comp_ref_cond), [])
        if not ref_texts:
            continue
        rc: dict[str, float] = {}
        for c in conds:
            if c == comp_ref_cond or META_ACTS[c].rx is None:
                continue
            if "{CAND}" in META_ACTS[c].rx:
                cnd = (u.get("cand") or {}).get(c)
                if cnd is None:
                    continue
                rc[c] = _p([complied(c, t, cand=cnd) for t in ref_texts])
            else:
                rc[c] = _p([complied(c, t) for t in ref_texts])
        ref_complied[ui] = rc

    # ★F1(0915) 수정: 예전엔 «모든 조건이 있는 단위만» 남겨(`all(got.values())`) 후보가 없어
    #   verification_first 하나를 스킵한 단위가 **다른 모든 조건의 통계에서도** 통째로 빠졌다
    #   (83 → 55 로 관측). 이제는 ref·control 둘 다 있는 단위만 남기고, 단위별로 **실제 생성된
    #   조건만** `cond` 에 넣는다 — `summarize()` 가 조건마다·짝마다 있는 단위만으로 잰다.
    recs = []
    for ui, u in enumerate(units):
        got = {c: agg.get((ui, c)) for c in conds if agg.get((ui, c))}
        if ref not in got or ctl not in got:
            continue                     # ref·control 은 모든 판정의 최소 요건이다
        rc = ref_complied.get(ui, {})
        recs.append({"unit_id": u["unit_id"], "tag": u["tag"], "group_id": u["group_id"],
                     "gold": u["gold"],
                     "cond": {c: {**{m: _p([x[fk] for x in got[c]]) for m, fk in
                                  (("acc", "r_corr"), ("trunc", "trunc"), ("tokens", "tokens"),
                                   ("complied", "complied"), ("no_new_boxed", "no_new_boxed"))},
                                  "complied_ref": rc.get(c, _NAN)}
                              for c in got}})   # 실제로 생성된 조건만 들어간다(위 주석 참조)
    pooled = summarize(recs, conds, ref=ref, control=ctl, k=k, seed=a.seed, n_boot=a.n_boot,
                       stage=a.stage, confirm_acts=confirm_acts)
    per_source = {t: summarize([r for r in recs if r["tag"] == t], conds, ref=ref, control=ctl,
                               k=k, seed=a.seed, n_boot=a.n_boot, stage=a.stage,
                               confirm_acts=confirm_acts)
                  for t, _ in srcs if any(r["tag"] == t for r in recs)}
    meta = {"mode": a.mode, "stage": a.stage, "split_seed": a.split_seed,
            "confirm_acts": confirm_acts, "model_path": a.model_path, "variant": a.variant,
            "rollouts": a.rollouts, "seed": a.seed, "max_tokens": a.max_tokens, "k": k,
            "sources": {t: len([r for r in recs if r["tag"] == t]) for t, _ in srcs},
            "n_units_selected": len(units), "n_units_complete": len(recs),
            "n_dropped_long": n_drop,
            # ★F1b(0915) 수정: 이전엔 이 둘을 더해 `n_skipped_no_cand` 하나로 찍어 같은 스킵을
            #   두 자리에서(선정 시점 + 요청 시점) 이중으로 세는 것처럼 보였다 — 사실 같은
            #   근본원인(형제 후보 없음)의 두 **다른 시점** 관측이라 더하는 게 틀린 건 아니지만
            #   구분 없이 합쳐 놓으면 «몇 문제가 verification_first 를 통째로 못 받았는지»(선정
            #   시점, 문제당 최대 1) 와 «몇 (문제,조건) 요청이 후보 없어 생성 자체를 안 했는지»
            #   (요청 시점, 문제당 최대 len(CAND_ACTS))를 못 가른다. 이제 셋 다 따로 낸다.
            "n_skipped_no_cand_at_selection": n_skip_cand,   # pick_non_plurality_wrong()이 None
            "n_skipped_no_cand_at_request": n_no_cand,       # 위와 같은 이유로 그 (단위,행위) 스킵
            "n_skipped_no_cand": n_no_cand + n_skip_cand,    # 하위호환(구 스키마) — 합계일 뿐
            "n_generations": len(gens),
            "acts": {n: v._asdict() for n, v in META_ACTS.items()}}

    with (out / "per_problem.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(
        {"pooled": pooled, "per_source": per_source, "meta": meta}, ensure_ascii=False, indent=2))
    md = render_report(pooled, per_source, mode=a.mode, meta=meta)
    (out / "gate_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
