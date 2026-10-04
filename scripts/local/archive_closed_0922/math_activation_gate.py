#!/usr/bin/env python
r"""math_activation_gate — «교정 능력은 있는가, 스스로 켜지는가»를 가른다.

이 4B non-reasoning 정책에서 지금까지 실측된 것: 문맥 안 재시도(in-context retry, 원 풀이 +
메타 씨앗 + "Second attempt:")는 1,248개 오답 롤아웃 중 **0개**를 구제했다(`math_plan_gate`/
`math_critique_resolve_gate` 의 (iv) 조건과 cd9 실측) — 심어 준 `redirect` 는 95.6% 가 새
`decision: verify` 로 조용히 덮인다. Self-Correction Bench(2507.02778)는 non-reasoning
모델 일반에서 **64.5%의 눈가림(blind spot)**을 보고한다: 남이 저지른 오류라고 붙이면 고치는
능력은 있는데, 그게 **자기 자신의** 오류라는 표식만 붙으면 고치지 못한다 — 능력이 없는 게
아니라 **자기활성화(self-activation)가 안 된다**는 것.

이 게이트가 그 둘을 가른다: 같은 계산 예산에서, 같은 오답 롤아웃에 대해
  (1) `blind`          — 문제만(재표본 기준선, 아무 신호 없음)
  (2) `wait`            — **자기 귀속** 신호: 자기 풀이를 이어 쓰다 "Wait, let me
                          double-check this." (in-context self-trigger)
  (3) `external`        — **외부 귀속** 신호: 새 user 턴으로 "There is an error… find it"
  (4) `blind_external`  — 오류 신호는 있지만 **원 풀이가 문맥에 없다**(문제 + "a previous
                          attempt was wrong" 만) — «보이는 오류를 고치는 것»과 «더 열심히
                          하라는 말»을 가른다.

★능력과 자기활성화를 가르는 관문:
  ACTIVATION(외부 신호가 실제로 켜는가) = (external − blind) 유의 ∧ (external − blind_external)
    유의(둘 다 CI 가 0 제외, 평균 > 0 — blind_external 대조가 «오류가 보여서 고쳤다»를
    «더 열심히 하라는 말이 효과였다»에서 가른다) ∧ 정답 행에서 external 이 거짓 경보로 뒤집는
    비율(flip_wrong_rate) ≤ 0.10(외부 신호가 «비판이면 무조건 답을 바꾼다»가 아니어야 한다).
  SELF-TRIGGER(자기 귀속 신호가 켜는가) = (wait − blind) 유의·양(+) ∧ 정답 행 flip_wrong_rate
    ≤ 0.10.
  BLIND-SPOT = rescue(external) − rescue(wait) — 능력은 있는데 자기활성화가 안 되는 간극의
  크기. Self-Correction Bench 의 64.5% 와 같은 종류의 수를 이 정책·이 무대에서 잰다.

★예산은 전부 **max_tokens 8192**다(스펙 지시) — cd9 의 G4 재풀이 게이트는 4,096 에서
  Level-5 재풀이의 54% 가 잘렸고, 그때 정확도 순서는 절단률 순서의 **정확히 역순**이었다
  (절단이 적은 조건이 우연히 높아 보였다). 4k 예산은 여기서 해석 불가능하므로 쓰지 않는다.
  조건별 절단률(`trunc_rate`)을 반드시 같이 본다.

★거짓 경보(false-alarm) 대조: 오답 롤아웃만 보면 "외부 신호는 답을 흔든다"를 "교정을
  자기활성화한다"로 오독한다(cd9 EVC/plan-ceiling 이 겪은 함정과 같은 종류) — 그래서
  **정답** 롤아웃(문제당 하나, MIXED 그룹에서)에도 같은 wait/external/blind 를 돌려
  flip_wrong_rate 를 잰다. external 은 여기서는 **거짓말**이다("이전 시도가 틀렸다"고
  말하지만 롤아웃은 맞았다) — 그런데도 답을 흔들면 그 신호는 «내용»이 아니라 «비판 형식
  그 자체»에 반응하는 것이다.

`wait` 프롬프트 조립 — «열린 assistant 턴을 이어 쓴다»는 요구는, 설치된 Qwen3 chat 템플릿
(`/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507/tokenizer_config.json`)을
직접 읽어 확인했다: 템플릿은 매 assistant 메시지 뒤에 무조건 `<|im_end|>\n` 을 붙인다(HF
`apply_chat_template` 의 `continue_final_message=True` 가 그 꼬리를 사후에 잘라낸다 — 실측
바이트 동일 확인함). 이 게이트는 **그 경로를 쓰지 않는다** — `render_generation_prompt` 가
이미 `<|im_start|>assistant\n` 에서 끝나는(add_generation_prompt=True) 문자열이므로, 그
뒤에 원 풀이 + cue 를 **그대로 이어 붙이면** `continue_final_message=True` 와 바이트
동일한 결과가 나온다(이 파일의 테스트가 실측으로 고정한다). 이 트릭은 이미
`math_cited_site_gate.continuation_prompt`/`math_critique_resolve_gate.incontext_prompt`
가 쓰는 것과 같은 규약이라 MockTok 으로도 테스트가 돌고, vllm 생성 프롬프트에도 그대로 쓸
수 있다(문자열 이어붙이기일 뿐, 특수 토크나이저 kwarg 가 필요 없다).

사용(예):
  python scripts/local/math_activation_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 \
      --variant math_opt --k 8 --max_sites 150 --seed 11 --max_tokens 8192 \
      --out_dir /hdd_data/seungpil/scratch/eval/activation_gate_s1 --gpu_util 0.45

★F1 «리셋 자리의 메타 내용»(0915) — `--conds` 로 조건을 고른다. 기본값은 위 네 조건이라
  G8 실행이 그대로 재현된다. F1 판:
  --conds blind,fact,fact_effort,fact_switch,fact_switch_donor,fact_notx,fact_pad
  (fact = blind_external 별칭 · 오답 본문은 어느 조건에도 들어가지 않는다 · 정답 행에는
  blind/fact_effort/fact_switch/fact_notx 만 돌아 거짓 경보(flip)를 잰다.)

★H2 «합의 상태 × 습관»(0916) — `--population {mixed,all_wrong,both}`.
  `mixed`(기본)는 옛 동작 그대로다. `all_wrong` 은 **8개 롤아웃이 전부 오답인 문제**
  (난이도-5 800개 중 195개, pass@8 = 0 — F1/S4 의 MIXED 모집단에는 한 번도 들어간 적이
  없고 그중 72%가 «고합의 오답»이다)에서 문제당 대표 오답 행 하나를 쓴다. 거기서는
  **기준선 구제율이 정의상 0** 이므로 맞는 재시도는 그대로 **새 해답**이고, 보고의 핵심은
  짝 차이가 아니라 **절대 구제율**이다. 행마다 그 문제의 합의 상태를 **gold 없이**(형제
  boxed 답의 동치 군집만으로) 붙여 모집단 × 상태 격자로 낸다. `both` 에서는 저합의
  (SPLIT/SCATTER) **정답** 행이 `pop_kind=correct_lowagree` 로 다시 태그돼 «시험 때 합의가
  갈렸다고 리다이렉트를 켜면 맞은 답을 얼마나 흔드는가»를 상태별로 낸다(추가 생성 없음 —
  그 행들은 이미 정답 모집단의 fact_switch/fact_notx 를 돌고 있다).
  ⛔`n_correct_sib`(정답 형제 수)는 **gold 기반**이라 기록·분석용일 뿐 — 어떤 프롬프트에도
  들어가지 않는다(`tests/test_math_activation_gate.py` 가 고정한다).
  예: --population both --conds fact,fact_effort,fact_switch,fact_notx --k 8
      --max_sites 200 --per_problem 1
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
from src.training import math_dis as _md  # noqa: E402  (plurality_answer 재사용)
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, grade_math, last_boxed, selftest_math_verify,
)

_NAN = float("nan")
PASS_DELTA_EXT = 0.05           # external − blind 의 하한(평균)
MAX_FLIP_WRONG = 0.10           # 정답 행 flip_wrong_rate 상한
CONDS_WRONG = ("blind", "wait", "external", "blind_external")
CONDS_CORRECT = ("blind", "wait", "external")

# ── F1 «리셋 자리의 메타 내용» (0915 추가) ────────────────────────────────────────
# G8 이 확정한 것: 오답 본문을 문맥에 두면 끌개가 되고(external −.104 vs blind_external),
# «이전 시도가 틀렸다»는 **사실만** 남기면 +.034 로 구제가 는다(blind .583 → .617).
# 남은 물음은 그 **리셋 자리에 무엇을 적을 때** 가장 좋은가다 — 오답 본문은 절대 안 보여 준다.
# 모든 새 조건은 `fact` 줄(=BLIND_EXTERNAL_NOTE) 뒤에 한 문장(들)을 덧붙일 뿐이다.
PASS_DELTA_CONTENT = 0.03       # (arm − fact) 평균 하한
MAX_TRUNC_CONTENT = 0.20        # 오답 행 trunc_rate 상한
# ★0916 수리 — F1 의 flip 절은 **상대적**이어야 한다. 절대 상한 .10 은 오측정이었다:
#   이 모집단(MIXED 그룹의 **정답** 롤아웃)을 재표본만 해도(blind, 아무 신호 없음)
#   flip_wrong = .341 이 나온다 — 맞힌 난이도-5 문제를 다시 풀면 3분의 1은 틀린다는 것이
#   이 정책의 기저율이다. 즉 .10 은 «거짓 경보를 일으키지 않는다»가 아니라 «blind 보다
#   3배 안정적이어라»를 요구했고, 어떤 팔도(심지어 신호가 **없는** 팔도) 통과할 수 없는
#   불가능 조건이었다. 물어야 할 것은 «리셋 자리의 내용이 blind 재표본보다 정답 행을 더
#   흔드는가»이므로 기준선은 같은 모집단의 blind flip 이고, 여유는 아래 마진뿐이다.
FLIP_MARGIN = 0.02              # arm flip_wrong ≤ (기준선 flip_wrong + 이 마진)
FLIP_BASELINE_KEYS = ("blind", "blind_external", "fact")  # 기준선 우선순위(blind 없으면 fact)
FACT_ALIAS = "blind_external"   # `fact` 는 blind_external 의 별칭 — 프롬프트 바이트 동일
CONDS_ALIASES = {"fact": FACT_ALIAS, "notx": "fact_notx", "reread": "fact_reread"}
# 내용 팔(검정 대상) / 대조 팔(검정에서 제외, 진단용)
CONTENT_ARMS = ("fact_effort", "fact_switch", "fact_notx")
CONTENT_CONTROLS = ("fact_switch_donor", "fact_pad")
# ── S4″/S4′-lite «정책이 스스로 쓴 회복 습관 노트»(0916) ────────────────────────
# F1 이 확정한 것: 리셋 자리에 **고정된** 한 문장을 넣으면 구제가 는다(effort/switch/notx).
# 다음 물음은 **정책이 스스로 그 한 줄을 쓸 수 있는가**, 그리고 **내부 신호(자기 형제 답
# 분포)를 보면 더 잘 쓰는가다. 훈련 없이(training-free) F1 과 같은 오답 모집단에서 잰다.
#   fact_reread  고정 팔 — «먼저 문제를 다시 읽고 주어진 값·제약을 모두 적어라»
#                (오프라인 상한 실험에서 제약 재독이 +1.4pp 였으나 **리셋 자리에서는 미검증**)
#   habit_self   정책이 쓴 한 줄(오답 본문만 보여 준다)
#   habit_sib    정책이 쓴 한 줄 + **gold 없는** 형제 답 분포 요약(정오는 절대 안 알려 준다)
S4_FIXED_ARMS = ("fact_reread",)
HABIT_ARMS = ("habit_self", "habit_sib")
S4_ARMS = S4_FIXED_ARMS + HABIT_ARMS
CONDS_CONTENT = CONTENT_ARMS + CONTENT_CONTROLS + S4_ARMS
# 정답 행(거짓 경보)에 돌릴 수 있는 조건 — 기본 선택은 예전 그대로 (blind, wait, external).
CONDS_CORRECT_ELIGIBLE = CONDS_CORRECT + CONTENT_ARMS
ALL_CONDS = CONDS_WRONG + CONDS_CONTENT

# ★사후 오류 감속(post-error slowing): «처음부터 다시 유도한 뒤에만 박스»
EFFORT_SENT = ("Before giving a final answer, re-derive the full solution from scratch and only "
               "then box it.")
# ★lose-shift: 모델이 스스로 붙인 방법 이름 L 만 옮긴다(숫자·답 금지 — 아래 누출 방지 참조)
#   `SWITCH_TMPL` 은 **`src.training.trial2` 에 있다**(아래 import) — S3 의 NOTE_MODE=switch
#   팔이 바이트 동일한 프롬프트를 써야 하므로 단일 진실 원천을 그쪽에 둔다.
# ★부정 지식(negative knowledge): 답이 X 가 아니라는 사실만
# ★fact_notx 는 «오답 값 자체를 실험 팔로 주입해 부정 지식이 해로운가(NuRL 의 예측)»를 재는
#   **실험용** 대조다. 라벨을 만들려면 **gold 를 알아야** 한다(오답 행은 그 롤아웃의 최종
#   답을, 정답 행은 다수결이 아닌 오답 형제의 답을 gold 대조로 골라 쓴다) — 그래서 이 팔은
#   **훈련 컴포넌트 후보가 아니다**(정책이 스스로 gold 없이 만들어낼 수 없는 신호를 넣기
#   때문). 행동으로 얻는 정보로서만 (NuRL 이 맞다면 이 팔이 해를 보이는가) 의미가 있다.
#   (`NOTX_TMPL` 도 `src.training.trial2` 에 있다 — 아래 import.)
# ★길이·어조 대조: 내용 없는 일반적 반성 세 문장(재유도 지시 없음·방법 없음·숫자 없음)
PAD_SENTS = ("Careful work matters here. It is worth being deliberate and steady. "
             "Take the problem on its own terms.")

# ★S4 고정 팔 — 제약 재독(constraint re-reading). 출처: 오프라인 상한 실험에서 «주어진 값·
#   제약을 다시 읽고 나열» 습관이 +1.4pp 였다. 그 실험은 **리셋 자리에서 쓴 적이 없다**
#   (거기선 풀이 앞에 붙였다) — 그래서 F1 자리에서 미검증 팔로 올린다.
REREAD_SENT = ("Before solving, re-read the problem statement and list every given quantity "
               "and constraint, then solve.")
# ★S4 정책-작성 팔의 사전 패스 질문 — «습관만, 풀이·수식·답 금지». 세 습관을 **보기로만**
#   제시하고 고르거나 섞게 한다(그러면 노트의 내용을 습관 부류로 분류할 수 있다).
HABIT_ASK = ("Write a one-line instruction to yourself for the next attempt at this problem. "
             "You may use these habits: rule out the failed answer; re-read the problem's "
             "constraints; re-derive from scratch. Choose one or combine. Do not include any "
             "solution steps, formulas, or a final answer.")
# ★그 한 줄을 리셋 자리에 싣는 틀(fact 줄 뒤 한 문장 규약 그대로)
HABIT_NOTE_TMPL = "Note to self: {note}"
# ★habit_sib 가 보여 주는 내부 신호 — **gold 없음·정오 표식 없음**. 자기 답과 형제 답들의
#   수학적 동치 군집 개수만 말한다(무응답 포함).
SIB_TMPL = "Your answer: {own}. Final answers across {n} attempts: {clusters}."
SIB_NOANS = "no answer"
#: 노트 부류 분류용 키워드(보고용 히스토그램 — 판정에 쓰지 않는다)
HABIT_KEYWORDS = {
    "exclude": ("rule out", "not ", "isn't", "is not", "exclude", "avoid the answer",
                "wrong answer", "incorrect answer"),
    "reread": ("re-read", "reread", "read the problem", "constraint", "given quantit",
               "given value", "what is asked"),
    "rederive": ("re-derive", "rederive", "from scratch", "start over", "redo", "again "
                 "from the beginning", "different approach", "another method"),
}
#: 사전등록된 «문제 안 학습 가능 신호» 관문 — 분산비 ≥ 1.3 ∧ p < .05
VAR_RATIO_MIN = 1.3
VAR_P_MAX = 0.05

# ── H2 «합의 상태 × 습관»(0916) ────────────────────────────────────────────────
# 분석(`agree_credit.py`)이 확정한 것: 난이도-5 800문제 중 **195개는 8개 롤아웃이 전부
# 오답**이고(pass@8 = 0 — F1/S4 의 MIXED 모집단에는 **한 번도 들어간 적이 없다**), 그중
# 72% 가 **고합의 오답**(ALL_SAME/DOMINANT — 8개가 같은 틀린 답으로 수렴)이다. 그 자리는
# «재표본으로는 절대 못 푸는» 자리이므로 리셋 자리의 습관이 가치를 낼 수 있는지가 미지다.
# 이 게이트는 두 모집단을 같은 습관 팔로 돌리고, 행마다 그 문제의 **합의 상태**를
# **gold 없이**(형제 8개의 boxed 답만으로) 붙인다.
#   ALL_SAME  최대 동치 군집 = K(= 8)
#   DOMINANT  5–7
#   SPLIT     3–4
#   SCATTER   ≤ 2
#   NOANS     boxed 답이 하나도 없다
# ★상태 정의는 `agree_credit.py:state_of` 와 **같은 정의**다(군집 = answers_equivalent,
#   분모 K = 그룹의 롤아웃 수 — 무응답도 K 에 센다). 새로 정의하지 않는다.
AGREE_STATES = ("ALL_SAME", "DOMINANT", "SPLIT", "SCATTER", "NOANS")
CONSENSUS_STATES = ("ALL_SAME", "DOMINANT")     # 고합의 = 재표본이 못 뚫는 자리
LOWAGREE_STATES = ("SPLIT", "SCATTER")          # 시험 때 «합의가 갈렸다»로 켜질 자리
POP_MIXED, POP_ALLWRONG = "mixed_wrong", "allwrong"
POP_CORRECT, POP_CORRECT_LOWAGREE = "correct", "correct_lowagree"
WRONG_POP_KINDS = (POP_MIXED, POP_ALLWRONG)
#: `--population` 선택 → 쓸 오답 pop_kind
POPULATION_CHOICES = {"mixed": (POP_MIXED,), "all_wrong": (POP_ALLWRONG,),
                      "both": (POP_MIXED, POP_ALLWRONG)}
#: 저합의 정답 행(거짓 경보 대조)에 돌리는 조건 — 시험 때 실제로 켜질 두 팔만
CONDS_CORRECT_LOWAGREE = ("fact_switch", "fact_notx")

# 방법 이름 L 을 얻는 사전 패스(pre-pass) — 오답 본문을 보고 «방법 이름만» 말하게 한다.
#   `LABEL_ASK`/`GENERIC_LABEL`/`LABEL_MAX_WORDS` 도 `src.training.trial2` 에 있다
#   (아래 import) — S3 의 switch 계열 사전 패스가 같은 질문을 써야 한다.

# ★(2) 자기 귀속 신호. render_generation_prompt(...) + 원 풀이 + 이 문자열 = 열린 assistant
#   턴을 이어 쓰는 것과 바이트 동일(위 docstring 실측 확인 참조).
WAIT_CUE = "\n\nWait, let me double-check this."

# ★(3) 외부 귀속 신호 — 새 user 턴. 오류가 있다는 사실 자체를 말할 뿐, 어디가 틀렸는지는
#   말하지 않는다(그러면 비평 값이 섞인다 — math_critique_resolve_gate 의 관심사).
EXTERNAL_CUE = ("There is an error in your solution. Find it and give the corrected final "
                "answer in \\boxed{}.")

# ★(4) blind_external — 오류 신호는 있지만 원 풀이가 문맥에 없다. «보이는 오류를 고치는가»와
#   «더 열심히 하라는 말이 효과였는가»를 가르는 대조.
# ★단일 진실 원천은 `src.training.trial2` 다(S3 2-시도 팔이 **바이트 동일**한 프롬프트를
#   써야 `.617` 참조가 그대로 적용된다 — 상수를 두 벌 두면 한쪽만 바뀐다).
from src.training.trial2 import (  # noqa: E402
    BLIND_EXTERNAL_NOTE, GENERIC_LABEL, GENERIC_NOTE, LABEL_ASK, LABEL_MAX_WORDS,
    NOTX_TMPL, SWITCH_TMPL, agreement_state, cap_attempt1_text, clean_note as _clean_note,
    clean_short_text, fact_prompt as _fact_prompt,
)

_SENTINEL = "<<<ACTIVATION_GATE_SOLUTION>>>"


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────────
def blind_prompt(tok, variant: str, problem: str) -> str:
    """(1) 문제만 — render_generation_prompt 와 **바이트 동일**(매치드-compute 앵커)."""
    return render_generation_prompt(tok, variant, problem)


def wait_prompt(tok, variant: str, problem: str, text: str) -> str:
    """(2) 자기 귀속 — 열린 assistant 턴을 원 풀이 + WAIT_CUE 로 이어 쓴다.
    ★`render_generation_prompt` 가 이미 `add_generation_prompt=True` 로 assistant 턴을 연
    채로 끝나므로, 그 뒤에 텍스트를 이어 붙이면 `continue_final_message=True` 렌더와 바이트
    동일하다(모듈 docstring 실측 확인 참조) — 별도의 sentinel 되꽂기가 필요 없다."""
    return render_generation_prompt(tok, variant, problem) + (text or "") + WAIT_CUE


def external_prompt(tok, variant: str, problem: str, text: str, ask: str = EXTERNAL_CUE) -> str:
    """(3) 외부 귀속 — 원 풀이를 assistant 턴으로 **바이트 동일**하게 담고 새 user 턴(ask).
    ★sentinel 되꽂기: Qwen3 템플릿은 assistant 본문의 끝 공백을 지운다 — 본문을 그냥 넣으면
    프롬프트가 원 롤아웃 문맥과 조용히 달라진다(math_critique_resolve_gate.build_critique_prompt
    와 같은 규약)."""
    msgs = build_math_prompt(problem, variant) + [
        {"role": "assistant", "content": _SENTINEL},
        {"role": "user", "content": ask},
    ]
    rendered = render_chat_messages(tok, msgs)
    if rendered.count(_SENTINEL) != 1:
        raise RuntimeError(f"[ACT] chat 템플릿에서 sentinel 을 {rendered.count(_SENTINEL)}번 "
                           "찾았다 — 1번이어야 한다(템플릿이 assistant 본문을 변형한다).")
    head, tail = rendered.split(_SENTINEL)
    return head + (text or "") + tail


def blind_external_prompt(tok, variant: str, problem: str) -> str:
    """(4) 오류 신호는 있지만 원 풀이가 문맥에 없다 — user 턴 끝에 BLIND_EXTERNAL_NOTE 만 붙는다.
    system 프롬프트·앞부분은 blind 와 같다(달라지는 것은 user 턴뿐)."""
    return note_prompt(tok, variant, problem)


def note_prompt(tok, variant: str, problem: str, extra: str = "") -> str:
    """★F1 공통 조립기 — user 턴 끝에 `BLIND_EXTERNAL_NOTE`(=fact 줄) + (있으면) 공백 하나 +
    `extra`. `extra=""` 이면 `blind_external_prompt` 와 **바이트 동일**(fact 별칭의 근거)."""
    # ★본체는 `src.training.trial2.fact_prompt` — S3 2-시도 팔과 한 벌만 쓴다.
    return _fact_prompt(tok, variant, problem, extra)


def label_prompt(tok, variant: str, problem: str, text: str) -> str:
    """fact_switch 사전 패스 — 오답 본문 + LABEL_ASK(새 user 턴). external_prompt 와 같은
    sentinel 규약을 쓴다(본문 끝 공백 보존)."""
    return external_prompt(tok, variant, problem, text, ask=LABEL_ASK)


def clean_label(raw: str, wrong_answer: str = "") -> tuple[str, str]:
    """사전 패스 출력 → (label, reason). ★누출 방지: 숫자·`\\boxed`·오답 문자열이 들어간
    라벨은 **버리고** 일반 라벨로 되돌린다(그러지 않으면 fact_switch 가 몰래 답을 알려 주는
    팔이 된다). reason ∈ {ok, empty, digit, boxed, answer}.
    ★본체는 `src.training.trial2.clean_short_text` — S3 반성문 가드와 한 벌만 쓴다."""
    return clean_short_text(raw, wrong_answer, max_words=LABEL_MAX_WORDS,
                           fallback=GENERIC_LABEL)


def rotate_donor(labels: Sequence[str], i: int, *, scope: str = "global",
                 sources: Sequence[str] | None = None) -> str:
    """기증자(donor) 라벨 — 결정적 회전. **자기 라벨은 절대 주지 않는다**: i+1, i+2 … 순으로
    돌며 자기 것과 **문자열이 다른** 첫 라벨을 쓰고, 전부 같으면(또는 그 범위에 후보가
    없으면) 일반 라벨로 되돌린다.
    ★fact_switch_donor 는 «라벨이 자기 문제와 무관해도 자기활성화가 켜지는가»를 재는
      **정박(anchoring) 대조**다 — 기증자 라벨은 설계상 **다른 문제**에서 온다(정답을
      요구하지 않으므로 gold 누출이 아니다). `scope="within_source"` 면 후보를 `sources`
      가 같은 행으로 좁힌다(기본 `scope="global"` 은 옛 동작 그대로 전체에서 돈다) —
      `sources` 가 없으면(이 게이트는 지금 `--rollouts` 를 하나만 받는다) within_source 는
      global 과 같아진다."""
    n = len(labels)
    if scope not in ("global", "within_source"):
        raise SystemExit(f"[act] 모르는 donor_scope: {scope!r} (아는 것: global, within_source)")
    same_src = (lambda j: True) if (scope == "global" or not sources) else (
        lambda j: sources[j] == sources[i])
    for off in range(1, n):
        j = (i + off) % n
        cand = labels[j]
        if cand != labels[i] and same_src(j):
            return cand
    return GENERIC_LABEL


def effort_prompt(tok, variant: str, problem: str) -> str:
    return note_prompt(tok, variant, problem, EFFORT_SENT)


def switch_prompt(tok, variant: str, problem: str, label: str) -> str:
    return note_prompt(tok, variant, problem, SWITCH_TMPL.format(label=label))


def notx_prompt(tok, variant: str, problem: str, answer: str) -> str:
    return note_prompt(tok, variant, problem, NOTX_TMPL.format(answer=answer))


def pad_prompt(tok, variant: str, problem: str) -> str:
    return note_prompt(tok, variant, problem, PAD_SENTS)


# ── S4″/S4′-lite: 고정 재독 팔 + 정책이 쓴 습관 노트 ──────────────────────────
def reread_prompt(tok, variant: str, problem: str) -> str:
    """fact_reread — fact 줄 뒤에 REREAD_SENT 한 문장(다른 fact_* 팔과 같은 조립)."""
    return note_prompt(tok, variant, problem, REREAD_SENT)


def habit_prompt(tok, variant: str, problem: str, note: str) -> str:
    """habit_self/habit_sib 의 **시도-2** 프롬프트 — 노트 한 줄만 옮긴다. 노트가 비면
    fact 전용으로 되돌린다(= 그 행은 대조 팔과 바이트 동일)."""
    n = (note or "").strip()
    return note_prompt(tok, variant, problem, HABIT_NOTE_TMPL.format(note=n) if n else "")


def sibling_answer_line(own: str, answers: Sequence[str]) -> str:
    """★gold 없는 내부 신호 — 자기 답과 형제 답들의 **수학적 동치 군집** 개수.
    정오는 어디에도 나오지 않고(gold 를 보지 않는다), 정답 문자열이 특별 대우를 받지도
    않는다. 군집 대표는 **먼저 나온 표기**이고 순서는 (빈도 내림, 첫 등장) 이다.
    빈 답은 `no answer` 로 센다. 후보가 없으면 "" (그 행에서 habit_sib 는 habit_self 와
    같은 프롬프트가 되므로 건너뛴다)."""
    xs = [(a or "").strip() for a in answers]
    if not xs:
        return ""
    clusters: list[list] = []      # [대표, 개수, 첫 등장]
    n_empty = 0
    for i, a in enumerate(xs):
        if not a:
            n_empty += 1
            continue
        for cl in clusters:
            if answers_equivalent(cl[0], a):
                cl[1] += 1
                break
        else:
            clusters.append([a, 1, i])
    parts = [f"{c[0]} ({c[1]})" for c in sorted(clusters, key=lambda c: (-c[1], c[2]))]
    if n_empty:
        parts.append(f"{SIB_NOANS} ({n_empty})")
    if not parts:
        return ""
    return SIB_TMPL.format(own=(own or SIB_NOANS), n=len(xs), clusters=", ".join(parts))


def own_is_minority(own: str, answers: Sequence[str]) -> bool:
    """자기 답이 형제 분포에서 **다수(plurality)가 아닌가**. 정오와 무관한 순수 분포 진술
    (habit_sib 의 기록용 — 판정에 쓰지 않는다)."""
    plur = _md.plurality_answer([a for a in answers if (a or "").strip()])
    own = (own or "").strip()
    if not own or not plur:
        return False
    return not answers_equivalent(plur, own)


def habit_ask_prompt(tok, variant: str, problem: str, text: str, sib_line: str = "") -> str:
    """습관 노트 **사전 패스** — 오답 본문(꼬리 캡)을 assistant 턴에 담고 새 user 턴으로
    (있으면) 형제 분포 한 줄 + HABIT_ASK. 오답 본문이 들어가는 **유일한** 자리다
    (시도-2 프롬프트에는 어느 팔에서도 들어가지 않는다)."""
    ask = (sib_line + " " + HABIT_ASK) if sib_line else HABIT_ASK
    return external_prompt(tok, variant, problem, cap_attempt1_text(tok, text), ask=ask)


def clean_habit(raw: str, wrong_answer: str = "") -> tuple[str, str]:
    r"""습관 노트 누출 가드 — `trial2.clean_note(..., allow_exclusion=True)` 를 부른다:
    자기 오답을 «그 답은 아니다»로 **한 번** 적는 것만 허용하고, 다른 숫자·수식·`\boxed`
    는 전부 거절한다(habit_sib 에서 형제 후보 Y 가 노트로 새는 것을 막는 절)."""
    return _clean_note(raw, wrong_answer, allow_exclusion=True)


def habit_class(note: str) -> str:
    """노트 → 습관 부류 {exclude, reread, rederive, mixed, generic}(보고용).
    두 가지 이상이면 mixed, 아무 키워드도 없으면(또는 일반 문구면) generic."""
    s = (note or "").strip().lower()
    if not s or s == GENERIC_NOTE:
        return "generic"
    hit = [k for k, kws in HABIT_KEYWORDS.items() if any(w in s for w in kws)]
    if not hit:
        return "generic"
    return hit[0] if len(hit) == 1 else "mixed"


def note_variance_stat(rows: Sequence[tuple[Sequence[int], Sequence[int]]], *, seed: int = 0,
                       n_perm: int = 2000) -> dict:
    """★«문제 안에 학습 가능한 신호가 있는가» 통계 — 노트별 재시도 정오의 **행 안 분산**을
    같은 행의 **단일 프롬프트** 팔(모든 K 회가 같은 프롬프트)과 비교한다.
    `rows` = [(습관팔 K 결과, 기준팔 K 결과)] (둘 다 같은 행·같은 K).

    ⚠️**사전등록된 «≥1.3» 은 문자 그대로는 잴 수 없다**(F1 의 flip .10 과 같은 종류의
      오측정): 이항 표본의 표본분산은 p̂(1−p̂) 로 **항상** 정해지므로 «노트마다 p 가 다르다»는
      과분산으로 나타나지 않는다. 노트당 재시도가 1회뿐이면 행 안 이질성은 오히려 계수를
      **Poisson-이항**(과소분산)으로 만든다. 그래서 이 함수가 내는 것은
        ratio = mean_r Var̂(습관팔) / mean_r Var̂(기준팔)   (Var̂ = K/(K−1)·p̂(1−p̂))
      즉 **행 수준 분산의 팔 간 비**이고, p 는 행마다 두 팔의 벡터를 맞바꾸는(교환가능성)
      **순열 검정**의 단측 p(귀무 비율 ≥ 관측 비율의 비율)다. 관문은 사전등록대로
      (ratio ≥ 1.3 ∧ p < .05) 로 찍되, 이 해석 제한을 보고에 같이 낸다."""
    def var_hat(xs: Sequence[int]) -> float:
        n = len(xs)
        if n < 2:
            return _NAN
        p = sum(float(x) for x in xs) / n
        return (n / (n - 1)) * p * (1.0 - p)

    pairs = [(var_hat(a), var_hat(b)) for a, b in rows
             if len(a) >= 2 and len(b) >= 2]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return {"ratio": _NAN, "p": _NAN, "n_rows": 0, "var_arm": _NAN, "var_ref": _NAN}
    va = sum(x for x, _ in pairs) / len(pairs)
    vb = sum(y for _, y in pairs) / len(pairs)
    ratio = (va / vb) if vb > 0 else _NAN
    rng = random.Random(seed)
    hits = 0
    for _ in range(int(n_perm)):
        sa = sb = 0.0
        for x, y in pairs:
            if rng.random() < 0.5:
                x, y = y, x
            sa += x
            sb += y
        r0 = (sa / sb) if sb > 0 else _NAN
        if math.isfinite(r0) and math.isfinite(ratio) and r0 >= ratio:
            hits += 1
    p = (hits + 1) / (int(n_perm) + 1)
    return {"ratio": ratio, "p": p, "n_rows": len(pairs), "var_arm": va, "var_ref": vb,
            "n_perm": int(n_perm)}


def variance_signal_pass(stat: dict) -> bool:
    """사전등록 관문 — ratio ≥ VAR_RATIO_MIN ∧ p < VAR_P_MAX."""
    r, p = (stat or {}).get("ratio", _NAN), (stat or {}).get("p", _NAN)
    return bool(isinstance(r, (int, float)) and math.isfinite(r) and r >= VAR_RATIO_MIN
                and isinstance(p, (int, float)) and math.isfinite(p) and p < VAR_P_MAX)


def resolve_correct_states(spec: str | None) -> list[str]:
    """`--correct_states` 파싱 — 정답(거짓-경보) 행을 합의 상태로 좁힌다. 비어 있으면
    **전체 상태**(= 필터 없음, 옛 동작 바이트 동일). 순서는 `AGREE_STATES` 로 정규화한다."""
    if not spec or not spec.strip():
        return list(AGREE_STATES)
    want = [x.strip().upper() for x in spec.split(",") if x.strip()]
    bad = [x for x in want if x not in AGREE_STATES]
    if bad:
        raise SystemExit(f"[act] 모르는 합의 상태 {bad} — 가능한 것: {', '.join(AGREE_STATES)}")
    return [s for s in AGREE_STATES if s in set(want)]


def resolve_conds(spec: str | None, default: Sequence[str] = CONDS_WRONG) -> list[str]:
    """`--conds` 파싱 — 별칭(fact→blind_external) 해소, 순서 보존·중복 제거."""
    if not spec:
        return list(default)
    out: list[str] = []
    for raw in spec.split(","):
        c = raw.strip()
        if not c:
            continue
        c = CONDS_ALIASES.get(c, c)
        if c not in ALL_CONDS:
            raise SystemExit(f"[act] 모르는 조건 {c!r} — 가능한 것: "
                             f"{', '.join(ALL_CONDS)} (별칭 fact={FACT_ALIAS})")
        if c not in out:
            out.append(c)
    if not out:
        raise SystemExit("[act] --conds 가 비었다")
    return out


# ── 롤아웃 선별 ─────────────────────────────────────────────────────────────────
def select_correct_rollouts(rolls: Sequence[dict], *, per_problem: int = 1) -> list[dict]:
    """MIXED 그룹(0 < 그룹 정답률 < 1)의 **정답·미잘림** 롤아웃, 문제당 최대 per_problem 개
    — select_wrong_rollouts 와 짝을 이루는 거짓-경보 모집단(math_cited_site_gate 의 선별
    규약을 그대로 따른다)."""
    acc: dict = {}
    for r in rolls:
        acc.setdefault(r["group_id"], []).append(int(r["r_corr"]))
    mixed = {g for g, v in acc.items() if 0 < sum(v) < len(v)}
    seen: dict = {}
    out = []
    for i, r in enumerate(rolls):
        if r["group_id"] not in mixed or not int(r["r_corr"]) or r.get("truncated"):
            continue
        if seen.get(r["group_id"], 0) >= per_problem:
            continue
        seen[r["group_id"]] = seen.get(r["group_id"], 0) + 1
        out.append({"roll_id": f"{r['group_id']}#{i}", "group_id": r["group_id"],
                    "problem": r["problem"], "gold": r["gold"], "text": r["text"]})
    return out


def select_all_wrong_rollouts(rolls: Sequence[dict], *, per_problem: int = 1) -> list[dict]:
    """★H2 모집단 — **8개 롤아웃이 전부 오답**인 문제(pass@8 = 0)에서 문제당 대표 오답 행.

    `select_wrong_rollouts` 는 MIXED 그룹만 보므로 이 195개 문제는 지금까지 한 번도
    시험되지 않았다. 대표 행은 **결정적**으로 고른다: 그 그룹의 행 순서대로 «절단되지 않았고
    boxed 답이 있는» 첫 행, 그런 행이 없으면 **첫 행**(절단·무응답이라도 쓴다 — 그러지
    않으면 절단이 심한 문제가 모집단에서 통째로 빠져 편향이 생긴다).
    per_problem 은 보통 1 이다(스펙: 문제당 한 행)."""
    groups: dict = {}
    for i, r in enumerate(rolls):
        groups.setdefault(r["group_id"], []).append((i, r))
    out = []
    for gid, rows in groups.items():
        if any(int(r.get("r_corr", 0)) for _, r in rows):
            continue                       # 하나라도 맞았으면 pass@8 > 0 — 이 모집단이 아니다
        ranked = [(i, r) for i, r in rows
                  if not r.get("truncated") and last_boxed(r.get("text", ""))]
        picks = (ranked or rows)[:max(1, int(per_problem))]
        for i, r in picks:
            out.append({"roll_id": f"{gid}#{i}", "group_id": gid, "problem": r["problem"],
                        "gold": r["gold"], "text": r["text"], "r_corr": 0})
    return out


#: ★`agreement_state` 의 본체는 `src.training.trial2` 다(위 import) — S3 의 `agree` 재시도
#  게이트가 **같은 상태 정의**를 써야 하고, Ray 롤아웃 워커는 scripts/local 을 import 할 수
#  없으므로 단일 진실 원천을 학습 쪽에 둔다. 임계값은 그 한 곳에만 있다.


def group_agreement(rolls: Sequence[dict]) -> dict:
    """group_id → `agreement_state`(그 그룹의 모든 boxed 답, K = 그룹 크기)."""
    gans = group_answers(rolls)
    return {gid: agreement_state(ans, k=len(ans)) for gid, ans in gans.items()}


def group_n_correct(rolls: Sequence[dict]) -> dict:
    """group_id → **gold 기반** 정답 형제 수. ⛔분석 기록용일 뿐 — 어떤 프롬프트에도
    들어가지 않는다(프롬프트 조립기 `build` 는 이 값을 보지 않는다)."""
    out: dict = {}
    for r in rolls:
        out[r["group_id"]] = out.get(r["group_id"], 0) + int(r.get("r_corr", 0))
    return out


def own_in_dominant(own: str, state: dict) -> bool:
    """자기 답이 그 문제의 **우세 군집** 대표와 수학적으로 같은가(정오와 무관)."""
    own = (own or "").strip()
    dom = ((state or {}).get("dominant_answer") or "").strip()
    if not own or not dom:
        return False
    return bool(answers_equivalent(dom, own))


def group_answers(rolls: Sequence[dict]) -> dict:
    """group_id → 그 그룹의 모든 최종 답(빈 답 포함, plurality_answer 가 빈 것은 알아서 뺀다)."""
    out: dict = {}
    for r in rolls:
        out.setdefault(r["group_id"], []).append(last_boxed(r.get("text", "")))
    return out


def group_wrong_answers(rolls: Sequence[dict]) -> dict:
    """group_id → **오답** 형제들의 최종 답(정답 행의 fact_notx 용 X 후보)."""
    out: dict = {}
    for r in rolls:
        if not int(r.get("r_corr", 0)):
            out.setdefault(r["group_id"], []).append(last_boxed(r.get("text", "")))
    return out


def nonplurality_wrong_answer(cands: Sequence[str], plurality: str) -> str:
    """정답 행의 X — «다수결이 아닌 오답 형제 답» 중 가장 흔한 것(동률이면 먼저 나온 것).
    후보가 없으면 "" (그 행에서는 fact_notx 를 건너뛴다 — 스펙)."""
    counts: dict = {}
    for a in cands:
        a = (a or "").strip()
        if not a:
            continue
        if plurality and answers_equivalent(plurality, a):
            continue
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ── 요약 ────────────────────────────────────────────────────────────────────────
def _p(vals: Sequence[int]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN


def _mean_key(recs: Sequence[dict], key: str) -> float:
    xs = [r[key] for r in recs if key in r]
    return (sum(xs) / len(xs)) if xs else _NAN


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down 보정(math_meta_content_gate.holm_adjust 와 같은 정의를 이 게이트에도
    둔다): 오름차순 p 에 (m−i) 를 곱하고 누적 최댓값으로 단조화한다."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[name] = running
    return out


def summarize_wrong(recs: Sequence[dict], *, k: int = 0, seed: int = 0,
                    n_boot: int = 2000, conds: Sequence[str] = CONDS_WRONG) -> dict:
    """오답 행 요약 — recs 는 선택한 조건 중 **하나라도** 있는 롤아웃(쌍마다 따로 짝짓는다).
    `conds` 가 기본값이면 G8 판(활성화 관문)과 키·난수 오프셋까지 동일하고, F1 조건을 넣으면
    `paired_{arm}_minus_fact` 계열이 추가된다.
    ★0915 수리: boxed 답이 없어 `fact_notx` 를 못 만든 행처럼 **일부 조건만 빠진** 행도
      나머지 조건의 비교에는 남는다 — `d(a, b)` 가 두 열이 **동시에 있는 행**만 골라
      짝짓고, `n_pairs_*` 로 그 표본 크기를 낸다(쌍마다 다를 수 있다)."""
    n = len(recs)
    conds = list(conds)
    out: dict = {"n_rollouts": n, "k": k, "conds": conds}
    for i, c in enumerate(conds):
        out[f"p_{c}"] = bootstrap_ci([r[f"p_{c}"] for r in recs if f"p_{c}" in r],
                                     seed=seed + i, n_boot=n_boot)

    def d(a: str, b: str) -> list:
        return [r[f"p_{a}"] - r[f"p_{b}"] for r in recs if f"p_{a}" in r and f"p_{b}" in r]

    def n_pairs(a: str, b: str) -> int:
        return sum(1 for r in recs if f"p_{a}" in r and f"p_{b}" in r)

    out["n_pairs"] = {}
    # ── G8 의 네 대조(해당 조건이 둘 다 있을 때만) ───────────────────────────────
    legacy = [("paired_external_minus_blind", "external", "blind", 4),
              ("paired_wait_minus_blind", "wait", "blind", 5),
              ("paired_external_minus_blind_external", "external", "blind_external", 6),
              ("paired_external_minus_wait", "external", "wait", 7)]
    for name, a, b, off in legacy:
        if a in conds and b in conds:
            out[name] = bootstrap_ci(d(a, b), seed=seed + off, n_boot=n_boot)
            out["n_pairs"][name] = n_pairs(a, b)
    if "external" in conds and "blind" in conds:
        out["sign_p_external_minus_blind"] = sign_test_p(d("external", "blind"))
    if "wait" in conds and "blind" in conds:
        out["sign_p_wait_minus_blind"] = sign_test_p(d("wait", "blind"))

    out["changed_rate"] = {c: _mean_key(recs, f"changed_{c}") for c in conds}
    out["to_plurality_rate"] = {c: _mean_key(recs, f"to_plurality_{c}") for c in conds}
    out["trunc_rate"] = {c: _mean_key(recs, f"trunc_{c}") for c in conds}
    out["repro_rate"] = {c: _mean_key(recs, f"repro_{c}") for c in conds}
    out["tokens_mean"] = {c: _mean_key(recs, f"tokens_{c}") for c in conds}
    if "wait" in conds:
        out["no_new_boxed_rate_wait"] = _mean_key(recs, "no_new_boxed_wait")

    # ── F1: fact 대비 짝 차이 + Holm ────────────────────────────────────────────
    content = [c for c in conds if c in CONDS_CONTENT]
    if FACT_ALIAS in conds and content:
        raw_p: dict = {}
        for j, arm in enumerate(content):
            out[f"paired_{arm}_minus_fact"] = bootstrap_ci(d(arm, FACT_ALIAS), seed=seed + 100 + j,
                                                           n_boot=n_boot)
            out["n_pairs"][f"paired_{arm}_minus_fact"] = n_pairs(arm, FACT_ALIAS)
            p = sign_test_p(d(arm, FACT_ALIAS))
            out.setdefault("sign_p_vs_fact", {})[arm] = p
            if arm in CONTENT_ARMS:
                raw_p[arm] = p
        if raw_p:
            out["holm_adjusted_p"] = holm_adjust(raw_p)
        out["tested_arms"] = [c for c in content if c in CONTENT_ARMS]
        # ★S4 팔은 **자기 가족 안에서만** Holm 보정한다(F1 의 3팔 보정을 건드리지 않는다).
        s4 = [c for c in content if c in S4_ARMS]
        if s4:
            out["tested_arms_s4"] = s4
            out["holm_adjusted_p_s4"] = holm_adjust(
                {arm: out["sign_p_vs_fact"][arm] for arm in s4})
        # 진단 대조 — 라벨 정박(donor) / 길이·어조(pad)
        if "fact_switch" in conds and "fact_switch_donor" in conds:
            out["diag_switch_minus_donor"] = bootstrap_ci(d("fact_switch", "fact_switch_donor"),
                                                          seed=seed + 200, n_boot=n_boot)
            out["n_pairs"]["diag_switch_minus_donor"] = n_pairs("fact_switch", "fact_switch_donor")
        if "fact_pad" in conds:
            out["diag_minus_pad"] = {
                arm: bootstrap_ci(d(arm, "fact_pad"), seed=seed + 210 + j, n_boot=n_boot)
                for j, arm in enumerate(c for c in content if c in CONTENT_ARMS)}

    for c in conds:
        out[f"rescue_{c}"] = out[f"p_{c}"]["mean"]
    if "external" in conds and "wait" in conds:
        out["blind_spot"] = (out["rescue_external"] - out["rescue_wait"]
                             if math.isfinite(out["rescue_external"])
                             and math.isfinite(out["rescue_wait"]) else _NAN)
    return out


def summarize_correct(recs: Sequence[dict], *, k: int = 0, seed: int = 0,
                      n_boot: int = 2000, conds: Sequence[str] = CONDS_CORRECT) -> dict:
    """정답(거짓-경보) 행 요약. ★조건마다 **그 조건이 있는 행**만 쓴다 — fact_notx 는 X 후보
    (다수결 아닌 오답 형제 답)가 없는 행에서 건너뛰므로 n 이 조건마다 다를 수 있다."""
    n = len(recs)
    conds = list(conds)
    out: dict = {
        "n_rollouts": n, "k": k, "conds": conds,
        "n_rows": {c: sum(1 for r in recs if f"flip_{c}" in r) for c in conds},
        "flip_wrong_rate": {c: bootstrap_ci([r[f"flip_{c}"] for r in recs if f"flip_{c}" in r],
                                            seed=seed + i, n_boot=n_boot)
                            for i, c in enumerate(conds)},
        "changed_rate": {c: _mean_key(recs, f"changed_{c}") for c in conds},
        "trunc_rate": {c: _mean_key(recs, f"trunc_{c}") for c in conds},
    }
    return out


def stratified_summary(wrecs: Sequence[dict], *, conds: Sequence[str],
                       arms: Sequence[str] = CONTENT_ARMS, base: str = FACT_ALIAS,
                       seed: int = 0, n_boot: int = 2000) -> dict:
    """★H2 요약 — **모집단(pop_kind) × 합의 상태(agree_state)** 격자.

    각 칸에 (a) 팔별 rescue(절대 구제율) CI, (b) 짝지은 (arm − base) CI + 부호검정 p,
    (c) 그 칸 안에서 `arms` 가족에 대한 Holm 보정 p 를 낸다. `pass@8 = 0` 모집단
    (`allwrong`)에서는 **기준선이 정의상 0**(8개 롤아웃 전부 오답)이므로 (a) 가 핵심
    지표다 — 거기서 맞는 재시도는 **새 해답**이다. 고합의 칸에는 `[CONSENSUS-WRONG]`
    플래그를 붙인다(재표본으로는 못 뚫는 자리).
    상태 "ALL" 은 그 모집단 전체(합)다."""
    conds = [c for c in conds]
    arms = [a for a in arms if a in conds]
    out: dict = {}
    pops = [p for p in (POP_MIXED, POP_ALLWRONG)
            if any(r.get("pop_kind") == p for r in wrecs)]
    for pi, pop in enumerate(pops):
        rows = [r for r in wrecs if r.get("pop_kind") == pop]
        cells: dict = {}
        states = ["ALL"] + [s for s in AGREE_STATES
                            if any(r.get("agree_state") == s for r in rows)]
        for si, st in enumerate(states):
            sub = rows if st == "ALL" else [r for r in rows if r.get("agree_state") == st]
            off = seed + 1000 * (pi + 1) + 37 * si
            cell: dict = {"n_rows": len(sub), "pop_kind": pop, "agree_state": st,
                          "dom_frac_mean": _mean_key(sub, "dom_frac"),
                          "own_in_dominant_rate": _mean_key(sub, "own_in_dominant"),
                          "n_correct_sib_mean": _mean_key(sub, "n_correct_sib")}
            cell["rescue"] = {c: bootstrap_ci([r[f"p_{c}"] for r in sub if f"p_{c}" in r],
                                              seed=off + i, n_boot=n_boot)
                              for i, c in enumerate(conds)}
            cell["trunc_rate"] = {c: _mean_key(sub, f"trunc_{c}") for c in conds}
            if pop == POP_ALLWRONG:
                # ★pass@8 = 0 → 기준선 0. 절대 구제율이 그대로 «새 해답» 비율이다.
                cell["absolute_rescue"] = dict(cell["rescue"])
            if base in conds:
                paired: dict = {}
                signp: dict = {}
                for j, arm in enumerate(a for a in conds if a != base):
                    d = [r[f"p_{arm}"] - r[f"p_{base}"] for r in sub
                         if f"p_{arm}" in r and f"p_{base}" in r]
                    paired[arm] = bootstrap_ci(d, seed=off + 200 + j, n_boot=n_boot)
                    signp[arm] = sign_test_p(d)
                cell["paired_minus_base"] = paired
                cell["sign_p"] = signp
                fam = {a: signp[a] for a in arms if a in signp}
                if fam:
                    cell["holm_adjusted_p"] = holm_adjust(fam)
            cell["flags"] = (["[CONSENSUS-WRONG]"]
                             if pop == POP_ALLWRONG and st in CONSENSUS_STATES else [])
            cells[st] = cell
        out[pop] = cells
    out["base"] = base
    out["tested_arms"] = arms
    return out


def state_correct_summary(crecs: Sequence[dict], *, conds: Sequence[str] = CONDS_CORRECT_LOWAGREE,
                          seed: int = 0, n_boot: int = 2000) -> dict:
    """저합의 정답 행(거짓-경보) — 상태별 flip_wrong. `pop_kind=correct_lowagree` 행만 본다."""
    rows = [r for r in crecs if r.get("pop_kind") == POP_CORRECT_LOWAGREE]
    out: dict = {"n_rows": len(rows), "conds": list(conds), "by_state": {}}
    states = ["ALL"] + [s for s in LOWAGREE_STATES
                        if any(r.get("agree_state") == s for r in rows)]
    for si, st in enumerate(states):
        sub = rows if st == "ALL" else [r for r in rows if r.get("agree_state") == st]
        out["by_state"][st] = {
            "n_rows": len(sub),
            "flip_wrong_rate": {c: bootstrap_ci([r[f"flip_{c}"] for r in sub if f"flip_{c}" in r],
                                                seed=seed + 500 + 11 * si + i, n_boot=n_boot)
                                for i, c in enumerate(conds)},
            "trunc_rate": {c: _mean_key(sub, f"trunc_{c}") for c in conds},
        }
    return out


def state_markdown(strat: dict, cstate: dict | None = None) -> str:
    """H2 격자 표 — 모집단 × 상태 × 팔. allwrong 의 절대 구제율을 먼저 크게 낸다."""
    if not strat:
        return ""
    base = strat.get("base", FACT_ALIAS)
    arms = strat.get("tested_arms") or []
    lines = ["", "### H2 — 합의 상태 × 리셋 자리 습관", "",
             f"상태는 **gold 없이** 형제 {8} 개의 boxed 답 동치 군집으로만 붙였다 "
             "(ALL_SAME=K · DOMINANT 5–7 · SPLIT 3–4 · SCATTER ≤2 · NOANS). "
             f"Holm 가족 = {arms} (모집단·상태 칸마다 따로).", ""]
    for pop in (POP_ALLWRONG, POP_MIXED):
        cells = strat.get(pop)
        if not cells:
            continue
        if pop == POP_ALLWRONG:
            lines += [f"#### {pop} — pass@8 = 0 (기준선 구제율 = 0, 맞는 재시도는 **새 해답**)",
                      "", "| state | n | 팔 | **절대 구제(CI)** | arm−fact | Holm p | trunc | flags |",
                      "|---|---|---|---|---|---|---|---|"]
        else:
            lines += [f"#### {pop} — MIXED 오답 행", "",
                      "| state | n | 팔 | rescue(CI) | arm−fact | Holm p | trunc | flags |",
                      "|---|---|---|---|---|---|---|---|"]
        for st, cell in cells.items():
            conds = list((cell.get("rescue") or {}).keys())
            for c in conds:
                lines.append(
                    f"| {st} | {cell['n_rows']} | {c} | "
                    f"{_f((cell.get('rescue') or {}).get(c))} | "
                    f"{_f((cell.get('paired_minus_base') or {}).get(c)) if c != base else '—'} | "
                    f"{_f((cell.get('holm_adjusted_p') or {}).get(c)) if c in arms else '—'} | "
                    f"{_f((cell.get('trunc_rate') or {}).get(c))} | "
                    f"{' '.join(cell.get('flags') or []) or '—'} |")
        lines += [""]
    if cstate and cstate.get("n_rows"):
        lines += ["#### correct_lowagree — 저합의(SPLIT/SCATTER) **정답** 행의 거짓 경보",
                  "", "| state | n | cond | flip_wrong(CI) | trunc |", "|---|---|---|---|---|"]
        for st, cell in (cstate.get("by_state") or {}).items():
            for c in cstate.get("conds") or []:
                lines.append(f"| {st} | {cell['n_rows']} | {c} | "
                             f"{_f((cell.get('flip_wrong_rate') or {}).get(c))} | "
                             f"{_f((cell.get('trunc_rate') or {}).get(c))} |")
        lines += [""]
    return "\n".join(lines)


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def _sig_pos(ci: dict | None) -> bool:
    ci = ci or {}
    if not _finite(ci, "lo", "hi", "mean"):
        return False
    return (ci["lo"] > 0 or ci["hi"] < 0) and ci["mean"] > 0


def activation_pass(wrong: dict, correct: dict) -> bool:
    """external−blind: CI 가 0 제외 ∧ 평균 ≥ +0.05 ∧ flip_wrong_rate(external, correct) ≤ 0.10
    ∧ external−blind_external: CI 가 0 제외 ∧ 평균 > 0."""
    eb = wrong.get("paired_external_minus_blind") or {}
    if not _finite(eb, "lo", "hi", "mean"):
        return False
    ok_eb = (eb["lo"] > 0 or eb["hi"] < 0) and eb["mean"] >= PASS_DELTA_EXT
    ok_ebe = _sig_pos(wrong.get("paired_external_minus_blind_external"))
    flip = (correct.get("flip_wrong_rate") or {}).get("external") or {}
    ok_flip = isinstance(flip.get("mean"), (int, float)) and math.isfinite(flip["mean"]) \
        and flip["mean"] <= MAX_FLIP_WRONG
    return bool(ok_eb and ok_ebe and ok_flip)


def self_trigger_pass(wrong: dict, correct: dict) -> bool:
    """wait−blind: CI 가 0 제외 ∧ 평균 > 0 ∧ flip_wrong_rate(wait, correct) ≤ 0.10."""
    ok_wb = _sig_pos(wrong.get("paired_wait_minus_blind"))
    flip = (correct.get("flip_wrong_rate") or {}).get("wait") or {}
    ok_flip = isinstance(flip.get("mean"), (int, float)) and math.isfinite(flip["mean"]) \
        and flip["mean"] <= MAX_FLIP_WRONG
    return bool(ok_wb and ok_flip)


def flip_baseline(correct: dict) -> tuple[float, str]:
    """F1 flip 절의 **기준선** — 같은 정답-행 모집단의 blind flip_wrong(없으면 fact).
    반환 (평균, 쓰인 조건 이름). 둘 다 없으면 (nan, "") — 그러면 flip 절은 판정 불가다."""
    flips = correct.get("flip_wrong_rate") or {}
    for key in FLIP_BASELINE_KEYS:
        m = (flips.get(key) or {}).get("mean", _NAN)
        if isinstance(m, (int, float)) and math.isfinite(float(m)):
            return float(m), key
    return _NAN, ""


def reset_content_pass(wrong: dict, correct: dict) -> tuple[bool, list, dict]:
    """RESET-CONTENT PASS ⇔ 어떤 내용 팔이 (arm − fact) CI 가 0 제외 ∧ 평균 ≥ +0.03 ∧
    Holm 보정 부호검정 p < .05 ∧ **정답 행 flip_wrong ≤ 기준선 flip_wrong + 0.02**
    ∧ 오답 행 trunc ≤ .20.

    ★flip 절은 0916 에 절대 상한(.10)에서 **상대 규칙**으로 고쳤다 — 위 `FLIP_MARGIN`
      주석 참조. 기준선은 `flip_baseline`(같은 정답-행 모집단의 blind, 없으면 fact)이고,
      기준선이 아예 없으면 이 절은 판정할 수 없으므로 **통과시키지 않는다**.
    반환: (통과 여부, 승자 목록, 팔별 진단 플래그)."""
    arms = [a for a in (wrong.get("tested_arms") or []) if a in CONTENT_ARMS]
    base, _base_key = flip_baseline(correct)
    flip_cap = base + FLIP_MARGIN if math.isfinite(base) else _NAN
    adj = wrong.get("holm_adjusted_p") or {}
    flips = correct.get("flip_wrong_rate") or {}
    truncs = wrong.get("trunc_rate") or {}
    winners: list = []
    flags: dict = {}
    for arm in arms:
        dd = wrong.get(f"paired_{arm}_minus_fact") or {}
        fl = (flips.get(arm) or {}).get("mean", _NAN)
        tr = truncs.get(arm, _NAN)
        p = adj.get(arm, 1.0)
        ok = (_finite(dd, "lo", "hi", "mean") and (dd["lo"] > 0 or dd["hi"] < 0)
              and dd["mean"] >= PASS_DELTA_CONTENT
              and isinstance(p, (int, float)) and math.isfinite(p) and p < 0.05
              and isinstance(fl, (int, float)) and math.isfinite(fl)
              and math.isfinite(flip_cap) and fl <= flip_cap
              and isinstance(tr, (int, float)) and math.isfinite(tr) and tr <= MAX_TRUNC_CONTENT)
        if ok:
            winners.append(arm)
        f: list = []
        if isinstance(fl, (int, float)) and math.isfinite(fl) and math.isfinite(flip_cap) \
                and fl > flip_cap:
            f.append("[FLIP>BLIND]")
        pad = (wrong.get("diag_minus_pad") or {}).get(arm)
        if pad is not None and not _sig_pos(pad):
            f.append("[TOKEN-CONFOUND?]")
        if arm == "fact_switch":
            don = wrong.get("diag_switch_minus_donor")
            if don is not None and not _sig_pos(don):
                f.append("[LABEL-ANCHOR?]")
        flags[arm] = f
    return bool(winners), winners, flags


def s4_content_pass(wrong: dict, correct: dict) -> tuple[bool, list, dict]:
    """S4″/S4′-lite 판정 — F1 과 **같은 판정선**을 S4 가족(fact_reread, habit_self,
    habit_sib)에 적용한다: (arm − fact) CI 가 0 제외 ∧ 평균 ≥ +0.03 ∧ Holm(S4 가족 안) p<.05
    ∧ 오답 행 trunc ≤ .20 ∧ (그 팔이 정답 행에도 돌았다면) flip ≤ 기준선+0.02.
    ★습관 팔은 정답 행에 돌지 않는다(그 행의 X·형제 분포가 gold 를 드러낸다) — 그래서
      flip 절은 «있으면 본다»다. 반환 (통과, 승자, 팔별 플래그)."""
    arms = list(wrong.get("tested_arms_s4") or [])
    adj = wrong.get("holm_adjusted_p_s4") or {}
    flips = correct.get("flip_wrong_rate") or {}
    truncs = wrong.get("trunc_rate") or {}
    base, _ = flip_baseline(correct)
    winners: list = []
    flags: dict = {}
    for arm in arms:
        dd = wrong.get(f"paired_{arm}_minus_fact") or {}
        tr = truncs.get(arm, _NAN)
        p = adj.get(arm, 1.0)
        fl = (flips.get(arm) or {}).get("mean", _NAN)
        ok_flip = True
        f: list = []
        if isinstance(fl, (int, float)) and math.isfinite(fl):
            ok_flip = math.isfinite(base) and fl <= base + FLIP_MARGIN
            if not ok_flip:
                f.append("[FLIP>BLIND]")
        ok = (_finite(dd, "lo", "hi", "mean") and (dd["lo"] > 0 or dd["hi"] < 0)
              and dd["mean"] >= PASS_DELTA_CONTENT
              and isinstance(p, (int, float)) and math.isfinite(p) and p < 0.05
              and isinstance(tr, (int, float)) and math.isfinite(tr) and tr <= MAX_TRUNC_CONTENT
              and ok_flip)
        if ok:
            winners.append(arm)
        if arm in HABIT_ARMS:
            vs = (wrong.get("note_variance") or {}).get(arm)
            if vs is not None:
                f.append("[VAR-SIGNAL]" if variance_signal_pass(vs) else "[NO-VAR-SIGNAL]")
        gen = (wrong.get("habit_hist") or {}).get(arm, {}).get("generic")
        n_all = sum((wrong.get("habit_hist") or {}).get(arm, {}).values()) or 0
        if gen and n_all and gen / n_all >= 0.5:
            f.append("[MOSTLY-GENERIC]")
        flags[arm] = f
    return bool(winners), winners, flags


def _f(v) -> str:
    if isinstance(v, dict):
        if "mean" in v:
            return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
        return "; ".join(f"{k}={_f(x)}" for k, x in v.items())
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(wrong: dict, correct: dict) -> str:
    act = activation_pass(wrong, correct)
    trig = self_trigger_pass(wrong, correct)
    wkeys = ["n_rollouts", "k", "p_blind", "p_wait", "p_external", "p_blind_external",
            "paired_external_minus_blind", "paired_wait_minus_blind",
            "paired_external_minus_blind_external", "paired_external_minus_wait",
            "sign_p_external_minus_blind", "sign_p_wait_minus_blind",
            "changed_rate", "to_plurality_rate", "trunc_rate", "no_new_boxed_rate_wait",
            "rescue_blind", "rescue_wait", "rescue_external", "blind_spot"]
    ckeys = ["n_rollouts", "k", "flip_wrong_rate", "changed_rate", "trunc_rate"]
    conds = wrong.get("conds") or list(CONDS_WRONG)
    wkeys += [f"rescue_{c}" for c in conds if f"rescue_{c}" in wrong and f"rescue_{c}" not in wkeys]
    wkeys += ["repro_rate", "tokens_mean"]
    wkeys += [f"paired_{a}_minus_fact" for a in conds if f"paired_{a}_minus_fact" in wrong]
    wkeys += [k for k in ("sign_p_vs_fact", "holm_adjusted_p", "diag_switch_minus_donor",
                          "diag_minus_pad", "label_leak") if k in wrong]
    lines = ["## math_activation_gate — 능력은 있는가, 스스로 켜지는가", "",
             "### 오답 행", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(wrong.get(k))} |" for k in wkeys if k in wrong]
    lines += ["", "### 정답 행(거짓-경보 대조)", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(correct.get(k))} |" for k in ckeys if k in correct]
    if "external" in conds and "blind" in conds:
        lines += ["", f"**ACTIVATION {'PASS' if act else 'FAIL'}** — (external−blind) CI 가 0 제외 "
                      f"∧ 평균 ≥ +{PASS_DELTA_EXT} ∧ flip_wrong_rate(external) ≤ "
                      f"{MAX_FLIP_WRONG} ∧ (external−blind_external) CI 가 0 제외 ∧ 평균 > 0",
                 f"**SELF-TRIGGER {'PASS' if trig else 'FAIL'}** — (wait−blind) CI 가 0 제외 "
                      f"∧ 평균 > 0 ∧ flip_wrong_rate(wait) ≤ {MAX_FLIP_WRONG}",
                 f"**BLIND-SPOT** = rescue(external) − rescue(wait) = "
                      f"{_f(wrong.get('blind_spot'))}", ""]
    if wrong.get("tested_arms"):
        ok, winners, flags = reset_content_pass(wrong, correct)
        base, bkey = flip_baseline(correct)
        cap = base + FLIP_MARGIN
        lines += ["", "### F1 — 리셋 자리의 메타 내용(오답 본문 없음)",
                  f"**RESET-CONTENT {'PASS' if ok else 'FAIL'}** — 어떤 내용 팔이 (arm − fact) "
                  f"CI 가 0 제외 ∧ 평균 ≥ +{PASS_DELTA_CONTENT} ∧ Holm p<.05 ∧ "
                  f"flip_wrong ≤ 기준선+{FLIP_MARGIN} ∧ trunc ≤ {MAX_TRUNC_CONTENT}; "
                  f"winners={winners}",
                  f"**[FLIP-RULE]** 기준선 flip_wrong({bkey or '없음'}) = {_f(base)} → 상한 "
                  f"{_f(cap)} (= 기준선 + {FLIP_MARGIN}). 절대 상한 {MAX_FLIP_WRONG} 은 "
                  "오측정이었다 — 신호 없는 blind 재표본조차 정답 난이도-5 행을 "
                  f"{_f(base)} 비율로 뒤집으므로 어떤 팔도 통과할 수 없었다.", "",
                  "| arm | arm−fact | Holm p | flip_wrong | trunc | arm−fact_pad | flags |",
                  "|---|---|---|---|---|---|---|"]
        for arm in wrong["tested_arms"]:
            lines.append(
                f"| {arm} | {_f(wrong.get(f'paired_{arm}_minus_fact'))} | "
                f"{_f((wrong.get('holm_adjusted_p') or {}).get(arm))} | "
                f"{_f(((correct.get('flip_wrong_rate') or {}).get(arm)) or {})} | "
                f"{_f((wrong.get('trunc_rate') or {}).get(arm))} | "
                f"{_f((wrong.get('diag_minus_pad') or {}).get(arm))} | "
                f"{' '.join(flags.get(arm) or []) or '—'} |")
        lines += ["", f"진단: (fact_switch − fact_switch_donor) = "
                      f"{_f(wrong.get('diag_switch_minus_donor'))} — 라벨 정박 대조", ""]
    if wrong.get("tested_arms_s4"):
        ok4, win4, fl4 = s4_content_pass(wrong, correct)
        lines += ["", "### S4″/S4′-lite — 정책이 쓴 회복 습관 노트(오답 본문 없음)",
                  f"**S4-LITE {'PASS' if ok4 else 'FAIL'}** — (arm − fact) CI 가 0 제외 ∧ "
                  f"평균 ≥ +{PASS_DELTA_CONTENT} ∧ Holm(S4 가족) p<.05 ∧ "
                  f"trunc ≤ {MAX_TRUNC_CONTENT}; winners={win4}", "",
                  "| arm | rescue | arm−fact | Holm(S4) p | trunc | flags |",
                  "|---|---|---|---|---|---|"]
        for arm in wrong["tested_arms_s4"]:
            lines.append(
                f"| {arm} | {_f(wrong.get(f'rescue_{arm}'))} | "
                f"{_f(wrong.get(f'paired_{arm}_minus_fact'))} | "
                f"{_f((wrong.get('holm_adjusted_p_s4') or {}).get(arm))} | "
                f"{_f((wrong.get('trunc_rate') or {}).get(arm))} | "
                f"{' '.join(fl4.get(arm) or []) or '—'} |")
        if wrong.get("habit_hist"):
            lines += ["", "노트 습관 히스토그램: "
                      + "; ".join(f"{a}={h}" for a, h in wrong["habit_hist"].items())]
        if wrong.get("habit_leak"):
            lines += ["노트 누출 통계: "
                      + "; ".join(f"{a}={h}" for a, h in wrong["habit_leak"].items())]
        if wrong.get("habit_minority"):
            lines += ["habit_sib 자기 답이 소수(minority)인 행 비율: "
                      + _f(wrong["habit_minority"])]
        for arm, vs in (wrong.get("note_variance") or {}).items():
            lines += [f"**[NOTE-VARIANCE {arm}]** ratio={_f(vs.get('ratio'))} "
                      f"(arm {_f(vs.get('var_arm'))} / ref {_f(vs.get('var_ref'))}) "
                      f"p={_f(vs.get('p'))} n_rows={vs.get('n_rows')} ref={vs.get('ref')} → "
                      f"{'신호 있음' if variance_signal_pass(vs) else '신호 없음'} "
                      f"(사전등록 ratio ≥ {VAR_RATIO_MIN} ∧ p < {VAR_P_MAX}). ⚠️노트당 재시도가 "
                      "1회이므로 «행 안 과분산»은 원리상 잴 수 없다 — 이 비는 같은 행의 "
                      "단일 프롬프트 팔과의 **분산비**다(note_variance_stat docstring)."]
        lines += [""]
    if wrong.get("agree_strata"):
        lines += [state_markdown(wrong["agree_strata"], correct.get("lowagree_states"))]
    return "\n".join(lines)


def resummarize_activation(gens_path: str, *, out_dir: str, seed: int = 11,
                           n_boot: int = 2000) -> int:
    r"""`--resummarize` — 생성 없이(GPU 없이) 이미 있는 `gens.jsonl` 에서 wrong/correct
    요약·짝지은 Δ·Holm·PASS 를 다시 만든다.
    ★한계(설계상): 이 게이트의 `gens.jsonl` 은 **개별 생성**만 담는다 — roll_id·group_id·
      population·cond·(원 롤아웃의) r_corr·(재생성의) gen_r_corr·text·truncated. 원 롤아웃의
      problem·gold·원래 답 문자열은 **없다**. 그래서 이 모드는 accuracy 기반 열(`p_c`,
      `trunc_c`, `flip_c`)만 다시 재고, 원문 비교가 필요한 열(`changed_*`, `repro_*`,
      `to_plurality_*`, `tokens_mean`, `no_new_boxed_rate_wait`)은 **내지 않는다**(재현이
      아니라 새 생성이 있어야 잴 수 있다 — 조용히 0/nan 으로 채우지 않고 아예 뺀다).
    ★그래도 짝지은 Δ·Holm·PASS 는 accuracy 열만으로 완결된다 — 0915 수리(모든 조건이 있어야
      쓰는 게 아니라 **쌍마다** 짝짓는 것)의 효과를 재생성 없이 이 모드로 확인할 수 있다.
    ★재채점(0916): `gens.jsonl` 의 `r_corr` 는 **1차 시도**의 정오다(재생성은 `gen_r_corr`).
      `scripts/local/regrade_jsonl.py --schema gate` 가 이 값을 재채점된 texts.jsonl 에서
      새로 고치므로, 이제 «오답» 모집단인데 1차 시도가 **사실은 정답**이던 단위가 드러난다.
      그 단위는 "직전 시도가 틀렸다" 는 **거짓 사실**을 받은 것이라 오답 모집단 통계에서
      **빼고**(`n_units_dropped_a1_correct`), 나머지는 재채점된 `gen_r_corr` 로 그대로 센다.
      재채점 전 파일에서는 오답 단위의 r_corr 가 전부 0 이라 드롭이 0 이고 숫자가 그대로다."""
    rows = [json.loads(ln) for ln in open(gens_path) if ln.strip()]
    if not rows:
        raise SystemExit(f"[act] 빈 gens 파일: {gens_path}")
    agg: dict = {}
    pop_of: dict = {}
    gid_of: dict = {}
    meta_of: dict = {}
    a1_of: dict = {}
    for r in rows:
        # ★H2: 행의 정체는 (roll_id, pop_kind) 다 — 같은 정답 롤아웃이 주 모집단과
        #   저합의 대조에 동시에 나올 수 있으므로 pop_kind 까지 키에 넣는다.
        key = (r["roll_id"], r.get("pop_kind", ""))
        rid, c = key, r["cond"]
        agg.setdefault((rid, c), []).append(r)
        pop_of.setdefault(rid, r.get("population", "wrong"))
        a1_of[rid] = max(a1_of.get(rid, 0), int(r.get("r_corr") or 0))
        gid_of.setdefault(rid, r.get("group_id", ""))
        meta_of.setdefault(rid, {"pop_kind": r.get("pop_kind", ""),
                                 "agree_state": r.get("agree_state", ""),
                                 "dom_frac": r.get("dom_frac", _NAN),
                                 "own_in_dominant": r.get("own_in_dominant", 0),
                                 "n_correct_sib": r.get("n_correct_sib", 0)})

    # ★재채점 뒤: «오답» 단위인데 1차 시도가 사실은 정답이면 거짓 사실을 받은 단위다 — 뺀다.
    dropped = {rid for rid, pop in pop_of.items() if pop == "wrong" and a1_of.get(rid, 0) == 1}
    for rid in dropped:
        pop_of.pop(rid, None)
    if dropped:
        rows = [r for r in rows if (r["roll_id"], r.get("pop_kind", "")) not in dropped]
        print(f"[act] 1차 시도가 재채점으로 정답이 된 오답 단위 {len(dropped)} 개 제외", flush=True)

    conds_seen = {c for (rid, c) in agg if rid not in dropped}
    conds = [c for c in ALL_CONDS if c in conds_seen]
    cconds = [c for c in CONDS_CORRECT_ELIGIBLE if c in conds_seen]
    clow_conds = [c for c in CONDS_CORRECT_LOWAGREE if c in conds_seen] \
        if any(m["pop_kind"] == POP_CORRECT_LOWAGREE for m in meta_of.values()) else []

    wrecs = []
    for rid, pop in pop_of.items():
        if pop != "wrong":
            continue
        rec: dict = {"roll_id": rid[0], "group_id": gid_of[rid], **meta_of[rid]}
        for c in conds:
            xs = agg.get((rid, c))
            if not xs:
                continue     # ★쌍마다 짝짓는다 — 이 조건만 없어도 나머지는 살아 있다
            rec[f"p_{c}"] = _p([x["gen_r_corr"] for x in xs])
            rec[f"trunc_{c}"] = _p([x.get("truncated", 0) for x in xs])
        if any(k.startswith("p_") for k in rec):
            wrecs.append(rec)

    crecs = []
    for rid, pop in pop_of.items():
        if pop != "correct":
            continue
        rec = {"roll_id": rid[0], "group_id": gid_of[rid], **meta_of[rid]}
        for c in cconds:
            xs = agg.get((rid, c))
            if not xs:
                continue
            rec[f"flip_{c}"] = _p([1 - x["gen_r_corr"] for x in xs])
            rec[f"trunc_{c}"] = _p([x.get("truncated", 0) for x in xs])
        if any(k.startswith("flip_") for k in rec):
            crecs.append(rec)

    wsumm = summarize_wrong(wrecs, k=0, seed=seed, n_boot=n_boot, conds=conds)
    csumm = summarize_correct(crecs, k=0, seed=seed, n_boot=n_boot, conds=cconds)
    if any(r.get("pop_kind") for r in wrecs):
        wsumm["agree_strata"] = stratified_summary(wrecs, conds=conds, seed=seed, n_boot=n_boot)
    if clow_conds:
        csumm["lowagree_states"] = state_correct_summary(crecs, conds=clow_conds, seed=seed,
                                                         n_boot=n_boot)
    if "external" in conds and "blind" in conds:
        wsumm["pass_activation"] = int(activation_pass(wsumm, csumm))
        wsumm["pass_self_trigger"] = int(self_trigger_pass(wsumm, csumm))
    # ★습관 팔이 있으면 gens.jsonl 의 노트 부류·누출 이유·분산비도 다시 낸다(재생성 불필요).
    hab_seen = [c for c in HABIT_ARMS if c in conds_seen]
    if hab_seen:
        hist: dict = {}
        hleak: dict = {}
        for r in rows:
            c = r["cond"]
            if c not in HABIT_ARMS:
                continue
            hist.setdefault(c, {})
            cls = r.get("note_class") or habit_class(r.get("note", ""))
            hist[c][cls] = hist[c].get(cls, 0) + 1
            if r.get("note_reason"):
                hleak.setdefault(c, {})
                hleak[c][r["note_reason"]] = hleak[c].get(r["note_reason"], 0) + 1
        wsumm["habit_hist"] = hist
        wsumm["habit_leak"] = hleak
        ref = next((c for c in ("fact_reread", FACT_ALIAS, "blind") if c in conds), "")
        wsumm["note_variance"] = {}
        for arm in hab_seen:
            rows_v = []
            for rid, pop in pop_of.items():
                if pop != "wrong":
                    continue
                xa, xb = agg.get((rid, arm)), agg.get((rid, ref))
                if xa and xb:
                    rows_v.append(([x["gen_r_corr"] for x in xa],
                                   [x["gen_r_corr"] for x in xb]))
            st = note_variance_stat(rows_v, seed=seed + 300)
            st["ref"] = ref
            wsumm["note_variance"][arm] = st
    if wsumm.get("tested_arms"):
        ok, winners, flags = reset_content_pass(wsumm, csumm)
        wsumm["pass_reset_content"] = int(ok)
        wsumm["reset_content_winners"] = winners
        wsumm["reset_content_flags"] = flags
    if wsumm.get("tested_arms_s4"):
        ok4, win4, fl4 = s4_content_pass(wsumm, csumm)
        wsumm["pass_s4_lite"] = int(ok4)
        wsumm["s4_lite_winners"] = win4
        wsumm["s4_lite_flags"] = fl4
    meta = {"resummarize_from": str(gens_path), "seed": seed, "conds": conds,
            "conds_correct": cconds, "n_units_dropped_a1_correct": len(dropped),
            "n_wrong_candidates": len(wrecs),
            "n_correct_candidates": len(crecs), "n_generations": len(rows),
            "note": "changed_*/repro_*/to_plurality_*/tokens_mean/no_new_boxed_rate_wait 는 "
                    "gens.jsonl 에 원 롤아웃 problem/gold/원래 답이 없어 못 재고 뺐다."}
    summ = {"wrong": wsumm, "correct": csumm, "meta": meta}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_row.jsonl").open("w") as fh:
        for r in wrecs:
            fh.write(json.dumps({**r, "population": "wrong"}, ensure_ascii=False) + "\n")
        for r in crecs:
            fh.write(json.dumps({**r, "population": "correct"}, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(wsumm, csumm)
    (out / "gate_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="", help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_sites", type=int, default=150)
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=8192,
                    help="★모든 재풀이 조건의 생성 예산 — 4k 는 절단률이 정확도 순서를 뒤집는다"
                         "(cd9 G4). 8192 아래로 내리지 말 것.")
    ap.add_argument("--max_prefix_tokens", type=int, default=8192,
                    help="★wait/external 프롬프트는 풀이 전체를 담는다")
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--conds", default=None,
                    help="쉼표로 구분한 조건(기본 = G8 의 blind,wait,external,blind_external — "
                         "기본값이면 옛 실행이 그대로 재현된다). F1: "
                         "blind,fact,fact_effort,fact_switch,fact_switch_donor,fact_notx,fact_pad. "
                         "S4-lite: blind,fact,notx,fact_reread,habit_self,habit_sib. "
                         "`fact`=blind_external · `notx`=fact_notx · `reread`=fact_reread "
                         "(별칭은 프롬프트 바이트 동일). habit_* 와 fact_reread 는 **오답 행만** "
                         "돈다(정답 행에서는 X·형제 분포가 gold 를 드러낸다).")
    ap.add_argument("--label_max_tokens", type=int, default=64,
                    help="fact_switch 사전 패스(방법 이름)의 생성 예산")
    ap.add_argument("--donor_scope", choices=("global", "within_source"), default="global",
                    help="fact_switch_donor 의 기증자 후보 범위. 기본 global 은 옛 동작 그대로"
                         "(전체 롤아웃에서 회전) — 이 게이트는 --rollouts 를 하나만 받으므로"
                         " within_source 는 지금은 global 과 동일하다(소스 표식이 없다).")
    ap.add_argument("--population", choices=tuple(POPULATION_CHOICES), default="mixed",
                    help="오답 모집단. mixed(기본) = 지금까지의 MIXED 오답 행. "
                         "all_wrong = 8개 롤아웃이 **전부 오답**인 문제(pass@8 = 0, 문제당 대표 "
                         "1행 — 지금까지 한 번도 시험되지 않은 195개). both = 둘 다 "
                         "(pop_kind 로 태그되고, 상태별 요약이 모집단마다 따로 나온다; "
                         "both 에서는 저합의(SPLIT/SCATTER) **정답** 행에 fact_switch/fact_notx "
                         "를 돌려 거짓 경보도 잰다).")
    ap.add_argument("--correct_states", default="",
                    help="정답(거짓-경보) 행을 합의 상태로 좁힌다(쉼표, 예 SPLIT,SCATTER). "
                         "기본(빈 값)은 전체 상태 = 필터 없음(옛 동작 바이트 동일). "
                         "H2 에서는 SPLIT,SCATTER 만 돌려 «시험 때 실제로 켜질 자리»의 거짓 "
                         "경보만 잰다(F1 이 이미 전 상태 flip 기준선을 냈다).")
    ap.add_argument("--resummarize", default="",
                    help="생성 없이(GPU 없이) 이 gens.jsonl 을 다시 요약해 --out_dir 에 쓴다.")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.resummarize:
        return resummarize_activation(a.resummarize, out_dir=a.out_dir, seed=a.seed,
                                      n_boot=a.n_boot)
    if not a.rollouts or not a.model_path:
        raise SystemExit("[act] --resummarize 가 없으면 --rollouts 와 --model_path 가 필수다")

    conds = resolve_conds(a.conds)
    cstates = resolve_correct_states(a.correct_states)
    cconds = [c for c in conds if c in CONDS_CORRECT_ELIGIBLE]
    need_label = any(c in conds for c in ("fact_switch", "fact_switch_donor"))
    print(f"[act] 조건(오답) {conds} / 조건(정답) {cconds}", flush=True)

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    # ★H2: 상태(gold 없음)와 정답 형제 수(gold 기반·기록용)를 행마다 붙인다.
    agree = group_agreement(rolls)
    ncorr = group_n_correct(rolls)

    def tag(rows: Sequence[dict], kind: str) -> list[dict]:
        for s in rows:
            st = agree.get(s["group_id"], {})
            s["pop_kind"] = kind
            s["agree_state"] = st.get("state", "NOANS")
            s["dom_frac"] = float(st.get("dom_frac", 0.0))
            s["own_in_dominant"] = int(own_in_dominant(last_boxed(s.get("text", "")), st))
            # ⛔gold 기반 — 분석용 기록일 뿐, 어떤 프롬프트에도 들어가지 않는다.
            s["n_correct_sib"] = int(ncorr.get(s["group_id"], 0))
        return list(rows)

    kinds = POPULATION_CHOICES[a.population]
    wrong: list[dict] = []
    if POP_MIXED in kinds:
        mixed_rows = select_wrong_rollouts(rolls, per_problem=a.per_problem)
        rng.shuffle(mixed_rows)
        wrong += tag(mixed_rows[:a.max_sites], POP_MIXED)
    if POP_ALLWRONG in kinds:
        aw_rows = select_all_wrong_rollouts(rolls, per_problem=1)
        rng.shuffle(aw_rows)
        wrong += tag(aw_rows[:a.max_sites], POP_ALLWRONG)

    pool = select_correct_rollouts(rolls, per_problem=1)
    rng.shuffle(pool)
    # ★`--correct_states` — 정답(거짓-경보) 행을 합의 상태로 좁힌다. 기본(전체 상태)이면
    #   필터가 아무 것도 버리지 않아 옛 동작과 바이트 동일하다.
    if cstates != list(AGREE_STATES):
        pool = [s for s in pool if agree.get(s["group_id"], {}).get("state") in cstates]
    # ★기본 mixed 판에서는 옛 동작 그대로(정답 행 수 = 오답 행 수). both 에서는 오답이 두
    #   모집단으로 늘어나므로 정답 행은 --max_sites 로 묶는다.
    n_corr_take = min(len(pool), max(1, min(len(wrong), a.max_sites)))
    correct = tag(pool[:n_corr_take], POP_CORRECT)
    # ★거짓-경보 대조(저합의 정답 행) — `--population both` 에서는 SPLIT/SCATTER 문제의
    #   정답 행을 `pop_kind=correct_lowagree` 로 **다시 태그**한다. 이 행들은 이미 정답
    #   모집단의 조건(fact_switch·fact_notx 포함)을 돌고 있으므로 **추가 생성이 없다** —
    #   시험 때 «합의가 갈렸다»로 리다이렉트를 켜면 맞은 답을 얼마나 흔드는지를 이 칸이 낸다.
    #   (별도 모집단으로 다시 돌리면 같은 프롬프트를 두 번 생성하고, 저합의 행을 주
    #   기준선에서 빼면 F1/S4 의 flip 기준선이 조용히 편향된다 — 둘 다 피한다.)
    clow: list[dict] = []
    if a.population == "both":
        for s in correct:
            if s["agree_state"] in LOWAGREE_STATES:
                s["pop_kind"] = POP_CORRECT_LOWAGREE
                clow.append(s)
    print(f"[act] 오답 롤아웃 {len(wrong)}개 "
          f"({ {k: sum(1 for s in wrong if s['pop_kind'] == k) for k in kinds} }) / "
          f"정답(거짓-경보) {len(correct)}개 (상태 {cstates} · 그중 저합의 {len(clow)}개)",
          flush=True)
    print("[act] 상태 분포(오답) "
          f"{ {s: sum(1 for r in wrong if r['agree_state'] == s) for s in AGREE_STATES} }",
          flush=True)
    if not wrong:
        raise SystemExit("[act] 오답 후보가 없다 — 입력 롤아웃에 MIXED 오답이 있는지 확인하라.")

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()
    lim = a.max_prefix_tokens + 512

    gans = group_answers(rolls)
    gwrong = group_wrong_answers(rolls)

    # ── fact_switch 사전 패스: 모델이 스스로 붙인 «방법 이름» L ────────────────
    # ★모집단 세 벌(wrong / correct / correct_lowagree)을 한 판으로 다룬다.
    pop_rows: dict = {"wrong": wrong, "correct": correct}
    labs: dict = {p: [GENERIC_LABEL] * len(r) for p, r in pop_rows.items()}
    lab_w, lab_c = labs["wrong"], labs["correct"]     # (옛 이름 유지 — 진단 출력용)
    leak = {"n": 0, "ok": 0, "empty": 0, "digit": 0, "boxed": 0, "answer": 0, "too_long_prompt": 0}
    if need_label:
        lreq, lix = [], []
        for pop, rows in pop_rows.items():
            for si, s in enumerate(rows):
                q = label_prompt(tok, a.variant, s["problem"], s["text"])
                if len(tok.encode(q)) > lim:
                    leak["too_long_prompt"] += 1
                    continue
                lreq.append(q)
                lix.append((pop, si))
        print(f"[act] 라벨 사전 패스 요청 {len(lreq)}개", flush=True)
        louts = llm.generate(lreq, SamplingParams(n=1, temperature=0.0, top_p=1.0,
                                                  max_tokens=a.label_max_tokens, seed=a.seed))
        lab_rows = []
        for (pop, si), o in zip(lix, louts):
            s = pop_rows[pop][si]
            raw = o.outputs[0].text
            lab, why = clean_label(raw, last_boxed(s["text"]))
            leak["n"] += 1
            leak[why] = leak.get(why, 0) + 1
            labs[pop][si] = lab
            lab_rows.append({"population": pop, "roll_id": s["roll_id"], "raw": raw,
                             "label": lab, "reason": why})
        with (out / "labels.jsonl").open("w") as fh:
            for r in lab_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[act] 라벨 누출 통계 {leak}", flush=True)

    # ── 요청 조립 ────────────────────────────────────────────────────────────
    def build(pop: str, si: int, s: dict, c: str) -> str | None:
        """조건 c 의 프롬프트(불가능하면 None — 그 행에서 그 조건을 건너뛴다)."""
        if c == "blind":
            return blind_prompt(tok, a.variant, s["problem"])
        if c == "wait":
            return wait_prompt(tok, a.variant, s["problem"], s["text"])
        if c == "external":
            return external_prompt(tok, a.variant, s["problem"], s["text"])
        if c == "blind_external":
            return blind_external_prompt(tok, a.variant, s["problem"])
        if c == "fact_effort":
            return effort_prompt(tok, a.variant, s["problem"])
        if c == "fact_pad":
            return pad_prompt(tok, a.variant, s["problem"])
        if c == "fact_reread":
            return reread_prompt(tok, a.variant, s["problem"])
        if c == "fact_switch":
            return switch_prompt(tok, a.variant, s["problem"], labs[pop][si])
        if c == "fact_switch_donor":
            return switch_prompt(tok, a.variant, s["problem"],
                                 rotate_donor(labs[pop], si, scope=a.donor_scope))
        if c == "fact_notx":
            if pop == "wrong":
                x = last_boxed(s["text"])          # 그 오답 시도의 최종 답
            else:                                   # 정답 행: 다수결이 아닌 오답 형제의 답
                x = nonplurality_wrong_answer(gwrong.get(s["group_id"], []),
                                              _md.plurality_answer(gans.get(s["group_id"], [])))
            return notx_prompt(tok, a.variant, s["problem"], x) if x else None
        raise RuntimeError(f"[act] 미구현 조건 {c}")

    # ★습관 팔은 «행마다 K개의 서로 다른 노트 × 노트당 재시도 1회»라 이 루프(조건당 한
    #   요청 × n=K)와 판이 다르다 — 아래 별도 블록에서 돈다.
    main_conds = [c for c in conds if c not in HABIT_ARMS]
    clow_conds = [c for c in CONDS_CORRECT_LOWAGREE if c in cconds] if clow else []
    reqs, ix, n_drop, n_skip = [], [], 0, 0
    for pop, rows, cs in (("wrong", wrong, main_conds), ("correct", correct, cconds)):
        for si, s in enumerate(rows):
            for c in cs:
                q = build(pop, si, s, c)
                if q is None:
                    n_skip += 1
                    continue
                if len(tok.encode(q)) > lim:
                    n_drop += 1
                    continue
                reqs.append(q)
                ix.append((pop, si, c))
    print(f"[act] 재풀이 요청 {len(reqs)}개 x K={a.k} (길이로 버린 것 {n_drop}개, "
          f"조건 불가로 건너뛴 것 {n_skip}개, 한도 {lim} 토큰)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    gens = []
    def gen_meta(s: dict) -> dict:
        """★per-gen 에 남기는 H2 태그 — `--resummarize` 가 상태를 잃지 않게 한다."""
        return {"pop_kind": s.get("pop_kind", ""), "agree_state": s.get("agree_state", ""),
                "dom_frac": s.get("dom_frac", _NAN),
                "own_in_dominant": s.get("own_in_dominant", 0),
                "n_correct_sib": s.get("n_correct_sib", 0)}

    for (pop, si, c), o in zip(ix, outs):
        s = pop_rows[pop][si]
        for x in o.outputs:
            corr = grade_math(x.text, s["gold"])
            ans = last_boxed(x.text)
            trunc = int(x.finish_reason == "length")
            agg.setdefault((pop, si, c), []).append(
                {"r_corr": corr, "ans": ans, "trunc": trunc,
                 "ntok": len(getattr(x, "token_ids", None) or [])})
            gens.append({"roll_id": s["roll_id"], "group_id": s["group_id"],
                        "population": ("wrong" if pop == "wrong" else "correct"),
                        "cond": c, "r_corr": s.get("r_corr", 0 if pop == "wrong" else 1),
                        "gen_r_corr": corr, "text": x.text, "truncated": trunc, **gen_meta(s)})
    # ── S4″/S4′-lite 습관 노트 팔 ────────────────────────────────────────────
    # 행마다 K개의 노트를 **온도 1.0** 으로 뽑고(서로 다른 노트), 노트마다 재시도 1회 →
    # 한 행 안에서 «노트가 다르면 결과가 다른가»를 볼 수 있다(note_variance).
    habit_hist: dict = {}
    habit_leak: dict = {}
    habit_minority: list = []
    habit_rows: list = []
    hab_arms = [c for c in conds if c in HABIT_ARMS]
    if hab_arms:
        sib_of = {si: sibling_answer_line(last_boxed(s["text"]),
                                          gans.get(s["group_id"], []))
                  for si, s in enumerate(wrong)}
        for arm in hab_arms:
            nreq, nix = [], []
            for si, s in enumerate(wrong):
                sib = sib_of[si] if arm == "habit_sib" else ""
                if arm == "habit_sib" and not sib:
                    n_skip += 1          # 형제 답이 없어 내부 신호를 만들 수 없는 행
                    continue
                q = habit_ask_prompt(tok, a.variant, s["problem"], s["text"], sib)
                if len(tok.encode(q)) > lim:
                    habit_leak.setdefault(arm, {})
                    habit_leak[arm]["too_long_prompt"] = \
                        habit_leak[arm].get("too_long_prompt", 0) + 1
                    continue
                nreq.append(q)
                nix.append(si)
            print(f"[act] {arm} 노트 사전 패스 {len(nreq)}개 x K={a.k}", flush=True)
            nouts = llm.generate(nreq, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                                      max_tokens=a.label_max_tokens,
                                                      seed=a.seed))
            notes: dict = {}
            for si, o in zip(nix, nouts):
                s = wrong[si]
                own = last_boxed(s["text"])
                sib_answers = gans.get(s["group_id"], [])
                minority = own_is_minority(own, sib_answers)
                if arm == "habit_sib":
                    habit_minority.append(float(minority))
                lst = []
                for j, x in enumerate(o.outputs):
                    note, why = clean_habit(x.text, own)
                    cls = habit_class(note)
                    lst.append((note, why, cls))
                    habit_leak.setdefault(arm, {})
                    habit_leak[arm][why] = habit_leak[arm].get(why, 0) + 1
                    habit_hist.setdefault(arm, {})
                    habit_hist[arm][cls] = habit_hist[arm].get(cls, 0) + 1
                    habit_rows.append({"arm": arm, "roll_id": s["roll_id"], "note_idx": j,
                                       "raw": x.text, "note": note, "reason": why,
                                       "habit_class": cls, "own_minority": int(minority),
                                       "sib_line": sib_of[si] if arm == "habit_sib" else ""})
                notes[si] = lst
            rreq, rix = [], []
            for si, lst in notes.items():
                for j, (note, _why, _cls) in enumerate(lst):
                    q = habit_prompt(tok, a.variant, wrong[si]["problem"], note)
                    if len(tok.encode(q)) > lim:
                        n_drop += 1
                        continue
                    rreq.append(q)
                    rix.append((si, j))
            print(f"[act] {arm} 재풀이 {len(rreq)}개 x 1 (노트당 1회, 예산 {a.max_tokens})",
                  flush=True)
            routs = llm.generate(rreq, SamplingParams(n=1, temperature=1.0, top_p=1.0,
                                                      max_tokens=a.max_tokens, seed=a.seed))
            for (si, j), o in zip(rix, routs):
                s = wrong[si]
                x = o.outputs[0]
                corr = grade_math(x.text, s["gold"])
                trunc = int(x.finish_reason == "length")
                agg.setdefault(("wrong", si, arm), []).append(
                    {"r_corr": corr, "ans": last_boxed(x.text), "trunc": trunc,
                     "ntok": len(getattr(x, "token_ids", None) or [])})
                note, why, cls = notes[si][j]
                gens.append({"roll_id": s["roll_id"], "group_id": s["group_id"],
                             "population": "wrong", "cond": arm, "r_corr": s.get("r_corr", 0),
                             "gen_r_corr": corr, "text": x.text, "truncated": trunc,
                             "note": note, "note_reason": why, "note_class": cls,
                             "note_idx": j, **gen_meta(s)})
        with (out / "habit_notes.jsonl").open("w") as fh:
            for r in habit_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[act] 습관 노트 누출 {habit_leak} / 부류 {habit_hist}", flush=True)

    with (out / "gens.jsonl").open("w") as fh:
        for r in gens:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── per-rollout(오답) ───────────────────────────────────────────────────
    # ★0915 수리: 예전에는 `all(got.values())` 로 **선택한 조건이 전부 있는 행만** 남겨서,
    #   boxed 답이 없어 fact_notx 를 못 만든 행이 **모든** 비교에서 통째로 사라졌다
    #   (`math_meta_content_gate` 가 앓던 것과 같은 표본 손실 버그). 이제는 `any()`로만
    #   거르고, 조건별로 **그 조건이 있는 값만** rec 에 채운다 — 짝짓기는 `summarize_wrong`
    #   의 `d(a, b)` 가 두 열이 **동시에 있는 행**만 골라서 한다(그래서 각 쌍의 n 이
    #   다를 수 있고, `n_pairs_*` 로 그 수를 낸다).
    wrecs = []
    for si, s in enumerate(wrong):
        got = {c: agg.get(("wrong", si, c)) for c in conds}
        if not any(got.values()):
            continue
        plur = _md.plurality_answer(gans.get(s["group_id"], []))
        orig_ans = last_boxed(s["text"])
        rec = {"roll_id": s["roll_id"], "group_id": s["group_id"], "problem": s["problem"],
              "gold": s["gold"], **gen_meta(s)}
        for c in conds:
            xs = got.get(c)
            if not xs:
                continue     # ★이 조건만 없다 — 나머지 조건의 짝짓기는 살아 있다
            rec[f"p_{c}"] = _p([x["r_corr"] for x in xs])
            rec[f"changed_{c}"] = _p([int(bool(x["ans"]) and not answers_equivalent(
                orig_ans, x["ans"])) for x in xs])
            rec[f"repro_{c}"] = _p([int(bool(orig_ans) and bool(x["ans"]) and answers_equivalent(
                orig_ans, x["ans"])) for x in xs])
            rec[f"tokens_{c}"] = _p([x["ntok"] for x in xs])
            rec[f"to_plurality_{c}"] = _p([int(bool(plur) and bool(x["ans"])
                                                and answers_equivalent(plur, x["ans"]))
                                           for x in xs]) if plur else 0.0
            rec[f"trunc_{c}"] = _p([x["trunc"] for x in xs])
        # ★no_new_boxed: wait 조건에서 이어쓰기에 새 \boxed 가 없으면 답은 원래 오답 그대로다
        #   (규칙: 그 경우 최종 답 = prefix 의 원래 오답으로 채점한다 — grade_math(x.text, gold)
        #   가 이미 그 규칙을 만족한다: 이어쓴 부분에 새 박스가 없으면 last_boxed 는 원문의
        #   마지막 박스, 즉 원래 오답을 그대로 본다).
        if "wait" in conds and got.get("wait"):
            rec["no_new_boxed_wait"] = _p([1 if not x["ans"] else 0 for x in got["wait"]])
        wrecs.append(rec)

    # ── per-rollout(정답, 거짓-경보) ────────────────────────────────────────
    crecs = []
    for pop, rows, cs in (("correct", correct, cconds),):
        for si, s in enumerate(rows):
            got = {c: agg.get((pop, si, c)) for c in cs}
            if not any(got.values()):
                continue
            orig_ans = last_boxed(s["text"])
            rec = {"roll_id": s["roll_id"], "group_id": s["group_id"], **gen_meta(s)}
            for c in cs:
                xs = got[c]
                if not xs:       # ★그 행에 그 조건이 없을 수 있다(fact_notx 의 X 부재)
                    continue
                rec[f"flip_{c}"] = _p([1 - x["r_corr"] for x in xs])  # 맞았는데 틀려진 비율
                rec[f"changed_{c}"] = _p([int(bool(x["ans"]) and not answers_equivalent(
                    orig_ans, x["ans"])) for x in xs])
                rec[f"trunc_{c}"] = _p([x["trunc"] for x in xs])
            crecs.append(rec)

    wsumm = summarize_wrong(wrecs, k=a.k, seed=a.seed, n_boot=a.n_boot, conds=conds)
    # ★거짓-경보 기준선(flip_baseline)은 **주 정답 모집단**만 써야 한다 — 저합의 대조 행이
    #   섞이면 F1/S4 의 flip 절 기준선이 조용히 달라진다.
    csumm = summarize_correct(crecs, k=a.k, seed=a.seed, n_boot=a.n_boot, conds=cconds)
    wsumm["agree_strata"] = stratified_summary(wrecs, conds=conds, seed=a.seed,
                                               n_boot=a.n_boot)
    if clow_conds:
        csumm["lowagree_states"] = state_correct_summary(crecs, conds=clow_conds,
                                                         seed=a.seed, n_boot=a.n_boot)
    if need_label:
        wsumm["label_leak"] = leak
    if hab_arms:
        wsumm["habit_hist"] = habit_hist
        wsumm["habit_leak"] = habit_leak
        wsumm["habit_minority"] = (sum(habit_minority) / len(habit_minority)
                                   if habit_minority else _NAN)
        ref = next((c for c in ("fact_reread", FACT_ALIAS, "blind") if c in conds), "")
        wsumm["note_variance"] = {}
        for arm in hab_arms:
            rows_v = [([x["r_corr"] for x in agg[("wrong", si, arm)]],
                       [x["r_corr"] for x in agg[("wrong", si, ref)]])
                      for si in range(len(wrong))
                      if agg.get(("wrong", si, arm)) and agg.get(("wrong", si, ref))]
            st = note_variance_stat(rows_v, seed=a.seed + 300)
            st["ref"] = ref
            wsumm["note_variance"][arm] = st
    if "external" in conds and "blind" in conds:
        wsumm["pass_activation"] = int(activation_pass(wsumm, csumm))
        wsumm["pass_self_trigger"] = int(self_trigger_pass(wsumm, csumm))
    if wsumm.get("tested_arms"):
        ok, winners, flags = reset_content_pass(wsumm, csumm)
        wsumm["pass_reset_content"] = int(ok)
        wsumm["reset_content_winners"] = winners
        wsumm["reset_content_flags"] = flags
    if wsumm.get("tested_arms_s4"):
        ok4, win4, fl4 = s4_content_pass(wsumm, csumm)
        wsumm["pass_s4_lite"] = int(ok4)
        wsumm["s4_lite_winners"] = win4
        wsumm["s4_lite_flags"] = fl4
    meta = {"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
            "seed": a.seed, "max_tokens": a.max_tokens, "conds": conds, "conds_correct": cconds,
            "population": a.population, "pop_kinds": list(kinds),
            "correct_states": cstates, "conds_correct_lowagree": clow_conds,
            "n_lowagree_candidates": len(clow),
            "n_by_pop_kind": {k: sum(1 for s in wrong if s["pop_kind"] == k) for k in kinds},
            "n_wrong_candidates": len(wrong),
            "n_correct_candidates": len(correct), "n_dropped_long": n_drop,
            "n_skipped_cond": n_skip, "n_generations": len(gens)}
    summ = {"wrong": wsumm, "correct": csumm, "meta": meta}

    with (out / "per_row.jsonl").open("w") as fh:
        for r in wrecs:
            fh.write(json.dumps({**r, "population": "wrong"}, ensure_ascii=False) + "\n")
        for r in crecs:
            fh.write(json.dumps({**r, "population": "correct"}, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(wsumm, csumm)
    (out / "gate_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
