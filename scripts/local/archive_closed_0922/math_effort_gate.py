#!/usr/bin/env python
r"""math_effort_gate — «노력을 외생화»해서 모니터링 실패와 통제 실패를 가른다.

★왜(`docs/analysis/CONTAM_activation_0915.md`): 150개 Level-5 오답 롤아웃을 자기 풀이 뒤 "Wait,
  let me double-check this." 로 이어 쓰면 구제 .233 뿐, 눈감고 새로 푸는 것(.583)에 크게
  진다(`math_activation_gate`). 재분석이 원인 후보 둘을 **기각**했다 — 주의 예산이 아니다(절단
  1.1% 대 13.2%, 이어쓰기 중위 733자 대 11,058자), 베낌도 아니다(길이를 맞추면 접두와 **덜**
  겹친다). 남은 것은 하나: **일찍 멈추고 원래 오답을 다시 내놓는다**(재생산 .796 대 눈감은 null
  .206). 노력으로 나누면 뒤집힌다(3k~6k자 .65, 6k~10k자 .85 — 같은 길이의 눈감은 풀이와 같거나
  그 위; 답을 바꾼 14.9% 는 90.5% 가 맞다) — ⚠그 조건화는 같은 생성물의 **결과**로 나눈 것이라
  선택 효과지 인과가 아니다. **이 게이트가 노력을 외생화한다**: 강제로 계속 쓰게 해 같은 토큰의
  눈감은 풀이 이상으로 오르면 실패는 모니터링(자기 오류를 못 봄)이 아니라 «그만 쓰기로 한»
  **통제**이고, 학습할 것은 «계속 일하기로 하는 결정» — 정책 자신의 롤아웃으로 라벨이 붙는다.

★설계(같은 오답 롤아웃·K=8·temp 1.0·`--max_tokens 8192`; 모집단은 `math_activation_gate` 와 같은
  MIXED 오답 + 강제 팔에만 짝지은 정답 행 거짓-경보 대조): (1) `blind` 문제만 —
  `render_generation_prompt` 와 **바이트 동일**한 앵커 · (2) `wait_free` 기존 이어쓰기(자기 cue),
  강제 없음 — .233 재현 · (3) `wait_forced_N` 같은 이어쓰기인데 **N 토큰 전에는 멈출 수 없다** ·
  (4) `blind_matched_N` 눈감은 풀이를 (3) 의 **평균 총 생성 토큰**으로 자른다 ·
  (5) `pad_forced_N` 같은 강제를 점검 cue 가 아니라 **내용 없는 cue** 뒤에 건다.

★(4)·왜 **2패스**인가: (4) 없이 «강제가 도왔다» 는 «토큰을 더 썼다» 와 구별되지 않는다. 그런데
  (4) 의 예산은 **사전에 알 수 없다** — N 바닥을 깔아도 총 생성량은 N 이 아니다. 총량은 (3) 을
  **돌려서 재는 수**라서 1패스(1·2·3·5)를 끝내고 평균을 읽은 **다음** 2패스(4)를 발사한다.
  ⚠눈감은 풀이는 ~11.4k자를 쓰므로 (4) 는 절단으로 죽을 수 있다 — trunc_rate 를 같이 읽고,
  (4) 대비 승리는 «같은 토큰 예산에서» 로만 읽는다((1) 대비 승리가 더 강한 주장이다).
★(5): 강제는 «더 쓰게 만드는 것» 이기도 하다 — 같은 강제를 내용 없는 cue 뒤에 걸면 갈린다.
  (도너 이어쓰기는 채점되는 문제와 문맥이 어긋나 비일관이므로 쓰지 않는다.)

★E1(0915)·PASS 앵커는 blind(1)지 blind_matched(4)가 아니다: (4) 없이는 «강제가 도왔다» 를
  «토큰을 더 썼다» 와 못 가른다고 위에서 적었지만, (4)를 **판정 자체**에 쓰면 새 함정에
  빠진다 — G4 게이트를 무효화한 것과 같은 함정이다. `blind_matched_N`은 눈감은 신선한 풀이를
  강제 팔의 **평균 총생성 토큰**으로 자른 것이라, 신선한 풀이가 그 예산에 잘리면 근사 0점을
  받고 (forced − blind_matched) ≥ +.03이 **자동으로** 통과해 버려 아무것도 증명하지 않는다.
  의미 있는 질문은 «오답 롤아웃이 나왔을 때, 강제로 이어 쓰는 것이 처음부터 다시 푸는 것보다
  나은가» 이고, 다시 풀기의 한계 비용은 **전체 예산** 신선한 풀이이므로 blind(전체 예산)가
  정직한 비교다. blind_matched는 강제 팔이 blind**보다 더** 쓸 때만 의미가 생기므로 REPORTED
  진단으로만 남기고(짝지은 Δ 표), blind_matched 자신의 절단률이 .20을 넘으면 그 대조 자체가
  해석 불가능하다는 경고를 낸다([WEAK-ANCHOR]).

★E2(0915)·퇴화(degenerate): 온도 1.0에서 EOS를 −inf로 막는 강제는 «멈추고 싶은» 모델을
  루프에 가둘 위험이 있다(알려진 퇴화 위험). 생성마다 두 저비용 지표 중 하나라도 걸리면
  `degenerate=1`: ① 20자 단위로 겹치지 않게 자른 조각 중 두 번 이상 나온 조각이 덮는 글자
  비율이 .5를 넘는다(LCS 급 정확한 반복탐지는 O(n²)라 이 O(n) 근사를 쓴다) ② 40자 단위
  조각이 바로 뒤이어 3번 이상 그대로 반복된다(하드 EOS 억제가 만드는 전형적 «같은 문장
  반복» 루프). 조건별 `degenerate_rate`를 표에 찍고 강제 팔 자신의 값이 .20을 넘으면
  통과 규칙 (d)에서 떨어뜨리며, .20을 넘는 모든 조건에 [DEGENERATE]를 찍는다.

★E3(0915)·강제가 산 것: `frac_stopped_at_floor` = 생성 토큰 수가 강제 바닥 N의 ±5% 안인
  비율 — 높으면 모델이 허락되는 순간 멈췄다는 뜻으로, «강제의 문자»만 지키고 «강제의 취지»
  는 안 지킨 것이다. null을 읽을 때 가장 먼저 볼 숫자다. `answer_changed_after_floor` =
  강제 생성 중 최종 답이 원래 오답과 달라진 비율(강제가 실제로 답을 바꿨는지).

★강제(vllm 0.29.0 실측 확인): `sampling_params.py:280 min_tokens: int = 0`("… before EOS or
  `stop_token_ids` can be generated")을 `v1/sample/logits_processor/builtin.py:165
  MinTokensLogitsProcessor` 가 집행한다 — 미달인 동안 `params.all_stop_token_ids`(EOS 포함,
  `update_from_generation_config` 가 넣는다)를 −inf 로 막고 도달하면 해제하며, 정지 **문자열**도
  `v1/engine/detokenizer.py:131` 이 `num_output_tokens() > min_tokens` 일 때만 본다(제약은
  `0 ≤ min_tokens ≤ max_tokens`, diffusion 만 미지원). 그래서 `min_tokens=N` + `ignore_eos=False`
  — N 전에는 못 멈추고 N **뒤에는** 멈춘다(바닥이지 천장이 아니다). 없는 환경이면 s1 식 대체
  경로(`--force_mode rounds`): N 미달이면 꼬리 end-of-turn 을 떼고 "\nWait," 를 붙여 **남은
  예산**으로 재표본, N 또는 3라운드까지(행별 라운드 수는 `mean_rounds`/`gens.jsonl`). 어느
  경로든 최종 텍스트는 접두 + 생성 조각 전부이고 **채점은 생성 부분만** 한다.
★새 박스 없음: 이어쓰기에 새 `\boxed` 가 없으면 최종 답 = 접두의 원 답이라 `effective_r_corr`
  가 원 롤아웃의 정오를 물려받는다 — 오답 행 구제 0(기존과 동일), **정답 행 뒤집힘 0**
  (`math_activation_gate` 는 여기서 과대 계상한다; flip_wrong_rate 가 통과 규칙이라 고친다).

★통과 규칙(0915 수정): **EFFORT PASS** ⟺ 어떤 `wait_forced_N` 이 다음을 **모두** 만족:
  (a) (forced − blind) CI 0 제외 ∧ 평균 ≥ +0.03 — blind 는 **전체 예산** 눈감은 풀이(위 E1) ·
  (b) 정답 행 `flip_wrong_rate` ≤ 0.10 · (c) 그 강제 팔 **자신**의 `trunc_rate` ≤ 0.20 ·
  (d) 그 강제 팔 **자신**의 `degenerate_rate`(E2) ≤ 0.20. nan 은 전부 FAIL. 후보가 여럿이면
  구제율(`rescue`)이 가장 높은 팔을 고른다 — 그래야 `pass_arm` 과 `CONTROL-IS-EFFORT` 가
  항상 같은 팔을 가리킨다. **CONTROL-IS-EFFORT** = (구제 최고 `wait_forced_N`) − `pad_forced_N`
  + CI 를 **따로** 찍는다 — ≈0 이면 강제는 «점검하게 만든 것» 이 아니라 «더 쓰게 만든 것» 이다
  (판정에는 안 넣는다). `(forced − blind_matched_N)` 은 REPORTED 진단으로만 남긴다(위 E1).

★O1(0917)·다리(bridge) 사전시험 — «모델은 자기 시도-1 뒤에 **다리 문장**이 오면 문맥 안에서
  회복하는가»: (6) `bridge_free` 접두 = 시도-1 전문, 이어 붙이는 것은 `build_self_traces.
  bridge_text(X)`(X = 그 행 자신의 시도-1 최종 답) — SFT 타깃과 **바이트 동일**하다(import 해서
  쓴다) · (7) `bridge_pad_free` 같은 자리에 길이만 맞춘 위약 `PLACEBO_BRIDGE`(부정도 X 도 없다).
  둘 다 `wait_free` 와 **같은 렌더 경로**(열린 assistant 턴 이어 쓰기)이고 강제가 없다.
  ⚠`wait_free` 와 달리 **새 박스가 없으면 물려받지 않고 r_corr 0**이다 — 다리는 앞 답을 명시적
  으로 버리므로 남는 답이 없다(`no_new_box_rate`/`n_no_new_box` 로 따로 센다). `reemit_x_rate`
  는 새 답이 X 와 동치인 비율. **BRIDGE PASS** ⟺ rescue(bridge_free) ≥ 0.6×rescue(blind) ∧
  Δ(bridge_free − bridge_pad_free) CI 하한 > 0. 조건은 `--conds` 로만 켠다.

사용(한 번 호출, 안에서 2패스 — 이 명령이 s1 발사판이다):
  python scripts/local/math_effort_gate.py --k 8 --max_sites 150 --seed 11 --variant math_opt \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 --gpu_util 0.45 \
      --max_tokens 8192 --force_tokens 1024,3072 --out_dir \
      /hdd_data/seungpil/scratch/eval/effort_gate_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_activation_gate import (  # noqa: E402  (조건 조립·모집단·표 서식을 한 곳에 둔다)
    WAIT_CUE, blind_prompt, select_correct_rollouts, wait_prompt,
)
from math_activation_gate import _f as _fmt  # noqa: E402
from build_self_traces import BRIDGE_TMPL, bridge_text  # noqa: E402,F401  (★O1 SFT 타깃과 동일)
from math_cited_site_gate import (  # noqa: E402  (통계·선별 정의를 한 곳에 둔다)
    bootstrap_ci, select_wrong_rollouts, sign_test_p,
)
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, grade_math, last_boxed, selftest_math_verify,
)

_NAN = float("nan")
PASS_DELTA = 0.03               # (forced − blind) 의 하한(평균) — ★E1: blind 는 전체 예산
MAX_FLIP_WRONG = 0.10           # 정답 행 flip_wrong_rate 상한
MAX_TRUNC = 0.20                # 통과 후보 팔 자신의 절단률 상한
MAX_DEGEN = 0.20                # ★E2: 통과 후보 팔 자신의 degenerate_rate 상한
MAX_ROUNDS = 3                  # rounds 대체 경로의 최대 라운드
FORCE_CUE = "\nWait,"           # ★s1 식 강제 이어쓰기 리터럴(대체 경로 전용)
EOT_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")
FORCED = ("wait_forced_", "pad_forced_")            # 강제가 걸리는 팔
_CONT = ("wait_free",) + FORCED                     # 이어쓰기 계열(새 박스 없음 규칙 적용)
PAD_CUE = "\n\nLet me look at this once more."      # ★(5) 점검 의도 없는 cue

# ★O1(0917)·«다리(bridge)가 문맥 안에서 구제하는가» — S6 축의 결정적 사전시험.
#   `bridge_free` = 시도-1 전문 + `build_self_traces.bridge_text(X)`(X = 그 행 자신의 시도-1
#   최종 답). 문자열은 SFT 타깃과 **바이트 동일**하다(그래서 import 한다 — 여기서 다시 쓰면
#   재현하려는 그 문장이 아니게 된다). `bridge_pad_free` = 같은 자리에 길이만 맞춘 위약 —
#   부정도 X 도 없다(= «다시 푸는 말» 만 남긴다). 둘 다 강제 없음(`--max_tokens` 전액).
#   (길이는 Qwen3 토크나이저 기준 44 토큰 대 다리 49 토큰 = 0.90, 글자 153 대 145 = 1.06 —
#    둘 다 ±15% 안이다. `\boxed{X}` 가 글자당 토큰을 많이 먹어 두 척도를 함께 맞추려면
#    쉼표·하이픈이 많은 이 문장 형태가 필요하다.)
PLACEBO_BRIDGE = ("\n\n<|meta|>Let me set this out, again, carefully, from the very beginning, "
                  "step-by-step, line-by-line, part-by-part, and give the final "
                  "answer.<|/meta|>\n\n")
BRIDGE_CONDS = ("bridge_free", "bridge_pad_free")
BRIDGE_RESCUE_FRAC = 0.6        # ★O1 게이트: rescue(bridge_free) ≥ 이 배수 × rescue(blind)
FLOOR_TOL = 0.05                # ★E3: frac_stopped_at_floor 의 바닥 대비 허용 오차(±5%)
SHINGLE_SIZE = 20                # ★E2 지표① 조각 길이
DEGEN_COVERAGE = 0.5             # ★E2 지표① 반복 조각이 덮는 글자 비율 상한
RUN_LEN = 40                     # ★E2 지표② 즉시-반복 조각 길이
RUN_COUNT = 3                    # ★E2 지표② 연속 반복 최소 횟수
_CIS = (("rescue", "p_"), ("flip_wrong_rate", "flip_"), ("reproduction_rate", "repro_"))
_MEANS = (("changed_rate", "changed_"), ("trunc_rate", "trunc_"), ("degenerate_rate", "degen_"),
          ("mean_gen_tokens", "ntok_"), ("mean_rounds", "rounds_"),
          ("forced_short_rate", "short_"), ("frac_stopped_at_floor", "stopfloor_"),
          ("answer_changed_after_floor", "aftfloor_"),
          ("reemit_x_rate", "reemit_"), ("no_new_box_rate", "nobox_"),      # ★O1
          ("n_no_new_box", "nnobox_"))

def is_continuation(cond: str) -> bool:
    """접두를 이어 쓴 조건인가 — 새 박스 없음 규칙의 적용 여부를 가른다.
    ★O1: `bridge_*` 는 접두를 이어 쓰지만 여기서 **False** 다. `wait_free` 는 «잠깐, 다시
    보자» 라 새 박스가 없으면 앞의 답이 그대로 최종 답이지만, 다리 문장은 앞의 답을 **명시적
    으로 버린다**("Discarding the work above … the answer is not X") — 그 뒤에 새 답을 안 내면
    남는 답이 없다. 그래서 물려받지 않고 r_corr 0 으로 세고(`nobox_*`/`nnobox_*` 로 따로
    계측한다), reproduction 도 «새 박스가 실제로 X 와 같을 때» 만 1 이다."""
    return cond.startswith(_CONT)

def force_floor(cond: str) -> int:
    """강제 팔이면 그 바닥 N, 아니면 0."""
    return int(cond.rsplit("_", 1)[1]) if cond.startswith(FORCED) else 0

def cond_names(force_tokens: Sequence[int]) -> tuple[list[str], list[str], list[str]]:
    """(1패스 조건, 2패스 조건, 정답 행 조건). 2패스 예산은 1패스 측정에 의존한다."""
    f = [f"wait_forced_{n}" for n in force_tokens]
    p = [f"pad_forced_{n}" for n in force_tokens]
    return ["blind", "wait_free"] + f + p, [f"blind_matched_{n}" for n in force_tokens], f + p


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────────
def pad_prompt(tok, variant: str, problem: str, text: str) -> str:
    """(5) `wait_prompt` 와 **cue 문자열만** 다르다(같은 규약: add_generation_prompt=True 로
    끝난 문자열 뒤에 이어 붙이면 continue_final_message=True 와 바이트 동일)."""
    return render_generation_prompt(tok, variant, problem) + (text or "") + PAD_CUE

def bridge_prompt(tok, variant: str, problem: str, text: str, *, placebo: bool = False) -> str:
    """★O1 — `wait_prompt` 와 **같은 렌더 경로**(열린 assistant 턴을 이어 쓴다. 새 user 턴이
    아니다). 붙이는 문자열만 다르다: `bridge_text(X)`(X = 이 행 시도-1 의 최종 `\\boxed` 답)
    또는 길이만 맞춘 위약 `PLACEBO_BRIDGE`."""
    app = PLACEBO_BRIDGE if placebo else bridge_text(last_boxed(text or "") or "")
    return render_generation_prompt(tok, variant, problem) + (text or "") + app

def build_prompt(tok, variant: str, cond: str, problem: str, text: str) -> str:
    """조건 → 프롬프트. blind/blind_matched_* 는 앵커와 바이트 동일, wait_forced_* 는
    `wait_free` 와 **바이트 동일**(다른 것은 표본 파라미터뿐), pad_forced_* 는 cue 만 다르다."""
    if cond == "blind" or cond.startswith("blind_matched"):
        return blind_prompt(tok, variant, problem)
    if cond == "wait_free" or cond.startswith("wait_forced"):
        return wait_prompt(tok, variant, problem, text)
    if cond.startswith("pad_forced"):
        return pad_prompt(tok, variant, problem, text)
    if cond in BRIDGE_CONDS:
        return bridge_prompt(tok, variant, problem, text, placebo=(cond == "bridge_pad_free"))
    raise ValueError(f"[EFF] 모르는 조건: {cond}")


# ── 퇴화(degenerate) 판정 ────────────────────────────────────────────────────────
def repeated_shingle_coverage(text: str, size: int = SHINGLE_SIZE) -> float:
    """★E2 지표①: `text`(GENERATED 부분만)를 `size`자 단위로 **겹치지 않게** 자른 조각
    (shingle) 중 두 번 이상 나온 조각이 덮는 글자 수의 비율. 진짜 반복탐지(LCS 등)는
    O(n²)라 대신 쓰는 저비용 O(n) 근사 — 경계에 안 맞는 반복은 놓칠 수 있으나 루프 생성은
    보통 조각 경계와 무관하게 다수 잡힌다."""
    n = len(text)
    if n < size:
        return 0.0
    chunks = [text[i:i + size] for i in range(0, n - size + 1, size)]
    counts = Counter(chunks)
    covered = sum(len(c) for c in chunks if counts[c] > 1)
    return covered / n

def has_consecutive_repeat_run(text: str, size: int = RUN_LEN, count: int = RUN_COUNT) -> bool:
    """★E2 지표②: `size`자 단위 조각이 바로 뒤이어 `count`번 이상 그대로 반복되는가 —
    하드 EOS 억제가 만드는 전형적 «같은 문장을 그대로 반복» 루프를 잡는다."""
    n = len(text)
    if n < size * count:
        return False
    chunks = [text[i:i + size] for i in range(0, n - size + 1, size)]
    run = 1
    for a, b in pairwise(chunks):
        run = run + 1 if a == b else 1
        if run >= count:
            return True
    return False

def is_degenerate(text: str) -> bool:
    """★E2: 온도 1.0에서 EOS 를 −inf 로 막는 강제는 «멈추고 싶은» 모델을 루프에 가둘 수
    있다(알려진 퇴화 위험) — 두 저비용 지표 중 하나라도 걸리면 퇴화로 본다."""
    text = text or ""
    return (repeated_shingle_coverage(text) > DEGEN_COVERAGE
            or has_consecutive_repeat_run(text))


# ── 강제 루프 ───────────────────────────────────────────────────────────────────
def strip_end_of_turn(text: str) -> str:
    """꼬리의 end-of-turn 표식과 공백을 뗀다 — 이어 붙이기 전 필수(대체 경로)."""
    s = (text or "").rstrip()
    while (m := next((x for x in EOT_MARKERS if s.endswith(x)), None)):
        s = s[: -len(m)].rstrip()
    return s

def _rec(g: dict, rounds: int = 1) -> dict:
    """생성 한 개 → 기록 한 줄(텍스트·모델 생성 토큰 수·절단 여부·쓴 라운드 수)."""
    return {"text": g["text"], "n_gen_tokens": int(g["n_tokens"]),
            "truncated": int(g.get("finish_reason") == "length"), "rounds": rounds}

def generate(sample: Callable, prompts: Sequence[str], *, k: int, max_tokens,
             force_tokens: int = 0, use_min_tokens: bool = True,
             max_rounds: int = MAX_ROUNDS) -> list[list[dict]]:
    r"""`prompts` 각각 K 개씩. `force_tokens>0` 이면 **N 토큰 전에는 멈출 수 없다**.

    `sample(prompts, n=, max_tokens=, min_tokens=)` → list[list[{text, finish_reason,
    n_tokens}]] (max_tokens 는 int 또는 프롬프트별 list). `use_min_tokens` 면 한 번에
    끝난다(rounds=1). 아니면 s1 식: N 미달·미절단 시퀀스마다 end-of-turn 을 떼고 FORCE_CUE 를
    붙여 **남은 예산**으로 재표본, N 또는 max_rounds 까지. 반환 텍스트는 생성 조각 + 주입한
    cue 전부, `n_gen_tokens` 는 **모델이 만든 토큰만** 센다(주입 cue 는 세지 않는다). 반환하는
    각 기록에는 최종(전 라운드 합) 텍스트로 판정한 `degenerate`(★E2)도 붙는다."""
    cap = max(max_tokens) if isinstance(max_tokens, (list, tuple)) else int(max_tokens)
    n_force = max(0, min(int(force_tokens), cap))
    floor = n_force if (n_force and use_min_tokens) else 0
    state = [[_rec(g) for g in row]
             for row in sample(prompts, n=k, max_tokens=max_tokens, min_tokens=floor)]
    for _ in range(max(0, max_rounds - 1) if (n_force and not floor) else 0):
        todo, qs, budgets = [], [], []
        for pi, row in enumerate(state):
            for ki, st in enumerate(row):
                if st["truncated"] or st["n_gen_tokens"] >= n_force or cap <= st["n_gen_tokens"]:
                    continue
                st["text"] = strip_end_of_turn(st["text"]) + FORCE_CUE
                todo.append((pi, ki))
                qs.append(prompts[pi] + st["text"])
                budgets.append(cap - st["n_gen_tokens"])
        if not todo:
            break
        for (pi, ki), row in zip(todo, sample(qs, n=1, max_tokens=budgets, min_tokens=0)):
            g, st = row[0], state[pi][ki]
            st["text"] += g["text"]
            st["n_gen_tokens"] += int(g["n_tokens"])
            st["truncated"] = int(g.get("finish_reason") == "length")
            st["rounds"] += 1
    for row in state:                                  # ★E2: 최종 텍스트(전 라운드 합) 기준
        for st in row:
            st["degenerate"] = int(is_degenerate(st["text"]))
    return state


# ── 채점 ────────────────────────────────────────────────────────────────────────
def effective_r_corr(gen_text: str, gold: str, *, orig_r_corr: int, cont: bool) -> int:
    """생성 부분만 채점한다. ★단 이어쓰기에 새 `\\boxed` 가 없으면 최종 답 = 접두의 원 답이므로
    원 롤아웃의 정오를 물려받는다(모듈 docstring «새 박스 없음» 참조)."""
    return (int(orig_r_corr) if cont and not last_boxed(gen_text or "")
            else grade_math(gen_text or "", gold))

def answer_stats(gen_text: str, orig_ans: str, *, cont: bool) -> tuple[int, int]:
    """(reproduction, changed). reproduction = 최종 답이 **원래 답과 수학적으로 동치**인가
    (핵심 기제 지표 — `answers_equivalent`; 새 박스 없는 이어쓰기는 원래 답 그대로라 1).
    changed 는 `math_activation_gate` 와 **같은** 정의(새 답이 있고 원래 답과 동치가 아니다)."""
    new = last_boxed(gen_text or "")
    if not new:
        return (1 if cont else 0), 0
    eq = bool(answers_equivalent(orig_ans, new))
    return int(eq), int(not eq)


# ── 요약 ────────────────────────────────────────────────────────────────────────
def _num(v, hi: float = math.inf) -> bool:
    """유한한 수이고 hi 이하인가 — nan/None/누락은 전부 False(=FAIL)."""
    return isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) <= hi

def _p(vals: Sequence[float]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN

def matched_budget(recs: Sequence[dict], cond: str, *, max_tokens: int) -> int:
    """★2패스 예산 — `cond` 팔의 **평균 총 생성 토큰**(롤아웃 평균), [1, max_tokens] 로 자른다.
    빈 입력이면 예산을 깎지 않는다(= max_tokens)."""
    v = [float(r[f"ntok_{cond}"]) for r in recs if _num(r.get(f"ntok_{cond}"))]
    return int(max_tokens) if not v else max(1, min(int(max_tokens), round(sum(v) / len(v))))

def contrasts(force_tokens: Sequence[int]) -> list[tuple[str, str]]:
    """짝지은 대조 — (a, b) 는 a − b: forced 대 wait_free/blind/blind_matched/pad_forced."""
    return [(f"wait_forced_{n}", b) for n in force_tokens
            for b in ("wait_free", "blind", f"blind_matched_{n}", f"pad_forced_{n}")]

BRIDGE_CONTRASTS = [("bridge_free", "blind"), ("bridge_free", "bridge_pad_free"),
                    ("bridge_free", "wait_free")]      # ★O1 — 조건이 없으면 summarize 가 건넌다

def summarize(recs: Sequence[dict], conds: Sequence[str], *, force_tokens: Sequence[int] = (),
              k: int = 0, seed: int = 0, n_boot: int = 2000) -> dict:
    """per-row 기록 → 요약. `recs` 는 모든 조건이 다 성립한 행만. `force_tokens` 를 주면 짝지은
    Δ·부호검정까지 찍는다(오답 행에서만 쓴다 — 정답 행은 flip_wrong_rate 만 본다)."""
    out: dict = {"n_rollouts": len(recs), "k": k, "conds": list(conds), "paired": {}, "sign_p": {}}
    for j, (key, pre) in enumerate(_CIS):
        out[key] = {c: bootstrap_ci([r[f"{pre}{c}"] for r in recs], seed=seed + 50 * j + i,
                                    n_boot=n_boot) for i, c in enumerate(conds)}
    for key, pre in _MEANS:
        out[key] = {c: _p([r[f"{pre}{c}"] for r in recs if f"{pre}{c}" in r]) for c in conds}
    out["n_no_new_box"] = {c: sum(int(r.get(f"nnobox_{c}", 0)) for r in recs) for c in conds}
    for j, (a, b) in enumerate(contrasts(force_tokens) + BRIDGE_CONTRASTS):
        if not all(f"p_{x}" in (recs[0] if recs else {}) for x in (a, b)):
            continue
        d = [r[f"p_{a}"] - r[f"p_{b}"] for r in recs]
        out["paired"][f"{a}_minus_{b}"] = bootstrap_ci(d, seed=seed + 200 + j, n_boot=n_boot)
        out["sign_p"][f"{a}_minus_{b}"] = sign_test_p(d)
    return out


# ── 통과 규칙 ───────────────────────────────────────────────────────────────────
def effort_pass(wrong: dict, correct: dict,
                force_tokens: Sequence[int]) -> tuple[bool, str | None]:
    """어떤 `wait_forced_N` 이 네 조건을 **다** 만족하면 PASS 후보다:
    (a) (forced − blind) CI 0 제외 ∧ 평균 ≥ +0.03 — blind 는 **전체 예산** 눈감은 풀이
    (★E1; blind_matched_N 은 REPORTED 진단일 뿐 여기서 안 쓴다) (b) 정답 행 flip_wrong_rate
    ≤ 0.10 (c) 그 팔 **자신**의 trunc_rate ≤ 0.20 (d) 그 팔 **자신**의 degenerate_rate
    (★E2) ≤ 0.20. nan 은 전부 FAIL. 후보가 여럿이면 **구제율(rescue)이 가장 높은** 팔을
    돌려준다 — 그래야 이 함수와 `control_is_effort` 가 항상 같은 팔을 가리킨다."""
    resc = wrong.get("rescue") or {}
    passing: list[tuple[float, str]] = []
    for n in force_tokens:
        c = f"wait_forced_{n}"
        d = (wrong.get("paired") or {}).get(f"{c}_minus_blind") or {}
        if (all(_num(d.get(x)) for x in ("lo", "hi", "mean"))
                and (d["lo"] > 0 or d["hi"] < 0) and d["mean"] >= PASS_DELTA
                and _num(((correct.get("flip_wrong_rate") or {}).get(c) or {}).get("mean"),
                         MAX_FLIP_WRONG)
                and _num((wrong.get("trunc_rate") or {}).get(c), MAX_TRUNC)
                and _num((wrong.get("degenerate_rate") or {}).get(c), MAX_DEGEN)):
            rescue_mean = (resc.get(c) or {}).get("mean")
            passing.append((rescue_mean if _num(rescue_mean) else -math.inf, c))
    if not passing:
        return False, None
    passing.sort(key=lambda t: t[0], reverse=True)
    return True, passing[0][1]

def bridge_pass(wrong: dict) -> tuple[bool, dict]:
    """★O1 **BRIDGE PASS** ⟺ (i) rescue(bridge_free) ≥ 0.6 × rescue(blind) — 같은 단위에서
    «문맥 안 회복» 이 «눈감고 새로 풀기» 의 6할은 되는가 — ∧ (ii) Δ(bridge_free −
    bridge_pad_free) CI 하한 > 0 — 회복이 «다리 문장» 때문이지 «다시 풀라는 말» 때문이
    아니다. nan 은 전부 FAIL."""
    resc = wrong.get("rescue") or {}
    rb = (resc.get("bridge_free") or {}).get("mean")
    r0 = (resc.get("blind") or {}).get("mean")
    d = (wrong.get("paired") or {}).get("bridge_free_minus_bridge_pad_free") or {}
    info = {"rescue_bridge_free": rb, "rescue_blind": r0,
            "vs_blind_ok": bool(_num(rb) and _num(r0) and rb >= BRIDGE_RESCUE_FRAC * r0),
            "delta_vs_placebo": d}
    info["pass_bridge"] = int(info["vs_blind_ok"] and _num(d.get("lo")) and d["lo"] > 0)
    return bool(info["pass_bridge"]), info

def weak_anchor_warnings(wrong: dict, force_tokens: Sequence[int]) -> list[str]:
    """★E1: `blind_matched_N` 은 눈감은 신선한 풀이를 강제 팔의 **평균 총생성 토큰**으로
    자른 것이라, 그 절단률이 높으면 (forced − blind_matched) 대조 자체가 해석 불가능해진다
    (신선한 풀이가 잘려 근사 0점을 받으면 승리가 «강제가 도왔다» 가 아니라 «앵커를 잘랐다»
    가 된다). 판정에는 안 쓰고 경고만 낸다 — `main()` 이 그대로 print 한다."""
    out = []
    for n in force_tokens:
        c = f"blind_matched_{n}"
        trunc = (wrong.get("trunc_rate") or {}).get(c)
        if isinstance(trunc, (int, float)) and math.isfinite(trunc) and trunc > MAX_TRUNC:
            out.append(f"[WEAK-ANCHOR] {c} trunc={trunc:.3f} — matched-anchor 대조가 해석 "
                       "불가능하다(눈감은 풀이가 강제 팔 평균 예산으로 잘려 나갔다)")
    return out

def degenerate_warnings(summ: dict, *, max_degen: float = MAX_DEGEN) -> list[str]:
    """★E2: `summ`(오답/정답 요약 어느 쪽이든)의 조건 중 `degenerate_rate` 가 상한을 넘는
    것마다 경고 한 줄. `main()` 이 그대로 print 한다."""
    out = []
    for c in summ.get("conds", []):
        rate = (summ.get("degenerate_rate") or {}).get(c)
        if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > max_degen:
            out.append(f"[DEGENERATE] {c} {rate:.3f}")
    return out

def control_is_effort(wrong: dict, force_tokens: Sequence[int]) -> dict:
    """구제율이 가장 높은 `wait_forced_N` 과 **같은 N** 의 `pad_forced_N` 의 짝지은 차 + CI.
    ≈0 이면 강제는 «점검» 이 아니라 «더 쓰기» 로 듣는 것이다."""
    empty = {"mean": _NAN, "lo": _NAN, "hi": _NAN, "n": 0}
    resc = wrong.get("rescue") or {}
    best = max(((float((resc.get(f"wait_forced_{n}") or {}).get("mean")), n) for n in force_tokens
                if _num((resc.get(f"wait_forced_{n}") or {}).get("mean"))), default=None)
    if best is None:
        return {"best_arm": None, "pad_arm": None, "delta": empty, "sign_p": _NAN}
    key = f"wait_forced_{best[1]}_minus_pad_forced_{best[1]}"
    return {"best_arm": f"wait_forced_{best[1]}", "pad_arm": f"pad_forced_{best[1]}",
            "delta": (wrong.get("paired") or {}).get(key) or empty,
            "sign_p": (wrong.get("sign_p") or {}).get(key, _NAN)}

def _table(summ: dict, keys: Sequence[str]) -> list[str]:
    rows = [f"| {c} | " + " | ".join(_fmt((summ.get(k) or {}).get(c)) for k in keys) + " |"
            for c in summ.get("conds", [])]
    return ["", f"| cond | {' | '.join(keys)} |", "|---" * (len(keys) + 1) + "|"] + rows

def to_markdown(wrong: dict, correct: dict, force_tokens: Sequence[int], meta: dict) -> str:
    ok, arm = effort_pass(wrong, correct, force_tokens)
    cie = control_is_effort(wrong, force_tokens)
    br_ok, br = bridge_pass(wrong)
    tail = ["changed_rate", "trunc_rate", "degenerate_rate", "mean_gen_tokens", "mean_rounds",
            "forced_short_rate", "frac_stopped_at_floor", "answer_changed_after_floor",
            "reemit_x_rate", "no_new_box_rate", "n_no_new_box"]
    lines = ["## math_effort_gate — 노력을 외생화해서 통제 실패를 가른다", "", "### 오답 행"]
    lines += _table(wrong, ["rescue", "reproduction_rate"] + tail)
    lines += ["", f"n_rollouts={wrong.get('n_rollouts')} · K={wrong.get('k')} · force_mode="
                  f"{meta.get('force_mode')}", "", "### 짝지은 Δ (오답 행, blind_matched 는 "
                  "REPORTED 진단 — ★E1 판정에는 안 쓴다)", "",
              "| contrast | Δ [95% CI] | sign p |", "|---|---|---|"]
    lines += [f"| {key} | {_fmt(ci)} | {_fmt((wrong.get('sign_p') or {}).get(key))} |"
              for key, ci in (wrong.get("paired") or {}).items()]
    lines += ["", "### 정답 행(거짓-경보, 강제 팔만)"] + _table(correct, ["flip_wrong_rate"] + tail)
    lines += ["", f"**EFFORT {'PASS' if ok else 'FAIL'}**{f' — {arm}' if arm else ''} — 어떤 "
                  f"wait_forced_N 이 (a) (forced − blind) CI 0 제외 ∧ 평균 ≥ +{PASS_DELTA} "
                  f"(b) flip_wrong_rate ≤ {MAX_FLIP_WRONG} (c) 그 팔 자신의 trunc_rate ≤ "
                  f"{MAX_TRUNC} (d) 그 팔 자신의 degenerate_rate ≤ {MAX_DEGEN} 을 다 만족 — "
                  "여럿이면 구제율(rescue) 최고 팔을 고른다(blind 는 전체 예산 눈감은 풀이)",
              f"**CONTROL-IS-EFFORT** = {cie.get('best_arm')} − {cie.get('pad_arm')} = "
              f"{_fmt(cie.get('delta'))} (sign p={_fmt(cie.get('sign_p'))}) — ≈0 이면 강제는 "
              "«점검하게 만든 것» 이 아니라 «더 쓰게 만든 것» 이다",
              f"**BRIDGE {'PASS' if br_ok else 'FAIL'}** — rescue(bridge_free)="
              f"{_fmt(br.get('rescue_bridge_free'))} ≥ {BRIDGE_RESCUE_FRAC}×rescue(blind)="
              f"{_fmt(br.get('rescue_blind'))} ? {br.get('vs_blind_ok')} ∧ Δ(bridge_free − "
              f"bridge_pad_free)={_fmt(br.get('delta_vs_placebo'))} 하한>0 — 다리가 문맥 안에서 "
              "구제하고, 그 구제가 위약(길이만 맞춘 «다시 풀자») 때문이 아니다",
              "", f"2패스 예산(blind_matched): {meta.get('matched_budgets')}", ""]
    return "\n".join(lines)


# ── 생성·기록 ───────────────────────────────────────────────────────────────────
def make_vllm_sampler(llm, sp_cls, seed: int) -> Callable:
    """vllm LLM → `sample(prompts, n=, max_tokens=, min_tokens=)`. max_tokens 는 int 또는
    프롬프트별 list(대체 경로가 «남은 예산» 을 프롬프트마다 다르게 준다)."""
    def sample(prompts, *, n, max_tokens, min_tokens=0):
        mt = max_tokens if isinstance(max_tokens, (list, tuple)) else [max_tokens] * len(prompts)
        sps = [sp_cls(n=n, temperature=1.0, top_p=1.0, max_tokens=max(1, int(m)),
                      min_tokens=max(0, min(int(min_tokens), max(1, int(m)))),
                      ignore_eos=False, seed=seed) for m in mt]
        return [[{"text": x.text, "finish_reason": x.finish_reason,
                  "n_tokens": len(x.token_ids)} for x in o.outputs]
                for o in llm.generate(list(prompts), sps)]
    return sample

def run_pass(sample, tok, rows, conds, *, variant: str, k: int, max_tokens: int, lim: int,
             budgets: dict | None = None, use_min_tokens: bool = True) -> tuple[dict, int]:
    """조건별로 프롬프트를 조립해 생성 → ({(row_index, cond): [gen, ...]}, 너무 길어 버린 수)."""
    got: dict = {}
    n_drop = 0
    for c in conds:
        pairs = [(si, build_prompt(tok, variant, c, s["problem"], s.get("text", "")))
                 for si, s in enumerate(rows)]
        keep = [(si, q) for si, q in pairs if len(tok.encode(q)) <= lim]
        n_drop += len(pairs) - len(keep)
        if not keep:
            continue
        res = generate(sample, [q for _, q in keep], k=k, force_tokens=force_floor(c),
                       use_min_tokens=use_min_tokens,
                       max_tokens=(budgets or {}).get(c, max_tokens))
        for (si, _), row in zip(keep, res):
            got[(si, c)] = row
    return got, n_drop

def rows_from(got: dict, rows, conds, *, population: str) -> tuple[list, list]:
    """(per-row 기록, gens 행). 모든 조건이 다 성립한 행만 기록에 넣는다. `degen_*`(★E2)는
    모든 조건에, `stopfloor_*`·`aftfloor_*`(★E3)는 강제 팔에만 붙는다."""
    recs, gens = [], []
    orig_corr = 1 if population == "correct" else 0
    for si, s in enumerate(rows):
        have = {c: got.get((si, c)) for c in conds}
        oa = last_boxed(s.get("text", ""))
        rec = {"roll_id": s["roll_id"], "group_id": s["group_id"], "population": population}
        for c, xs in have.items():
            cont = is_continuation(c)
            rcs = [effective_r_corr(x["text"], s["gold"], orig_r_corr=orig_corr, cont=cont)
                   for x in (xs or [])]
            gens += [{"cond": c, "group_id": s["group_id"], "roll_id": s["roll_id"],
                      "population": population, "text": x["text"], "r_corr": rc,
                      "truncated": x["truncated"], "n_gen_tokens": x["n_gen_tokens"],
                      "rounds": x["rounds"], "degenerate": x.get("degenerate", 0)}
                     for x, rc in zip(xs or [], rcs)]
            if not xs:
                continue
            st = [answer_stats(x["text"], oa, cont=cont) for x in xs]
            rec[f"p_{c}"], rec[f"flip_{c}"] = _p(rcs), _p([1 - v for v in rcs])
            rec[f"repro_{c}"], rec[f"changed_{c}"] = _p([a for a, _ in st]), _p([b for _, b in st])
            rec[f"trunc_{c}"] = _p([x["truncated"] for x in xs])
            rec[f"ntok_{c}"] = _p([x["n_gen_tokens"] for x in xs])
            rec[f"rounds_{c}"] = _p([x["rounds"] for x in xs])
            rec[f"degen_{c}"] = _p([x.get("degenerate", 0) for x in xs])         # ★E2
            nb = [last_boxed(x["text"] or "") for x in xs]                       # ★O1
            rec[f"nobox_{c}"] = _p([int(not b) for b in nb])
            rec[f"nnobox_{c}"] = sum(int(not b) for b in nb)
            rec[f"reemit_{c}"] = _p([int(bool(b) and bool(answers_equivalent(oa, b))) for b in nb])
            n = force_floor(c)
            if n:
                rec[f"short_{c}"] = _p([int(x["n_gen_tokens"] < n) for x in xs])
                # ★E3: 강제의 문자만 지켰는가(바닥 ±5% 안에서 멈췄다) + 강제가 답을 바꿨는가.
                rec[f"stopfloor_{c}"] = _p([int(abs(x["n_gen_tokens"] - n) <= FLOOR_TOL * n)
                                           for x in xs])
                rec[f"aftfloor_{c}"] = _p([b for _, b in st])
        if all(have.values()):
            recs.append(rec)
    return recs, gens


# ── 재요약 ──────────────────────────────────────────────────────────────────────
def resummarize_effort(gens_path: str, rollouts: str, *, out_dir: str, seed: int = 11,
                       n_boot: int = 2000, k: int = 0) -> int:
    r"""`--resummarize` — 생성 없이(GPU 없이) 이미 있는 `gens.jsonl` + 원 `texts.jsonl` 로
    요약을 다시 만든다. 이 게이트의 gens 행은 `r_corr` 가 **생성**의 정오라 1차 시도의 정오가
    행에 없다 — 그래서 `--rollouts`(재채점된 texts.jsonl)를 **같이** 받아 roll_id 로 잇는다.
    ★그러면 `rows_from` 을 그대로 다시 태울 수 있어(원문 problem/gold/원 답이 거기 있다)
      `repro_*`·`changed_*` 까지 전부 같은 정의로 다시 난다 — 통계 정의는 하나도 안 바뀐다.
    ★«오답» 모집단인데 1차 시도가 재채점으로 **정답**이 된 단위는 "직전 시도가 틀렸다" 는
      거짓 사실을 받은 것이라 뺀다(`n_units_dropped_a1_correct`)."""
    gens = [json.loads(ln) for ln in open(gens_path) if ln.strip()]
    if not gens:
        raise SystemExit(f"[EFF] 빈 gens 파일: {gens_path}")
    roll_of: dict = {}
    for i, ln in enumerate(open(rollouts)):
        if not ln.strip():
            continue
        r = json.loads(ln)
        roll_of[f"{r['group_id']}#{i}"] = r

    conds_seen = {g["cond"] for g in gens}
    force_tokens = sorted({force_floor(c) for c in conds_seen if c.startswith("wait_forced_")})
    p1, p2, ccond = cond_names(force_tokens)
    p1 = [c for c in p1 if c in conds_seen] + [c for c in BRIDGE_CONDS if c in conds_seen]
    p2 = [c for c in p2 if c in conds_seen]
    ccond = [c for c in ccond if c in conds_seen]

    out_recs, out_gens, n_dropped = [], [], 0
    per_pop: dict = {}
    for g in gens:
        per_pop.setdefault(g.get("population", "wrong"), []).append(g)
    for pop, conds in (("wrong", p1 + p2), ("correct", ccond)):
        rows, index_of = [], {}
        got: dict = {}
        for g in per_pop.get(pop, []):
            rid = g["roll_id"]
            src = roll_of.get(rid)
            if src is None:
                continue
            if pop == "wrong" and int(src.get("r_corr") or 0) == 1:
                if rid not in index_of:
                    index_of[rid] = None
                    n_dropped += 1
                continue
            if rid not in index_of:
                index_of[rid] = len(rows)
                rows.append({"roll_id": rid, "group_id": g.get("group_id", src["group_id"]),
                             "problem": src["problem"], "gold": src["gold"],
                             "text": src.get("text", "")})
            si = index_of[rid]
            if si is None:
                continue
            got.setdefault((si, g["cond"]), []).append(
                {"text": g["text"], "truncated": int(g.get("truncated", 0)),
                 "n_gen_tokens": int(g.get("n_gen_tokens", 0)),
                 "rounds": int(g.get("rounds", 1)), "degenerate": int(g.get("degenerate", 0))})
        recs, gg = rows_from(got, rows, conds, population=pop)
        out_recs += recs
        out_gens += gg
        per_pop[pop + "_recs"] = recs
    wrecs, crecs = per_pop.get("wrong_recs", []), per_pop.get("correct_recs", [])

    wsumm = summarize(wrecs, p1 + p2, force_tokens=force_tokens, k=k, seed=seed, n_boot=n_boot)
    csumm = summarize(crecs, ccond, k=k, seed=seed, n_boot=n_boot)
    ok, arm = effort_pass(wsumm, csumm, force_tokens)
    wsumm["pass_effort"], wsumm["pass_arm"] = int(ok), arm
    wsumm["control_is_effort"] = control_is_effort(wsumm, force_tokens)
    _, wsumm["bridge"] = bridge_pass(wsumm)                                    # ★O1
    for line in weak_anchor_warnings(wsumm, force_tokens):
        print(line, flush=True)
    for line in degenerate_warnings(wsumm) + degenerate_warnings(csumm):
        print(line, flush=True)
    meta = {"resummarize_from": str(gens_path), "rollouts": rollouts, "seed": seed,
            "force_tokens": force_tokens, "force_mode": "resummarize", "matched_budgets": {},
            "n_units_dropped_a1_correct": n_dropped, "n_generations": len(out_gens)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_row.jsonl").open("w") as fh:
        for r in out_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(
        {"wrong": wsumm, "correct": csumm, "meta": meta}, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(md := to_markdown(wsumm, csumm, force_tokens, meta))
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--resummarize", default="",
                    help="이미 있는 gens.jsonl 로 요약만 다시 만든다(GPU 없이). "
                         "--rollouts 는 그대로 필요하다(1차 시도의 정오가 거기 있다)")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="math_opt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_sites", type=int, default=150)
    ap.add_argument("--per_problem", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force_tokens", default="1024,3072", help="강제 바닥 N 목록(콤마)")
    ap.add_argument("--force_mode", default="auto", choices=("auto", "min_tokens", "rounds"),
                    help="auto 는 min_tokens 지원을 실측해 고른다")
    ap.add_argument("--max_tokens", type=int, default=8192,
                    help="★생성 예산 — 4k 는 절단률이 정확도 순서를 뒤집는다(cd9 G4)")
    ap.add_argument("--max_prefix_tokens", type=int, default=8192)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--conds", default="",
                    help="★O1 1패스 조건을 직접 고른다(콤마). 기본값(빈 문자열)이면 기존 "
                         "cond_names 그대로. bridge_free/bridge_pad_free 는 여기서만 켠다. "
                         "wait_forced_* 를 하나도 안 고르면 강제 팔·2패스·정답 행이 없다")
    ap.add_argument("--skip_matched", action="store_true",
                    help="2패스(blind_matched_N)를 건너뛴다 — REPORTED 진단만 포기하고 "
                         "생성량을 줄인다(★E1 판정은 blind 앵커라 영향 없음)")
    a = ap.parse_args()

    selftest_math_verify()
    if a.resummarize:
        return resummarize_effort(a.resummarize, a.rollouts, out_dir=a.out_dir, seed=a.seed,
                                  n_boot=a.n_boot, k=a.k)
    if not a.model_path:
        raise SystemExit("[EFF] --resummarize 가 없으면 --model_path 가 필수다")
    force_tokens = [int(x) for x in str(a.force_tokens).split(",") if str(x).strip()]
    if not force_tokens and not a.conds:
        raise SystemExit("[EFF] --force_tokens 가 비었다.")
    p1, p2, ccond = cond_names(force_tokens)
    if a.conds:                       # ★O1: 1패스 조건을 직접 고른다(기본 경로는 안 바뀐다)
        want = [c.strip() for c in str(a.conds).split(",") if c.strip()]
        if bad := [c for c in want if c not in list(p1) + list(BRIDGE_CONDS)]:
            raise SystemExit(f"[EFF] 모르는 조건: {bad}")
        p1 = want
        force_tokens = sorted({force_floor(c) for c in want if c.startswith("wait_forced_")})
        _, p2, ccond = cond_names(force_tokens)
        ccond = [c for c in ccond if c in want]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rolls = [json.loads(line) for line in open(a.rollouts)]

    def pick(cands, cap):                       # 선별 규약은 두 게이트와 같다 + seeded 셔플
        rng.shuffle(cands)
        return cands[:cap]
    wrong = pick(select_wrong_rollouts(rolls, per_problem=a.per_problem), a.max_sites)
    correct = pick(select_correct_rollouts(rolls, per_problem=1), len(wrong))
    if not wrong:
        raise SystemExit("[EFF] 오답 후보가 없다 — MIXED 오답이 있는지 확인하라.")
    print(f"[eff] 오답 {len(wrong)} / 정답 {len(correct)} · 1패스 {p1} · 2패스 {p2}", flush=True)

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    use_min = a.force_mode != "rounds"
    if a.force_mode == "auto":
        try:
            SamplingParams(n=1, max_tokens=8, min_tokens=4)
        except Exception as e:                                  # noqa: BLE001
            print(f"[eff] min_tokens 미지원({e}) → rounds 대체 경로", flush=True)
            use_min = False
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + a.max_prefix_tokens + 1024, enforce_eager=True)
    tok = llm.get_tokenizer()
    sample = make_vllm_sampler(llm, SamplingParams, a.seed)
    kw = {"variant": a.variant, "k": a.k, "max_tokens": a.max_tokens,
          "lim": a.max_prefix_tokens + 512, "use_min_tokens": use_min}

    gw, dw = run_pass(sample, tok, wrong, p1, **kw)                 # 1패스(오답)
    gc, dc = run_pass(sample, tok, correct, ccond, **kw)            # 1패스(정답 거짓-경보)
    # ── 2패스: blind_matched_* — 예산은 1패스를 **재야만** 안다(docstring 참조) ──
    if a.skip_matched:
        print("[eff] --skip_matched: 2패스 생략(REPORTED blind_matched 진단 없음)", flush=True)
        budgets, p2, d2 = {}, [], 0
    else:
        w1, _ = rows_from(gw, wrong, p1, population="wrong")
        budgets = {f"blind_matched_{n}": matched_budget(w1, f"wait_forced_{n}",
                                                        max_tokens=a.max_tokens)
                   for n in force_tokens}
        print(f"[eff] 2패스 예산 {budgets}", flush=True)
        g2, d2 = run_pass(sample, tok, wrong, p2, budgets=budgets, **kw)
        gw.update(g2)

    wrecs, wgens = rows_from(gw, wrong, p1 + p2, population="wrong")
    crecs, cgens = rows_from(gc, correct, ccond, population="correct")
    for name, payload in (("gens.jsonl", wgens + cgens), ("per_row.jsonl", wrecs + crecs)):
        with (out / name).open("w") as fh:
            for r in payload:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    wsumm = summarize(wrecs, p1 + p2, force_tokens=force_tokens, k=a.k, seed=a.seed,
                      n_boot=a.n_boot)
    csumm = summarize(crecs, ccond, k=a.k, seed=a.seed, n_boot=a.n_boot)
    ok, arm = effort_pass(wsumm, csumm, force_tokens)
    wsumm["pass_effort"], wsumm["pass_arm"] = int(ok), arm
    wsumm["control_is_effort"] = control_is_effort(wsumm, force_tokens)
    _, wsumm["bridge"] = bridge_pass(wsumm)                                    # ★O1
    for line in weak_anchor_warnings(wsumm, force_tokens):        # ★E1
        print(line, flush=True)
    for line in degenerate_warnings(wsumm) + degenerate_warnings(csumm):   # ★E2
        print(line, flush=True)
    meta = {"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
            "seed": a.seed, "max_tokens": a.max_tokens, "force_tokens": force_tokens,
            "force_mode": ("min_tokens" if use_min else "rounds"), "matched_budgets": budgets,
            "wait_cue": WAIT_CUE, "pad_cue": PAD_CUE, "conds": p1,
            "bridge_tmpl": BRIDGE_TMPL, "placebo_bridge": PLACEBO_BRIDGE,     # ★O1
            "n_wrong": len(wrong),
            "n_correct": len(correct), "n_dropped_long": dw + dc + d2,
            "n_generations": len(wgens) + len(cgens)}
    (out / "gate_summary.json").write_text(json.dumps(
        {"wrong": wsumm, "correct": csumm, "meta": meta}, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(md := to_markdown(wsumm, csumm, force_tokens, meta))
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
