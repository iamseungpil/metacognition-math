#!/usr/bin/env python
r"""math_verify_perform_gate — V1 «검산 **수행**의 외생화»: 말하게 두지 말고 시켜라.

왜: `meta_content_prompt_screen` 재분석이 cd9 의 검산 null 을 한 겹 벗겼다. «다른 방법으로 다시
  풀어라»는 지시는 조건 수준에서 null 인데, 그 조건의 생성물을 뜯어 보면 정책은 검산을 **말만**
  한다 — 말한 비율 **.955** 대 실제로 수행한 비율 **.386**, 절단 **.313**. 그리고 **같은 문제
  안에서** 수행한 생성과 말만 한 생성을 갈라 보면 Δ 는 recompute **+.17** · magnitude **+.16** ·
  backward_mask **+.11** 이다. 즉 조건 수준의 null 은 «검산이 무용하다»가 아니라 «지시가 수행을
  못 만들어냈다 + 예산이 모자랐다» 와 구분이 안 된다.
  ★그 Δ 는 **관찰**이다: 수행 여부는 정책이 고른 것이라 문제 난이도·그 시행의 상태와 섞여 있다
    (쉬운 문제에서 더 자주 수행했다면 Δ 는 수행의 효과가 아니라 난이도의 효과다).
  V1 은 그 교란을 **외생화**로 끊는다 — (a) 구조화된 출력 템플릿(섹션 이름과 «Method 2 result:
  <value>» 같은 채워야 하는 슬롯)으로 수행을 요구하고, (b) 예산을 **12,288** 로 올려 절단
  .313 을 걷어내고, (c) 같은 문제 안에서 **구조가 동일하고 내용만 없는** 템플릿과 짝지어 잰다.

★중심 대조는 `tmpl_null` 이다(`plain` 이 아니다). 어떤 템플릿이든 붙이면 응답은 길어지고 섹션이
  생기고 분포가 흔들린다 — 그 흔들림을 «검산이 도왔다»로 읽은 것이 cd9 의 반복된 함정이다.
  `tmpl_null` 은 **같은 세 섹션 구조**(반추 + 최종)를 요구하되 새 계산을 금지한다 —
  Gandhi 외 arXiv:2503.01307 의 length-matched placeholder 논리 그대로다(그 논문은 이런 대조를
  null 로 보고한다). 그래서 **주장은 (act − tmpl_null)**, `plain` 은 절대 수준을 읽는 앵커다.
★수행률을 통과 규칙에 넣는다: `performed_rate < .60` 이면 이 게이트는 **아무것도 시험하지
  않은 것**이고 그때의 Δ 는 해석 불가다(`[PERFORM-LOW]`). 절단도 마찬가지로 `.20` 초과면
  `[BUDGET?]` — cd9 G4 에서 절단률 순서가 정확도 순서를 **정확히 역순**으로 뒤집었다.
★수행 판정은 `src/training/verify_terms.py` 한 곳에서만 온다(어휘 일치가 아니라 «계산줄이 실제로
  있는가 ∧ 그 결과를 판정했는가»). 그 모듈은 나중에 보상 항으로 그대로 재사용한다 —
  PAL 경고(arXiv:2211.10435, 머릿속 시뮬레이션 23.2% vs 실제 실행 72.0%)가 그 설계의 근거다.
★Δ 옆에 **범주 히스토그램**(confirm_right/confirm_wrong/revise_right/revise_wrong)을 같이 찍는다.
  Δ≈0 이어도 revise_right ≈ revise_wrong 이면 결론은 «검산이 무용»이 아니라 «검산이 방향 없이
  흔든다»다 — 그 둘은 다음 수가 전혀 다르다.

★0915 재분석이 두 가지를 고쳤다(V1 → V1b):
  ① **수행 판정기**: 템플릿 조건은 `performed_template`(섹션 ∧ 슬롯 ∧ Compare 안의 판정)로
     잰다. V1 은 자유서술용 어휘 탐지기로 템플릿 출력을 재서 backward 수행률을 **.19** 로
     찍었는데, 템플릿 인식으로 다시 재면 **.79** 다 — 그 한 줄이 «수행이 안 일어났다»는
     잘못된 그림을 통째로 만들었다. 대조군만 `performed_check`(자연발생 수행)를 유지한다.
  ② **두 번째 대조 `reread_null`**: backward_tmpl 의 +.044 중 **85%가 답을 바꾸지 않은 행**
     에서 왔고, 그 행들의 'Recovered:' 값은 대개 **문제가 이미 준 수**였다. 그렇다면 이득의
     정체는 검산이 아니라 «답을 쓴 뒤 문제를 다시 읽은 것»(재주목)일 수 있다. `tmpl_null` 은
     새 계산을 금지할 뿐 되읽기를 시키지 않아 그것을 통제하지 못한다. `reread_null` 은
     **문제의 수량을 한 줄씩 나열**만 시킨다(새 계산 금지).

통과 규칙(찍고 json 에도 남긴다): **PERFORM-CAUSAL PASS** ⇔ 어떤 `*_tmpl` 이
  (act − tmpl_null) 의 95% CI 가 0 을 제외 ∧ 평균 ≥ +0.03 ∧ `performed_rate` ≥ .60 ∧
  `trunc_rate` ≤ .20. **`--stage confirm`** 이면 여기에 (act − reread_null) 의 CI 가 0 제외 ∧
  평균 ≥ +0.03 이 추가된다. `[REREAD-CONFOUND?]` 는 (act − tmpl_null) > 0 인데
  (act − reread_null) ≤ 0 인 팔에 붙는다 — 그 이득은 «검산»이라고 부를 수 없다.

짝짓기(구 게이트의 버그를 안 옮긴다): `math_meta_content_gate` 는 «모든 조건이 다 있는 단위만»
  요약에 넣어서, 한 조건이 길이 초과로 빠지면 그 문제가 **모든 비교에서** 통째로 사라졌다.
  여기서는 **조건 쌍마다** 짝을 짓는다 — 어떤 단위는 자기가 못 가진 쌍에서만 빠진다.

모집단: 두 코퍼스(mathL5 / math500)의 MIXED 문제를 `select_mixed_problems` 로 고르고
  `--max_problems` 를 출처에 고르게 나눈다. 두 파일이 group_id 를 «g0»부터 각자 붙이므로
  `source_tag` 로 네임스페이스를 가른다(안 그러면 서로 다른 문제가 한 단위로 뭉개진다).

사용(예):
  python scripts/local/math_verify_perform_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --rollouts /hdd_data/seungpil/scratch/eval/math500_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 --variant math_opt \
      --k 8 --max_problems 120 --seed 11 --max_tokens 12288 --gpu_util 0.45 \
      --out_dir /hdd_data/seungpil/scratch/eval/verify_perform_s1

확인 실행(s2 — **안 쓴 문제**에서, 되읽기 대조를 넣고):
  python scripts/local/math_verify_perform_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --conds tmpl_null,reread_null,backward_tmpl,recompute_tmpl \
      --exclude_units_from /hdd_data/seungpil/scratch/eval/verify_perform_s1/per_problem.jsonl \
      --max_problems 101 --k 8 --seed 23 --max_tokens 12288 --stage confirm \
      --out_dir /hdd_data/seungpil/scratch/eval/verify_perform_s2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import bootstrap_ci, sign_test_p  # noqa: E402  (통계 정의는 한 곳)
from math_plan_gate import select_mixed_problems  # noqa: E402  (MIXED 선별 규약 재사용)
from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_meta import (  # noqa: E402
    boxed_spans, grade_math, last_boxed, selftest_math_verify,
)
from src.training.verify_terms import (  # noqa: E402
    performed_check, performed_template, recovered_value_appears_in_problem, self_correction,
)

_NAN = float("nan")
PASS_DELTA = 0.03       # (act − tmpl_null) 평균의 하한
PASS_DELTA_REREAD = 0.03  # confirm 단계: (act − reread_null) 평균의 하한
MIN_PERFORMED = 0.60    # 수행률 하한 — 이 아래면 Δ 는 해석 불가([PERFORM-LOW])
MAX_TRUNC = 0.20        # 절단율 상한 — 넘으면 [BUDGET?] (cd9 G4: 절단이 순서를 뒤집는다)
CATEGORIES = ("confirm_right", "confirm_wrong", "revise_right", "revise_wrong", "none")
VERDICTS = ("same", "different", "none")
STAGES = ("screen", "confirm")
FINAL_HEAD_RX = re.compile(r"^#{1,3}\s*Final\b.*$", re.I | re.M)   # '## Final' 섹션 머리


class Cond(NamedTuple):
    """한 조건. `suffix`=""는 기준선(render_generation_prompt 과 **바이트 동일**).
    `act` 는 `verify_terms` 의 행위 이름(None = 수행을 시험하지 않는 기준·대조군 — 그래도
    수행률은 `probe` 로 잰다: 대조군에서도 자연발생 수행을 봐야 Δ 가 읽힌다)."""
    suffix: str      # 사용자 턴 끝에 붙는 접미(조건 사이의 **유일한** 차이)
    act: str | None  # verify_terms 의 수행 판정에 쓸 행위(None = 대조군)
    probe: str       # 수행률을 잴 때 쓰는 행위(대조군도 같은 잣대로 재려고 둔다)


# ★★ 조건의 단일 진실 원천 — 문장을 바꾸려면 여기만 고친다. 요약·표·판정이 전부 이 dict 를
#   따라온다(테스트가 전파를 고정). 항목마다 출처 주석을 단다. ★★
VERIFY_CONDS: dict[str, Cond] = {
    # 기준선 — `render_generation_prompt` 과 바이트 동일(절대 수준 앵커). 수행률은 recompute
    #   잣대로 잰다(«지시 없이도 이미 수행하는 비율»). provenance: no prior(기준선 그 자체).
    "plain": Cond(suffix="", act=None, probe="recompute"),
    # 내용 없는 구조 대조 — **같은 섹션 구조**(리뷰 + 최종)를 요구하되 새 계산을 금지한다.
    #   provenance: **Gandhi 외 arXiv:2503.01307** 의 length-matched placeholder(그 논문이
    #   null 로 보고하는 대조). ★주장은 (act − 이것) 이다 — 템플릿을 붙였다는 사실 자체의
    #   효과(길이·섹션·어조)를 여기서 빼낸다.
    "tmpl_null": Cond(
        suffix=("\n\nAfter your solution, add a section '## Review' with three sentences of "
                "general reflection on careful work (no new calculations), then a section "
                "'## Final' containing your final answer in \\boxed{}."),
        act=None, probe="recompute"),
    # ★두 번째 대조 — «답을 쓴 뒤 문제를 다시 읽는다». 구조(섹션 + 최종)는 같고, 새 계산은
    #   금지하되 **문제의 수량을 한 줄씩 나열**하게 한다. provenance: 0915 재분석 —
    #   backward_tmpl 의 +.044 중 **85%가 답을 바꾸지 않은 행**에서 왔고 그 행들의
    #   'Recovered:' 값은 대개 **문제가 이미 준 수**였다. 즉 이득의 정체가 «검산»이 아니라
    #   «되읽기/재주목»일 수 있다. tmpl_null 은 새 계산을 금지할 뿐 되읽기를 시키지 않으므로
    #   그 교란을 못 가른다 — 이 조건이 그것만 따로 준다.
    "reread_null": Cond(
        suffix=("\n\nAfter your solution, add a section '## Recheck': list every numeric "
                "quantity and condition given in the problem statement, in one line each, "
                "with no new calculations. Then a section '## Final' containing your final "
                "answer in \\boxed{}."),
        act=None, probe="recompute"),
    # 재계산 — 두 번째 유도를 **슬롯으로** 강제한다('Method 2 result: <value>' 를 채워야 한다).
    #   provenance: 관찰(이 코퍼스의 수행 내 Δ +.17) + self-consistency 계열의 일반 근거.
    #   말만 하기 어려운 이유가 슬롯이다 — 값을 쓰려면 계산해야 한다.
    "recompute_tmpl": Cond(
        suffix=("\n\nAfter your solution, add '## Method 2': solve the problem again by a "
                "different method, ending with the line 'Method 2 result: <value>'. Then "
                "'## Compare': state whether the two results are the same or different; if "
                "different, determine which is correct. Then '## Final' with \\boxed{}."),
        act="recompute", probe="recompute"),
    # 역방향 — 답이 맞다고 가정하고 문제의 수량 하나를 복원해 원문 값과 대조한다.
    #   provenance: **arXiv:2212.09561**(self-verification) / **arXiv:2312.06867** — 실 이득을
    #   보고하되 ~7B 미만에서는 비일관이라고도 보고한다(우리 4B 는 그 구간 — 기대치를 낮춰 둔다).
    #   관찰 Δ +.11.
    "backward_tmpl": Cond(
        suffix=("\n\nAfter your solution, add a section '## Backward check': assuming your "
                "answer is correct, recover one numeric quantity given in the problem and "
                "write 'Recovered: <value> vs stated: <value>'. Then '## Compare': same or "
                "different; if different, resolve. Then '## Final' with \\boxed{}."),
        act="backward", probe="backward"),
    # 크기·부호 — **풀기 전에** 기대를 적고 푼 뒤 대조한다(사후 합리화를 자리로 막는다).
    #   provenance: 관찰(수행 내 Δ +.16) — no prior. 가장 싼 행위다(재유도가 없다).
    "magnitude_tmpl": Cond(
        suffix=("\n\nBefore solving, write '## Expectation': the expected sign and rough size. "
                "After solving, '## Compare': does the answer match the expectation "
                "(same/different); if different, resolve. Then '## Final' with \\boxed{}."),
        act="magnitude", probe="magnitude"),
}

REF = "plain"          # 절대 수준 앵커(바이트 동일 기준선)
CONTROL = "tmpl_null"  # ★주장의 기준 — 구조는 같고 내용만 없는 대조
CONTROL2 = "reread_null"  # ★두 번째 기준 — 되읽기만 시키는 대조(0915 재분석)
ACT_CONDS = tuple(n for n, c in VERIFY_CONDS.items() if c.act is not None)


def conds_all() -> tuple[str, ...]:
    """조건 이름(정의 순서) — VERIFY_CONDS 가 단일 진실 원천이다."""
    return tuple(VERIFY_CONDS)


# ── 프롬프트 조립 ───────────────────────────────────────────────────────────────
def cond_prompt(tok, variant: str, problem: str, name: str) -> str:
    """한 조건의 프롬프트. suffix 가 ""이면 `render_generation_prompt` 과 **바이트 동일**,
    아니면 **사용자 턴 끝에 접미만** 붙는다 — system 문구를 새로 만들면 «템플릿의 효과»와
    «프롬프트가 달라진 효과»가 섞인다(math_meta_content_gate.act_prompt 와 같은 계약)."""
    suf = VERIFY_CONDS[name].suffix
    if not suf:
        return render_generation_prompt(tok, variant, problem)
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + suf}
    return render_chat_messages(tok, msgs)


# ── 채점 ───────────────────────────────────────────────────────────────────────
def final_answer(text: str) -> str:
    r"""최종 답 = **'## Final' 섹션 안의 마지막 \boxed**, 그 섹션이 없으면 전체의 마지막 \boxed.
    ★왜 섹션을 먼저 보는가: 이 템플릿들은 '## Method 2' 안에도 \boxed 를 쓸 수 있다. 전체의
      마지막만 보면 «두 번째 방법의 중간값»이 최종 답으로 채점되는 사고가 난다. 섹션이
      비어 있거나(모델이 머리만 쓰고 박스를 안 씀) 없으면 전체 마지막으로 폴백한다."""
    t = text or ""
    ms = list(FINAL_HEAD_RX.finditer(t))
    if ms:
        tail = t[ms[-1].end():]
        sp = boxed_spans(tail)
        if sp:
            return sp[-1][0]
    return last_boxed(t)


def grade_final(text: str, gold: str) -> int:
    r"""최종 답만 math_verify 로 채점한다(`final_answer` 로 뽑은 값을 다시 박스에 싸서 —
    `grade_math` 의 파서가 그대로 쓰이도록)."""
    ans = final_answer(text)
    if not ans:
        return 0
    return grade_math(f"\\boxed{{{ans}}}", gold)


def score_text(text: str, cond: str, *, corr: int, truncated: int, tokens: int,
               problem: str = "") -> dict:
    """생성 하나(이미 채점된 정오 `corr`) → 기록. 수행·판정·변경은 전부 `verify_terms` 한
    곳에서 온다 — 이 함수 하나가 최초 생성 경로(`gen_record`)와 `--resummarize`(저장된
    text 를 CURRENT 탐지기로 다시 재는 경로)가 **같은 코드**를 타게 한다(둘이 갈라지면
    "재요약"이 "재현"이 아니게 된다).
    ★템플릿 조건(act ≠ None)은 `performed_template`(섹션 ∧ 슬롯 ∧ Compare 안의 판정 ∧
      본 섹션의 계산줄 하한)으로 잰다. 대조군(act=None)은 «지시 없이도 자연발생으로
      수행하는가»를 재는 것이므로 `performed_check(probe)` 를 쓴다(잣대가 다르니 분모도
      다르다 — 두 열을 한 숫자로 섞지 말 것). 대조군에는 `performed_tmpl_sanity` 도 같이
      남긴다: 새 계산을 금지한 대조가 템플릿 수행으로 읽히면 대조가 새는 것이다(0 이어야
      정상).
    ★범주는 `verdict_grade` 를 그대로 쓰지 않고 여기서 조립한다 — 정오는 이 게이트의
      섹션 인식 채점기(`grade_final`)로 재야 하고, `verdict_grade` 는 전체 마지막 박스를
      쓰기 때문이다(그 차이를 조용히 섞지 않는다). `problem` 이 없으면(예: 저장된
      gens.jsonl 에 문제 본문이 없는 재요약) `recovered_in_problem` 은 생략한다."""
    t = text or ""
    act = VERIFY_CONDS[cond].act
    sc = self_correction(t)
    corr = int(bool(corr))
    rec: dict = {"r_corr": corr, "trunc": int(truncated), "tokens": int(tokens),
                 "changed": float(sc["changed"]), "n_boxed": sc["n_boxed"]}
    if act is not None:
        pt = performed_template(t, act)
        rec.update({
            "performed": float(pt["performed"]), "verdict": pt["verdict"],
            "sections": pt["sections"], "slot_values": pt["slot_values"],
            "sec_main": float(pt["sections"]["main"]), "sec_slot": float(pt["sections"]["slot"]),
            "sec_compare": float(pt["sections"]["compare"]),
            "n_eq_lines_main": pt["n_eq_lines_main"],
            # compliance(형식 준수) ≠ performed(수행) — sections 전부 1 이면 1, 계산줄
            # 하한과는 무관하다. 요약에서 이 둘을 나란히 보여줘야 «형식은 맞췄는데 계산은
            # 안 한» 행이 보인다.
            "compliant": float(int(all(pt["sections"].values()))),
            "detector": "template"})
        if act == "backward":
            rec["recovered_matches_stated"] = float(bool(pt.get("recovered_matches_stated")))
            # ★«무효 검산» 신호 — 복원했다는 값이 문제에 이미 적혀 있으면 아무것도 복원하지
            #   않은 것이고, 그 행의 이득은 검산이 아니라 되읽기일 수 있다. 문제 본문이 없으면
            #   못 잰다(재요약 모드에서 gens.jsonl 이 problem 을 안 들고 있을 수 있다).
            if problem:
                rec["recovered_in_problem"] = float(recovered_value_appears_in_problem(
                    problem, (pt["slot_values"] or {}).get("recovered", "")))
    else:
        pc = performed_check(t, VERIFY_CONDS[cond].probe)
        rec.update({"performed": float(pc["performed"]), "verdict": pc["verdict"],
                    "detector": "freetext",
                    # 온전성: 새 계산을 금지한 대조가 템플릿 수행으로 읽히면 안 된다.
                    "performed_tmpl_sanity": float(
                        performed_template(t, VERIFY_CONDS[cond].probe)["performed"])})
    if not rec["performed"]:
        cat = "none"
    elif sc["changed"]:
        cat = "revise_right" if corr else "revise_wrong"
    else:
        cat = "confirm_right" if corr else "confirm_wrong"
    rec["category"] = cat
    rec["revised"] = float(bool(rec["performed"]) and bool(sc["changed"]))
    return rec


def gen_record(text: str, cond: str, gold: str, *, truncated: int, tokens: int,
               problem: str = "") -> dict:
    """생성 하나(생성 시점 — gold 로 새로 채점) → 기록. `score_text` 를 감싼다."""
    return score_text(text, cond, corr=grade_final(text or "", gold), truncated=truncated,
                      tokens=tokens, problem=problem)


def _p(vals: Sequence[float]) -> float:
    v = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return (sum(v) / len(v)) if v else _NAN


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def paired_delta(recs: Sequence[dict], a: str, b: str, *, seed: int, n_boot: int) -> tuple:
    """(a − b) 짝지은 Δ — **그 쌍을 둘 다 가진 단위만** 쓴다.
    ★구 게이트(math_meta_content_gate)는 «모든 조건이 다 있는 단위»만 요약에 넣어서, 한 조건이
      길이 초과로 빠지면 그 문제가 **모든 비교에서** 사라졌다(표본이 조용히 줄고, 어떤 쌍은
      멀쩡했는데도 못 쓰였다). 여기서는 쌍마다 따로 짝을 짓는다."""
    d = [r["cond"][a]["acc"] - r["cond"][b]["acc"]
         for r in recs if a in r["cond"] and b in r["cond"]]
    return bootstrap_ci(d, seed=seed, n_boot=n_boot), sign_test_p(d), len(d)


def summarize(recs: Sequence[dict], conds: Sequence[str], *, k: int = 0, seed: int = 0,
              n_boot: int = 2000, stage: str = "screen") -> dict:
    """단위(문제)별 기록 → 조건별 지표 + 짝지은 Δ(vs tmpl_null, vs plain) + 통과 판정.
    recs[i]["cond"][c] = {"acc","trunc","tokens","performed","changed","acc_changed","cat"}
    — **조건이 다 있을 필요는 없다**(쌍마다 짝짓는다, `paired_delta` 참조)."""
    acc, trunc, tokens, perf, chg, acc_chg, cats, n_c = {}, {}, {}, {}, {}, {}, {}, {}
    d_ctl, p_ctl, n_ctl, d_ref, p_ref = {}, {}, {}, {}, {}
    d_rr, p_rr, n_rr = {}, {}, {}
    slot, comp, compl, rmatch, rinprob, verd, revise, acc_rev, tsan = (
        {}, {}, {}, {}, {}, {}, {}, {}, {})
    for i, c in enumerate(conds):
        have = [r for r in recs if c in r["cond"]]
        n_c[c] = len(have)
        # 진단 열 — 수행이 어디서 깨지는가(본 섹션은 썼는데 슬롯을 안 채웠는가 등).
        for dst, key in ((slot, "sec_slot"), (comp, "sec_compare"), (compl, "compliant"),
                         (rmatch, "recovered_matches_stated"),
                         (rinprob, "recovered_in_problem"), (revise, "revised"),
                         (acc_rev, "acc_revised"), (tsan, "performed_tmpl_sanity")):
            dst[c] = _p([r["cond"][c].get(key, _NAN) for r in have])
        verd[c] = {v: _p([r["cond"][c].get("verdict_hist", {}).get(v, _NAN) for r in have])
                   for v in VERDICTS}
        d_rr[c], p_rr[c], n_rr[c] = paired_delta(recs, c, CONTROL2, seed=seed + 300 + i,
                                                 n_boot=n_boot)
        acc[c] = bootstrap_ci([r["cond"][c]["acc"] for r in have], seed=seed + i, n_boot=n_boot)
        trunc[c] = _p([r["cond"][c]["trunc"] for r in have])
        tokens[c] = _p([r["cond"][c]["tokens"] for r in have])
        perf[c] = _p([r["cond"][c]["performed"] for r in have])
        chg[c] = _p([r["cond"][c]["changed"] for r in have])
        # acc|changed — 답을 바꾼 생성만 놓고 본 정확도. 이게 낮으면 검산은 «고치되 틀리게
        # 고친다»는 뜻이라, Δ≈0 의 해석이 «무용»에서 «해롭게 흔든다»로 바뀐다.
        acc_chg[c] = _p([r["cond"][c].get("acc_changed", _NAN) for r in have])
        cats[c] = {g: _p([r["cond"][c]["cat"].get(g, _NAN) for r in have]) for g in CATEGORIES}
        d_ctl[c], p_ctl[c], n_ctl[c] = paired_delta(recs, c, CONTROL, seed=seed + 100 + i,
                                                    n_boot=n_boot)
        d_ref[c], p_ref[c], _ = paired_delta(recs, c, REF, seed=seed + 200 + i, n_boot=n_boot)
    out = {
        "n_units": len(recs), "k": k, "ref": REF, "control": CONTROL, "control2": CONTROL2,
        "stage": stage, "conds": list(conds),
        "n_units_per_cond": n_c, "n_pairs_vs_control": n_ctl, "n_pairs_vs_reread": n_rr,
        "acc": acc, "trunc_rate": trunc, "resp_tokens": tokens, "performed_rate": perf,
        "changed_rate": chg, "acc_given_changed": acc_chg, "category_hist": cats,
        "slot_rate": slot, "compare_rate": comp, "compliance_rate": compl,
        "recovered_match_rate": rmatch,
        "recovered_in_problem_rate": rinprob, "verdict_hist": verd,
        "revise_rate": revise, "acc_given_revise": acc_rev,
        "performed_tmpl_sanity": tsan,
        "delta_vs_control": d_ctl, "sign_p_vs_control": p_ctl,
        "delta_vs_reread": d_rr, "sign_p_vs_reread": p_rr,
        "delta_vs_ref": d_ref, "sign_p_vs_ref": p_ref,
        # (act − tmpl_null) 내림차순. 대조군은 승자 후보가 아니다.
        "acts_ranked": sorted(
            [c for c in conds if VERIFY_CONDS[c].act is not None],
            key=lambda c: -d_ctl[c]["mean"] if _finite(d_ctl[c], "mean") else math.inf),
        "budget_flag": [c for c in conds
                        if isinstance(trunc[c], float) and math.isfinite(trunc[c])
                        and trunc[c] > MAX_TRUNC],
        "perform_low": [c for c in conds if VERIFY_CONDS[c].act is not None
                        and not (isinstance(perf[c], float) and math.isfinite(perf[c])
                                 and perf[c] >= MIN_PERFORMED)],
    }
    out["reread_confound"] = reread_confound_flags(out)
    ok, winners = perform_causal_pass(out, stage=stage)
    out["pass_perform_causal"] = int(ok)
    out["winners"] = winners
    return out


def reread_confound_flags(summ: dict) -> list[str]:
    """`[REREAD-CONFOUND?]` — (act − tmpl_null) 평균 > 0 인데 (act − reread_null) 평균 ≤ 0 인
    행위. 즉 «검산이 도왔다»로 읽히던 이득이 **되읽기 대조 앞에서 사라진다**는 뜻이다.
    ★0915 재분석이 이 플래그를 만든 이유: backward_tmpl 의 +.044 중 85%가 답을 안 바꾼 행에서
      왔고 그 'Recovered:' 값은 대개 문제가 이미 준 수였다. 그 패턴의 자연스러운 설명이
      «답을 쓴 뒤 문제를 다시 읽었다»이고, tmpl_null 은 그것을 통제하지 못한다.
    ★nan 은 플래그를 **안 세운다**(대조가 없는 실행에서 헛경보를 내지 않는다)."""
    d1 = summ.get("delta_vs_control") or {}
    d2 = summ.get("delta_vs_reread") or {}
    out: list[str] = []
    for c in summ.get("conds") or []:
        if VERIFY_CONDS.get(c) is None or VERIFY_CONDS[c].act is None:
            continue
        a, b = d1.get(c) or {}, d2.get(c) or {}
        if _finite(a, "mean") and _finite(b, "mean") and a["mean"] > 0 and b["mean"] <= 0:
            out.append(c)
    return out


def perform_causal_pass(summ: dict, *, stage: str = "screen") -> tuple[bool, list[str]]:
    """PERFORM-CAUSAL — 어떤 `*_tmpl` 이 (act − tmpl_null) CI 가 0 제외 ∧ 평균 ≥ +0.03 ∧
    performed_rate ≥ .60 ∧ trunc_rate ≤ .20 이면 통과. nan 은 전부 FAIL 쪽이다.
    ★수행률을 규칙에 넣는 이유가 이 실험의 전부다: 수행이 안 일어났으면 «검산의 인과 효과»를
      잰 게 아니라 «또 말만 하게 한 것»을 잰 것이고, 그때의 Δ 는 어느 쪽으로도 못 읽는다.
    ★`stage="confirm"` 이면 **두 번째 대조를 추가로 넘어야 한다**: (act − reread_null) 의 CI 가
      0 을 제외하고 평균 ≥ +0.03. 되읽기 대조를 못 넘는 이득은 «검산의 효과»라고 부를 수 없다
      — screen 단계의 승자가 여기서 떨어지는 것이 이 확인 실험의 정상적인 결과다."""
    def _num(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    if stage not in STAGES:
        raise SystemExit(f"[verify] 모르는 stage: {stage!r} (아는 것: {STAGES})")
    winners: list[str] = []
    for c in summ.get("acts_ranked") or []:
        d = (summ.get("delta_vs_control") or {}).get(c) or {}
        pf = (summ.get("performed_rate") or {}).get(c, _NAN)
        tr = (summ.get("trunc_rate") or {}).get(c, _NAN)
        if not (_finite(d, "lo", "hi", "mean") and (d["lo"] > 0 or d["hi"] < 0)
                and d["mean"] >= PASS_DELTA and _num(pf) and pf >= MIN_PERFORMED
                and _num(tr) and tr <= MAX_TRUNC):
            continue
        if stage == "confirm":
            r = (summ.get("delta_vs_reread") or {}).get(c) or {}
            if not (_finite(r, "lo", "hi", "mean") and (r["lo"] > 0 or r["hi"] < 0)
                    and r["mean"] >= PASS_DELTA_REREAD):
                continue
        winners.append(c)
    return bool(winners), winners


def _f(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}]"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def to_markdown(summ: dict, *, title: str = "pooled") -> str:
    ref, ctl = summ["ref"], summ["control"]
    rr = summ.get("control2", CONTROL2)
    lines = [f"### {title} — n_units={summ['n_units']} · K={summ['k']} · "
             f"stage={summ.get('stage')}", "",
             f"| cond | n | acc | Δ vs {ctl} | sign_p({ctl}) | Δ vs {rr} | sign_p({rr}) | "
             f"Δ vs {ref} | compliance | performed | trunc | tokens | changed | acc\\|changed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in summ["conds"]:
        lines.append(
            f"| {c} | {summ['n_units_per_cond'][c]} | {_f(summ['acc'][c])} | "
            f"{_f(summ['delta_vs_control'][c])} | {_f(summ['sign_p_vs_control'][c])} | "
            f"{_f(summ['delta_vs_reread'][c])} | {_f(summ['sign_p_vs_reread'][c])} | "
            f"{_f(summ['delta_vs_ref'][c])} | {_f(summ['compliance_rate'][c])} | "
            f"{_f(summ['performed_rate'][c])} | "
            f"{_f(summ['trunc_rate'][c])} | {_f(summ['resp_tokens'][c])} | "
            f"{_f(summ['changed_rate'][c])} | {_f(summ['acc_given_changed'][c])} |")
    # 수행이 **어디서** 깨지는가 + 되읽기 의심 신호(backward 의 'Recovered:' 가 문제에 이미
    # 있던 수인가). 이 두 열이 Δ 의 해석을 가른다.
    lines += ["", "**수행 해부**(템플릿 조건은 template 탐지기, 대조군은 free-text 탐지기 + "
              "tmpl_sanity(0 이어야 정상))", "",
              "| cond | slot | compare | verdict same/diff/none | revise | acc\\|revise | "
              "rec=stated | rec∈problem | tmpl_sanity |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in summ["conds"]:
        vh = summ["verdict_hist"][c]
        lines.append(
            f"| {c} | {_f(summ['slot_rate'][c])} | {_f(summ['compare_rate'][c])} | "
            f"{_f(vh['same'])}/{_f(vh['different'])}/{_f(vh['none'])} | "
            f"{_f(summ['revise_rate'][c])} | {_f(summ['acc_given_revise'][c])} | "
            f"{_f(summ['recovered_match_rate'][c])} | "
            f"{_f(summ['recovered_in_problem_rate'][c])} | "
            f"{_f(summ['performed_tmpl_sanity'][c])} |")
    lines += ["", "**범주 히스토그램**(confirm_right / confirm_wrong / revise_right / "
              "revise_wrong / none) — Δ≈0 이어도 revise_right ≈ revise_wrong 이면 결론은 "
              "«검산 무용»이 아니라 «방향 없이 흔든다»다.", "",
              "| cond | " + " | ".join(CATEGORIES) + " |",
              "|---|" + "---|" * len(CATEGORIES)]
    for c in summ["conds"]:
        lines.append(f"| {c} | " + " | ".join(_f(summ["category_hist"][c][g])
                                              for g in CATEGORIES) + " |")
    lines += ["", f"**RANK (act − {ctl})**: " + " > ".join(
        f"{c}({_f(summ['delta_vs_control'][c].get('mean'))})" for c in summ["acts_ranked"]), ""]
    extra = (f" ∧ **(act − {rr}) CI 가 0 제외 ∧ 평균 ≥ +{PASS_DELTA_REREAD}**"
             if summ.get("stage") == "confirm" else "")
    lines.append(
        f"**PERFORM-CAUSAL {'PASS' if summ.get('pass_perform_causal') else 'FAIL'}** "
        f"(stage={summ.get('stage')}) — (act − {ctl}) CI 가 0 제외 ∧ 평균 ≥ +{PASS_DELTA} ∧ "
        f"performed_rate ≥ {MIN_PERFORMED} ∧ trunc_rate ≤ {MAX_TRUNC}{extra}; "
        f"winners={summ.get('winners')}")
    if summ.get("reread_confound"):
        lines.append(
            f"**[REREAD-CONFOUND?]** {summ['reread_confound']} — (act − {ctl}) > 0 인데 "
            f"(act − {rr}) ≤ 0 이다. 그 이득은 «검산»이 아니라 «답 쓴 뒤 문제를 다시 읽은 것»과 "
            "구분되지 않는다. 이 행을 승자로 부르지 말 것.")
    if summ.get("budget_flag"):
        lines.append(f"**[BUDGET?]** trunc_rate > {MAX_TRUNC} 인 조건: {summ['budget_flag']} — "
                     "cd9 G4 처럼 절단률이 정확도 순서를 뒤집을 수 있다. 이 표를 그대로 읽지 말 것.")
    if summ.get("perform_low"):
        lines.append(f"**[PERFORM-LOW]** performed_rate < {MIN_PERFORMED} 인 행위: "
                     f"{summ['perform_low']} — 수행이 안 일어났으므로 그 Δ 는 **해석 불가**다"
                     "(«검산이 무용»이 아니라 «또 말만 했다»와 구분되지 않는다).")
    lines.append("**[PAL] 말한 검산 ≠ 수행한 검산** (arXiv:2211.10435) — 이 게이트의 존재 이유가 "
                 "그 구분이다. performed 는 verify_terms 의 «계산줄 ∧ 판정 진술» 정의로만 잰다.")
    return "\n".join(lines + [""])


def render_report(pooled: dict, per_source: dict, *, meta: dict) -> str:
    lines = ["## math_verify_perform_gate — V1 «검산 **수행**의 외생화»", "",
             f"model={meta.get('model_path')} · variant={meta.get('variant')} · "
             f"seed={meta.get('seed')} · max_tokens={meta.get('max_tokens')} · "
             f"sources={meta.get('sources')} · n_generations={meta.get('n_generations')}", "",
             to_markdown(pooled, title="pooled")]
    lines += [to_markdown(s, title=f"source={t}") for t, s in per_source.items()]
    return "\n".join(lines)


def source_tag(path: str) -> str:
    """출처 표식 = 롤아웃 파일의 상위 디렉터리 이름. ★두 코퍼스가 group_id 를 «g0»부터 각자
    붙이므로 이 표식으로 네임스페이스를 안 나누면 서로 다른 문제가 한 단위로 뭉개진다."""
    return Path(path).resolve().parent.name or Path(path).name


def parse_conds(spec: str | None) -> tuple[str, ...]:
    """`--conds` 쉼표 목록 → 조건 이름(빈 값이면 전부). **정의 순서**로 되돌려 준다 —
    표·시드 오프셋이 순서에 걸려 있어서 호출자가 순서를 흔들면 재현이 깨진다.
    ★모르는 이름은 fail-loud. 오타 하나가 조건을 조용히 빼는 것이 이 게이트의 최악 사고다."""
    if not spec or not spec.strip():
        return conds_all()
    want = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [w for w in want if w not in VERIFY_CONDS]
    if bad:
        raise SystemExit(f"[verify] 모르는 조건: {bad} (아는 것: {list(VERIFY_CONDS)})")
    return tuple(c for c in conds_all() if c in set(want))


def load_excluded_units(path: str | None) -> set[str]:
    """이전 실행의 `per_problem.jsonl` → 이미 쓴 unit_id 집합.
    ★확인 실험은 **안 쓴 문제**에서 돌아야 한다. screen 에서 승자를 고른 그 문제로 다시 재면
      그 선택의 잡음을 그대로 확증하게 된다(cd9 가 반복한 함정)."""
    if not path:
        return set()
    out: set[str] = set()
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            u = (json.loads(ln) or {}).get("unit_id")
            if u:
                out.add(u)
    return out


def resummarize_gens(gens_path: str, *, out_dir: str, seed: int = 11, n_boot: int = 2000,
                     stage: str = "screen", conds_spec: str = "") -> int:
    """`--resummarize` — 이미 있는 `gens.jsonl`(생성된 text)에서 **CURRENT** 탐지기·stage
    규칙으로 `per_problem.jsonl`/`gate_summary.{json,md}` 를 다시 만든다. GPU·생성 없음.
    ★재요약이지 재생성이 아니다: 정오(`r_corr`)는 gens.jsonl 에 이미 채점되어 저장된 값을
      그대로 쓴다(gens.jsonl 에 gold·problem 원문이 없다 — 채점 로직은 이 감사의 대상이
      아니다). 다시 재는 것은 **수행·판정·범주**뿐이다(`score_text` 를 `text` 에 다시 태운다).
    ★`problem` 이 없으므로 `recovered_in_problem` 은 못 잰다 — 그 열은 이 모드에서 항상
      nan 이다(원 실행의 gate_summary 와 나란히 놓을 때 이 결손을 명시할 것)."""
    gens: list[dict] = [json.loads(ln) for ln in open(gens_path) if ln.strip()]
    if not gens:
        raise SystemExit(f"[verify] 빈 gens 파일: {gens_path}")
    conds = parse_conds(conds_spec)
    present = {g["cond"] for g in gens}
    conds = tuple(c for c in conds if c in present)

    agg: dict = {}
    unit_meta: dict[str, dict] = {}
    for g in gens:
        uid, c = g["unit_id"], g["cond"]
        if c not in VERIFY_CONDS:
            continue     # ★모르는 조건은 조용히 건너뛴다 — gens 는 옛 조건 이름을 담을 수 있다
        rec = score_text(g.get("text", ""), c, corr=g.get("r_corr", 0),
                         truncated=g.get("trunc", 0), tokens=g.get("tokens", 0),
                         problem=g.get("problem", ""))
        agg.setdefault((uid, c), []).append(rec)
        unit_meta.setdefault(uid, {"unit_id": uid, "tag": g.get("tag", ""),
                                   "group_id": uid.split("::")[-1], "gold": ""})

    recs = []
    for uid, meta in unit_meta.items():
        cond_map = {}
        for c in conds:
            got = agg.get((uid, c))
            if not got:
                continue
            changed = [g["r_corr"] for g in got if g["changed"]]
            revised = [g["r_corr"] for g in got if g.get("revised")]
            cond_map[c] = {
                "acc": _p([g["r_corr"] for g in got]),
                "trunc": _p([g["trunc"] for g in got]),
                "tokens": _p([g["tokens"] for g in got]),
                "performed": _p([g["performed"] for g in got]),
                "changed": _p([g["changed"] for g in got]),
                "acc_changed": (_p(changed) if changed else _NAN),
                "revised": _p([g.get("revised", _NAN) for g in got]),
                "acc_revised": (_p(revised) if revised else _NAN),
                **{key: _p([g.get(key, _NAN) for g in got])
                   for key in ("sec_slot", "sec_compare", "compliant",
                               "recovered_matches_stated",
                               "recovered_in_problem", "performed_tmpl_sanity")},
                "verdict_hist": {v: _p([1.0 if (g.get("verdict") or "none") == v else 0.0
                                        for g in got]) for v in VERDICTS},
                "cat": {gname: _p([1.0 if g["category"] == gname else 0.0 for g in got])
                        for gname in CATEGORIES}}
        if cond_map:
            recs.append({**meta, "cond": cond_map})

    tags = sorted({r["tag"] for r in recs})
    k = max((len(v) for v in agg.values()), default=0)
    pooled = summarize(recs, conds, k=k, seed=seed, n_boot=n_boot, stage=stage)
    per_source = {t: summarize([r for r in recs if r["tag"] == t], conds, k=k, seed=seed,
                               n_boot=n_boot, stage=stage)
                  for t in tags if any(r["tag"] == t for r in recs)}
    meta = {"resummarize_from": str(gens_path), "seed": seed, "k": k, "stage": stage,
            "conds_selected": list(conds), "n_units_scored": len(recs),
            "n_generations": len(gens),
            "sources": {t: len([r for r in recs if r["tag"] == t]) for t in tags},
            "conds": {n: v._asdict() for n, v in VERIFY_CONDS.items()},
            "note": "recovered_in_problem 은 gens.jsonl 에 problem 원문이 없어 nan 이다."}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_problem.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(
        {"pooled": pooled, "per_source": per_source, "meta": meta}, ensure_ascii=False, indent=2))
    md = render_report(pooled, per_source, meta=meta)
    (out / "gate_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


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
    ap.add_argument("--resummarize", default="",
                    help="생성 없이(GPU 없이) 이 gens.jsonl 을 CURRENT 탐지기로 다시 요약해 "
                         "--out_dir 에 쓴다. 주면 --rollouts/--model_path 는 무시된다.")
    ap.add_argument("--rollouts", action="append", default=[],
                    help="math_rollout 산출물 texts.jsonl (여러 번 — 출처별 표도 찍는다)")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_problems", type=int, default=120, help="출처에 고르게 배분")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=12288,
                    help="★8k 에서 recompute 절단이 .313 이었다. 내리지 말 것.")
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--conds", default="",
                    help="쉼표 목록(기본 전부). 예: tmpl_null,reread_null,backward_tmpl")
    ap.add_argument("--exclude_units_from", default="",
                    help="이전 실행의 per_problem.jsonl — 거기 나온 unit_id 는 제외한다"
                         "(확인 실험은 **안 쓴 문제**에서 돈다)")
    ap.add_argument("--stage", choices=list(STAGES), default="screen",
                    help="confirm 이면 (act − reread_null) 관문이 통과 규칙에 추가된다")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.resummarize:
        return resummarize_gens(a.resummarize, out_dir=a.out_dir, seed=a.seed,
                                n_boot=a.n_boot, stage=a.stage, conds_spec=a.conds)
    if not a.rollouts or not a.model_path:
        raise SystemExit("[verify] --resummarize 가 없으면 --rollouts 와 --model_path 가 필수다")

    selftest_math_verify()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs = load_sources(a.rollouts)
    conds = parse_conds(a.conds)
    used = load_excluded_units(a.exclude_units_from)
    if used:
        print(f"[verify] 제외 단위 {len(used)}개 ← {a.exclude_units_from}", flush=True)

    units: list[dict] = []
    cap = max(1, a.max_problems // max(1, len(srcs)))
    for tag, rolls in srcs:
        pool = [{"unit_id": f"{tag}::{g['group_id']}", "tag": tag, **g}
                for g in select_mixed_problems(rolls, max_problems=0)]
        n_all = len(pool)
        pool = [u for u in pool if u["unit_id"] not in used][:cap]
        units += pool
        print(f"[verify] {tag}: 문제 {len(pool)}개 (MIXED {n_all}개 중, 출처 상한 {cap}, "
              f"제외 {len(used)}개 적용)", flush=True)
    if not units:
        raise SystemExit("[verify] 후보가 없다 — 입력 롤아웃에 MIXED 그룹이 있는지 확인하라.")
    print(f"[verify] 단위 {len(units)} x 조건 {len(conds)} x K={a.k} = "
          f"{len(units) * len(conds) * a.k}개 생성 예정", flush=True)

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util, max_model_len=a.max_tokens + 2048,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    lim = 2048 - 256   # 프롬프트 한도(max_model_len − max_tokens 안에 들어가야 한다)

    reqs, ix, n_drop = [], [], 0
    for ui, u in enumerate(units):
        for c in conds:
            q = cond_prompt(tok, a.variant, u["problem"], c)
            if len(tok.encode(q)) > lim:
                n_drop += 1
                continue     # ★그 (단위, 조건) 만 빠진다 — 단위 전체를 버리지 않는다
            reqs.append(q)
            ix.append((ui, c))
    print(f"[verify] 요청 {len(reqs)}개 (버린 것 {n_drop}개, 프롬프트 한도 {lim} 토큰)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    gens = []
    for (ui, c), o in zip(ix, outs):
        u = units[ui]
        for x in o.outputs:
            rec = gen_record(x.text, c, u["gold"], truncated=int(x.finish_reason == "length"),
                             tokens=len(x.token_ids), problem=u["problem"])
            agg.setdefault((ui, c), []).append(rec)
            gens.append({"unit_id": u["unit_id"], "tag": u["tag"], "cond": c, "text": x.text,
                         **{k2: v for k2, v in rec.items() if k2 != "cat"}})
    with (out / "gens.jsonl").open("w") as fh:
        for r in gens:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    recs = []
    for ui, u in enumerate(units):
        cond_map = {}
        for c in conds:
            got = agg.get((ui, c))
            if not got:
                continue     # ★쌍마다 짝짓는다 — 이 조건이 없어도 나머지 쌍은 살아 있다
            changed = [g["r_corr"] for g in got if g["changed"]]
            revised = [g["r_corr"] for g in got if g.get("revised")]
            cond_map[c] = {
                "acc": _p([g["r_corr"] for g in got]),
                "trunc": _p([g["trunc"] for g in got]),
                "tokens": _p([g["tokens"] for g in got]),
                "performed": _p([g["performed"] for g in got]),
                "changed": _p([g["changed"] for g in got]),
                "acc_changed": (_p(changed) if changed else _NAN),
                "revised": _p([g.get("revised", _NAN) for g in got]),
                "acc_revised": (_p(revised) if revised else _NAN),
                # 진단 열 — 없는 조건은 nan 으로 남는다(_p 가 유한값만 센다).
                **{key: _p([g.get(key, _NAN) for g in got])
                   for key in ("sec_slot", "sec_compare", "compliant",
                               "recovered_matches_stated",
                               "recovered_in_problem", "performed_tmpl_sanity")},
                "verdict_hist": {v: _p([1.0 if (g.get("verdict") or "none") == v else 0.0
                                        for g in got]) for v in VERDICTS},
                "cat": {gname: _p([1.0 if g["category"] == gname else 0.0 for g in got])
                        for gname in CATEGORIES}}
        if cond_map:
            recs.append({"unit_id": u["unit_id"], "tag": u["tag"], "group_id": u["group_id"],
                         "gold": u["gold"], "cond": cond_map})

    pooled = summarize(recs, conds, k=a.k, seed=a.seed, n_boot=a.n_boot, stage=a.stage)
    per_source = {t: summarize([r for r in recs if r["tag"] == t], conds, k=a.k, seed=a.seed,
                               n_boot=a.n_boot, stage=a.stage)
                  for t, _ in srcs if any(r["tag"] == t for r in recs)}
    meta = {"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
            "seed": a.seed, "max_tokens": a.max_tokens, "k": a.k, "stage": a.stage,
            "conds_selected": list(conds), "exclude_units_from": a.exclude_units_from,
            "n_units_excluded": len(used),
            "sources": {t: len([r for r in recs if r["tag"] == t]) for t, _ in srcs},
            "n_units_selected": len(units), "n_units_scored": len(recs),
            "n_dropped_long": n_drop, "n_generations": len(gens),
            "conds": {n: v._asdict() for n, v in VERIFY_CONDS.items()}}

    with (out / "per_problem.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(
        {"pooled": pooled, "per_source": per_source, "meta": meta}, ensure_ascii=False, indent=2))
    md = render_report(pooled, per_source, meta=meta)
    (out / "gate_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
