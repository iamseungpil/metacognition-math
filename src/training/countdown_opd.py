r"""OPD — 힌트 교사 (메타 구간 on-policy distillation). 설계: docs/DESIGN_opd_hint_teacher.md.

이 파일은 **순수 함수만** 담는다(countdown_rewards.py 와 같은 규약 — torch·verl 의존 0,
CPU 로 전부 테스트 가능). 실제 teacher forward(GPU 필요)는 `src/training/verl_sdc.py::
_compute_countdown_opd` 가 한다: 그 함수가 여기 네 함수를 호출해 (1) 힌트 문자열을
만들고, (2) 힌트-조건 채팅 메시지를 조립하고, (3) KL 을 잴 문자 구간을 정하고,
(4) 잰 KL 을 보상으로 바꾼다.

왜 별도 모듈인가 (countdown_rewards.py 에 안 넣은 이유). `TERMS`/`ARM_SPECS`/`arm_reward`
는 countdown_rewards.py 에 그대로 둔다(그 파일이 "팔 정체의 단일 정의처"라는 규약을
지키기 위해서다 — `r_opd_meta` 는 그 파일에 추가한다). 이 파일은 힌트·삽입·스팬처럼
`countdown_sites`/`countdown_selfcontrol` 을 끌어와야 하는 **테스트 대상 로직**만 묶은
자리다 — countdown_rewards.py 가 countdown_sites 를 이미 알듯(FT/MT 의 family_dead
계산 참조), 여기서도 같은 재사용을 하되 앞으로 `verl_sdc.py` 가 부를 표면만 좁힌다.

⚠누출 금지(설계 §1.1): `build_hint` 는 `countdown_sites.oracle_for_site` 의
`family_dead`/`live_new_moves` **두 필드만** 쓴다. `witness`/`decoy`/최종식/
`\boxed{...}`/target 연산 결과는 절대 넣지 않는다.
"""
from __future__ import annotations

from typing import Optional

from src.training import countdown_sites as _cds
from src.training.countdown_rewards import parse_meta
from src.training.countdown_selfcontrol import _ARITH_EQ

__all__ = ["build_hint", "hinted_messages", "opd_spans", "opd_reward"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 힌트 문자열 — 설계 §1.1
# ══════════════════════════════════════════════════════════════════════════════

def build_hint(nums, target, prefix_text: str) -> str:
    r"""§1.1 힌트 문자열 그대로.

        Hint: your current line of attack is {dead|alive}.
        Hint: first moves that still reach the target: {live_list|none found}.

    `family_dead is None`(prefix_text 안에 완결 시도가 아예 없다 — 판정 불가)이면
    힌트 자체를 생략한다("" 를 돌려준다, `countdown_sites.family_dead_label` 의
    "시도가 없으면 None" 규약과 같은 이유).

    ★오라클(`countdown_sites.oracle_for_site`)이 돌려주는 것 중 `family_dead` 와
    `live_new_moves` **두 필드만** 읽는다. `witness`(정답 식)는 이 함수의 반환값
    어디에도 등장하지 않는다 — 그것이 이 항이 T1(도치 자, 정답 힌트)과 구조적으로
    다른 지점이다(설계 §5).
    """
    oracle = _cds.oracle_for_site(prefix_text, nums, target)
    fd = oracle["family_dead"]
    if fd is None:
        return ""
    dead_or_alive = "dead" if int(fd) else "alive"
    live = oracle["live_new_moves"]
    live_list = ", ".join(live) if live else "none found"
    return (f"Hint: your current line of attack is {dead_or_alive}.\n"
            f"Hint: first moves that still reach the target: {live_list}.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 힌트 삽입 — 설계 §1.2 (프리픽스 바로 앞, user 메시지의 연장)
# ══════════════════════════════════════════════════════════════════════════════

def hinted_messages(prompt_messages: list[dict], hint: str, prefix: str) -> list[dict]:
    r"""§1.2. `hint` 를 **마지막 user 메시지**의 끝에 이어붙이고(빈 문자열이면 user
    메시지를 손대지 않는다), 그 뒤에 `assistant` 메시지로 `prefix` 를 추가한다.

    `countdown_sites.render_prefix_prompt` 와 같은 뼈대(프롬프트 뒤 assistant 프리픽스
    이어붙이기)에 힌트 삽입 한 단계만 더한 것이다 — 그 함수를 복제하지 않고 여기서
    새로 짜는 이유는 힌트가 **user 쪽**에 들어가 그 함수의 "system+user 그대로, 뒤에
    assistant 만 붙인다"는 계약과 다르기 때문이다.

    입력 리스트/딕셔너리를 변형하지 않는다(얕은 복사 후 반환).
    """
    msgs = [dict(m) for m in prompt_messages]
    if hint:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                content = msgs[i].get("content") or ""
                msgs[i]["content"] = f"{content}\n\n{hint}" if content else hint
                break
        else:
            raise ValueError(
                "hinted_messages: prompt_messages 에 user 메시지가 없다 — "
                "countdown_task.build_prompt 의 출력이 아닌 것 같다.")
    msgs.append({"role": "assistant", "content": prefix})
    return msgs


# ══════════════════════════════════════════════════════════════════════════════
# 3. KL 스팬 — 설계 §2.1 (META 구간 ∪ 첫 post-meta attempt 줄)
# ══════════════════════════════════════════════════════════════════════════════

def opd_spans(text: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    r"""(meta_start, meta_end, first_attempt_end) 문자 오프셋, 반열린 구간.

    META 구간 = `parse_meta(text,"new")` 의 `[start,end)`(태그 포함 — OSD/inv/
    pmi_shift 자들과 같은 경계 정의, 설계 §2.1-1).

    완결된 메타가 없으면 **셋 다 None**(잴 자리가 없다).

    첫 post-meta attempt 줄 = META `end` 이후 첫 `_ARITH_EQ`(`a op b = c`,
    `countdown_selfcontrol._ARITH_EQ` 그대로 재사용 — 복제 금지 규약) 매치가 속한
    줄의 끝(그 줄 다음 줄바꿈 뒤, 없으면 텍스트 끝)까지. 매치가 없으면
    `first_attempt_end == meta_end` 를 돌려준다 — 즉 두 구간의 합집합이 META 구간과
    바이트 동일해진다(설계 §2.1: "이 구간은 비운다 — 실패가 아니다").
    """
    m = parse_meta(text, "new")
    if not m.get("emitted") or m.get("start") is None:
        return None, None, None
    meta_start, meta_end = int(m["start"]), int(m["end"])
    after = text[meta_end:]
    mv = _ARITH_EQ.search(after)
    if not mv:
        return meta_start, meta_end, meta_end
    match_end = meta_end + mv.end()
    nl = text.find("\n", match_end)
    line_end = nl + 1 if nl != -1 else len(text)
    return meta_start, meta_end, line_end


# ══════════════════════════════════════════════════════════════════════════════
# 4. 보상식 — 설계 §2.2 / §4
# ══════════════════════════════════════════════════════════════════════════════

def opd_reward(mean_kl: Optional[float], C: float) -> float:
    r"""R_opd_meta = −clip(mean_kl, 0, C) / C  ∈ [−1, 0].

    `mean_kl` 이 `None` 이면(잴 스팬이 없었다 — `opd_spans` 가 `(None,None,None)`
    이었거나 union 스팬이 비었다) **0.0**(침묵 — `r_timing` 의 `family_dead is None
    → 0` 과 같은 관례). 음의 `mean_kl`(학생이 이미 힌트-조건 분포보다 확신에 차 있던
    자리)은 0 에서 클립돼 벌이 없다 — 이 항은 **단측**이다(설계 §2.2: "음의 KL 을
    보상으로", 즉 KL 이 작을수록/음수일수록 벌이 없고, 클수록 최대 −1 까지 벌).
    """
    if mean_kl is None:
        return 0.0
    c = float(C)
    if not (c > 0):
        raise ValueError(f"opd_reward: C={C!r} 는 0보다 커야 한다.")
    x = max(0.0, min(float(mean_kl), c))
    return -x / c
