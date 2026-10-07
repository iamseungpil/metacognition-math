r"""크레딧 — 순수 함수만(torch 없음). H4 형제 다수결 라벨 · 근사오답(후보 집합) · CH-Fork 토큰 가중치(표준화·방향 비례)."""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence


def _num_eq(a: str, b: str) -> bool:
    """순수 숫자 후보끼리만 비교하는 보수적 필터(LaTeX 는 파싱하지 않는다)."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, TypeError):
        return False


def near_miss_decoy(gold: str, seed: int = 17, checker=None) -> str:
    r"""X ≡ gold 인 행의 A− — **규칙 기반 근사오답**. A+ ≡ A− 면 PMI 신호가 구조적으로 0 이다.

    ★`src/training/_decoy_utils._rule_based_decoy` 를 옮긴 것. 보장: ①gold 와 문자열 다름
    ②숫자면 수치로도 다름 ③(gold, seed)에 **결정적**(md5) ④checker 가 동치로 보는 후보는
    버린다. 전략: 정수·실수 ±, LaTeX 상수, 분수 뒤집기·분자 ±1, 부호 반전, 폴백 `+1`."""
    if checker is None:
        from mc.grade import answers_equivalent      # noqa: PLC0415
        checker = answers_equivalent
    rng = random.Random(int(hashlib.md5(f"{gold}|{seed}".encode()).hexdigest()[:8], 16))
    s = str(gold).strip()
    cand: list[str] = []
    if re.fullmatch(r"-?\d+", s):
        n = int(s)
        cand += [str(n + d) for d in (1, -1, 2, -2, 10, -10, 5, -5)]
    elif re.fullmatch(r"-?\d+\.\d*|-?\.\d+|-?\d+\.?", s):
        try:
            v = float(s)
            cand += [str(round(v + d, 2)) for d in (0.1, -0.1, 1.0, -1.0, 0.5, -0.5)]
        except ValueError:
            pass
    if "\\pi" in s:
        cand += [s.replace("\\pi", ""), s.replace("\\pi", "\\pi/2")]
    if "\\sqrt" in s:
        cand.append(re.sub(r"\\sqrt\{(\d+)\}", r"\1", s))
    m = re.match(r"\\?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and m.group(1) != m.group(2):
        cand.append(f"\\frac{{{m.group(2)}}}{{{m.group(1)}}}")
        cand.append(f"\\frac{{{int(m.group(1)) + 1}}}{{{m.group(2)}}}")
    if s.startswith("-") and s != "-0":
        cand.append(s[1:])
    elif s not in {"0", "0.0", "-0", "-0.0"}:
        cand.append("-" + s)
    valid = [c for c in cand if c != s and c.strip() and not _num_eq(c, s)]
    try:
        valid = [c for c in valid if not checker(c, s)]
    except Exception:
        pass
    if valid:
        return rng.choice(valid)
    fb = s + " + 1"                                   # 기호적 폴백(sympy 로도 gold ≠ fb)
    try:
        return fb if not checker(fb, s) else s + " + 1000000"
    except Exception:
        return fb


def majority_label(answers: Sequence[str], self_idx: int | None = None) -> str | None:
    r"""**leave-one-out** 형제 다수답(H4 의 gold 대체). 동률·무답이면 None(라벨 없음).

    `self_idx` 를 주면 그 행을 뺀 나머지로만 센다 — 자기 표가 캐스팅보트가 되는 자기충족
    채널을 막는다. 답 있는 형제가 2 미만이면 None.
    """
    from mc.grade import answers_equivalent  # noqa: PLC0415
    rest = [a for i, a in enumerate(answers) if self_idx is None or i != self_idx]
    xs = [str(a or "").strip() for a in rest]
    xs = [a for a in xs if a]
    if len(xs) < 2:
        return None
    clusters: list[list[str]] = []
    for a in xs:
        for c in clusters:
            if answers_equivalent(c[0], a):
                c.append(a)
                break
        else:
            clusters.append([a])
    top = max(len(c) for c in clusters)
    best = [c for c in clusters if len(c) == top]
    return best[0][0] if len(best) == 1 else None


def label_correct(answer: str, *, gold: str = "", label: str = "gold",
                  sib_answers: Sequence[str] | None = None,
                  self_idx: int | None = None, options: dict | None = None) -> float | None:
    """R = 1[answer 가 라벨과 같다]. `label="majority"` 면 LOO 형제 다수답이 라벨이다.
    라벨이 없으면(동률·무답) None — 그 행은 보상 항이 **정의되지 않는다**. `options`(객관식 선택지,
    `mc.grade.parse_options`)가 있으면 글자 답을 선택지 값으로 바꿔 채점한다."""
    from mc.grade import choice_value, grade_answer  # noqa: PLC0415
    a = str(answer or "").strip()
    if label == "gold":
        return float(grade_answer(a, gold, options)) if a else 0.0
    if label != "majority":
        raise ValueError(f"[MC] label={label!r} 은 gold|majority 중 하나여야 한다.")
    a = choice_value(a, options)
    mj = majority_label([choice_value(x, options) for x in sib_answers or []], self_idx=self_idx)
    if mj is None:
        return None
    from mc.grade import answers_equivalent  # noqa: PLC0415
    return float(bool(a) and answers_equivalent(mj, a))


def fork_weights(scores, sign: float, lam: float = 1.0, cap: float = 3.0, min_n: int = 16) -> list[float]:
    r"""수정 28 CH-Fork — 행 안 토큰 가중(평균 1 = 행 총량 보존). z = 행 안 표준화(sign·대조 PMI), w = 1 + λ·clip(z, 0,
    cap) 를 평균 1 로 나눔: 결과 방향(성공 +, 실패 −)으로 성공·실패 이웃을 가른 토큰에 행의 결과 adv 를 몰아준다.
    분산 0 이거나 min_n 토큰 미만이면 균등(= B; 몇 토큰짜리 z 는 잡음)."""
    x = [sign * float(v) for v in scores]
    n = len(x)
    if n < max(2, min_n):
        return [1.0] * n
    m = sum(x) / n
    sd = (sum((v - m) ** 2 for v in x) / n) ** 0.5
    if sd <= 1e-12:
        return [1.0] * n
    w = [1.0 + lam * min(max((v - m) / sd, 0.0), cap) for v in x]
    mw = sum(w) / n
    return [v / mw for v in w]

