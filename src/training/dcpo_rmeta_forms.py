"""R_meta 공식 아홉 -- 오프라인 스윕과 온라인 학습이 **같은 코드**를 쓴다.

왜 별도 모듈인가. 이 저장소는 "선언된 판정식과 채택된 수치가 불일치"한 전례가 있다
(원장 0726: Stage G 선언대로면 0.000 FAIL, 채택된 per_cell 로는 0.785 통과). 오프라인에서
공식을 고르고 온라인에서 다른 식을 돌리면 같은 사고가 재발한다. 그래서 공식은 여기 한 곳에만
있고, `sweep_rmeta.py`(고르기)와 `verl_sdc.py`(학습)가 둘 다 여기서 import 한다.

torch 를 쓰지 않는다 -- 순수 dict -> float 이므로 어느 환경에서도 import 되고 CPU 로 테스트된다.

원재료(호출자가 채워 넣는다). 전부 롤아웃 하나에 대한 스칼라:

    lp_meta         logp(메타 텍스트 | 프롬프트+메타앞본문)
    n_meta_tok      메타 토큰 수
    pmi_open        메타 **앞** 위치의 gold-decoy logp
    pmi_close       메타 **뒤** 위치
    pmi_ans_real    **답 직전** 위치, 메타 = 이 롤아웃의 것
    pmi_ans_donor   **답 직전** 위치, 메타 = 다른 문제의 것 (길이 맞춤)
    lp_post_real    메타 뒤 본문의 logp, 문맥 = 진짜 메타
    lp_post_donor   **같은 본문**, 문맥 = 도너 메타
    n_post          그 본문의 토큰 수
    meta_len        메타 길이 (음성 대조군 전용)

계열 셋:

  위치 차   pmi_close - pmi_open   -- 현행. 메타를 덧붙이면 믿음이 오르나.
            확신에 찬 문장이면 그냥 오른다(rq3v2g_b4p2 step295: 정형문 +0.289 / 그 외 -0.145,
            R_corr 은 0.864 vs 0.839 로 사실상 같았다).
  문맥 차   real - donor           -- 메타를 바꿔치면 무엇이 무너지나. 문제 정보가 없는 문장은
            남의 메타로 바꿔도 변화가 없으므로 정형문이 0 에 붙는다.
  놀라움    -lp_meta / n_meta_tok  -- RLRT(arXiv 2605.10781) 계열. 사전확률이 낮은 메타.
            정형문은 정의상 최빈 메타라 이 축에서도 0 에 붙는다 -- **다른 이유로 같은 결론**.
"""
from __future__ import annotations

from typing import Callable, Mapping

__all__ = ["FORMULAS", "rmeta_from_ingredients", "INGREDIENT_KEYS"]

INGREDIENT_KEYS = (
    "lp_meta", "n_meta_tok", "pmi_open", "pmi_close",
    "pmi_ans_real", "pmi_ans_donor", "lp_post_real", "lp_post_donor",
    "n_post", "meta_len",
)


def _clip(x: float, c: float = 2.0) -> float:
    return max(-c, min(c, float(x)))


def _rev(o: float, c: float, save: float = 1.0, derail: float = 2.0,
         eps: float = 0.0) -> float:
    """부호 뒤집기 보너스. eps 밖에서 양쪽 다 넘어야 세므로 0 근처 잔떨림은 안 센다."""
    if o < -eps and c > eps:
        return abs(save)          # 오답쪽 -> 정답쪽 = SAVE
    if o > eps and c < -eps:
        return -abs(derail)       # 정답쪽 -> 오답쪽 = DERAIL
    return 0.0


def _surprise(g: Mapping) -> float:
    return -float(g["lp_meta"]) / max(1, int(g["n_meta_tok"]))


def _drive(g: Mapping) -> float:
    return (float(g["lp_post_real"]) - float(g["lp_post_donor"])) / max(1, int(g["n_post"]))


FORMULAS: dict[str, Callable[[Mapping], float]] = {
    # ── 위치 차 (현행 계열)
    "pos":          lambda g: g["pmi_close"] - g["pmi_open"],
    "pos_clip":     lambda g: _clip(g["pmi_close"] - g["pmi_open"]),
    "pos_rev_only": lambda g: _rev(g["pmi_open"], g["pmi_close"]),
    "pos_full":     lambda g: _clip(g["pmi_close"] - g["pmi_open"])
                              + _rev(g["pmi_open"], g["pmi_close"]),
    # 열 때 이미 정답 쪽이면 고칠 게 없다 -> 승리 세리머니에 상을 주지 않는다
    "pos_cond":     lambda g: (g["pmi_close"] - g["pmi_open"]) if g["pmi_open"] < 0 else 0.0,

    # ── 문맥 차 · 답 기준
    "ans":          lambda g: g["pmi_ans_real"] - g["pmi_ans_donor"],
    "ans_clip":     lambda g: _clip(g["pmi_ans_real"] - g["pmi_ans_donor"]),
    "ans_cond":     lambda g: (g["pmi_ans_real"] - g["pmi_ans_donor"])
                              if g["pmi_ans_donor"] < 0 else 0.0,

    # ── 문맥 차 · 후속 본문 기준 ("그 다음 사고를 이끌었나")
    "drive":        _drive,
    "drive_raw":    lambda g: float(g["lp_post_real"]) - float(g["lp_post_donor"]),

    # ── RLRT 계열. 성공 가중은 곱하지 않는다 -- 정답 헤드가 그 일을 따로 하고,
    #    여기서 곱하면 오프라인 채점에서 결과가 새어 순환이 된다.
    "surprise":     _surprise,

    # ── 혼합
    "ans+drive":    lambda g: _clip(g["pmi_ans_real"] - g["pmi_ans_donor"]) + 2.0 * _drive(g),
    "surprise+ans": lambda g: _clip(g["pmi_ans_real"] - g["pmi_ans_donor"]) + 0.5 * _surprise(g),

    # ── 음성 대조군. 학습에 쓰라고 있는 게 아니다: 이것이 상위권에 오르면 위의 전부가
    #    길이의 대리일 뿐이라는 뜻이고, 스윕 결과 전체를 폐기해야 한다.
    "NEG_len_only": lambda g: float(g["meta_len"]),
}


def rmeta_from_ingredients(g: Mapping, formula: str) -> float:
    """이름으로 공식을 골라 적용한다. 오타는 즉사 -- 조용히 0 을 흘리면 무효 팔이 된다.

    (`_v4_rmeta_source_strict` 와 같은 fail-closed 자세: 이 저장소는 '선언은 있고 배선은
    없는' 레버로 여러 번 실험을 날렸다.)
    """
    if formula not in FORMULAS:
        raise ValueError(
            f"dcpo_rmeta_formula={formula!r} 는 없는 이름이다. 가능: {sorted(FORMULAS)}")
    missing = [k for k in INGREDIENT_KEYS if k not in g]
    if missing:
        raise KeyError(f"원재료 누락: {missing}")
    v = float(FORMULAS[formula](g))
    if v != v or v in (float("inf"), float("-inf")):   # NaN/inf -> fail-closed 0
        return 0.0
    return v
