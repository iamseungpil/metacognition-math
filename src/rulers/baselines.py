r"""무모델(model-free) 기준선 — 새 자가 이걸 못 이기면 자가 아니라 문자열 겹침을
재는 것이다(`docs/VERDICT_cd6_pair_rulers.md` "Two cheapest next experiments" §1의
S2 처방). `table.py`의 partial-correlation·"beats every baseline" 관문이 이 값들을
쓴다. 전부 `needs_model=False` — CPU 전용, forward 없음.

포트 출처:
  S2 성분 (VERDICT 인용: "next-in-witness, first-step match, operator-in-witness,
  value-in-intermediates, bad-move-is-large-product") — `scripts/pair_rulers.py`의
  `move_in_path()`(참조된 이름, 그 파일이 만든 항진 진단 `next_in_witness`/
  `next_in_decoy` 컬럼과 같은 개념)와 VERDICT 본문의 서술을 근거로 여기서 재구현.
  `pair_rulers.py`에 `move_in_path`의 완전한 정의를 확인하지 못해(그 파일에 함수
  정의가 없고 import만 있었다), 여기서는 VERDICT가 서술한 다섯 성분을 문자 그대로
  독립 재구현한다 — 이름은 같지만 **재현이지 복제가 아니다**(다르면 반드시 함께 고칠
  것 — S2가 논문/판정문에 인용될 때는 이 파일의 버전이 원천이다).

  `prefix_features()` — `src/training/countdown_selfcontrol.py:97-125`을 **import**
  한다(포트 대상이 아니다: 그 함수는 "정답이 필요 없는 순수 텍스트 특징 추출"이라
  이미 이 패키지가 원하는 형태의 현행 정본이고, 복제하면 두 곳이 갈릴 위험만 생긴다
  — `CLAUDE.md`가 금지하는 종류의 중복이다. 이 패키지가 "포트하지 말고 import하지
  말라"고 금지한 대상은 `scripts/pair_rulers.py` 등 cd6 자 실험의 **낡은 스코어러**
  들이지, 여전히 쓰이는 현행 유틸리티가 아니다).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from src.rulers.base import MetaSample, Site

__all__ = ["s2_overlap_score", "meta_length_baseline", "stated_confidence_baseline",
           "decision_one_hot", "prefix_feature_baselines", "ALL_BASELINES"]

_MOVE_RE = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)")
_NUM_RE = re.compile(r"\d+")


def _first_move(text: str) -> Optional[tuple[int, str, int]]:
    m = _MOVE_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def s2_overlap_score(site: Site, sample: MetaSample) -> float:
    r"""VERDICT §1 처방의 "0-parameter overlap score S2". 5개 0/1 성분의 합(0..5):
      1. next-in-witness         next: 의 (a,b) 가 witness 문자열의 (a,b)와 일치
      2. first-step match        메타 뒤 첫 시도가 witness의 첫 걸음과 정확히 같다
      3. operator-in-witness     next: 의 연산자가 witness에 등장하는 연산자다
      4. value-in-intermediates  next: 의 두 값 중 하나가 witness의 정수 리터럴에 있다
      5. bad-move-is-large-product  next: 가 곱셈이고 두 값 다 두 자리 이상(큰 곱은
                                    흔히 "그럴듯해 보이는" 오답의 형태라는 VERDICT의
                                    관찰)
    각 성분은 대칭 정보가 없어도(정답을 몰라도) 문자열만 보고 계산된다 — "정답 경로
    누출"이 없다는 것 자체가 이 기준선의 핵심(어떤 자도 이걸 못 이기면 정답 흉내를
    재는 것뿐이라는 게 VERDICT의 논지).
    """
    witness = site.witness or ""
    nm = sample.next_move or ""
    mv = _first_move(nm)
    score = 0
    if mv:
        a, op, b = mv
        pair = frozenset((a, b))
        w_move = _first_move(witness)
        if w_move and frozenset((w_move[0], w_move[2])) == pair:
            score += 1                                    # next-in-witness
        w_ops = set(re.findall(r"[+\-*/]", witness))
        if op in w_ops:
            score += 1                                    # operator-in-witness
        w_nums = {int(x) for x in _NUM_RE.findall(witness)}
        if a in w_nums or b in w_nums:
            score += 1                                    # value-in-intermediates
        if op == "*" and a >= 10 and b >= 10:
            score += 1                                     # bad-move-is-large-product
    cont = sample.continuation or ""
    me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else 0
    after = cont[me:me + 64]
    fm = _first_move(after)
    w_move = _first_move(witness)
    if fm and w_move and frozenset((fm[0], fm[2])) == frozenset((w_move[0], w_move[2])) and fm[1] == w_move[1]:
        score += 1                                          # first-step match
    return float(score)


def meta_length_baseline(sample: MetaSample) -> float:
    """메타 길이(문자 수) — "길이의 대리" 대조군(`countdown_rewards.r_g`의 G 팔과
    같은 취지: 자가 사실 길이만 재고 있는지 가르는 최소 기준선)."""
    return float(len(sample.meta_raw or ""))


def stated_confidence_baseline(sample: MetaSample) -> float:
    """모델이 스스로 적은 confidence 값 그대로 — "지각은 좋다"(POSTMORTEM §0)는
    선행 관찰의 최솟값 기준선."""
    c = sample.confidence
    return float(c) if c is not None else float("nan")


def decision_one_hot(sample: MetaSample) -> dict:
    """decision 필드의 원-핫. redirect/verify/none 세 값."""
    d = (sample.decision or "").strip().lower()
    return {"decision_redirect": 1.0 if d == "redirect" else 0.0,
            "decision_verify": 1.0 if d == "verify" else 0.0,
            "decision_none": 1.0 if d not in ("redirect", "verify") else 0.0}


def prefix_feature_baselines(site: Site, sample: MetaSample) -> dict:
    """`countdown_selfcontrol.prefix_features`를 그대로 부른다(위 모듈 docstring
    참조 — import, 포트 아님)."""
    from src.training.countdown_selfcontrol import prefix_features  # noqa: PLC0415

    text = site.prefix + (sample.continuation or "")
    feats = prefix_features(text, site.nums)
    return {"n_att_pre": float(feats.get("n_att_pre", 0)),
            "pos_frac": float(feats.get("pos_frac", 0.0)),
            "has_boxed_pre": float(feats.get("has_boxed_pre", 0))}


ALL_BASELINES = ("s2_overlap", "meta_length", "stated_confidence",
                  "decision_redirect", "decision_verify",
                  "n_att_pre", "pos_frac", "has_boxed_pre")


def compute_all_baselines(site: Site, sample: MetaSample) -> dict:
    """`table.py`가 부르는 단일 진입점 — 위 성분을 한 dict로 모은다."""
    out = {
        "s2_overlap": s2_overlap_score(site, sample),
        "meta_length": meta_length_baseline(sample),
        "stated_confidence": stated_confidence_baseline(sample),
    }
    out.update(decision_one_hot(sample))
    try:
        out.update(prefix_feature_baselines(site, sample))
    except Exception:
        out.update({"n_att_pre": float("nan"), "pos_frac": float("nan"),
                    "has_boxed_pre": float("nan")})
    return out
