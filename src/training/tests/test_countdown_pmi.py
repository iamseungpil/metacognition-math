"""`countdown_pmi` 단위 테스트 — ⛔GPU 없음, verl 없음.

두 층으로 나눴다.
  · **가짜 토크나이저** 층: 오프셋을 우리가 정하므로 스팬 산술을 정확한 정수로 못박는다.
    경계에 걸친 토큰(straddle)도 여기서 **의도적으로** 만든다.
  · **진짜 Qwen3-4B 토크나이저** 층: 평문 `<meta>` 가 단일 토큰이 아니라는 사실
    (`['<meta','>']`, `['</','meta','>']`, `'>\\n'` 한 토큰)이 실제로 처리되는지 본다.
    모델 가중치는 안 받는다 — 토크나이저 파일만 있으면 된다. 없으면 skip.

여기서 재는 것 넷(지시 항목 5):
    ① 메타 스팬 검출  ② 부분열 쌍 폴백  ③ 메타 없는 행  ④ 빈 결과
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.training.countdown_pmi import (  # noqa: E402
    CONFIG_REQUIREMENTS,
    boxed,
    build_pmi_arms,
    divergent_spans,
    find_meta_token_span,
    read_pmi_from_ref_logprobs,
    score_pmi_shift,
)
from src.training.countdown_task import META_CLOSE_TAG, META_OPEN_TAG  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 가짜 fast 토크나이저 — k글자 고정 청크. 오프셋이 우리 손에 있다.
# ══════════════════════════════════════════════════════════════════════════════

class ChunkTokenizer:
    """텍스트를 k글자씩 잘라 토큰으로 만든다. `is_fast=True` 를 흉내낸다.

    k=1 이면 문자=토큰이라 경계가 절대 안 걸리고, k>1 이면 태그 경계가 청크 안으로
    파고들어 **straddle 경로**가 실제로 돈다.
    """

    is_fast = True

    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = [], []
        for a in range(0, len(text), self.k):
            b = min(a + self.k, len(text))
            ids.append(1000 + (hash(text[a:b]) % 50000))
            offs.append((a, b))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class SlowTokenizer(ChunkTokenizer):
    is_fast = False


def _tok_real():
    """Qwen3-4B 토크나이저. 못 받으면 None (스킵 사유)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception:
        return None


REAL = _tok_real()
needs_real = pytest.mark.skipif(REAL is None,
                                reason="Qwen3-4B 토크나이저를 로컬에서 못 받았다")


def _resp(body_before="Try 3+4 first.\n", inner="\nconfidence: 0.4\nweak family\ndecision: redirect\n",
          after="\nNow \\boxed{(3+4)*5}"):
    return f"{body_before}{META_OPEN_TAG}{inner}{META_CLOSE_TAG}{after}"


# ══════════════════════════════════════════════════════════════════════════════
# ① 메타 스팬 검출
# ══════════════════════════════════════════════════════════════════════════════

def test_span_char_level_exact_indices():
    """k=1 이면 토큰 인덱스 = 문자 인덱스. 스팬 산술을 정수로 못박는다."""
    tok = ChunkTokenizer(1)
    text = _resp()
    span = find_meta_token_span(tok, text)
    assert span is not None
    i = text.index(META_OPEN_TAG)
    j = text.index(META_CLOSE_TAG)
    assert span.open_start == i
    assert span.inner_start == i + len(META_OPEN_TAG)
    assert span.inner_end == j
    assert span.close_end == j + len(META_CLOSE_TAG)
    assert span.n_inner_tok == j - (i + len(META_OPEN_TAG))
    assert not span.straddle_open and not span.straddle_close
    assert not span.meta_first


def test_span_never_leaks_meta_into_open_and_never_truncates_close():
    """★핵심 계약. 청크가 태그를 가로질러도 OPEN 엔 태그가 없고 CLOSE 엔 다 들어온다."""
    for k in (1, 2, 3, 5, 7):
        tok = ChunkTokenizer(k)
        text = _resp()
        span = find_meta_token_span(tok, text)
        assert span is not None, k
        # 토큰 인덱스 → 문자 위치로 되돌려 검사한다(k글자 청크라 정확히 k*idx).
        open_chars = text[:min(len(text), k * span.open_start)]
        close_chars = text[:min(len(text), k * span.close_end)]
        assert META_OPEN_TAG not in open_chars, (k, repr(open_chars[-20:]))
        assert META_CLOSE_TAG in close_chars, k
        assert span.open_start <= span.inner_start <= span.inner_end <= span.close_end


def test_span_straddle_flags_are_reported_not_swallowed():
    """경계에 걸치면 조용히 넘어가지 말고 **기록**해야 한다."""
    tok = ChunkTokenizer(4)
    # 앞 본문 길이를 1글자로 두면 4글자 청크가 '<' 를 가로지른다.
    text = "x" + META_OPEN_TAG + "hi" + META_CLOSE_TAG + "yz"
    span = find_meta_token_span(tok, text)
    assert span is not None
    assert span.straddle_open or span.straddle_close


def test_span_first_block_when_two():
    tok = ChunkTokenizer(1)
    text = ("a" + META_OPEN_TAG + "one" + META_CLOSE_TAG
            + "b" + META_OPEN_TAG + "two" + META_CLOSE_TAG + "c")
    span = find_meta_token_span(tok, text)
    assert span is not None
    assert span.open_start == 1
    assert span.close_end == 1 + len(META_OPEN_TAG) + 3 + len(META_CLOSE_TAG)


def test_span_meta_first_flag():
    tok = ChunkTokenizer(1)
    span = find_meta_token_span(tok, _resp(body_before=""))
    assert span is not None and span.meta_first and span.open_start == 0


def test_span_unclosed_block_is_none():
    tok = ChunkTokenizer(1)
    assert find_meta_token_span(tok, "abc" + META_OPEN_TAG + "no close here") is None


def test_span_requires_fast_tokenizer_and_raises_loudly():
    """느린 토크나이저에 None 을 흘리면 '아무도 메타를 안 냈다'로 오독된다."""
    with pytest.raises(TypeError):
        find_meta_token_span(SlowTokenizer(1), _resp())


def test_span_rejects_mismatched_response_ids():
    tok = ChunkTokenizer(1)
    with pytest.raises(ValueError):
        find_meta_token_span(tok, _resp(), response_ids=[1, 2, 3])


def test_span_accepts_matching_response_ids():
    tok = ChunkTokenizer(1)
    text = _resp()
    ids = tok(text)["input_ids"]
    assert find_meta_token_span(tok, text, response_ids=ids) is not None


@needs_real
def test_span_real_tokenizer_plain_meta_is_not_one_token():
    """실측 근거: 평문 `<meta>` 는 어휘에 없고 문맥마다 경계가 바뀐다."""
    assert REAL.convert_tokens_to_ids(META_OPEN_TAG) is None
    assert len(REAL(META_OPEN_TAG, add_special_tokens=False)["input_ids"]) > 1
    assert len(REAL(META_CLOSE_TAG, add_special_tokens=False)["input_ids"]) > 1


@needs_real
def test_span_real_tokenizer_decoded_contract():
    text = _resp()
    span = find_meta_token_span(REAL, text)
    assert span is not None
    open_txt = REAL.decode(list(span.ids[:span.open_start]))
    close_txt = REAL.decode(list(span.ids[:span.close_end]))
    assert "<meta" not in open_txt
    assert META_CLOSE_TAG in close_txt
    assert META_OPEN_TAG in close_txt
    assert span.n_inner_tok > 0
    # `'>\n'` 한 토큰이 닫는 태그 끝을 넘어가므로 straddle_close 가 서야 한다.
    assert span.straddle_close


@needs_real
def test_span_real_tokenizer_inner_count_is_a_lower_bound():
    """내용이 한 글자면 경계 토큰에 흡수돼 n_inner_tok=0 이 된다 — 알고 쓰는 편향."""
    tiny = find_meta_token_span(REAL, META_OPEN_TAG + "c" + META_CLOSE_TAG + "tail")
    assert tiny is not None and tiny.n_inner_tok == 0
    real = find_meta_token_span(REAL, _resp())
    assert real is not None and real.n_inner_tok >= 5


@needs_real
def test_span_real_tokenizer_meta_first_rollout():
    span = find_meta_token_span(REAL, _resp(body_before=""))
    assert span is not None and span.meta_first


# ══════════════════════════════════════════════════════════════════════════════
# ② 부분열 쌍 폴백
# ══════════════════════════════════════════════════════════════════════════════

def test_pair_normal_uses_div_path():
    tok = ChunkTokenizer(1)
    pair = divergent_spans(tok, "(3+4)*5", "(3-4)*5")
    assert pair is not None and pair.path == "div"
    assert pair.gold_slice.stop > pair.gold_slice.start
    assert pair.decoy_slice.stop > pair.decoy_slice.start


def test_pair_subsequence_falls_back_to_full():
    """★수리 지점. '2' 는 '12' 의 부분열이라 잘라내면 한쪽 슬라이스가 빈다."""
    tok = ChunkTokenizer(1)
    pair = divergent_spans(tok, "12", "2")
    assert pair is not None
    assert pair.path == "full"
    assert pair.gold_slice == slice(0, len(pair.gold_ids))
    assert pair.decoy_slice == slice(0, len(pair.decoy_ids))
    assert pair.gold_slice.stop > 0 and pair.decoy_slice.stop > 0


def test_pair_falls_back_in_both_directions():
    """gold 가 짧은 쪽일 때도(= **gold** 슬라이스가 비는 쪽) 폴백이 걸려야 한다."""
    tok = ChunkTokenizer(1)
    pair = divergent_spans(tok, "1", "12")
    assert pair is not None and pair.path == "full"
    assert pair.gold_slice.stop > 0 and pair.decoy_slice.stop > 0


def test_pair_identical_is_none():
    tok = ChunkTokenizer(1)
    assert divergent_spans(tok, "(3+4)", "(3+4)") is None


def test_pair_empty_is_none():
    tok = ChunkTokenizer(1)
    assert divergent_spans(tok, "", "(3+4)") is None
    assert divergent_spans(tok, "(3+4)", "") is None
    assert divergent_spans(tok, None, "(3+4)") is None
    assert divergent_spans(tok, "(3+4)", None) is None


def test_boxed_matches_the_existing_definition():
    """두 곳이 갈리면 PMI 가 다른 문자열을 재게 된다."""
    from src.training.dcpo_directional import boxed_answer_string
    for v in ("(3+4)*5", " 12 ", "1/2"):
        assert boxed(v) == boxed_answer_string(v)


@needs_real
def test_pair_real_tokenizer_length_mismatch_is_kept_not_dropped():
    """실측 3.7% 자리. 기존 `_pmi_position_scalar` 는 여기서 NaN 으로 죽는다."""
    g, d = "((11+(6*21))*(19-18))", "((11+(6*21))-(19-18))"
    gi = REAL(boxed(g), add_special_tokens=False)["input_ids"]
    di = REAL(boxed(d), add_special_tokens=False)["input_ids"]
    assert len(gi) != len(di), "이 쌍이 더 이상 길이 불일치가 아니면 예시를 갱신하라"
    pair = divergent_spans(REAL, g, d)
    assert pair is not None
    assert pair.gold_slice.stop > pair.gold_slice.start
    assert pair.decoy_slice.stop > pair.decoy_slice.start


# ══════════════════════════════════════════════════════════════════════════════
# ③ 메타 없는 행 처리
# ══════════════════════════════════════════════════════════════════════════════

def test_no_meta_row_is_skipped_and_counted():
    tok = ChunkTokenizer(1)
    prompts = ["P", "P"]
    resps = ["no meta at all \\boxed{(1+2)}", _resp()]
    ap, ar, attempts, diag = build_pmi_arms(
        tok, prompts, resps, ["(3+4)*5", "(3+4)*5"], ["(3-4)*5", "(3-4)*5"])
    assert diag["no_meta"] == 1
    assert len(attempts) == 1 and attempts[0].row == 1
    assert len(ap) == len(ar) == 4


def test_missing_witness_or_decoy_is_counted_separately():
    tok = ChunkTokenizer(1)
    resps = [_resp(), _resp(), _resp()]
    _ap, _ar, attempts, diag = build_pmi_arms(
        tok, ["P"] * 3, resps,
        ["", "(3+4)*5", "(3+4)*5"],
        ["(3-4)*5", "", "(3+4)*5"])          # 3번째는 gold==decoy
    assert diag["no_witness"] == 1
    assert diag["no_decoy"] == 1
    assert diag["bad_pair"] == 1
    assert attempts == []


def test_length_mismatch_between_inputs_raises():
    tok = ChunkTokenizer(1)
    with pytest.raises(ValueError):
        build_pmi_arms(tok, ["P"], [_resp(), _resp()], ["a"], ["b"])


def test_rows_for_meta_less_batch_are_all_unscored():
    tok = ChunkTokenizer(1)
    rows, diag = score_pmi_shift(
        tokenizer=tok, trainer=None,
        prompt_texts=["P", "P"],
        response_texts=["nothing here", "still nothing"],
        witnesses=["(3+4)*5"] * 2, decoys=["(3-4)*5"] * 2,
        debug=False, _ref_scorer=lambda p, r: pytest.fail("스코어러가 불리면 안 된다"))
    assert diag["attempted"] == 0 and diag["no_meta"] == 2
    for r in rows:
        assert r["emitted"] == 0 and r["scored"] is False
        assert math.isnan(r["pmi_open"]) and math.isnan(r["pmi_close"])
        assert r["meta_n_tok"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# ④ 빈 결과 / 스코어링 읽기
# ══════════════════════════════════════════════════════════════════════════════

def test_read_pmi_empty_attempts_gives_empty_list():
    assert read_pmi_from_ref_logprobs([], []) == []


def test_end_to_end_with_injected_ref_scorer():
    """팔 넷의 순서와 슬라이스 합산이 맞는지 손계산으로 확인한다."""
    tok = ChunkTokenizer(1)

    def fake_ref(arm_prompts, arm_resps):
        # 팔 r 의 모든 토큰 logp = -(r % 4 + 1). gold@open=-1, decoy@open=-2,
        # gold@close=-3, decoy@close=-4.
        return [[-float(r % 4 + 1)] * len(resp) for r, resp in enumerate(arm_resps)]

    rows, diag = score_pmi_shift(
        tokenizer=tok, trainer=None,
        prompt_texts=["P"], response_texts=[_resp()],
        witnesses=["(3+4)*5"], decoys=["(3-4)*5"],
        debug=False, _ref_scorer=fake_ref)
    assert diag["attempted"] == 1 and diag["scored"] == 1
    r = rows[0]
    assert r["emitted"] == 1 and r["scored"] is True and r["path"] == "div"
    pair = divergent_spans(tok, "(3+4)*5", "(3-4)*5")
    ng = pair.gold_slice.stop - pair.gold_slice.start
    nd = pair.decoy_slice.stop - pair.decoy_slice.start
    assert r["pmi_open"] == pytest.approx(-1.0 * ng - (-2.0 * nd))
    assert r["pmi_close"] == pytest.approx(-3.0 * ng - (-4.0 * nd))


def test_ref_rows_with_nonfinite_fail_closed_to_nan():
    tok = ChunkTokenizer(1)

    def bad_ref(arm_prompts, arm_resps):
        return [[float("inf")] * len(resp) for resp in arm_resps]

    rows, diag = score_pmi_shift(
        tokenizer=tok, trainer=None,
        prompt_texts=["P"], response_texts=[_resp()],
        witnesses=["(3+4)*5"], decoys=["(3-4)*5"],
        debug=False, _ref_scorer=bad_ref)
    assert diag["attempted"] == 1 and diag["scored"] == 0
    assert math.isnan(rows[0]["pmi_open"]) and math.isnan(rows[0]["pmi_close"])
    assert rows[0]["scored"] is False


def test_read_pmi_accepts_numpy_rows():
    np = pytest.importorskip("numpy")
    tok = ChunkTokenizer(1)
    _ap, _ar, attempts, _d = build_pmi_arms(
        tok, ["P"], [_resp()], ["(3+4)*5"], ["(3-4)*5"])
    assert len(attempts) == 1
    lens = [len(attempts[0].pair.gold_ids), len(attempts[0].pair.decoy_ids)] * 2
    width = max(lens) + 3            # ref_lp 는 배치 폭으로 패딩돼 온다
    ref = np.zeros((4, width), dtype=np.float32)
    ref[:] = -1.0
    pmis = read_pmi_from_ref_logprobs(ref, attempts)
    assert len(pmis) == 1
    # 전 팔이 같은 값이므로 PMI = -(#gold) + (#decoy) — 패딩이 새어 들어오면 깨진다.
    ng = attempts[0].pair.gold_slice.stop - attempts[0].pair.gold_slice.start
    nd = attempts[0].pair.decoy_slice.stop - attempts[0].pair.decoy_slice.start
    assert pmis[0][0] == pytest.approx(-1.0 * ng + 1.0 * nd)


def test_rows_match_countdown_rewards_row_contract():
    """`arm_reward` 가 요구하는 키(`pmi_open`/`pmi_close`/`meta_n_tok`)를 실제로 낸다."""
    from src.training.countdown_rewards import TERMS
    tok = ChunkTokenizer(1)
    rows, _ = score_pmi_shift(
        tokenizer=tok, trainer=None,
        prompt_texts=["P"], response_texts=[_resp()],
        witnesses=["(3+4)*5"], decoys=["(3-4)*5"],
        debug=False,
        _ref_scorer=lambda p, r: [[-1.0] * len(x) for x in r])
    produced = set(rows[0])
    for term in ("meta_pos", "meta_mul", "len"):
        for need in TERMS[term]["needs"]:
            if need in ("pmi_open", "pmi_close", "meta_n_tok", "emitted"):
                assert need in produced, f"{term} 의 원재료 {need} 를 안 낸다"


def test_config_requirement_is_declared():
    assert CONFIG_REQUIREMENTS["trainer.use_legacy_worker_impl"] == "disable"


@needs_real
def test_end_to_end_real_tokenizer_countdown_instance():
    """진짜 인스턴스·진짜 토크나이저로 한 바퀴. GPU 없이 도는지까지 본다."""
    from src.training.countdown_task import gen_instances
    inst = gen_instances(3, seed=7)
    resps = [_resp(after=f"\n\\boxed{{{r['witness']}}}") for r in inst]
    rows, diag = score_pmi_shift(
        tokenizer=REAL, trainer=None,
        prompt_texts=["Numbers/Target prompt"] * 3, response_texts=resps,
        witnesses=[r["witness"] for r in inst],
        decoys=[r["decoy"] for r in inst],
        debug=False,
        _ref_scorer=lambda p, r: [[-0.5] * len(x) for x in r])
    assert diag["attempted"] == 3 and diag["scored"] == 3
    assert all(row["emitted"] == 1 and row["meta_n_tok"] > 0 for row in rows)
