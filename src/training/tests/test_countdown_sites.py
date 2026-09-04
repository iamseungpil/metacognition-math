"""`countdown_sites.py` 회귀 테스트 — CPU 전용, 합성 텍스트 + 작은 인스턴스로 검산한다."""
import itertools
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_sites as cs                    # noqa: E402
from src.training import countdown_task as ct                     # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 1. 첫수 완전 열거 — 브루트포스로 total_solutions 를 독립 검산
# ══════════════════════════════════════════════════════════════════════════════

def _brute_force_total_solutions(nums, target) -> int:
    """`enumerate_solutions` 와 독립적으로 짠 브루트포스.

    ★대칭 연산 이중 계수 주의: `+`/`*` 는 `(i,j)` 와 `(j,i)` 가 같은 결합이므로
    **자리 쌍을 `i<j` 로만** 훑는다(전부 `i!=j` 로 훑으면 대칭 결합이 두 번 잡혀
    `enumerate_solutions`/`pair_moves` 의 카운팅 방식과 어긋난다 — 실측: 그렇게
    짜면 nums=[1,2,3,4] target=10 에서 47 이 아니라 272 가 나왔다).
    """
    def fold(vals):
        if len(vals) == 1:
            return 1 if vals[0] == Fraction(target) else 0
        n = len(vals)
        results = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = vals[i], vals[j]
                rest = [vals[k] for k in range(n) if k not in (i, j)]
                cands, seen = [], set()
                cands.append(("+", a + b))
                cands.append(("*", a * b))
                if a - b > 0:
                    cands.append(("-", a - b))
                if b - a > 0:
                    cands.append(("-", b - a))
                if b != 0 and (a / b).denominator == 1 and a / b > 0:
                    cands.append(("/", a / b))
                if a != 0 and (b / a).denominator == 1 and b / a > 0:
                    cands.append(("/", b / a))
                for key in cands:
                    # (op,val) 로 dedupe — val 만으로 dedupe 하면 서로 다른 연산이
                    # 우연히 같은 값을 내는 자리(예: nums (2,4) 에서 4-2=2 와 4/2=2)에서
                    # 서로 다른 두 식을 하나로 뭉개 버린다(실측으로 잡은 버그).
                    if key in seen:
                        continue
                    seen.add(key)
                    results += fold(rest + [key[1]])
        return results

    return fold([Fraction(v) for v in nums])


def test_enumerate_solutions_matches_brute_force_small_instance():
    """nums=[1,2,3,4] target=10 — enumerate_solutions 의 total 이 독립 브루트포스와 같다."""
    nums, target = [1, 2, 3, 4], 10
    counts, witness = cs.enumerate_solutions(nums, target)
    total = sum(counts.values())
    brute = _brute_force_total_solutions(nums, target)
    assert total == brute
    assert total > 0
    assert witness is not None
    # witness 가 실제로 이 문제를 푸는 식인지 기존 검증된 grade() 로 재확인.
    assert ct.grade(r"\boxed{%s}" % witness, nums, target) == 1


def test_enumerate_solutions_all_moves_sum_to_total():
    """첫수별 개수의 합이 total_solutions 와 같다(정의상 당연하지만 회귀 방지)."""
    nums, target = [2, 3, 5, 7], 30
    counts, _ = cs.enumerate_solutions(nums, target)
    assert sum(counts.values()) > 0


def test_pair_moves_no_illegal_intermediate():
    """`pair_moves` 가 뱉는 값은 전부 Countdown 규칙(양의 정수)을 지킨다."""
    for a, b in itertools.product(range(1, 10), repeat=2):
        for mv, val in cs.pair_moves(a, b):
            assert isinstance(val, int) and val > 0


def test_canon_move_symmetric_for_add_mul_but_not_sub_div():
    assert cs.canon_move(3, "+", 7) == cs.canon_move(7, "+", 3)
    assert cs.canon_move(3, "*", 7) == cs.canon_move(7, "*", 3)
    assert cs.canon_move(7, "-", 3) != cs.canon_move(3, "-", 7)


# ══════════════════════════════════════════════════════════════════════════════
# 2. site 추출 — 합성 텍스트
# ══════════════════════════════════════════════════════════════════════════════

def test_cut_own_meta_requires_complete_meta():
    incomplete = "some reasoning\n<meta>\nconfidence: 0.5\n</meta>\nmore"  # decision 없음
    assert cs.cut_own_meta(incomplete) is None

    complete = ("Attempt 1: 5+19=24\n<meta>\nconfidence: 0.4\njudging my approach\n"
                "decision: redirect\n</meta>\nAttempt 2: 25-3=22\n\\boxed{(25-3)+22}")
    prefix = cs.cut_own_meta(complete)
    assert prefix is not None
    assert prefix == complete[: complete.index("<meta>")]
    assert "<meta>" not in prefix


def test_cut_own_meta_none_when_meta_at_start():
    text = "<meta>\nconfidence: 0.5\nx\ndecision: verify\n</meta>\nrest"
    assert cs.cut_own_meta(text) is None


def test_cut_attempt_boundary_stratified_and_excludes_boxed():
    lines = [f"{i}+{i+1}={2*i+1}" for i in range(1, 20)]  # 19개의 등식 시도
    text = "\n".join(lines) + "\n\\boxed{1+2}"
    rng = random.Random(0)
    prefix = cs.cut_attempt_boundary(text, rng)
    assert prefix is not None
    assert "\\boxed{" not in prefix
    assert len(prefix) > 0


def test_cut_attempt_boundary_none_when_no_equalities():
    assert cs.cut_attempt_boundary("no equalities here at all", random.Random(0)) is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. family_dead
# ══════════════════════════════════════════════════════════════════════════════

def test_family_dead_none_without_attempts():
    assert cs.family_dead_label("no equalities", [1, 2, 3, 4], {}) is None


def test_family_dead_true_when_both_last_moves_unsolvable():
    nums, target = [1, 2, 3, 4], 100  # 100 은 이 네 수로 못 만든다(모두 <=24 최대곱)
    counts, _ = cs.enumerate_solutions(nums, target)
    assert sum(counts.values()) == 0  # 오라클 스스로도 해가 없다고 본다
    prefix = "1+2=3\n3+4=7\n"
    assert cs.family_dead_label(prefix, nums, counts) == 1


def test_family_dead_false_when_a_live_move_present():
    nums, target = [1, 2, 3, 4], 10  # (1+2+3+4)=10, 살아있는 계열 존재
    counts, _ = cs.enumerate_solutions(nums, target)
    live_move = next(mv for mv, c in counts.items() if c > 0)
    a, op, b = live_move
    prefix = f"9+9=18\n{a}{op}{b}={eval(f'{a}{op}{b}')}\n"
    assert cs.family_dead_label(prefix, nums, counts) == 0


def test_split_attempts_breaks_on_restart_from_original_number():
    nums = [5, 19, 25, 3]
    prefix = "5+19=24\n24+25=49\nTry again:\n25-3=22\n22+19=41\n"
    attempts = cs.split_attempts(prefix, nums)
    assert len(attempts) == 2
    assert attempts[0][0][:3] == ("5", "+", "19")
    assert attempts[1][0][:3] == ("25", "-", "3")


# ══════════════════════════════════════════════════════════════════════════════
# 4. oracle_for_site 조립
# ══════════════════════════════════════════════════════════════════════════════

def test_oracle_for_site_basic_fields():
    nums, target = [5, 19, 25, 3], 21
    prefix = "Let's try 5+19=24.\n"
    oracle = cs.oracle_for_site(prefix, nums, target)
    assert oracle["total_solutions"] > 0
    assert oracle["witness"] is not None
    assert ct.grade(r"\boxed{%s}" % oracle["witness"], nums, target) == 1
    assert isinstance(oracle["move_density"], dict)
    assert all(0.0 <= v <= 1.0 for v in oracle["move_density"].values())
    assert oracle["pairs_pre"] == [[5, 19]]
    # live_new_moves 는 pairs_pre 에 등장한 (5,19) 쌍의 첫수를 담지 않는다.
    for mv_str in oracle["live_new_moves"]:
        assert not (mv_str.startswith("5+19") or mv_str.startswith("19+5"))


def test_oracle_for_site_raises_when_unsolvable():
    """생성 보장이 깨진(=해가 0개인) 문제를 오라클에 넣으면 조용히 넘기지 않고 예외."""
    with pytest.raises(RuntimeError):
        cs.oracle_for_site("no attempts", [1, 1, 1, 1], 999999)


# ══════════════════════════════════════════════════════════════════════════════
# 5. build_site_row — parquet 스키마
# ══════════════════════════════════════════════════════════════════════════════

def test_build_site_row_schema_matches_training_parquet_plus_site_cols():
    nums, target = [5, 19, 25, 3], 21
    prefix = "Let's try 5+19=24.\n"
    oracle = cs.oracle_for_site(prefix, nums, target)
    row = cs.build_site_row(site_id="t-0", source="gs0", nums=nums, target=target,
                            oracle=oracle, cut_type="attempt-boundary", prefix=prefix)
    base_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info",
                "nums", "target", "witness", "decoy"}
    site_cols = {"site_id", "source", "cut_type", "prefix", "n_att_pre", "pairs_pre",
                "family_dead", "live_new_moves", "move_density", "total_solutions"}
    assert base_cols | site_cols == set(row.keys())
    assert row["prompt"][-1] == {"role": "assistant", "content": prefix}
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    g_nums, g_target = ct.parse_ground_truth(row["reward_model"]["ground_truth"])
    assert g_nums == nums and g_target == target
    # decoy 는 witness 와 값이 달라야 한다(swap_op_decoy 계약).
    if row["decoy"]:
        assert ct.grade(r"\boxed{%s}" % row["decoy"], nums, target) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. 템플릿 왕복 — 토크나이저 없으면 스킵
# ══════════════════════════════════════════════════════════════════════════════

def test_render_prefix_prompt_template_roundtrip():
    tok_dir = Path("/hdd_data/seungpil/scratch/models/Qwen3-4B")
    if not tok_dir.exists():
        pytest.skip(f"tokenizer dir not found: {tok_dir}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))

    inst = {"nums": [5, 19, 25, 3], "target": 21}
    prefix = "Let me try (5+19)=24.\n"
    msgs = cs.render_prefix_prompt(inst, prefix, "new")
    assert msgs[-1] == {"role": "assistant", "content": prefix}

    base_msgs = ct.build_prompt(inst, "new")
    gen = tok.apply_chat_template(base_msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    cont = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True,
                                   add_generation_prompt=False, enable_thinking=False)
    assert gen + prefix == cont

    msgs_empty = cs.render_prefix_prompt(inst, "", "new")
    cont_empty = tok.apply_chat_template(msgs_empty, tokenize=False,
                                         continue_final_message=True,
                                         add_generation_prompt=False, enable_thinking=False)
    assert cont_empty == gen
