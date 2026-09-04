r"""Prefix-anchored site 빌더 — 부분 응답 접합점에서 GRPO 를 재개하기 위한 데이터셋.

동기. GRPO 는 지금 응답을 **처음부터** 굴린다. 그런데 정말 갈리는 지점은 대개
"막힌 뒤 메타를 쓰기 직전"이나 "탐색 도중 특정 시도 경계"다. 그 지점을 프롬프트에
직접 박아 넣으면(=`prompt` 컬럼에 assistant 프리픽스를 이어 붙이고 verl 이 그 뒤부터
생성하게 하면) 롤아웃 예산을 그 지점에 집중시킬 수 있다 — 이 파일은 그 "지점"(site)
을 실제 롤아웃에서 뽑고, 오라클(완전 열거)로 "그 지점에서 정답이 아직 살아 있는가"를
라벨링한다.

**site 두 종류**
  A "own-meta"         — 첫 `<meta>` **앞**까지. 모델이 스스로 멈춰 메타를 쓴 지점.
  B "attempt-boundary" — 시도(등식) 한 줄이 끝난 뒤. n_att_pre 로 층화해 무작위로 하나
                         고른다(막히기 전/중/후 지점을 고루 뽑기 위해서다).

**오라클 라벨** (CPU, 완전 열거 — `enumerate_first_moves`)
  Countdown 은 수 4개뿐이라 "첫 결합"의 공간이 작다(최대 6개 자리 × 최대 4개 연산
  ≈ 20개). 그 전부를 열거해 "이 첫 결합에서 시작하는 해가 몇 개인가"를 정확히 센다.
  `move_density`/`live_new_moves`/`family_dead` 는 전부 이 카운트 위에서 정의한다.

**설계 결정 — 문서화된 가정 (스펙이 명시하지 않아 이 파일이 직접 고른 것)**
  1. "첫 결합"의 정규형: `+`/`*` 는 대칭이라 `(min(a,b), op, max(a,b))` 로 접는다.
     `-`/`/` 는 방향이 결과를 바꾸므로 실제로 쓰인 순서 `(a, op, b)` 그대로 둔다.
  2. `family_dead` 의 "attempt" 경계. 스펙은 "마지막 두 시도의 첫 결합"이라고만
     적었고 "시도"를 텍스트에서 어떻게 자르는지는 규정하지 않았다. 여기서는
     `_ARITH_EQ`(`a op b = c`) 매치들을 순서대로 훑어, **왼쪽 피연산자가 직전 매치의
     결과와 같으면 같은 시도(사슬)로, 원래 문제의 수 중 하나로 되돌아가면 새 시도**로
     간주한다(`_split_attempts`). 이 규칙 밖의 재시작(중간값을 다시 인용하는 등)은
     새 시도로 오분류될 수 있다 — 완전한 자연어 파싱은 정규식으로 원리적으로 불가능
     하므로 이는 최소 방어선이다(`countdown_rewards._multiset_has_pair` 의 같은 계열
     주석 참조).
  3. `move_density` 의 분모는 그 문제의 `total_solutions`(오라클, 모든 정답 트리 수).
     "distinct expression trees is fine"(과제 지시) — 같은 값 두 자리가 있으면
     자리가 다른 같은 결합을 별개 트리로 셀 수 있다(과다 계수 가능성은 인정하고
     넘어간다. 목적은 "이 첫수가 상대적으로 얼마나 많이 쓰이는 해로 이어지는가"라는
     상대 비교이지 절대 유일 카운트가 아니다).
  4. site 의 `witness`/`decoy` — 학습 parquet 스키마가 요구하는 두 컬럼이지만, eval
     롤아웃 jsonl 에는 생성 시점의 witness 가 없다(그 소스는 `nums/target/text/
     r_corr/group_id` 만 준다). 그래서 오라클 열거 도중 찾은 **첫 정답 트리**를
     witness 로 역산하고, `countdown_task.swap_op_decoy` 로 decoy 를 새로 만든다
     (기존 검증된 함수 재사용 — 복제하지 않는다).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Optional

from src.training import countdown_task as ct
from src.training.countdown_rewards import parse_meta
from src.training.countdown_selfcontrol import _ARITH_EQ, prefix_features

__all__ = [
    "N_ATT_BUCKETS", "bucket_of", "canon_move", "move_key_str",
    "pair_moves", "all_candidate_first_moves", "enumerate_solutions",
    "cut_own_meta", "cut_attempt_boundary", "split_attempts", "family_dead_label",
    "oracle_for_site", "extract_sites_from_rollout", "render_prefix_prompt",
    "build_site_row",
]

# ══════════════════════════════════════════════════════════════════════════════
# 0. n_att_pre 층화 버킷 — 과제 지시의 4구간(2-4/5-9/10-16/17+) + 0-1 을 보태
#    시도 0~1회인 own-meta 류 site 도 요약·판정 세트 층화에서 빠지지 않게 한다.
# ══════════════════════════════════════════════════════════════════════════════

N_ATT_BUCKETS = ("0-1", "2-4", "5-9", "10-16", "17+")


def bucket_of(n_att: int) -> str:
    if n_att <= 1:
        return "0-1"
    if n_att <= 4:
        return "2-4"
    if n_att <= 9:
        return "5-9"
    if n_att <= 16:
        return "10-16"
    return "17+"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 첫수 완전 열거 — Countdown 4수의 오라클 (nums·target 만으로 결정된다)
# ══════════════════════════════════════════════════════════════════════════════

def canon_move(a: int, op: str, b: int) -> tuple:
    """첫수 정규형. `+`/`*` 는 대칭 정렬, `-`/`/` 는 실제 순서를 보존한다."""
    if op in ("+", "*"):
        lo, hi = (a, b) if a <= b else (b, a)
        return (lo, op, hi)
    return (a, op, b)


def move_key_str(move: tuple) -> str:
    a, op, b = move
    return f"{a}{op}{b}"


def pair_moves(a: int, b: int) -> list[tuple[tuple, int]]:
    """두 값 a·b 에서 만들 수 있는 Countdown-합법 결합 전부: [(정규형 첫수, 값), ...].

    중간값 양의 정수 규칙은 `countdown_task._fold_countdown` 과 동형이다(따로
    임포트하지 않는다 — 여기 필요한 건 "두 값" 수준의 한 스텝 판정뿐이라 전체 AST
    평가기를 끌어오는 게 오히려 배선을 흐린다. 규칙 자체는 셋 다 같다: 덧셈·곱셈은
    항상 합법, 뺄셈은 결과가 양수일 때만, 나눗셈은 나누어떨어지고 몫이 양수일 때만).
    """
    out: list[tuple[tuple, int]] = []
    out.append((canon_move(a, "+", b), a + b))
    out.append((canon_move(a, "*", b), a * b))
    if a - b > 0:
        out.append(((a, "-", b), a - b))
    if b - a > 0:
        out.append(((b, "-", a), b - a))
    if b != 0 and a % b == 0 and a // b > 0:
        out.append(((a, "/", b), a // b))
    if a != 0 and b % a == 0 and b // a > 0:
        out.append(((b, "/", a), b // a))
    # a==b 인 뺄셈/나눗셈은 두 조건문이 같은 튜플을 두 번 만들 수 있다 — 중복 제거.
    seen, dedup = set(), []
    for mv, val in out:
        if mv in seen:
            continue
        seen.add(mv)
        dedup.append((mv, val))
    return dedup


def all_candidate_first_moves(nums) -> list[tuple]:
    """원래 4수에서 나올 수 있는 **합법** 첫수 전부(해로 이어지는지는 안 본다)."""
    nums = [int(v) for v in nums]
    seen, out = set(), []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for mv, _ in pair_moves(nums[i], nums[j]):
                if mv not in seen:
                    seen.add(mv)
                    out.append(mv)
    return out


def enumerate_solutions(nums, target) -> tuple[dict, Optional[str]]:
    """완전 열거 — {정규형 첫수: 그 첫수로 시작하는 해의 개수}, 그리고 첫 해의 식 문자열.

    ⚠추측 아님·완전 열거: 4수에서 매 스텝 두 값을 골라 접는 모든 경로를 전부 밟는다
    (가지치기 없음). 4수 기준 분기 수가 작아(최대 6쌍×최대 4연산, 3수로 줄면
    3쌍×4, 2수면 1쌍×4) 수백만 이상으로 폭발하지 않는다.
    """
    counts: Counter = Counter()
    witness_holder: list[str] = []

    def rec(vals: list[tuple[int, str]], first_move):
        if len(vals) == 1:
            v, expr = vals[0]
            if v == int(target):
                counts[first_move] += 1
                if not witness_holder:
                    witness_holder.append(expr)
            return
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                (a, ea), (b, eb) = vals[i], vals[j]
                rest = [vals[k] for k in range(n) if k not in (i, j)]
                for mv, val in pair_moves(a, b):
                    op = mv[1]
                    if op in ("+", "*"):
                        expr = f"({ea}{op}{eb})"          # 대칭 — 순서 무관
                    else:
                        # `-`/`/` 는 `pair_moves` 가 이미 실제 계산 순서(mv[0] 이
                        # 왼쪽 피연산자 값)를 돌려준다 — 그 값이 a 인지 b 인지만
                        # 보고 식 문자열의 좌우를 맞춘다(재추정 아님, 그대로 읽음).
                        expr = f"({ea}{op}{eb})" if mv[0] == a else f"({eb}{op}{ea})"
                    fm = first_move if first_move is not None else mv
                    rec(rest + [(val, expr)], fm)

    rec([(int(v), str(int(v))) for v in nums], None)
    return dict(counts), (witness_holder[0] if witness_holder else None)


# ══════════════════════════════════════════════════════════════════════════════
# 2. site 컷 — 롤아웃 텍스트 한 줄에서 프리픽스 두 종류를 뽑는다
# ══════════════════════════════════════════════════════════════════════════════

def cut_own_meta(text: str) -> Optional[str]:
    """A 컷 — 첫 `<meta>` **앞**. **완결된**(confidence·decision 둘 다 있는) 메타가
    없으면 None(스펙: "완결된 메타 없는 행은 스킵"). 위치 0(=<meta> 가 응답 맨
    앞)이면 프리픽스가 빈 문자열이라 재개 지점으로 의미가 없어 역시 None.
    """
    m = parse_meta(text, "new")
    if not m.get("emitted") or m.get("start") is None:
        return None
    start = int(m["start"])
    if start <= 0:
        return None
    return text[:start]


def cut_attempt_boundary(text: str, rng: random.Random) -> Optional[str]:
    r"""B 컷 — 시도(등식) 줄이 끝나는 지점 중 하나를 n_att_pre 로 층화해 고른다.

    후보 = `_ARITH_EQ`(`a op b = c`) 매치가 속한 줄의 **끝(그 줄 다음 줄바꿈 뒤)**.
    0번 위치·`\boxed{` 이후는 제외한다(스펙 그대로). 후보를 `N_ATT_BUCKETS` 로
    묶어 비어 있지 않은 버킷 중 하나를 고른 뒤 그 안에서 하나를 뽑는다 — 이렇게
    해야 "시도 2~4회 뒤" 부터 "17회 넘게 헤맨 뒤"까지 site 분포가 한쪽으로
    쏠리지 않는다.
    """
    boundaries: set[tuple[int, int]] = set()
    for m in _ARITH_EQ.finditer(text):
        nl = text.find("\n", m.end())
        if nl == -1:                       # "줄 다음 줄바꿈" 이 없으면 경계가 아니다
            continue
        boundary = nl + 1
        if boundary <= 0 or boundary >= len(text):
            continue
        cand_prefix = text[:boundary]
        if "\\boxed{" in cand_prefix:
            continue
        n_att = len(_ARITH_EQ.findall(cand_prefix))
        boundaries.add((boundary, n_att))

    buckets: dict[str, list[int]] = {b: [] for b in N_ATT_BUCKETS}
    for boundary, n_att in boundaries:
        buckets[bucket_of(n_att)].append(boundary)

    nonempty = [b for b in N_ATT_BUCKETS if buckets[b]]
    if not nonempty:
        return None
    chosen_bucket = rng.choice(nonempty)
    boundary = rng.choice(sorted(buckets[chosen_bucket]))
    return text[:boundary]


# ══════════════════════════════════════════════════════════════════════════════
# 3. family_dead — 마지막 두 시도의 첫수가 아직 살아 있는 계열인가
# ══════════════════════════════════════════════════════════════════════════════

def split_attempts(prefix: str, nums) -> list[list[tuple[str, str, str, str]]]:
    """`prefix` 를 시도(사슬) 단위로 나눈다. 각 시도는 `(a,op,b,c)` 매치 리스트.

    ★규칙(문서화된 가정, 모듈 docstring §설계 결정 2 참조): 매치의 왼쪽 피연산자가
    **직전 매치의 결과와 같으면** 같은 사슬(같은 시도)로 잇고, 다르면 새 시도로
    끊는다. "원래 문제의 수로 되돌아간다"는 이 조건의 특수한 경우일 뿐이다 —
    되돌아간 수는 직전 결과와 (거의 항상) 다르므로 이미 `a != prev_c` 로 걸린다.
    반대로 별도 검사를 추가하면, 우연히 직전 결과가 원래 수 중 하나와 같은 값일
    때(예: `1+2=3` 다음 `3+4=7` — 3 은 원래 nums 에도 있다) 정상적으로 이어지는
    사슬을 잘못 끊어버린다 — `nums` 인자는 그래서 시그니처에는 남기되(다른
    site 헬퍼들과 형태를 맞추려는 목적) 이 판정에는 쓰지 않는다.
    """
    matches = list(_ARITH_EQ.finditer(prefix))
    if not matches:
        return []
    attempts: list[list[tuple]] = []
    prev_c: Optional[int] = None
    for m in matches:
        a, op, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
        starts_new = (prev_c is None) or (int(a) != prev_c)
        if starts_new or not attempts:
            attempts.append([])
        attempts[-1].append((a, op, b, c))
        prev_c = int(c)
    return attempts


def family_dead_label(prefix: str, nums, solution_counts: dict) -> Optional[int]:
    """마지막 두 시도의 첫수 중 **어느 하나도** 해로 이어지지 않으면 1, 하나라도
    살아 있으면 0, 시도가 아예 없으면 None(스펙 그대로).
    """
    attempts = split_attempts(prefix, nums)
    if not attempts:
        return None
    last2 = attempts[-2:]
    moves = []
    for att in last2:
        a, op, b, _c = att[0]
        moves.append(canon_move(int(a), op, int(b)))
    alive = any(solution_counts.get(mv, 0) > 0 for mv in moves)
    return 0 if alive else 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. 오라클 조립 — site 하나의 라벨 전부
# ══════════════════════════════════════════════════════════════════════════════

def oracle_for_site(prefix: str, nums, target) -> dict:
    """site 하나에 대한 완전 열거 오라클 라벨 전부.

    Returns dict: n_solutions_first_moves(정규형 첫수 문자열→개수), family_dead,
    live_new_moves(문자열 리스트), move_density(문자열→비율), total_solutions,
    witness(오라클이 찾은 첫 해), n_att_pre, pairs_pre(정렬된 [a,b] 리스트), pos_frac.
    """
    pf = prefix_features(prefix, nums)
    counts, witness = enumerate_solutions(nums, target)
    total = sum(counts.values())
    if total == 0:
        # 생성 과정(gen_instance)이 해 존재를 구성으로 보장하므로 여기 걸리면
        # 오라클 자체가 고장난 것 — 조용히 넘기지 않는다(fail-loud).
        raise RuntimeError(
            f"oracle_for_site: nums={nums} target={target} 에 해가 0개 — 생성 "
            "보장(gen_instance)이 깨졌거나 enumerate_solutions 에 결함이 있다.")

    candidates = all_candidate_first_moves(nums)
    pairs_pre = pf["pairs_pre"]
    move_density = {move_key_str(mv): counts.get(mv, 0) / total for mv in candidates}
    live_new_moves = [
        move_key_str(mv) for mv in candidates
        if counts.get(mv, 0) > 0 and (min(mv[0], mv[2]), max(mv[0], mv[2])) not in pairs_pre
    ]
    fam_dead = family_dead_label(prefix, nums, counts)

    return {
        "n_solutions_first_moves": {move_key_str(k): v for k, v in counts.items()},
        "family_dead": fam_dead,
        "live_new_moves": live_new_moves,
        "move_density": move_density,
        "total_solutions": total,
        "witness": witness,
        "n_att_pre": pf["n_att_pre"],
        "pairs_pre": sorted([list(p) for p in pairs_pre]),
        "pos_frac": pf["pos_frac"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 롤아웃 하나 → site 0~2개
# ══════════════════════════════════════════════════════════════════════════════

def extract_sites_from_rollout(row: dict, rng: random.Random) -> list[dict]:
    """롤아웃 한 줄({group_id,nums,target,r_corr,text}) → site dict 리스트(0~2개).

    각 site: {cut_type, prefix} + `oracle_for_site` 의 라벨 전부. `nums`/`target`
    은 호출자가 이미 갖고 있어 다시 넣지 않는다(호출자가 합친다).
    """
    text = row["text"]
    nums, target = row["nums"], row["target"]
    out = []
    a_prefix = cut_own_meta(text)
    if a_prefix is not None:
        out.append({"cut_type": "own-meta", "prefix": a_prefix,
                    **oracle_for_site(a_prefix, nums, target)})
    b_prefix = cut_attempt_boundary(text, rng)
    if b_prefix is not None:
        out.append({"cut_type": "attempt-boundary", "prefix": b_prefix,
                    **oracle_for_site(b_prefix, nums, target)})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. 프리픽스 프롬프트 렌더링 — chat 메시지 리스트 (템플릿 적용은 verl/트레이너 몫)
# ══════════════════════════════════════════════════════════════════════════════

def render_prefix_prompt(inst, prefix: str, variant: str = "new") -> list[dict]:
    """`countdown_task.build_prompt` 뒤에 assistant 프리픽스 메시지를 이어붙인다.

    ★템플릿 적용(`apply_chat_template(..., continue_final_message=True,
    add_generation_prompt=False, enable_thinking=False)`)이 원래 생성 프롬프트 +
    프리픽스 바이트와 정확히 같다는 것은 이 모듈이 검증하지 않는다 — 그건 토크나이저가
    할 일이고, `scripts/local/build_sites.py` 가 실행 시점에 실측해 `summary.json` 에
    적는다(모듈 docstring 이 아니라 실측치를 남겨야 토크나이저 버전이 바뀌어도
    거짓말을 안 한다).
    """
    msgs = ct.build_prompt(inst, variant)
    msgs.append({"role": "assistant", "content": prefix})
    return msgs


# ══════════════════════════════════════════════════════════════════════════════
# 7. site → parquet 행
# ══════════════════════════════════════════════════════════════════════════════

def build_site_row(*, site_id: str, source: str, nums, target, oracle: dict,
                    cut_type: str, prefix: str, variant: str = "new",
                    split: str = "train", index: int = 0) -> dict:
    """`countdown_task.build_records` 와 같은 형(prompt/ability/reward_model/
    extra_info/nums/target/witness/decoy) + site 전용 컬럼.

    `decoy` 는 `countdown_task.swap_op_decoy` 로 오라클 witness에서 새로 만든다
    (기존 검증 함수 재사용). witness 를 못 찾았으면(이론상 `oracle_for_site` 가
    이미 total=0 을 fail-loud 로 막으므로 일어나지 않지만) decoy 도 빈 문자열.
    """
    nums = [int(v) for v in nums]
    target = int(target)
    inst = {"nums": nums, "target": target}
    witness = oracle.get("witness") or ""
    decoy = ""
    if witness:
        d = ct.swap_op_decoy(witness, nums, target, random.Random(hash(site_id) & 0xFFFFFFFF))
        decoy = d if d is not None else ""

    extra = {
        "index": index, "split": split, "nums": nums, "target": target,
        "witness": witness, "decoy": decoy, "prompt_variant": variant,
        "search_budget": ct.SEARCH_BUDGET,
    }
    return {
        "data_source": "countdown",
        "prompt": render_prefix_prompt(inst, prefix, variant),
        "ability": "countdown",
        "reward_model": {"style": "rule", "ground_truth": ct.make_ground_truth(nums, target)},
        "extra_info": extra,
        "nums": nums, "target": target, "witness": witness, "decoy": decoy,
        # site 전용 컬럼
        "site_id": site_id, "source": source, "cut_type": cut_type, "prefix": prefix,
        "n_att_pre": int(oracle["n_att_pre"]),
        "pairs_pre": json.dumps(oracle["pairs_pre"], separators=(",", ":")),
        "family_dead": oracle["family_dead"],
        "live_new_moves": json.dumps(oracle["live_new_moves"], separators=(",", ":")),
        "move_density": json.dumps(oracle["move_density"], separators=(",", ":")),
        "total_solutions": int(oracle["total_solutions"]),
    }
