"""`build_sites.split_train_judge` 회귀 테스트 — 문제당 site 상한 (0904 수리).

CPU 전용, 합성 site 딕셔너리만 쓴다 — 실 롤아웃 jsonl·토크나이저를 건드리지 않는다.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/local
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root

import build_sites as bs                                           # noqa: E402


def _make_sites(n_problems: int, sites_per_problem: int) -> list[dict]:
    """문제 하나당 site `sites_per_problem`개를 만든다 — family_dead/n_att_pre 를
    문제마다 다르게 흩어 stratum 도 여러 개 나오게 한다(층화 로직이 실제로 갈리는
    입력이어야 상한이 층화와 상호작용해도 깨지지 않는지 확인할 수 있다).
    """
    sites = []
    for p in range(n_problems):
        problem = ((p, p + 1, p + 2, p + 3), 10 + p)
        for k in range(sites_per_problem):
            sites.append({
                "_problem": problem,
                "_source": "gs0",
                "cut_type": "own-meta" if k % 2 == 0 else "attempt-boundary",
                "prefix": f"prefix-{p}-{k}",
                "n_att_pre": k % 20,
                "family_dead": k % 2,
                "pairs_pre": set(),
                "live_new_moves": [],
                "move_density": {},
                "total_solutions": 1,
            })
    return sites


def test_split_train_judge_respects_per_problem_caps():
    # 문제 30개 × site 40개 = 1200 후보. 상한 없이 돌리면 문제 몇 개만으로도
    # n_judge/n_train 을 다 채울 수 있는 크기다(0904 에 실제로 그랬다: 문제 24개).
    sites = _make_sites(n_problems=30, sites_per_problem=40)
    cap_train, cap_judge = 5, 3

    train_sites, judge_sites = bs.split_train_judge(
        sites, n_train=100, n_judge=60, seed=7,
        max_sites_per_problem_train=cap_train,
        max_sites_per_problem_judge=cap_judge)

    train_by_problem = Counter(s["_problem"] for s in train_sites)
    judge_by_problem = Counter(s["_problem"] for s in judge_sites)

    assert train_by_problem, "train 이 비었다"
    assert judge_by_problem, "judge 가 비었다"
    assert max(train_by_problem.values()) <= cap_train
    assert max(judge_by_problem.values()) <= cap_judge

    # 상한을 걸었으니 같은 n_judge=60 을 채우려면 문제가 최소 60/cap_judge=20개는 있어야 한다.
    assert len(judge_by_problem) >= 60 // cap_judge

    # 문제 단위 disjoint 는 상한과 무관하게 여전히 지켜야 한다.
    assert set(train_by_problem) & set(judge_by_problem) == set()


def test_split_train_judge_cap_forces_more_problems_than_uncapped():
    """같은 site 후보·같은 n_judge 에서, 상한을 걸면 상한 없을 때보다 judge 문제 수가
    늘어나야 한다(0904 결함이 정확히 이 방향으로 관측됐다 — 문제 24개로 1000행)."""
    sites = _make_sites(n_problems=30, sites_per_problem=40)

    _, judge_uncapped = bs.split_train_judge(
        sites, n_train=100, n_judge=200, seed=7,
        max_sites_per_problem_train=10_000, max_sites_per_problem_judge=10_000)
    _, judge_capped = bs.split_train_judge(
        sites, n_train=100, n_judge=200, seed=7,
        max_sites_per_problem_train=10_000, max_sites_per_problem_judge=4)

    n_problems_uncapped = len({s["_problem"] for s in judge_uncapped})
    n_problems_capped = len({s["_problem"] for s in judge_capped})
    assert n_problems_capped > n_problems_uncapped


if __name__ == "__main__":
    test_split_train_judge_respects_per_problem_caps()
    test_split_train_judge_cap_forces_more_problems_than_uncapped()
    print("OK")
