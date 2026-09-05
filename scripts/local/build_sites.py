#!/usr/bin/env python
r"""CLI — prefix-anchored site 데이터셋 빌드 (`src/training/countdown_sites.py` 사용).

    python scripts/local/build_sites.py --out_dir $WORK/data/sites_v1 \
        --n_train 3000 --n_judge 1000 --seed 7 \
        --max_sites_per_problem_train 12 --max_sites_per_problem_judge 8

산출물 (모두 `--out_dir` 아래):
  sites_train.parquet   재개 학습용 site 3000행(기본)
  sites_judge.parquet   판정용 site 1000행(기본) — train 과 (nums,target) 문제 겹치지 않음,
                        family_dead∈{0,1} × n_att 버킷으로 층화
  mixed_train.parquet   normal 50% + site(=sites_train) 50%, 섞은 것
  summary.json          컷 종류·family_dead·버킷별 개수 + 템플릿 왕복 검사 결과

★0904 수리(문제-클러스터 CI 결함): `--n_judge 1000` 하나만 있을 때 judge 가 단 24개
  문제에서만 뽑혔다(한 문제가 롤아웃 8개 × 3소스 × 2컷 ≈ 최대 48개 site 를 낸다 —
  층화 목표를 채우는 데 문제 몇 개면 충분했다). 문제 수가 적으면 "문제 단위" 신뢰구간이
  사실상 표본 24개짜리라 무의미하다. `--max_sites_per_problem_judge`(기본 8)·
  `--max_sites_per_problem_train`(기본 12) 로 문제당 site 수를 상한 걸어, 같은
  `--n_judge`/`--n_train` 을 채우려면 **더 많은 서로 다른 문제**를 끌어와야 하게 만든다.

CPU 전용 — GPU 를 만지지 않는다(학습이 GPU 를 쓰고 있으므로).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training import countdown_sites as cs  # noqa: E402
from src.training import countdown_task as ct  # noqa: E402

ROLLOUT_SOURCES = [
    ("gs0", "eval/gs0_Qwen3-4B_new/s11/texts.jsonl"),
    ("cd7_A", "eval/cd7_A_new_s2/step_30/texts.jsonl"),
    ("cd7_SC", "eval/cd7_SC_new_s1/step_30/texts.jsonl"),
]
TOKENIZER_DIR = "models/Qwen3-4B"
TRAIN_PARQUET = "data/countdown_train_4num_new.parquet"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_template_roundtrip(work: Path) -> dict:
    """토크나이저로 두 왕복 검사를 실측한다(모듈 docstring 이 아니라 여기서 실측 —
    토크나이저 버전이 바뀌면 이 함수의 실측치도 같이 바뀌어야 한다).
    """
    tok_dir = work / TOKENIZER_DIR
    result = {"tokenizer_found": tok_dir.exists()}
    if not tok_dir.exists():
        result["skipped_reason"] = f"tokenizer dir not found: {tok_dir}"
        return result

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    inst = {"nums": [5, 19, 25, 3], "target": 21}
    msgs = ct.build_prompt(inst, "new")
    prefix = "Let me try (5+19)=24.\n"

    gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    cont = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": prefix}], tokenize=False,
        continue_final_message=True, add_generation_prompt=False, enable_thinking=False)
    check1_ok = (gen + prefix) == cont

    cont_empty = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": ""}], tokenize=False,
        continue_final_message=True, add_generation_prompt=False, enable_thinking=False)
    check2_ok = cont_empty == gen

    result.update({
        "check1_nonempty_prefix_ok": check1_ok,
        "check1_gen_plus_prefix_tail": (gen + prefix)[-140:],
        "check1_cont_tail": cont[-140:],
        "check2_empty_prefix_ok": check2_ok,
        "check2_gen_tail": gen[-140:],
        "check2_cont_empty_tail": cont_empty[-140:],
    })
    return result


def collect_all_sites(work: Path, seed: int) -> list[dict]:
    """세 소스 전부를 훑어 site 후보 전부를 뽑는다. (nums,target,prefix) 로 전역 중복 제거."""
    seen: set[tuple] = set()
    sites: list[dict] = []
    n_incomplete_meta_skipped = 0
    n_no_boundary = 0
    for source, relpath in ROLLOUT_SOURCES:
        path = work / relpath
        rows = _load_jsonl(path)
        for ridx, row in enumerate(rows):
            # ★행별로 결정적 rng — 같은 입력이면 항상 같은 B컷을 고른다(재현성).
            rng = random.Random((hash((source, ridx, row["group_id"])) ^ seed) & 0xFFFFFFFF)
            key = (tuple(row["nums"]), row["target"])
            raw_sites = cs.extract_sites_from_rollout(row, rng)
            got_a = any(s["cut_type"] == "own-meta" for s in raw_sites)
            got_b = any(s["cut_type"] == "attempt-boundary" for s in raw_sites)
            if not got_a:
                n_incomplete_meta_skipped += 1
            if not got_b:
                n_no_boundary += 1
            for s in raw_sites:
                dedup_key = (key[0], key[1], s["prefix"])
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                s["_problem"] = key
                s["_source"] = source
                sites.append(s)
    return sites, {"n_incomplete_meta_skipped": n_incomplete_meta_skipped,
                   "n_no_boundary": n_no_boundary}


def stratum_of(site: dict) -> tuple:
    fam = site["family_dead"]
    fam_key = "none" if fam is None else str(int(fam))
    return (fam_key, cs.bucket_of(site["n_att_pre"]))


def split_train_judge(sites: list[dict], n_train: int, n_judge: int, seed: int, *,
                     max_sites_per_problem_train: int = 12,
                     max_sites_per_problem_judge: int = 8):
    """문제(nums,target) 단위로 disjoint 하게 나누고, judge 는 층화로 채운다.

    전략: 문제를 무작위 순서로 훑으며, 그 문제의 site 들(문제당 최대
    `max_sites_per_problem_judge`개로 미리 잘라 둔다) 이 "아직 부족한 층"을
    채우는 데 도움이 되는 동안은 judge 로, 그 뒤로는 train 후보 풀로 돌린다.
    엄격한 최적 배분(정수계획법)은 하지 않는다 — 이 자리는 "대략 고르게"면
    충분하고, 실제 분포는 `summary.json` 에 정직하게 남긴다.

    ★문제당 상한이 왜 필요한가(0904 수리): 상한 없이는 한 문제가 최대 48개
    (롤아웃 8 × 소스 3 × 컷 2) site 를 낼 수 있어, `n_judge` 를 채우는 데 문제
    몇 개면 충분했다(실측: 1000행이 문제 24개에서만 나왔다) — "문제 단위"
    신뢰구간을 표본 24개로 좁혀버린다. 상한을 걸면 같은 `n_judge`/`n_train` 을
    채우기 위해 반드시 더 많은 서로 다른 문제를 끌어와야 한다.
    """
    rng = random.Random(seed)
    by_problem: dict[tuple, list[dict]] = defaultdict(list)
    for s in sites:
        by_problem[s["_problem"]].append(s)
    problems = list(by_problem.keys())
    rng.shuffle(problems)

    strata = sorted({stratum_of(s) for s in sites})
    target_per_stratum = max(1, n_judge // max(1, len(strata)))
    judge_counts: Counter = Counter()
    judge_problems: set[tuple] = set()
    train_problems: set[tuple] = set()
    judge_pool: list[dict] = []

    judge_total = 0
    for p in problems:
        if judge_total >= n_judge:
            train_problems.add(p)
            continue
        p_sites = list(by_problem[p])
        rng.shuffle(p_sites)
        capped = p_sites[:max_sites_per_problem_judge]   # ★문제당 상한 — judge
        helps = any(judge_counts[stratum_of(s)] < target_per_stratum for s in capped)
        if helps:
            judge_problems.add(p)
            for s in capped:
                if judge_total >= n_judge:
                    break
                judge_counts[stratum_of(s)] += 1
                judge_total += 1
                judge_pool.append(s)
        else:
            train_problems.add(p)

    train_pool: list[dict] = []
    for p in train_problems:
        p_sites = list(by_problem[p])
        rng.shuffle(p_sites)
        train_pool.extend(p_sites[:max_sites_per_problem_train])  # ★문제당 상한 — train

    rng.shuffle(judge_pool)
    rng.shuffle(train_pool)

    judge_sites = judge_pool[:n_judge]
    train_sites = train_pool[:n_train]
    return train_sites, judge_sites


def drop_overlong_rows(rows: list[dict], work: Path, cap: int) -> list[dict]:
    """E-131(2026-09-06): 렌더링 후 프롬프트가 `cap` 토큰을 넘는 자리 행을 버린다.

    학습 시 `data.max_prompt_length`(2048) 를 넘는 행이 배치에 하나만 있어도 verl agent-loop 가
    배치 폭을 늘려 **모든 행**의 응답 꼬리가 채점에서 사라진다. 그래서 한도보다 넉넉히 아래(기본 1900)
    에서 자른다. 토크나이저가 없으면 실측 불가 → 명시적으로 실패한다(조용히 통과 금지).
    """
    tok_dir = work / TOKENIZER_DIR
    if not tok_dir.exists():
        raise RuntimeError(f"[build_sites] 토크나이저가 없어 프롬프트 길이 상한({cap})을 검사할 수 없다: {tok_dir}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    kept, dropped = [], []
    for r in rows:
        text = tok.apply_chat_template(list(r["prompt"]), tokenize=False, continue_final_message=True,
                                       add_generation_prompt=False, enable_thinking=False)
        n = len(tok(text).input_ids)
        (kept if n <= cap else dropped).append((r, n))
    if dropped:
        print(f"  E-131 상한 {cap} 초과로 {len(dropped)}행 제거: "
              + ", ".join(f"{r['site_id']}({n})" for r, n in dropped[:8]))
    return [r for r, _ in kept]


def to_rows(sites: list[dict], split: str) -> list[dict]:
    rows = []
    for i, s in enumerate(sites):
        nums, target = s["_problem"]
        site_id = f"{split}-{s['_source']}-{s['cut_type']}-{i}"
        rows.append(cs.build_site_row(
            site_id=site_id, source=s["_source"], nums=list(nums), target=target,
            oracle=s, cut_type=s["cut_type"], prefix=s["prefix"], split=split, index=i))
    return rows


def build_mixed_train(work: Path, site_rows: list[dict], seed: int) -> list[dict]:
    """normal 50% + site(=sites_train) 50%, 섞는다. normal 쪽도 빈 assistant 메시지를
    붙여 `apply_chat_template_kwargs` 가 배치 전체에 균일하게 먹도록 맞춘다.
    """
    import pandas as pd
    rng = random.Random(seed + 1)
    df = pd.read_parquet(work / TRAIN_PARQUET)
    n = len(site_rows)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    normal_rows = []
    for i in idx:
        r = df.iloc[i].to_dict()
        prompt = list(r["prompt"]) + [{"role": "assistant", "content": ""}]
        normal_rows.append({
            "data_source": r["data_source"], "prompt": prompt, "ability": r["ability"],
            "reward_model": dict(r["reward_model"]), "extra_info": dict(r["extra_info"]),
            "nums": list(r["nums"]), "target": int(r["target"]),
            "witness": r["witness"], "decoy": r["decoy"],
            "site_id": "", "source": "normal", "cut_type": "", "prefix": "",
            "n_att_pre": 0, "pairs_pre": "[]", "family_dead": None,
            "live_new_moves": "[]", "move_density": "{}", "total_solutions": 0,
        })
    mixed = normal_rows + list(site_rows)
    rng.shuffle(mixed)
    return mixed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_judge", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_sites_per_problem_train", type=int, default=12,
                    help="train 문제 하나가 최대 몇 개의 site 를 낼 수 있나 (0904 수리)")
    ap.add_argument("--max_sites_per_problem_judge", type=int, default=8,
                    help="judge 문제 하나가 최대 몇 개의 site 를 낼 수 있나 (0904 수리)")
    ap.add_argument("--work", default=None, help="WORK 루트 (기본: $WORK 환경변수)")
    ap.add_argument("--max_prompt_tokens", type=int, default=1900,
                    help="E-131: 렌더링 후 프롬프트 토큰 상한(학습 max_prompt_length 2048 보다 넉넉히 아래)")
    args = ap.parse_args()

    import os
    work = Path(args.work or os.environ["WORK"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 롤아웃 세 소스에서 site 후보 수집 중...", flush=True)
    sites, extract_stats = collect_all_sites(work, args.seed)
    print(f"  후보 site {len(sites)}개 (own-meta 스킵 {extract_stats['n_incomplete_meta_skipped']}, "
          f"boundary 없음 {extract_stats['n_no_boundary']})")

    print("[2/5] train/judge 문제 단위 분할 중...", flush=True)
    train_sites, judge_sites = split_train_judge(
        sites, args.n_train, args.n_judge, args.seed,
        max_sites_per_problem_train=args.max_sites_per_problem_train,
        max_sites_per_problem_judge=args.max_sites_per_problem_judge)
    train_problems = {s["_problem"] for s in train_sites}
    judge_problems = {s["_problem"] for s in judge_sites}
    overlap = train_problems & judge_problems
    if overlap:
        raise RuntimeError(f"train/judge 문제가 {len(overlap)}개 겹친다 — 분할 로직 결함.")
    print(f"  train site {len(train_sites)} (문제 {len(train_problems)}개), "
          f"judge site {len(judge_sites)} (문제 {len(judge_problems)}개)")

    print("[3/5] parquet 행 조립 중...", flush=True)
    train_rows = drop_overlong_rows(to_rows(train_sites, "train"), work, args.max_prompt_tokens)
    judge_rows = drop_overlong_rows(to_rows(judge_sites, "judge"), work, args.max_prompt_tokens)

    import pandas as pd
    pd.DataFrame(train_rows).to_parquet(out_dir / "sites_train.parquet", index=False)
    pd.DataFrame(judge_rows).to_parquet(out_dir / "sites_judge.parquet", index=False)

    print("[4/5] mixed_train.parquet 조립 중...", flush=True)
    mixed_rows = build_mixed_train(work, train_rows, args.seed)
    pd.DataFrame(mixed_rows).to_parquet(out_dir / "mixed_train.parquet", index=False)

    print("[5/5] 템플릿 왕복 검사 + summary.json 기록 중...", flush=True)
    template_check = check_template_roundtrip(work)

    def counts_by(rows, key):
        c = Counter(r[key] for r in rows)
        return dict(c)

    def fam_bucket_counts(rows):
        c = Counter()
        for r in rows:
            fam = "none" if r["family_dead"] is None else str(int(r["family_dead"]))
            c[(fam, cs.bucket_of(r["n_att_pre"]))] += 1
        return {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}

    def stratum_problem_counts(pre_row_sites):
        """층(stratum)별 **서로 다른 문제 수** — site 개수가 아니라, 0904 수리가
        직접 겨냥한 지표(문제-클러스터 CI 는 이 수가 표본 크기다).
        """
        d: dict[tuple, set] = defaultdict(set)
        for s in pre_row_sites:
            d[stratum_of(s)].add(s["_problem"])
        return {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(d.items())}

    summary = {
        "n_candidate_sites": len(sites),
        "extract_stats": extract_stats,
        "n_train": len(train_rows), "n_judge": len(judge_rows),
        "n_mixed": len(mixed_rows),
        "n_train_problems": len(train_problems), "n_judge_problems": len(judge_problems),
        "max_sites_per_problem_train": args.max_sites_per_problem_train,
        "max_sites_per_problem_judge": args.max_sites_per_problem_judge,
        "train_cut_type_counts": counts_by(train_rows, "cut_type"),
        "judge_cut_type_counts": counts_by(judge_rows, "cut_type"),
        "train_family_dead_bucket_counts": fam_bucket_counts(train_rows),
        "judge_family_dead_bucket_counts": fam_bucket_counts(judge_rows),
        "train_stratum_problem_counts": stratum_problem_counts(train_sites),
        "judge_stratum_problem_counts": stratum_problem_counts(judge_sites),
        "template_roundtrip_check": template_check,
        "seed": args.seed,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
