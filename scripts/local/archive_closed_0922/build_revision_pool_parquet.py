#!/usr/bin/env python
r"""수정(revision) 팔용 **비만장일치 학습 풀** — 스크리닝 롤아웃에서 골라 parquet 으로.

왜: 기본 학습 풀(math_train_math_opt.parquet, 7,295문제)은 수정 행이 ~1.4% 뿐이라 512행
스텝당 크레딧 받는 행이 1~3개다(사전등록 수정 6 의 처치가 사실상 관측 불가). 수정은
**비만장일치** 문제에 몰린다(L5 실측: DOMINANT 7.6% vs ALL_SAME 1.4%). 그래서 이미 있는
스크리닝 롤아웃(문제당 8 롤아웃)에서 그룹 합의 상태를 재서 DOMINANT(변형 A) /
DOMINANT+SPLIT(변형 B) 문제만 남긴다.

상태 정의는 `src.training.trial2.agreement_state`(**엄격** 동치 — 학습과 같은 정의)이고,
행 구성은 `build_math_parquet.make_record` 를 그대로 쓴다(스키마 단일 진실 원천).
길이 게이트: 그 문제의 8 롤아웃 중 **최대** n_tok 이 RESP_LEN(기본 8192)을 넘으면 버린다 —
학습에서 잘릴 문제를 풀에 넣으면 수정 행만 골라 잘라내게 된다.

    python scripts/local/build_revision_pool_parquet.py \
        --rollouts A.jsonl B.jsonl --train-pool <math_train_math_opt.parquet> \
        --eval-parquet <math_eval_L5_800.parquet> --out-dir <DIR> [--max-tok 8192]

런처 기본값은 **건드리지 않는다** — 산출 parquet 은 `DATA_TRAIN=` 으로 넘긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_math_parquet import make_record, norm_problem  # noqa: E402
from src.training.trial2 import agreement_state  # noqa: E402

VARIANT_A = {"DOMINANT"}
VARIANT_B = {"DOMINANT", "SPLIT"}


def load_groups(paths) -> dict:
    """문제 정규화 키 → {problem, gold, finals[], n_tok[]} (여러 파일을 합친다)."""
    out: dict = defaultdict(lambda: {"problem": "", "gold": "", "finals": [], "n_tok": []})
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                k = norm_problem(str(r.get("problem") or ""))
                g = out[k]
                g["problem"] = str(r.get("problem") or "")
                g["gold"] = str(r.get("gold") or "")
                g["finals"].append(str(r.get("final_answer") or ""))
                g["n_tok"].append(int(float(r.get("n_tok") or 0)))
    return dict(out)


def select(groups: dict, states: set, max_tok: int, all_same_frac: float = 0.0) -> tuple[list, dict]:
    """상태 필터 + 길이 게이트. 반환 (선택된 키 리스트, 통계).

    `all_same_frac F` > 0 이면 ALL_SAME 문제를 무작위(seed 0)로 뽑아 **최종 풀의 F 몫**이
    되게 덧붙인다 — 비만장일치만으로 좁힌 풀이 쉬운 문제를 한 개도 안 보는 분포 이동을
    막는다(정확도 회귀 감시용 대조 질량). 길이 게이트는 덧붙이는 쪽에도 똑같이 건다.
    """
    stats = {"by_state": Counter(), "state_kept": 0, "dropped_len": 0, "n_groups": len(groups),
             "all_same_frac": float(all_same_frac), "all_same_added": 0,
             "all_same_dropped_len": 0, "all_same_available": 0}
    kept, easy_pool = [], []
    for k, g in sorted(groups.items()):
        st = str(agreement_state(g["finals"])["state"])
        stats["by_state"][st] += 1
        too_long = max(g["n_tok"] or [0]) > max_tok
        if st not in states:
            if st == "ALL_SAME" and all_same_frac > 0:
                stats["all_same_available"] += 1
                if too_long:
                    stats["all_same_dropped_len"] += 1
                else:
                    easy_pool.append(k)
            continue
        stats["state_kept"] += 1
        if too_long:
            stats["dropped_len"] += 1
            continue
        kept.append(k)
    if all_same_frac > 0 and kept:
        f = min(max(float(all_same_frac), 0.0), 0.95)
        # 최종 풀에서 ALL_SAME 이 F 몫: n_easy = F/(1−F) × n_hard
        n_easy = min(len(easy_pool), int(round(f / (1.0 - f) * len(kept))))
        random.Random(0).shuffle(easy_pool)
        kept += easy_pool[:n_easy]
        stats["all_same_added"] = n_easy
    stats["by_state"] = dict(stats["by_state"])
    stats["kept"] = len(kept)
    return kept, stats


def build(keys, groups, meta: dict) -> pd.DataFrame:
    """선택된 문제 → s3c_all 과 **같은 스키마**의 행(extra_info 에 agree_state/group_pass_rate 포함)."""
    rows = []
    for k in keys:
        g = groups[k]
        m = meta.get(k, {})
        rec = make_record(g["problem"], g["gold"], "math_opt",
                          level=str(m.get("level", "")), subject=str(m.get("subject", "")))
        rec["extra_info"] = {
            "problem": g["problem"], "gold": g["gold"],
            "level": str(m.get("level", "")), "subject": str(m.get("subject", "")),
            "prompt_variant": "math_opt",
            "group_pass_rate": float(m.get("pass_rate", 0.0)),
            "agree_state": str(m.get("state", "")),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--train-pool", required=True, help="level/subject 를 join 할 원 학습 parquet")
    ap.add_argument("--eval-parquet", required=True, help="겹침이 0 이어야 하는 held-out 평가 세트")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tok", type=int, default=8192)
    ap.add_argument("--all_same_frac", type=float, default=0.0,
                    help="ALL_SAME 문제를 최종 풀의 이 몫만큼 덧붙인다(무작위 seed 0, 기본 0=안 섞음)")
    ap.add_argument("--only", default=None, help="한 변형만 만든다(rev_dom|rev_domsplit)")
    ap.add_argument("--suffix", default="", help="출력 파일명 접미(예: _easy20)")
    a = ap.parse_args(argv)

    groups = load_groups(a.rollouts)
    pool = pd.read_parquet(a.train_pool)
    meta = {}
    for _, r in pool.iterrows():
        ei = dict(r["extra_info"])
        meta[norm_problem(str(r["problem"]))] = {"level": ei.get("level", ""),
                                                 "subject": ei.get("subject", "")}
    # pass rate + 상태는 롤아웃에서(학습 텔레메트리와 같은 재료)
    for k, g in groups.items():
        st = agreement_state(g["finals"])
        meta.setdefault(k, {})
        meta[k]["state"] = str(st["state"])
        meta[k]["pass_rate"] = 0.0     # r_corr 은 아래에서 채운다
    for p in a.rollouts:
        acc: dict = defaultdict(list)
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    acc[norm_problem(str(r.get("problem") or ""))].append(
                        float(r.get("r_corr") or 0))
        for k, v in acc.items():
            if k in meta and v:
                meta[k]["pass_rate"] = sum(v) / len(v)

    ev = pd.read_parquet(a.eval_parquet)
    ev_keys = {norm_problem(str(p)) for p in ev["problem"]}

    os.makedirs(a.out_dir, exist_ok=True)
    out_stats = {"max_tok": a.max_tok, "rollout_files": list(a.rollouts),
                 "eval_parquet": a.eval_parquet, "variants": {}}
    for name, states in (("rev_dom", VARIANT_A), ("rev_domsplit", VARIANT_B)):
        if a.only and name != a.only:
            continue
        keys, st = select(groups, states, a.max_tok, all_same_frac=a.all_same_frac)
        overlap = sorted(set(keys) & ev_keys)
        st["eval_overlap"] = len(overlap)
        # ★평가 세트와 한 문제도 겹치면 안 된다 — 조용히 넘어가지 말고 즉사시킨다.
        assert not overlap, f"{name}: held-out 평가 세트와 {len(overlap)} 문제가 겹친다"
        df = build(keys, groups, meta)
        path = os.path.join(a.out_dir, f"math_train_mixed_{name}{a.suffix}.parquet")
        df.to_parquet(path, index=False)
        st["path"] = path
        st["rows"] = int(len(df))
        out_stats["variants"][name] = st
        print(f"[revision_pool] {name}: rows={len(df)} states={sorted(states)} "
              f"kept={st['kept']} dropped_len={st['dropped_len']} overlap={len(overlap)} -> {path}",
              flush=True)
    _sname = f"revision_pool_stats{a.suffix}.json"
    with open(os.path.join(a.out_dir, _sname), "w", encoding="utf-8") as fh:
        json.dump(out_stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out_stats["variants"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
