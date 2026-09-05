#!/usr/bin/env python
"""판정 자리(sites_judge) 이어쓰기 결과의 기제 표 — 팔·스텝별로 같은 자리 성공률과
«죽은/산 계열» 별 발화율·redirect 비율·새 쌍 시도율·메타 위치·잘림을 한 표로 낸다.

    python scripts/local/judge_mechanism.py OPT_opt_s1_step30 OPT_MT_opt_s1_mixed_step30

입력: $WORK/conts_v1/judge_cd7_<tag>.parquet (ckpt_keeper 가 판정 지점마다 만든다).
사전등록 3차 지표(docs/PREREGISTRATION_countdown_anchored_2x2.md §3)를 그대로 계산한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

WORK = Path(os.environ.get("WORK", "/hdd_data/seungpil/scratch"))


def family_name(v) -> str:
    if v in (1, True, "1", "True", 1.0):
        return "dead"
    if v in (0, False, "0", "False", 0.0):
        return "alive"
    return "none"


def mechanism_row(tag: str, sites: pd.DataFrame) -> dict:
    df = pd.read_parquet(WORK / "conts_v1" / f"judge_cd7_{tag}.parquet").merge(sites, on="site_id")
    df["fam"] = df.family_dead.map(family_name)
    df["em"] = df.emitted.astype(bool)
    emitted = df[df.em]
    row = {"tag": tag, "succ": df.r_corr.mean(), "trunc": df.truncated.mean(),
           "ntok": df.n_tokens.mean(), "emit": df.em.mean()}
    for fam in ("dead", "alive"):
        s, se = df[df.fam == fam], emitted[emitted.fam == fam]
        row[f"emit_{fam}"] = s.em.mean()
        row[f"redir_{fam}"] = (se.decision == "redirect").mean()
        row[f"novel_{fam}"] = se.novel.astype(float).mean()
        row[f"pos_{fam}"] = (se.meta_start_in_cont / se.n_tokens.clip(lower=1)).mean()
        row[f"succ_{fam}_meta"] = se.r_corr.mean()
        row[f"succ_{fam}_nometa"] = s[~s.em].r_corr.mean()
        row[f"trunc_{fam}_meta"] = se.truncated.mean()
    return row


def main(tags: list[str]) -> None:
    sites = pd.read_parquet(WORK / "data/sites_v1/sites_judge.parquet")[["site_id", "family_dead"]]
    table = pd.DataFrame([mechanism_row(t, sites) for t in tags]).set_index("tag").round(3).T
    pd.set_option("display.width", 250)
    print(table.to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["OPT_opt_s1_step30"])
