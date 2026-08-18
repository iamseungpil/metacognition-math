"""트랙 1 — 학습 중 redirect/verify 에 무슨 일이 있었나. 생성 0회, 기존 롤아웃만 읽는다.

왜 이것부터인가. 우리 의도문은 "좋은 메타인지 **행동**이 정답률을 높일 때 그 행동을 보상"인데,
실측은 `decision: redirect` 가 2.9% -> 0.2% 로 사라졌다. 그리고 step 5 에서 redirect 의
R_meta 가 **−0.385** 였다. 보상이 우리가 원한 행동에 벌을 주고 있었을 수 있다.

가설 (기전까지 적는다). PMI-shift 는 메타 **경계**에서 믿음을 잰다:

    shift = PMI(메타 뒤) − PMI(메타 앞)

`verify` 는 "이대로 가되 확인하자" 라 그 순간 믿음이 흔들리지 않는다.
`redirect` 는 "이 길을 버리자" 라 **그 순간에는 현재 답의 믿음이 무너지고**, 새 길의 성과는
아직 안 나왔다. 즉 **redirect 의 보상은 나중에 오는데 PMI-shift 는 너무 일찍 잰다.**

  -> 예측 1  R_meta(redirect) < R_meta(verify), 전 구간
  -> 예측 2  그런데 **결과**로 보면 redirect 가 손해가 아니어야 한다
             (R_corr | redirect  vs  R_corr | verify)
  -> 예측 3  둘의 간극이 클수록 "측정 창이 이르다" 가설이 강해진다

예측 2 가 뒤집히면(=redirect 가 결과로도 나쁘면) 가설 기각이고, 보상은 정직하게
나쁜 행동에 벌을 준 것이 된다.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics as st

DEC = re.compile(r"decision:\s*(verify|redirect)", re.I)
META = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.S)


def rows_of(path):
    d = json.load(open(path))
    C = {c: i for i, c in enumerate(d["columns"])}
    for r in d["data"]:
        t = r[C["main_tail"]] or ""
        m = META.findall(t)
        dec = None
        if m:
            g = DEC.search(m[0])
            dec = g.group(1).lower() if g else None
        yield {
            "step": r[C["step"]], "group": r[C["group"]], "dec": dec,
            "R_meta": r[C["R_meta"]], "R_corr": r[C["R_corr"]],
            "R_cal": r[C["R_cal"]], "conf": r[C["conf"]],
            "fmt": r[C["fmt_class"]],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rq3v2g_b4p2")
    ap.add_argument("--every", type=int, default=10, help="몇 스텝마다 볼까")
    ap.add_argument("--out", default="track1.json")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    run = [r for r in api.runs("gistdslab/metacot-dcpo-v4") if r.name == args.run][0]
    files = [f for f in run.files() if f.name.startswith("media/table/dcpo/rollouts_")]
    num = lambda f: int(f.name.split("rollouts_")[1].split("_")[0])
    want = sorted({n for n in map(num, files) if n % args.every == 0 or n in (1, 5)})
    print(f"[0] 테이블 {len(files)} 중 {len(want)} 스텝을 본다: {want[:8]}...{want[-3:]}")

    out = []
    for n in want:
        f = next(f for f in files if num(f) == n)
        p = f.download(root=f"./t1/{n}", replace=True, exist_ok=True).name
        rs = list(rows_of(p))
        meta_rows = [r for r in rs if r["dec"]]
        if not meta_rows:
            continue
        by = collections.defaultdict(list)
        for r in meta_rows:
            by[r["dec"]].append(r)

        def agg(k, key):
            v = [r[key] for r in by[k] if r[key] is not None]
            return round(st.mean(v), 4) if v else None

        row = {"step": n, "n_meta": len(meta_rows)}
        for k in ("verify", "redirect"):
            row[f"{k}_share"] = round(len(by[k]) / len(meta_rows), 4)
            row[f"{k}_R_meta"] = agg(k, "R_meta")
            row[f"{k}_R_corr"] = agg(k, "R_corr")
            row[f"{k}_n"] = len(by[k])
        out.append(row)
        print(f"  step {n:>3}  redirect {row['redirect_share']*100:5.2f}% (n={row['redirect_n']:3d})"
              f"  R_meta v/r {row['verify_R_meta']:+.3f}/{str(row['redirect_R_meta']):>7}"
              f"  R_corr v/r {row['verify_R_corr']:+.3f}/{str(row['redirect_R_corr']):>7}")

    json.dump(out, open(args.out, "w"), indent=2)

    # ── 판정: 예측 1·2·3
    def col(k):
        return [r[k] for r in out if r.get(k) is not None]
    rm_v, rm_r = col("verify_R_meta"), col("redirect_R_meta")
    rc_v, rc_r = col("verify_R_corr"), col("redirect_R_corr")
    print("\n=== 판정 ===")
    if rm_r:
        print(f"예측1  R_meta  verify {st.mean(rm_v):+.4f}  vs  redirect {st.mean(rm_r):+.4f}"
              f"   차이 {st.mean(rm_v)-st.mean(rm_r):+.4f}   "
              f"{'보상이 redirect 에 벌을 준다' if st.mean(rm_r) < st.mean(rm_v) else '아니오'}")
    if rc_r:
        print(f"예측2  R_corr  verify {st.mean(rc_v):+.4f}  vs  redirect {st.mean(rc_r):+.4f}"
              f"   차이 {st.mean(rc_v)-st.mean(rc_r):+.4f}   "
              f"{'★결과로는 손해가 아니다 -> 측정 창 가설 지지' if st.mean(rc_r) >= st.mean(rc_v) - 0.05 else '결과로도 나쁘다 -> 가설 기각'}")
    sh = [r["redirect_share"] for r in out]
    print(f"추세    redirect 점유율  {sh[0]*100:.2f}% -> {sh[-1]*100:.2f}%")
    print(f"        (전반 {st.mean(sh[:len(sh)//2])*100:.2f}%  후반 {st.mean(sh[len(sh)//2:])*100:.2f}%)")


if __name__ == "__main__":
    main()
