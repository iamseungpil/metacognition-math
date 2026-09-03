"""probe_monitor — cd6 프로브 30분 보고용 진행 상황 요약기 (CPU 전용, 수초 완료).

인자 없이 실행하면 stdout 으로 한 번에:
  1. stageA 진행 (로그 마지막 줄 · stageA_shard*/sites_shard* parquet 행수)
  2. stageA 결과 요약 (조건별 정답률 · Δ̂=p_orig−p_ctrl 분포/히스토그램 · post=1 슬라이스)
  3. stageB 진행/결과 (동일 요령)
  4. held-out 평가 진행 ($W/cd6_work/eval/R2H_* 의 telemetry.json → never_rate/acc 표)
  5. GPU 한 줄 + 디스크 여유

파일이 없거나 절반만 있어도 죽지 않고 «미도달» 로 표기한다.

실행: HF_HUB_OFFLINE=1 PYTHONPATH=$REPO $PY $REPO/scripts/probe_monitor.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

W = "/home/jovyan/beomi/splee"
OUT = Path(f"{W}/cd6_work/probe")
EVAL = Path(f"{W}/cd6_work/eval")
LOGS = f"{W}/cd6_work/logs"

MISS = "«미도달»"


def hr(title: str):
    print(f"\n══ {title} " + "═" * max(0, 60 - len(title)))


# ── 1·3 공용: 로그/파케 진행 ────────────────────────────────────────────────
def last_meaningful_line(path: str, tail_bytes: int = 20000) -> str:
    """tail 에서 마지막 의미 있는 줄. tqdm 은 \r 로 덮어쓰므로 \r 도 줄바꿈 취급."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        return f"(읽기 실패: {e})"
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", raw) if ln.strip()]
    if not lines:
        return "(빈 로그)"
    # tqdm/단계 문구 우선 — 뒤에서부터 진행표시가 있는 줄을 찾는다.
    pat = re.compile(r"%\||it/s|it\]|est\. speed|\[stage|harvest|요청|wrote|Traceback|Error",
                     re.I)
    for ln in reversed(lines):
        if pat.search(ln):
            return ln[:200]
    return lines[-1][:200]


def show_logs(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        print(f"  로그: {MISS} ({patterns[0]})")
        return
    for f in files:
        print(f"  log {os.path.basename(f):<32} {last_meaningful_line(f)}")


def parquet_rows(path: str):
    """행수만 필요하므로 metadata 로 센다 (빠름). 실패 시 pandas fallback."""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        try:
            import pandas as pd
            return len(pd.read_parquet(path))
        except Exception:
            return None


def show_parquets(pattern_list, label: str):
    files = sorted({f for p in pattern_list for f in glob.glob(p)})
    if not files:
        print(f"  {label}: {MISS}")
        return []
    for f in files:
        n = parquet_rows(f)
        n_s = f"{n:,}행" if n is not None else "행수 읽기 실패"
        print(f"  {label} {os.path.basename(f):<32} {n_s}")
    return files


def load_concat(files):
    import pandas as pd
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  (경고: {os.path.basename(f)} 읽기 실패 — {e})")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ── 2: 조건별 정답률 + Δ̂ 분포 ──────────────────────────────────────────────
def cond_table(df):
    """cond → (평균정답률, n). df 는 [site_id, cond, correct] 를 가정."""
    print(f"  {'cond':<6}{'p(correct)':>12}{'n_rows':>9}{'n_sites':>9}")
    for cond, g in df.groupby("cond"):
        print(f"  {cond:<6}{g['correct'].mean():>12.4f}{len(g):>9,}"
              f"{g['site_id'].nunique():>9,}")


def site_deltas(df, sites=None):
    """site_id 별 p_orig − p_ctrl. sites(=sites_shard 병합)가 있으면 post 를 붙인다."""
    g = df.groupby(["site_id", "cond"])["correct"].mean().unstack("cond")
    if "orig" not in g.columns or "ctrl" not in g.columns:
        return None
    d = (g["orig"] - g["ctrl"]).dropna().rename("delta").reset_index()
    if sites is not None and "post" in sites.columns:
        d = d.merge(sites[["site_id", "post"]].drop_duplicates("site_id"),
                    on="site_id", how="left")
    return d


def dist_summary(vals, indent="  "):
    import numpy as np
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    if n == 0:
        print(indent + f"Δ̂: {MISS} (표본 0)")
        return
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    k = max(1, n // 10)
    print(indent + f"Δ̂=p_orig−p_ctrl  n={n}  mean={v.mean():+.4f}  "
                   f"Q1={q1:+.3f} med={q2:+.3f} Q3={q3:+.3f}  "
                   f"bot10%평균={v[:k].mean():+.4f}  top10%평균={v[-k:].mean():+.4f}")


def text_hist(vals, bins=20, width=40, indent="  "):
    import numpy as np
    v = np.asarray(vals, float)
    if len(v) == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    cnt, edges = np.histogram(v, bins=bins, range=(lo, hi))
    peak = max(1, cnt.max())
    for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
        bar = "█" * max(0, round(c / peak * width))
        print(indent + f"[{e0:+.2f},{e1:+.2f}) {c:>5} {bar}")


def stage_results(stage_files, sites_df, stage_name: str):
    df = load_concat(stage_files)
    if df is None or not len(df):
        print(f"  {stage_name} 결과: {MISS}")
        return
    need = {"site_id", "cond", "correct"}
    if not need <= set(df.columns):
        print(f"  {stage_name}: 예상 컬럼 {need} 없음 — 실제 {list(df.columns)[:8]}")
        return
    cond_table(df)
    d = site_deltas(df, sites_df)
    if d is None or not len(d):
        print(f"  Δ̂: {MISS} (orig/ctrl 쌍 없음)")
        return
    if "post" in d.columns:
        clean = d[d["post"] != 1]
        post = d[d["post"] == 1]
    else:
        clean, post = d, d.iloc[0:0]
    print(f"  ── Δ̂ 분포 (post=1 제외, site={len(clean)}) " + "─" * 20)
    dist_summary(clean["delta"])
    text_hist(clean["delta"])
    if len(post):
        print(f"  ── post=1 슬라이스 (site={len(post)}) " + "─" * 20)
        dist_summary(post["delta"], indent="    ")
    else:
        print("  post=1 슬라이스: 표본 없음")


# ── 4: held-out 평가 ────────────────────────────────────────────────────────
def heldout():
    dirs = sorted(glob.glob(str(EVAL / "R2H_*")))
    if not dirs:
        print(f"  R2H_* 폴더: {MISS} ({EVAL}/R2H_*)")
        return
    rows = []
    for d in dirs:
        t = Path(d) / "telemetry.json"
        name = os.path.basename(d)
        if not t.exists():
            rows.append((name, None))
            continue
        try:
            j = json.loads(t.read_text())
        except Exception as e:
            rows.append((name, f"telemetry 파싱 실패: {e}"))
            continue
        rows.append((name, {
            "acc": j.get("acc"),
            "never": (j.get("rescue") or {}).get("never_rate"),
            "emit": j.get("emit_rate"),
            "n": j.get("n_problems"),
        }))
    print(f"  {'cell':<22}{'telemetry':>10}{'never_rate':>12}{'acc':>9}"
          f"{'emit':>8}{'n_prob':>8}")
    for name, v in rows:
        if v is None:
            print(f"  {name:<22}{'진행중':>10}{'-':>12}{'-':>9}{'-':>8}{'-':>8}")
        elif isinstance(v, str):
            print(f"  {name:<22}{'오류':>10}  {v}")
        else:
            f4 = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "-"
            print(f"  {name:<22}{'있음':>10}{f4(v['never']):>12}{f4(v['acc']):>9}"
                  f"{f4(v['emit']):>8}{str(v['n'] or '-'):>8}")


# ── 5: GPU / 디스크 ─────────────────────────────────────────────────────────
def gpu_line():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            print(f"  GPU: nvidia-smi 실패 — {r.stderr.strip()[:120]}")
            return
        parts = []
        for ln in r.stdout.strip().splitlines():
            try:
                i, util, mu, mt = [x.strip() for x in ln.split(",")]
                parts.append(f"g{i} {util}% {int(mu)//1024}/{int(mt)//1024}G")
            except ValueError:
                parts.append(ln.strip())
        print("  GPU: " + " · ".join(parts))
    except FileNotFoundError:
        print("  GPU: nvidia-smi 없음")
    except Exception as e:
        print(f"  GPU: 확인 실패 — {e}")


def disk_line():
    try:
        u = shutil.disk_usage(W)
        print(f"  디스크({W}): 여유 {u.free/2**30:.1f}G / 전체 {u.total/2**30:.1f}G "
              f"({u.free/u.total*100:.1f}% free)")
    except Exception as e:
        print(f"  디스크: 확인 실패 — {e}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    # 1. stageA 진행
    hr("1. stageA 진행")
    show_logs([str(OUT / "logs" / "stageA_*.log"), f"{LOGS}/*stageA*.log",
               f"{LOGS}/probe_stageA*.log"])
    a_files = show_parquets([str(OUT / "stageA_shard*.parquet")], "stageA")
    s_files = show_parquets([str(OUT / "sites_shard*.parquet")], "sites ")

    sites_df = load_concat(s_files) if s_files else None
    if sites_df is not None:
        sites_df = sites_df.drop_duplicates("site_id")

    # 2. stageA 결과
    hr("2. stageA 결과")
    if a_files:
        stage_results(a_files, sites_df, "stageA")
    else:
        print(f"  {MISS}")

    # 3. stageB 진행/결과
    hr("3. stageB 진행/결과")
    show_logs([str(OUT / "logs" / "stageB_*.log"), f"{LOGS}/*stageB*.log",
               f"{LOGS}/probe_stageB*.log"])
    b_files = show_parquets([str(OUT / "stageB_shard*.parquet"),
                             str(OUT / "stageB" / "stageB_shard*.parquet")], "stageB")
    if b_files:
        stage_results(b_files, sites_df, "stageB")
    else:
        print(f"  결과: {MISS}")

    # 4. held-out 평가
    hr("4. held-out 평가 (R2H_*)")
    heldout()

    # 5. GPU / 디스크
    hr("5. 자원")
    gpu_line()
    disk_line()
    print()


if __name__ == "__main__":
    sys.exit(main())
