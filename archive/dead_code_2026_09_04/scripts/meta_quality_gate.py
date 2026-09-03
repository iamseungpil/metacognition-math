"""수확된 메타의 정성 검문 준비 — quality_dump.md 생성 (CPU 전용, GPU 불필요).

meta_splice_probe.py stageA 가 쓴 sites_shard*.parquet 을 모두 읽어,
층화 표집 48개 사이트에 자동 태그를 달고 검토 에이전트가 읽을 마크다운
덤프($OUT/quality_dump.md)를 만든다. 이 덤프는 «수확된 메타가 실제 평가문인가,
프롬프트 지시문의 메아리/일반론 재탕인가»의 품질 통과/불통과 판정 재료다.

층화 표집 48:
  post==1 에서 8 · (post==0 중) pos<0.3 에서 12 · 0.3~0.6 에서 14 · >0.6 에서 14.
  각 층 안에서 (decision, confidence 구간) 그룹을 라운드로빈으로 돌며 뽑아
  confidence·decision 이 다양하게 섞이도록 한다. 표집은 결정적(seed 고정).

자동 태그 (표본만이 아니라 **전체 사이트**에 달아 요약 비율도 계산):
  found_claim  prefix 에서 정답을 찾았다고 주장 (기존 컬럼, 없으면 재계산)
  echo         프롬프트 지시문 지문("one or two sentences" 등)을 그대로 되뇜
  generic      일반론 지문("common strategy" 등)만 있고 문제의 실제 숫자
               (nums·target) 언급이 하나도 없는 경우 — 즉 일반론«만»으로 끝남
  repeat       같은 4단어 구절이 본문에 2회 이상
  concrete     문제의 실제 숫자 nums·target 중 하나 이상을 언급

실행: PYTHONPATH=$REPO HF_HUB_OFFLINE=1 $PY scripts/meta_quality_gate.py
      (선택: --n-samples 48 --out .../quality_dump.md)
입력이 아직 없으면 (stageA 미완료) 친절한 메시지 후 exit 0.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# 단일 정의처 재사용 (복제 금지): 경로 상수 · 결정적 시드 · 정오 판정.
from scripts.meta_splice_probe import OUT, det_seed, rollout_correct  # noqa: E402

# ── 태그용 지문 ────────────────────────────────────────────────────────────
_FOUND_CLAIM = re.compile(r"found|valid solution|correct (?:solution|expression)", re.I)
_ECHO_PHRASES = (
    "one or two sentences",
    "do not solve",
    "do not do arithmetic",
    "metacognitive block",
)
_GENERIC_PHRASES = (
    "common strategy",
    "straightforward way",
    "common operations",
    "standard approach",
    "worth continuing",
)

CTX_CHARS = 120          # 표본별 앞뒤 문맥 길이
N_SAMPLES_DEFAULT = 48
QUOTA = {"post": 8, "early": 12, "mid": 14, "late": 14}   # 층별 표집 수


# ══ 태그 계산 ═══════════════════════════════════════════════════════════════
def body_text(row) -> str:
    b = row.get("meta_body")
    if isinstance(b, str) and b.strip():
        return b
    r = row.get("meta_raw")
    return r if isinstance(r, str) else ""


def tag_concrete(body: str, nums, target) -> int:
    """본문이 문제의 실제 숫자(nums·target) 중 하나 이상을 언급하는가."""
    mentioned = {int(x) for x in re.findall(r"\d+", body)}
    wanted = {int(v) for v in nums} | {int(target)}
    return int(bool(mentioned & wanted))


def tag_repeat(body: str) -> int:
    """같은 4단어 구절이 2회 이상 나오는가."""
    words = re.findall(r"[a-z0-9']+", body.lower())
    if len(words) < 8:
        return 0
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return int(any(c >= 2 for c in grams.values()))


def compute_tags(row) -> list[str]:
    body = body_text(row)
    low = body.lower()
    nums = [int(v) for v in row["nums"]]
    tags = []
    fc = row.get("found_claim")
    if (int(fc) if fc is not None and not (isinstance(fc, float) and np.isnan(fc))
            else int(bool(_FOUND_CLAIM.search(body)))):
        tags.append("found_claim")
    if any(p in low for p in _ECHO_PHRASES):
        tags.append("echo")
    concrete = tag_concrete(body, nums, row["target"])
    # generic = 일반론 지문이 있고, 구체 숫자 언급이 «하나도 없어» 일반론만으로 끝남.
    if any(p in low for p in _GENERIC_PHRASES) and not concrete:
        tags.append("generic")
    if tag_repeat(body):
        tags.append("repeat")
    if concrete:
        tags.append("concrete")
    return tags


# ══ 층화 표집 ═══════════════════════════════════════════════════════════════
def stratum_of(row) -> str:
    if int(row["post"]) == 1:
        return "post"
    p = float(row["pos"])
    if p < 0.3:
        return "early"
    if p <= 0.6:
        return "mid"
    return "late"


def conf_bin(c) -> str:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return "na"
    c = float(c)
    if c < 1 / 3:
        return "lo"
    if c < 2 / 3:
        return "mid"
    return "hi"


def diverse_pick(rows: list[dict], k: int, stratum: str) -> list[dict]:
    """(decision, confidence 구간) 그룹 라운드로빈으로 k개 — 다양성 확보, 결정적."""
    if len(rows) <= k:
        return list(rows)
    groups: dict[tuple, list] = {}
    for r in sorted(rows, key=lambda r: r["site_id"]):
        groups.setdefault((str(r.get("decision")), conf_bin(r.get("confidence"))),
                          []).append(r)
    keys = sorted(groups)
    for gk in keys:                      # 그룹 안 순서는 결정적 셔플
        rng = np.random.RandomState(det_seed("qgate", stratum, *gk))
        rng.shuffle(groups[gk])
    picked = []
    while len(picked) < k and any(groups[gk] for gk in keys):
        for gk in keys:
            if groups[gk] and len(picked) < k:
                picked.append(groups[gk].pop(0))
    return picked


def stratified_sample(recs: list[dict], n_total: int) -> tuple[list[dict], dict]:
    strata: dict[str, list] = {s: [] for s in QUOTA}
    for r in recs:
        strata[r["stratum"]].append(r)
    scale = n_total / sum(QUOTA.values())
    picked, plan = [], {}
    for s, q in QUOTA.items():
        want = max(1, round(q * scale)) if n_total != sum(QUOTA.values()) else q
        got = diverse_pick(strata[s], want, s)
        plan[s] = {"pool": len(strata[s]), "quota": want, "picked": len(got)}
        picked += got
    # 층이 모자라면 남은 사이트에서 결정적으로 채운다 (총 n_total 목표).
    if len(picked) < n_total:
        chosen = {r["site_id"] for r in picked}
        rest = [r for r in recs if r["site_id"] not in chosen]
        fill = diverse_pick(rest, n_total - len(picked), "backfill")
        plan["backfill"] = {"pool": len(rest), "quota": n_total - len(picked),
                            "picked": len(fill)}
        picked += fill
    return picked, plan


# ══ 덤프 작성 ═══════════════════════════════════════════════════════════════
def fence(text: str) -> str:
    """본문에 ``` 가 있어도 안 깨지는 코드펜스."""
    ticks = "`" * max(4, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{ticks}\n{text}\n{ticks}"


def ctx_snip(text: str, lo: int, hi: int) -> str:
    return text[lo:hi].replace("\r", "")


def render(recs, picked, plan, path: Path):
    n = len(recs)
    tag_names = ["found_claim", "echo", "generic", "repeat", "concrete"]
    tag_cnt = Counter(t for r in recs for t in r["tags"])
    per_prob = Counter(r["prob_idx"] for r in recs)
    dist = Counter(per_prob.values())

    L = []
    L.append("# 메타 품질 검문 덤프 (quality gate)\n")
    L.append("검토 에이전트용. 아래 표본을 읽고 «수확된 메타가 상황 평가문인가, "
             "지시문 메아리(echo)/일반론(generic) 재탕인가»로 품질 통과/불통과를 "
             "판정하라. 태그는 자동 부착된 «후보»일 뿐 — 원문을 직접 읽고 판단하라.\n")
    L.append("## 요약 통계\n")
    L.append(f"- 전체 사이트 수: **{n}** (문제 {len(per_prob)}개, "
             f"post=1 {sum(1 for r in recs if r['post'])}개)")
    L.append("- 태그별 비율 (전체 사이트 기준):")
    for t in tag_names:
        c = tag_cnt.get(t, 0)
        L.append(f"  - {t}: {c}/{n} = {c / max(1, n):.1%}")
    L.append("- 문제당 사이트 수 분포 (사이트수: 문제수):")
    for k in sorted(dist):
        L.append(f"  - {k}개: {dist[k]}문제")
    L.append("- 층화 표집 계획/실적 (pool→quota→picked):")
    for s, v in plan.items():
        L.append(f"  - {s}: {v['pool']} → {v['quota']} → {v['picked']}")
    L.append("")

    L.append(f"## 표본 {len(picked)}개\n")
    for i, r in enumerate(picked, 1):
        t = r["response_text"]
        s, e = int(r["meta_start"]), int(r["meta_end"])
        tags = ", ".join(r["tags"]) if r["tags"] else "(없음)"
        L.append(f"### [{i:02d}] {r['site_id']}  · 층={r['stratum']}")
        L.append(f"- nums={list(map(int, r['nums']))} target={int(r['target'])} | "
                 f"이 롤아웃 정오={r['correct']} (문제 8중 {int(r['n_correct_of8'])} 정답) | "
                 f"prefix내 정답식={int(r['solved_in_prefix'])}")
        L.append(f"- conf={r.get('confidence')} dec={r.get('decision')} "
                 f"pos={float(r['pos']):.2f} post={int(r['post'])} "
                 f"meta_len={int(r['meta_len'])}")
        L.append(f"- 태그: {tags}")
        L.append("- 메타 원문:")
        L.append(fence(str(r["meta_raw"])))
        L.append(f"- 앞 문맥 ({CTX_CHARS}자):")
        L.append(fence(ctx_snip(t, max(0, s - CTX_CHARS), s)))
        L.append(f"- 뒤 문맥 ({CTX_CHARS}자):")
        L.append(fence(ctx_snip(t, e, e + CTX_CHARS)))
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


# ══ main ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    ap.add_argument("--out", default=str(OUT / "quality_dump.md"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(OUT / "sites_shard*.parquet")))
    if not files:
        print(f"[meta_quality_gate] 아직 사이트 파일이 없다: {OUT}/sites_shard*.parquet\n"
              "  meta_splice_probe.py stageA 가 끝나면 다시 돌려라. (에러 아님)")
        sys.exit(0)

    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in files],
                   ignore_index=True).drop_duplicates("site_id")
    print(f"[meta_quality_gate] {len(files)}개 샤드에서 사이트 {len(df)}개 로드")

    recs = []
    for r in df.to_dict("records"):
        r["tags"] = compute_tags(r)
        r["stratum"] = stratum_of(r)
        # 이 롤아웃 자체의 정오 (마지막 \boxed 식 채점 — 프로브와 같은 판정기)
        r["correct"] = rollout_correct(r["response_text"],
                                       [int(v) for v in r["nums"]],
                                       int(r["target"]))
        recs.append(r)

    picked, plan = stratified_sample(recs, args.n_samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(recs, picked, plan, out_path)
    print(f"[meta_quality_gate] wrote {out_path} (표본 {len(picked)}개 / 전체 {len(recs)})")


if __name__ == "__main__":
    main()
