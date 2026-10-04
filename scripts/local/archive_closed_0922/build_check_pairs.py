#!/usr/bin/env python
r"""CLI — 검산(`<check>`) 선호 쌍 parquet 빌드 (DPO 재료).

    python scripts/local/build_check_pairs.py \
        --sites /hdd_data/seungpil/scratch/data/sites_v4/sites_train.parquet \
        --conts /hdd_data/seungpil/scratch/conts_v4/base_train.parquet \
        --out   /hdd_data/seungpil/scratch/data/sites_v4/check_pairs_train.parquet

왜 «구성된» 쌍인가. `scripts/local/chk_pair_probe.py` 실측: 자연 GRPO 그룹(같은 문제
8롤아웃) 안에 `chk_solved` 와 `over_claim`/미검산이 **함께** 있는 그룹은 0.4~5.0%다.
수학 단계의 R18b(자연 발생 메타 대조학습)가 굶은 것과 같은 조건이라, 쌍은 자연 그룹이
아니라 **같은 자리(site)에서 뽑은 K개 이어쓰기**(`scripts/local/gen_continuations.py`)
안에서 만든다 — 같은 프리픽스를 공유하므로 프리픽스가 통제된 대조가 된다.

분류는 `src/training/countdown_rewards.check_row` 를 **그대로 재사용**한다(복제 금지).
자리마다 한 쌍만 낸다:
  chosen  = `chk_solved` → (없으면) `chk_persist` ∧ 정답 → (없으면) 정답 ∧ 검산 있음
  rejected= `over_claim` → (없으면) 검산 없음 ∧ 오답
**양쪽이 실제로 존재할 때만** 쌍을 낸다 — «X 를 무(無)보다 선호» 같은 조작된 쌍은 안 만든다.

CPU 전용. 이 산출물은 아직 어떤 학습 팔에도 연결돼 있지 않다(`src/training/dpo_check.py`
참조 — 트레이너 배선은 미완).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training.countdown_rewards import check_row  # noqa: E402

# 자리 중 쌍이 이 비율 미만이면 굶은 것으로 보고 크게 실패한다(R18b 재발 방지).
MIN_PAIR_RATE = 0.01
MIN_PAIRS = 20


def classify(cont: dict, nums, target) -> dict:
    """이어쓰기 한 행의 검산 라벨. `full_text` 전체를 채점한다(박스가 프리픽스 밖에 있을 수 있다)."""
    cr = check_row(cont.get("full_text") or "", nums, target, int(cont.get("r_corr") or 0))
    cr["r_corr"] = int(cont.get("r_corr") or 0)
    return cr


def _pick_chosen(rows: list[dict]):
    for kind, ok in (
        ("chk_solved", lambda c: c["chk_solved"]),
        ("chk_persist_correct", lambda c: c["chk_persist"] and c["r_corr"]),
        ("correct_with_check", lambda c: c["r_corr"] and c["n_checks"] > 0),
    ):
        hit = [r for r in rows if ok(r["_chk"])]
        if hit:
            return hit[0], kind
    return None, None


def _pick_rejected(rows: list[dict]):
    for kind, ok in (
        ("over_claim", lambda c: c["over_claim"]),
        ("nocheck_wrong", lambda c: c["n_checks"] == 0 and not c["r_corr"]),
    ):
        hit = [r for r in rows if ok(r["_chk"])]
        if hit:
            return hit[0], kind
    return None, None


def build_pairs(sites_rows: list[dict], cont_rows: list[dict]) -> tuple[list[dict], dict]:
    """자리별 선호 쌍을 만든다. 두 입력 모두 dict 리스트(테스트에서 합성 픽스처로 쓴다)."""
    site_by_id = {s["site_id"]: s for s in sites_rows}
    by_site: dict[str, list[dict]] = {}
    n_orphan = 0
    for c in cont_rows:
        sid = c.get("site_id")
        if sid not in site_by_id:
            n_orphan += 1
            continue
        by_site.setdefault(sid, []).append(c)

    pairs, kinds = [], {}
    for sid, rows in by_site.items():
        site = site_by_id[sid]
        nums, target = site["nums"], site["target"]
        for r in rows:
            r["_chk"] = classify(r, nums, target)
        chosen, c_kind = _pick_chosen(rows)
        rejected, r_kind = _pick_rejected(rows)
        if chosen is None or rejected is None or chosen is rejected:
            continue
        kinds[f"{c_kind}|{r_kind}"] = kinds.get(f"{c_kind}|{r_kind}", 0) + 1
        pairs.append({
            "site_id": sid,
            "nums": list(nums),
            "target": int(target),
            "prefix": site.get("prefix") or "",
            "prompt": site.get("prompt"),
            "chosen_text": chosen.get("continuation") or "",
            "rejected_text": rejected.get("continuation") or "",
            "chosen_kind": c_kind,
            "rejected_kind": r_kind,
            "chosen_r_corr": chosen["_chk"]["r_corr"],
            "rejected_r_corr": rejected["_chk"]["r_corr"],
        })
    stats = {
        "n_sites_with_conts": len(by_site),
        "n_pairs": len(pairs),
        "n_orphan_conts": n_orphan,
        "pair_rate": (len(pairs) / len(by_site)) if by_site else 0.0,
        "kinds": kinds,
    }
    return pairs, stats


def audit_print(pairs: list[dict], n: int = 5, seed: int = 0) -> None:
    """무작위 5쌍 대장 기록 — 프리픽스 꼬리 + chosen/rejected 육안 검사용."""
    rng = random.Random(seed)
    for p in rng.sample(pairs, min(n, len(pairs))):
        print("-" * 78)
        print(f"[audit] site={p['site_id']} nums={p['nums']} target={p['target']} "
              f"({p['chosen_kind']} > {p['rejected_kind']})")
        print(f"  prefix…: …{p['prefix'][-240:]}")
        print(f"  CHOSEN  : {p['chosen_text'][:400]}")
        print(f"  REJECTED: {p['rejected_text'][:400]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="사이트 parquet (sites_train.parquet)")
    ap.add_argument("--conts", required=True, help="이어쓰기 parquet (gen_continuations.py 출력)")
    ap.add_argument("--out", default=None, help="출력 parquet (없으면 쓰지 않고 통계만 낸다)")
    ap.add_argument("--mode", default="meta", help="이어쓰기 모드 필터(기본 meta; all 이면 전부)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    sites = pd.read_parquet(args.sites).to_dict("records")
    conts_df = pd.read_parquet(args.conts)
    if args.mode != "all" and "mode" in conts_df.columns:
        conts_df = conts_df[conts_df["mode"] == args.mode]
    conts = conts_df.to_dict("records")
    print(f"[build_check_pairs] sites={len(sites)} conts={len(conts)} (mode={args.mode})")

    pairs, stats = build_pairs(sites, conts)
    print(f"[build_check_pairs] 쌍 {stats['n_pairs']} / 자리 {stats['n_sites_with_conts']} "
          f"({stats['pair_rate']:.1%}), orphan_conts={stats['n_orphan_conts']}")
    print(f"[build_check_pairs] kinds={json.dumps(stats['kinds'], ensure_ascii=False)}")

    if stats["n_pairs"] < MIN_PAIRS or stats["pair_rate"] < MIN_PAIR_RATE:
        raise SystemExit(
            f"[build_check_pairs] FAIL — 쌍이 굶었다: n_pairs={stats['n_pairs']} "
            f"(<{MIN_PAIRS}) 또는 pair_rate={stats['pair_rate']:.3%} (<{MIN_PAIR_RATE:.1%}). "
            "R18b 와 같은 조건이므로 여기서 멈춘다 — 자리/이어쓰기 조성을 먼저 고쳐라.")

    audit_print(pairs, n=5, seed=args.seed)

    if args.out:
        pd.DataFrame(pairs).to_parquet(args.out, index=False)
        print(f"[build_check_pairs] wrote {len(pairs)} pairs -> {args.out}")
    else:
        print("[build_check_pairs] --out 없음 — 통계/감사만 출력하고 저장하지 않았다.")


if __name__ == "__main__":
    main()
