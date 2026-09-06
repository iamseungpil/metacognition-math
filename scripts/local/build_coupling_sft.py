#!/usr/bin/env python
r"""CLI — hint 모드 이어쓰기(Task A `gen_continuations.py --modes hint` 출력)에서
"메타(힌트) → 구체적으로 다른 다음 수 → 정답" 결합을 가르치는 SFT parquet 을 뽑는다.

동기(`docs/RESULTS_cd7.md` "같은 자리 인과 검사"). 정책이 실제로 내는 메타는 같은
지점에서 이어쓰기 성공률을 바꾸지 못한다 — 메타와 "구체적으로 다른 다음 수"가
결합돼 있지 않기 때문이다. 이 결합을 가르치려면, **힌트를 받은 채**(가족이 죽었는지,
살아있는 첫수가 뭔지 알고) 이어쓴 궤적 중 실제로 **새로운**(novel) 수를 시도해서(=
힌트를 따라서, followed) 정답까지 간 것(r_corr=1)만 golden demo 로 남기고, 그 데모를
**힌트 없이**(학생이 스스로 이 결합을 재현해야 하므로) 학습시킨다.

파이프라인
  1. `--continuations` (Task A 산출물, mode=hint 행만 있다고 가정하되 방어적으로
     필터한다) 을 읽는다 — continuation/full_text/r_corr/emitted/novel/followed/
     truncated/decision/hint_text 등.
  2. `--sites` (Task A 가 읽은 것과 같은 sites parquet) 을 site_id 로 조인해
     원본(힌트 없는) `prompt`([system,user,assistant-프리픽스])·`prefix`·
     `family_dead`·`nums`·`target` 을 가져온다.
  3. 필터 단계별로 순서대로 걸러 요약표에 남긴다: r_corr==1 → emitted==1 →
     novel==1 → followed==1 → not truncated → (옵션) family_dead==1 인 site 는
     decision=="redirect" 만.
  4. site 당 최대 `--max_per_site`, 문제(nums,target) 당 최대 `--max_per_problem`
     행만 남긴다(둘 다 k_index 오름차순으로 앞에서부터 — 결정적, 재현 가능).
  5. SFT 행 조립: `messages` = [원본 site 의 system·user(힌트 없음), assistant(=
     `full_text` — 힌트 모드에서 `full_text == prefix + continuation` 이 이미
     보장된다, `gen_continuations.build_fed_prefix_text("hint", ...)` 가 프리픽스를
     그대로 돌려주므로)], `wrong_prefix` = site 의 `prefix`, `scenario` = "redirect"
     (`src/training/sft.py::_should_mask_prefix` 가 프리픽스만 loss-mask 하고
     메타/구체 수/정답은 학습하게 만드는 유일한 스위치 — REDIRECT 취급이 정확한
     이유는 이 프리픽스가 "힌트 없이 재현해야 하는, 아직 안 풀린 시작점"이라는 점에서
     REDIRECT SFT 행의 wrong_prefix 와 같은 역할이기 때문이다: 학생이 이 프리픽스
     **자체**를 내도록 배우면 안 되고, 그 뒤에 오는 결합만 배워야 한다).

이 스크립트는 순수 함수(필터·조립)와 I/O(parquet 읽기/쓰기)를 분리한다 — 순수
함수는 `tests/test_build_coupling_sft.py` 가 합성 parquet 없이도 딕셔너리로 검증한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# gen_continuations.build_fed_prefix_text 재사용 — hint 모드의 fed prefix 가
# "prefix 그대로"라는 불변식이 깨지면(예: 나중에 다른 사람이 hint 모드도 donor 처럼
# 뭔가를 덧붙이게 바꾸면) 여기서 즉시 어긋난다(복제 대신 재사용 — 규약).
from scripts.local.gen_continuations import build_fed_prefix_text  # noqa: E402

FILTER_STAGES = ("mode_hint", "r_corr", "emitted", "novel", "followed", "not_truncated",
                 "redirect_for_dead")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 순수 함수 — 행 단위 필터
# ══════════════════════════════════════════════════════════════════════════════

def row_passes_filters(row: Mapping, *, require_redirect_for_dead: bool) -> tuple[bool, str]:
    """행 하나가 필터를 통과하는가. (통과여부, 실패한 첫 단계 이름) — 통과하면
    두번째 값은 `""`.

    단계 순서는 모듈 docstring §3 과 같다 — 어느 단계에서 떨어졌는지가 요약표의
    "단계별 kept/total" 을 만드는 재료다.
    """
    if row.get("mode") != "hint":
        return False, "mode_hint"
    if int(row.get("r_corr") or 0) != 1:
        return False, "r_corr"
    if int(row.get("emitted") or 0) != 1:
        return False, "emitted"
    if int(row.get("novel") or 0) != 1:
        return False, "novel"
    if int(row.get("followed") or 0) != 1:
        return False, "followed"
    if int(row.get("truncated") or 0) != 0:
        return False, "not_truncated"
    if require_redirect_for_dead:
        family_dead = row.get("family_dead")
        is_dead = family_dead is not None and not (
            isinstance(family_dead, float) and family_dead != family_dead  # NaN
        ) and int(family_dead) == 1
        if is_dead and row.get("decision") != "redirect":
            return False, "redirect_for_dead"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. 순수 함수 — site/문제 당 상한
# ══════════════════════════════════════════════════════════════════════════════

def cap_per_key(rows: Sequence[Mapping], key_fn, max_n: int) -> list[Mapping]:
    """`key_fn(row)` 로 묶어 그룹당 최대 `max_n` 개만 남긴다. 그룹 안 순서는 입력
    순서를 그대로 보존한다(호출자가 미리 k_index 오름차순으로 정렬해 결정적으로
    만든다). `max_n <= 0` 이면 상한 없음(전부 통과)."""
    if max_n is None or max_n <= 0:
        return list(rows)
    seen: dict = {}
    out = []
    for row in rows:
        key = key_fn(row)
        n = seen.get(key, 0)
        if n >= max_n:
            continue
        seen[key] = n + 1
        out.append(row)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. 순수 함수 — 필터링된 이어쓰기 한 행 + 원본 site 정보 → SFT 행
# ══════════════════════════════════════════════════════════════════════════════

def build_sft_row(cont_row: Mapping, site_row: Mapping) -> dict:
    r"""SFT 행 하나 조립.

    `messages` = 원본(힌트 없는) site 프롬프트의 system·user + assistant(=full_text).
    `full_text` 는 hint 모드에서 `prefix + continuation` 과 바이트가 같아야 한다 —
    그렇지 않으면(스키마 오염·다른 모드 섞임) 여기서 즉사(fail-loud, 조용히 다른
    값을 쓰지 않는다).
    """
    prefix = site_row["prefix"]
    expected_fed = build_fed_prefix_text("hint", prefix, None)
    if expected_fed != prefix:  # pragma: no cover - hint 불변식이 깨지면 즉시 드러난다
        raise AssertionError("build_sft_row: hint 모드의 fed prefix 불변식이 깨졌다.")
    full_text = cont_row["full_text"]
    if not full_text.startswith(prefix):
        raise ValueError(
            f"build_sft_row: site {cont_row.get('site_id')!r} 의 full_text 가 site 의 "
            f"prefix 로 시작하지 않는다 — hint/other 모드가 섞였을 가능성.")

    original_prompt = list(site_row["prompt"])
    if not original_prompt or original_prompt[-1].get("role") != "assistant":
        raise ValueError("build_sft_row: site 의 prompt 마지막 메시지가 assistant 가 아니다.")
    base_messages = [dict(m) for m in original_prompt[:-1]]     # 힌트 없는 system/user
    messages = base_messages + [{"role": "assistant", "content": full_text}]

    return {
        "site_id": cont_row["site_id"],
        "messages": messages,
        "wrong_prefix": prefix,
        "scenario": "redirect",
        "cut_type": site_row.get("cut_type"),
        "family_dead": site_row.get("family_dead"),
        "decision": cont_row.get("decision"),
        "r_corr": int(cont_row.get("r_corr") or 0),
        "novel": int(cont_row.get("novel") or 0),
        "followed": int(cont_row.get("followed") or 0),
        "n_tokens": int(cont_row.get("n_tokens") or 0),
        "hint_text": cont_row.get("hint_text") or "",
        "nums": list(site_row["nums"]) if site_row.get("nums") is not None else None,
        "target": site_row.get("target"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 순수 함수 — 요약표
# ══════════════════════════════════════════════════════════════════════════════

def summarize_stages(rows: Sequence[Mapping], *, require_redirect_for_dead: bool) -> dict:
    """단계별 kept/total (누적 통과), family_dead 별 최종 kept 분포, 최종 kept 의
    평균 n_tokens."""
    total = len(rows)
    stage_kept = {}
    survivors = list(rows)
    checks = [
        ("mode_hint", lambda r: r.get("mode") == "hint"),
        ("r_corr", lambda r: int(r.get("r_corr") or 0) == 1),
        ("emitted", lambda r: int(r.get("emitted") or 0) == 1),
        ("novel", lambda r: int(r.get("novel") or 0) == 1),
        ("followed", lambda r: int(r.get("followed") or 0) == 1),
        ("not_truncated", lambda r: int(r.get("truncated") or 0) == 0),
    ]
    if require_redirect_for_dead:
        def _redirect_for_dead(r):
            fd = r.get("family_dead")
            is_dead = fd is not None and not (isinstance(fd, float) and fd != fd) and int(fd) == 1
            return (not is_dead) or r.get("decision") == "redirect"
        checks.append(("redirect_for_dead", _redirect_for_dead))

    for name, pred in checks:
        survivors = [r for r in survivors if pred(r)]
        stage_kept[name] = len(survivors)

    by_fam: dict[str, int] = {}
    for r in survivors:
        fd = r.get("family_dead")
        key = "none" if fd is None or (isinstance(fd, float) and fd != fd) else str(int(fd))
        by_fam[key] = by_fam.get(key, 0) + 1

    mean_n_tokens = (sum(int(r.get("n_tokens") or 0) for r in survivors) / len(survivors)
                    if survivors else float("nan"))

    return {
        "total_rows": total,
        "kept_by_stage": stage_kept,
        "final_kept": len(survivors),
        "kept_by_family_dead": dict(sorted(by_fam.items())),
        "mean_n_tokens": mean_n_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. I/O
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--continuations", required=True,
                    help="gen_continuations.py --modes hint 출력 parquet")
    ap.add_argument("--sites", required=True, help="sites_{train,judge}.parquet (원본 프롬프트용)")
    ap.add_argument("--out", required=True, help="SFT parquet 출력 경로")
    ap.add_argument("--max_per_site", type=int, default=2)
    ap.add_argument("--max_per_problem", type=int, default=4)
    ap.add_argument("--require_redirect_for_dead", action="store_true",
                    help="family_dead==1 인 site 는 decision==redirect 인 행만 남긴다")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    cont_df = pd.read_parquet(args.continuations)
    sites_df = pd.read_parquet(args.sites)
    site_lookup = sites_df.set_index("site_id").to_dict(orient="index")

    all_rows = cont_df.to_dict(orient="records")
    for r in all_rows:
        r["family_dead"] = site_lookup.get(r.get("site_id"), {}).get("family_dead")

    summary = summarize_stages(all_rows, require_redirect_for_dead=args.require_redirect_for_dead)

    kept = []
    for r in all_rows:
        ok, _ = row_passes_filters(r, require_redirect_for_dead=args.require_redirect_for_dead)
        if ok:
            kept.append(r)
    kept.sort(key=lambda r: (r["site_id"], int(r.get("k_index") or 0)))
    kept = cap_per_key(kept, lambda r: r["site_id"], args.max_per_site)
    kept = cap_per_key(
        kept, lambda r: (tuple(int(v) for v in site_lookup[r["site_id"]]["nums"]),
                        int(site_lookup[r["site_id"]]["target"])),
        args.max_per_problem)

    sft_rows = [build_sft_row(r, site_lookup[r["site_id"]]) for r in kept]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(sft_rows)
    out_df.to_parquet(out_path, index=False)

    summary["after_caps"] = len(sft_rows)
    summary["max_per_site"] = args.max_per_site
    summary["max_per_problem"] = args.max_per_problem
    summary["require_redirect_for_dead"] = args.require_redirect_for_dead
    summary["out_path"] = str(out_path)

    print(f"[build_coupling_sft] {summary['total_rows']} input rows -> "
          f"{summary['final_kept']} pass filters -> {summary['after_caps']} after caps")
    print("kept_by_stage:")
    for name, n in summary["kept_by_stage"].items():
        print(f"  {name}: {n}")
    print(f"kept_by_family_dead: {summary['kept_by_family_dead']}")
    print(f"mean_n_tokens (post-filter, pre-cap): {summary['mean_n_tokens']:.1f}")
    print(f"wrote {len(sft_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
