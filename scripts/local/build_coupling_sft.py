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
     (옵션, 기본 on) novel==1 → (옵션, 기본 on) followed==1 → not truncated →
     (옵션) site_gain(힌트 성공률 - 무힌트 성공률 >= --min_site_gain, 그리고
     무힌트 성공률 <= --max_base) → (옵션) family_dead==1 인 site 는
     decision=="redirect" 만.

     0907 확장 동기: `novel==1 ∧ followed==1` 을 기본으로 강제하면 hint 전체
     45,504 행 중 1,210(novel)/18,018(followed) 의 교집합만 남아 ~450 행으로
     너무 좁다(hint 모드에서 novel 은 2.7%). "메타가 실제로 도움이 됐는가"를
     novel/followed 라는 **행 단위** 신호가 아니라 **site 단위** 신호
     (`--baseline_conts` 의 무힌트 성공률 대비 힌트 성공률 상승, `--min_site_gain`)
     로 대체할 수 있게 `--no_require_novel`/`--no_require_followed` 를 추가했다.
     이러면 "이 정확한 이어쓰기가 새 수를 썼는가" 대신 "이 site 에서 힌트가
     통계적으로 효과가 있었는가" 를 golden demo 채집 조건으로 쓴다 — 여전히
     r_corr==1(그 이어쓰기 자체는 정답에 도달) ∧ emitted==1(메타를 실제로 냄) 은
     행 단위로 유지한다.
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
                 "site_gain", "redirect_for_dead")


def _is_dead(family_dead) -> bool:
    """`family_dead` 컬럼의 1/0/None/NaN 을 방어적으로 bool 로 정규화한다."""
    return (family_dead is not None
            and not (isinstance(family_dead, float) and family_dead != family_dead)  # NaN
            and int(family_dead) == 1)


# ══════════════════════════════════════════════════════════════════════════════
# 0. 순수 함수 — site 단위 "힌트가 실제로 도움이 됐는가" 게이트
# ══════════════════════════════════════════════════════════════════════════════

def compute_site_success_rate(rows: Sequence[Mapping], *, site_key: str = "site_id",
                               value_key: str = "r_corr") -> dict:
    """`site_key` 로 묶어 `value_key` 의 평균(=성공률)을 낸다.

    행 단위 novel/followed 대신 "이 site 에서 이 조건(힌트 있음/없음)이 통계적으로
    얼마나 잘 통했는가"를 재는 재료 — `compute_site_gain_ok` 가 두 조건(힌트 성공률,
    무힌트 성공률)을 여기서 뽑아 비교한다.
    """
    sums: dict = {}
    counts: dict = {}
    for r in rows:
        key = r[site_key]
        sums[key] = sums.get(key, 0.0) + float(r.get(value_key) or 0)
        counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def compute_site_gain_ok(hint_rates: Mapping[str, float], baseline_rates: Mapping[str, float],
                          *, min_site_gain: float, max_base: Optional[float] = None) -> dict:
    """site 별로 `(hint_rate - baseline_rate) >= min_site_gain` 이고 (옵션)
    `baseline_rate <= max_base` 인지 판정한다.

    baseline 이 없는 site(그 site 가 `--baseline_conts` 의 nometa 모드에 없음) 는
    gain 을 계산할 근거가 없으므로 **보수적으로 탈락**시킨다(조용히 통과시키지
    않는다 — 근거 없는 golden demo 를 만들지 않기 위해).
    """
    out = {}
    for site_id, hint_rate in hint_rates.items():
        base_rate = baseline_rates.get(site_id)
        if base_rate is None:
            out[site_id] = False
            continue
        ok = (hint_rate - base_rate) >= min_site_gain
        if max_base is not None:
            ok = ok and base_rate <= max_base
        out[site_id] = bool(ok)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. 순수 함수 — 행 단위 필터
# ══════════════════════════════════════════════════════════════════════════════

def row_passes_filters(row: Mapping, *, require_redirect_for_dead: bool,
                        require_novel: bool = True, require_followed: bool = True,
                        site_gain_ok: Optional[Mapping[str, bool]] = None) -> tuple[bool, str]:
    """행 하나가 필터를 통과하는가. (통과여부, 실패한 첫 단계 이름) — 통과하면
    두번째 값은 `""`.

    단계 순서는 모듈 docstring §3 과 같다 — 어느 단계에서 떨어졌는지가 요약표의
    "단계별 kept/total" 을 만드는 재료다. `require_novel`/`require_followed` 를
    끄면 해당 단계를 건너뛴다(기본은 기존 동작과 바이트 동일하게 True).
    `site_gain_ok` 를 주면(= `--min_site_gain` 사용) site 가 그 매핑에서 True 여야
    통과한다 — 매핑에 없거나 값이 False 면 탈락.
    """
    if row.get("mode") != "hint":
        return False, "mode_hint"
    if int(row.get("r_corr") or 0) != 1:
        return False, "r_corr"
    if int(row.get("emitted") or 0) != 1:
        return False, "emitted"
    if require_novel and int(row.get("novel") or 0) != 1:
        return False, "novel"
    if require_followed and int(row.get("followed") or 0) != 1:
        return False, "followed"
    if int(row.get("truncated") or 0) != 0:
        return False, "not_truncated"
    if site_gain_ok is not None and not site_gain_ok.get(row.get("site_id"), False):
        return False, "site_gain"
    if require_redirect_for_dead:
        if _is_dead(row.get("family_dead")) and row.get("decision") != "redirect":
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

def summarize_stages(rows: Sequence[Mapping], *, require_redirect_for_dead: bool,
                      require_novel: bool = True, require_followed: bool = True,
                      site_gain_ok: Optional[Mapping[str, bool]] = None) -> dict:
    """단계별 kept/total (누적 통과), family_dead 별 최종 kept 분포, 최종 kept 의
    평균 n_tokens, redirect 비중."""
    total = len(rows)
    stage_kept = {}
    survivors = list(rows)
    checks = [
        ("mode_hint", lambda r: r.get("mode") == "hint"),
        ("r_corr", lambda r: int(r.get("r_corr") or 0) == 1),
        ("emitted", lambda r: int(r.get("emitted") or 0) == 1),
    ]
    if require_novel:
        checks.append(("novel", lambda r: int(r.get("novel") or 0) == 1))
    if require_followed:
        checks.append(("followed", lambda r: int(r.get("followed") or 0) == 1))
    checks.append(("not_truncated", lambda r: int(r.get("truncated") or 0) == 0))
    if site_gain_ok is not None:
        checks.append(("site_gain", lambda r: bool(site_gain_ok.get(r.get("site_id"), False))))
    if require_redirect_for_dead:
        def _redirect_for_dead(r):
            return (not _is_dead(r.get("family_dead"))) or r.get("decision") == "redirect"
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
    redirect_share = (sum(1 for r in survivors if r.get("decision") == "redirect") / len(survivors)
                       if survivors else float("nan"))

    return {
        "total_rows": total,
        "kept_by_stage": stage_kept,
        "final_kept": len(survivors),
        "kept_by_family_dead": dict(sorted(by_fam.items())),
        "mean_n_tokens": mean_n_tokens,
        "redirect_share": redirect_share,
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
    ap.add_argument("--baseline_conts", default=None,
                    help="conts_train_gs0.parquet 같은, mode 컬럼에 'nometa' 를 포함하는 "
                         "이어쓰기 parquet — site 별 무힌트 성공률을 낸다. "
                         "--min_site_gain 을 쓰려면 필수.")
    ap.add_argument("--min_site_gain", type=float, default=None,
                    help="site 의 (힌트 성공률 - 무힌트 성공률) 이 이 값 이상인 site 만 "
                         "남긴다. --baseline_conts 가 필요하다.")
    ap.add_argument("--max_base", type=float, default=None,
                    help="site 의 무힌트 성공률 상한(옵션) — 예: 0.5 로 주면 무힌트로는 "
                         "대체로 실패하는 site 만 남긴다. --min_site_gain 없이는 무시된다.")
    ap.add_argument("--no_require_novel", action="store_true",
                    help="novel==1 필터를 끈다(기본은 켜짐 — 기존 동작과 동일)")
    ap.add_argument("--no_require_followed", action="store_true",
                    help="followed==1 필터를 끈다(기본은 켜짐 — 기존 동작과 동일)")
    args = ap.parse_args()
    if args.min_site_gain is not None and not args.baseline_conts:
        ap.error("--min_site_gain 은 --baseline_conts 없이는 쓸 수 없다.")
    return args


def main() -> None:
    args = parse_args()
    import pandas as pd

    cont_df = pd.read_parquet(args.continuations)
    sites_df = pd.read_parquet(args.sites)
    site_lookup = sites_df.set_index("site_id").to_dict(orient="index")

    all_rows = cont_df.to_dict(orient="records")
    for r in all_rows:
        r["family_dead"] = site_lookup.get(r.get("site_id"), {}).get("family_dead")

    require_novel = not args.no_require_novel
    require_followed = not args.no_require_followed

    site_gain_ok = None
    if args.min_site_gain is not None:
        baseline_df = pd.read_parquet(args.baseline_conts)
        baseline_rows = baseline_df[baseline_df["mode"] == "nometa"].to_dict(orient="records")
        if not baseline_rows:
            raise ValueError(
                f"--baseline_conts {args.baseline_conts!r} 에 mode=='nometa' 행이 없다.")
        baseline_rates = compute_site_success_rate(baseline_rows)
        # hint 성공률은 mode=='hint' 행 전체(필터 이전)로 낸다 — "그 site 에서 힌트를
        # 준 이어쓰기들의 성공률" 이 golden demo 채집 이전에 이미 확정돼야 하므로.
        hint_rows_for_rate = [r for r in all_rows if r.get("mode") == "hint"]
        hint_rates = compute_site_success_rate(hint_rows_for_rate)
        site_gain_ok = compute_site_gain_ok(
            hint_rates, baseline_rates, min_site_gain=args.min_site_gain, max_base=args.max_base)

    summary = summarize_stages(
        all_rows, require_redirect_for_dead=args.require_redirect_for_dead,
        require_novel=require_novel, require_followed=require_followed,
        site_gain_ok=site_gain_ok)

    kept = []
    for r in all_rows:
        ok, _ = row_passes_filters(
            r, require_redirect_for_dead=args.require_redirect_for_dead,
            require_novel=require_novel, require_followed=require_followed,
            site_gain_ok=site_gain_ok)
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
    summary["require_novel"] = require_novel
    summary["require_followed"] = require_followed
    summary["min_site_gain"] = args.min_site_gain
    summary["max_base"] = args.max_base
    summary["n_sites_after_caps"] = len({r["site_id"] for r in kept})
    final_redirect_share = (sum(1 for r in sft_rows if r.get("decision") == "redirect") / len(sft_rows)
                             if sft_rows else float("nan"))
    summary["after_caps_redirect_share"] = final_redirect_share
    summary["after_caps_mean_n_tokens"] = (sum(r.get("n_tokens") or 0 for r in sft_rows) / len(sft_rows)
                                            if sft_rows else float("nan"))
    summary["out_path"] = str(out_path)

    print(f"[build_coupling_sft] {summary['total_rows']} input rows -> "
          f"{summary['final_kept']} pass filters -> {summary['after_caps']} after caps")
    print("kept_by_stage:")
    for name, n in summary["kept_by_stage"].items():
        print(f"  {name}: {n}")
    print(f"kept_by_family_dead: {summary['kept_by_family_dead']}")
    print(f"mean_n_tokens (post-filter, pre-cap): {summary['mean_n_tokens']:.1f}")
    print(f"redirect_share (post-filter, pre-cap): {summary['redirect_share']:.3f}")
    print(f"n_sites_after_caps: {summary['n_sites_after_caps']}")
    print(f"after_caps redirect_share: {summary['after_caps_redirect_share']:.3f}  "
          f"mean_n_tokens: {summary['after_caps_mean_n_tokens']:.1f}")
    print(f"wrote {len(sft_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
