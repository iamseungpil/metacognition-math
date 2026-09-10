#!/usr/bin/env python
r"""CLI — **절제(restraint) 보충 코퍼스**. `build_coupling_sft.py` 가 만드는 결합
코퍼스(막힌 자리에서 메타→다른 수→정답)는 **양성만** 담는다. 그래서 그걸로 SFT 한
정책은 "언제 메타/검산을 **안 해야 하는지**"를 한 행도 배우지 못하고, 자리를 가리지
않고 32~99% 행에서 블록을 낸다. 이 스크립트는 그 반대편 절반을 채운다.

근거. `docs/FINDINGS_cd6.md` §건강층 — 이미 멀쩡한(건강) 자리에 블록을 억지로 끼우면
정확도가 −17~−23%p 다(수학 도메인, 여러 재현). 즉 불필요한 메타는 중립이 아니라
**비용**이다. Countdown 의 site 데이터에는 같은 구분이 이미 있다:
`sites_*.parquet::family_dead` (1=막힘/dead, 0=살아있음/healthy).

두 종류 행을 만든다(둘 다 `mode=="meta"` 이어쓰기에서만 뽑는다 — 그래야 site 의
원본 system 프롬프트(메타 지시문 포함)와 **같은 조건**에서 정책이 스스로 고른 행동이
된다. `nometa` 모드는 다른 system 프롬프트로 생성됐으므로 절제의 증거가 못 된다):

  · **kind="restraint"** (건강 자리 절제 양성) — `family_dead==0` ∧ `emitted==0`
    ∧ `r_corr==1` ∧ not truncated. 메타를 **안 내고** 곧장 정답까지 간 이어쓰기.
    `wrong_prefix` = site 의 prefix, `scenario` = "redirect" →
    `src/training/sft.py::_should_mask_prefix` 가 프리픽스만 loss-mask 하고
    "블록 없이 바로 푸는 뒷부분"을 학습시킨다(결합 코퍼스와 바이트 동일한 규약).

  · **kind="decorative"** (건강 자리 불필요 검산 음성) — `family_dead==0` ∧
    `emitted==1` ∧ not truncated ∧ 메타가 프리픽스 **뒤**에 있는 행.
    `wrong_prefix` = `full_text[:</meta> 끝]`, `scenario` = "redirect" →
    같은 마스크 경로가 **장식 블록까지 통째로 loss-mask** 한다. 즉 이 행은
    "이 블록을 내라"를 절대 가르치지 않고, 블록 뒤 풀이만 가르친다.
    ⚠️정직한 한계: 기존 SFT 배선에는 음의 손실이 없다. 이 규약은 장식 블록의
    확률을 **깎지는 못하고**, 다만 그 블록이 올라가지 않게 막을 뿐이다(부정 예시의
    "약한" 형태). 강한 음성이 필요하면 SFT 쪽 배선을 새로 깔아야 한다 — 이
    스크립트는 `sft.py` 를 건드리지 않는다.

출력 스키마는 `data/sites_v4/coupling_sft_v4.parquet` 과 **같은 14 컬럼**에
`kind` 하나만 더한다 → `pd.concat` 으로 그대로 섞을 수 있다(`kind` 는 결합 코퍼스
쪽에서 NaN 이 되며 학습 경로는 이 컬럼을 읽지 않는다).

순수 함수(필터·조립)와 I/O(parquet)를 분리한다 — `tests/test_build_restraint_sft.py`
가 딕셔너리만으로 검증한다.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.local.build_coupling_sft import _is_dead, cap_per_key  # noqa: E402
from src.training.countdown_rewards import parse_meta  # noqa: E402

KINDS = ("restraint", "decorative")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 순수 함수 — 행 단위 필터
# ══════════════════════════════════════════════════════════════════════════════

def _is_alive(family_dead) -> bool:
    """`family_dead==0` 만 살아있는(건강) 자리. None/NaN 은 **탈락**(시도 0회 —
    막힘인지 건강인지 근거가 없으므로 조용히 통과시키지 않는다)."""
    if family_dead is None:
        return False
    if isinstance(family_dead, float) and family_dead != family_dead:  # NaN
        return False
    return not _is_dead(family_dead)


def row_passes_restraint(row: Mapping) -> tuple[bool, str]:
    """건강 자리 절제 **양성**인가. (통과여부, 실패한 첫 단계)."""
    if row.get("mode") != "meta":
        return False, "mode_meta"
    if not _is_alive(row.get("family_dead")):
        return False, "alive_site"
    if int(row.get("emitted") or 0) != 0:
        return False, "no_meta_emitted"
    if int(row.get("r_corr") or 0) != 1:
        return False, "r_corr"
    if int(row.get("truncated") or 0) != 0:
        return False, "not_truncated"
    return True, ""


def row_passes_decorative(row: Mapping, *, require_correct: bool = False) -> tuple[bool, str]:
    """건강 자리 **불필요 검산 음성**인가.

    `require_correct=True` 면 정답에 도달한 행만 남긴다 — 뒷부분(풀이)을 학습시키는
    행이므로 기본값은 True 가 안전하지만, 장식 블록 자체를 마스크하는 게 목적이라
    호출자가 고를 수 있게 둔다. 메타 끝 오프셋이 프리픽스 안(또는 없음)이면 탈락:
    그런 행은 "프리픽스 뒤에 붙은 장식" 이라는 전제가 성립하지 않는다."""
    if row.get("mode") != "meta":
        return False, "mode_meta"
    if not _is_alive(row.get("family_dead")):
        return False, "alive_site"
    if int(row.get("emitted") or 0) != 1:
        return False, "meta_emitted"
    if int(row.get("truncated") or 0) != 0:
        return False, "not_truncated"
    if require_correct and int(row.get("r_corr") or 0) != 1:
        return False, "r_corr"
    if meta_end_after_prefix(row.get("full_text"), row.get("prefix")) is None:
        return False, "meta_after_prefix"
    return True, ""


def meta_end_after_prefix(full_text, prefix) -> int | None:
    r"""`full_text` 안에서 `</meta>` 가 끝나는 문자 오프셋 — 단, 그 블록이 site
    프리픽스 **뒤**에서 시작할 때만. 아니면 None."""
    text = full_text or ""
    prefix = prefix or ""
    if not text.startswith(prefix):
        return None
    m = parse_meta(text, "new")
    start, end = m.get("start"), m.get("end")
    if start is None or end is None:
        return None
    if int(start) < len(prefix):
        return None
    return int(end)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 순수 함수 — SFT 행 조립
# ══════════════════════════════════════════════════════════════════════════════

def build_restraint_row(cont_row: Mapping, site_row: Mapping, *, kind: str) -> dict:
    r"""coupling_sft_v4 와 같은 컬럼 + `kind`. `messages` = site 의 원본 프롬프트
    (system·user) + assistant(=full_text). `wrong_prefix` 는 kind 에 따라 다르다
    (restraint = prefix, decorative = prefix + 장식 블록까지)."""
    if kind not in KINDS:
        raise ValueError(f"build_restraint_row: unknown kind {kind!r}")
    prefix = site_row["prefix"]
    full_text = cont_row["full_text"]
    if not full_text.startswith(prefix):
        raise ValueError(
            f"build_restraint_row: site {cont_row.get('site_id')!r} 의 full_text 가 "
            f"site 의 prefix 로 시작하지 않는다 — 다른 모드가 섞였을 가능성.")

    if kind == "decorative":
        end = meta_end_after_prefix(full_text, prefix)
        if end is None:
            raise ValueError(
                f"build_restraint_row: site {cont_row.get('site_id')!r} 는 프리픽스 "
                f"뒤에 </meta> 가 없다 — decorative 행으로 만들 수 없다.")
        wrong_prefix = full_text[:end]
    else:
        wrong_prefix = prefix

    original_prompt = list(site_row["prompt"])
    if not original_prompt or original_prompt[-1].get("role") != "assistant":
        raise ValueError("build_restraint_row: site 의 prompt 마지막 메시지가 assistant 가 아니다.")
    messages = [dict(m) for m in original_prompt[:-1]] + [
        {"role": "assistant", "content": full_text}]

    return {
        "site_id": cont_row["site_id"],
        "messages": messages,
        "wrong_prefix": wrong_prefix,
        "scenario": "redirect",     # _should_mask_prefix 가 마스크하는 유일한 값
        "cut_type": site_row.get("cut_type"),
        "family_dead": site_row.get("family_dead"),
        "decision": cont_row.get("decision"),
        "r_corr": int(cont_row.get("r_corr") or 0),
        "novel": int(cont_row.get("novel") or 0),
        "followed": int(cont_row.get("followed") or 0),
        "n_tokens": int(cont_row.get("n_tokens") or 0),
        "hint_text": "",
        "nums": list(site_row["nums"]) if site_row.get("nums") is not None else None,
        "target": site_row.get("target"),
        "kind": kind,
    }


def summarize_kinds(rows: Sequence[Mapping], *, require_correct_decorative: bool) -> dict:
    """kind 별 단계 통과 수(누적)."""
    out: dict = {"total_rows": len(rows)}
    for kind, fn in (("restraint", lambda r: row_passes_restraint(r)),
                     ("decorative", lambda r: row_passes_decorative(
                         r, require_correct=require_correct_decorative))):
        stages: dict = {}
        for r in rows:
            ok, stage = fn(r)
            key = stage or "_pass"
            stages[key] = stages.get(key, 0) + 1
        out[kind] = {"pass": stages.pop("_pass", 0), "rejected_at": dict(sorted(stages.items()))}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. I/O
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--continuations", required=True,
                    help="gen_continuations.py 출력 parquet (mode 컬럼에 'meta' 포함)")
    ap.add_argument("--sites", required=True, help="sites_{train,judge}.parquet")
    ap.add_argument("--out", required=True, help="restraint SFT parquet 출력 경로")
    ap.add_argument("--coupling", default=None,
                    help="(옵션) coupling_sft_v4.parquet — 섞었을 때의 배치 조성을 찍는다")
    ap.add_argument("--max_per_site", type=int, default=2)
    ap.add_argument("--max_decorative", type=int, default=0,
                    help="decorative(음성) 행 상한. 0 이면 만들지 않는다(기본).")
    ap.add_argument("--decorative_any_r_corr", action="store_true",
                    help="decorative 행에서 r_corr==1 요구를 끈다(기본은 정답 행만)")
    ap.add_argument("--min_rows", type=int, default=100,
                    help="이 수 미만이면 **즉사**한다(조용한 빈 코퍼스 금지)")
    ap.add_argument("--sample", type=int, default=5, help="감사용 무작위 표본 행 수")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    cont_df = pd.read_parquet(args.continuations)
    sites_df = pd.read_parquet(args.sites)
    site_lookup = sites_df.set_index("site_id").to_dict(orient="index")

    all_rows = cont_df.to_dict(orient="records")
    for r in all_rows:
        info = site_lookup.get(r.get("site_id"), {})
        r["family_dead"] = info.get("family_dead")
        r["prefix"] = info.get("prefix")

    require_correct = not args.decorative_any_r_corr
    summary = summarize_kinds(all_rows, require_correct_decorative=require_correct)

    kept_r = [r for r in all_rows if row_passes_restraint(r)[0]]
    kept_r.sort(key=lambda r: (r["site_id"], int(r.get("k_index") or 0)))
    kept_r = cap_per_key(kept_r, lambda r: r["site_id"], args.max_per_site)

    kept_d: list = []
    if args.max_decorative > 0:
        kept_d = [r for r in all_rows
                  if row_passes_decorative(r, require_correct=require_correct)[0]]
        kept_d.sort(key=lambda r: (r["site_id"], int(r.get("k_index") or 0)))
        kept_d = cap_per_key(kept_d, lambda r: r["site_id"], 1)[:args.max_decorative]

    sft_rows = ([build_restraint_row(r, site_lookup[r["site_id"]], kind="restraint")
                 for r in kept_r]
                + [build_restraint_row(r, site_lookup[r["site_id"]], kind="decorative")
                   for r in kept_d])

    if len(sft_rows) < args.min_rows:
        raise SystemExit(
            f"[build_restraint_sft] FATAL: 사용 가능한 행이 {len(sft_rows)} 개뿐이다 "
            f"(--min_rows {args.min_rows}). 단계별 탈락: {summary}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sft_rows).to_parquet(out_path, index=False)

    n_sites = len({r["site_id"] for r in sft_rows})
    print(f"[build_restraint_sft] {len(all_rows)} input rows -> {len(sft_rows)} rows "
          f"({len(kept_r)} restraint / {len(kept_d)} decorative) over {n_sites} sites")
    for kind in KINDS:
        print(f"  {kind}: {summary[kind]}")

    # 배치 조성 — 결합 코퍼스와 섞었을 때의 비율.
    if args.coupling:
        n_c = len(pd.read_parquet(args.coupling))
        tot = n_c + len(sft_rows)
        print(f"batch composition if concatenated with {args.coupling}: "
              f"coupling(redirect) {n_c} ({n_c / tot:.1%}) + restraint {len(kept_r)} "
              f"({len(kept_r) / tot:.1%}) + decorative {len(kept_d)} "
              f"({len(kept_d) / tot:.1%}) = {tot}")

    # 감사 표본 — 사람이 눈으로 읽는 용도(사전등록 §감사 규약).
    rng = random.Random(args.seed)
    for i, row in enumerate(rng.sample(sft_rows, min(args.sample, len(sft_rows)))):
        asst = row["messages"][-1]["content"]
        tail = asst[len(row["wrong_prefix"]):]
        print(f"\n───── audit sample {i + 1} [{row['kind']}] site={row['site_id']} "
              f"family_dead={row['family_dead']} r_corr={row['r_corr']} ─────")
        print(f"  MASKED wrong_prefix (…tail 200자): ...{row['wrong_prefix'][-200:]!r}")
        print(f"  TRAINED continuation (앞 400자): {tail[:400]!r}")
        print(f"  contains <meta> in trained span: {'<meta>' in tail}")
    print(f"\nwrote {len(sft_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
