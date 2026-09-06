#!/usr/bin/env python
r"""CLI — OPT_CF(반사실 쌍둥이) 학습 parquet 빌드.

    python scripts/local/build_cf_twins.py \
        --in  $WORK/data/sites_v1/mixed_train_v3c_opt.parquet \
        --out $WORK/data/sites_v1/mixed_train_v3c_cf_opt.parquet

왜 필요한가 (`docs/RESULTS_cd7.md` "같은 자리 인과 검사", 09-07 01:10). 학습 전
정책에서 스스로 낸 메모는 nometa 대비 같은 자리 성공률을 바꾸지 못한다(−0.001,
95% CI 가 0 을 포함) — FT/MT/OPD/OPDC 는 전부 "오라클 타이밍이 맞았는가" 또는
"힌트 교사와 얼마나 가까운가"만 재고, "메모가 이 문제에서 실제로 정답률을
바꿨는가"는 안 잰다. OPT_CF(`src/training/countdown_rewards.py` §8)는 그것을
직접 잰다: 같은 자리(site_id)를, 메타를 허가한 프롬프트(main)와 메타 문장 자체가
없는 plain 프롬프트(twin, 반사실 기준선)로 **둘 다** 학습 배치에 태워, main 의
정답률을 twin 그룹의 평균 정답률과 비교한다.

입력: `scripts/local/build_sites.py` 가 만든 `mixed_train_v3c_opt.parquet`
(site 행 = `extra_info.site_id` 가 비어있지 않은 행, opt 프롬프트 — 메타 허가
문장 포함, `src/training/countdown_task.PROMPT_VARIANTS["opt"]`).

행 변환:
  * site 행(main) 하나마다 **twin** 행 하나를 만든다 — main 과 바이트 동일하되
    시스템 메시지만 `PROMPT_VARIANTS["plain"]`(메타 문장 자체가 없음)으로 바꾸고,
    `extra_info.cf_role="twin"`, `extra_info.cf_key=<site_id>`,
    `extra_info.prompt_variant="plain"` 을 준다. 최상위 `site_id`/`source`/
    `cut_type`/`prefix` 등 site 컬럼은 **main 과 동일하게 유지**한다(twin 은
    "같은 자리, 메타 허가만 뺀 것"이지 "자리가 다른 행"이 아니다) — 다만
    `data_source`/`nums`/`target`/`witness`/`decoy`/`prompt` 의 user·assistant
    부분은 main 과 완전히 같다(오직 system 메시지 + cf_* 필드만 다르다).
  * main 행 자신은 `extra_info.cf_role="main"`, `extra_info.cf_key=<site_id>`
    를 얻는다(다른 것은 안 바뀐다).
  * 일반(normal) 행은 `extra_info.cf_role="none"`, `extra_info.cf_key=""`.

행 순서: **모든 (main, twin) 쌍을 먼저**, 그다음 normal 행을 이어붙인다 — 각
쌍은 정확히 2행이고 짝의 시작 인덱스가 항상 짝수이므로, `data.train_batch_size`
(64, 짝수) 로 순서대로 자르면 어떤 쌍도 배치 경계에 걸리지 않는다(전제: 셔플이
꺼져 있을 때 — `data.shuffle=True` 가 기본이면 verl 이 이 순서를 에폭마다
다시 섞어 쌍이 갈라질 수 있다. `scripts/local/run_arm.sh` 는 ARM=OPT_CF 일 때
`data.shuffle=false` 를 강제한다 — 그 파일의 주석 참조).

CPU 전용.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training import countdown_task as ct  # noqa: E402


def _twin_prompt(prompt, variant: str = "plain"):
    """`prompt`(system/user/assistant 메시지 리스트)에서 system 메시지만
    `PROMPT_VARIANTS[variant]` 로 바꾼 **새** 리스트를 돌려준다. user/assistant
    는 바이트 동일하게 보존한다(자리 프리픽스는 assistant 메시지에 있다).
    """
    msgs = [dict(m) for m in prompt]
    if not msgs or msgs[0].get("role") != "system":
        raise ValueError("prompt[0] 이 system 메시지가 아니다 — 스키마 가정이 깨졌다.")
    msgs[0] = {"role": "system", "content": ct.PROMPT_VARIANTS[variant]}
    return msgs


def build_cf_twins(df):
    """`df`(mixed_train_v3c_opt.parquet 형) 에서 site 행마다 twin 을 하나씩 만들어
    (main, twin) 쌍을 먼저 두고 normal 행을 뒤에 붙인 새 DataFrame 을 돌려준다.
    """
    import pandas as pd

    pair_rows: list[dict] = []
    normal_rows: list[dict] = []
    n_site = 0
    for _, row in df.iterrows():
        r = row.to_dict()
        site_id = r.get("site_id") or ""
        ei = dict(r["extra_info"])
        if site_id:
            n_site += 1
            # ── main ──────────────────────────────────────────────────────
            main = copy.deepcopy(r)
            main_ei = dict(ei)
            main_ei["cf_role"] = "main"
            main_ei["cf_key"] = site_id
            main["extra_info"] = main_ei
            # ── twin: system 메시지만 plain, cf_role/cf_key/prompt_variant 갱신 ──
            twin = copy.deepcopy(r)
            twin["prompt"] = _twin_prompt(list(r["prompt"]), "plain")
            twin_ei = dict(ei)
            twin_ei["cf_role"] = "twin"
            twin_ei["cf_key"] = site_id
            twin_ei["prompt_variant"] = "plain"
            twin["extra_info"] = twin_ei
            pair_rows.append(main)
            pair_rows.append(twin)
        else:
            ei = dict(ei)
            ei["cf_role"] = "none"
            ei["cf_key"] = ""
            r["extra_info"] = ei
            normal_rows.append(r)

    # ★배치 조성(2026-09-07 감사): 쌍을 앞에 몰아두면 shuffle=false 배치 64개가 «자리만」/«일반만」 으로
    #   갈려 OPT_MT(자리 절반) 와 비교가 안 된다. (main, twin, normal, normal) 주기 4 로 엮어 모든 배치가
    #   자리 절반·일반 절반이 되게 한다(64 는 4 의 배수라 쌍이 배치 경계를 넘지 않는다). 남는 쪽은 뒤에 붙인다.
    out_rows = []
    pairs = [pair_rows[i:i + 2] for i in range(0, len(pair_rows), 2)]
    ni = 0
    for pr in pairs:
        out_rows.extend(pr)
        out_rows.extend(normal_rows[ni:ni + 2]); ni += 2
    out_rows.extend(normal_rows[ni:])
    return pd.DataFrame(out_rows), {
        "n_in": len(df), "n_site": n_site, "n_normal": len(normal_rows),
        "n_pairs": n_site, "n_out": len(out_rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="입력 parquet (예: mixed_train_v3c_opt.parquet)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="출력 parquet (예: mixed_train_v3c_cf_opt.parquet)")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.in_path)
    out_df, stats = build_cf_twins(df)
    out_df.to_parquet(args.out_path, index=False)

    print(f"[build_cf_twins] in={args.in_path} n_in={stats['n_in']} "
          f"(site={stats['n_site']}, normal={stats['n_normal']})")
    print(f"[build_cf_twins] out={args.out_path} n_out={stats['n_out']} "
          f"(pairs={stats['n_pairs']} x2 + normal={stats['n_normal']})")


if __name__ == "__main__":
    main()
