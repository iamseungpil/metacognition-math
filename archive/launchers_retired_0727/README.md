# Retired launchers (moved 2026-07-27)

These were live for part of 0727 and are kept for the record, not for use.

## Why each is here

`a100g1_sft_*` / `a100g1_sft_*_eb16` — the 1-GPU SFT2 path. Created to escape an
11-12h queue for 4-GPU slots, then abandoned when the quota table showed the
A100-80GB **user max is 1 GPU** and, separately, that a 1-GPU run needs ~7.7h
against an observed ~3.5h preemption window. It cannot finish. The `_eb16`
pair additionally exists because the first 1-GPU conversion silently kept
`gradient_accumulation_steps: 4`, which on one process is effective batch 4
rather than T1's 16 — the fix that produced them is the reason they are worth
reading.

`a100g2_sft_*` — the 2-GPU variant. Never submitted; superseded by the same
user-max finding. Note its output paths collide with the live 4-GPU pair, so
running it would overwrite the real artifacts.

`h100std_probe_*` — one-shot gate probes that completed.

## What replaced them

The live path is `h100std_sft_{b0p2,b2p2}_rvfull.yaml` (SFT2 pair) and
`h100std_rq3v2f_{b0p,b2p,b3p}.yaml` (RL), listed in CLAUDE.md. (Updated 2026-08-03:
the a100 twins this section used to name were themselves retired into
`archive/launchers_retired_0803/`.) Nothing here should
be resubmitted; the compute assumptions they were built around no longer hold.

---

## 2026-07-27 2차 이동 — RQ3 think-off 세대 10종

`h100std_rq3_b0/b2/b3/b3_dbg/b3nopmi.yaml`, `h100std_sft_b0_gold/b0p/b2p/b23_unmasked/b2p2_rvseg.yaml`.

이들은 **think-off RQ3 세대**다. 현행은 think-on RQ3v2이며, 두 세대는 arm 이름(B0/B2/B3 vs
b0p/b2p/b3p), init, SFT 단수(1단 vs 2단 스택)가 모두 다르다. 은퇴 사유는 세대 교체만이 아니다 —
구세대 SFT2 필터가 **위장된 시나리오 선택자**로 판명됐다(E-093: `think-closed` 조건이 redirect를
554→67로 걸러냈다). 그러므로 이 런처들로 낸 결과는 복제 결과로 보고할 수 없다.

`h100std_sft_b0p.yaml`·`h100std_sft_b2p.yaml`은 예외적으로 값이 있다: **현행 SFT1 init**
(`b0p_v8base_strict_sft`·`b2p_v8meta_strict_sft`)을 만든 런처이므로 재현 기록으로 보존한다.
현행 SFT2 런처가 게이트 기준을 이 파일에서 인용한다.

이동 시 참조 16건에 아카이브 경로를 부여했고(ARCHITECTURE·SUBMISSION_RUNBOOK·CODE_MAP·
LOCAL_RUN·EXPERIMENT_PLAN·base_rl_recipe·EXPERIMENT_LOG·experiments/README), 살아있는 SFT2
런처 2종의 선례 인용도 갱신했다. `docs/reports/`의 런로그는 역사 기록이므로 손대지 않았다.
