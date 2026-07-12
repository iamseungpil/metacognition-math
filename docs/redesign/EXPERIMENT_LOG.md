# EXPERIMENT LOG — RQ3 매치드 래더 (Qwen3-8B-Base 재설계)

> rq3 재설계의 시행착오를 시간순으로 남긴다. 각 항목은 "무엇이 깨졌고, 어떻게
> 원인을 확정했고, 무엇으로 고쳤는가"를 기록한다. 진단 원칙과 게이트는
> `docs/CONSTITUTION.md`, 현행 레시피는 `docs/redesign/base_rl_recipe.md` 참조.
> 마지막 갱신: 2026-07-12.

## 1. v1 전멸 — base 엔트로피 붕괴 (2026-07-08)

v1 발사분(B0/B2/B3)이 전부 gs50–100에서 엔트로피 붕괴(entropy→0.01, clip→0.9,
response length 717→3582 폭주). **원인 = instruct 세대 하이퍼파라미터를 base
substrate에 그대로 복사**한 것: temp 0.6 + top_k 20, max_response 4096 캡,
GDPO advantage std-정규화. 인과 확정 근거: 동일 레시피의 instruct 쌍둥이 런은
300스텝 완주 → substrate 인과. (SFT는 무죄 — eos 불변식 정상, 코퍼스 4096 초과
0%, gs25까지는 정상이었음.) **처방 = v2 레시피**: temp 1.0 / top_k -1 /
top_p 1.0 / max_response 8192 / max_model_len·max_num_batched_tokens 10240 /
`norm_adv_by_std=false`(Dr.GRPO) / logprob micro-bs 2. SimpleRL-Zoo·Dr.GRPO·
DAPO 문헌의 base-RL 권고와 일치.

## 2. pmi_shift 불발 — 하류 가설 기각, 상류(SFT emission)로 추적 (2026-07-10)

B3 계열에서 pmi_shift가 불발(pmishift_attempted 0.03, n_save 0). 보상 스케일 /
anchor_norm / len_cost가 원인이라는 하류 가설을 먼저 검증 — 전부 **기각**
(anchor_norm은 불발 신호를 증폭하는 증상 증폭기일 뿐, 원인 아님). 진짜 원인은
상류: **base meta-SFT가 under-trained** — 코퍼스는 100% meta인데 RL 시작 시
emission이 ~52%뿐이고 RL이 40%까지 침식(예전 instruct 세대는 99% emission으로
pmi_shift가 정상 발화했던 것과 대조). 메타 보상은 재료(meta 블록)가 없으면
정의상 불발이다.

## 3. 근본 원인 = sft.py wrong_prefix segment-mask → unmask fix (2026-07-10)

emission 저조의 근인을 `src/training/sft.py`에서 확정: **wrong_prefix
segment-mask가 응답 시작 부분을 학습에서 제외**해, 모델이 "응답을 meta로
시작하는 습관" 자체를 못 배움(emission 38%). **fix = b23 parquet의
`wrong_prefix` 필드를 비워 whole-response 학습**(unmask) →
`data/b23_rv_unmasked_sft.parquet`. emission 38% → **92%**.

## 4. B0 baseline 데이터 교체 — 오염된 matched_clean → 공개 HF gold (2026-07-10)

B0(no-meta) SFT 데이터를 기존 `v8_base_matched_clean` 계열에서 **공개 HF gold**
로 교체: gsm8k 637 + MATH 653 = 1,290행(`data/b0_gold_sft.parquet`), RV 문제
부분집합, 정답은 math_verify로 검증. → `models/b0_gold_sft`.

## 5. rq3 3-arm 발사 + B3 gs25 게이트 통과 (2026-07-11)

`h100std_rq3_b0/b2/b3.yaml` 발사 (init = `models/b0_gold_sft` /
`models/b23_rv_unmasked_sft`). **B3 gs25 게이트 통과 확정**:
meta_emit **0.89** (≥0.8) · pmishift_attempted **0.40** (≥0.3) ·
n_save **7** (>0) · acc_with **0.70** ≫ acc_without **0.28**.

## 6. 운영 교훈 (Standard 선점 클러스터)

- **선점/긴 validation을 행(hang)으로 오진하지 말 것.** Standard tier는 선점이
  잦고, in-training validation(594문제)은 ~1시간 걸리며, wandb는 단조 step
  거부로 조용할 수 있다 — 셋 다 "죽은 것처럼" 보인다. **진짜 행 판별법 =
  stdout을 재다운로드해서 줄 수가 불변인지 확인.**
- 진짜 행도 1회 실재했음: **gs40 체크포인트 저장 직후 vLLM 데드락** 1회.
- wandb는 `WANDB_RUN_ID` 고정 + `WANDB_RESUME=allow`로 run 파편화 방지
  (v1에서 18조각으로 파편화됐던 문제의 해결).
- save_freq: B0=5, B2/B3=10. test_freq=25. ckpt는 HF model repo
  `iamseungpil/metacot-h200-triobj-dcpo-v3` 경유 릴레이.

## 7. 예비 결과 (2026-07-12 현재 — 단일 시드·진행 중·**미확정**)

- **RQ1 (B2−B0)**: 매칭 val 3점 **+0.151(gs25) / +0.164(gs50) / +0.189(gs75)**,
  9개 데이터셋 전부 양성.
- **RQ2 (B3−B2)**: gs25 **+0.042** 한 점. 어려운 과목 집중 —
  intermediate_algebra +0.125, counting +0.089, precalculus +0.081;
  쉬운 gsm8k -0.02.
- **B3 meta emission이 RL 중 0.89 → 0.54로 침식 중.** answer 스팬만
  correctness를 받는 region-split의 구조적 압력으로 해석. acc_with ≫
  acc_without은 유지(행동 자체는 건재) — 관찰 지속.
- val은 in-training 594문제(verl_val_meta_mix), reward +1/−1 스케일이므로
  acc = (r+1)/2. 최종 판정(gs300 held-out 1030) 전까지 모든 숫자 PRELIMINARY.

## 8. 열린 항목

- [ ] B3 meta emission 침식(0.89→0.54) 계속 관찰 — 바닥 확인 / emission-floor
      개입 판단.
- [ ] **B2-R arm** 추가 (region-split, meta advantage=0) — RQ2의 순수 격리
      (현행 RQ2=B3−B2는 "region-split 구조 + pmi_shift" 패키지 차이).
- [ ] gs300 완주 → held-out 1030 최종 판정.
