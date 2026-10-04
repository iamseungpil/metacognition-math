# archive_closed_0922 INDEX

2026-09-22 정리(A1/A2): scripts/local/ 을 현행 축(gpu_queue/math_rollout/math_revision_eval/
math_pmi_shift_probe/regrade_jsonl/run_math_arm.sh + env.sh/README.md/math_cited_site_gate.py)만
남기고 나머지를 이곳으로 옮겼다. 형식: `파일명 — 서빙한 축 — docs/RESULTS_cd9.md 절(줄) / HYPOTHESIS_LEDGER 절(줄)`.
"원장 언급 없음"은 두 문서 어디에도 파일명이 등장하지 않는다는 뜻(스크립트 자체가 폐기됐거나
언급 없이 쓰였던 유틸일 수 있음).

- build_cf_twins.py — 반사실 짝(counterfactual twin) 데이터 생성 — 원장 언급 없음
- build_check_pairs.py — 점검용 짝 데이터 생성 — 원장 언급 없음
- build_coupling_sft.py — 결합(coupling) SFT 데이터 생성 — 원장 언급 없음
- build_decision_traces.py — 결정 토큰 궤적 데이터 생성 — 원장 언급 없음
- build_gate_sites.py — 관문용 자리(site) 데이터 생성 — 원장 언급 없음
- build_math_dis_parquet.py — M_DIS(재표집 다수결 대조) 학습 parquet 생성 — 원장 언급 없음
- build_math_parquet.py — 기본 수학 학습/평가 parquet 빌더(--forced_frac 등) — RESULTS_cd9.md:174
- build_restraint_sft.py — 자제(restraint) SFT 데이터 생성 — 원장 언급 없음
- build_retry_labels.py — 재시도 판단 라벨 생성 — 원장 언급 없음
- build_revision_pool_parquet.py — 수정(revision) 풀 parquet 빌더 — 원장 언급 없음
- build_revision_traces.py — 자기 수정 성공 궤적 교사 데이터 생성 — RESULTS_cd9.md:868
- build_self_traces.py — 자기 궤적(self trace) 데이터 생성 — 원장 언급 없음
- build_sites.py — 메타 자리(site) 데이터 생성 — 원장 언급 없음
- build_sites_v4.sh — build_sites.py v4 파이프라인 실행기 — 원장 언급 없음
- chain_submit.sh — gpu_queue 잡 연쇄 제출기 — 원장 언급 없음
- check_no_val_overlap.py — 학습/검증 문제 중복 점검 — 원장 언급 없음
- chk_pair_probe.py — chk(check) 짝 탐침 — 원장 언급 없음
- chk_tax_diagnostic.py — chk 분류(taxonomy) 진단 — 원장 언급 없음
- ckpt_keeper.py — 체크포인트 보존/정리 유틸 — 원장 언급 없음
- disk_guard.py — 디스크 용량 감시 유틸 — 원장 언급 없음
- eval_rev_ckpt.sh — 수정(revision) 체크포인트 평가 실행기 — 원장 언급 없음
- gate_judgment.py — 판단(judgment) 관문 계산 — 원장 언급 없음
- gen_continuations.py — 이어쓰기(continuation) 생성 유틸 — 원장 언급 없음
- hf_upload.py — HuggingFace 업로드 유틸 — 원장 언급 없음
- judge_mechanism.py — 판단 메커니즘 분석 — 원장 언급 없음
- ladder_next.sh — 사다리(ladder) 다음 단계 실행기 — 원장 언급 없음
- make_data.sh — 범용 데이터 생성 실행기 — 원장 언급 없음
- make_opt_variant.py — math_opt 프롬프트 변형 생성 — 원장 언급 없음
- math_activation_gate.py — 활성화(activation) 관문 — RESULTS_cd9.md:697
- math_agree_probe.py — M_AGREE 합의 탐침 — 원장 언급 없음
- math_alloc_gate.py — 예산 배분(allocation) 관문 — 원장 언급 없음
- math_anti_teacher_ruler.py — anti-teacher 자(ruler) 측정 — 원장 언급 없음
- math_critique_eval.py — 비평(critique) 분해 평가(first_acc/rescue 등) — RESULTS_cd9.md:344
- math_critique_ig_ruler.py — 비평 정보이득(IG) 자 — 원장 언급 없음
- math_critique_resolve_gate.py — 비평 해소(resolve) 관문 — 원장 언급 없음
- math_decision_eval.py — 결정 토큰 평가 — 원장 언급 없음
- math_diff_eval.py — 캘리브레이션/배분 대조 평가(N=8 표집) — 원장 언급 없음
- math_dis_eval.py — 재표집 다수결 대조 진단 평가 — 원장 언급 없음
- math_disagree_gate.py — 불일치 진단 SELECT/CONTENT 관문 — RESULTS_cd9.md:413
- math_dist_effect.py — 분포 이동(distribution shift) 효과 측정 — 원장 언급 없음
- math_effort_gate.py — 노력(effort) 배분 관문 — 원장 언급 없음
- math_end_probe.py — 종료부 tertile AUC 탐침 — RESULTS_cd9.md:161
- math_geometry_probe.py — 형제 기하(geometry) 탐침(폐기된 우선순위 3 축) — 원장 언급 없음
- math_meta_content_gate.py — 메타 내용(content) 관문 — 원장 언급 없음
- math_meta_probe.py — 메타 발화 일반 탐침 — 원장 언급 없음
- math_plan_gate.py — 계획(plan) 관문 — 원장 언급 없음
- math_protocol_eval.py — 프로토콜(장풀이/게이트 리셋 등) 평가 — RESULTS_cd9.md:761
- math_retry_eval.py — 재시도(첫답/판단/재시도 분해) 평가 — RESULTS_cd9.md:145
- math_ruler_pivot.py — 자(ruler) 피벗 분석 — 원장 언급 없음
- math_site_emission_eval.py — 자리(site) 발화 평가 — 원장 언급 없음
- math_sites.py — 자기 인용 자리 반사실 생성(donor 조건 포함) — RESULTS_cd9.md:67
- math_trial2_eval.py — S3 two-trial 평가 — 원장 언급 없음
- math_uncertainty_ruler.py — 불확실성 자(ruler) — 원장 언급 없음
- math_verify_perform_gate.py — math_verify 성능 관문 — 원장 언급 없음
- meta_content_rulers.py — 메타 내용 자(ruler) 모음 — 원장 언급 없음
- opd_probe.py — OPD(온폴리시 분포) 탐침 — 원장 언급 없음
- promote_smoke.sh — 스모크 통과 시 승격 실행기 — 원장 언급 없음
- restart_worker_when_idle.sh — 유휴 워커 재시작 감시기 — 원장 언급 없음
- ruler_table.py — 여러 자(ruler) 결과 표 합성 — 원장 언급 없음
- run_arm.sh — Countdown 세대 팔 런처(수학 이전) — 원장 언급 없음
- run_arm_retry.sh — run_arm.sh 재시도 변형 — 원장 언급 없음
- run_sft.sh — SFT 학습 실행기 — 원장 언급 없음
- screen_by_rollouts.py — 롤아웃 기준 문제 선별 — 원장 언급 없음
- test_build_sites.py — build_sites.py 단위 테스트 — 원장 언급 없음

## 2026-09-23 추가 이동(수정 17 정리)
mc/·tests/mc 어디서도 import·실행하지 않음을 확인하고 옮겼다(mc/shift_check.py·mc/run.sh 는 주석으로만 언급).
⚠️ 옛 트리 시험 `tests/test_regrade_jsonl.py`·`tests/test_pmi_shift_probe.py`(import) 와
`tests/test_math_retry_sl.py`·`tests/test_trial2_score.py`(run_math_arm.sh 실행)는 옛 경로를 가리킨다 — tests/mc 밖이다.
- run_math_arm.sh — 옛 src.training.math_meta 팔 발사기(M_* 팔) — 현행 발사기는 mc/run.sh
- math_pmi_shift_probe.py — 2턴 PMI SHIFT 탐침 — mc/shift_check.py 가 산술을 옮겨 대체
- regrade_jsonl.py — 옛 jsonl 재채점(.orig 보존) — 현행 재채점은 `python -m mc.eval --regrade_only`
