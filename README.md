# metacognition-math

**한 줄 요약**: Qwen3-8B-Base가 풀이 도중 `<|meta|>` 블록으로 메타인지를 외재화하도록
학습하고, 그 메타 구간에 모델 자신의 gold/decoy 로그확률 신호를 증류하는
**메타인지적 자기증류(metacognitive self-distillation)** 로 OOD(어려운 도메인,
AIME)에 강건한 수학 추론을 만든다.

North-star 가설: 메타인지 행동(막힘 감지, 가정 점검, 접근 전환, 검산)은
in-distribution 정답 암기보다 분포 밖 문제에서 더 강건하게 일반화한다. 목표
지표는 정확도이고, calibration(ECE/Brier/과신율)은 보조 지표로 항상 함께
측정한다.

현행 실험은 **RQ3 매치드 래더**(2026-07-11~)다. 진짜 `Qwen/Qwen3-8B-Base` 위에서
선언된 축만 다른 3-arm 매치드 RL을 돌린다:

```
B0:  Qwen3-8B-Base → no-meta gold SFT(models/b0_gold_sft)        → RL(VANILLA_GRPO, correctness-only)
B2:  Qwen3-8B-Base → meta SFT(models/b23_rv_unmasked_sft)        → RL(VANILLA_GRPO, correctness-only)
B3:  Qwen3-8B-Base → 같은 meta SFT(models/b23_rv_unmasked_sft)   → RL(region-split, pmi_shift만 활성·다른 meta head 전부 0)
```

과학 질문: **RQ1 = B2 − B0**(메타 SFT 효과), **RQ2 = B3 − B2**(메타 보상을
correctness→pmi_shift로 교체한 효과; 순수 격리는 B2-R arm(region-split,
meta advantage=0) 추가 예정).

## 현재 결과 (PRELIMINARY — 단일 시드·진행 중, 최종 판정 아님)

- **RQ1(B2−B0)**: 매칭 val 3점 **+0.151(gs25) / +0.164(gs50) / +0.189(gs75)**,
  9개 데이터셋 전부 양성.
- **RQ2(B3−B2)**: gs25 **+0.042** 한 점. 어려운 과목에 집중 —
  intermediate_algebra +0.125, counting +0.089, precalculus +0.081;
  쉬운 gsm8k는 -0.02.
- B3 게이트(gs25) 통과: meta_emit 0.89 · pmishift_attempted 0.40 · n_save 7 ·
  acc_with 0.70 vs without 0.28. 단 RL 중 meta emission이 0.89→0.54로 침식 중
  (answer 스팬만 correctness를 받는 구조적 압력; acc_with≫without은 유지) —
  관찰 중.
- 최종 판정은 gs300 held-out 1030문제 비교이며 아직 열려 있다. 이 시점의 모든
  숫자는 PRELIMINARY로 취급할 것.

### pre-rq3(instruct) 세대 결과 (보존)

이전 세대는 instruct `Qwen/Qwen3-8B` 기반 2-arm(meta=pmishift vs
base_matched)이었다. 당시 pmishift arm이 same-step 비교에서
`val-aux/*/correctness/mean@1` 기준 8/8 도메인 리드였고, held-out T1에서
matched-base 대비 6/6 셀 유의 승리(MATH500 +18.8pp 등, 단일 시드·패키지
효과)를 기록했다. 이 숫자들은 instruct 세대에만 유효하며 현행 rq3 래더와
섞지 않는다.

## 5분 재현 가이드

```bash
git clone https://github.com/iamseungpil/metacognition-math && cd metacognition-math
cp .env.example .env                          # HF_TOKEN / GH_TOKEN / WANDB_API_KEY 채우기
source experiments/common/load_secrets.sh     # .env 로드 + placeholder 검사

# held-out 1030 eval (GSM8K 500 + MATH-500 500 + AIME 30), vLLM, 논문 프로토콜
python scripts/eval_vllm_1030.py \
    --model_path <merged_ckpt_dir> --model_name my_eval \
    --output_dir results/eval_1030_my_eval/ \
    --max_tokens 16384 --temperature 0.7 --num_samples 8 --seed 42

# RL 학습 (MSR 클러스터, amlt) — rq3 매치드 래더 3-arm
set -a; source .env; set +a
amlt run h100std_rq3_b0.yaml rq3-b0-<날짜> -d "B0 no-meta baseline RL"
amlt run h100std_rq3_b2.yaml rq3-b2-<날짜> -d "B2 meta-SFT + vanilla GRPO"
amlt run h100std_rq3_b3.yaml rq3-b3-<날짜> -d "B3 region-split pmi_shift"
# (SFT init 재생성이 필요하면: h100std_sft_b0_gold.yaml / h100std_sft_b23_unmasked.yaml)
```

데이터 parquet은 HF dataset `iamseungpil/metacot`, RL 체크포인트는
`iamseungpil/metacot-h200-triobj-dcpo-v3`를 경유해 릴레이된다. 자세한 배선은
아래 experiments 가이드 참조.

## 실험 가이드 → [experiments/README.md](experiments/README.md)

**현행 연구 질문(rq3 매치드 래더): RQ1 = B2 − B0(메타 SFT 효과), RQ2 = B3 −
B2(pmi_shift 메타 보상 효과).** 구세대(pre-rq3) RQ1–4 넘버링(효과/분해/층화/
calibration, T1–T4)은 experiments/README.md 참조 — 같은 "RQ" 표기가 세대마다
다른 뜻으로 쓰여 왔으니 혼동 주의. 폴더 구조(science/infra 분리), 실행 예시
3종(SFT/eval/RL), 협업자 트랙 A(클러스터 학습)/B(분석, GPU 불필요)/C(SFT v2
데이터)/D(집필·사이트) 온보딩이 전부 거기 있다. **새로 온 사람은 그 문서부터
읽는다.**

## 지표 규약 요약 (전문은 experiments/README.md 3절)

1. 학습 중 정확도는 wandb `val-aux/<ds>/correctness/mean@1`만 — `val-core`/
   `reward`는 메타 shaping 합성 지표라 arm 비교에 쓰면 가짜 격차가 생긴다.
2. 최종 판정은 held-out 1030문제, 채점은 **math_verify** (`check_correctness`는
   버그 문서화됨, 사용 금지).
3. 논문 eval: 16k tokens, avg@8(AIME avg@16), temp 0.7, 두 arm을 같은 job·같은
   seed로.
4. 난이도 층화 정확도 필수 보고 (집계만 보면 Simpson 함정).
5. 메타 방출은 닫힌 `<|meta|>...<|/meta|>` 블록만 센다.

## 설명 사이트

프로젝트 해설 사이트: **https://metacog-explainer.pages.dev** (소스: `docs/site/`)

## 더 보기

- `CLAUDE.md` — 에이전트/데이터 레지스트리
- `NODE_POLICY.md` — AMLT 노드 소유권 규칙
- `scripts/README.md`, `scripts/ANALYSIS_INDEX.md` — 스크립트·분석 산출물 색인
- `docs/`, `archive/`, `legacy/` — 이전 세대 실험(SDC/RLSD/CTSD) 계획과 기록

## 보안

토큰은 **.env에만** 둔다 (gitignore됨). 코드·yaml·문서에 실제 토큰을 절대
커밋하지 않는다 — yaml은 `${HF_TOKEN}` 환경변수 치환만 쓴다.

## 연락처

이승필 — iamseungpil@gmail.com (HF/GitHub: `iamseungpil`)
