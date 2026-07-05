# metacognition-math

**한 줄 요약**: Qwen3-8B가 풀이 도중 `<|meta|>` 블록으로 메타인지를 외재화하도록
학습하고, 그 메타 구간에 모델 자신의 gold/decoy 로그확률 신호를 증류하는
**메타인지적 자기증류(metacognitive self-distillation)** 로 OOD(어려운 도메인,
AIME)에 강건한 수학 추론을 만든다.

North-star 가설: 메타인지 행동(막힘 감지, 가정 점검, 접근 전환, 검산)은
in-distribution 정답 암기보다 분포 밖 문제에서 더 강건하게 일반화한다. 목표
지표는 정확도이고, calibration(ECE/Brier/과신율)은 보조 지표로 항상 함께
측정한다.

두 arm은 메타 메커니즘만 빼고 byte-identical하다:

```
meta arm:  Qwen3-8B → SFT-1(v8_meta_inside_strict) → SFT-2(rv_functional)
                    → RL(TRIOBJ_DCPO_V4 + pmi_shift 보상)
base arm:  Qwen3-8B → 같은 데이터의 meta 제거판 SFT → RL(VANILLA_GRPO, correctness-only)
```

## 현재 결과 (PRELIMINARY — 최종 판정 아님)

- pmishift(meta arm)가 same-step 비교에서 `val-aux/*/correctness/mean@1` 기준
  **8/8 도메인 리드** 중.
- 단, 이것은 학습 중 지표다. **최종 판정인 held-out 1030문제 비교(base gs300 vs
  pmishift gs300)는 base arm gs300 체크포인트 확보를 기다리는 중**이라 아직
  열려 있다. 이 시점의 모든 숫자는 PRELIMINARY로 취급할 것.

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

# RL 학습 (MSR 클러스터, amlt) — meta arm / base arm
set -a; source .env; set +a
amlt run h100std_pmishift.yaml pmishift-<날짜> -d "meta arm RL"
amlt run h100std_base_matched_rl.yaml base-matched-<날짜> -d "base arm RL"
```

데이터 parquet은 HF dataset `iamseungpil/metacot`, RL 체크포인트는
`iamseungpil/metacot-h200-triobj-dcpo-v3`를 경유해 릴레이된다. 자세한 배선은
아래 experiments 가이드 참조.

## 실험 가이드 → [experiments/README.md](experiments/README.md)

연구 질문은 4개다 — RQ1 효과(PMI-shift가 정확도를 올리는가, 메인 표 T1),
RQ2 분해·메커니즘(효과는 무엇이며 어디서 오는가, T2), RQ3 난이도·유형·OOD
층화(T3), RQ4 calibration(T4). RQ1–4와 테이블 T1–T4 매핑, 폴더 구조
(science/infra 분리), 실행 예시 3종(SFT/eval/RL), 협업자 트랙 A(클러스터
학습)/B(분석, GPU 불필요)/C(SFT v2 데이터)/D(집필·사이트) 온보딩이 전부 거기
있다. **새로 온 사람은 그 문서부터 읽는다.**

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
