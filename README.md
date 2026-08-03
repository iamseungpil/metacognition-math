# metacognition-math

**메타인지 강화학습으로 수학 추론의 "분포 밖" 일반화를 얻는다.**

모델이 풀이 도중 `<|meta|>…<|/meta|>` 블록으로 자기 상태를 점검하게 만들고(**meta-CoT 형식**),
그 블록을 **모델 자신의 gold-vs-decoy 증거를 얼마나 움직였는가**로 보상한다(**PMI-shift**).
외부 교사는 없다 — 보상을 읽는 참조 모델은 정책의 **동결 사본**이다.

**North-star**: 메타인지 습관은 **습관을 심은 분포 밖**에서 더 강건하게 일반화한다.
습관을 심는 SFT2 코퍼스는 easy 870 · medium 893 · **hard 0건**이므로,
**MATH500 level 4–5 (262문항)가 곧 그 "분포 밖" 축**이다.

> **처음 오셨나요? 이 셋만 읽으면 됩니다.**
> 1. 이 파일 — 무엇을 왜 하는가 · **지금 어디인가**
> 2. [`docs/CLAIMS.md`](docs/CLAIMS.md) — **무엇이 참이고 무엇이 닫혔는가**
> 3. [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md) — 진단 원칙 · 지표 대시보드 · 발사 게이트

---

## 지금 어디인가 (2026-08-03)

| arm | 무엇 | 상태 | 판정 |
|---|---|---|---|
| **b0p** | 메타 제거 쌍둥이 SFT2 + vanilla GRPO (**통제군**) | 🟢 `solid-gibbon` 진행 중 | — |
| **b2p** | 메타 SFT2 + vanilla GRPO (**프라이밍만**) | 🟢 `hip-hound` gs300 재생성 중 | ✅ 정상 — 메타 구조 300스텝 내내 평평 |
| **b3p** | 메타 SFT2 + PMI-shift 7헤드 패키지 | 🔴 완주했으나 **무효** | ❌ **Outcome C** — gs150 이후 발화율 1.00→0.018, **처치 소멸** |

**한 줄**: instruct 기질에서는 방법이 작동한다(검증 완료). **base 기질 복제는 처치가 스스로 사라져 아직 판정 불가.**

## 검증된 것 — 보존 산출물 독립 재채점 (전체는 [`docs/CLAIMS.md`](docs/CLAIMS.md))

| 주장 | 값 |
|---|---|
| **우리 방법 > base SFT+GRPO** | MATH500 **+14.00pp** (p<.001) · AIME +8.75pp · **모든 난이도 레벨에서 유의** |
| **분포 밖에서 더 크다** (north-star) | L1–2 +10.53pp → **L4–5 +17.70pp** · 기울기 **+7.17pp** (p=.009, 잡음바닥 ±3.08) |
| PMI-shift 보상 **단독** 기여 | **+4.38pp** (p=.0001) · 4k·16k 양쪽 · **종결 구제 아님**(종결자만 봐도 +4.46pp) |
| 성분 분해 | 프라이밍 +4.85 · **PMI-shift +4.38** · 나머지 헤드 +4.77 = +14.00 |
| ⚠ 일반화 기울기의 **출처** | **프라이밍**(+8.62, p=.003). RL 보상의 기울기는 0(−1.44, n.s.) |

⚠ 전 arm **단일 학습 시드**. `seed43_*` 파일은 디코딩 시드다.

## 다음 실험

| 순위 | 실험 | 누가 | 상태 |
|---|---|---|---|
| 1 | **b3s** — base `shiftonly` 두 팔(`meta_floor` 0.0 vs 0.05), 통제군 b2p 공유 | 우리 | 승인 대기 |
| 2 | **채점 격자 36셀** — 보존 산출물 독립 재채점 (**GPU 0**) | 협업자 | 착수 가능 |
| 3 | b3p(구) gs300 + b2p gs300 **OOD eval** (L4–5 분할, 응답 로깅) | 우리 | b2p 완주 후 |
| 4 | instruct `shiftonly + gandhi` 재학습 — 삭제된 체크포인트 복원 + **2번째 시드** | 협업자 | 2 이후 |
| 5 | 스케일 축 (4B / 14B) | 협업자 | 1·4 이후 |

## 판정 기준 (사전 선언)

- **주 지표**: MATH500 **L4–5(n=262)**에서 `Δacc(L4–5) − Δacc(L1–2)`
  바닥 0.00 · 천장 +29.97 · **잡음바닥 ±3.08pp** (같은 모델 8샘플 4/4 분할 A-vs-A 실측)
- **개선 인정**: CI 하한 > +3.0pp **그리고** 판정 지점까지 `dcpo/meta_emit_rate ≥ 0.80`
- **⛔ in-training val594는 판정에 쓰지 않는다** — 셀당 21~38문항이라 한 문제가 2.6~4.8pp
- 채점은 **`math_verify`**. `check_correctness`는 버그 문서화됨, 사용 금지
- 논문 eval: 16k tokens · avg@8 (AIME avg@16) · temp 0.7 · 두 arm을 같은 job·같은 seed로
- 난이도 층화 **필수** — 집계만 보면 Simpson 함정

## 저장소 배치

```
core/KNOBS.yaml          하중 노브 등록부 — dcpo_* 85개 전수 (live 38 / default-only 7 / dead 40)
src/                     라이브러리 (학습·보상·평가)
configs/                 Hydra 상속 체인: verl_sdc_e21r_shared → verl_e4_selfdistill → arm leaf
h100std_rq3v2f_*.yaml    라이브 RL 런처 3개
h100std_sft_b*2_rvfull.yaml  그 init을 만든 SFT2 런처 2개
docs/                    CLAIMS · CONSTITUTION · PREREGISTRATION · reports/
archive/                 은퇴한 것 전부 — 각 디렉터리에 "왜 여기 있는지" README
paper/                   논문
```

⚠ **`--keep 1`이 판정 지점 체크포인트를 프루닝합니다.** b3p의 처치 살아있던 gs100–150이
그렇게 사라져 이제 평가할 수 없습니다. 새 런은 `--keep 3` + 판정 지점 명시 보존.

## 재현

```bash
git clone https://github.com/iamseungpil/metacognition-math && cd metacognition-math
cp .env.example .env                        # HF_TOKEN / GH_TOKEN / WANDB_API_KEY
set -a; source .env; set +a

# held-out eval (GSM8K 500 + MATH-500 500 + AIME 30) — 응답 텍스트를 parquet에 남긴다
python scripts/eval_vllm_1030.py \
    --model_path <ckpt_dir> --model_name my_eval --output_dir results/eval_1030_my_eval/ \
    --max_tokens 16384 --temperature 0.7 --num_samples 8 --seed 42

# 학습 (MSR amlt). 순서 필수: SFT2 쌍이 HF에 착지해야 RL이 init을 스테이징한다.
amlt run h100std_sft_b0p2_rvfull.yaml sft2-b0p2-<날짜>   # 통제군
amlt run h100std_sft_b2p2_rvfull.yaml sft2-b2p2-<날짜>   # 메타
# models/{b0p2,b2p2}_rvfull_sft 가 4샤드 착지한 뒤에만:
amlt run h100std_rq3v2f_b0p.yaml rq3v2f-b0p-<날짜>
amlt run h100std_rq3v2f_b2p.yaml rq3v2f-b2p-<날짜>
amlt run h100std_rq3v2f_b3p.yaml rq3v2f-b3p-<날짜>
```

발사 전 판정 기준은 [`docs/PREREGISTRATION_rq3v2_base_replication.md`](docs/PREREGISTRATION_rq3v2_base_replication.md)에 동결돼 있다.

## HF 자산 (전부 PUBLIC)

- 데이터·init 모델·env — [`datasets/iamseungpil/metacot`](https://huggingface.co/datasets/iamseungpil/metacot) · [`datasets/iamseungpil/metacot-rv`](https://huggingface.co/datasets/iamseungpil/metacot-rv)
- RL 체크포인트 — [`iamseungpil/metacot-h200-triobj-dcpo-v3`](https://huggingface.co/iamseungpil/metacot-h200-triobj-dcpo-v3)
- ⚠ **instruct 세대 RL 체크포인트는 삭제됨.** 생성 산출물(`eval/*_1030_v2`)만 보존 —
  **재채점은 가능, 재실행은 불가.** 복원하려면 `models/v8_rv_functional_sft`에서 재학습.

## 규율

실험 설계·발사·판정은 `stacked-research` 스킬을 따른다. 요점 넷:

1. **실험 = 디렉터리 + MANIFEST.** 새 실험은 새 파일이 아니라 **config 델타**다.
2. **발사 전 게이트 G1~G6** 전부 기계 검사 (주장·발화·해상도·통제군·회귀·링크).
3. **CLAIMS 갱신 없이 판정문 금지** — 닫는 것 / 여는 것 / 재확인 계수기.
4. **이동 커밋과 수정 커밋을 절대 섞지 않고, 매 이동 후 회귀 벤치(G5).**

## 더 보기

- [`CLAUDE.md`](CLAUDE.md) — 에이전트·데이터 레지스트리
- [`NODE_POLICY.md`](NODE_POLICY.md) — AMLT 노드 소유권 규칙
- [`docs/CODE_MAP.md`](docs/CODE_MAP.md) — live vs legacy, 호출 사슬, rmeta config-flip 함정
- [`experiments/README.md`](experiments/README.md) — 폴더 구조·협업자 트랙
- 설명 사이트 — https://metacog-explainer.pages.dev (소스 `docs/site/`)
  ⚠ 사이트 수치는 인증 취소된 pre-rq3 세대다. 배너 참조.

## 보안

토큰은 **`.env`에만** 둔다(gitignore됨). 코드·yaml·문서에 실제 토큰을 절대 커밋하지 않는다 —
yaml은 `${HF_TOKEN}` 환경변수 치환만 쓴다.

## 연락처

이승필 — iamseungpil@gmail.com (HF/GitHub: `iamseungpil`)
