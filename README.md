# metacognition-math

**메타인지 강화학습으로 수학 추론의 "분포 밖" 일반화를 얻는다.**

모델이 풀이 도중 `<|meta|>…<|/meta|>` 블록으로 자기 상태를 점검하게 만들고(**meta-CoT 형식**),
그 블록을 **모델 자신의 gold-vs-decoy 증거를 얼마나 움직였는가**로 보상한다(**PMI-shift**).
외부 교사는 없다 — 보상을 읽는 참조 모델은 정책의 **동결 사본**이다.

**North-star**: 메타인지 습관은 **습관을 심은 분포 밖**에서 더 강건하게 일반화한다.
습관을 심는 SFT2 코퍼스는 easy 870 · medium 893 · **hard 0건**이므로,
**MATH500 level 4–5 (262문항)가 곧 그 "분포 밖" 축**이다.

> **처음 오셨나요? 이 넷만 읽으면 됩니다.**
> 1. 이 파일 — 무엇을 왜 하는가 · **지금 어디인가**
> 2. [`docs/PREREGISTRATION_countdown_sc_round.md`](docs/PREREGISTRATION_countdown_sc_round.md) — **현재 라운드**(cd7)의 설계·판정 기준
> 3. [`docs/CLAIMS.md`](docs/CLAIMS.md) — 과거(instruct·base 복제) 세대에서 무엇이 참이고 무엇이 닫혔는가
> 4. [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md) — 진단 원칙 · 지표 대시보드 · 발사 게이트
>
> 아래 본문의 "메타-CoT 형식 + PMI-shift" 서술은 **instruct 세대**(2026-07)의 방법이다.
> 그 방법을 Qwen3-8B-Base로 복제하는 시도, 그리고 그 다음 Countdown 과제로 전환해 모델
> 내부 신호로 메타를 가려내려는 시도(cd6)는 모두 막혔다 — 자세한 경위는
> [`docs/POSTMORTEM_cd6_rulers_2026-09-03.md`](docs/POSTMORTEM_cd6_rulers_2026-09-03.md).

---

## 지금 어디인가 (2026-09-04 — cd7, Countdown SC 라운드)

instruct 세대(§검증된 것)와 Qwen3-8B-Base 복제(§아래 한 줄)는 **모두 과거 라운드**다.
base 복제가 처치 소멸로 막힌 뒤, 과제를 Countdown(다중해 산술 탐색)으로 바꾸고 "모델
속 신호로 좋은 메타를 가려내는 자(ruler)" 25개를 검증했으나 **전부 탈락**했다
(`docs/POSTMORTEM_cd6_rulers_2026-09-03.md`). 지금 도는 것은 그 다음 수:
모델 **자신의** 신호(막힘·과신·행동)만으로 메타인지를 보상하는 **cd7 SC 라운드**다.

- **머신**: 로컬 H100 80GB × 4 (GPU 0~3), 클러스터가 아니라 `scripts/local/` 큐.
- **팔**: N0(맨 GRPO, 메타 없음) · A(메타 요구, 무채점) · **SC**(막힘→탐색·과신→검증
  보상) · G(길이 위약) + SC_GH(정답 항 뺀 굿하트 압력시험, 20스텝, 학습 주장 미사용).
- **규모**: Qwen3-4B, SFT 없음, GRPO 100스텝, 판정 지점 30/50/100, 씨앗 1~3.
- **상태**: **결과 0건** — 아래는 학습 시작 전 gs0 기준선.

| 프롬프트 | 정답률 | 발화율 | 발화 행 중 stuck | early | hi | novel | checked |
|---|---|---|---|---|---|---|---|
| plain (N0 기준) | 0.426 | 0 | — | — | — | — | — |
| new (SC 기반) | 0.390 | 0.398 | 0.869 | 0.031 | 0.297 | 0.047 | 0.414 |
| p3 (⛔ 부적합 — 메타가 시도 전) | 0.099 | 0.882 | 0.033 | 0.960 | 0.029 | 0.831 | 0.228 |

new에서 explore 발동률 0.0145(발화 행 기준) — RL이 20스텝 안에 이걸 못 올리면
"침묵 항" 판정(사전등록 §5). 전체 설계·판정 밴드는
[`docs/PREREGISTRATION_countdown_sc_round.md`](docs/PREREGISTRATION_countdown_sc_round.md).

**한 줄**: instruct 기질에서는 방법이 작동했다(아래 검증됨). **하지만 base 기질
복제는 프라이밍이 널(C-026), 우리 보상 패키지는 통제군보다 음수(C-029)였고**, 그 다음
Countdown으로 과제를 바꿔 모델 내부 신호로 메타를 가려내려 한 시도도 25개 전부
탈락했다(cd6 postmortem). 지금 검증 중인 것은 근거-진리 없이 모델 **자신의** 신호만으로
메타인지를 보상해도 정답률이 오르는가이며, 아직 답이 없다.

## 검증된 것 — 보존 산출물 독립 재채점 (전체는 [`docs/CLAIMS.md`](docs/CLAIMS.md))

⚠ **아래는 instruct 세대(2026-07) 결과다.** 같은 방법을 Qwen3-8B-Base로 복제하자
프라이밍은 널(+0.18pp, C-026), 보상 패키지는 음수(−2.48pp, C-029)였다 — instruct
이득이 기질 고유의 성질이었을 가능성이 있다는 뜻. Countdown 자 탐색(cd6)도 전멸했다
(위 "지금 어디인가" 참조). 아래 숫자를 "현재 방법이 낸 결과"로 인용하지 말 것.

| 주장 | 값 |
|---|---|
| **우리 방법 > base SFT+GRPO** | MATH500 **+14.00pp** (p<.001) · AIME +8.75pp · **모든 난이도 레벨에서 유의** |
| **분포 밖에서 더 크다** (north-star) | L1–2 +10.53pp → **L4–5 +17.70pp** · 기울기 **+7.17pp** (p=.009, 잡음바닥 ±3.08) |
| PMI-shift 보상 **단독** 기여 | **+4.38pp** (p=.0001) · 4k·16k 양쪽 · **종결 구제 아님**(종결자만 봐도 +4.46pp) |
| 성분 분해 | 프라이밍 +4.85 · **PMI-shift +4.38** · 나머지 헤드 +4.77 = +14.00 |
| ⚠ 일반화 기울기의 **출처** | **프라이밍**(+8.62, p=.003). RL 보상의 기울기는 0(−1.44, n.s.) |

⚠ 전 arm **단일 학습 시드**. `seed43_*` 파일은 디코딩 시드다.

## 판정 기준 (사전 선언 — instruct 세대. cd7 판정 기준은 아래 별도)

- **주 지표**: MATH500 **L4–5(n=262)**에서 `Δacc(L4–5) − Δacc(L1–2)`
  바닥 0.00 · 천장 +29.97 · **잡음바닥 ±3.08pp** (같은 모델 8샘플 4/4 분할 A-vs-A 실측)
- **개선 인정**: CI 하한 > +3.0pp **그리고** 판정 지점까지 `dcpo/meta_emit_rate ≥ 0.80`
- **⛔ in-training val594는 판정에 쓰지 않는다** — 셀당 21~38문항이라 한 문제가 2.6~4.8pp
- 채점은 **`math_verify`**. `check_correctness`는 버그 문서화됨, 사용 금지
- 논문 eval: 16k tokens · avg@8 (AIME avg@16) · temp 0.7 · 두 arm을 같은 job·같은 seed로
- 난이도 층화 **필수** — 집계만 보면 Simpson 함정

cd7의 주/부 지표와 판정 밴드는
[`docs/PREREGISTRATION_countdown_sc_round.md`](docs/PREREGISTRATION_countdown_sc_round.md)
§4·§5에 별도로 동결돼 있다(held-out 500×8 정답률, 판정 30/50/100스텝, σ_run 2.89pp).

## 저장소 배치

```
core/KNOBS.yaml          하중 노브 등록부 — dcpo_* 85개 전수 (live 38 / default-only 7 / dead 40)
src/                     라이브러리 (학습·보상·평가) — 현재 라이브는 src/training/countdown_*.py
scripts/local/           ★현재 실행 경로 — GPU 큐 워커·팔 러너·데이터 빌더·HF 업로드
configs/countdown_6arm.yaml  cd7 RL config (arm은 run_arm.sh 가 CLI 로 고른다)
h100std_rq3v2f_*.yaml    math-DCPO(instruct/base) 라이브 RL 런처 3개 — 클러스터 복구 시에만
h100std_sft_b*2_rvfull.yaml  그 init을 만든 SFT2 런처 2개
docs/                    CLAIMS · CONSTITUTION · PREREGISTRATION · POSTMORTEM · reports/
archive/                 은퇴한 것 전부 — 각 디렉터리에 "왜 여기 있는지" README
                         (예: archive/dead_code_2026_09_04/, archive/launchers_retired_0904/)
paper/                   논문
```

⚠ **`--keep 1`이 판정 지점 체크포인트를 프루닝합니다.** b3p의 처치 살아있던 gs100–150이
그렇게 사라져 이제 평가할 수 없습니다. 새 런은 `--keep 3` + 판정 지점 명시 보존.

## 재현 (cd7, 로컬 H100)

```bash
source scripts/local/env.sh                      # .env 로드 + simplerl 활성화 + Ray/vLLM 노브
bash scripts/local/make_data.sh                   # countdown_{train,val}_4num_<variant>.parquet 생성
python scripts/local/gpu_queue.py start-workers 0 1 2 3   # GPU당 워커 1개
python scripts/local/gpu_queue.py submit --name cd7_SC_p3_s0 --priority 10 \
    --cmd "bash scripts/local/run_arm.sh SC 0 100 p3"     # 팔 하나 = 잡 하나 = GPU 한 장
python scripts/local/gpu_queue.py status          # pending/running/done/failed 확인
```

세부 사용법(디스크 가드·HF 업로드 경로·정지)은 [`scripts/local/README.md`](scripts/local/README.md).

math-DCPO(클러스터 amlt) 재현은 아래 "HF 자산" 절 이전 세대 기준으로 여전히 유효하나
**현재 두 amlt VC 모두 제약 상태**다 — `CLAUDE.md`의 Compute 절 참조.

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

## 협업

협업자 온보딩과 실험 요청: [`docs/COLLABORATION_REQUEST.md`](docs/COLLABORATION_REQUEST.md)

## 더 보기

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 세 세대 요약 · 현재 라이브 경로 spine · 모듈 지도
- [`CLAUDE.md`](CLAUDE.md) — 에이전트·데이터·컴퓨트 레지스트리
- [`NODE_POLICY.md`](NODE_POLICY.md) — ⚠ DEPRECATED (pre-rq3 세대 AMLT 노드 소유권 규칙)
- [`docs/CODE_MAP.md`](docs/CODE_MAP.md) — math-DCPO 호출 사슬·rmeta config-flip 함정(§1~8) + Countdown cd7 경로(§9)
- [`scripts/local/README.md`](scripts/local/README.md) — cd7 실행 방법(큐·러너·평가·업로드)
- [`experiments/README.md`](experiments/README.md) — 폴더 구조·협업자 트랙
- 설명 사이트 — https://metacog-explainer.pages.dev (소스 `docs/site/`)
  ⚠ 사이트 수치는 인증 취소된 pre-rq3 세대다. 배너 참조.

## 보안

토큰은 **`.env`에만** 둔다(gitignore됨). 코드·yaml·문서에 실제 토큰을 절대 커밋하지 않는다 —
yaml은 `${HF_TOKEN}` 환경변수 치환만 쓴다.

## 연락처

이승필 — iamseungpil@gmail.com (HF/GitHub: `iamseungpil`)
