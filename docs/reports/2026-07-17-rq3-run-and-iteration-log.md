# RQ3 실행·반복 로그 — 문제·해결 형식 (2026-07-17 시작)

> **형식 규약** (update-study 스킬 참조). 진행한 모든 실험/사건을 아래에 **append**한다.
> 나중에 이 로그를 논문 §Experiments의 서사로 정리한다.
> 각 엔트리는 세 부분:
> - **추상적 문제 (Abstract Problem)**: 이 단계가 답하려는 일반적·개념적 질문. 구현 세부 없이.
> - **구체적 해결 (Concrete Solution)**: 실제 config/코드/명령까지 내려간 구현.
> - **통찰 (Insight)**: 이 사건이 헌법/논문 주장에 대해 알려준 것. insight가 핵심.
>
> 판정 기준: **held-out 1030(GSM8K500+MATH500+AIME30) at gs300**, base(B0) 대비. 헌법 게이트(Part VI):
> emit≥0.8 · pmishift_attempted≥0.3 · n_save>0 · entropy>0.1 (gs~25). 게이트 실패 = GPU 낭비 전 상류 수정.
> pre-register 해석표(Codex 수렴): B3pkg>B2 & B3pkg>B3-noPMI → 패키지+조건부PMI / B3pkg>B2 & B3pkg≈B3-noPMI → form/패키지, PMI 철회 / 셋≈ → mechanism·form-vs-function 논문 / B3pkg<B2 or noPMI>B3pkg → PMI 유해 음성결과.

## 실험 상태판 (running tally) — 갱신 0727 16:45

> **이 파일을 여는 사람에게**: 아래가 현행이다. 그 다음 절의 0719 상태판은 **은퇴한
> RQ3(think-off) 세대**이며 부록 전용으로 보존된 것이다. 두 판을 섞어 읽지 말 것.

**현행 세대 = RQ3v2 think-on matched ladder.** 그 안에 두 lineage가 있고, 구분이 결과 해석을
좌우한다:

| lineage | RL init | 런처 | 지위 |
|---|---|---|---|
| **rvseg (구)** | `models/b2p2_rvseg_sft` (378행) | `h100std_rq3v2_{b2p,b3p}.yaml` | ⛔**부록 전용**. E-093에서 init이 위장된 시나리오 필터로 redirect 기아(554→67)임이 확정됐다. **복제 결과로 보고 금지.** |
| **rvfull (본선)** | `models/b2p2_rvfull_sft` (미생성) | `h100std_rq3v2f_{b0p,b2p,b3p}.yaml` | 본선. SFT2 쌍이 먼저 필요하다. |

| Arm | exp | lineage | 상태 | durable / 트레이너 스텝 |
|---|---|---|---|---|
| b2p | `inviting-jackal` (basicvc H100x4) | **rvseg(부록)** | running 3일차 | gs220 / **228**·300 (~6.0분/스텝, ~7h) |
| b3p | — | rvseg | **미발사** | — |
| b0p·b2p·b3p (본선) | — | rvfull | **미발사** — 컴퓨트 차단 | — |
| SFT2 쌍 (b0p2·b2p2 rvfull) | — | — | **미발사** — 본선의 선행조건 | — |

> ⚠️ **인프라**: `msrresrchbasicvc`가 0726 05:49부터 신규 제출을 전면 거부(그룹정책 멤버십
> 공백·행정 조치 대기). 카나리아 38회 연속 RED. `msrresrchvc`는 A100-80GB 사용자 한도 1장이라
> 4-GPU 잡이 영원히 스케줄되지 않는다. 그래서 **본선 5종이 전부 대기 중**이고, 지금 도는
> 유일한 잡이 부록 lineage인 것이다.

> **판정 규약**: `docs/PREREGISTRATION_rq3v2_base_replication.md`. 결과 3종(전이 /
> substrate-특이 / **무효런**), 셀별 주장 금지, `format_fair` 채점기 동결.

---

## [은퇴] RQ3(think-off) 세대 상태판 — 0719 07:29 시점, 부록 보존용

| Arm | exp(현재) | 상태 | durable/in-mem gs | 판정 축 |
|---|---|---|---|---|
| B0 (base gold-SFT + vanilla GRPO) | — | gs300 완주(기준선) | 300 | 기준 |
| B2 (meta-SFT + vanilla GRPO) | internal-grub | running·**durability 교착**(선점마다 gs158 리셋) | 158 / ~167 | RQ1 = B2−B0 |
| B3pkg (meta-SFT + full triobj + PMI-shift) | smart-bat | running·**durability 교착**(선점마다 gs43 리셋) | 43 / ~139 | RQ2 = B3pkg−B2 |
| B3-noPMI (패키지 − PMI-shift) | rq3-b3nopmi-2 | **완료**(gs149·wandb) | 149 | 순수 PMI = B3pkg−B3-noPMI |

> ⚠️ **인프라 상태(E-057~E-064)**: HF/Xet 대용량 업로드 차단→두 arm 선점마다 durable anchor로 리셋→**gs300 미도달**. gs300 held-out 미확보. 아래 결과는 **in-training val(gs50/100)** 기준 예비.

### 예비 결과 요약 — held-out val 정확도 9과목 평균 (greedy@1·(rew+1)/2·E-055/059)
| gs | B0 | B2 | B3pkg | B3noPMI | RQ1(B2−B0) | RQ2(B3pkg−B2) | PMI(B3pkg−noPMI) |
|---|---|---|---|---|---|---|---|
| 50 | 56.3 | **64.9** | 60.4 | 50.0 | +8.6 | −4.5 | **+10.4** |
| 100 | 59.2 | **66.6** | 62.4 | 64.1 | +7.4 | −4.2 | **−1.7** |

> **판정(예비)**: **RQ1 견고**(B2>B0 +7~9pp·meta-SFT init이 정확도↑). **RQ2 음성**(B3pkg<B2, gs50 −4.5→gs100 −4.2)=triobj 패키지가 correctness-only 미달. **PMI 이득 소멸/역전**(gs50 +10.4→gs100 −1.7)=emission collapse(91%→4.5%·E-056/058/059)로 메커니즘 사망→PMI 무대 붕괴. ∴pre-register 표의 **"B3pkg<B2 / PMI 철회·음성결과+기전(emission collapse) 서사"** 분기. 처방=E-063 개선실험(emission-protected advantage).

### B0 gs300 held-out 기준선 (비교 anchor, HF v3 `eval/base_matched_1030_v2/`, meta_emission≈0=진짜 base)
| 벤치 | @16k acc | @4k acc |
|---|---|---|
| GSM8K | **0.8905** | 0.8868 |
| MATH500 | **0.5440** | 0.5385 |
| AIME(2024) | **0.0458** | 0.0500 |
> meta arm(B2/B3pkg/B3-noPMI)의 gs300 held-out을 이 수치와 비교해 RQ1/RQ2 판정. (avg@8, AIME는 avg@16 seed변형 포함.)

---


> 📁 **E-000 ~ E-111은 분리·동결됐다** →
> [`2026-07-17-rq3-run-and-iteration-log-part1.md`](2026-07-17-rq3-run-and-iteration-log-part1.md).
> 이 파일은 **E-112 이후**만 담는다. 새 엔트리는 여기 append.

---

## E-112 — 1-GPU 전환이 실효 배치를 4배 줄인 것을 발견·수정, eb16 쌍 제출 (0727 04:30–04:50 UTC)

**근인.** 0727에 A100 4장 큐(11–12h 대기)를 피하려고 SFT2를 1-GPU로 옮기면서
`gradient_accumulation_steps: 4`를 그대로 뒀다. T1의 SFT2는 `bs1 × ga4`를 **4 프로세스**로
돌려 실효 배치가 **16**이었는데(`configs/archive/sft_rv_functional.yaml:20-21` +
`configs/accelerate_sft.yaml:13`), 같은 값이 1 프로세스에서는 실효 배치 **4**가 된다.
스텝 수가 이를 확증한다 — 1763행 × 3ep ÷ 4 = 1322스텝이고 실제 잡은 1212스텝을 보고했다
(길이 필터 후). 배치 16이면 ~330스텝이다. 이 런의 존재 이유가 "T1을 문자 그대로 복제해서
음성 결과를 우리가 만든 편차 탓으로 돌릴 수 없게 하는 것"이므로, 4배 편차는 그 목적을 무효화한다.

**수정.** `configs/sft_{b0p2,b2p2}_rvfull_g1eb16.yaml` — ga 16, 별도 `output_dir`.
메모리는 불변(누적은 마이크로배치를 순차 실행, per-device bs는 1 유지).
두 파일에 불변식을 박아뒀다: **실효 배치는 어느 world size에서도 16**
(1 GPU → ga 16, 2 → 8, 4 → 4), 그리고 **두 arm은 같은 world size에서 돌아야 한다**.

**낡은 텍스트 5건**(sed-clone 함정 5연속). 전수 diff 통독으로만 잡혔다:
1. `_safe` 클론이 물려받은 resume 주석이 `resume_from_checkpoint: auto`를 약속하는데
   그 키는 의도적으로 제거된 상태였다(sft.py:375-377이 None을 반환 → 매번 콜드 스타트).
   런처에 HF에서 체크포인트를 되받는 단계가 없어 resume은 "죽지 않은 노드"에서만 의미가
   있는데 그건 resume이 필요 없는 경우다. 주석을 사실대로 고쳤다 — `save_steps 25`가 실제로
   사는 건 수동 탈출구(시작 15분 내 로컬 ckpt 존재 → 위험 감지 시 손으로 push).
2. b0p2 config 서두 "Dose ... bs1 x ga4" (2곳), 3. b2p2 config 동일 문구,
4. 런처 description의 push 경로가 구 lineage(`models/b0p2_rvfull_sft/`)를 가리킴,
5. b2p2 런처의 "OUTPUT NAME IS NEW (b2p2_rvfull_sft…)".

**충돌 회피(파괴조작 3율).** 신규 잡은 `b{0,2}p2_rvfull_eb16_sft`로 **별도 lineage**를 쓴다.
덕분에 **돌고 있는 ga4 컨트롤을 죽이지 않고** 나란히 큐에 넣을 수 있다 — 기회편승(Basic)
티어에서 슬롯은 희소 자원이므로, 대체가 확보되기 전에는 폐기하지 않는다. ga4 컨트롤은
큐 헤지로만 남기고 eb16이 노드를 잡으면 취소한다.
반면 **큐에만 있던 ga4 메타 잡**(`a100g1-sft-b2p2-rvfull`, compute 미점유)은 즉시 취소했다 —
그건 잘못된 배치로 학습할 뿐 아니라 산출물 이름이 4-GPU 잡(`rq3v2-sft2-rvfull`, 배치 16으로
올바름)과 **충돌**했다.

**타르볼.** asset **491027629** = `metacognition_rq3v2_0727_eb16.tar.gz`,
md5 `570c3e322c9acaee4c867ceb6acab406`. 노드가 쓰는 **동일 asset URL로 되받아** md5 왕복
일치 확인, 비밀 스캔 0건, 필수 12파일 존재 확인, 되받은 타르볼 안 ga=16 재확인.
490894146의 superset(g1eb16 config 2종 + sft.py resolver 추가). 폴백 시 argparse에서 죽으므로
조용히 틀린 배치로 학습할 길은 없다.

**제출 결과** (둘 다 오류 없음):
| 실험 | 잡 | 상태 |
|---|---|---|
| `worthy-snail` | `sft_b2p2_rvfull_g1eb16` (메타) | queued |
| `a100g1-sft-b0p2-eb16` | `sft_b0p2_rvfull_g1eb16` (컨트롤) | queued |

**기타 관측.**
- ga4 컨트롤(헤지): 232/1212, ETA 11h. `save_strategy: epoch`이라 첫 로컬 저장은 step 404.
- b2p: durable **gs140 (model 4 / optim 4 / extra 4 = RESUMABLE)**. 02:45 이후 새 global_step
  커밋 없음. **04:31에 새 wandb run**(`run-20260727_043127`)이 생겨 프로세스가 재시작됐다.
  amlt는 `failed`(1.6 kB)로 표시하고 SSH도 거부하는데 HF 커밋은 계속 도착한다 —
  대리변수 함정 재확인. gs160 도달 여부가 resume 성공의 판정 신호다.
- basicvc 카나리아 **RED 21회째** (동일 `The virtual cluster does not exist`).
- pytest `tests/`: 804 passed, 8 skipped.

### E-113 — b2p resume 성공 확증, 그리고 SSH가 막힌 잡을 관측하는 법 (0727 05:00 UTC)

**정정.** 직전 틱에서 "02:45 이후 gs160 커밋이 없다"를 이상 신호로 다뤘는데, 성급했다.
학습 속도가 **~373 s/it**라 20스텝 간격 체크포인트는 약 2시간에 하나다. gs160은 정상적으로
06:30 UTC 무렵에 온다.

**관측 채널.** `amlt status`가 `failed`(1.6 kB)를 보고하고 `amlt ssh`는
"No running/queued job found"로 거부하며 `amlt logs list`는 Azure artifact API 502/503을
낸다. 그런데 노드의 백그라운드 pusher는 살아 있어서 **wandb 디렉터리를 HF로 계속 올린다**.
`list_repo_tree(..., expand=True)`로 `last_commit.date`를 보면 `wandb/rq3v2_b2p/output.log`가
04:54에 갱신된 것이 보이고, 그 파일을 받으면 학습 stdout을 그대로 읽을 수 있다.
**amlt 3경로가 전부 막힌 잡의 상태를 읽는 유일한 방법**으로 기록해 둔다.

**판정 (output.log 원문).**
```
Found checkpoint: /scratch/checkpoints/rq3v2_b2p/global_step_140
Load from checkpoint folder: /scratch/checkpoints/rq3v2_b2p/global_step_140
Setting global step to 140
Resuming from /scratch/checkpoints/rq3v2_b2p/global_step_140
Training Progress:  48%|████▊  | 143/300 [17:29<14:57:50, 343.13s/it]
```
04:31의 재시작은 **gs140에서 정확히 resume**했다 — gs0 재시작도, 죽은 것도 아니다.
durable이 model/optim/extra 각 4로 완비돼 있던 것이 값을 했다(E-106에서 gs10이 optim 0/4라
resume 불가였던 것과 대비된다). 잔여 157스텝 × ~6분 ≈ 15.7h.

**기타.** ga4 컨트롤(헤지) 254/1212 정상. eb16 두 잡 여전히 queued(14분).
카나리아 **RED 22회째**. eb16 산출물 HF 0파일(정상).

### E-114 — 노드 재활용 수술: 잘못된 배치 런을 죽이고 같은 GPU에서 eb16 컨트롤 기동 (0727 05:41–05:50 UTC)

**왜.** eb16 두 잡이 1시간 가까이 queued였고(Basic=기회편승), 그 사이 노드 하나는 **실효 배치 4**의
쓸모없는 학습을 계속 돌리고 있었다. 산출물이 T1 대조에 못 쓰이므로 기다릴수록 순손실이다.

**⛔ 하마터면 노드를 날릴 뻔한 함정.** 사전 계획은 `pkill -f "sft.py --config"`였다. 실제로
그 패턴이 매치한 첫 PID는 **런처 bash 셸(1877)**이었다 — 셸의 cmdline에 yaml 명령 블록 전문이
들어 있어 `sft.py --config` 문자열을 포함하기 때문이다. 그대로 실행했으면 훈련이 아니라 **잡
자체를 죽여 노드를 잃었다**. 교훈: **원격 노드에서 pkill 패턴 종료 금지. `readlink /proc/<pid>/exe`로
python 프로세스를 특정해 PID로 죽인다.**

실제 프로세스 지도:
| PID | 정체 | 조치 |
|---|---|---|
| 1877 | 런처 bash (`sleep 43200`로 노드 유지) | **보존** |
| 1890 / 1970 / 3236 | gpu_keeper (548–568 MB) | **보존** |
| 3237 | `accelerate.commands.launch` | 종료 |
| 3610 | 실제 트레이너 (36,162 MB) | 종료 |

**절차.** (1) `/scratch/eb16stage`에 asset 491027629를 **별도 전개**(돌고 있는 잡의
`/scratch/metacognition`은 무손상, mtime 02:00:46 확인). (2) SSH 셸에 HF/WANDB/GH 토큰이
살아 있음을 **길이만 출력해** 확인. (3) 3610·3237만 PID 지정 종료 → GPU가 keeper 3종(1.7 GB)만
남고 비었으며 1877 생존 확인. (4) 같은 GPU에서 `configs/sft_b0p2_rvfull_g1eb16.yaml`로 기동.

**판정 신호 — 총 스텝 303.** 배치 4일 때는 1212였다. 303 × 16 = 1212 × 4 = 4848로 동일 샘플
수이며, 실효 배치가 16으로 복원됐음을 산술로 확증한다. 속도도 개선됐다: **~101 s/step × 303 ≈
8.5h**로, 배치 4의 40 s/step × 1212 ≈ 13.5h보다 **빠르다** — CPU 오프로드 Adam 스텝이 병목이라
옵티마이저 스텝 수가 1/4로 줄면 벽시계가 줄어든다.

**push 구멍 메움.** 워크플로가 지적한 "SFT 런처엔 학습 중 push가 없다"를 그대로 두지 않고,
`scripts/push_sft_ckpts_to_hf.py`를 전용 레포 `iamseungpil/metacot-sft2-eb16`(private) 대상으로
`--interval 120 --keep 2` 데몬 기동. `--once` 스모크로 인증·시딩 검증 후 백그라운드 전환.
이제 노드가 죽어도 최근 체크포인트는 HF에 남는다.

**정리.** 큐에 있던 중복 컨트롤 잡 `a100g1-sft-b0p2-eb16` 취소 — 수술로 대체가 **확보된 뒤에**
폐기했다(3율). 메타 arm(`worthy-snail`)은 큐에 유지하며, 슬롯 경쟁이 하나 줄어 유리해졌다.

**현재.** 컨트롤 eb16: 노드 위 running 1/303 → ETA 8.5h, 산출물 `b0p2_rvfull_eb16_sft`.
메타 eb16: queued. b2p: 149/300 정상. 카나리아 RED 24회.

### E-115 — 대리변수 함정 4번째: "진행바 정지 = 멈춤"이 아니다 (0727 06:05 UTC)

b2p의 `Training Progress` 줄이 44분 동안 `149/300 [50:48]`에 고정돼 있어 정지를 의심했다.
HF 위 `wandb/rq3v2_b2p/output.log`의 `last_commit`은 05:55로 신선했으므로 파일은 갱신되고
있었다. **필터를 걷고 원문 꼬리를 보니** 답이 나왔다:

```
test_gen_batch meta info: {... 'validate': True, 'global_steps': 150}
validation generation end                      (수십 회 반복)
```

gs150의 주기 held-out 검증을 돌고 있었다. 검증 배치는 tqdm 학습 진행바를 갱신하지 않으므로
**진행바는 정지처럼 보이지만 잡은 건강하다**. durable이 gs140에 머문 것도 정상이다 — 저장은
20스텝 간격이라 다음이 gs160이다.

교훈: 로그를 볼 때 **내가 찾을 것을 정해 놓고 grep하면 그 밖의 상태는 보이지 않는다**.
`grep "Training Progress"`가 아니라 꼬리 원문을 먼저 읽을 것. 이것으로 대리변수 함정 목록은
네 개가 됐다 — `timing_s/step`(마지막값) · wandb `crashed` · `amlt status=failed` · **정지한 진행바**.

**같은 틱의 나머지.** 컨트롤 eb16 13/303(~87 s/step, ETA 7h) — 총 스텝 303 유지, pusher(PID 29740)
생존, 로컬 체크포인트는 아직 없음(첫 저장 step 25, ~06:23 예정). 메타 eb16 여전히 queued(1h).
카나리아 RED 25회.

### E-116 — SFT push 구멍이 실제로 닫혔다 (0727 06:28 UTC)

E-114에서 띄운 `push_sft_ckpts_to_hf.py` 데몬이 **end-to-end로 검증됐다**.
컨트롤 eb16이 step 25에 도달해 06:24에 로컬 `checkpoint-25/`를 저장했고, 데몬이 이를 집어
`iamseungpil/metacot-sft2-eb16`에 올렸다 — 4샤드 + index + tokenizer + `trainer_state.json`까지
17개 파일 전부 착지. 0726 사이드카 SFT2가 ~40/309 스텝에서 노드와 함께 사라져 HF에 0파일을
남겼던 실패 양식이 이제 구조적으로 막혔다.

**한 번 헷갈릴 뻔한 지점.** 06:21 시점에 `list_repo_files`는 `.gitattributes` 하나만 반환했다.
이때 pusher 로그 꼬리를 보니 샤드가 92~98% 업로드 중이었다 — **HF 커밋은 원자적이라 업로드가
끝날 때까지 파일 목록에 아무것도 안 보인다.** "레포가 비었다 = 실패"로 판정했으면 멀쩡한
데몬을 고치려 들었을 것이다. 업로드 여부는 레포 목록이 아니라 **pusher 로그 원문**으로 먼저 본다.

**같은 틱의 나머지.** 컨트롤 eb16 27/303(체크포인트 저장·업로드가 I/O를 나눠 써서 ETA가 일시적으로
10h로 늘었다 — 업로드가 끝나면 회복될 값이다). 메타 eb16 여전히 queued(1h 15m). 카나리아 RED 26회.
b2p는 gs150 검증이 1시간 넘게 이어지는 중이고 HF 로그 최종 갱신은 06:10 — 다음 틱에 갱신이
끊기면(06:45까지 새 커밋 없음) 그때는 사망 신호로 다룬다.

### E-117 — b2p가 4시간째 durable 진전 0: save_freq가 윈도우 수명보다 길다 (0727 06:51 UTC)

**증상.** 06:51에 b2p 로그를 보니 진행이 **150/300에서 141/300으로 되돌아가** 있었다. 로그 자체는
신선했다(last_commit 06:40). wandb run 디렉터리 이름으로 재시작 이력이 드러난다:

| run | 시작 | 마지막 push | 결과 |
|---|---|---|---|
| `run-20260726_230835` | 23:08 | 04:04 | 5.3h 생존 — **gs140 저장 성공**(02:45) |
| `run-20260727_043127` | 04:31 | 06:10 | 2.0h — gs150 검증 중 사망, **저장 0** |
| `run-20260727_062856` | 06:28 | 진행 중 | gs140에서 또 resume |

**근인.** `h100std_rq3v2_b2p.yaml:243`이 `trainer.save_freq=20`이다. 윈도우 아키텍처는 preemption
시 **HF의 최신 완전 체크포인트를 다시 당겨와 나머지를 재수행**하므로, 한 윈도우가 20스텝을
은행에 넣지 못하면 그 윈도우의 계산은 전량 폐기된다. 측정된 ~380 s/step 기준 20스텝은 순수
학습만 **2.1시간**이고, 구간 안에 `test_freq=50` 검증이 걸리면 더 든다. 관측된 윈도우 수명은
5.3h / 2.0h / 2.0h — 즉 **두 번 연속 문턱을 못 넘어 4시간이 통째로 버려졌다.**

정정: 이것은 영구 livelock이 아니라 **한계선 문제**다. 긴 윈도우(5.3h)는 성공했다. 다만 짧은
윈도우가 이어지면 진전이 0으로 수렴한다.

**수정.** `save_freq=20 → 5`(≈32분/저장)로 낮췄다 — 관측된 **모든** 윈도우가 넘는 값이다.
적용 대상은 `h100std_rq3v2_b2p.yaml` · `a100_rq3v2f_b2p.yaml` · `a100_rq3v2f_b0p.yaml`.
b3p 계열은 이미 `save_freq=10`(≈63분)이라 관측 윈도우를 넘으므로 **손대지 않았다.**
저장 빈도가 늘어도 pusher가 `--keep 1`로 프루닝하므로 정상 상태 스토리지는 그대로다.

**지금은 적용할 수 없다.** 돌고 있는 b2p는 basicvc에 있고 그 VC는 0726 05:49부터 **모든 신규
제출을 거부**한다(E-091, 카나리아 26연속 RED). `amlt ssh`도 거부한다(status=failed). 즉
**취소하면 다시 못 올린다.** 따라서 취소하지 않고 그대로 둔다 — 긴 윈도우가 걸리면 gs160을
은행에 넣을 수 있고, 잃을 것은 없다. 수정된 런처는 basicvc 복구 시(또는 A100 경로로 갈 때)
첫 제출부터 효력을 갖는다.

**상태 정정.** RQ2의 b2p arm은 **gs140에서 사실상 동결**로 보는 것이 정확하다. 이전 틱들에서
"정상 진행"이라 보고한 것은 로그의 진행바만 봤기 때문이며, durable 기준으로는 02:45 이후
전진이 없다. **진행 신호는 오직 HF 커밋**이라는 규칙을 내가 로그 진행바에 한해 느슨하게
적용했던 것이 원인이다.

### E-118 — b3p가 durable을 못 남긴 근인: E-052 OOM 수정이 b3p에 이식되지 않았다 (0727 07:30 UTC)

**질문.** b3p는 `rq3v2_b3p` 계열 durable 체크포인트가 **HF에 하나도 없다**. 그런데 `main-mink`는
17시간을 돌았다. 왜 아무것도 안 남았나.

**로그.** `main-mink :rq3v2_b3p`는 **retry_013까지** 있다(14 윈도우). 가장 멀리 간 retry_013을
받아 읽었다:

```
[YAML] existing GRPO resume gs (model+extra+optim>=4) = 20 1
[YAML] RGS(HF)=20 ANY=1 LOCAL_GS(pulled)=20      ← 당시엔 완전한 gs20이 HF에 있었다
...
local_global_step_folder: /scratch/checkpoints/rq3v2_b3p/global_step_40
local_global_step_folder: /scratch/checkpoints/rq3v2_b3p/global_step_60
...
File ".../fsdp/transformer_impl.py", line 595, in forward_backward_batch
    loss.backward()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.54 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 3.35 GiB is free.
Including non-PyTorch memory, this process has 70.52 GiB memory in use.
Training Progress: 26%|██▌ | 78/300 [6:31:33<24:58:45, 405.07s/it]
```

즉 **선점이 아니라 CUDA OOM**이다. 6시간 31분을 돌아 step 78에서 backward가 터졌다.
(다른 윈도우는 선점도 섞여 있다 — retry_010은 부트스트랩 도중 `Caught signal 15`로 죽어
학습을 시작조차 못 했다. 두 살해자가 공존한다.)

**근인.** b2p 런처 55–80행에 3라운드짜리 OOM 추적 기록이 그대로 남아 있다 —
E-047(expandable_segments는 vLLM이 hard-assert해서 되돌림) → E-051(micro_batch 1로도 부족,
gs162 재발) → **E-052(진짜 피크는 8192토큰 BACKWARD ACTIVATION이며,
`enable_activation_offload=true`가 load-bearing lever)**. b2p는 0718에 고쳐졌다.
**그 수정이 b3p에는 한 번도 이식되지 않았다.**

| 레버 | b2p | b3p (수정 전) |
|---|---|---|
| `log_prob_micro_batch_size_per_gpu` | 1 | **2** |
| `rollout.gpu_memory_utilization` | 0.35 | **없음** |
| `fsdp_config.optimizer_offload` | true | **없음** |
| `model.enable_activation_offload` | true | **없음** |
| `rollout.enforce_eager` | true | **없음** |

두 실패의 footprint까지 일치한다 — b2p가 재발했을 때 70.26→70.46 GB, b3p는 70.52 GB.
게다가 b3p는 triobj 패키지가 롤아웃 위에 반사실 decoy까지 채점하므로 b2p보다 **더** 쓴다.
즉 b3p는 더 무거운 작업을 **수정 이전 메모리 설정으로** 계속 돌려온 셈이다.

**durable이 0인 경위.** gs20은 실재했고 retry_013이 거기서 resume했다. 이후 로컬로 gs40·gs60을
저장했지만 HF에는 남지 않았다 — 0726에 optim 0/4인 불완전 `rq3v2_b3p gs10`을 RGS 기아를 풀려고
삭제했고(E-106), 그 시점 이후로 완전한 원격 체크포인트가 재생성된 적이 없다. **먼저 고쳐야 할
것은 OOM이다**: 매 윈도우가 step 78 벽에 부딪히면 은행에 넣는 양이 벽 이전으로 제한된다.

**수정.** b2p의 네 레버를 b3p 런처 3종(`h100std_rq3v2_b3p.yaml`·`a100_rq3v2_b3p.yaml`·
`a100_rq3v2f_b3p.yaml`)에 그대로 이식하고 micro_batch를 2→1로 내렸다. 전부 메모리 전용이며
gradient-identical이므로 matched 비교는 유지된다 — 오히려 b2p가 이미 갖고 있던 레버라
**넣는 쪽이 더 matched**다. 4종 플래그 존재를 b2p와 대조해 5/5 일치 확인, yaml 파싱과
`bash -n` 통과.

**교훈.** 한 arm에서 근인을 규명해 고쳤으면 **같은 사다리의 다른 arm에도 이식했는지 그 자리에서
확인**해야 한다. E-097(컨트롤 arm의 SFT2 누락)과 정확히 같은 종류의 비대칭이며, 이번에는
9일 동안 b3p의 모든 실행을 조용히 죽여 왔다.

### E-119 — OOM 수정본 b3p 제출, 미수정본 정리 (0727 07:22 UTC)

E-118 수정을 실제로 태웠다. **순서가 중요했다**: 두 잡은 같은 HF lineage(`rq3v2_b3p`)에 쓰므로
동시에 살아 있으면 이중 writer가 된다(0726 감사가 지목했던 위험). 둘 다 아직 queued여서 잃을
compute가 없었으므로, 이번만은 **취소 → 제출** 순서가 옳다(대체 확보 후 폐기의 예외 —
"대체"가 아직 실행되지 않은 큐 항목일 때는 병존 자체가 위험이다).

- 취소: `rq3v2-b3p-a100-r2`(queued 10h, 수정 이전 설정 — 노드를 잡아도 step 78 벽 확정)
- 제출: **`rq3v2-b3p-a100-oomfix`** ← `a100_rq3v2_b3p.yaml`(E-118 레버 4종 + micro_batch 1),
  타르볼은 **488239754 그대로** 유지했다. 메모리 수정은 전부 yaml 인자에 있고 그 타르볼은
  b3p가 실제로 돌던 검증된 버전이므로, 변경을 최소화한다.
- 상태: `preparing` (오류 없음)

**낡은 텍스트 2종 추가 발견**(sed-clone 함정 6연속). 제출 출력의 description을 읽다 잡았다:
`log_prob micro-batch 2 (8192 OOM guard)`가 두 자리에 남아 있었고(이제 1), A100 런처 두 개가
`H100x4 Standard`라고 주장하고 있었다(실제 A100x4 Basic). 세 파일 모두 수정·커밋.

**콜드스타트.** `rq3v2_b3p` durable이 0이므로 RGS ANY=0 → gs0 정상 시작한다(E-106의 기아
조건은 "lineage 파일이 있는데 pull이 빈손일 때"이므로 해당 없음).

**같은 틱.** 컨트롤 eb16 **58/303**, pusher 회전 정상(`checkpoint-25`+`checkpoint-50`, `--keep 2`).
메타 eb16 queued 2h. b2p durable gs140 그대로. 카나리아 RED 27회.

### E-120 — 수술 런의 마지막 빈틈: 최종 모델 push 감시자 설치 (0727 08:10 UTC)

E-114의 노드 재활용 수술로 컨트롤 eb16을 **손으로** 띄웠기 때문에, 이 런에는 학습 종료 후
최종 모델을 HF로 올리는 주체가 없다. 원래 런처 셸(PID 1877)은 아직 살아 있지만 그 push 루프는
**구 경로** `/scratch/checkpoints/b0p2_rvfull_sft`를 밀고 있고 이 런은 그 경로에 쓰지 않는다.
E-116의 체크포인트 데몬은 크래시 대비를 해주지만 회전하는 `checkpoint-N/`을 **다른 레포**
(`metacot-sft2-eb16`)에 올릴 뿐, RL 런처가 init을 스테이징하는 곳
(`iamseungpil/metacot` 데이터셋의 `models/<name>/`)에는 아무것도 남기지 않는다.
즉 5시간 뒤 학습이 끝나도 **다음 단계가 쓸 산출물은 노드에만 존재**하게 된다 —
0726 사이드카가 전손된 것과 같은 구조다.

**설치.** `/scratch/final_push_eb16.sh` — 트레이너 PID 28843이 사라질 때까지 대기했다가
`push_models_hf.py`로 `models/b0p2_rvfull_eb16_sft`에 올리고, 4샤드가 실제로 착지할 때까지
최대 20회 재시도한다. `setsid nohup`으로 띄워 SSH 세션과 무관하게 산다.

**게이트.** 밀기 전에 `$CKPT/config.json` 존재를 확인한다. `sft.py:556`의
`trainer.save_model(output_dir)`이 실제로 실행돼야만 생기는 파일이므로, 선점으로 죽은 런에서는
없다. **부분 디렉터리를 올리는 것은 아무것도 안 올리는 것보다 나쁘다** — 다음 단계가 그것을
정상 init으로 착각한다.

**검증.** 노드에서 `wc -l` 30줄, `bash -n` 통과, 핵심 줄(TPID/CKPT/model_name/repo_id/게이트)
육안 확인, 감시자 프로세스 생존 확인. heredoc이 원격으로 전송되며 잘리지 않았음을 확인한 것이
요점이다(0718 `amlt bash -c` 따옴표 절단 전례).

**같은 틱.** 컨트롤 **91/303**, pusher 회전 정상(`checkpoint-25` 프루닝, 50·75 유지, 업로드
176초). b3p oomfix queued 50m, 메타 eb16 queued 3h, 4GPU 메타 queued 18h, b2p durable gs140.
카나리아 RED 29회.

### E-121 — RL 런처가 존재하지 않을 init을 가리키고 있었다 + fail-closed 게이트 (0727 08:30 UTC)

**발견.** 컨트롤 완주를 기다리며 다음 단계를 미리 점검하다 잡았다. eb16 전환(E-112)에서 산출물
lineage를 **의도적으로 분리**했는데(`b*p2_rvfull_eb16_sft`), 그 사실이 RL 런처에 반영되지 않았다:

| RL 런처 | 스테이징하던 init | 실제로 생길 것 |
|---|---|---|
| `a100_rq3v2f_b0p.yaml` | `models/b0p2_rvfull_sft` | **없음** → `b0p2_rvfull_eb16_sft` |
| `a100_rq3v2f_b2p.yaml` | `models/b2p2_rvfull_sft` | **없음** → `b2p2_rvfull_eb16_sft` |
| `a100_rq3v2f_b3p.yaml` | `models/b2p2_rvfull_sft` | **없음** → `b2p2_rvfull_eb16_sft` |

(`a100_rq3v2_b3p.yaml`은 구 init `b2p2_rvseg_sft`를 쓰며 그건 HF에 실재하므로 그대로 둔다.)

**왜 세 arm 모두 eb16 이름인가.** matched 비교는 **두 arm이 같은 world size**여야 성립한다
(E-112 불변식). 컨트롤은 1-GPU 수술 런이므로, 짝이 되는 메타도 1-GPU eb16이어야 한다.
18시간째 대기 중인 4-GPU 메타(`rq3v2-sft2-rvfull`)가 먼저 착지하면 그 산출물은
`b2p2_rvfull_sft`가 되는데, 그건 **1-GPU 컨트롤과 world size가 어긋난 짝**이다. 실효 배치는
양쪽 16으로 같으니 배치-4 때보다 훨씬 작은 편차지만 여전히 미검증 비대칭이므로,
기본 경로는 1-GPU 쌍으로 잡고 4-GPU 메타는 대체 경로로 둔다.

**더 나쁜 것 — 가드가 없었다.** 스테이징 블록에 착지 검증이 전혀 없었다. `set +e`가 걸려 있어
아티팩트가 없으면 `snapshot_download`/`copytree`가 실패해도 스크립트가 계속 진행하고, verl이
없는 경로로 떠서 죽고, 윈도우는 `sleep 86400`까지 흘러가 **A100×4를 하루 놀린다.** 이것이 내가
매 틱 손으로 지켜온 "SFT 산출물 HF 착지 전 RL 제출 금지" 게이트의 실체였다. 사람이 아니라
런처에 넣었다:

```
ls -l /scratch/models/<init>/config.json || { echo "[YAML] FATAL init ... ABORT window"; sleep 300; exit 1; }
```

**검증.** 세 파일 모두 구 이름 잔존 0, eb16 이름 10~11회, 가드 1개, yaml 파싱 + `bash -n` 통과.

**같은 틱.** 컨트롤 **101/303**(로컬 `checkpoint-100`, HF 50/75). 노드 3종 프로세스(트레이너·
체크포인트 pusher·최종 push 감시자) 전부 생존. b3p oomfix queued 1h, 메타 eb16 queued 3h,
4GPU 메타 queued 18h, b2p durable gs140.

### E-122 — 컨트롤 노드 선점: 수술 런 소실, pusher가 건진 것, 그리고 창(window) 산술 (0727 09:22 UTC)

**사건.** 09:22 틱에서 SSH 출력이 깨지고 로컬 체크포인트가 비어 보였다. 확인 결과
`a100g1-sft-b0p2-rvfull`이 `running` → `queued`로 돌아가 있었고 retry 로그 항목이 **0 → 32**로
늘었다. **노드가 선점됐다.** E-114 수술로 손수 띄운 트레이너·체크포인트 pusher·최종 push
감시자 세 프로세스가 `/scratch`와 함께 전부 사라졌다.

**건진 것.** HF `iamseungpil/metacot-sft2-eb16`에 **`checkpoint-100`과 `checkpoint-125`가 남았다.**
E-116에서 붙인 학습 중 pusher가 올려둔 것이 전부다. 그게 없었으면 0726 사이드카처럼 전손이었다.
"대리변수가 아니라 durable만 본다"는 규칙이 실제로 값을 한 첫 사례다.

**놓칠 뻔한 후속 위험.** 선점된 잡은 큐로 돌아갔다가 **새 노드를 잡아 다시 `running`이 됐는데**,
런처가 실행하는 것은 `configs/sft_b0p2_rvfull.yaml`, 즉 **실효 배치 4**다. eb16은 런처가 아니라
수동 수술이었으므로 재시작으로 재현되지 않는다. 그대로 뒀으면 잘못된 배치로 처음부터 8시간을
다시 돌 뻔했다. 취소했다.

**창 산술 — 이것이 핵심이다.**
| 구성 | 필요 시간 | 관측된 선점 창 | 완주 가능? |
|---|---|---|---|
| 1-GPU (ga16) | ~7.7h (관측: 3.4h에 122/303) | **~3.5h** | **아니오** |
| 4-GPU (ga4×4proc) | ~2h | ~3.5h | **예** |

1-GPU 경로는 **구조적으로 완주할 수 없다**. 슬롯은 잘 잡히지만(2~4h) 런타임이 창의 2배가 넘는다.
크로스노드 resume으로 메울 수도 없다 — HF에 올라간 체크포인트는 **weights-only**다.
`push_sft_ckpts_to_hf.py`가 DeepSpeed `global_step*/` 옵티마이저 디렉터리(8B Adam·ZeRO-3 기준
~96 GB)를 의도적으로 제외하기 때문이고, 그걸 매 저장마다 올리는 것은 현실적이지 않다.

**조치 3종.**
1. **eb16 런처 2종에 학습 중 pusher 내장**(E-122 커밋). 전에는 `accelerate launch`가 반환된
   뒤에만 밀었다 — 창을 못 넘기는 런에게는 "영원히 안 밂"과 같다. arm별 `--path_prefix`로
   두 arm이 서로 덮지 않게 했고, 학습이 끝나면 pusher를 종료해 최종 push와 경합하지 않게 했다.
2. **잘못된 배치로 재시작한 ga4 잡 취소 → 1-GPU eb16 재제출**(`a100g1-sft-b0p2-eb16-r2`).
3. **4-GPU 컨트롤 제출**(`a100-sft-b0p2-4g`) — 창 안에 완주 가능한 **유일한** 경로.
   실효 배치 16 확인(bs1 × ga4 × 4proc), 산출물 `b0p2_rvfull_sft`.

**이제 경로가 둘이고 각각 내부적으로 matched다.**
- 경로 A(1-GPU): 컨트롤 `b0p2_rvfull_eb16_sft` + 메타 `b2p2_rvfull_eb16_sft`
- 경로 B(4-GPU): 컨트롤 `b0p2_rvfull_sft` + 메타 `b2p2_rvfull_sft`(`rq3v2-sft2-rvfull`, 19h 대기)

⚠️**미해결**: E-121에서 RL 런처를 경로 A 이름으로 고정했다. 경로 B가 먼저 완주하면 이름이
어긋난다. 다음 틱에 **"eb16이 있으면 그것, 없으면 평문 이름"**으로 양자 수용하도록 고쳐
베팅을 없앨 것.

**같은 틱.** b2p durable이 **gs140 → gs160**으로 움직였다(동결 해제). 긴 윈도우가 걸렸다는 뜻이며
E-117의 `save_freq=5` 수정은 여전히 다음 제출에서 효력을 갖는다. 카나리아 RED 30회.

### E-123 — RL 런처의 init 이름 베팅 제거 (0727 09:58 UTC)

E-121에서 RL 런처 3종을 경로 A(1-GPU eb16) 이름으로 **고정**했다. E-122에서 경로가 둘로
갈리면서 그 고정은 베팅이 됐다 — 경로 B(4-GPU)가 먼저 완주하면 존재하지 않는 이름을 스테이징한다.
런처가 할 필요 없는 베팅이므로 없앴다.

**해소 방식.** 실행 시점에 HF를 조회해 `models/<name>/config.json`이 실재하는 첫 후보를 고른다
(eb16 우선, 없으면 평문). 그리고 **고정 로컬 경로 `/scratch/models/sft2_init`**로 복사하므로
하위 단계(tokenizer 패치·fail-closed 게이트·`model.path`)는 전부 이름과 무관해진다.
두 후보가 모두 없으면 `sys.exit(1)` → E-121 게이트가 윈도우를 ABORT한다.

| 런처 | 1순위 | 2순위 |
|---|---|---|
| `a100_rq3v2f_b0p.yaml` | `b0p2_rvfull_eb16_sft` | `b0p2_rvfull_sft` |
| `a100_rq3v2f_b2p.yaml` | `b2p2_rvfull_eb16_sft` | `b2p2_rvfull_sft` |
| `a100_rq3v2f_b3p.yaml` | `b2p2_rvfull_eb16_sft` | `b2p2_rvfull_sft` |

⚠️불변식은 그대로다 — **arm은 같은 경로 안에서 짝지어야 한다.** 1-GPU 컨트롤 대 4-GPU 메타는
미검증 world-size 비대칭이다. 이 자동 해소는 이름 문제만 없애고 짝짓기 판단을 대신하지 않는다.

**검증 방식이 요점.** yaml 블록 스칼라 → bash 큰따옴표 → `python -c` 세 겹의 이스케이프를 지나면
embedded python은 **노드에서만** 깨진다. 그래서 yaml을 로드하고 `$$`→`$`를 되돌린 뒤 bash가
풀어줄 `\"`→`"`까지 적용해 **python이 실제로 받게 될 소스를 뽑아 `compile()`** 했다. 3종 모두 통과.
지금 HF에는 후보가 하나도 없으므로 해소 로직은 의도대로 ABORT 경로를 탄다.

**같은 틱.** 5개 잡 전부 queued(1GPU 컨트롤 23m·4GPU 컨트롤 22m·1GPU 메타 5h·4GPU 메타 20h·
b3p oomfix 2h). b2p durable gs160 유지. 노드를 잡은 것이 없어 새 런타임 검증 없음.

### E-124 — 큐 정체의 근인 후보: A100 사용자당 동시 실행 1 (0727 10:30 UTC)

한 시간 넘게 5개 잡이 전부 queued인데 아무것도 시작하지 않아 쿼터를 조회했다.
`amlt target list sing -v`:

```
TARGET       SERIES      ACCEL        QUOTAS (USED/LIMIT)
                                      Premium      Standard   Basic      User max
msrresrchvc  NC_A100_v4  A100 80GB     4 / 4        0 / 0      0 / 0     . / 1
msrresrchvc  NDAMv4      A100 80GB     .   .        .   .      1 / 32    . / 1
msrresrchvc  NDv4        A100 40GB   382 / 384      .   .      .   .     . / 12
```

**A100 계열 두 시리즈 모두 `User max = 1`이다.** VC 레벨로는 Basic NDAMv4가 1/32밖에 안 쓰여
용량이 남아 있는데도 우리 잡이 안 붙는다는 사실과 맞물린다. 관측 증거도 일치한다 —
컨트롤 1개가 running인 동안 4-GPU 메타는 **20시간**, b3p는 3시간, 1-GPU 메타는 5시간을
나란히 대기했다. 즉 **A100에서는 한 번에 잡 하나만 돌 수 있고, 컨트롤이 그 한 자리를 계속
쥐고 있었다.** (아직 정황 증거다. `NC_A100_v4`가 Premium 4/4로 포화라는 점도 겹쳐 있어
단독 원인이라고 단정하지 않는다.)

**따르는 결론.** 컨트롤과 메타를 **동시에** 돌릴 수 없다 — 순차 실행이다. 그러면 구성 선택이
바뀐다:
| 구성 | 1회 소요 | 창 안 완주 | 쌍(컨트롤+메타) 총 소요 |
|---|---|---|---|
| 1-GPU | ~7.7h | **불가**(창 ~3.5h) | 완주 자체가 안 됨 |
| 4-GPU | ~2h | 가능 | **~4h 순차** |

**조치.** 1-GPU SFT 2종 취소(`a100g1-sft-b0p2-eb16-r2`, `worthy-snail`). 취소 근거는 쿼터
해석과 **독립적으로** 성립한다 — 1-GPU는 선점창 안에 완주할 수 없다. 여기에 동시 실행이 1이라면
완주 불가한 잡이 완주 가능한 4-GPU 잡의 유일한 자리를 뺏는다는 이유가 더해진다.
이로써 경로 A(1-GPU eb16)는 사실상 폐기되고 **경로 B(4-GPU)가 단일 경로**가 된다.
E-123의 런타임 init 해소가 이 전환을 자동으로 흡수한다(eb16이 없으면 평문 이름으로 내려간다).

**남은 큐 3개 — 그리고 이들도 서로 경쟁한다:**
| 잡 | 필요 | 비고 |
|---|---|---|
| `a100-sft-b0p2-4g` | ~2h | 컨트롤 SFT2. 복제축의 전제 |
| `rq3v2-sft2-rvfull` | ~2h | 메타 SFT2. 짝 |
| `rq3v2-b3p-a100-oomfix` | **~2일**(300스텝) | RQ2 arm. **구 init 사용 → SFT2에 의존하지 않음** |

⚠️**미결 우선순위**: b3p가 한 자리를 잡으면 이틀간 SFT 쌍이 굶는다. 반대로 SFT 쌍을 먼저 돌리면
4시간 뒤 b3p가 시작된다. 사용자 판단이 필요한 지점이라 물었다.

### E-126 — 두 자문의 수확, 그리고 b2p 재가동 (0727 13:15 UTC)

**fable(코드 클린 점검)이 P0을 잡았다 → E-125로 이미 수정·커밋.** 내 E-117/E-118 커밋이 주석을
backslash 이어쓰기 체인 **한가운데** 넣어 런처 6개의 verl 명령을 잘랐다. 잃은 인자가 하필 그
커밋들이 존재한 이유(OOM 레버 4종·save_freq·hydra.searchpath)였다. `bash -n`도 yaml 파싱도
diff 통독도 통과하는 클래스라 `tests/test_launcher_yaml_lint.py`로 인접성 규칙을 박았다
(음성 대조: 수정 전 트리에서 정확히 그 6파일만 실패). 859 passed.

**fable의 미처리 지적 3종**(다음 틱 처리 대상):
1. **E-122의 학습 중 pusher가 죽은 파일에만 있다** — eb16 런처 2개에 넣었는데 그 둘은 E-124에서
   폐기됐고, 살아남은 경로 B 런처는 여전히 `save_strategy: epoch` + 학습 후 push다.
2. `CLAUDE.md`의 Compute 절이 은퇴한 think-off 세대를 현행 런처로 안내한다.
3. 런처 33개 중 17개가 사망. `a100g2_*` 2종은 제출된 적도 없는데 산출물 경로가 살아있는 경로와
   충돌한다.
또한 stale-text가 6건이 아니라 **7건**이었다 — 8d1c787의 sed가 config 주석을 자기모순으로
오염시켰다("T1은 1 GPU에서 ga16으로 ... FOUR processes에서 실효배치 16"). E-125에서 함께 복원.

**codex(가설 사전등록)의 핵심.** 데이터가 생기기 전에 판정식을 고정했다.
- 메커니즘 지지 조건 7종 동시 성립: 규약 설치(`V_SFT2 ≥ .80`, `ΔV ≥ .70`) / pmi 신호 유효
  (`T ≥ .60`, 부트스트랩 하한 > .50) / pmi 고유 이득(`Δ_pmi-only ≥ +3pp`, 요인 주효과
  `I_pmi ≥ +5pp` 하한 > 0) / 풀패키지 `≥ +10pp` / **형식 효과 아님**(형식 무관 채점기에서도 양수)
  / **길이 효과 아님**(길이 보정 후 50% 잔존) / **보조 헤드만의 효과 아님**.
- "재현 실패"는 진단 지표가 건강한데도 pmi 상한 CI < +3pp일 때만. 구간이 넓으면 **결론 유보**.
- **검정력 경고**: 단일 시드·MATH500 500문항은 15~20pp에는 충분하나 5pp엔 한계, 난이도 6셀은
  셀당 83문항이라 **셀별 주장 금지**. 결론은 "이 런에서 사전등록 조건 하에" 까지만.
- **가장 스스로를 속이기 쉬운 방식 = 한쪽 arm에만 들어가는 구현 드리프트.** 모든 보상 곡선이
  멀쩡해 보이는 채로 거짓 양성/음성이 나온다. 오늘만 이 유형이 3번 나왔다(비대칭 사다리 E-097,
  한쪽에만 이식된 OOM 수정 E-118, 죽은 파일에만 들어간 pusher 수정 E-122).
- 처방: **결과를 보기 전에 동결·보관** — 데이터셋 매니페스트와 행 해시, 런처 명령과 해석된 최종
  config, 유효 토큰 수와 옵티마이저 업데이트 수, **arm별 시퀀스 길이·절단 분포**, 체크포인트 해시,
  채점기 버전·파서, 평가 항목 ID 전체.

**b2p 재가동.** durable이 **gs160 → gs180 → gs200**(300 중 200). 오전 4시간 동결(E-117)과 달리
긴 윈도우를 받고 있다. `save_freq=20`인 채로도 전진 중이므로 E-117의 `save_freq=5` 수정은 여전히
다음 제출에서만 효력을 갖는다.

**컴퓨트.** 카나리아 RED 32회. 4-GPU 잡 2건은 A100-80GB **사용자 한도 1장** 때문에 영원히
스케줄되지 않지만, 큐에 두는 비용이 없고 쿼터가 바뀌면 살아나므로 취소하지 않는다(단 이 경로를
계획에 계상하지 않는다). T1 vs base 1대1 대조 워크플로(fable 에이전트) 진행 중.

### E-127 — 시퀀스 길이 교란을 숫자로 확정: 학습 구간은 7.6%가 아니라 **29.5%** 차이 (0727 14:00 UTC)

codex가 지목한 미해결 교란("메타 arm은 시퀀스가 길어 같은 배치라도 토큰 수가 다르다")을
손으로 재기 위해 `scripts/freeze_run_manifest.py`를 만들고 SFT2 쌍에 돌렸다.

**설정 대칭성은 깨끗하다.** 두 config가 다른 키는 **4개뿐**이고 전부 의도된 것이다 —
`dataset_path` / `model_name_or_path` / `output_dir` / `run_name`. 하이퍼파라미터 드리프트 없음.
코퍼스도 1763행 동일, 시나리오 분할 동일(redirect 554 / verify 1209), 내용 해시만 다르다.

**그런데 토큰은 다르다.** 로컬에 토크나이저를 스테이징해 실측했다:

| | assistant 평균 | **학습 구간 평균** | **학습 구간 합계** |
|---|---|---|---|
| control(메타 제거 twin) | 520.5 | **165.8** | **292,353** |
| meta | 571.1 | **214.8** | **378,614** |

전체 기준으로는 +9.7%지만 **손실이 실제로 걸리는 구간은 +29.5%**다. sft.py가 prompt와
wrong_prefix를 마스킹하므로(`src/training/sft.py:105-118`) 학습 구간은 훨씬 짧고, 그래서 메타
블록이 차지하는 **비중**이 커진다. 평균 차 +48.9 토큰은 메타 블록 길이와 일치하므로 이 차이는
드리프트가 아니라 **처치 그 자체**다.

**정규화 방식이 결론의 방향을 바꾼다.** transformers 4.57.6의 `Trainer.compute_loss`는
`num_items_in_batch`를 사용한다(확인함) — 즉 grad-accum 창 전체의 **토큰 단위 정규화**이고,
예제 단위가 아니다. 따라서:
- 스텝당 그래디언트 크기는 두 arm이 비교 가능하다(각자 자기 토큰 수로 나눈다).
- 그러나 **공유 콘텐츠(recovery 텍스트)가 받는 그래디언트 비중이 다르다.** 메타 arm은 매 스텝
  그래디언트의 약 77%만 공유 콘텐츠에 쓰고 나머지 23%는 메타 블록에 쓴다. 컨트롤은 100%다.
- 즉 "메타 arm이 더 많이 학습한다"가 아니라 **"메타 arm이 공유 행동을 덜 학습한다"**가 맞는
  방향이다. 내 직관과 반대였다.

**이것이 교란인가.** 가법적 처치 설계에는 내재적이다 — 내용을 더하면서 공유 부분의 비중을
유지할 수는 없다. 결정적인 것은 **T1이 같은 코퍼스 쌍을 썼는지**다. 썼다면 이 성질은 새로
도입된 것이 아니라 복제된 것이고, 복제 실험으로서는 오히려 맞다. **미확인이며 다음 점검 대상.**
안 썼다면 T1과 base의 비교 자체에 이 항이 추가로 들어간다.

**동결 도구.** `scripts/freeze_run_manifest.py`는 결과를 보기 전에 다음을 JSON으로 고정한다 —
arm별 코퍼스 행수·내용 해시·시나리오 분할·토큰 길이 분포, 해석된 config와 **arm 간 config 차이
전량**, 목적함수를 결정하는 6개 소스 파일의 해시, git commit과 dirty 여부, 그리고 **채점 모드**
(`format_fair` / `strict_boxed`)와 채점기 모듈 해시. `--compare a.json b.json`으로 두 시점 사이에
무엇이 움직였는지 본다. 첫 산출물: `docs/manifests/sft2_pair_20260727.json`.

**채점기에 관한 정정.** 워크플로는 "format-fair 채점기가 in-tree에 없다"고 했으나 이는 문자열
검색 결과였고, 실제 구현은 `experiments/analysis/analysis_common.py:120-148`의 `Grader.grade()`에
있다. boxed가 없으면 런타임 `answer_extracted`를 같은 `robust_grade` 경로로 채점하는 fallback이며,
주석에 근거까지 적혀 있다(gandhi arm은 GSM8K에서 ~85%만 boxed, base/pmishift는 ~100% → boxed
부재를 오답 처리하면 arm별 형식 편향). 따라서 필요한 조치는 **신규 작성이 아니라 동결**이었고,
manifest가 모듈 해시와 선언된 모드를 함께 기록한다.

### E-128 — 길이 교란 해소: T1이 같은 코퍼스 쌍을 썼고 네 config가 dose-matched다 (0727 14:15 UTC)

E-127이 남긴 미확인 항목("T1도 메타 corpus + 메타제거 twin 쌍을 썼는가")을 확정했다.

**T1 메타 arm**: `archive/launchers_pre_rq3/h100std_rv_functional_sft.yaml:98`이
`configs/sft_rv_functional.yaml`로 `data/rv_redirect_verify_functional.parquet`을 학습.
**T1 컨트롤 arm**: `archive/launchers_pre_rq3/h100std_base_matched_pipeline.yaml:133-135`가
`configs/sft_base_rv.yaml`로 SFT2를 돌리고, 그 대상이 **`data/v8_base_rv_sft.parquet`** —
우리 base 컨트롤이 쓰는 바로 그 메타 제거 twin이다.

컨트롤 config는 `d565b52`("repo cleanup: archive pre-rq3 launchers/configs/trainers")에서
삭제돼 트리에 없었다. git 이력에서 복원해 대조했고, 참조 가능하도록
`configs/archive/sft_base_rv.yaml`로 되살렸다.

| | T1 메타 | T1 컨트롤 | base 메타 | base 컨트롤 |
|---|---|---|---|---|
| dataset | rv_redirect_verify_functional | **v8_base_rv_sft** | rv_redirect_verify_functional | **v8_base_rv_sft** |
| epochs / lr | 3 / 1e-5 | 3 / 1e-5 | 3 / 1e-5 | 3 / 1e-5 |
| bs × ga | 1 × 4 | 1 × 4 | 1 × 4 | 1 × 4 |
| max_length | 4096 | 4096 | 4096 | 4096 |
| save_strategy | epoch | epoch | epoch | epoch |

**결론.** E-127에서 잰 학습 구간 29.5% 토큰 차이는 **T1에서 복제된 성질이지 새로 도입된 교란이
아니다.** T1의 승리도 같은 비대칭 위에서 나왔으므로, 복제 실험으로서는 이 성질을 유지하는 것이
옳다. 네 config가 전부 dose-matched이고 남은 차이는 init과 데이터, 즉 처치뿐이다.

**방법론 메모.** 이 확인은 삭제된 파일을 git 이력에서 복원해야 가능했다. T1 대조 워크플로는
`sft_base_rv.yaml`을 "트리에 없음"으로 처리하고 넘어갔는데, 아카이브 정리로 사라진 파일이
있으면 워킹 트리 grep만으로는 과거 레시피를 복원할 수 없다. **T1 계보를 다룰 때는
`git log --all -- <path>`를 기본 절차로 둔다.**

### E-129 — H100 런처 5종 준비·검증, 사전등록 문서 동결 (0727 14:55 UTC)

**H100 런처를 실제로 제출해 봤다.** 아직 발사가 아니라 **온전성 검정**이다: 거부 사유가
SKU·YAML·쿼터 오류가 아니라 `The virtual cluster does not exist` 하나뿐이면, amlt가 타깃·SKU·
티어를 전부 해석한 뒤 마지막 관문에서만 막힌 것이므로 파일은 정상이다. 5종 모두 그랬다.
**그룹정책만 풀리면 그대로 붙는다.**

| 런처 | 제출 결과 |
|---|---|
| `h100std_sft_b0p2_rvfull.yaml` | VC 차단(예상) → 런처 정상 |
| `h100std_sft_b2p2_rvfull.yaml` | 〃 |
| `h100std_rq3v2f_{b0p,b2p,b3p}.yaml` | 〃 |

**wandb 지표 배선 확인.** `src/`에 `dcpo/` 접두 지표가 **36개** 정의돼 있고, 사전등록 판정에 쓸
것이 전부 실재한다 — `dcpo/meta_emit_rate`(verl_sdc.py:1017) ·
`dcpo/pmishift_attempted_rate` · `dcpo/pmishift_n_save`(:2564) · `dcpo/pmishift_n_derail` ·
`dcpo/wellformed_rate`(:1045) · `dcpo/acc_with` · `dcpo/acc_without`. 엔트로피와 클립은 verl
내부 지표(`actor/entropy`, `actor/pg_clipfrac`)이며 0727 b2p 로그에서 관측을 확인했다.

**사전등록 문서 동결**: `docs/PREREGISTRATION_rq3v2_base_replication.md`.
- **결과 3종**을 못박았다. 특히 **Outcome C(무효 런)** — gs25에서 emit<0.80 / attempted<0.10 /
  n_save=0 / entropy<0.1 중 하나면 보상이 굶은 것이고, **중단·상류 수정·재발사**이며 이를
  "재현 실패"로 부르는 것이 사전등록된 오류다. 직전 b3p가 정확히 이 상태였다.
- **검정력 제약을 명시**했다: 단일 시드·500문항은 15~20pp엔 충분하나 5pp엔 한계, 6셀은 셀당
  83문항 → **셀별 주장 금지**, "Qwen3-8B-Base 일반"으로 확대 금지.
- **채점기 동결**: `format_fair`를 판정용으로, `strict_boxed`는 **함께** 보고(대체 아님).
  A5 조건이 존재하는 이유가 두 채점기 사이의 A축 부호 반전이다.
- E-127/E-128의 학습구간 29.5% 비대칭을 **수용된 기지 비대칭**으로 기록했다 — T1도 같은 성질을
  가졌으므로 복제된 것이며, 나중에 불리한 결과를 설명하는 "발견"으로 제시될 수 없게 못박았다.
- 문서가 인용한 지표 키 7종과 파일 경로 6종을 **전수 실재 검증**했다.

**정리 진행**: 죽은 런처 8종은 `archive/launchers_retired_0727/`로 이동 완료(README 포함).
구세대 11종(`h100std_rq3_*`·`h100std_sft_*`)은 README·ARCHITECTURE·CODE_MAP·SUBMISSION_RUNBOOK·
CONSTITUTION이 참조하므로 문서 갱신과 묶어 별도 처리한다. H100 SKU라 현재 제출 자체가 불가능해
오발사 위험은 없다.

### E-130 — 문서 4종이 은퇴한 세대를 "현행 실행 절차"로 안내하고 있었다 (0727 15:20 UTC)

구세대 런처 11종을 아카이브하려고 참조 관계를 조사하다, **파일 이동보다 급한 문제**를 찾았다.
README·ARCHITECTURE·SUBMISSION_RUNBOOK·CONSTITUTION이 참조하는 것은 기록이 아니라 **지시**였다:

- `README.md:75-79` — `amlt run h100std_rq3_b0.yaml ...` 를 **실행 명령으로** 제시.
- `docs/CONSTITUTION.md:18` — "**현행 실험명**: RQ3 매치드 래더 (`h100std_rq3_b0/b2/b3.yaml`)"
  라고 **선언**. 헌법이 은퇴한 세대를 현행으로 규정하고 있었다.
- `docs/SUBMISSION_RUNBOOK.md:16,47,54-57` — 제출 절차와 4-arm 표 전부 구세대.
- `ARCHITECTURE.md:24-31` — SFT/LAUNCH 블록이 구세대 런처와 init.

파일명만 낡은 게 아니라 **세대 전체가 낡았다**. 구세대는 think-off RQ3(B0/B2/B3, init
`b0_gold_sft`/`b23_rv_unmasked_sft`)이고 현행은 think-on RQ3v2(2단 SFT 스택 위의 b0p/b2p/b3p)다.
게다가 구세대 SFT2 필터는 E-093에서 **위장된 시나리오 선택자**로 판명돼 폐기된 것이다.

**조치.** 각 문서를 읽고 성격에 맞게 처리했다.
- `README.md`: 실행 블록을 현행 순서로 교체 — SFT2 쌍 먼저, **HF 4샤드 착지 후** RL 3종.
  구세대 런처는 "실행하지 말 것"으로 명시하고 사전등록 문서를 가리키게 했다.
- `docs/CONSTITUTION.md`: 현행 실험 정의를 RQ3v2로 갱신하고, 구세대는 은퇴 사유(필터가 시나리오
  선택자)와 함께 괄호로 남겼다.
- `docs/SUBMISSION_RUNBOOK.md` / `ARCHITECTURE.md`: 본문을 다시 쓰지 않고 **세대 주석**을 머리에
  달았다. 두 문서의 실질(토큰 주입 시점 substitution·제출 전 점검·데이터 흐름·트레이너 진입점)은
  세대와 무관하게 여전히 정확하기 때문이다. 이름만 옮겨 읽으라고 명시했다.

**파일 이동은 아직 하지 않았다.** 문서가 올바른 런처를 가리키게 된 이상 구세대 파일이 루트에
남아 있는 위험은 크게 줄었고, 11종은 H100 SKU라 현재 제출 자체가 불가능하다. 이동은 `docs/CODE_MAP.md`
(참조 11건)까지 함께 처리할 때 한다.

**같은 틱.** 카나리아 RED 34회. b2p durable gs200 유지. 전체 스위트 854 passed.

### E-131 — 구세대 런처 아카이브 완료, 참조 32건 해소 (0727 15:35 UTC)

E-130에서 문서 4종의 **지시**를 고친 뒤, 남은 `docs/CODE_MAP.md`(참조 11건)를 처리하고 구세대
런처 10종을 옮겼다.

**CODE_MAP 처리 원칙.** 이 문서의 메커니즘 서술 — 모드 분기, config 상속 순서, §2의 "rmeta 소스는
yaml이 아니라 런처가 결정한다"는 함정, §3의 RGS 완전성 규칙 — 은 **전부 그대로 유효하다.** 현행
런처가 같은 장치를 쓴다. 그래서 본문을 다시 쓰지 않고 (a) 헤더에 세대 주석, (b) 런처 이름을 현행
으로 갱신, (c) §7 루트 런처 인벤토리만 재작성했다. 줄번호 인용
(`h100std_rq3_b3.yaml:191`)은 **인용 자체를 제거**했다 — 오늘 그 클래스가 하루 만에 썩는 것을
두 번 봤다.

**이동.** `h100std_rq3_b0/b2/b3/b3_dbg/b3nopmi.yaml` +
`h100std_sft_b0_gold/b0p/b2p/b23_unmasked/b2p2_rvseg.yaml` → `archive/launchers_retired_0727/`.
`h100std_env_builder.yaml`은 **실험이 아니라 conda env 빌더**이므로 루트에 남겼다.

**참조 해소 32건.** 이동 후 존재하지 않는 경로를 가리키는 참조를 전수 탐색해 아카이브 경로를
부여했다 — ARCHITECTURE(6) · SUBMISSION_RUNBOOK(9) · CODE_MAP · LOCAL_RUN(1) ·
EXPERIMENT_PLAN(1) · base_rl_recipe(6) · EXPERIMENT_LOG(1) · experiments/README(3), 그리고
**살아있는 SFT2 런처 2종의 선례 인용**(`# GATE = b0p precedent (...)`). `docs/reports/`의 런로그는
역사 기록이라 손대지 않았다. 자동 재검사 결과 **미해결 0건**.

**`h100std_sft_b0p/b2p.yaml`은 예외적으로 값이 있다** — 현행 SFT1 init
(`b0p_v8base_strict_sft`·`b2p_v8meta_strict_sft`)을 만든 런처이고 현행 SFT2 런처가 게이트 기준을
여기서 인용한다. 삭제가 아니라 아카이브인 이유다.

**루트 런처 최종 상태(14종).** 현행 A100 6 + H100 5 + 구 init RQ2 부록 1(`a100_rq3v2_b3p.yaml`) +
`h100std_rq3v2_{b2p,b3p}.yaml`(폐기 lineage, CLAUDE.md가 금지 표기) + env 빌더 1.
0727 시작 시점 27종에서 줄었다. 834 passed.

### E-132 — 학습 구간 토큰 계산을 manifest에 내장 (0727 15:45 UTC)

E-127에서 학습 구간 29.5% 차이를 **손으로** 계산했다. 손계산은 다음 발사 때 반복되지 않으므로
`scripts/freeze_run_manifest.py`가 자동으로 재게 했다.

**왜 학습 구간이어야 하는가.** `sft.py`는 prompt와 wrong_prefix를 마스킹하므로 손실이 걸리는
구간은 `prefix_split_char` 이후다. assistant 전체를 재면 비율이 **1.0972**로 나오고 이건
"9.7% 차이, 무시 가능"으로 읽힌다. 실제로 그래디언트가 걸리는 구간은 **1.2951**이다. 마스킹이
양쪽에서 공유 prefix를 걷어내면 남은 것 중 메타 블록의 비중이 훨씬 커지기 때문이다. **어느 쪽을
재느냐가 "볼 만한 값인가"를 가른다.**

manifest는 이제 `token_len_assistant`와 `token_len_trained`를 **둘 다** 기록하고,
`token_exposure`에 arm 간 비율과 평균 차를 계산해 넣는다. 실행 시 표준출력에도 찍는다:
`trained-token exposure meta/control = 1.2951 (mean delta +48.9 tok)` — 손계산 값과 일치.
사람이 나중에 나눠보기를 기대하지 않는다. **manifest에 없으면 아무도 안 본다.**

`token_exposure`에는 해석 규칙도 함께 적었다 — 차이가 메타 블록 길이와 같으면 그것은 처치이고,
**복제 대상 런에 그 성질이 없었을 때만** 교란이다. E-128에서 T1도 같은 코퍼스 쌍을 썼음을 확인했
으므로 현재는 처치다.

**테스트 10개**(`tests/test_freeze_run_manifest.py`): 분할 경계, `prefix_split_char=0`(VERIFY 행은
마스킹 없음), messages가 list로 오는 경우, 컬럼 부재, 0727 측정값 1.2951 **회귀 핀**,
`--tokenizer` 없을 때 **침묵하지 않고 이유를 남기는지**(침묵은 "arm이 일치함"과 구별 불가),
arm 간 config 차이 열거, 쌍이 아닌 입력 거부, 채점 모드 2종 선언. 844 passed.

**부수**: fable P3 지적대로 pusher docstring의 "token comes from HF_TOKEN **ONLY**"를 고쳤다.
`--token` 플래그가 없다는 **인터페이스** 서술이지 토큰 **출처**에 대한 서술이 아니었는데, 실제로는
`token=None`이 `HUGGING_FACE_HUB_TOKEN`/캐시 로그인 폴백을 허용한다. 0727의 실제 PAT 노출 사고도
근거로 덧붙였다.

### E-133 (0727 16:30) — 푸셔의 영구 고장이 건강과 구별되지 않던 문제를 부팅 프로브로 닫음

`push_sft_ckpts_to_hf.py`는 데몬 루프 안의 모든 예외를 삼킨다. 일시적 HF 오류가 내구성을
죽이면 안 되기 때문에 의도한 설계지만, 그 대가로 **영구 고장**(오타 난 `repo_id`, 쓰기 권한
없는 토큰, 타인 소유 레포)이 건강한 유휴 데몬과 완전히 똑같아 보인다. 프로세스는 살아 있고
로그엔 `daemon start`가 찍히고 런처는 `PUSH_PID`를 보고 "보호됨"이라 믿는데, 실제로는 창이
끝날 때까지 아무것도 올라가지 않는다.

`write_probe()`가 부팅 시 한 번 `<prefix>/.push_probe`에 몇 바이트를 쓰고 지운다. 여기서
실패하는 것은 전부 런 내내 실패하므로 `sys.exit(1)`. `--probe-only`는 런처가 학습 전에
**동기적으로** 목적지를 검사하게 한다. 세 경로 실측: 쓰기 가능(exit 0, `write+delete ok`),
타인 소유(exit 1, 403), 없는 네임스페이스(exit 1, 404). SFT 런처 4종에 배선.

### E-134 (0727 16:30) — `bash -c '…'` 안의 아포스트로피 하나가 잡 전체를 절단한다(신규 결함류)

E-133 배선 중 주석에 `typo'd`를 썼다가 **런처 4종이 전부 깨졌다**. 근인: 이 런처들의 명령은
통째로 `bash -c '<스크립트>'`의 **단일따옴표 안**에 들어간다. 그래서 스크립트 어디든 —
**주석 안이라도** — 아포스트로피 하나가 바깥 따옴표를 닫고 그 뒤가 전부 쓰레기가 된다.
yaml은 파싱을 통과시키고, diff는 멀쩡해 보이고, `tests/test_launcher_yaml_lint.py`의 기존
연속행 검사도 통과한다. 조립된 명령에 `bash -n`을 돌려야만 보인다(0718 메모리의
`amlt bash -c` 따옴표 함정과 같은 계열).

문구를 고쳐 해소했고, 이 결함류를 잡는 린트를 추가했다: 모든 런처 yaml의 모든 job command가
`bash -n`을 통과해야 한다 + 음성 대조군(아포스트로피를 넣으면 실제로 실패하는지). 루트 런처
14종 전부 통과. 861 passed.

### E-135 (0727 16:30) — b2p 틱: durable gs220, 학습은 정상, 엔트로피 상승은 아직 열화 아님

- **durable** `checkpoints/rq3v2_b2p/global_step_220` (15:56~16:02 UTC 착지, gs200 프루닝됨).
  잡 레벨 `running` 3일차.
- **wandb 상태 `crashed`는 오진**이다. heartbeat 16:03:42, 조회 시각 16:26 → 22분 공백.
  그런데 3일에 220스텝 = **스텝당 ~19.6분**이므로 22분 공백은 정확히 한 스텝 간격이다.
  스텝 220 지표가 실제로 기록돼 있다. 남은 80스텝 ≈ **26시간**.
- **학습 신호**: `critic/score/mean` 0.281(s20) → 0.401(s200) → **0.515(s220)**.
- **held-out val은 전 과목 개선**(gs50→gs200): algebra 0.418→0.558, geometry 0.012→0.188,
  prealgebra 0.587→0.610, gsm8k 0.908→0.925, omni-math −0.473→−0.460.
- **관찰: `actor/entropy`가 0.53(s40) → 2.74(s220)로 5배 상승.** 0629에 확정한 실패 모드
  (디코딩 degeneration/비종료)의 선행 지표와 형태가 같다. 다만 지금은 **동반 증거가 없다**:
  `response_length/clip_ratio`는 s200에 2.5%로 튀었다가 s220에 0.98%로 복귀했고, 길이
  평균은 857→564로 줄었으며, 무엇보다 val이 전 과목 개선 중이다. 판정: **경보 아님, 감시
  대상**. 반증 조건 — clip_ratio가 두 연속 측정에서 5%를 넘거나 val 과목 절반이 하락하면
  그때 degeneration으로 재분류한다.
- **카나리아 37회째 RED** (`msrresrchbasicvc` `UserError: does not exist`). A100 4-GPU 2건은
  여전히 미스케줄(사용자 한도 1장).

### E-136 (0727 16:45) — b2p는 정상(스텝 228/300, ~7시간 남음). E-135의 두 진술을 정정한다

42분간 wandb 스텝이 안 늘어 조사했다. **원인: wandb 동기화가 끊긴 것이고 학습은 멀쩡하다.**
잡 로그는 `Training Progress: 76%|228/300 [10:10:56<7:11:55, 359.94s/it]`인데 wandb는 221에
멈춰 있다 — **7스텝 뒤처짐**. wandb 디렉터리 자체는 계속 HF로 밀리고 있어(16:28 커밋) 데이터는
유실되지 않았고 나중에 재동기 가능하다.

E-135에서 쓴 두 가지를 정정한다:
1. **스텝 속도**: "3일에 220스텝 = 19.6분/스텝, 완주 ~26시간"이라 썼는데, 이건 07-24 12:19
   런 시작부터의 **평균**이라 초반 재시작·선점 구간을 포함한다. **현재 실측은 359.94s/it =
   6.0분/스텝**이고 트레이너 자체 ETA는 **7시간 12분**. 26시간이 아니다.
2. **wandb `crashed` 해석**: "heartbeat 공백 = 정확히 한 스텝 간격이라 정상"이라 했는데,
   6분/스텝이므로 그 근거는 성립하지 않는다. 실제로는 **진짜 동기화 중단**이다. 다만 결론
   (학습 정상)은 그대로이며, 근거가 wandb 타이밍이 아니라 **잡 로그의 스텝 카운터**로 바뀐다.

**교훈(운영)**: wandb는 진행 판정의 1차 소스가 될 수 없다 — durable(HF)과 마찬가지로 조용히
낡을 수 있다. 스텝 레벨 진실은 `amlt log view <exp> :<job>`이고, HB 스피너가 로그를 뒤덮으므로
`grep -vE "^\+|^\[HB |MemAvailable"`로 걸러야 트레이너 출력이 보인다.

**스텝 228 지표**(로그 직접): entropy 2.62 · score/mean 0.323 · response_length/mean 548 ·
clip_ratio 0.0117 · **aborted_ratio 0.0** · grad_norm 0.574 · pg_clipfrac 0.00096 · ppo_kl
0.00022. score/mean은 스텝간 변동이 크다(214~228에서 0.28~0.68) — 단일 스텝 값으로 판정 금지.

**엔트로피 감시 갱신**: s200 이후 2.3~2.9 대에서 **평탄화**(더 안 오름). clip_ratio ~1%,
aborted_ratio 0. E-135의 반증조건 어느 것도 발동 안 함 → degeneration 가설은 **여전히 미지지**.
카나리아 38회째 RED.

### E-137 (0727 17:00) — 틱: 스텝 231/300 정상. 런로그 회전(E-000~E-111 동결)

- **b2p 트레이너 스텝 231/300**, 330.43s/it(**5.5분/스텝**), ETA **6시간 20분**. durable gs220.
  지표: entropy 2.456 · score/mean 0.400 · response_length/mean 539 · **clip_ratio 0.98%** ·
  **aborted_ratio 0.0** · grad_norm 0.469. E-135 반증조건 미발동 → degeneration 여전히 미지지.
- **카나리아 39회째 RED.**
- **런로그 회전**: 3,660줄이 되어 한 번에 읽기 어려워졌다. E-000~E-111(110건)을
  `2026-07-17-rq3-run-and-iteration-log-part1.md`로 분리·동결하고, 본편은 헤더+현행 상태판+
  E-112 이후(24건, 859줄)만 담는다. 분리 전후 엔트리 수 134건 일치 확인(중복 0·유실 0).
  본편 머리에 part1 포인터를 넣어 탐색 경로를 끊지 않았다. E 번호는 두 파일에 걸쳐 유일하므로
  기존 [[E-xxx]] 상호참조는 그대로 유효하다.

### E-138 (0727 18:56) — 침묵의 정체는 선점이었다. resume가 작동해 gs220에서 재개(손실 16스텝)

18:10~18:30 사이 "노드발 아웃바운드 전면 침묵 vs 외부 조회는 running"으로 미판정 상태였던
b2p의 원인이 확정됐다. **Standard 티어 선점**이다.

증거 3종이 한꺼번에 양성으로 돌아섰다:
- `retry_047` — 직전 틱까지 최대가 046이었다. 새 retry = 선점 후 재시작.
- **새 wandb 런 디렉터리** `run-20260727_182402-rq3v2-b2p-1`(18:24:02 시작). 이전은
  `run-20260727_062856-...`. HF 커밋 18:28:25로 사이드카 푸시도 재개.
- 진행바 `73%|220/300 [00:00<?, ?it/s]` — **gs220 durable에서 부팅 직후**.

**타임라인**: 마지막 확증 스텝 236 @ 17:23 → 선점(17:23~17:30 추정) → 18:24 재시작 →
gs220에서 resume. **손실 = 스텝 221~236, 16스텝(~1.5h)**. 재개 시점 기준 남은 80스텝은
5.6분/스텝이면 ~7.5시간.

**잘 된 것.** ①**손대지 않은 판단이 옳았다** — 침묵 78분 시점에 취소·재제출했다면 이미
복구 중이던 잡을 죽이고 gs220부터 다시 시작하는 같은 결과를 훨씬 늦게 얻었을 것이다.
②durable 릴레이가 설계대로 작동했다(gs220 → 자동 resume, 오resume 없음). ③`--keep 1`
저장예산에도 불구하고 anchor가 살아 있었다.

**진단에서 배운 것.** 초기에 나는 로그 API 503을 근거로 "관측 경로 장애"로 기울었고, 그
근거로 "로그가 커서 게이트웨이 타임아웃"을 들었다. 그 설명은 로그 API에는 맞았지만
**HF 사이드카 침묵은 설명하지 못했고**, 그 미설명 잔차가 실제 원인(선점)을 가리키고 있었다.
교훈: **여러 신호가 동시에 죽었을 때, 하나의 편리한 설명이 나머지를 덮게 두지 말 것.**
설명되지 않는 신호 하나가 진짜 원인이다. [[singularity-preemption-not-hang-retry-check-0711]]

**운영 규칙 추가**: b2p 같은 Standard 잡의 침묵을 볼 때 **가장 먼저 `amlt log list`의 retry
최대번호**를 확인한다(로그 본문보다 가볍고, 본문 조회가 503일 때도 종종 응답한다). 번호가
오르면 선점이고, 그때는 기다리는 것이 유일하게 옳은 조치다.

카나리아 45회째 RED.

### E-139 (0727 19:20) — 선점 주기가 체크포인트 주기보다 짧아졌다: 교착 위험, 그리고 손댈 수 없음

E-138의 선점 복구 25분 뒤 **retry_048**이 관측됐다(18:56 시점 047). 진행바는 다시
`73%|220/300 [00:00<?, ?it/s]` — **또 gs220에서 부팅**했다.

**재시작 이력**(retry 번호 기준): ≤17:45 retry_046 → 18:24 retry_047(gs220 resume) →
~19:0x retry_048(gs220 resume). 즉 최근 두 창은 **40~55분**짜리다.

**이것이 왜 교착인가.** 체크포인트는 20스텝마다이고 5.6분/스텝이면 **약 112분 + 저장 ~6분**이
필요하다. 창이 40~55분이면 **다음 체크포인트에 영원히 도달하지 못한다** — 매 창이 컴퓨트를
태우고 durable은 gs220에 고정된다. 은퇴 세대 상태판에 적힌 "선점마다 gs158/gs43 리셋"과
같은 실패 모드가 재발한 것이다.

**단, 이번 창이 짧아진 것은 최근 현상이다.** 직전 런은 06:28~17:23까지 **11시간 무중단**으로
돌아 gs200·gs220을 찍고 스텝 236까지 갔다. 그러니 이건 "이 잡은 원래 못 돈다"가 아니라
**최근 1시간 사이 클러스터 압력이 급증했다**는 관측이다. 압력이 가라앉으면 다시 진행한다.

**⛔치명적 제약: 이 잡은 교체 불가능하다.** `msrresrchbasicvc`는 0726 05:49부터 **신규 제출을
전면 거부**한다(카나리아 46회 연속 RED). 따라서 save_freq를 20→5로 줄이는 자명한 처방은
**재발사를 요구하므로 실행 불가**다 — 취소하는 순간 노드를 영영 잃는다. 지금 도는 이 잡은
차단 이전에 진입해 살아남은 유일한 H100 자원이다.

**그러므로 조치는 없다. 기다리는 것이 유일한 선택지다.** 최악의 경우에도 gs220은 durable하게
남아 있어 부록용 체크포인트로 쓸 수 있다. 잡에 손대는 것은 순손실만 낳는다.

**감시 지표 변경**: 이제부터 스텝 수보다 **retry 번호 증가율**이 핵심 지표다. 048에서
멈추면(= 창이 길어지면) 진행 재개, 계속 오르면 gs220 고정으로 확정하고 부록 판정으로 넘어간다.

### E-140 (0727 19:53) — E-139의 교착 우려는 실현되지 않았다: 창이 길어져 스텝 225 진행 중

retry는 **048에서 멈췄고**(E-139가 세운 판별 기준), 진행바는
`75%|225/300 [32:13<8:06:18, 389.04s/it]` — 이번 창은 32분째 지속 중이며 그 사이 5스텝을
진행했다. E-139에서 우려한 "선점 주기 < 체크포인트 주기" 교착은 **현재 데이터로는 미성립**이다.
40~55분 창 두 번은 클러스터 압력의 일시적 급증이었고, 두 틱 연속 상승이 아니었으므로 E-139의
확정 조건도 발동하지 않았다.

속도는 389.04s/it(**6.5분/스텝**)으로 선점 전(5.6분)보다 다소 느리다. gs240까지 15스텝 ≈
**97분**(이 창이 유지되면 ~21:30).

**⚠️단일 관측 하나는 확인이 필요하다**: 재개 후 s225에서 `actor/entropy` **3.165** —
관측 이래 최고치다(직전 최고 2.87 @s217, 선점 직전 s236은 2.18). `clip_ratio`도 2.93%로
s236의 0.39%에서 올랐다. 방향만 보면 degeneration 가설이 예측하는 조합이다.

다만 **판정하기엔 이르다**. ①엔트로피는 스텝간 변동이 크다(s214 2.83 → s215 2.45 관측).
②선점 후 재개 직후라 s221~225를 다른 데이터 순서로 다시 밟는 중이다. ③E-135의 반증조건
(clip_ratio **두 연속** >5%)에 미달한다. 다음 두 틱에서 엔트로피가 3 위에 머물고 clip_ratio가
계속 오르면 그때 재분류한다.

카나리아 47회째 RED.

### E-141 (0727 21:36) — gs240 착지. 선점 후 첫 durable 전진, 그리고 열려 있던 우려 두 건 종결

`checkpoints/rq3v2_b2p/global_step_240`이 21:24~21:26에 착지하고 gs220은 프루닝됐다.
**선점(17:23) 이후 첫 durable 전진**이며, 이로써 열어 두었던 두 우려를 모두 닫는다.

**① E-139의 교착 시나리오 — 실증적으로 기각.** "선점 주기(40~55분) < 체크포인트 주기(~118분)
이므로 durable이 gs220에 영구 고정된다"고 우려했는데, retry_048 창은 **2시간 17분 무중단**으로
22스텝을 진행하고 체크포인트까지 완주했다. 40~55분 창 두 번은 클러스터 압력의 일시적 급증이
맞았다. **교훈: 두 표본으로 주기를 추정해 구조적 결론을 내리지 말 것** — E-139에서 "두 틱 연속
상승"이라는 확정 조건을 미리 걸어 둔 덕분에 성급한 확정을 피했다.

**② E-140의 엔트로피 스파이크 — 재개 직후 일시 교란으로 확정.** s225 entropy 3.165 ·
clip 2.93%(관측 이래 최악 조합)에서 s230 1.923/1.17% → s235 2.451/0.39% → s239 2.561/0.98%
→ s242 2.953/1.76%로, 1.9~3.0 사이를 오가며 **추세 없이 진동**한다. clip_ratio는 2% 아래를
유지하고 aborted_ratio는 계속 0.0이다. E-135의 반증조건(clip_ratio 두 연속 >5%)은 한 번도
발동하지 않았다. **degeneration 가설은 기각되지 않았을 뿐, 지지된 적도 없다** — 최종 판정은
gs300 held-out에 남긴다.

**③ 정정.** E-139에서 retry_048의 재시작 시각을 "~19:0x"로 추정했는데, wandb 디렉터리
`run-20260727_191309-rq3v2-b2p-1`로 확인한 실제 값은 **19:13**이다.

**현황**: 스텝 **242/300**, 392.58s/it. 남은 58스텝 ≈ 6시간 20분(선점이 없다면 ~04:00 UTC).
다음 durable은 gs260. 카나리아 51회째 RED.

### E-142 (0728 00:00) — gs250 val 도착. **E-135의 반증조건이 발동했다.** 재분류하지 않는 이유와, 그 조건이 잘못 설계됐다는 인정

**먼저 오해 하나를 정정한다.** 스텝 250에서 `1175.94s/it`을 보고 "학습이 3.7배 느려졌다"고
읽을 뻔했다. 실제로는 `timing_s/step`이 **416초(6.9분)로 정상**이고, 스텝 250이 **검증
스텝이라 `timing_s/testing`에 2,563초(43분)**가 잡힌 것이다. tqdm의 s/it은 이 검증 시간을
포함한다. **속도 저하는 없었다.**

**gs200 이후 처음으로 val이 나왔다**(선점으로 wandb가 끊겨 로그 본문에서 직접 읽음).

| step | gsm8k | algebra | prealgebra | geometry | omni-math |
|---|---|---|---|---|---|
| 50 | 0.908 | 0.418 | 0.587 | 0.012 | −0.473 |
| 100 | 0.763 | 0.395 | 0.435 | −0.085 | −0.517 |
| 150 | 0.780 | 0.411 | 0.416 | 0.164 | −0.610 |
| 200 | **0.925** | **0.558** | **0.610** | **0.188** | **−0.460** |
| **250** | 0.896 | 0.445 | 0.529 | 0.018 | −0.489 |

**E-135에 적어 둔 반증조건은 "clip_ratio 두 연속 >5% **또는 val 과목 절반 하락**"이었다.
gs200 대비 5/5가 하락했으므로 이 조건은 문자 그대로 발동했다.** 이것을 조용히 넘기지 않고
기록한다.

**그런데 재분류하지 않는다. 근거 셋:**

1. **비교 기준을 바꾸면 결론이 뒤집힌다.** gs150 대비로는 4/5가 **상승**했다(gsm8k +0.116,
   prealgebra +0.113, omni +0.121, algebra +0.034, geometry −0.146). gs200이 봉우리였고
   gs250은 그 되돌림이다.
2. **이 진동폭은 원래 있던 것이다.** gs50→100에서도 4/5가 하락했고 그때 아무 문제 없었다.
   계열 전체가 ±0.15 규모로 흔들린다. 즉 "절반 하락"은 **정상 상태에서도 수시로 참**이 되는
   조건이었다.
3. **직접 측정치가 반대다.** 보상 안에 `degeneration_penalty`라는 **바로 그 실패 모드의
   직접 계측**이 있다. gs250: gsm8k 0.000 · algebra 0.000 · prealgebra ~0.000 ·
   geometry −0.019 · omni −0.157. gs200(−0.163)·gs100(−0.085)·gs50(−0.112)과 비교해
   **추세가 없다**. 나는 그동안 엔트로피와 clip_ratio라는 **대리지표**로 degeneration을
   논해 왔는데, 정작 직접 지표가 계속 로깅되고 있었다.

**그러므로 조건 자체가 잘못 설계됐음을 인정한다.** "절반 하락"은 (a) 비교 기준을 명시하지
않았고 (b) 이미 데이터에 보이던 잡음 크기를 반영하지 않았으며 (c) 손에 쥔 직접 지표 대신
대리지표를 썼다. **조건을 아래로 교체한다**:

> **degeneration 재분류 조건(v2)**: 연속된 두 val 지점에서 ①`degeneration_penalty`가
> gsm8k·algebra·prealgebra 중 둘 이상에서 −0.01 아래로 내려가거나 omni-math가 −0.20 아래로
> 내려가고, **동시에** ②score가 5과목 중 3과목 이상에서 **gs150 수준 아래**로 떨어질 때.

교훈: 반증조건은 미리 적는 것만으로 충분하지 않다. **그 조건이 정상 상태에서 얼마나 자주
참이 되는지**를 함께 확인해야 하고, 대리지표보다 직접 지표가 있으면 그것을 써야 한다.
[[E-135]]·[[E-140]]

**기타**: retry 049 유지(이번 창 1h47m·10스텝). durable gs240. 스텝 250/300. 카나리아 55회 RED.

### E-143 (0728 00:47) — gs260 착지. 무선점 2h50m 창에서 21스텝, 완주까지 39스텝

`checkpoints/rq3v2_b2p/global_step_260`이 00:44~00:47에 업로드 중이다(gs240은 곧 프루닝될
것). retry는 **049에서 유지** — 이번 창이 **2시간 50분 무중단**으로 21스텝을 진행했다.
E-139의 교착 우려가 완전히 무근했음을 두 번째 체크포인트로 재확인한다.

스텝 **261/300**, `timing_s/step` **341.1초(5.7분)** — 선점 전 최고 속도 수준으로 돌아왔다.
남은 39스텝 중 val 1회(~43분)를 포함하면 **약 4시간 25분**, 무선점 시 완주 ~05:10 UTC.

지표: entropy 2.816 · score/mean 0.430 · clip_ratio 1.17% · aborted 0.0. E-142에서 정한
대로 **엔트로피·clip은 대리지표이므로 판정에 쓰지 않는다**. 다음이자 마지막 val은 gs300이며,
거기서 `degeneration_penalty`로 재분류 조건 v2를 평가한다.

**선점 이력 정리**(참고용, 주기로 단정하지 않음): 17:23(손실 16스텝) · 19:13(손실 2스텝) ·
21:56. 창 길이는 1h50m · 2h43m · 2h50m+로, durable 릴레이가 매번 작동해 손실이 체크포인트
간격 이내로 제한됐다. 카나리아 57회째 RED.

### E-144 (0728 02:43) — gs280 착지. 남은 20스텝, 완주까지 약 2시간

`checkpoints/rq3v2_b2p/global_step_280` 업로드 중(02:43). retry **049 유지** — 이번 창이
**4시간 44분 무중단**으로 세 번째 체크포인트를 연속 통과했다. 21:56 재시작 이후 선점 없음.

스텝 **280/300**, `timing_s/step` 382.2초. 남은 20스텝 + 마지막 val(~43분) ≈ **2시간**,
완주 예상 **~04:45 UTC**.

s280 지표: entropy 3.207 · score/mean 0.387 · clip 1.37% · aborted 0.0.

**완주 시 할 일**(사전 정리):
1. **gs300 val의 `degeneration_penalty`로 E-142 재분류 조건 v2 평가** — 이것이 degeneration
   가설의 최종 판정이다. 대리지표(entropy/clip)는 쓰지 않는다.
2. gs300 durable 착지 확인 후 held-out eval(1030문항 = GSM8K500+MATH500+AIME30,
   `format_fair`·`strict_boxed` 양쪽, 항목ID 동결).
3. ⚠️**결과 보고 시 반드시 명시**: 이 런은 **rvseg 부록 lineage**(init `b2p2_rvseg_sft`,
   E-093에서 redirect 기아로 폐기 판정)다. **복제 결과로 보고 금지**, 부록 수치로만 쓴다.
   본선 rvfull은 SFT2 쌍이 선행조건이고 컴퓨트 차단으로 미발사 상태다.

카나리아 60회째 RED.

### E-145 (0728 04:45) — **b2p gs300 완주.** durable 착지, 최종 검증 진행 중

`checkpoints/rq3v2_b2p/global_step_300`이 HF에 착지했다. 로그는
`test_gen_batch meta info: {... 'validate': True, 'global_steps': 300}`와
`validation generation end`를 반복하며 **마지막 val을 돌고 있다**(9과목 순회).

**완주 경위.** 07-24 12:19 발사 → 3일 16시간, retry 049회. 마지막 창은 21:56 재시작 이후
**6시간 49분 무선점**으로 gs280·gs300 두 체크포인트를 연속 통과했다. 선점 3회(17:23·19:13·
21:56)의 실제 손실은 16 + 2 + 0 스텝뿐이다 — durable 릴레이가 매번 최신 체크포인트에서
자동 resume했고, 그 사이 나는 잡에 **한 번도 손대지 않았다**. 78분 침묵 구간에서 취소·재제출을
했다면 이미 복구 중이던 잡을 죽이고 같은 지점부터 훨씬 늦게 다시 시작했을 것이다.
[[E-138]]·[[E-139]]·[[E-141]]

**다음 단계는 val 수치가 나온 뒤에 한다** — 지금 판정하지 않는다:
1. gs300 val의 `degeneration_penalty` 5과목으로 **E-142 재분류 조건 v2 평가**.
   엔트로피는 s294에서 3.590(관측 최고)까지 갔지만 **대리지표이므로 근거로 쓰지 않는다**.
2. held-out eval 1030문항(GSM8K500+MATH500+AIME30), `format_fair`·`strict_boxed` 양쪽,
   항목ID 동결.
3. ⚠️**이 런은 rvseg 부록 lineage**(init `b2p2_rvseg_sft`, E-093에서 redirect 기아로 폐기
   판정)다. gs300이 나왔다고 해서 복제 결과가 되지 않는다 — **부록 수치로만** 쓴다. 본선
   rvfull은 SFT2 쌍이 선행조건이고 컴퓨트 차단으로 여전히 미발사다.

카나리아 63회째 RED.

### E-146 (0728 05:33) — **b2p 최종: 300/300 완주, degeneration 가설 기각.** `failed` 상태는 teardown 아티팩트

**먼저 상태 표시를 정정한다.** `amlt status`가 `failed`로 바뀌었지만 **학습 실패가 아니다**:
- 진행바 `Training Progress: 100%|300/300 [7:18:20]` — 전 스텝 완주.
- gs300 val 완료, wandb가 `Synced 5 W&B file(s)` 후 정상 종료. **wandb `state: finished`,
  `_step: 300`**.
- 로그 마지막 줄이 `Terminated` — 즉 파이썬 프로세스가 정상 종료한 **뒤** 컨테이너가 SIGTERM을
  받았다. `failed`는 그 종료 코드일 뿐이다.
- durable `checkpoint/rq3v2_b2p/global_step_300` 이미 착지 완료.

**gs300 val (wandb 동기화분, 9과목 중 추적 5과목)**

| step | gsm8k | algebra | prealgebra | geometry | omni-math |
|---|---|---|---|---|---|
| 150 | 0.780 | 0.411 | 0.416 | 0.164 | −0.610 |
| 200 | **0.925** | **0.558** | **0.610** | **0.188** | −0.460 |
| 250 | 0.896 | 0.445 | 0.529 | 0.018 | −0.489 |
| **300** | 0.841 | 0.384 | 0.490 | −0.024 | **−0.438** |

`degeneration_penalty/mean@1` (0 = 열화 없음)

| step | gsm8k | algebra | prealgebra | geometry | omni-math |
|---|---|---|---|---|---|
| 200 | 0.0000 | −0.0008 | −0.0036 | −0.0089 | −0.1632 |
| 250 | 0.0000 | −0.0035 | 0.0000 | −0.0192 | −0.1573 |
| **300** | −0.0035 | −0.0084 | 0.0000 | −0.0286 | **−0.0861** |

**E-142 재분류 조건 v2 평가 — 두 절 모두 미충족, degeneration 가설 기각.**
- **①(직접지표)**: gs250·gs300 어느 쪽에서도 gsm8k·algebra·prealgebra가 −0.01 아래로 간 적이
  **없다**(최악이 gs300 algebra −0.0084). omni도 −0.20 아래로 간 적 없고, gs300에서 오히려
  **−0.0861로 gs150 이후 최선**이다. → FALSE.
- **②(score)**: gs300에서 gs150 수준 아래인 과목은 algebra(0.384<0.411)·geometry
  (−0.024<0.164) **2과목뿐**이고, gsm8k·prealgebra·omni는 gs150보다 위다. 3과목 기준 미달
  → FALSE.
- 두 절이 AND이므로 **조건 미발동**. 엔트로피는 s294에서 3.590, 최종 요약 3.0519로 관측 최고
  대역에 머물렀지만 **대리지표라 판정에 넣지 않았다**(E-142). 결과적으로 대리지표를 버린 것이
  옳았다 — 엔트로피만 봤다면 열화로 오판했을 것이고, 직접지표는 정반대를 가리킨다.

**val 궤적 요약**: gs200이 봉우리, 이후 gs250·gs300으로 완만히 되돌린다. gs300은 gs150 대비
3과목 상승·2과목 하락. 즉 **후반 100스텝에서 순개선이 없다** — 열화도 아니지만 학습이 더
얻어낸 것도 없다. gs200 근처가 이 런의 최적점으로 보인다.

**다음**: held-out 1030문항(GSM8K500+MATH500+AIME30, `format_fair`·`strict_boxed` 양쪽,
항목ID 동결). ⚠️컴퓨트 차단으로 eval 잡 제출 경로부터 확인해야 한다.
⚠️**이 런은 rvseg 부록 lineage**(init `b2p2_rvseg_sft`, E-093 폐기 판정) — **복제 결과로
보고 금지**, 부록 수치로만.

### E-147 (0728 06:07) — held-out eval은 현재 컴퓨트로 **불가능**하다. 축소판을 돌리지 않는 이유

b2p gs300 held-out eval(E-146의 다음 단계)의 실행 경로를 조사했다. 결론: **지금 쓸 수 있는
컴퓨트로는 비교 가능한 eval을 돌릴 수 없고, 축소해서 돌리면 비교 자체가 무의미해진다.**

**기준 프로토콜**(B0 기준선 `eval/base_matched_1030_v2`를 만든 `runs/archive/h100std_basearm_1030_eval.yaml`):
- 1,030문항(gsm8k+math500+aime2024, 각 max 500)
- **`--num_samples 8`(avg@8)**, `--tp_size 4`, **@4k와 @16k 두 패스**
- **H100×4 · basicvc**

**가용 컴퓨트**: basicvc는 제출 전면 거부. msrresrchvc는 H100 없음 + **A100-80GB 사용자당
1장**, 선점창 ~3.5h. 8B 모델 자체는 80GB에 tp=1로 충분히 올라가지만, 작업량이
**avg@8 × 1,030문항 × 2토큰예산**이다. tp 4→1과 H100→A100을 함께 감안하면 기준 대비 대략
**한 자릿수 배 느려진다**. 3.5시간 창에 들어갈 수 없다.

**축소판을 돌리지 않는 이유.** `num_samples 8→1`, `@16k` 생략으로 줄이면 1-GPU 창에 넣을 수는
있다. 그러나 B0 기준선이 **avg@8 @4k/@16k**이므로 그렇게 얻은 수치는 **기준선과 비교할 수
없다**. 비교 불가능한 숫자를 만드는 데 희소하고 불안정한 컴퓨트를 쓰는 것은 순손실이다.
게다가 이 런은 애초에 **rvseg 부록 lineage**라, 부록 수치 하나를 위해 본선 발사용 자원을
소모할 이유가 더욱 없다.

**결정: eval은 컴퓨트가 열릴 때까지 보류.** 대신 그 순간 바로 쏠 수 있도록 **기준 프로토콜과
동일한 eval 런처를 미리 만들어 검증해 둔다**(다음 틱). 발사 우선순위는 그대로
①SFT2 쌍(rvfull) → ②본선 RL 3종 → ③이 부록 eval.

**참고**: `scripts/eval_vllm_1030.py`는 현행 SFT 런처(`h100std_sft_*`)에도 이미 배선돼 있어,
SFT2 쌍이 돌면 그 산출물에 대한 축약 eval은 자동으로 따라온다(math500+aime2024, 100문항,
@8k). 이건 게이트용이지 held-out 판정용이 아니다.

카나리아 65회째 RED.

### E-148 (0728 06:40) — 차단 진단 대수정: 0727 그룹정책 판정 철회, 클라이언트 버전 기각, 그리고 내가 지어낸 ID 하나

사용자가 **정상 동작하는 H100 잡의 전체 스펙**을 제공했고, 그것이 우리 진단 두 개를 뒤집었다.

**① 0727 진단 철회 — `GroupPolicy` 태그는 정책이 아니라 제출자 ID였다.**
그 잡의 `tags.GroupPolicy`, `createdBy.userObjectId`, `userId`가 **모두 같은 값**
(`e9deff52-...`)이다. 즉 amulet이 제출자의 AAD object id를 태그로 찍은 것이고, 가입할 수 있는
정책 객체가 아니다. "그 정책에 우리를 넣어달라"는 요청은 **"그 사람이 되게 해달라"**는 말이었다.
독립 검토가 결정적 증거를 더했다: VC의 `groupPolicies`는 17개 객체이고 각각 개별 사용자 OID
이름을 갖는데 **우리 OID도 그 사용자 OID도 거기 없다**. 0727에 근거로 삼은
`expand-sku -t "...:e9deff52"` 실패는 **모든 OID에 대해 똑같이 실패**하므로 아무것도 증명하지
않았다. 교훈: **한 번의 실패 프로브를 근거로 쓰기 전에 그 프로브가 정상 대조군에서도 실패하는지
확인할 것.** 음성 대조군 없는 프로브는 증거가 아니다.

**② 클라이언트 버전 기각 — 세 버전 동일 실패.** 되는 잡이 `amlt 11.14.2`였고 우리는 그 버전만
안 써봤다. 내부 인덱스에서 격리 venv로 설치해 **같은 신원·같은 카나리아**로 제출했다.

| amlt | 결과 |
|---|---|
| 11.9.1 (차단 시작 시점) | `UserError: The virtual cluster does not exist` |
| **11.14.2** (정상 잡의 버전) | **동일** |
| 11.16.0 (현행) | 동일 |

스택은 `handle_job_submission`의 `HttpResponseError` — **서비스 거부**다. ⚠️중간에 11.14.2가
"target could not be found"를 내서 내가 "돌파구"라고 성급히 보고했는데, 그건 새 venv의 빈 타깃
캐시 때문이었다. `amlt target list sing`으로 캐시를 채우니 같은 서버 에러였다. **다른 에러 문구를
다른 원인으로 속단하지 말 것.**

**③ 내 오류: 요청서에 검증하지 않은 object id를 썼다.** 0728 요청서 초안에 제출자 OID를
`3e6b95a6`로 적었는데 **그런 값은 없다**. 실제는 `az ad signed-in-user show`로 확인한
`a22660cc-8fa9-4dd1-b5fc-eafa9718e257`(`sc-vhr286860@microsoft.com`). 그대로 발송됐다면 지원팀이
조회할 수 없는 요청이 됐다. 즉시 정정. **교훈: 외부로 나가는 문서의 식별자는 예외 없이 명령으로
확인하고 쓸 것 — 기억이나 추정으로 쓰지 말 것.**

**남은 확정 사실**: VC ARM id는 정상 잡과 바이트 동일 · ARM 읽기(쿼터 GET)는 지금도 성공하며
per-user 쿼터 행이 `defaultGroupPolicyOverallQuotas`에서 나온다(기본 정책이 우리를 **호가**는
한다) · 실패는 admission 단계에서만 발생. **유력 가설(미확인)**: 0716 GCR 재할당 때 우리 신원이
새 allocation으로 이관되지 않았고 구 경로가 0726 05:49에 폐기됨 — 0715 재할당 준비 중 동일 에러가
일시 출현했고, allocation 갱신은 각 랩 GPU delegate 소관인데 우리 쪽은 접촉된 적이 없다.
[[gcr-bonete-maintenance-realloc-0716]]

**산출물**: 요청서에 correlation ID 2쌍(0726·0728) 추가, 0727 절은 삭제 대신 **SUPERSEDED 표시**
(증상 서술은 여전히 유효하므로), CLAUDE.md Compute 절도 같은 오진을 담고 있어 정정.

### E-149 (0728 07:35) — 워크스페이스 축을 처음 시험했다. `gcrllm2ws`에서 **다른 실패 모드**가 나왔다

사용자가 "그럴 리 없다"고 밀어붙여 재검토했고, 그 결과 **한 번도 안 본 축**이 두 개 있었다.

**① 프로젝트 등록은 원인이 아니다.** 새 amlt 프로젝트(`vcprobe0728`, version 11.16.0으로 깨끗
하게 생성)에서 제출해도 동일. 우리 `.amltconfig`의 `"version": "11.9.1"` 흔적은 무관하다.

**② 워크스페이스가 세 개나 있었다.** 24개 구독 전수 조회 결과 접근 가능한 AML 워크스페이스는
`msra-sh-aml-ws`(westus2·우리가 쓰던 것), **`singularity`**(eastus, RG `aml`), **`gcrllm2ws`**
(westus3, RG `gcrllm`, "3rd Party LLMs" 구독). 셋 다 등록하고 `amlt run -w <ws>`로 각각 basicvc에
카나리아를 던졌다.

| workspace | 결과 |
|---|---|
| `msra-sh-aml-ws` (westus2) | `UserError: The virtual cluster does not exist` |
| `singularity` (eastus) | **동일** — correlation op `0334ddc5e99bf24d35be991fa7611382` / req `5afe7ea439f0dbe4` |
| **`gcrllm2ws`** (westus3) | **다름!** VC 해석을 통과해 실험까지 생성(`Created new experiment ... on msrresrchbasicvc`)하고, **데이터스토어 쓰기 권한**에서 막힘 |

`gcrllm2ws`의 에러 전문:

```
You don't have permission to perform this operation on the AzureML workspace.
Action: Microsoft.MachineLearningServices/workspaces/datastores/write.
Scope: /subscriptions/ca45784d-.../resourceGroups/gcrllm/.../workspaces/gcrllm2ws/datastores/...
Ensure you have the 'AzureML Data Scientist' role ...
```

**왜 중요한가.** 이 경로는 "VC가 없다"가 **아니다**. 코드 패키지 업로드 단계에서 권한으로 멈춘
것이고, 그 앞 단계는 통과했다. 다만 **여기서 VC admission까지 갔다고 단정할 수는 없다** — 업로드는
제출보다 앞이므로 admission은 아직 시험되지 않았다. 그럼에도 **요청 가능한 구체적 대상**이 처음
생겼다: `gcrllm2ws`에 **`AzureML Data Scientist` 역할** 하나. "Singularity allocation을 고쳐달라"
보다 훨씬 작고 명확한 요청이다.

**부수 확인**: `singularity` 워크스페이스는 **eastus**에서 같은 에러를 냈다. 즉 westus2·eastus 두
리전에서 동일 → 리전/서비스 인스턴스 문제가 아니라 신원 축이라는 기존 판단과 정합.

**남은 대기**: 사용자 요청으로 30분 간격 재신청 루프로 전환.

---

## E-150 (0731 07:44 UTC) — 차단 해제. SFT2 쌍 발사

**차단이 풀렸다.** 0726 05:49에 시작된 `msrresrchbasicvc` 제출 차단이 **5일 2시간**만에 해소됐다.
0728 08:06~0729 07:37 사이 마흔다섯 틱 연속 `exit 1 1 1`이었고, 0729 07:37 이후 루프가 끊긴 구간을
지나 0731 07:44 틱에서 처음으로 `exit 0`이 나왔다.

```
exit: 0 0 1
  ① h100std_sft_b0p2_rvfull  → 제출 성공
  ② canary (singularity ws)  → 제출 성공
  ③ canary (gcrllm2ws)       → 여전히 AzureML Data Scientist 권한 부족
```

**해제 시점은 특정할 수 없다.** 0729 07:37과 0731 07:44 사이 어딘가에서 열렸고, 그 구간에 틱이
없었다. 원인도 확인되지 않았다 — 우리가 바꾼 것은 아무것도 없고(같은 클라이언트·같은 yaml·같은
신원), 서비스 측에서 무언가 복구된 것으로 보인다. **"일시적"이라던 사용자의 초기 판단이 결과적으로
맞았다**; 다만 5일은 그 가설로 예측할 수 있는 길이가 아니었다.

**즉시 발사** — 사전 합의된 순서대로:

| 잡 | 실험명 | SKU | 상태(발사 +2m) |
|---|---|---|---|
| `h100_sft_b0p2_rvfull` (control) | wired-kiwi | 80G4-H100 | queued |
| `h100_sft_b2p2_rvfull` (meta) | exact-tick | 80G4-H100 | queued |

**노드는 아직 안 잡혔다.** 둘 다 `queued` = 제출은 수락됐고 Standard 티어에서 노드 배정 대기 중.
`preparing → queued`까지가 확인된 진행이다.

**매니페스트 동결**: `docs/manifests/sft2_pair_20260731.json`. 검사 통과 —
arm 간 차이가 **설계된 4개뿐**(dataset_path / model_name_or_path / output_dir / run_name),
다섯 번째 차이 없음. trained-token 노출비 meta/control = **1.2951**(mean +48.9 tok)로 0727 측정치와
일치. grader는 `format_fair`로 동결.

**카나리아 ②는 `failed`**. 1-CPU `echo` 잡이므로 실패할 코드가 없다 — Basic 티어 선점이거나
E-146에서 본 teardown 아티팩트로 보인다. 카나리아의 목적은 admission 시험이었고 **제출이 수락된
시점에 그 목적은 달성**됐다. 본선 판정에는 무관.

**③ `gcrllm2ws`는 여전히 막혀 있다** — 같은 권한 에러. 이 경로는 이제 불필요하지만, 요청 (b)가
아직 열려 있다는 사실은 남는다.

**다음**: 두 SFT 완주(각 ~2h) → `models/{b0p2,b2p2}_rvfull_sft` 4샤드 HF 착지 확인 →
`h100std_rq3v2f_{b0p,b2p,b3p}` 발사.

---

## E-151 (0731 10:46 UTC) — E-150 발사분 전멸. 근인 = 폐기된 프로필 GH_TOKEN

**E-150에서 발사한 SFT2 쌍은 둘 다 실패했다.** 13분·11분 돌고 죽었고, 나는 3시간 동안
`queued`만 보고 확인하지 않았다. 사용자가 물어서 알았다.

**근인은 컴퓨트가 아니다.** `user_logs/std_log.txt` 첫 줄:

```
curl: (22) The requested URL returned error: 401
tar (child): /tmp/metacognition.tar.gz: Cannot open: No such file or directory
```

코드 tarball을 못 받아 `/scratch/metacognition`이 통째로 없었고, 이후 가드
(corpus 확인 → pusher 확인)가 순차 발동해 `ABORT window; exit 1`. **가드는 제대로 작동했다** —
스테일/부재 tarball로 비내구 실행하는 것을 막았다.

**401의 진짜 원인 — 토큰 두 개가 존재하고, 죽은 쪽이 이겼다:**

| 출처 | sha256[:12] | asset 491027629 |
|---|---|---|
| 셸 프로필(기본 export) | `e966d6aa…` | **HTTP 401** |
| `.env` | `c95ca4e4…` | HTTP 200 |

0727 GH_TOKEN 유출 후 GitHub이 유출분을 폐기했고 새 토큰은 `.env`에 들어갔지만,
**프로필의 죽은 값이 그대로 남아 `.env` 값을 가렸다**. 내 30분 루프 틱 명령에는
`set -a; source .env; set +a`가 빠져 있었고, 그래서 죽은 토큰이 노드로 갔다.
런처는 `GH_TOKEN: ${GH_TOKEN}`으로 제출 시점 환경을 확장하므로 이 누락이 곧바로 401이 된다.

**놓친 신호**: 제출 로그에 `automatically extracting WANDB_API_KEY from your .netrc file`
경고가 있었다. 환경변수가 없어서 폴백했다는 직접 증거였고, 나는 지나쳤다.
**교훈: 이 경고는 "환경이 로드되지 않았다"의 카나리아다.**

**조치 1 — 발사 전 게이트를 넣고 재발사**(musical-mutt / clever-frog, 둘 다 queued):
세 토큰 존재 + `GH_TOKEN`으로 asset이 실제 HTTP 200인지 확인한 뒤에만 `amlt run`.
재발사 로그에서 `.netrc` 경고가 사라진 것으로 환경 전달을 확인했다.

**조치 2 — HF 점검에서 두 번째 지뢰 발견**: `iamseungpil/metacot-sft2-4g`가
**존재하지 않았다**(`RepositoryNotFoundError`). 두 SFT2 런처 모두 중간 체크포인트를 이 repo에
푸시하고, 발사 직후 `--probe-only`로 쓰기 가능성을 **동기 확인**한 뒤 실패하면 abort한다.
즉 토큰을 고쳐도 노드를 잡자마자 여기서 또 죽을 상태였다. private model repo로 생성하고
런처와 동일한 probe를 두 prefix(`b0p2_4g`/`b2p2_4g`)로 실행해 `write+delete ok` 확인,
잔여 파일 없음(`.gitattributes`만).

**HF 현황**(0731 점검):

| repo | type | files | last | 비고 |
|---|---|---|---|---|
| `iamseungpil/metacot` | dataset | 1534 | 0726 11:18 | `models/`에 rvfull **0개** |
| `iamseungpil/metacot` | model | 97 | 0724 08:50 | checkpoint-253/506/759 |
| `iamseungpil/metacot-sft2-4g` | model | — → 1 | **신규 생성** | private, probe 통과 |
| `iamseungpil/metacot-h200-triobj-dcpo-v3` | model | 388 | 0728 05:14 | RL 체크포인트 |

dataset `models/`에 있는 것: `b0p_v8base_strict_sft`, `b2p_v8meta_strict_sft`(SFT1 init 쌍),
`b2p2_rvseg_sft`(부록 lineage). **rvfull SFT2는 여전히 없다** — 0726 이후 아무것도 올라가지 않았고,
차단 기간과 정확히 겹친다.

**b3p를 먼저 못 하는 이유 — 실측**. b3p의 init 해석 블록을 그대로 실행한 결과:

```
검사 PAIR_eb16_sft   -> [False, False]
검사 PAIR_sft        -> [False, False]
[init] no suffix has BOTH b0p2_rvfull* and b2p2_rvfull* on HF
       - refusing to pair mismatched arms
   exit=1
```

지금 b3p를 발사하면 노드를 잡은 직후 이 지점에서 죽는다. 이는 버그가 아니라 설계된 거부다 —
짝이 맞지 않는 arm끼리 비교하는 것을 막는다.

---

## E-152 (0731 14:10 UTC) — 두 번째 전멸의 근인: 런처가 tarball보다 새 인터페이스를 호출

**E-151에서 토큰을 고쳐 재발사한 쌍도 전멸했다**(14분·13분). 토큰은 이번엔 맞았다 —
`staged SFT1 init OK`, corpus 존재, pusher 스크립트 존재까지 전부 통과했고, 다음 줄에서 죽었다:

```
push_sft_ckpts_to_hf.py: error: unrecognized arguments: --probe-only
[YAML][pusher] destination not writable - this run would be NON-DURABLE; ABORT window
```

**pin된 asset이 런처보다 낡았다.** asset 491027629는 **0727 04:43** 스냅샷인데,
`--probe-only`는 내가 **0727 16:15**에 추가했다. 런처(0728 편집)는 새 플래그를 호출하고
tarball 안 스크립트는 그것을 모른다. argparse가 거부 → 가드 발동 → abort.
차단 5일 동안 발사가 없었으므로 이 불일치가 드러날 기회가 없었다.

**내 사전 점검의 구멍**: E-151에서 나는 tarball에 파일이 **존재하는지**만 확인하고
그 파일이 **런처가 호출하는 인터페이스를 지원하는지**는 확인하지 않았다.
존재 확인은 인터페이스 호환 확인이 아니다.

**조치 — 최소 교체로 asset 재빌드**:

| | |
|---|---|
| 새 asset | `496769353` = `metacognition_rq3v2_0731_probe.tar.gz` |
| md5 | `b43789e0c89b93451597ec767198ed27` (35,736,417 B) |
| 변경 | `scripts/push_sft_ckpts_to_hf.py` **한 개만** |
| 검증 | 두 asset을 모두 풀어 **459개 파일 전수 diff** → 그 1개만 differ, 파일 수 동일 |

두 런처의 `CODE_TAR_REVISION`을 동시에 갱신(양 arm 동일 asset 유지). lint 45개 통과,
추가 주석에 아포스트로피 0(E-134 방어).

**전수 인터페이스 점검**(이번에 새로 한 것). 런처가 호출하는 모든 진입점을 새 tarball 버전으로
대조했다:

| 스크립트 | 런처가 쓰는 인자 | tarball 지원 |
|---|---|---|
| `push_sft_ckpts_to_hf.py` | `--probe-only` 외 | OK (신규 asset) |
| `push_models_hf.py` | `--model_path --model_name --repo_id` | OK |
| `verify_eos_invariant.py` | positional CKPT | OK |
| `eval_vllm_1030.py` | `--benchmarks --max_problems --max_tokens --max_model_len --temperature --tp_size --num_samples` | OK |

term-eval 단계까지 미리 본 이유: 학습 2시간을 태운 뒤 거기서 죽으면 손실이 훨씬 크다.

**발사 전 게이트 G1~G4**(전부 통과해야 `amlt run`):
G1 세 토큰 로드 · G2 `GH_TOKEN`으로 새 asset이 실제 HTTP 200 · G3 두 prefix로 write probe ·
G4 SFT1 init 쌍이 HF에 실재. 이 게이트는 E-151과 E-152를 죽인 두 조건을 각각 잡는다.

**발사**: `fond-panda`(b0p2 control) / `neat-hyena`(b2p2 meta), 둘 다 80G4-H100 Standard.

**누적 손실**: H100 4-GPU 윈도 4개(각 ~13분). 전부 학습 시작 전에 죽어 체크포인트 손실은 없다.

---

## E-153 (0731 15:2x UTC) — SFT2 쌍 완주·착지, RL 3-arm 발사

**rvfull lineage의 SFT2가 프로젝트 최초로 완주했다.** E-097이 지목한 사다리의 빠진 단이 메워졌다.

| arm | 실험 | 스텝 | 최종 loss | HF 착지 |
|---|---|---|---|---|
| control `b0p2` | fond-panda | 303 (3ep) | 0.053 | `models/b0p2_rvfull_sft/` 4샤드+config+tokenizer (14파일) |
| meta `b2p2` | neat-hyena | 309 (3ep) | — | `models/b2p2_rvfull_sft/` 동일 14파일 |

양쪽 std_log에서 `[YAML][push] all 4 shards durable` 확인. H100 4장에서 arm당 **약 20분**
(예상 2시간의 1/6). term-eval(`verify_eos_invariant` → `eval_vllm_1030` → `measure_sft_gate rc=0`)도
양쪽 통과. durable pusher는 `metacot-sft2-4g`에 b0p2_4g 32파일 / b2p2_4g 16파일을 남겼다.

**GPU quota 정리**: basicvc H100은 16 GPU/user이고 SFT 두 잡이 8개를 쥔 채 런처 말미의
`sleep 43200`(12h) keep-alive로 놓지 않는다. RL 3-arm은 12개가 필요하므로(8+12=20>16)
**HF에서 4샤드 착지를 확인한 뒤에만** 두 잡을 취소했다(파괴조작 3율: LIST→decide→execute).
취소 후 `killed`, 산출물은 전부 durable.

**RL 발사 게이트 G1~G5 — E-151/E-152 재발 방지를 명시적 검사로 전환**:

| 게이트 | 결과 |
|---|---|
| G1 세 토큰 로드(`.env`) | PASS |
| G2 asset 490407111이 GH_TOKEN으로 HTTP 200 | PASS |
| G3 **런처가 호출하는 모든 스크립트 인자를 그 asset 버전으로 대조** | PASS |
| G4 b3p init 해석 블록 로컬 실행 | `resolved PAIR_sft` exit 0 |
| G5 `test_launcher_yaml_lint.py` | 45 passed |

**세 RL 런처는 동일 asset 490407111을 pin**한다(매치드 유지). G3에서 특히 확인한 것:
런처는 `push_ckpts_to_hf.py`의 `--token` 인자를 sed로 패치한 뒤 grep으로 검증하고 실패 시
abort하는데, **이 asset에는 sed 대상 문자열이 없다**(이미 패치된 형태). sed는 no-op이지만
grep이 검사하는 것은 **최종 상태**이므로 통과한다. E-152의 probe는 인터페이스의 존재를
가정했기 때문에 죽었고, 이쪽은 결과를 검사하기 때문에 산다 — 같은 종류의 가드라도 설계가 다르다.

**발사**(전부 80G4-H100 Standard, queued):

| arm | 실험 | 잡 |
|---|---|---|
| b0p (control GRPO) | mature-swift | `:h100_rq3v2f_b0p` |
| b2p (meta-SFT2 GRPO) | steady-mule | `:h100_rq3v2f_b2p` |
| b3p (full triobj PMI-shift) | fair-calf | `:h100_rq3v2f_b3p` |

셋을 **동시에** 띄운 이유: RQ2는 `b3p − b2p`, 복제축은 `b3p − b0p`이므로 arm이 서로 다른 시점의
노드·드리프트를 타면 매치드 설계가 깨진다.

**미해결(다음 여유에 규명)**: 두 SFT2 arm의 총 스텝이 **303 vs 309**(epoch당 101 vs 103).
corpus는 둘 다 정확히 1763행 1:1임을 parquet에서 직접 확인했으므로, 유효 샘플이 1616 vs 1648로
~2% 어긋난다. `max_length 4096`의 truncation인지 drop인지 `src/training/sft.py`로 가려야 한다.
런처가 주장하는 dose-matched에 직접 걸리는 지점이다.

---

## E-154 (0731 16:1x UTC) — SFT2 스텝 불일치(303 vs 309) 규명: dose는 1:1이 아니다

**관측**: SFT2 두 arm의 총 스텝이 303(control) vs 309(meta)였다. corpus는 parquet에서 직접
세어 둘 다 정확히 1763행이므로, 학습 파이프라인 안에서 갈라진다는 뜻이다.

**재현**: 로컬에서 `prepare_sft_dataset`을 노드와 동일한 토크나이저·`max_length=4096`으로 돌려
관측 스텝을 **정확히 재현**했다.

| arm | raw | kept | dropped | eval(5%) | train | steps/ep(ceil/16) | ×3ep |
|---|---|---|---|---|---|---|---|
| control | 1763 | 1701 | **62** | 85 | 1616 | 101 | **303** |
| meta | 1763 | 1725 | **38** | 86 | 1639 | 103 | **309** |

**드롭 경로는 truncation이 아니다.** `max_length` 초과는 `full_ids[:max_len]`로 자를 뿐이고,
행이 사라지는 곳은 그 다음이다:

```python
ds = ds.filter(lambda row: row["num_target_tokens"] > 0)
```

**시나리오별 분해 — 드롭은 verify에 집중되고 control에서 23행 더 난다**:

| arm | scenario | raw | kept | dropped |
|---|---|---|---|---|
| control | verify | 1209 | 1149 | **60** |
| control | redirect | 554 | 552 | 2 |
| meta | verify | 1209 | 1172 | **37** |
| meta | redirect | 554 | 553 | 1 |

**메커니즘**: 모든 행이 `wrong_prefix`를 갖고 있어(매니페스트 확인) 전부 segment-mask 경로를 탄다.
control은 **meta 블록이 제거된 twin**이므로 prefix를 마스킹하고 나면 학습 대상 토큰이 하나도
남지 않는 행이 생긴다 — verify에서 60행. meta arm에서는 meta 블록 자체가 학습 대상으로 남아
그 행들이 살아남는다. **버그가 아니라 meta-removed twin 설계의 필연적 귀결이다.**

**타당성에 대한 판정**:
- 런처/매니페스트의 "1763 rows EXACT 1:1"은 **raw 수준에서만 참**이다. 학습 수준에서는 1701 vs 1725,
  실제 dose는 1616 vs 1639 — **1.4% 차이**.
- 따라서 "dose matched (3 epochs, lr 1e-5, ...)" 주장은 **step-matched가 아니다**. 보고 시 각주 필요.
- 다만 차이의 유일한 원천이 meta 블록이므로 **독립적 교란은 아니다**. meta 블록을 제거하면
  일부 행의 학습 신호가 필연적으로 사라진다.
- **시나리오 구성 왜곡은 미미**하다: 학습되는 verify 비율이 control 67.55% vs meta 67.94%
  (원본 68.58%) — 0.4%p 차이. 매니페스트의 "identical scenario split" 주장은 raw에서는 참이고
  학습 후에도 실질적으로 유지된다.

**남는 선택지**(지금 결정하지 않음): ①현 상태로 진행하고 1.4% dose 차이를 각주로 보고,
②control에서 드롭되는 60행을 meta에서도 제외해 강제 step-match. ②는 meta arm의 데이터를
control의 결함에 맞춰 깎는 것이라 그 자체가 새로운 선택 편향이다. 현재는 ①로 진행 중이며
RL은 이미 이 init 위에서 돌고 있다.

**동시 관측(RL 3-arm, 0731 16:03)**: 전부 running. b0p의 초기 `grad_norm 66.99`(E-153에서
주시 대상으로 걸어둔 값)는 **warmup 스파이크였고 해소**됐다 — s11에서 1.97. 세 arm 안정 구간:

| arm | step | grad_norm | correctness/mean | durable ckpt |
|---|---|---|---|---|
| b0p | 11 | 1.97 | +0.113 | gs5, gs10 |
| b2p | 9 | 1.00 | +0.293 | gs5 |
| b3p | 9 | 0.40 | +0.382 | gs5 |

---

## E-155 (0731 17:33 UTC) — b0p 사망을 amlt는 running으로 보고했다. wandb가 잡았다

**b0p이 죽어 있었고 amlt는 `running`이었다.** heartbeat는 10초 간격인데 마지막이 **17:15:41**,
발견 시각 17:33 기준 **18분 침묵**. 같은 시각 b2p 17:32:20 / b3p 17:34:06은 정상이었다.

| arm | last HB | progress | `Caught signal 15` |
|---|---|---|---|
| **b0p** | **17:15:41** | 31/300 | 1 |
| b2p | 17:32:20 | 35/300 | 2 |
| b3p | 17:34:06 | 30/300 | 0 |

**SIGTERM 자체는 즉사가 아니다** — b2p는 두 번 받고도 계속 돌았다. b0p만 신호 뒤 프로세스가
멎었고, 노드는 살아 있어(런처 말미 keep-alive) **amlt가 running을 유지했다. 그래서 자동 retry도
걸리지 않았다.**

**탐지 경로가 결정적이었다.** `amlt status`만 봤다면 놓쳤다. wandb에서 `rq3v2f_b0p`의 state가
`crashed`이고 step이 30에서 정체된 것을 보고 역추적했다. → **틱 절차에 heartbeat 신선도 검사를
추가한다. `running`은 프로세스 생존의 증거가 아니다.**

**복구**: durable ckpt가 `global_step_30`까지 있고 사망은 step 31 직후 → **손실 1스텝**.
gs30 확인 후(파괴조작 3율) 잡 취소 → 게이트 통과 후 재제출(`sharp-llama`).
`pull_resume_ckpt.py --config_name rq3v2f_b0p`가 HF에서 gs30을 끌어와 이어받는다.

**부수 확인 — wandb project가 둘로 갈린다**: SFT2는 `metacot-math`, RL은 **`metacot-dcpo-v4`**
(그룹 `rq3_matched_ladder`). 처음에 `metacot-math`를 조회해 "RL run이 없다"고 잘못 볼 뻔했다.

**같은 step에서의 arm 비교(train reward, 공통 구간 step 18~29)**:

| | b0p | b2p | b3p |
|---|---|---|---|
| 구간 평균 | 0.406 | 0.399 | **0.467** |
| 최고를 차지한 step 수 | 1 | 0 | **11** |

복제축 `b3p − b0p = +0.061`, RQ2 `b3p − b2p = +0.068`. **방향은 Instruct T1과 일치**하지만
이는 **train reward**이며 held-out이 아니다. 단일 step은 ±0.2 튀고(16:19의 b3p 0.660은 노이즈),
12 step 평균이라 순위가 안정적으로 보일 뿐이다. **판정은 gs50 val부터.**

**총 스텝은 300**(`Training Progress: n/300`), 스텝당 약 220s → 완주까지 약 16시간.

---

## E-156 (0731 17:56 UTC) — 같은 사망이 b2p에서 반복. 패턴 확정

**E-155의 b0p 사망과 동일한 일이 b2p에서 22분 뒤 재현됐다.** 우연이 아니라 구조적 실패다.

| arm | wandb state | last step | heartbeat | amlt |
|---|---|---|---|---|
| b0p (구 run) | crashed | 30 | 17:15:45 | running(→취소) |
| **b2p** | **crashed** | 34 | **17:32:18** | **running** |
| b3p | running | 33 | 17:54:58 | running |

**메커니즘**: Standard 티어에서 SIGTERM이 오고 학습 프로세스가 죽는다. 그러나 런처 말미의
keep-alive(`sleep 43200`)가 노드를 붙잡으므로 **잡은 failed로 떨어지지 않는다**. 따라서
**amlt의 자동 retry가 영원히 걸리지 않는다.** `amlt status: running`은 프로세스 생존의 증거가
아니며, 사람이 heartbeat를 보고 감지해 재제출해야만 복구된다.

**복구**: durable ckpt는 5스텝 간격이라 **스텝 손실은 거의 없다** — b0p은 gs30에서 1스텝,
b2p는 gs35에서 0스텝. 실제 비용은 **감지 지연 + 재초기화(~15분)**다.

| arm | 재제출 | resume |
|---|---|---|
| b0p | `sharp-llama` (17:36) | gs30 |
| b2p | `golden-gnu` (17:56) | gs35 |

**대응 선택**: 근본 해결은 런처의 keep-alive를 제거해 학습 사망 시 잡도 죽게 만드는 것이다
(그러면 amlt가 자동 retry). 그러나 그러려면 세 arm을 전부 재발사해야 하고 현재 진행분(gs30~35)을
버린다. 완주까지 300스텝 × 220s ≈ 16시간이 남은 시점에서, **감지 주기를 25분 → 15분으로 좁히는
쪽이 손실이 작다**고 판단해 그렇게 조정한다. 사망 1회당 손실은 대략 (평균 감지지연 7.5분 +
재초기화 15분) ≈ 22분이다.

**틱 절차에 고정**: 매 틱 세 arm의 `[HB ...]` 최신 시각을 확인하고, **5분 이상 정지 시 사망으로
간주**한다. wandb `state=crashed`도 동일 신호. `amlt status`만으로는 절대 판정하지 않는다.

**정정(E-155/E-156 후속, 0731 18:13)**: b0p의 실제 resume 지점은 **gs30이 아니라 gs25**였다
(`Training Progress: 25/300`, 로그에 `global_step_25`). durable에는 gs25·gs30이 모두 있었으나
`pull_resume_ckpt`는 newest-first로 고르되 **torn checkpoint를 건너뛴다** — step 31 도중 사망하며
gs30이 불완전하게 남은 것으로 보인다. 따라서 **b0p 손실은 1스텝이 아니라 6스텝(31→25)**이다.
E-155의 "손실 1스텝"을 이 값으로 정정한다. 교훈: durable 목록에 gsN이 보인다고 해서 그 지점에서
재개된다는 보장은 없다 — **재개 후 실제 progress를 확인해야 한다.**

0731 18:13 상태: b0p running(gs25, HB 신선) · b2p queued 16분(노드 대기) · b3p running(38/300, HB 신선).

---

## E-157 (0731 18:30 UTC) — b3p 선점. 다만 이번엔 자동 복구 경로다

**b3p가 3시간 만에 선점됐다**(step 38). 그러나 **E-155/E-156과 죽는 방식이 다르다**:

| | b0p·b2p (E-155/E-156) | b3p (지금) |
|---|---|---|
| 학습 프로세스 | 死 | 死 |
| 잡 상태 | **running 유지** | **queued로 하강** |
| amlt 자동 retry | **안 걸림** | **걸림** |
| 복구 주체 | 사람(감지→재제출) | amlt |

같은 SIGTERM이라도 노드가 회수되면 잡이 queued로 내려가 재스케줄되고, 노드가 살아남으면
keep-alive가 잡을 running으로 붙들어 retry를 막는다. **후자만 사람이 개입해야 한다.**
b3p는 durable `gs35`가 있으므로 노드를 다시 잡으면 거기서 재개한다 — 손실 3스텝(38→35).

**0731 18:30 상태 — 세 arm 중 하나만 가동**:

| arm | 상태 | progress | durable |
|---|---|---|---|
| b0p | running (HB 18:30:17) | **29**/300 (25→29 전진, resume 정상) | gs25, gs30 |
| b2p | queued 32분 | — | gs30, gs35 |
| b3p | queued (선점) | 38에서 중단 | gs35 |

**노드 확보가 느려졌다.** H100 12장을 동시에 요구하는 상황에서 basicvc가 혼잡하다.
b2p는 재제출 후 32분째 대기다. 기다리는 것 외에 취할 수단이 없다 — 취소·재제출은 대기열
순서만 뒤로 밀 뿐이다.

---

## E-158 (0731 22:44 UTC) — 첫 held-out 비교(gs50): b3p가 복제축에서 5승 3무 1패

**wandb가 되살아나 val을 읽을 수 있게 됐다.** 22:44 시점에 세 run 모두 `state=running`으로
복귀했고, b0p·b3p의 gs50 `val-aux/*/correctness/mean@1`이 회수됐다. std_log에서는 이미 밀려나
사라진 값이라 **wandb가 유일한 복구 경로였다**. 스냅샷을 `docs/reports/valsnap_rq3v2f_gs50.tsv`로
고정했다.

**gs50 held-out (보상 스케일 −1~+1, 정확도 아님)**:

| 벤치마크 | b0p | b3p | b3p−b0p |
|---|---|---|---|
| prealgebra | 0.581 | **0.677** | +0.097 |
| omni-math | −0.556 | **−0.460** | +0.095 |
| precalculus | −0.459 | **−0.405** | +0.054 |
| number_theory | 0.524 | **0.571** | +0.048 |
| intermediate_algebra | −0.042 | **0.000** | +0.042 |
| gsm8k | 0.899 | 0.899 | 0.000 |
| algebra | 0.474 | 0.474 | 0.000 |
| geometry | 0.030 | 0.030 | 0.000 |
| counting_and_probability | 0.771 | **0.657** | −0.114 |

**5승 3무 1패.** 이긴 다섯은 전부 어려운 축이고 진 하나는 counting_and_probability다.
방향은 **Instruct T1(matched-base 6/6셀 유의승리, MATH +18.8 / AIME +14, 0706)과 일치**한다.

**그러나 재현으로 보고하면 안 된다. 네 가지 제약**:
1. **gs50은 전체의 17%**다. rvseg lineage에서 정점은 gs200이었다 — 지금은 시작 단계다.
2. **동점 3개가 소수점까지 완전히 같다**(gsm8k 0.8990, algebra 0.4737, geometry 0.0303).
   같은 SFT2 init에서 출발해 아직 갈라지지 않았다는 뜻이고, 동시에 **val 세트가 작아 해상도가
   낮다**는 뜻이다(gsm8k는 99문항, 나머지는 30~40문항 수준).
3. **유의성 검정이 아니라 단일 시점 점추정**이다. 셀당 표본이 30~99개면 ±0.1은 우연히 나온다.
4. **RQ2(b3p−b2p)는 아직 불가** — b2p가 49/300으로 gs50 직전이다. b2p 없이는 이 차이가
   **메타 헤드의 효과인지 SFT2 데이터 차이의 효과인지 분리되지 않는다.**

**0731 22:44 상태**: b0p 64 · b2p **49**(gs50 직전) · b3p **52**. 세 arm 모두 HB 신선.
발사 7.5시간, 중단 9회, 전부 복구.

**운영 교훈**: `log view`는 꼬리만 주고 HB가 10초마다 찍혀 **val 라인이 수 분 내 밀려난다**.
이번엔 wandb가 살아나 복구됐지만 그건 운이었다. **val은 관측 즉시 파일로 고정할 것.**

---

## E-159 (0731 23:02 UTC) — 세 arm gs50 완성: 복제축은 b3p 우위, 그러나 RQ2는 반반

b2p가 gs50에 도달해 **세 arm 비교가 처음으로 성립**했다. 스냅샷:
`docs/reports/valsnap_rq3v2f_gs50_three_arm.txt`.

| 벤치마크 | b0p | b2p | b3p | RQ2(3−2) | 복제축(3−0) |
|---|---|---|---|---|---|
| prealgebra | 0.581 | 0.645 | **0.677** | +0.032 | +0.097 |
| omni-math | −0.556 | −0.619 | **−0.460** | +0.159 | +0.095 |
| precalculus | −0.459 | −0.459 | **−0.405** | +0.054 | +0.054 |
| number_theory | 0.524 | 0.429 | **0.571** | +0.143 | +0.048 |
| intermediate_algebra | −0.042 | **0.042** | 0.000 | −0.042 | +0.042 |
| algebra | 0.474 | **0.526** | 0.474 | −0.053 | 0.000 |
| gsm8k | **0.899** | 0.859 | **0.899** | +0.040 | 0.000 |
| geometry | 0.030 | **0.212** | 0.030 | −0.182 | 0.000 |
| counting_and_probability | **0.771** | **0.771** | 0.657 | −0.114 | −0.114 |

- **복제축 `b3p − b0p`: 5승 3무 1패** — 방향은 Instruct T1(6/6 유의승리)과 일치.
- **RQ2 `b3p − b2p`: 5승 0무 4패** — **거의 반반**.
- 추가 계산 **`b2p − b0p`: 4승 2무 3패** — 역시 혼조.

**핵심 해석**: b3p는 control 대비 일관되게 앞서지만, **그 이득이 메타 헤드에서 오는지 SFT2 메타
코퍼스에서 오는지가 gs50 시점에서는 분리되지 않는다.** geometry에서 b2p(0.212)가 b0p·b3p(0.030)를
크게 앞서는 것이 그 방향을 시사한다. E-158에서 복제축만 보고했던 것은 **불완전한 그림**이었고,
RQ2를 함께 놓아야 "메타 헤드의 순효과는 아직 보이지 않는다"는 정확한 서술이 된다.

**제약(E-158과 동일, 유효)**: gs50 = 전체의 17% · 셀당 표본 30~99 · 유의성 검정 없음 ·
rvseg lineage의 정점은 gs200이었음. **판정하기엔 이르다.**

**0731 23:02 상태**: b0p 64 · b2p 50 · b3p 56, 세 arm 모두 `running`. wandb 세 run 모두
`state=running`으로 복귀해 val 조회가 안정화됐다.

---

## E-160 (0731 23:56 UTC) — heartbeat가 신선한 채로 죽는다. 판정 기준 3차 개정

**b0p이 54분간 step 64에서 멈춰 있었는데 heartbeat는 계속 신선했다.** E-155/E-156에서 세운
"HB 신선 = 생존" 규칙의 반례다.

**증거 세 가지가 함께 갔다**:

| 신호 | 값 |
|---|---|
| `gpu0used` | **35198MB로 54분간 1MB도 불변** |
| 비-HB 로그 라인 | **0개**(heartbeat만 반복) |
| wandb step | 64에서 정체(23:02→23:56) |

**원인**: heartbeat는 `kill -0 <PID>`로 **프로세스의 존재만** 확인하는 별도 백그라운드 루프다.
학습이 좀비가 되어도 PID가 남아 있는 한 HB는 계속 찍힌다. **HB는 감시 루프의 생존을 증명할 뿐
학습의 진전을 증명하지 않는다.**

**판정 기준 개정(3차)** — 아래를 모두 만족하면 사망으로 간주하고 복구한다:
1. `Training Progress` 또는 wandb `_step`이 **두 틱 연속 정지**
2. `gpu0used`가 **완전히 불변**(학습 중이면 vLLM sleep/wake로 변동한다)
3. 비-HB 로그 라인이 없음
4. 새 `retry_NNN`이 생기지 않음(생겼으면 amlt 자동 복구 중 → 개입 금지)

**복구**: durable `gs65`가 있고 wandb step은 64였으므로 **손실 0**. 취소 후 게이트 통과하여
재제출(`right-teal`).

**0731 23:56 상태**: b0p 재제출(gs65) · b2p **retry_005**(5번째 자동 retry, 초기화 중) ·
b3p **67**/300 정상. durable: b0p gs65 · b2p gs60 · b3p gs65.
발사 8.7시간, 누적 중단 10회.

---

## E-161 (0801 05:07 UTC) — b0p 4번째 중단은 **자동 복구되는 종류**였다(retry_003)

**관측**: 04:37 step 81(정상) → 05:07 `amlt status=queued`, wandb `state=crashed`, HB 24분 경과.
b0p이 step **83**에서 끊겼다. 발사 이후 b0p 단독 4번째 중단, 전체 13번째.

**직전 3회와 다른 점**: `log list`에 `system_logs/lifecycler/retry_003/`과
`snapshot_capability/retry_003/`이 **새로 생겼다**. 즉 amlt가 스스로 재스케줄 중이다.
E-160 판정 기준 4번("새 `retry_NNN`이 생겼으면 amlt 자동 복구 중 → 개입 금지")에 정확히
해당하므로 **수동 재제출을 하지 않는다**. 직전 3회는 retry가 생기지 않아 사람이 재제출해야
했던 반면, 이번은 선점(preemption) → 자동 재큐 경로다.

**손실 계산**: durable 최신이 `gs80`이고 wandb는 83에서 끊겼으므로 **손실 3스텝**.

**HF durable 건강검진**(같은 시각, `iamseungpil/metacot-h200-triobj-dcpo-v3`):

| arm | 최신 durable | model / optim / extra | 판정 |
|---|---|---|---|
| rq3v2f_b0p | gs80 | 4 / 4 / 4 | 완전 |
| rq3v2f_b2p | gs120 | 4 / 4 / 4 | 완전 |
| rq3v2f_b3p | gs115 | 4 / 4 / 4 | 완전 |

세 arm 모두 샤드가 빠짐없이 올라가 있고 `keep=1` 프루닝도 의도대로 동작한다.
**체크포인트 릴레이 자체에는 결함이 없다** — 중단은 전부 인프라 선점이지 파이프라인 결함이 아니다.

**부수 기록(도구)**: `amlt log view`의 파일 지정은 위치인자가 아니라 `-F <path>`이며
`<exp> :<job>`은 그 뒤에 온다. 위치인자로 주면 "Expected no further arguments"로 죽는다.
또 이 명령은 페이저를 꺼도(`-P`) 10분 안에 반환하지 않는 경우가 있어, 생사 판정은
`status` + `log list`의 `retry_NNN` 유무 + wandb로 끝내는 편이 빠르다.

**0801 05:07 상태**: b0p **83**/300에서 선점 → retry_003 자동 재큐(val지점 1) ·
b2p **119**/300(val 2) · b3p **114**/300(val 2). 발사 14시간, 누적 중단 13회.
gs100 3-arm 비교는 b0p 복귀 후로 미룬다.

---

## E-162 (0801 08:06 UTC) — 세 arm gs100 비교: **총합은 유지, 분해는 뒤집혔다**

b0p이 gs100에 도달해(step 103) 세 arm이 같은 두 지점(gs50·gs100)에서 비교 가능해졌다.
지표는 `val-aux/*/correctness/mean@1`이며 **보상 스케일(−1~+1)이지 정확도가 아니다**.
스냅샷: `docs/reports/valsnap_rq3v2f_gs100_three_arm.tsv`(두 지점 × 9벤치 × 3 arm).

### gs100 held-out 9벤치

| bench | b0p | b2p | b3p | 복제 b3p−b0p | RQ2 b3p−b2p | b2p−b0p |
|---|---|---|---|---|---|---|
| gsm8k | 0.8788 | 0.8788 | 0.8788 | 0.0000 | 0.0000 | 0.0000 |
| algebra | 0.5263 | 0.4737 | 0.5526 | +0.0263 | +0.0789 | −0.0526 |
| counting | 0.8286 | 0.8857 | 0.7714 | −0.0571 | −0.1143 | +0.0571 |
| geometry | 0.1515 | 0.0909 | 0.1515 | 0.0000 | +0.0606 | −0.0606 |
| inter_alg | 0.0000 | 0.1250 | 0.0833 | +0.0833 | −0.0417 | +0.1250 |
| num_theory | 0.4286 | 0.5714 | 0.6190 | +0.1905 | +0.0476 | +0.1429 |
| prealg | 0.5806 | 0.7097 | 0.6129 | +0.0323 | −0.0968 | +0.1290 |
| precalc | −0.3514 | −0.4595 | −0.3514 | 0.0000 | +0.1081 | −0.1081 |
| omni | −0.4286 | −0.6508 | −0.5238 | −0.0952 | +0.1270 | −0.2222 |
| **평균** | **0.2905** | **0.2917** | **0.3105** | **+0.0200** | **+0.0188** | **+0.0012** |

### 두 지점 비교

| 축 | gs50 승-무-패 (평균) | gs100 승-무-패 (평균) |
|---|---|---|
| 복제 `b3p−b0p` | 5-3-1 (+0.0246) | **4-3-2 (+0.0200)** |
| RQ2 `b3p−b2p` | 5-0-4 (+0.0042) | **5-1-3 (+0.0188)** |
| `b2p−b0p` | 4-2-3 (+0.0204) | **4-1-4 (+0.0012)** |

**총합은 안정적이다**: b3p가 통제군을 앞서는 폭이 두 지점에서 +0.0246 → +0.0200으로 거의 그대로다.
한 점이었다면 노이즈로 봐야 했지만 두 점이 같은 부호·같은 크기이므로 **방향 자체는 실재한다고 볼 근거가 생겼다**.

**그런데 그 이득의 출처는 두 지점에서 정반대로 나온다.** gs50에서는 SFT2 메타코퍼스가 대부분을
설명했고(`b2p−b0p` +0.0204 vs `b3p−b2p` +0.0042), gs100에서는 그 관계가 뒤집혀 메타헤드가
거의 전부를 설명한다(+0.0012 vs +0.0188). 총합이 같은데 분해가 뒤집힌다는 것은
**두 성분을 가르는 해상도가 아직 없다**는 뜻이다. E-159에서 "이득의 출처 미분리"라고 적은 판단은
두 번째 점에서도 그대로 유효하며, 오히려 근거가 강해졌다 — 분해가 점마다 뒤집히는 것이야말로
표본이 부족할 때 나타나는 전형적 양상이다.

### Instruct T1과의 대조

보상이 항목당 ±1이라면(−1~+1 범위가 함의한다) Δ보상 = 2·Δ정확도이므로
복제축 +0.0200은 **9벤치 평균 약 +1.0%p**에 해당한다. Instruct 계열 T1(0706)은
matched-base 대비 **6/6 셀 유의 승리, MATH +18.8%p, AIME +14%p**였다.
**방향은 같지만 크기는 자릿수가 다르다.** 현시점에서 "base에서 instruct 성능이 재현된다"고
말할 수 없고, 말할 수 있는 것은 "부호가 같은 약한 우위가 두 지점에서 유지된다"까지다.

### 제약(반복 기재)

셀당 표본 30~99개 · **유의성 검정 없음**(단일 시드, 쌍대 부트스트랩 미적용) ·
gs100은 사전등록 300스텝의 33% 지점 · rvseg 선행 실험의 정점은 gs200이었으므로
**지금 판정하면 정점 이전에서 자르는 것**이다.

**0801 08:06 상태**: b0p **103**/300(val 2) · b2p **139**/300(val 2) · b3p retry_002 재초기화 중
(wandb 132/HB 35분은 잔재, val 2). 발사 17.3시간, 누적 중단 16회.

---

## E-163 (0801 08:40 UTC) — b3p **retry 소진 → `failed`**, 수동 재제출(`pure-stag`)

**관측**: 08:06까지 `running`이던 b3p(`fleet-ray`)가 08:40에 `failed`로 전환됐다.
`log list`의 최신 retry는 `retry_003`에서 멈춰 있고 더 늘지 않는다.
E-161에서 정한 자동복구 경로("새 `retry_NNN`이 생기면 개입 금지")가 **끝난 상태**다 —
amlt의 재시도 한도를 소진했고, 이제부터는 사람이 다시 넣어야 한다.
0801에 관측된 세 가지 중단 양상이 이로써 다 나왔다: ①retry 생성 후 자동복귀(b0p·b2p),
②retry 없이 좀비(E-160), ③**retry 소진 후 terminal `failed`(이번)**.

**손실**: durable `rq3v2f_b3p` gs130이 완전(model/optim/extra = 4/4/4)하고 wandb는 132에서
끊겼으므로 **2스텝**.

**발사 전 게이트(전부 통과)**:

| 항목 | 결과 |
|---|---|
| `GH_TOKEN` | HTTP 200 |
| `HF_TOKEN` | HTTP 200 |
| `WANDB_API_KEY` | 40자 존재 |
| 코드 asset `490407111` | HTTP 200 |
| durable `rq3v2f_b3p` | gs130 완전 |

**재제출**: `h100std_rq3v2f_b3p.yaml -t msrresrchbasicvc` → 실험명 **`pure-stag`**
(이전 `fleet-ray`는 종료). b3p의 네 번째 윈도.

**부수 관측 — durable 프론티어에 '쓰는 중' 스텝이 보인다**: 같은 시각 LIST에서
b0p은 gs105(완전)와 gs110(model 4 / **optim 1** / extra 4), b2p은 gs140(완전)과
gs145(model 4 / **optim 2** / extra 4)가 함께 보였다. 최신 스텝이 불완전한 것은 결함이 아니라
**푸셔가 업로드 중**이라는 뜻이며, `pull_resume_ckpt.py`가 3종 완비(≥4샤드) 스텝만 고르므로
재개 지점은 자동으로 직전 완전 스텝이 된다. 불완전 스텝을 보고 릴레이가 깨졌다고 판단하면 안 된다.

**0801 08:40 상태**: b0p **110**/300(val 2, durable gs105) · b2p **145**/300(val 2, durable gs140) ·
b3p 재제출 `pure-stag` 준비 중(durable gs130, val 2). 발사 17.9시간, 누적 중단 17회.
gs150 3-지점 비교는 세 arm이 모두 세 번째 평가를 찍은 뒤로 미룬다.

---

## E-164 (0801 13:15 UTC) — **durable 프론티어가 학습에 34스텝까지 뒤처져 있었다**(자기 정정) · b3p는 gs150 평가 중 선점

### 1. optimizer 상태는 정상이다(두 겹으로 확인)

**저장**: 완전 체크포인트 한 벌은 `model` 4샤드 32.8 GB + `optim` 4샤드 **65.5 GB** + `extra_state` 4샤드로,
optimizer가 모델의 정확히 2배다(Adam 1·2차 모멘트 fp32). 크기가 맞으므로 optimizer가 통째로 올라간다.

**복원**: 더 중요한 증거는 재개 이후 lr이 이어진다는 것이다. b3p는 오늘 오전 gs130에서 재개했는데
코사인 감쇠가 끊기지 않는다. 콜드 스타트였다면 lr이 warmup으로 되돌아가고 grad_norm이 튄다.

| gs | b0p lr | b3p lr | b0p grad_norm | b3p grad_norm |
|---|---|---|---|---|
| 125 | 7.520e-07 | 7.520e-07 | 0.72 | 0.31 |
| 130 | 7.282e-07 | 7.282e-07 | 0.67 | 0.34 |
| 135 | 7.039e-07 | 7.039e-07 | 0.80 | 0.41 |

세 arm의 lr이 같은 gs에서 소수점까지 일치한다 → **스케줄이 matched로 유지된다**(비교 타당성 근거).

### 2. 자기 정정 — E-161/E-163의 "손실 2~3스텝"은 그 시점에만 맞았다

`files_metadata=True`로 샤드 수를 세어 보니 durable **완전** 프론티어가 학습보다 크게 뒤처져 있다:

| arm | 학습 gs | durable 완전 gs | 격차 |
|---|---|---|---|
| b0p | 162 | 125 | **37** |
| b2p | 182 | 150 | **32** |
| b3p | 149 | 130 | **19** |

원인은 대역폭이다. 체크포인트 한 벌이 **98 GB**인데 세 arm이 같은 레포에 동시에 밀어 넣는다.
`global_step_135`~`160` 자리에 `fsdp_config.json`·토크나이저 config 등 **7개 작은 파일만 남은 빈
디렉터리**가 줄줄이 있는 것이 증거다 — 푸셔가 작은 파일만 올리고 큰 샤드를 못 올린 채 다음 스텝으로
넘어갔다. 프루닝 자체는 정상이고(잔여물 총 수십 KB) 레포는 601파일 598.5 GB(체크포인트 581.7 GB).

**⚠️ LIST 방법 주의**: `list_repo_files`만 쓰면 arm당 한 스텝만 보여 "keep=1 정상"으로 오독하기 쉽다.
**반드시 `repo_info(files_metadata=True)`로 스텝별 샤드 수와 바이트를 세라.** 0801 05:07·08:40의
"릴레이 무결함" 판정은 이 오독 위에 있었다 — 무결함인 건 맞지만 **지연은 못 봤다**.

### 3. 그 위험이 30분 만에 현실화됐다

b3p는 12:58에 gs150 평가를 수행 중이었고(`validation generation end` + `global_steps: 150` 확인),
13:15에 **평가 도중 선점**됐다(18번째 중단). `retry_001` 생성 → 자동복구 경로이므로 개입하지 않는다.
durable이 gs130이므로 **19스텝(~1시간 30분)을 다시 걷는다**. gs130에서 재개하면 gs150에서 다시
평가하므로 지점이 영구히 비지는 않는다.

### 4. 방법론 교훈 — 평가 구간에서 E-160 사망 기준은 거짓 양성을 낸다

b3p는 12:41~12:58 사이 **gs 세 틱 정지 + `gpu0used` 35310MB 완전 불변 + 새 retry 없음**으로
E-160 4기준 중 3개를 만족했으나 살아서 평가 중이었다. **val 지점 직전의 정지는 먼저 평가를 의심하고
`log view`로 `validation`/`test_gen_batch` 라인을 확인하라.** 그때 개입했으면 평가를 날렸다.

**0801 13:15 상태**: b0p **gs163**(val 50·100·150) · b2p **gs182**(val 50·100·150) ·
b3p **retry_001 재부팅 중**(durable gs130, val 50·100). 발사 22.5시간, 누적 중단 18회.
3-arm gs150 비교는 b3p가 gs150을 다시 찍은 뒤로 미룬다.

---

## E-165 (0801 14:42 UTC) — 재개한 wandb run이 **replay 구간을 통째로 버린다**(b3p gs131~150)

**관측**: b3p(`pure-stag` retry_001)는 정상 재개해 **gs136까지 학습 중**이다
(`Training Progress: 45%|136/300`). 그런데 std_log에 다음이 반복된다:

```
wandb: WARNING Tried to log to step 136 that is less than the current step 150.
Steps must be monotonically increasing, so this data will be ignored.
```

**기전**: 런처는 `WANDB_RESUME=allow`로 **같은 run id에 이어 쓴다**. 직전 시도가 gs149까지 가고
gs150 평가에 진입하면서 wandb 내부 스텝 카운터가 **150**까지 올라갔다. 그 시도는 평가 도중
선점됐고(E-164), durable은 gs130이었다. 이제 재개본이 gs131부터 다시 로깅하는데 wandb는
단조증가만 허용하므로 **131~150의 모든 로그를 무시한다**.

**귀결**: b3p가 gs150에 다시 닿아 평가해도 **그 평가는 wandb에 남지 않는다**(150은 현재 150보다
크지 않다). 즉 **3-arm gs150 비교에서 b3p 칸이 wandb만으로는 영영 비어 있다.**

**복구 경로 둘**:
1. **평가는 실제로 수행되고 std_log에 찍힌다** → b3p가 gs150을 지날 때
   `$AMLT log view`에서 `val-aux/*/correctness/mean@1` 라인을 직접 긁어 표를 채운다.
2. gs151부터는 카운터를 넘어서므로 정상 기록된다 → **gs200 지점은 온전하다.**

**일반화**: durable 프론티어가 뒤처진 상태에서 선점되면(E-164) 재주행 구간이 생기고,
`WANDB_RESUME=allow`인 한 그 구간의 지표는 **조용히 사라진다**. 로그에 WARNING이 뜨긴 하지만
wandb 대시보드에는 아무 흔적이 없다. **재개 후에는 `valgs`가 기대대로 채워지는지 반드시 확인하고,
비면 잡 로그를 1차 출처로 삼아야 한다.** 두 결함이 곱해진 사례다 — 지연된 체크포인트가
재주행을 낳고, 재주행이 지표 소실을 낳았다.

**부수 관측(속도)**: 이 시각 세 arm 모두 느려져 있다 — b0p 324 s/step, b2p 405 s/step,
b3p 475 s/step(로그 표기). b3p가 가장 느리지만 같은 대역이며, 초기 ~220 s/step 대비 전반적 저하다.
b3p가 gs150에 닿기까지 14스텝 ≈ 1.8시간.

**0801 14:42 상태**: b0p **gs179**(val 50·100·150) · b2p **gs195**(val 50·100·150) ·
b3p **gs136**(val 50·100, gs131~150 로그 소실 중). 발사 23.9시간, 누적 중단 18회.

---

## E-166 (0801 15:30 UTC) — 근인 확정: **HF 공개 저장소 용량 초과로 LFS 쓰기가 403**

### 서버가 직접 말했다

2 MB 무작위 바이너리를 해당 레포에 올려 보는 프로브 한 번으로 끝났다:

```
403 Forbidden: You have exceeded your public storage space.
Cannot access content at: https://huggingface.co/iamseungpil/metacot-h200-triobj-dcpo-v3.git/info/lfs/objects/batch
```

### 관측 패턴이 정확히 이걸로 설명된다

스텝 디렉터리별 present 파일을 LFS 여부로 나누면 신호가 한 점 흐림 없이 갈린다:

| 유형 | 파일 수 | LFS | 비-LFS |
|---|---|---|---|
| 완전 스텝(b0p gs125 · b2p gs150 · b3p gs130) | 23 | 14 | 9 |
| **부분 스텝 전부**(b0p gs130~180, b2p gs160~240, b3p gs135~155) | **7** | **0** | 7 |
| 과도기 1건(b2p gs155) | 16 | 7 | 9 |

**비-LFS(작은 JSON)는 통과하고 LFS(.pt 샤드, tokenizer.json)는 하나도 못 올라간다.**
푸셔는 작은 JSON 7개를 올린 뒤 첫 `.pt`에서 403을 3회 맞고 그 스텝을 통째로 포기한다
(`_upload_step_dir` → `return False`, done 표시 안 함). 그래서 프론티어가 얼어붙었다.

### 자가 교착 구조

`push_ckpts_to_hf.py`의 `--squash_every`(기본 20)는 **성공한 업로드 수**를 셀 때만 증가한다.
업로드가 전부 실패하니 카운터가 안 올라가고 → `super_squash_history`가 영영 안 돌고 →
LFS 히스토리가 안 풀리고 → 용량이 안 나서 업로드가 계속 실패한다. **용량을 되찾을 유일한 장치가
용량 부족 때문에 작동하지 않는다.** 0724 메모("delete_folder는 tip만 지우고 LFS는 squash로만
해방된다")가 경고했던 주기 정리를 건너뛴 결과다.

### 비용

프론티어 동결 시점 이후의 모든 전진이 무보험이다. 0801 15:15 b0p이 gs179에서 선점되자
**gs125로 되돌아갔다 — 54스텝(약 5시간)**. b3p는 19스텝, b2p는 현재 노출 48스텝.
24시간 중단 19회(평균 75분마다) 기준으로 75분에 약 13스텝 전진하는데 한 번에 30~54스텝을 잃으므로
**현 상태로는 진행보다 손실이 크다 — 남은 스텝을 끝낼 수 없다.**

### 자기 정정

같은 틱에 "`save_freq` 5→25"를 권고했으나 **철회한다.** 용량이 없으면 저장 주기와 무관하게 403이다.
원인을 확인하기 전에 처방을 낸 것이 오류였다. **원인 프로브(2 MB 쓰기 한 번)가 코드 정독보다 쌌다.**

### 재고 실측(현재 tip 기준, LFS 히스토리 별도)

| 레포 | 크기 | 주요 항목 |
|---|---|---|
| `metacot-h200-triobj-dcpo-v3` | 549.3 GB | 현행 rq3v2f_b0p/b2p/b3p 303 GB + **구 계보 `rq3_b0` 98.3 · `rq3v2_b2p` 98.3 · `rq3_b2` 33.0** |
| `metacot`(dataset) | 162.0 GB | SFT 모델 9벌 × 16.4 GB 등 |
| `metacot-sft2-4g` | 49.2 GB | SFT2 체크포인트 3벌 |
| **합계** | **760.5 GB** | |

**즉시 회수 후보는 구 RL 계보 약 230 GB**(`rq3_b0`, `rq3v2_b2p`, `rq3_b2`)다. 다만 `delete_folder`는
tip만 지우므로 **삭제 후 `super_squash_history`를 돌려야 실제로 해방된다**. squash는 히스토리를
되돌릴 수 없게 만들므로 사용자 승인 후 집행한다.

**0801 15:30 상태**: b0p **retry_005로 gs125부터 재주행**(gs179에서 선점) · b2p **gs198**(durable gs150) ·
b3p **gs141**(durable gs130). 발사 24.5시간, 누적 중단 19회.

---

## E-167 (0801 16:00 UTC) — 정리 집행: 계정 **966.2 → 671.4 GB**(525 GB 회수) · 그러나 **403 지속**

사용자 승인("히스토리와 같은 체크포인트들은 정리해줘") 아래 집행. 파괴조작 3율대로
LIST → 참조검사 → 집행 순으로 진행했고, 각 삭제 전에 **대체물 존재를 확인**했다.

### 1차: 구 RL 계보 (`metacot-h200-triobj-dcpo-v3`)

| 폴더 | 크기 | 삭제 근거 |
|---|---|---|
| `checkpoints/rq3_b0` | 98.3 GB | 런처가 `archive/launchers_retired_0727/`에 은퇴 |
| `checkpoints/rq3v2_b2p` | 98.3 GB | v2f 계보로 대체됨 |
| `checkpoints/rq3_b2` | 33.0 GB | 은퇴 계보의 중간 스텝 |

**대체물 확인**: 세 계보 모두 wandb 기록이 레포에 남아 있고 `docs/reports`에 각 3건씩 결과가
문서화돼 있다 → **지워진 것은 가중치뿐, 수치와 결론은 보존**. 현행 yaml의 언급은 전부 주석·설명이며
resume 소스로 쓰이지 않음을 grep으로 확인. 결과 549.3 → **319.8 GB**.

### 2차: 레거시 SFT의 DeepSpeed optimizer 상태 (`iamseungpil/metacot` **model**)

같은 이름의 **dataset** 레포만 보다가 놓쳤던 곳이다 — 계정 전수조사로 발견했다(344 GB).
SFT 체크포인트 3벌이 있는데 벌당 **모델 가중치 16.4 GB + DeepSpeed optimizer 98.3 GB** 구조였다.
`checkpoint-{253,506,759}/global_step*/`만 삭제해 **가중치는 남기고 294.9 GB 회수**.
현행 코드·런처에서 `checkpoint-253/506/759` 참조 0건이며, 런처가 쓰는 `iamseungpil/metacot`은
`repo_type="dataset"`으로 **별개 레포**임을 확인했다. 결과 344.0 → **49.2 GB**.

### 결과와 미해결

| | 전 | 후 |
|---|---|---|
| 계정 전체(현재 tip) | 966.2 GB | **671.4 GB** |
| LFS 쓰기 프로브 | 403 | **여전히 403**(3회 재시도) |

**해석**: HF의 용량 산정은 현재 tip이 아니라 **아직 GC되지 않은 LFS 객체까지 포함**한다.
`super_squash_history`는 즉시 성공하지만(1초) 실제 해방은 서버측 GC를 기다려야 한다.
즉 **더 지운다고 지금 당장 풀리지 않는다** — 그래서 3차 정리(dataset 레포의 레거시 SFT 9벌 중
현행 미사용분 ≈115 GB)는 **보류하고 사용자 판단을 기다린다**.

**⚠️ 조사 방법 교훈**: 같은 이름이 model·dataset **두 종류로 존재**할 수 있다
(`iamseungpil/metacot`). 한쪽만 보면 344 GB를 통째로 놓친다. **용량 문제는 처음부터
`list_models`+`list_datasets`로 계정 전수조사할 것.** 세 레포만 본 초기 판단이 늦어진 원인이다.

**남은 선택지**: ①GC를 기다린다(수시간) ②HF에 공개 프로젝트 용량 상향 요청
(에러 메시지가 `website@huggingface.co` 안내) ③구독 상향(`canPay=false`라 결제수단 등록 필요).
그때까지 세 arm은 **무보험 상태로 계속 돈다** — 선점 시 동결 프론티어까지 되돌아간다.

**0801 16:00 상태**: b0p **gs125부터 재주행 중**(retry_005) · b2p **gs198** · b3p **gs141**.
발사 25시간, 누적 중단 19회.

---

## E-168 (0802 12:00 UTC) — 403의 진짜 크기: tip 671 GB vs **HF 계산 11,379 GB**. E-167 판단 정정

E-167에서 나는 "HF가 아직 GC 안 된 LFS 객체까지 센다"고 썼다. 방향은 맞았지만 **크기를 두
자릿수 틀렸다**. `expand[]=usedStorage`로 계정 전 레포를 직접 물으니:

| 레포 | HF 계산 usedStorage | tip 크기 | 히스토리 잔량 |
|---|---|---|---|
| `metacot-h200-triobj-dcpo-v3` | **9,785.8 GB** | 319.8 GB | **9,466 GB** |
| `metacot` (model) | 614.2 GB | 49.2 GB | 565 GB |
| `metacot-h200-triobj-dcpo-v2` | 524.1 GB | **0.0 GB** | 524 GB |
| `metacot-sft2-eb16` | 81.9 GB | 32.8 GB | 49 GB |
| `metacot-h200-e9-bci-rlvr` | 65.5 GB | **0.0 GB** | 65.5 GB |
| `sopbench-trackb-h200` | 44.0 GB | 6.8 GB | 37 GB |
| **계정 합계** | **11,378.8 GB** | 671.4 GB | **10,707 GB** |

계정은 `isPro: True`(PRO, `periodEnd` 2026-08 말). **E-167에서 지운 525 GB는 전체 문제의 약 5%였다.**
그래서 아무 변화가 없었다. tip만 보고 원인을 판정한 것이 오류다.

### 인과 사슬 (확정)

1. RL 체크포인트 1벌 = **98.3 GB**(가중치 16.4 + optimizer 65.5 + extra). 5스텝마다 1벌 업로드.
2. 푸셔 `--keep 1`은 **tip만** 잘라낸다(`_prune_old_verl_ckpts` → `delete_folder`). 밀려난 판본의
   LFS 객체는 **히스토리에 그대로 남고 HF는 그것을 센다**.
3. 유일한 회수 수단은 `super_squash_history`이고 `--squash_every` **기본값 20**이다.
   `_squash_history` 독스트링은 "Each checkpoint upload adds a ~16GB commit"이라고 적혀 있다 —
   **SFT 체크포인트(16 GB) 기준으로 정해진 값**이다. 실제 RL 체크포인트는 98 GB이므로
   한 squash 주기에 **20 × 98 ≈ 1.96 TB**가 쌓인다. **PRO 공개 한도(1 TB)의 두 배다.**
   ⇒ 이 런은 **설계상 도중에 403을 맞게 되어 있었다.**
4. 403이 뜨면 `uploads_since_squash += 1`이 **성공 업로드에서만** 증가하므로 squash가 영영 안 돈다(자기교착, E-166).

### 이번에 집행한 것

- **squash 3건**(한 번도 squash된 적 없던 레포): `dcpo-v2`(152→1 커밋) · `e9-bci-rlvr`(27→1) ·
  `sft2-eb16`(13→1). 세 건 모두 **squash 전후 tip 파일 목록 완전 일치**를 확인했다(4/2/33개).
- 결과: **`usedStorage`는 11,378.8 GB에서 한 바이트도 안 움직였다.** LFS 프로브도 여전히 403.

### 그래서 확정된 사실

`super_squash_history`는 1초 만에 성공하고 커밋을 1개로 접지만, **HF의 저장공간 산정은 그와 동기하지 않는다.**
어제 05:05/05:08에 squash한 `dcpo-v3`·`metacot`도 7시간 뒤인 지금까지 각각 9,785.8 GB·614.2 GB로 그대로다.
⇒ **더 지우거나 더 squash해도 지금 당장은 안 풀린다.** 남은 지렛대는 셋:
①HF 재계산/GC를 기다린다 ②`website@huggingface.co`에 요청(에러 메시지가 지정한 창구) ③레포 삭제 후 재생성
(=9.8 TB 확실 제거, 단 durable 체크포인트 3벌 295 GB를 먼저 로컬로 내린 뒤에만 — 로컬 여유 536 GB).

### 런처 수정 (사용자 승인)

세 yaml의 `pkill -f push_ckpts_to_hf`를 **PID 기반**으로 교체했다. 이 한 줄이 `bash -c` 명령줄
자신과 매칭되어 스크립트를 자살시켰고, 그 뒤의 FINAL SYNC PUSH가 한 줄도 실행되지 않아
**b2p가 300스텝을 완주하고도 gs300 가중치를 잃었다**. `PUSH_PID=$$!` 포착 → `kill` → `kill -9` →
`kill -0` 생존검증 순. yaml 파싱·홑따옴표 짝(4/4/6)·shlex·잔존 pkill 0건 확인.

**미제안 승인 대기**: `--squash_every`를 20 → 3(≈300 GB)으로. 이걸 안 고치면 재발한다.

### b3p 메타 발화 붕괴 (신규 발견)

`dcpo/meta_emit_rate`: gs5~115 **0.98~1.00** → gs135 0.744 → gs170 0.170 → (durable gs130 재개로
gs175 0.940 복원) → **gs199 0.186**. `pmishift_attempted_rate`도 0.82 → 0.098. **두 번 돌려 두 번 붕괴.**
`actor/entropy`는 0.20 → 0.42로 **상승** — 축퇴가 아니라 규약에서의 능동 이탈이다.

| 구간 | SAVE | DERAIL | 비율 | 평균 발화율 | 보너스 기댓값(β_save 1.0, β_derail 2.0) |
|---|---|---|---|---|---|
| gs1–50 | 255 | 302 | 0.84 | 0.997 | **−349** |
| gs51–100 | 241 | 290 | 0.83 | 0.999 | **−339** |
| gs101–150 | 243 | 237 | 1.03 | 0.892 | **−231** |
| gs151–199 | 173 | 128 | 1.35 | 0.626 | **−83** |

**해석**: 블록 하나를 쓸 때의 기대 보너스가 전 구간 음수다. 정책은 200스텝에 걸쳐 "안 쓰는 게 이득"을
학습했다. 비율이 0.84→1.35로 좋아지는 것은 **도움 안 되던 블록이 먼저 사라졌기** 때문이지 메타가
좋아져서가 아니다. 복제축이 +0.0246(gs50) → +0.0200(gs100) → +0.0094(gs150)로 반감하는 것이
발화율 1.00 → 1.00 → 0.89과 나란히 간다.

**사전등록 대비**: Outcome C 게이트는 gs25 기준이고 gs25는 전부 통과(emit 0.998·attempted 0.82·
n_save 8·entropy 0.20) ⇒ **무효 실행 아님**. A1의 "gs80까지 0.80 미만으로 침식되지 않을 것"도 통과
(gs80 0.998). 다만 A1의 "sustained"는 gs135부터 깨졌고 A2(attempted ≥0.30)는 현재 0.098로 깨졌다.

**결론**: gs300은 비교 지점이 될 수 없다 — 그 시점의 b3p는 메타를 거의 안 하는 모델이다.
처치가 실재하는 구간은 **gs100~150**이다. 사전등록의 "gs300에서 비교" 조항은 개정이 필요하며
이는 사용자 판단 사항이다.

**반증 가능한 예측**: 지금 도는 gs200 평가에서 b3p 9벤치 평균이 b2p의 gs200(0.3755) 이하로 나오고
`b3p − b2p`가 gs150의 +0.0046보다 0에 가까워질 것. 틀리면 위 인과 설명을 폐기한다.

**0802 12:00 상태**: b0p **사망**(retry 7소진, 마지막은 sft2_init 다운로드 IncompleteRead) ·
b2p **300/300 완주**(가중치 소실, 지표 6지점 확보) · b3p **gs199**(gs200 평가 중).

---

## E-169 (0803 04:10 UTC) — HF 이전 완료: **10,130 GB 즉시 회수**. 시드 가드가 b0p를 지켜냄

### 1. 원인 검증 (추측 → 증명)

`dcpo-v3`의 **현재 히스토리에 남은 90개 커밋의 트리를 전부 순회**해 도달 가능한 LFS 객체의 합집합을
계산했다. 결과: **고유 83개 / 319.7 GB**. HF 청구액은 9,785.8 GB. 즉 **9,466 GB(96.7%)가 어떤 커밋에서도
도달 불가능한 고아**였다. 계정 전체로는 청구 11,377.9 GB 중 **10,669.9 GB(93.8%)가 고아**,
도달 가능분은 708 GB뿐이었다.

### 2. 두 가지 기전 확정

| 조작 | 반영 속도 |
|---|---|
| `super_squash_history` | **즉시 성공하지만 usedStorage는 안 움직임**(3개 레포 squash 직후 11,378.8 GB 그대로, dcpo-v3는 7시간 뒤에도 그대로) |
| `delete_repo` | **즉시 반영**(dcpo-v2+e9-bci-rlvr 삭제 → 11,378.8 → 10,789.2 GB, 정확히 −589.6 GB) |

그리고 그 589.6 GB만으로 **403이 풀렸다** ⇒ **한도는 1 TB가 아니라 10.8~11.4 TB 사이다.**
E-168에서 "PRO 공개 한도 1 TB"라고 쓴 것은 틀렸다. 정정한다.

### 3. 집행 (레포 이전)

로컬 여유 536 GB를 확인하고 순서대로 진행했다.
1. 보존 대상 다운로드 → **원격과 파일별 바이트 대조**(누락 0). 도중 403이 풀려 푸셔가 되살아나면서
   b3p 프론티어가 gs130 → **gs205**로 올라가 대상이 바뀌었고, gs205를 추가로 받았다.
2. 삭제 직전 3벌 모두 23파일 model/extra/optim **4/4/4** 재확인.
3. `delete_repo` → `create_repo`(동명) → **재개 가드 시드**: arm별 `extra_state_world_size_4_rank_0.pt`
   1개씩(15 KB)만 먼저 올려 `RGS_ANY=1`·`RGS_STEP=0` 상태를 만듦.
4. `upload_large_folder`로 3벌 + eval/models/wandb/reports 복원.

**결과: 계정 11,133.1 → 1,003.4 GB (−10,129.7 GB, 즉시).** 복원 완료 후 푸셔 정상화,
b3p 프론티어는 현재 gs295. 계정 2,985.4 GB로 여유 충분.

### 4. 시드 가드가 실제로 일했다

복원 업로드(295 GB 해싱+전송) 도중 b0p(`loving-osprey`)가 재개를 시도했고 로그가 이렇게 남았다:

```
[resume] ignoring PARTIAL steps on HF: [125]
[YAML] RGS(HF)=0 ANY=1 LOCAL_GS(pulled)=NONE
[YAML] ABORT: ... refusing gs0 cold-start. Next window retries.
```

**설계대로 gs0 콜드스타트를 거부했다.** 시드가 없었다면 `ANY=0`이 되어 b0p가 gs0부터 새 계보를
만들고 gs125를 덮어썼을 것이다. 대가는 재시도 소진 후 **약 14시간 유휴**였다 — 데이터 대신 시간을 냈다.
★교훈: **레포를 비우는 조작 전에는 재개 가드가 참이 되는 최소 파일을 먼저 심어라.**
`delete → create → seed(수초) → bulk`. 시드는 가드가 세는 패턴과 정확히 일치해야 한다.

### 5. 현 상태와 지표

| arm | 잡 | gs | durable |
|---|---|---|---|
| b0p | `solid-gibbon` 재제출 | gs125부터 | gs125 |
| b2p | 완주 | 300/300 | gs150 |
| b3p | `pure-stag` retry_005 부팅 | gs298→gs295 재개 | gs295 |

9벤치 평균 (보상 스케일 −1~+1):

| gs | b0p | b2p | b3p | b3p−b0p | b3p−b2p |
|---|---|---|---|---|---|
| 50 | 0.2469 | 0.2673 | 0.2715 | +0.0246 | +0.0042 |
| 100 | 0.2905 | 0.2917 | 0.3105 | +0.0200 | +0.0188 |
| 150 | 0.3357 | 0.3405 | 0.3451 | +0.0094 | +0.0046 |
| 200 | — | 0.3755 | 0.2839 | — | **−0.0915** |
| 250 | — | 0.3749 | 0.3567 | — | **−0.0182** |
| 300 | — | 0.3820 | (대기) | — | — |

**RQ2 축의 부호가 후반부에 뒤집혔다.** 같은 구간에서 `dcpo/meta_emit_rate`는 1.00 → **0.0137**(gs298)로
사실상 소멸했다. b3p의 응답 길이는 779.3으로 b2p(732.2)보다 길고 `clip_ratio`도 0.0645 vs 0.0371로 높다
— **메타 블록이 없는데도 더 길고 더 자주 잘린다** ⇒ 발화 소멸만이 아니라 퇴화가 동반될 가능성.

### 6. E-168 해석의 정정

E-168에서 SAVE/DERAIL 비대칭(β_derail 2.0)을 발화 붕괴의 원인으로 제시했으나, 배치 구성을 확인하니
**스텝당 롤아웃 512개**(`train_batch_size 64 × rollout.n 8`, b2p 실측 `(732.19+70.06)×512=410,752`로 확증)
대비 SAVE/DERAIL은 스텝당 5~6건, 즉 **시도 블록의 약 3%**에만 보너스가 붙는다.
따라서 비대칭 보너스는 유력한 후보이지 확정 원인이 아니다. 확정하려면 연속 SHIFT 항의 평균 부호와
`len_cost`(0.08) 기여를 따로 재야 한다. **미측정.**

---

## E-170 (0803 05:30 UTC) — 발화 붕괴의 근인 확정: **anti-collapse floor가 꺼져 있다**. 설정 파일이 이 붕괴를 예언해 놓았다

사용자 질문("instruct에서는 발화율이 안 무너졌잖아")이 정확한 지점을 찍었다. 같은 보상·같은 패키지인데
기질에 따라 결과가 갈린다면 원인은 보상이 아니라 **보상이 놓인 조건**이다. 찾았다.

### 1. 발화를 지키는 힘이 하나도 서 있지 않다

`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml`의 헤드 구성을 힘의 방향으로 다시 읽으면:

| 헤드 | 가중치 | 발화에 대한 힘 | 왜 |
|---|---|---|---|
| `pmi_shift` | 0.8 | **없음** | PMI-scored 행에서만 중심화 ⇒ 평균 0. **어느 블록이 나은지**만 채점하고 **쓸지 말지**는 안 건드림 |
| `w_format` | 0.35 | **조건부** | 발화해야 받는 보상. 침묵은 0점이지 감점이 아님 ⇒ **침묵이 안전한 탈출구** |
| `w_emit` | 0.1 | **불활성** | 그룹 중심화. 8개 롤아웃이 전부 발화하면 그룹 내 상수 ⇒ advantage 0 |
| **`dcpo_meta_floor`** | **0.0** | **꺼짐** | 유일한 무조건 상시 견인력 |
| `len_cost` | 0.08 | **반대(무조건)** | 모든 토큰에 부과 |
| 절단 페널티 | 0.3 | **반대** | 캡에 닿으면 |
| `R_corr` | **1.0** | **반대** | 절단 응답은 −1 |

⇒ **발화를 지키는 것은 전부 조건부이거나 중심화로 불활성이고, 공격하는 것은 무조건이다.**

### 2. 설정 파일이 이 결과를 미리 적어놨다

같은 파일 30~38행:
> `FLOOR STAYS at a reduced 0.05 (review I1: centered delta is mean-zero => no standing emission pull;`
> **`removing the floor reopens the v3l collapse).`** `Floor removal only as a LATE stage-2 action gated on measured emission stability.`

그리고 151~157행:
> `# FLOOR STAYS (review I1), reduced 0.1 -> 0.05: ... hold the channel open under the length/format pressure`
> `dcpo_meta_floor: 0.0   # cf_group (2026-06-21) 0.05 -> 0.0: the counterfactual answer-delta is the emit-when-useful gradient`

**주석은 "FLOOR STAYS ... 0.05"라고 쓰여 있고 바로 다음 줄의 값은 0.0이다.** 0.0의 근거는
*"counterfactual answer-delta가 emit-when-useful gradient를 준다"*인데, **이 런은 cf_group이 아니라
pmi_shift를 쓴다.** 즉 floor를 뺀 근거가 이 구성에는 성립하지 않는다. 그리고 review I1이
그 결과(`v3l collapse` 재발)를 이름까지 붙여 예측해 놨다.

### 3. instruct는 왜 안 무너졌나 — 설계가 아니라 우연이 지켜줬다

논문 표 3(T1, instruct): **메타 arm의 절단률이 base arm보다 낮다** — MATH500 ≈6% vs 19%,
AIME ≈50% vs 73%. instruct의 고장 양상은 **비종결(degeneration)**이고, 메타 블록이 커밋하고
끝낼 구실을 줬다 ⇒ **발화가 정확도를 올린다 ⇒ 가중치 1.0짜리 correctness 헤드가 floor 역할을 대신했다.**

base는 부호가 뒤집힌다. 같은 스텝에서 b3p의 `clip_ratio`가 쌍둥이 b2p의 **2~3배**다:

| gs | b3p clip | b2p clip | 배율 | b3p emit |
|---|---|---|---|---|
| 100 | 0.0059 | 0.0020 | 3.0× | 0.998 |
| 150 | 0.0117 | 0.0039 | 3.0× | 0.897 |
| 200 | 0.0469 | 0.0273 | 1.7× | **0.197** |
| 290 | 0.1055 | 0.0371 | 2.8× | 0.041 |

base는 이미 잘 종결한다. 블록은 토큰만 더 쓰고 8192 캡 쪽으로 민다 ⇒ 절단 ⇒ `R_corr=−1`
⇒ **correctness 헤드가 발화를 처벌한다.** floor가 꺼져 있으니 맞설 것이 없다.

**시점도 맞는다**: 발화는 `clip_ratio≈0`인 gs≤100까지 1.00을 유지하다가, clip이 1%를 넘고
`resp_len`이 462→746으로 뛰는 **gs110~120에 침식이 시작**되고, clip이 4%를 넘는 gs200에 붕괴한다.

⚠**단서**: gs290에서 발화가 0.04인데 clip은 10.6%로 최고다. 즉 블록이 절단의 **유일한** 원인은
아니다. 두 arm 모두 길이가 자란다(b2p 333→944). 블록은 공통 드리프트를 **2~3배 증폭**해
base에서 임계를 먼저 넘게 만든 **증폭기**이지 단독 원인이 아니다.

### 4. 판정에 미치는 영향 — 내 앞 판단을 정정한다

E-169와 직전 보고에서 나는 이 런이 **Outcome B(기질 특이적)**로 가고 있다고 썼다. **정정한다.**

- **gs50~150**: 발화율 ≥0.89로 처치가 실재 ⇒ **유효한 검정**. 복제축 +0.0246 → +0.0200 → +0.0094
  (≈ +1.2%p → +0.5%p). 효과는 작지만 실재하고 감소 중.
- **gs150~300**: 처치가 사라짐(발화 0.0137) ⇒ **이 구간은 메커니즘의 검정이 아니다.**
  사전등록 A1("≥0.80 유지")이 gs135에, A2("attempted ≥0.30")가 gs190에 깨졌다.

따라서 후반부는 **B가 아니라 C(무효 실행)에 가깝다** — 사전등록의 C 처방은
*"Abort, fix upstream, relaunch"*다. **"base 기질에서는 효과가 없다"고 결론내면 틀린다.
가드가 꺼진 채로 돌린 런에서 처치가 자멸한 것을 기질 탓으로 돌리는 것이기 때문이다.**

### 5. 남는 질문 (미측정)

`dcpo_meta_floor: 0.05`로 켜면 발화가 유지되는가, 아니면 지연될 뿐인가. floor는 무조건 견인력이라
**발화를 붙잡되 쓸모없는 발화까지 붙잡는다** — 그게 0.0으로 내린 원래 이유(boilerplate 억제)다.
즉 floor 복원은 붕괴를 고치면서 **"always-on 메타"라는 반대편 실패**를 부를 수 있다.
이 트레이드오프는 이 프로젝트가 이미 여러 번 만난 자리다(사이트 메커니즘 1번 selectivity vs 2번 always-on).

---

## E-171 (0803 07:00 UTC) — 가드는 **양쪽 다** 꺼져 있었다. git 이력으로 확정 + 재현성 점검

E-170을 검증하려고 "T1도 floor가 꺼져 있었나"를 물었고, 답이 나왔다.

### 1. 보상 설정 이력 (`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml`)

| 커밋 | 날짜 | `w_emit` | `meta_floor` |
|---|---|---|---|
| `1eae74e` | 2026-06-15 | 0.15 | **0.05** ← 가드 ON |
| `f1f6cec`/`6537775` | **2026-06-22** | 0.1 | **0.0** ← **가드 OFF**(cf_group 작업의 부수효과) |
| `6a12ae4` | 2026-07-12 | 0.1 | 0.0 |

T1의 런처 `archive/launchers_pre_rq3/h100std_pmishift.yaml`은 **같은 config**를 쓰고
(`--config-name=triobj_dcpo_v4_stage3b_h100_4x4k`) override는 `rmeta_source=pmi_shift`,
`w_over=0.0` **두 개뿐**이다. 런처 이력상 T1 RL은 06-23 이후 ~07-08에 돌았다
(`docs/reports/2026-07-08-RQ2-...md`).

⇒ **T1도 `meta_floor=0.0`·`w_emit=0.1`로 돌았다. 가드는 양쪽 다 꺼져 있었다.**

### 2. 그래서 E-170의 결론은 유지되고, 더 날카로워진다

설정 차이가 아니다. **처치의 생존이 통제되지 않은 결합(블록↔절단)의 부호에 의존했고,
그 결합을 막아줄 가드는 두 런 모두에서 꺼져 있었다.** instruct는 부호가 유리해서 살아남았고
(블록이 절단을 줄임: 6% vs 19%), base는 부호가 불리해서 죽었다(b3p 절단이 쌍둥이의 2~3배).

**즉 T1의 성공은 재현 가능한 설계가 아니라 기질에 우연히 유리했던 조건 위에 서 있었다.**
그리고 우리는 **그 조건이 하중을 받고 있다는 사실 자체를 몰랐다** — 매치드 감사는 런처를
바이트 단위로 비교했지만, `meta_floor`는 상속된 config에 있어 비교 대상이 아니었다.

### 3. ⚠️추가 발견 — instruct의 "가장 깨끗한 증거"도 다른 보상이었다

`docs/reports/2026-07-08-RQ2-isolated-pmishift-net-shiftonly-vs-gandhi.md`의 shiftonly arm은
*"all other heads zeroed: cal/format/emit/len_cost/over = 0"*이다. 즉 MATH500 **+5.6~+5.9pp**
(p<.001)를 낸 구성은 **len_cost도 emit 헤드도 없는** 보상이다. 지금 b3p가 쓰는 7-헤드 패키지와
다르다. 이 수치를 "패키지 안에서의 PMI-shift 효과"로 인용하면 안 된다.

### 4. 재현성 점검 결과

| | base 사다리(rq3v2f) | instruct(T1) |
|---|---|---|
| 코드 스냅샷 | asset `490407111` ✅ | **어느 asset인지 기록 없음** ⚠ (릴리스 72개) |
| RL config | `configs/` ✅ | `configs/triobj_dcpo_v4_stage3b` ✅ (동일 파일) |
| 런처 | 루트 3개 ✅ | `archive/launchers_pre_rq3/` ✅ |
| SFT2 코퍼스 | `rv_redirect_verify_functional`·`v8_base_rv_sft` ✅ | `rv_redirect_verify_functional` ✅ |
| SFT init | HF `metacot`(dataset) `models/b0p2_rvfull_sft`·`b2p2_rvfull_sft` ✅ | HF `metacot-rv` `v8_rv_functional_sft` ✅ |
| RL 체크포인트 | b0p gs125·b2p gs150·b3p gs295 ✅ | **전삭제** ❌ |
| 평가 산출물 | 없음(아직) | `eval/{base_matched,pmishift,shiftonly,gandhi}_1030_v2` ✅ |

⇒ **base는 완전 재현 가능. instruct는 "가중치만" 없다** — SFT init·코퍼스·config가 남아 있어
RL 재주행(≈30h/arm)으로 복원 가능하고, 기존 수치는 저장된 생성물로 재채점 가능하다.
**단 코드 asset 매핑이 산문에만 있다**(런→asset 표가 없음). 이게 가장 큰 재현성 구멍이다.

### 5. 구조적 교훈

이 사고의 형태는 **"06-22에 다른 실험(cf_group) 때문에 끈 스위치가, 6주 뒤 다른 실험(pmi_shift)의
후반부를 무효화했다"**이다. 외부 연구가 이 실패를 이렇게 부른다:
*"a bad assumption at step 3 can quietly contaminate step 50"*(agentic AI failure modes, ICML 2026 워크숍).
우리 사례에서 step 3과 step 50 사이는 **6주**였다.

**처방은 floor를 켜는 게 아니라(그건 always-on 메타라는 반대편 실패를 부른다),
런마다 `resolved config`를 동결·기록하고 매치드 쌍은 resolved 층에서 diff하는 것이다.**
런처만 비교하면 상속된 노브는 영원히 안 보인다.

---

## E-172 (0803 08:00 UTC) — G2 발화검사 통과, **그러나 헌법의 대시보드는 이 런을 "broken"으로 읽고 있었다**

stacked-research 스킬의 G1~G4를 실제로 돌렸다.

### G2 발화검사 — PMI-shift는 **배선돼 있고 발화한다**

| 지표 | warmup(≤80) | mid(81–150) | late(>150) |
|---|---|---|---|
| `gdpo/meta_region_utility/std` | 1.038 | 1.042 | **0.441** |
| `dcpo/pmishift_attempted_rate` | 0.711 | 0.634 | **0.151** |
| `dcpo/pmishift_rmeta_mean_scored` | **−0.141** | +0.136 | −0.022 |
| `dcpo/rmeta_pos_rate` / `neg_rate` | 0.207 / **0.261** | 0.248 / 0.196 | 0.074 / 0.038 |
| `dcpo/eff_scale_meta` | 0.336 | 0.461 | 0.589 |

meta 헤드의 advantage 표준편차가 1.04로 실재한다 ⇒ **무효 레버 아님. G2 PASS.**
다만 `std`가 후반 0.44로 반토막 나고 `attempted`가 0.15로 떨어진다 — **발화가 죽으면 채점할 행이
줄어 헤드가 스스로 흐려지는 양의 되먹임**이 돈다.

★그리고 지난 틱에 "미측정"이라 했던 **연속 SHIFT 항의 평균 부호**가 여기 있다:
warmup에서 `rmeta_mean_scored = −0.141`, `neg_rate(0.261) > pos_rate(0.207)`.
**규약이 굳어야 할 첫 80스텝 동안 채점된 메타 보상의 평균이 음수였다.**

### ★★ 헌법 Part IV가 이미 이 수치의 정상 범위를 적어놨다

`docs/CONSTITUTION.md:128`:
> `dcpo/pmishift_rmeta_mean_scored` | Healthy **+1.0–+1.2** | Broken **~−0.2** | *Near-0 ⇒ no signal.*

**우리 값 −0.141은 Broken 열이다.** 그런데 Part VI의 발사 게이트 목록에는 이 지표가 없다:
> emit ≥0.8 ✅ · attempted ≥0.3 ✅ · n_save>0 ✅ · entropy>0.1 ✅ · clip<0.2 ✅ — **전부 통과**

⇒ **대시보드는 숫자를 알고 있었고, 게이트는 그 숫자를 안 봤다.**
stacked-research §7: *"게이트를 통과했는데 실패가 났으면 게이트를 추가한다."*

또한 Part V 실패모드 1번은 처방을 이렇게 적어놨다:
> *"an RL-side meta-emission **floor** to resist erosion"*
**헌법이 처방한 floor가 설정에서 0.0이다**(E-170/171).

### G3 해상도 — **FAIL: 기준선 행이 없다**

세 런처가 **전부** `++trainer.val_before_train=False`로 gs0 평가를 껐다
(`b0p:289`, `b2p:283`, `b3p:269`). 두 config는 `val_before_train: true`인데 런처가 덮었다.
⇒ **각 arm의 출발점이 val 세트에서 미측정**이고, arm별 RL 기여를 계산할 수 없다.
SFT 단계 실측으로 b2p 58.0% vs b0p 55.5%(MATH500-100)의 **초기 격차가 존재**하는데,
그 격차가 gs50 이후 모든 비교에 섞여 있고 빼낼 방법이 없다.
[[number-needs-its-baseline-beside-it-0731]] 그대로다 — **네 번째 칸이 비었다.**

### G4 통제군 — 부분 통과

b0p(메타 제거 쌍둥이+vanilla)·b2p(메타 SFT2+vanilla)로 **"메타 있으나 보상 안 함"** 통제군은 있다.
없는 것은 **학습 0 팔**(SFT2 init 그대로의 val 성능) — 이는 G3의 gs0 결손과 같은 구멍이다.

### G1 CLAIMS — 대장이 없다

`docs/CONSTITUTION.md`(224행)는 훌륭한 진단 문서지만 **주장 대장이 아니다**.
"닫는 것 / 여는 것 / 재확인 계수기" 필드가 0건. 그래서 판정이 로그 2,300행 뒤에 묻히고,
같은 결론을 다시 사게 된다(이번 세션만 floor 관련 재확인 2회).

### 헌법 위반 1건 추가 발견

Part VII: *"Matched-arm isolation: ... **Same config path**, same len_cost, same recipe."*
현행은 b0p/b2p = `base_matched_grpo_h100_4x4k`, b3p = `triobj_dcpo_v4_stage3b_h100_4x4k`로
**config path가 다르다.** 상속 노브(`meta_floor`·`len_cost`)가 비교 대상 밖에 놓인 구조적 이유다.

### E-1xx 2026-08-03 — b2p 재개가 1시간째 큐 대기 · b0p 5.7시간 정지 구간

- **b2p `musical-ant` = `queued`**(제출 후 1h). 노드를 아직 못 잡았다. wandb `rq3v2f_b2p` 는
  이전 런(`hip-hound`)이 `crashed` @ gs178 로 남아 있고 **gs176+ 행이 없다** — 재개가
  아직 시작조차 안 됐다는 뜻. Standard 티어 큐 대기이므로 개입하지 않는다(중단 4양상 ①).
- **b0p `solid-gibbon` gs195, running.** 스텝 간격에 **20,509s(5.7h) 한 칸**이 있다
  (≈gs180 부근). 이 한 칸 때문에 끝점 기반 s/step 이 1,226s 로 부풀었다 —
  **중앙값은 최근 30스텝 365s / 최근 100스텝 298s**. 실제 ETA ≈ **10.6h**.
  단 298→365 로 **완만히 느려지는 중**.
- **b3s `musical-wombat` gs22, 290 s/step, ETA 22.4h.** gs25 게이트 4종 사전 확인:
  `meta_emit_rate 0.9961` ✅ · `pmishift_attempted_rate 0.7832` ✅ · `actor/entropy 0.2286` ✅ ·
  `rmeta_mean_scored` **−0.3599(gs20) → +0.0189(gs22)** 로 부호 복귀
  (`pos_rate 0.2441` vs `neg_rate 0.2383` 로 균형). 헌법 건전 밴드 +1.0~+1.2 에는 아직 멀지만
  워밍업 구간이라 절대값은 판정하지 않는다. **판정 지점은 gs50 kill 게이트.**
- 조치 없음. 다음 틱에서 (1) b2p 가 `running` 으로 바뀌었는지 (2) b3s gs25 게이트 확정.

### E-1xx 2026-08-03 — b3s gs25 게이트 통과·발화는 살았으나 **rmeta 는 음수** · b2p 는 98GB 재개 다운로드 중

- **b2p `musical-ant` = running(3h), 정지 아님.** 로그가 `pull_resume_ckpt.py ... | tail -5` 에서
  서 있는데, 출력이 파이프에 물려 **완료 전엔 한 줄도 안 찍힌다.** 재개 지점은 정상 인식
  (`existing GRPO resume gs = 175`). HF 실측 `rq3v2f_b2p/global_step_175` = **23파일 98.3GB** 이므로
  받는 양이 그만큼이다. 학습이 한 스텝도 안 돌았으니 wandb 에 gs176+ 가 없는 게 맞다. 개입 없음.
- **b3s gs44 — gs25 게이트 4종 통과.** `meta_emit_rate 0.9980`(최근 6점 0.996~0.998로 평평) ·
  `attempted 0.8008`(0.633→0.801 상승) · `entropy 0.1805`(>0.1, 단 0.24→0.18 하락 추세).
  **`meta_floor=0.05` 가 발화를 잡고 있다** — b3p 가 같은 구간에서 무너진 것과 대비된다.
- ⚠**단, 메타 보상은 음수다.** `rmeta_mean_scored −0.2515`, `neg_rate 0.3047 > pos_rate 0.2051`.
  헌법 건전 밴드는 +1.0~+1.2 이고 −0.2 는 "깨짐" 자리다. 즉 **발화는 강제로 살렸지만, 나오는 메타
  블록이 gold-vs-decoy 마진을 평균적으로 깎고 있다.** gs50 kill 게이트는 발화율만 보므로 통과하겠지만
  **통과가 곧 처치 건전을 뜻하지 않는다.** `critic/score/mean` 은 0.19→0.69 로 오르는 중이라
  정답성 자체는 개선 중 — 다음 판정 지점에서 두 수를 나란히 볼 것.
- **b0p gs206**, 421 s/step(365→421 로 계속 느려짐), ETA 11.0h. **b3s** 295 s/step, ETA 21.0h.
- HF 체크포인트 실측: b0p gs205 · b2p gs175 · **b3p gs300** · b3s gs30/35/40(+45 업로드 중),
  각 98.3GB. `--keep 3` 이 b3s 판정 지점을 보존하고 있다. **b3p gs300 이 살아 있으므로 eval 가능.**

### E-1xx 2026-08-03 14:36 UTC — b3s **gs50 kill 게이트 통과** · b2p 재개 성공(단 wandb 가 gs176~179 폐기)

**b3s (`musical-wombat`) gs49 — 게이트 4종 전부 통과. 계속 진행.**

| 지표 | gs49 | 최근 6점 | 게이트 |
|---|---|---|---|
| `dcpo/meta_emit_rate` | **1.0000** | 0.998 1.0 0.996 0.994 0.996 1.0 | ≥0.80 ✅ **침식 0** |
| `dcpo/pmishift_attempted_rate` | 0.7480 | 0.80 0.74 0.68 0.63 0.72 0.75 | ≥0.30 ✅ |
| `actor/entropy` | 0.2363 | 0.18 0.26 0.22 0.23 0.21 0.24 | >0.1 ✅ |
| `dcpo/pmishift_rmeta_mean_scored` | **+0.0039** | −0.25 −0.28 −0.13 **+0.18** −0.13 **+0.00** | (게이트 아님) |
| `rmeta_pos_rate` / `neg_rate` | 0.2617 / 0.2441 | — | 양수가 앞섬 |
| `critic/score/mean` | 0.5000 | 0.70 0.46 0.41 0.40 0.59 0.50 | — |

⚠**정정**: 직전 틱에 `rmeta_mean_scored` 를 "−0.2515 → −0.2819 로 악화 중"이라고 보고했는데,
**틀렸다.** 네 스텝 뒤 +0.0039 로 돌아왔고 계열이 스텝마다 ±0.28 로 진동한다
(−0.25 → −0.28 → −0.13 → **+0.18** → −0.13 → +0.00). **스텝 단위 rmeta 는 5~6점으로 추세를 읽을
해상도가 없다.** 판정에 쓰려면 창 평균이 필요하다. 음수 구간이 있었다는 사실만 남기고
"악화 중"은 철회한다.

★**핵심**: `meta_floor=0.05` 가 발화를 완전히 잡고 있다(gs49 까지 침식 0). b3p 는 같은 노브가
0.0 인 채로 gs150 이후 1.00→0.018 로 무너졌다. **단 붕괴는 gs150 부터였으므로 진짜 시험은 아직
앞에 있다** — gs150 게이트까지 통과해야 이 노브가 원인이라고 말할 수 있다.

**b2p (`musical-ant`) 재개 성공.** 98.3GB 다운로드 완료 후 gs175 에서 재개해 14:28 UTC 에 **gs177**
(374 s/it, ETA 12.8h). 앞서 "3시간째 정지"로 보인 것은 `pull_resume_ckpt.py | tail -5` 의 출력
버퍼링이었고, 실제로는 받고 있었다. b0p 의 같은 재개가 ~1h 였던 데 비해 ~2h 걸렸다.

⚠**단 wandb 가 gs176·177 을 버렸다** — `Tried to log to step 177 that is less than the current
step 179`. 이전 크래시 런이 wandb 를 gs179 까지 밀어놨고 `WANDB_RESUME=allow` 가 되감기를 거부한다.
**E-165 와 같은 실패**(그때는 b3p). 경고 2건 = gs176·177. gs180 부터는 정상 기록될 것이다.
⇒ **gs180 전까지 b2p 는 wandb 가 아니라 로그의 `Training Progress` 로 읽는다.**

⚠`amlt status` 가 같은 시각에 `queued` 로 표시됐다 — 로그가 8분 전 학습 스텝을 찍고 있으므로
**status 표시가 늦거나 튀는 것**이다. 진행 판단은 로그를 정본으로 한다.

**b0p (`solid-gibbon`) gs210**, 436 s/step(421→436 로 계속 느려짐), ETA 10.9h.

### E-1xx 2026-08-03 15:10 UTC — b2p **세 번째 재시작**(패턴: 다운로드 2~3h → 학습 13분 → 죽음) · b3s 발화 침식 0

**b2p (`musical-ant`) 가 또 재시작했다.** 14:28 에 gs177 을 찍었는데 15:10 현재 로그가 다시
`pull_resume_ckpt.py` 로 되감겼다(새 pod, 322줄). 즉 **98.3GB 를 또 받고 있다.**
관측된 주기: **다운로드 2~3h → 학습 ~13분 → 죽음.** 이 주기가 반복되면 b2p 는 전진하지 못한다.
직전 pod 의 실패 사유는 새 pod 로그에 남지 않아 확인 불가(이전 세대 `hip-hound` 는 host RAM OOM,
`RewardLoopWorker.compute_score` 527GB).

★**중요 — b2p 는 이미 gs300 을 완주한 적이 있다.** wandb 에 같은 이름의 런이 둘이다:

| run id | 상태 | gs | 생성 |
|---|---|---|---|
| `rq3v2f-b2p-1` | **finished** | **300** | 2026-07-31 15:29 |
| `rq3v2f-b2p-2` | crashed | 178 | 2026-08-03 04:45 |

즉 **학습 곡선과 in-training val 은 gs300 까지 이미 있다.** 잃은 것은 gs300 **가중치**뿐이다
(pusher self-kill, `pkill -f push_ckpts_to_hf` 가 자기 `bash -c` 명령줄에 매칭). 지금 12시간 넘게
재주행하는 유일한 목적은 **held-out eval 용 가중치 재생성**이다. 이 비용이 값한지는 판단이 필요하다.
⇒ 대안이 없다는 점은 분명하다 — 평가하려면 가중치가 있어야 한다.

⚠ 현 `musical-ant` 는 `WANDB_RESUME=allow` 로 **`rq3v2f-b2p-2`(gs178)** 에 붙는다. 그래서
gs176·177 이 폐기됐다. gs180 을 넘겨야 기록이 재개된다 — **이번 시도도 거기까지 못 갔다.**

**b3s (`musical-wombat`) gs56 — 발화 침식 0.** 창 평균으로 다시 읽었다:

| 지표 | last | 최근10 평균 | 최근20 평균 |
|---|---|---|---|
| `meta_emit_rate` | 0.9980 | **0.9980** | **0.9979** |
| `pmishift_rmeta_mean_scored` | +0.0929 | −0.0172 | −0.0712 |
| `rmeta_pos_rate` / `neg_rate` | 0.2676 / 0.2461 | 0.2232 / 0.2334 | 0.2107 / 0.2423 |
| `actor/entropy` | 0.2107 | 0.2315 | 0.2269 |
| `gdpo/correctness/mean` | 0.5996 | 0.4842 | 0.4848 |

rmeta 창 평균은 **0 바로 아래에서 0 쪽으로 올라오는 중**(−0.0712 → −0.0172). 건전 밴드 +1.0~+1.2
에는 여전히 멀다. 발화는 `meta_floor` 가 완전히 잡고 있으나 **블록의 유용성은 아직 0 근처**다.

HF 실측: b0p gs215 · b2p gs175(정체) · b3p gs300 · b3s gs40/45/50/55 — `--keep 3` 정상 작동.
**b0p** gs215, 443 s/step, ETA 10.5h.

### E-1xx 2026-08-03 15:52 UTC — **b0p 재시작(콜드)** · b2p gs180 도달·wandb 회복 · b3s gs62

- ⚠**b0p (`solid-gibbon`) 가 gs219 에서 죽고 콜드 재시작 중.** 로그가 190줄, 단계는
  `[bootstrap] fast-path: pulling conda-pack env from HF` — conda env·코드·init 모델·98GB 체크포인트를
  전부 새로 받아야 한다. wandb `rq3v2f-b0p-1` = crashed @ gs219 (b0p 런은 이것 하나뿐이라
  재개 시 같은 런에 붙고, gs220 전까지 몇 스텝이 폐기될 것이다).
  ★**HF 의 gs220 은 18/23 파일 24.6GB 로 부분 업로드다.** `pull_resume_ckpt.py` 가 model·optim·extra
  각 4/4 를 요구하므로 **gs215 로 재개된다** — 5스텝 손실. 부분 체크포인트 가드가 의도대로 작동한 사례다.
- ✅**b2p (`musical-ant`) gs180 도달.** 로그에
  `local_global_step_folder: /scratch/checkpoints/rq3v2f_b2p/global_step_180` +
  `Saved model to .../global_step_180/actor/model_world_size_4_rank_1.pt`. wandb 도 gs179 까지 올라와
  **폐기 구간을 벗어났다**(gs176~179 는 영구 결측). 370 s/step, ETA 12.4h.
- **b3s (`musical-wombat`) gs62**, 321 s/step, ETA 21.2h. 창 평균:

| 지표 | 최근10 | 최근20 |
|---|---|---|
| `meta_emit_rate` | **0.9990** | **0.9982** |
| `rmeta_mean_scored` | −0.1183 | −0.0779 |
| `pos_rate` / `neg_rate` | 0.2031 / 0.2529 | 0.2114 / 0.2463 |
| `entropy` | 0.2578 | 0.2416 |
| `correctness` | 0.4390 | 0.4772 |

발화는 여전히 침식 0. rmeta 는 **0 아래에서 진동하며 음수율이 양수율보다 꾸준히 높다**
(10/20 창 모두). 건전 밴드 +1.0~+1.2 와는 자릿수가 다르다.

- **HF 푸시 상태**: 마지막 체크포인트 커밋 15:41(b0p gs220 부분) · `prune global_step_45 (keep latest 3)`
  15:39 정상. b3s gs50·55·60 완결 23/23 98.3GB. b2p gs175 완결. b3p gs300 보존.

### E-1xx 2026-08-03 16:20 UTC — **b0p 선점 루프**(부트스트랩 완료 → 다운로드 시작 → SIGTERM) · b2p·b3s 순항

⚠**b0p (`solid-gibbon`) 가 두 번째로 선점됐다. 이번엔 다운로드 시작 직후다.**

```
[bootstrap] env install complete → /scratch/simplerl_v4.done      ← 환경 재구축 성공
[bootstrap] code at pinned revision 490407111
[YAML] existing GRPO resume gs (model+extra+optim>=4) = 215 1      ← 부분 gs220 을 정확히 배제 ✅
+ python .../pull_resume_ckpt.py --config_name rq3v2f_b0p          ← 98GB 받기 시작
amlt-code-runner - WARNING - Caught signal 15                      ← SIGTERM (선점)
```

**주기가 안 맞는다**: 재시작 ~15:52 → 부트스트랩 완료 ~16:19 → 선점 16:20. **약 28분**을 붙어 있었는데
98GB 다운로드는 콜드에서 1~3시간이 걸린다. **선점 간격 < 준비 시간**이면 b0p 는 영원히 전진하지 못한다.
이번 사이클에서 학습 스텝은 0.

★잃은 것은 없다 — `gs215` 는 HF 에 23/23 98.3GB 완결로 있고, 부분 `gs220`(18/23)은
`pull_resume_ckpt.py` 의 완결성 검사가 이번에도 정확히 걸러냈다(로그 `= 215`).

**조치**: 없음. 중단 4양상 ① (자동 retry) 이므로 대기. 단 **같은 사이클이 2~3회 더 반복되면**
선택지를 올린다 — (a) 계속 대기 (b) 취소·재제출로 다른 노드 추첨 (c) **체크포인트를 노드 로컬
`/scratch` 대신 이미 마운트된 blob(`/scratch/AzureBlobStorage_CODE/...`, 6.5TB 여유)에 저장**해
재다운로드 자체를 없앤다. (c) 는 런처 yaml 변경이라 승인 사항이고, blobfuse 쓰기 속도가 학습을
막을 위험이 있어 작은 검증이 먼저다.

**b2p gs183** (370 s/step, ETA 12.0h) · **b3s gs67** (334 s/step, ETA 21.6h) — 둘 다 순항.
HF 푸시 정상: `prune global_step_50 (keep latest 3)` 16:12, b3s gs55·60·65 완결 23/23.

b3s 창 평균 (gs67):

| 지표 | 10평균 | 20평균 |
|---|---|---|
| `meta_emit_rate` | **0.9988** | **0.9987** |
| `rmeta_mean_scored` | −0.1048 | −0.0768 |
| `pos_rate` / `neg_rate` | 0.2105 / 0.2521 | 0.2139 / 0.2457 |
| `entropy` | 0.2759 | 0.2547 |
| `correctness` | 0.4392 | 0.4627 |

발화 침식 0 유지. rmeta 는 gs45~67 내내 **0 아래에서 진동하고 음수율이 양수율보다 높다** —
20스텝 창으로 봐도 반전되지 않는다.
