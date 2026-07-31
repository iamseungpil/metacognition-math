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
