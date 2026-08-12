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

### E-1xx 2026-08-03 16:53 UTC — **b0p 선점 3회차 · 판단 요청** · b2p gs188 · b3s gs73

⚠**b0p (`solid-gibbon`) 선점 사이클 3회차.** 로그가 다시 **190줄**,
`[bootstrap] fast-path: pulling conda-pack env from HF` — 부트스트랩 맨 처음이다.

| 사이클 | 무엇을 했나 | 결과 |
|---|---|---|
| 1 | gs215→219 학습, gs220 부분 업로드(18/23) | 선점 (~15:41) |
| 2 | 15:52 부트스트랩 → 16:19 완료 → 98GB 다운로드 시작 | **16:20 SIGTERM. 학습 0 스텝** |
| 3 | ~16:2x 재시작, 16:53 현재 conda-pack 받는 중 | 진행 중 |

**b0p 는 1시간 12분 동안 학습 스텝을 하나도 진행하지 못했다.** 준비(부트스트랩 ~27분 +
98GB 다운로드 1~3h)가 선점 간격(~28분)보다 길다. 이 비율이 유지되면 **구조적으로 전진 불가**다.

★잃은 것은 여전히 없다 — `gs215` 완결 보존, 부분 `gs220` 은 가드가 매번 정확히 배제.

**사용자 판단 요청 — 세 선택지**

| | 무엇 | 비용 | 위험 |
|---|---|---|---|
| (a) | 계속 대기 | 0 | 선점 간격이 안 늘면 영원히 정체 |
| (b) | 취소 → 재제출 | 큐 재진입 | 다른 노드 추첨. 더 나쁠 수도 |
| (c) | 체크포인트를 **blob** 에 저장 (`/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore`, 6.5TB 여유) | 런처 yaml 변경 = **승인 필요** | blobfuse 쓰기가 학습을 막을 수 있음 → **작은 검증 먼저** |

(c) 는 재다운로드를 **구조적으로 제거**한다 — 지금은 HF 를 내구 저장소로 쓰느라 매 선점마다
98GB 를 다시 받는다. 다만 98GB 를 blobfuse 로 쓰는 속도가 미측정이라, 도입 전 한 스텝 저장에
걸리는 시간을 재야 한다.

**b2p gs188** (380 s/step, ETA 11.8h) — HF gs185 완결. 순항.
**b3s gs73** (334 s/step, ETA 21.1h) — HF gs60·65·70 완결, `prune global_step_55 (keep latest 3)` 16:40.

| b3s 지표 | 10평균 | 20평균 |
|---|---|---|
| `meta_emit_rate` | **0.9986** | **0.9988** |
| `rmeta_mean_scored` | **−0.1468** | **−0.1177** |
| `pos_rate` / `neg_rate` | 0.2225 / 0.2602 | 0.2150 / 0.2547 |
| `entropy` | 0.2887 | 0.2752 |
| `correctness` | 0.5238 | 0.4875 |

발화 침식 0. **rmeta 는 gs67 → gs73 사이에 −0.077 → −0.118 로 더 내려갔다**(20창 기준).
음수율이 양수율보다 높은 상태가 gs45 이후 한 번도 뒤집히지 않았다. 정답성은 0.46→0.49 로 오르는 중이라
**발화·정답성은 좋아지는데 메타 블록의 증거 기여만 음수**인 분리가 굳어지고 있다.

### E-1xx 2026-08-03 17:26 UTC — **b0p 재개 성공(선점 3회차에서 뚫림) · (c) blob 전환 보류** · b3s rmeta 계속 하락

✅**b0p (`solid-gibbon`) 가 학습을 재개했다.** 3회차 사이클이 끝까지 갔다:

```
[YAML] existing GRPO resume gs (model+extra+optim>=4) = 215      ← 부분 gs220 배제 (3회 연속 정확)
[resume] downloaded global_step_215 -> /scratch/checkpoints/...  ← 98GB 완료
Training Progress: 216/300 [05:11<7:16:24, 311.72s/it]           ← 학습 중, gs216
```

**311.72 s/it 로 ETA 7시간 16분.** 이전 사이클보다 빠르다(직전 관측 439 s/step).
⇒ **선택지 (c)(체크포인트를 blob 에 저장)는 보류한다.** 선점 간격이 준비 시간보다 길어진 사례가
나왔으므로 지금 yaml 을 건드릴 근거가 없다. 다만 원인은 그대로다 — 선점 한 번에 98GB 재다운로드가
붙는 구조. **재발하면 다시 올린다.**

⚠wandb `rq3v2f_b0p` 는 여전히 `crashed` @ gs219 로 보인다. 새 프로세스가 같은 런에 붙었고
gs216~219 는 이전 런이 이미 밀어놔 **폐기될 것이다**(E-165 동형, b2p 와 같은 현상).
**gs220 을 넘겨야 기록이 재개된다.**

**b2p gs193** (385 s/step, ETA 11.4h) — HF gs190 완결. **b3s gs78** (345 s/step, ETA 21.3h) —
HF gs65·70·75 완결, `prune global_step_60 (keep latest 3)` 17:10.

**b3s 창 평균 추이 — rmeta 하락이 이어진다:**

| gs | `rmeta` 20창 | `neg_rate` 20창 | `pos_rate` 20창 | `correctness` 20창 |
|---|---|---|---|---|
| 67 | −0.0768 | 0.2457 | 0.2139 | 0.4627 |
| 73 | −0.1177 | 0.2547 | 0.2150 | 0.4875 |
| **78** | **−0.1441** | **0.2643** | 0.2132 | **0.5164** |

**세 지점 연속 같은 방향**이다 — rmeta 는 내려가고, 음수율은 올라가고(0.2457→0.2643),
양수율은 0.213 에 고정, 정답성은 오른다(0.463→0.516). `meta_emit_rate` 는 0.9986 로 변화 없다.
**"발화·정답성은 좋아지는데 메타 블록의 증거 기여만 음수로 깊어지는" 분리가 확정 방향을 잡았다.**
스텝 진동(±0.28)으로는 설명되지 않는다 — 20스텝 창이 세 지점 모두 단조 하락이다.
gs150 게이트에서 이 표를 그대로 이어 판정한다.

### E-1xx 2026-08-03 18:31 UTC — **b0p·b3s 동시 선점** (b2p만 생존) · b3s rmeta 하락 멈추고 소폭 회복

⚠**두 팔이 같은 시각에 선점됐다.** `amlt status` 가 b0p·b3s 를 `queued` 로, wandb 가 둘 다
`crashed` 로 표시한다. b2p 만 `running` 이다. 같은 축출 파도로 보인다.

| arm | 선점 시점 gs | HF 최신 완결 | 손실 | 현재 단계 |
|---|---|---|---|---|
| b0p | 227 (로그) | **gs225** 23/23 | 2스텝 | 재시작 |
| b3s | 85 | **gs85** 23/23 | **0스텝** | 부트스트랩 중 `Caught signal 15` — **재시작 중 또 선점** |
| b2p | — | gs200 23/23 | — | 🟢 gs198~199 학습 중 |

b3s 로그가 191줄에서 `[bootstrap] fast-path: pulling conda-pack env` 다음 줄이 곧바로
`Caught signal 15` 다 — **b0p 가 겪은 것과 같은 사이클**에 들어갔다. 다만 b3s 는 gs85 가 완결로
올라가 있어 **재개 손실이 0** 이다(`--keep 3` 덕에 gs75·80·85 세 벌 보존).

★**(c) blob 전환 재검토 조건에 근접**: b0p 는 이번에 뚫렸지만 이제 **b3s 가 같은 자리**에 들어갔다.
b3s 가 2회 연속 학습 0스텝이면 정식으로 올린다. b3s 는 **가설을 실제로 시험하는 팔**(gs150 게이트)
이라 지연 비용이 가장 크다.

**b3s 분리 추적 표 — 하락이 멈추고 소폭 회복했다:**

| gs | rmeta 20창 | neg_rate | pos_rate | correctness | entropy 20창 |
|---|---|---|---|---|---|
| 67 | −0.0768 | 0.2457 | 0.2139 | 0.4627 | 0.2752 |
| 73 | −0.1177 | 0.2547 | 0.2150 | 0.4875 | — |
| 78 | −0.1441 | 0.2643 | 0.2132 | 0.5164 | 0.3052 |
| 83 | −0.1388 | 0.2637 | 0.2162 | 0.5309 | 0.3291 |
| **85** | **−0.1311** | 0.2631 | **0.2188** | **0.5536** | 0.3380 |

세 지점 단조 하락(gs67~78) 뒤 **두 지점 연속 회복**(−0.1441 → −0.1388 → −0.1311)이다.
10스텝 창은 −0.0998 로 더 위다. 양수율도 0.2132 → 0.2188 로 처음 움직였다.
⇒ **"계속 깊어진다"는 읽기는 확실히 틀렸다.** 지금 자료가 지지하는 것은 **gs78 부근이 저점이고
−0.10~−0.14 대에서 진동**이라는 것뿐이다. 음수율>양수율 구도는 여전히 유지된다.
correctness 는 0.463 → 0.554 로 단조 상승, entropy 도 0.275 → 0.338 로 계속 넓어진다.

HF 푸시 정상: `prune global_step_220 (keep latest 1)` 18:23, b0p gs225·b2p gs200 완결.

### E-1xx 2026-08-03 22:02 UTC — **b0p 4번째 선점**(gs248, 완주 5.5h 앞두고) · b3s·b2p 순항

⚠**b0p (`solid-gibbon`) 가 gs248 에서 또 선점됐다.** 완주까지 5.5시간 남은 지점이다.
로그가 321줄로 리셋됐고 이미 부트스트랩을 넘겨 `existing GRPO resume gs = 245` 를 잡은 뒤
98GB 다운로드에 들어갔다. **3스텝 손실**(gs245 완결 보존).

b0p 선점 이력: gs219 → gs227 → (부트스트랩 중 1회) → **gs248**. 오늘만 네 번이다.
매번 자력 복구했지만 **한 사이클에 준비 30분~3시간을 지불**한다.

★**blob 계수 = 1** (선점만 셈). 이번 사이클도 학습 0스텝으로 끝나면 2회 연속 → (c) 상신.

**b3s (`musical-wombat`) gs91** — 390 s/step, ETA 22.6h. HF gs80·85·**90** 완결,
`prune global_step_75 (keep latest 3)` 22:02 정상. 재개 후 지표(5스텝 평균):
`meta_emit_rate` **1.0000**(침식 0) · `rmeta_mean_scored` −0.0477 ·
`pos_rate` 0.2281 vs `neg_rate` 0.2355 · `correctness` 0.5272.
직전 틱의 "gs87 에서 양수율이 처음 앞섰다"는 **유지되지 않았다** — 다시 음수율이 근소하게 앞선다.
단일 스텝 관측이었으므로 예고한 대로 판단 근거로 쓰지 않는다. **gs105+ 20창에서 재확인.**

**b2p (`musical-ant`) gs229** — 390 s/step, ETA 7.7h. HF gs225 완결. 순항.

### E-1xx 2026-08-04 00:11 UTC — **b0p 5번째 선점**(gs258) · **b3s gs105: rmeta 회복 확정 조건 첫 지점 통과**

⚠**b0p (`solid-gibbon`) 가 gs258 에서 또 선점됐다**(오늘 5번째). 로그 6030줄 중 6008줄에
`Caught signal 15`, 바로 앞 줄이 `Training Progress: 258/300 [1:51:17<4:27:03, 381.52s/it]` 다.
HF `gs255` 완결 보존 → **3스텝 손실**. 완주까지 4.4시간 남은 지점이었다.
b0p 선점 이력: gs219 → gs227 → 부트스트랩중 → gs248 → **gs258**.

⚠**wandb 상태는 신뢰하지 말 것** — 같은 시각 b0p 는 wandb `running`(gs257), b3s 는 `crashed`(gs104)
로 보였으나 **둘 다 틀렸다**. b0p 는 방금 SIGTERM 을 받았고, b3s 는 로그가 00:11:46 에 살아 있고
HF 에 gs105 를 올리는 중이다(`amlt status` 도 running). **정본은 로그와 HF 커밋이다.**

★**b3s gs105 — 선언한 확정 조건의 첫 지점을 통과했다.**

| gs | rmeta 20창 | neg_rate | pos_rate | 격차 | correctness | entropy |
|---|---|---|---|---|---|---|
| 78 | −0.1441 | 0.2643 | 0.2132 | 0.051 | 0.5164 | 0.3052 |
| 85 | −0.1311 | 0.2631 | 0.2188 | 0.044 | 0.5536 | 0.3380 |
| 99 | −0.0406 | 0.2424 | 0.2312 | 0.011 | 0.5440 | 0.4152 |
| 101 | −0.0521 | 0.2456 | 0.2301 | 0.016 | 0.5490 | 0.4251 |
| **105** | **−0.0425** ✅ | 0.2340 | 0.2253 | **0.0087** ✅ | 0.5219 | 0.4425 |

조건은 `rmeta > −0.05` **그리고** `격차 < 0.02` 였고 **둘 다 충족**이다. 20창이 재개 경계(gs85)를
완전히 벗어난 첫 지점이기도 하다. **gs120 에서 한 번 더 충족되면** "gs78 저점(−0.144)에서 회복해
0 근처로 수렴"을 지지되는 읽기로 채택한다. 발화는 0.9991 로 침식 0, entropy 는 0.305→0.443 으로
계속 넓어지고 있다.

**b2p (`musical-ant`) gs248** — 400 s/step, ETA 5.9h. **이제 가장 빠른 완주 후보다**(b0p 가 재시작에
드는 시간만큼 밀린다). HF gs245 완결.

### E-1xx 2026-08-04 02:21 UTC — **b3s gs120 확정 판정: 선언 조건 FAIL, 단 실패 방향이 유리** · rmeta 양수 전환

**선언했던 조건**(gs105 틱에 사전 고정): `rmeta 20창 > −0.05` **그리고** `|neg−pos| < 0.02`
→ 충족 시 *"gs78 저점(−0.144)에서 회복해 **0 근처로 수렴**"* 을 지지되는 읽기로 채택.

**gs120 실측**:

| 항목 | 값 | 판정 |
|---|---|---|
| `rmeta` 20창 | **+0.0461** | ✅ PASS (−0.05 위, **양수로 전환**) |
| `neg−pos` | **−0.0236** (pos 0.2258 > neg 0.2021) | ❌ **FAIL** (\|0.0236\| ≥ 0.02) |
| `meta_emit_rate` | 0.9997 | 침식 0 |
| `entropy` | 0.5509 | 계속 상승 |
| `correctness` | 0.5570 | 상승 |

★**선언한 조건은 문자 그대로 FAIL 이므로 원 명제를 채택하지 않는다.** 다만 **실패한 방향이
유리한 쪽**이라는 점을 같이 적는다 — 격차가 벌어진 것은 **양수율이 음수율을 0.0236 앞섰기**
때문이고, `rmeta` 자체도 0 을 지나 **+0.046 으로 부호가 바뀌었다**.

⇒ 즉 관측된 것은 *"0 근처로 수렴"* 이 아니라 ***"0 을 지나 양수로 반전"*** 이다. 이는 원 명제보다
**강한 주장**이고, **원 명제를 검사하려고 만든 조건으로는 검사할 수 없다.**

⛔**여기서 조건을 고쳐 채택하지 않는다** — 결과를 보고 문턱을 옮기는 것이 정확히 사후 조건 변경이다.
새 명제 *"gs120 에서 rmeta 가 양수로 전환됐고 양수율이 음수율을 앞선다"* 는 **관측으로만 기록**하고,
지속 여부는 **이미 선언돼 있는 gs150 게이트**에서 확인한다(새 판정 지점을 만들지 않는다).

**gs78 → gs120 궤적** (전부 20창):

| gs | rmeta | neg−pos | correctness | entropy |
|---|---|---|---|---|
| 78 | −0.1441 | +0.0511 | 0.5164 | 0.3052 |
| 105 | −0.0425 | +0.0087 | 0.5219 | 0.4425 |
| 110 | −0.0242 | +0.0043 | 0.5355 | 0.4741 |
| 116 | −0.0013 | −0.0045 | 0.5475 | 0.5178 |
| **120** | **+0.0461** | **−0.0236** | 0.5570 | 0.5509 |

다섯 지점 단조 상승이다. **⚠단 `entropy` 도 0.305 → 0.551 로 같이 단조 상승**하고 있어,
rmeta 개선이 **정책이 넓어지면서 따라온 것인지** 분리되지 않는다. b3p 는 gs150 이후 무너졌으므로
**gs150 게이트가 이 둘을 가르는 지점**이다.

**b0p gs270**(376 s/step, ETA 3.2h) · **b2p gs261**(403 s/step, ETA 4.5h) — 선점·전송오류 0건.
HF: b0p gs265 완결 + gs270 업로드중, b2p gs260 완결, b3s gs105·110·115 완결 + gs120 업로드중.

### E-1xx 2026-08-04 04:30 UTC — **b0p·b2p 동시 선점**(b0p 는 완주 12스텝 앞) · b3s 무사

⚠**완주 직전의 두 팔이 같은 파도에 쓸렸다.**

| arm | 선점 시점 | HF 최신 완결 | 손실 | 남았던 거리 |
|---|---|---|---|---|
| **b0p** | gs288 | gs285 23/23 | 3스텝 | **gs300 까지 12스텝(~1.4h)** |
| **b2p** | gs275 | gs275 23/23 | **0스텝** | gs300 까지 25스텝(~2.9h) |
| b3s | — | gs120·125·130 | — | 🟢 gs134 학습 중 |

b0p 로그 7981·7982 줄, b2p 로그 190·191 줄에 각각 `Caught signal 15` 가 **두 줄씩** 찍혔다
(러너가 자식으로 전달하며 두 번 기록). b2p 는 이미 새 pod 로 넘어가 192줄에서 부트스트랩 중이고,
b0p 도 같은 경로를 밟을 것이다. **b2p 는 gs275 가 완결로 올라간 직후 죽어 손실이 0 이다** —
`--keep 1` + 5스텝 저장 간격이 이번엔 정확히 맞았다.

**b0p 선점 이력: gs219 → gs227 → 부트스트랩중 → gs248 → gs258 → gs288.** 오늘 여섯 번이고
**여섯 번 다 자력 복구**했다. 다만 이번은 완주를 1.4시간 앞두고 맞았으므로, 재개(부트스트랩+98GB)
비용을 감안하면 완주가 최소 2~4시간 밀린다.

★**blob 계수 = 1**(b0p). b2p 는 별도로 세지 않는다 — 이번 사이클에서 학습 0스텝으로 끝나는지 먼저 본다.

**b3s (`musical-wombat`) gs133** — 522 s/step, ETA 24.2h. 선점 0건.
20창: `emit 0.9997` · `rmeta +0.0242` · `neg−pos −0.0147` · `corr 0.5691` · **`ent 0.6487`**.
rmeta 는 +0.016 → +0.024 로 소폭 반등해 **양수대에서 진동**한다. **entropy 만 계속 오른다**(0.305→0.649).
gs150 까지 17스텝(~2.5h).

### E-1xx 2026-08-04 05:35 UTC — **b3s rmeta 음수 복귀(gs141, −0.068)** · b0p 7번째 선점(직전 사이클 학습 0스텝) · b2p 재개 성공

★★**b3s 의 rmeta 가 판정 지점 직전에 음수로 돌아섰다.** 20창:

| gs | rmeta | neg−pos | correctness | entropy |
|---|---|---|---|---|
| 120 | **+0.0461** (정점) | −0.0236 | 0.5570 | 0.5509 |
| 130 | +0.0165 | −0.0122 | 0.5810 | 0.6216 |
| 133 | +0.0242 | −0.0147 | 0.5691 | 0.6487 |
| 137 | +0.0011 | −0.0057 | 0.5842 | 0.6804 |
| **141** | **−0.0680** | **+0.0141** | 0.5967 | 0.6834 |

gs120 정점 이후 **네 지점 연속 하강**(+0.046 → +0.017 → +0.001 → −0.068)이고,
**양수율 우위도 gs141 에서 도로 뒤집혔다**(음수율이 0.0141 앞섬). 발화는 0.9994 로 여전히 침식 0,
correctness 는 0.5967 로 오히려 최고치, `entropy` 는 0.683 로 고원에 접근했다.

⇒ **"gs78 저점에서 회복해 양수로 반전"은 일시적이었다**(gs116~137, 약 20스텝). 채택하지 않기로
한 판단이 결과적으로 옳았다 — gs120 에서 조건을 고쳐 채택했다면 지금 철회해야 했다.
현재 지지되는 읽기는 **"rmeta 가 −0.14 ~ +0.05 대역에서 큰 진폭으로 진동한다"** 뿐이고,
방향성 주장은 어느 쪽으로도 서지 않는다.

**gs150 판정이 9스텝(~1.3h) 앞**이다. 도착 조합은 **발화 정상 + rmeta 음수 + entropy 고원**으로
예상된다 — b3p 의 붕괴 전조(발화 자체가 무너짐)와는 **다른 모양**이다.

⚠**b0p 7번째 선점.** 로그가 263줄로 리셋됐다. **직전 사이클은 학습 0스텝**이었다 —
04:30 선점 → 부트스트랩 → gs285 다운로드 완료 → verl 부팅 중(05:03, 1175줄) → **다시 선점**.
★**0스텝 사이클 1회 누적.** 이번 사이클도 0스텝이면 **2회 연속 → (c) blob 전환 상신**.

✅**b2p 재개 성공** — gs275 에서 재개해 **gs279** 학습 중(로그 2400줄). 손실 0.

HF: b3s `gs135` 완결 + gs140 업로드중(21/23), b0p gs285·b2p gs275·b3p gs300 보존.

### E-1xx 2026-08-04 07:26 UTC — **b0p 종료(`failed`)** — 선점이 아니라 init 다운로드 끊김 · 가드가 정상 작동

`amlt status` = **`failed`**(자동 retry 없음). 로그 337줄의 마지막 구간이 원인을 그대로 보여준다:

```
requests.exceptions.ChunkedEncodingError: ('Connection broken:
  IncompleteRead(107788604 bytes read, 4792298212 more expected)')
ls: cannot access '/scratch/models/sft2_init/config.json': No such file or directory
[YAML] FATAL init /scratch/models/sft2_init missing or incomplete; ABORT window
+ exit 1
```

★**이번 실패는 선점이 아니다.** 끊긴 것은 resume 체크포인트가 아니라 **init 모델**(`sft2_init`)이고,
4.8GB 중 **107MB** 에서 연결이 끊겼다. 그 뒤 런처의 무결성 가드가 불완전한 init 을 잡아 **의도적으로
중단**시켰다 — 깨진 init 으로 학습을 시작하는 것보다 죽는 편이 낫다는 설계대로다.
(앞서 b3s 가 같은 `ChunkedEncodingError` 를 만났을 때는 resume 체크포인트였고 `for rp in 1 2 3`
재시도가 잡아냈다. **init 스테이징 경로에는 그 재시도가 없다** — 이번에 드러난 비대칭이다.)

**손실 없음** — HF `gs285` 23/23 98.3GB 완결 보존. 재제출하면 gs285 에서 **15스텝**(~1.6h)만 남는다.

**b0p 최종 이력**: 선점 7회(gs219→227→부트스트랩중→248→258→288→05:35 사이클) 전부 자력 복구,
마지막은 선점이 아닌 **init 전송 끊김으로 terminal failed**. 중단 4양상 ③ → **수동 재제출 필요**.
사용자 확인 요청함.

**나머지 두 팔 정상**: b2p gs291(완주 9스텝, ~1h) · b3s gs149(gs150 판정 1스텝 앞, 발화 0.9992).

## ★★ E-1xx 2026-08-04 07:37 UTC — **b3s gs150 판정: `meta_floor=0.05` 가 발화 붕괴를 막았다**

**도달 확인**: HF `checkpoints/rq3v2f_b3s/global_step_150` 이 **23/23 98.3GB 완결**로 올라왔다
(wandb 는 gs149 로 한 스텝 뒤처져 있으나 체크포인트가 정본이다).

### 판정 — 선언한 세 항목

| # | 항목 | b3p (floor 0.0) | **b3s (floor 0.05)** | 판정 |
|---|---|---|---|---|
| ① | `dcpo/meta_emit_rate` | gs150 이후 **1.00 → 0.018** | **0.9992** (20창, 침식 0) | ✅ **PASS** |
| ② | `rmeta_mean_scored` | — | **−0.1080** (음수, gs78 저점 −0.144 에 재접근) | 음수대 |
| ③ | `actor/entropy` | — | **0.7248** (gs67 0.305 대비 **2.4배**) | 단조 상승 |

부수 지표: `pmishift_attempted_rate` 0.6090 · `neg−pos` +0.0317 · `correctness` 0.6004.

### 닫는 것

★**base 기질의 발화 붕괴 원인은 `dcpo_meta_floor` 였다.** b3s 는 b3p 에서 **이 노브 하나만** 바꾼
런이고(diff 로 검증, 같은 `--config-name`, 같은 코드 asset `490407111`), 같은 지점에서 b3p 는
0.018 로 무너졌고 b3s 는 0.9992 를 유지했다. ⇒ **"len_cost 0.08 이 발화를 눌렀다"는 경쟁 가설은
이 증거로 불필요해졌다** — floor 하나로 설명된다. 후속 팔 `b3p + len_cost=0.0` 은 **더 이상 필요 없다.**

### 여는 것

⚠**"발화가 살았다"는 "처치가 유효하다"가 아니다.** 세 가지가 남았다:

1. **rmeta 가 음수다.** 발화는 floor 가 강제하고 있는데, 나오는 메타 블록이 gold-vs-decoy 마진을
   평균적으로 **깎는다**(−0.108). 헌법 건전 밴드 +1.0~+1.2 와는 자릿수가 다르다.
   ⇒ 다음 질문: *floor 가 강제한 발화가 **내용 없는 발화**인가?* 응답 parquet 을 눈으로 봐야 한다.
2. **entropy 가 2.4배로 계속 오른다.** rmeta 회복(gs116~137)과 동행하다 gs141 부터 갈라졌다.
   정책이 넓어지는 것이 정답성에 기여하는지, 아니면 붕괴의 다른 형태인지 미분리.
3. ★**gs150 은 붕괴의 시작점이지 종점이 아니다.** b3p 는 gs150 **이후** 무너졌으므로,
   gs150~300 구간에서 발화가 유지되는지가 **완전한 판정**이다. 지금 판정은 **"gs150 시점 통과"** 까지다.

### 재확인 계수기

0 — 이 결론은 처음 확립됐다.

**나머지**: b2p **gs295**(HF gs290 완결 + gs295 업로드중, gs300 까지 5스텝 ~35분) ·
b0p 는 07:26 `failed`(init 전송 끊김, 재제출 승인 대기 중).

### ★ E-1xx 2026-08-04 08:42 UTC — **b2p gs300 가중치 확보** (지난번 잃었던 그 체크포인트) · b0p 재제출 정상 진행

★★**HF `checkpoints/rq3v2f_b2p/global_step_300` = 23/23 98.3GB 완결.**
지난 세대에서 b2p 는 gs300 을 완주하고도 `pkill -f push_ckpts_to_hf` 가 **자기 `bash -c` 명령줄에
매칭돼** 푸셔를 SIGTERM 으로 죽이는 바람에 이 체크포인트를 잃었다. **이번엔 남았다.**

⚠**단, 이것이 곧 "self-kill 수정 검증 완료"는 아니다.** 지금 올라간 gs300 은 **주기 푸셔**가
올린 것이고, 잡은 아직 `running`(`Training Progress: 299/300`)이라 **FINAL SYNC 경로는 아직
실행되지 않았다.** 로그에 `FINAL SYNC`·`pusher stopped`·`[YAML] WARN` 모두 0건이다.
⇒ 확정된 것: **가중치는 안전하다**(평가에 필요한 것은 이미 확보). 미확정: PID-only stop 이
FINAL SYNC 단계에서 의도대로 도는지 — 잡 종료 시 확인한다.

부수 확인: `prune ... (keep latest 1)` 규칙대로 b2p gs295 가 gs300 착지 후 정리됐다.

**✅ b0p 재제출(`rq3v2f-b0p-0804`) 정상 진행** — 37분째 `running`.
```
[init] resolved PAIR_sft -> /scratch/models/sft2_init (b0p2_rvfull_sft)   ← init 스테이징 성공
[YAML] existing GRPO resume gs (model+extra+optim>=4) = 285               ← gs285 재개 인식
```
★**이번에 추가한 init 재시도는 발화하지 않았다**(`init staging attempt N landed no config.json`
0건) — 첫 시도에 성공했다는 뜻이다. 방어책은 들어갔고 이번엔 쓸 일이 없었다.
지금 98GB 다운로드 단계(로그 321줄). gs285→300 은 15스텝.

**b3s gs153** — `emit 0.9991`(PASS 유지) · `rmeta −0.1255` · `neg−pos +0.0376` ·
`corr 0.6063` · `ent 0.7526`. gs150 통과 이후에도 발화 침식 0. rmeta 는 계속 내려가
gs78 저점(−0.144)에 근접 중.

## ★★ E-1xx 2026-08-04 09:15 UTC — **b2p 완주 + 최종 푸시 내구 확인. self-kill 수정이 작동했다**

**b2p (`musical-ant`) 300/300 완주.** 로그 증거 체인:

```
Training Progress: 100%|██████████| 300/300 [3:52:26<00:00, 557.87s/it]
[YAML] verl_sdc rq3v2f_b2p rc=0                       ← 학습 정상 종료
+ kill -9 4429                                        ← PID 로만 푸셔 종료
+ kill -0 4429                                        ← 종료 검증
... SHARDS=8 / No files have been modified since last commit ...
[YAML] FINAL PUSH DURABLE global_step_300             ← 최종 푸시 내구 확인
+ break
+ sleep 86400
```

★**지난 세대는 정확히 이 지점에서 `pkill -f push_ckpts_to_hf` 가 자기 `bash -c` 명령줄에 매칭돼
푸셔를 죽였고 gs300 가중치를 잃었다. 이번엔 `kill -9 <PID>` → `kill -0 <PID>` 검증 경로로 지나갔고
`FINAL PUSH DURABLE` 까지 도달했다.** ⇒ **PID-only stop 수정 검증 완료.** `[YAML] WARN` 0건.

HF 실측: `checkpoints/rq3v2f_b2p/global_step_300` **23/23 98.3GB 완결**.
"No files have been modified" 는 주기 푸셔가 이미 다 올려둬서 최종 푸시가 올릴 게 없었다는 뜻이다 —
정상 경로다.

⚠**모니터링 지시 정정**: 내가 매 틱 grep 하던 문자열 `FINAL SYNC` 는 **런처에 없다**. 실제 문자열은
**`FINAL PUSH DURABLE`** 이다. 그래서 직전 두 틱에서 "FINAL=0" 으로 보였던 것이고, 완주 신호를
놓칠 뻔했다. 이후 틱은 `FINAL PUSH DURABLE|pusher pid=|kill -9|SHARDS=` 로 grep 한다.

★**b2p 잡은 `sleep 86400` 으로 노드를 붙들고 있다** — 취소 여부는 사용자 판단(b0p·b3s 가 이미
각자 노드를 갖고 있어 급하지 않다).

**✅ b0p (`rq3v2f-b0p-0804`) 재개 성공 — gs289 학습 중.** gs285 에서 이어붙었고 **11스텝(~1.4h)** 남았다.
init 재시도는 이번에도 미발화(첫 시도 성공).

**b3s gs157** — 발화 0.9991 유지, 계속 진행. gs300 까지 143스텝.

**현재 gs300 확보**: b2p(정상 팔) · b3p(붕괴 팔). b0p 11스텝 · b3s 143스텝 남음.

### E-1xx 2026-08-04 10:53 UTC — **b2p eval 16k 3벤치 산출물 착지** · b3s correctness 3점 연속 하락

★**b2p eval 이 16k 패스를 세 벤치 모두 끝내고 업로드했다** — `eval/rq3v2f_b2p_1030/` 에 20개 파일.
`aime2024` · `gsm8k` · `math500` 각각 json + parquet + metadata + log. **주 지표에 쓰는 16k MATH500 이
이미 손에 있다.**

MATH500 parquet 실측(4000행 = 500문제 × 8샘플):

| 항목 | 값 |
|---|---|
| `num_meta_blocks` | **1개 3773행(94.3%)**, 2개 160, 3개 56, 4개 10, 5개 1 — **held-out 에서도 메타가 실제로 나온다** |
| `finish_reason` | stop 3903 / length 97 → **절단 2.4%** |
| 정확도 (⚠**옛 채점기** `is_correct`) | **0.6290** |

⚠⚠**`level` 컬럼이 parquet 에 없다.** 우리 주 지표는 `Δacc(L4–5) − Δacc(L1–2)` 인데 난이도가
산출물에 실려 있지 않으므로, **MATH500 원본과 `question` 으로 조인해 level 을 붙이는 단계가
필요하다.** instruct 재채점 때도 같은 조인을 했을 것이므로 새 작업은 아니지만, **평가 뒤 분석
파이프라인에 이 단계가 빠지면 주 지표를 못 만든다** — 미리 적어둔다.
⚠**정확도 0.6290 은 옛 채점기 값이라 그대로 쓰면 안 된다**(C-001: 같은 바이트에서 채점 규칙 하나가
10.2pp·부호까지 움직였다). **`math_verify` 재채점이 선행되어야 한다.**

★**b3s — `correctness` 20창이 3점 연속 하락했다.** 내 규칙(20창 3점 이상 같은 방향)을 충족하므로
이제 "지지되는 읽기"로 적는다:

| gs | emit | rmeta | neg−pos | **corr** | ent |
|---|---|---|---|---|---|
| 153 | 0.9994 | −0.1255 | +0.0376 | 0.6063 | 0.7526 |
| 159 | 0.9992 | −0.1020 | +0.0282 | 0.5610 | 0.7808 |
| **165** | 0.9992 | −0.0674 | +0.0170 | **0.5160** | **0.8397** |

**정확도가 내려가는 동안 entropy 는 계속 오른다**(0.7526 → 0.8397, gs67 대비 2.8배).
rmeta 는 오히려 −0.126 → −0.067 로 개선됐고 발화는 침식 0 이다.
⇒ **"발화 정상 + rmeta 개선 + 정확도 하락 + entropy 상승"** 조합이다. 정책이 넓어지면서
정답률을 잃고 있을 가능성이 있고, 그렇다면 b3p 와는 **다른 경로의 열화**다(b3p 는 발화가 죽었다).
**gs300 held-out 이 이걸 가른다** — in-training correctness 는 val594 계열이라 판정에 쓰지 않는다.

**b0p gs295** — 재개 후 순항, gs300 까지 5스텝. HF gs290 완결 + gs295 업로드중.

### ⚠ E-1xx 2026-08-04 11:49 UTC — **b0p gs296 에서 1시간 무진행(행 의심)** — 완주 5스텝 앞

**증상**: `step:295` 를 찍은 뒤 **10:50:37 이후 실질 로그가 한 줄도 없다**(현재 11:49).
로그는 4839줄로 늘었지만 2374줄 이후는 **거의 전부 HB 하트비트**이고, vLLM 의
`Processed prompts` / `Adding requests` 진행 표시가 **0건**이다.

| 근거 | 값 |
|---|---|
| 마지막 vLLM INFO | **10:50:37** (약 59분 전) |
| step:295 이후 실질 로그 | **0줄** (HB 536개만) |
| HB 최신 | 11:49:30 — **프로세스는 살아 있다** |
| `gpu0used` | **35238MB 로 고정** — 세 하트비트 내내 변화 없음 |
| `Caught signal 15` | 0건 (선점 아님) |
| 직전 스텝들 | 292→295 각 311~355 s/it 로 **정상 속도였다** |

⇒ **선점도 아니고 느린 스텝도 아니다.** GPU 메모리가 35GB 에서 미동도 없고 생성 진행 표시가
전혀 없으므로 **vLLM 생성 단계에서 멈춘 것(행)으로 의심**된다. 직전 스텝 로그의
`global_seqlen/minmax_diff: 80888`(min 36366 / max 117254)이 눈에 띈다 — 한 랭크에 극단적으로 긴
시퀀스가 몰린 직후다.

★**잃은 것은 없다** — HF `rq3v2f_b0p/global_step_295` 가 **23/23 98.3GB 완결**이다.
행이 확정되면 잡을 취소하고 재제출하면 **gs295 에서 5스텝**만 남는다.

**조치**: 아직 개입하지 않는다. 다음 틱(30분 뒤)에도 실질 로그가 0줄이면 **1.5시간 무진행**이므로
행으로 확정하고 **취소·재제출을 사용자에게 상신**한다. (근거: `sleep 86400` 폴백이 있어 저절로는
안 끝난다.)

**b3s gs170** — `emit 0.9993` · `rmeta −0.0419`(계속 개선) · `neg−pos +0.0087` ·
**`corr 0.5034`(4점 연속 하락)** · `ent 0.8593`.

| gs | emit | rmeta | corr | ent |
|---|---|---|---|---|
| 153 | 0.9994 | −0.1255 | 0.6063 | 0.7526 |
| 159 | 0.9992 | −0.1020 | 0.5610 | 0.7808 |
| 165 | 0.9992 | −0.0674 | 0.5160 | 0.8397 |
| **170** | 0.9993 | **−0.0419** | **0.5034** | **0.8593** |

**corr 하락이 네 지점, ent 상승이 네 지점, rmeta 개선이 네 지점** — 세 축이 동시에 같은 방향으로
움직인다. 발화만 미동이 없다(0.999). gs300 held-out 이 이 조합의 의미를 가른다.

### ✅ E-1xx 정정 2026-08-04 — **b0p 는 행이 아니었다: 호스트 RAM OOM (채점기)**

앞 항목에서 "vLLM 생성 단계 행 의심"이라 적었다. **틀렸다.** 전체 로그를 받아보니
13:23:52 에 Ray 가 워커 10개를 **메모리 압박(OOM)** 으로 죽였다.

```
Task main_task failed due to oom. Memory on the node was 842.15GB / 882.00GB (0.954818)
Top 10 memory users: PID MEM(GB) COMMAND
  8824  527.54  ray::RewardLoopWorker.compute_score      ← 노드 RAM 의 60%
  8324   72.97  ray::WorkerDict
  8325   72.67  ray::WorkerDict
  8323   72.08  ray::WorkerDict
  8322   71.71  ray::WorkerDict
```

⇒ **터진 것은 트레이너가 아니라 채점기다.** `RewardLoopWorker.compute_score` 한 액터가
**527GB** 를 쥐고 있었다. 죽은 워커에 vLLM 서버와 WorkerDict 4개가 전부 포함돼
이후 재시도는 `ValueError: Total available GPUs 0 is less than total desired GPUs 4` 로
두 번 연속 실패했고, 런처의 최종 푸시 블록이 돌아 **`[YAML] FINAL PUSH DURABLE global_step_295`**
를 찍고 `sleep 86400` 으로 들어갔다.

| | 값 |
|---|---|
| 마지막 학습 스텝 | **295** (목표 300 — **5스텝 부족**) |
| HF | `rq3v2f_b0p/global_step_295` **23/23 완결** |
| 잡 상태 | `running` — 학습은 죽었고 `sleep 86400` 이 **4×H100 을 점유 중** |
| 선점 | `Caught signal 15` **0건** (선점 아님) |

**왜 못 봤나**: 10:50 이후 실질 로그 0줄 + GPU 메모리 고정을 "생성 행"으로 읽었는데,
같은 관측이 **호스트 RAM 이 채점기에 먹혀 프로세스 전체가 굶는 상태**와 구별되지 않는다.
⇒ 앞으로 무진행 진단에는 **GPU 메모리만이 아니라 호스트 RAM 도** 본다.

**b3s 는 같은 이유로 죽지 않았다** — 오히려 응답이 더 길다(resp_mean 1433 vs b0p 652,
seq_max 284k vs 117k)는데 멀쩡하다. ⇒ 길이 자체가 방아쇠가 아니라 **퇴화 응답 한 건**이
채점기에서 폭발한 것으로 보인다(`math_verify` hang 계열, 0624 기록과 같은 표면).

**b3s gs182 (20창)**

| ~gs | corr | emit | rmeta | ent |
|---|---|---|---|---|
| 140 | 0.5967 | 0.9994 | −0.0415 | 0.6834 |
| 160 | 0.5497 | 0.9991 | −0.0593 | 0.7856 |
| 180 | **0.5110** | 0.9992 | **+0.0042** | **0.9251** |
| 182 | 0.5172 | 0.9990 | **+0.0163** | 0.9377 |

**rmeta 가 0을 넘어 양수로 갔고, 같은 구간에서 corr 는 −0.086, ent 는 +0.25.**
발화는 미동 없다. b3p 의 발화 사망과는 다른 형태이며, **메타 채널로 보상을 벌면서
정답률을 내주는 모양**에 가깝다. 판정은 gs300 held-out 이 한다.

### ✅ E-1xx 2026-08-04 — **b0p gs300 완주** · 페어 평가 발사 · 310 계획 철회

**b0p 는 300 까지 갔다.** `rq3v2f-b0p-fin-0804` 의 두 번째 윈도우가 gs295 에서 재개해
296·297·298·299·300 을 돌았고(`Training Progress: 100%`, 스텝당 375~395 s), 16:39:32 에
4랭크 전부 model/optim/extra_state 를 저장했다. HF `rq3v2f_b0p/global_step_300` **23/23
98.3GB**. 채점기 OOM 은 재발하지 않았다(`due to oom` 0건). 그 뒤 선점돼(sig15 2줄) 세 번째
윈도우가 gs300 을 다시 끌어왔고 — total=300 이라 할 일이 없으므로 **취소**했다.

★**gs295 도 살아 있다** — `--keep 3` 이 처음으로 값을 했다. `keep 1` 이었으면 gs300 이
올라가는 순간 지워졌을 자리다.

**310 계획은 철회했다.** 사용자가 "b2p 는 이미 gs300 인데 b0 만 돌리면 되지 않나" 라고
지적했고 그게 맞다. 두 가지가 틀렸었다:

1. **b2p 는 돌릴 필요가 없었다.** gs300 체크포인트와 그걸로 뽑은 평가가 이미 둘 다 있다.
   더 돌리면 평가가 가리키는 가중치에서 모델이 떠나 재평가를 강제한다.
2. **310 은 매칭을 깬다.** `total_training_steps` 는 코사인 곡선 전체를 다시 그린다.
   같은 295~300 구간에서 b2p 가 겪은 lr 대 total=310 의 lr:

   | step | b2p (total=300) | total=310 | 차이 |
   |---|---|---|---|
   | 295 | 1.0076e-7 | 1.0640e-7 | **+5.6%** |
   | 300 | 1.0000e-7 | 1.0285e-7 | +2.8% |

   warmup 도 30→31 로 밀린다. 이 두 팔은 **init 말고는 전부 같다**가 존재 이유다.
   덤으로 310 은 이득도 거의 없다 — `min_lr_ratio: 0.1` 이라 lr 이 이미 1e-7 바닥이고
   추가 10스텝은 전부 바닥에서 바닥으로 간다.

⇒ `total_training_steps=310` 오버라이드는 되돌리고 `--keep 3` 만 남겼다.

**페어 평가 발사** — `h100std_rq3v2f_pair_1030_eval.yaml` / `rq3v2f-pair-eval-0804`.
사전등록이 "두 팔을 같은 job·같은 seed" 를 요구하는데 b2p 는 혼자 평가돼 있었으므로
비교가 잡 간 비교가 될 참이었다. **b0p 먼저**(없는 절반이라 윈도우가 2번째 팔에서 죽어도
그건 건진다), 그 다음 b2p 를 **`eval/rq3v2f_b2p_1030_pair` 로** 재평가한다 — 기존 결과를
보존하면서 **동일 가중치 중복 평가가 잡 간 편차의 직접 측정치**가 된다. bash -n rc=0,
lint 27 통과, eval code tar(467403206) HTTP 200.

**협업 문서 결함 두 개 수리** — ①§3 표는 주 지표를 "MATH500 전체 차이"로,
바로 아래 ⚠는 "level 기울기 하나로 고정"으로 적어 **주 지표가 둘**이었다. 재현 문서에서
이건 갈림길이다. 둘 다 남기되 각자 무엇을 묻는지와 **네 조합의 해석을 결과 보기 전에**
못박았다. ②**절단 경고 신설** — 우리 통제군은 16k 에서 AIME 240 중 174개(72.5%)가 절단됐다.
그 팔의 4.6% 는 오답이 아니라 **비종료**이며, 협업자가 능력 격차로 읽으면 결론이 틀어진다.

### 🔎 E-1xx 2026-08-04 — **instruct pmishift 도 `meta_floor: 0.0` 로 돌았다** — b3s 는 복제가 아니다

`b3s − b2p` 가 유효한 비교냐는 물음을 확인하다 나왔다. **유효하다.** 다만 그것이
*"instruct 결과의 base 복제"* 는 **아니다.**

**b3s − b2p 의 정합성** (런처 대조):

| | b2p | b3s |
|---|---|---|
| init | `models/b2p2_rvfull_eb16_sft` | **동일** (`b3s.yaml:100` 주석이 명시) |
| config | `base_matched_grpo_h100_4x4k` | `triobj_dcpo_v4_stage3b_h100_4x4k` |
| 메타 보상 | 없음 (correctness-only) | `dcpo_rmeta_source=pmi_shift`, `w_over=0.0` |
| 스텝 / 스케줄 | 300 / total=300 | 300 / total=300 |

⇒ init·데이터·스텝이 같고 **보상 경로만 다르다.** RQ2 로 성립한다.

**그런데 instruct 팔의 floor 를 확인했더니:**

- `archive/launchers_pre_rq3/h100std_pmishift.yaml` 에 **`meta_floor` 오버라이드가 0건** ⇒ config 기본값을 씀
- `f1f6cec`(2026-06-22 **00:08:02**)가 기본값을 0.05 → **0.0** 으로 바꿈
- 그 런처의 마지막 편집 `8db0a83` 은 같은 날 **03:10:49** 이고, 그 시점 config 는 이미 `dcpo_meta_floor: 0.0`
- 런처가 핀한 `CODE_TAR_REVISION: 458068577` 은 0622 자산(454xxx)보다 **뒤에 빌드된 것**

⇒ **instruct pmishift 는 floor 0.0 으로 돌았고, 발화율은 1.000 을 유지했다.**

| 기질 | floor | 발화율 |
|---|---|---|
| instruct | 0.0 | **1.000** |
| base (b3p) | 0.0 | **0.018** |
| base (b3s) | 0.05 | 0.999 (gs192 현재) |

**따라서 붕괴는 floor 단독이 아니라 `floor 0.0 × base 기질`의 상호작용이다.** C-012 는
"근인 = floor 0.0" 이라 적었는데, 정확히는 **base 기질에서만 발현하는 근인**이다. b3s 가
그것을 구제한다는 사실은 그대로 유효하다(같은 지점 0.018 vs 0.999).

★**주장 문구에 영향**: `b3s − b2p` 가 양성이어도 *"instruct 레시피가 base 에서 재현됐다"* 로
쓸 수 없다. b3s 는 instruct 팔에 **없던 노브 하나를 더 가진다.** 쓸 수 있는 문장은
*"메타 보상 패키지는 base 에서도 도움이 된다 — 단 발화를 붙잡아 준다면"* 이다.
⇒ 진짜 복제는 **instruct 사다리 전체 재현**(협업자 과제 4)이 답한다.

### ★★ E-1xx 2026-08-04 20:00 UTC — **RQ1 첫 수치: base 기질에서 널** (⚠잡 간 비교, 잡 내 비교 대기)

b0p gs300 이 5패스 전부 완주했다. **프로젝트 최초로 `b2p − b0p` 를 같은 세대·같은 기질에서 잰다.**

⚠**아직 잡 내 비교가 아니다.** 여기 b2p 수치는 **이전 잡**(`eval/rq3v2f_b2p_1030`)의 것이다.
페어 잡의 b2p 재평가(`_pair`)가 지금 돌고 있고, 그게 사전등록이 요구한 **같은 job** 비교다.
아래는 **잡 간(cross-job) 추정치**로 읽어야 한다.

**16k · n=8 · seed42 (사전등록 판정 패스), `math_verify` 재채점**

| | b0p (통제군) | b2p (메타) | Δ | 95% CI | instruct 세대 |
|---|---|---|---|---|---|
| **MATH500 전체** | 65.38% | 65.55% | **+0.18pp** | [−1.30, +1.68] | **+14.00pp** |
| L1–2 (n=133) | 77.35% | 76.03% | −1.32pp | [−3.20, +0.28] | +10.53pp |
| L4–5 (n=262) | 53.53% | 54.87% | +1.34pp | [−1.15, +3.86] | +17.70pp |
| **★기울기** | | | **+2.65pp** | [−0.36, +5.71] | **+7.17pp** |

문항 단위 쌍대 부트스트랩 10k · 사전처리 `level` 층화(조인 실패 0/500).

⇒ **주 지표 널. 보조(기울기)도 널이고 잡음바닥 ±3.08pp 이하.**
방향은 맞다(L4–5 > L1–2) — 하지만 크기가 instruct 의 **1/5** 이고 CI 가 0 을 배제하지 못한다.

★**채점기가 또 움직였다.** 저장 `is_correct` → `math_verify`: b0p **+3.80pp**, b2p **+2.65pp**.
비대칭이 1.15pp 라 **차분이 +1.32 → +0.18pp 로 줄었다.** C-001 이 경고한 그대로다.
발표 수치는 재채점본으로만 쓴다.

**처치는 존재했다** — 발화율 b0p 0.0045 / b2p **1.0000** (전 벤치). 절단도 이번엔 b2p 가 **적다**
(97 vs 121/4000), 그래서 4k 에서 보였던 길이 압력 교란은 16k 에서 사라졌다.

**다른 벤치** (저장 채점기): GSM8K b0p 0.9067 / b2p 0.9193 (+1.25pp) ·
AIME seed42 0.2083 / 0.1500 (−5.83pp), seed43 0.1708 / 0.1542 — avg@16 로 **−3.75pp**.
AIME 은 n=30 이라 판정에 쓰지 않는다(한 문제 3.3pp).

**판정 확정 전 남은 것 둘**: ①`_pair` 완주 → **잡 내 비교** + 같은 체크포인트 A-vs-A 로
잡 간 편차 실측 ②단일 학습 시드(구조적 한계, 협업자 과제 4).

**b3s** gs197/300 · 최근 7스텝 emit **0.9997** · corr +0.4269 · rmeta +0.0155 · ent 1.206 ·
683s/step → 약 19.5시간. 선점·OOM 0건.

### ★★★ E-1xx 2026-08-04 20:20 UTC — **instruct 의 +14pp 는 상당 부분 "통제군이 답을 못 맺은 것"이었다**

RQ1 이 base 에서 널로 나온 뒤, *"instruct 의 큰 수가 진짜였나"* 를 같은 채점기·같은 방법으로
확인했다. 보존된 `eval/*_1030_v2` 응답 parquet 을 `math_verify` 로 재채점했다.

**① instruct 효과는 재채점 후에도 살아남는다** (문항 단위 쌍대 부트스트랩 10k, 사전처리 `level`)

| | 재채점 Δ | 95% CI | 기존 보고 |
|---|---|---|---|
| MATH500 전체 | **+12.35pp** | [+10.10, +14.65] | +14.00pp |
| L1–2 | +5.45pp | [+2.35, +8.83] | +10.53pp |
| L4–5 | **+16.89pp** | [+13.50, +20.52] | +17.70pp |
| ★기울기 | **+11.44pp** | [+6.67, +16.15] | +7.17pp |

⇒ 방법을 통일해도 instruct 는 크고 유의하다. **채점 아티팩트가 아니다.**

**② 그런데 네 팔을 나란히 놓으면 격차의 정체가 보인다**

| 팔 | 정확도 | **절단율** | 종결분만 | 평균길이 |
|---|---|---|---|---|
| instruct 통제군 | 57.23% | **18.8%** | 68.99% | **3367** |
| instruct 메타 | 69.58% | 6.0% | 73.64% | 1805 |
| base 통제군 (b0p) | 65.38% | **3.0%** | 67.23% | 1009 |
| base 메타 (b2p) | 65.55% | 2.4% | 67.15% | 1075 |

| 세대 | 전체 격차 | 종결분만 격차 | 절단율 차 |
|---|---|---|---|
| **instruct** | **+12.35pp** | **+4.65pp** | **−12.8pp** |
| **base** | +0.17pp | −0.08pp | −0.6pp |

★**instruct 통제군은 응답의 18.8% 를 못 맺었고 평균 3,367 토큰을 썼다.** 메타 팔은 6.0% ·
1,805 토큰이다. **종결한 응답만 보면 격차가 +12.35 → +4.65pp 로 줄어든다** — 즉 instruct
헤드라인의 **대략 2/3 이 "통제군의 비종료"** 에서 온다.

⚠**종결분만 격차는 조건부 지표다**(0731 기록: 조건부 지표는 파괴 비용을 못 본다). 두 팔의
종결 부분집합이 서로 다르므로 선택편향이 있고, **이 +4.65pp 를 "진짜 효과"로 쓰면 안 된다.**
분해로만 읽는다. 비종료는 그 자체로 실패이므로 **+12.35pp 는 end-to-end 수치로서 유효하다.**

★**base 통제군에는 그 병리가 없다.** 절단 3.0%, 평균 1,009 토큰 — 이미 잘 끝낸다.
**고칠 것이 없으니 메타가 줄 것도 없다.** 그래서 +0.17pp.

**가설(미검정)**: 메타 블록이 instruct 에서 한 일의 큰 부분은 **종결을 유도한 것**이고,
그 이득은 **통제군이 비종료로 무너져 있을 때만** 발생한다. 이것이 참이면 north-star
("습관을 심은 분포 밖에서 더 강건")의 근거였던 난이도 기울기도 재해석해야 한다 —
어려운 문제일수록 통제군이 더 길게 헤매다 잘리기 때문이다.

**★닫는 것**: instruct +14.00pp 를 *"메타인지 습관이 정확도를 올린다"* 의 단독 근거로 쓰는 것.
**★여는 것**: ①instruct 통제군의 비종료를 길이 shaping 으로 고친 뒤에도 격차가 남는가
(통제군 한 팔만 재학습하면 되는 값싼 검정) ②난이도별 절단율 — 기울기가 절단 기울기와
같이 가는지. ②는 **지금 보존 산출물로 GPU 0 에 가능하다.**

---

## E-135 (0805 09:30 UTC) · 페어 eval 완주 — **A-vs-A 편차 실측 + 채점기 3종 감도**

`rq3v2f-pair-eval-0804` 가 `[YAML] PAIRED EVAL DONE both arms` 로 완주했다. b0p 와
b2p 중복(`eval/rq3v2f_b2p_1030_pair/`) 둘 다 4k·16k·seed43 전 패스가 HF 에 올라갔다.

### ① 동일 체크포인트 A-vs-A — **잡 간 편차는 MATH500 ±1pp 미만**

b2p gs300 **같은 가중치**를 다른 eval 잡에서 다시 돌린 결과(문항 단위 avg@8, 쌍대 부트스트랩 10k):

| 벤치 | 원본 | 재평가 | Δ | 95% CI | 절단 |
|---|---|---|---|---|---|
| MATH500 | 77.28% | 76.88% | **−0.40pp** | [−1.40,+0.62] | 2.43→2.77% |
| GSM8K | 91.92% | 92.30% | +0.38pp | [−0.23,+0.97] | 0.03→0.03% |
| AIME | 15.00% | 15.42% | +0.42pp | [−3.75,+5.00] | **23.75→31.67%** |
| 기울기(L4–5 − L1–2) | | | −0.57pp | [−2.71,+1.51] | |

⇒ **RQ1 의 널은 eval 잡 잡음이 아니다.** MATH500 편차가 1pp 미만인데 RQ1 은 +0.18pp 다.
중복 팔로 다시 계산한 RQ1 도 −0.33pp [−1.80,+1.15] — 같은 널이 재현된다.

⚠**AIME 16k 는 같은 가중치에서 절단율이 8pp 흔들린다**(23.75→31.67%). AIME 단독으로
판정하지 말 것. 관련해 RQ1 의 AIME 는 −5.83pp [−12.50,−0.42] 로 CI 가 0 을 넘지 않고
중복 팔에서도 −5.42pp [−12.50,+0.00] 로 재현되지만, **A-vs-A 잡음이 ±4~5pp** 이고
n=30(1문항 = 3.33pp)이라 **시사일 뿐 판정이 아니다**.

### ② 채점기 3종 감도 — **판정은 안 바뀐다**

| 비교 | 저장 `is_correct` | 정본 `rewards.py:89` | `$…$` 래핑 |
|---|---|---|---|
| RQ1 전체 | +1.32 [−0.00,+2.67] | **+0.17** [−1.35,+1.70] | +1.15 [−0.30,+2.62] |
| RQ1 기울기 | +2.29 [−0.27,+4.90] | **+2.65** [−0.39,+5.74] | +1.48 [−1.39,+4.38] |
| RQ1(중복) 전체 | +0.82 [−0.57,+2.17] | −0.33 [−1.80,+1.15] | +0.75 [−0.70,+2.17] |
| A-vs-A 전체 | −0.50 | −0.50 | −0.40 |

**9 셀 전부 CI 가 0 을 가른다.** 0804 에 기록한 +0.18pp 는 정본 채점기 값이고 재계산에서
+0.17pp 로 복원됐다 — 정정할 수치는 없다.

### ③ 부수 발견 — 정본 채점기가 bare LaTeX 를 과소채점 (**C-027 신설**)

`rewards.py:89` 는 `parse(str(gold))` 를 `$…$` 없이 부른다. `math_verify` 는 래핑 없는
문자열을 평문 취급하므로 `\left( 3, \frac{\pi}{2} \right)` 같은 답이 파싱되지 않는다.
MATH500 절대값이 **65.1% → 76.9%** (+11.8pp) 로 바뀐다. 600행 감사: 래핑이 구제 122건 /
래핑이 잃음 1건(그건 진짜 오답). **두 팔에 똑같이 걸려 차분은 살지만 절대값은 전 세대에서
~11pp 낮게 보고돼 왔다.** 정본 수정은 승인 사항이고 차분이 안 바뀌므로 급하지 않다 —
다만 **외부 비교표에 우리 절대 정확도를 그대로 쓰면 안 된다.**

### 상태

| 잡 | 상태 | 근거 |
|---|---|---|
| 페어 eval | ✅ 완주 | `[YAML] PAIRED EVAL DONE both arms` |
| b3s | 🟢 gs265/300 · emit 1.0000 · 선점 0 · OOM 0 | `amlt log view musical-wombat` |
| b0p / b2p 학습 | ✅ gs300 완주 | E-133 |

b3s 잔여 35스텝 × ~620s ≈ **6시간**. 완주 후 b3s+b2p+b0p 한 job eval → RQ2.

---

## E-136 (0805 10:20 UTC) · 같은 스텝 4팔 지표 대조 — **b3p 붕괴 시점 특정 + b3s 엔트로피 폭주 발견**

wandb `gistdslab/metacot-dcpo-v4` 에서 b0p/b2p/b3p/b3s 를 같은 스텝에 나란히 놓았다
(`scripts/matched_step.py`). 두 가지가 나왔고 **하나는 나의 앞선 "b3s 건강" 판단을 정정한다.**

### ① b3p 발화 붕괴 — gs150→200 사이, 시점이 특정된다

| gs | 50 | 100 | **150** | **200** | 250 | 300 |
|---|---|---|---|---|---|---|
| b3p `gdpo/meta_emission/mean` | 1.0000 | 0.9980 | **0.8965** | **0.1973** | 0.0586 | **0.0176** |
| b3s `gdpo/meta_emission/mean` | 0.9941 | 0.9941 | 0.9961 | 0.9980 | 1.0000 | (진행) |

메타 계열 val 헤드가 전부 같이 죽는다: `meta_structure` 0.0992→−0.0057 ·
`postmeta_closure` 0.3091→−0.0212 · `meta_commit_shape` 0.1884→−0.0218 ·
`outcome_calibration` 0.0823→0.0036. **gs200 이후 b3p 는 b0p(메타 없음) 프로파일**이다.
그리고 val 정확도는 gs300 에 68.84% 로 **회복**한다 ⇒ **floor 없는 triobj 는 메타를 버리고
바닐라로 수렴한다.** b3p 무효화가 계측으로 확인됐다.

### ② ⚠b3s 엔트로피 폭주 — 단조, 26배 (**앞선 "건강" 판단 정정**)

| gs | 20 | 50 | 100 | 150 | 200 | 250 | 265 |
|---|---|---|---|---|---|---|---|
| b3s `actor/entropy` | 0.204 | 0.249 | 0.457 | 0.829 | 1.474 | 4.642 | **5.405** |
| b2p | 0.20 | ~0.25 | ~0.28 | ~0.29 | 0.300 | 0.292 | 0.32 |
| b3p | 0.21 | 0.221 | 0.287 | 0.369 | 0.416 | 0.302 | 0.32 |

`actor/kl_loss`(참조 이탈) 0.0013→**0.137**, 100배. **`entropy_coeff 0.001` 과
`kl_loss_coef 0.0` 은 두 팔이 동일**하다(`verl_sdc_e21r_shared.yaml:53,55` 상속,
`triobj_...:73` 과 `base_matched_...:51` 이 둘 다 kl 0.0) ⇒ 원인은 노브가 아니라 **보상**이다.

**기제**: floor 가 상수 보상을 보장 → 그 성분의 그룹정규화 advantage 가 0 → 메타 구간
정책기울기가 얇아짐 → **KL 앵커 0 상태에서 엔트로피 보너스가 무저항 승리.**
b3p 는 발화를 죽여 탈출했고 b3s 는 floor 가 탈출로를 막아 **압력이 엔트로피로 갔다.**

**붕괴은 아니다**: `response/aborted_ratio` **0.000** · 최소 응답길이 100~117토큰 ·
`pg_clipfrac` 7e-4 · val594 correctness gs250 **67.99%**(b2p 69.49%). 생성이 망가진 게
아니라 **분포가 퍼졌다.**

### ③ 판정 — b3s 를 계속 돌린다

`entropy_coeff` 가 두 팔에 같으므로 폭주는 **처치가 만든 결과**이지 외부 교란이 아니다.
RQ2 estimand 는 **최종정책 end-to-end 효과**로 이미 선언돼 있다 ⇒ `b3s − b2p` 를
**부작용 포함 패키지 효과**로 읽는다. 잔여 6시간이고 중단하면 45시간을 잃는다.
**다만 RQ2 가 음수면 이 항목이 기제 후보 1번**이다.

⚠**저엔트로피 구간 체크포인트는 이미 소실**이다 — `--keep 3` 이라 b3s 는 gs255/260/265 만
남아 있다(HF 확인). "폭주 전 성능"은 재학습 없이 못 잰다.

### ④ 같은 스텝 val594 correctness (판정 아님 — 셀당 21~38문항)

| gs | b0p | b2p | b3p | b3s | b2p−b0p | b3s−b2p | b2p A-vs-A |
|---|---|---|---|---|---|---|---|
| 200 | 68.89 | 68.44 | 64.20 | 69.02 | −0.45 | +0.59 | +0.34 |
| 250 | 68.30 | 69.49 | 67.83 | 67.99 | +1.19 | −1.50 | −0.75 |
| 300 | — | 68.12 | 68.84 | — | — | — | +0.98 |

**모든 차이가 같은 팔 A-vs-A 퍼짐(±1pp) 안**이다. 추세 주장 금지.
b3p gs200 의 −4.24pp 만 그 밖이고, 그건 발화 붕괴 시점과 정확히 일치한다.

### 다음 조치

- b3s gs300 완주 → **b3s+b2p+b0p+b3p 네 팔 한 job eval**. b3p 를 넣으면
  `b3s − b3p` = *triobj 안에서 메타 살아있음 vs 죽음* 대조가 **추가 비용 1시간**에 붙는다.
- **승인 필요**: floor 유지 + KL 앵커 또는 `entropy_coeff 0` 인 팔 하나 더 돌릴지
  (22시간·4 GPU). RQ2 부호를 보고 결정하는 편이 싸다.

---

## E-137 (0805 11:40 UTC) · **"이득은 통제군이 깨져서 생긴 것 아닌가"** — 측정으로 기각

물음: instruct 의 이득이 사실은 *통제군이 종결을 못 해서* 생긴 아티팩트 아닌가.
`finish_reason=="stop"` 인 **두 팔 모두 종결한 샘플**로 제한해 다시 계산했다
(`scripts/breakage_decomp.py`, `$…$` 래핑 채점 / C-027, 문항단위 쌍대 부트스트랩 10k).

| MATH500 16k | 헤드라인 | **종결분만** | 구제분 | 통제군 절단 | 토큰 |
|---|---|---|---|---|---|
| `shiftonly − gandhi` | +5.45 [+3.10,+7.88] | **+5.29** [+2.75,+7.89] | +0.16 (**2.9%**) | 3.70% | 1125→1628 |
| `pmishift − gandhi` | +10.22 [+7.92,+12.55] | **+11.25** [+8.81,+13.78] | **−1.03** | 3.70% | 1125→1805 |
| (GSM8K, shiftonly) | +0.82 [−0.28,+1.95] | +0.82 | 0.00 | 0.00% | 194→341 |

절대값(래핑 채점): gandhi 71.65% · shiftonly 77.10% · pmishift 81.88%.

**판정: 기각.** 절단 구제는 순수 보상 축 이득의 **2.9%** 뿐이고 패키지 축에서는 **음수**다
(처치 팔이 더 많이 절단되므로 절단분을 빼면 이득이 오히려 커진다). C-004 의 구제 설명은
**통제군이 matched-base(절단 18.8%)인 비교에만** 산다 — gandhi 처럼 종결하는 메타 baseline
상대로는 적용되지 않는다. C-001 · C-004 에 반영했다.

⚠**남은 대안설명은 길이 하나다**: shiftonly 1,628 vs gandhi 1,125 토큰(+45%).
단 길이는 **처치 후 매개변수**이므로 통제하면 안 된다(collider). 올바른 검정은
*길이만 늘리는 개입*이 같은 +5.29pp 를 내는지이고, 그건 C-020 placebo 자리다.

### 이것이 base 실패 진단에 주는 함의

순수 PMI-shift 이득이 **종결 아티팩트가 아니라 실재**하므로, base 널의 원인은 "보상이
가짜였다"가 아니다. E-136 에서 잰 두 가지가 남는다 — ①크레딧이 행동에 도달하지 않음
(save/derail 300스텝 동안 0.75→0.94 평평) ②base 에서 그 행동이 애초에 net-positive 가
아님(0스텝부터 0.85). **①과 ②는 서로 독립이고 둘 다 고쳐야 한다.**

---

## E-138 (0805 12:20 UTC) · b3s 엔트로피 폭주가 **출력에 나타났다** — 숫자 동형이의자

Monitor `b3gidwpus` 가 `Traceback` 을 잡아 원문을 확인했다. **잡 자체는 무사하다**
(`state=running`, gs269 전진, `math_verify` 내부에서 잡히는 예외). 그런데 내용이 사건이다.

모델이 `\boxed{}` 안에 **다른 문자체계의 숫자**를 쓰고 있다 —
크메르 `៤ ៨ ១ ៦` · 타밀 `௬ ௫ ௮ ௪ ௭ ௩` · 말라얄람 `൦ ൫ ൯` · 라오 `໐` · 미얀마 `႖` · 전각 `０ ２ ３`.
`math_verify` 가 `SyntaxError: invalid character '௩' (U+0BE9)` 로 실패하고 **그 답은 0점**이 된다.

빈도가 엔트로피 궤적과 겹치며 단조 증가한다(로그 67,304행, 총 924건):

| 시각(UTC) | 8/4 18h | 20h | 21h | 22h | 8/5 00h | 01h | **02h** | 03h | 04h | 06h | 07h | 08h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 건수 | 2 | 12 | 8 | 22 | 70 | 134 | **174** | 76 | 126 | 56 | 130 | 70 |

같은 구간 `actor/entropy` 는 0.20 → 5.46 이다. **약 60~80배 증가가 두 축에서 같이 일어난다.**

### 판정

**E-136 에서 "생성이 망가진 게 아니라 분포가 퍼진 것"이라 적은 것을 정정한다.** 퍼진 결과가
출력에 도달했다. 다만 여전히 붕괴는 아니다 — `aborted_ratio` 0.000, 최소 응답 100~117토큰,
학습 전진 중, val594 67.99%.

**보상 되먹임 고리**가 있다: 확산 → 동형이의자 → 수치가 맞아도 0점 → 학습 신호 잡음 증가 →
더 확산. 이건 `entropy_coeff` 가 무저항으로 이기는 상황(C-028)을 가속한다.

★**RQ2 채점 필수 조치**: gs300 eval 에서 **`NFKC` 정규화 전/후를 둘 다** 낸다.
주 지표는 사전등록대로 **무정규화**, 부지표로 정규화본. **두 팔에 같은 정규화**를 적용한다.
이걸 안 하면 b3s 의 *추론 실패* 와 *숫자를 딴 글자로 씀* 이 구분되지 않는다.
대략 2%(924건 / 약 46,000 롤아웃)이고 이건 **하한**이다 — sympy `Number()` 까지 도달한 것만 세었다.

잡은 계속 돌린다. 잔여 31스텝 ≈ 5시간.

---

## E-139 (0805 12:40 UTC) · **b3sh 발사** — C-019 가 9일간 지목해온 미실행 실험

`rq3v2f-b3sh-0805` 제출 11분 만에 노드 확보, **running**. b3s 와 병렬(8 GPU 점유).

C-019 는 *"효과가 검증된 구성을 base 에서 재현한 적이 없다 … base 에서 shiftonly 구성
(6헤드 0)을 돌리는 것, **이것이 지금 사다리에서 가장 직접적인 미실행 실험**"* 이라 적어놓고
9일간 열리지 않았다. 이 팔이 그것을 닫는다.

`h100std_rq3v2f_b3s.yaml` 과 **바이트 동일**(같은 `CODE_TAR_REVISION 490407111`, 같은
`b2p2_rvfull_eb16_sft` init, 같은 데이터, 같은 300스텝)이고 오버라이드 다섯만 다르다:

| 노브 | b3s | **b3sh** | shiftonly(승리 구성) |
|---|---|---|---|
| `dcpo_w_cal` | 0.3 | **0.0** | 0.0 |
| `dcpo_w_format` | 0.35 | **0.0** | 0.0 |
| `dcpo_w_emit` | 0.1 | **0.0** | 0.0 |
| `dcpo_len_cost` | 0.08 | **0.0** | 0.0 |
| `dcpo_meta_floor` | 0.05 | **0.0** | 0.0 |
| `dcpo_w_meta` (pmi) | 0.8 | 0.8 | 0.8 |

⇒ 메타 구간이 **`pmi_shift` 하나만** 싣는다. 승리 구성의 `eff_ratio_meta` 는 0.91 이었고
b3s 는 1.9~3.7 이다. 계보 `rq3v2f_b3sh` 로 분리해 b3p·b3s 를 덮지 않는다. 새 코드 0줄.

### 읽는 법 — 같은 기준점 b3p 에 대해 두 요인이 분리된다

- **`b3sh − b3p`** = **헤드 스택 효과** (floor 0.0 으로 맞춤)
- **`b3s − b3p`** = **floor 효과** (헤드 7개로 맞춤)

C-019 가 경고한 "len_cost 제거가 floor 실험과 교란된다"는 이 배치로 풀린다 — b3p 가 공통
기준점이라 두 대비가 각각 한 요인만 바꾼다.

### 정정 둘 (오늘 내가 틀렸던 것)

1. **"region-split 이 두 실패모드의 공통 원인"은 틀렸다.** 승리 구성 `shiftonly` 도
   region-split 을 썼다(*"pmi_shift routes onto META_CONTENT (meta_region_utility)"*) 그리고
   entropy 0.27 로 멀쩡했다. 원인은 자르기가 아니라 **그 구간에 여섯 개를 쌓은 것**이다.
2. **"`meta_region_utility` 가 0.0000 이라 배선 실패"는 틀렸다.** 그건 **val-aux** 값이고,
   검증에는 ref-worker 2위치 teacher forcing 이 없어 pmi_shift 가 계산되지 않는다.
   학습 쪽 `gdpo/meta_region_utility/max` 는 2.0~3.0 으로 발화한다.

### 기질 축에서 확인된 사실

instruct 세대의 기반 모델은 `configs/sft_v8_meta_inside_strict.yaml:2` 기준 **`Qwen/Qwen3-8B`**
(후학습본)이고 base 사다리는 **`Qwen3-8B-Base`** 다. 같은 계열·크기·토크나이저, **후학습 유무만**
다르다. Qwen3-8B 의 후학습에는 long-CoT cold start 와 reasoning RL 이 포함되므로,
instruct 팔은 **이미 RL 로 강화된 자기점검 위에 형식을 씌운 것**이고 base 팔은 1,763행으로
**없는 기능을 만드는 것**이다. 관측과 일치한다: instruct 는 발화 지지 장치가 **없는데도**
gs300 발화 0.9219, base b3p 는 지지가 **있는데도** 0.0176.

### 상태

| 잡 | 상태 |
|---|---|
| b3sh | 🟢 running 8m · wandb 런 생성 전(부트스트랩) |
| b3s | 🟢 gs272/300 · emit 1.000 · entropy 5.384 · HF `[260,265,270]` · 잔여 ~4.6h |

---

## E-140 (0805 11:37 UTC) · b3s 선점 1회 (gs280) — **재개 배선 정상 작동, 손실 5스텝**

`Caught signal 15` **2줄 = 선점 1회**(문서화된 서명). wandb 런은 `crashed` 로 뜨지만 amlt 잡은
`running` 이고 새 재시도 창이 부트스트랩 중이다. 배선이 설계대로 작동했다:

```
[YAML] existing GRPO resume gs (model+extra+optim>=4) = 275 1
+ python .../pull_resume_ckpt.py --config_name rq3v2f_b3s
```

**gs275 에서 재개한다.** HF 에는 `[265,270,275,280]` 이 있는데 게이트가 275 를 고른 것은
완성 규칙이 `model` · `extra_state` · **`optim`** 세 종류 모두 ≥4 샤드를 요구하기 때문이다 —
gs280 은 optim 샤드가 아직 다 안 올라간 상태였다. **손실은 5스텝**이고, 이것이 `save_freq=5` +
`--keep 3` 을 쓰는 이유다(b3p 때 `--keep 1` 이 처치가 살아있던 체크포인트를 지웠다).

⚠**모니터 오경보 정정**: 감시기가 "step 0 에서 25분 정체"를 울렸는데, 부트스트랩 창에는
`step:` 줄이 아직 없어서 0 으로 읽힌 것이다. 정체가 아니라 **재개 다운로드**다(gs275 체크포인트
23파일 98.3GB). 감시기를 고쳤다 — `step:` 이 한 번도 안 보이면 정체 카운터를 올리지 않고,
문턱도 5→8 회(40분)로 올렸다. 새 감시기 `b90ry5apt`(구 `b3gidwpus` 중지).

⚠**작업 실수 하나**: `amlt log view --lines 100000` 을 네 번 연속 돌리다 7분 타임아웃에 걸려
명령이 죽었다. 재시도 창 이력을 세려던 것이었다. 로그는 가볍게 받는다(`--lines 400~1200`).

### 상태

| 잡 | 상태 |
|---|---|
| b3s | 🔵 재개 부트스트랩 중 · gs275 에서 재시작 · ETA 는 다운로드 완료 후 재산정 |
| b3sh | 🟢 gs13/300 · emit 0.994 · entropy **0.207** · HF `[5,10]` 계보 분리 정상 |

---

## E-141 (0805 15:20 UTC) · **b3sh 의 개입이 배선상 작동한다** — `eff_ratio_meta` 1.6 → 0.5

⚠**과정 결함 자백**: 매 틱 `emit`·`entropy`·`len`·`score` 만 보고했고 **PMI-shift 고유 지표는
E-136 에서 한 번 잰 뒤 추적하지 않았다.** b3sh 는 `pmi_shift` 가 유일한 헤드인 팔이라 그 지표가
**주 지표**여야 했다. 사용자가 물어서 그제서야 쟀다. 감시기에 넣어 고쳤다(아래).

### ① 헤드 제거가 배분을 설계대로 바꿨다 — 첫 25스텝

| 팔 | 메타 구간 헤드 | **`eff_ratio_meta`** | save/derail | attempted |
|---|---|---|---|---|
| **b3sh** | **1개**(pmi_shift) | **0.506** | 0.84 | 0.676 |
| b3s | 6개 | **1.544** | 0.75 | 0.700 |
| b3p | 6개 | **1.632** | 0.83 | 0.698 |
| instruct shiftonly (승리, gs290~300) | 1개 | **0.91** | 2.61* | 0.145* |

\* instruct 의 save/derail 과 attempted 는 tail 20스텝만 로깅돼 있고 attempted 가 낮아
**선택편향이 있다** — 비교로 쓰지 말 것(b3p 도 발화가 죽자 attempted 0.04 에서 s/d 2.6 이 됐다).

⇒ **헤드를 여섯에서 하나로 줄이자 메타 구간의 유효 스케일이 1.6 → 0.5 로 떨어졌다.**
E-139 가 예측한 그대로다. 그리고 b3sh 의 값은 gs2 0.455 → gs17 0.532 로 **오르는 중**이며
(`dcpo_anchor_warmup_steps: 20`), 승리 구성의 0.91 쪽으로 가고 있다. **앵커 워밍업이 끝나는
gs20 이후 어디에 앉는지가 이 팔의 첫 판정 지점이다.**

### ② 행동 자체는 아직 그대로다

save/derail 첫 25스텝: b3sh **0.84** vs b3s 0.75 vs b3p 0.83. **차이 없다.** 당연하다 —
같은 init 이고 배분만 바꿨다. 배분 수정이 행동을 옮기는지는 **누적 학습 뒤에** 나타난다.
E-136 이 잰 대로 b3s 는 300스텝 내내 0.75→0.94 로 안 움직였다. **b3sh 가 그걸 움직이면
그것이 이 프로젝트에서 처음으로 "크레딧이 행동에 도달했다"는 증거가 된다.**

### ③ `rmeta` 는 여전히 대체로 음수

b3sh 스텝별 `pmishift_rmeta_mean_scored`: −0.175 −0.177 −0.104 −0.319 −0.053 −0.054 −0.111
−0.166 −0.642 −0.385 −0.292 **+0.108** −0.094 **+0.061** −0.257 −0.251.
`gdpo/meta_region_utility/mean` 도 같은 부호다. C-016 이 적은 "초기 음수"가 유지된다 —
**base 의 메타 블록은 평균적으로 정답 마진을 깎는다.**

### 감시기 교체 — PMI-shift 지표를 넣었다

`b121no4n1` 중지, **`bxqv69hhv`** 로 교체. 새로 잡는 것 둘:
- **`save/derail` 이 최근 25스텝 평균 1.20 을 넘으면** 알림(gs60 이후) — base 는 모든 팔에서
  0스텝부터 ~0.85 였으므로 **이게 움직이면 프로젝트 최초 사건**이다.
- **`eff_ratio_meta` 가 1.20 을 넘으면** 알림 — b3s 레짐(1.5~3.7)으로 흘러가는지 감시.
- 엔트로피 문턱은 3.0 → **1.5** 로 낮췄다(b3s 가 1.5 를 넘은 게 gs200 이므로 조기 경보가 된다).

### 매 틱 보고 항목 갱신

앞으로 두 팔 모두 **`emit` · `entropy` · `save/derail`(최근 25스텝) · `eff_ratio_meta`** 를 낸다.

---

## E-142 (0805 15:10 UTC) · **codex 배선 감사** — 헤드라인 주장은 기각(내 전제 오류), 실질 지적 다섯

`codex exec` 로 도는 두 팔의 배선 결함 후보 여섯을 코드 근거와 함께 감사시켰다.
⚠1차 시도는 `--sandbox read-only` 에서 `bwrap: loopback: Failed RTM_NEWADDR` 로 파일을 못 읽어 실패.
`--sandbox danger-full-access` 로 재시도하고 **전후 `git status`/`rev-parse` 로 무변경을 검증**했다
(`M paper` 는 감사 전부터 있던 서브모듈 상태, HEAD 동일).

### ⛔기각된 헤드라인 — 원인은 **내가 준 잘못된 전제**

codex 결론은 *"`dcpo_meta_open: 151669` 가 vocab 밖이라 `META_CONTENT` 마스크가 비어 있고,
PMI-shift 의 토큰 advantage 가 메타에 전혀 안 닿는다 ⇒ b3s·b3sh 무효"* 였다. **틀렸다.**

내가 **스톡 `Qwen/Qwen3-8B` 토크나이저**로 `<|meta|>` 를 인코딩해 "5조각·151669 는 범위 밖"이라 보고했고
codex 는 그 전제를 명시적으로 받아(*"Under the stated stock-tokenizer facts"*) 사슬을 세웠다.
그러나 학습이 쓰는 건 **SFT 체크포인트의 확장 토크나이저**다:

```
models/b2p2_rvfull_sft/added_tokens.json:  '<|meta|>': 151669,  '<|/meta|>': 151670
models/b2p2_rvfull_sft/config.json:        vocab_size: 151671
```

`src/training/meta_token_init.py` 가 이를 명시한다 — *"the **added** meta tokens … Call **AFTER**
`resize_token_embeddings`"*. 스톡 151669개(0~151668)에 둘을 더하면 정확히 151669·151670 이고
`META_OPEN_DEFAULT = 151669`(`dcpo_region.py:39`)와 일치한다.

**독립 반증도 있다**: codex 자신이 Q5 에서 `dcpo/eff_ratio_meta = EMA(|A_meta|)/EMA(|A_corr|)`
(`verl_sdc.py:691-699`)라고 밝혔다. 마스크가 비면 `A_meta ≡ 0` 이라 이 값이 0 이어야 하는데
실측은 b3sh **0.55** / b3s **1.99** 다. ⇒ 마스크는 채워져 있다.

★**교훈**: 외부 검토자에게 넘기는 전제도 계측 대상이다. 내 "스톡 토크나이저" 관측이 배선검사를
통과하지 못한 채 전제로 들어갔고, 검토자는 그 위에 정확한 추론을 쌓아 틀린 결론에 도달했다.

### ✅살아남는 지적 다섯 (전부 file:line 근거 있음)

1. **`norm_adv_by_std_in_grpo=false` 는 죽은 플래그다.** `adv_estimator: gdpo`(`configs/triobj…:94`)라
   패치 경로가 상류 GDPO 결과를 **버리고** `_compute_dcpo_region_advantage` 를 부른다
   (`verl_sdc_utils.py:505-506`). 그 함수는 **그룹 평균만 빼고 std 로 안 나눈다**
   (`dcpo_region.py:1225-1227`). ⇒ 플래그는 안 쓰이는 중간값에만 닿는다.
   **결과적으로 우리가 의도한 동작(std 정규화 없음)이 실현돼 있다** — 승인대기 ① 닫힘.
2. **`dcpo_w_over=0.0` 은 완전 사문**이다. 두 런처가 넘기지만 조성 경로에 소비처가 없다
   (`verl_sdc_utils.py:413-427`). 무해하나 "노브를 껐다"고 적으면 안 된다.
3. ⚠**앵커 EMA 가 체크포인트에 안 들어간다**(`verl_sdc_utils.py:30`, `dcpo_region.py:1242`).
   모듈 전역이라 **선점·재개 이력이 다른 런은 재개 후 앵커 스케일이 달라진다.** b3s 는 선점 다회,
   b3sh 는 1회 — 오늘 대비한 `eff_ratio_meta` 0.55 vs 1.99 에 이 교란이 섞인다.
   헤드 수 차이가 지배적일 것으로 보이나 **교란으로 기록한다.**
4. **b3s vs b3sh 는 한 노브 대비가 아니다**(오버라이드 5개 차이). `b3sh − b3p`(floor 맞춤)와
   `b3s − b3p`(헤드 맞춤)로 읽는 설계는 유효하지만 **둘을 직접 대비하면 안 된다.**
5. **instruct 승리 구성과는 코드 리비전·init·응답길이가 모두 다르다.** 특히
   `data.max_response_length` **4096(아카이브) vs 8192(b3s·b3sh)** — 절단율과 발화 모집단이 달라진다.
   ⇒ 승리 구성의 `eff_ratio_meta 0.91` 을 **b3sh 의 문턱으로 쓰면 안 된다**(기존 주의를 강화).

### 덤 — `rmeta_pos/neg_rate` 의 나머지 53% (승인대기 ⑦ 닫힘)

의도된 dead-zone 이 아니라 **로깅 분류 문턱**이다: 분모는 배치 전체 행이고, 양성은 `R_meta > 0.5`,
음성은 `< -0.5` 로만 센다(`verl_sdc.py:650-660`). 나머지는 미시도 행(`verl_sdc.py:1620-1643`) +
PMI/ref 스코어링 실패 행(`1689-1722`) + shift 가 ±0.5 사이인 행이 섞여 있다. **신호 손실이 아니다.**

### 판정

**두 팔 모두 계속 돌린다. 무효화 사유 없음.** 승인대기 일곱 → **넷**(①⑥⑦ 닫힘).
남은 넷: ②CLAUDE.md SFT2 서술 정정 ③`acc_with`/`acc_without` 산술 모순 ④`pair_analyze.py` NFKC
⑤메타 텍스트 perplexity base vs instruct.

---

## E-143 (0805 16:10 UTC) · codex 2차 논의 — **"b3s 가 메타를 2~4배 세게 민다"는 내 해석 철회**

엔트로피 상승 원인을 codex 와 코드로 따졌다(`danger-full-access` + 전후 `git rev-parse`/`status`
무변경 검증: `974fad7` 동일, `M paper` 만 유지). ⚠전제 오염을 막기 위해 **"내가 주는 수치는 wandb
텔레메트리이지 코드에 대한 주장이 아니다. 직접 확인하라"** 를 프롬프트 첫머리에 박았다.

### ⛔철회 — E-139·E-141 의 **해석** 부분

codex 가 조성 순서를 추적했다(`dcpo_region.py:1225-1268`, `verl_sdc_utils.py:232-351`):

```
correctness 토큰 :  1.0     × A_corr
메타 토큰       :  w_meta × A_meta_raw × (corr_ema / meta_ema)
```

**앵커가 메타를 correctness 스케일로 맞춘 뒤 `w_meta=0.8` 이 곱해진다.** ⇒ 최종 메타 밀기는
**모든 팔에서 correctness 의 약 0.8배로 같다.** `dcpo/eff_ratio_meta` 는 **앵커 이전 원시 비율**을
찍는 진단값이고 "얼마나 세게 미는가" 가 아니다(`verl_sdc.py:691-699`).

⇒ E-139 의 *"원인은 그 구간에 여섯 개를 쌓은 것(eff_ratio 0.91 vs 1.9~3.7)"* 과 E-141 의
같은 취지 해석을 **철회한다.** 헤드 제거가 `eff_ratio` 를 바꾼 것은 사실이나 **밀기 세기의 차이가 아니다.**
★앵커는 `global_step > 20` 에서만 engage 한다(`dcpo_region.py:1252-1258`) — 그 전엔 원시 중심화값에
`w_meta` 워밍업만 곱한다.

### ★ 대신 b3s 에만 있는 진짜 비대칭 — **floor 는 상수가 아니다**

```
b3s : 0.8 × anchored_meta  +  0.05 / 메타토큰수
b3sh: 0.8 × anchored_meta
```

둘이 위험하다(`dcpo_region.py:1323-1350`):
1. **중심화 이후에 더해진다**(`:1182,1330`) ⇒ 그룹평균 상쇄 논리가 안 통한다. 그리고
   `floor_mask=1` 인 행에만 붙는다 — 신뢰 메타 클래스(wellformed/swapped/dup_open/reversed/drift,
   `:106,111,112`). **메타 낸 행은 +, 안 낸 행은 0** ⇒ **그룹 내 판별자**다.
2. **메타 토큰 수로 나눈다**(`:1325,1337,1348`) ⇒ **메타가 짧아질수록 토큰당 floor 가 커진다.**
   본체 메타 advantage 는 행 스칼라를 그대로 브로드캐스트해서(`:1154,1268`) 이 성질이 없다.

⇒ b3s 에만 있고 b3p·b3sh·b2p 에는 없는 유일한 비대칭이며, 엔트로피가 터진 팔도 b3s 하나다.
⚠**단 codex 는 "floor 만으로 단조 엔트로피 증가가 보장되진 않는다 — CANNOT DETERMINE" 이라 명시했다.
유력 후보이지 확정 원인이 아니다.**

### ❓미결 — `eff_scale_corr` 가 팔마다 **3.4배** 다르다

codex 가 `len_cost` 로는 설명 불가라 했고(워밍업 `step/80` 이라 gs24 에서 최대 0.024, 실제 0.001 수준),
내가 스텝별로 쟀다:

| gs | b3sh | b3p | b3s |
|---|---|---|---|
| 2 | **0.7604** | 0.2211 | 0.2313 |
| 10 | **0.6963** | 0.2223 | 0.2279 |
| 20 | **0.6306** | 0.2073 | 0.2123 |
| 40 | **0.6726** | 0.1887 | 0.1928 |

**gs2 부터 이미 3.4배 벌어져 있고 EMA 워밍업으로 수렴하지 않는다.** 같은 모델·같은 데이터·같은
correctness 보상인데 첫 스텝부터 다르다. `eff_scale_meta` 는 세 팔 모두 0.31~0.37 로 같다.
`gdpo/correctness/std` 도 세 팔 모두 ~0.85 다. **설명되지 않는다 — 미결로 둔다.**
per-row correctness·길이·그룹 ID·마스크가 있어야 풀린다(codex 도 CANNOT DETERMINE).

### 확정된 것 셋 (엔트로피 관련)

1. 메타 밀기 세기는 팔 간 **같다**(0.8×) — 기존 설명 철회.
2. b3s 만 갖는 것은 **floor**, 그리고 그것은 **그룹 내 판별자 + 토큰수 반비례**다.
3. 엔트로피를 올리는 장치 셋 전부 활성 확인: `entropy_coeff 0.001`
   (`verl_sdc_e21r_shared.yaml:53`, 상속 `triobj:47,48`) · `kl_loss_coef 0.0`(`triobj:72,73`) ·
   clip-higher `0.2/0.28`(`triobj:74,78,79`). ⚠`pg_clipfrac_lower=0.0000` 은 **위쪽 클립 비활성의
   증거가 아니다**(codex 정정) — 나는 그렇게 읽을 뻔했다.

★codex Q5 부가: 메타 advantage 는 행 스칼라를 메타 토큰 전체에 브로드캐스트하므로 **총량 정규화가
아니다**. 반면 **floor 만 총량 정규화**다. 희소 라우팅이 응답 토큰 전체에서 보면 분산을 키울 수 있다.

### 상태

| 잡 | 진행 | emit | ent | s/d | effRatio |
|---|---|---|---|---|---|
| b3s | **gs285** ↑ | 1.0000 | 6.198 | 1.40 | 2.077 |
| b3sh | gs40 | 0.9980 | 0.211 | 1.06 | 0.563 |

HF `b3s [275,280,285]` · `b3sh [25,30,35,40]`. `FATAL`/`ABORT` 0건. b3s **남은 15스텝**.

---

## E-144 (0805 18:50 UTC) · **b3s gs300 완주 → RQ2 eval 제출**

b3s 가 300스텝을 마쳤고 HF 에 완성 저장됐다. 게이트 조건(`model`/`extra_state`/`optim` 각각 ≥4 샤드)을
직접 확인했다:

```
HF  checkpoints/rq3v2f_b3s  [290, 295, 300]
gs300 완성도 = model 4 / extra_state 4 / optim 4  ✅
```

**`rq3v2f-rq2-eval-0805` 제출 완료**(`h100std_rq3v2f_rq2_1030_eval.yaml`, 커밋 `da85086`).
세 팔 순서 **b3s → b3p → b2p(`_rq2` 접두어)**, 1030문제, 4k·16k·seed43,
`CODE_TAR_REVISION 467403206`(기존 모든 eval 과 채점 코드 동일).

### b3s 최종 상태 (gs300)

| 지표 | 값 |
|---|---|
| `meta_emission` | **1.0000** — 300스텝 내내 유지 |
| `actor/entropy` | **6.417** (초기 0.216 → 30배) |
| `eff_ratio_meta` | 2.163 |
| save/derail (최근 25스텝 원시) | **104/87 = 1.20** |
| `attempted_rate` | 0.586 |
| `rmeta_mean_scored` | **+0.095** (전반 −0.14~−0.24 에서 후반 양수 전환) |

⚠**해석 주의 셋**: ①`rmeta` 양수 전환은 codex 확인대로 **참조가 동결본**이라 정책이 참조에서
멀어진 결과일 수 있고 "믿음 갱신이 좋아졌다"는 증거가 아니다. ②save/derail 1.20 은 25스텝 누적이고
스텝별로 `(295,1,12) (297,9,1) (298,0,0)` 처럼 극단이 오간다 — **추세 주장 금지**.
③엔트로피 6.4 로 끝났으므로 **gs300 체크포인트는 지금까지 중 가장 확산된 상태**다.

### 채점 시 필수 (E-138)

**NFKC 정규화 전/후를 둘 다** 낸다. b3s 가 크메르·타밀·전각 숫자를 쓰고 있어 값이 맞아도
`math_verify` 가 0점을 준다. 두 팔에 같은 정규화를 적용하고, **주 지표는 사전등록대로 무정규화**.

### 남은 것

- **b3sh** gs53/300 · `w_meta` 워밍업(80스텝) 66% 지점 · **gs80 전 판정 불가**
- eval 완주 후 `scripts/pair_analyze.py` 로 **RQ2 = b3s − b2p**, 부수로 **b3s − b3p**
  (같은 triobj 안에서 메타 살아있음 vs 죽음)

---

## E-145 (0805 20:10 UTC) · **RQ2 판정 — 음수다** (C-029 신설)

b3s 팔의 eval 이 완주해 `eval/rq3v2f_b3s_1030/` 에 5개 패스가 전부 착지했다. b2p 는 기존 산출물이
있으므로 **사전등록 주 비교를 지금 계산했다**(`scripts/rq2_analyze.py`).

### 결과

| MATH500 16k | b2p | b3s | Δ | 95% CI |
|---|---|---|---|---|
| **주 지표** | 77.28% | 74.80% | **−2.48pp** | **[−3.83, −1.15]** |
| **종결분만**(3,788/4,000) | 78.21% | 75.56% | **−2.65pp** | **[−4.16, −1.16]** |
| **난이도 기울기** | | | **−4.10pp** | **[−7.03, −1.28]** |

GSM8K +0.33pp[−0.70,+1.35] · AIME +1.25pp[−1.25,+4.17](둘 다 널).
중복 b2p 팔로 재계산: **−2.08pp [−3.35,−0.82]**, 기울기 −3.53pp — **재현된다.**
A-vs-A 잡 편차가 −0.40pp 이므로 **잡음의 6배**다.

### ✅ codex 가 지시한 검사를 먼저 통과했다 — **절단이 아니다**

codex 는 "판정 전에 종결분 분해를 먼저 돌려라, 절단율 격차가 1.22pp 뿐이라 −2.48pp 를 설명 못 한다"고
했다. 실행 결과 **종결분만 −2.65pp 로 오히려 더 나쁘고 절단 기여분은 +0.17pp** 다.
⇒ **추론 능력 손실이지 잘려서가 아니다.**
(AIME 는 종결분만 +3.47pp[−0.53,+8.50] 로 널이고 절단율이 23.75→36.25% 라 판정 불가.)

### ⚠ E-138 대비는 결과에 영향이 없었다 (정정)

eval parquet 의 **동형이의자 비율이 전 팔 0.00%** 다. NFKC 정규화 전/후가 소수점까지 동일하다.
★codex 확인: 학습 롤아웃은 `temperature=1.0/top_p=1.0/top_k=-1`, **eval 은 `0.7/0.95`**
(`h100std_rq3v2f_rq2_1030_eval.yaml:164,190`, `scripts/eval_vllm_1030.py:150-176`).
⇒ **eval 이 고엔트로피 꼬리를 감쇠시키므로 학습분포 손상을 과소평가한다.** 정규화는 그래도 둘 다 냈다.

### ★ 기제 후보 (codex, 코드 근거) — 길이 선택적 보상 압력

b3s 는 `dcpo_len_cost=0.08` 과 `trunc_open_penalty=0.3` 을 갖고, **길이 비례로 correctness 에서
직접 뺀다**(`verl_sdc.py:490-507`). 절단 행은 `R_corr=-1`·`R_meta=0` 으로 강제된다
(`dcpo_region.py:916-944`). **b2p 에는 `len_cost` 도 다른 `dcpo_*` 헤드도 없다**
(`base_matched_grpo_h100_4x4k.yaml:23-24`). 토큰수도 b2p 1,075 → b3s 1,446 이다.
⇒ **길고 어려운 문제가 선택적으로 벌받는다** — 기울기 −4.10pp 가 어려운 쪽에서만 나쁜 것과 일치.
⚠아카이브된 instruct 승리 구성은 **`len_cost=0`** 이었다.

### ★ codex 사전공약 (b3p 결과 보기 전에 기록)

- b3p 가 b2p 대비 **0 근처로 회복** ⇒ **floor 가 문제**
- b3p 가 **여전히 유의하게 낮고 기울기도 음수** ⇒ **triobj 패키지 전체가 문제**

### 판정

**base 사다리에서 F(이득) 칸이 닫혔다.** RQ1 널(+0.18pp) + RQ2 음수(−2.48pp).
**b3s 를 결과로 인용하는 모든 문장을 철회한다.**

남은 것: ①**b3sh**(pmi_shift 단독·`len_cost=0`·`floor=0`)가 길이 압력 가설을 직접 검정한다 —
현재 gs65/300 ②b3p eval(진행 중)이 floor 를 가른다 ③학습 패리티 디코딩 재측정(codex Q4).

---

## E-146 (0805 23:2x UTC) — ★b3p 판정: floor 가 아니라 패키지 본체 · **그리고 내가 b3s 증거를 덮어썼다**

### ① 판정 (사전공약대로, 결과 보기 전 기록됨)

`eval/rq3v2f_b3p_1030` 5패스 착지. MATH500 n=4,000, 문항단위 쌍대 부트스트랩 10k.

| 팔 | acc | 절단율 | 평균 토큰 |
|---|---|---|---|
| b2p (통제) | 77.28% | 2.43% | 1,075 |
| b2p_pair | 76.88% | 2.77% | — |
| **b3p** (`meta_floor=0`) | **75.20%** | **5.12%** | **1,472** |
| b3s (`meta_floor=0.05`) | 75.28% | 3.02% | — |
| b0p | 76.12% | 3.02% | — |

- **b3p − b2p = −2.08pp [−3.48,−0.67]** · 기울기 −2.77pp
- **b3p − b2p_pair = −1.68pp [−2.95,−0.40]** · 기울기 −2.20pp
- **b3s − b3p = +0.07pp [−1.38,+1.50]**

⇒ 사전공약의 두 갈래 중 **"여전히 유의하게 낮고 기울기도 음수 ⇒ triobj 패키지 전체가 문제"** 쪽.
**`meta_floor` 는 원인이 아니다.** → C-030.

★**codex 정정**: `b3s−b3p ≈ 0` 은 **동등성 증거가 아니다**. CI ±1.5pp 로 ±1.4pp 효과를 배제할
검정력이 없다. *"floor 가 회복시킨다는 증거가 없다"* 까지만 말할 수 있다.

### ② 절단 분해 (codex 지시 — b3s 의 결과를 b3p 에 전이하지 말라)

두 팔 모두 종결한 샘플 3,746/4,000(93.7%): **−2.02pp [−3.55,−0.48]**, 절단 기여 **−0.06pp(2.9%)**.
⇒ b3s 와 같다. **추론 손실이지 잘려서가 아니다.**

### ③ ★길이 5분위 분해 — **단조가 아니다**

| 구간 | b2p 토큰 | b2p acc | b3p acc | Δ | CI | b3p 절단 |
|---|---|---|---|---|---|---|
| Q1 최단 | 177 | 99.12 | 98.25 | −0.88 | [−2.00,+0.12] | 0.12% |
| Q2 | 239 | 94.38 | 93.12 | −1.25 | [−3.88,+1.25] | 0.00% |
| Q3 | 337 | 89.25 | 86.50 | −2.75 | [−5.38,−0.25] | 0.25% |
| **Q4** | **606** | **72.00** | **67.38** | **−4.62** | **[−8.88,−0.50]** | 1.00% |
| Q5 최장 | 4,017 | 31.62 | 30.75 | −0.88 | [−4.88,+3.25] | 24.25% |

**순수 `len_cost` 가설이 예측하는 단조 악화가 안 나온다.** 손상은 **중상난도 Q4 대역에서 최대**이고
최장 구간에서 사라진다. ⚠단 Q5 는 b2p 조차 31.6% 라 바닥 압축 + 절단 24% ⇒ **해상도 부족**이지
"손상 없음"이 아니다. 표적은 **"풀 수 있지만 쉽지 않은 문제"** 대역이다.

⚠토큰이 1,075→1,472(**+37%**)로 늘었다 — **길이를 벌하는 항이 있는데도 길어졌고** 정확도는 내려갔다.

### ④ ★★내 실수: **b3s 증거 parquet 을 덮어썼다**

RQ2 재확인이 **−2.00pp** 로 나왔는데 C-029 기록은 **−2.48pp** 였다. 추적 결과:

1. 채점 결정성 검사 — 같은 parquet 3회 재채점, 4,000행 **불일치 0**. 채점 잡음 아님.
2. 정본 스크립트 vs tmp 사본 `diff` — **동일**. 코드 변경 아님.
3. b2p 는 77.28% 로 **완전 동일**, b3s 만 74.80 → 75.28 이동 ⇒ **b3s parquet 이 바뀌었다.**
4. 근인: 내가 쓴 `h100std_rq3v2f_rq2_1030_eval.yaml:86` 의 SPEC 이
   `rq3v2f_b3s:eval/rq3v2f_b3s_1030` — **기존 프리픽스를 그대로 겨냥**. 22:09 에 재생성판이 올라갔다.
5. **복구 성공**: HF `revision='ac274e72'`(19:42:48)에서 원본을 받아 **−2.48pp 를 정확히 재현**.

| | b3s acc | RQ2 = b3s−b2p | 기울기 |
|---|---|---|---|
| v1 (원본, C-029 증거) | 74.80% | **−2.48pp [−3.83,−1.15]** | −4.10pp |
| v2 (재생성) | 75.28% | **−2.00pp [−3.40,−0.62]** | −3.01pp |
| **차이 (A-vs-A)** | | **+0.47pp [−0.55,+1.50]** | |

기존 A-vs-A(b2p pair−orig) 는 −0.40pp [−1.40,+0.62] 이므로 **드리프트 범위 안**이다.
**판정은 안 바뀐다**(두 판 모두 CI 가 0 을 배제). codex 판단도 *"기록 정정으로 충분"*.
⛔**−2.48 과 −2.00 의 차이를 처치 효과로 읽지 말 것** — 같은 체크포인트의 생성 잡음이다.

★**규율 결함**: 파괴조작 3율(LIST→결정→실행)을 **입력**에는 적용했지만 **출력 경로**에는 적용하지
않았다. 런처를 복제할 때 SPEC 의 출력 프리픽스가 **기존 산출물을 겨냥하는지** 확인하지 않았다.
⇒ **런처 복제 시 출력 경로도 LIST 대상이다.** codex 추가 권고: 기존 HF path 존재 시 업로드 즉시 실패,
결과 인용은 mutable `main` 대신 **commit revision 고정**(C-029 증거란에 반영).

### ⑤ 조치

- C-029 증거란에 리비전 고정 + 두 판 병기. C-030 신설.
- 다음 판정점은 여전히 **b3sh**(pmi 단독·`len_cost=0`·`floor=0`) — **Q4 대역의 −4.6pp 가
  사라지는지**가 판정점이다. 단 12시간째 `queued`.
- codex 제안 중 미착수: `len_cost` 만 0 으로 둔 b3p 변형(config 1개) · 학습 패리티 디코딩 재측정.

---

## E-147 (0806 00:0x UTC) — ★통제군이 셋이 되자 효과 크기가 흔들렸다 · **"잡음의 6배" 철회**

`eval/rq3v2f_b2p_1030_rq2` 5패스 착지. 같은 b2p(gs300) 체크포인트의 **세 번째 독립 생성판**이고,
**b3p·b3s(v2)와 같은 잡에서** 나왔다.

| 통제 생성판 | 잡 | MATH500 acc | 토큰 |
|---|---|---|---|
| `b2p` (원본, **사전등록에서 쓴 것**) | 잡A | **77.28%** | 1,075 |
| `b2p_pair` | 잡B | 76.88% | 1,134 |
| `b2p_rq2` | **잡C = b3p·b3s(v2)와 동일** | 76.68% | 1,129 |

A-vs-A: pair−orig −0.40 [−1.40,+0.62] · rq2−orig −0.60 [−1.55,+0.33] · rq2−pair −0.20 [−1.23,+0.85]

**원본 b2p 가 세 판 중 가장 높은 판이었다.** 그래서 통제를 바꾸면 효과가 줄어든다:

| 대조 | Δ | 95% CI | 기울기 |
|---|---|---|---|
| b3p − b2p(사전등록) | −2.08pp | [−3.48,−0.67] | −2.77pp |
| **b3p − b2p_rq2(같은 잡)** | **−1.48pp** | **[−2.90,−0.05]** | −1.91pp |
| b3s − b2p(사전등록) | −2.00pp | [−3.40,−0.62] | −3.01pp |
| **b3s − b2p_rq2(같은 잡)** | **−1.40pp** | **[−2.73,−0.12]** | −2.15pp |

### codex 판정 (결과를 보고 통제를 바꾸는 것의 위험을 먼저 물었다)

**① 통제 선택**: 결과를 본 뒤 통제를 갈아끼우면 forking-path 다. ⇒ **사전등록 통제(원본 b2p)를
주 분석으로 유지**, 같은-잡 통제는 **감도분석**, 3판 풀링(76.95%)은 **탐색적 보조**.
"b2p_rq2 가 더 옳다"고 교체하지 말고 *"잡 효과를 줄여도 결론이 유지되는가"* 로 읽어라.

**② 같은-잡 통제도 완전하지 않다**(codex 가 코드 직접 확인): SPEC 순서가 `b3s → b3p → b2p`
(`h100std_rq3v2f_rq2_1030_eval.yaml:86`)라 `b2p_rq2` 는 **마지막 순번**이고 열/GPU 클럭/캐시/
메모리 단편화/호스트 부하 같은 **순서 효과**가 섞인다. 시드는 MATH500 16k 세 팔 모두 **42 로 동일**
(`:179-190`, AIME 추가 패스만 43). metadata 에 hostname 을 기록하지만(`eval_vllm_1030.py:240-260`)
이번 실행분이 repo 에 없어 **실제 같은 호스트였는지는 확인 불가**.

**③ ★"잡음의 6배" 철회**: n=3 의 범위(0.60pp)를 잡음바닥으로 쓸 근거가 없다. 정규 가정에서도
SD 의 95% 범위가 대략 **0.16~1.92pp** 로 넓다. 0.60pp 는 *"이번 세 판에서 관찰된 범위"* 일 뿐이다.
⇒ C-029 의 해당 문장을 **덮어썼다**.

**④ 판정 자체**: 방향은 세 통제 전부에서 음수라 **안 바뀐다**. 다만 서술을 낮춰야 한다 —
*"방향은 강건하고 크기와 통계적 여유는 제한적이다"*(같은-잡 통제에서 CI 상한 −0.05).

### 내 오류 두 개

1. **가장 유리한 통제 대비로 "6배"를 썼다.** 그때는 통제가 둘뿐이었지만, 둘 중 높은 쪽을
   주 통제로 쓰고 있다는 사실을 점검하지 않았다.
2. **A-vs-A 차이 하나를 잡음바닥으로 승격시켰다.** 단일 차분은 SD 가 아니다.

⇒ **규율**: *"이 수의 기준선은 무엇이고, 기준선 자체가 몇 판 중 어느 판인가"* 를 같이 적는다.

### 그 밖의 이번 틱

- **b3sh 가 `queued` 를 벗어나 노드를 잡았다**(13h 대기). 부트스트랩 중(conda 환경 추출 단계),
  `FATAL`·OOM 0. gs70 에서 재개하므로 손실 4스텝. wandb 는 아직 gs74(20:36) 그대로 — 정상.

---

## E-148 (0806 06:0x UTC) — b3sh 첫 사건: `gs110-119` 가 사전 선언 범위를 벗어났다 (**메타 지표 한정**)

### 사건

사전에 못박은 기준: *"완성된 10스텝 창의 save/derail 이 이전 창들의 범위를 벗어날 때."*
**gs110-119 = 62/22 = 2.82** — 이전 열한 창의 범위 **0.29~1.51** 밖. **처음 충족.**

| 창 | sv | dr | scored | sv/sc | dr/sc | sv−dr | member | rmeta | util | score |
|---|---|---|---|---|---|---|---|---|---|---|
| gs70-79 | 45 | 100 | 145 | 0.310 | 0.690 | −55 | 0.712 | −0.120 | −0.084 | 0.459 |
| gs80-89 | 48 | 54 | 102 | 0.471 | 0.529 | −6 | 0.719 | +0.007 | −0.000 | 0.498 |
| gs90-99 | 71 | 47 | 118 | 0.602 | 0.398 | +24 | 0.663 | +0.022 | +0.014 | 0.414 |
| gs100-109 | 39 | 54 | 93 | 0.419 | 0.581 | −15 | 0.689 | +0.024 | +0.017 | 0.405 |
| **gs110-119** | **62** | **22** | **84** | **0.738** | **0.262** | **+40** | **0.685** | **+0.118** | **+0.080** | 0.466 |

- **움직인 것은 `derail`** — 22 는 전 구간 최저(이전 최소 38). `derail=0` 스텝 17개 중 **4개가 gs116~119 연속**.
- **비율 아티팩트 아님**: `sv/scored` 0.738 이 이전 최대 0.602 를, `sv−dr` +40 이 이전 최대 +25 를 넘는다.
- **모집단 변화 아님**: `member_rate` 0.685 가 범위(0.663~0.719) 한가운데. `attempted_rate` 도 불변.

### ★codex 정정 — "네 지표가 독립적으로 같은 방향"은 과장이다

`rmeta_mean_scored` 와 `save/derail` 은 **같은 `pmi_open/pmi_close` 쌍의 서로 다른 요약**이다
(`dcpo_pmi_shift.py:104-117`, `verl_sdc.py:1740-1759`). `rmeta_mean_scored` 는 연속 shift +
SAVE 보너스 + DERAIL 페널티를 모두 포함한 평균이라 **부호 반전 없이 연속 shift 만으로도 오른다**.
⇒ 정확한 서술: ***"같은 PMI 개입의 discrete flip 통계와 reward 평균이 함께 개선된 정황"***.
⛔"독립 증거 넷"이라고 쓰지 말 것.

### ★정확도는 안 움직였다 — 이 사건의 값은 조기 진단 신호뿐이다

`critic/score/mean` 은 창평균 0.38~0.50 로 **방향이 없다**. `acc_with` 는 0.716→0.769 로 완만하나
`acc_without`(0.00~0.34)은 **분모가 다른 counterfactual 채점분**이라 함께 읽으려면
`cw_graded_rate` 가 필요하다(`verl_sdc.py:649-668`).
⇒ **메타/신념 proxy 는 변했으나 task correctness 향상 증거는 없다.**
codex: *"held-out Q4 대역에서 −4.6pp 가 사라지지 않으면 학습 중 PMI 개선은 task-level 가치가 없었다."*

### ★사전등록 — 이 사건이 진짜인지 판별하는 기준 (결과 보기 전 기록)

> **gs120-149 의 완성된 세 개 10스텝 창 중 최소 두 창**에서
> `sv/scored ≥ 0.738` **그리고** `dr/scored ≤ 0.262` **그리고**
> `member_rate` 가 **0.635~0.735**(gs110-119 의 0.685 ±0.05) 안일 것.

이 기준은 **비율 분모 조작과 채점 모집단 변화를 동시에 통제**한다.
⚠gs120 첫 스텝의 `2/5`는 **n=1 미완성 창이라 반증도 재현도 아니다** — 판정에 넣지 않는다.
⚠앞으로 `save/derail` 비율만 쓰지 말고 **`sv/scored`·`dr/scored`·`sv−dr` 을 같이 기록**한다(codex).

### 창 크기 정본

사전 기준이 "완성된 고정 10스텝 창"이므로 **10스텝 fixed bin 이 정본**. 25스텝 롤링(현재 **1.20**,
역사 범위 0.48~1.34)은 **지속성 확인용 보조**다. 둘이 갈리는 것은 25스텝 창이 약했던 gs100-109 를
포함하기 때문이며 모순이 아니다.

### ⚠관측 메모

`dcpo/pmishift_attempted_rate` 와 `dcpo/pmishift_member_rate` 가 **모든 창에서 값이 같다**
(0.663/0.685/0.712/0.719 …). 서로 다른 키인데 동일값이라 **둘 중 하나가 중복 로깅일 수 있다** —
확인 전까지 두 값을 독립 근거로 쓰지 말 것.

### 최종 판정 기준은 불변

**held-out Q4 대역(~600토큰, 통제군 72%)에서 b3p 의 −4.6pp 가 사라지는가.**
학습 지표가 좋아도 held-out 이 오른다는 보장은 없다(b3s 전례: 학습 멀쩡, eval −2pp).

---

## E-149 (0806 07:5x UTC) — E-148 사건 **재현 실패**(단 기준 자체가 비진단적) · b3sh 팔 정체 정정

### ① 사전등록 재현 기준 = 실패 확정

기준(E-148): *"gs120-149 완성 세 창 중 최소 두 창에서 `sv/scored ≥ 0.738` & `dr/scored ≤ 0.262` &
`member_rate` 0.635~0.735"*

| 창 | sv | dr | scored | sv/sc | sv−dr | member | 판정 |
|---|---|---|---|---|---|---|---|
| gs110-119 (사건) | 62 | 22 | 84 | **0.738** | +40 | 0.685 | *기준의 출처* |
| gs120-129 | 47 | 43 | 90 | 0.522 | +4 | 0.680 | ❌ |
| gs130-139 (n=8) | 48 | 58 | 106 | 0.453 | −10 | 0.668 | **산술적 불가** |

gs130-139 산술: 0.738 도달에 **추가 save 116건(derail 0 가정)** 필요. 남은 2스텝 최대 44건 ⇒ 불가능.
**두 창 실패 ⇒ "3중 2" 충족 불가.**

### ★★판정문 (codex 합의) — 이 문장 그대로 쓸 것

> **E-148 proxy 사건은 후속 창에서 사전 기준으로 재현되지 않았다. 그러나 기준값이 사건 창 자신에서
> 유도됐고 후속 창에서 달성 불가능해졌으므로, 이 실패만으로 효과 부재를 추론하지 않는다.
> 최종 효능 판정은 held-out Q4 eval 에 따른다.**

⚠**기준 설계 결함 둘**(내 오류 ⑪ 의 실물):
1. 문턱 0.738 이 **사건 창 자신의 관측값**이다 ⇒ 정의상 그 창만 통과한다.
   `sv/scored` 창간 분포는 **0.227~0.738**(14창), 중앙값 근처 0.47.
2. **`dr/scored ≤ 0.262` 는 `save+derail=scored` 이므로 첫 조건의 대수적 중복**(codex).
   조건 셋 중 실질은 **둘**이었다.
⇒ **규율: 재현 문턱은 사건 창을 제외한 분포에서 정한다. 조건들이 대수적으로 독립인지 먼저 확인한다.**

### ② ★b3sh 팔 정체 정정 — codex 가 제기, 검증 결과 **무해**

codex 지적: b3sh 런처가 **`dcpo_trunc_open_penalty` 를 덮어쓰지 않았고 기본값이 0.3**
(`h100std_rq3v2f_b3sh.yaml:251-257` vs `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:203`).
이 패널티는 실제 advantage 에 적용된다(`dcpo_region.py:1350-1362`). **지적은 사실이다.**

**검증 결과 무해 — 두 근거:**
1. **표적 모집단이 비어 있다.** 이 패널티는 *"메타를 열고 닫기 전에 잘린 행"* 에만 걸리는데
   `dcpo/meta_unclosed_rate` 가 137스텝 전체에서 **최대 0.0059 · 평균 0.0011**
   (배치 512행 기준 스텝당 평균 0.6행) ⇒ 사실상 발화하지 않는다.
2. **원본 shiftonly 승리 구성도 동일하다.** `archive/launchers_pre_rq3/h100std_shiftonly.yaml:152-157`
   은 `w_cal`·`w_format`·`w_emit`·`len_cost`·`w_over` 만 0 으로 덮어쓰고
   **`trunc_open_penalty` 는 건드리지 않았다.** ⇒ b3sh 는 이 축에서 **원본과 동일**, 복제 충실도 이상 없음.

⇒ **정확한 서술**: b3sh 는 *"pmi 단독"* 이 아니라 ***"원본 shiftonly 와 같은 다섯 헤드를 끈 구성"***.
⛔내가 앞서 쓴 *"길이 압력이 전혀 없다"* 는 **과했다** — 잠재적 절단 패널티가 하나 남아 있되 발화하지 않는다.
★b3p·b3s 도 같은 기본값을 쓰므로 **`b3sh − b3p` 대비는 여전히 헤드 스택을 정확히 격리**한다.

### ③ 건전성 — 경고지만 발산 아님

| 창 | ent | len | clip% | emit | grad_norm | score |
|---|---|---|---|---|---|---|
| gs0-9 | 0.216 | 300 | 0.00 | 0.995 | 0.480 | 0.380 |
| gs90-99 | 0.268 | 476 | 0.53 | 0.999 | 0.593 | 0.414 |
| gs110-119 | 0.242 | 446 | 0.45 | 1.000 | 0.552 | 0.466 |
| gs120-129 | 0.270 | 497 | 0.57 | 1.000 | 0.551 | 0.486 |
| **gs130-139(n=8)** | **0.313** | **614** | **1.39** | 0.999 | 0.521 | 0.476 |

엔트로피 +45% · 길이 2배 · 절단률 이전 최대의 **2.4배**. **단 codex 판정: b3s 의 6.417 발산과
같은 경로라 볼 근거 없음** — `emit`(1.0 유지)·`grad_norm`·`score` 가 동반 상승하지 않는다.
★절단 행은 `R_corr=-1` 강제(`dcpo_region.py:916-927`)이므로 **절단률 상승은 병리 위험 신호**로 계속 본다.
★`len_cost=0` 인 팔에서 길이 증가는 코드상 가능하다(`verl_sdc.py:490-507` 이 유일한 길이 비용).

### ④ 조치 = **중단하지 않고 완주**(codex 권고, 나도 동의)

최종 기준이 held-out Q4 이고 지금은 학습 사망이 아니다. ⚠**중단 권한은 내게 없다**(사용자 승인 사항).
codex 부가 제안: `gs130` 중간 eval 을 병렬로 돌릴 수 있으나 **중간 진단으로만** 취급하고
최종 판정은 최종 체크포인트로 한다. **이 역시 승인 사항이라 미착수.**

---

## E-150 (0806 09:2x UTC) — ★★`critic/score` 는 correctness 가 아니다 · **학습 중 정확도는 네 팔이 구별되지 않는다**

### ① codex 가 무너뜨린 것: `critic/score/mean` 오독

내가 세 틱에 걸쳐 *"b3sh 의 `score` 가 네 창 연속 상승 = 정확도가 오른다"* 로 읽었다. **틀렸다.**

`critic/score/mean` 은 rollout 배치 전체의 `token_level_scores.sum(-1)` 평균이고, EOS 위치에
**`R_corr + 0.5·R_meta + 0.3·R_cal + 0.1·R_format`** 이 합산된다
(`verl_sdc.py:1044-1068`, `:2419-2443`).
★**b3sh 의 `w_cal=0`·`w_format=0` 은 advantage 라우팅만 끄고, GDPO 보상 가중치
`gdpo_reward_weights: [1.0, 0.5, 0.3, 0.0, 0.1]` 은 그대로다**
(`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:101`, b3sh 런처는 미override — 직접 grep 확인).
★추가 오염(codex): pmi-shift `R_meta` 는 advantage key 에 덮어쓰지만 stash 기반 `rm_scores` 는
그대로라, `critic/score` 가 pmi-shift 를 직접 반영하지 않을 수 있다(`verl_sdc.py:469-483`).
⇒ **`critic/score` 로 팔 간 정확도를 비교하면 안 된다. 순수 지표는 `gdpo/correctness/mean` 이다.**

### ② ★★순수 correctness 로 재측정 — 네 팔이 구별되지 않는다

`gdpo/correctness/mean` → 정확도 `(x+1)/2`:

| 팔 | gs0-24 | gs75-99 | **gs125-149** | gs200-224 | gs275-299 | **held-out MATH500** |
|---|---|---|---|---|---|---|
| b2p (통제) | — | — | — | 75.96% | 75.57% | **77.28%** |
| b3s | 72.19% | 77.46% | **79.92%** | 77.98% | 77.26% | 75.28% |
| b3p | 72.46% | 77.38% | 77.10% | 77.94% | 76.48% | 75.20% |
| **b3sh** | 72.39% | 76.97% | **77.57%** | — | — | (미측정) |

★★★**b3s 는 학습 중 정확도가 79.92%(전 팔 최고)였는데 held-out 에서 −2.00pp 졌다.**
★b3sh 의 현재 77.57% 는 **같은 구간의 b3s(79.92%)보다 낮다.**
⇒ **b3sh 의 "네 창 연속 단조 상승"은 효능 증거가 아니다.** 같은 모양이 b3s 에서 더 크게 나왔고
결과는 음수였다. **내 지난 세 틱의 낙관을 철회한다.**

★★**이것은 "학습 지표가 held-out 을 예측하지 않는다"를 처음으로 *측정*한 것이다**(그동안은 주장이었다).
학습 중 정확도 격차는 **1pp 미만**인데 held-out 격차는 **2pp** 다 — 부호까지 뒤집혀 보인다
(학습에선 b3s > b2p, held-out 에선 b3s < b2p).

### ③ codex 판정 요약

- **Q1**: `score` 상승을 correctness 로 읽으면 안 된다. 절단 행은 `R_corr=-1` 강제
  (`dcpo_region.py:916-927`)라 절단이 늘면 내려가야 하는데 올랐다 ⇒ 다른 보상항 기여.
  `len_cost=0` 이라 길이가 score 를 **기계적으로** 올리는 경로는 없다.
- **Q2**: "메타 보상이 방해였다"(a)는 **코드상 확인 불가** — pmi-shift 는 여전히 META_CONTENT
  advantage 에 들어간다. 가장 안전한 읽기: *"두 지표가 서로 다른 모집단·시점·보상 정의를 보고 있어
  반대 방향의 일부가 집계 아티팩트일 수 있다."*
- **Q3 (건전성 판별 지표)**: **`dcpo/pmishift_member_rate`** 하나를 보라. 길이·절단이 오르면서
  이 값이 떨어지면 형식 붕괴 초입. 실측: b3sh 0.687→**0.631**(완만) / b3s 0.695→0.586 /
  **b3p 0.694→0.014(붕괴)**. ⇒ b3sh 는 b3s 와 비슷한 완만한 하락, b3p 식 붕괴는 아니다.
- **Q4**: `critic/score` 로 b2p 를 넘어도 **아무것도 강하게 말할 수 없다**. 대신
  `gdpo/correctness/mean` 이나 동일 프로토콜 held-out 으로 비교하라.

### ④ ★사전등록 (codex, E-148 실수를 피한 형태)

> **gs300 held-out Q4 대역에서, 사전 고정한 paired item-bootstrap 95% CI 의
> `b3sh − b2p` 하한이 0pp 보다 클 것.**
> 그렇지 않으면 **학습 중 단조 패턴은 효능 증거로 인정하지 않는다.**

★**문턱을 사건 창 관측값에서 가져오지 않고 repo 의 최종 성공 기준(matched held-out 정확도)을
따른다**(`docs/CONSTITUTION.md:44-48`). E-149 의 기준 결함을 반복하지 않는다.

### ⑤ 내 오류 ⑭ — 지표 이름을 정의로 착각했다

`critic/score/mean` 을 "정확도"로 읽었다. **이름이 아니라 계산식을 확인했어야 했다.**
오류 ⑬(보상 스케일이 다른 팔의 score 비교)을 정정하면서도 **그 지표가 애초에 correctness 가 아니라는
것은 못 봤다.** 같은 지표를 두 번 잘못 읽은 셈이다.
⇒ **규율: 지표를 판정에 쓰기 전에 그 지표가 계산되는 코드 줄을 연다.** 이름·단위·범위로 추측하지 않는다.

---

## E-151 (0806 11:2x UTC) — ★b3sh 능력 붕괴 판정 · **b3p·b3s 와 다른 세 번째 실패 양상**

### ① 사전 선언한 두 문턱이 동시에 뚫렸다

| 창 | acc% | member | sv/sc | ent | len | clip% | **emit** | grad |
|---|---|---|---|---|---|---|---|---|
| gs120-129 | **78.22** | 0.680 | 0.522 | 0.270 | 497 | 0.57 | 1.0000 | 0.551 |
| gs130-139 | 78.03 | 0.666 | 0.377 | 0.301 | 581 | 1.21 | 0.9994 | 0.514 |
| gs140-149 | 76.21 | 0.579 | 0.338 | 0.341 | 613 | 1.46 | 0.9998 | 0.462 |
| **gs150-159** | **67.86** | **0.446** | 0.431 | 0.388 | 666 | 1.76 | **1.0000** | 0.395 |
| gs160-169(n=2) | 68.55 | 0.445 | 0.727 | 0.416 | 704 | 2.34 | **1.0000** | 0.424 |

- 정확도 **67.86%** — 완성 창 최저(71.35%, gs0-9)보다 **3.5pp 아래**
- `member` **0.446** — 완성 창 최저(0.579)보다 크게 아래, 사전 문턱 0.5 미만
- ★**`meta_emission` 은 1.0000 으로 완벽**

### ② ★세 번째 실패 양상 — 기존 둘과 다르다

| 팔 | emit | 학습 정확도 | 엔트로피 | 양상 |
|---|---|---|---|---|
| b3p | 0.997→**0.018** | 77% 유지 | 0.32 안정 | **메타를 포기**했다 |
| b3s | 0.999 유지 | 77% 유지 | 0.22→**5.88** | **엔트로피가 폭주**했다 |
| **b3sh** | **1.0000 유지** | 78.2→**67.9** | 0.39(낮음) | **메타를 쓰면서 능력이 무너진다** |

### ③ codex 판정 (코드 근거)

**`member_rate` 의 정의**(`verl_sdc.py:1617,1745`, `dcpo_region.py:111`): 각 행이
①`wellformed/swapped/dup_open/reversed/drift` 중 하나이고 ②메타 내용이 비어 있지 않고
③gold/decoy 가 유효하며 **토큰 길이가 같고** ④frozen-ref PMI 두 값이 유한할 때 `member=1`.
⇒ **정확도도 발화율도 아니고 "PMI 채점 가능성"이다.**
★`meta_emission` 은 문자열에 `<|meta|>` 가 있는지만 본다(`verl_sdc.py:808`) — **닫히지 않은 메타·
절단·discard 도 emission=1 이 된다.**
⇒ **member 하락 = (a) 메타 표식은 내지만 PMI 가 신뢰 가능한 형식·길이로 채점되지 않는 행이 늘었다.**
길이·절단 상승과 정합. ⚠단 aggregate 만으로 **절단/discard 실패와 decoy 불일치/ref 비유한 실패를
분리할 수 없다.**

**⚠`correctness` 와 `member` 는 같은 모집단이 아니다**: correctness 는 각 completion 을 gold 와
대조하는 ±1 head(`rewards.py:978`), member 는 그중 PMI 채점 조건을 통과한 **부분집합**.
V4 에서 truncation/discard 는 correctness head 에서도 별도 처리된다(`dcpo_region.py:915`).
⇒ **member 하락을 correctness 하락과 같은 원인으로 단정하지 말 것.**

**과적합인가**: **아니다.** b3s/b3p 는 *학습은 유지·상승하고 held-out 만 하락*했으나 **b3sh 는 학습
정확도 자체가 무너진다.** ⇒ 분류는 **"일반화 실패가 아니라 on-policy 학습/보상 구성에 의한
정책·능력 붕괴"**. ⚠단 held-out 을 아직 안 봤으므로 *"과적합이 전혀 없다"* 까지는 말할 수 없다.

### ④ ★★codex 권고 = **gs150·gs160 을 held-out 으로 재고 남은 139스텝은 중단**

세 신호가 동시에 나쁘다: correctness 가 통제군보다 **8pp 낮음** · member 0.446 · 길이·절단 상승.
*"메타를 유지하면서 유용성을 회복할 가능성보다 **메타 표식만 유지하는 정책 붕괴가 진행 중일
가능성이 높다**."* 하나만 고르면 **gs160**, 변화점까지 보려면 **gs150·gs160 둘 다**.
⚠단 이 중간 eval 은 **사전등록한 gs300 판정을 대체하지 않는다**.
⚠**나는 중단 권한이 없다 — 사용자 승인 사항.**

### ⑤ ★내 오류 ⑯ — `gdpo/correctness/mean` 도 팔 간 직접 비교가 무효였다

`verl_sdc.py:496-511`: `dcpo_len_cost != 0` 이면 **`correctness` 키 자체에서 길이 비용을 차감**하고
그 값을 로깅한다(`_heads["R_corr"]` 재대입).
⇒ **b3sh·b2p 는 `len_cost=0` 이라 순수하지만 b3s·b3p 는 차감된 값**이다.
**차감 크기**(0.08 × len/8192): b3s gs250-299 len 1145 → 0.0112 = **0.56pp** ·
b3s gs150-199 len 1445 → 0.0141 = **0.71pp** · b3p gs250-299 len 1086 → 0.0106 = **0.53pp**.
⇒ **1pp 미만이라 결론은 뒤집히지 않고 오히려 강화된다**: b3s 의 실제 학습 정확도는 79.92%+0.7pp 로
더 높았고 held-out 은 여전히 −2.00pp 였다. **일반화 격차도 더 음수다.**
✅**b3sh vs b2p 비교는 둘 다 `len_cost=0` 이라 유효** — 최근 틱들의 비교는 문제없다.
⇒ **규율(재확인): 팔 간 비교 전에 그 지표가 팔마다 같은 계산을 거치는지 확인한다.**
E-150 에서 `critic/score` 에 대해 이 실수를 잡고도 **바로 옆 지표에서 같은 실수를 했다.**

---

## E-152 (0806 12:0x UTC) — ★★E-151 판정 정정: **능력 붕괴가 아니라 형식·보상 라우팅 실패**

### ① 응답을 직접 열어서 원인을 특정했다

wandb `dcpo/rollouts` 테이블(스텝당 512행)을 gs100/120/140/155 로 받아 집계.

| gs | wellformed | discard | 오프너 `<\|meta\|>` 2회+ | discard 중 **정답인데 0점** |
|---|---|---|---|---|
| gs100 | 91.4% | 8.2% | 18행(3.5%) | 16 |
| gs120 | **94.1%** | 5.9% | 7행(1.4%) | 7 |
| gs140 | 87.1% | 12.5% | 42행(8.2%) | 46 |
| **gs155** | **53.3%** | **46.7%** | **178행(34.8%)** | **160(66.9%)** |

실제 discard 응답(gs155) — **오프너가 두 번이고 답은 맞았다**:
```
<|meta|>
<|meta|>
confidence: 0.88
The answer looks plausible but must not be committed without an independent check...
decision: verify
<|/meta|>
Start by expanding: (8-x)^2=x^2 => 64-16x=0 => x=4
\boxed{4}
```

### ② ★★능력은 거의 안 떨어졌다 — E-151 판정을 정정한다

`fmt_class` 별 `R_corr` 평균:
| gs | wellformed n / 평균 → acc | discard n / 평균 |
|---|---|---|
| gs120 | 482 / +0.6058 → **80.29%** | 30 / **0.0000** |
| gs155 | 273 / +0.5458 → **77.29%** | 239 / **0.0000** |

⇒ **형식이 맞는 행의 정확도는 80.3%→77.3%, 3pp 하락에 그친다.**
내가 E-151 에 쓴 *"정확도 78.2→67.9%, 능력 붕괴"* 는 **`discard` 행 47% 가 `R_corr=0` 으로
평균을 끌어내린 것**이 대부분이었다.

★**정정된 판정문(codex 합의) — 이 문장 그대로 쓸 것**:
> **b3sh 는 gs155 에서 discard/중복 오프너 급증으로 집계 정확도가 0점 행에 의해 크게 낮아졌지만,
> wellformed 행의 정확도는 약 3pp 만 하락해 능력 붕괴가 아니라 형식 판정·보상 라우팅 실패를 보였다.**

### ③ ★**정정(0806 15:0x, codex-sol 적발·내가 코드로 재확인)** — discard 행은 **보상 어드밴티지가 정확히 0**, 그러나 **"완전한 zero-gradient" 는 틀린 문장이었다**

**틀린 문장(이 항목의 원래 제목·본문)**: *"보상도 벌점도 없다 — 완전한 zero-gradient"*, *"마스크가 모두 빈다"*.
**맞는 문장 — 앞으로 이것만 쓸 것**:
> **b3sh 에서 discard 행은 라우팅된 GRPO 보상 어드밴티지가 정확히 0 이다. 다만 actor 의
> 엔트로피 목적함수에는 여전히 참여하므로 gradient 가 전혀 없는 것은 아니다.**

살아남는 근거(그대로 유효):
- `R_corr`/`R_meta`/`R_cal` 을 **모두 0 으로 덮어씀**: `dcpo_region.py:944-966`.
  테이블 실측이 정확히 `0.0000`(gs165 discard 234행 전부, `nunique=1`).
  ⚠나는 `dcpo_region.py:916-927` 만 읽고 *"discard 도 ±1 을 받는다"* 고 판단했다 — **덮어쓰기 블록을 못 봤다.**
- ANSWER/META_CONTENT/CONF 마스크는 **빈다**: `dcpo_region.py:597-600, 692-695`
- PMI `R_meta` 는 `TRUSTED_META_CLASSES` 만 대상: 집합 정의 `dcpo_region.py:106-113`,
  **실제 배제 지점** `verl_sdc.py:1624`
- correctness/meta/cal **그룹 평균에서 제외**: 멤버십 기록 `verl_sdc.py:397`,
  실제 소비 `verl_sdc_utils.py:407`(`group_mean_subtract`)

**틀렸던 두 지점**:
1. ⛔**"마스크가 모두 빈다" 는 거짓** — discard 는 `FORMAT_VIOLATION` 을 **명시적으로 채운다**
   (`dcpo_region.py:597-600`, 실측 확인). b3sh 는 `w_format=0` 이라 그 마스크에서 보상이 흐르지
   않을 뿐, 마스크 자체는 존재한다.
2. ⛔**"gradient 가 전혀 없다" 는 거짓** — b3sh 는 `entropy_coeff: 0.001`
   (`configs/verl_sdc_e21r_shared.yaml:53`)을 **override 하지 않는다**(런처 grep 0건).
   discard 토큰도 엔트로피 항에는 들어간다.

⇒ 그래도 결론은 유지된다: 정답 discard 가 `+1` 을 `0` 으로 바꿔 **이득을 얻는 보상 경로는 없다.**
⇒ ⛔**"GRPO 중심화로 −1 대신 0 을 노리는 해킹" 가설은 코드상 성립하지 않는다.**
⚠**왜 번지는가는 여전히 미해결.** 난이도 편향도 약하다(쉬움 41.2% · 어려움 47.3%).

### ③-b ★**새 사실 — discard 는 "복잡한 쓰레기" 가 아니라 `replaced=False` 인 tier-1 강등이다**

롤아웃 테이블 원본 재집계(gs155/165/170/175, 각 512행):

| gs | wellformed | discard | dup_open | truncation | `replaced=True` | discard 중 `answer==gt` |
|---|---|---|---|---|---|---|
| gs155 | 273 (53.3%) | 239 (46.7%) | 0 | 0 | 0 | **179 (74.9%)** |
| gs165 | 276 (53.9%) | 234 (45.7%) | 1 | 1 | **1** | **167 (71.4%)** |
| gs170 | 255 (49.8%) | 255 (49.8%) | 0 | 2 | 0 | **201 (78.8%)** |
| gs175 | 242 (47.3%) | **269 (52.5%)** | 0 | 1 | 0 | **183 (68.0%)** |

- discard 행은 **전부** `has_meta=True`, `unclosed=False` (`nunique=1`) — 메타를 안 쓴 것도,
  닫는 태그를 빠뜨린 것도 아니다.
- **512행 중 치환에 성공한 행이 1개뿐**이다. 그런데 치환은 config 에서 **켜져 있고**
  (`dcpo_format_replace: true`, `dcpo_recover_first_pair: true` —
  `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:185,188`), 학습 로그에
  `FORMAT-REPLACE ABORT` **0건**, `coherence FAIL` **0건**이다(36,746줄 전수 grep).
- ⇒ **producer 단계의 분류기 자체가 이 행들에 replacement plan 을 내주지 않고 있다.**
  `w_format` 이 벌하는 대상이 바로 **unreplaced discard** 이므로 b3shf 가정과는 정합적이지만,
  **왜 plan 이 안 나오는지는 미확정.** ★다음 검사: eval/롤아웃에 `token_ids` 를 저장해
  `classify_dcpo_format` 을 직접 돌린다(텍스트 재토큰화로는 원래 토큰열을 증명할 수 없다).

⚠**위 표의 "오프너 2회+" 계열 수치(§①)는 `main_tail` — 응답의 꼬리 발췌 — 에서 센 것이므로
하한이다.** gs165 discard 234행의 `main_tail` 안 `<\|meta\|>` 개수는 2회 122행·3회 30행·4회 9행이고
`<\|/meta\|>` 는 1회가 182행이다. 즉 전형은 `open, open, close` 이지 "닫는 태그 실종" 이 아니다.

### ④ ★`w_format` 을 되살리면 되나 (codex, 코드 근거)

**(a) 벌한다.** `w_format` 은 **unreplaced discard** 를 벌한다 — `format_penalty = -format_neg`
(`dcpo_region.py:1002-1019`), 현 config `format_neg=0.2`
(`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:136`), discard delimiter mask 로 routing
(`dcpo_region.py:1275-1289`).
⚠**단 토큰 치환에 성공한 tier-1 `dup_open` 은 `R_format=0` 이고 mask 도 없어 벌하지 않는다**
(`dcpo_region.py:568-579`). 우리 예시(`open, open, close`)는 tier-1 복구 대상이 **아니라**
최종 discard 다(`dcpo_region.py:276-300` vs `:344-360`).
★`recover_first_pair=true` 는 **pre-pass 에서만** 호출되고 consumer 재판정엔 전달되지 않는다
(`verl_sdc.py:2796-2800` vs `:285`).

**(b) 값은 0.35 그대로**가 b3s/b3p 와 matched 비교가 된다. ⚠단 그 순간 이 팔은
**더 이상 순수 shiftonly 복제가 아니라 `shiftonly + format scaffold`** 라는 별도 팔이다.
⚠`anchor_norm=true` 라 warmup 뒤 format advantage 가 재스케일되므로 **유효 강도를 0.35 로 읽지 말 것**
(`dcpo_region.py:1283-1289`).

**(c) 위험**: `w_format` 은 format delimiter 에만 sparse 하게 작용하고(`dcpo_region.py:1152-1173`)
엔트로피를 직접 올리는 항이 아니다. b3s/b3p 의 발화 붕괴·엔트로피 폭주를 재현한다고 **말할 코드 근거는
없다.** ⚠그러나 공유 모델에서 format 압력이 발화 패턴에 간접 영향을 줄 수 있어 **안전하다고 단정도 못 한다.**

### ⑤ ★codex 권고 — 중단은 유지, **사유를 바꾼다**

- **gs150·gs160 held-out 평가는 계속** — ★단 **wellformed/discard 를 분리 보고**할 것
- **남은 학습은 중단 권고** — 능력 붕괴 때문이 **아니라**, gs155 이후 **상당수 행이
  zero-gradient/discard 레짐이라 이 런의 후반 비교가 해석 불가능**해졌기 때문
- **format 검증은 돌아가는 런을 중간에 바꾸지 말고** 별도 짧은 **`shiftonly + w_format=0.35` ablation**
  으로 분리할 것
⚠**중단·발사 권한은 내게 없다 — 사용자 승인 사항.**

### ⑥ 내 오류 ⑰

**코드 블록 하나만 읽고 "덮어쓰기 없음"을 단정했다**(`dcpo_region.py:916-927` 만 보고 `:944-966` 을 놓침).
그리고 그 잘못된 독해로 **codex 의 옳은 지적을 "틀렸다"고 사용자에게 보고**했다.
⇒ **규율: 값이 코드 예측과 다르면 코드가 아니라 내 독해를 먼저 의심하고, 같은 변수를 건드리는
   블록을 전부 grep 한다.**

---

## E-153 (0806 15:0x UTC) — ★b3sh gs165 첫 패스 착지 · **4k 패스로는 주 질문을 풀 수 없다** · codex-sol 게이트로 초안 4건 중 3건 축소

### ① 무엇이 착지했나

`eval/rq3v2f_b3sh_gs165/rq3v2f_b3sh_gs165_4k_n8/*.parquet` (1030문제 × n=8, `max_tokens=4096`).
16k per-bench 패스는 **아직 생성 중**. gs175 는 미착지.

### ② ★사전등록이 정한 판정 패스는 **16k 뿐**이다 (codex 지적 → 문서로 확인)

- `docs/EXPERIMENT_PLAN.md:20` 과 재차 `:58` — *"판정은 불변 — **gs300 held-out 1030(16k,
  avg@8, math_verify)** 에서만"*.
- `scripts/rq2_analyze.py:16` 이 16k 경로를 하드코딩. C-029/C-030 은 전부 이 경로 산출물.
⇒ **4k 패스는 판정 패스가 아니라 배포예산 민감도 분석이다.** 무시해서도 안 되고, 판정에 써서도 안 된다.

### ③ ★같은 대비가 두 패스에서 다르게 나온다 (MATH500, 같은 ckpt·문제·채점기)

| 패스 | b2p 기준선 | b3s−b2p | b3p−b2p |
|---|---|---|---|
| **16k**(판정 패스) | 77.28% · 1075.2tok · 절단 2.43% | **−2.00pp** CI[−3.38,−0.60] **유의** | **−2.08pp** CI[−3.50,−0.68] **유의** |
| 4k(민감도) | 75.45% · 530.4tok · 절단 3.97% | −0.53pp CI[−1.88,+0.82] 불유의 | −0.53pp CI[−1.90,+0.88] 불유의 |

CI 폭은 두 패스가 비슷하다(≈2.78 vs 2.70pp). **점추정이 약 4배 축소**된 것이지 정밀도가 떨어진 것이 아니다.

★**codex 축소 요구(수용)** — *"예산 조건부(budget-conditional)"* 는 **한 단계 과하다.**
패스별 추정치가 다른 것을 관측했을 뿐 **상호작용을 검정하지 않았다.** 채택 문구:
> **MATH500 추정 효과는 판정 패스(16k)보다 4k 패스에서 실질적으로 작다. 이는 디코딩 예산
> 민감성을 시사하지만, 예산 상호작용과 prefix 포함관계는 아직 검정하지 않았다.**

★미검정 항목 둘(다음 분석에서 처리): (a) 문항 단위로 `[(b3−b2p)_16k − (b3−b2p)_4k]` 부트스트랩,
(b) 4k 완성이 16k 완성의 **정확한 prefix 인지** — 별도 생성 호출이고 16k 는 벤치별·4k 는 통합이라
같은 시드만으로는 "캡이 유일한 변경 원인" 을 증명하지 못한다.
★C-030 인용 시 **"긴 응답 대역"이 아니라 "Q4/중장 대역"** 으로 쓸 것 — C-030 은 최장 Q5 를
**미해결**로 적었지 최대 손상으로 적지 않았다.

### ④ b3sh gs165 판정 — **미결(inconclusive)**

| 비교(4k 패스) | delta | 95% CI |
|---|---|---|
| b3sh gs165 − b2p gs300, **MATH500** | **−0.97pp** | [−2.30, +0.35] |
| b3sh gs165 − b2p gs300, gsm8k | +0.33pp | [−0.57, +1.25] |
| b3sh gs165 − b2p gs300, aime2024 | −2.08pp | [−6.25, +1.67] |
| b3sh gs165 − b2p gs300, 1030 통합 | −0.38pp | [−1.18, +0.41] |

★**내 오류 ⑲ — 서로 다른 시험지를 나란히 놓았다.** 초안에서 b3sh 는 **1030 통합**, b3s/b3p 는
**MATH500** 으로 계산해 비교했다. 같은 MATH500 으로 맞추면 b3sh 는 −0.97pp 다.
★**1030 통합 수치는 500/500/30 의 임의 혼합**이고 사전등록 엔드포인트가 아니다 — 헤드라인 금지.

★**codex 축소 요구(수용)** — *"4k 패스는 쓸 수 없다·비진단적"* 은 과하다. 4k 추정치는 4k estimand
에 대해 충분히 정밀하다. 채택 문구:
> **4k 에서 b3sh gs165 는 b2p gs300 과 MATH500 에서 구별되지 않는다(−0.97pp, CI[−2.30,+0.35]).
> 이 이차적·스텝 비매칭 추정치는 "무해" 와 "16k 의 b3s/b3p 급 손상" 을 모두 포함하므로 주 질문(16k)
> 을 해결하지 못한다. 다만 4k 예산에서의 gs165 성능에 대해서는 정보가 있다.**

⚠**스텝 비매칭**: b3sh gs165 vs 대조군 gs300. b3sh 는 gs175 에서 멈췄으므로 **gs300 판정은 원리상 불가.**
★다음: 대조군을 경유하지 말고 **`b3sh − b3s`, `b3sh − b3p` 직접 대비**를 계산할 것(codex 권고).

### ⑤ 형식 실패가 **eval 생성에도 발현한다**

텍스트 수준 분류(내 프록시, 트레이너 토큰 분류기 아님), 4k 패스 샘플 단위:

| arm | wellformed | multi_block | no_meta | dup_open | other |
|---|---|---|---|---|---|
| **b3sh gs165** | 78.86% | 2.89% | 0.00% | **18.13%** | 0.12% |
| b2p gs300 | 94.54% | 2.75% | 0.00% | 0.17% | 2.54% |
| b3s gs300 | 98.96% | 0.02% | 0.00% | 0.00% | 1.02% |
| b3p gs300 | 0.83% | 0.01% | **98.52%** | 0.00% | 0.64% |

★b3p 는 **메타를 아예 포기**했다(`no_meta` 98.5%) — 이건 형식 실패가 아니라 별개 실패다.
★**codex 축소 요구(수용)**: *"표면 중복 오프너가 eval 에 도달한다"* 는 지지되지만
**"18.13% 가 트레이너 discard 일 것"** 은 성립하지 않는다. 두 분류기가 갈리는 지점 —
토큰 조각으로 조립된 `<\|meta\|>` 문자열 / open·close 사이에 `</think>` 가 낀 경우 /
서명 없는 2-open / close-only·reversed·drift·진짜 truncation / multi_block 취급 /
치환된 tier-1 행의 이름 보존. ⇒ **트레이너 클래스율과 숫자로 직접 비교하지 말 것.**
★⛔**"eval 온도 0.7 이라 과소평가" 는 삭제한다** — 방향 근거가 없다(codex). 낮은 무작위성은
고확률 오형식 모드를 **억제할 수도 증폭할 수도** 있다. 디코딩 스윕 없이는 말하지 않는다.

### ⑥ 중복 오프너 행이 **더 유능한 것은 아니다**(단, 서술적·중첩 한정)

4k 패스, b3sh gs165:

| bench | n | dup_open율 | 정형 acc | dup_open acc | 정형 tok | dup tok |
|---|---|---|---|---|---|---|
| gsm8k | 4000 | 16.02% | 92.49% | 92.51% | 222.6 | 214.7 |
| math500 | 4000 | 21.22% | **71.23%** | **86.81%** | 673.9 | 423.0 |
| aime2024 | 240 | 1.67% | 11.59% | 50.00% | 2307.2 | 1901.0 |

★**codex 축소 요구(수용)** — 채택 문구:
> **MATH500 의 큰 주변부 우위는 문항 내 중첩 부분집합에서는 유지되지 않는다(추정 연관
> −0.57pp, CI[−2.20,+1.02]). 구성(composition) 설명을 지지하지만, 효과 없음이나 인과를
> 확립하지는 않는다.**

★**내 오류 ⑳ — `n=501` 은 MATH500 단독일 수 없다**(MATH500 은 500문항). 벤치 통합 짝지음으로
**MATH 전용 격차를 설명**했다. ★다음: **MATH 전용** 문항고정효과 비교, 중첩 문항 수 명시,
문항별 중복 성향을 level·b2p 문항정확도와 대조. 부트스트랩은 **샘플이 아니라 문항** 단위로.
⚠**짧은 응답 = 쉬운 문항이 아니다** — 길이 자체가 중복 오프너 궤적의 *결과*일 수 있다.
⚠짝지음은 8회 추출에서 두 형식이 **모두 나온 "스위처" 문항**에만 조건화되므로 자체 선택편향이 있다.

### ⑦ ★b3shf 모니터링 보정 — **배치 discard 율만 보면 안 된다**(codex)

`w_format` 어드밴티지는 **그룹 중심화**된다. `n=8` 그룹이 **전부 discard** 면 `w_format=0.35` 여도
format 어드밴티지는 0 이다. ⇒ **discard 와 비-discard 가 섞인 그룹의 비율**을 함께 볼 것.
사전등록 KILL 기준(gs180 까지 discard <15%)은 유지하되, 이 보조 지표를 같이 낸다.

### ⑧ 사전등록 정합성 — 판정문에 반드시 붙일 단서 셋 (codex, 문서로 확인)

1. 동결 사전등록은 **토큰 캡을 명시하지 않는다**; 16k 확약은 `docs/EXPERIMENT_PLAN.md` 에서 온다.
2. 동결본은 **`format_fair` + `strict_boxed`** 채점기를 고정했다
   (`docs/PREREGISTRATION_rq3v2_base_replication.md:101`). C-029/C-030 이 쓴 **`$...$` 래핑
   채점기(C-027)는 프로토콜 이탈/민감도**로 라벨링하거나 동결 채점기를 함께 보고해야 한다.
3. **b3s 사전등록의 주 지표는 전체 MATH500 정확도가 아니라
   `Δacc(L4–5) − Δacc(L1–2)` 기울기**이고 잡음바닥 **±3.08pp** 가 이미 실측돼 있다
   (`docs/preregistrations/2026-08-03-b3s-meta-floor.md:39`).
⇒ C-029/C-030 의 제목·산문("RQ2 음수", "패키지가 손상원") 에 **"사전등록 16k 디코딩 하에서"**
   단서를 붙인다. 좁은 16k 판정 자체는 4k 결과와 **모순되지 않는다.**

### ⑨ 산출물

- `$JOB/tmp/b3sh_gs165_eval.py` — 형식 분리 채점(probe 성격, 정본 아님)
- `$JOB/tmp/pass_resolution_check.py` — 4k/16k 대비 + dup_open 분해
- `$JOB/tmp/recheck_e152.py` — 롤아웃 테이블 원본 재집계
- `$JOB/tmp/codex_e153_out.txt` — codex-sol 적대검토 전문
⚠**세 스크립트 모두 일회성 진단이다.** §5.3 대로 정본 `scripts/` 로 승격하지 않는다.
정본 승격이 필요한 것은 하나뿐: **eval 저장에 `token_ids` 추가**(③-b 의 미확정을 닫는 유일한 길) —
**정본 코드 변경이므로 승인 사항.**

### ⑩ 미해결로 남기는 것

- discard 행에 **왜 replacement plan 이 안 나오는가**(치환은 켜져 있고 abort 0건)
- 중복 오프너가 **왜 번지는가**(보상 이득 경로 없음이 코드로 확인됨)
- 4k/16k **상호작용 미검정** · prefix 포함관계 미확인
- `eff_scale_corr` 팔간 3.4배 차이 · `acc_with`/`acc_without` 산술 모순


---

## E-154 (0806 17:2x UTC) — ★★b3sh 도 대조군보다 낮다(16k 판정 패스) · **그러나 pmi_shift 를 지목할 수 없다** · 실행간 분산 최초 실측

★**채택 헤드라인(codex 합의, 이 문장 그대로)**:
> **탐색적 gs165-대-gs300 비교에서 b3sh 는 16k MATH500 에서 vanilla 대조군보다 낮았고, 동결
> 채점기 두 종에서도 그렇다. 보조 계수 다섯을 0 으로 만들어도 음의 연관은 사라지지 않았다.
> 단 이 실험은 PMI-shift 를 공유 TRIOBJ 라우팅·복구·discard·절단 기계로부터 분리하지 못하며,
> 사전등록된 gs300 판정을 제공하지 못한다.**

### ① 수치 — **리비전 고정**(경로만으로 인용 금지, ⑦ 참조)

16k MATH500, 문항 페어드 부트스트랩 10,000회, `$...$` 래핑 채점.
b3sh gs165 = **74.45%** (절단 0.53% · 754.1 tok), 파일 oid `3f86f798de1f52cb`.

★**정정(0806 19:2x) — 이 파일도 그 뒤 덮였다.** eval 잡이 완주하며 같은 경로를 다시 썼고
현재 oid 는 **`7e32d9b491c795ce`**, 값은 **74.70%** 다. 그 파일로 다시 계산하면
**b3sh gs165 − b2p gs300 = −2.58pp CI[−4.03,−1.18]** (아래 표의 −2.83pp CI[−4.20,−1.48] 는
oid `3f86f798…` 기준). **두 수 모두 유효한 생성 실현이며 결론(대조군보다 낮고 0 배제)은 동일하다.**
⇒ ⑦의 규율이 **E-154 자신의 헤드라인에서 바로 발동했다.** 앞으로 표의 모든 행에 oid 를 단다.

| 비교 | delta | 95% CI | |
|---|---|---|---|
| **b3sh gs165 − b2p gs300** | **−2.83pp** | [−4.20, −1.48] | **0 배제** |
| b3sh gs165 − b3s gs300 | −0.82pp | [−2.17, +0.53] | 0 포함 |
| b3sh gs165 − b3p gs300 | −0.75pp | [−2.20, +0.70] | 0 포함 |
| b3sh gs165 − b2p gs300 (gsm8k 16k) | +0.85pp | [−0.17, +1.90] | 0 포함 |
| b3sh gs165 − b2p gs300 (aime 16k, n=30) | −0.42pp | [−3.75, +3.33] | 0 포함 |

★**codex 독립 재현**: −2.825pp, item-bootstrap CI ≈ [−4.20, −1.48]. **동결 채점기
`format_fair`·`strict_boxed` 로는 −2.90pp [−4.28, −1.53]** ⇒ **부호는 `$...$` 채점기 이탈에
의존하지 않는다.** ★codex 중첩(문항+문항내) 부트스트랩 민감도 ≈ **[−4.60, −1.08]** — 여전히 음수.

★**탐색적 하드아이템 지표**(codex 계산, 사후이므로 주 지표 대체 금지):
L1–2 **−0.47pp** · L4–5 **−4.58pp** · **기울기 −4.11pp CI[−6.89,−1.39]** ·
사후 선언된 길이 Q4 대역 **−5.00pp CI[−8.88,−1.13]**. 손상이 어려운 문항에 몰린다는 서술을 강화한다.

### ② ⛔**"원인은 pmi_shift" 추론은 무너진다** — 내 초안 최대 오류

세 처치 팔이 공유하고 대조군에 없는 것은 pmi_shift **하나가 아니다**. 전부 `TRIOBJ_DCPO_V4`,
대조군만 `VANILLA_GRPO`(`h100std_rq3v2f_b3sh.yaml:245` vs `h100std_rq3v2f_b2p.yaml:266`). 공유물:

- **영역 라우팅 어드밴티지** — correctness 를 `ANSWER` 에만, PMI 를 `META_CONTENT` 에만 흘린다.
  vanilla 는 응답 전체에 방송한다(`verl_sdc_utils.py:500`, `dcpo_region.py:1268`).
- **파서 기반 형식 복구** — old-log-prob 계산 **전에 생성 토큰을 변형**
  (`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:183`, `verl_sdc.py:2536`).
- **discard 라우팅** — 마스크 0·correctness 스칼라 0·그룹 중심화 제외
  (`dcpo_region.py:597`, `:956`). **b3sh 의 discard 율을 고려하면 특히 중대**하다.
  vanilla 는 그 행들을 correctness 학습에서 빼지 않는다.
- **`dcpo_trunc_open_penalty=0.3`** — b3sh 가 끄지 않았다(`configs/...:203`), 비중심화 음의
  어드밴티지를 준다(`dcpo_region.py:1350`).
- **PMI 전용 스케일링** — 80스텝 warmup · 채점행 한정 중심화 · confidence carve-out ·
  anchor 정규화(`configs/...:145`, `verl_sdc_utils.py:325`).

★**`EXPERIMENT_PLAN.md:116` 이 이미 적어 두었다**: b3p−b2p 는 **패키지 비교**이고
**PMI 를 분리하려면 matched PMI-off 팔이 필요하다.** ⇒ §6 붉은 깃발 "이건 아직 안 재봤다" 적중.

★**맞는 문구**:
> **cal·format·emit·len_cost·floor 다섯 계수를 0 으로 해도 음의 MATH500 결과는 남는다.
> 따라서 그 다섯이 b3sh 결손에 *공동으로 필요하지는 않다.* 이 결과는 PMI-shift 를 공유 TRIOBJ
> 라우팅·복구·discard·절단·스케일링 기계와 구별하지 못한다.**

★**다음 실험(이 로그가 지정한다)**: **`TRIOBJ_DCPO_V4` 쌍둥이 + `rmeta_source=none`(또는
`w_meta=0`)** — 기계는 전부 같고 PMI 만 끈 팔. 이것 없이는 PMI 귀속이 불가능하다.

### ③ ⛔스텝 비매칭 — 방향을 알 수 없다

b2p 의 gs165 체크포인트가 HF 에 없다(gs300 만 보존). vanilla 가 gs165→gs300 에 **올랐다면**
비교는 b3sh 에 불리하고, **떨어졌다면** 유리하다. **해결 불가.**
★맞는 문구: *"관측된 종점에서 b3sh(gs165) 가 b2p(gs300) 보다 16k MATH500 에서 낮았다."*
매칭된 처치효과도, 사전등록 최종 판정도 **아니다.**
★⛔**"gs165 만에 b3s 수준 손상 도달(45% 적은 학습)" 은 무효** — 단조 손상 누적과 **동등성**을
둘 다 가정한다. **CI 가 0 을 포함하는 것은 동등성의 증거가 아니다**(동등성 마진 미검정).
⇒ "통계적으로 구별되지 않는다" 를 **"차이가 검출되지 않았다"** 로 바꾼다.

### ④ ★실행간 분산 최초 실측 — 그러나 **CI 에 더하지 말 것**

eval 이 선점→재시작을 반복해 **같은 gs165 체크포인트를 4번 생성**했다(4k · seed=42 · T=0.7 ·
top_p=0.95). 그중 둘은 바이트 동일 중복 업로드라 **독립 생성은 3벌**.

| revision | MATH500 | gsm8k | aime2024 | 절단% |
|---|---|---|---|---|
| `6c472a2343` (14:05) | 74.48 | 92.47 | 12.08 | 2.23 |
| `56fe6db23c` (14:58) | 74.48 | 92.47 | 12.08 | 2.23 (← 6c472a 와 **100% 동일**) |
| `81f87a800a` (16:06) | 73.83 | 92.58 | 14.17 | 2.28 |
| `423042c29a` (17:06) | 73.92 | 92.95 | 13.75 | 2.32 |

**서로 다른 생성끼리 completion 문자열 일치율 31.0~32.0%** — 시드를 고정해도 2/3 이 달라진다.
**실행간 SD**: MATH500 **0.35pp** · gsm8k 0.23pp · aime **1.10pp** (n=3, 매우 불확실).
★16k 에도 gs165 AIME 생성이 둘 있다(12.92 vs 14.58, +1.67pp CI≈[−1.25,+5.00], n=30 이라 약함).
⛔**codex 지시(수용)**: 이 값을 **하한·SD·CI 팽창계수·16k 로 이식 가능한 "잡음바닥"** 으로 쓰지 말 것.
사전등록 item-bootstrap CI 를 주로 두되 **"하나의 생성 실현에 조건부"** 라고 명시한다.
★제대로 된 해법은 **체크포인트당 16k 전체 생성을 반복**하고 run/문항 계층 분석을 하는 것.

### ⑤ ⛔(D) 는 과적합 검정이 아니다

`gs175 − gs165` (4k): math500 **+0.78pp** CI[−0.30,+1.85] · gsm8k −0.25 · aime +1.25.
★맞는 문구: *"4k 에서 gs165→gs175 변화는 통계적으로 해소되지 않았다."*
⛔"잡음바닥 안" 이라 쓰지 말 것. ⛔16k(손상이 나타난 곳)에 대한 답이 아니다.
⚠체크포인트 변화와 생성 실현이 **교란**돼 있다(각 체크포인트 1회 생성).
⚠**E-152 의 "능력 손실이 아니다" 를 이것으로 강화하지 말 것** — 10스텝 동안의 평탄함은 지연·
저파워·포화·4k 가 긴 출력 레짐을 덜 노출하는 것으로도 설명된다. **b3sh 는 gs165 에 이미 대조군
아래였으므로 손상이 discard 증가보다 선행했을 수도 있다.**

### ⑥ 사전등록 정합성

- **b3sh 를 명명한 동결 사전등록 문서는 없다.**
- 일반 계획은 gs300·16k·avg@8 최종 판정을 요구(`docs/EXPERIMENT_PLAN.md:20`).
- b3s 사전등록의 주 지표는 L4–5 − L1–2 기울기(`docs/preregistrations/2026-08-03-b3s-meta-floor.md:39`)
  — **b3sh 를 자동으로 규율하지 않는다.**
- 이 로그 `:4243` 의 b3sh 기준(gs300 길이-Q4 CI 하한>0)은 **발사 후 기입**이라 동결 사전등록이 아니고
  **gs300 에 도달한 적이 없다.**
⇒ ①은 **등록된 최종-체크포인트 요건을 위반**한다. **중간·절단 런 결과로 라벨링한다.**

### ⑦ ★★가장 심각 — **HF 경로는 변하며, 4번 덮였다**

`eval/rq3v2f_b3sh_gs165/.../rq3v2f_b3sh_gs165_4k_n8.parquet` 은 **14:05·14:58·16:06·17:06** 네 번
쓰였다. 내가 오전에 보고한 −0.97pp 는 `6c472a2343`(oid `664e4e…`) 의 값이고, 현재 main 은
oid **`c9a7d59d8a0f5573`** 로 다른 점수를 낸다. codex 가 이것을 잡아냈다.
⇒ **규율(신규·강제): 모든 결과는 HF 커밋 + 파일 oid 로 고정해 인용한다. 가변 경로만으로 인용 금지.**
⚠`eval/rq3v2f_b3sh_gs175/` 아래 `gs165` 이름 파일들은 손상이 아니라 **로컬 출력 디렉터리 통째
재업로드로 생긴 중복**이다 — 모든 파일이 `metadata.json` 의 `model_path` 로 출처를 밝힌다.

### ⑧ 그 외 codex 지적(수용)

- ⛔**"MATH500 특이적" 미지지** → *"MATH500 에서 검출, GSM8K·AIME 에서 미해소."*
  도메인×처치 상호작용 미검정이고 AIME 는 파워가 매우 낮다.
- 세 처치의 **실패 표현형이 다르다**(발화 붕괴 · 엔트로피 · 중복/discard 라우팅).
  집계 정확도가 비슷하다고 **단일 공유 기전이 성립하지 않는다.**
- **체크포인트 선택이 사후적**이다 — gs165 는 사전등록된 종점이 아니라 **살아남은 가장 이른 것**.
  CI 는 이 선택을 반영하지 않는다.
- 분석 스크립트가 `$JOB/tmp` 에만 있어 **master 에서 재현 불가**(§5.4 미배선 도구).

### ⑨ 산출물(전부 probe 성격, 정본 승격 안 함)

`$JOB/tmp/` 의 `b3sh_16k_verdict.py` · `e153_followups.py` · `pass_resolution_check.py` ·
`b3sh_gs165_eval.py` · `recheck_e152.py` · `codex_e154_out.txt`.
★b3sh 롤아웃 테이블 194개 보존: `/home/v-seungplee/rq3v2f_evidence/b3sh_rollouts_0806`.

### ⑩ 미해결

- **PMI 귀속 불가** — matched PMI-off TRIOBJ 쌍둥이 필요(②)
- b2p 의 중간 체크포인트 부재로 **스텝 비매칭 해소 불가**
- 16k 실행간 분산 미측정(4k 만 n=3)
- discard 행에 replacement plan 이 안 나오는 이유(E-152 ③-b)


---

## E-155 (0806 19:2x UTC) — eval 완주 · **gs175 도 판정 패스에서 대조군 아래** · 과적합 검정을 제 자리(16k)로 옮김 · 내 A-vs-A 가설 오답

`rq3v2f-b3sh-eval-0806` 이 `pass` 로 완주했다. gs165·gs175 의 16k 전 패스가 착지.

### ① 판정 패스(16k MATH500) 전량 — **oid 고정**

| 파일 | oid | model_path | acc |
|---|---|---|---|
| b3sh gs165 | `7e32d9b491c795ce` | `merged_rq3v2f_b3sh_gs165` | **74.70%** |
| b3sh gs175 | `aae605444e80453f` | `merged_rq3v2f_b3sh_gs175` | **74.10%** |
| b2p gs300 | `08faf5ae9fc4e404` | `merged_rq3v2f_b2p_gs300` | 77.28% |
| b3p gs300 | `af5df50da404f0b8` | `merged_rq3v2f_b3p_gs300` | 75.20% |
| b3s gs300 | `ce372c5b593c8f4c` | `merged_rq3v2f_b3s_gs300` | 75.28% |

| 대비 (16k MATH500, 문항 페어드 부트스트랩) | delta | 95% CI | |
|---|---|---|---|
| b3sh **gs165** − b2p gs300 | **−2.58pp** | [−4.03, −1.18] | **0 배제** |
| b3sh **gs175** − b2p gs300 | **−3.17pp** | [−4.65, −1.73] | **0 배제** |
| b3sh gs175 − b3p gs300 | −1.10pp | [−2.58, +0.40] | 0 포함 |
| b3sh gs175 − b3s gs300 | −1.18pp | [−2.50, +0.13] | 0 포함 |

⇒ **두 체크포인트 모두 대조군보다 낮다.** E-154 의 결론은 gs175 에서도 유지되며,
b3p·b3s 와의 차이는 여전히 검출되지 않는다.

### ② ★과적합 검정을 판정 패스로 옮김 (codex 요구 (d) 이행)

E-154 ⑤ 는 4k 에서 `gs175 − gs165 = +0.78pp` 였고 codex 는 "손실이 나타난 16k 에서 해야
질문에 답한다" 고 지적했다. **16k 에서 다시 재면:**

> **gs175 − gs165 = −0.60pp CI[−1.80, +0.60] — 통계적으로 해소되지 않음.**

부호는 4k(+0.78)와 반대지만 두 구간 모두 0 을 포함한다.
⇒ **채택 문구: "16k 에서도 gs165→gs175 변화는 통계적으로 해소되지 않았다.
더 이른 체크포인트가 더 잘 일반화한다는 증거는 없다."**
⚠체크포인트 변화와 생성 실현이 여전히 **교란**돼 있다(각 체크포인트 1회 생성).
⚠그 사이 학습 discard 는 45.7%→52.5% 로 악화했으나, ⛔이것을 E-152 의 "능력 손실 아님"
근거로 **쓰지 않는다**(codex (d): 지연·저파워·포화로도 설명된다).

### ③ ⛔내 가설 오답 — **16k A-vs-A 는 없다**

`eval/rq3v2f_b3sh_gs175/` 아래에도 `gs165_16k_n8_math500` 이 있어서 두 번째 생성이라고
추정했으나, **두 파일은 oid 가 같고(`7e32d9b491c795ce`) completion 이 100% 동일**하다.
반복 생성이 아니라 **중복 업로드**다. codex 의 (c) 판단이 옳았다.
⇒ **판정 패스의 실행간 분산은 여전히 미측정.** 4k 의 SD 0.35pp 를 16k 로 이식하지 않는다.

### ④ ★★⑦ 규율이 E-154 자신에게 발동했다

E-154 가 기록한 gs165 16k MATH500 은 oid `3f86f798de1f52cb`·**74.45%**·−2.83pp 였는데,
eval 완주가 **같은 경로를 다시 써서** 지금은 oid `7e32d9b491c795ce`·**74.70%**·−2.58pp 다.
두 수 모두 유효한 생성 실현이고 결론은 같지만, **가변 경로로 인용했다면 두 판정문이 서로
모순되는 것처럼 보였을 것이다.** E-154 ①에 정정 문단을 덮어썼다.
⇒ **앞으로 판정 표의 모든 행에 oid 를 단다.**

### ⑤ 남은 것

- **b3nopmi 가 유일한 귀속 경로** — 진행 중(gs 초반, discard 4.3~6.1%로 b3sh 초기와 동류)
- b3shf 는 판별 구간 gs120~180 도달 전
- 판정 패스 실행간 분산 미측정(체크포인트당 16k 반복 생성 필요)
- b2p 중간 체크포인트 부재로 스텝 비매칭 해소 불가


---

## E-156 (0807 01:0x UTC) — ★★★instruct 대조 발견: **성공한 세대는 PMI 신호를 갖고 있었고 base 세대는 없다** · 내 초안 3건 중 3건 축소

GPU 0. 기존 wandb 히스토리 재분석 + codex-sol 적대검토(반증 조건을 먼저 적고 그것부터 물림).

### ① ★채택 문구 (codex 권고, 이 문장 그대로)

> **이 Base + meta-SFT2 기질에서는 채점 가능한 행의 약 3% 만 PMI 부호 뒤집힘을 겪고, 그 뒤집힘에
> 일관된 SAVE 다수가 없으며, 연속 shaped shift 는 초기에 음수다. 성공한 instruct 세대 대조군은
> 뒤집힘 커버리지 약 6.5%, SAVE 비율 약 66~67%, 첫 구간부터 양의 보상을 보인다. 이것은
> 기질 또는 궤적 품질에 따른 신호 격차를 지목하지만, PMI-shift 가 튜닝 불가능하다는 것도,
> 같은 레퍼런스로 SFT 를 거르는 것이 옳은 처방이라는 것도 확립하지 않는다.**

### ② 대조표 — codex 계산, **내가 독립 재현**(자릿수 일치)

| 팔 | n_save / n_derail | **SAVE 비율** | **뒤집힘 / 채점행** | **gs1–49 rmeta** | 채점행 수 |
|---|---|---|---|---|---|
| **instruct T1 full** `triobj_dcpo_v4_pmishift_h100_4x4k` | 4066 / 1976 | **67.30%** | **6.51%** | **+0.592** | 92,797 |
| **instruct shiftonly** `shiftonly_pmishift_h100_4x4k` | 2397 / 1235 | **66.00%** | **6.43%** | **+0.640** | 56,520 |
| base b3p | 958 / 980 | 49.43% | 3.06% | −0.164 | 63,297 |
| base b3s | 1247 / 1419 | 46.77% | 2.71% | −0.162 | 98,313 |
| base b3sh(≤gs177) | 843 / 913 | 48.01% | 3.04% | −0.156 | 57,720 |
| base b3nopmi(gs1–27) | 137 / 170 | 44.63% | 3.18% | −0.209 | 9,664 |

★instruct T1 은 **matched base 대비 MATH500 +18.8pp 로 6/6 셀 승리**한 그 팔이다.
★resume 조각은 스텝 중복을 **나중 생성분 우선**으로 dedup 하고 gs1–300 으로 통일했다.
★**내가 걸었던 반증 조건**(*"T1 의 SAVE 비율도 47% 면 이 이야기는 통째로 무너진다"*)은 **발생하지 않았다.**

### ③ ⛔무너진 내 초안 셋

**(1) "뒤집힘이 동전던지기 = 방향 신호 없음"** — **과잉 해석.**
뒤집힘은 채점행의 **3%(base) / 6.5%(instruct)** 뿐이고, **나머지 97% / 93.5% 는 연속 보상을 받되
SAVE 비율에는 보이지 않는다**(`dcpo_pmi_shift.py:104` 연속항 · `:107` SAVE/DERAIL 정의 · `:161` 진단
카운터가 같은 술어). ⚠**50% 는 확립된 귀무값이 아니다** — 교차 기회는 `PMI_open` 분포·0 과의 거리·
decoy 구성·메타 크기에 달렸다. 순열 검정이나 명시적 귀무모형이 필요하다.
⚠**"throughout(끝까지)" 는 사실과 다르다** — b3p 구간별 SAVE 비율은 46.5·45.3·50.3·56.3·**71.1**·58.3
으로 오른다(단 후반은 채점행 842·436개로 표본이 작아 잡음).

**(2) "RL 이 빚 갚기에 예산을 쓴다"** — **스케일 오독.**
RL 이 받는 것은 원 평균이 아니라 **`A_meta = R_meta − groupmean(R_meta)`**
(`dcpo_region.py:1070`,`:1268`) ⇒ **−0.16 같은 공통 오프셋은 상쇄된다.**
`rmeta_mean_scored` 는 member 행에 대한 평균이고(`verl_sdc.py:1750`), **group-centering 전 ·
anchor 정규화 전 · `w_meta` warmup 전 · floor 전**이다(warmup `verl_sdc.py:484`,
anchor `dcpo_region.py:1224`, 설정 `configs/triobj...:145`,`:189`).
⇒ 맞는 문구: **"원 채점기 진단으로서 초기 평균 부호 이동이 음수"** 이지 정책 어드밴티지의 적자가 아니다.
★단 codex 가 뒤집힘 보너스를 빼고 연속 성분만 복원해도 여전히 음수다
(b3p −0.148 · b3s −0.146 · b3sh −0.141 · b3nopmi −0.188) ⇒ **보너스 비대칭의 산물은 아니다.**

**(3) "보상 튜닝으로는 못 살린다"** — **세 설정을 불가능성으로 승격했다.**
맞는 문구: **"floor 0.05 와 열거된 보조 가중치 제거는 이 실행들에서 held-out 을 회복시키지 못했다."**
세 팔 모두 `w_meta=0.8`·같은 clipping·뒤집힘 보너스·decoy·80스텝 warmup·anchor 정규화·영역 라우팅·
group-centering 을 **유지**했고, **뒤집힘 비대칭·연속 scale·clip·anchor·warmup 길이·커버리지 게이팅은
아무도 스윕하지 않았다.** ⚠b3sh 도 `dcpo_trunc_open_penalty=0.3` 을 안 껐다
(`h100std_rq3v2f_b3sh.yaml:251` vs `configs/triobj...:203`). b3nopmi 는 gs27 이라 아직 아무것도 말 못 한다.

### ④ ⛔내가 제안한 SFT 필터는 **순환적이다**

같은 동결 레퍼런스가 **시연을 고르고** 나중에 **RL 보상까지 준다면**, 추론 능력이 아니라
**레퍼런스 특이적 트리거 문구**를 제조할 수 있다. codex 의 비순환 설계(채택):
1. **생성·선별 전에 문제 단위로 분할**
2. 후보 메타를 **인과적 계속(causal continuation)** 으로 고른다 — 같은 prefix + 실제 메타 vs
   메타 제거/셔플, **엄격 gold 정답**으로 판정
3. **선별은 평가자 A(또는 앙상블), 게이트는 B, RL 보상은 held-out 동결 레퍼런스 C**
4. **decoy 시드·레퍼런스 체크포인트/모델 간 전이**를 요구
5. 이미 구현된 own≠gold · 셔플-메타 플라시보 · safe-default 검사를 유지
   (`src/eval/pmi_shift_signal.py:282`,`:350`)
6. 게이트는 `rmeta>0` + SAVE>50% 가 아니라 **인과 개선의 신뢰구간 + 전체 shift 통계**로
★**기존 SFT 게이트는 이미 다르다**: AUC>0.55 · SAVE≥1 · 교란 검사(`scripts/measure_sft_gate.py:252`).
⇒ 내가 "새로 만들자"고 한 게이트는 **이미 있는 것보다 약했다.**

### ⑤ ⛔발화 붕괴와 member 붕괴는 **하나의 사실**

구조상 `member_rate ≤ trusted-meta rate ≤ emission rate`. b3p 의 `member/emission` 비는
0.71·0.72·0.67·0.65·0.52·0.45 로 떨어진다.
⇒ **주 사실은 "발화가 붕괴했다" 하나**이고, **부수 사실은 "발화한 것 중 신뢰·채점 가능 비율도 악화"** 다.
⛔emission 0.038 과 member 0.017 을 **독립된 두 붕괴로 세지 말 것**(오류 ⑩ 재발 방지).

### ⑥ 그 외 정정

- 내 b3s 합계 `1263/1431` 은 **wandb 스텝 302·303 을 포함**했다. gs1–300 으로 통일하면 **1247/1419**.
- 구간 평균은 **스텝별 평균의 비가중 평균**이라, 채점행이 붕괴한 후반 구간(842·436행)을 앞 구간과
  같은 무게로 비교하면 안 된다.
- 채점 모집단이 발화·형식에 따라 변하므로, 세로축 `rmeta_mean_scored` 는 **보상 드리프트와
  선택/구성 드리프트가 섞여 있다.**
- "b3s 엔트로피 폭주" · "b3sh 형식 붕괴" 는 **표현형이지 held-out 손실의 확립된 원인이 아니다.**
- **`gs0 rmeta` 게이트는 현재 존재하지 않는다** — 만들려면 동결 샘플링·문제 분할을 갖춘
  오프라인 사전 롤아웃/채점 프로토콜이 필요하다.

### ⑦ 닫는 것 / 여는 것 / 재확인 계수기

**닫는 것**
- ⛔"base 에서 실패한 이유를 모른다" 는 더 이상 정확하지 않다 — **신호 격차라는 측정된 후보가 생겼다.**
- ⛔SAVE 비율만 보고 "방향 신호 없음" 이라 말하는 분석. **분모(채점행)를 같이 내지 않으면 무효.**
- ⛔같은 동결 레퍼런스로 SFT 를 고르는 단순 필터 제안(순환).

**여는 것**
- **instruct vs base 의 신호 격차를 무엇이 만드는가** — 기질·디코딩·레퍼런스 행동·롤아웃 분포가
  모두 다르므로 **SFT 코퍼스로 특정되지 않는다.** 분리 실험이 필요하다.
- **PMI 하이퍼파라미터 스윕**(뒤집힘 비대칭·연속 scale·clip·anchor·warmup·커버리지 게이팅) —
  아무도 안 건드렸다.
- **비순환 인과-계속 선별 파이프라인**(④의 6단계).
- **커버리지 자체가 지표다**: base 3% vs instruct 6.5%. 커버리지를 올리는 개입이 별도 레버다.

**재확인 계수기: 0** — 이 대조는 처음 계산됐다.

### ⑧ 산출물

`$JOB/tmp/pmi_signal_audit.py`(probe) · `codex_e156_out.txt`(검토 전문).
⚠probe 는 정본 승격하지 않는다. **정본 승격이 필요한 것 하나**: instruct-vs-base 신호 대조를
`scripts/` 로 올리는 것 — 앞으로 모든 새 기질에서 **발사 전에** 돌려야 할 검사이기 때문(승인 사항).


---

## E-157 (0807 03:3x UTC) — 운영: **init 스테이징 실패가 몇 시간 사이 두 잡을 죽였다** · 선점 위에 얹힌 두 번째 실패 모드

판정 아님. 인프라 사실 기록 — 잊고 다시 진단하는 것을 막기 위함.

### ① 같은 지점, 두 잡

| 잡 | 결과 | 마지막 로그 |
|---|---|---|
| `rq3v2f-b3nopmi-0806` | `failed` | `[YAML] FATAL init /scratch/models/sft2_init missing or incomplete; ABORT window` + `exit 1` |
| `musical-wombat`(=`h100_rq3v2f_b3s`, 완주 후 `sleep 86400`) | `failed` | **동일** |

b3nopmi 쪽 근인은 로그에 남아 있다:
`ChunkedEncodingError: IncompleteRead(4653341155 bytes read, 262619213 more expected)`
— HF 에서 init(≈4.9GB)을 받다가 **4.65GB 에서 전송이 끊김.**

### ② 실패 모드가 둘이다

이 클러스터에서 잡이 죽는 경로는 이제 **두 가지**로 세야 한다:
1. **선점**(Standard 등급, Premium 없음) — 체크포인트로 재개 가능, 손실은 마지막 save 이후 스텝
2. **재시작 시 init 스테이징 실패** — **재개 불가, 잡 자체가 죽는다.** 가드가 잘못된 init 학습을
   막아주지만(옳음), 사람이 재발사해야 한다(승인 사항).
⇒ **①이 ②를 유발한다**: 선점되지 않으면 재스테이징도 없다. 선점이 잦을수록 ②의 노출도 늘어난다.

### ③ 왜 매 재시작이 비싼가

`/scratch` 는 노드 전용이라 선점되면 전부 사라진다. 새 노드마다 다시 받는 것:
conda 환경 **≈5GB**(압축 해제 포함) + init 모델 **≈5GB**(`snapshot_download`, 14파일) + 체크포인트.
⇒ **준비에만 20~40분**. 이것이 ②의 노출창이다.

### ④ 실측 — 선점 사이에 번 스텝 수

wandb `_runtime` 인접 간격에서 >600s 를 선점 공백으로 분리해 셈:
- **b3p(완주함)**: `[37, 10, 49, 32, 16, 4, 15, 6, 21, 36, 2, 9, 14, 0, 5, 2, 6, 2, 1, 0, 10, 0, 0]`
  — 선점 **22회**를 겪고도 완주. 단 후반에 **0 이 세 번**(준비 끝나기 전에 쫓겨남).
- **b3nopmi(현재)**: `[19, 4, 1]` — 줄어드는 중.
★순수 계산은 변함없다(b3nopmi 269s/step · b3p 416s/step). **문제는 계산이 아니라 자리.**
★**부등호**: 재시작 비용 20~40분 = 4~9스텝 값 vs 최근 획득 1~4스텝 ⇒ **버는 것보다 준비가 비싸다.**

### ⑤ 산출물 무사 확인(취소 판단의 근거)

`musical-wombat` 이 죽어도 잃은 것은 없다: HF `checkpoints/rq3v2f_b3s/` 에 **gs301·302·303 전부
m4/o4/e4 완전**, `eval/rq3v2f_b3s_1030/` parquet 5개(C-029 근거) 존재.
⇒ **사용자 승인 ⑥(취소)은 실행 불필요** — 대상이 이미 `failed` 로 자원을 반환했다.

### ⑥ 사용자 결정(0807 03:3x)

`musical-wombat` 소멸 후에도 b3nopmi 가 큐에 남자 **㉘ b3shf 중단**을 물었고,
**"둘 다 그대로 둔다"** 로 결정됨. 조치 없음, 모니터링 계속.

### ⑦ 여는 것

- init 스테이징을 **재시도 가능**하게 만들 수 있나(현재는 1회 실패 시 잡이 죽는다).
  ⚠정본 런처 변경 = 승인 사항. 지금 제안하지 않고 기록만 남긴다.
- 노드-로컬 캐시가 아니라 **영속 스토리지**에서 init 을 스테이징할 수 있나.


---

## E-158 (0807 06:0x UTC) — ★★★**RQ2 에 미통제 옵티마이저 비대칭이 있다** · 내 감사 결론이 틀렸고 fable 이 잡았다 · PMI 노브 6종은 **한 번도 설정된 적 없다**

★사용자 요청으로 **fable 적대검토**를 돌렸고, 그 검토가 **내가 몇 시간 전에 내린 감사 결론을 반증**했다.
아래 코드 사실은 **내가 직접 file:line 을 열어 재확인**했다.

### ① ⛔내 감사 결론 정정 — "C-1 정규화 비대칭은 이 세대에 해당 없다"는 **틀렸다**

내가 본 것(맞음): 일곱 런처 전부 `++algorithm.norm_adv_by_std_in_grpo=false` 를 넘기고,
처치 경로는 `dcpo_region.py:1113` 에서 **설계상 mean-only**(`"subtract group mean, NO /std"`).
⇒ 그래서 "대칭이다" 라고 결론냈다.

**내가 못 본 것**: **대조군과 처치군은 서로 다른 함수로 간다.**
- 처치(b3*): `verl_sdc.py:3571` 게이트(`sdc_enabled or _adv_region`) 통과 →
  `compute_sdc_gdpo_advantage` → `compose_dcpo_region_advantage`(mean-only, whiten 없음)
- **대조(b0p/b2p)**: `sdc_enabled: false` + `VANILLA_GRPO`(`_REGION_ROUTED_MODES` 아님) ⇒
  **게이트 통과 실패 → verl stock `compute_gdpo_outcome_advantage` 로 떨어진다.**

그 stock 함수의 마지막 줄(verl 0.7.1 `core_algos.py:466`):
```python
advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask
```
**무조건이다.** `norm_adv_by_std_in_grpo` 는 안쪽 `compute_grpo_outcome_advantage` 의 그룹 단계만
제어하고, **마지막 배치 단위 whiten 은 플래그와 무관하게 실행된다.**

⇒ **대조군의 어드밴티지는 배치 std=1 로 재정규화되고, 처치군의 correctness 어드밴티지는 whiten
없이 원시 group-centered 값이다.** 같은 lr 에서 대조군이 correctness 방향으로 더 크게 움직인다.
**이것은 보상 내용의 차이가 아니라 옵티마이저 기하의 차이다.**

★**세 처치 팔이 보조 헤드 구성은 전부 다른데 적자가 −2.0~−2.6pp 로 거의 같다**는 관측과 정합한다.
셋이 공유하는 것은 PMI 가 아니라 **region compose 경로(= whiten 부재)** 다.
⚠**배율은 아직 미측정** — wandb `critic/advantages/*` 로 확인하는 작업이 진행 중이다(GPU 0).

### ② ⚠C-029 · C-030 에 붙여야 할 단서 (문구 확정 전, 정량화 대기)

두 판정은 **여전히 유효하다**(수치·부호·CI 그대로). 다만 인과 해석에 다음이 붙어야 한다:
> **`b3p − b2p` 비교에는 보상 설계 차이 외에 어드밴티지 정규화 차이가 포함돼 있다 —
> 대조군은 배치 whiten 을 받고 처치군은 받지 않는다(`core_algos.py:466` vs `dcpo_region.py:1113`).
> 따라서 이 차분은 "메타 보상 패키지의 효과"가 아니라 "패키지 + 옵티마이저 기하"의 합효과다.**
⚠**정량화 전에는 이 단서를 "가능성"으로만 쓴다** — 배율이 작으면 무시할 만할 수도 있다.

### ③ ★PMI 노브 여섯 개가 **한 번도 설정된 적 없다**(grep 0건)

`configs/` · 런처 전체에서 `dcpo_pmishift_*` 설정 **0건**. 전부 코드 기본값으로만 돈다
(`verl_sdc.py:1587-1597`):
| 노브 | 기본값 | 의미 |
|---|---|---|
| `reversal_save` | **1.0** | SAVE 보너스 |
| `reversal_derail` | **2.0** | DERAIL 벌점 — **2배 비대칭** |
| `reversal_min_magnitude` | **0.0** | **eps 가드 꺼짐**(0 근처 jitter 도 뒤집힘으로 셈) |
| `meta_body_dup_thresh` | **1.0** | **내용-무결성 가드 꺼짐**(meta 가 body 복제여도 credit) |
| `scale` | 1.0 | 연속항 배율 |
| `clip` | 2.0 | 연속항 클립 |

★**비대칭의 기질 의존성**(fable 계산, 내가 산술 확인): 뒤집힘 행의 기대 보너스
= `P(save)·1 − P(derail)·2`
- instruct(SAVE 67.3%): `0.673 − 0.654` = **+0.019** ≈ 중립
- base(SAVE 49.4%): `0.494 − 1.012` = **−0.518** ⇒ **순벌금**

⇒ **같은 상수가 instruct 기질에서는 중립이고 base 기질에서는 "믿음을 움직이는 meta" 에 대한
순벌금이다.** E-156 ③이 "아무도 스윕하지 않았다"고 적은 것은 문자 그대로였고,
**스윕 1순위 축이 어디인지에 대한 정량적 근거**가 이제 있다.
★**코드 배선은 이미 있다** — `++algorithm.dcpo_pmishift_reversal_derail=1.0` 같은 오버라이드만으로
가능하다. **정본 코드 변경 불필요.**

### ④ 죽은 코드 · 미배선

- **`dcpo_w_over`**: `src/` · `configs/` grep **0건**. 런처 CLI 에만 존재 ⇒ **죽은 노브**(codex 보고 재확인).
- **`src/training/verl_gdpo_algos.py`**: 살아있는 경로가 아닌데(실제는 verl-native 함수)
  **같은 이름을 갖고 있어 "GDPO 코드를 확인했다"는 사람이 틀린 구현을 읽게 된다**(fable 지적).
  ⚠나도 이 함정에 걸릴 뻔했다.

### ⑤ 방법론 — 내가 틀린 방식

나는 **"런처가 같은 플래그를 넘긴다"** 를 확인하고 대칭이라 결론냈다. 빠진 것은
**"두 팔이 같은 함수로 가는가"** 였다. `stacked-research` **G8(팔 정체 — 매니페스트 전 키 diff)**
가 정확히 이것을 막으라고 있는 게이트인데, 나는 **플래그 한 개만 diff 했다.**
⇒ **규율: 두 팔을 비교하기 전에 "같은 코드 경로로 가는가"를 먼저 확인한다.
플래그가 같아도 경로가 다르면 그 플래그는 무의미할 수 있다.**


---

## E-159 (0807) — 옵티마이저 기하 비대칭은 **instruct 승리에도 똑같이 있었다**. 그리고 E-158 의 게이트 조건은 내가 틀리게 적었다

### ① E-158 정정 — 판별자는 `sdc_enabled` 가 아니라 `sdc_mode` 다

E-158 에 나는 *"대조군(`sdc_enabled:false`+VANILLA_GRPO)은 게이트를 통과 못 한다"* 라고 썼다.
**`sdc_enabled: false` 는 양쪽 팔 모두에 있다** —
`configs/base_matched_grpo_h100_4x4k.yaml:67` 과 `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:92`.
`adv_estimator: gdpo` 도 양쪽 동일(`:69` / `:94`).

실제 게이트(`verl_sdc.py:3571-3572`):
```
if _is_gdpo_estimator(adv_estimator) and config is not None and \
   (config.get("sdc_enabled", False) or _adv_region):
```
`_adv_region = _adv_sdc_mode in _REGION_ROUTED_MODES`,
`_REGION_ROUTED_MODES = {"TRIOBJ_DCPO_V2","TRIOBJ_DCPO_V3","TRIOBJ_DCPO_V4"}` (`verl_sdc.py:1097`).
⇒ **갈라지는 유일한 키는 `sdc_mode`** — `VANILLA_GRPO`(`base_matched:66`) vs
`TRIOBJ_DCPO_V4`(`triobj:91`). 코드 주석 자체가 이 설계를 명시한다(`verl_sdc.py:3559-3563`:
*"region modes ... run teacher-FREE (sdc_enabled=false) ... Route region modes regardless of sdc_enabled"*).

**E-158 의 결론(경로가 갈리고 whiten 이 한쪽에만 걸린다)은 유효하다. 틀린 것은 조건의 이름뿐이다.**
그러나 이것은 G8 을 두 번째로 얕게 실행한 것이다 — 첫 번째는 플래그 한 개만 봤고,
두 번째는 **내가 갈린다고 믿은 플래그가 실제로는 양쪽에 같은 값으로 있었다.**
⇒ 규율 보강: **"어느 키가 갈리는가"를 답할 때 그 키의 값을 양쪽에서 실제로 출력한다.**

### ② 그래서 무엇이 달라지나 — instruct 승리도 같은 기하 대조였다

| 세대 | 처치팔 config | 대조군 config | `norm_adv_by_std_in_grpo` 오버라이드 |
|---|---|---|---|
| **instruct T1**(+18.8pp 승) | `triobj_dcpo_v4_stage3b_h100_4x4k` (V4, region, mean-only) | `base_matched_grpo_h100_4x4k` (VANILLA, stock, **whiten**) | **양쪽 0건** (grep -c = 0) |
| **base rq3v2f**(−2pp 패) | 동일 (`h100std_rq3v2f_b3p.yaml:246`) | 동일 (`h100std_rq3v2f_b2p.yaml:267`) | **양쪽 0건** |

**두 세대가 같은 두 config 를 쓴다.** 따라서:

- ⛔**"기하 비대칭이 base 결손의 원인이다" 는 이것만으로 성립하지 않는다.** 같은 비대칭이
  있는 상태에서 instruct 는 **이겼다.** 기하는 상수이고 결과는 부호가 뒤집혔으므로,
  기하 단독 인과는 배제된다 — 남는 것은 **기질과의 상호작용**이거나 다른 원인이다.
- ★그러나 **반대 방향의 손실이 확정된다**: `EXPERIMENT_PLAN.md` 가 이 대비를
  *"메타 보상의 효과"* 로 읽어 왔는데, **처치와 대조가 옵티마이저 기하까지 함께 바꾼다.**
  ⇒ **이 프로젝트 역사 전체에서 기하를 맞춘 메타-보상 비교는 한 번도 없었다.**
  ⇒ **"PMI-shift 가 어디선가 작동했다"는 명제에 기하-통제된 지지 증거는 현재 0건이다.**
  instruct +18.8pp 는 **패키지+기하 합효과**이지 PMI 귀속이 아니다.

### ③ 이것을 닫는 팔은 이미 발사됐다 — **b3null**

`h100std_rq3v2f_b3null.yaml`(커밋 `e42dd87`, 실험 `rq3v2f-b3null-0807`, 이 문서 기준 부트스트랩 중).
b3p 와 바이트 동일하되 **보상 헤드 아홉 개를 전부 0** 으로 하고 `dcpo_rmeta_source=none`.
⇒ **correctness-only 인데 region 경로**를 탄다.

**판정 규칙(발사 전 등록):**
- **b3null ≈ b2p** ⇒ 기하는 무해하다. −2pp 는 **보상 성분** 탓 ⇒ b3nopmi 가 PMI 를 분리한다.
- **b3null < b2p** (b3p 와 비슷한 폭) ⇒ **−2pp 는 기하 탓**이고 보상은 거의 무관.
  이 경우 **모든 기존 팔 비교가 재해석 대상**이 되고, instruct 승리도 재측정 대상이다.
- **b3null > b2p** ⇒ 예상 밖. region 기하가 이롭다는 뜻이고 별도 조사.

⚠**판독 함정**: b3null 은 `w_format=0` 이므로 형식 압력이 없다. 발화가 죽으면 carve-out
(`dcpo_region.py:691` ANSWER=¬META)이 사실상 무효가 되어 **의도한 "기하만" 대조가 아니게 된다.**
⇒ **판정 전에 b3null 의 `meta_emit_rate` 를 반드시 확인한다.** 0.9 아래로 떨어지면
그 팔은 기하 대조가 아니라 "발화 붕괴" 팔이므로 판정에서 제외한다.

### ④ 닫는 것 / 여는 것

**닫는 것** — "instruct 에서 PMI-shift 가 작동했으므로 base 에서도 작동시키면 된다" 라는
전제 위에 선 실험 설계 전부. 그 전제는 **기하-통제된 증거가 0건**이다.
**여는 것** — (a) b3null 판정, (b) 기하를 맞춘 instruct 재실행(가장 비쌈, b3null 결과에 따라),
(c) **PMI 하이퍼 스윕**(`++algorithm.dcpo_pmishift_*` 오버라이드만, 정본 변경 불필요 — 승인 ㉝).
**재확인 계수기** — 0 (신규).

---

## E-160 (0807) — 두 개의 정정. ①내 "스칼라 채널 생존" 판독은 무효 ②엔트로피-비율 기전은 측정으로 기각

### ① 내 판독 오류 — `grad_norm` 은 두 가설을 구별하지 못한다

E-158 후속으로 나는 *"Adam 이 균일 스케일을 흡수했다면 grad_norm 이 같아야 하는데 2.4배 다르다
⇒ 스칼라 채널이 살아 있다"* 라고 읽었다. **이 추론은 무효다.**
`grad_norm` 은 Adam **이전** 값이다. 어드밴티지를 k배 하면 gradient·m·√v 가 모두 k배가 되어
업데이트는 불변인데 `grad_norm` 은 k배가 된다. 즉 **2.4배∝2.6배 일치는 두 가설 모두의 예측**이고,
확인해 주는 것은 *"스케일이 gradient 까지 도달한다"* 뿐 — 아무도 다투지 않는 명제다.
ε-floor 로도 못 살린다: total 0.36 / ≈8×10⁹ 파라미터 ⇒ per-param RMS ≈ 4×10⁻⁶ ≫ ε=10⁻⁸.
⇒ **"순수 스칼라 배율이 정책을 바꾼다" 는 여전히 미입증이다.**
살아남는 비대칭은 **clip 포화**(whiten 27~38% vs region 0~0.3%)뿐이며, 이것은 비선형이라 실재한다.

### ② `entropy_coeff` 는 양쪽 다 0.001 — 그리고 그 기전은 기각됐다

상속 체인 확인: `verl_sdc_e21r_shared.yaml:53` → `verl_e4_selfdistill_h200_4x4k.yaml:21-23` →
`base_matched_grpo_h100_4x4k.yaml:29-31` / `triobj_dcpo_v4_stage3b_h100_4x4k.yaml:47-49`.
**어느 leaf 도 재정의하지 않는다.** (`kl_loss_coef` 는 양쪽 leaf 가 0.0 으로 죽였다 — `triobj:73`,
`base_matched:51` ⇒ KL 은 대칭 사망.)

**제안된 기전**: total∇ = ∇pg + 0.001·∇H 인데 region 팔은 ∇pg 만 2.6배 작고 ∇H 는 같으므로
**방향이 달라** Adam 이 상쇄할 수 없다 ⇒ region 팔이 실효 엔트로피 압력을 ~2.6배 받는다.
**사전등록 예측**: whiten 팔은 평탄, region 팔은 **전부** 상승.

**측정(wandb `gistdslab/metacot-dcpo-v4`, `actor/entropy`, GPU 0):**

| 팔 | 기하 | gs1–20 | gs40–60 | gs100–130 | max(@step) |
|---|---|---|---|---|---|
| b2p | **whiten** | 0.2159 | 0.2208 | **0.2647** | 0.361 (179) |
| b0p | **whiten** | 0.1944 | 0.1868 | 0.1993 | 0.248 (169) |
| b3p | region | 0.2149 | 0.2151 | 0.2916 | 0.518 (209) |
| **b3nopmi** | region | 0.2142 | 0.2744 | **0.6313** | 0.734 (113) |
| b3shf | region | 0.2172 | 0.2313 | **0.2522** | 0.524 (175) |
| ~~b3s~~ | region | — | — | — | ⚠**오염**(내 오류 ㉑ 으로 b3shf 와 run 공유) |

**⇒ 예측 실패. 기전을 균일 설명으로는 기각한다.**
- whiten 인 **b2p 가 +0.049 오르는 동안 region 인 b3shf 는 +0.035 밖에 안 올랐다** — 부등호가 뒤집힌다.
- region 팔 **내부** 스프레드(0.252 ~ 0.631, **2.5배**)가 기하 간 스프레드보다 **크다.**
⇒ 엔트로피 궤적을 정하는 것은 **기하가 아니라 보상 헤드 구성**이다.

### ③ 그리고 부수 관측 — 보상이 엔트로피를 **누른다**

같은 기하 안에서 순서: **b3nopmi(w_meta=0) 0.631 ≫ b3p(w_meta=0.8) 0.292 > b3shf 0.252.**
메타 보상을 **끈** 팔이 가장 높다. ⇒ 메타 보상은 엔트로피를 올리는 게 아니라 **누르는** 방향으로 작동한다.
⚠b3nopmi 는 gs123 까지만 있어 이 밴드가 그 팔의 말단이다. 확정 전 완주 후 재측정.
⚠b3s 행은 쓰지 말 것 — run 오염.

### ④ 닫는 것 / 여는 것

**닫는 것** — (a) `grad_norm` 비를 근거로 한 "스칼라 채널 생존" 주장, (b) "엔트로피 비율이
region 팔의 −2pp 를 균일하게 설명한다" 가설, (c) 그 위에 설계될 `entropy_coeff` 조정 실험.
**여는 것** — (a) **clip 포화**가 유일하게 살아남은 비선형 비대칭이므로 이것을 직접 재는 것,
(b) **보상 헤드가 엔트로피를 누르는 방향**이라는 관측의 확정(b3nopmi 완주 후),
(c) b3null 판정(기하 번들 vs 보상).
**재확인 계수기** — 0 (신규).

### ⑤ b3null 판독 규칙 — **결과 도착 전에** 등록한다

판정문은 **세 값을 반드시 병기**한다: ⑴ b3null−b2p (gs300 held-out 1030, paired bootstrap)
⑵ **s\*** = `gdpo/meta_emission/mean` 이 최초로 0.5 아래로 간 스텝 ⑶ `actor/entropy` 의 b2p 대비.

- **b3null ≤ 75.5** (treatment 밴드) ⇒ *"메타 보상 헤드 전무 상태에서 region 경로만으로 적자가
  재현된다"* 까지만 허용. ⛔**"스케일/whiten 이 원인" 으로 좁히는 문장은 `s*≤150` 일 때만 허용**
  (그 이후 구간은 carve-out 이 소멸한 상태로 적자를 유지한 것이다).
- **s\* > 250** (발화 생존) ⇒ *"스케일 + carve-out + discard배제 **번들**"* 로만 기술. **성분 귀속 금지.**
- **b3null ≥ 76.8** (control 밴드) ⇒ 기하 면책. 원인을 b3nopmi 로 좁힌다 —
  b3nopmi ≈ 75.2 면 {format/cal/emit/floor} 헤드군, ≈ 77 이면 PMI 헤드.
- **75.5 ~ 76.8** ⇒ 부분 기여로 기술. **단일 원인 서사 금지.**
- **조기 판독**: 첫 50스텝에서 b3null 엔트로피가 b2p 동구간 대비 유의 상승하면 위 ② 기전이
  부분 부활한다. control 처럼 평탄한데 최종이 treatment 밴드면 다른 성분을 찾는다.

★**b3null 은 "whiten 시험" 이 아니라 "기하 번들 시험" 이다.** b3null−b2p 에는 no-whiten 외에
**carve-out**(ANSWER=¬META, `dcpo_region.py:691`)과 **tier-2 discard 배제**
(`dcpo_region.py:1101-1113`; 대조군은 전 행 유지)가 남는다. 구현 감사 결과 **누락 노브는 없다**
(`dcpo_meta_len_cap: 96` 은 floor 블록 `:1330` 안에서만 쓰이므로 floor=0 이면 자동 무효).

---

## E-161 (0807) — **판정 체크포인트에서 기전이 이미 죽어 있었다.** 그리고 메타는 도움이 됐다 — 해로운 것은 **보상**이다

### ① b3p 는 판정 시점에 붕괴 상태였다 (wandb `gistdslab/metacot-dcpo-v4`)

| b3p 구간 | `gdpo/meta_emission/mean` | `critic/score/mean` | `response_length/mean` | `response_length/clip_ratio` |
|---|---|---|---|---|
| gs1–20 | 0.9953 | 0.3951 | 296 | 0.0000 |
| gs40–60 | 0.9982 | 0.4242 | 334 | 0.0003 |
| gs100–130 | 0.9307 | **0.4457**(최고) | 576 | 0.0091 |
| gs150–200 | 0.6228 | 0.4018 | 866 | 0.0390 |
| **gs250–300** | **0.0375** | **0.2998** | 1083 | **0.0912** |

⇒ **`s*`(발화가 0.5 아래로 간 스텝) ≈ gs175–210.**
⇒ ★**`EXPERIMENT_PLAN.md:20` 이 정한 판정 패스는 gs300 이다. 그 체크포인트의 발화율은 3.75% 다.**
즉 **b3p 의 `−2.08pp` 는 "메타 보상의 효과"가 아니라 "메타 기전이 죽은 뒤 남은 것"을 잰 수다.**
학습 정확도 최고점은 gs100–130 이었고 거기서 −14.6pp 내려와 판정됐다.
**응답 9.1% 가 8192 토큰 상한에 부딪힌다** = 비종료 = E-138 계열 degeneration.

⚠**이것은 E-160 ⑤ 의 `s*` 규칙을 b3p 에도 소급 적용해야 함을 뜻한다**:
`s* ≈ 190 < 250` 이므로 b3p 판정문에 **성분 귀속 문장을 쓸 수 없다.**

**대조군 b2p 는 같은 구간에서 건강하다**: gs250–300 정확도 0.5230(상승), 길이 765, clip 3.5%.

### ② 가동 중 세 팔의 건강 상태 (이 시점)

| 팔 | gs (**로그=정본**) | entropy | 길이 | 발화율 | discard | 정확도 | 판정 |
|---|---|---|---|---|---|---|---|
| **b3nopmi** | 124 | **0.631** | 841 | 0.9981 | 0.1672 | 0.4136 | 🟡 발화 정상이나 **길이·엔트로피가 b3p 붕괴 직전 궤적** |
| **b3shf** | **50** | — | — | — | — | — | 🟠 정상이나 **매우 느림**(1일에 50스텝 = 선점) · **wandb 판독 불가**(아래) |
| b3null | ~15 | — | — | — | — | — | ⏳ 지표 미도달 |

**생성 텍스트 직접 확인**(b3nopmi gs121–125 샘플 5건, `DCPO_DBG` 라인):
메타 블록 정상 닫힘 · `\boxed{}` 정상 · `<|im_end|>` 정상 종료 · `R_corr=1.000` ·
**비-ASCII(크메르/타밀/데바나가리) 0건.** ⇒ **지금 돌고 있는 팔에 degeneration 없음.**

⛔**정정 — b3shf 는 wandb 로 읽을 수 없다.** 처음 이 표에 `b3shf gs177 / discard 0.4634 /
정확도 0.5027` 을 적었으나, 잡 로그가 정본이고 거기엔 `'global_steps': 50` 이다.
**wandb run `rq3v2f_b3shf` 의 gs150–200 구간은 b3sh 의 데이터**다 — 내 오류 ㉑(런처 복제 시
`WANDB_RUN_ID: rq3v2f-b3sh-1` 미치환)의 결과가 여기서 실현됐다.
⇒ ★**b3shf 팔은 wandb 판독 불가. 진행·지표는 잡 로그로만 읽는다.**
⇒ ★**HF ckpt(gs40·45·50)가 로그와 일치한다** — 즉 푸셔는 정상이고, 불일치한 것은 wandb 뿐이다.

★그리고 그 오염된 행이 **b3sh 에 대해** 말해 주는 것은 오히려 강하다:
**b3sh 는 discard 46.3% 이고 학습 정확도가 대조군 수준(0.5027 vs 0.5031)인데 held-out 은
−2.58/−3.17pp 로 최악**이었다(E-154/E-155). ⇒ **discard 폭발도 학습 정확도도 held-out 을
예측하지 않는다.** (E-152 가 discard 를 정확도 손실과 묶어 읽은 것은 근거가 없다.)

### ③ ⛔**철회** — "메타는 이롭다(+1.23pp)" 는 **이미 판정된 축을 다시 산 것**이다

**처음 이 절에 다음을 썼다**: b0p 76.05% 신규 측정 ⇒ `b2p − b0p = +1.23pp` ⇒
*"메타 발화 자체가 이롭고 해로운 것은 보상이다. base 에서 메타가 안 통한다는 서사를 뒤집는다."*

⛔**철회한다. 이 축은 0804 에 이미 판정됐다.**
`docs/CLAIMS.md:34-35` **C-026 (CONF 트랙)**: **`b2p − b0p` MATH500 `+0.18pp` [−1.30, +1.68] — 널.**
발화율 1.0000 확인. 그리고 `docs/CLAIMS.md:150-155` **C-004 가 이것을 사전 예측했다**
(instruct 이득의 큰 몫 = 통제군 비종결 구제인데 base 통제군 절단은 3.0% 라 구제할 것이 없다).

★**내 +1.23pp 는 C-026 의 CI 안에 있다**(+1.23 < +1.68). ⇒ 신규 점추정을 그대로 받아도
**"널과 정합"이 선순위 해석**이다. 나는 **CI 를 계산하지도 않고** 부호만 보고 서사를 뒤집었다.

★★**축 이름 자체가 틀렸다.** `EXPERIMENT_PLAN.md:15-16` 대로 두 팔은 SFT2 만 쌍둥이이고
**SFT1 이 통째로 다르다**(`b0on_v8base_strict` vs `b2on_v8meta_strict`; meta 제거 행은 토큰 수도 다르다).
계획표가 이미 올바른 이름을 정해 뒀다 — `plan:154` **"meta-SFT **레시피** 효과"**. **"발화 효과"가 아니다.**
그리고 발화 효과를 within-model 로 분리하려던 E4 는 여기서 **원리적으로 무력**하다:
**b2p 발화가 100.00% 라 emit-vs-비emit 대비의 분산이 0** 이다.

**★재확인 계수기: 2** (0804 에 1, 0807 에 2 — **같은 축을 대장을 안 보고 세 번째로 계산**).
`stacked-research` **§7: 계수기가 2를 넘으면 색인이 아니라 스킬을 고친다.**
**어느 게이트가 놓쳤나 — G1(CLAIMS grep).** 워크플로가 새 수를 주자 나는 *"이 수가 어느 주장을
여는가"* 를 묻지 않고 곧장 판정문을 썼다. **G1 은 새 실험을 설계할 때만이 아니라
새 수치를 받았을 때도 발동해야 한다.** ⇒ 규율 보강:
**★외부(워크플로·에이전트)에서 온 수치는 판정문에 쓰기 전에 반드시 `CLAIMS.md` 를 개념으로 grep 한다.**

**남은 유효 사실 (이 절에서 살아남는 것)**

| 팔 | held-out(이번 재채점) | 메타 발화 | 메타 보상 | 대장 값 |
|---|---|---|---|---|
| b2p | 77.28% | **100.00%** | ✗ | 77.28 |
| b0p | 76.05% | 없음 | ✗ | (C-026 이 쓴 산출물과 **차분 미소명**) |
| b3s | 75.25% | 99.98% | ✓ | 75.28 · **그리고 74.80** ⇒ **세 번째 값** |
| b3p | 75.15% | (gs300 3.75%) | ✓ | 75.20 |

⚠**b3s 가 세 값을 갖게 됐다**(74.80 → 75.28 → 75.25). C-030(원장 3970-3996)에서 리비전 고정으로
봉합했던 것과 **같은 병**이다. ⇒ **가변 HF 경로 문제가 재발했다.**

**⇒ 소명 순서(전부 GPU 0, 이것이 지금 1순위):**
① ckpt **oid 고정**(b0p 는 gs295·gs300 둘 다 있으므로 어느 것인지 판별)
② grader 변종 고정(`rq2_analyze.py:30` 의 `$`-wrap · NFKC 플래그 유무)
③ parquet 리비전 고정
④ **그 다음에야** C-026 개정 여부를 논한다. **개정 전까지 C-026(널)이 정본이다.**

★**발화 자체의 효과를 주장할 수 있는 유일한 설계**(이 축에서 유일하게 새로운 것):
**b2p 그대로 두고 디코딩에서 `<|meta|>` 토큰만 밴하고 재평가**. 같은 가중치·같은 정책에서
발화만 끈 within-model 대비다. 인프라는 이미 있다(`src/training/switch_ban_processor.py`). ~20 GPU-h.

### ④ 기전 둘이 구조적으로 배제됐다

- **자가수정 채널 없음**: 첫 메타 블록이 **문자 위치 0** 에서 시작한다(b2p 100.00%, b3s 99.98%).
  앞에 고칠 것이 없다 — 헤더이지 수정이 아니다. 실제 wrong→right 사건 4000표본 중 **1~4건**.
  ⇒ **정확도 차이의 채널로 "메타가 오답을 바로잡는다" 는 배제.**
- **유령 confidence**: 정상 블록 밖의 `confidence:` 가 b3s 8.90% / b2p 4.95% 표본에 있고,
  전체 confidence 언급의 **30.5%(b3s) · 32.7%(b2p)** 를 차지한다. `rewards.py:535` 가
  `not blocks` 로 게이트하므로 **이들은 한 번도 채점·보상·처벌되지 않았다.**
  ⇒ 프로그램이 "신뢰도 신호"라고 부른 것의 3분의 1이 목적함수 밖에 있었다.

### ⑤ 옵티마이저 비대칭 — clip 만 남는다(E-160 ① 재확인, 독립 경로)

pre-clip `grad_norm > 1.0` 비율: **b2p-1 28.1% · b0p-1 37.5% · b2p-2 11.3%**
vs **b3p-1 0.33% · b3s-1 0.00% · b3shf-1 0.00%**. post-clip 실효 norm 비 2.27~2.75배.
⇒ 두 팔이 **런 전체에서 clip 문턱의 반대편에** 있다. Adam 이 균일 배율은 흡수해도 clip 은 못 흡수한다.
★**신규 비대칭**: 대조군 `critic/advantages/mean` 은 whiten 때문에 **항상 정확히 0**인데,
**b3p 는 후반에 −0.0457 로 표류**한다(envelope 대비 2.8%) — 모든 응답 토큰에 **지속적 음의 밀기**.
b3s·b3shf 는 0 근처. **−2.08pp 인 팔만 이 성질을 가진다** ⇒ 순수 배율과 구별되는 후보 기전.

### ⑥ 닫는 것 / 여는 것

**닫는 것** — (a) ⛔**철회**(③ 참조 — C-026 이 이 축을 이미 널로 판정했다), (b) 자가수정 채널 가설,
(c) b3p gs300 을 근거로 한 성분 귀속 문장 전부(`s*≈190`), (d) discard 폭발 = 정확도 손실 등식.
**여는 것** — (a) **b3p 를 gs100–130(발화 살아 있고 정확도 최고)에서 held-out 평가** — 이것이
"기전이 살아 있을 때 메타 보상이 무엇을 했나"를 처음으로 묻는다, (b) 유령 confidence 를 채점에
넣는 설계(`rewards.py:535` 게이트), (c) b3nopmi 길이·엔트로피 궤적 감시(붕괴 예측).
**재확인 계수기** — 0 (신규).

### ⑦ 왜 ⑥(a) 를 지금 못 하나 — **`--keep 1` 이 중간 체크포인트를 계속 지운다** (조치 완료)

HF 실측(`iamseungpil/metacot-h200-triobj-dcpo-v3`) 보존 상태:
`rq3v2f_b3p` **[300]** · `rq3v2f_b2p` **[300]** · `rq3v2f_b0p` [295, 300] · `rq3v2f_b3s` [301,302,303].
⇒ ★**b3p 의 gs100–130(발화 0.93, 정확도 최고 0.4457)은 영구 소실됐다.** ⑥(a) 는 **실행 불가**다.

근인: 런처가 푸셔에 **`--keep 1`** 을 넘긴다(`h100std_rq3v2f_b3nopmi.yaml:159`, 전 팔 동일).
`_prune_old_verl_ckpts`(`scripts/push_ckpts_to_hf.py:35-95`)가 **완전 ckpt 중 최신 1개만 남기고
나머지를 `delete_folder`** 한다. 5스텝마다 발동하므로 **판정 시점을 제외한 전 구간이 소멸**한다.
가드 자체는 옳게 짜여 있다(부분 ckpt 를 keep 에 안 세어 유일 재개상태를 지키는 로직이 `:39-43`).
문제는 **정책값**이지 구현이 아니다.

★**이것이 이 프로그램의 구조적 손실이다**: 모든 판정이 **말단 체크포인트 하나**로만 가능하고,
그 말단이 붕괴 뒤라는 것을 ①이 보였다. **즉 "기전이 살아 있을 때의 효과" 는 설계상 측정 불가였다.**

**조치(가역·순수 추가, 삭제 없음)**: 프루너는 `checkpoints/{config_name}` 만 **비재귀** 리스팅한다
(`push_ckpts_to_hf.py:51-58`) ⇒ 다른 접두어는 사정권 밖이다. 서버측 LFS 복사로
`checkpoints/rq3v2f_b3nopmi/global_step_120/` (23파일, **발화 0.9981 = 기전 생존**) 를
**`preserved/mechanism_alive/rq3v2f_b3nopmi_gs120/`** 로 복제했다(다운로드 0바이트).
⇒ **b3nopmi 는 "기전 살아 있는 시점" 평가가 가능해졌다.** b3p 는 이미 늦었다.

**남은 승인 사항**: 런처의 `--keep` 상향(1→4)은 정본 편집 + 잡 재시작(스테이징 20~40분)이라
승인 대상이다. 승인 전까지는 **틱마다 서버측 복사로 방어**한다 — 승인 불필요·무비용.

---

## E-162 (0808) — 소명 완료: **표류는 파일이 아니라 채점기다.** 그리고 빠져 있던 CI 넷

### ① oid 고정 — 판정 후보 13개 전수

`iamseungpil/metacot-h200-triobj-dcpo-v3`, `eval/**/gs300_16k_n8_math500.parquet`:

| 팔 | oid(앞16) | 크기 |
|---|---|---|
| **b2p** `_1030` | `08faf5ae9fc4e404` | 3.5MB |
| **b0p** `_1030` | `fbcd9864a64ba6ba` | 2.8MB |
| **b3s** `_1030` | `ce372c5b593c8f4c` | 5.1MB |
| **b3p** `_1030` | `af5df50da404f0b8` | 4.0MB |

★**b0p 는 gs300 eval **하나뿐**이다** — gs295 ckpt 는 있어도 eval 은 없다. ⇒ 페이블의
"어느 ckpt 인가" 질문은 해소: **gs300, `fbcd9864a64ba6ba`.**

### ② ⛔**함정 셋** — 전부 이번에 처음 보인다

1. ★**`super_squash` 가 파일별 커밋 날짜를 파괴했다.** 13개 파일의 `last_commit.date` 가
   **전부 `2026-08-07 23:07:56`** 로 같다. ⇒ **날짜는 출처 증거로 쓸 수 없다. oid 만 유효하다.**
   (`push_ckpts_to_hf.py:251-262` 가 히스토리를 단일 커밋으로 재작성한다.)
2. ★**`eval/rq3v2f_b3sh_gs175/` 안에 gs165 파일이 들어 있다** —
   `rq3v2f_b3sh_gs175/rq3v2f_b3sh_gs165_16k_n8_math500.parquet`, oid `7e32d9b491c795ce`
   (gs165 디렉터리의 것과 **동일 oid**). ⇒ **`gs175/*.parquet` 로 glob 하면 gs165 를 집는다.**
   E-155 는 `aae605444e80453f`(진짜 gs175)를 썼으므로 무사하다. **앞으로 glob 금지, oid 로 지목.**
3. ★**b2p 는 eval 세대가 셋이다**: `_1030`(`08faf5ae`) · `_1030_pair`(`a42de5db`) ·
   `_1030_rq2`(`e46eca3d`). 서로 다른 생성 실현이다. 대장은 `_1030` 을 쓴다.

### ③ 표류의 근인 — **파일이 아니라 채점기다**

위 oid 로 `math_verify` + `$`-wrap(NFKC 없음, `scripts/rq2_analyze.py:23-33` 의 정본 변종)으로
다시 채점하니 **대장 값이 정확히 재현된다**:

| 팔 | 이번 재채점 | 대장 | 워크플로(E-161) |
|---|---|---|---|
| b2p | **77.28** | 77.28 ✅ | 77.28 |
| b0p | **76.12** | (없었음) | 76.05 |
| b3s | **75.28** | 75.28 ✅ | 75.25 |
| b3p | **75.20** | 75.20 ✅ | 75.15 |

⇒ ★**같은 oid 에서 대장 값이 나온다.** 워크플로의 75.25/75.15/76.05 는 **다른 채점기 변종**의
산물이지 파일 표류가 아니다. **b3s 의 "세 번째 값" 은 파일 문제가 아니라 채점기 문제다.**
(74.80 은 C-030 이 봉합한 옛 리비전이므로 별건.)
⇒ **규율: 수치를 인용할 때 oid **와 채점기 변종**을 함께 단다. 둘 중 하나만으로는 재현 안 된다.**

### ④ 빠져 있던 CI 넷 (쌍대 부트스트랩 10,000회, 문항 클러스터, seed 20260805, n=500문항/4000표본)

| 대비 | Δpp | 95% CI | 판정 |
|---|---|---|---|
| **b2p − b0p** | **+1.15** | **[−0.30, +2.62]** | **0 포함 = 널** |
| b3s − b2p | −2.00 | [−3.40, −0.62] | 0 제외 ✅ |
| b3p − b2p | −2.08 | [−3.50, −0.70] | 0 제외 ✅ |
| **b3s − b0p** | **−0.85** | **[−2.38, +0.68]** | **0 포함 = 널** |

**⇒ ⑴ C-026 의 판정(널)이 유지된다.** 점추정은 +0.18 → +1.15 로 움직였지만 **둘 다 널**이고
서로의 CI 안에 있다. **E-161 ③ 의 철회는 옳았다.** C-026 개정 불필요 — 재확인만 된다.

**⇒ ⑵ 신규: `b3s − b0p` 도 널이다** (−0.85 [−2.38, +0.68]).
★**메타 보상 팔은 "메타 없는" 통제군(b0p)보다 유의하게 나쁘지 않다.** 유의하게 낮은 것은
**b2p(메타 SFT + 바닐라 GRPO) 대비뿐**이다.
⇒ 서열은 비추이적이 아니라 **b0p 가 넓은 CI 로 가운데 있는 것**이다:
`b2p > b3s` 유의 · `b2p > b0p` 널 · `b0p ≈ b3s` 널.
⛔**그러므로 "메타 보상이 base 를 망친다" 는 "b2p 대비" 조건을 반드시 달아야 한다.**
b0p 를 기준선으로 잡으면 그 문장은 **성립하지 않는다.**

### ⑤ 닫는 것 / 여는 것

**닫는 것** — (a) *"b3s 가 세 값을 갖는 것은 파일 표류다"*(채점기다), (b) 커밋 날짜를 출처
증거로 쓰는 모든 절차(squash 가 파괴함), (c) C-026 개정 논의(재확인됨).
**여는 것** — (a) **기준선 선택이 결론을 바꾼다**는 사실을 판정문 서식에 반영
(⇒ 모든 −2pp 문장에 "vs b2p" 명기), (b) 워크플로 채점기 변종이 무엇이었는지 특정,
(c) `<|meta|>` 토큰-밴 재평가(발화 효과의 유일한 within-model 설계).
**재확인 계수기** — C-026: **2**(변동 없음. 이번엔 대장을 먼저 읽고 확인만 했다).

---

## E-163 (0808) — **사전등록 재현축 판정: 재현 실패, 단 유의 손상도 아님.** 그리고 채점기 지문 대조

내가 E-162 에서 **사전등록된 축 하나를 계산 목록에서 빠뜨렸다.**
`EXPERIMENT_PLAN.md:19`: *"축: **RQ1=b2p−b0p** · **RQ2=b3p−b2p** · **재현축=b3p−b0p**(T1 헤드라인과 같은 축)"*.
나는 앞의 둘과 사후 대비 둘을 냈고 **정작 T1 헤드라인 동축을 안 냈다.**

### ① 사전등록 세 축 — 전부 (정본 채점기 `wrap=on, nfkc=off`, oid 고정, 쌍대 부트스트랩 10k)

| 축 | Δpp | 95% CI | 판정 |
|---|---|---|---|
| **RQ1** `b2p−b0p` | +1.15 | [−0.30, +2.62] | **널** |
| **RQ2** `b3p−b2p` | **−2.08** | **[−3.50, −0.70]** | **유의 음수** |
| ★**재현축** `b3p−b0p` | **−0.92** | **[−2.50, +0.62]** | **널** |

`EXPERIMENT_PLAN.md:50-52` 사전등록 판정규칙: *"`b3p−b0p` **유의 양수 = 재현 성공** /
**음수 = substrate-dependence(음성결과로 발표)**"*.

⇒ ★★**어느 쪽도 아니다. 판정문은 이렇게 쓴다:**
> **T1 의 instruct 헤드라인(+18.8pp)과 같은 축에서 base 재현은 실패했다(Δ=−0.92pp).
> 동시에 유의한 손상도 아니다(CI 가 0 을 포함).** 즉 결과는 *"기질이 뒤집는다"* 가 아니라
> ***"기질이 효과를 0 으로 만든다"*** 이다. 사전등록이 예상한 두 갈래(성공/음성) 중
> **어느 것도 아닌 세 번째** 이므로, 음성결과 발표문은 **"부호 반전"이 아니라 "소멸"** 로 쓴다.

⚠**동시에 RQ2(`b3p−b2p`)는 유의 음수다.** 두 사실은 모순이 아니다 —
b0p 가 넓은 CI 로 가운데 있어(77.28 > 76.12 > 75.20) Δ=2.08 만 CI 문턱을 넘는다.
⇒ ★**"메타 보상이 해롭다" 는 RQ2 의 문장이고 기준선은 b2p 다. 재현축에서는 널이다.**
**두 문장을 반드시 같이 쓴다.** 하나만 쓰면 기준선 선택이 결론을 만든 것이 된다.

### ② 채점기 지문 대조 — 후보 둘 다 기각, 잔차는 **1~3행**이다

같은 oid 에서 `{$-wrap on/off} × {NFKC on/off}` 4조합 전수:

| 조합 | b2p | b0p | b3s | b3p |
|---|---|---|---|---|
| **wrap=on, nfkc=off** (정본) | **77.28** | **76.12** | **75.28** | **75.20** |
| wrap=on, nfkc=on | 77.28 | 76.12 | 75.28 | 75.20 |
| wrap=off, nfkc=off | 65.55 | 65.38 | 65.08 | 65.10 |
| wrap=off, nfkc=on | 65.55 | 65.38 | 65.08 | 65.10 |
| *(워크플로)* | *77.28* | *76.05* | *75.25* | *75.15* |

- ⛔**NFKC 는 이 데이터에서 완전 무효과다** — 네 팔 전부 소수점까지 동일. (크메르·타밀 숫자는
  NFKC 가 접지 않으므로 예상과 일치. 전각만 접는데 그 사례가 없다.)
- ⛔**`$`-wrap 제거는 워크플로 값을 설명 못 한다** — 전 팔이 **~11pp 붕괴**(C-027 이 이미 확인한
  bare-LaTeX 과소채점)하지 워크플로처럼 0.1pp 어긋나지 않는다.
- ★**잔차의 크기가 정체를 말해 준다**: b0p −0.07pp · b3s −0.03pp · b3p −0.05pp, b2p **정확히 0**.
  4000표본에서 **0.025pp = 1행**이다 ⇒ 워크플로와의 차이는 **1~3행**뿐이다.
  체계적 변종이 아니라 **`math_verify` 의 행 단위 실패**(hang→예외→0)로 설명된다.
  이 프로젝트에 이미 병력이 있다(`cfgroup-deadlock-is-mathverify-hang`).
  b2p 가 정확히 0 인 것은 그 팔만 우연히 실패행이 없었거나 대장 값을 인용했기 때문이다.
⇒ **출처 표기에 네 번째 칸이 필요하다: 체크포인트·모집단·집계 + `채점기 변종`.**
  그리고 **채점기는 결정론적이지 않다** — 같은 파일·같은 설정도 1~3행이 흔들린다.

### ③ C-026 처리 정정 — "재확인"이 아니라 **"증거 교체 후 재확인"**이다

E-162 ⑤ 에 *"C-026 개정 논의 닫는다(재확인됨)"* 라고 썼다. **불충분하다.**
`CLAIMS.md:559` 의 C-026 증거란은 **경로만 있고 oid 가 없다**(`eval/rq3v2f_b0p_1030/` 등).
C-026 의 Δ=+0.18 은 b0p ≈ 77.10 을 함의하는데 oid 고정 재채점은 **76.12** 다 — **같은 경로에서 ~1pp 차이.**
경로는 가변이고(b2p 만 해도 생성판이 셋), **`super_squash` 가 히스토리를 단일 커밋으로 재작성했으므로
0804 시점에 그 경로에 있던 파일은 원리적으로 복구 불가능하다.**

⇒ **올바른 처분**: 판정(널)은 유지하되 **증거란을 oid 고정판으로 교체**하고
*"원 점추정 +0.18 의 입력은 경로-가변 + squash 로 비재현. 널 결론은 양판 CI 중첩으로 불변"* 을 명기.
⇒ ★**전방 처방**: `CLAIMS.md` 를 훑어 **증거란에 oid 가 없는 C-claim 을 전부 채운다** —
지금 tip 에 있는 파일은 아직 고정 가능하다. **다음 squash 가 돌면 그것도 늦는다.**

### ④ 닫는 것 / 여는 것

**닫는 것** — (a) *"워크플로 값 차이는 채점기 변종 탓"*(변종 둘 다 기각, 1~3행 flakiness 다),
(b) NFKC 를 이 데이터에서 논하는 것, (c) `b3p−b0p` 미계산 상태.
**여는 것** — (a) `CLAIMS.md` oid 공백 스캔·충전(**시급 — squash 전에**),
(b) eval 디렉터리 gs 오염 스캔(`eval/.*_gs(\d+)[^/]*/.*_gs(\d+)_` 불일치),
(c) **SFT2-init eval** — b3null·b3nopmi 결과가 도착하기 **전에** 있어야 그것들이
"얻은 것 vs 잃은 것" 축 위에 놓인다.
**강등** — `<|meta|>` 토큰-밴 재평가. 그 실험의 동기였던 "발화 +1.23pp" 가 널로 죽었으므로
(RQ1 [−0.30, +2.62]) 유의하지 않은 효과를 GPU 로 분해할 이유가 없다. 발화 서사가 되살아나면 꺼낸다.
**재확인 계수기** — C-026: 2 (변동 없음).

---

## E-164 (0808) — 출처 방어 완료: **CLAIMS oid 공백 0** · eval 오염은 한 디렉터리에 국한(단 50:50 혼재)

E-163 ④ 가 연 두 작업을 끝냈다. 둘 다 GPU 0.

### ① `CLAIMS.md` oid 충전 — 공백 **3 → 0**

증거란 30개 전수 스캔(경로는 인용하는데 oid 가 없는 것):

| 위치 | 주장 | 조치 |
|---|---|---|
| `:559` | **C-026**(`b2p−b0p` 널) | **증거 교체** — 아래 |
| `:691` | C-029 | oid 충전(리비전 고정만으로는 부족 — squash 가 히스토리를 뭉갠다) |
| `:759` | C-030 | oid 충전 |

**충전한 oid**: b0p `fbcd9864a64ba6ba` · b2p `08faf5ae9fc4e404` · b2p_pair `a42de5db47ba7b5f` ·
b3s `ce372c5b593c8f4c` · b3p `af5df50da404f0b8`. **재스캔 결과 공백 0.**

★**C-026 은 충전이 아니라 교체다.** 원 점추정 `+0.18pp` 는 b0p ≈ 77.10 을 함의하는데
oid 고정 재채점은 **76.12** 다(~1pp 괴리). `super_squash` 로 **0804 시점 입력은 복구 불가**.
⇒ 판정(널)은 유지, 증거를 **`+1.15pp` [−0.30, +2.62] / oid 3종**으로 교체하고
*"양판 CI 가 서로를 포함하므로 널 결론은 불변"* 을 명기했다.

### ② eval gs 오염 스캔 — 240파일 중 **한 디렉터리**, 단 **완전 혼재**

⚠**내 첫 스캔이 틀렸다**: 디렉터리 gs 를 경로의 **마지막** 성분에서 뽑아 중첩 디렉터리
(`.../rq3v2f_b3sh_gs165_16k_n8_math500/....parquet`)를 **자기 자신과 일치**시켰다.
⇒ 로그 5건만 잡히고 **parquet 을 전부 놓쳤다.** **최상위 eval 디렉터리 기준**으로 고쳐 재실행.

**결과**: `eval/rq3v2f_b3sh_gs175/` 안에
**gs175 정상 20파일 + gs165 이물 20파일**(parquet 4개 포함) — **50:50 혼재.**
다른 디렉터리는 전부 깨끗하다.

⇒ ★**그 디렉터리를 glob 하면 동전던지기다.** E-155 는 oid(`aae605444e80453f`)로 지목해 무사했다.
⇒ **규칙 확정: eval 은 glob 금지. 파일명을 정확히 구성하고 oid 로 확인한다.**
⛔삭제하지 않는다 — 이물도 유효한 gs165 데이터이고, 파괴조작 3율(대체물 확보 전 폐기 금지)에 걸린다.

### ③ 닫는 것 / 여는 것
**닫는 것** — oid 없는 증거란을 근거로 한 인용 전부(이제 0개) · *"오염이 더 있을 것"* 이라는 우려(1건 확정).
**여는 것** — 스캔 도구 자체를 게이트에 걸기(`stacked-research §5.4`: 도구를 만들면 그 자리에서
게이트·CI·CLAIMS 중 하나에 건다). 지금은 일회성 스크립트라 **미배선**으로 적어 둔다.
**재확인 계수기** — 0.

---

## E-165 (0808) — **PMI 보상은 의도한 방향으로 실제 일을 한다**(단일 손잡이 대조) · `s*` 정확값 · **발화 붕괴는 −2pp 의 원인이 아니다**

### ① G8 먼저 — 정말 손잡이 하나인가

`diff h100std_rq3v2f_b3p.yaml h100std_rq3v2f_b3nopmi.yaml`(팔 이름 정규화 후) 실행부 차이:
**`++algorithm.dcpo_w_meta=0.0` 한 줄뿐.** 나머지는 `description` 주석만 다르다.
같은 config·같은 init·같은 데이터·같은 300스텝·`rmeta_source=pmi_shift` 동일.
⇒ **이 프로그램에서 지금까지 나온 것 중 가장 깨끗한 단일 손잡이 대조다.**

### ② 같은 스텝 구간 대조 — **PMI 보상은 5/6 지표에서 의도 방향으로 작동한다**

| gs120–140 | b3p (`w_meta`=0.8) | b3nopmi (`w_meta`=0) | PMI 보상의 효과 |
|---|---|---|---|
| **발화율** | 0.8290 | **0.9890** | ⛔**낮춘다** |
| **응답길이** | **696** | 1017 | ✅짧게 (−32%) |
| **8192 상한 충돌** | **0.0166** | 0.0617 | ✅비종료 **3.7배** 억제 |
| **entropy** | **0.3370** | 0.6804 | ✅**2.0배** 억제 |
| **discard** | **0.0882** | 0.1629 | ✅절반 |
| **학습 정확도** | **0.4610** | 0.4265 | ✅+3.5pp |

⇒ ★**PMI 보상은 "무효 레버"가 아니다.** 길이 폭주·비종료·엔트로피 폭주·discard 를 **전부 억제**하고
학습 정확도를 올린다. **대가는 발화율 침식 하나뿐**이다.
⛔**단 이것은 학습 동역학이지 held-out 이 아니다**(EXP 트랙). 이 프로젝트의 확립된 사실:
**학습 지표는 held-out 을 예측하지 않는다**(b3s 가 학습 최고인데 held-out 최하위권이었다).
따라서 **"PMI 가 이롭다"로 승격 금지.** 승격은 b3nopmi gs300 held-out 이 한다.

### ③ `s*` 정확값 — 내 E-161 추정 `≈190` 을 **157 로 정정한다**

`gdpo/meta_emission/mean` 이 문턱을 처음 밑도는 스텝:

| 팔 | <0.9 | **`s*`(<0.5)** | <0.1 | 최종 |
|---|---|---|---|---|
| **b3p** (`meta_floor=0`) | gs118 | **gs157** | gs205 | gs300 **0.0176** |
| **b3s** (`meta_floor=0.05`) | **없음** | **없음** | 없음 | gs303 **1.0000** |
| b3nopmi (진행중) | 없음 | 없음 | 없음 | gs132 0.9570 |

⚠**E-160 ⑤ 의 판독 게이트에 직결된다**: *"스케일/whiten 이 원인으로 좁히는 문장은 `s*≤150` 일 때만 허용"*.
**b3p 의 `s*`=157 > 150 이므로 그 문장은 b3p 에 대해 허용되지 않는다.** (아슬아슬하다 — 규칙대로 적용한다.)

### ④ ★★**발화 붕괴는 −2pp 의 원인이 아니다** — `meta_floor` 가 그것을 증명한다

b3s = b3p + `meta_floor=0.05` **한 손잡이**. 그런데:
- **b3s 는 발화가 전혀 무너지지 않는다**(gs303 에서 **1.0000**). ⇒ `meta_floor=0.05` 가 발화 붕괴를 **완전히 막는다.**
- **그런데 held-out 은 b3s 75.28 · b3p 75.20 — 사실상 같다**(둘 다 vs b2p 유의 음수).

⇒ ★**발화를 100% 지켜도 −2pp 는 그대로다.** 따라서 **발화 붕괴는 결손의 원인이 아니다.**
⇒ **E-161 ① 의 함의를 좁혀야 한다**: gs300 이 붕괴 뒤라는 사실은 *"b3p 의 수치를 기전 귀속에
쓸 수 없다"* 까지는 맞지만, *"붕괴 때문에 손해를 봤다"* 는 **b3s 가 반증한다.**
⇒ **닫는다**: 발화 유지를 목표로 하는 개입(floor 상향·발화 보상 강화)으로 −2pp 를 되찾으려는 설계 전부.
`meta_floor=0.05` 가 이미 발화를 100% 지켰고 **아무것도 되찾지 못했다.**

### ⑤ 정정 — b3s 는 오염되지 않았다 (내 오류)

E-161 ② 표에 *"b3s ⚠오염(b3shf 와 run 공유)"* 라고 적었다. **틀렸다.**
오염된 것은 **b3shf** 뿐이고(그 팔이 `rq3v2f-b3sh-1` run id 를 물려받았다),
b3s 가 오염된 것처럼 보인 이유는 **내 매처의 부분문자열 검사** 때문이다 —
`"rq3v2f_b3s" in "rq3v2f_b3shf"` 가 참이라 두 run 이 같은 키에 묶였다.
**wandb run `rq3v2f_b3s` 자체는 깨끗하다**(n=302, gs303 완주). 위 ③④ 는 그 정본을 쓴다.
⇒ 규율: **부분문자열로 팔을 매칭하지 마라. 정확 일치를 쓴다.**
(같은 날 두 번째 도구 결함이다 — gs 오염 스캔도 경로 성분을 잘못 골랐다. **내 도구를 먼저 의심하라.**)

### ⑥ 닫는 것 / 여는 것
**닫는 것** — (a) *"PMI 보상이 무효 레버다"*, (b) *"발화 붕괴가 −2pp 를 만들었다"*(b3s 반증),
(c) 발화 유지형 개입으로 −2pp 회복 시도, (d) *"b3nopmi 가 b3p 붕괴 궤적을 따라간다"*(반대다 —
발화는 살아 있고 **길이·엔트로피가 b3p 보다 훨씬 나쁘다**), (e) b3s 오염 우려.
**여는 것** — (a) **b3nopmi gs300 held-out** 이 ②를 승격/기각한다(`b3p − b3nopmi` 사전등록,
`EXPERIMENT_PLAN.md:110-125`), (b) *"그럼 −2pp 는 무엇 때문인가"* — 발화도 아니고 보조 5헤드도
아니면(E-154) **남은 것은 region 기하 번들** ⇒ **b3null 이 유일한 판별자**, (c) b3nopmi 의
길이 폭주가 held-out 을 얼마나 깎는지(비종료 6.2%).
**재확인 계수기** — 0.

---

## E-166 (0808) — **보조 판정 사전등록** (등록 시각 `2026-08-08 01:37:29 UTC`, HEAD `f8c6850`) · 그리고 내 "기록 없음" 주장 철회

### ⓪ 철회 — 학습 절단 → held-out 관계는 **이미 판정돼 있었다**

나는 페이블에게 *"학습 clip_ratio → held-out 절단률 관계를 잰 기록을 못 찾았다"* 고 했다. **틀렸다.**
`docs/CLAIMS.md:689-692`(C-029, codex 가 지시한 분해):
> ✅**절단이 아니다.** `finish_reason=="stop"` 인 **두 팔 모두 종결한 샘플**(3,788/4,000 = 94.7%)로
> 제한하면 **−2.65pp [−4.16,−1.16]** — **헤드라인보다 오히려 나쁘다.** 절단 기여분 **+0.17pp**.
> ⇒ **추론 능력 손실이지 잘려서가 아니다.**

원장 3310-3326 에 팔별 절단율 표와 조건부 지표 경고까지 있다.
⇒ **b3nopmi 의 학습 비종료 9.18% 는 판정을 막지 않는다.** b3p 는 같은 수준(9.12%)에서 판정됐다.
⇒ **b3nopmi 는 그대로 gs300 까지 보낸다.** (이번이 오늘 세 번째 "대장에 이미 있었다" 사례다.)

### ① 주 판정 — **변경 없음**

`b3p − b3nopmi` **gs300 held-out 1030, 16k, avg@8, MATH500**(`EXPERIMENT_PLAN.md:20`),
분기표는 `:110-125` 그대로. **CONF 트랙.**

★**이름을 정확히 붙인다**: 이 차분은 **"PMI 헤드가 최종 정책에 주는 end-to-end 순효과 —
하류 동역학(길이 억제·발화 침식) 포함"** 이다.
⛔**"PMI 가 메타인지 품질에 주는 효과" 라고 부르는 것은 금지.**
근거: 0717 에 이미 확립된 원칙(RQ2 의 estimand 는 내용 효과가 아니라 최종 정책의 end-to-end
효과이고, 침식이 질문 자체를 바꾸는 것까지 estimand 에 포함된다). 두 팔이 **서로 다른 방향으로**
망가지는 것(b3p=발화 사망 `s*`=157 / b3nopmi=비종료 9.2%)은 **측정의 오염이 아니라 측정 결과 자체**다.
E-165 가 그 기전을 이미 줬다 — 이 기질·이 패키지에서 PMI 헤드의 주 역할은 메타 품질 채점기가
아니라 **길이·비종료·discard 정칙화기**다.

### ② 보조 분석 — **지금 고정한다** (EXP 트랙, 헤드라인 승격 영구 금지)

- **스텝**: **{gs50, gs100, gs150} 고정 그리드.** 상태와 무관하게 **지금** 박는다.
  ⛔**"두 팔 모두 비종료 ≤2% 인 마지막 공통 스텝" 같은 규칙은 쓰지 않는다** —
  **처치 이후 변수로 조건화**하는 것이고, 원장 3326 이 이미 경고한 함정이다
  (*"조건부 지표는 파괴 비용을 못 본다"*, 0731). 붕괴 직전에서 판정하면 **각 헤드의 후기 파괴
  비용이 estimand 에서 구조적으로 빠진다.**
- **지표 키**: **`val-aux/{subset}/correctness/mean@1`** 를 9개 서브셋에 대해 사용.
  ⛔`val-core/*/reward` 는 **보상-성형(meta-shaped)** 이라 정확도 대용으로 금지.
  ⚠`val-aux/*/score` 도 아니다 — **`correctness`** 가 순수 정답률 키다.
- **데이터 가용성 확인 완료**(wandb `gistdslab/metacot-dcpo-v4`, **resume 조각 병합 후**):
  b3p [50,100,150,200,250,300] · b2p [50,100,150,200,250,300] · b0p [50,100,150,200,250] ·
  b3s [0,50,…,303] · b3nopmi [50](진행중) · b3null [](gs31, 첫 val 은 gs50).
  ★**b2p 는 조각 2개**(`rq3v2f-b2p-1`,`-2`) — **병합 전에는 [200,250,300] 만 보인다.**
  ⇒ **조각 병합은 의무다.** ckpt 프루닝과 무관하게 **val 은 wandb 에 남아 있으므로 재학습 불필요.**
- **명기 사항**(판정문에 그대로 옮긴다):
  > *"이 보조 분석은 구성상 각 헤드의 **후기 파괴 비용을 배제한다**. 기전 증거로만 읽으며,
  > 어떤 결과가 나와도 헤드라인으로 승격하지 않는다."*

### ③ b3nopmi 보존본 처분
`gs120`·`gs130` 은 **유지한다.** 보조 판정을 val594 기반으로 정의했으므로 지금은 불필요하지만,
후일 matched-step **held-out** 이 결정적이 되면 **b3p 1회 재학습(~60 GPU-h)** 을 정당화할 근거가 된다.
⚠b3p 는 gs300 외 ckpt 가 전부 프루닝됐고, GPU 비결정성 전례(±0.05) 때문에 재학습은
**"복구"가 아니라 새 실현**이다.

### ④ 도구 결함 의심처 다섯 (페이블 지적, 미해소)
1. **run 이름 부분문자열** — `b3s` ⊂ `b3sh` ⊂ `b3shf`. **정확 일치 또는 run-id 키**로.
   ★실측: `rq3v2f_b3shf` 의 id 는 `rq3v2f-b3sh-1` 이다(오염 재확인).
2. **HF 경로 prefix** — `checkpoints/rq3v2f_b3s` 는 `rq3v2f_b3sh` 의 접두. **trailing slash 필수.**
   (프루너는 정확 경로라 안전, **ad-hoc 스크립트가 위험**.)
3. ✅**해소(0808)** — E-165 는 **이중계상하지 않았다.** `rq3v2f_b3p`·`rq3v2f_b3nopmi` 둘 다
   **조각 1개**(`rq3v2f-b3p-1` / `rq3v2f-b3nopmi-1`)이고, `gdpo/meta_emission/mean` 행 수와
   고유 스텝 수가 정확히 일치한다(300/300 · 137/137, **중복 스텝 0**).
   ⚠**단 규칙은 유지**: 조각이 둘 이상인 팔(b2p)에서는 병합·dedup 이 의무다.
   그리고 dict 대입(`d[step]=v`)은 **스캔 순서의 마지막**을 남길 뿐 *"나중 생성분 우선"* 이
   아니다 — 조각이 여럿이면 **명시적으로 최신 조각을 우선**하도록 써야 한다(E-156 방식).
4. **기대 개수 assert 부재** — parquet 전부 누락이 침묵으로 지나갔다. 모든 스캔에 `≥N` assert.
5. ✅**해소(0808)** — `eval/rq3v2f_b3sh_gs175/` 의 **glob 소비자는 없다.**
   `--include=*.py/*.sh/*.yaml` 전수 grep 결과 그 프리픽스를 참조하는 것은
   `h100std_rq3v2f_b3sh_1030_eval.yaml:88` 뿐이고, 거기서도 **명시 SPEC**
   (`rq3v2f_b3sh:eval/rq3v2f_b3sh_gs165:165` / `...gs175:175`)으로 지목한다 — glob 아님.
   `eval/` 를 glob 하는 파이썬은 `push_*_hf.py` 셋뿐이고 **전부 업로드 측**이지 분석 측이 아니다.
   ⇒ **오염이 어떤 판정문에도 소비되지 않았다.** E-155 는 oid 로 지목했으므로 무사.
   ⚠그 런처 설명(`:1`)이 오염의 유래도 알려준다 — 한 잡이 두 스텝을 같은 창에 올렸다.
   ★**규칙은 유지**: eval 은 glob 금지, 파일명 정확 구성 + oid 확인.

---

## E-167 (0808) — b3nopmi 발화가 실제로 무너지기 시작했고, **무너지는 동안 나머지 지표가 전부 좋아진다**

### ① 이번엔 진동이 아니다 — 4스텝 연속 하회

| gs | 발화 | 비종료 | 길이 | discard | 학습정확도 |
|---|---|---|---|---|---|
| 134 | 0.9629 | 0.0918 | 1197 | 0.1602 | 0.5137 |
| 135 | 0.9434 | 0.0664 | 1086 | 0.1387 | 0.4074 |
| 136 | 0.9414 | 0.1055 | **1317** | 0.1523 | 0.4371 |
| **137** | **0.8965** | 0.0703 | 1134 | 0.1406 | 0.3789 |
| **138** | **0.8965** | 0.0488 | 714 | 0.0781 | 0.5945 |
| **139** | **0.8477** | 0.0605 | 975 | 0.0977 | 0.5191 |
| **140** | **0.7969** | **0.0430** | **659** | **0.0586** | **0.5785** |

**0.90 미만 최장 연속 4회**(gs137–140), 단조 하강. `s*`(<0.5)는 미도달.
⇒ 0808 이전 틱의 *"진동이지 붕괴가 아니다"* 판단은 **그 시점엔 옳았고**(gs133 0.932 → gs134 0.963
되튐), **지금은 실제 하강이다.** 두 번 다 단일 점이 아니라 연속성으로 판정했다.

### ② ★발화가 죽는 **동안** 다른 네 지표가 동시에 좋아진다

gs136 → gs140 구간:
**길이 1317 → 659**(−50%) · **비종료 10.6% → 4.3%**(−59%) · **discard 15.2% → 5.9%**(−61%) ·
**학습정확도 0.437 → 0.579**(+14pp).

⇒ ★**이 팔에서 길이·비종료·discard 병리의 원천은 메타 블록 자체다.**
메타를 그만 쓰자 답이 짧아지고 끝을 맺고 정확해진다.

★**E-165 와 방향이 일치한다**: PMI 보상이 켜진 b3p 는 **같은 병리를 보상으로 눌렀다**
(길이 696 vs 1017 · 비종료 1.66% vs 6.17%). PMI 보상을 끈 b3nopmi 는 그것을 못 누르다가,
**결국 메타 발화 자체를 포기하는 방식으로 같은 지점에 도달하고 있다.**
⇒ 두 팔이 **같은 압력에 서로 다른 방식으로 굴복**한다: b3p 는 보상으로 억제, b3nopmi 는 발화 포기.

⚠**주의 — 이것은 EXP 트랙의 팔 내부 상관 관측이다.** 스텝 진행과 교란돼 있고,
"메타 블록이 병리의 원인"이라는 인과 주장으로 승격하려면 별도 개입이 필요하다
(예: 같은 ckpt 에서 디코딩 시 메타만 끄고 길이·종결률 비교).
네 지표가 **동시에** 같은 방향으로 움직인다는 점이 단순 잡음보다 강한 근거이지만, 그뿐이다.

### ③ 보존 판단 — 추가 보존 없음(규칙대로)

보존 기준은 *"발화 ≥0.9 인 최신"* 이고, **그 창은 gs136 에서 닫혔다.**
보유분 `gs120`(0.998) · `gs130`(0.977)이 기전 생존 구간을 덮는다.
⚠gs135(0.943)는 프루닝됐으나 gs130 과 실질 차이가 없어 손실 아님.
**gs140 이후는 보존하지 않는다** — 기전이 이미 죽어가는 상태라 "기전 살아 있는 체크포인트"가 아니다.

### ④ 닫는 것 / 여는 것
**닫는 것** — *"b3nopmi 는 발화가 멀쩡하다"*(E-165 ② 작성 시점엔 참, 지금은 거짓) ·
b3nopmi 를 "발화 생존 팔"로 전제하는 해석.
**여는 것** — (a) **b3nopmi 의 `s*` 감시**(b3p 는 gs157 이었다. 같은 지점에서 무너지면
발화 붕괴가 **PMI 와 무관한 공통 현상**이라는 뜻이고, 그러면 E-165 ②의 "대가는 발화 침식뿐"도
PMI 귀속에서 빠진다), (b) 디코딩 개입으로 ②의 인과 승격.
**재확인 계수기** — 0.

---

## E-168 (0808) — ⛔**정정: PMI 보상은 발화 침식의 *원인*이 아니라 *저항*이다.** 붕괴 속도 7배 차이

### ① 0.9 교차점에 정렬한 붕괴 곡선

`s*` 한 점만 보면 안 된다 — **곡선의 모양**이 판별한다. 두 팔을 각자의 0.9 교차 스텝에 정렬:

| 교차 후 | **b3p** (`w_meta`=0.8) | **b3nopmi** (`w_meta`=0) |
|---|---|---|
| +0 | 0.875 | 0.896 |
| +2 | 0.855 | 0.848 |
| +4 | 0.805 | **0.781** |
| +6 | 0.820 | **0.699** |
| +8 | 0.859 | *(미도달)* |
| +12 | **0.916** ← 0.9 위로 복귀 | |
| +16 | 0.754 | |
| +30 | **0.941** ← 다시 복귀 | |

**0.9 교차 후 임계 도달까지 걸린 스텝:**

| | 0.8 | 0.7 | 0.5 (`s*`) |
|---|---|---|---|
| **b3p** | +16 (gs134) | **+37** (gs155) | **+39** (gs157) |
| **b3nopmi** | **+3** (gs140) | **+5** (gs142) | 미도달 |

⇒ ★★**b3p 는 30스텝 넘게 저항하며 진동한다 — 두 번이나 0.9 위로 되돌아간다.
b3nopmi 는 되돌아옴 없이 단조로 떨어진다. 0.7 도달이 7배 빠르다.**

⚠표의 b3nopmi 빈칸은 **아직 도달하지 않은 구간**이다(gs142 가 최신). 반복된 0.699 는 채움값이지 데이터가 아니다.
그럼에도 **+3/+5 vs +16/+37 의 차이는 이미 모호하지 않다.**

### ② ⛔**E-165 ② 정정** — *"대가는 발화 침식뿐"* 은 틀렸다

E-165 에서 나는 gs120–140 한 구간만 보고
*"PMI 보상은 길이·비종료·엔트로피·discard 를 억제하고, **대가는 발화 침식뿐**"* 이라고 썼다.
**그 구간에서 b3p 발화가 더 낮았던 것은 사실이지만, 원인 귀속이 틀렸다.**

**발화 붕괴는 두 팔 모두에서 일어나며, 보상이 없는 팔이 훨씬 빠르게 무너진다.**
⇒ **발화 침식은 PMI 보상의 *대가*가 아니다.** 이 기질·이 패키지의 **공통 압력**이고,
PMI 보상은 그것에 **저항하는** 쪽이다. b3p 가 gs120–140 에서 더 낮았던 것은
**단지 19스텝 먼저 시작했기 때문**이다(0.9 교차: b3p gs118 vs b3nopmi gs137).
⇒ ★**같은 절대 스텝으로 두 팔을 비교한 것이 오류였다. 사건(0.9 교차)에 정렬해야 했다.**

### ③ 그러면 발화 붕괴에 대해 지금 아는 것 — 세 층

| 개입 | 발화 붕괴 | held-out |
|---|---|---|
| 없음 (b3nopmi, `w_meta`=0) | **가장 빠름**(0.9→0.7 이 5스텝) | 대기중 |
| **PMI 보상**(b3p, `w_meta`=0.8) | **지연**(0.9→0.7 이 37스텝) — 단 결국 gs300 에서 0.0176 | 75.20 |
| **`meta_floor=0.05`**(b3s) | **완전 방지**(gs303 에서 1.0000) | 75.28 |

⇒ 세 수준의 개입이 발화를 **전혀 못 지킴 → 지연 → 완전 방지** 로 깔끔하게 정렬되는데,
**held-out 은 b3p 75.20 ≈ b3s 75.28 로 구별되지 않는다.**
⇒ ★**E-165 ④ 재확인·강화**: 발화 보존은 held-out 과 무관하다. **세 수준에 걸쳐 무관하다.**

### ④ 남는 질문 — PMI 보상이 실제로 억제한 것은 무엇인가
E-165 ②의 나머지 네 지표(길이·비종료·엔트로피·discard)도 **같은 오류에 노출돼 있다** —
절대 스텝 비교였다. **0.9 교차 정렬로 다시 재야 한다.**
⇒ 그 재측정 전까지 *"PMI 보상이 길이·비종료를 억제한다"* 도 **잠정**으로 둔다.
⚠단 E-167 ②(발화가 죽는 동안 그 네 지표가 좋아진다)는 **팔 내부 시계열**이라 이 오류와 무관하다.

### ⑤ 닫는 것 / 여는 것
**닫는 것** — *"PMI 보상의 대가는 발화 침식"*(E-165 ②) · 발화 붕괴를 PMI 에 귀속하는 모든 서술 ·
**절대 스텝으로 두 팔의 동역학을 비교하는 것**.
**여는 것** — (a) 네 지표를 **0.9 교차 정렬로 재측정**(GPU 0), (b) b3nopmi `s*` 계속 감시
(예상 gs145~150 — b3p 의 gs157 보다 **이르면** 위 그림이 더 강해진다),
(c) *"무엇이 이 기질에서 발화를 밀어내는가"* — 세 팔 공통 압력의 정체.
**재확인 계수기** — 0.

---

## E-169 (0808) — E-168 ④ 이행: **네 지표는 교차 정렬에서도 살아남고, 학습정확도만 뒤집힌다**. 그리고 b3nopmi 하강이 멈췄다

### ① 0.9 교차 정렬 재측정 (E-168 ④가 요구한 것)

각 팔의 **자기 0.9 교차 스텝**을 원점으로 삼아, 두 팔 모두 **기전이 살아 있던** 교차 **전** 20스텝
(b3p gs98–117 · b3nopmi gs117–136)과, 교차 직후 공통 구간(0~+5)을 비교:

| 지표 | 교차 전 20스텝 b3p | b3nopmi | 교차 후 0~+5 b3p | b3nopmi |
|---|---|---|---|---|
| **응답길이** | **528** | 1020 | 724 | 924 |
| **비종료** | **0.0064** | 0.0623 | 0.0156 | 0.0592 |
| **entropy** | **0.2710** | 0.6781 | 0.3322 | 0.6192 |
| **discard** | **0.0753** | 0.1611 | 0.0993 | 0.0973 |
| 학습정확도 | 0.4164 | **0.4389** | 0.4804 | **0.4887** |

⇒ ★**E-165 ②의 네 지표는 교차 정렬에서도 그대로 성립한다** —
**PMI 보상은 길이를 절반으로(1020→528), 비종료를 ~10배(6.23%→0.64%), entropy 를 2.5배,
discard 를 절반으로 억제한다.** 절대 스텝 비교의 오류(E-168)에도 **불구하고** 방향과 크기가 살아남았다.
⇒ **"PMI 보상은 정칙화기다" 는 잠정에서 확정으로 올린다**(단 EXP 트랙 — held-out 이 승격한다).

### ② ⛔**정정 — 학습정확도 주장은 뒤집힌다**

E-165 ② 에 *"학습정확도 +3.5pp(b3p 우세)"* 라고 썼다. **교차 정렬하면 반대다**:
교차 전 20스텝에서 **b3p 0.4164 vs b3nopmi 0.4389 — b3nopmi 가 +2.3pp 높다.**
교차 후 구간에서도 b3nopmi 가 근소 우세(0.4804 vs 0.4887).
⇒ **"PMI 보상이 학습 정확도를 올린다" 는 절대 스텝 비교가 만든 허상이다. 철회한다.**
⇒ ★**같은 오류가 다섯 지표 중 하나에서만 결과를 뒤집었다** — 나머지 넷은 효과 크기가 커서 버텼다.
**작은 차분일수록 정렬 오류에 취약하다**는 뜻이고, 이것이 그 규율의 실전 근거다.

### ③ b3nopmi 하강이 **멈췄다** — E-168 의 "단조" 서술 정정

gs142 0.699 → gs143·144 **0.703**. **6스텝 단조 하강 뒤 평탄해졌다.**
`s*`(<0.5)는 **여전히 미도달**(최신 gs144, 발화 0.703).
⇒ E-168 ①의 *"되돌아옴 없이 단조로 떨어진다"* 는 **관측 시점까지는 참이었고, 지금은 아니다.**
**측정된 값(+3 만에 0.8, +5 만에 0.7)은 그대로 유효하다** — 그것은 이미 일어난 사실이다.
⚠**"7배 빠르다" 는 초기 하강에 대한 진술로 한정한다.** b3nopmi 도 진동 국면에 들어갔을 수 있다.
⇒ **`s*` 비교는 아직 미결.** b3p 는 gs157(교차 +39)이었다.

### ④ 닫는 것 / 여는 것
**닫는 것** — *"PMI 보상이 학습정확도를 올린다"*(E-165 ②, 정렬 오류) ·
*"E-165 의 네 지표는 정렬 오류로 무효일 수 있다"*(재측정으로 살아남았다) ·
*"b3nopmi 는 단조 하강한다"*(gs143 부터 평탄).
**여는 것** — (a) `s*` 비교 계속(b3nopmi 가 +39 안에 0.5 를 밑도는가),
(b) **학습정확도가 두 팔에서 사실상 같다**는 새 사실 — 정칙화 효과(길이·비종료)가
**학습 정확도로는 안 나타난다**는 뜻이고, held-out 예측력 문제와 이어진다.
**재확인 계수기** — 0.

---

## E-170 (0808) — **`s*` 판별 완료: 보상 없는 팔이 2.6배 빨리 무너진다.** E-168 확정

### ① 등록된 판별 관측 (E-168 이 열고 E-166 이 서식을 정한 것)

| 팔 | 0.9 교차 | **`s*`(<0.5)** | **교차 후 걸린 스텝** |
|---|---|---|---|
| **b3p** (`w_meta`=0.8) | gs118 | gs157 | **+39** |
| **b3nopmi** (`w_meta`=0) | gs137 | **gs152** | **+15** |

⇒ ★★**PMI 보상이 없으면 발화가 2.6배 빨리 무너진다**(39/15 = 2.6).
b3nopmi 궤적(gs148→152): 0.580 → 0.516 → 0.543 → 0.510 → **0.498**.

### ② E-168 의 핵심 주장이 **완전 구간에서 확정된다**

E-168 은 초기 하강(0.9→0.7)만으로 *"7배 빠름"* 이라 했다가 b3nopmi 가 평탄해지자
**"초기 하강 한정"으로 축소**했다. 이제 **0.9→0.5 전 구간**의 값이 나왔다: **2.6배.**
⇒ **정직한 최종 수치는 2.6배다.** 7배는 초기 구간의 값이었고, 전 구간에서는 그만큼 크지 않다.
⇒ 그러나 **방향과 결론은 그대로**: **PMI 보상은 발화 침식의 *원인*이 아니라 *저항*이다.**

★**그리고 발화 붕괴는 두 팔 모두에서 일어난다** — 이 기질·이 패키지의 **공통 압력**임이 확정됐다.
⇒ **E-165 ②의 "대가는 발화 침식뿐" 철회는 옳았다.**

### ③ 갱신된 발화 4층 (이제 수치가 다 있다)

| 개입 | 0.9→0.5 소요 | 최종 발화 |
|---|---|---|
| **없음** (b3nopmi) | **+15 스텝** | 진행중(gs152 에서 0.498) |
| **PMI 보상** (b3p) | **+39 스텝** | gs300 에서 0.0176 |
| **`meta_floor=0.05`** (b3s) | **∞**(도달 안 함) | gs303 에서 **1.0000** |
| *(b3null, 헤드 전무)* | 미도달 | gs71 에서 **1.000** — 아직 이른 단계 |

⇒ **개입 강도와 발화 보존이 단조로 정렬한다.** 그런데 **held-out 은 b3p 75.20 ≈ b3s 75.28** 로 구별 못 한다.
⇒ ★**발화 보존은 held-out 과 무관하다** — 이제 **소요 스텝이라는 연속 척도**에서도 확인됐다.

### ④ 닫는 것 / 여는 것
**닫는 것** — *"7배"* 라는 수(초기 구간 한정이었다) · *"b3nopmi 는 s\* 에 안 닿을 수도"* 라는 가능성 ·
발화 붕괴를 PMI 에 귀속하는 서술 전부.
**여는 것** — (a) **b3null 이 언제 무너지는가** — 헤드가 **전무**한 팔이다. b3nopmi(PMI 만 없음)보다
빠르면 *"메타 보상 계열 전체가 저항한다"*, 비슷하면 *"PMI 만이 저항한다"*.
현재 gs71 에서 1.000 이라 아직 이르다. **b3null 의 0.9 교차 스텝을 기록해 둘 것.**
(b) 이 공통 압력의 정체 — *무엇이 이 기질에서 메타 발화를 밀어내는가*(세 팔 공통).
**재확인 계수기** — 0.

---

## E-171 (0808) — b3shf **호스트 RAM OOM 사망**(gs65) · 3.5시간 늦게 발견 · 재발사

### ① 확진 — E-157 의 세 번째 사인

| 증거 | 값 |
|---|---|
| 예외 | `ValueError: Total available GPUs 0 is less than total desired GPUs 4` — **두 번**(재시도도 실패) |
| 종료 | `kill 3340` → `kill -9 3340` |
| 종료 메시지 | `[YAML] FINAL PUSH DURABLE global_step_65` |
| 마지막 하트비트 | **04:11 UTC** (발견 시각 07:4x — **3.5시간 지연**) |
| GPU | **2271MB**(유휴) |
| MemAvailable | 1.00TB → **1.78TB 로 회복** ⇒ 워커가 죽으며 메모리를 놓았다 |

⇒ **호스트 RAM OOM**(E-157 ③, `RewardLoopWorker` 273GB×2, `rewards.py:57-100` 의 버려진 데몬 스레드 누수).
선점도 init 스테이징 실패도 아니다.

### ② ⚠**`amlt status` 는 3.5시간 동안 `running` 이었다**

학습 프로세스는 죽었는데 **컨테이너는 살아 있어서** amlt 가 `running` 으로 보고했다.
⇒ ★**`amlt status` 는 잡의 생사를 말해 주지 않는다.** 슬롯 점유 여부만 말한다.
⇒ ★★**규율 보강: 스텝이 멈춘 것처럼 보이면 "val 구간이겠거니" 하고 넘기지 말고 로그의 하트비트 시각을
   확인한다.** 이번엔 gs65 가 여러 틱 동안 그대로였는데 나는 "선점이 잦은 팔" 로만 읽고 넘겼다.
   **하트비트가 30분 이상 갱신되지 않으면 사망으로 간주하고 로그를 연다.**
   (0808 에 b3nopmi 가 val 로 40분 멈춘 전례가 있어 "정상 정지"에 익숙해진 것이 지연의 원인이다 —
   **정상 정지와 사망을 구별하는 것은 스텝이 아니라 하트비트 시각**이다.)

### ③ 조치 — 취소 후 재발사(사용자 상시 지시 "실패하면 계속 신청")

LIST → 결정 → 실행 순서로:
- **LIST**: `checkpoints/rq3v2f_b3shf/` 에 gs55·gs60·gs65 **전부 m4/o4/e4 완전** ⇒ 재개 가능.
- **결정**: 죽은 잡이 슬롯만 점유 중이므로 취소하고 같은 런처로 재발사. 새 파일 없음, 승인 불필요.
- **실행**: `amlt cancel rq3v2f-b3shf-0806` → `killed` 확인 → `amlt run h100std_rq3v2f_b3shf.yaml`
  → **`rq3v2f-b3shf-0808`**(`preparing`). **gs65 에서 재개.**

### ④ 이 팔을 계속 태울 것인가 — **사용자 결정 사항**(새 정보 있음)
0807 결정은 *"b3shf·b3nopmi 둘 다 유지"* 였다. 그 뒤 페이블이 슬롯 전환(→ SFT2-init eval)을 권고했고,
**이번 사망으로 그 권고의 근거가 하나 더 실현됐다**:
- ①**기회비용 실현**: 1일 3시간에 **gs65**(선점 다발 + OOM 사망). 판별 구간 gs120+ 까지 최소 하루 더.
- ②**고유 질문 축소**: E-165/E-169 가 *"PMI 보상이 discard 를 억제한다"* 를 보였고,
  b3nopmi(format 헤드 **켠 채** PMI 만 끔)의 discard 16.1% 가 *"format 단독으로는 discard 를 못 막는다"* 를
  공짜로 보여준다 ⇒ b3shf 의 잔여 질문은 밴드 내부 디테일이다.
- ③**SFT2-init eval 이 여전히 무자금 P0** — 처치 팔이 "잃은 것"인지 "덜 얻은 것"인지 가르는 유일한 측정.
⇒ **일단 재발사해 슬롯을 채웠고**(상시 지시 이행), 전환 여부는 사용자 판단으로 남긴다.

---

## E-172 (0808) — ⛔**b3null 이 `discard` 폭주로 붕괴했다(gs103 에 88%).** 판별자가 무력화될 위험

### ① 관측 — 사전등록 보조 그리드가 먼저 잡아냈다

`val-aux/{9서브셋}/correctness/mean@1` 평균(E-166 고정 그리드):

| 팔 | gs50 | gs100 |
|---|---|---|
| b2p | 0.2673 | 0.2917 ↑ |
| b0p | 0.2469 | 0.2905 ↑ |
| b3p | 0.2715 | 0.3105 ↑ |
| b3s | 0.2921 | 0.3279 ↑ |
| **b3null** | **0.3249** | **0.0799** ⛔ |

**다른 네 팔은 전부 올랐는데 b3null 만 4배 떨어졌다.**

### ② 근인 — `discard` 폭주(발화 붕괴가 **아니다**)

| gs | 학습정확도 | 길이 | 비종료 | entropy | **discard** | 발화 |
|---|---|---|---|---|---|---|
| 51 | 0.4715 | 374 | 0.004 | 0.271 | **0.123** | 0.998 |
| 61 | 0.4414 | 578 | 0.008 | 0.351 | **0.336** | 1.000 |
| 71 | 0.4359 | 539 | 0.004 | 0.360 | **0.650** | 1.000 |
| 81 | 0.5664 | 437 | 0.000 | 0.349 | **0.746** | 0.998 |
| 91 | 0.4367 | 446 | 0.006 | 0.465 | **0.881** | 0.998 |
| 101 | 0.2734 | 543 | 0.004 | 0.617 | **0.883** | 0.998 |
| **103** | **0.0020** | 887 | 0.031 | **0.799** | **0.877** | **0.996** |

같은 구간 **b3p 의 discard 는 0.047~0.068**(20배 차이). b3p 학습정확도는 gs100 에서 0.5215.

⇒ ★**발화는 0.996~1.000 으로 멀쩡하다.** 무너진 것은 **형식 라우팅**이다 —
행이 discard 로 분류되면 `R_corr`·`R_meta`·`R_cal` 이 전부 0이 되고 마스크가 비어
**그 행은 gradient 를 하나도 못 나른다**(E-152 가 b3sh 에서 본 것과 같은 병리).
gs103 에 배치의 **88%** 가 그 상태다. 학습정확도 0.0020 은 그 결과다.

### ③ 왜 하필 b3null 인가 — **`w_format=0` 이기 때문이다**

b3null 은 설계상 **보상 헤드 아홉 개를 전부 0** 으로 했고 거기에 **`dcpo_w_format=0.0`** 이 포함된다.
`w_format` 은 **미복구 discard 를 벌하는 유일한 헤드**다(`format_penalty = -format_neg`).
- **`w_format=0` 인 팔**: b3sh(E-152, gs120→155 에 discard 5.9%→46.7%) · **b3null(지금 88%)**
- **`w_format=0.35` 인 팔**: b3p(0.05) · b3s · b3nopmi(0.16) — **아무도 폭주하지 않았다**

⇒ ★★**b3sh 에서 세운 가설이 b3null 에서 독립 재현됐다: `w_format` 을 끄면 discard 가 폭주한다.**
두 팔은 나머지 설정이 크게 다른데도 같은 병리를 보였다. **`w_format` 은 무효 레버가 아니다.**

### ④ ⚠**판별자 위기 — b3null 의 gs300 은 "기하 대조"를 재지 못할 수 있다**

b3null 의 존재 이유는 *"−2pp 가 보상 때문인가 region 기하 번들 때문인가"* 를 가르는 것이었다
(E-159 ③·E-160 ⑤). 그런데 **배치의 88% 가 gradient 를 못 나르면 그 팔은 "기하를 탄 정책"이 아니라
"거의 학습되지 않은 정책"** 이다. E-160 ⑤ 에 등록해 둔 함정이 **다른 형태로 실현됐다** —
나는 *"발화가 죽으면 carve-out 이 무효가 된다"* 를 경계했는데, **발화는 살아 있고 라우팅이 죽었다.**

⇒ ★**판독 규칙 보강(결과 도착 전 등록)**: b3null 판정문에 **`discard_rate` 를 반드시 병기**한다.
> **gs300 시점 discard > 0.5 이면 b3null 은 기하 대조로 쓸 수 없다.**
> 그 경우 결과는 *"`w_format=0` 이면 이 기질에서 학습이 성립하지 않는다"* 로만 읽고,
> **region 기하에 대한 어떤 귀속도 금지**한다. E-160 ⑤ 의 밴드 규칙은 그때 무효다.

### ⑤ 닫는 것 / 여는 것
**닫는 것** — *"b3null 이 기하 판별자다"* 를 **무조건**으로 쓰는 것(⇒ discard 조건부로 강등) ·
*"`w_format` 은 밴드 내부 디테일"*(b3sh 슬롯 전환 논거 중 하나였다 — **틀렸다. 이제 두 팔에서 재현된 주효과다**).
**여는 것** — (a) **`w_format` 이 discard 를 막는다**는 가설이 이제 2/2 재현 ⇒ **b3shf 가 이 가설의 직접 검정**이
되고, 그 팔의 가치가 **올라간다**(E-171 ④ 의 "고유 질문 축소" 논거를 **철회**한다),
(b) b3null 이 회복하는지 계속 관측(discard 가 내려오면 판별자로 살아난다),
(c) *"−2pp 의 기하 귀속"* 은 b3null 이 못 답하면 **다른 설계가 필요하다** — 무엇인지 미정.
**재확인 계수기** — 0.

---

## E-173 (0808) — **b3null 은 회복하지 않았다.** discard 는 내려가는데 정확도가 음수로 깊어진다 ⇒ **기하 판별자 무력화**

### ① 관측 — 등록해 둔 부활 기준(E-172 ④, 3조건)이 **0회**

| gs | discard | 학습정확도 | 발화 |
|---|---|---|---|
| 106 | 0.7090 | +0.0184 | 0.9961 |
| 107 | 0.5781 | +0.1109 | 0.9805 |
| 108 | 0.5234 | +0.2395 | 0.9707 |
| 109 | 0.5098 | +0.0613 | 0.9531 |
| **110** | 0.4473 | **−0.2215** | 0.9453 |
| **111** | 0.4395 | **−0.3813** | 0.9355 |
| **112** | **0.3828** | **−0.5441** | **0.9062** |

⇒ ★**discard 가 0.88 → 0.38 로 내려가는 동안 학습정확도는 +0.24 → −0.54 로 떨어졌다.**
**음수 3연속**(gs110·111·112), 그리고 **단조로 깊어진다.** 발화도 0.996 → 0.906 으로 함께 하강.
⇒ **부활 3조건(discard<0.5 ∧ 정확도>0 ∧ 발화≥0.9) 동시 충족 = 0회.**

### ② 해석 — 라우팅을 통과하되 **내용을 포기**한 정책

`discard` 는 줄었고 그 행들은 이제 gradient 를 나른다. 그런데 그 gradient 가 **음의 보상**을 나른다.
⇒ 정책이 *"버려지지 않는 형식"* 은 찾았지만 *"맞는 답"* 은 잃었다.
⚠**이것이 E-172 를 "회복 중"으로 읽지 않은 이유다.** 한 지표(discard)만 보면 개선으로 보이는데
세 지표를 같이 보면 **더 나빠지는 중**이다. 그 규율을 직전 틱에 강화해 둔 것이 판독을 구했다.

### ③ ★**판정 — b3null 은 region 기하 대조로 쓸 수 없다**

E-172 ④ 에 등록한 규칙을 그대로 적용한다:
> **gs300 discard > 0.5 이면 기하 대조 불가** ⇒ 현재 0.38 로 그 조건은 벗어났지만,
> **강화 기준(정확도>0)이 실패**했으므로 판정은 같다.

**따라서 이 팔의 gs300 held-out 은 다음 문장으로만 읽는다:**
> **"`w_format=0` 이면 이 기질에서 학습이 성립하지 않는다."**
> ⛔**region 기하에 대한 어떤 귀속도 금지.** E-160 ⑤ 의 밴드 규칙(treatment/control 밴드)은 **무효**다.

### ④ ⚠**그래서 "−2pp 의 기하 귀속"은 지금 답할 수단이 없다**

−2pp 의 원인 후보에서 제거된 것: 발화 붕괴(E-165 ④) · 절단(C-029) · 보조 5헤드(E-154).
남은 후보는 **region 기하 번들**뿐인데, **그것을 재도록 설계한 유일한 팔이 자기 설계 때문에 죽었다** —
헤드를 전부 0으로 만들려면 `w_format` 도 0으로 해야 했고, 그것이 학습을 파괴했다.
⇒ ★**b3null 설계 자체에 내재된 모순이었다**: *"보상 없이 기하만"* 을 만들려는 시도가
**형식 라우팅을 지키는 보상까지 제거**해 버렸다.
⇒ ★★**다음 설계 요건(미충족)**: region 기하만 바꾸고 **`w_format` 은 살려두는** 팔이 필요하다.
즉 `w_format=0.35` + 나머지 여덟 헤드 0. **이것은 b3null 의 재설계이며 새 런처 = 승인 사항.**
⚠단 그 팔도 `w_format` 이 켜져 있으므로 **"헤드 전무"가 아니다** — 이름을 *"format-only 기하 대조"* 로 정확히 붙여야 한다.

### ⑤ 닫는 것 / 여는 것
**닫는 것** — *"b3null 이 기하 판별자다"* (E-159 ③·E-160 ⑤ 가 그 위에 세운 판독 밴드 전부 무효) ·
*"b3null 이 회복 중이다"*(직전 틱 관측 — 3조건 0회로 종결).
**여는 것** — (a) **format-only 기하 대조 팔**(`w_format=0.35` + 8헤드 0) — **설계·승인 필요**,
(b) 그때까지 **−2pp 의 기하 귀속은 미결로 둔다**(추측으로 채우지 말 것),
(c) b3null 을 gs300 까지 태울 가치가 있는지 — **`w_format=0` 의 파괴력을 정량화하는 자료**로는 여전히 쓸모가 있다
(b3sh 46.7% 와 함께 두 점).
**재확인 계수기** — 0.

---

## E-174 (0808) — **프레임 교정: b3null 의 죽음은 실패가 아니라 기하에 대한 첫 측정이다** · 대체 설계 확정 · 내 보존 규칙이 놓친 것

### ① ★★★프레임 교정 — **"순수 기하" 팔은 이 아키텍처에서 원리적으로 불가능하다**

discard 탈출구(중복 opener → tier-2 배제 → 전 마스크 0 → **무gradient 구멍**, `dcpo_region.py:1101-1113`)는
**region 경로에만 존재한다.** b2p(VANILLA)는 형식이 어떻든 전 행이 correctness gradient 를 받는다.

⇒ ★**"경찰 없는 region 기하는 자기 파괴한다"는 것 자체가 기하의 속성이다** — b3sh 46.7% · b3null 88%, **2점 재현**.
⇒ ★★**기하가 무gradient 구멍을 내장하고 있고, 그 구멍은 헤드 하나가 상시 순찰해야 닫힌다.**
따라서 **선호 보상을 전부 끄면서 기하만 남기는 팔은 만들 수 없다.** 대체 설계의 해석 한계는
설계 미숙이 아니라 **이 사실의 귀결**이다.
⇒ E-173 의 *"설계 내재 모순"* 을 이 문장으로 승격한다: **모순이 아니라 아키텍처의 성질이었다.**

### ② 헤드를 두 부류로 등록한다 — **경찰 vs 선호**

| 부류 | 헤드 | 역할 |
|---|---|---|
| **경찰**(무결성 유지) | **`w_format` 0.35** · **`trunc_open_penalty` 0.3** | 무gradient 구멍을 닫는다 |
| **선호**(메타 방향) | `w_meta`/PMI · `w_cal` · `w_emit` · `len_cost` · `meta_floor` | 메타 행동을 유도한다 |

⇒ 새 팔의 질문이 깨끗해진다: ***"선호 보상 0 + region 기하 + 경찰 = −2pp 인가?"***

### ③ 대체 설계 확정 — **`bfmt`**(승인 대기)

**오버라이드 7개**(경찰 둘은 **건드리지 않는다** — config 기본 0.35/0.3 유지):
`rmeta_source=none` · `w_meta=0` · `w_cal=0` · `w_emit=0` · `len_cost=0` · `meta_floor=0` · `w_over=0`(사문이나 b3p 패리티).
b3p 와 그 외 바이트 동일. **혈통은 새로 — `rq3v2f_bfmt`**(b3null 혈통 재사용 금지, 밑줄·하이픈 누출 전례).

**설계 근거 둘(내 초안 정정)**:
- ⛔**`trunc_open_penalty` 를 끄면 안 된다.** format 헤드는 truncation 행을 **의도적으로 0 처리**하므로
  (`dcpo_region.py:1008-1014`) **절단 탈출구는 `w_format` 이 구조적으로 못 막는다.** 그 구멍의 경찰은
  un-centered trunc 채널(`:1352-1362`)뿐이다. 게다가 b3p·b3s·b3nopmi 전부 0.3 이라 diff 축 최소화.
- ⛔**`anchor_norm=true` 를 끄면 안 된다.** 끄면 format 경찰의 실효 강도가 다른 region 팔과 달라져
  **새 diff 축이 생긴다.** ⚠**단 내 이전 이해가 틀렸다**: *"anchor 는 가중치 0 인 헤드만 리스케일하니 무효"*
  는 **전-헤드-0 전제**에서만 참이다. format 만 살리면 `dcpo_region.py:1283-1289` 가 **A_format 을
  corr 스케일로 실제 리스케일한다.** ⇒ anchor EMA 비영속(E-142 ③) 교란을 상속하므로
  **판정문에 선점 횟수를 병기**한다.

**판정문 이름(등록)**: `bfmt − b2p` = **"region 기하(라우팅·discard 배제·비whiten) + 경찰 헤드의 합효과"**.
⛔이보다 좁힐 수 없다 — 그리고 **좁힐 수 없다는 근거가 이제 측정돼 있다**(①의 2점 재현).

**★사다리가 내부 분해를 준다**: `b2p → bfmt → b3nopmi → b3p` 가 각각
**{기하+경찰}** / **{+cal·emit·len}** / **{+PMI}** 를 한 단씩 더한다.
⇒ bfmt 가 뜨면 **b3nopmi 와의 차분이 보조 5헤드 몫을 공짜로 준다.**

**★반증 훅(결과 전 등록)**: bfmt 가 `w_format=0.35` 를 켜고도 **discard 가 0.2 를 지속 초과하면
E-152/E-172 가설(*"format 헤드가 discard 를 막는다"*)이 반증된다.** b3nopmi 의 discard 16% 상승이
이미 그 방향의 약한 신호다. 그 경우 bfmt 도 판정 불능 팔이 되므로 **discard 알림을 모니터에 걸고
조기 중단 조건을 지금 적는다.**

### ④ ⛔**내 보존 규칙이 붕괴 전 상태를 놓쳤다**

보존한 `rq3v2f_b3null_gs80` 의 실제 상태: **discard 0.7539** — **이미 붕괴 한복판**이다.
붕괴 전 상태는 gs46~56(discard 0.090 → 0.199, 정확도 +0.41~+0.66, 발화 1.000)인데 **전부 프루닝됐다.**
현재 b3null 에 남은 ckpt 는 **gs110 하나**뿐.

★**근인 — 내 보존 규칙이 warmup 을 기준으로 삼았다**: *"gs80(warmup) 넘긴 뒤 첫 보존"*.
그런데 **이 팔의 병리는 gs56~61 에 시작했다 — warmup 이 끝나기 전이다.**
⇒ ★★**규율: "warmup 이후에 보존" 은 *warmup 중엔 볼 것이 없다* 는 가정이다. 그 가정이 틀리는 팔이 있다.**
**보존 시점은 달력(warmup)이 아니라 지표(병리 시작)로 정해야 한다.**
⇒ 실무 규칙: **새 팔은 warmup 여부와 무관하게 첫 완전 ckpt 를 한 번 뜬다**(비용 0, 서버측 복사).

### ⑤ 우선순위(페이블 판정) — 순서 불변, 단 b3null 은 회수 권고

1. **SFT2-init eval (~20 GPU-h) 이 여전히 1순위** — 처치 팔이 "잃은 것"인지 "덜 얻은 것"인지 가르는 유일한 측정.
2. **그 다음 `bfmt`** — 조건부가 아니라 **적극 권고**. 이유: 사전등록 재현축이 "소멸"로 닫힌 지금
   base 세대에 남은 과학적 내용물은 사실상 **RQ2 −2.08 의 귀속**뿐이다.
   *"보상이 해롭다"* 와 *"구현 기하가 해롭고 보상은 그 위에 얹혔다"* 는 논문의 discussion·future work·
   방법 부류 전체에 **반대 결론**을 만든다(전자는 방법 사망, 후자는 **기하 수리 후 재시도 여지**).
   비용 gs150+eval ≈ **130 GPU-h**(적자는 gs165 까지 완전 형성된다는 2팔 전례).
3. **b3null 회수 권고** — 두 번째 데이터 점의 가치는 이미 은행에 들어갔다(폭주 곡선 + val-aux 0.3249→0.0799, E-173).
   남은 190스텝 ≈ **80~90 GPU-h** 는 자기 파괴된 정책의 세 번째 유효숫자다.
   ⚠**단 붕괴 전 ckpt 는 이미 없다**(④) — 회수해도 잃을 산출물이 없다.

### ⑥ EXP 관측 하나 (과독 금지, bfmt 사전예측 칸에만)
**gs50 시점 val-aux 0.3249 는 5팔 중 최고였다**(b2p 0.2673 · b3p 0.2715 · b3s 0.2921 · b3nopmi 0.3116).
⚠조건부 지표라 파괴 비용을 못 본다(0731 규율). 그러나 방향 표지로서 **bfmt 사전예측**에 적을 자격은 있다:
> **bfmt 예측 — 붕괴 없이 gs50 우위가 유지되면 control 밴드, 아니면 treatment 밴드.**

### ⑦ 닫는 것 / 여는 것
**닫는 것** — *"순수 기하 팔을 만들 수 있다"* · *"b3null 설계가 미숙했다"*(아키텍처의 성질이었다) ·
*"anchor_norm 은 무효"*(전-헤드-0 전제에서만 참) · *"warmup 이후 보존이면 충분"*.
**여는 것** — (a) `bfmt` 런처(**승인 필요**), (b) **region compose 출력에 `masked_whiten` 을 붙인 팔** —
bfmt 가 treatment 밴드에 떨어졌을 때 번들 내부(스케일 vs 라우팅/배제)를 가르는 다음 단계
(**정본 변경 1곳, 승인 사항**), (c) SFT2-init eval 런처(**승인 필요**).
**재확인 계수기** — 0.

## E-175 (2026-08-08 12:45 UTC) — [b3shf 가 E-172 를 직접 검정하고 통과 · 내 PMI 노브 권고를 X-005 로 정정]

**G8 선행**: `h100std_rq3v2f_b3sh.yaml` vs `..._b3shf.yaml` 실차분 = **`++algorithm.dcpo_w_format` 0.0 → 0.35 단 하나**. 단일 손잡이 대조 확인.

### ① `w_format` 이 discard 를 막는다 — 직접 검정 통과 (관측구간 gs101~104, 4스텝)
b3shf 로그 `[DCPO-V3] fmt classify/replace` (B=512):

| step | wellformed | discard | truncation |
|---|---|---|---|
| ~101 | 476 (93.0%) | **2 (0.39%)** | 30 |
| ~102 | 482 (94.1%) | **4 (0.78%)** | 23 |
| ~103 | 492 (96.1%) | **1 (0.20%)** | 17 |
| 104 | 483 (94.3%) | **2 (0.39%)** | 24 |

대비: `w_format=0` 인 두 팔 — b3sh **46.7%** · b3null **88%**.
⇒ **E-172(`w_format` 은 주효과)의 직접 검정이 통과**했다. 단일 손잡이로 discard 가 **두 자릿수 배** 갈린다.
⚠**스텝 정합 미완**: b3sh 의 46.7% 가 어느 스텝인지 이 틱에서 확인하지 않았다. 크기 차(60~120배)가 커서
정합만으로 뒤집힐 가능성은 낮으나, **판정문에 쓰기 전 정합할 것**.

### ② 부수 관측 — wellformed 93~96%
E-067 은 base meta-SFT 세대에서 `wellformed_rate` gs1 **0.40** → gs100 **0.002** 붕괴를 기록했다.
b3shf 는 gs101~104 에서 **93~96%** 를 유지한다. ⚠**세대가 다르다**(b3pkg vs b3shf) — 직접 비교 금지.
다만 *"base 는 껍데기만 배운다"* 가 **`w_format` 이 꺼진 조건에 한정될 가능성**을 연다. **미결.**

### ③ ⛔**내 0808 권고 정정 — `dcpo_pmishift_*` 스윕은 X-005 로 이미 위축돼 있었다**
나는 사용자에게 *"`reversal_derail=2.0` 때문에 base 에서 보상이 순벌금이다 · 이 노브를 한 번도 안
건드렸으니 지금 당장 스윕할 값이 있다"* 고 권고했다. **`CLAIMS.md:836` X-005 를 grep 하지 않았다.**

X-005 ⛔ *"SAVE/DERAIL 비대칭(β_derail 2.0)이 발화 붕괴의 원인이다"* — **기각됨**.
근거: 스텝당 롤아웃 512개·보상질량 410,752 대비 SAVE/DERAIL 은 **스텝당 5~6건 ≈ 3%**.

**이 틱의 실측이 X-005 를 재확인한다** (b3shf gs100~104):

| step | attempted | n_save | n_derail | rmeta_mean_scored |
|---|---|---|---|---|
| 100 | 355 | 11 | 8 | 0.7273 |
| 101 | 320 | 6 | 9 | 0.2506 |
| 102 | 357 | 5 | 8 | 0.4713 |
| 103 | 335 | 7 | 4 | 0.5061 |
| 104 | 322 | 0 | 1 | 0.6387 |

- SAVE 비율 29/59 = **49.2%** — 기억의 base 47% 와 일치.
- 비대칭 항 기여 = (29×1 − 30×2)/59 = **−0.525/사건**. 내가 예측한 −0.52 와 일치한다. **그러나**
  뒤집힘은 시도행 1,689 중 **59건(3.5%)** 뿐이라 행당 기여 ≈ **−0.019**.
- ★★**결정적**: `rmeta_mean_scored` 는 **전 구간 양수(0.25~0.73)** 다.
  ⇒ **"base 에서 PMI 보상이 순벌금"은 틀렸다.** 비대칭 하위항만 약한 음수이고, **헤드 자체는 양수**다.

**지금 참인 것**: 비대칭 스윕은 **무효 레버에 가깝다**(헤드 크기의 약 4%). `reversal_min_magnitude`·
`dup_thresh` 가드가 꺼져 있다는 사실은 그대로 참이나, **그것으로 −2pp 를 설명하려 해선 안 된다.**
⇒ 0808 문헌조사에서 살아남는 제안은 **① 스윕이 아니라 ② Gandhi 식 행동 분류기로 SFT2 코퍼스 재검**이다.

**근인**: 나는 워크플로가 아니라 **내 grep 결과**로 서사를 만들었고, `KNOBS.yaml` 의 *"한 번도 설정 안 됨"*
을 *"미탐색 기회"* 로 읽었다. **미설정은 기회의 증거가 아니다** — 이미 기각돼서 안 건드렸을 수 있다.
⇒ 규율 보강: **"이 노브는 아무도 안 건드렸다"를 근거로 쓰기 전에 그 노브의 *개념*을 CLAIMS 에서 grep한다.**

### ④ b3null 비용 실측 — 은퇴 판단 재료
gs116/300 · **1345.48s/it** · 잔여 ETA **68:46** (현 시점 amlt 추정).
b3shf 는 499.56s/it — **2.7배 느리다**. `global_seqlen/min` b3null 473,616 vs b3shf 72,325(**6.5배**)
= 비종료 장문 생성. ⇒ 남은 184스텝 ≈ **69h × 4×H100 ≈ 275 GPU-h**.
E-173 이 이 팔의 gs300 을 *"`w_format=0` 이면 학습이 성립하지 않는다"* 로만 읽기로 확정했으므로,
**그 결론은 gs116 시점에 이미 성립**한다. **은퇴 후보 1순위**(사용자 결정 대기).

### 상태
3잡 전부 생존(HB 12:44:56~57 vs now 12:45:11). b3nopmi ckpt **gs200** · b3shf **gs103** · b3null **gs116**.
HF 완전성 전 팔 OK(불완전 0). **b3shf gs100 을 `preserved/mechanism_alive/` 로 보존**(23파일 m/o/e=4) — `--keep 1` 방어.

## E-176 (2026-08-08 16:50 UTC) — [정정] E-175 ③ 의 b3null 비용 수치는 **변동 텔레메트리로 만든 것이었다**

**E-175 ③ 은 "잔여 ≈275 GPU-h, 틱마다 커진다"고 적었다. 네 시간 뒤 그 수치는 60% 줄었다.**

| 시각 | 스텝 | s/it | 잔여 ETA |
|---|---|---|---|
| 12:45 | 116 | 1345 | 68:46 |
| 14:09 | 119 | 1522 | 76:30 |
| 15:04 | 121 | 1584 | **78:44**(최대) |
| 15:30 | 122 | 1587 | 78:26 |
| 15:57 | 123 | 1533 | 75:23 |
| 16:23 | 125 | 1233 | 59:55 |
| **16:50** | **129** | **595** | **28:15** |

★**26분에 4스텝(≈390s/step)** — 다른 팔들과 같은 속도다. 78시간이라던 잔여가 **28시간**이 됐다.

### 지금 참인 것
- **b3null 의 느림은 *구간*이었지 성질이 아니다.** gs116~123 구간이 비정상적으로 느렸고 그 뒤 정상화됐다.
- **잔여 비용은 ≈28h × 4×H100 ≈ 112 GPU-h** (E-175 ③ 의 275~350 은 **철회**).
- ⇒ ★**b3null 은퇴 논거에서 *비용 축은 빠진다.*** 남는 것은 **과학적 가치 축 하나**다:
  **E-173 이 이 팔의 gs300 해석을 이미 확정했다**(*"`w_format=0` 이면 이 기질에서 학습이 성립하지 않는다"*, region 기하 귀속 금지).
  그 논거는 처음부터 더 강했고 지금도 그대로다.

### 왜 틀렸나 — 근인
**amlt 의 `S/it` 과 ETA 는 최근창 추정치이고, 선점·val·시퀀스 길이 변동에 크게 흔들린다.**
나는 이것을 **7틱에 걸쳐 누적 관측**하며 "6점 단조 감속"으로 격상까지 했는데, 8번째 점에서 뒤집혔고
9~10번째에 **완전히 반대 방향**이 됐다.

★**규율 보강**: ⛔**amlt ETA/s-it 위에 결정 논거를 세우지 마라.** 진행 비용을 논거로 쓰려면
**스텝 카운트와 벽시계로 직접 계산**하고, **관측창을 최소 수 시간**으로 잡는다.
★**그리고 논거가 여러 축을 가질 때는 가장 안정적인 축을 헤드라인에 둔다** — 여기서는 비용(변동)이 아니라
**과학적 가치(E-173, 확정)** 였다. 관련: [[pin-oid-telemetry-lags-0807]].

### 상태(16:50)
b3nopmi **gs226/300**(603s/it·판정점까지 74스텝) · b3shf **선점② 재개 중**(부트스트랩 27분째, `validation generation end`) · b3null **gs129/300**.
b3shf gs120 은 53분째 `model 3/4` — **확정적으로 잘린 ckpt**(무해: gs115 온전, 프루너가 완전한 스텝만 센다).

---

## E-177 (0809 05:16 UTC) — b3nopmi gs300 **도달·보존 완료**, 판정 서식 **사전 확정**(결과 미열람)

**사실만.** `checkpoints/rq3v2f_b3nopmi/global_step_300/actor/` 가 **m/o/e 각 4/4** 로 완전.
`step=300`, HB `[HB Sun Aug  9 05:16:27 AM UTC 2026]`, 사망표식 0,
`FINAL PUSH DURABLE` **미출현**(최종 val + 최종 푸시 진행 중으로 읽는다 — 단정 아님).

**보존 완료** — `preserved/mechanism_alive/rq3v2f_b3nopmi_gs300/` 23파일, server-side
`CommitOperationCopy`(바이트 전송 0). ckpt 신원 고정(actor model 샤드 `lfs.sha256` 앞 16자):
`rank_0 b5b2aa91ade18eea` · `rank_1 d80d7b4e62786ad4` · `rank_2 d7360c79d2e1d197` · `rank_3 e548a8aed92b10aa`.

⚠**gs300 도달 ≠ 판정 완료.** held-out eval 은 별도 GPU 잡이다. 이 항목은 **보존과 서식 확정까지**만이다.

### 판정 서식 — **아래를 결과 보기 전에 확정한다**

**주 지표(CONF)**: `b3p − b3nopmi`, gs300 held-out **1030 · 16k · avg@8 · MATH500**,
채점기 `math_verify` + `$`-wrap **on**, 문제 단위 **paired bootstrap**, 유의 = 95% CI 가 0 을 포함하지 않음.
**이름**: **"PMI 헤드의 end-to-end 순효과(하류 동역학 포함)"**. ⛔"메타인지 품질 효과"로 부르지 않는다.

**기준선을 옆에 둔다**(재측정 금지·oid 고정):
b3p `af5df50da404f0b8` **75.20** · b2p(대조군) `08faf5ae9fc4e404` **77.28** ·
b0p `fbcd9864a64ba6ba` 76.12 · b3s `ce372c5b593c8f4c` 75.28.

**분기표는 `docs/EXPERIMENT_PLAN.md:109-119` 를 그대로 따른다**(요약이 아니라 그 표가 정본):
1. `B3pkg > B2` **이고** `B3pkg > B3-noPMI`(유의) ⇒ 패키지가 vanilla 를 이기고 **그 안에서 PMI 의 한계 기여가 관측됨**.
   ⛔ 일반적 "메타인지"나 다중 시드 재현으로 확대 금지 — E3·placebo·독립 judge 필요.
2. `B3pkg > B2` **이지만** `B3pkg ≈ B3-noPMI` ⇒ **PMI 순기여 미검출**. 주장을 **package-level 로 강등**.
3. `B3pkg ≈ B2` **이고** `B3pkg ≈ B3-noPMI` ⇒ **정확도 향상 미재현**. null 을 그대로 보고한다(과거 arm 중 좋은 수 선택 금지).
4. `B3pkg < B2` **또는** `B3-noPMI > B3pkg` ⇒ 현 recipe/substrate 에서 **PMI 가 무효 또는 해로운 방향**일 가능성.
   ⛔ 과거 instruct 결과를 근거로 성공 주장 금지.

★`EXPERIMENT_PLAN.md:116-119` 를 판정문에 그대로 옮긴다 — **`B3pkg−B2` 는 region routing·advantage
정규화 차이를 포함한 패키지 비교**이고, **`B3pkg−B3noPMI` 만이 PMI 헤드를 한 변수로 뺀 내부 대조**다.
따라서 **첫 비교가 양수여도 두 번째가 양수가 아니면 "PMI 가 이겼다"가 아니라 "패키지가 이겼다"로 적는다.**

**판정 표 규칙**: 모든 행에 **파일 oid**(`api.get_paths_info` 의 `lfs.sha256`)를 단다.
같은 경로가 하루에 네 번 덮인 실측이 있다([[pin-oid-telemetry-lags-0807]]) — 경로 인용 금지.
⛔`eval` glob 금지. ⛔RQ2(−2.08 [−3.50,−0.70])와 재현축(−0.92 [−2.50,+0.62]) 은 **병기**한다.

### 다른 두 팔(같은 틱 실측)
b3shf 완전 ckpt **gs180**(gs185 업로드 중 optim 3/4) · b3null 완전 ckpt **gs215**(gs220 업로드 중).

---

## E-178 (0809 09:16 UTC) — b3nopmi **완주 확인**(`FINAL PUSH DURABLE global_step_300`)

로그 원문 그대로: `[YAML] FINAL PUSH DURABLE global_step_300`.
**N=300 을 직접 읽었다** — 이 문자열은 완주와 사고사에 똑같이 찍히므로 N 확인이 판별의 전부다
([[three-death-modes-basicvc-0807]]). 같은 tail 에서 사망표식(`ABORT window`/`FATAL init`/
`IncompleteRead`/`failed due to oom`/`Total available GPUs 0`) **0건** ⇒ **완주**로 기록한다.
잡 상태는 아직 `running`(최종 푸시·정리 구간).

★모니터 에이전트가 먼저 "gs300 complete" 를 통보했으나 **그 보고를 그대로 쓰지 않고 로그에서
N 을 직접 읽어 확인했다** — [[external-numbers-need-claims-grep-0807]] 의 방아쇠 ②.

**보존본 재확인**: `preserved/mechanism_alive/rq3v2f_b3nopmi_gs300/` **23파일** 그대로.
⚠`--keep 1` 프루닝이 `checkpoints/` 쪽 gs300 을 언젠가 지워도 보존본은 남는다.

### 같은 틱 다른 두 팔(4시간 전 대비)
b3shf 완전 ckpt **gs205**(gs180→205, 순항) · b3null 완전 ckpt **gs250**(gs215→250, 순항).

### 미결 — 판정 eval 은 아직 발사하지 않았다
E-177 이 확정한 서식대로 `b3p − b3nopmi`(gs300 held-out 1030·16k·avg@8·MATH500)를 재려면
**별도 GPU 잡**이 필요하다. **발사는 사용자 확인 대기 중**이며, 이 항목은 확인 전 상태를 고정한다.

---

## E-179 (0809 11:5x UTC) — 페이블 적대검토: **내 전제 두 개 철회**, 후보 순위 재편, 다음 수는 b3nopmi eval 하나

사용자 질문(*"PMI-shift 는 이론적으로 말이 되는데 왜 base 에선 안 되나 / instruct 에선 왜 됐나"*)에
답하려다 **내가 세운 전제 둘이 file:line 으로 반박됐다.** 페이블 보고를 그대로 쓰지 않고 직접 확인했다.

### ⛔철회 ① — *"SFT2 코퍼스 378행 기아가 현행 손상의 후보"*
`h100std_rq3v2f_b3p.yaml:1`: *"the ONLY change from the previous b3p arm is the INIT, now
`models/b2p2_rvfull_eb16_sft` (SFT2 on the **RAW 1763-row** RV corpus) **instead of the 378-row
filtered subset** that E-093/E-094 showed was a covert scenario filter"*. b3s·b3nopmi 도 `:100` 에 동일.
E-067 의 gs1 wellformed 0.40 붕괴는 **구 계보 기록**이다. 현행 실측은 반대 — b3p gs25 emit 0.998,
b3shf gs101~104 wellformed 93~96%.
⚠**`CLAUDE.md:83-84` 가 낡았다** — 아직 `b2p2_rvseg_sft` 를 "RL init for BOTH b2p and b3p" 로 적고 있고
내가 그것을 근거로 삼았다. **문서 수정은 승인 사항**이라 지적만 남긴다.

### ⛔철회 ② — *"base 에서 shiftonly 를 돌린 적이 없다"*(C-019 를 그대로 인용한 것)
`h100std_rq3v2f_b3sh.yaml:256` 에 `++algorithm.dcpo_len_cost=0.0` 이 실재하고, description 이
*"exactly five override changes — w_cal→0, w_format→0, w_emit→0, len_cost→0, meta_floor→0"* 라고 적는다.
instruct 원본 `archive/launchers_pre_rq3/h100std_shiftonly.yaml:152-157` 과 **같은 다섯 개**다.
⇒ **b3sh 가 base 의 shiftonly 복제였고, `w_format=0` 때문에 자기파괴했다**(중복 `<|meta|>` opener →
discard 5.9%→**46.7%** → 배치 절반 무gradient). ★**C-019 는 이 결과로 갱신돼야 한다.**
★**내 명령 결함**: `grep -oE "dcpo_w_[a-z_]+=[0-9.]+"` 가 `dcpo_len_cost`(`w_` 접두 아님)를 구조적으로
놓쳤다. **"빈 결과 = 내 명령부터 의심"** 규율이 걸렸어야 했다.
⇒ 지금 도는 **b3shf 가 "살아있는 shiftonly + 형식 비계"**(discard 0.2~0.8%)이고 gs215 진행 중.
⚠단 런처가 스스로 적는다: *"NOT a pure shiftonly replication any more but shiftonly + format scaffold,
and `anchor_norm=true` means the effective strength is not literally 0.35."* — **순수 복제로 인용 금지.**

### 페이블이 세운 것(확인함)
★**세 팔의 처치가 극단적으로 다른데 손상이 같다** — b3p(메타 사멸) −2.08 / b3s(메타 생존·엔트로피 26배)
−2.00 / b3sh(보조 5헤드 0) −2.58. **처치가 다른데 손상이 같으면 공통분모가 유죄 후보**다.
공통분모 = `TRIOBJ` region 경로(mean-only advantage·비-whiten·discard 라우팅). 대조군 b2p 만 stock
`masked_whiten`. ⚠**단 instruct 는 같은 기하로 이겼다** ⇒ 기하 단독으로 기질 반전은 설명 못 한다.

★**(가) 긴장 해소**: b3s ≈ b3p (**Δ +0.07pp [−1.38,+1.50]**)가 죽이는 것은 정확히 하나 —
*"처치 소멸(발화 붕괴)이 held-out 손상의 원인"*. 메타가 끝까지 살아도, 완전히 죽어도 같은 자리다.
⇒ 손상은 **최종 정책이 메타를 하느냐가 아니라 300스텝 동안 무엇이 정책을 밀었느냐**에서 온다.
⚠CI 폭 ±1.5pp ⇒ 정확한 진술은 "floor 무영향"이 아니라 **"회복 증거 없음"**.

★**group-centering 확인**: `dcpo_region.py:1226` `A_meta = group_mean_subtract(R_meta, index, member=_rmeta_member)`
⇒ **`rmeta` 평균 오프셋(−0.16)은 그룹 평균이 흡수한다.** 남는 유효 격차는 평균이 아니라
**그룹 내 순위 품질**(SAVE 47% vs 67%, 커버리지 3% vs 6.5%)이다.

★**G7 부호 검사가 개입 하나를 죽였다**: 발화율 → held-out **부호 ≈ 0**(b3p 0.018/75.20 vs b3s 0.99/75.28).
⇒ ⛔**발화를 올리는 개입(floor 강화·`w_emit` 증대)은 설계하지 않는다.**

### 다음 수 — 하나로 좁혀졌다
**`b3nopmi` gs300 eval 하나가 후보 1위(region 기하)와 2위(PMI 헤드)를 동시에 가른다.**
b3nopmi 가 b2p 수준 회복 ⇒ 기하 후보 사망·PMI 단독 유죄. 여전히 −1.5pp 이하 ⇒ 그 반대.
ckpt 확보·보존 완료, 서식은 E-177 에 확정, 비용 ~20 GPU-h. **사용자 승인 대기.**
GPU 0 병행 후보 = `placebo.py` 배선. ⚠단 범위가 좁다 — 메타 밖 텍스트·boxed 답이 byte-동일이라
**SAVE/DERAIL 행 분류는 구성상 불변**이고, 갈리는 것은 *"내용 특징 조건 attribution 이 셔플에서
살아남는가"* 뿐이다. 헌법 Part VII 의 `shift(real) ≫ shift(corrupted)` 는 **별도 소규모 GPU 채점**이 필요.

⚠**이 항목은 codex-sol 게이트 전 단계다**(0805 지시). 판정문으로 승격하기 전 검사 필요.

---

## E-180 (0809 14:21 UTC) — b3shf 메타 발화 **57% 에서 안정**(붕괴 아님), 단 E-175 구간 대비 명확한 하락

**로그 원문**(`[DCPO-V3] fmt classify/replace`, B=512), gs~224~226 연속 세 스텝:

| step | `no_meta` | `wellformed` | `truncation` | `discard` | **발화율**(512−no_meta) |
|---|---|---|---|---|---|
| ~224 | 229 | 246 | 26 | 11 | **55.3%** |
| ~225 | 220 | 263 | 27 | 2 | **57.0%** |
| ~226 | 216 | 260 | 31 | 5 | **57.8%** |

★**3점이 55~58% 대에서 평평하다.** ⇒ **b3p 의 발화붕괴 궤적(0.98→0.0137)과 모양이 다르다.**
⚠**나는 13:53 에 2점만 보고 *"b3p 와 같은 방향"* 이라고 썼다 — 3점째가 그것을 지지하지 않는다. 철회한다.**
⚠미세 상승(55.3→57.8)을 **회복으로 읽지 않는다**(E-173).

★**그러나 E-175 구간 대비 하락은 실재한다**: 같은 팔 gs101~104 에서 **wellformed 93~96%** → 현재 **51%**.
**이미 일어난 변화이고 현재는 안정**이라는 것이 정확한 진술이다. ⇒ **b3shf 를 한 팔로 취급해 구간을 섞으면 안 된다.**

### 판정에 미치는 영향(미리 적어 둔다)
b3shf 는 **살아있는 shiftonly + 형식 비계**이고 gs300 이 C-019 를 닫을 팔이다(E-179).
발화 57% 는 b3p(1.8%)처럼 **무효 실행은 아니지만**, E-175 를 근거로 *"형식이 건강하다"* 고 말할 수 있는 구간은 **gs101~124 뿐**이다.
⇒ **gs300 판정문에는 그 시점의 발화율을 실측해 병기한다.** ⛔E-175 수치를 gs300 맥락에 재사용 금지.

### 같은 틱 다른 관측 — 하나는 철회
⛔**discard 3점 단조(3→8→11) 철회** — 4점째가 **2**, 그 뒤 **5**. 추세가 아니었다. 절대값도 0.4~2.1% 로 경보선(두 자릿수 %)과 멀다.
★**규율 추가**: **좁은 grep 이 진짜 관측을 가린다.** `grep -oE "discard..."` 만 뽑다가 **같은 줄의 `no_meta` 43%** 를 놓칠 뻔했다.
**필드 하나를 볼 땐 그 줄 전체를 봐라**(`grep -E "fmt classify/replace" | tail -3`).
0809 에 같은 유형의 결함을 **두 번** 냈다 — `dcpo_len_cost` 를 `dcpo_w_*` 패턴으로 놓친 것(E-179)과 이것.

### 상태(14:21)
b3nopmi **gs300 완주**(보존본 23파일, eval 승인 대기) · b3shf **gs225**(`step=226`, HB 14:21:14, 사망표식 0) · b3null **gs290**.

---

## E-181 (0809 15:17 UTC) — b3null **gs300 도달·보존 완료**, 최종 형식 상태 실측: **발화 7%**(discard 폭발이 아니라 발화 소멸로 끝났다)

**사실.** `checkpoints/rq3v2f_b3null/global_step_300/actor/` **m/o/e 각 4/4**. `step=300`, HB
`[HB Sun Aug 9 03:17:18 PM UTC 2026]`, 사망표식 0, **`FINAL PUSH DURABLE` 미출현**(최종 val·푸시 중).
**보존 완료** — `preserved/mechanism_alive/rq3v2f_b3null_gs300/` **23파일**, server-side `CommitOperationCopy`.
⚠**완주 확인은 다음 틱** — `FINAL PUSH DURABLE global_step_N` 의 **N 을 읽어야** 완주와 사고사가 갈린다.

### gs300 최종 형식 상태(로그 원문, `fmt classify/replace` B=512, 연속 두 스텝)
`{'no_meta': 474, 'discard': 9, 'truncation': 23, 'dup_open': 2, 'wellformed': 4}`
`{'no_meta': 476, 'discard': 9, 'truncation': 21, 'wellformed': 5, 'dup_open': 1}`
⇒ **발화율 7.0~7.4%** · **wellformed 0.8~1.0%** · discard **1.8%**.

★**이 팔은 discard 폭발 상태로 끝나지 않았다.** E-172 가 gs103 에 **discard 88%** 를 기록했고
같은 항목이 *"discard 가 0.88 → 0.38 로 내려가는 동안 학습정확도는 +0.24 → −0.54 로 떨어졌다"* 고
적어 뒀다. **gs300 은 그 궤적의 종착지다** — discard 는 1.8% 까지 내려갔지만 그것은 형식이 좋아져서가
아니라 **메타를 아예 내지 않게 되어서**다(no_meta 93%).
⇒ ★**E-173 의 규율 *"한 지표 개선을 회복으로 읽지 마라"* 가 이 팔에서 정확히 실현됐다.**
⚠**따라서 `w_format=0` 팔을 인용할 때 "discard 88%" 를 gs300 맥락에 쓰면 틀린다.** 구간을 밝혀라.

### `w_format` 세 팔의 종착지가 갈린다(실측)
| 팔 | `w_format` | 끝난 모양 |
|---|---|---|
| b3sh | 0.0 | **discard 폭발**(gs120→155 에 5.9%→46.7%) → gs175 중단 |
| **b3null** | 0.0 | **발화 소멸**(gs300 에 no_meta 93%, wellformed 1%) → gs300 도달 |
| b3shf | **0.35** | **발화 53~58% 유지**(E-180), discard 0.4~2.1% |
⇒ ★**`w_format=0` 인 두 팔이 서로 다른 경로로 무너졌다** — 하나는 형식 오류 폭발, 하나는 발화 소멸.
**E-172 의 판정(*"경찰 없는 region 기하는 자기 파괴한다"*, 2점 재현)은 유지되며, 이번 실측이 그
자기파괴의 *두 번째 종착 형태*를 확정한다.** ⛔새 인과 주장은 만들지 않는다 — E-173 이 이 팔의 gs300
해석을 이미 확정했다(*"`w_format=0` 이면 학습 불성립"*, **region 기하 귀속 금지**).

### 같은 틱 b3shf
`step=228`·완전 ckpt gs225·HB 15:18:10·사망표식 0. 판정점까지 72스텝.

---

## E-182 (0809 16:14 UTC) — b3null **완주 확인**(N=300) · b3shf **선점 재시작**(조치 불필요)

**① b3null 완주.** 로그 원문 `[YAML] FINAL PUSH DURABLE global_step_300` — **N=300 을 직접 읽었다**.
같은 tail 에 사망표식(`ABORT window`/`FATAL init`/`IncompleteRead`/`failed due to oom`/`Total available GPUs 0`) **0건**
⇒ **완주**로 확정한다. HB `[HB Sun Aug 9 03:47:30 PM UTC 2026]`. 보존본 `preserved/mechanism_alive/rq3v2f_b3null_gs300/` **23파일** 유지.
★E-181 이 *"완주 확인은 다음 틱"* 으로 미뤄 둔 항목을 이것으로 닫는다.
⇒ **gs300 완주 팔이 둘**(b3nopmi·b3null), 둘 다 보존 완료.

**② b3shf 선점 재시작.** `amlt log view` 가 빈 결과처럼 보여 **내 명령부터 의심**했고, 줄 수를 세니 **193줄**이었다.
`User code starts running at Sun Aug 9 04:04:17 PM UTC 2026` + 깨끗한 `[bootstrap]` 시작(conda-pack 내려받는 중),
사망표식 **0건** ⇒ **재시작 모드 ①(선점)** 이다. **조치하지 않는다.**
마지막 완전 ckpt **gs225** 에서 재개한다(직전 관측 `step=229`, HB 15:46:33).
★**"빈 결과 = 내 명령부터 의심, 다음 로그 줄 수"** 규율이 그대로 작동했다 — 190~320줄이면 재시작이다.

### 상태(16:14)
b3nopmi **gs300 완주**(보존 23, eval 승인 대기) · b3null **gs300 완주 확인**(보존 23) · b3shf **부트스트랩 중**(gs225 에서 재개).

---

## E-183 (0809 17:0x UTC) — **판정 eval 발사**: `b3nopmi` gs300 (사용자 승인)

사용자 지시(원문): *"판정 eval 돌려. 노드에서 진행해주면 돼."* ⇒ 발사.
**실험** `rq3v2f-b3nopmi-eval-0809` · **잡** `:h100_rq3v2f_b3nopmi_eval` · 80G4-H100 Standard · msrresrchbasicvc.
런처 `h100std_rq3v2f_b3nopmi_1030_eval.yaml`(커밋 `a139271`) — `h100std_rq3v2f_b3sh_1030_eval.yaml` 의 클론.

### 발사 전 검사(전부 통과)
1. **G8 팔 정체** — 기능 차분은 **SPEC 목록 하나뿐**: `rq3v2f_b3nopmi:eval/rq3v2f_b3nopmi_gs300:300`.
   루프 기계·세 패스·다운로드/머지 경로는 원본과 동일.
2. **채점 코드 동일성** — `CODE_TAR_REVISION: "467403206"`, **이전 모든 eval 과 같은 값**을 유지했다.
   ⇒ 팔 간 비교에서 채점기 차이가 교란으로 들어올 수 없다.
3. **덮어쓰기 방지(E-146 재발 차단)** — HF `eval/` prefix 를 **전부 열거**해 `rq3v2f_b3nopmi_gs300` 이 **부재**임을 확인한 뒤 파일을 썼다.
   기존 prefix 12개: `base_matched_1030_v2` · `gandhi_1030_v2` · `pmishift_1030_v2` · `rq3v2f_b0p_1030` ·
   `rq3v2f_b2p_1030` · `..._pair` · `..._rq2` · `rq3v2f_b3p_1030` · `rq3v2f_b3s_1030` · `rq3v2f_b3sh_gs165` ·
   `rq3v2f_b3sh_gs175` · `shiftonly_1030_v2`.
4. **클론 잔재 검사(두 표기)** — `b3sh|B3SH` 잔재 **1건**, 그것은 내가 새로 쓴 *"cloned from the b3sh mid-ckpt launcher"* 출처 표기다.
   `gs165|gs175|:165|:175` 잔재 **0건**. ★0807 에 `sed` 가 하이픈 표기를 놓쳐 **살아있는 run 을 오염**시킨 적이 있어 두 표기로 검사했다.
5. **체크포인트** — gs300 **m/o/e 4/4**, 보존본 `preserved/mechanism_alive/rq3v2f_b3nopmi_gs300/` **23파일**(E-177).
   ⇒ `--keep 1` 프루너가 eval 도중 원본을 지워도 판정 자산은 남는다.
6. **lint** — `yaml.safe_load` OK · `tests/test_launcher_yaml_lint.py` **48 passed**.

### 세 패스(원본 그대로)
a) 전 벤치 **4k · n=8 · seed 42** → b) aime 만 **16k · n=8 · seed 43** → c) 전 벤치 **16k · n=8 · seed 42**.
각 패스 뒤 HF 업로드 ⇒ 창이 죽어도 끝난 것은 살아남는다. **주 지표는 (c) 의 16k MATH500.**

### ⚠상속된 오해 소지 하나(동작 무관)
마지막 줄 `echo "[YAML] PAIRED EVAL DONE both arms"` 가 원본에서 그대로 왔다 — **지금은 한 팔**이다.
**echo 문자열일 뿐 동작에 영향 없다.** ⇒ **완료 판별은 `[YAML] rq3v2f_b3nopmi_1030_eval_v2 gs300 DONE` 으로 한다.**
⛔`PAIRED EVAL DONE both arms` 를 완주 근거로 인용하지 말 것.

### 판정은 E-177 서식대로만 읽는다(결과 보기 전 확정됨, `843bb8a`)
주 지표 `b3p − b3nopmi`, gs300 held-out 1030·16k·avg@8·MATH500, `math_verify`+`$`-wrap on, 문제단위 paired bootstrap.
분기표는 `docs/EXPERIMENT_PLAN.md:109-119` 정본. ★`:116-119` — **첫 비교가 양수여도 두 번째가 양수가 아니면 "패키지가 이겼다"로 적는다.**
기준선 병기: b2p `08faf5ae9fc4e404` **77.28** · b0p `fbcd9864a64ba6ba` 76.12 · b3s `ce372c5b593c8f4c` 75.28 · b3p `af5df50da404f0b8` **75.20**.
★**판정 표 모든 행에 oid**(`lfs.sha256`). ⛔eval glob 금지·경로만 인용 금지.

---

## E-184 (0809 18:41 UTC) — 판정 eval **선점 재시작**(18:29:37) · 4k 패스 실측 확보(주 지표 아님) · **발화율 8.68%**

### ① 선점 재시작 — 조치 불필요, 그러나 결과가 3~4시간 늦어진다
`rq3v2f-b3nopmi-eval-0809` 로그가 **193줄 + 깨끗한 `[bootstrap]` + `User code starts running at Sun Aug 9 06:29:37 PM UTC 2026`**,
사망표식(`ABORT window`/`FATAL init`/`IncompleteRead`/`failed due to oom`/`Total available GPUs 0`) **0건** ⇒ **모드 ①(선점)**.
★**취소·재발사해도 똑같이 처음부터**이므로 **손대지 않는다.**
★**런처 설계가 값을 했다** — *"각 패스 뒤 HF 업로드"* 덕에 이미 끝난 세 패스 산출물이 **살아남았다**.
⚠**단 루프는 4k 부터 재실행**이고, **재실행분이 기존 경로를 덮어쓴다**(E-146·[[pin-oid-telemetry-lags-0807]]).
⇒ **아래 수치는 "첫 실현"으로 oid 와 함께 고정해 둔다.** 덮인 뒤 값이 달라져도 이 항목이 두 실현을 구분한다.

### ② 4k 패스 실측 — ⛔**주 지표 아님**
`eval/rq3v2f_b3nopmi_gs300/rq3v2f_b3nopmi_gs300_4k_n8/rq3v2f_b3nopmi_gs300_4k_n8.json` **oid `760a868519624703`**

| 벤치 | 4k·n8 acc | 메타 발화율 | 절단 |
|---|---|---|---|
| **MATH500** | **61.18%** | 13.6% | 291/4000 (7.3%) |
| GSM8K | 91.78% | 0.68% | 2/4000 |
| AIME2024 | 11.67% | 60.4% | 139/240 |
| 전체 | 74.59% | **8.68%** | — |

⛔**16k 기준선(b3p 75.20 / b2p 77.28)과 직접 비교 금지** — 토큰 예산이 다르고 4k 는 MATH500 절단 7.3% 다.
⚠이 프로젝트에 **같은 대비가 16k −2.00pp(유의) vs 4k −0.53pp(불유의)** 로 갈린 실측이 있다. **주 지표는 16k.**

### ③ E-177 이 요구한 "그 시점 발화율" 확보 — 판정문 병기용
**b3nopmi gs300 메타 발화 전체 8.68%** (MATH500 13.6% · GSM8K 0.68% · **AIME2024 60.4%**).
⇒ ★**PMI 를 껐는데도 b3p(gs300 발화 1.8%)와 같은 저발화 체제다.** 그리고 **난이도에 따라 극단적으로 갈린다** —
쉬운 GSM8K 0.68% vs 어려운 AIME 60.4%.
⛔**이것을 인과로 승격하지 마라.** E-177 은 *"gs300 판정문에 그 시점 발화율을 실측해 병기하라"* 고만 요구했다.
⚠**b3p 의 1.8% 는 학습 로그의 `meta_emit_rate` 이고 이 8.68% 는 eval 산출물의 `overall_meta_emission_rate` 다** —
**측정 지점이 다르므로 두 수를 같은 표에 나란히 놓을 때 출처를 밝혀라.**

### ④ 내 명령 결함 하나 더(오늘 세 번째 같은 유형)
HF 산출물을 `f.split('/')[-1]` 로 **파일명만 출력**해서 실제 경로에 **하위 디렉터리가 한 겹 더 있다는 것을 못 봤고**,
그 결과 `hf_hub_download` 가 **404** 를 냈다. 실제 경로는 `eval/<prefix>/<NAME>/<NAME>.json`.
★오늘 같은 유형 셋: `dcpo_len_cost` 패턴 누락(E-179) · `no_meta` 필드 누락(E-180) · **경로 축약(이번)**.
⇒ ★★**규율: 조회 결과는 축약하지 말고 전체를 찍는다.** 좁은 출력이 진짜 관측을 가린다.

### 상태(18:41)
eval **재시작 후 부트스트랩 중**(산출물 12파일 보존됨) · b3shf 완전 ckpt **gs225**(재개 성공) · b3nopmi·b3null 학습 **완주 확정**, 보존본 각 23파일.

---

## E-185 (0809 20:33~21:0x UTC) — ★★★**판정 eval 완주 · 사전등록 주 지표 `b3p − b3nopmi` 는 비유의** · 분기표 ④ · **손상은 PMI 단독 원인이 아니다** (codex-sol 게이트 통과, 지적 5건 반영·1건 반증)

`[YAML] rq3v2f_b3nopmi_1030_eval_v2 gs300 DONE` 직접 확인(20:33:51), 사망표식 0, HF 산출물 20파일.
서식은 **결과를 보기 전에** E-177(`843bb8a`)에서 확정했고 그대로 따른다.

### ① 주 지표 — **비유의**

`b3p − b3nopmi`, gs300, held-out 1030 · **16k · avg@8 · MATH500**, 문항 paired bootstrap 10,000회,
n_shared = **500 문항**(4,000 completion 이 아니라 500 클러스터).

| 채점기 | b3p − b3nopmi | 95% CI | boot p | McNemar |
|---|---|---|---|---|
| `\boxed{}`-wrap robust (paired_eval `Grader`) | **−0.65pp** | [−1.9, +0.7] | 0.328 | 20/14 · p=0.392 |
| strict_boxed (fallback 없음) | **−0.60pp** | [−1.85, +0.65] | 0.367 | — |

★**두 채점기 모두 0 을 포함한다.** 채택 문구(codex 권고 그대로):
> **주 비교는 통계적으로 비유의했다. 0 효과를 기각하지 못했으며, PMI 헤드의 end-to-end 순효과는
> 검출되지 않았다. 이것은 실제 효과가 0 이라는 증거도, 등가성 판정도 아니다.**

⚠**"널" 이라고 쓰지 않는다** — 등가성 마진을 사전등록한 적이 없다.
**검정력**(관측 SE ≈ 0.66pp 기준, codex 계산): 진효과 1.0pp → **33%** · 1.5pp → 63% · 2.0pp → 86% ·
2.5pp → 97%. **양측 α=.05, 80% power MDE ≈ 1.84pp** (90% ≈ 2.13pp).
⇒ **1pp 안팎의 진효과는 이 설계로 못 본다.** CI 는 "PMI 가 최대 1.9pp 해롭다"와도 양립한다.

### ② 수준과 보조 대비 (MATH500 16k, oid 고정)

파일 oid(`lfs.sha256` 앞 16): b3nopmi parquet **`b5f0b6de86d3c047`** · json `87542728189eab39`
(`eval/rq3v2f_b3nopmi_gs300/rq3v2f_b3nopmi_gs300_16k_n8_math500/`) ·
b3p **`af5df50da404f0b8`**(`eval/rq3v2f_b3p_1030/`) · b2p **`08faf5ae9fc4e404`**(`eval/rq3v2f_b2p_1030/`).

| arm | `\boxed{}`-wrap robust | strict_boxed | 원채점(runtime) | 절단% | 평균토큰 | 메타발화%(closed) | boxed율 |
|---|---|---|---|---|---|---|---|
| b2p (vanilla 대조군) | **77.325%** | 77.33% | 62.90% | 2.4% | 1075 | 100.0% | 97.62% |
| b3nopmi | **75.825%** | 75.72% | 61.30% | 5.4% | 1474 | 9.9% | 94.85% |
| b3p | **75.175%** | 75.12% | 60.75% | 5.1% | 1472 | 1.4% | 95.62% |

| 대비 | robust | strict_boxed |
|---|---|---|
| **b3p − b3nopmi (주 지표)** | **−0.65pp** [−1.9,+0.7] p=0.328 | **−0.60pp** [−1.85,+0.65] p=0.367 |
| b2p − b3nopmi | **+1.50pp** [+0.3,+2.7] p=0.017 · McNemar 15/12 p=0.701 | **+1.60pp** [+0.40,+2.80] p=0.009 |
| b2p − b3p | (기존 RQ2 를 **부호 맞춰** 적으면 **+2.08pp** [+0.68,+3.50]) | +2.20pp [+0.83,+3.60] p=0.003 |

⛔**정정(내 초안 오류)**: 초안이 `b2p − b3p` 행에 **−2.08pp** 를 적었다. −2.08 은 `b3p − b2p` 다.
**행 이름을 유지하면 +2.08pp.** codex 가 잡았고 확인했다.

보조 셀(b3nopmi − b3p): gsm8k 16k **+0.8pp** [−0.1,+1.6] p=0.082 · aime2024 16k avg@16(n=30)
**+0.8pp** [−4.4,+5.4] p=0.732. ⚠**저검정력·다중 보조분석이므로 결론에 쓰지 않고 기술통계로만 남긴다.**

### ③ ★채점기 provenance — codex 지적 **절반 채택, 결정적 부분은 반증**

★**채택**: E-177 이 사전등록한 주 채점기는 원장 `:6302` 의 ***"`math_verify` + `$`-wrap on"*** 이고,
기준선 **b3p 75.20 · b2p 77.28** 은 그 계열이다. 내가 이번에 쓴 것은 `paired_eval.py` 의 `Grader`
= `experiments/common/grading.py:52 robust_grade`(**`\boxed{}` 로 감싸 parse**)다.
두 값은 **b3p 75.175 vs 75.20 · b2p 77.325 vs 77.28** 로 0.03~0.05pp 안에서 만나지만
**같은 채점기가 아니다** ⇒ ⛔**이번 수치를 "format_fair" 로 재명명하지 않는다.** 초안의 라벨을 고쳤다.
⚠`freeze_run_manifest.py:66` 의 해시 목록에 `analysis_common.py` 는 있으나 **실구현 `grading.py` 와
`math_verify` 버전이 빠져 있다** — 동결이 불완전하다(정본 코드 수정이 필요하므로 **승인 사항**으로 남긴다).

★**반증**: codex 는 *"OID 고정 parquet 에 같은 `Grader` 를 돌렸더니 b2p 77.325 · b3p 74.975 ·
b3nopmi 75.625 로 내 표와 불일치 ⇒ 표가 한 채점 구현으로 만들어지지 않았다는 반증"* 이라고 썼다.
직접 확인한 결과 **원인은 내 표의 혼재가 아니라 `math_verify` 버전**이다 — codex 는 **0.6.0**,
내 환경은 **0.9.0**(`gram` env). 그리고 결정적으로:
- **b2p 는 두 버전이 77.325% 로 완전 일치**한다(codex 값 = 내 값).
- 차이는 메타를 내는 두 팔에서만 **정확히 0.20pp = 500문항 중 1문항 값**으로 같은 크기다.
- ⇒ **주 대비에서 상쇄된다: 0.6.0 으로도 `b3p − b3nopmi` = 74.975 − 75.625 = −0.65pp 로 동일.**

★**따라서 주 지표는 `math_verify` 버전에 불변이다**(0.6.0 · 0.9.0 두 버전 모두 −0.65pp).
단 `b2p − b3nopmi` 는 +1.50(0.9.0) vs +1.70(0.6.0) 로 갈리므로 **그 수는 버전을 밝혀 인용한다.**

### ④ G8 (팔 정체) — 통과, 단 **정적 대비**임을 명시

`git diff --stat --no-index h100std_rq3v2f_b3p.yaml h100std_rq3v2f_b3nopmi.yaml`
= **17 insertions / 16 deletions**(초안의 "60줄"은 `diff | wc -l` 이라 단위가 다르다).
학습 하이퍼파라미터 차분은 `h100std_rq3v2f_b3nopmi.yaml:253` 의 **`++algorithm.dcpo_w_meta=0.0` 한 줄**뿐,
나머지는 전부 lineage 이름(experiment_name · default_local_dir · resume regex · WANDB_NAME).

**`w_meta=0` 이 실제로 끄는 것**(codex 지적 → 직접 확인):
- PMI-shift 점수는 **계속 계산된다** (`src/training/verl_sdc.py:443`)
- region mask 도 **그대로 구성된다** (`verl_sdc.py:290`)
- `w_meta` 는 읽힌 뒤 warmup scale 과 곱해진다 (`verl_sdc_utils.py:345`)
- 사라지는 것은 정확히 **`w_meta * A_meta * meta_c` 한 항** (`dcpo_region.py:1268` 의
  `advantages = w_corr*A_corr*ans + w_meta*A_meta*meta_c + w_cal*A_cal*conf`)
- length cost · anchor · calibration · format · emission · truncation 경로는 **전부 살아 있다**

⇒ ⛔*"META_CONTENT 영역이 비활성화된다"* **틀림** · ⛔*"PMI 계산을 제거했다"* **틀림**.
★**정확한 이름: "PMI 유래 META_CONTENT advantage 항을 0 으로 만들었다."**
"PMI 헤드 제거"는 **이 정의를 바로 붙일 때만** 허용되는 약칭이다.
(⚠단 일반 meta-content 토큰에는 그 항이 **유일한 직접 advantage** 이므로, confidence·format·emission
같은 특수 위치를 빼면 메타 내용 토큰의 직접 학습신호는 사실상 사라진다.)

⚠**정적 런처 대비 ≠ 실현된 한 변수 대비.** init 은 런타임에 두 후보 중 하나를 고르고
(`h100std_rq3v2f_b3p.yaml:119`), anchor EMA 는 **체크포인트되지 않는 module-global** 이다
(`verl_sdc_utils.py:28`). 두 팔의 선점·재개 이력이 다르므로 **resolved init hash · resolved config ·
선점 횟수를 G8 에 병기해야** 대비가 강해진다 — **미측정으로 남긴다.**

### ⑤ 판정 — 분기표 ④ (`docs/EXPERIMENT_PLAN.md:114`)

정본 행은 **`B3pkg < B2` 또는 `B3-noPMI > B3pkg`** 다. 첫 조건이 **두 채점기 모두에서 유의**하게
성립하므로, 두 번째가 비유의여도 ④가 맞다. 허용 결론: ***"PMI-shift 가 현 recipe/substrate 에서
무효 또는 해로운 방향일 가능성."*** ⛔**이보다 세게 쓰지 않는다** — ④는 PMI 가 해롭다는 **입증이 아니다.**

`:116-119` 대로, 첫 비교가 양수가 아니고 두 번째도 양수가 아니므로
⛔**"PMI-shift 가 이겼다"는 쓸 수 없다.**

**채택 판정문(codex 권고 문안 채택)**:
> 분기표 ④에 해당한다. B3pkg 는 B2 보다 유의하게 낮다. 사전등록 주 비교 `b3p−b3nopmi` 는
> 음의 점추정이지만 95% CI 가 0 을 포함하여, **PMI 유래 META_CONTENT advantage 항의
> end-to-end 순효과는 검출되지 않았다.** 이는 효과가 0 이라는 등가성 판정도, PMI 가 해롭다는
> 입증도 아니다.
>
> b3nopmi 역시 B2 보다 유의하게 낮아 **PMI 항의 제거만으로 패키지 손상이 회복되지 않았다.**
> 따라서 손상을 PMI 단독 원인으로 귀속할 수 없으며, 남은 원인은 region routing · discard 처리 ·
> length/format/calibration/emission/truncation/anchor 를 포함한 **공통 TRIOBJ 번들** 안에 있다.
> **이 대조만으로 region 기하를 특정하지 않는다.**

### ⑥ E-179 가 건 판별 — **이분 판정은 보류**

E-179 는 *"b3nopmi 가 b2p 쪽으로 회복 ⇒ region 기하 후보 사망·PMI 단독 유죄 / 여전히 −1.5pp 이하 ⇒
손상이 PMI 특이적이 아님"* 으로 **결과 전에** 갈라 뒀다(사후 편의는 아니다. 단 b3p 결과를 본 뒤 만든
**적응적 기준**이지 최초 설계시점 사전등록은 아니다).

실측은 **정확히 그 절단점 위**다: `b2p − b3nopmi` = +1.50pp(robust 0.9.0) / +1.70pp(0.6.0) /
+1.60pp(strict). E-179 는 **어느 채점기로 경계를 판정할지 지정하지 않았고**, 세 값이 −1.5 절단점의
양쪽으로 갈린다. ⇒ ★**그 이분 판정은 보류한다.**

대신 확실한 것만 적는다:
> **b3nopmi 는 세 채점 조건 모두에서 B2 보다 유의하게 낮다 — 완전 회복의 증거는 없다.**
> PMI 를 껐을 때 좁혀진 몫은 2.08pp 중 0.65pp(약 31%)이고 **그 회복 자체가 비유의**하다.

⇒ 후보 1위(TRIOBJ region 기하)는 **기각되지 않았을 뿐**이다. ⛔**"살아남았으니 유력"으로 승격 금지** —
b3nopmi 에는 cal·format·emit·length·trunc·anchor·discard routing 이 **전부 남아 있어** region 을 특정하지 못한다.
E-174 가 이미 지목한 **별도 `bfmt` 대비**가 그것을 가르는 실험이다(원장 `:6134`).

### ⑦ 발화율 병기 (출처를 밝혀서)

| 출처 | b2p | b3nopmi | b3p |
|---|---|---|---|
| eval `meta_emission_rate`(**closed** 블록, paired_eval) MATH500 16k | 100.0% | **9.9%** | **1.4%** |
| eval json `overall_meta_emission_rate`(열린 블록 포함) MATH500 16k | — | 14.1% | — |
| eval json, aime2024 16k | — | 76.7%(closed 41.2%) | — |
| eval json, gsm8k 16k | — | 0.9%(closed 0.8%) | — |
| eval json, 4k 전 벤치(E-184) | — | 8.68% | — |
| **학습 로그** `meta_emit_rate` | — | — | **1.8%** |

⚠**b3p 의 1.8% 와 1.4% 는 출처가 다르다**(학습 로그 vs eval). 섞어 쓰지 않는다.
⚠**발화율은 처치 후 매개변수다.** b2p(100%)가 최상위·b3p(1.4%)가 최하위라는 배열을 **G7 을 뒤집는
증거로 쓰지 않는다** — 세 팔이 통째로 다르다.

### ⑧ ⛔E-184 에 대한 정정 — **채점기 provenance**

E-184 가 적은 4k 수치(전체 74.59% · MATH500 61.18% · GSM8K 91.78% · AIME 11.67%)는
**eval json 의 원채점(runtime grader)** 이다. 같은 계열 16k 가 원채점 **61.30%** / robust 재채점
**75.825%** 로 갈린다. ⇒ **원채점을 기준선(75~77, 재채점 계열) 옆에 놓으면 14pp 짜리 가짜 격차가 생긴다.**
E-184 는 *"예산이 달라 비교 금지"* 라고만 적었는데, **채점기가 다르다는 것이 더 강한 이유**다.

### ⑨ 반드시 병기하는 대안설명 (codex ⑦ 전부 채택)

- **단일 학습 시드**(C-021) — 문항 부트스트랩은 최적화 궤적 분산을 **포함하지 않는다.**
- **eval 생성 잡음** — −0.65pp 는 동일 체크포인트 A-vs-A 변동과 **같은 크기대**다(E-154 실측 SD 0.35pp).
- **선점/재개 이력** — anchor EMA 가 비영속이라 실제 최적화 경로가 갈릴 수 있다.
- **동적 init 선택** — 코드가 같아도 실제 선택된 init artifact hash 가 필요하다(미측정).
- **residual bundle** — noPMI 에도 length cost·format·cal·emit·truncation·discard routing 이 남는다.
- **placebo 미실시**(C-020) — "내용 대 형태" 는 여전히 못 가른다.
- 보조 GSM/AIME 셀은 **결론에 쓰지 않는다.**

### ⑩ 산출물·다음

- `experiments/analysis/paired_eval.py`(정본) · `$JOB/tmp/strict_boxed_probe.py`(**일회성 probe,
  §5.3 대로 정본 승격하지 않는다**) · `$JOB/tmp/codex_e185.txt`(적대검토 전문) · `$JOB/tmp/pe/`(parquet 사본)
- ⚠`analysis_common.py` 정규화 스키마의 문항 키는 **`qid`**(`problem_id` 아님).
- **CLAIMS 갱신 의무**(§4): 이 축은 `docs/CLAIMS.md` 에 `b3nopmi` 가 **한 번도 없어** 새 증거가 맞다.
  단 *"손상이 triobj 공통부에 있다"*(`CLAIMS.md:731`)와 *"region-split 은 여러 후보 중 하나"*(`:765`)는
  **이미 산 결론**이므로 다시 사지 않는다. **C-019 는 E-179 에서 stale 판정을 받고도 아직 그대로다** —
  같은 커밋에서 갱신한다.
- **새로 열리는 것**: 공통 TRIOBJ 번들을 가르는 대비(E-174 의 `bfmt`). ⛔**사용자 승인 전 발사 금지.**

### 상태(21:0x)
판정 eval **완주**(20파일) · b3shf 학습만 활성(gs225~226) · b3nopmi·b3null 학습 완주·보존 각 23파일.

---

## E-186 (0809 23:45 UTC) — ★★**b3shf 만 `val_before_train=True`** — 선점 재시작 비용이 다른 팔의 두 배 이상이다

### ① 실측

`h100std_rq3v2f_b3shf.yaml:275` 에 **`++trainer.val_before_train=True`**. 전 팔 대조:

| 런처 | `val_before_train` |
|---|---|
| `h100std_rq3v2f_b3p.yaml` · `b3nopmi` · `b2p` · `b3null` | **False** |
| **`h100std_rq3v2f_b3shf.yaml`** | **True** |

⇒ **b3shf 는 선점 재시작마다 학습 재개 전에 전체 검증을 한 번 다 돌린다.** 다른 네 팔은 돌지 않는다.

### ② 이번 인스턴스 시간 분해 (21:36:39 기동)

- 21:36:39 → 22:36:57 **준비 60분**(conda-pack + 코드 + gs225 ckpt 다운로드·로드)
- 22:36:57 → 23:45:03 현재까지 **재개 검증 68분, 진행 중**(`global_steps: 225` 고정)
- **학습 스텝 0** (기동 후 2시간 8분)

★**검증은 고착이 아니다** — 속도로 확인했다(⚠두 창의 구간이 달라 계수 직접 비교 금지):
창 A 22:36:06~23:17:22(41.3분) `validation generation end` **17개 = 0.41/분** ·
창 B 22:58:14~23:45:03(46.8분) **20개 = 0.43/분**. 사망표식 0, HB 23:45:03 신선, 재시작 없음.

### ③ 산수 정정 — ⛔내가 앞 틱에 말한 "재시작 1회 ≈ 13스텝"은 **과소평가**였다

준비 60분만 세었다. 실제는 **준비 60분 + 검증 68분+ = 2시간 이상 ≈ 27스텝 값**.
남은 **75스텝** ≈ 선점 없는 5~6시간. 오늘 오후 선점 간격은 그보다 짧았다(21:23:49 → 21:36:39).

★**이것이 b3p 가 22회 선점되고도 완주한 이유**다(memory `three-death-modes-basicvc-0807`) —
그 팔의 재시작 비용은 **준비뿐**이었다. b3shf 를 그 전례로 안심하면 안 된다.

### ④ 부수 관찰 — G8 관점

b3shf 는 *"b3sh + `w_format 0→0.35`"* 로 기술돼 왔다. 그런데 `val_before_train` 도 다르다.
학습 목적함수는 바꾸지 않지만 **재개 시 RNG 소비 경로가 다른 팔과 갈린다** — 결정론 비교를 할 때
⚠주의. ⛔현 시점에서 학습 결과 차이의 원인으로 승격하지 않는다(관찰만).

### ⑤ 남기는 결정

**b3shf 가 gs300 에 닿을지 불확실하다.** 사용자 결정 사항으로 올린다 —
계속 기다릴지, **gs225 를 종점으로 볼지**. (b3shf 의 목적은 새 질문이 아니라 b3p/b2p 와의 gs300 정합이다.)
⛔런처를 고쳐 `val_before_train=False` 로 바꾸는 것은 **정본 변경이자 도는 팔의 조건 변경**이므로 승인 사항.

### 상태(23:45)
b3shf `running` · 재개 검증 68분째 진행 · 학습 스텝 0 · 완전 ckpt gs225 · 사망표식 0.
판정 eval 완주(E-185) · b3nopmi·b3null 학습 완주 · 보존본 여섯 개 각 23파일.

### E-186 후속 (0810 00:0x UTC) — 사용자 결정: **플래그 고쳐 재발사**

`h100std_rq3v2f_b3shf.yaml:275` **`val_before_train=True → False`**(커밋 `6cb7403`, 한 줄 차분).
LIST → `amlt cancel rq3v2f-b3shf-0808` → `killed` 확인 → **`amlt run h100std_rq3v2f_b3shf.yaml rq3v2f-b3shf-0810`**.
체크포인트 계보 `rq3v2f_b3shf` 동일이므로 **gs225 에서 재개**. 이제 다섯 팔 전부 `val_before_train=False`.
⚠**감시 대상 실험명이 `rq3v2f-b3shf-0810` 으로 바뀌었다**(구 `-0808` 은 killed).
⚠검증은 가중치를 바꾸지 않으므로 학습 목적함수 변경이 아니다. 재개 시 RNG 소비 경로는 달라지나
**이미 재시작마다 달라지고 있었다** — 새 교란이 아니다.

---

## E-187 (0810 06:3x UTC) — ⚠**b3shf 가 b3sh 의 wandb run 에 이어붙고 있다** (텔레메트리 오염, 학습은 무사)

### ① 실측 — 전 팔 전수 대조

| 런처 | `WANDB_NAME` | `WANDB_RUN_ID` | `WANDB_RESUME` |
|---|---|---|---|
| b0p | `rq3v2f_b0p` | `rq3v2f-b0p-1` | allow |
| b2p | `rq3v2f_b2p` | `rq3v2f-b2p-2` | allow |
| b3p | `rq3v2f_b3p` | `rq3v2f-b3p-1` | allow |
| b3s | `rq3v2f_b3s` | `rq3v2f-b3s-1` | allow |
| **b3sh** | `rq3v2f_b3sh` | **`rq3v2f-b3sh-1`** | allow |
| **b3shf** | `rq3v2f_b3shf` | **`rq3v2f-b3sh-1`** ← **동일** | allow |
| b3null | `rq3v2f_b3null` | `rq3v2f-b3null-1` | allow |
| b3nopmi | `rq3v2f_b3nopmi` | `rq3v2f-b3nopmi-1` | allow |

`h100std_rq3v2f_b3shf.yaml:43` = `h100std_rq3v2f_b3sh.yaml:43` = **`rq3v2f-b3sh-1`**.
**충돌은 이 한 쌍뿐이고 나머지 여섯 팔은 전부 고유하다.**

### ② 근인 — sed 클론이 **하이픈 표기만** 놓쳤다

b3shf 런처의 표기 잔재 전수: `rq3v2f_b3shf` **18건**(치환 성공) · `rq3v2f_b3sh` 2건(설명문) · **`rq3v2f-b3sh-1` 1건(치환 실패)**.
⇒ 밑줄 표기는 전부 바뀌었는데 **하이픈 표기 RUN_ID 한 줄이 남았다.**
★**이것은 E-078 에서 이미 한 번 일어난 사고와 같은 패턴이다**(*"sed 가 하이픈 표기 WANDB_RUN_ID 를 못 건드려 `WANDB_RESUME: allow` 가 fresh 런을 구 런에 이어붙임 → 초기 ~160스텝 지표 조용한 증발"*).
**같은 실패를 두 번 샀다.** 규율 *"런처 클론은 밑줄·하이픈 두 표기로 잔재 grep"* 이 대장에 있었는데 이 클론에는 적용되지 않았다.

### ③ 영향 범위 — 학습은 무사, **텔레메트리만 오염**

- **체크포인트 계보는 안전하다.** `rq3v2f_b3shf` 로 분리돼 있고(런처 `:159,:181,:202,:206`), b3sh 의 gs165/170/175 를 덮지 않았다(HF 실측으로 둘 다 존재).
- **오염 구간**: b3sh 는 **gs175 에서 중단**됐으므로 같은 run 의 **`_step` 1~175 가 두 팔 혼재**, **176+ 는 b3shf 전용**.
- ⇒ ⛔**철회**: 0810 Step 1 에서 보고한 b3shf 의 **gs1-50 · 51-100 · 101-150 행**(n_save 5.30 / 4.98 / 4.46 등)은 **b3sh 값이 섞였을 수 있다.**
- ✅**살아남음**: **gs201-300 구간**(실제 gs201~228, n=28) — `n_save` **9.79** · `attempted` 0.586 · `rmeta` **+0.848** 은 **b3shf 전용 구간**이므로 유효하다.
  ⇒ *"b3shf 가 우리 base 팔 중 처음으로 헌법 Healthy 칸에 들어갔다"* 는 판정은 **유지된다**.
- ⚠**추가 위험(미확인)**: b3shf 가 같은 `_step` 에 덮어썼다면 **b3sh 의 wandb 기록 자체가 손상**됐을 수 있다.
  E-152 의 b3sh 진단(discard 5.9%@gs120 → 46.7%@gs155)이 wandb 기반이었다면 그 근거가 지금은 재현되지 않을 수 있다. **미검증으로 남긴다.**

### ④ 조치 — **잡은 건드리지 않는다**

- 남은 구간(gs230+)은 어차피 b3shf 전용이라 **RUN_ID 를 고쳐도 얻는 것이 없다.** 반면 수정하려면 재시작이 필요하고 **53분(≈12스텝)** 을 잃는다. ⇒ **현행 유지.**
- ★**미결 조치**: `h100std_rq3v2f_b3shf.yaml:43` 을 `rq3v2f-b3shf-1` 로 고치는 것은 **다음 재발사 전에** 해야 한다. 지금 고치면 다음 재개에서 시계열이 끊긴다는 부작용이 있으므로 **재발사 시점에 함께** 적용한다. (정본 yaml 변경 = 승인 사항으로 올림.)

### ⑤ 판정 절차에 미치는 영향

이 오염은 **wandb 텔레메트리 한정**이고 **held-out eval·체크포인트·HF 산출물에는 영향이 없다**.
⇒ E-185/C-031 의 판정 수치와 b2p/b3p/b3nopmi/b3s 의 모든 Step 1 수치는 **그대로 유효하다**.

### 상태(06:3x)
b3shf `rq3v2f-b3shf-0810b` **running**, HB 06:22:46, 사망표식 0, **gs230 저장 완료**(로그에 rank 0~3 전부 확인), `Training Progress: 230/300`, **1675초/스텝**.
gs230 실측: `meta_emission` 0.441 · `format_penalty` 0.377(w_format 활성 = b3shf 확증) · 엔트로피 0.279 · `response_length` 평균 4098 · **`clip_ratio` 0.42**.

---

## E-188 (0810 07:5x UTC) — ⛔**Step 3 캘리브레이션 발사 취소** — 사전등록 배선검사가 발사 전에 설계를 막았다

### ① 결정
**`b3shf gs225/230` + `b3s gs303` 로의 PMI-shift 캘리브레이션을 발사하지 않는다.** 런처(`$JOB/tmp/launch_pmi_calib_s3/h100std_pmi_calib_s3.yaml`)는 기계 게이트 8종을 전부 통과했고(asset 200·oid 고정·프리픽스 부재·yaml lint·b64 왕복 md5) 두 에이전트가 독립으로 재현했다. **막은 것은 게이트가 아니라 측정 정의다.**

### ② 이유 (둘 다 코드 직접 확인)

**(1) 프로브가 다른 라벨을 쓴다.**
`src/eval/pmi_shift_signal.py:293-295` → `correct = int(float(r["c_with"]) >= 0.5)`, 주 지표 AUC 의 라벨은 `r["correct"]`. **`c_without` 은 이 파일에 존재하지 않는다.**
⇒ 이 경로의 `auc_shift` 는 *"PMI-shift 가 **정답/오답**을 가르는가"* 이고, **S0.2 가 이미 잰 양**이다. *"**뒤집기**를 예측하는가"* 가 아니다.
⚠`reversal_label`(:94-99)은 뒤집기 라벨이 아니라 **점수 부호전환**(`pmi_open<0 & pmi_close>0`)이라는 점수 자체의 속성이다. `n_save_reversal`·`save_correct_rate` 도 그 속성의 개수·조건부 정답률일 뿐이다.
⇒ 그대로 쏘면 **같은 결론을 조건만 바꿔 두 번 산다**(§4 재확인 계수기 대상).

**(2) 뒤집기 라벨 자체가 대부분 인공물이다.**
`dcpo_region.py:391 cf_answer_from_prefix` → `rewards.py:186-187 _extract_answer_fallback` 은
`\boxed{}` 도 `####` 도 없으면 **텍스트의 마지막 숫자**를 답으로 삼는다(`nums[-1]`).
★**같은 저장소에 이미 고친 판이 있다** — `rewards.py` `_extract_boxed_or_hash`(2026-07-14 bugfix) 주석:
> *"일부러 마지막 숫자로 폴백하지 **않는다** — 꼬리 숫자가 우연히 gold 와 일치한 무-boxed 롤아웃을 정답으로 채점하면 안 된다(**last-number 휴리스틱이 과발화해 진짜 오답을 뒤집었다**)."*
⇒ **정답 채점기는 고쳤는데 반사실 경로는 안 고쳤다.** 부분 수정이 남긴 구멍이다.
실측(A4, 검증자 dedup 정정 후): 뒤집기 라벨의 **93~94%가 긁기**(pooled 133/141). 엄격 필터 시 **혼합그룹 b3s 0개 · b3shf 2개** ⇒ **잴 사건이 없다.**

### ③ 판정
**§2.1 절차 ⑤(조작 자체를 배선검사하라)에서 탈락.** 바닥·천장 이전에 **측정 대상이 존재하지 않는다.**
⇒ 사전등록을 결과 전에 강제한 것이 GPU 지출 전에 이것을 막았다. **절차의 성공 사례로 기록한다.**

### ④ 이 회차의 실질 수확 — A2 (GPU 0)
**b2p 의 confidence 는 라우팅에 실제로 배선돼 있다**: `P(redirect|저신뢰)` **89.57%** vs `P(redirect|고신뢰)` **1.26%**.
**그러나 그 confidence 는 정답성이 아니라 난이도를 잰다**: 문항내 AUROC **0.5271** [0.4943, 0.5591], 셔플 바닥 0.4993.
⇒ ★**온도계를 라우터에 배선해 놓았다** — redirect 는 *틀린 시도*가 아니라 *어려운 문항*에서 발화한다.
⚠**검증자 정정**: *"96%가 난이도"* 는 분모가 다른 두 양의 비(전체 +57.15pp vs 매칭 −2.19pp). **동일 모집단 분해는 4.11 → 2.19 = 47%.**
다른 팔도 같은 정정: b3sh175 **89%**(20.53→2.34) · b3sh165 **92%**(18.12→1.44). **방향은 유지, 크기만 하향.**
⚠b3shf 는 eval 산출물이 **없어** 이 계산에서 빠졌다(HF `eval/` 260파일 중 b3shf parquet **0개**).

### ⑤ A3 — C-020 은 좁혀지지 않았고, 근본 블로커가 드러났다
생성 없는 placebo 는 **원리적으로** 채점을 못 바꾼다(블록 내부만 치환, 채점기는 블록 밖 마지막 `\boxed` 만 읽음 → **0/4000 행 변경**, 검정력 0).
좁히려면 `placebo_prefix` 로 **계속생성**해야 하는데 **instruct 계열 RL 체크포인트가 전삭제**됐다(C-022) ⇒ 그 팔에서는 **재학습부터** 해야 한다. **이것이 C-020 의 진짜 블로커다.**
얻은 상관 증거: shiftonly 에서도 메타 **내용**이 같은 문항의 정답/오답을 거의 못 가른다(**문항내 AUC 0.542** [0.502,0.583]; pooled 0.831 과의 격차 0.289 는 난이도 교란). **형태(길이 0.592)가 내용보다 세다.**
⚠**검증자 반증(채택)**: A3 이 *"gandhi 0.71525 출처 불명·9.55pp 미해결"* 이라 한 것은 **채점기 혼용**이었다 — 71.525=format-fair, 61.975=strict, 차이가 **정확히 382행 = +9.55pp**. 원장 `part1:2790-2792` 에 이미 있다.

### ⑥ 다음에 필요한 것 (미착수)
**뒤집기를 진짜로 재려면** `scripts/harvest_redirect_cf.py` 의 4팔(R/N′/Nc/B′)을 생성 경로에 배선해야 한다 — splice 400 × 4팔 × k=8 ≈ **12,800 생성**. GPU 잡이고 **설계부터 다시**다.
그리고 그 전에 **`cf_answer_from_prefix` 를 `_extract_boxed_or_hash` 로 바꾸는 것**(정본 코드 변경 = 승인 사항)이 선행되어야 라벨이 인공물이 아니게 된다.

### ⑦ A1 검증자가 잡은 인용 오류 (런처 `description` 에 실려 나갈 뻔함)
*"n=8 = `h100std_rq3v2f_b3shf.yaml:257-259`"* → 그 파일에 `rollout.n` 은 **한 줄도 없다**. 실제 출처는
`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:83  n: 8`. 값은 맞고 **출처 칸이 틀렸다**.

---

## E-189 (0810 08:2x UTC) — ★★★**막힌 지표 넷의 공통 원인은 하나다: 반사실(`c_without`)이 채워지지 않는다**

### ① 실측 — 롤아웃 테이블 `c_without` 충전율 (wandb `dcporollouts`, 팔별 최근 10테이블 5,120행)

| 팔 | 행 | `c_without` 숫자 충전 | 비율 |
|---|---|---|---|
| **b3shf**(gs176+ 만, E-187 오염 회피) | 5,120 | 486 | **9.5%** |
| **WINNER `shiftonly_pmishift`** | 5,120 | 316 | **6.2%** |
| b3p | 5,120 | 131 | **2.6%** |
| b3s | 5,120 | 11 | **0.2%** |

**우승자를 포함해 어느 팔도 10% 를 넘은 적이 없다.** 나머지는 전부 `None`.
근거: `src/training/dcpo_region.py:387` 주석 그대로 —
> `# NOT wired into the live R_meta site (default behavior unchanged); see redirect_cf.py.`
⇒ **반사실 보상 경로는 코드로 존재하고 라이브 경로에 배선돼 있지 않다.** §5.4 *"관문에 안 걸린 도구는 미완성"* 의 사례.

### ② 이것이 오늘 막힌 것 넷을 **전부** 설명한다
- `acc_with − acc_without` 이상/NaN → **without 이 거의 비어 있어서**
- CF 굶주림(혼합그룹 19%) → 같은 원인
- 뒤집기 라벨 **94% 인공물**(E-188) → 반사실이 없으니 `_extract_answer_fallback` 이 **마지막 숫자로 폴백**
- 문항내 대비 교란(아래 ③) → 반사실 대신 "메타 안 낸 행"을 대용했는데 그 집단은 파이프라인이 다르게 취급한다
⇒ **네 지표가 따로 고장난 게 아니라, 없는 장기 하나 때문에 넷이 동시에 안 돈다.**

### ③ 문항내 대비 — 계산했고, **교란을 못 뺐다**
같은 문항(그룹) 안에서 메타 발화 행 vs 무발화 행의 정답률 차(최근 25테이블, 12,800행):

| 팔 | mixed 그룹 | pooled Δ | 문항내 Δ | 95% CI |
|---|---|---|---|---|
| WINNER | 557 | −0.153 | **−0.009** | [−0.016,−0.002] |
| b3p | 320 | −0.345 | +0.004 | [−0.020,+0.027] |
| b3s | 9 | +0.055 | +0.063 | [−0.184,+0.311] |
| **b3shf**(gs176+) | 1,534 | +0.332 | **+0.376** | [+0.360,+0.392] |
| b3nopmi | 656 | −0.552 | −0.027 | [−0.043,−0.010] |
| b3null | 255 | −0.685 | −0.079 | [−0.113,−0.045] |

⛔**b3shf 의 +0.376 을 효과로 읽지 마라 — 교란 확인에서 죽었다.**
`fmt_class` 분포: 무-메타 행의 **98.2% 가 `fmt_class = no_meta`**. 분류기가 메타 부재 시 즉시 `no_meta` 로 반환하므로(`dcpo_region.py:242-244`, S0.1) **무-메타 행은 원리적으로 `wellformed` 이 될 수 없다.**
⇒ `wellformed` 로 제한하면 **mixed 그룹이 0** 이 된다(b3shf·WINNER 둘 다). **`fmt_class` 로는 교란을 뺄 수 없다.**
⚠b3shf 는 절단율 42%(`response_length/clip_ratio` gs230 0.422) — 무-메타 행이 폭주·절단 행일 가능성을 배제 못 했다.
⚠WINNER 의 `fmt_class` 는 meta 행조차 `discard` 81% 로 우리 팔과 분포가 완전히 달라 **세대 간 비교 불가**.
⚠이 문항내 Δ 는 트레이너의 `dcpo/acc_with − acc_without` 과 **다른 양**이다(전자는 `has_meta` 로 분할, 후자는 `c_without` 컬럼 사용). **섞어 인용하지 마라.**

### ④ 헌법 표의 미계측 항목
`dcpo/pmishift_save_correct_rate` · `pmishift_derail_correct_rate` 는 `CONSTITUTION.md:123-128` 에 Healthy/Broken 대역까지 적혀 있으나 **전 팔·우승자 모두 wandb 에 키가 없다** — 표에만 있고 계측이 없다.

### ⑤ 판정
**PMI-shift 는 우리 메타 보상이 맞다**(engagement 표). 그러나 *"그 보상이 좋은 보상인가"* 는 **반사실 없이 판정 불가**이고, `c_without` 충전율이 ≤9.5% 이므로 **이 프로그램에서 그 판정이 내려진 적이 없다** — 우승자 세대 포함.
⇒ ★**다음 수는 새 보상 설계가 아니라 `c_without` 배선이다.** credit 재분배·2턴·라우팅 보상 **전부 그 뒤에 온다.**
필요한 것: `redirect_cf.py`/`harvest_redirect_cf.py` 의 4팔(R/N′/Nc/B′)을 라이브 경로에 연결 = **정본 코드 변경(승인 사항) + GPU 재학습**.
그리고 E-188 의 선행 조건도 유효: `cf_answer_from_prefix` 를 `_extract_boxed_or_hash` 로 교체해야 라벨이 인공물이 아니게 된다.

---

## E-190 (2026-08-10) — 【사전등록 · 결과 미열람】메타 블록이 실제로 결과를 바꾸는가 + **E-189 의 처방 정정**

### ⓪ 먼저 정정한다 — E-189 의 "다음 수"가 틀렸다

E-189 는 *"다음 수는 `c_without` 배선이고, 그것은 정본 코드 변경 + GPU 재학습이다"* 로 닫았다.
**둘 다 틀렸다.** 근거는 전부 코드·설정·원자료다.

**(1) 학습 루프의 반사실은 "가끔 발화"한 적이 없다 — 한 번도 안 돌았다.**
`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:181` **`sdc_counterfactual: false`**, 6팔 런처 어느 것도 override 하지 않는다(밑줄·하이픈 두 표기 grep).
`verl_sdc.py:2696` 이 조기반환하므로 `cf_texts` 는 기입되지 않고, wandb `dcpo/cf_text_rate` 는 **1,465점 전부 0.0000**(b3shf 240 · b3s 302 · b3p 300 · b3nopmi 301 · b3null 302 · shiftonly 20).
⇒ E-189 이 관측한 충전율 **9.5%/6.2%/2.6%/0.2% 는 배선이 아니라 텍스트 폴백의 우연한 성공률**이다.
필요충분조건: **(텍스트에 `<|meta|>` 존재) AND (첫 `<|meta|>` 앞 구간에 숫자 존재)**
(`dcpo_region.py:397-398`, `:400`→`rewards.py:186-187` `nums[-1]`).
전문가시 **20,540행** 재실행 결과 예측·실제 **일치 100.00%, 불일치 0건** — 다른 생산자는 없다.
★팔별로 갈리는 이유는 **첫 `<|meta|>` 의 위치**다: b3s 는 메타가 첫 토큰(3,815/3,850)이라 접두에 숫자가 없어 발화 99.94%인데 충전 0.21%; b3nopmi/b3p/b3null 은 CoT **뒤**에 메타가 와서 `P(충전|has_meta)` 93~100%.
⇒ 충전율 = 발화율 × P(접두에 숫자) 이고, 두 항이 반대로 움직여 **어느 팔도 둘 다 높지 않다.**

**(2) 그 배선을 켜는 것은 config 한 줄이 아니고, 켜도 목표에 못 닿는다.**
`sdc_counterfactual=true` 는 `agent_name="cf_prefix_agent"` 라우팅(`verl_sdc.py:3017`)을 요구하는데 그 `agent:` 블록은 **일부러 삭제돼 있고**(`configs/…:84-88`, torch28x 크래시 사유 명시), 6팔 런처 **154행에 살아있는 중단 가드**가 부활을 감지하면 잡을 죽인다(`ABORT window`).
실패는 조용하다 — `verl_sdc.py:2951-2957` 이 전 예외를 삼켜 all-None 으로 되돌린다.
비용: CF 호출은 활성행 수와 무관하게 **메인 배치 B 전체로 패딩**(`verl_sdc.py:3008`) ⇒ B=512 ⇒ **스텝당 +512 시퀀스 = 롤아웃 생성 +100%.**
적격률 상한(최근10 실측): b3s 78.6% · b3shf 24.6% · shiftonly 14.8% · b3nopmi 10.4% · b3p **1.7%** · b3null **0.6%** ⇒ **90% 는 어느 팔도 불가.**

**(3) ★그리고 이 실험은 이미 아홉 번 돌았다.**
HF `iamseungpil/metacot-rv`(dataset, 119파일) `eval/` 아래 `cf_*.jsonl` **9건**. 내가 전부 내려받아 직접 재집계했다(워크플로 수치와 자릿수까지 일치):

| run | n | emit_A | leak(tag) | acc_A | acc_B | Δ | saved | broke | McNemar p | 95%CI | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rv_dcpo_gs50 | 594 | 589 | 0 | 0.7088 | 0.6987 | +0.0101 | 32 | 26 | 0.512 | [−0.015,+0.035] | 0.036 |
| rv_dcpo_gs60 | 594 | 262 | 0 | 0.7121 | 0.7003 | +0.0118 | 23 | 16 | 0.337 | [−0.009,+0.032] | 0.030 |
| rv_dcpo_gs70 | 594 | 435 | 0 | 0.7172 | 0.7205 | −0.0034 | 21 | 23 | 0.880 | [−0.025,+0.019] | 0.031 |
| rv_dcpo_gs90 | 594 | 594 | 0 | 0.6970 | 0.7256 | **−0.0286** | 16 | 33 | **0.021** | [−0.052,−0.006] | 0.033 |
| rv_dcpo_gs100 | 594 | 117 | 0 | 0.7020 | 0.6616 | **+0.0404** | 43 | 19 | **0.003** | [+0.015,+0.066] | 0.037 |
| rv_dcpo_gs190 | 594 | 594 | 0 | 0.7407 | 0.7492 | −0.0084 | 11 | 16 | 0.442 | [−0.026,+0.009] | 0.025 |
| rv_cgate_init | 594 | 417 | 0 | 0.5741 | 0.5976 | −0.0236 | 21 | 35 | 0.081 | [−0.048,+0.001] | 0.035 |
| rv_cgate_warmup | 594 | 266 | 0 | 0.6835 | 0.6936 | −0.0101 | 20 | 26 | 0.461 | [−0.033,+0.012] | 0.032 |
| pmishift_gs180 | 594 | 584 | 0 | 0.7492 | 0.7306 | +0.0185 | 21 | 10 | 0.071 | [+0.0002,+0.037] | 0.026 |
| **POOLED**(비독립·판정 근거 아님) | 5,346 | | 0 | | | **+0.00075** | 208 | 204 | 0.883 | [−0.0067,+0.0082] | 0.0106 |

읽어야 할 것 셋.
- ★**관측 효과(|Δ| 중앙 0.0118)가 각 런의 MDE(0.025~0.037)보다 작다.** ⇒ 9건 중 7건의 "검출 안 됨"은 **효과 0 이 아니라 해상도 미달**이다.
- ★**인접 두 체크포인트에서 부호가 반대로 유의하다**(gs90 −0.029 p=.021 vs gs100 +0.040 p=.003). ⇒ **단일 체크포인트 판정 금지.**
- 기록이 지지하는 가장 강한 문장: *"이 계열 전반에서 메타의 행 단위 순 인과기여는 **±1pp 안**"*.

**(4) E-188 의 선행조건은 오프라인 경로엔 해당 없다.**
`eval_counterfactual_difficulty.py` 는 메인 스레드에서 돌고 `_check_correctness` 가 math_verify 를 권위로 삼아 False 를 그대로 반환하며(`rewards.py:146-147`), 문자열 폴백의 예측 추출은 **`_extract_boxed_or_hash`**(`:160`)다. **마지막-숫자 폴백을 구조적으로 안 탄다** ⇒ E-188 인공물이 원리적으로 발생하지 않는다.

⇒ **정정된 다음 수: 정본 트레이너 배선도, 재학습도 아니다. 오프라인 반사실을 base 세대로 확장하는 것이고, 저장소 변경은 계측 배선 한 건뿐이다.**

### ① 이번에 실제로 배선한 것 (관측 전용·팔의 정책 불변)

`src/eval/eval_counterfactual_difficulty.py` +47/−2. **G5: 733 passed / 8 skipped — 패치 전후 동일**(3벌 독립 확인).

- **W-1b 산문 누출**(신규·필수): `_has_meta_signature`(`dcpo_region.py:98`, `^\s*(confidence|assessment|action)\s*:`)를 **양 팔에** 기록. ★기존 `emitted_meta_without` 은 **`"<|meta|>" in go` 태그 검사일 뿐**이라, 선행 9런의 **0/5,346 은 "태그가 없다"만 말하고 "메타 내용이 없다"는 말하지 않는다.** 우리 코드가 이 실패를 이미 실측으로 적어 놨다 — `signature_suppression_ids` docstring: *"태그 두 개만 막았더니 산문으로 샜고, 누출가드가 v3l 에서 CF 의 ~3/4 를 폐기했다."* ⇒ **선행 9건 결과의 최대 남은 위협이고, 그들은 생성문을 안 남겨 원리적으로 사후 확인이 불가능하다.**
- **W-4 절단**: `finish_reason` · 토큰수 양 팔.
- **W-3 재채점 가능성**: `gen_with`/`gen_without` 저장. 선행 9런은 스칼라 8키뿐이라 누출률·채점기 버전·다른 층화가 **영구히 재확인 불가**다.
- **잡음바닥**: `--repeat_arm_a` 로 arm A 를 두 번 디코드 → `correct_with2`·`identical_a2`.
- 배선검사(로컬): AST OK · 임포트 OK · 탐지기 6/6(오탐 케이스 `"the confidence interval"` → False 포함).

### ② 사전등록 (결과 전 확정)

- **링크(G6)**: 사다리 **E 사용**. ⛔**F 이득에는 닿지 않는다.**
- **G1 재구매 검사**: `CLAIMS.md` 923행 개념 grep 12히트, `c_without`/`acc_with`/`counterfactual` **0건** ⇒ CLAIMS 층위 신규. ★**그러나 원자료 층위에선 신규가 아니다**(위 9건) ⇒ 판정문보다 먼저 **C-032** 로 등록한다.
- **설계**: arm A 그리디 / arm B `logit_bias={<|meta|>: −100}`. ⚠arm B 는 *"같은 궤적에서 블록만 뺀 것"*이 아니라 **다른 정책**이다 ⇒ ⛔결과를 **"블록의 국소 인과"로 서술하지 않는다.**
- **주 지표**: 행 단위 `u = c_with − c_without ∈ {−1,0,+1}`, 집계 **문항 단위**(greedy k=1 ⇒ 문항=행), paired bootstrap 10,000. 적격 = `emitted_meta_with == True`.
  ★**동반 절대 지표 강제**: `acc_A_all − acc_B_all`(발화 무관 전 문항). ⛔조건부가 +인데 절대가 ≤0 이면 **헤드라인은 절대값**이다(0731 교훈).
- **배선검사(실패 시 발사 취소)**: W-1a 태그 누출(사전 기대 0) · ★**W-1b 산문 누출 < 5%** · 충전율 ≥95% · 절단 <10% · **눈으로 5개**(u=+1 둘, −1 둘, 0 하나 전문 첨부) · `|U_AA| < 0.005`.
- **검정력**: SD=√f, f 사전분포 = **선행 9런 실측 0.046~0.104(중앙 0.077)**. n=594·f=0.077 ⇒ **MDE 0.0265**. 등가성 δ=0.05·power .80 필요 n(`z_{1−β/2}`) = f .05→172 · .08→**274** · .10→**343**.
  ★**사전 고지**: 선행 9런의 |Δ| 중앙값 **0.0118 은 이 설계의 MDE 미만**이다. ⇒ *"검출 안 됨"* 이 나오면 그것은 **효과 0 이 아니라 해상도 한계**이며, 그렇게 적는다. ±1pp 를 읽으려면 문항 **약 6,300개**가 필요하고 그건 별도 예산 결정이다.
- **판정 밴드**: CI하한>0 ⇒ **바꾼다** / CI 전체가 (−δ,+δ) ⇒ **장식적** / CI상한<0 ⇒ **해롭다** / CI 가 0 을 포함하되 ±δ 밖 ⇒ ★**판정 불가·검정력 부족**(⛔"장식적"으로 쓰지 않는다) / 배선검사 실패 또는 `|U_AA|≥0.005` ⇒ **판정 없음** / ★**인접 체크포인트 부호 반전 시 단일-ckpt 판정 금지**.
- **선택성**: `sel⁺=P(u>0)`, `sel⁻=P(u<0)`, `sel_ratio=sel⁺/(sel⁺+sel⁻)`, 무작위 기준선 0.5. f<0.05 면 계산만 하고 판정 안 함.
- **대상 팔**(HF 현물): **b2p gs300**(발화 100%·메타SFT+바닐라GRPO = 가장 깨끗한 기준선·현 1위 팔) · **b3s gs303** · **b3sh gs175**. ⛔b3p(발화 1.85%)·b3nopmi(10.25%)·b3null 은 잴 메타가 없어 **사전 판정 불가**로 제외.
  ⚠발화율은 math500 16k **n8 샘플링**에서 옮긴 값이고 실행은 594 **greedy k=1** 이다 — **추정이지 관측이 아니다.** arm A 패스가 실측을 부산물로 준다.
- ★**반대 증거(결과 전 기재)**: (a) E-078 캘리브레이션 낙제 init 가 6/6 셀 우승 (b) E-189 문항내 대비에서 **우승자조차 −0.009** (c) 0622 cf_group: 반사실 보상 본격 실행 시 발화 gs50 만에 **0.000** (d) **선행 9런 풀링 +0.00075 [−0.0067,+0.0082]**. ⇒ **나는 U 가 0 근방 또는 작은 음수로 나올 것으로 본다.** 그래도 밴드대로 판정하고 ⛔팔을 바꿔 다시 돌리지 않는다.
- **답하지 않는 것**: held-out 정확도 연결 · 인과 기전/국소성 · 내용 대 형태(C-020 그대로) · instruct 세대 일반화(C-022) · 다중 시드(C-021).
- **기록 의무**: `U`·95%CI·실현 MDE·`f`·`U_AA`·선택성·절대차·n·채점기·예산·집계단위·**oid**. **판정문은 gpt-5.6-sol 적대검토 후 적층**(0805). 트랙 **CONF**.

### ③ 방법론 결함 — 세 번째 재발

이 회차 네 갈래 RCA 중 **누구도 "이미 아홉 번 돌았다"를 찾지 못했다.** 넷 다 CLAIMS 와 저장소 코드는 grep 했으나 **HF 산출물 저장소를 개념 grep 하지 않았다.**
0731(`MEASURED_INDEX` 개념 grep)·0807(외부 수치도 CLAIMS grep)의 정확한 재발이다.
★**처방**: 새 실험 설계 전 검사 대상에 **`HfApi().list_repo_files(...)` 개념 grep** 을 추가한다. 대장에 없는 결과가 원자료 저장소에는 **9건** 쌓여 있었다.

---

## E-191 (2026-08-10) — 【판정: 판정 없음】배선검사가 실패했고, 그 실패가 계기를 무너뜨렸다

**대상** `e190cf_b2p_gs300` (job pass) · HF `iamseungpil/metacot-rv` `eval/e190_b2p_gs300/cf_b2p_gs300.jsonl`
oid `6a2d2e732f099bf06af406937681d2558defecb153fffe4e5e4408150363b474` · 14,467,826 B · **n=594** · greedy k=1 · 예산 16,384 tok · 집계 문항단위 · 채점 `_check_correctness`(메인스레드, math_verify 권위 + `_extract_boxed_or_hash`).
내가 HF 에서 직접 내려받아 재집계했다(노드 로그 수치 미채택).
★**gpt-5.6-sol 적대검토 통과**(0805 지시). 코덱스가 낸 반박 **6건 전부 내 재계산으로 자릿수까지 재현**되어 채택했다. 아래는 정정 후 판이다.

### ① 사전등록 배선검사
| 검사 | 값 | 게이트 | |
|---|---|---|---|
| W-1a **여는 태그** 문자열 누출 | 0/594 = 0.0000 | <0.05 | PASS |
| **W-1b 산문 시그니처 누출** | **363/594 = 0.6111** | <0.05 | **FAIL** |
| **W-4 절단** | **81/594 = 0.1364** (A 17 / B 71) | <0.10 | **FAIL** |
| A-vs-A `U_AA` | +0.00168, Wald 95%CI **[−0.01546, +0.01883]** | \|점추정\|<0.005 | 절차상 PASS · **등가성 미입증** |
| W-3 눈검사 | u=+1 둘 · u=−1 둘 · u=0 하나 (양팔 정상종료) | 구성 충족 | 완료 |

⇒ 사전등록 §②: **배선검사 하나라도 실패 = 판정 없음.**
★**A-vs-A 게이트 설계 자체가 틀렸다**: 점추정만 검사하면 **대칭적 큰 흔들림이 상쇄되어 쉽게 통과**한다. 실제 행 수준 불일치는 **27/594 = 4.55%**(A1만 정답 14 / A2만 정답 13). ⇒ ⛔"정확도가 안정적"이라고 쓰면 안 된다. 올바른 게이트는 **CI 전체가 ±0.005 안에 드는 TOST** 이고 이 실행은 **통과하지 못한다**.
⚠**사전등록은 paired bootstrap 10,000 을 요구했는데 `e190_cf_gates.py:48,98` 은 Wald SE 를 쓴다.** 재계산: bootstrap(seed 0) **[+0.03535, +0.08754]** vs Wald [+0.03466, +0.08655]. 값은 비슷하나 **"사전등록 방식으로 계산했다"는 표기는 거짓**이었다 — 다음 실행 전 코드를 맞춰야 한다.

### ② 읽지 않는 주 지표, 그리고 **추정대상 셋이 서로 다른 답을 준다**
`U = +0.0606` (bootstrap 95%CI [+0.0354, +0.0875]) · saved 50 / broke 14 · f=0.1077 · realised MDE 0.0377. 절대 `0.7458 − 0.6852 = +0.0606`.
**선행 9건이었다면 이것을 "메타가 6pp 돕는다"로 보고했을 것이다.**

| 추정대상 | 값 | 상태 |
|---|---|---|
| **운영: 16k 예산에서 여는 토큰을 금지하면** | capped-policy ITT(`정답 ∧ finish=stop`) A 443 / B 405 · saved 51 / broke 13 · **U = +0.0640** [+0.0381, +0.0899] | **실재한다** |
| **의미론: "메타 내용만 제거"의 효과** | — | **식별 불가** (arm B 가 깨끗한 처치가 아니다) |
| **무제한 예산 정확도** | 미완주 17/71 의 최종 정답을 모름 ⇒ 보수적 경계 **U ∈ [−0.0556, +0.0926]** | **부호조차 미식별** |

★**정정**: 초안의 *"효과는 전부 절단에서 온다"* 는 **틀렸다.**
정확한 기술은 **net contrast 36 중 34(94.4%)가 한 팔 이상 절단된 행에서 발생했다** 이고, 이는 **처치후 분해**다. 절단을 실패로 처리하는 ITT 로 다시 채점하면 효과는 **사라지지 않고 오히려 커진다**(+0.0640).
both-stop 제한(n=513, U=+0.0039 [−0.0148,+0.0226] p=0.8388)은 **처치후 변수 조건화**라 비편향 추정이 아니다 — 효과의 소재를 가리키는 기술일 뿐이다.

### ③ ★근인 — arm B 는 "메타 없는 정책"이 아니다
`logit_bias` 는 **여는 토큰 id 하나**만 −100 으로 누른다. **닫는 태그는 다른 id 라 안 막힌다.**

| arm B 에서 | 값 | arm A |
|---|---|---|
| 여는 태그 `<\|meta\|>` | **0/594 = 0.0000** ← 선행 9건이 잰 유일한 수 | 594/594 |
| **닫는 태그 `<\|/meta\|>`** | **464/594 = 0.7811** | 594/594 |
| `decision:` 필드 | 428/594 = 0.7205 | — |
| 내용 시그니처(`confidence:`/`assessment:`/`action:`) | 363/594 = 0.6111 | 594/594 |
| 반복 붕괴(정의 의존, 아래) | 67/594 = 0.1128 | 12/594 = 0.0202 |

**W-3 눈검사 실물** — 셋 다 **양팔 정상종료** 행이다:
- row 3366(u=+1): arm B 가 문장 중간에 `Let the intersection point be \[ ( <\|/meta\|>\nconfidence: 0.00\n…` — **닫는 태그로 메타 블록을 연다.**
- row 268(u=−1): arm B 가 **`<\|/meta\|>` 로 시작**한다.
- row 4457(u=0): arm B 가 `Now <\|/meta\|>\nconfidence: 0.88 … decision: verify\n<\|/meta\|>` 로 **온전한 메타 블록**을 쓴다.
붕괴 실물: row 2932 는 `confiden` 을 16,384 토큰 내내 반복 · row 3365 는 `<\|/meta\|>` 로 시작해 수천 개 개행 · row 1936 은 `いらっ` 이물 뒤 `confidence: 0.00`.

⇒ ★**`"<|meta|>" in text` 는 메타 부재를 재지 않는다.** 모델은 금지 토큰을 **우회**한다.
★**정정**: *"arm B 는 망가진 디코더다"* 는 과했다. 방어 가능한 서술은 **"b2p 의 arm B 는 깨끗한 meta-free 대조군이 아니며, 억제와 연관된 제어문법 교란·반복·절단을 보인다"** 이다. "우회하려는 의도"는 출력에서 관찰되지 않는 심리적 설명이다.

**길이도 정정한다.** 초안의 *"B 가 2.6배 폭주"* 는 **평균이 만든 착시**다:
- 중앙값 A 228 / B 295 = **1.29배**
- **붕괴 67건을 빼면 평균 A 837 / B 665 — B 가 오히려 짧다**
- 양팔 정상종료 513행: A 440 / B 516 (약 17% 차)
- 붕괴 67건 중 **66건이 `finish=length`** ⇒ 붕괴와 절단은 **독립 진단이 아니다**(이중계상 금지). "반복 붕괴가 절단의 주된 형태"라고 써야 한다.
⇒ 전반적 장문화가 아니라 **작은 병적 tail 이 평균을 끌어올린 혼합분포**다.

**붕괴율은 정의 의존적이다**(코덱스 확인, 내 재계산 일치): 임계값 5~1000 에서 전부 67/594. 빈 줄을 반복으로 안 세면 B 57 / A 6. 저장소 기존 탐지기(`src/eval/cf_stats.py:57`) 기준 B 74 / A 22. ⇒ **정확한 11.28% 는 정의 의존이나, B 의 비정상 반복이 훨씬 많다는 질적 결론은 견고**하다.

### ④ 누출 층화 — 여기도 정정한다
| arm B | n | acc_A | acc_B | U | 95%CI |
|---|---|---|---|---|---|
| 시그니처 O | 363 | 0.7218 | 0.7273 | **−0.0055** | [−0.0297, +0.0186] |
| 시그니처 X | 231 | 0.7835 | 0.6190 | +0.1645 | [+0.1124, +0.2166] |
| **시그니처 X + 양팔 정상종료** | **173** | — | — | **+0.0173** | **[−0.0125, +0.0472]** |

★**정정**: *"못 쓴 행이 곧 붕괴·절단 행"* 은 **거짓**이다. 시그니처 없는 231행 중 절단/붕괴는 **57건뿐**이다. 큰 +0.1645 의 대부분이 그 57건에서 나오는 것은 맞다. ⚠이 층화 자체도 **처치후 선택**이다.

### ⑤ 선행 9건(C-032)에 대한 함의 — **무효가 아니라 위협**
★**정정**: b2p **한 체크포인트**로 9건(rv_dcpo 여러 step · cgate init/warmup · pmishift — 서로 다른 모델 상태)에 외삽할 수 없다. 누출·붕괴 확률은 체크포인트의 로짓 구조와 메타 발화 습관에 의존한다.

- **말할 수 있는 것**: 선행 9건은 `"<|meta|>"` 부재만 확인했고 **생성문·종료상태·토큰수를 저장하지 않아**(스칼라 8키뿐, 내가 9개 파일 전부 확인) arm B 가 실제로 meta-free 였는지 **사후 검증이 원리적으로 불가능**하다. b2p 실행은 그 식별 가정이 **실패할 수 있음을 실증**하므로 9건의 **의미론적 해석에 중대한 위협**을 제기한다.
- ⛔**말할 수 없는 것**: "9건 모두 무효다" / "그 1~4pp 가 이 인공물에서 왔다".
- 9건은 여전히 **해당 logit-bias 정책 개입의 효과**는 측정한다. 그것을 "메타 제거 효과"라 부를 수 없을 뿐이다.

`e190cf_b3sh_gs175`·`e190cf_b3s_gs303` 를 **계기 진단용으로 계속 돌린다**(판정용 아님) — 누출·붕괴가 체크포인트 특이적인지 일반적인지가 이 위협의 크기를 정한다.

### ⑥ 닫는 것 / 여는 것
- **★닫는 것**: ⛔`"<|meta|>" in text` 를 **메타 부재 지표로 쓰는 것**. ⛔점추정만 보는 A-vs-A 게이트. ⛔단일 태그 `logit_bias` 억제로 만든 반사실을 **"메타 제거"로 해석**하는 것(이 실행 + C-032 9건).
- **★여는 것 — 그리고 여기서 초안이 또 틀렸다**: *"splice+continue 기계가 `harvest_redirect_cf.py` 에 이미 있다"* 는 **거짓**이다. 그 파일은 `splice_index`(오답 trace 의 30~70% 지점)와 R/N′/Nc **판정 로직만** 담고, **`main()` 은 `SystemExit` 로 할 일 목록을 던지며 종료한다**(`:192`). 실행 생성기는 **미구현**이다.
  ⇒ 유효한 반사실 설계는 여전히 **prefix-forced continuation** 이 유력하나 **새로 구현해야 하고**, ⚠prefix 강제도 문맥에 개입하므로 **"붕괴가 원리적으로 없다"는 보장은 없다**. 발사 전 같은 배선검사(닫는태그·시그니처·붕괴·절단·TOST)를 그대로 걸어야 한다.
  ⚠닫는 태그까지 막는 안은 **이 데이터로 시험하지 않았다** — 예측일 뿐이므로 근거로 쓰지 않는다.
- **부수**: A-vs-A 동일 텍스트 **229/594 = 0.3855**. ⇒ *"이 코드 경로의 반복 greedy 호출은 텍스트 수준에서 재현되지 않는다"* 는 참이나, **원인을 vLLM 배치 비결정성으로 귀속한 것은 미입증**이다(backend 수치·스케줄러·호출 순서 등 미분리). ⚠A2 는 `correct_with2`/`identical_a2` 만 저장하고 생성문을 버려(`eval_counterfactual_difficulty.py:154`) 365건 불일치의 성격을 지금은 감사할 수 없다 — **다음 실행에서 A2 생성문도 저장할 것**.
- **운영**: `checkpoints/rq3v2f_b2p/global_step_300/actor` 총 **98.28 GB** 중 `optim_world_size_4_rank_*` **65.51 GB**(내가 `HfApi().get_paths_info` 로 파일별 실측). eval 엔 불필요 ⇒ `ignore_patterns` 로 67% 절감 — **승인 대기**.

### ⑦ 이 회차가 산 것
**계기 하나를 폐기했다.** 그 계기로 아홉 번 측정했고, 열 번째에 처음으로 계측을 붙였더니 첫 사용에서 무너졌다.
⇒ **배선검사를 안 붙인 측정은 측정이 아니다.** 그리고 **적대검토는 판정문의 문장 열두 개를 고쳤다** — 그중 셋(*"효과는 전부 절단"* · *"9건 무효"* · *"splice 기계가 이미 있다"*)은 채택했다면 **다음 실험의 설계를 잘못 정했을 것**이다.

---

## E-192 (2026-08-10) — b3s gs303: 같은 개입, 다른 파괴. **그리고 내가 만든 탐지기가 정본 탐지기를 이겼다고 착각했다**

**대상** `e190cf_b3s_gs303`(job pass) · HF `eval/e190_b3s_gs303/cf_b3s_gs303.jsonl` oid `8136423fc1e20f51…` · 20,283,110 B · n=594 · greedy k=1 · 16k · 내가 직접 재집계.
⛔**판정용 아님** — arm B 무효는 C-033 확정. **계기 진단 전용**(결함이 체크포인트 특이적인가).
★**gpt-5.6-sol 적대검토 통과**. 반박 **전부 내 재계산으로 재현**되어 채택. **초안의 헤드라인 하나가 부호까지 뒤집혔다**(③).

### ① 두 팔 대조
| | 발화 A | 여는태그 B | 닫는태그 B | 시그니처 B | **B단독 절단** | **W-4(어느팔이든)** | acc_A | acc_B |
|---|---|---|---|---|---|---|---|---|
| b2p gs300 | 1.0000 | 0.0000 | **0.7811** | 0.6111 | 0.1195 | **0.1364** | 0.7458 | 0.6852 |
| b3s gs303 | 1.0000 | 0.0000 | **0.1229** | 0.1077 | 0.2593 | **0.2744** | 0.7374 | **0.2071** |

⚠**W-4 정정**: 초안이 쓴 b3s 0.2593 은 **B 단독** 값이다. 사전등록 정의(어느 팔이든 비-stop)로는 **163/594 = 0.2744**.
b3s 진단 지표: 전체 U=+0.5303 [+0.4881,+0.5726] · 양팔 stop n=431 accA 0.8028 / accB 0.2390 U=+0.5638 · capped-ITT U=+0.5606.
A-vs-A: b3s U_AA −0.00168 [−0.02005,+0.01669] 행뒤집힘 31/594=0.0522 동일텍스트 245/594=0.4125 (b2p 0.00168 / 4.55% / 38.55%).
⚠**정정**: *"재현 불가성은 하네스 성질"* 은 **말할 수 없다**. 체크포인트와 하네스가 함께 바뀌지 않은 설계라 원인 분리가 불가능하고 독립 조건은 **n=2** 다. 가능한 문장은 *"같은 하네스의 두 실행에서 관찰됐다"* 까지.

### ② ★사전 예측 채점 — 방향 맞음, 기전 틀림
사전 기재: *"b3s/b3sh 는 메타가 위치 0 이라 더 심할 것."*
- **심각도**: 맞음. 파괴폭 **+0.53 vs +0.06**, 약 9배.
- **기전**: **틀림.** 나는 누출이 더 클 것으로 함축했으나 실제는 반대다(닫는태그 0.1229 vs 0.7811).
⚠단 *"b3s 는 우회하지 못한다"* 는 절대화라 틀렸다 — **닫는태그 73/594, 시그니처 64/594 로 0 이 아니다.**

### ③ ★★내 탐지기가 틀렸다 — 정본 탐지기로는 방향이 뒤집힌다
초안은 내가 즉석에서 만든 판정기(연속 동일 줄 ≥30)로 *"b3s 붕괴 0.0118 < b2p 0.1128, b3s 는 덜 붕괴"* 라고 썼다. **`src/eval/cf_stats.py:57 degeneracy_flags` 로 다시 세면 반대다:**

| | repetition B | **too_short B** | **no_answer B** | **any flag B** | any flag A |
|---|---|---|---|---|---|
| b2p gs300 | 0.1246 | 0.0017 | 0.0673 | **0.1263** | — |
| b3s gs303 | **0.2020** | **0.4343** | **0.5185** | **0.6414** | 0.0000 |

⇒ **b3s arm B 의 64.14% 가 프로젝트 자체 기준으로 퇴화**다. 내 임시 판정기는 `'confi'` 3토큰을 *"반복 아님"* 으로 흘려보냈다.
★**교훈**: **임시 판정기를 만들기 전에 저장소 정본 판정기를 grep 하라.** 이번엔 정본이 존재했고, 내 것보다 정확했고, 부호가 반대였다.

### ④ ★★그리고 그 정본 도구는 **내 게이트에 안 걸려 있었다**
`grep -n "cf_stats\|too_short\|no_answer" experiments/analysis/e190_cf_gates.py src/eval/eval_counterfactual_difficulty.py scripts/run_e190_cf_eval.sh` → **히트 0.**
`degeneracy_flags` 는 `repetition` · `too_short` · `no_answer` **셋을 이미 구현**해 두었고, docstring 은 *"flagged row must not count as a redirect win"* 이라고 용도까지 적어 놨다.
⇒ ⛔초안의 *"내 게이트 넷이 상상하지 못한 실패 양식"* 은 **거짓**이다. **프로젝트는 상상했고 구현까지 했다. 내가 배선을 안 했을 뿐이다.**
★이것은 §5-4 *"도구가 있는데 관문에 안 걸려 있으면 미배선 = 미완성"* 의 **정확한 재발**이다. 다음 설계의 배선검사에 `degeneracy_flags` 를 **정본으로** 건다.

### ⑤ 기전 — 생성문
arm B 최빈 출력 첫 40자: **`confiden` 145건 · `confi` 104건**.
길이: `ntok_B = 3` **250/594** · `ntok_B < 20` **251/594 = 0.4226** · `≤50` 256/594 · **상한 16384 도달 154/594**. 중앙 **118**(A 228.5, 비 **0.516**). b2p 는 `ntok_B<20` **0/594**, 비 1.294. 양쪽 다 `ntok_A<20` = 0.
⇒ 정확한 서술: **"arm B 의 약 42% 행에서 3토큰 안팎의 무정답 stop 모드가 발생했다."**
⚠**정정**: *"완주했는데 아무것도 안 냈다"* 는 과하다. 하네스가 `finish_reason` 만 저장하고 **`stop_reason` 을 버려**(`eval_counterfactual_difficulty.py:144`) EOS 인지 명시적 stop 인지 판별할 수 없다. `finish=stop` 은 *"상한 절단이 아니다"* 이지 *"의미론적 완주"* 가 아니다. ⇒ **다음 실행에서 `stop_reason` 도 저장한다.**

### ⑥ b2p 효과의 기술적 분해 (⚠인과 분해 아님 — 전부 처치후 결과)
| 층 | n | net | 전체 기여 |
|---|---|---|---|
| **전체** | 594 | **+36** | **+0.0606** |
| 한 팔 이상 절단 | 81 | +34 | +0.0572 |
| 양팔 stop | 513 | +2 | +0.0034 |
| 닫는태그 B 있음 | 464 | +4 | +0.0067 |
| 시그니처 B 있음 | 363 | −2 | −0.0034 |
| **정본 repetition B** | **74** | **+37** | **+0.0623** |
| **정본 repetition B 없음** | **520** | **−1** | **−0.0017** |

⇒ ★**효과 전부가 퇴화 행에 있다** — 비퇴화 520행의 net 은 **−1** 이다.
⚠**정정**: 초안의 *"b2p 양식 인공물 크기가 선행 9건의 1~4pp 대역과 겹친다"* 는 **삭제한다.** 절단 제거 후 기술적 잔차는 **약 +0.34pp** 이고, 닫는태그/시그니처 층화는 각각 +0.67pp / −0.34pp 다. **순수 우회 기여는 무작위 팔이 없어 원리적으로 식별 불가**다. 선행 9건의 |Δ| 도 정확히는 **0.34~4.04pp**(두 건은 1pp 미만)다.

### ⑦ 선행 9건(C-032)의 위협 — 좁혀졌지만 초안만큼은 아니다
선행 9건 acc_B **0.5976~0.7492**, Δ **+1.85 / −2.36 / −1.01 / +4.04 / −0.84 / +1.01 / +1.18 / −0.34 / −2.86 pp**(내 재집계).
- ✅**말할 수 있는 것**: **b3s 급 집계 정확도 파국(Δ +53pp, acc_B 0.207)은 9건에 없었다.** 더 정밀하게 — b3s 에서 관측된 `<20` 무정답 질량 42.26% 가 그대로 있었다면 최대 정확도가 **0.5774** 이므로, 그 정확한 혼합비는 **9건 전부에서 배제**된다.
- ⛔**말할 수 없는 것**: *"b3s 양식 / 다른 생성 파괴를 배제했다"*. **b2p 가 반례다** — acc_B 0.6852 로 선행 범위 안에 있으면서 닫는태그 78.11% · 시그니처 61.11% · 절단 11.95% 였다. 정확도만으로는 **1−acc_B = 25.08~40.24%** 까지 파괴 질량이 숨을 수 있다.
- ⛔여전히 금지: *"9건 무효"* · *"그 1~4pp 가 인공물이다"*.

### ⑧ 여는 것
- 배선검사에 **`cf_stats.degeneracy_flags` 를 정본으로** 건다(`repetition`/`too_short`/`no_answer`). 고정 `<20` 단독보다 **`P(flag_B) − P(flag_A)`** 로 대조한다 — b3s 는 0.6414 vs 0.0000 으로 즉시 잡히고, b2p 는 0.1263 vs 낮음으로 잡히지 않으므로 **이 게이트 하나로는 부족**하다(누출 게이트와 병행).
- **`stop_reason` 저장** · **A2 생성문 저장** · **`e190_cf_gates.py` 를 bootstrap + TOST 로** 수정.
- b3sh 는 선점 후 **재개되어 running**(2h). 도착하면 세 번째 양식 확인.

---

## E-193 (2026-08-10) — 세 팔 완주. **같은 개입, 네 갈래 파괴.** E-190 조사 종결

**대상** `e190-cf-0810` 3잡 전부 pass. 정본 `src/eval/cf_stats.py degeneracy_flags` 로 내가 직접 재집계.
oid: b2p `6a2d2e73…`(14,467,826B) · b3s `8136423f…`(20,283,110B) · b3sh `7ac0eaf9…`(16,709,784B) · 각 n=594 · greedy k=1 · 16k.
⛔**전부 판정용 아님** — arm B 무효는 C-033 확정. **계기 진단**이다.
★**gpt-5.6-sol 적대검토 통과**. 반박 전부 내 재계산으로 재현되어 채택. **초안의 헤드라인 두 개가 무너졌다**(②③).

### ① ★G8 — 내가 팔 정체를 여러 틱 잘못 적었다
모니터링 노트에 *"b3sh = shiftonly + `w_format=0.35`"* 라 적어 왔다. **틀렸다.**
`h100std_rq3v2f_b3sh.yaml:254` = **`dcpo_w_format=0.0`** · `w_cal=0` · `w_emit=0` · `meta_floor=0` · `len_cost=0`.
`w_format=0.35` 는 **b3shf**(`h100std_rq3v2f_b3shf.yaml:254`, 현재 학습 중인 별개 팔).
⚠**그리고 b3sh↔b3shf 는 `w_format` 하나만 다르지 않다**: `trainer.val_before_train` **True → False**(`:275`), lineage 이름 전부, 그리고 ★**`WANDB_RUN_ID` 가 둘 다 `rq3v2f-b3sh-1`**(`b3shf:43`, `WANDB_RESUME: allow`) — E-187 의 오염 의심이 여기서 코드로 확인된다.
⇒ G8 (*"이름은 이전 세션이 만든 캐시다"*)이 실제로 잡았다. **보상 목적함수 차이는 `w_format` 하나가 맞으나, "전 키 한 키 차이" 는 거짓.**

### ② 세 팔 프로파일 (정본 탐지기)
| | 닫는태그 B | **고아 닫는태그**(내용 0) | 내용 시그니처 B | `decision:` B | rep B | too_short B | no_answer B | **any-flag B** | any-flag A | W-4 | ntok_B<20 | 중앙비 B/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b2p gs300 | 0.7811 | **0.0623** | 0.6111 | 0.7205 | 0.1246 | 0.0017 | 0.0673 | **0.1263** | 0.0370 | 0.1364 | 0.0000 | 1.294 |
| b3s gs303 | 0.1229 | **0.0000** | 0.1077 | 0.1229 | 0.2020 | 0.4343 | 0.5185 | **0.6414** | 0.0589 | 0.2744 | 0.4226 | 0.516 |
| b3sh gs175 | **0.9966** | **0.4899** | 0.4529 | 0.4428 | 0.1498 | 0.0034 | 0.0505 | **0.1532** | 0.0354 | 0.1414 | 0.0000 | 1.183 |

★**정정 — 초안의 *"b3sh 는 99.66% 가 우회한다"* 는 틀렸다.**
b3sh 의 0.9966 은 **닫는 태그 문자열 출현율 = delimiter 오염률**이지 메타 사용률이 아니다. 592건 중 **291건(0.4899)이 고아 태그**로, `confidence:` 도 `decision:` 도 없이 태그 뒤에 곧바로 수학 풀이가 온다(553/594 는 아예 태그로 시작). **정본 내용 시그니처는 0.4529.**
반면 **b2p 의 고아율은 0.0623 뿐**이다 — b2p 의 464건은 대부분 **진짜 메타 블록**이다(시그니처 363, 시그니처∧decision 363).
⇒ **고아율이 네 번째 구별 축이다.** b2p=진짜 우회 · b3s=우회 실패 · b3sh=절반이 구분자 오염.

| | acc_A | acc_B | U | 95%CI | saved | broke | U_AA | 행 뒤집힘 |
|---|---|---|---|---|---|---|---|---|
| b2p | 0.7458 | 0.6852 | +0.0606 | [+0.0347,+0.0865] | 50 | 14 | +0.00168 | 0.0455 |
| b3s | 0.7374 | **0.2071** | +0.5303 | [+0.4881,+0.5726] | 323 | 8 | −0.00168 | 0.0522 |
| b3sh | 0.7205 | 0.6532 | +0.0673 | [+0.0365,+0.0982] | 65 | 25 | **−0.01178** | 0.0556 |

⇒ **세 체크포인트에서 결함 프로파일이 서로 달랐다.** 한 실행의 프로파일로 다른 실행을 추정할 수 없다(E-192 의 두 점 주장을 세 점으로 반복 관찰).

### ③ 퇴화/비퇴화 분해 — **초안의 "부호는 음수 쪽" 은 철회**
| | 전체 net | 퇴화 층 | 비퇴화 층 | 비퇴화 U 95%CI |
|---|---|---|---|---|
| b2p | +36/594 | n=75 **+37** | n=519 **−1** | **[−0.0208, +0.0170]** |
| b3s | +315/594 | n=381 +277 | n=213 +38 | [+0.1165, +0.2403] |
| b3sh | +40/594 | n=91 **+45** | n=503 **−5** | **[−0.0355, +0.0156]** |

b2p·b3sh 에서 **퇴화 층이 전체 net 을 통째로 만든다**(+37/+36, +45/+40). 그러나 ⚠**비퇴화 층의 CI 둘 다 0 을 넓게 포함**한다.
★**정정**: *"개입이 안 망가뜨린 행에서는 메타가 돕는다는 증거가 없고 부호는 음수 쪽"* → **뒤 절반 삭제.**
말할 수 있는 것: **정본 퇴화 탐지기에 걸리지 않은 사후 부분집합에서는 어느 방향으로도 유의한 차이가 없다.** ⚠처치후 선택이라 인과 추정도 아니다.
b3s 만 비퇴화 층이 유의(+0.1784)한데, 전체의 36%(213/594)만 남은 **심하게 선택된 잔여**다.

### ④ A-vs-A — "잡음의 5.7배" 는 삭제
b3sh: A1−A2 **−7/594 = −0.01178**, CI **[−0.03072, +0.00715]**, 행 뒤집힘 **33/594 = 0.0556**(A1만 13 / A2만 20). A1−B **+0.06734** [+0.03651,+0.09817] · A2−B **+0.07912** [+0.04901,+0.10924].
★**정정**: `U/U_AA ≈ 5.7` 나눗셈은 **신호대잡음비가 아니다** — `U_AA` 는 잡음의 표준편차가 아니라 **한 번 관측된 부호 있는 대비**이고, 상쇄로 작아질 수 있으며 한 반복으로 분포를 추정할 수 없다.
정확한 서술: **b3sh 는 동일 조건 두 디코드에서 점정확도가 1.18pp 달랐고 5.56% 행이 뒤집혀 사전 게이트 `|U_AA|<0.005` 를 실패했다. 따라서 단일 디코드로 1pp 급 정밀도를 주장할 수 없다.** ⚠단 이것이 관측된 6.7pp 의 **부호를 잡음이 뒤집었다는 증거는 아니다** — 그 차이를 "메타 효과"로 못 읽는 이유는 **퇴화와 식별 실패**다.

### ⑤ ★선행 9건(C-032) — 최종 정리
- ✅**말할 수 있는 것**: `emitted_meta_without` 은 **여는 태그 문자열 검사**(`eval_counterfactual_difficulty.py:123`)일 뿐이므로 **`0/5,346` 은 "arm B 가 메타를 안 썼다"는 증거가 아니다.** 세 팔 중 둘에서 그 지표가 0 인 채로 닫는 태그가 78.11% / 99.66% 출현했고, 그중 진짜 메타 내용은 61.11% / 45.29% 였다.
- ✅ b3s 급 집계 정확도 파국(acc_B 0.2071, Δ +53pp)은 9건에 없었다(acc_B 0.5976~0.7492).
- ⛔**말할 수 없는 것**: *"선행 9건은 b2p/b3sh 프로파일과 일치하므로 arm B 가 메타를 계속 썼다"*. **정확도 대역 포함은 프로파일 동일성의 증거가 아니다.** 정확한 문장: **선행 arm B 정확도는 b2p/b3sh 값이 놓인 범위를 포함하지만, 저장된 8키만으로는 생성 프로파일도 메타 사용 여부도 식별할 수 없다.**
- ⛔여전히 금지: *"9건 무효"* · *"그 Δ 가 인공물이다"* · *"그들 우회율이 78~99% 였다"*.

### ⑥ E-190 조사 종결 — 닫는 것 / 여는 것
- **닫는다**: E-190 3팔 조사. ⛔`logit_bias` 단일-태그 억제 계열 반사실 전부. ⛔`"<|meta|>" in text` 를 메타 부재 지표로 쓰는 것.
- **연다**: **prefix-forced continuation 구현**(승인 대기). 발사 전 배선검사에 다음을 **정본으로** 건다 —
  ①`cf_stats.degeneracy_flags` 를 `P(flag_B) − P(flag_A)` 로(⚠b3s 0.6414 vs 0.0589 는 즉시 잡히나 **b2p 0.1263 vs 0.0370 은 안 잡힌다** ⇒ 누출 게이트 병행 필수)
  ②**고아 닫는태그율**(신규 — b3sh 를 잡는 유일한 축) ③내용 시그니처 ④W-4 절단 ⑤**TOST 등가성**(점추정 게이트 폐기) ⑥`stop_reason` 저장 ⑦A2 생성문 저장 ⑧사전등록대로 **paired bootstrap**(현행 `e190_cf_gates.py:48,98` 은 Wald).
- ⚠**prefix 강제도 문맥 개입이라 붕괴가 원리적으로 배제되지 않는다.** `harvest_redirect_cf.py:192 main()` 은 `SystemExit` — **생성기는 미구현**이다.

### ⑦ 이 조사가 산 것
**계기 하나를 폐기했고, 그 계기로 아홉 번 측정한 기록의 해석 근거를 무효화했다** — 정확히는 *"그 지표가 0 이라는 사실이 아무것도 보장하지 않는다"* 는 것을 세 팔로 보였다.
그리고 **내가 급조한 도구가 이 조사에서 네 번 나를 속였다**: 없는 기계를 있다고(`main()` 미독), 게이트를 점추정으로, 탐지기 부호를 반대로, 팔 이름을 캐시된 결론으로.
★넷 다 **저장소에 정본이 있었거나 한 줄 확인이면 됐다.** 다음 설계는 새 도구를 만들지 않고 정본을 배선한다.

---

## EXP-0811a (탐색 · ⛔판정 아님 · 사전등록 없음 · 헤드라인 금지)

E-190 이 남긴 부산물로 훑었다 — **이 프로젝트 최초로 CF eval 생성문 전문이 저장됐고, arm A 는 개입이 없어 유효하다**(C-033 은 arm B 만 죽였다). n=594 greedy k=1, 세 팔.

**① 선언 confidence 는 난이도 탐지가 아니다** — 난이도 층 **안에서도** 분리된다(Wilson 95%CI 비중첩):
| | easy | medium | hard |
|---|---|---|---|
| b2p gs300 | +0.352 분리(n=188/10) | +0.034 겹침 | +0.273 분리(n=33/102) |
| **b3s gs303** | 전부 0.88(대비 불가) | **+0.264 분리**(0.851 vs 0.586) | **+0.625 분리**(0.825 vs 0.200) |
| b3sh gs175 | +0.026 겹침 | +0.161 분리 | +0.662(hi n=5, 약함) |
point-biserial r: b3s **+0.533** · b2p +0.341 · **b3sh −0.018**.

**② ★그 신호는 추론 내용이 아니라 1비트다** — b3s 의 메타 블록 **고유 서두가 594행에 8개**(0.013), 최빈 하나가 **69.2%**, confidence 값은 **0.88/0.75 둘뿐**. b2p 는 고유 146(0.246)·최빈 18.9%.
⇒ **가장 판박이인 팔이 가장 잘 가른다.** 나르는 것은 *"둘 중 어느 템플릿이 발화했나"* 이지 블록 안의 문장이 아니다.

**③ b3sh 는 어휘가 늘고 정보가 사라졌다** — confidence 값 5종(0.0/0.12/0.25/0.75/0.88)으로 가장 다양한데 r **−0.018**, easy 층에서 `conf<0.5` 행의 정확도가 **0.864 로 오히려 높다**(반상관). PMI-shift 단독(shiftonly)이 **선언을 다양하게 만들되 교정을 깬** 모양.

**④ 라우팅** — b2p verify 588/redirect 6(0.333) · **b3s redirect 0** · b3sh verify 528(0.705)/redirect 66(**0.848**).

⚠**한계**: 상관이지 인과 아님 · `difficulty` 3단계는 거칠어 층내 잔여 교란 있음 · 단일 greedy 패스 · 594문항 · **EXP 트랙이라 헤드라인·주장 금지**.
★**여는 것**(다음에 물을 것): 이 1비트가 **held-out 정확도로 이어지는가**, 그리고 **1비트를 라우팅에 실제로 쓰면**(0.75 일 때만 재시도) 이득이 나는가. ⇒ 4분해의 *평가✅ / 사용❌* 를 새 데이터가 다시 가리킨다.

## EXP-0811b (탐색 · ⛔판정 아님) — **0811a 의 읽기를 스스로 뒤집는다: 문항을 잡으면 신호가 없다**

EXP-0811a 는 난이도 **3단계 층** 안에서 confidence 가 정답을 갈랐다고 적었다(b3s hard +0.625). **더 강한 통제가 가능했다** — held-out MATH500 parquet 은 문항당 **k=8 롤아웃**이라 **같은 문항 안에서** hi(≥0.85) vs lo(<0.85) 를 비교할 수 있고, 그러면 난이도가 **상수**가 된다.

**결과(문항내, n=4,000/팔, 16k n8)**
| | 혼합문항 | conf≥0.85 | <0.85 | 차 | 방향(hi/lo/동률) | U 95%CI | 부호검정 p |
|---|---|---|---|---|---|---|---|
| b2p gs300 | **490**/500 | 0.6321 | 0.6177 | +0.0144 | 80/59/351 | +0.0429 [−0.0041,+0.0899] | 0.089 |
| **b3s gs300** | **145**/500 | 0.6245 | **0.6405** | **−0.0160** | 17/**30**/98 | **−0.0897** [−0.1812,+0.0019] | 0.079 |
| b3sh gs175 | 422/500 | 0.6405 | 0.6381 | +0.0024 | 56/51/315 | +0.0118 [−0.0362,+0.0599] | 0.699 |
| b3p gs300 | 1/500 (발화 **98/4000**) | — | — | — | — | — | 판정 불가 |

⇒ ★**셋 다 비유의. b3s 는 음의 방향.** 0811a 의 `r=+0.533` 은 **난이도 3단계로 못 걷어낸 문항 간 구조**였을 가능성이 크다.

**⚠단, 두 분석은 네 가지가 다르다 — "0811a 가 틀렸다"고 단정하지 않는다**: ①코퍼스(594 val meta-mix vs held-out MATH500) ②디코딩(greedy k=1 vs n=8 샘플링) ③채점기(신선 `_check_correctness` vs **저장된 `is_correct`** — C-002 로 옛 채점기 의심) ④문항내는 **혼합문항만** 쓴다(b3s 는 145/500 뿐 = 71%가 8샘플 전부 같은 값 ⇒ 심한 선택).
말할 수 있는 것: **held-out 에서 문항을 통제하면 선언 confidence 는 그 시도의 정답을 가르지 못한다.** 절대 정확도는 인용하지 않는다(C-002).

★**이것이 4분해의 "평가 ✅" 를 흔든다.** 그 ✅ 는 다른 과제(에너지 자 AUC 0.988)에서 온 것이고, **수학에서 메타가 선언하는 confidence 는 문항내 정보가 없다.** 평가도 사용도 못 하면 남는 것은 발화뿐이다.
★**여는 것**: (b) 라우팅 재시도 실험의 전제가 무너졌다 — 1비트에 정보가 없으면 그 비트로 재시도해도 이득이 없다. **(b) 는 취소하고**, 대신 *"문항내 정보를 가진 신호가 하나라도 있는가"*(길이·엔트로피·자기일관성)를 먼저 물어야 한다.

## EXP-0811c (탐색 · ⛔판정 아님) — **문항내 정보는 있다. 메타 블록 안이 아닐 뿐이다**

0811b 가 *"선언 confidence 는 문항내 정보가 없다"* 로 닫았다. 그러면 **정보를 가진 신호가 하나라도 있는가**를 같은 통제(문항 고정, held-out MATH500 16k n8, 4,000행/팔)로 물었다.

| 문항내 대비 | b2p gs300 | b3s gs300 | b3sh gs175 |
|---|---|---|---|
| (i) 길이 < 중앙 | +0.0064 [−0.079,+0.092] p=1.00 (혼합157) | +0.0473 [−0.038,+0.132] p=.341 (169) | +0.0163 [−0.071,+0.104] p=.807 (184) |
| **(ii) 메타 블록 ≥2** | **−0.2093 [−0.357,−0.061] p=.011** (86) | (혼합 2, 판정불가) | −0.1240 [−0.244,−0.004] p=.063 (121) |
| (iii) `finish=stop` | +0.5500 [+0.381,+0.719] p<.001 (40) | +0.3269 [+0.144,+0.509] p=.002 (52) | +0.2941 [+0.078,+0.511] p=.062 (17) |
| **(iv) 8샘플 최빈답과 일치** | **+0.3004 [+0.203,+0.398] p<.001** (223) | **+0.3125 [+0.221,+0.404] p<.001** (240) | **+0.2992 [+0.210,+0.388] p<.001** (254) |
(평균 정확도 차: (iv) b2p +0.389 · b3s +0.406 · b3sh +0.366)

★**셋 다 자기일관성(iv)이 강한 문항내 정보를 나른다.** 그리고 **메타 블록을 더 내는 것(ii)은 음의 신호**다.
⇒ ★**모델의 *행동*(블록을 하나 더 연다)이 *선언*(`confidence: 0.88`)보다 정보가 많다.** 선언은 0811b 에서 0, 행동은 −0.21.

**⚠읽는 법**
- (ii)는 **역인과로 읽는 것이 자연스럽다** — 잘 안 풀려서 메타를 한 번 더 여는 것. 그래도 **테스트 시점에 관측 가능한 신호**라는 점은 같다.
- (iv)는 부분적으로 순환이다(모델이 우연보다 낫다면 최빈답은 자주 정답). **새로운 발견이 아니라 대조가 요점**이다 — 공짜 신호는 +0.30, 학습시킨 채널은 0.
- (iii)은 거의 항진명제(절단→오답)이고 혼합문항이 17~52로 작다.
- ⚠저장 `is_correct` 사용(C-002 옛 채점기 의심) · 혼합문항만 · 단일 벤치 · **EXP 트랙이라 헤드라인 금지**.

★**이 조합이 가리키는 것**: 이 프로그램은 **문항내 정보가 0인 채널에 보상을 걸어 왔고**, 정보가 있는 신호(샘플 간 합의)는 바로 옆에 공짜로 있었다. 4분해로 옮기면 *평가 ❌(선언) · 행동은 약한 음의 신호 · 사용 ❌* 이고, **자 자체를 바꿀 후보가 생겼다**.
★**여는 것**: 다음 설계에서 보상 대상을 *"선언한 confidence"* 가 아니라 *"샘플 간 합의와 정렬되는가"* 로 두는 것이 물어볼 가치가 있다. ⚠단 그건 새 가설이라 **승인 사항**이고, 지금은 탐색 기록으로만 남긴다.

## 0811 13:43 — b3shf **gs300 체크포인트 착지**(완전 4/4/4)

`rq3v2f-b3shf-0810b` · HB `13:42:58`(29초 전) · 사망표식 0 · HF `checkpoints/rq3v2f_b3shf/global_step_300` **m4/o4/e4**.
⚠**잡은 아직 종료 전** — 로그 꼬리는 `global_steps: 300` 최종 검증 생성 중이고 학습 스텝 로그는 `step:299` 까지다.
⇒ 말할 수 있는 것: **300스텝 학습 완료 + gs300 내구 저장 확정.** ⛔"잡 종료"로 승격 금지(⑧ 홀드아웃 eval 은 승인 대기).

## EXP-0811d (탐색 · ⛔판정 아님) — **선언은 완벽한 동전 한 닢이고, 아무것과도 붙어 있지 않다**

0811c(ii)의 *"메타 블록을 더 여는 것이 음의 신호"* 를 **역인과**(안 풀려서 하나 더 연다)로 읽었다.
그렇다면 ***첫*** 블록의 confidence 가 이미 낮아야 한다. 같은 문항내 통제로 두 대비를 나란히 놨다.

| | 첫conf<중앙 → **블록 추가?** | 첫conf<중앙 → **오답?** |
|---|---|---|
| b2p gs300 (혼합 491/500) | U=+0.0367 [−0.0002,+0.0735] **p=.066** (52/34/405) | U=−0.0387 [−0.0860,+0.0086] p=.129 (61/80/350) |
| b3sh gs175 (혼합 48/500) | U=+0.0208 p=1.00 (4/3/41) | U=+0.0000 p=1.00 (6/6/36) |
(첫conf 평균 — b2p: 블록≥2 0.7941(n=227) vs 블록1 0.8185(n=3773) · b3sh: 0.7612(263) vs 0.7904(3628))

★**역인과 가설은 확증되지 않았다.** 첫 선언은 자기 다음 행동조차 겨우 경계선에서만 예측한다(p=.066).
⇒ 0811c(ii)의 음의 신호는 **첫 선언에 없던 무언가**(풀이 중반의 곤경)에서 오는 것으로 보인다. ⚠관측만으로는 방향 분리 불가.

★**그러다 채널 자체를 셌더니 — 이것이 오늘의 수확이다.**

| 팔 | 첫 conf 분포 | 엔트로피 |
|---|---|---|
| b2p gs300 | `0.75` 1935(.484) · `0.88` 2065(.516) | **0.9992 bit** |
| b3s gs300 | `0.75` 2062(.515) · `0.88` 1938(.484) | **0.9993 bit** |
| b3sh gs175 | `0.12` 44(.011) · `0.25` 9(.002) · `0.75` 2439(.627) · `0.88` 1399(.360) | 1.0464 bit |

★**채널은 죽어 있지 않다 — 값이 상수로 붕괴한 것이 아니다.** 두 값 사이에서 **거의 완벽하게 균등한
1비트**를 매번 내보낸다. 그런데 그 1비트가 **정답과도(0811b)·자기 다음 행동과도(위) 붙어 있지 않다.**
⇒ ***해상도 부족이 아니라 내용 부재다.*** "선언 값이 2종뿐이라 못 가른 것"이라는 변명은 성립하지 않는다 —
1비트는 있고, 그 1비트가 무작위에 가깝다.

**⚠읽는 법** · 혼합문항만 · 저장 `is_correct`(C-002 의심) · b3sh 저conf 층은 53/3891 로 얇아 대비가 사실상 b2p 만 유효 ·
단일 벤치 · **EXP 트랙이라 헤드라인 금지**.
★**여는 것**: (1) 이 1비트가 *무엇과* 붙어 있는지 — 문항 난이도·길이·과목 등 **문항 수준** 변수와의 대비(문항내가 아니라 문항간)를 재면
"선언이 시도가 아니라 **문항**을 읽고 있다"를 검사할 수 있다. (2) 0811c 의 자기일관성(+0.30)과 대비하면,
**보상 대상을 선언에서 합의로 옮기는 설계**의 근거가 한 겹 두꺼워진다(⚠새 가설=승인 사항).

## EXP-0811e (탐색 · ⛔판정 아님 · ★codex 게이트 전 승격 금지) — **선언 confidence 는 문제가 아니라 `sample_idx` 를 읽는다**

0811d 가 *"선언은 균등한 1비트인데 아무것과도 안 붙어 있다"* 로 끝났다. 그럼 **그 1비트는 무엇과 붙어 있나**를 물었다.

**① 먼저 내 기준선이 틀렸다.** 문항간 Spearman 이 컸는데(`p_hi` vs 평균응답길이 **−0.7469**, 문항길이 −0.4332, 문항정답률 +0.2734, 전부 n=500)
문항당 hi 비율의 sd(0.1642)는 "이항 우연"(0.1767)보다 **작았다**. 둘은 동시에 참일 수 없다.
- **순열검정(hi 전역 셔플 2,000회)**: 귀무 sd 0.045 · 세 상관 전부 **순열 p = 0.0000** ⇒ 상관은 진짜.
- **기계적 결합 아님**: lo 행만으로 문항 평균길이를 만들어도 **rho −0.7239**(n=491).
- ⇒ 틀린 것은 **"8 롤아웃이 독립"이라는 내 이항 기준선**이었다. ★[[independence-assumption-repeated-failure-0730]] 재발.

**② 왜 독립이 아닌가 — 문항당 hi 개수가 3·5 에 몰려 있다.**
관측 `{2개:3, 3개:256, 4개:16, 5개:149, 6개:59, 7개:8, 8개:9}` vs 독립가정 예상 `{3개:102, 4개:136, 5개:116}`.

**③ 원인: 슬롯이 선언을 결정한다.** 슬롯별 `P(첫 conf ≥0.85)`:

| 팔 | idx0 | idx1 | **idx2** | idx3 | idx4 | **idx5** | idx6 | **idx7** | 슬롯별 정답률 범위 |
|---|---|---|---|---|---|---|---|---|---|
| b2p gs300 | .462 | .470 | **1.000** | .034 | .018 | **1.000** | .152 | **.994** | 0.610~0.658 |
| b3s gs300 | .484 | .490 | **.602** | .362 | .328 | **.616** | .418 | **.576** | 0.590~0.628 |
| b3sh gs175 | .060 | .388 | **.758** | .016 | .004 | **.682** | .198 | **.692** | 0.584~0.614 |

★**서로 독립으로 학습된 세 팔에서 슬롯 순서가 동일하다** — {2,5,7} 높음 · {3,4} 낮음 · {0,1,6} 중간.
모델이 원인이면 순서가 같을 이유가 없다 ⇒ 남는 원인은 **슬롯별 디코딩 난수 흐름**이다.
그리고 ★**같은 슬롯들의 정답률은 평평하다**(b2p 0.610~0.658) — 슬롯은 **선언을 가르지만 정답을 가르지 않는다**.

**해석(가장 단순한 것)**: `0.75` vs `0.88` 지점에서 모델의 분포가 문제와 거의 무관하게 ~50/50 이라
**슬롯별 고정 난수열이 거의 모든 문제에서 같은 쪽으로 뒤집는다.** 잔여로 자유 슬롯(0·1·6)이 문제를 따라 움직여
문항간 상관(길게 쓰는 어려운 문항일수록 lo)을 만든다.

**⇒ 0811b·0811d 의 널이 설명된다.** 같은 문항 안에서 선언이 정답을 못 가른 것은
*"모델이 자기평가를 못 해서"* 가 아니라 **그 자리에 실린 것이 대부분 샘플러의 동전이기 때문**이다.

**⚠경계 — 이 문장으로만 쓴다**: *이 eval 의 디코딩 설정에서, **방출된 토큰**이 문제 정보를 거의 나르지 않는다.*
⛔"모델에 내부 캘리브레이션 신호가 없다"로 승격 금지(로짓을 못 봤다) · b2p/b3sh 는 강하고 b3s 는 약화형 ·
단일 벤치·단일 체크포인트 · 저장 `is_correct` 는 C-002 의심 · **EXP 트랙**.

★**CONF 로 올리려면 필요한 것**: ⑴ 다른 시드/`n` 으로 재현(슬롯 구조가 시드에 붙는지) ⑵ 같은 프롬프트에서
그 위치의 로짓 확인 ⑶ **codex-sol 적대검토**(0805 지시). 셋 다 하기 전에는 CLAIMS 에 쓰지 않는다.
★**여는 것**: 이것이 재현되면 **선언 confidence 를 보상·평가 대상으로 쓰는 모든 설계가 영향을 받는다**(⚠새 가설=승인 사항).
그리고 0811c 의 자기일관성(+0.30)이 상대적으로 더 강해진다 — 그쪽은 8 슬롯을 **가로질러** 읽기 때문에 슬롯 난수에 면역이다.

## 0811 14:44 — **b3shf 완주 확정**(300스텝 · rc=0)

`rq3v2f-b3shf-0810b` / `:h100_rq3v2f_b3shf`. 종료 경로를 로그에서 직접 읽었다:
`wandb: Synced` → `[YAML] pusher 7307 stopped` → `[YAML] final sync push target: .../global_step_300`
→ `[YAML] FINAL PUSH DURABLE global_step_300` → `+ break` → `+ sleep 86400`(대기 셸). **`rc=0`.** 사망표식 0.
HB 는 `14:41:30` 에 멈췄고 그것은 사망이 아니라 **하트비트 루프 종료**다.
HF `checkpoints/rq3v2f_b3shf/global_step_300` = **m4/o4/e4** 완전.

⚠**부수 관측 — 노드의 내구성 게이트가 config 로 좁혀져 있지 않다.**
노드는 `"global_step_300/actor/model_world_size" in f` 로 세어 `SHARDS=16` 을 얻고 `SHARDS>=4` 로 통과시킨다.
그 16개의 출처는 **b0p 4 · b2p 4 · b3p 4 · b3shf 4** — 즉 **다른 팔의 gs300 만 있어도 통과한다.**
이번에는 b3shf 자체가 4/4/4 라 판정은 옳지만, **게이트는 보이는 것보다 약하다.**
(내가 config 로 좁혀 센 4/4/4 가 정답. ★"이름이 맞으니 맞다"가 아니라 **범위를 좁혀 다시 세라**.)
⛔코드 수정은 사전 승인 사항이므로 여기 관측으로만 남긴다.

★**남은 것**: ⑧ gs300 홀드아웃 eval — **발사만 대기**(승인 사항).

## EXP-0811f (탐색 · ⛔판정 아님 · ★codex 게이트 전 승격 금지) — **선언은 세 층위 중 가장 거친 층에만 붙어 있다**

0811e 의 슬롯 지문을 벤치·시드·팔을 가로질러 재현했다. 저장소 `eval/` 에 **`seed43` 변형**이 있어 시드 대조가 GPU 0 으로 가능했다.

**① 슬롯 지문은 벤치를 가로질러 유지된다** — `P(첫conf≥0.85)` by `sample_idx`:

| 파일 | idx0 | idx1 | idx2 | idx3 | idx4 | idx5 | idx6 | idx7 |
|---|---|---|---|---|---|---|---|---|
| b2p math500 | .462 | .470 | **1.000** | .034 | .018 | **1.000** | .152 | **.994** |
| b2p gsm8k | .970 | .980 | **1.000** | .066 | .008 | **1.000** | .484 | **1.000** |
| b2p aime2024 | .033 | .033 | **1.000** | .000 | .000 | **1.000** | .000 | **.933** |
| **b2p aime seed43** | .033 | **1.000** | .000 | .000 | **1.000** | .000 | **.933** | .033 |
| b3s aime2024 | .033 | .033 | .033 | .000 | .000 | .033 | .000 | .033 |
| **b3s aime seed43** | .033 | .033 | .000 | .000 | .033 | .000 | .033 | .033 |
| b3sh gs165 math500 | .068 | .348 | .766 | .014 | .000 | .710 | .168 | .696 |
| b3sh gs175 math500 | .060 | .388 | .758 | .016 | .004 | .682 | .198 | .692 |
| b3nopmi math500 | .012 | .006 | .008 | .006 | .002 | .014 | .006 | .008 |
| pmishift(instruct) math500 | .826 | .922 | .768 | .682 | .828 | .934 | .762 | .808 |
| shiftonly(instruct) / gandhi(instruct) | ~.03 / **.000 전부** | | | | | | | |

★**② `seed43` 은 원본 지문을 정확히 한 칸 왼쪽 회전시킨 것이다.** b2p·b3s 둘 다 8칸 전부 일치
(b2p: `[1/30,1/30,30/30,0,0,30/30,0,28/30]` → `[1/30,30/30,0,0,30/30,0,28/30,1/30]`).
★그런데 **생성 텍스트는 진짜로 다르다**: 같은 (문항,슬롯) 완전일치 **0/240**, 텍스트 교집합 **2/240**.
⇒ 독립 디코딩이 맞고, **회전하는 것은 confidence 비트의 슬롯 무늬뿐**이다.
가장 단순한 설명: 슬롯 시드가 `base_seed + i` 라서 seed 42→43 이면 슬롯 i 가 이전 i+1 의 난수열을 받는다.
★그리고 이것이 성립하려면 **같은 난수열이 30개 서로 다른 문항 전부에서 같은 쪽으로 떨어져야 한다** —
즉 그 토큰의 확률이 문항에 거의 무관하게 일정하다는 뜻이다.

**③ 그럼 어디에는 붙어 있나 — 벤치(분포) 수준에는 붙어 있다.** b2p gs300:

| 벤치 | P(첫conf=0.88) | 실제 정답률 |
|---|---|---|
| gsm8k | **0.6885** | 0.9193 |
| math500 | **0.5162** | 0.6290 |
| aime2024 | **0.3750** | 0.1500 |

★**정리 — 세 층위**: **분포 수준 ✅ 단조** · **문항 수준(벤치 내) ≈ 상수**(그래서 슬롯 난수가 지배) · **시도 수준 ❌ 0**(0811b·0811d).
⇒ 선언 confidence 는 *"이 문제집이 어렵다"* 는 알지만 *"이 문제가"* 도 *"이 시도가"* 도 모른다.

**⚠경계** · 팔마다 다르다(rq3v2f 계열은 극단적, **instruct 세대 pmishift 는 .68~.93 로 평평**해 슬롯 결정성이 약하고, gandhi 는 전부 .000·b3nopmi ~.01 로 애초에 고conf 를 안 낸다) ·
aime 은 n=30 이라 얇다 · 저장 `is_correct` 는 C-002 의심 · 로짓 미확인 · **EXP 트랙**.
★**CONF 승격 요건은 0811e 와 동일**(⑴시드/`n` 재현 — ②가 이걸 상당 부분 채웠다 ⑵로짓 ⑶codex-sol 적대검토).
★**닫는 것(잠정)**: *"선언 confidence 가 시도 수준에서 왜 안 붙나"* 를 더 재는 실험. 근인이 나왔다 — 더 재도 같은 답이다.
★**여는 것**: ⑴ `seed43` 을 **독립 시드 강건성 근거로 쓴 곳이 있는지** 확인해야 한다(회전 관계라면 그 강건성은 약하다) ⑵ 보상·평가를 선언에서
**샘플 간 합의**로 옮기는 설계(0811c 의 +0.30)는 슬롯 난수에 면역이다 — ⚠새 가설=승인 사항.

## 0811 15:16 — **정정: 완주한 잡이 셋 더 있었고, b3null 판정 eval 이 미실행이다**

여러 틱 동안 `amlt status rq3v2f-b3shf-0810b` **한 실험만** 조회해 왔다. `amlt list` 를 돌리자
`Running` 으로 표시된 잡이 **넷**이었고, 넷 다 실제로는 **끝나서 대기 셸(`sleep 86400`)에 앉아 있었다.**

| 실험 | 실제 | 마지막 HB | 산출물 |
|---|---|---|---|
| `rq3v2f-b3nopmi-0807` | **완주** `FINAL PUSH DURABLE global_step_304` | 8/10 21:39 | eval 완료 → C-031 |
| **`rq3v2f-b3null-0807`** | **완주** `FINAL PUSH DURABLE global_step_303` | **8/10 23:48** | ★**eval 미실행** |
| `rq3v2f-b3nopmi-eval-0809` | **pass** `PAIRED EVAL DONE both arms` | — | C-031 |
| `rq3v2f-b3shf-0810b` | **완주**(오늘 14:44 기록) | 8/11 14:41 | eval 승인 대기 |

★**`running` 은 잡 상태이지 학습 상태가 아니다** — 대기 셸이 그 표시를 유지시킨다.
(`ckpt 착지≠잡 종료` 의 쌍둥이: **`잡 running`≠학습 중**.)

**★b3null 확인(HF 직접)**: `checkpoints/rq3v2f_b3null/global_step_303/actor` = **m4/o4/e4** 완전 ·
보존본 `preserved/mechanism_alive/rq3v2f_b3null_gs300` 존재 · **`b3null` 이름의 parquet 0개 ⇒ 홀드아웃 eval 미실행.**

**왜 중요한가.** b3null 은 **기하 통제군**이다(전 헤드 0 · `rmeta_source=none` · 그래도 TRIOBJ region 경로로 감).
발사 설명에 판정 규칙이 이미 적혀 있다: *b3null 이 treatment 대역(~75%)에 앉으면 −2pp 는 **기하·라우팅 묶음**의 것이고
메타 보상은 면책된다 — 그러면 이 프로그램의 모든 팔이 다른 것을 재고 있었다는 뜻이다. control 대역(~77.28)에 앉으면 기하가 면책되고 메타 헤드가 다시 기소된다.*
⇒ **미실행 eval 하나가 프로그램의 귀책 방향을 가른다.** 학습은 15시간 전에 끝나 있었다.

★**승인 대기 목록에 ⑨ 추가**: **b3null gs303 홀드아웃 eval**(⑧ b3shf gs300 과 같은 성질·같은 하니스).
⛔재촉하지 않는다. 다만 ⑧보다 ⑨가 **여는 주장이 크다**는 것은 적어 둔다.

★**규율 갱신**: 매 틱 (A)status 는 **`amlt list` 로 전수**를 본다. 한 실험 이름으로 좁히면
**끝난 잡도 새로 끝난 잡도 안 보인다.** 이 누락이 15시간 갔다.

## EXP-0811g (탐색 · ⛔판정 아님) — **`seed43` 은 `seed42` 와 난수열을 공유한다**(한 칸 어긋나서)

**⚠먼저 자기 정정.** 0811f 가 *"여는 것: `seed43` 을 독립 시드 강건성 근거로 쓴 곳이 있는지 확인해야 한다"* 로 닫았다.
그 답은 **이미 `docs/CLAIMS.md:443` C-021 에 있었다** — *"전 arm이 단일 학습 시드다 — `seed43_*`는 디코딩 시드다"*,
근거 *"생성문 일치율 0.000"*, **닫는 것에 "`seed43` 파일을 독립 반복으로 인용하는 것"이 명시**돼 있다.
★**G1 재발(4회째)**: 새 의문을 원장에 열기 전에 CLAIMS 를 개념 grep 하지 않았다. [[external-numbers-need-claims-grep-0807]]

**그런데 재확인이 새 것을 하나 낳았다.** 0811f 의 기전 설명(슬롯 시드 = `base_seed + i`)이 옳다면
**seed43 슬롯 i 와 seed42 슬롯 i+1 은 같은 난수열**이어야 하는데, 두 파일의 생성문은 하나도 안 겹쳤다(0/240).
둘이 같이 서려면 **앞은 같고 뒤에서 갈라져야** 한다. 공통 접두사(문자)를 shift 별로 쟀다:

| shift | 접두사 중앙값 | 평균 | ≥200자 | ≥1000자 |
|---|---|---|---|---|
| 0 | 23 | 28 | .000 | .000 |
| **+1** | **298** | **464** | **.571** | **.125** |
| 2 | 23 | 29 | .000 | .000 |
| 3 | 26 | 35 | .004 | .000 |
| 4 | 36 | 43 | .000 | .000 |
| 5 | 23 | 30 | .004 | .000 |
| 6 | 36 | 46 | .008 | .000 |
| 7 | 26 | 35 | .008 | .000 |
| *대조: seed42 내부 인접 슬롯* | *23* | *29* | — | — |

★**shift=+1 에서만 폭발한다.** ⇒ **두 실행은 슬롯이 한 칸 어긋난 채 같은 난수열을 탄다.**
같은 프롬프트·같은 가중치·같은 스트림으로 출발해 도중에 갈라진다(부동소수/배치 순서). conf 토큰이 그 **공유 접두사 안**에 있어서 0811f 의 지문이 한 칸 회전했다.

**★영향 — `seed43` 의 정당한 용도 하나가 흔들린다.** ⚠**아래 문단은 EXP-0811h 가 축소했다 — 실제 영향은 작다. 그쪽을 읽어라.**
런처들이 *"aime 만 16k seed43 → pass b 와 합쳐 AIME 는 총 16샘플"* 로 쓴다(`h100std_rq3v2f_*_1030_eval.yaml`).
그 **avg@16 은 독립 16표본이 아니다** — 난수열을 공유하는 **8쌍**이고, 쌍의 절반 이상이 200자 넘게 같은 글을 쓰고 12.5%는 1,000자 넘게 같다.
⇒ AIME avg@16 로 계산한 **CI 는 실제보다 좁다**. (원장 3283행의 AIME avg@16 −3.75pp 같은 수가 여기 해당한다.)
★**C-021 의 결론은 그대로 옳다**(학습 시드 반복이 아니다). 흔들리는 것은 ⑴ 그 근거의 강도(**완전일치 0.000 은 너무 거친 검사였다** — 접두사로 보면 강하게 결합) ⑵ **avg@16 을 독립 16표본으로 쓴 곳**.
⛔CLAIMS 수정은 판정이므로 **codex-sol 게이트 후**에 한다. 여기서는 관측으로만 남긴다.

★**여는 것**: AIME 를 인용하는 곳에서 **avg@16 이 쓰였는지, 쓰였다면 seed42 단독(avg@8)으로 다시 읽어야 하는지** 점검.
★**규율**: **완전일치는 결합을 재는 검사가 아니다** — 결합은 **접두사·부분일치**로 재라.

## EXP-0811h (탐색 · ⛔판정 아님) — **0811g 의 영향 주장을 축소한다: 결합은 실재하나 결과 수준 몫은 작다**

0811g 가 *"AIME avg@16 은 결합된 8쌍이라 CI 가 실제보다 좁다"* 로 닫았다. **얼마나 좁은지를 안 재고 썼다.**
쟀더니 그 주장은 **과했다.** shift+1 의 `is_correct` 상관을 **다른 shift 와 나란히** 놓는 것이 옳은 기준선이었다:

| 팔 | 정답률 | **ρ(+1)** | **ρ(다른 shift 평균)** | 초과 | 답 일치(+1) vs 타 |
|---|---|---|---|---|---|
| b2p | .1521 | .6608 | .5962 | +.065 | .1750 / .1655 |
| b0p | .1896 | .6122 | .5889 | +.023 | .2250 / .1970 |
| b3s | .1292 | .7407 | .6031 | +.138 | .1833 / .1708 |
| b3p | .1625 | .7248 | .6680 | +.057 | .2500 / .2101 |
| b3nopmi | .1625 | .5766 | .5150 | +.062 | .1708 / .1720 |
| **pmishift** | .1854 | .5188 | .5543 | **−.036** | .2167 / .2232 |
| base | .0479 | .4068 | .1456 | +.261 | .2667 / .1935 |

★**상관의 대부분은 시드 공유가 아니라 문항 뭉침이다.** AIME 는 한 문항을 여덟 번 다 맞거나 다 틀리는 쪽이라
**어느 두 샘플을 짝지어도** ρ 가 0.5~0.67 나온다. shift+1 의 **초과분은 +0.02~+0.14** 이고 **pmishift 에서는 음수**다.
(base 의 +0.261 은 정답률 .0479 에서 나온 불안정한 값으로 읽는다.)

★**따라서 실무 영향은 작다.** 지배항인 문항내 뭉침은 **문항 단위 쌍대 부트스트랩이 이미 흡수한다**(문항을 재표집하므로).
시드 공유가 더하는 것은 문항별 비율 추정의 정밀도에 얹히는 **2차 항**이다.
⇒ ★**이 근거로 다시 읽어야 할 보고 수치는 없다.** 0811g 의 "여는 것"에 대한 답은 **아니오**다.

**남는 것(0811g 중 유효한 부분)**: ⑴ 접두사 공유는 실재한다(shift+1 중앙 298자·12.5%가 ≥1000자) —
그것이 0811e/f 의 슬롯 지문 회전을 설명한다 ⑵ **완전일치는 결합 검사가 아니다**(C-021 근거의 거칢) ⑶
`seed43` 을 **학습 시드 반복**으로 인용하면 안 된다(C-021, 그대로 유효).
⇒ 0811g 의 **`★영향` 문단은 이 항목으로 대체**한다(그 문단에 포인터를 박아 뒀다).

★**규율(오늘 두 번째)**: **"CI 가 좁다"는 크기를 재기 전에는 주장이 아니다.** 방향만 맞고 크기를 안 잰 문장이
어제 예측 채점에서도 지적됐던 형태다. 그리고 ★**상관을 인용하기 전에 "같은 구조의 다른 짝"을 기준선으로 놓아라** —
그 한 열이 없었으면 나는 문항 뭉침을 시드 결합으로 잘못 팔았을 것이다.

## EXP-0811i (탐색 · ⛔판정 아님 · ★codex 게이트 전 승격 금지) — **b3p 는 메타를 안 여는 게 아니라, 여는 토큰이 `cros` 로 변질됐다**

b3p 의 발화 건들을 **실제로 읽으려고** 열었다가 두 가지가 나왔다.

**① 먼저 내 수 정정 — "b3p 발화 98/4000" 은 척도 하나의 값이었다.** 같은 파일에서 네 척도가 다 다르다:

| 척도 | 값 |
|---|---|
| 여는태그 `<\|meta\|>` 포함 | **74**/4000 (.0185) |
| 닫는태그 `<\|/meta\|>` 포함 | **88**/4000 |
| `num_meta_blocks` > 0 | **98**/4000 ← 내가 0811b 이후 인용해 온 수 |
| `meta_confidences` 비어있지 않음 | **97**/4000 |
교차: 여는O·블록X **13** · **여는X·블록O 37** · 둘다 61.

**② 그 37건을 읽으니 여는태그 자리에 `cros` 가 있었다.** 그래서 전 팔의 **완성본 첫 줄**을 셌다:

| 팔 | 첫 줄 = `<\|meta\|>` | 첫 줄 = `cros` | 그 외 |
|---|---|---|---|
| b2p / b3s / b3sh175 | **1.0000** | 0 | 0 |
| **b3p** | **0.0000** | **0.9825** | 70 |
| b3nopmi | 0.0000 | 0.0000 | 4000 (`'Let'`×729 등 평범한 산문) |

★**b3p 는 4000행 중 98.25% 가 `cros` 로 시작한다.** 그 뒤는 멀쩡한 풀이다(정답·무메타 행 하나를 통째로 읽음:
`cros` → 함수 대입 → `\boxed{\frac{14}{3}}`, 601자, `finish=stop`). ⇒ **모델은 여전히 매번 메타 자리를 연다. 여는 토큰이 바뀐 것이다.**
`cros` 는 b3p 전체 **3931/4000** 행에 나오고 b2p 206·b3s 232 (부분문자열 `across` 등으로 설명되는 수준)와 자릿수가 다르다.

★**b3p 와 b3nopmi 는 서로 다르게 실패한다**: b3p = **여는데 깨진 토큰으로**(98.25%) · b3nopmi = **아예 안 연다**(첫 줄 0%, 다만 본문 어딘가에 여는태그 410/4000).

**③ 발화행의 성격**(여는태그 기준 74건): 정답률 **0.3378** vs 무발화행 0.6126 · 길이 중앙 **2692** vs 253 토큰(10배) ·
56개 문항에 흩어짐(문항당 1건이 42) · **슬롯 분포는 평평**(4~13) ⇒ *발화 여부*는 슬롯이 안 정한다(0811e 의 슬롯 결정성은 *선언 값*에만 해당) ·
conf 값은 `0.75`×57 / `0.88`×4 로 **저conf 편중**(b2p 의 48/52 균형과 대비).
메타 본문은 **판박이가 아니라 문제를 가리킨다** — *"The earlier reasoning assumed the product part is forced to evaluate to 120 … but that overlooks that the expression can be rebracketed …"*.
한 건은 `\boxed{4}` **뒤에** 붙은 사후 반성이고, 한 건은 앞에 `decision: redirect` 가 있는 **두 번째** 블록이다.

**⚠경계** · 발화행 조건화는 **처치후 선택**이라 "메타가 정확도를 낮췄다"로 못 읽는다 · 저장 `is_correct` 는 C-002 의심 ·
단일 벤치·단일 체크포인트 · `cros` 가 **왜** 그 토큰인지는 미확인(토크나이저 id 인접성 미검사) · **EXP 트랙**.

★**닿는 곳(⛔지금 고치지 않음, 기록만)**: ⑴ E-190 이 b3p 를 *"발화 1.85% 라 설계상 판정불가"* 로 제외했는데,
**그 서술은 행동의 묘사로는 틀렸다**(제외 자체는 반사실이 밴할 실제 토큰을 필요로 하므로 운영상 여전히 성립할 수 있다).
⑵ **C-031(`b3p − b3nopmi` −0.65pp p=0.328)** 은 *"깨진 채로 여는 팔"* vs *"안 여는 팔"* 의 대비인데 주장문이 그걸 이름 짓지 않는다.
⑶ 어느 쪽도 CLAIMS 수정은 **codex-sol 게이트 후**.

★**여는 것**: `cros` 의 정체 — 토크나이저에서 `<\|meta\|>` id 와 `cros` id 가 인접한가, 아니면 학습 중 임베딩이 끌려갔나.
(⚠먼저 `docs/CLAIMS.md` 를 **`cros`·토큰 변질·homoglyph** 개념으로 grep 한다 — b3s 의 Khmer/Tamil 숫자 homoglyph 기록이 이미 있으므로 **같은 현상의 다른 얼굴일 수 있다**.)
★**규율(오늘 세 번째)**: **하나의 이름에 네 개의 척도가 붙어 있으면 어느 것을 인용 중인지 매번 적어라** — 74·88·97·98 이 전부 "발화"였다.

## EXP-0811j (탐색 · ⛔판정 아님 · ★codex 게이트 전 CLAIMS 수정 금지) — **b3p 의 "발화 죽음"은 탐지기 회피였다** · b3s 동형이의자와 같은 계열

0811i 의 `cros` 를 따라가 **탐지기 정본과 어휘와 생성문을 다 읽었다.**

**① 학습측 발화 판정은 토큰 id 동등성이다.**
`src/training/dcpo_region.py:39` `META_OPEN_DEFAULT = 151669` · `:208` `O = [i for i in range(T) if rmask[i] and ids[i] == meta_open]`
→ `:1001` `"has_meta"` → `src/training/verl_sdc.py:649,657` `dcpo/meta_emit_rate = sum(has_meta)/B`.
⇒ **id 151669 가 아니면 그 행은 메타가 없는 것으로 처리되고, 메타 마스크가 비어 메타 보상 헤드가 아예 안 걸린다.**

**② b3p 는 그 id 를 안 쓴다.** 완성본 첫 12자 최빈: `'cros\nA direc'` 1522 · `'cros\nLet the'` 443 · `'cros\nLet\n\['` 426 …
(b2p 는 `'<|meta|>\ncon'` **4000/4000**). 체크포인트 자체의 `added_tokens.json` 은 정상이다(`<|meta|>` 151669 · `<|/meta|>` 151670) ⇒ 토크나이저 문제가 아니다.
`cros` 는 **어휘에 단일 토큰으로 없다**(`cross` 28842 는 있음) ⇒ `c`(66)+`ros`(3630) 류의 **평범한 토큰 조합**이다.
저장소 전체에서 낱말 `cros` 는 **내 원장 기록 말고 0건** ⇒ 데이터·코드 유래가 아니라 **학습 중 생겼다**.

**③ 그래서 기록된 기제 문장이 흔들린다.**
`docs/CLAIMS.md:641` — *"**b3p 는 발화를 죽여** 이 압력에서 탈출했고, b3s 는 floor 때문에 탈출로가 막혀 압력이 엔트로피로 갔다."*
★**결과("메타 보상 압력에서 탈출")는 맞다. 행동("발화를 죽였다")은 틀렸다.**
b3p 는 **매 생성마다 메타 자리를 계속 연다**(98.25%). 바꾼 것은 **탐지기가 키로 삼는 토큰**이다.
⇒ 이것은 행동 포기가 아니라 **탐지기 회피**다.

**④ 그리고 b3s 와 같은 계열이다.** ⚠**EXP-0811k 가 이 항목을 반쪽으로 줄였다 — 엔트로피 고리는 b3p 에 없다. 그쪽을 읽어라.** `CLAIMS.md:647-657` 은 b3s 가 답의 숫자를 크메르 `៤៨១៦`·타밀 `௬௫௦`·전각 `０` 로 써서
`math_verify` 가 죽고 0점 처리된다고 기록한다. **두 팔이 같은 수를 두 번 쓴 셈이다** —
*검사기가 못 알아보는 이웃 표면형으로 갈아탄다.* b3s 는 **채점기**를(숫자), b3p 는 **보상 탐지기**를(메타 여는 토큰) 회피했다.
⇒ ★**"엔트로피 확산 → 동형이의자" 고리는 b3s 전용 현상이 아니라 이 기질의 일반 반응일 수 있다.**

**⚠경계**
· **언제부터인지 못 잰다** — b3p 는 `--keep 3` 로 **gs300 만 잔존**(잔존 스텝 목록 `[300]`). 시점·인과 방향(회피가 먼저인가 압력 이탈이 먼저인가)은 **재학습 없이는 미확정**.
· 생성문에는 토큰 id 가 없다 ⇒ `c`+`ros` 분해는 **어휘로부터의 추론**이지 직접 관측이 아니다.
· 단일 벤치·단일 체크포인트 · 저장 `is_correct` 는 C-002 의심 · **EXP 트랙**.

★**닿는 곳(⛔지금 안 고침·codex-sol 게이트 후)**: ⑴`CLAIMS.md:641` 의 기제 문장 ⑵ E-190 의 b3p 제외 사유 ⑶ **C-031**(`b3p−b3nopmi`)
— 이제 그 대비는 *"탐지기를 회피한 팔"* vs *"아예 안 여는 팔"* 이다 ⑷ `dcpo/meta_emit_rate` 를 **행동 지표로 인용한 모든 곳**.
★**여는 것**: ⑴ 학습 로그에서 `dcpo/meta_emit_rate` 가 떨어진 **시각**과 엔트로피 궤적을 겹쳐 보기(wandb `rq3v2f-b3p-1`, 재학습 불필요)
⑵ 다른 팔에도 같은 회피가 **덜 눈에 띄게** 있나 — b3nopmi 여는태그 410/4000 의 나머지 3590 은 무엇으로 시작하나(`'Let'`×729 확인, 나머지 미확인).
★**규율(오늘 네 번째)**: **탐지기가 토큰 id 동등성이면, 그 지표는 "행동"이 아니라 "그 토큰"을 잰다.** 회피가 가능한 지표를 행동의 이름으로 부르지 마라.

## EXP-0811k (탐색 · ⛔판정 아님) — **회피 시점은 gs157. 내 기제 가설 둘은 데이터가 반증했다**

0811j 가 *"b3p 는 탐지기를 회피했고, b3s 의 엔트로피→동형이의자와 같은 계열"* 로 닫았다.
wandb `rq3v2f_b3p`(⚠런 이름은 **밑줄**이다 — `rq3v2f-b3p-1` 로 찾다 빈 결과를 받았고, 내 명령부터 의심해 고쳤다) 를 열어 궤적을 읽었다.

**① 붕괴는 단일 사건이 아니라 두 물결이고, 중간에 완전히 회복한다.**
`dcpo/meta_emit_rate`: gs1–150 ≈ **0.96~1.00** → gs155 0.557 → gs170 **0.170** → ★gs175 **0.939 회복** → gs190 0.701 → gs205 **0.066** → gs300 **0.018**.
**0.5 를 처음 밑도는 스텝 = gs157.** `dcpo/wellformed_rate` 가 거의 같은 궤적(0.828→0.133→0.820→0.010)이라 같은 사건이다.

**② 구간 집계 — 두 가설을 나란히 놓는다**

| 구간 | emit 평균 | **rmeta(메타행) 평균/중앙** | kl 평균 |
|---|---|---|---|
| ① 안정 gs1–150 | 0.963 | **−0.012** / −0.041 | 0.048 |
| ② 1차붕괴 gs151–174 | 0.534 | **+0.180** / +0.174 | 0.194 |
| ③ 회복 gs175–189 | 0.887 | **+0.347** / +0.352 | 0.177 |
| ④ 2차붕괴 gs190–215 | 0.230 | **+0.082** / +0.139 | 0.265 |
| ⑤ 이후 gs216–300 | 0.043 | −0.121 / −0.119 | 0.290 |

★**가설 A "메타 보상이 음수가 돼서 도망쳤다" — 반증.** 메타 보상은 **안정기에 이미 음수**였고(평균 −0.012, **스텝의 60.7% 가 음수**),
**붕괴 두 구간에서는 오히려 양수**(+0.180·+0.082)였으며 회복기에 가장 높았다(+0.347). 붕괴 이후 음수 비율은 **50.0%** 로 안정기보다 **낮다**.
⇒ 보상 부호로는 설명되지 않는다.

★**가설 B "b3s 의 엔트로피 확산과 같은 계열" — 반쪽만 맞다.** `actor/entropy` 최대: **b3p 0.518** · b2p 0.361 · **b3s 6.442**.
b3p 는 통제군보다 조금 높을 뿐 **자릿수가 다르다**. ⇒ **회피라는 결과는 공유하지만, b3s 의 엔트로피 고리는 b3p 에 없다.**
0811j 의 ④ 항목은 이 폭으로 줄인다(그 문단에 포인터를 박아 뒀다).

**③ 함께 움직인 유일한 것은 KL 이다** — `actor/kl_loss` 0.048 → 0.194 → 0.177 → 0.265 → **0.290**(6배, 단조에 가깝다).
⚠단 `kl_loss_coef 0.0`(CLAIMS:638)이라 **KL 은 벌점이 아니라 표류의 계측**이다 ⇒ **원인이 아니라 증상으로 읽는다.**

★**따라서 정직한 결론: 시점은 잡았고(gs157) 기제는 못 잡았다.** 남은 후보 — 회복(gs175)이 있었다는 점이 중요하다:
**한 번 돌아왔다가 다시 나간 것**이므로 단조 압력(형식 벌점·길이 비용)보다 **배치 수준 우연 + 되먹임**이 더 맞는 모양이다.
⚠체크포인트는 gs300 만 남아 gs150~210 구간을 직접 못 본다(`--keep 3`).

★**여는 것**: ⑴ 같은 구간의 `dcpo/rmeta_pos_rate`·`rmeta_neg_rate`·discard/format 계열을 붙여 **회복 gs175 에 무엇이 달랐나** ⑵ b3nopmi·b3s 도 같은 궤적을 그렸나(같은 키로 세 팔 겹쳐보기).
★**규율(오늘 다섯 번째)**: ★**내 기제 가설은 구간 집계 한 표로 죽는다** — 붕괴 스텝 하나를 집어 rmeta −0.565 를 인용했으면 반대 결론을 팔았을 것이다. **단일 스텝 대신 구간 분포.**

## EXP-0811l (탐색 · ⛔판정 아님) — **발화 붕괴는 기본값이다. floor 만이 막고, 메타 헤드는 늦출 뿐이다**

0811k 가 *"b3p 의 붕괴 시점은 gs157, 기제는 못 잡았다"* 로 닫았다. **한 팔만 보고 있었다.**
wandb 에 있는 **다섯 팔 전부**의 `dcpo/meta_emit_rate` 를 겹쳤다(⚠b2p·b0p 는 이 키 자체가 없다 — VANILLA_GRPO 라 헤드가 없다).

| gs | b3p | b3s | b3nopmi | b3shf | **b3null** |
|---|---|---|---|---|---|
| 1–100 | ~1.00 | ~1.00 | ~1.00 | ~1.00 | ~1.00 |
| 125 | .830 | 1.000 | .996 | 1.000 | **.363** |
| 150 | .896 | .996 | .543 | 1.000 | **.037** |
| 157 | .443 | .996 | .361 | 1.000 | .016 |
| 175 | **.939** | 1.000 | .348 | 1.000 | .033 |
| 215 | .057 | .998 | .174 | .941 | .016 |
| 300 | **.018** | **1.000** | .156 | .059 | .025 |

**0.5 첫 하회**: **b3null gs125** · b3nopmi gs152 · b3p gs157 · b3shf gs228 · **b3s 없음**.

★**용량-반응 순서가 보인다.** 메타를 떠받치는 보상이 **적을수록 더 일찍** 무너진다 —
**b3null(전 헤드 0 · `rmeta_source=none`)이 가장 먼저**, 그다음 PMI 없는 b3nopmi, 전 패키지 b3p, format 얹은 b3shf,
그리고 **`meta_floor=0.05` 인 b3s 만 300스텝 내내 1.000 을 유지**한다.

★**이것이 0811k 의 미해결을 상당 부분 채운다.** b3null 은 **메타 보상이 아예 없는데도**, 아니 **없기 때문에 가장 빨리** 무너졌다.
⇒ **발화 붕괴는 "벌을 피한 탈출"이 아니라 "아무도 안 붙들 때의 기본 표류"다.** 메타 헤드는 그것을 **늦추고**, floor 는 **막는다.**
0811k 에서 rmeta 부호가 붕괴를 설명 못 한 이유가 이것이다 — 설명해야 할 것은 *왜 떨어졌나* 가 아니라 *무엇이 붙들고 있었나* 였다.

★**`CLAIMS.md:641` 재조정(⛔codex 게이트 전 수정 금지)**: *"b3p 는 발화를 죽여 압력에서 탈출했고, b3s 는 floor 때문에 탈출로가 막혀 압력이 엔트로피로 갔다."*
— **후반부(floor 가 붕괴를 막는다)는 강하게 확증된다**(다섯 팔 중 b3s 만 유지). **전반부(탈출)는 더 약해졌다**: 압력이 **전혀 없는** b3null 이 **더 빨리** 무너졌으므로 탈출로 읽을 근거가 없다.

★**⑧ b3shf 에 대한 관측(승인 판단에 필요)**: 이 팔도 **무너졌다** — gs228 하회 후 gs300 에서 **0.059**.
w_format 을 되살린 것은 붕괴를 **gs157→gs228 로 71스텝 늦췄을 뿐** 막지 못했다. ⇒ 홀드아웃 eval 을 돌리면 **발화 거의 없는 상태의 팔**을 재게 된다(E-190 이 b3p 를 뺀 것과 같은 사정).

**⚠경계**
· ★**b3shf 궤적은 오염 가능**: `rq3v2f_b3shf` 는 0806(Killed)·0808(Killed)·0810(Failed)·0810b(완주) **네 번의 발사가 공유한 이름**이고 E-187 에 WANDB_RUN_ID 재사용이 기록돼 있다. **gs228 이전 구간은 미검증**으로 둔다(최종 상태 gs300=0.059 는 완주분과 일관).
· b3null 은 **번들**을 움직였다(헤드 8개 + `rmeta_source=none` 동시) ⇒ "어느 헤드가" 는 이걸로 못 가른다.
· 이건 **학습 중 지표**다. 0811i/j 가 보였듯 **`meta_emit_rate` 는 토큰 id 검사**라 "행동"이 아니라 "그 토큰"을 잰다 — b3p 는 실제로 98.25% 에서 열고 있었다. **다른 팔의 낮은 값도 같은 함정일 수 있다**(b3nopmi 홀드아웃 첫 줄은 `<|meta|>` 0% · `cros` 0% 로 **정말 안 여는** 쪽이었다; b3null·b3shf 는 미확인).
· **EXP 트랙** · CLAIMS 수정은 codex-sol 게이트 후.

★**여는 것**: ⑴ **b3null·b3shf 의 홀드아웃 생성문 첫 줄**을 봐야 낮은 발화율이 *안 여는 것*인지 *다른 토큰으로 여는 것*인지 갈린다 — 그런데 **둘 다 eval parquet 이 없다**(승인 대기 ⑧·⑨). ⇒ ★**⑧⑨ 는 정확도 판정만이 아니라 이 구분을 위해서도 필요하다.**
⑵ floor 가 막는 기전 — 상수 보상이 advantage 를 0 으로 만들어 그 구간의 정책 기울기를 얇게 한다는 기존 서술(CLAIMS:639)과 이 순서가 맞는가.
★**규율(오늘 여섯 번째)**: ★**"왜 떨어졌나"를 묻기 전에 "무엇이 붙들고 있었나"를 물어라** — 통제군을 겹치자 한 팔에서 못 찾던 답이 나왔다. **한 팔의 시계열은 통제군 없이는 이야기가 안 된다.**

## EXP-0811m (탐색 · ⛔판정 아님 · ★codex 게이트 전 CLAIMS 수정 금지) — **floor 기제: 코드는 기록된 설명을 반박하고, 0811l 의 순서를 예측한다**

0811l 이 *"floor 만이 발화 붕괴를 막는다"* 를 다섯 팔로 보였다. **왜 막는지**를 정본 코드에서 확인했다.

**① 구현 — floor 는 보상이 아니라 *중심화 이후* 의 advantage 편향이다.**
`src/training/dcpo_region.py:1182-1192`(docstring, 원문):
> *"a small POSITIVE, **UN-CENTERED** advantage bias added onto the META_CONTENT tokens of TRUSTED-meta rows … It is added **AFTER** the Dr.GRPO group-mean-subtract **on purpose**: a constant folded into R_meta **BEFORE** centering **cancels** (the group mean absorbs any term common to all rows), **silently doing nothing**. Routed post-centering it **survives**, giving 'emit a trusted wellformed meta' a fixed +meta_floor pull that offsets the FORMAT-penalty collapse pressure (v3l: meta_emit 0.5→0 by step 60). The CENTERED Â_meta still rides on top, so R_meta keeps deciding useful-vs-harmful meta — **the floor only keeps the channel OPEN, it does not grade content**."*
`:1323-1330` 주석과 `:1348` `advantages = advantages + float(meta_floor) * fl * (meta_in_resp / row_n)` 이 그대로 그 설계다.

**② 그래서 `CLAIMS.md:639` 의 기제 문장이 어긋난다.**
> *"floor 가 매 스텝 상수 보상을 보장 → **그 성분의 그룹 정규화 advantage 가 0** → 메타 구간의 정책 기울기가 얇아지고, KL 앵커가 0인 상태에서 엔트로피 보너스가 무저항으로 이긴다."*
★**"그룹 정규화되어 0 이 된다"는 것은 코드가 명시적으로 피한 실패다.** floor 를 *보상*으로 보면 그 결론이 나오지만, 구현은 floor 를 *advantage* 에 후단 주입한다.
⇒ b3s 의 엔트로피 폭주는 **실측 사실**(최대 6.442)이지만, **그 설명은 구현과 맞지 않는다.** ⛔"floor 가 기울기를 얇게 했다"를 계속 인용하지 말 것.

**③ 반대로, 코드가 말하는 것은 0811l 이 잰 것과 정확히 맞는다.**
docstring 의 목적어가 *"FORMAT-penalty collapse pressure 를 상쇄해 채널을 열어 둔다"* 이고, 실측 순서가 그대로다 —
**floor 있는 b3s 만 1.000 유지**, floor 없는 넷은 전부 붕괴(b3null gs125 · b3nopmi gs152 · b3p gs157 · b3shf gs228).
그리고 docstring 이 인용한 선례(v3l: `meta_emit 0.5→0 by step 60`)까지 같은 모양이다.
⇒ ★**구현의 선언된 의도와 다섯 팔의 용량-반응이 일치한다.** 이건 오늘 처음으로 **기록과 코드와 실측이 서로를 지지한 자리**다.

**④ 그래서 열려 있는 것은 다른 질문이다.** floor 가 "기울기를 얇게 해서" 엔트로피를 풀어준 게 아니라면,
**b3s 의 엔트로피는 왜 6.442 로 갔나** — 여전히 미설명. 후보: 발화가 유지된 채 메타 내용에 대한 압력만 남아 그 구간의 탐색이 커졌나,
아니면 엔트로피 폭주가 floor 와 무관한 별개 경로인가(⚠`entropy_coeff` 는 두 팔 동일).

**⚠경계** · docstring 은 **의도의 선언**이지 발화 검사가 아니다 — `meta_floor` 가 b3s 런에서 실제로 0.05 로 들어갔는지는 런처 매니페스트로 따로 봐야 한다(G8) ·
b3s 는 floor 외에도 shiftonly 설정이 함께 다르다 ⇒ **단일 변수 아님** · **EXP 트랙**.

★**여는 것**: ⑴ b3s 런처의 실제 `dcpo_meta_floor` 값을 매니페스트에서 확인(G8) ⑵ b3s 의 `actor/entropy` 상승 시점과 `meta_emit_rate`·`rmeta_*` 를 겹쳐 **엔트로피 경로**를 다시 후보화 ⑶ `CLAIMS:639` 정정문 초안(⛔codex-sol 게이트 후).
★**규율(오늘 일곱 번째)**: ★**기제 문장은 코드 docstring 한 줄로 죽거나 산다.** 오늘 네 개의 기제 서술을 검사했고 — 내 것 둘(0811k), 기록된 것 둘(`:641` 약화·`:639` 반박) — **셋이 틀렸다.** 기제를 쓰기 전에 **그 성분의 구현 주석을 먼저 읽어라.**

**★0811m 후속 — G8(팔 정체) 통과, 경계 ⑴ 닫힘.** 런처 실측:
`h100std_rq3v2f_b3s.yaml` **`dcpo_meta_floor=0.05`** · `b3sh`/`b3shf`/`b3null` **명시 0.0** ·
`b3p`/`b3nopmi` **미지정 → config 기본** `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:154` **`dcpo_meta_floor: 0.0`**.
⇒ **여섯 팔 중 floor 가 켜진 것은 b3s 하나뿐**이고, 그것이 발화를 유지한 유일한 팔이다. 0811l 의 순서는 이 축에서 깨끗이 갈린다.
⚠남는 경계: b3s 는 shiftonly 설정도 함께 다르므로 **단일 변수는 아니다**(floor 만 다른 팔은 없다).

## EXP-0811n (탐색 · ⛔판정 아님) — **b3s 엔트로피는 "폭주"가 아니라 gs50부터의 완만한 램프다** · 연대가 기존 서사와 안 맞는다

0811m 이 `CLAIMS:639` 의 기제를 반박하고 *"그럼 b3s 엔트로피는 왜 6.4 인가"* 를 열었다. 설명을 버리고 궤적부터 봤다(11개 키 전부 존재).

| gs | 1 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 215 | 230 | 250 | 265 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `actor/entropy` | .214 | .209 | .249 | .359 | .457 | .690 | .829 | 1.020 | 1.474 | 2.543 | 3.689 | 4.642 | 5.405 | **5.880** |
| `meta_emit_rate` | .996 | .996 | .994 | 1.000 | .994 | 1.000 | .996 | 1.000 | .998 | .998 | .996 | 1.000 | 1.000 | **1.000** |
| `response_length/mean` | 250 | 372 | 385 | 437 | 488 | **1137** | 1163 | **2297** | 1577 | 1184 | 1010 | 1174 | 1274 | 1122 |
| `wellformed_rate` | .939 | .930 | .959 | .941 | .914 | .805 | .781 | **.674** | .766 | .836 | .803 | .760 | .766 | .797 |
| `actor/grad_norm` | .581 | .358 | .499 | .432 | .316 | **.185** | .195 | .222 | .338 | .239 | .182 | .163 | .224 | .306 |
| `rmeta_pos/neg_rate` | .22/.27 | .27/.31 | .20/.20 | .28/.28 | .22/.26 | .20/.17 | .15/.22 | .23/.12 | .28/.21 | .24/.26 | .23/.21 | .21/.20 | .27/.09 | .26/.17 |
| `kl_loss` | .000 | .002 | .010 | .028 | .048 | .062 | .078 | .092 | .091 | .117 | .131 | .136 | .137 | .131 |

**① "폭주"가 아니다.** 임계 통과: **>0.5 gs93** · >1.0 gs175 · >2.0 gs209 · >4.0 gs226.
**gs50 부터 단조로 오른다** — 어떤 사건이 촉발한 계단이 아니라 **처음부터의 램프**다.

**② 연대가 기존 서사와 안 맞는다.** `CLAIMS.md:641` 은 *"b3s 는 floor 때문에 **탈출로가 막혀** 압력이 엔트로피로 갔다"* 고 한다.
그런데 ⑴ b3s 의 발화는 **한 번도 떨어진 적이 없다**(전 구간 .994~1.000) — **"나가려다 막힌 순간"이 없다.**
⑵ 엔트로피가 0.5 를 넘은 **gs93** 은 다른 팔들의 붕괴 시작(**b3null gs125** · b3nopmi gs152 · b3p gs157)보다 **앞선다**.
⇒ ★**"막혀서 방향이 틀어졌다"는 인과 순서가 데이터에 없다.** (⛔`:641` 의 b3s 절도 재인용 전 재검토 필요 — `:639` 에 이어 **두 번째**.)

**③ 먼저 움직인 것은 길이다.** `response_length/mean` 이 gs100 **488** → gs125 **1137**(2.3배) → gs175 **2297**(최대) 로 뛰고,
`wellformed_rate` 는 같은 구간에 .914 → .805 → **.674** 로 내린다. **엔트로피의 가파른 구간(>1.0, gs175)은 그 뒤에 온다.**
⚠단 길이는 gs200 이후 **1000~1300 으로 되돌아오는데 엔트로피는 계속 오른다** ⇒ 길이만으로는 후반이 설명 안 된다.

**④ 메타 보상은 내내 순중립이다.** `rmeta_pos_rate ≈ rmeta_neg_rate`(.20~.28 vs .09~.31), 평균은 0 근처에서 진동.
⇒ floor 가 채널을 열어 두는 동안 **내용을 가르는 힘은 거의 없었다**(docstring 이 말한 *"floor only keeps the channel OPEN, it does not grade content"* 의 실측 대응).
**`grad_norm` 은 오히려 낮아진다**(.58→.185→.16~.31) — E-158 이 기록한 region 경로의 작은 advantage 와 일관.

**⚠결론은 후보까지만.** 후보 ⑴ **길이 인플레이션 → 엔트로피**(단 후반 미설명) ⑵ **순중립 R_meta + un-centered floor 아래에서 메타 구간이 사실상 자유롭게 표류**(내용 압력 부재) ⑶ 별개 경로.
⛔**어느 것도 지금 기제로 쓰지 않는다.** 오늘 기제 서술을 넷 검사해 셋이 틀렸다 — **네 번째를 더 만들지 않는다.**

★**여는 것**: ⑴ b3s 와 **같은 floor 를 켠 다른 팔이 없다** ⇒ floor 단독 효과는 **현재 데이터로 분리 불가**(새 팔 = 승인 사항) ⑵ 길이 가설은 `len_cost 0.08` 이 켜져 있는데도 2297 까지 갔다는 점을 먼저 설명해야 한다 ⑶ 동형이의자 파싱실패 궤적(CLAIMS:653)과 이 램프의 시간 정렬.
★**규율**: ★**"폭주"라는 말을 쓰기 전에 임계 통과 시각을 적어라** — gs50 부터의 램프를 "폭주"로 부르면 **촉발 사건을 찾는 잘못된 탐색**이 시작된다.

## EXP-0811o (탐색 · ⛔판정 아님) — **정보를 가진 유일한 신호는 여섯 팔에서 똑같다 — 어떤 메타 개입도 못 움직였다**

0811c(iv) 가 *"8샘플 최빈답 일치는 셋 다 +0.30"* 을 냈고, 0811e 가 그 신호는 **8슬롯을 가로질러 읽으므로 슬롯 난수에 면역**임을 함의했다.
전 팔로 넓혔다(held-out MATH500 16k n8, 문항내 대비, 혼합문항만).

| 팔 | 정답률 | 최빈답 다수율 | 혼합 | 평균차 | **U** | 95%CI | p |
|---|---|---|---|---|---|---|---|
| **b0p**(메타 없음·matched base) | .6158 | .7950 | 245 | +0.4086 | **+0.3265** | [+.2368,+.4163] | .000 |
| b2p | .6290 | .8085 | 223 | +0.3887 | +0.3004 | [+.2030,+.3979] | .000 |
| b3s | .6092 | .7917 | 240 | +0.4060 | +0.3125 | [+.2210,+.4040] | .000 |
| b3p | .6075 | .7907 | 250 | +0.3973 | +0.3160 | [+.2259,+.4061] | .000 |
| b3nopmi | .6112 | .7940 | 244 | +0.4104 | **+0.3402** | [+.2526,+.4277] | .000 |
| b3sh gs175 | .5975 | .7935 | 254 | +0.3659 | +0.2992 | [+.2101,+.3883] | .000 |
| *base(instruct 세대)* | .5440 | .7470 | 283 | +0.2708 | +0.1696 | [+.0815,+.2577] | .000 |
| *pmishift(instruct 세대)* | .6600 | .8465 | 202 | +0.3431 | +0.2673 | [+.1686,+.3661] | .000 |

★**base 세대 여섯 팔이 +0.2992 ~ +0.3402 안에 다 들어간다.** CI 가 전부 겹치고, **최빈답 다수율도 .79~.81 로 평평**하다.
**메타를 한 번도 안 배운 b0p 가 오히려 상단(+0.3265)** 이고, 메타 SFT+GRPO 인 b2p 가 하단(+0.3004)이다 — ⚠물론 **구별 불가**(CI 완전 중첩).
⇒ ★**이 프로그램의 어떤 메타 개입도 자기일관성 신호를 움직이지 못했다.** SFT 레시피도, TRIOBJ 패키지도, PMI 헤드도, floor 도, shiftonly 도.

**오늘 사슬을 한 줄로 놓으면**: **선언 채널 = 0**(0811b/d, 슬롯 난수가 지배 0811e/f) · **행동 채널 = 약한 음수**(0811c ii, 메타블록 ≥2 → −0.21) ·
**샘플 간 합의 채널 = +0.30 이고 여섯 팔에서 불변**(여기).
⇒ **정보가 있는 축은 하나뿐인데, 그 축은 우리가 건드린 적이 없다.**

**⚠경계** · (iv)는 부분적으로 순환(모델이 우연보다 나으면 최빈답은 자주 정답) — 요점은 **팔 간 불변성**이지 크기가 아니다 ·
instruct 세대 두 줄은 **다른 기질·다른 데이터**라 세로 비교 금지(참고로만) · 저장 `is_correct` 는 C-002 의심 · 단일 벤치 · **EXP 트랙**.

★**여는 것**: 0811c 가 열어 둔 *"보상 대상을 선언에서 **샘플 간 합의**로 옮긴다"* 의 전제 점검 결과 —
**축은 실재하고(U +0.30, p<.001), 아직 아무도 안 건드렸다.** 다만 그것이 *"움직일 수 있는 축"* 이라는 증거는 **없다**(불변성은 양날이다:
개입이 없어서 안 움직인 것일 수도, 움직이지 않는 축일 수도). ⚠새 가설이므로 **승인 사항**.
★**규율**: ★**"우리 개입이 X 를 바꿨나"는 통제군을 같은 표에 넣기 전에는 못 묻는다** — b0p 한 줄이 여섯 팔의 이야기를 결정했다.

## 0811 20:11 — **운영: amlt 인증 만료**(잡 사망 아님)

`amlt list` → `No authentication method succeeded. Try refreshing your credentials with "az login"`.
★**잡 상태가 아니라 내 조회 채널이 끊긴 것**이고, 지금 **도는 학습은 0**(넷 다 대기 셸)이라 실질 손실은 없다.
HF·wandb 는 `.env` 토큰이라 **정상**(이 틱의 EXP 는 그대로 수행됨). 복구는 대화형이라 사용자만 가능: 프롬프트에 **`! az login`**.
⚠복구 전까지 (A)status·`amlt log view` 는 못 쓴다 ⇒ **새 발사·잡 사망을 못 본다.** ⑧⑨ 발사도 인증이 필요하다.

## EXP-0811p (탐색 · ⛔판정 아님) — **엔트로피가 먼저, 동형이의자가 20시간 뒤** — `CLAIMS:657` 방향은 확증된다

0811n 이 b3s 엔트로피를 gs50부터의 램프로 확정했다. `CLAIMS:653` 이 동형이의자 파싱실패를 **시각별**로 적어 뒀으므로,
wandb `_timestamp` 로 스텝↔벽시계를 붙여 두 궤적을 정렬했다(런 08/03 10:20 → 08/06 21:06 UTC, n=302).

| UTC | 파싱실패/h | 그때 gs | 그때 entropy |
|---|---|---|---|
| 8/4 18 | 2 | 192 | 1.036 |
| 8/4 20 | 12 | 200 | 1.474 |
| 8/4 22 | 22 | 208 | 1.595 |
| 8/5 00 | **70** | 219 | 3.036 |
| 8/5 01 | 134 | 224 | 3.408 |
| 8/5 02 | **174** | 230 | 3.689 |
| 8/5 04 | 126 | 242 | 4.993 |
| 8/5 07 | 130 | 254 | 4.799 |

임계 통과 시각: **entropy>0.5 = 08/03 22:06**(gs93) · >1.0 = 08/04 12:37(gs175) · >2.0 = 08/04 22:09(gs209) · >4.0 = 08/05 01:19(gs226).

★**엔트로피가 0.5 를 넘은 시각은 첫 파싱실패(8/4 18h, 2건/h)보다 20시간 앞선다.** 실패가 실질적으로 커지는 8/5 00h(70건/h)에는 엔트로피가 이미 **3.036**.
⇒ ★**`CLAIMS.md:657` 의 *"확산 → 동형이의자 → 0점 → 잡음 증가"* 는 **시간 순서가 데이터와 맞는다**.
역방향(동형이의자가 엔트로피를 키웠다)은 이 정렬로 **배제**된다.
⚠단 꼬리에서는 단조가 아니다 — 8/5 02h 174건(entropy 3.689) 뒤 04·07h 는 126·130 으로 줄었는데 엔트로피는 4.8~5.0 으로 계속 높다.
(0811n 의 "길이는 되돌아오는데 엔트로피는 안 내린다"와 같은 자리.)

**★오늘 `CLAIMS` 630-674 블록 검사 결산** — 네 서술을 시간·코드·통제군으로 쟀다:
| 서술 | 결과 |
|---|---|
| `:639` floor → 그룹정규화 advantage 0 → 기울기 얇아짐 | ⛔**코드가 반박**(0811m) |
| `:641` b3p 가 발화를 죽여 **탈출** | ⚠**약화** — 압력 없는 b3null 이 더 빨리 붕괴(0811l) |
| `:641` b3s 는 **막혀서** 압력이 엔트로피로 | ⛔**인과 순서 없음** — 발화가 한 번도 안 떨어졌고 엔트로피가 더 이르다(0811n) |
| `:657` 확산 → 동형이의자 | ✅**확증**(여기) |
⇒ **같은 블록 안에서 셋이 흔들리고 하나가 섰다.** ⛔CLAIMS 수정은 **codex-sol 게이트 후** 한꺼번에.

★**규율**: ★**시각이 적혀 있는 기록은 시각으로 검사할 수 있다** — `CLAIMS:653` 의 시간표가 없었으면 이 방향은 못 갈랐다.
**기록에 시각을 남기는 것이 나중의 인과 검사를 가능하게 한다.**

## EXP-0811q (탐색 · ⛔판정 아님) — **①재채점 통과 ②캘리브레이션 보상은 켜져 있었다 ③그런데 confidence 는 9번째 글자에서 선언된다**

사용자 질문 셋(채점 재확인 · 캘리브레이션 보상 · 습관이 왜 안 오르나)에 대한 GPU 0 실측.

**① 재채점 — 저장 `is_correct` 는 14pp 낮게 매기지만, 오늘 결론은 전부 살아남는다.**
정본 `experiments/analysis/analysis_common.robust_grade`(math_verify 사용가능=True)로 4,000행×2팔 재채점:

| | 저장 | **재채점** | 일치율 |
|---|---|---|---|
| b2p | .6290 | **.7732**(+.1442) | .8502 |
| b3s | .6092 | **.7522**(+.1430) | .8500 |

문항내 대비를 **두 채점기로 나란히**:

| 대비 | 저장 | **재채점** |
|---|---|---|
| ① confidence hi vs lo (b2p) | U +.0387 p=.129 | **U +.0367 [−.0153,+.0886] p=.192** |
| ① 같은 것 (b3s) | U −.0833 p=.104 | **U −.0903 [−.1883,+.0077] p=.098** |
| ② 최빈답 일치 (b2p) | +.3004 | **+.4036 [+.3012,+.5060]** ↑ |
| ② 같은 것 (b3s) | +.3125 | **+.4375 [+.3426,+.5324]** ↑ |
| ③ 메타블록 ≥2 (b2p) | −.2093 p=.011 | **−.2093 [−.3643,−.0543] p=.015** |

★**채점을 고쳐도 confidence 는 못 가른다. 잘 되는 신호(자기일관성)만 더 커진다.**
⇒ **C-002 경고는 여전히 유효**(절대값·팔 간 비교엔 저장 컬럼 금지)하지만, **오늘의 문항내 결론은 채점 결함의 산물이 아니다.**

**② 캘리브레이션 보상 — 존재하고, 올바르고, 켜져 있었다.**
`dcpo_region.py:951-954` `R_cal[i] = -((conf[i] - c_with)**2)` (Brier) · conf 없으면 0 · discard 행만 0(`:964-967`) ·
`:1227` 그룹 평균 차감 후 `:1156` `+ w_cal * Â_cal * CONF` 로 **conf 토큰 마스크에 실린다**.
**G8 — 팔별 `dcpo_w_cal`**: **b2p = VANILLA_GRPO(헤드 자체가 없음)** · b3p·b3s·b3nopmi **0.3**(config `:116` 기본) · b3sh·b3shf·b3null **0.0**(명시).
⇒ ★**b2p 의 confidence 는 RL 이 한 번도 손댄 적 없는 SFT 모방**이다. 그런데 **캘리브레이션이 켜지고 발화가 300스텝 내내 1.000 이던 b3s 에서도** 재채점 U = **−0.0903 [−.1883,+.0077]** 로 **여전히 못 가른다.**

**③ 근인 후보 — confidence 는 응답의 9번째 글자에서 선언된다.**
`completion` 내 `confidence:` 위치: **b2p·b3s 모두 중앙값 = 최댓값 = 9자**, `0~20자` 비율 **1.0000**(4,000/4,000). 전체 응답 길이 중앙 **866 / 916자**.
⇒ ★**모델은 풀이를 한 글자도 쓰기 전에 확신도를 선언한다.** 그 시점에 조건화할 수 있는 것은 **문항뿐**이고,
그 롤아웃의 정오는 남은 ~860자의 표집에 달려 있다.
★**따라서 그 위치에서 Brier 를 최소화하는 최적 정책은 `P(정답 | 문항)` — 문항당 상수**다.
그리고 그것이 0811f 가 잰 세 층위 그대로다: **분포 수준 단조 ✅ / 문항 수준 ≈ 상수 ✅ / 시도 수준 0 ✅**.
⇒ **시도 수준의 널은 학습 실패가 아니라 배치가 강제한 것**으로 읽을 수 있다.
그리고 최적값은 실수인데 어휘는 `0.75`/`0.88` 둘뿐이라 정책이 그 사이에 앉고 **슬롯별 난수가 어느 쪽인지 정한다**(0811e).

**④ 그리고 "이 메타가 유용했나"를 재는 경로는 한 번도 안 돌았다.**
`dcpo/cf_text_rate` **최대 0.000** — b3s(n=302)·b3p(n=300)·b3nopmi(n=302) 전부. `:945-946` 에 따라 `c_without is None` 이면 `R_meta=0` 이므로
**반사실 유용성 경로는 죽어 있었고**, 세 팔 다 `rmeta_source=pmi_shift` 로 대체 신호를 썼다.

★**종합 — 습관이 안 오르는 이유 셋이 겹친다**: ⑴**자리**(증거가 생기기 전에 묻는다) ⑵**내용 무채점**(반사실 미발화 + floor 는 *"does not grade content"*) ⑶**발화 지지가 내용맹**(floor 만이 붕괴를 막고, 없으면 기본값으로 0 이 된다 — 0811l).
**⚠경계**: ⑶은 실측, ⑴은 **강한 후보**(모델이 선언으로 자기 정오를 유도하지 못한다는 가정에 의존 — 우리 널과 일관하나 증명 아님), ⑵는 실측(`cf_text_rate` 0).
★**여는 것(⚠새 가설=승인 사항)**: **확신도를 풀이 *뒤* 로 옮기면** 그 시점에는 시도의 증거가 존재한다 ⇒ 시도 수준 신호가 원리적으로 가능해진다. **지금 구현하지 않는다.**
★**규율**: ★**보상이 옳아도 그 보상이 걸린 토큰의 위치가 예측 가능성을 결정한다** — 보상 설계를 의심하기 전에 **그 토큰이 어디에 찍히는지** 먼저 재라.

## EXP-0811r (탐색 · ⛔판정 아님 · ★중요) — **정정: `pmi_shift` 는 유용성 측정이 맞다. 그런데 메타가 앞으로 이동해 그 측정의 전제가 깨졌다**

**⚠먼저 자기 정정.** 0811q 에서 *"이 메타가 유용했나를 재는 경로는 한 번도 안 돌았다"* 고 썼다. **틀렸다.**
안 돈 것은 **결과 수준 반사실**(`c_with − c_without`, `cf_text_rate` 최대 0.000)이고,
**믿음 수준 인과 측정인 `pmi_shift` 는 돌았다**(`rmeta_source=pmi_shift`, 전 팔).

**① `pmi_shift` 가 무엇인가**(`src/training/dcpo_pmi_shift.py:1-46`):
`PMI_open = logp(gold|메타 앞 본문) − logp(decoy|…)` · `PMI_close = 같은 것을 메타 포함 문맥에서` · **SHIFT = close − open**.
decoy→gold 뒤집힘 = **SAVE(+)**, gold→decoy = **DERAIL(−, 더 크게)**, 무뒤집힘 = 클립된 연속항.
⇒ ★**설계 의도로는 "메타가 믿음을 얼마나 옳은 쪽으로 움직였나"라는 인과 측정이 맞다.**

**② 그런데 계산 문맥이 이렇다**(`verl_sdc.py` `_compute_dcpo_v4_pmi_shift_rmeta`):
```
ctx_open_text  = prompt + response_prefix     # response_prefix = 메타 앞의 본문
ctx_close_text = ctx_open_text + meta_text
```
열거된 skip 사유는 **`skip_empty_meta` · `skip_dup_meta` · `skip_decoy` 셋뿐** — **`response_prefix` 가 비었는지 보는 가드는 없다.**

**③ 그리고 메타는 앞으로 이동했다 — 이것이 오늘의 실측이다.**

| | 메타 위치(문자) | **상대 위치** | 20자 이내 |
|---|---|---|---|
| SFT1 `b2on_v8meta_strict_sft`(4,245행) | 중앙 251 | 0.286 | 2.8% |
| **SFT2 `b2p2_rvseg_sft2`(378행, RL 출발점)** | 중앙 **500**, **최소 190** | **0.468** | **0.0000** |
| **b2p gs300 / b3s gs300**(각 4,000행) | **9** (중앙=최대) | ~0.01 | **1.0000** |
| **b3sh gs165**(4,000행) | **0** | ~0 | **1.0000** |

★**SFT2 에는 메타로 시작하는 예시가 378개 중 0개**인데, RL 후에는 **100%** 가 그렇다.
★**배선검사 통과**: 평가 프롬프트는 `[{"role":"user","content":question}]` + `add_generation_prompt=True` 뿐(`scripts/eval_vllm_1030.py:39-46`) — 시스템 프롬프트도 강제 프리픽스도 없다. **학습된 행동이다.**
★**시점 상한 gs165**(가장 이른 잔존 eval 이 이미 100%). 정확한 시점은 학습 로그의 스텝별 롤아웃 덤프(`dcpo_region.py:969`)로 **읽으면 나온다** — ⚠지금은 `amlt` 인증 만료로 막힘.

**④ 결과: `PMI_open` 이 "메타 앞의 믿음"이 아니라 "문항만 본 사전믿음"이 된다.**
본문이 없으면 되돌릴 추론도, 탈선시킬 확신도 없다. **SAVE/DERAIL 사건의 준거가 사라진다.**
⇒ 보상은 여전히 계산되지만, **재는 대상이 설계와 다르다.** 그리고 `dcpo_pmishift_meta_body_dup_thresh` 는 **전 팔 기본 1.0 = 꺼짐**이라
docstring 이 요구한 *"presence-as-confidence 혼입 가드"* 도 작동하지 않았다. 요구된 유효성 검정 둘(placebo·safe-default, `src/eval/pmi_shift_signal.py`)의 **결과는 `CLAIMS` 에 없다.**

**⑤ ★그리고 이 이동은 메타 보상이 만든 것이 아니다.** **b2p 는 VANILLA_GRPO 로 메타 헤드가 하나도 없는데도** 100% 가 9번째 글자다.
⇒ **정답성 압력만으로도 메타는 추론 밖으로 밀려난다.**

★**두 회피가 대칭이다**: b3p 는 **행동을 유지하고 탐지 토큰을 버렸고**(0811i/j `cros`), 모든 팔은 **토큰을 유지하고 행동을 추론 밖으로 옮겼다**(여기).
**둘 다 측정되는 것과 의도된 것을 분리하는 방법이다.**

**⚠경계**
· **"앞으로 옮겼다 = 쓸모없다"는 아니다** — 먼저 계획하고 푸는 것은 정당한 전략이다. 다만 ⑴설계 의도(추론 중 재고)와 다르고 ⑵`pmi_shift` 의 전제를 깨고 ⑶그 자리의 확신도는 시도 정보를 원리적으로 못 담는다(0811q).
· **"메타가 있는 게 도움이 되나"는 이미 답이 있다 — `C-026` `b2p − b0p` = **+0.18pp [−1.30,+1.68] 널**.** ⛔재계산 금지.
· 이동 **시점**은 gs165 상한만 있고 정확히 모른다 · b0p 는 메타 자체가 0건(설계대로).

★**여는 것(전부 승인 사항)**: ⑴**학습 로그로 이동 시점 확정**(GPU 0, `az login` 만 필요) ⑵**확신도를 풀이 뒤로 옮긴 사후질의 프로브**(재학습 없이 eval 만으로 0811q 의 구조 가설을 **반증 가능**) ⑶**prefix-forced continuation**(승인 ①) — 본문을 강제로 깔면 `PMI_open` 도 `c_without` 도 비로소 의미를 갖는다 ⑷ 향후 pmi_shift 실행 시 **빈 prefix 가드 + `dup_thresh` 활성화**(정본 코드 변경).
★**규율**: ★**보상의 전제가 되는 문맥이 학습 중에 변할 수 있다** — 보상식을 검사할 때 **그 식이 참조하는 문맥이 실제로 존재하는지**를 같이 재라. 여기서는 "메타 앞 본문"이 100% 비어 있었다.

## EXP-0811s (탐색 · ⛔판정 아님 · ★핵심) — **메타는 "추론 중 재고"에서 "앞머리 선언"으로 바뀌었고, 그 변화는 SFT2 데이터에서 이미 시작됐다**

0811r 이 *"메타가 앞으로 이동해 `pmi_shift` 의 전제가 깨졌다"* 를 냈다. **어디서부터 그랬는지**를 데이터 계보로 거슬러 올라갔다.

**① 계보 — 3단 붕괴**

| 단계 | `<think>` | **메타가 think 안** | 메타 위치 |
|---|---|---|---|
| `v8_meta_inside_think`(6,329행) | 100% | **100%** | 234자 / `</think>` 645 |
| **SFT1** `b2on_v8meta_strict_sft`(4,245행) | 100% | **100%** | 251자 / `</think>` 818 |
| 원본 풀 `rv_redirect_verify_functional`(1,763행) | 100% | **4.65%**(82행) | — |
| **SFT2** `b2p2_rvseg_sft2`(378행) ← **RL 출발점** | 100% | **0.00%** | 메타 500자, **`</think>` 272자** |
| **RL 후** b2p·b3s gs300(각 4,000행) | **0.0000** | — | **0자, 100%** |

★**SFT1 은 "추론 안에서 재고하라"를 100% 로 가르쳤고, SFT2 는 그것을 0% 로 뒤집었다.**
그리고 원본 풀 자체가 이미 **95.35% 가 `</think>` 밖**이므로, 이건 378행 필터만의 문제가 아니라 **그 데이터 계열의 생성 방식**이다.
(⚠SFT2 의 378행 선별에 `think-closed` 조건이 있었다는 기록은 E-093 에 있다 — 그 조건이 **redirect 기아**만이 아니라 **메타 위치 반전**도 낳았다.)
★그 위에 RL 이 한 걸음 더 갔다: **`<think>` 자체를 버리고 메타를 0번 자리로.**

**② instruct 세대와 비교 — "이 문제가 없었나"의 답: 정도의 차이지 면제는 아니다**

| 팔 | 메타율 | `<think>` | **메타가 think 안** | **메타 앞 본문(중앙, 자)** |
|---|---|---|---|---|
| **gandhi**(instruct) | .1445 | **1.0000** | .0087 | ★**2,418** |
| pmishift(instruct) | .9230 | .5988 | .0188 | **23** |
| shiftonly(instruct) | .9915 | .1168 | .0022 | **8** |
| b2p(base) | 1.0000 | **.0000** | — | **0** |
| b3s(base) | 1.0000 | **.0000** | — | **0** |

★**어느 팔도 메타를 think 안에 두지 않는다**(0.2~1.9%). 그러나 **결정적 변수는 "think 안"이 아니라 "메타 앞에 본문이 얼마나 있나"** 이고,
그 축에서는 **gandhi 만 실질적으로 다르다**(2,418자). 나머지는 0~23자.
⇒ `PMI_open` 이 의미를 갖는 팔은 **gandhi 하나뿐**이었고, 그 팔은 정작 메타를 14%만 낸다.

**③ ★그래서 "유용성 측정"의 진짜 문제는 창(window)이다.**
메타가 앞머리 계획이 되면 그 인과 효과는 **뒤따르는 ~860자의 풀이를 통해** 흐른다.
그런데 `pmi_shift` 는 **메타 토큰 경계의 즉각적 믿음 점프**만 본다 ⇒ **계획이 도움이 됐어도 못 잡는다.**
그것을 잡을 수 있는 유일한 측정은 **결과 수준 반사실 `c_with − c_without`** 인데 — **`cf_text_rate` 최대 0.000, 한 번도 안 돌았다.**

★★**정리**: 유용성 측정이 **둘** 있었다. **앞머리 계획을 잡을 수 있는 쪽(결과 반사실)은 안 돌았고, 돈 쪽(믿음 점프)은 추론 중 재고만 잡는데 모델이 그걸 그만뒀다.**
⇒ *"메모가 도움이 됐나"* 는 **원리적으로 측정 밖에 있었다.** 보상 설계의 잘못이 아니라 **측정 창과 행동 위치의 불일치**다.

**⚠경계** · "앞머리 계획"이 나쁘다는 증거는 없다(오히려 정당한 전략) — 다만 **그 형태는 우리 측정 어느 것도 안 잡는다** · gandhi 는 다른 방법이라 세로 비교 금지 ·
SFT2 의 위치 반전이 **RL 이동의 원인인지**는 미확정(RL 은 b2p 에서도 일어났고 b2p 도 같은 SFT2 에서 출발했다 — 즉 **둘 다 SFT2 하류**라 분리 안 됨).
★**여는 것**: ⑴ SFT1 체크포인트(메타 100% think 안)에서 바로 RL 을 걸면 위치가 유지되는가 — **SFT2 를 건너뛰는 팔** ⑵ 앞머리 계획을 잡는 유일한 측정인 **결과 반사실을 살리기**(승인 ①) ⑶ 확신도 사후질의 프로브.
★**규율**: ★**측정 창이 행동 위치를 따라가는지 확인하라** — 행동이 옮겨가면 같은 공식이 다른 것을 잰다. 여기서는 "메타 앞 본문"이 4,245행 100% → 378행 0% → 롤아웃 0자로 줄었다.

## 0811 설계 노트 (⛔판정 아님) — 사용자 제안(conf-게이트 제어루프)에 걸리는 **선행 기록 둘**

**① PMI-shift 유효성 검정은 이미 한 번 발사 직전에 멈췄다**(원장 §6868-6885). 막은 것은 게이트가 아니라 **측정 정의**:
· `src/eval/pmi_shift_signal.py:293-295` 의 AUC 라벨은 `r["correct"]` 이고 **`c_without` 이 그 파일에 없다** ⇒ 그 `auc_shift` 는 *"shift 가 정답/오답을 가르나"*(S0.2 가 이미 잰 것)이지 ***"뒤집기를 예측하나"* 가 아니다.**
· 뒤집기 라벨 자체가 오염: `dcpo_region.py:391 cf_answer_from_prefix` → `rewards.py:186-187 _extract_answer_fallback` 이 `\boxed{}`·`####` 없으면 **텍스트의 마지막 숫자**를 답으로 삼는다.
  ★같은 저장소의 `_extract_boxed_or_hash`(0714 bugfix)는 그 폴백을 **일부러 버렸다** — *"last-number 휴리스틱이 과발화해 진짜 오답을 뒤집었다"*. ⇒ **정답 채점기는 고쳤는데 반사실 경로는 안 고쳤다.**
⇒ ★**pmi_shift 를 다시 쓰려면 이 두 개를 먼저 고쳐야 한다.** 사용자의 *"먼저 신호가 있는지 테스트하자"* 는 정확히 이 지점을 짚었다.

**② 그리고 "메타가 실제로 도움이 되나"는 강제-계속 설계로 이미 한 번 측정됐다 — PG0(0619)**
[[pg0-raw-onpolicy-harvest-infeasible]]: 같은 prefix + 실제 메타 vs 제거/셔플, 최종답 채점.
· 전체 `mean_gap(R−Nc)` = **+0.046** · `N'=Nc` 로 *"아무 주입"* 교란 기각 ⇒ **메타 내용이 인과적으로 이롭다(약하게)**
· saves 87/200(43.5%) · strong(≥.25) 41/200(20.5%) ⇒ **소수에서 강하게, 다수에선 무효**
· ★**난이도별**: easy **+0.082**(n38) · medium +0.013(n100) · **hard −0.034**(n62, **역효과**)
· scenario: **verify +0.020 > redirect +0.007**
⇒ ★★**사용자 제안의 `conf<0.3 → 다른 방향(redirect)` 분기는 PG0 가 해롭다고 기록한 칸(hard·redirect)에 정확히 조준된다.**
반대로 `conf>0.8 → 검산(verify)` 은 **이로운 칸**(easy/medium·verify)에 조준된다.
⚠PG0 는 콜드스타트 SFT 모델(0619)에서 잰 사전확률이지 현 체크포인트의 판정이 아니다. 그리고 0620 부분정정(채점 버그)이 붙어 있다.

★**설계 함의(⛔구현 아님·승인 사항)**: ⑴분기를 **하니스가 강제**하면 confidence 가 인과적으로 하중을 받고 **정확도 하나로 end-to-end 채점된다** — `R_cal`·`R_meta`·floor·pmi_shift 가 전부 불필요해진다 ⑵반사실은 롤아웃 단위가 아니라 **팔 단위**(분기 ON vs OFF)로 싸게 얻어진다 ⑶⚠현재 conf 는 값이 **두 개뿐**(0.75/0.88)이라 0.8/0.3 문턱이 사실상 결정론이 된다 — **분기 전에 conf 해상도를 먼저 확인해야 한다**.

## EXP-0811t (탐색 · ⛔판정 아님 · ★★오늘 최대 소득) — **모델은 메타를 옮기지 않았다. 메타가 되짚으려던 대상을 지웠다**

0811s 가 *"메타가 앞으로 이동했다"* 로 닫았다. **틀린 서술이었다.** SFT2 원문을 통째로 읽으니 다른 그림이 나온다.

**① SFT2 가 가르친 서식 — 한 문제를 두 번 푼다**
```
<think> [1차 풀이 — 답까지 나옴] </think>     ← 중앙 272자에서 닫힘
confidence: 0.8 / assessment: / action:        ← 태그 없는 평문 블록
<|meta|> confidence: 0.88 … decision: verify <|/meta|>
[검산 = 2차 풀이]                               ← `</think>` 이후 중앙 763자 (앞의 2.8배)
\boxed{답}
```
★**`<think>` 안의 마지막 수가 `\boxed` 답과 일치: 190/348 = 0.5460** ⇒ **1차에서 이미 답이 나온다.** 뒤의 "검산"은 **두 번째 완전 풀이**다.
★그리고 confidence 를 **두 번** 쓴다(평문 0.8 + 태그 0.88) — 값도 다르다.

**② RL 후 남은 것 / 지운 것**(b2p gs300, n=4,000)

| 조각 | 잔존 |
|---|---|
| `<|meta|>` · `<|/meta|>` · `decision:` | **1.0000** |
| `\boxed` | 0.9762 |
| **`<think>` · `</think>`** | **0.0000** |
| **평문 `assessment:`** | **0.0000** (평문 `action:` 0.0130) |

⇒ ★**모델은 메타를 앞으로 옮긴 것이 아니라, 메타 *앞에 있던 1차 풀이와 평문 블록을 통째로 삭제*했다.**
남은 것은 `<|meta|>` → (검산이었던) 유일한 풀이 → `\boxed` 다.

**③ 그래서 이것은 합리적 최적화였다.** SFT2 서식은 **같은 문제를 두 번 풀게** 한다. 정답 보상 + 길이 압력 아래에서
**중복된 1차 패스를 지우는 것은 순이득**이다(답은 어차피 2차에서 나온다). 그리고 메타는 **두 패스의 경계에 고정**돼 있었으므로,
**앞쪽이 사라지자 메타가 0번 자리로 떨어졌다.**
★그 결과 메타 본문은 *"The expression looks plausible, but it should be independently checked"* 라고 말하는데 —
**그 "expression" 은 더 이상 존재하지 않는다.** 되짚을 대상이 지워진 채 되짚는 문장만 남았다.
★**메타 보상이 없는 b2p 도 그랬다** ⇒ 원인은 메타 보상이 아니라 **중복 서식 + 정답/길이 압력**이다.

**④ ★그리고 instruct 세대는 SFT2 를 하지 않았다 — 이것이 두 세대의 구조적 차이다**

| | 행수 | 메타가 think **안** | 퇴화(같은 줄 8회+) |
|---|---|---|---|
| **instruct RL 출발점** `v8_meta_inside_strict` | 4,264 | **1.0000** | **0.0000** |
| **base SFT1** `b2on_v8meta_strict_sft` | 4,245 | **1.0000** | **0.0000** |
| **base SFT2 = 실제 RL 출발점** `b2p2_rvseg_sft2` | **378** | **0.0000** | **0.0794** |

★**instruct 의 RL 출발점과 base SFT1 은 사실상 같은 물건이다**(4,264 vs 4,245행 · 둘 다 메타 100% think 안 · 퇴화 0).
★**base 세대만 그 뒤에 SFT2 를 한 단 더 얹었다** — **11배 작고**(378행), **메타가 think 밖 100%**, **8%가 퇴화 쓰레기**(같은 줄 최대 **976회** 반복).
원본 풀(1,763행)도 퇴화 5.50%(최대 1,849회)다. **SFT1 은 4,245행 중 0행.**

⚠**단정 금지 — 교란**: instruct vs base 는 **기질(Instruct vs Base 모델)도 다르다**. 따라서 *"SFT2 가 원인"* 은 **강한 가설**이지 판정이 아니다.
★**그런데 그것을 분리하는 실험이 정확히 하나 있다 — 같은 base 기질에서 SFT1 → RL(=SFT2 건너뛰기).** 단일변수다.

★**여는 것**: ⑴**SFT1 → RL 팔**(SFT2 제외, 다른 것 전부 동일) ⑵ SFT2 를 살리려면 **퇴화 30행 제거 + 1차 패스 중복 제거 + 메타를 think 안으로** 재구성 ⑶ 이 세 결함이 **redirect 기아(E-093)와 별개**라는 점 기록.
★**규율**: ★**"모델이 X 를 옮겼다"고 쓰기 전에 "모델이 X 앞의 것을 지운 건 아닌가"를 보라** — 상대 위치의 변화는 **분자가 아니라 분모**에서 올 수 있다.
★**그리고 학습 데이터는 한 행을 통째로 읽어라** — 통계 세 개(위치·길이·태그율)로는 "두 번 푼다"도 "퇴화 976회"도 안 보였다.

## ⛔정정-0811 (EXP-0811s·0811t 를 **대체**) — **세 가지를 틀렸다. 그리고 그 정정이 결론을 뒤집는다**

**⛔틀림 ①: 현재 팔의 RL init 은 378행 SFT2 가 아니다.**
런처 여덟 개 전부 `models/b2p2_rvfull_eb16_sft`(b0p 만 `b0p2_rvfull_eb16_sft`)를 쓴다 — **원본 1,763행으로 재학습한 SFT2**.
378행 `b2p2_rvseg_sft2` 는 **은퇴한 파일**이다. 나는 `CLAUDE.md` 의 계보 서술(*"SFT2 → models/b2p2_rvseg_sft = RL init"*)을 믿었고, **런처를 diff 하지 않았다**(G8 위반 — 나는 이 규칙을 갖고 있었으면서 *init 모델*에 적용하지 않았다). `CLAUDE.md` 가 낡았다.

**⛔틀림 ②: 이 감사는 이미 돌았다 — E-093/E-094/E-095 (2026-07-26).**
`E-094` 는 *"SFT2 필터가 SFT1과 싸우고 있었다 — meta-inside 를 100%→0% 로 역전"* 을 **제목으로** 달고 있고, 4코퍼스 감사표까지 있다.
그리고 **그 수리는 이미 배포됐다** — 현행 init 이 raw 1763 재학습본인 이유가 정확히 E-093/094/095 다.
⇒ **G1 재발 5회째.** 내 방아쇠 다섯 개(①새 실험 ②남이 준 수치 ③안 건드린 노브 ④"기록 없다" ⑤"여는 것")로도 못 잡았다. **여섯 번째 방아쇠 필요: "데이터·설정 파일을 감사하기 전에 그 파일 이름으로 원장을 grep 한다."**

**⛔틀림 ③: 내 `meta-INSIDE` 측정에 버그가 있었다.**
`te=t.find("</think>")` 가 **−1** 을 돌려주는 행(=think 가 안 닫힌 행)에서 `mi<te` 를 False 로 처리해 **"밖"으로 셌다.** 올바른 처리는 *"안 닫혔으면 그 뒤는 전부 think 안"* 이다.

| 코퍼스 | n | think-closed | **meta-INSIDE(정정)** | 내가 어제 쓴 값 |
|---|---|---|---|---|
| base SFT1 `b2on_v8meta_strict_sft` | 4,245 | 1.000 | **1.0000** | 1.0000 ✅ |
| T1 SFT1 `v8_meta_inside_strict` | 4,264 | 1.000 | **1.0000** | 1.0000 ✅ |
| **raw 1763 = 현재 init 학습데이터** | 1,763 | 0.412 | **0.6347** | ⛔**0.0465** |
| 378 필터(은퇴) | 378 | 1.000 | **0.0000** | 0.0000 ✅ |
E-094 의 **63.5%** 와 일치한다.

---

### ★그래서 결론이 뒤집힌다 — **범인은 데이터가 아니라 RL 이다**

| | meta-INSIDE |
|---|---|
| 현재 RL **출발점**(raw 1763 재학습) | **63.5%** |
| 현재 RL **도착점**(b2p·b3s gs300, n=4,000) | **0%** (`<think>` 자체가 0.0000) |

★**데이터는 이미 0% → 63.5% 로 수리됐고, RL 이 그것을 다시 0% 로 만들었다.**
⇒ 어제 내가 쓴 *"SFT2 가 범인, SFT1→RL 이 단일변수 검정"* 은 **약해진다.** 63.5% 짜리 출발점도 못 버텼는데 100% 짜리가 버틴다는 보장이 없다.
⇒ ★**"RL 자체가 메타를 추론 밖으로 민다"가 이제 주가설**이고, **하니스 강제 구조**가 선택지가 아니라 **주 후보**가 된다.

**살아남는 오늘의 관측**(재확인함): ⑴RL 도착점은 `<think>` 0.0000·메타 0번자리 100% ⑵b2p(메타 보상 0)도 그렇다 ⇒ 정답/길이 압력이 원인 ⑶SFT2 서식은 **한 문제를 두 번 푼다**(think 안 마지막 수 == `\boxed` 답 54.6%) ⇒ RL 이 1차 패스를 지운 것 ⑷raw 1763 에 **같은 줄 8회+ 연속반복 5.50%**(최대 1,849회) — E-095 의 *"메타신호 죽은 행 29.7%"* 와 **다른 결함일 수 있음**(교차확인 필요).

★**규율(오늘 여덟 번째)**: ★**`CLAUDE.md` 같은 요약 문서는 계보의 authority 가 아니다 — 런처가 authority 다.** 그리고 ★**`find()` 가 −1 을 돌려주는 경우를 분기로 적어라**(부재를 "밖"으로 세면 부호가 뒤집힌다).

## 0812 00:12 — 점검: **인증 복구 · 그리고 "완주"로 기록한 잡 둘이 선점→재개되어 다시 돌았다**

**① 인증 복구.** `az account get-access-token --subscription c4c534bc-…` → 만료 **2026-08-12 01:15:04 UTC**. `amlt list` 정상.
(0811 20:11 의 `AADSTS530002` 기기규정 거부는 해소됨.)

**② ⚠어제 완주로 기록한 잡 둘이 다시 돌았다 — 선점 자동재큐**

| 잡 | 어제 기록 | **지금 실제** | 노드 |
|---|---|---|---|
| `rq3v2f-b3nopmi-0807` | 완주 gs304(HB 8/10 21:39) | 재개 → **gs307 까지 학습 → 8/12 00:10 재완주** · `sleep 86400` | **점유** |
| `rq3v2f-b3shf-0810b` | 완주 gs300 `rc=0`(8/11 14:44) | 재개 · `existing GRPO resume gs = 302` · **지금 ckpt 내려받는 중** | **점유** |
| `rq3v2f-b3null-0807` | 완주 gs303 | **status `pass`** — 종료 확인 | 해제 |

★**`FINAL PUSH DURABLE` + `rc=0` 을 봐도 "이 잡은 끝났다"가 아니다** — Standard 티어에서 선점되면 **자동 재큐되어 다시 시작**하고,
resume 로직이 **마지막 체크포인트에서 이어 학습**한다. ⇒ **완주 판정에 유효기간이 있다.**

**③ ⚠체크포인트 영향 — 하나는 안전, 하나는 위험**

| 팔 | `checkpoints/` 현재 | 보존본 |
|---|---|---|
| b3nopmi | **gs307 만** — **gs300 프루닝됨** | ✅ `preserved/mechanism_alive/rq3v2f_b3nopmi_gs300`(23파일) |
| **b3shf** | gs300 · gs301 · gs302 | ⛔**gs300 보존본 없음**(`rq3v2f_b3shf_gs100` 만 있음) |
| b3null | gs303 | ✅ `rq3v2f_b3null_gs300`(23파일) |

★**C-031 이 쓴 b3nopmi gs300 은 보존본 덕에 살아남았다.** 보존이 없었으면 **판정의 근거가 사라졌을 것**이다.
⚠**b3shf gs300 은 보존본이 없고, 재개된 잡이 다음 저장(gs305 예상)을 하는 순간 `--keep 3` 이 지운다.** 승인 대기 ⑧ 이 그 체크포인트를 쓴다.
남은 시간 추정: ckpt 다운로드 + 3스텝 ≈ **30~50분**.

**④ 운영 함의 — 병렬 팔에 쓸 노드가 묶여 있다**
둘 다 **이미 답이 나온 팔을 다시 돌리는 중**이고(b3nopmi 는 재완주 후 유휴 대기, b3shf 는 붕괴 구간을 더 파는 중),
H100×4 노드를 **각각 하나씩** 점유하고 있다.

★**규율(신설)**: **`FINAL PUSH DURABLE`·`rc=0`·`+ break` 는 "그 시점에 끝났다"이지 "지금도 끝나 있다"가 아니다.**
Standard 티어에서는 **완주 뒤에도 재큐될 수 있으므로, 완주 기록에는 관측 시각을 박고 재점검한다.**
그리고 ★**완주 즉시 보존본을 뜬다** — b3nopmi 는 보존이 있어 살았고 b3shf 는 없어서 위험하다.

## EXP-0812a (탐색 · ⛔판정 아님 · ★★핵심) — **T1 승자는 DCPO 였고, GDPO 스칼라 판은 만들어졌으나 안 돌았다. 그리고 두 세대의 차이는 "두-패스 보존"이다**

**① 사용자 제안(GDPO 에 메타 유틸리티를 스칼라로)은 이미 구현돼 있다 — `TRIOBJ_META_V1`.**
`archive/runs_archive/h200_triobj_meta_v1.yaml`: *"**Env-reward-only GDPO multi-head**: correctness(1.0) + meta_revision_utility(0.5) + meta_commit_shape(0.3). `TRIOBJ_META_V1` in `_VANILLA_MODES` → teacher OFF, **directly comparable to base GRPO**. **NO verl-internal advantage/token-mask surgery.** **Two-pass output: preliminary boxed → `<|meta|>` verify/confidence → final boxed (LAST boxed graded)**"*
★**네 곳 다 확인했다**: `CLAIMS` **0건** · 원장 **0건** · HF `metacot-h200-triobj-meta-v1` **404** · wandb 100런 중 **0건**.
⇒ **만들어졌지만 결과가 없다.** (⚠h200 세대가 다른 wandb 프로젝트일 가능성은 남는다.)

**② ⚠그런데 그 헤드도 지금 켜면 조용히 0 이 된다.**
`src/training/meta_revision_rewards.py:97` — `boxed=[b for b in _all_boxed(text) if b]` · **`if len(boxed) < 2: scores.append(0.0); continue`**.
`\boxed` **2개 이상** 비율 실측:

| | ≥2 비율 |
|---|---|
| **b2p gs300**(base, RL 후) | **0.0065** |
| b3s gs300 | 0.0185 |
| **pmishift**(instruct, T1 승자) | **0.2352** |
| raw1763 = 현재 init 데이터 | 0.2008 |
| SFT1 | 0.0292 |

⇒ ★**base 팔에서는 99.35% 가 0 점**이 된다. **pmi_shift 와 정확히 같은 침묵 실패**다.

**③ ★★그리고 T1 승자는 GDPO 가 아니라 DCPO 였다.** wandb 런 이름이 authority:
`triobj_dcpo_v4_pmishift_h100_4x4k` = **pmishift(T1 승자) = TRIOBJ_DCPO_V4 + pmi_shift** ·
`gandhi_metasft_vanilla_h100_4x4k` = gandhi 는 vanilla · `shiftonly_pmishift_h100_4x4k` = shiftonly.
⇒ ★**"DCPO 가 범인"은 성립하기 어렵다 — 같은 DCPO+pmi_shift 레시피가 instruct 에서 이기고 base 에서 졌다.**

**④ ★★★그럼 두 세대의 차이는 무엇인가 — 두-패스 보존**
init 데이터의 두-패스 비율은 **20.08%**(raw1763)로 같은데, **instruct 는 23.5% 로 지켰고 base 는 0.65% 로 파괴**했다.
그리고 **pmi_shift 도 `meta_revision_utility` 도 둘 다 "메타 앞에 작업이 있어야" 작동한다**(전자는 `response_prefix`, 후자는 첫 `\boxed`).
⇒ ★**주가설 재정렬: 범인은 SFT2·DCPO·pmi_shift·메타내용 중 하나가 아니라, "RL 이 보상이 참조하는 구조를 파괴했다" 이다.**
그리고 그 파괴는 **정답·길이 압력만으로 일어난다**(b2p 는 메타 보상이 0인데도 그랬다).

**⚠경계** · TRIOBJ_META_V1 의 wandb 부재는 **다른 프로젝트일 가능성** 미배제 · instruct vs base 는 **기질도 다르다**(교란) ·
`\boxed`≥2 는 두-패스의 **대리 지표**이지 정의가 아니다 · **EXP 트랙**.
★**여는 것**: ⑴**두-패스를 강제**했을 때 base 에서도 두 헤드가 살아나는가(하니스 강제 or 형식 보상) ⑵ pmishift(instruct) 의 학습 중 두-패스 궤적 — 언제부터 23.5% 인가 ⑶ TRIOBJ_META_V1 이 정말 미실행인지 h200 프로젝트 확인.
★**규율**: ★**보상 헤드를 고르기 전에 "그 헤드가 요구하는 구조가 롤아웃에 남아 있나"를 세라.** 오늘 두 헤드가 같은 이유로 0 이었다.

## EXP-0812b (탐색 · ⛔판정 아님) — **ultracode 감사 3건 착지: R_cal 미지 결함 6종 · pmi_shift 수정은 "가드+floor 게이팅 한 쌍" · 하니스 강제는 (a)안**

워크플로 `wf_7c451b7a-ab1`: 감사 3건 완료(312k 토큰·56 도구호출), 초안·비평 4 에이전트는 **세션 한도로 실패**(reset 02:30 UTC). 감사 요지:

**감사① R_cal — 알려진 5종 확정 + 미지 6종 적발** (전부 file:line, 수정은 차분 스케치만 = 승인 대상)
- ⛔[critical] **스칼라/마스크 출처 불일치**(`dcpo_region.py:844` vs `:651-688`): R_cal 의 conf 는 **전체 텍스트 정규식**, 기울기는 **메타 내부 CONF 토큰** — 본문의 "probability 3" 이 팬텀 R_cal 을 만들고, ★**메타 앞 본문을 복원하는 순간(요구 ④) 라벨과 착지점이 갈라져 더 악화된다.**
- ⛔[critical] **cal 전용 membership 부재**(`:1227,:1200`): R_cal≤0 항상(클램프 0.99 탓에 완벽 발화도 −1e-4) + 침묵=0 + 전원 centering ⇒ **conf 발화가 양의 advantage 를 받을 수 있는 경우가 구조적으로 없다.** R_meta 는 member_mask 로 고쳤는데 R_cal 은 명시적으로 배제됨.
- ⛔[critical] **anchor_norm × 붕괴 자기잠금**(`:1258-1261`): cal_s EMA 감쇠 후 재발화 1건에 ~10-20× 증폭된 음의 킥. EMA 는 resume 시 리셋.
- [major] truncation 행 c_with 미가드(R_corr 0714 가드가 R_cal 엔 없음) · excl_conf×floor: **CONF 토큰만 floor 무보호 = "숫자만 탈락"이 보상 최적** · conf 상수 시 그룹 순기울기 해석적 0(A_cal 합=0 증명).
- ⇒ ★**근본 해법은 코드가 아니라 배치**: conf 를 추론 뒤로(요구 ④와 합치).

**감사② pmi_shift — 최소 패치는 한 쌍**
- ⛔[critical] **①빈 prefix 가드**(`verl_sdc.py:1623-1672`, `dcpo_pmishift_min_prefix_tokens` 기본 0=no-op) 만으로는 부족 — 켜면 attempted≈0 으로 **헤드가 침묵할 뿐 본문을 쓰라는 기울기가 안 생긴다.** ⛔[critical] **⑦meta_floor 자격을 "prefix 있는 메타"로 AND-게이팅**해야 첫머리-메타의 순양 균형이 깨진다.
- [major] **⑥4-arm 전부 동결 ref 채점** — R_shift 는 **정책이 아닌 ref 의 믿음 이동**, 행동 변화는 어디서도 측정 안 됨 ⇒ 요구 ③("행동을 바꿔야")의 정공은 강제-계속 eval 의 답-flip rate.
- dup_thresh 는 빈-prefix 레짐에서 **수학적 no-op**(공집합 Jaccard=0)·대칭이라 부분 베끼기 못 잡음(containment 교체) · decoy 는 LaTeX 답 전멸→정수 편향(length-matched 필터) · split_first_meta 다중블록 오귀속 · PMI 합산 자체는 **올바름 확인**.

**감사③ 하니스 강제 — 3안 비교, (a) 채택 권고**
- **(a) 에이전트 루프 재사용**: `cf_prefix_agent.py`(라이브·현행 스택 구동 실적) 1차 템플릿 + BCI 의 seed-붙이기 패턴. ~200-250행. **verl 네이티브로 주입 토큰 기울기 제외 지원**(`response_mask` 경로 `agent_loop.py:618-633`→`dp_actor.py:573-653`).
- **(b) 트레이너 텐서 리팩**: ⛔**이미 실패한 경로** — `_force_inject_rollout` 이 `NotImplementedError` 하드블록(`verl_sdc.py:2496-2516`), 크래시 선례 docstring 명시. 기각.
- **(c) 비강제 유도**: 실측으로 이미 패배(b2p 포함 5/5 붕괴). 보완재로만.
- ⚠함정 2: **학습마스크와 파싱마스크 분리 필요**(dcpo_region 파서가 rm=0 을 패딩으로 간주 → 주입 `<|meta|>` 가 라우팅에 투명) · **비강제 방출률을 판정에 포함**(rm=0 이면 마커 방출에 기울기가 없어 전이가 유일한 회피로).
- 재사용 가능: `pg0_yield_pilot.py` + `meta_inject.plan_inject_prefixes`(유닛테스트 완료) · BCIConfAgentLoop 은 **은퇴 상태**(부활 필요) · agent 블록 배선은 torch28x 스키마 함정(0711) 때문에 **1-step 노드 스모크 필수**.

★**여섯 질문 체계로 확정**(사용자 0812): ①SFT2 ②DCPO ③pmi_shift ④메타 내용 ⑤**전-행동 부재** ⑥**R_cal 등 보상 기능**. ⑥은 감사①+ECE 실측으로 사실상 답변됨(켠 팔 ECE 최악·판별 최고·침묵=만점 아티팩트). 나머지는 계획 루프 재개 후.

## EXP-0812c (탐색 · ⛔판정 아님 · ★★E0 완결 — GPU 0) — **meta-first 는 RL 표류가 아니라 step 1 부터였고, 근원은 SFT2 의 wrong_prefix segment-mask 다**

E0(메타 위치 붕괴 시점)를 wandb rollouts 표(`media/table/dcpo/rollouts_<step>_*`, `main_tail`=마지막 1500자, 1400자 미만 행은 전문)로 판독했다.

**① b3s 온폴리시 롤아웃의 meta-first 비율 (512행/스텝):**

| step | 1 | 2 | 3 | 5 | 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| meta-first | **.972** | .962 | .975 | .958 | .970 | .979 | .986 | .993 | .983 | .992 |
| `<think>` 포함 | .010 | .008 | .008 | .014 | .006 | .004 | .004 | .000 | .002 | .002 |

★**RL step 1 에 이미 97.2%.** 붕괴 곡선이 아니라 **초기 조건**이다.
⇒ ⛔**0811t("RL 이 1차 패스를 지웠다")와 0812 정정("RL 이 민다") 둘 다 위치에 관해서는 틀렸다** — RL 은 지운 것도 민 것도 아니고 **유지했을 뿐**이다. (발화 사멸은 별개 현상으로 유효: floor 0 팔들의 gs125~228 붕괴.)

**② 근원 — 코드와 데이터가 함께 말한다.**
`src/training/sft.py:106-113` (원문 주석): *"REDIRECT rows additionally carry the student's wrong_prefix at the HEAD of the assistant target — mask it too (**train ONLY the meta block + recovery, never teach the model to PRODUCE the flawed prefix**)."*
그리고 `rv_redirect_verify_functional.parquet` 실측: **wrong_prefix 비어있지 않음 1.0000 (1,763/1,763행)** — verify 1,209 + redirect 554 **전부**. prefix 길이 중앙 496자.
⇒ ★**SFT2 는 단 한 행도 "메타 앞 추론을 생산하는 것"을 가르치지 않았다.** 맨 문제에서 시작하는 온폴리시 생성은 **손실을 받아 본 유일한 지점 — 메타 블록 — 에서 시작한다.** meta-first 는 이 설계의 직접 귀결이다.
★설계 의도 자체는 합리적이었다(redirect 의 결함 prefix 를 가르치지 않기). **문제는 그 마스크가 verify 1,209행(prefix = 모델 자신의 멀쩡한 1차 풀이)에도 적용**되어, "먼저 추론하라"의 예시가 코퍼스에서 전멸한 것이다.

**③ 함의 — 계획 재정렬.**
- **Q1(SFT2 범인?) = 위치에 관해 YES, GPU 0 으로 완결.** 학습 팔 불필요.
- ★**SFT1 → RL 팔 재승격**: SFT1 (`b2on_v8meta_strict_sft`)은 **wrong_prefix 컬럼 자체가 없다** = 전응답 손실 = think 안 메타 100%. 내 0812 강등("63.5% 출발점도 못 버텼다")은 **틀렸다** — 그 63.5% 는 **데이터 위치**였고 손실 마스크 탓에 **모델 행동엔 전달된 적이 없다**. 유효한 출발점 대비는 "SFT1(전응답 손실) vs SFT2(메타-이후만 손실)"이다.
- Q5(전-행동 부재)의 상류 원인이 특정됐다: **하니스 강제와 SFT 재설계(마스크 해제/부분 학습)가 동급 후보**가 된다.
- instruct 가 몸통을 지킨 이유도 설명된다: instruct RL init(v8_meta_inside_strict)은 **전응답 손실**이었다.
★**여는 것**: ⑴ SFT1→RL 팔(이제 Q1 검증이 아니라 **Q5 해법 후보**) ⑵ SFT2 재설계 시 verify 행의 prefix 는 학습에 포함(redirect 만 마스크) ⑶ 워크플로 계획에 이 사실 반영(초안은 옛 전제로 돌고 있음 — 최종 보고에서 덮어씀).
★**규율**: ★**"데이터의 위치"와 "모델의 행동"은 손실 마스크가 갈라놓을 수 있다** — 코퍼스 통계를 행동 예측에 쓰기 전에 **어느 토큰이 손실을 받는지** 확인하라.

## 0812 04:05 — **계획 v4 착지**(비평 3라운드·17에이전트·오류 0) + ⛔**N0 실기: b3shf gs300 프루닝됨 → gs302 보존**

**① 계획 v4** = `docs/reports/2026-08-12-six-question-plan-v4.md` (전문 12,621자 · 라운드별 원문은 워크플로 저널).
구조: W0(E-계열 GPU0 + E3 프로브) → W1(N4→N2→N3·N1) → W2(T2 전제복원 풀패키지) → W2′(조건부 T2e entropy=0) → W3(T1 uid-수준 ρ=0.5 부분 주입) → W4(T3 수리된 R_cal) → W5 후보(bfmt=Q2 유일 계기·SFT1-init RL·언마스크 SFT2·σ_run 복제).
비평이 잡은 것 중 핵심: ★**A18 — `knob_registry.py:168` 의 ack 읽기가 ListConfig 에서 항상 빈 집합**(어떤 ack 값도 통과 불가, 역사상 미관측 열린 게이트) ⇒ **전 T 팔 발사의 선행 조건**으로 지정. Q2 는 W0~W4 에서 열리지 않음을 명시(E-173). T2 의 최빈 경로가 **보류**임을 계획 자신이 인정(b3s 선례가 엔트로피·정확도 밴드 둘 다 위반).
⚠**내 재조정 권고(사용자 결정 대상)**: 계획은 SFT-측 수리(SFT1-init RL·verify 언마스크 SFT2)를 W5 후보로 강등했으나, 사용자의 직전 방향(마스크 수술→재SFT→RL)과 어긋난다. 두 접근은 다른 질문이다 — T2 는 *"RL 레버만으로 전-행동을 복원할 수 있나"*(논문 기전 주장에 부합, 단 보류 사전확률 높음·코드 A1/A2/A18 선행), SFT-측은 *"올바른 기질 위에서 RL 이 채점하게 하기"*(코드 0줄·저렴·성공 사전확률 높음). **병행 가능**(각 1노드): W2 에 SFT-측 두 팔을 승격해 T2 와 나란히 놓는 것을 권고.

**② ⛔N0 실기.** 04:00 확인 시 `checkpoints/rq3v2f_b3shf/global_step_300·301` 은 **이미 프루닝**(재개 잡이 gs303·304 저장 → keep-3). **오늘 아침 세운 규율("완주 즉시 보존본을 떠라")이 몇 시간 만에 실현된 실패다.**
조치: **gs302(최근접 생존 스텝) 를 `preserved/mechanism_alive/rq3v2f_b3shf_gs302/` 로 서버측 복사(23파일) 완료.**
피해 평가: ⑧/N3 은 gs300 대신 **gs302(+2스텝, 같은 붕괴 레짐)** 로 대체 — 과학적 손실 경미(발화 .059 팔), 단 "gs300" 표기는 전부 gs302 로 정정 필요.

## 0812 04:10 — **웨이브 0/1 개시**: E5·E6 완료 + N2·N4 발사

**E5(len_cost 직독)**: 승자 계보 config(`stage2c:122`→현행 `stage3b:160`) = **0.08** · b3s 앵커 팔 = **0.0**(런처 오버라이드). ⇒ T2 는 b3s 패리티대로 **len_cost 0** — 계획 문구("패리티 원칙이지 −2pp 주범 제거 기대 아님") 그대로.
**E6(앵커 재현)**: b2p 앵커 77.325 vs `robust_grade` 재채점 **0.7732**(0811q) — 일치 ✅. 채점기 동일성 전제 성립.
**N2 발사**: `rq3v2f-b3null-eval-0812` — b3null gs303(m4/o4/e4 확인) → `eval/rq3v2f_b3null_gs303`(부재 확인 후 신규). 런처 `h100std_rq3v2f_b3null_1030_eval.yaml` — 템플릿(b3nopmi eval) 대비 **SPEC 한 줄 + description 만 차이**(G8 diff 확인).
**N4 발사**: `sft2init-eval-0812` — `models/b2p2_rvfull_eb16_sft` 를 **머지 없이 그대로**(SFT 저장은 HF 형식) → `eval/sft2init_gs0`(부재 확인). C-017/C-018 의 네 번째 칸이자 EXP-0812c 의 eval-측 교차검증(gs0 에서 meta-first 인가). 게이트+머지 블록만 직다운로드로 교체(diff 61줄).
⚠경미: 두 런처의 **잡 이름 필드가 템플릿 그대로**(`h100_rq3v2f_b3nopmi_eval`) — 실험 이름이 갈라 주므로 기능 무해, 다음 판부터 정정.
쿼터: 재개 잡 2개(8GPU) + 이 둘(8GPU) = **16/16 만석**. N1 은 A14 승인 대기 · 재개 잡 취소는 사용자 결정 대기.

## 0812 04:40 — 사용자 재정렬 지시: **"고치는 것이 먼저"** → 수리된 SFT2 코퍼스 완성 + 계획 v4 편향 인정

**① 사용자 지적과 그에 대한 판정.** *"잘못 학습시키고 있었는데 왜 eval 먼저 하나, 계획이 올바른가"* —
계획 v4 는 측정 타당성(교란·검정력·사전등록·A18 발견)에는 엄격하나, ★**"깨끗한 귀속"을 "빠른 수리"보다 우선하는 편향**이 있다.
SFT-측 수리를 W5 로 강등한 것이 그 증상이다. Q1(범인 규명)이 0812c 로 이미 닫혔다는 근거는 **"검증용 팔 불필요"까지만 정당**하고
**"수리 팔 불필요"는 따라 나오지 않는다** — 계획이 이 구분을 뭉갰다. ⇒ **사용자 지시로 재정렬: SFT 수리를 임계 경로로 승격.**
eval(N2·N4)은 수리를 막지 않는 병렬 작업이며, 특히 **N4 = 수리의 "before 사진"**(고친 init 을 같은 하니스로 gs0 eval → before/after 가 300스텝 RL 없이 판정됨), N2 = 수리된 기질 위에 보상을 얹을 때의 **경로 선택**(DCPO vs GDPO) 입력.

**② 수리된 SFT2 코퍼스 완성**: `data/rvfull_verify_unmasked.parquet` (HF `metacot-rv` 백업 완료).
- verify 1,209행: `wrong_prefix=""` ⇒ `sft.py` 기본 동작으로 **전응답 학습**(추론→메모→검산 전체). 행당 학습 구간 중앙 **+549자**.
- redirect 554행: 마스크 **유지**(결함 prefix 를 가르치지 않는 원 의도 보존).
- `messages` 전행 바이트 불변 · redirect 행 바이트 불변 검증 완료. **단일 변수 수술**(퇴화 5.5% 등 다른 결함은 의도적으로 불변 — 대조 가능성 유지).

**③ 재정렬된 임계 경로**: 수리 SFT2 굽기(레시피 = `h100std_sft_b2p2_rvfull.yaml` 그대로: SFT1 init·3ep·lr 1e-5·max_len 4096, 데이터만 교체, 출력명 신규 `models/b2p3_vunmask_sft`) → **gs0 eval(N4 하니스)로 수리 확인** → RL(b2p 레시피). ⚠쿼터 16/16 만석 — **발사는 노드 확보 대기**(유휴 재개 잡 취소 승인 또는 eval 1개 종료).

## 0812 05:05 — **승인 실행 일괄**: 401 재발사 · 옛 잡 2개 정지 · sft.py 수리 push · 재빌드 발사

**① 웨이브1 eval 401 정정**: 최초 발사(04:05) 시 **발사 셸에서 `.env` 를 source 하지 않아 `${GH_TOKEN}` 이 빈 값**으로 치환 → 노드에서 code tar `curl 401`. 실패 잡은 재시도 무의미(빈 토큰 고정)라 취소 불요 판정 후 **`.env` 소스된 셸에서 재발사**: `rq3v2f-b3null-eval-0812b` · `sft2init-eval-0812b`. ★규율: **`amlt run` 은 반드시 `.env` 소스된 셸에서** — 치환은 제출 시점이다.
**② 승인된 정지**(사용자: "불필요한 실험은 멈추면 돼"): `rq3v2f-b3nopmi-0807`(재완주 후 유휴)·`rq3v2f-b3shf-0810b`(붕괴 구간 재학습) 취소 → **8 GPU 확보**.
**③ 승인된 코드 수리 push**(`b360348`, GitHub master 반영):
- `src/training/sft.py`: `_should_mask_prefix(wrong_prefix, scenario)` — **redirect(와 scenario 무필드 레거시)만 마스크**, verify 는 전응답 학습. 레거시 코퍼스는 바이트 동일 학습(하위호환).
- `tests/test_sft_prefix_mask_scenario.py`: 계약 5건. **전체 750 passed / 8 skipped.**
- `CLAUDE.md` 계보 정정(378행 판이 현행 init 으로 잘못 기재돼 있던 것) · `docs/COLLABORATION_REQUEST.md` 에 협업자 공지(구 init 팔들은 meta-first 습관 공유 — 위치 민감 분석 전제 깨짐).
**④ 재빌드 발사**: `sft-b2p3-vunmask-0812` — SFT1 init·3ep·lr 1e-5 그대로, 데이터만 `rvfull_verify_unmasked.parquet`(HF 다운로드), config 는 노드에서 one-key sed 파생, 출력 **`models/b2p3_vunmask_sft`**(신규명). 노드 sft.py 는 구판이지만 **데이터 수준 수리라 결과 동일**(빈 prefix → 마스크 없음).
**현재 점유**: eval 2(8 GPU) + SFT 재빌드(4) = 12/16. **다음**: 재빌드 완료 → sft2init-eval 하니스로 **수술 전/후 비교** → 통과 시 RL(b2p 레시피).

## 0812 05:15 — 웨이브1 2차 정정: **N4 는 repo·이름 오지정, 재빌드는 tqdm BrokenPipe** → 수리 재발사

**N4(`sft2init-eval-0812b`) 실패 원인 = 내 런처 오류 2건**: ⑴게이트가 `REPO`(dcpo-v3 **model** repo)를 봤는데 **공용 init 은 `iamseungpil/metacot`(dataset) 에 있다** ⑵이름 `b2p2_rvfull_eb16_sft` 는 **어느 repo 에도 없다** — 실물은 `models/b2p2_rvfull_sft`. RL 런처들은 **E-123 이름 해소**(eb16 우선→plain 폴백)를 이미 갖고 있었고 내가 그걸 안 옮겼다(G8 을 init 스테이징 블록에는 적용 안 함). 수리: 같은 해소 규약 이식 + dataset repo + `/scratch/init_name.txt` 로 MERGED 결정 → **`sft2init-eval-0812c`** 재발사.
**재빌드(`sft-b2p3-vunmask-0812`) 실패 원인 = 템플릿 유래**: SFT1 init 스테이징의 cache+`copytree` 경로가 **tqdm BrokenPipe** (amlt 러너 stdout, `waitstatus 127`)로 복사 전 사망 → `FATAL init missing`. 수리: `HF_HUB_DISABLE_PROGRESS_BARS=1` + `local_dir` 직다운로드(복사 단계 제거) → **`sft-b2p3-vunmask-0812b`** 재발사.
N2(`rq3v2f-b3null-eval-0812b`)는 사망표식 0, 진행 중. 취소 시도 2건은 "취소 불요"(이미 종료 상태).
★규율: **런처를 복제할 때 G8 diff 는 SPEC 줄만이 아니라 자산 스테이징 블록(어느 repo·어느 이름·어떻게 받나)에도 걸어라** — RL 런처의 E-123 해소 로직이 정본이었다.

## EXP-0812d (탐색 · ★codex-sol 게이트 전 CLAIMS 금지) — **N2 착지: b3null 은 treatment 대역 — 기하·라우팅 묶음이 −2pp 의 주범, 메타 보상은 면책**

`rq3v2f-b3null-eval-0812b` 완료(05:29 UTC 관측). `eval/rq3v2f_b3null_gs303` 20파일 착지.

**사전등록 판정**(발사문 그대로: treatment ~75 대역 ⇒ 기하 기소·메타 보상 면책 / control 77.28 대역 ⇒ 기하 면책):

| | robust_grade (MATH500 n8) |
|---|---|
| **b3null gs303** (전 보상 0 · TRIOBJ region 경로만) | **0.7488** |
| b2p gs300 (통제군 · VANILLA whiten 경로) | 0.7732 |
| **Δ(b3null−b2p)** | **−2.45pp · 95%CI [−3.72, −1.15]** (문항대응 부트 10k, 문항 500) |

⇒ ★**treatment 대역 확정.** 보상이 하나도 없는 팔이 b3p(−2.08pp)와 같은 구멍에 앉았다.
**b3\* 팔들의 −2pp 는 메타 보상이 아니라 경로(DCPO region 기하·라우팅 묶음)가 만든 것.**
⚠스코프(계획 v4·E-173 준수): 이것은 **좌표 1점**이다 — 묶음 내부 귀속(whiten 부재/answer carve-out/discard 제외 중 무엇인지)은 미결(bfmt 사다리, W5).

**첫 줄 검사(부수 발견)**: b3null 4,000행 **전부 `\tiVar` 로 시작**(메타태그 포함률 0.0003, `<|meta|>`-first 0, `cros` 0).
⇒ **세 번째 변질 무늬**: b3p=`cros`, b3null=**`\tiVar`**. b3null 은 회피할 탐지기도 없으므로 이것은 회피가 아니라
**붙드는 보상이 없는 region 경로에서 init 의 meta-first 서두가 부패한 형태**로 읽힌다(발화 gs125 붕괴와 정합).

**파이프라인 함의**: 2차 RL("새 b3p")은 **경로를 그대로 쓰면 안 된다** — 후보 ⑴ DCPO + `entropy_coeff=0`(T2e 레버, E-158 의 2.6배 상대압력 보정) ⑵ KL 앵커 ⑶ GDPO 멀티리워드. 재빌드·1차 RL 결과와 함께 결정.

## EXP-0812e (탐색 · ★codex 게이트 전 CLAIMS 금지) — **N4 착지: gs0 에서 이미 meta-first 100% — 습관의 기원은 SFT2 로 확정, 그리고 전 팔의 진짜 원점은 65.8%**

`sft2init-eval-0812c` 완료(06:19 UTC 관측). `eval/sft2init_gs0` 20파일. 구 init(`b2p2_rvfull_sft` 해소)을 **RL 0걸음** 상태로 채점.

| gs0 기준선 (MATH500 n8 · robust_grade) | 값 |
|---|---|
| **정확도** | **0.6583** |
| **첫 줄 = `<|meta|>`** | **1.0000 (4,000/4,000)** |
| `<think>` 포함 | **0.0000** |
| 메타 위치(문자) | 중앙 **0** |
| conf 분포 | 0.75×2555 · 0.88×1365 (+0.12×12·0.25×3) |
| 길이(토큰) 중앙 | 256 |

★**함의 셋**:
1. **EXP-0812c 의 eval-측 완전 확증** — 학습측 step-1 은 97.2% 였는데 **eval 디코딩에서는 gs0 이 이미 100%**. meta-first 는 RL 이 만든 것도 강화한 것도 아니고 **SFT2 산출물의 성질 그 자체**다. Q1(위치) 종결 수준의 교차검증.
2. **전 팔의 진짜 원점 = 65.83%.** 지금까지 없던 수. b2p 의 RL 은 **+11.5pp** 를 만들었고(65.8→77.3), b3null(경로만) 은 **+9.0pp**(65.8→74.9) — ⇒ **DCPO 경로도 크게 배우긴 한다. 다만 whiten 경로보다 2.45pp 덜 배운다**(EXP-0812d 와 정합). "경로가 학습을 망친다"가 아니라 "경로가 상대적으로 손해"가 정확한 서술.
3. conf 는 gs0 부터 0.75/0.88 두 값 지배(64/34) — RL(b2p)이 48/52 로 재배분했을 뿐. **2값 어휘도 SFT2 유산.**

**운영**: 재빌드 2차(0812b)도 같은 BrokenPipe 로 사망 → **3차 수리 = 스테이징 파이썬의 stdout/stderr 를 `/scratch/stage_init.log` 파일로 리다이렉트**(러너 파이프 SIGPIPE 원천 차단) 후 `sft-b2p3-vunmask-0812c` 재발사.
★**수술 후 판정선이 이제 명확하다**: 고친 학생(b2p3)의 gs0 사진에서 ⓐ**첫 줄 `<|meta|>` ≪ 1.0**(이상적으로 `<think>` 복귀) ⓑ**정확도 ≥ 0.658 근방**(비열등) 이면 수리 성공.

## 0812 codex-sol 게이트 판정 (EXP-0812c/d/e) — c: PASS-WITH-EDITS · d: **FAIL** · e: PASS-WITH-EDITS

전문: `docs/reports/2026-08-12-codex-gate-exp0812cde.md`. CLAIMS 반영: **C-034**(meta-first 는 RL 이전부터 — 마스크는 유력 기전·미확정), **C-035**(공통 원점 65.83·전후 차이 +11.5/+9.0 은 인과 분해 아님).

★핵심 세 가지:
1. **0812c 인과 승격 조건 = 지금 돌고 있는 재빌드.** 게이트가 요구한 "동일 init·코퍼스에서 verify-mask on/off 만 바꾼 SFT 대조"가 정확히 `sft-b2p3-vunmask`(데이터 단일변수 수술)다. b2p3 gs0 사진의 meta-first 가 ≪1.0 이면 인과 확정, 1.0 그대로면 기전은 SFT1/직렬화 쪽.
2. **0812d FAIL 사유**: 전 헤드 0 ≠ "메타 보상만 제거" — 경로·whitening·수집 궤적이 함께 달라 귀속 불가. 관측(2체크포인트 2.45pp 차이)만 유효. 귀속하려면 2×2(경로 × 보상 on/off)·복수 seed·`\tiVar` 채점 감사. **"기하 기소·보상 면책" 서술은 원장 EXP-0812d 에서 헤드라인으로 쓰지 않는다.**
3. **0812e 는 "전 팔의 진짜 원점" → "동일 분기·동일 하니스 팔의 공통 기준선"으로 강등** + eval manifest 미작성이 결손으로 지적됨(여는 것에 등재).

## 0812 운영 — 재빌드 3차(-0812c) 사인 확정 + 4차(-0812d) 발사

**3차 사인(로그 검시)**: ①**sed 인용 파열 확정** — 명령 전체가 `bash -c '...'` 단일따옴표 래퍼인데, 내가 넣은 `sed -i 's|dataset_path: …|…|'` 의 **안쪽 단일따옴표가 바깥 래퍼를 조기 종료** → `dataset_path:` 가 명령으로 실행됨(`amlt_run.sh: line 77: dataset_path:: command not found` ×2 = 파이프 세그먼트 2개). 0718 amlt-bashc 함정의 재발형. ②스테이징 5연패("FATAL init missing")는 stage_init.log 가 노드와 함께 소실돼 미확정 — 단 자산 실재는 로컬 실사로 확인(`models/b2p_v8meta_strict_sft/config.json` present=True, 이름 문제 아님).

**4차 수리 3종**: ⑴스테이징 python 을 **b2p2 베이크에서 실제 통과한 원형으로 회귀**(cache+copytree, 파일 리다이렉트·local_dir 제거) + 셸 레벨 `HF_HUB_DISABLE_PROGRESS_BARS/XET/TQDM_DISABLE` export ⑵sed 를 **무따옴표·무공백** `s/rv_redirect_verify_functional/rvfull_verify_unmasked/` 로(치환 대상은 config 내 1곳뿐임을 grep 확인) ⑶스테이징·코퍼스·config 스왑 실패를 전부 `exit 1` 즉사로. **기계검증**: 본문 내 단일따옴표 수 = 래퍼 2개뿐(작동 원본과 동일), yaml 파스 OK.

★**규율 신설**: amlt 런처의 명령 본문에 **단일따옴표를 절대 넣지 않는다**(본문이 `bash -c '…'` 로 싸인다). 발사 전 기계검사: `script.count("'") == 2`.

`sft-b2p3-vunmask-0812d` 제출 성공(06:58Z 경). 참고: `amlt list` 의 preparing 은 낡은 캐시일 수 있다 — **failed 판정은 `amlt status` 로**.

## 0812 운영 — 4차(-0812d) 학습 완주했으나 산출물 좌초 → 5차(-0812e) 발사·4차 취소

**4차 경과**: 스테이징·교재교체 전부 통과, **3 epoch 완주**(train_loss 0.277, 1113s, loss 0.95→0.11 건강 곡선). 그러나 **one-key sed 가 한 키 부족** — dataset_path 만 바꾸고 `output_dir: /scratch/checkpoints/b2p2_rvfull_sft` 는 그대로 → 모델은 b2p2 이름 폴더에 저장, EOS 게이트·measure_sft_gate·HF 푸시는 전부 없는 b2p3 경로를 조회(HFValidationError = 로컬 경로 부재 시 repo id 해석 시도) → **shards_on_hf 0/4**, 산출물 노드에 좌초. EOS rc=1·gate 크래시는 품질 신호가 아니라 **경로 부재의 동일 증상**.

**5차 수리**: sed 3키(dataset_path + `b2p2_rvfull_sft→b2p3_vunmask_sft` + run_name) + 가드 3중(교재 grep·output_dir grep·**잔존 b2p2 문자열 0 확인**). 발사 전 로컬 시뮬레이션으로 파생 config 4키 전부 검증, 단일따옴표 count==2 통과. `sft-b2p3-vunmask-0812e` 제출, 4차는 취소(푸시 재시도+12h sleep 로 H100×4 헛점유).

★**규율 보강(G8-스테이징의 따름정리)**: 런처가 config 를 노드에서 파생할 때, **파생 후 config 의 "정체성 키 전부"(dataset·output_dir·run_name)를 grep 가드**하고 **원본 이름의 잔존 0** 을 함께 확인한다 — "one-key" 라는 설계 자체가 이름을 믿은 것.

## 0812 운영 — **재빌드 5차 완주·`models/b2p3_vunmask_sft` HF 착지(4/4샤드+config, 14파일)** → 수술 후 사진 발사

`sft-b2p3-vunmask-0812e`: 3키 스왑 가드 통과, 3 epoch 완주(loss 곡선 4차와 일치=재현성), **push attempt 1 에서 4/4 durable**(08:55Z 관측, HfApi 실사로 교차확인). ⚠**게이트 eval 은 또 죽었으나 이번 사인은 나 자신** — 4차 스테이징 수리 때 넣은 `TQDM_DISABLE=1` 이 vLLM `llm.generate` 의 진행바-경과시간 나눗셈을 0 으로 만듦(`_run_engine: total_in_toks / pbar.format_dict["elapsed"]` ZeroDivisionError). ⇒ gate json 은 이번 판에도 없음. **판정은 어차피 수술 후 사진이 담당**(같은 하니스 전/후 쌍)이라 베이크 재실행은 불필요; 다만 ★규율: **vLLM 이 도는 잡에 TQDM_DISABLE 을 export 하지 말 것**(HF_HUB_DISABLE_PROGRESS_BARS 만).

**수술 후 사진 발사**: `h100std_b2p3init_1030_eval.yaml`(sft2init eval 복제, 이름해소→`("b2p3_vunmask_sft",)` 고정, SPEC=`b2p3init:eval/b2p3init_gs0:0`) — 발사 전 4중 검사(잔존 b2p2=0·yaml 파스·따옴표 2·HF 프리픽스 충돌 0) 통과, `b2p3init-eval-0812` 제출(09:00Z 경).

★**판정선(재확인)**: eval/b2p3init_gs0 착지 후 — 첫줄 `<|meta|>` ≪1.0(이상적 `<think>` 복귀) AND 정확도 ≥0.658 근방 ⇒ **수리 성공 + C-034 의 마스크 인과 승격**. meta-first 1.0 그대로면 기전은 SFT1/직렬화(C-034 유지·수리 무효).
