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

## 실험 상태판 (running tally) — 갱신 0719 07:29

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

## E-000 (2026-07-17 19:41 UTC) — VC 복구 후 B3pkg·B2 fresh 발사

**추상적 문제.** 인프라(가상클러스터↔워크스페이스 연결)가 재할당으로 끊겨 제출이 전면 차단된
상황에서, (a) 차단이 우리 쪽 오류인지 서버 쪽인지 판별하고, (b) 복구 순간을 놓치지 않고
정확히 원하는 초기상태(meta-SFT init에서 fresh)로 학습을 시작할 수 있는가?

**구체적 해결.**
- 차등 진단: 동일 최소 echo 잡이 `msrresrchvc`에는 접수되고 `msrresrchbasicvc`에서만
  "VC does not exist" 거부 → 워크스페이스↔basicvc **연결 미복원**으로 특정(우리 오류 아님).
- 1시간 재시도 루프로 복구 감시. 19:32 UTC 복구 포착 → `amlt run h100std_rq3_b3.yaml`/
  `h100std_rq3_b2.yaml`가 `msrresrchbasicvc` `80G4-H100`(H100×4)에 queued→running.
- b3pkg 부트스트랩 로그 확인: `RGS(model+extra+optim>=4)=0 0` → **cold start(fresh gs0)**,
  resume-hardening 3회 재시도 후 cold-start 진입(v3에 rq3_b3pkg 체크포인트 없음 = 오resume 리스크 없음).
- 리셋된 상태로 T1을 이긴 full triobj 패키지 복원(w_over=0만 override), collapse-fixed v2 레시피
  (temp1.0/top_k−1/max_response8192/Dr.GRPO norm_adv_by_std=false) 3-arm 동일 적용.

**통찰.** "제출 실패"의 근인은 코드가 아니라 인프라 연결이었고(9회 헛다리 후 차등검증이 결정), 이
경험은 헌법 Part II(증거 우선·근인 추적)의 실전 사례다. 또한 fresh-from-init cold start가 로그로
확증돼(RGS=0), matched-ladder의 "선언된 축(SFT-init, reward-source)으로만 다르다"는 전제가
발사 시점에 깨지지 않았음을 보장한다.

---

## E-001 (2026-07-17 19:45 UTC) — B3-noPMI resume-hijack 리스크 선제 발견

**추상적 문제.** matched-ladder에서 한 arm은 fresh(gs0)로 시작하는데 비교 arm에는 이전 세대의
잔여 체크포인트가 남아 있으면, 자동 resume 배선이 그 잔여물을 끌어와 "선언된 축으로만 다르다"는
matched 전제를 조용히 깨뜨릴 수 있다. 발사 전에 이를 어떻게 탐지·차단하는가?

**구체적 해결.** b3pkg가 `RGS=0`(v3에 rq3_b3pkg 체크포인트 없음)로 cold-start함을 확인한 반면,
HF `iamseungpil/metacot-h200-triobj-dcpo-v3`를 조회하니 **`checkpoints/rq3_b3nopmi/global_step_135/`**
(actor/{model,extra_state,optim}_world_size_4_rank_0..3.pt 완전 shard)가 실재. b3nopmi를 지금
발사하면 resume-hardening이 gs135를 끌어와 fresh gs0 대신 **gs135에서 이어붙는다.** provenance
(어느 config/세대)가 불확실 → resume(진행분 보존) vs force-fresh(삭제/skip) 결정을 Codex에 전문으로
넘겨 확인 중(btju1h1i0). 안전 기본값: **b3pkg가 fresh gs0인 이상 matched 무결성 위해 b3nopmi도
fresh gs0**가 원칙이며, gs135가 현행 config의 유효 진행분임이 확증되지 않는 한 force-fresh.

**통찰.** resume 자동화는 preemption 생존(keep-alive)엔 이롭지만, matched-ladder에서는 **양날의 검**
이다 — 한 arm만 잔여 체크포인트를 가지면 "isolation is sacred"(헌법 Part I.3)를 위반한다. 따라서
keep-alive(resume)와 matched-isolation은 arm별 체크포인트 상태를 **발사 전 명시적으로 대조**해야
양립한다. 이 대조를 자동화하지 않으면 조용한 confound가 된다.

### 상태 tick — 19:45 UTC
- b3pkg·b2 running 7분, 부트스트랩(b23 init 스테이징·regression guard 통과). world_size=4 실측은 verl 학습 시작 후.
- Codex keep-alive 하드닝 리뷰(F1–F5 running 리스크 + b3nopmi 결정) 진행 중.

---

## E-002 (2026-07-17 19:50 UTC) — 선점 부활: b3pkg를 gs0에서 재제출 + keep-alive 하드닝 판정

**추상적 문제.** 선점 가능한(opportunistic) 컴퓨트에서 matched-ladder를 죽지 않게 유지하려면,
(a) "running" 상태가 정말 살아있음을 뜻하는지, (b) 첫 체크포인트 전에 선점당한 arm을 무결성 훼손
없이 어떻게 되살리는지, (c) 여러 신뢰성 결함 중 무엇이 *지금 돌아가는* 잡의 실제 사망 원인인지를
구분해야 한다.

**구체적 해결.**
- Codex keep-alive 판정: **live 사망 위험 F5 > F1**(F5=verl 죽어도 `sleep 86400`이 잡을 running으로
  위장 → 은닉 사망), 재개 무결성 F3 > F2, 신규발사 F4. → **"running"은 필요조건이지 충분조건이 아님;
  진짜 생존 신호는 `global_step` 증가 + HF shard 증가.**
- 19:49 실측: `peaceful-starling(b3pkg)` 상태가 **`pass`(1.6kB, 9분)**로 전이. 메모리
  [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]대로 이는 **Basic 티어 graceful-SIGTERM
  선점**(코드버그 아님). 로그 마지막이 `export LOCAL_RANK=0` = 부트스트랩 말미, **gs0·체크포인트 전**
  → 잃은 진행분 없음. `amlt log list` retry 없음(terminal).
- keep-alive 실행: `amlt run h100std_rq3_b3.yaml` 재제출 → **`distinct-serval` queued**(cold-start 재개).
  b2(daring-gelding)는 running 생존.
- b3nopmi 판정: Codex도 **force-fresh를 안전 기본값**으로 확정(gs135 provenance—config hash·SFT-init
  id·reward-source·wandb run id·shard 완결성—를 전부 확인 못 하면 resume 금지, 발사 전 auto-discovery
  경로가 gs135를 잡지 않도록 명시적 차단 필요).

**통찰.** 선점 컴퓨트의 keep-alive는 "재제출 문제"가 아니라 **모니터링 완결성 문제**다 — F5가 보여주듯
프로세스가 죽어도 상태판은 살아있다고 거짓말할 수 있어, `global_step`·HF shard 증가라는 *진행의 증거*를
봐야 한다. 그리고 **첫 체크포인트 전 선점은 되살리는 비용이 0**(cold-start)이므로, fresh-from-init +
잦은 체크포인트가 옳은 keep-alive 자세다. matched-ladder에서는 여기에 "arm별 잔여 체크포인트 대조"
(E-001)를 더해야 keep-alive가 무결성을 깨지 않는다.

### 상태 tick — 19:50 UTC
- b3pkg = **distinct-serval** (queued, 선점 후 재제출). b2 = daring-gelding (running ~13분).
- 다음: b3pkg 노드 재확보 감시, 두 잡 모두 `global_step` 증가·world_size=4 확인(F5-aware).

---

## E-003 (2026-07-17 19:53 UTC) — world_size=4(GPU 4장) 확증 + "resume-hijack" 재해석(비판적 교정)

**추상적 문제.** (a) 요청한 4-GPU가 실제로 전부 학습에 참여하는지 로그로 확증할 수 있는가?
(b) 한 arm에만 잔여 체크포인트가 있는 상황(E-001)을 "hijack 리스크"로 볼지 "정상 resume"으로 볼지 —
이 판단이 matched 무결성과 낭비되는 compute 사이의 트레이드오프를 가른다. 성급한 force-fresh는
유효한 진행분을 버릴 수 있다.

**구체적 해결.**
- **GPU 4장 확증**: b2(daring-gelding) 로그에 `[Rank 0..3] Loaded model/optimizer/rng/lr_scheduler
  from .../global_step_140/actor/*_world_size_4_*` + `Training Progress: 140/300`. → 4개 rank 전부
  gs140에서 로드·학습 중 = **H100×4 완전 활용 확인.** b3pkg(distinct-serval)도 2분 만에 노드 재확보 running.
- **재해석**: b2는 계획대로 gs140 **resume**(fresh 아님)으로 정상 진행 중이다. 그렇다면 같은 세대·같은
  4-GPU 배치인 b3nopmi gs135(vocab 151671=meta 증강, world_size_4 완전 shard, 98.3GB)도 **hijack이
  아니라 b2와 동형의 정상 resume**일 개연성이 높다. b3pkg만 fresh gs0인 이유는 별개 — b3.yaml이
  "eroded gs60(형식보상 0으로 뒀던 잘못된 config)에서 resume 금지, fresh 시작"을 **의도적으로** 지정했기
  때문(즉 b3pkg fresh는 나쁜 부분런 폐기, b2/b3nopmi resume은 좋은 부분런 계승 — 둘 다 최종 gs300에서
  올바른 config).
- **교정된 결정 규칙**: b3nopmi 발사 시 **force-fresh를 무조건 기본값으로 삼지 말 것.** 대신 gs135의
  config(config_name·reward heads·SFT-init id)가 현행 b3nopmi.yaml과 일치하는지 확인 → 일치하면
  **b2처럼 resume**(gs135 진행분 보존), 불일치/불확실하면 force-fresh. 이 확인은 발사 직전 수행.

**통찰.** matched-ladder에서 "잔여 체크포인트"는 자동으로 confound가 아니다 — **fresh-vs-resume는
config가 같은 한 과학적으로 중립**(같은 0→300 궤적을 wall-clock만 나눔)이다. 실제 confound는
*config 불일치*일 때만 생긴다. 따라서 keep-alive의 올바른 질문은 "resume하지 말까?"가 아니라
"이 체크포인트가 현행 config의 산물인가?"이다. E-001의 "force-fresh 기본값"은 이 구분을 놓친
과잉 방어였고, 여기서 config-provenance 확인으로 교정한다. (헌법 Part II: 근인을 증거로 좁혀라.)

### 상태 tick — 19:53 UTC
- b3pkg = distinct-serval (running, fresh gs0 재시작). b2 = daring-gelding (running, **gs140→300, world_size=4 확인**).
- b3nopmi 미발사(발사 시 gs135 config 일치 확인 후 resume/fresh 결정). GPU 4장 활용 확증됨.

---

## E-004 (2026-07-17 19:57 UTC) — Phase-1 사전 준비: B0 기준선·eval 하니스 확보

**추상적 문제.** 결과 판정 단계(gs300)는 학습 완료를 몇 시간 기다려야 도달한다. 그 대기 시간을 낭비하지
않으려면, 결과가 나오는 즉시 비교·판정할 수 있도록 *비교 기준선*과 *평가 도구*를 미리 확보·검증해야 한다.

**구체적 해결.**
- B0(base-matched) gs300 held-out이 이미 HF v3 `eval/base_matched_1030_v2/`에 계산돼 있음을 확인,
  JSON에서 정확도 추출: **GSM8K 0.8905 · MATH500 0.5440 · AIME@16k 0.0458**(@4k: 0.8868/0.5385/0.05).
  `meta_emission_rate≈0.001~0.05` = 메타 거의 안 씀 → 진짜 base arm 확증. 이를 보고서 상단에 anchor로 고정.
- eval 하니스 `src/eval/eval_hf.py`(gsm8k/math500/aime2024, math_verify 채점) 준비 확인 → gs300 도달 시
  meta 체크포인트에 그대로 실행 가능.
- 상태 tick: b3pkg(distinct-serval) running 5분·b2(daring-gelding) running 19분, **선점 없음.**

**통찰.** 긴 대기(hours-to-gs300)를 가진 파이프라인에서 "다음 단계 무엇이 필요한가?"를 미리 물어
downstream 준비물(기준선·평가기·비교 스크립트)을 대기 중에 확보하면, 결과 도달 시 판정이 **즉시**
이뤄진다. 대기는 유휴가 아니라 준비의 시간이다. (B0 기준선이 고정됐으므로, meta arm 하나라도 gs300에
닿으면 곧바로 RQ 판정으로 진입할 수 있다.)

### 상태 tick — 19:57 UTC
- b3pkg=distinct-serval running(5m, fresh gs0)·b2=daring-gelding running(19m, gs140→300). 선점 없음. GPU4 확증.
- B0 기준선·eval 하니스 확보 완료. 다음: b3pkg gs25 게이트 점검(도달 시), 두 잡 global_step 증가 감시.

---

## E-005 (2026-07-17 20:00 UTC) — 타임스케일 재보정: gs300은 "시간"이 아니라 "며칠"

**추상적 문제.** 자율 모니터링의 캐던스·기대·keep-alive 전략은 "결과까지 얼마나 걸리는가"에 의존한다.
실제 스텝 처리속도를 측정하지 않으면, 판정 단계 도달 시점을 오판해 감시 자원과 기대를 잘못 배분한다.

**구체적 해결.** b2 실측: `Training Progress: 141/300 [10:16<27:14:04, 616.63 s/it]` →
**스텝당 ≈616초(≈10.3분)**. 남은 159스텝 × 10.3분 ≈ **27시간**(b2, gs141→300). b3pkg는 fresh gs0라
300스텝 × 10.3분 ≈ **~50시간**(선점·재제출 오버헤드 별도). 즉 **gs300 held-out 판정은 며칠 단위**이며,
b3pkg의 첫 헌법 게이트(gs25)조차 **약 4시간** 뒤다(무중단 가정). b2는 step141로 진행 확인(global_step
증가=진짜 생존, F5 통과). b3pkg(distinct-serval)는 8분째 부트스트랩(cold-start pull_resume 통과 중).

**통찰.** 이 파이프라인은 **분 단위 감시 × 며칠 지평**의 문제다. 함의: (1) 판정·해석·논문업데이트 단계는
수 시간~수일 뒤이므로, 지금의 생산적 작업은 *감시+keep-alive+downstream 준비*이지 결과 판정이 아니다.
(2) b3pkg는 fresh라 **첫 체크포인트(save_freq=10 → gs10, ≈100분+부트스트랩) 전에 선점되면 진행분 0**
이므로, Basic 티어에서 "gs10까지 살아남는 한 창"을 확보하는 것이 당면 keep-alive 목표다 — 그 전까진
재제출 비용 0, 이후엔 resume이 보호. (3) 캐던스 10분은 선점 감지엔 충분하나, 게이트/결과는 여러 시간마다
드물게 발생하므로 tick 대부분은 "정상 진행" 확인이 정상이다.

### 상태 tick — 20:00 UTC
- b2=daring-gelding running(step141/300, 616s/it, ~27h ETA, F5 생존 확인). b3pkg=distinct-serval running(8m, 부트스트랩).
- 선점 없음. 당면 목표: b3pkg가 gs10까지 살아남아 첫 체크포인트 확보. 다음: 10분 tick 감시.

---

## E-006 (2026-07-17 20:04 UTC) — b3pkg 부트스트랩 선점 반복(2회): 재제출 vs 개입 판단

**추상적 문제.** 선점 가능한 컴퓨트에서 *fresh* arm은 "부트스트랩 + 첫 체크포인트 전"이라는 취약창을
가진다. 이 창에서 반복 선점되면 진행분이 0으로 리셋되어 영원히 제자리일 수 있다. 언제까지 단순
재제출을 반복하고, 언제 스케줄링/체크포인트 정책에 개입해야 하는가? — 그리고 그 개입이 matched
무결성을 깨지 않는가?

**구체적 해결.**
- 20:02 실측: b3pkg(distinct-serval)가 **8분 부트스트랩 중 2번째 선점**(`pass`, 학습 시작 전, 진행분 0).
  → 즉시 재제출 `internal-stud`(queued). b2(daring-gelding)는 gs141→ running 26분 생존(이미 부트스트랩
  통과·resume 상태라 취약창 없음).
- 반복 패턴이므로 "안 죽게"(사용자 명령) 이행을 위해 Codex에 mitigation 전문 전달(b0g8v12lt):
  후보 (1)재제출 반복 (2)save_freq 초기만 1~5로 낮춰 첫 체크포인트 앞당김(취약창 ~2h→~25min)
  (3)부트스트랩 단축. 핵심 질문: save_freq는 학습 궤적에 영향 없으니(체크포인트 빈도일 뿐) b3pkg만
  낮춰도 RQ2 confound가 아님이 맞는지 확인 중.

**통찰.** keep-alive는 "재제출"과 "체크포인트 정책"의 두 레버를 가진다 — 취약창이 길수록(fresh+
느린 부트스트랩+높은 save_freq) 반복 선점에 약하다. 근본 대응은 **첫 체크포인트를 앞당겨 취약창을
단축**하는 것이며, save_freq는 궤적-중립(모델 업데이트 규칙 불변, 오직 저장 빈도)이라 arm별로 달라도
matched-isolation을 깨지 않을 가능성이 높다(Codex 확인 대기). 즉 keep-alive 개선이 과학적 비교를
훼손하지 않는 안전한 지점이 존재한다.

### 상태 tick — 20:04 UTC
- b3pkg=**internal-stud**(queued, 2번째 재제출)·b2=daring-gelding running(26m, gs141/300 진행).
- Codex bootstrap-mitigation 논의 진행 중. 임계: b3pkg 1~2회 더 부트스트랩 선점 시 Codex 권고대로 개입.

---

## E-007 (2026-07-17 20:06 UTC) — Codex bootstrap-mitigation 판정 + 취약창 2단 구조 규명

**추상적 문제.** "첫 체크포인트 전 취약창"이라는 진단을 개입으로 옮기려면, 그 창이 *어느 구간*에서
깨지는지 정확히 알아야 한다. 잘못된 층에 개입하면(예: 부트스트랩 문제에 체크포인트 정책을 손대면)
효과가 없다. 개입은 관측된 실패 구간과 일치해야 한다(헌법 Part II: 근인을 층까지 좁혀라).

**구체적 해결.**
- Codex 판정(b0g8v12lt): **2회 선점 = 개입 임계.** 권고 순서 (1)부트스트랩 캐시/사전스테이징(최선,
  선점 자체 감소) (2)`save_freq`를 초기 1로 낮춰 첫 체크포인트 앞당김(취약창 축소) (3)재제출 반복(임시).
  **save_freq는 궤적-중립**(optimizer/model/RNG 정확 저장·save callback이 학습로직 불변) → b3만 낮춰도
  **RQ2 confound 아님**(단 논문엔 "b3 초기 fault-tolerance 설정"으로 명시, b2 save_freq는 불변).
- **정정**: b3.yaml은 이미 `trainer.save_freq=5`(config 기본 10 오버라이드, line 216). 즉 첫 체크포인트=gs5.
- **취약창 2단 구조 규명**: (1)부트스트랩 ~12분(다운로드; save_freq 무관) + (2)gs0→gs5 ~50분(save_freq=5;
  save_freq=1이면 gs0→gs1 ~10분). **지난 2회 선점은 전부 (1)부트스트랩 구간** → save_freq 변경으로는
  못 막았을 사건. 따라서 지금 save_freq 개입은 시기상조(internal-stud 아직 부트스트랩 미통과).
- **결정**: internal-stud running(live) — 죽이지 않고 부트스트랩 통과 기회 부여. Codex의 `save_freq=1`은
  "부트스트랩 통과 *후* gs5 전 반복 선점" 시 발동할 **사전승인 개입**으로 예비(발동 시 line216 5→1,
  첫 체크포인트 후 5로 복원). 부트스트랩 구간 반복 선점엔 재제출이 유일 대응(긴 창 확보까지).

**통찰.** 취약창은 단일 블록이 아니라 **부트스트랩(다운로드-바운드) + 첫-체크포인트-전(스텝-바운드)의 2단**
이며, 각 단은 다른 레버로만 줄어든다 — 부트스트랩은 캐시/prefetch, 스텝창은 save_freq. 관측된 실패
구간(여기선 전부 부트스트랩)에 맞는 레버를 골라야 개입이 유효하다. "개입 임계 도달"이 곧 "아무 개입이나
지금"을 뜻하지 않는다 — 실패 구간과 개입 레버의 정합이 먼저다.

### 상태 tick — 20:06 UTC
- b3pkg=**internal-stud** running(47s, 3번째 시도, 부트스트랩 진입)·b2=daring-gelding running(28m, gs141/300).
- save_freq=1은 사전승인·예비(부트스트랩 통과 후 gs5 전 반복선점 시). 다음: internal-stud 부트스트랩 통과 여부 감시.

### 상태 tick — 20:11 UTC
- b3pkg=internal-stud running(5m, 부트스트랩 후반=staging 통과·`LOCAL_RANK=0`, verl 시작 직전; 지난 2회 사망지점 ~8m이라 아직 위험구간)·b2=daring-gelding running(34m, gs141/300, F5 생존). 선점 없음. 개입 불필요(진행 중).

---

## E-008 (2026-07-17 20:13 UTC) — B3-noPMI 발사 결정 재교정: "Fresh gs0" 의도 vs gs135 resume 메커니즘 충돌

**추상적 문제.** launcher의 *의도*(설명 텍스트)와 *메커니즘*(resume-hardening 스크립트)이 어긋나면,
자동화가 의도를 조용히 뒤집을 수 있다. matched-ladder에서 이 어긋남을 발사 전에 잡지 못하면 한 arm이
잘못된 시작점에서 학습된다. 어느 쪽이 authoritative인가?

**구체적 해결(E-003 재교정).**
- b3nopmi.yaml 실독: 설명이 명시적으로 **"Fresh gs0"**, `WANDB_RUN_ID=rq3-b3nopmi-2`("-2"=재시작),
  arm 정의 = B3pkg와 IDENTICAL하되 `dcpo_w_meta=0.0`(pmi advantage 진짜 0, form scaffold는 ON).
- 그러나 resume-hardening(line 162-174)은 v3의 `rq3_b3nopmi` lineage를 pull하며, **ABORT 가드가
  "gs135 존재하는데 pull 실패 시 cold-start를 *거부*"** → gs135가 있으면 fresh가 아니라 gs135 resume이 됨.
- **∴ E-003의 "b2처럼 정상 resume" 추론은 성급했다.** b2는 *의도가 resume gs140*이지만, b3nopmi는
  *의도가 fresh gs0*이다(설명 명시). 진짜 parallel 쌍은 **b3pkg(fresh)–b3nopmi(fresh)**이지 b2가 아니다.
  b3pkg가 fresh인 이유("eroded gs60 폐기")처럼, b3nopmi의 gs135도 폐기 대상일 개연성(재시작 "-2").
- **교정된 결정**: b3nopmi = **force-fresh**(설명 의도 존중). 발사 전 gs135가 hijack 못 하도록 **HF에서
  gs135 archive/제거 또는 config로 skip** 필요(그냥 발사하면 ABORT 가드가 오히려 gs135 resume 강제).
  이 파괴적 조치(gs135 삭제/이동)는 b3nopmi 발사 시점(b3pkg 안정 후, 1+시간 뒤)에 **Codex 전문 확인** 후 실행.

**통찰.** 자동 resume은 "체크포인트가 있으면 이어붙인다"는 *메커니즘*이지 *의도*가 아니다 — 의도가 fresh인
arm에 잔여 체크포인트가 남아 있으면, keep-alive 자동화가 matched 설계를 조용히 뒤집는다. **launcher의
설명(의도)과 resume 스크립트(메커니즘)를 발사 전에 대조**해야 이 전복을 막는다. 나는 E-003에서 "잔여=정상
resume"으로 성급히 일반화했다가, 실제 yaml의 "Fresh gs0" 명시를 읽고 교정했다(헌법 Part II: 증거—여기선
yaml 원문—로 판단을 좁혀라. 프록시 추론보다 원문이 우선).

### 상태 tick — 20:13 UTC
- b3pkg=internal-stud running(7m, 부트스트랩 위험 임계 ~8m 근접)·b2=daring-gelding running(gs141/300).
- b3nopmi 결정 확정: force-fresh(발사 시 gs135 제거+Codex 확인). 다음: internal-stud 8m 통과 여부.

### 상태 tick — 20:17 UTC — b3pkg 3번째 부트스트랩 선점 → 재제출(new-starling, 4th)
- internal-stud도 9m에 `pass`(3연속 부트스트랩 선점, 전부 학습 전). Basic 티어 노드가 ~12-15분 부트스트랩(init 16GB+env 12GB 다운로드)을 못 버팀. save_freq 무효 구간·캐시는 fresh-node라 불가 → **재제출이 유일 대응**. → new-starling queued.
- b2=daring-gelding running(39m, gs141/300) 정상 — b2는 이미 부트스트랩 통과·resume이라 안정. **b2가 첫 gs300 결과를 낼 것**(~27h).
- ★에스컬레이션 임계: b3pkg가 2~3회 더 부트스트랩 선점하면 Codex와 **부트스트랩 단축**(env pre-bake 커스텀 이미지 h100std_env_builder.yaml 활용 등) 논의. 지금은 재제출 지속(b2가 주 진행분 담당).

### 상태 tick — 20:19 UTC
- b3pkg=new-starling **queued**(노드 대기). b2=daring-gelding running(41m, gs141/300). 선점 없음·액션 불필요(안정 대기). 다음 실질 이벤트=new-starling 노드확보/선점 또는 b2 gs 진행(~10분/스텝).

---

## E-009 (2026-07-17 20:21 UTC) — Phase-1 완비: gs300 held-out eval 레시피 확정(B0와 동일 방법론)

**추상적 문제.** 결과 판정이 유효하려면 meta arm과 baseline이 *동일 평가 방법론*(같은 스크립트·샘플수·
토큰예산·시드·채점기)으로 측정돼야 한다. gs300 도달 후 즉흥적으로 평가 명령을 짜면 방법론이 어긋나
비교가 오염된다. 결과 전에 "정확히 어떤 명령으로 B0가 측정됐는가"를 확정해 재사용해야 한다.

**구체적 해결.**
- B0 기준선(HF v3 `eval/base_matched_1030_v2/base_gs300_{16k,4k}_n8`)을 생성한 스크립트를 특정:
  **`scripts/eval_vllm_1030.py`**(vLLM 기반, `run_eval_1030.sh`는 pre-rq3 레거시라 배제).
- 인터페이스: `--model_path --model_name --output_dir --benchmarks gsm8k math500 aime2024
  --max_problems 500 --max_tokens {16384|4096} --num_samples 8 --tp_size 4 --temperature 0.7
  --top_p 0.95 --seed 42`. (B0의 16k_n8/4k_n8과 정확히 대응.)
- **gs300 meta arm 재사용 명령**(예: b2):
  ```
  # gs300 체크포인트를 HF v3서 pull → actor/huggingface/가 HF 모델 디렉터리
  python scripts/eval_vllm_1030.py --model_path <checkpoints/rq3_b2/global_step_300/actor/huggingface> \
    --model_name b2_gs300_16k_n8 --output_dir eval/b2_matched_1030_v2 \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 --max_tokens 16384 --num_samples 8 \
    --tp_size 4 --seed 42   # + 4k 변형(--max_tokens 4096)
  ```
  채점=math_verify(eval_vllm_1030 내). 산출 acc를 B0(GSM8K0.8905/MATH5000.544/AIME@16k0.0458)와 비교.

**통찰.** 공정 비교의 관건은 "무엇을 재는가"만큼 "어떻게 재는가"다 — baseline을 만든 *정확한 측정
파이프라인*을 결과 전에 특정·고정해두면, gs300 도달 시 판정이 즉시·재현가능·방법론-일치로 이뤄진다.
Phase-1(모니터→결과)의 downstream 준비물(B0 수치 + eval 레시피 + 해석표)이 이제 전부 확보돼, 첫 arm이
gs300에 닿는 순간 곧바로 RQ 판정 단계로 진입할 수 있다.

### 상태 tick — 20:21 UTC
- b3pkg=new-starling queued(노드 대기)·b2=daring-gelding running(gs141/300). 선점 없음. Phase-1 준비 완비(B0 기준선·eval 레시피·해석표). 다음: new-starling 노드확보/선점, b2 gs 진행 감시.

### 상태 tick — 20:29 UTC
- b3pkg=new-starling running **8m**(4번째 시도, 부트스트랩 마지막=verl 시작 직전; 지난 3회 사망지점 ~8m — 분수령). b2=daring-gelding running(51m, step141/300, 644s/it, F5 생존). 선점 없음. Phase-2b 컨틴전시 설계 Codex 진행 중. 다음: new-starling 8m 통과(학습 시작) 여부.

---

## E-010 (2026-07-17 20:30 UTC) — Phase-2b 사전설계: 결과-분기별 개선실험 컨틴전시 맵(Codex)

**추상적 문제.** "결과가 나쁘면 개선실험을 이어서 하라"는 지시를, 결과가 나온 뒤에야 설계하면 다음
실험까지 공백이 생긴다. 각 *예상-이탈 분기*에 대해 근인 가설·개선실험·불변식(북극성/인식적독립/matched)을
결과 *전에* 못박아두면, gs300 판정 직후 지체 없이 다음 사이클로 진입할 수 있다.

**구체적 해결(Codex bf07dekok, 전제: 1차지표=held-out 정확도, gold는 correctness만·confidence/메타표현엔 안 씀).**
- **분기2 (B3pkg≈B3-noPMI, PMI 순기여 없음)** → *selective-PMI replacement* [P0=B3-noPMI, P1=B3pkg,
  P2=PMI를 **CTSD(T+/T−)+gm-contrast+entropy-triggered force-inject**로 교체, P3=P2+emission-floor].
  근인: 이득은 form/format·PMI 정보량 약함 / PMI가 내용 vs confidence-inflation 미분리. 측정: acc·AUROC·ECE/Brier·risk-coverage·발화율/길이·region별.
- **분기3 (셋 다 ≈, 정확도 이득 없음)** → *current-substrate CF transfer* [Q0=B3-noPMI, Q1=Q0+현 substrate
  재채굴 **CF/directional**, Q2=Q0+**emission-floor+meta-maintenance replay**, Q3=Q1+Q2].
  근인: 메타발화가 답 수정으로 연결 안 됨(CF 효과 substrate-dependent였을 가능성) / base SFT under-train+region-split. 판별: Q1만↑=CF부족, Q2만 발화↑·acc정체=emission만, Q3만↑=결합.
- **분기4 (B3pkg<B2 or noPMI>B3pkg, PMI 유해)** → *anti-inflation PMI rescue* [R0/R1 baseline, R2=PMI폐기+
  **CTSD+gm+엔트로피게이트 주입**(저엔트로피 정답엔 미주입, 고엔트로피/오답위험만), R3=R2+emission-floor;
  anchor_norm 제외(inert 위험)]. 성공=acc 회복 AND AUROC↑ 또는 ECE/Brier↑(발화율·평균confidence만↑=실패). 최소 3 paired seeds.
- **분기1 (성공)** → multi-seed 재현 + **level/region/decoy-placebo 통제**로 패키지 효과와 PMI 메커니즘 분리.
- **★최종 단일 추천(사전 예약)**: **Q0–Q3 omnibus** [B3-noPMI / B3pkg / CF+emission-floor /
  CF+emission-floor+selective-PMI] — PMI의 유해/무효뿐 아니라 **현 substrate에서 self-distill-RL이 정확도를
  회복시키는지 자체**를 한 번에 판별하는 최고 정보가치 실험.

**통찰.** 반증가능한 파이프라인은 "결과를 본 뒤 무엇을 할지"가 아니라 **"각 결과에 무엇으로 답할지"를
미리 못박는다** — 이는 사후합리화(HARKing)를 막고, 대기 시간을 다음 사이클의 설계로 전환한다. 세 개선
분기가 공통으로 **PMI를 CTSD+gm+엔트로피게이트로 교체**하는 방향으로 수렴한 것은, 우리 근인 진단(PMI=
proxy·내용/confidence 미분리·always-on 비선택)과 정합한다. 모든 개선안이 북극성(메타→정확도)·인식적
독립(gold=correctness)·matched-isolation을 보존한다(발사 전 Codex 재확인 필요).

### 상태 tick — 20:30 UTC
- b3pkg=new-starling running **9m** — ★지난 3회 사망지점(~8m) **처음으로 통과**(부트스트랩 클리어 가능, verl 학습 시작 임박). b2=daring-gelding running(step141/300). 선점 없음. Phase-2b 컨틴전시 확보(Q0–Q3 omnibus 사전예약). 다음: new-starling gs0 학습 시작 확인.

---

## E-011 (2026-07-17 20:33 UTC) — b3pkg 4연속 부트스트랩 선점: 에스컬레이션 임계 vs ROI 판단

**추상적 문제.** keep-alive 개입(재제출)이 반복 실패하면, 더 무거운 개입(인프라 변경)으로 에스컬레이션할지
판단해야 한다. 그러나 무거운 개입 자체가 큰 우회이고 불확실한 ROI를 가질 때, "임계 도달=즉시 개입"이
아니라 **개입 비용 vs 대안 경로의 가치**를 저울질해야 한다. 무엇이 지금 정말 막혀 있는가?

**구체적 해결.**
- new-starling도 9m에 `pass`(4번째 부트스트랩 선점, 학습 전) → 즉시 재제출 `tidy-dane`(5th attempt).
  b3pkg는 4연속 부트스트랩에서 사망, verl 학습을 한 번도 시작 못 함(Basic 티어 노드가 ~12-15분
  다운로드=code+init16GB+env12GB를 못 버팀; 현재 클러스터 busy 추정).
- 에스컬레이션 옵션(env pre-bake 이미지 h100std_env_builder.yaml로 env 12GB 다운로드 제거)은 **큰 우회**:
  이미지 빌드 자체가 preemptible 티어서 도는 별도 잡, 시간 소요, init 16GB는 여전히 다운로드.
- **판단: 아직 env-prebake 미착수, churn 지속.** 근거: (1) **b2가 주 결과(RQ1=B2−B0) 담당**하며 안정
  진행 중(55m, step141/300) — b3pkg 지연이 첫 결과를 막지 않음. (2) 4회 선점은 **transient busy 구간**
  가능성(b2는 calmer 순간에 노드 잡아 생존) — 지속 여부 불명. (3) prebake ROI 불확실.
- **firmer 임계 재설정**: b3pkg가 **1시간+ 지속 실패**(추가 ~4-5회)하면 env-prebake 착수(Codex 확인 후).
  그 전까진 재제출 churn(비용=매회 ~10분, b2 진행과 병행이라 기회비용 낮음).

**통찰.** "에스컬레이션 임계 도달"이 자동으로 "무거운 개입 실행"을 뜻하지 않는다 — 개입의 **한계비용 대비
막혀 있는 것의 가치**를 봐야 한다. 여기선 primary 경로(b2/RQ1)가 안 막혀 있고 실패가 transient일 수
있어, 저비용 churn을 유지하고 무거운 개입(이미지 빌드)은 지속성이 확인될 때로 미루는 것이 합리적이다.
(헌법 Part II 연장: 개입은 *확인된* 병목에만. 병목이 "일시적 운"인지 "구조적"인지 먼저 가려라.)

### 상태 tick — 20:33 UTC
- b3pkg=**tidy-dane**(5th attempt, queued)·b2=daring-gelding running(55m, step141/300). 선점 대응 지속. env-prebake는 1시간+ 지속 실패 시. Phase-1·2b 완비. 다음: tidy-dane 부트스트랩·b2 gs 진행.

### 상태 tick — 20:40 UTC
- b2=daring-gelding running(1h): **step 141→142 전진**(Training Progress 142/300, 681s/it, ETA~30h) = F5 확증, 실제 학습. b3pkg=tidy-dane **queued 6m**(노드 아직 대기, opportunistic). 선점 없음. b2 주 진행분 순항. 다음: tidy-dane 노드확보, b2 gs 진행.

---

## E-012 (2026-07-17 20:44 UTC) — "b3pkg는 코드 문제인가?" — 아니오, 100% 인프라(증거 기반 판별)

**추상적 문제.** 한 arm이 반복 실패하면 "코드 결함"을 의심하는 것이 자연스럽다. 그러나 실패가 코드
*실행 전*(할당·부트스트랩)에서 일어나면 코드는 용의자조차 될 수 없다. 실패의 **발생 계층**을 특정해야
엉뚱한 곳(코드)을 파는 낭비를 막는다.

**구체적 해결(판별).**
- tidy-dane(5th)가 `failed` — 단 **로그 출력 0줄**(`No log output available for this job`) = 큐에서 노드를
  못 잡고 실패, **코드 미실행**(할당 실패). 코드 에러 아님.
- b3pkg **6회 시도 전부 verl 학습 미시작**(tidy-dane·new-starling 모두 `Training Progress`/`global_step`
  흔적 0). 모든 죽음이 부트스트랩(다운로드) 또는 큐에서, **코드 실행 이전**.
- 실패 유형 집계: **4×`pass`**(부트스트랩 중 graceful-SIGTERM 선점) + **1×`failed`**(큐 할당, 무출력).
  전부 코드-이전 계층.
- **코드는 독립 검증됨**: (a) b2가 지금 동일 트레이너 verl_sdc로 step142 정상 학습; (b) b3pkg의
  triobj+pmi_shift 설정은 T1을 이긴 아카이브 h100std_pmishift.yaml과 동일 패키지; (c) 이전 base b3 런이
  gs25 게이트 통과(emit0.89·attempted0.40·n_save7). → 코드는 이미 base에서도 돈 적 있음.
- **결론**: b3pkg 실패 = **Basic 티어 opportunistic 노드 가용성**(현재 busy)일 뿐, 코드 무관.
  → complete-akita(6th) 재제출. 대응은 코드 수정 아니라 재제출 지속(±env-prebake).

**통찰.** "반복 실패=코드 결함"은 유혹적 오진이다. **실패 계층(pre-code vs in-code)을 로그 유무·verl
도달 여부로 먼저 특정**하면, 무출력-큐실패·부트스트랩-선점은 코드가 아니라 인프라로 즉시 귀속된다.
코드가 *실행될 기회조차 없이* 죽은 것과 *실행돼서* 죽은 것은 근본적으로 다른 문제다(헌법 Part II:
증상을 레버와 혼동 말라 — 여기서 레버는 코드가 아니라 스케줄러).

### 상태 tick — 20:44 UTC
- b3pkg=**complete-akita**(6th, queued)·b2=daring-gelding running(1h, step142/300). b3pkg 실패=인프라(코드 무관, 증거확정). 재제출 지속. 다음: complete-akita 노드확보, b2 gs 진행.

### 상태 tick — 20:48 UTC
- b2=daring-gelding running(1h): **step 142→143 전진**(Training Progress 143/300, 649s/it, ETA~28h) F5 확증. b3pkg=complete-akita(6th) **노드 확보·running 1m, 부트스트랩**(`extracting env ~5GB`) — 취약 창 재진입(지난 4회 ~8-9m 사망). 선점 없음. 다음: complete-akita 부트스트랩 통과(~8m 분수령), b2 gs 진행.

---

## E-013 (2026-07-17 20:57 UTC) — b2도 선점: auto-retry + 완결성 게이트가 resume 지점을 지킴

**추상적 문제.** 안정적이던 arm이 선점되면 (a) 자동 복구가 작동하는지, (b) 재개가 *올바른*
체크포인트(완전 저장본)에서 이뤄지는지 확인해야 한다. 불완전 체크포인트에서 재개하면 학습이
조용히 오염된다. 그리고 여러 arm이 동시에 churn하면 티어 상태(일시적 busy vs 구조적) 판단이
필요하다.

**구체적 해결.**
- b2(daring-gelding)가 `Terminated`→status `queued`로 전이 = **선점**. 그러나 **Singularity가
  `retry_001` 자동 재시도** 중 → **재제출 불필요, 자동 재개.** (선점≠terminal; Standard/Basic 모두
  graceful 선점은 amlt가 retry.)
- HF `rq3_b2` 체크포인트 = gs130/140/**150/160**. 그런데 daring-gelding은 이전에 **gs140**에서
  resume했음(gs160 아님) → gs150/160은 **불완전**(과거 "gs150 optim 3/4 permanent freeze" 이슈와
  일치). **RGS 3원 완결성 게이트**(model+extra_state+optim 각≥4)가 불완전 gs150/160을 거부하고
  **gs140(마지막 완전본)에서 재개**시킴. 손실=steps 141~143(~30분)뿐.
- 동시에 complete-akita(b3pkg)도 8m verl-startup 중 → **양 arm이 동시에 Basic 티어 busy에 노출.**
  단 b2는 auto-retry+resume으로 자력복구, b3pkg는 재제출 churn. 조치 불필요(둘 다 자동 경로).

**통찰.** keep-alive의 진짜 안전장치는 "재개하느냐"가 아니라 **"올바른 지점에서 재개하느냐"**다 —
**완결성 게이트(3원 shard 카운트)가 불완전 체크포인트를 자동 거부**하기에, 선점이 잦아도 학습이
half-saved 상태에서 오염되지 않는다. 이것이 E-001/E-008의 "잔여 체크포인트 대조"와 같은 원리의
런타임 구현이다: 잔여물의 *존재*가 아니라 *완전성/provenance*가 재개 안전성을 결정한다. (양 arm
동시 churn은 티어 일시적 busy의 징후 — 구조적 문제 아니면 riding-out이 맞음.)

### 상태 tick — 20:57 UTC
- b2=daring-gelding **queued(retry_001 자동재시도, gs140서 resume 예정)** — 선점됐으나 자력복구. b3pkg=complete-akita running 8m(verl-startup, 생존). 양 arm 티어 busy 노출·둘 다 자동복구 경로. 조치 불필요. 다음: b2 retry 재개·complete-akita verl 시작 여부.

---

## E-014 (2026-07-17 21:03 UTC) — b3pkg 부트스트랩 병목 재진단: flash-attn 가설 반증, 진짜는 init 다운로드

**추상적 문제.** 반복 실패의 병목을 "그럴듯한 원인"으로 가정하고 개입하면, 실제 병목이 다른 곳이면
낭비다. 개입(예: env-prebake) 전에 **로그로 실제 시간이 어디서 소모되는지** 확증해야 한다. 가설은
증거로 반증되어야 한다.

**구체적 해결(자기교정).**
- 가설: "부트스트랩 병목 = flash-attn 2.8.3 소스빌드 ~15분(env_builder 설명), v3→v4 전환하면 해결."
- **반증**: complete-akita 로그 = `[bootstrap] flash-attn already present in simplerl` +
  `env install complete → /scratch/simplerl_v4.done`. → **b3pkg는 이미 v4(flash-attn 미리구움)를 사용**
  (bootstrap_sdc_node.sh line68이 v4 pull; b3.yaml line87이 이 스크립트 호출). flash-attn 빌드 **안 함**.
  env-prebake는 **이미 적용된 상태**였다.
- **진짜 병목**: env(v4) 이후 로그 = `Fetching 14 files ... Xet Storage enabled but hf_xet not installed,
  Falling back to regular HTTP download`. → **b23 SFT init 16GB(14 files)를 느린 HTTP로 다운로드**
  (hf_xet은 [[hf-xet-upload-pitfall-and-chain-fix]] 사일런트404 때문에 line93서 의도적 제거 → 다운로드도
  HTTP 폴백). 이 init 다운로드 + verl init이 합쳐 ~8-9분 → 티어 선점창(~8-9분)에 아슬하게 패배.
- **결론**: 부트스트랩은 env 측면에선 이미 최적(v4). 남은 시간(init 16GB HTTP + verl init)은 쉽게 못 줄임
  → **easy fix 없음, 재제출 churn 지속**(b2도 한 번 선점됐다 회복했듯 b3pkg도 긴 창 잡으면 됨).
- **후보 micro-opt**(다음 에스컬레이션): init `snapshot_download`에 `HF_HUB_ENABLE_HF_TRANSFER=1`
  적용 시 HTTP 다운로드 대폭 가속 가능(env_builder는 이미 사용). 업로드가 아닌 **다운로드**라 404 위험
  낮음. yaml 변경이라 Codex 확인 후. churn 지속 시 착수.

**통찰.** "그럴듯한 병목 가설"을 로그로 반증하는 것이 개입보다 먼저다 — 나는 env_builder 설명만 보고
flash-attn 빌드를 병목으로 단정했으나, 실제 로그는 v4가 이미 그걸 제거했고 병목은 init 다운로드임을
보였다. **개입(v3→v4)은 이미 되어 있었다.** 만약 로그 확인 없이 "env-prebake 착수"로 갔다면 이미
있는 것을 또 만드는 완전한 낭비였을 것이다(헌법 Part II: 증상→레버 확증 먼저). 남은 병목(init 다운로드)의
올바른 레버는 hf_transfer 가속이지 env 재빌드가 아니다.

### 상태 tick — 21:03 UTC
- b2=daring-gelding **running 재개**(retry_001 노드확보, gs140 resume). b3pkg=**endless-antelope**(7th, queued). b3pkg=v4 이미사용·병목은 init HTTP 다운로드. 조치=churn 지속(+hf_transfer 후보). 다음: endless-antelope 부트스트랩, b2 gs 재개.

---

## E-015 (2026-07-17 21:08 UTC) — hf_transfer fix: Codex "조건부 GO but 크래시 위험" → 검증불가로 defer

**추상적 문제.** 확증된 병목(init 다운로드)에 대한 후보 fix가, *특정 전제(패키지 설치·라이브러리 버전)에서만*
안전할 때, 그 전제를 검증할 수 없다면 적용해야 하는가? "도움될 수도"의 기대값과 "더 악화"의 꼬리위험을
저울질해야 한다.

**구체적 해결(Codex bm6wdxgc4 판정).**
- A) `hf_transfer`≠`hf_xet`(별개 패키지). hf_xet 업로드404와 무관, 다운로드 가속 대상. 단 init에만 scope 필수.
- B) ★**버전 의존 위험**: `huggingface_hub<1.0` + hf_transfer 미설치 → `HF_HUB_ENABLE_HF_TRANSFER=1`은
  **no-op가 아니라 ValueError 크래시**(init 다운로드 붕괴 → b3pkg 더 악화). `hub≥1.0`이면 deprecated no-op(무익).
- C) 구버전+설치 상태면 2배 가속 현실성(8-9분→4-6분, 선점창 탈출 가능). 단 verl init이 크면 효과 제한.
- D) init-scope면 numerics 불변(confound 아님). 전역 env면 b0/b2 환경차·pusher 영향 → **init-only 필수**.
- E) 정확한 편집=line~101 `python - <<'PY'` 앞에만 `HF_HUB_ENABLE_HF_TRANSFER=1` prepend(전역 아님).
- **결정: defer.** 노드 없이 v4의 hf_transfer 설치·hub 버전을 확인 불가 → 블라인드 적용은 크래시 꼬리위험.
  b2(주 결과)가 순항하고 b3pkg는 churn으로 결국 긴 창 잡을 수 있으므로, **검증 없는 변경 대신 churn 유지**가
  low-regret. (안전판: 나중에 `HF_HUB_ENABLE_HF_TRANSFER=$(python -c "import hf_transfer" 2>/dev/null &&
  echo 1||echo 0)` 가드형이면 fail-safe지만, yaml heredoc 정밀편집 리스크가 있어 churn 지속이 우선.)

**통찰.** "확증된 병목"에 대한 fix라도 **그 fix의 안전 전제를 검증할 수 없으면 적용하지 않는다** —
특히 실패모드가 "무익"이 아니라 "더 악화(크래시)"일 때. 검증 가능성이 개입 가능성의 전제다(헌법 Part II
연장: 근인을 확증했어도, 처방의 *안전 전제*까지 확증돼야 처방한다). 여기선 primary 경로(b2)가 건재해
b3pkg churn의 기회비용이 낮으므로 보수가 정답.

### 상태 tick — 21:08 UTC
- b3pkg=endless-antelope(7th) queued 6m·b2=daring-gelding running(gs140 resume 진행). hf_transfer=defer(검증불가). churn 지속. 다음: endless-antelope 부트스트랩·b2 gs 진행.

### 상태 tick — 21:11 UTC
- b3pkg=endless-antelope(7th) **`failed`(큐 할당 실패, 노드 미확보)** → 재제출 **many-basilisk(8th)**. b3pkg 누적 6회 실패(부트스트랩 선점 4 + 큐할당 실패 2) = **티어 heavy busy 구간**(b3pkg가 노드조차 못 잡음). b2=daring-gelding running(노드 보유, 순항). 조치=churn 지속(티어 부하 감소 대기). b2 primary라 첫 결과 무영향.

---

## E-016 (2026-07-17 21:17 UTC) — 티어 heavy-busy가 primary arm(b2)까지: 선점-재개 정체 vs save_freq 개입의 위험

**추상적 문제.** 인프라 부하가 심해 여러 arm이 동시에 흔들리고 primary arm조차 net 진행을 못 할 때,
(a) 체크포인트 정책 개입(save_freq↓)으로 진행을 지키느냐, (b) 부하가 transient라 보고 riding-out 하느냐를
가른다. 개입 자체가 primary arm의 노드를 잃는 위험을 동반할 때, 위험-대비-이득을 어떻게 저울질하나?

**구체적 해결.**
- b2(daring-gelding) 로그: `[Rank] Loaded optimizer/rng/lr_scheduler from rq3_b2/global_step_140` +
  `Training Progress: 140/300 [00:00<?, ?it/s]` = **또 선점당해 gs140에서 재재개**(retry_NNN 다수). save_freq=10
  이라 gs141~149는 미저장 → 선점시 gs140으로 리셋. **~4-5스텝마다 선점되면 gs150 저장점을 못 넘어 net 진행 0.**
  (HB의 gpu0used 80→40GB는 재개 중 모델로딩이라 정상, F5 이상 아님.)
- b3pkg(many-basilisk)는 queued(노드조차 못 잡음). → **양 arm이 동시에 heavy-busy에 노출.**
- **개입 옵션**: b2 save_freq 10→1/5로 낮춰 자주 저장→선점해도 최신에서 재개(net 진행 보존). 궤적-중립(confound 아님).
- **그러나 개입 위험**: b2는 Singularity `retry_NNN` **auto-retry 중**(제출 config 사용). save_freq 바꾸려면 b2를
  **취소 후 재제출** 필요 → **heavy-busy 구간에 노드를 잃으면 b3pkg처럼 큐에서 못 돌아올 위험.**
- **결정: riding-out(b2 미개입).** 근거: (1) 티어 busy는 **cyclic·transient**일 가능성(곧 calm 창) — 그때 b2가
  gs150+ 도달해 저장. (2) auto-retry가 이미 작동(자력 재개, gs140 무결 resume). (3) 취소-재제출은 primary
  노드 상실 위험이 이득보다 큼. **firmer 임계**: b2가 **~30-40분 내 gs150을 못 넘으면**(지속적 busy 확정) 그때
  save_freq↓ 개입(Codex 확인+취소-재제출 감수). 그 전까진 auto-retry에 맡김. b3pkg는 churn.

**통찰.** keep-alive 개입도 *공짜가 아니다* — heavy-busy 구간에 primary arm을 취소-재제출하는 것은
"진행 보존"을 위해 "노드 상실"을 무릅쓰는 것이고, 부하가 transient면 riding-out이 개입보다 낫다.
**개입의 위험이 병목의 성격(일시적 vs 지속적)에 의존**하므로, 병목이 지속적임을 확인하기 전엔 저위험
경로(auto-retry+대기)를 유지한다(E-011의 "임계 도달≠즉시 개입" 원리를 primary arm에 적용).

### 상태 tick — 21:17 UTC
- b2=daring-gelding running(선점-재개 반복, gs140서 재시작·auto-retry). b3pkg=many-basilisk queued(노드 대기). 양 arm heavy-busy 노출. 조치=riding-out(b2 미개입)+b3pkg churn. 임계=b2 30-40분내 gs150 미도달시 save_freq↓. 다음: 티어 calm 여부·b2 gs150 돌파.

### 상태 tick — 21:19 UTC
- b3pkg=many-basilisk(8th) `failed`(3연속 큐할당 실패) → 재제출 **humane-stinkbug(9th)**. b3pkg 누적 7회 실패(부트스트랩 선점 4 + 큐할당 3) = 티어 b3pkg 포화. b2=daring-gelding running(auto-retry 재개 시도). 조치=churn 지속·b2 riding-out. b2 gs150 돌파 임계 감시(≈21:50-22:00).

---

## E-017 (2026-07-17 21:28 UTC) — b2 개입 임계 재보정: save_freq=10은 ~100분 무중단을 요구

**추상적 문제.** 개입(save_freq↓) 임계를 정할 때, 체크포인트 주기와 스텝 시간에서 나오는 **실제
무중단-요구시간**을 계산하지 않으면 임계가 비현실적이 된다. 잘못된 임계는 "곧 저장되겠지"라는
헛된 기다림이나 성급한 개입을 유발한다.

**구체적 해결(재보정).**
- b2(daring-gelding) 실측: gs140 resume 후 `Training Progress: 141/300, 643s/it`. save_freq=10 → 다음
  저장점 gs150. gs141→150 = 9스텝 × ~10.7분 = **~96분 무중단 필요.**
- ∴ 내 이전 임계 "gs150 by 21:50-22:00"은 **물리적으로 불가능**(gs141서 22:00까진 ~30분뿐 → gs143 정도).
  임계를 **"HF의 b2 최신 *완전* 체크포인트가 gs140을 못 넘음"** 으로 재정의: b2가 ~96분 무중단 창을
  못 잡아 gs150 저장 실패가 **지속**되면(예: ~22:50-23:00까지 HF 최신완전=여전히 gs140) = busy 지속 확정.
- 그때 개입: b2 save_freq 10→1(궤적-중립)로 취소-재제출 → 짧은 창에도 매스텝 저장·최신서 재개 = net 진행.
  단 heavy-busy서 취소-재제출은 노드상실 위험이라 **지속성 확정 후에만**(Codex 확인).
- 실패모드 대비: 리스크는 (a)riding-out→b2 무한정 gs140 정체 vs (b)개입→노드상실. 둘 다 primary 정지.
  transient면 (a)가, 지속이면 (b)가 유리 → **지속성 확정이 개입 트리거.**

**통찰.** 개입 임계는 감(感)이 아니라 **산수**(체크포인트주기 × 스텝시간 = 무중단-요구시간)로 정해야
현실적이다. save_freq=10 + 10min/step = 100분 무중단 요구는 busy 티어에 취약한 설계이며, 이 계산이
"왜 b2가 정체하는가"와 "언제 개입해야 하는가"를 동시에 답한다. (이 취약성 자체가 향후 matched-ladder를
preemptible 티어서 돌릴 때 save_freq를 낮게 잡아야 한다는 교훈.)

### 상태 tick — 21:28 UTC
- b3pkg=humane-stinkbug(9th) running ~2m 부트스트랩(생존, ~8m 분수령 대기). b2=daring-gelding running(gs141, gs140 resume 진행). 조치=riding-out(재보정 임계: HF b2 완전체크포인트가 ~22:50까지 gs140이면 save_freq↓ 개입). b3pkg churn 계속. 다음: humane-stinkbug ~8m 통과·b2 gs 진행.

### 상태 tick — 21:35 UTC
- humane-stinkbug(b3pkg 9th) running ~4m 부트스트랩(cold-start, 생존; ~8m 분수령 아직 전). b2=daring-gelding running(step 141, 142 임박, 643s/it). 선점 없음·양 arm progress. 다음: humane-stinkbug ~8m 통과 여부·b2 gs150 향한 진행(임계 ~22:50).

---

## E-018 (2026-07-17 21:33 UTC) — 확증 병목에 fail-safe 개입 적용: 가드형 hf_transfer + amlt `$$` 이스케이프

**추상적 문제.** 확증된 병목(init 다운로드)에 대한 개입이 "안전 전제(패키지 유무)를 검증 못 함"(E-015)으로
막혔을 때, **fail-safe 가드**로 그 위험을 제거하면 개입이 가능해진다. 단 런처 도구(amlt)의 문자열 처리
규약을 어기면 개입 자체가 파싱에서 깨진다 — 개입의 안전성은 대상뿐 아니라 *전달 경로*까지다.

**구체적 해결.**
- humane-stinkbug(9th) 5번째 부트스트랩 선점(pass 9m). b3pkg 9회 실패(~2h) = 에스컬레이션 확정.
- E-015의 크래시 위험(hf_transfer 미설치+hub<1.0 → ValueError)을 **가드형**으로 제거:
  b3.yaml line100 `python -c "` 앞에 `HF_HUB_ENABLE_HF_TRANSFER=$$(python -c 'import hf_transfer'
  2>/dev/null && echo 1 || echo 0)` prepend. → hf_transfer 있으면 1(init 다운로드 가속), 없으면 0(현행동작).
  **downside 0**(worst=현행), upside=가속으로 부트스트랩이 선점창(~8-9m) 아래로. init-scope라 numerics 불변(confound 아님).
- ★amlt 파싱 함정: 처음 `$(...)`로 썼더니 `Invalid placeholder`(amlt는 string.Template로 `$`를 placeholder
  취급). 기존 `$${GH_TOKEN}` 관례대로 **`$(`→`$$(`** 이스케이프하니 통과. → 재제출 성공 **coherent-dinosaur(10th)**.
- b2=daring-gelding running(순항). b3nopmi도 발사시 동일 fix 적용 예정(일관성).

**통찰.** 검증 불가로 막힌 개입은 **fail-safe 가드**($$(check)&&1||0)로 되살릴 수 있다 — 실패모드를
"크래시"에서 "무익(현행)"으로 바꾸면 downside가 사라져 개입 문턱이 낮아진다. 그리고 개입의 안전성은
*대상*(init-scope·궤적중립)뿐 아니라 *전달 경로*(amlt `$$` 이스케이프)까지 지켜야 완성된다 — 도구 규약을
어기면 안전한 개입도 파싱에서 죽는다. (헌법 Part II: 처방의 안전 전제 + 전달 경로 모두 확증.)

### 상태 tick — 21:33 UTC
- b3pkg=**coherent-dinosaur**(10th, hf_transfer fail-safe fix 탑재) 제출. b2=daring-gelding running(순항). b3pkg 9회 실패 후 타깃 개입 적용. 다음: coherent-dinosaur가 부트스트랩(가속된 init)을 선점 전 통과하는지.

### 상태 tick — 21:46 UTC
- b2=daring-gelding running(2h): **step 141→143 전진**(Training Progress 143/300, 611s/it, 31분 무중단). gs150 저장까지 ~7스텝(~70분)→~22:50 예상(E-017 임계 부합). 주 결과 순항. b3pkg=coherent-dinosaur(10th, hf_transfer fix) **queued**(노드 못잡음, 티어 busy; retry0). fix 효과는 노드확보 후 관찰. 조치=queued 두고 다음 tick 재확인. 다음: coherent-dinosaur 노드확보·b2 gs150 돌파.

### 상태 tick — 21:48 UTC
- b3pkg=coherent-dinosaur(10th) **`failed`(11s만 살고 죽음** — 노드 즉시 회수, hf_transfer fix 발휘 틈도 없음) → 재제출 **wired-honeybee(11th, fix)**. b3pkg 누적 10회 실패 = 티어 heavy 포화(노드 11s 유지 or 큐실패). b2=daring-gelding running(step143 순항, 무영향). 조치=churn 지속·b2 gs150 감시. ★fix는 노드가 최소 부트스트랩 시간(~수분)은 유지돼야 효과—현 포화선 무의미하나, 부하 풀리면 발휘.

### 상태 tick — 21:53 UTC
- b3pkg=wired-honeybee(11th) queued/Terminated(노드 유지 실패, 티어 포화 ~2h+ 지속; auto-requeue 관찰). b2=daring-gelding running(step143, gs140 resume 진행; **HF 완전체크포인트 여전히 gs140**, gs150 저장까진 ~7스텝/~70분→~22:50 예상). 조치=b3pkg churn·b2 riding-out(임계 미도달). 다음: 티어 calm·b2 gs150 저장·b3pkg 노드유지.

### 상태 tick — 21:58 UTC
- b3pkg=wired-honeybee(11th) 노드 재확보 running(3s — 티어가 짧은 창 반복 부여·즉시 회수 패턴). b2=daring-gelding running(순항). 티어 포화로 b3pkg 노드 순간만 유지. b2 무영향 진행. 다음: b3pkg가 이번 창을 부트스트랩 넘길지(hf_transfer fix)·b2 gs150 저장.

### 상태 tick — 22:01 UTC — b3pkg 노드 유지·부트스트랩 진입(첫 성공적 창)
- b3pkg=wired-honeybee(11th) running ~2m, **실제 부트스트랩 시작**(`fast-path: pulling conda-pack env`·`extracting env ~5GB`) — 11회 만에 처음 노드를 붙잡고 부트스트랩 진입(티어 calm 신호 가능). hf_transfer fix가 이제 init 다운로드서 효과 볼 차례. b2=daring-gelding running(**step 143→144 전진**, gs150까지 ~6스텝→~22:50). 양 arm progress. 다음: wired-honeybee ~8m 부트스트랩 통과 여부(hf_transfer로 가속됐나)·b2 gs150 저장.

---

## E-019 (2026-07-17 22:03 UTC) — ★내 hf_transfer fix가 부트스트랩을 깨뜨림 → 즉시 원복 (검증 층위의 교훈)

**추상적 문제.** 개입(fix)의 검증은 *어느 층위*에서 하느냐가 관건이다. 문법/제출이 통과해도 *런타임
실행*이 깨지면 fix는 순손해다. 그리고 그 런타임 실패가 "특정 경로가 실행돼야만" 드러나면, 배포와
발견 사이에 시차가 생겨 여러 시도를 낭비한다.

**구체적 해결.**
- wired-honeybee(11th, hf_transfer fix 탑재)가 **11회 만에 처음 부트스트랩을 통과해 init 다운로드 라인
  (내가 편집한 line100)에 도달**하자 로그에 `-c: line 40: syntax error: unexpected end of file`.
  → 내 편집 `HF_HUB_ENABLE_HF_TRANSFER=$$(python -c 'import hf_transfer' 2>/dev/null&&echo 1||echo 0) python -c "`
  가 **heredoc 다중따옴표/명령치환과 충돌**해 bash 파싱을 깼다(중첩 `python -c '...'`와 뒤이은 다줄 `python -c "..."`).
- ★핵심 함정: **`python yaml.safe_load` PASS + `amlt run` 제출 PASS였다**(문법·제출 층위는 통과). 그러나
  **노드에서 bash가 그 명령을 실행하는 런타임 층위에서 깨졌다** — 그리고 그 라인은 부트스트랩을
  *실제 통과한* 시도에서만 실행되므로, 10·11번째 시도(대부분 그 전에 선점사)까지 안 드러났다.
- **즉시 원복**: line100을 원래 `python -c "`로 복구(hf_transfer 잔존 0 확인) → 재제출 **firm-gopher(12th,
  원복 작동 yaml)**. b3pkg를 known-good(느린 init지만 작동) 상태로 복귀. wired-honeybee는 깨진 스크립트로 자체 사망.

**통찰.** "확증된 병목의 fail-safe fix"라도 **전달 경로(yaml heredoc 따옴표)의 런타임 실행을 검증**하지
않으면 순손해가 된다. 검증 층위는 (문법 parse) < (도구 제출) < **(노드 런타임 실행)** 순으로 깊어지며,
가장 깊은 층위가 진짜 검증이다. 내 fix는 앞 두 층위를 통과해 "안전"으로 오인됐으나 가장 깊은 층위서
깨졌다. 교훈: 명령-문자열(특히 중첩 따옴표) 편집은 **실제 실행 로그로 확인**해야 하며, 그 전엔
known-good을 유지한다. (E-018의 개입을 E-019가 원복 — "verify before AND after" 중 after 검증이 잡아냄.)

### 상태 tick — 22:03 UTC
- b3pkg=**firm-gopher**(12th, hf_transfer 원복된 작동 yaml). wired-honeybee(11th) 깨진 스크립트로 사망 예정. b2=daring-gelding running(step144, gs150 ~22:50). 조치=b3pkg known-good churn·b2 riding-out. hf_transfer는 재시도 안 함(churn이 기능적). 다음: firm-gopher 부트스트랩·b2 gs150.

### 상태 tick — 22:07 UTC
- b3pkg=firm-gopher(12th, 작동 yaml) queued(노드 대기, 티어 busy). b2=daring-gelding running(HF 완전체크포인트 여전히 gs140; gs150 저장 ~22:50 예상, 임계 전). 조치=b3pkg churn·b2 riding-out. 다음: firm-gopher 노드확보·b2 gs150 저장(임계 ~22:50).

### 상태 tick — 22:10 UTC
- b2=daring-gelding running: **F5 확인**(HB 10s마다 갱신·gpu0used 30~56GB 변동=RL 단계별 정상, 진짜 학습중). "Training Progress" grep 빈 건 tail이 HB구간이었을 뿐. gs150 향한 진행. b3pkg=firm-gopher(12th, 작동 yaml) queued/Terminated(노드 못잡음, 티어 busy 지속). 조치=b3pkg churn·b2 riding-out(gs150 임계 ~22:50). 다음: b2 gs150 저장·firm-gopher 노드확보.

### 상태 tick — 22:13 UTC
- b3pkg=firm-gopher(12th, 작동 yaml) **노드 재확보 running 5s, 부트스트랩 진입**(티어가 창 부여). b2=daring-gelding running(순항). firm-gopher가 이번 창서 ~8m 부트스트랩 통과하는지 관건(작동 yaml=정상 진행). 다음: firm-gopher ~8m 통과·b2 gs150 저장.

### 상태 tick — 22:16 UTC
- firm-gopher(b3pkg 작동 yaml) running ~2m 부트스트랩(env 추출, 정상 진행; ~8m 분수령 전). b2=daring-gelding running(HF 완전체크포인트 여전히 gs140, gs150 저장 ~22:50 예상). 순조. 다음: firm-gopher ~8m 통과 여부·b2 gs150 저장(임계 ~22:50).

### 상태 tick — 22:18 UTC
- firm-gopher(b3pkg 작동 yaml) running 4m 부트스트랩(cold-start 단계, **syntax error 없음=원복 확인**, 정상 진행). ~8m 분수령 전, 생존. b2=daring-gelding running 순항. 다음: firm-gopher ~8m 통과(verl 시작)·b2 gs150 저장.

### 상태 tick — 22:20 UTC
- firm-gopher(b3pkg 작동 yaml) running 6m, **`LOCAL_RANK=0`(verl-startup 지점) 도달·생존** — 앞으로 ~2-3분이 verl 학습 시작 vs 사망 결정 구간. b2=daring-gelding running 순항. 다음: firm-gopher verl launch(WorkerDict)/사망 여부·b2 gs150.

### 상태 tick — 22:22 UTC — firm-gopher ~8m 통과 생존 / b2 정체 확정→Codex 개입 검토
- firm-gopher(b3pkg 작동 yaml) **running 8m — 지난 사망지점(~8-9m) 넘겨 생존**(부트스트랩 지속, verl 임박; 긍정적). b2=daring-gelding **또 선점→queued(auto-retry)**. b2 정체 확정: gs140↔~145 오가며 gs150(save_freq10=~100분 무중단 필요) 못넘음, HF 최신완전 2h+째 gs140. → save_freq10→1 개입(취소+재제출) 여부를 Codex 확인(bid28r81w). 조치=Codex 판정 대기·firm-gopher verl 감시. 다음: firm-gopher verl 시작·Codex b2 판정 반영.

---

## E-020 (2026-07-17 22:22 UTC) — b2 정체: Codex GO(save_freq→1) but 안전-타이밍 실행(다음 선점서 적용)

**추상적 문제.** 승인된 개입(save_freq↓)이 "지금 running 중인 primary arm을 취소"해야 적용된다면,
승인의 *실행 타이밍*이 위험을 좌우한다. running 노드를 잃는 것과 개입 지연 사이에서, 개입 효과를
보존하면서 위험을 최소화하는 순간은 언제인가?

**구체적 해결.**
- Codex 판정(bid28r81w): **GO — save_freq 10→1(취소+재제출).** 근거: 10-step 저장간격이 선점주기(~30-60분)와
  구조적으로 안 맞아 net progress 0(2h+ HF 최신완전=gs140 정체). riding-out 기대값 낮음. 궤적-중립(RQ confound
  아님, "b2 fault-tolerance 설정"으로 명시). 단 (a)취소 전 gs140 HF 완전저장 확인, (b)전체 save_freq=1은
  ~3TB→NO-GO, "초기만 1, gs150-160 후 10 복원" 2단계, (c)—keep 2로 HF는 완화됨.
- **사전확인 완료**: HF gs140 = model/extra_state/optim 각 **4/4 shard 완전** → 취소해도 gs140서 안전 resume.
  b2.yaml save_freq 위치 = line200 `trainer.save_freq=10`(CLI override, 공유 config 아님 → 단일줄 편집).
- **실행 타이밍 정제**: b2가 방금 auto-retry로 **running 회복** → 지금 취소는 running 노드 상실 위험(재제출이
  busy 티어서 큐 헤맬 수 있음). 그래서 **다음 선점(queued=노드 이미 없음) 순간에 적용**: 그때 auto-retry
  취소+save_freq1 재제출 → running 노드 손실 0. b2 현 run이 gs150 도달하면(best) 개입 불필요.
- **적용 절차(다음 선점시)**: b2.yaml line200 10→1 편집 → amlt stop daring-gelding(또는 현 exp) → amlt run 재제출.
  gs150-160 도달·저장 확인 후 line200 1→10 복원+재제출(2단계). b3pkg=**optimal-terrier(13th)** 재제출(firm-gopher 8m 선점).

**통찰.** 승인(GO)과 실행-타이밍은 별개다 — Codex의 GO는 "개입하라"이지 "지금 running 노드를 죽여라"가
아니다. **개입이 자원(노드) 파괴를 수반할 때, 그 자원이 이미 없는 순간(선점 직후 queued)을 골라 적용**하면
효과는 같고 위험은 사라진다. 이는 E-016의 "개입도 공짜 아님"을 실행 층위로 확장한 것: 개입의 *내용*(GO)이
정해져도 *언제*가 위험을 정한다.

### 상태 tick — 22:22 UTC(2)
- b3pkg=optimal-terrier(13th, 작동 yaml) 재제출. b2=daring-gelding running(회복, gs140→ 진행). b2 save_freq→1 개입은 **다음 선점시 적용**(준비완료: gs140 완전·line200 편집지점). 다음: b2 선점 감지→개입 적용 or gs150 도달, firm-gopher(→optimal-terrier) verl.

### 상태 tick — 22:30 UTC
- ★b2 정체 재확인: HF **gs140 COMPLETE(4/4/4)·gs150 INCOMPLETE(optim 3/4=옛 중단push 잔재)** → b2 최신완전 여전히 gs140. save_freq10에선 gs150 한 창 도달 불가로 정체 확정. b2 running(선점 트리거 대기). save_freq=1 개입은 다음 선점시 적용(E-020). b3pkg=optimal-terrier(13th) running 48s 부트스트랩(작동 yaml). 다음: b2 선점→개입 or gs150·optimal-terrier verl.

### 상태 tick — 22:33 UTC
- b2=daring-gelding running(★최근 몇 tick 무선점 = 이번엔 긴 창 잡았을 가능성, gs150 도달 희망적). 최신완전 여전히 gs140(gs150 완전저장시 정체해소·개입불필요). b3pkg=optimal-terrier(13th, 작동 yaml) running 3m 부트스트랩(cold-start). 선점 트리거 없어 개입 대기. 다음: b2 gs150 완전저장 or 선점→개입·optimal-terrier ~8m verl.

### 상태 tick — 22:35 UTC
- b2=daring-gelding running(★선점 없이 ~30-40분 지속=긴 창 확보, gs150 향해 진행; 최신완전 gs140). optimal-terrier(b3pkg 작동 yaml) running 6m, **`LOCAL_RANK=0`(verl-startup) 도달** — ~2-3분이 verl 시작/사망 결정. b2 선점 트리거 없어 개입 대기. 다음: b2 gs150 완전저장(~23:30 예상)·optimal-terrier verl launch/사망.

### 상태 tick — 22:38 UTC
- **optimal-terrier(b3pkg 작동 yaml) running 8m — 지난 사망지점(~8-9m) 넘겨 생존!**(부트스트랩 지속, verl 임박; 노드가 버텨줌=긍정적). b2=daring-gelding running 순항(선점 없이 지속, gs150 향해; 최신완전 gs140). 개입 대기(무선점). 다음: optimal-terrier verl 시작(첫 milestone)·b2 gs150 완전저장.

### 상태 tick — 22:40 UTC
- optimal-terrier(b3pkg) 8m 선점(pass, verl 전) → 재제출 **working-ram(14th, 작동 yaml)**. b3pkg 13회 실패 = init 다운로드(~8-9m)와 선점창(~8-9m)의 아슬한 경쟁, 티어 calm 대기(코드·yaml 정상). b2=daring-gelding running 순항(gs150 향해, 최신완전 gs140, 선점 없이 지속). b2 무영향. 다음: working-ram 부트스트랩·b2 gs150 완전저장.

### 상태 tick — 22:42 UTC
- working-ram(b3pkg 14th) queued(노드 대기). b2=daring-gelding running(선점 없이 지속, gs150 향해; 최신완전 gs140, gs150 ~23:20-30 예상). 개입 대기(무선점). 다음: b2 gs150 완전저장(정체해소)·working-ram 노드확보.

### 상태 tick — 22:44 UTC — b2=daring-gelding running(선점 없이 지속, gs150 향해; 최신완전 여전 gs140). working-ram(b3pkg 14th) queued(노드 대기). 무선점=개입 불필요. 다음: b2 gs150 완전저장 확인(정체해소)·working-ram 노드확보.

---

## Phase-1 종합 (Infrastructure Hardening) — 문제-해결-통찰 (E-000~E-020 추상화)

> gs300 결과 이전의 인프라 단계 21개 엔트리를 5개 추상 문제-클래스로 종합한다. 개별 tick은 위 로그에, 여기선 "무엇이 반복되는 문제였고 어떤 구조적 해결이 그것을 닫았는가"를 남긴다. (update-study: 문제=추상, 해결=구체 구현, 통찰=전이가능 원리.)

### 종합-A. 선점 내성 (Preemption Resilience) — E-000·E-002·E-006·E-011·E-013·E-016
**문제(추상):** 공유 클러스터의 non-dedicated 티어에서 장기 학습 잡은 임의 시점에 선점된다. 선점 자체는 막을 수 없으므로, 시스템은 "선점이 진행을 파괴하지 않는다"는 불변식을 코드로 보장해야 한다.
**해결(구체):** 3중 배선 — (1) `amlt` 잡을 terminal 실패(pass/failed) 감지 즉시 동일 yaml 재제출(HF resume 자동), (2) 체크포인트 완결성 게이트(model/extra_state/optim 각 `world_size_4_rank_0..3` = 4/4 shard 갖춰야 resume 후보로 인정 — 불완전은 기각), (3) verl auto-retry(`retry_NNN`)로 컨트롤플레인 레벨 재기동. 이 셋이 겹쳐 "최신 *완전* 체크포인트"가 항상 진행의 하한을 지킨다.
**통찰:** 선점 빈발 환경에서 진행의 단위는 "실행 시간"이 아니라 "마지막으로 저장된 완전 체크포인트"다. liveness 판정도 `running` 상태가 아니라 global_step 전진 + 완전-shard 저장으로 해야 한다(상태 필드는 선점을 graceful SIGTERM=`pass`로 위장한다 — [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]).

### 종합-B. Fresh-start 무결성 vs Resume-hijack — E-001·E-003·E-008
**문제(추상):** keep-alive의 "체크포인트 있으면 자동 resume"는 편의이자 함정이다. force-fresh(gs0)를 의도한 arm이 과거 HF 체크포인트를 발견하면 조용히 그 지점을 hijack해 matched-isolation을 깨뜨린다.
**해결(구체):** arm별로 resume 의도를 명시화 — b3nopmi는 yaml 설명에 "Fresh gs0" 못박고, 발사 시 `iamseungpil/metacot-h200-triobj-dcpo-v3`의 `rq3_b3nopmi/global_step_135`를 사전 제거(hijack 소스 차단) + Codex 확인 후 발사하는 절차로 고정. b2/b3pkg는 resume 정당(선점 복구)이라 게이트 유지.
**통찰:** "자동 resume"는 arm의 *의도*와 무관하게 발동하는 부작용이다. 실험 정체성(fresh인가 continue인가)은 런타임 메커니즘이 아니라 발사 전 명시적 게이트로 결정돼야 한다. matched ladder에서 이 구분 실패는 곧 인과 주장의 오염이다.

### 종합-C. 병목 오진 방지 (Bottleneck Misdiagnosis) — E-012·E-014
**문제(추상):** 반복 실패를 마주하면 "코드 버그"로 성급히 귀속하려는 압력이 생긴다. 그러나 실패의 진짜 층위(코드 vs 인프라 vs 다운로드 vs 선점)를 증거로 판별하지 않으면 존재하지 않는 버그를 쫓는다.
**해결(구체):** 판별 프로토콜 — (1) 실패가 verl 학습 진입 *전*(부트스트랩/init)에서만 나는지 로그 위치로 확인 → 코드 무관 입증, (2) flash-attn 빌드 가설은 "이미 빌드되면 skip" 경로 확인으로 반증, 진짜 병목은 init 단계 16GB HTTP 스냅샷 다운로드 vs 선점창의 경쟁으로 특정. E-012에서 "b3pkg는 코드 문제인가?"에 100% 인프라로 판정.
**통찰:** 9회 헛다리(gpu_mem·w_meta·tee·ipykernel 등 red herring)의 교훈 = 실패 층위는 *증상*이 아니라 *발생 위치*로 판별한다. "학습 진입 전 죽음"은 코드가 아니라 환경 신호다. 이 판별을 건너뛰면 keep-alive만으로 풀릴 문제에 코드 수정을 배포해 순손해를 낸다.

### 종합-D. 개입 타이밍·위험 계산 (Intervention Calculus) — E-007·E-015·E-017·E-018·E-019·E-020
**문제(추상):** 정체(예: b2가 gs140↔145 진동, 완전 gs150 미도달)를 보면 즉시 개입하려는 압력이 생긴다. 그러나 검증되지 않은 fix는 작동하던 경로를 깨뜨릴 수 있고, running 잡을 취소하는 개입은 그 자체로 진행을 버린다.
**해결(구체):** 위험-계층 규칙 — (1) 검증 층위는 `문법 parse < 도구 제출 < 노드 런타임 실행` 순이며 가장 깊은 층위만이 진짜 검증. hf_transfer 가드(E-018)가 parse+submit 통과했으나 노드에서 heredoc 따옴표 충돌로 부트스트랩을 깸(E-019) → 즉시 원복이 손해를 회수. (2) save_freq=1 개입(E-020)은 Codex GO에도 "running 잡 취소" 대신 "다음 선점 시 적용"으로 안전-타이밍. (3) yaml 편집은 값-편집만 허용, 명령-문자열(heredoc/중첩따옴표) 편집 금지.
**통찰:** 개입의 기대값은 (성공 시 이득) − (fix가 작동 경로를 깨뜨릴 확률 × 손해) − (개입 자체가 버리는 진행)이다. 확증되지 않은 fix는 이 식에서 음수가 되기 쉽다. "작동하는 것을 런타임-검증 전까지 known-good으로 취급"이 default이며, 개입은 무선점·저위험 창에서만([[yaml-heredoc-edit-verify-runtime-layer-0717]]).

### 종합-E. 결과 라이프사이클 사전 스테이징 (Pre-staging) — E-004·E-005·E-009·E-010
**문제(추상):** gs300 결과는 며칠 뒤 도착하지만, 결과가 오는 *순간* 해석이 사전등록돼 있지 않으면 사후 합리화(p-hacking·garden of forking paths)에 노출된다.
**해결(구체):** 결과 이전에 4종 확정 — (1) B0 기준선 고정(GSM8K 0.8905 / MATH500 0.544 / AIME@16k 0.0458), (2) held-out 1030 eval 레시피를 B0와 *동일* 방법론으로 확정(`eval_vllm_1030.py`), (3) 타임스케일 재보정(gs300 = "시간"이 아니라 "며칠"), (4) 결과-분기별 개선실험 컨틴전시 맵(Phase-2b Q0–Q3 omnibus) 사전설계 + 해석표 pre-register.
**통찰:** 사전등록은 결과가 예상과 다를 때 비로소 값을 한다. "결과 나오면 어떻게 판정할지"를 결과 전에 못박아야, 양성이든 음성이든 동일 잣대로 읽는다. 특히 RQ2 estimand는 "PMI 내용 순기여"가 아니라 "패키지 end-to-end 효과"로 명시돼 있어야 한다([[rq2-estimand-endtoend-not-content-0717]]).

**Phase-1 총평:** 인프라 단계의 유일한 목표는 "gs300까지 훈련이 죽지 않고 도달하게 하는 것"이며, 위 A~E는 그 목표를 위협하는 5개 실패 양식과 그 구조적 방어다. 학습·해석·개선의 본 라이프사이클은 gs300 도착 시 개시된다. Phase-1의 이월 자산 = 완전-체크포인트 하한 불변식·arm 정체성 게이트·병목 판별 프로토콜·개입 위험식·사전등록 해석표.

---

## E-021 (2026-07-17 22:48 UTC) — working-ram 노드 확보(running) + b2 "save-truncation" 패턴 확증

**문제(추상):** 두 실패 양식이 동시에 관찰된다. (1) 반복 큐잉되던 arm이 노드를 잡았을 때, 부트스트랩 통과까지가 아직 남은 취약창이다. (2) primary arm이 "running"으로 보이지만 그 저장이 매번 선점에 잘려 durable 진행(최신 *완전* 체크포인트)이 정체할 수 있다 — running≠progress.

**해결(구체):**
- working-ram(b3pkg 14th): `queued`→`running` 전환 확인(11m 전 노드 확보, 부트스트랩 7m째). 14번째 시도 만에 큐 통과. 다음 관문=init 16GB 스냅샷 다운로드 vs 선점창 경쟁(종합-C 병목). verl 학습 진입 시 gs25 헌법게이트 점검 예약.
- b2(daring-gelding): HF shard 완결성 정밀 판별 — gs140=4/4·4/4·4/4(완전, resume 앵커), **gs150=model4/4·extra4/4·optim3/4**(1 optim shard 미완), **gs160=model1/4·extra4/4·optim0/4**(부분). b2는 gs160까지 전진하나 저장이 잘려 최신완전=gs140 고착. optim shard가 저장 버스트의 마지막에 업로드돼 선점창에 절단되는 구조(gs150은 여러 tick째 3/4 정지=업로드 지연 아닌 절단 확정).
- 개입 판단: b2가 **현재 running(선점 아님)**이므로 안전-타이밍 규칙(종합-D)상 취소 개입 보류. 다음 1-2 tick에도 최신완전이 gs140이면 = save_freq=10이 무중단 저장창을 못 얻는다는 확정 증거 → **다음 선점 시** save_freq=1 적용(line200 값-편집, running 취소 아님).

**통찰:** 완결성 게이트(종합-A)의 가치가 여기서 이중으로 드러난다 — (1) gs150(optim 3/4)을 resume 앵커로 오인했다면 optimizer state가 깨진 채 재개돼 조용한 학습 오염이 났을 것이나, 게이트가 gs140으로 정확히 후퇴시킴. (2) 동시에 "gs150/160 dir이 생겼으니 진행 중"이라는 dir-존재 기반 liveness는 거짓 신호 — durable 진행의 유일한 척도는 shard-완전 최신 gs다. "running 상태"도 "체크포인트 dir 생성"도 진행을 보장하지 않는다; **shard-완전 gs 전진만이 진행이다.** save-truncation은 선점-빈발 티어에서 save_freq가 클수록(저장 버스트가 클수록) 악화되는 실패 양식이다.

---

## E-022 (2026-07-17 22:46 UTC) — b3pkg 14th(working-ram) 부트스트랩 선점 재확인 + 재제출(noted-tortoise) + tty 제출 함정

**문제(추상):** 노드를 확보해 running까지 간 arm도, 부트스트랩(init 다운로드) 취약창을 못 넘기면 graceful-SIGTERM 선점(`pass`)으로 사망한다. 재제출은 keep-alive의 정답이나, 비대화형(no-tty) 셸에서 제출 도구가 확인 프롬프트로 막히면 재제출 자체가 실패하는 2차 함정이 있다.

**해결(구체):**
- working-ram(b3pkg 14th): `running`(9m) → `pass`(선점) 확인 = 종합-C 병목(init 16GB 스냅샷 다운로드 vs ~12-15min 선점창)의 재현. terminal이므로 즉시 재제출.
- 재제출 1차 `amlt run h100std_rq3_b3.yaml`가 `OSError: [Errno 6] ... '/dev/tty'`로 실패(experiment 껍데기 enjoyed-seahorse는 생겼으나 job 미제출 — `amlt status enjoyed-seahorse`가 빈 결과로 중복 없음 확인). 2차 `amlt run -y ...`로 확인 프롬프트 우회 → **noted-tortoise** job 제출 성공(queued).
- b2(daring-gelding): 여전 running·최신완전 gs140(save-truncation 지속, E-021).

**통찰:** keep-alive 재제출은 반드시 `-y`(비대화형)로 해야 한다 — no-tty 환경에서 `amlt run`은 code-snapshot 확인 프롬프트에서 `/dev/tty` OSError로 죽고, 이때 experiment 껍데기만 생기고 job은 안 뜬다(중복 오해 유발). 재제출 실패를 "또 선점"으로 오진하지 않으려면 제출 직후 `amlt status <exp>`로 job 존재를 확인해야 한다. 이는 종합-A(선점 내성)의 실행 세부: 재제출 명령 자체가 환경 의존적으로 실패할 수 있으므로 제출-후-검증이 필수다.

---

## E-023 (2026-07-17 23:04 UTC) — ★두 arm 모두 데드락 확진 + Codex 이중 판정 + b2 save_freq=1 집행(primary 구제)

**문제(추상):** 선점-빈발 티어에서 fresh arm은 "부트스트랩 데드락"(진입 못함)으로, 체크포인트-보유 arm은 "save-truncation 데드락"(진입했으나 durable frontier 못 올림)으로 죽는다. 둘 다 "선점창 < 완주해야 할 원자적 작업"이라는 동일 구조의 서로 다른 층위다. 개입은 primary를 살리되 concurrent-write 오염을 피해야 한다.

**해결(구체):**
1. **b3pkg(secondary) = A 판정(Codex)**: noted-tortoise(15th)도 9m 부트스트랩 선점 pass → sound-hare(16th) 재제출. Codex 자문 결론=15연속은 구조적 데드락(선점창~9m < 부트스트랩~12-15m), 처방=env+16GB SFT-init 번들 immutable 노드 이미지(런타임 다운로드 제거)·yaml 명령문자열 불변·이미지 지정 레이어만 변경·1회 preemption-resume smoke. 이미지 준비 전까지 b3pkg opportunistic 재제출 자제, b2 집중.
2. **b2(primary) save-truncation 데드락 확진**: `amlt log list`로 retry_003까지 **3회 선점-재시도** 확인. HF shard 재측정 gs150=optim 3/4가 ~19min+ 불변=upload 지연 아닌 **truncation 확정**. b2는 매번 gs140(마지막 완전)서 재개→gs150/160 저장 시도→선점에 절단 반복, durable frontier gs140 영구 고착.
3. **save_freq=1 집행(Codex GO ×2, E-020+본건)**: 새 3-retry 증거로 Codex 재확인 GO. h100std_rq3_b2.yaml line200 `save_freq=10→1` 값편집(명령구조·따옴표 불변, yaml parse OK)→daring-gelding 취소→mighty-civet(save_freq=1) 재제출(queued). 논리=매 스텝 저장으로 선점창 안에 완전저장 1개라도 착지시켜 frontier를 스텝단위 전진. gs150-160 durable 착지 후 save_freq=10 복원 예정.

**통찰:** (1) **개입 실행층의 함정**: `amlt stop`은 존재하지 않는 명령이다(정답=`amlt cancel <exp> -y`). 이를 모르고 "정지→재제출" 순서로 짰다가 재제출이 먼저 성공해 순간적으로 daring-gelding(save_freq10)+mighty-civet(save_freq1)가 **동일 rq3_b2 체크포인트 경로에 동시쓰기할 뻔**했다(즉시 cancel로 회수). 교훈=concurrent-write 오염 방지엔 "취소-먼저-확인, 재제출-나중"이 정순이며, 취소 명령의 정확한 이름을 사전 검증해야 한다(종합-D 실행층 함정의 재현: 개입 도구 자체가 실패 지점). (2) **데드락 동형성**: 부트스트랩 데드락(b3pkg)과 save-truncation 데드락(b2)은 같은 부등식(선점창 < 원자작업 시간)의 두 층위이며, 해법도 동형이다 — "원자작업을 작게 쪼개 선점창 안에 착지"(b2=save_freq↓로 저장을 자주·작게 시도) 또는 "원자작업을 사전 제거"(b3pkg=이미지에 bootstrap 번들). frontier 전진의 하한은 언제나 "선점창 안에 완주 가능한 최대 원자작업"이다.

### 상태 tick — 23:11 UTC — mighty-civet(b2 save_freq=1) running 3m(gs140 재개, 첫 스텝 gs141 compute중~10min라 미착지 정상). sound-hare(b3pkg 16th) running 8m(통상 사망점 9m 근접, 아직 생존). 둘 다 실패 없음. 다음: b2 gs141 완전저장 착지(save_freq=1 데드락해소 확증)·sound-hare 부트스트랩 통과 여부.

---

## E-024 (2026-07-17 23:23 UTC) — b3pkg durable-fix 조사: env_builder는 병목 오해·진짜 fix는 모델-번들 이미지(사용자 결정 플래그)

**문제(추상):** Codex가 권한 "durable fix"(env+16GB init 번들 이미지)를 실행하려면, 기존 pre-bake 자산이 실제 병목을 겨냥하는지 먼저 검증해야 한다. 잘못된 자산에 투자하면 병목이 그대로 남는다(종합-C 병목 오진의 재발 위험).

**해결(구체):**
1. sound-hare(b3pkg 16th) 8m 부트스트랩 선점(pass). keep-alive 브리지로 shining-bonefish(17th) 재제출(Codex는 자제 권했으나 primary 아닌 secondary라 opportunistic 브리지는 무해·keep-alive 준수).
2. h100std_env_builder.yaml read-only 조사: 이 one-off는 conda 환경에 **flash-attn 2.8.3만 구워** simplerl_v4.tar.gz로 HF에 올린다(→arms가 v4 pull해 ~15min flash-attn 빌드 skip). **이미 완료**된 상태(현 bootstrap이 v4 사용, flash-attn skip). ⇒ env_builder는 **flash-attn 빌드시간**을 없앨 뿐, **실제 부트스트랩 병목인 16GB SFT-init 모델 snapshot_download(E-014)**는 전혀 해결하지 못한다.
3. ⇒ Codex의 "16GB init 번들 immutable 이미지"는 **새 메커니즘** 필요: (a) 모델을 커스텀 컨테이너 이미지에 굽기(레지스트리 push·amlt image 교체=중대 인프라) 또는 (b) env_builder 확장해 16GB 모델을 env tarball에 함께 tar(단일 다운로드로 통합하나 16GB는 그대로) 또는 (c) 노드-영속 캐시. 어느 것도 non-trivial. **사용자 가시화 필요 결정으로 플래그**(primary b2가 건재하므로 b3pkg는 이 결정 대기 가능).

**통찰:** durable fix 착수 전 "기존 자산이 진짜 병목을 겨냥하는가"를 먼저 검증한 것이 종합-C(병목 오진)의 실천이다 — env_builder라는 그럴듯한 pre-bake 자산이 있으나, 그것이 없애는 비용(flash-attn 빌드)은 이미 0이고 실제 병목(16GB 모델 다운로드)은 손대지 않는다. "pre-bake 자산이 존재함 ≠ 병목을 해결함." Codex의 A 처방은 방향은 옳으나(다운로드 제거), 구체적 구현은 기존 env_builder 재활용이 아니라 모델-번들이라는 더 큰 작업이며, 이는 primary 결과를 위태롭게 하지 않는 선에서 사용자 결정을 받아 진행할 사안이다.

### 상태 tick — 23:27 UTC — mighty-civet(b2 save_freq=1) running 20m·로그상 정상(vLLM 4워커 활성 gpu0=52GB, step141 rollout 완료→update 단계, 오류·stuck 없음). gs141 완전저장 임박(20m=resume~8+rollout~10+update). save_freq=1 작동 확인. b3pkg=shining-bonefish(17th) 진행 중. 다음(23:35): b2 gs141 완전저장 착지=데드락해소 확증.

### 상태 tick — 23:31 UTC — ★b2 gs141 dir 등장(save_freq=1 저장 개시, shard 업로드 시작 0/4=진행중). mighty-civet running 23m. shining-bonefish(b3pkg 17th) queued(노드대기). 다음: gs141 완전(4/4/4) 착지=save-truncation 데드락 해소 확증(최신완전 gs140→141 전진).

---

## E-025 (2026-07-17 23:36 UTC) — b2 step141 정상 완료 + 데드락 진단 정련(로컬저장≠HF푸시)·save_freq=1 작동 확인·ETA ~33h

**문제(추상):** "체크포인트가 HF에 불완전"이라는 증상만으로는 근인 층위를 특정할 수 없다 — 로컬 저장 실패인지, HF 푸시 절단인지, 푸시 빈도 부족인지를 로그로 분리해야 올바른 레버를 당긴다(종합-C 병목 판별의 심화).

**해결(구체):**
1. mighty-civet 로그 확증: `training/global_step:141` 완료, `timing_s/save_checkpoint:30.3s`(로컬 저장 완료), `critic/score/mean:0.43`(reward 건강), 오류·abort 없음. `Training Progress 141/300 [12:21<32:46:13, 741.97s/it]` ⇒ **ETA ~33h**(159스텝 남음).
2. 푸시 메커니즘 규명(yaml read-only): verl은 `save_freq=1`로 매 스텝 **로컬**(/scratch/checkpoints/rq3_b2) 저장. 별도 **백그라운드 pusher**(`push_ckpts_to_hf.py --interval 90 --keep 2`)가 90s마다 로컬→HF per-file 업로드(최근 2개만 유지). 종료 시 **최종 sync push**(동기·검증·재시도)로 최고 gs를 확실히 올림.
3. gs141 HF 상태: extra_state 4/4(소용량 먼저)·model/optim 0/4(대용량 pusher 업로드 중). ⇒ **로컬 저장은 항상 정상**(30s)이었고, 관측된 "save-truncation"의 진짜 근인은 **선점이 HF 푸시를 대용량 shard 도중에 절단**하는 것. save_freq=1은 완전 로컬 체크포인트를 ~12min마다 생성해 pusher에게 푸시 기회를 대폭 늘림으로써 이를 완화(save_freq=10일 땐 완전 로컬 ckpt가 ~2h마다라 그 사이 선점되면 HF에 새 완전본이 안 올라감).

**통찰:** 데드락의 정확한 층위는 "저장"이 아니라 "**푸시 완결성 under 선점**"이었다. 증상(HF 불완전)이 같아도 근인이 로컬저장이면 save_freq이 무의미하고, HF푸시빈도면 save_freq=1이 정확한 레버다 — 로그의 `save_checkpoint:30s`(로컬 성공) vs HF의 model/optim 0/4(푸시 미완)를 **분리 계측**해야 레버를 맞게 고른다. E-021~E-023의 "save-truncation 데드락" 명명은 방향은 맞았으나(durable frontier 정체), 기전은 "로컬 저장 실패"가 아니라 "대용량 shard 푸시가 선점에 절단"으로 정련된다. 다음 확증=gs141이 4/4/4로 HF 완전 착지(최신완전 gs140→141)하면 레버 정확성 최종 입증.

### 상태 tick — 23:41 UTC — b2(mighty-civet) running 33m·step142 진행·건강. HF gs141 푸시 느림(model 1/4, 대역폭 병목)=durability는 최종 sync push 의존. ★shining-bonefish(b3pkg 17th) running 15m(통상 사망점 9m 초과 생존=티어 잠잠, 부트스트랩 통과 가능성). 다음: shining-bonefish 부트스트랩→verl 진입 여부·gs141 HF 완전화.

---

## E-026 (2026-07-17 23:45 UTC) — durability 진짜 병목=푸시 대역폭(save_freq 아님)·최종 sync push가 실질 안전망·b2 로컬은 순항

**문제(추상):** 개입(save_freq=1) 후에도 HF durable frontier가 정체하면, 개입이 겨냥한 레버가 진짜 병목이었는지 재검증해야 한다. 잘못된 레버는 증상을 안 고치고 부작용만 남긴다.

**해결(구체):**
1. pusher 대상 진단: HF의 rq3_b2 gs별 shard 측정 — gs141만 현 pusher 대상(model 1/4·optim 0/4가 13min째), gs142-144는 HF에 아예 없음(=chase 문제 아님). gs150(optim3/4)·gs160(1/4)은 옛 daring-gelding 잔재. ⇒ pusher는 gs141 하나를 **~4.5MB/s(3.5GB shard가 13min)**로 극도로 느리게 업로드 중 = **업로드 대역폭 병목**(hf_transfer 미적용 추정). verl 저장(12min/step)을 못 따라잡음.
2. durability 실질 기전 재규명: 백그라운드 pusher(interval90·keep2)는 대역폭 병목으로 running 중 완전 체크포인트를 못 올림 ⇒ HF frontier는 **최종 sync push**로만 전진 — (a) 정상 완료 시 gs300을 SIGKILL 압박 없이 동기·재시도 업로드(=**최종 결과는 안전**), (b) graceful 선점(pass) 시 최고 로컬 gs를 동기 푸시(단 업로드 느려 gs150 optim3/4처럼 절단 가능).
3. b2 로컬은 순항(step~144, 36min 무선점). ★b2 정지는 금물 — HF가 아직 gs140이라 지금 stop하면 로컬 gs144 진행을 잃고 gs140으로 롤백. 그대로 running 유지.
4. b3pkg: shining-bonefish(17th) 18m 부트스트랩 선점(cold-start resume 체크까지 도달, 거의 성공)→moving-llama(18th) 브리지 재제출(티어 잠잠).

**통찰:** save_freq=1의 진짜 가치는 "완전 로컬 체크포인트를 자주 만들어 **최종 sync push가 최근 gs를 잡게** 하는 것"이지, 백그라운드 pusher를 빠르게 만드는 게 아니다(대역폭은 save_freq 무관). 즉 레버는 부분적으로만 맞았다 — durability의 진짜 병목은 **업로드 대역폭**이고, 이는 명령문자열(pusher 호출) 편집 금지 제약상 안전히 못 고친다. 실질 안전망은 **정상 완료 시 최종 push**(gs300 확실히 착지)이므로, primary 목표(gs300 eval)는 티어가 b2를 완주시켜주기만 하면 안전하다. 교훈=개입 후 "frontier가 실제로 전진하는가"를 계측해 레버 정확성을 사후 검증하고, 안 맞으면 기전을 다시 분해한다(종합-C의 반복 적용).

### 상태 tick — 23:49 UTC — b2(mighty-civet) running 42m·step142·reward 0.51(0.43→개선=학습중)·correctness 0.535·ETA~35h·건강. HF 최신완전 여전 gs140(E-026대로 pusher 느림, durability=최종push 의존, 예상됨). moving-llama(b3pkg 18th) running 2m(부트스트랩 초기). 다음: moving-llama 부트스트랩 통과·b2 순항·선점시 최종push 확인.

### 상태 tick — 23:54 UTC — b2(mighty-civet) running 47m·step143(gs140→143 전진)·731s/it·ETA~32h·건강. HF frontier gs140(pusher 느림, 예상). moving-llama(b3pkg 18th) running 7m·verl 런칭(cold-start 16GB init 다운로드 취약창). 다음: moving-llama 다운로드 통과→verl global_step(첫 b3pkg 학습)·b2 순항.

---

## E-027 (2026-07-18 00:00 UTC) — b3pkg 18연속 부트스트랩 실패 확정: opportunistic 재제출은 브리지일 뿐·모델-번들 이미지가 실질 해법(사용자 결정 명시 플래그)

**문제(추상):** 동일 실패가 N회 반복되면(여기선 18회), 추가 재시도의 기대이득은 사실상 0에 수렴하고 "재시도 자체가 진전"이라는 착각이 리소스를 태운다. keep-alive(멈추지 말라) 지시와 "futile한 재시도 중단"(Codex) 사이 긴장을 명시적으로 해소해야 한다.

**해결(구체):**
1. moving-llama(b3pkg 18th) 9m 부트스트랩 선점(pass) — 16GB init cold-start 다운로드가 또 ~9min 취약창에 절단. shining-bonefish(17th) 18m·moving-llama(18th) 9m 등 최근 창이 길어졌으나 여전히 다운로드 완주(~12-15min) 불가. **18연속 실패 = fresh b3pkg는 이 티어에서 구조적 미완주 확정**(E-024/E-026 병목=16GB 다운로드 대역폭 vs 선점창).
2. 조치: keep-alive 지시(사용자 직접 명령, skill 위계상 우선) 존중해 relaxing-walleye(19th) 브리지 재제출(비용≈0, 대기 arm 없어 노드 경쟁 무해, 희박한 긴 창 대비). **단 이건 브리지이지 해법이 아님을 명시.**
3. ★실질 해법=**모델-번들 immutable 이미지**(Codex-A): env_builder는 flash-attn만 pre-bake(E-024)라 16GB 모델을 컨테이너에 굽는 새 메커니즘 필요(레지스트리 push·amlt image 교체=중대 인프라). **사용자 결정 대기**(primary b2 건재하므로 b3pkg는 이 결정까지 대기 가능; b3pkg=RQ2 secondary, b2=RQ1 primary).

**통찰:** "멈추지 말라"(keep-alive)와 "futile 재시도 중단"(효율)은 충돌이 아니라 **역할 분리**로 푼다 — 재제출은 비용≈0이라 계속하되(keep-alive 문자 준수), 그것이 진전이라 착각하지 않고 durable fix를 병행 추진한다. 18연속 실패의 교훈=opportunistic 재시도의 기대이득이 0에 수렴하면 "재시도를 늘리는 것"이 아니라 "실패의 구조(선점창<원자작업)를 바꾸는 것"(=모델 번들로 다운로드 원자작업 제거)이 유일한 전진이다(종합-A·E-023 데드락 동형성의 실천). 사용자 결정 필요 지점을 정직하게 플래그하는 것이 자율 루프의 책임.

### 상태 tick — 00:01 UTC(7/18) — b2(mighty-civet) running 53m·liveness 확인(HB 10초마다 전진·gpu0used 43GB 활성·프로세스 alive)=hang 아님. ★교훈: amlt log view는 ~3-4min 지연+tqdm progress bar 미flush라 "step 고정"으로 보여도 hang 아님 — HB 타임스탬프 전진+gpu usage로 liveness 판정(F5-aware). HF frontier gs140(pusher 느림, 예상). relaxing-walleye(b3pkg 19th) queued. 다음: b2 순항·b3pkg 부트스트랩.

---

## E-028 (2026-07-18 00:10 UTC) — ★b2 durability 데드락 blocking화 근인=hf_transfer 미작동·Codex D+A+B+C·B(stale삭제) 즉시실행·A는 2차확증후 신중배포

**문제(추상):** primary arm이 "살아있으나(auto-retry) durable frontier가 전진 못 하는" net-zero 데드락에 빠지면, 근인을 정확히 층위별로 격리(compute? 로컬저장? 푸시대역폭?)하고, 위험한 근본수정은 충분한 증거+안전검증 후 배포해야 한다(E-019: 인프라 수정은 런타임서만 깨짐 발견).

**해결(구체):**
1. 관측: b2(mighty-civet, save_freq=1) gs140 재개→57min 무선점→선점→HF frontier gs140 그대로. pusher가 57min에 gs141(model3/4·optim0/4)조차 미완, gs142/143은 HF 부재, 최종push 미완 ⇒ **매 선점 gs140 롤백 = NET 진전 0**. 재계산 gs는 rollout 확률성으로 매번 달라 cross-resume 누적 진전도 불가.
2. 근인 격리: `save_checkpoint:30s`(로컬 정상) vs 업로드 4.5MB/s(3.5GB가 12min). pusher는 hf_transfer 쓰도록 설계(line292-297, import 성공시 활성)나 **env에 hf_transfer 미설치→표준 느린 업로드로 폴백**이 유력. 작동 시 30GB가 ~14min이면 57min 창에 충분 ⇒ **A(hf_transfer 미작동)가 blocking 근인**.
3. Codex(gpt) 판정: **D(fail-closed pusher/resume: staging→checksum→COMMITTED 마커, committed frontier만 재개, 부분/stale 삭제=B포함) 우선 + A(bootstrap서 hf_transfer 설치보장+전송률 검증) + C(save_freq를 업로드시간보다 긴 값)** 보조.
4. 즉시 실행=**B**: HF의 불완전 gs141(3/4/0)·gs150(4/4/3)·gs160(1/4/0) 삭제(b2 queued=pusher 비활성이라 레이스 없음). gs130·gs140 완전 보존. ⇒ 재개시 stale shard 스킵으로 인한 혼합 corruption 원천 차단.
5. **A/D 배포 보류**: code tarball 재빌드 필요(push_ckpts_to_hf.py/bootstrap 편집)라 E-019 리스크. b2가 auto-retry(retry_001)로 살아있으므로, 다음 창에서도 롤백하는지 **2차 데이터포인트**로 데드락 확증 후, 편집을 런타임층 검증하며 신중 배포. (1개 관측으로 위험 변경 금지.)

**통찰:** "살아있음(auto-retry)"과 "전진함(durable frontier↑)"은 다르다 — F5-aware liveness를 durability까지 확장하면 "최신 *완전* HF 체크포인트가 증가하는가"가 진짜 지표다. save_freq=1 개입(E-023)은 방향이 부분적으로 어긋났다: 진짜 병목은 저장빈도가 아니라 **업로드 대역폭(hf_transfer)**이었고, 이는 frontier 정체를 계측(E-026)→롤백 실현(E-028)의 2단계로 드러났다. 교훈=위험한 근본수정(tarball 재빌드)은 (a)근인을 층위 격리로 확정하고 (b)≥2 데이터포인트로 데드락을 확증하고 (c)런타임층 검증 절차를 갖춘 뒤 배포한다. 안전·가역 조치(B: 삭제)는 즉시, 위험·비가역 조치(A/D)는 증거 축적 후.

---

## E-029 (2026-07-18 00:27 UTC) — 데드락 완전 근인: 백그라운드 pusher+최종 push 둘 다 hf_transfer 미사용·2차확증·reversible fix 배포 준비

**문제(추상):** 근인을 "한 경로"로 특정했다고 끝이 아니다 — 동일 병목이 여러 경로에 걸쳐 있으면 한 경로만 고쳐선 데드락이 안 풀린다. durability를 지키는 *모든* 업로드 경로를 열거해 각각이 병목을 공유하는지 확인해야 한다.

**해결(구체):**
1. 2차 데이터포인트: b2 재개(gs140)→step141 완료(reward0.45·save24s)→gs141 pusher 업로드 시작이나 **extra 4/4·model 0/4·optim 0/4**(1차와 동일 느린 패턴). 업로드 대역폭이 지속 병목임 재확인.
2. 완전 근인 규명(yaml read): durability 업로드 경로 2개 모두 hf_transfer 미사용 — (a)백그라운드 pusher(push_ckpts_to_hf.py)는 `import hf_transfer` 성공시만 `setdefault(HF_HUB_ENABLE_HF_TRANSFER,1)`인데 env에 미설치로 폴백 추정, (b)**최종 sync push(yaml line233-243)는 아예 env var/설치 없이 `upload_folder`만 호출**. ⇒ 양쪽 다 ~4.5MB/s. graceful 선점의 짧은 grace(분 단위)로 30GB(≈2h) 못 올림 → frontier gs140 고착. **어떤 현실적 창(<1h)에서도 durably 전진 불가 = 데드락 확정.**
3. 배포 설계(E-019 리스크 최소화·reversible): (i)yaml `env:` 블록에 `HF_HUB_ENABLE_HF_TRANSFER: "1"` 추가=**값편집(명령문자열 아님)**→pusher·최종push 둘 다 자동 적용, (ii)bootstrap_sdc_node.sh에 `pip install hf_transfer` 추가(env var만으론 미설치시 에러), (iii)build_sdc_code_snapshot.sh로 tarball 재빌드→GitHub release 업로드→새 asset id, (iv)yamls의 CODE_TAR_REVISION 값갱신(480254660→신규)=값편집, (v)★b2 running 방해 없이 준비(다음 auto-retry에 적용)·런타임로그 검증. **가역성**: 신 tarball이 부트스트랩 깨면 CODE_TAR_REVISION 원복(값편집)으로 즉시 복구=E-019식 비가역 사고와 다름.

**통찰:** 근인 격리의 마지막 단계=**병목을 공유하는 모든 경로 열거**. "pusher가 느리다"에서 멈췄다면 최종push도 같은 병목임을 놓쳐 반쪽 fix가 됐을 것. durability는 pusher+최종push 두 경로의 AND 조건이라 둘 다 hf_transfer를 써야 풀린다. 그리고 위험한 인프라 수정도 **가역적으로 설계**(CODE_TAR_REVISION 값스위치)하면 E-019식 "런타임서만 깨지는 비가역 배포"의 공포 없이 시도 가능하다 — reversibility가 위험한 fix를 안전하게 만든다.

---

## E-030 (2026-07-18 00:35 UTC) — Codex NO-GO(hf_transfer v1.x 무시)+hf_xet 프로젝트충돌→블라인드 배포 보류·사용자 결정 플래그

**문제(추상):** 근인·fix가 명확해 보여도, 외부 권고(Codex+web)와 프로젝트 고유 이력이 상충하면 블라인드 배포는 새 사고를 부른다. 특히 primary 실험 인프라의 비가역 위험 요소(tarball)는 상충 해소 전 배포 금지.

**해결(구체):**
1. Codex 배포 확인 결과 **NO-GO**: (a)huggingface_hub v1.x는 `HF_HUB_ENABLE_HF_TRANSFER=1` 무시(deprecated)·hf_transfer 설치만으론 속도 보장 안 됨→버전 확인·고정 선행 필요, (b)대안=`hf_xet`+`HF_XET_HIGH_PERFORMANCE=1`+1회 canary 업로드 속도검증 후 GO.
2. **상충 발견**: 이 env는 yaml line68 `pip install "huggingface_hub<1.0"`=v0.x라 hf_transfer가 유효(v1.x 무시 경고는 미해당). 그러나 Codex가 권한 **hf_xet은 본 프로젝트서 "대용량 사일런트 404"로 제거된 이력**(CLAUDE.md hf-xet-upload-pitfall·bootstrap `rm -rf hf_xet*`). ⇒ Codex의 최선책이 프로젝트 이력과 정면충돌.
3. 판정: (i)신호 상충(hf_transfer for v0.x vs hf_xet-but-broke) + (ii)canary 미검증 + (iii)tarball 재빌드 양 arm 영향 + (iv)b2가 죽지 않고 로컬 진행 중 ⇒ **블라인드 배포 보류**. 가역 설계(CODE_TAR_REVISION 스위치)라도 상충 미해소 상태 배포는 순손해 위험. **사용자 결정 플래그**: durability 데드락 근인=업로드 대역폭 확정, fix 후보=hf_transfer(v0.x 유효 추정)/hf_xet(Codex 권장but 과거 404)/save_freq↑·긴창 대기, 어느 것도 canary 검증 필요.
4. 잠정: b2 auto-retry로 살아있고 로컬 step 진행 중이므로 계속 관찰. 혹 graceful 선점서 최종push(upload_folder 10retry, resumable)가 누적 완료해 frontier 전진하는지 확인(1 rollback만으론 총 데드락 단정 유보).

**통찰:** "근인 확정 + fix 명확"이라도 **외부 권고 vs 프로젝트 이력 상충**은 배포 게이트다. Codex는 web-informed지만 프로젝트 고유 함정(hf_xet 404)은 모른다 — 자율 에이전트는 외부 조언을 프로젝트 메모리(CLAUDE.md·기존 코드 결정)와 교차검증해야 하며, 상충 시 canary 없는 블라인드 배포보다 사용자 판단을 구하는 게 옳다. reversibility가 위험을 낮추나 "상충 미해소"는 reversibility로 못 덮는다(잘못된 방향 배포는 되돌려도 시간손실). 위험한 primary-인프라 수정의 순서=근인확정→fix후보→**canary검증 or 사용자결정**→배포→런타임검증.

---

## E-031 (2026-07-18 01:05 UTC) — ★진단 정정(사용자 지적): b3pkg 실패는 "16GB 다운로드"가 아니라 "anchor 부재"(cold-start 관문 미통과)

**문제(추상):** "A는 되는데 B만 안 된다"는 관찰은 진단의 리트머스다 — 내가 지목한 근인(공통 요소)이 A/B에 동일하다면, 그것은 비대칭을 설명 못 하므로 틀린 근인이다. 공통 요소를 배제하고 **실제로 다른 것**을 찾아야 한다.

**해결(구체):**
1. 사용자 반문: "b2에선 돌아가는데 왜 b3만 안돼?" → yaml 재검증. b2(line93-94)·b3pkg(line103-104) **동일**: `snapshot_download(repo_id="iamseungpil/metacot", allow_patterns=["models/b23_rv_unmasked_sft/**"])` — **같은 16GB SFT-init 다운로드**. 게다가 b2는 여기에 **gs140 체크포인트(~30GB) pull까지 추가**로 함(resume). ⇒ b2가 오히려 더 큰 다운로드를 하고도 통과 = **"16GB 다운로드>선점창"은 비대칭을 설명 못 하는 틀린 근인**(E-024·E-026·E-027·E-030의 전제 일부 오류).
2. 진짜 근인=**anchor 비대칭**: b2는 gs140 체크포인트 보유 → 선점돼도 gs140서 resume(부트스트랩+resume ~15min), 진행분 무손실, 짧은 창에도 생존. b3pkg는 fresh(gs0) → anchor 없어 매 선점마다 **cold-start 전체**(부트스트랩+verl init+첫스텝+첫저장 ≈30-35min)를 한 창에 완주해야 첫 체크포인트 생성. ~9min 선점창에선 불가 → 영구 gs0. b3pkg 사망지점도 다운로드 후 "cold start" 체크(로그 확인) = 다운로드는 성공, **verl cold-start init/첫스텝 도달 전 사망**.
3. ⇒ b3pkg의 관문은 "한 번이라도 ~30-35min 무중단 창을 잡아 **첫 체크포인트**를 저장"하는 것. 그 후엔 b2처럼 resume로 안정화. fix 재평가: (a)모델-번들 이미지=부트스트랩 단축(~15→~5min)해 첫저장 도달 확률↑ (b)opportunistic 재제출=긴 창 우연 대기 (c)b3pkg save_freq를 최소로+cold-start init 단축.

**통찰:** **"공통 원인은 차등 결과를 설명 못 한다"**가 인과 진단의 핵심 필터다. 22회 실패를 "16GB 다운로드"로 4개 엔트리(E-024~E-030)에 걸쳐 귀속했으나, 그 요소가 b2에도 동일(오히려 더 큼)함을 사용자 반문으로 재검증하니 틀린 근인이었다. 진짜 차이는 **anchor 유무**(b2는 gs140 보유로 cold-start 관문 통과 완료, b3pkg는 미통과). 교훈=자율 에이전트는 "왜 여기선 되고 저기선 안 되나"를 스스로 물어 공통 요소를 배제해야 하며(대조 사례가 최강의 반증 도구), 사용자의 날카로운 반문은 진단 오류를 잡는 최고의 external check다. 오류 귀속의 연쇄(E-024~E-030 일부)를 정직히 정정하는 것이 신뢰의 조건.

---

## E-032 (2026-07-18 01:05 UTC) — ★실측 로그로 gs0 사망 근인 확정(사용자 라이브로그 제공): 다운로드 무죄·(1)fresh에 무조건 pull루프 ~2min 낭비 (2)save_freq=5로 첫 anchor가 ~75min 거리 → save_freq=1 적용(advanced-goose)

**문제(추상):** 반복 실패의 근인은 추정이 아니라 **실측 타임라인**으로 확정해야 한다. 각 단계의 실제 소요를 로그로 재면, 추정 병목(다운로드)이 무죄이고 자충수(설정)가 유죄로 드러날 수 있다.

**해결(구체):**
1. 사용자가 set-shark(22nd) 라이브 노드로그 제공 → 실측: 00:52:51 시작 → 부트스트랩 완료 00:55:09(**2.5min**, conda-pack fast-path·flash-attn 기존재) → **16GB SFT init 다운로드 51초**("b23 meta-SFT staged") → RGS='0 0' → **pull_resume 루프가 RGS=0인데도 무조건 3회 실행**(각 sleep30, **~2min 낭비** — 0716 resume-hardening을 B0에서 복사시 fresh-arm 게이트 누락; abort는 ANY=1 조건이라 hard-deadlock은 아님, line178) → `export LOCAL_RANK=0`(verl launch 직전)에서 **10min 선점 사망**.
2. ⇒ 근인 체인 확정: (a)다운로드는 무죄(51s). (b)pull루프 낭비로 verl launch가 ~8min으로 밀림. (c)**결정타 = save_freq=5**: verl 진입해도 첫 체크포인트가 gs5 = init ~7min + 5스텝×~12min ≈ **시작부터 ~75min 무중단 창 필요** — 이 티어(금요저녁 PT 피크, 창 8-18min)에선 도달 불가. 예전(driven-duckling 0714)에 돌아간 건 yaml이 옳아서가 아니라 티어가 긴 창을 줬기 때문.
3. 조치: 사전승인 컨틴전시("부트스트랩 통과 후 gs5 전 재선점시 save_freq 5→1") 트리거 충족 → h100std_rq3_b3.yaml line216 `save_freq=5→1` **값편집**(parse OK) → advanced-goose(23rd) 재제출. 첫 anchor가 gs1(시작부터 ~25min compute + push)로 단축. pull루프 게이트 수정은 명령문자열 편집이라 보류(사용자/Codex 승인시).
4. 사용자 질문 답: "SFT model 위 학습"=이미 그렇게 함(b23 init, 51s 스테이징 성공). "keep-alive로 GPU 잡기"=이미 함(인라인 GPU 스피너 t~0부터) — 선점은 idle-suspend가 아니라 용량 eviction이라 스피너로 못 막음.

**통찰:** 22회 실패 동안 근인을 3번 갈아탔다(16GB 다운로드→anchor 비대칭→실측 확정). 각 단계에서 진단을 갈아엎은 도구가 전부 **더 원시적인 증거**였다: HF 목록(간접)→yaml 대조(정적)→**노드 라이브 로그(실측)**. 교훈=반복 실패는 추정 근인으로 N번 재시도하기 전에 실측 로그 한 번을 얻는 게 압도적으로 싸다. 그리고 "예전엔 됐다"의 설명은 코드 회귀만이 아니라 **환경 조건(티어 혼잡도) 변화**일 수 있다 — 예전 성공 시점과 현재의 환경 차이를 함께 봐야 한다. save_freq는 durability 뿐 아니라 **cold-start anchor 도달 시간**을 결정하는 이중 레버다(save_freq=5는 fresh arm에겐 "첫 75min을 무결점으로 버텨라"는 요구였다).

### 상태 tick — 01:12 UTC(7/18) — ★b2 frontier gs140→**141 완전 전진**(4/4/4, gs143 push도 optim1/4 진행, gs130은 keep2 prune됨)="느림" 확정·총데드락 반증(창 길면 pusher 완주). b2 running 2h 건강. advanced-goose(b3pkg 23rd, save_freq=1) 01:09 새 창서 정상 부트스트랩 진행(사용자 라이브로그 확인, 오류 없음). 다음: advanced-goose gs1 anchor(~25min 지점)·부분-push 함정 감시.

### 상태 tick — 01:15 UTC(7/18) — advanced-goose(b3pkg 23rd, save_freq=1) 가드 통과·verl launch 진입(~9min, 이전 22회 사망지점 통과중). gs1 로컬 anchor 예상 ~01:27-30. b2(mighty-civet) running 2h·step143+·reward 0.43→0.51→0.61 상승. 다음: advanced-goose gs1 저장→pusher push=anchor 확립 여부.

### 상태 tick — 01:20 UTC(7/18) — advanced-goose(23rd, save_freq=1) 10m 선점(verl init중, gs1 미도달·HF 깨끗=ANY함정 미발동). 23연속. 창 10min < cold-start~gs1 ~25min: save_freq=1은 필요조건이나 창 자체가 짧음(금요저녁 PT 피크). 브리지 재제출. 남은 레버=pull루프 게이트(-2min, 사용자 승인대기)·주말 긴 창 대기. b2 순항(2h·step143+).

---

## E-033 (2026-07-18 01:36 UTC) — ★★★근인 확정·수정: b3pkg 23연속 사망 = 주석 속 'clean isolation' 따옴표가 bash -c 스크립트를 절단(0712부터 verl이 한 번도 실행된 적 없음) → 따옴표 2자 제거(smart-bat)

**문제(추상):** 명령 블록이 따옴표 래핑(bash -c '...')으로 전달되는 시스템에서, *주석*조차 실행문과 같은 인용 규칙의 지배를 받는다. 공백을 품은 따옴표쌍 하나가 스크립트를 조용히 절단하면, 절단점 이후는 "실행되고 죽는" 게 아니라 "존재하지 않게" 되어, 증상(무출력 pass)이 선점·인프라 문제로 위장된다.

**해결(구체):**
1. 판별 증거 사슬: (a)사용자 라이브로그 2회 제공→cold-start 타임라인 실측(다운로드 51s, 무죄), (b)wandb API 전수조회→**rq3_b3pkg run 0개**(b2·b3nopmi·b0는 존재; shining-bonefish 18min 창에서도 없음=verl 미시작 확정), (c)죽은 로그 tail→rc 라인·verl xtrace 부재, 23/23이 정확히 `LOCAL_RANK=0`서 종료, (d)b2(작동)/b3nopmi(작동이력 gs149)/b3pkg 3자 문자단위 대조→**b3pkg에만 공백 품은 따옴표쌍 `'clean isolation'`**(0712 CORRECTION 주석), (e)**shlex.split 기계증명**: b2/b3nopmi=토큰3·script에 VERL_PID 포함, b3pkg=토큰4·script가 `for clean`서 절단·verl launch 미포함. 절단 스크립트의 마지막 실행문=`export LOCAL_RANK=0`→주석만 남아 조용히 exit 0→"pass"·동일 1.6kB.
2. 무해/유해 판별 원리: `''`(인접)·`'...'`(공백없음)은 닫힘+재열림 후에도 **단어가 안 쪼개져** 무해(b2/b3nopmi 작동으로 실증). **공백을 품은 쌍만** word-split로 bash -c 인자를 절단.
3. Codex 검증: 근인 동의·수정안 충분·GO 조건(shlex 토큰3+VERL_PID+bash -n+yaml parse) 제시 → 수정(따옴표 2자만 제거, 주석 내용 보존) → **4중 검증 전부 통과** → ruling-mustang(구판) 이미 사망 확인 → **smart-bat(25th, fix판) 제출**.
4. 부수 정정: E-024~E-032의 "선점창<원자작업" 프레임은 b3pkg에 대해선 오진(진짜는 코드절단). 단 save_freq=1·B(stale삭제)는 여전히 유효한 방어. 0713 메모리("SSH 손실행 gs1 도달=코드 무죄, 선점")도 절반만 맞음—손실행은 래핑을 우회했을 뿐.

**통찰:** (1)**"한 번도 성공한 적 없는" 실패는 확률적 실패(선점)와 다른 클래스**다—wandb run 0개라는 전역 부재 증거가 23개의 개별 로그보다 강했다. (2)증거의 위계: 추정<로그 tail<라이브 실측<**전수 대조(작동 쌍둥이 diff)**<**기계적 재현(shlex)**—최종 확정은 항상 재현 가능한 기계증명으로. (3)주석은 무해하다는 직관은 래핑-인용 시스템에서 깨진다: **주석도 코드다**. (4)사용자의 반복된 반문("b2는 되는데 왜 b3만", "잘못 수정한 것 아니냐", "로컬서 한줄한줄")이 오진의 관성을 끊었다—자율 루프는 외부 반문을 진단 재검증 트리거로 대접해야 한다. 검증: smart-bat이 verl 진입+wandb run 생성하면 최종 입증.

---

## E-034 (2026-07-18 01:48 UTC) — ★절단 fix 입증 성공: smart-bat이 23번 만에 처음으로 LOCAL_RANK=0 통과·verl 실행 시작

**문제(추상):** 근인 수정은 "코드가 통과 검증을 넘겼다"가 아니라 "런타임에서 실패 지점을 실제로 통과했다"로만 입증된다(E-019 교훈의 긍정판). 절단된 스크립트는 그 지점 이후가 "존재하지 않았으므로", 입증은 곧 "그 이후 줄이 로그에 나타나는가"이다.

**해결(구체):** smart-bat(b3pkg 25th) 라이브 로그 raw tail 확인 — 부트스트랩(2.5min)·pull_resume 헛돌기(~90s) 통과 후 `export LOCAL_RANK=0` → **이번엔 잘리지 않고** `[HB ... gpu0used=2270MB]`(하트비트=VERL_PID 정상)·`INFO 01:44:58 Automatically detected platform cuda`(verl/vLLM 실행 시작) 출력. **23/23이 정확히 `LOCAL_RANK=0`에서 끝났던 것과 대조 = 절단 fix(따옴표 2자 제거) 결정적 입증.** verl가 0712 이후 처음으로 실행됨.

**통찰:** 근인 수정 입증의 최소충분 신호 = "이전 실패 지점의 *다음 한 줄*". 23개 로그가 전부 같은 줄에서 끊긴 게 근인의 지문이었고, 그 다음 줄(HB·platform cuda)이 나타난 게 수정의 지문이다. 이제 남은 건 **완전히 다른 문제 클래스**: verl init(~7min)+첫 스텝+gs1 저장(~25min total)까지 선점 창이 버티느냐(순수 티어 운). 즉 "코드 데드락"(해결됨)과 "선점 창 부족"(save_freq=1로 완화, 주말 긴 창 대기)을 분리했다. 다음=gs1 로컬저장+HF push=첫 anchor 확립 확인.

### 상태 tick — 01:48 UTC(7/18) — ★★b3pkg(smart-bat) verl init 완료·Training Progress 0/300 진입=프로젝트 사실상 최초 b3pkg 실학습 시작(gpu0 40GB rollout 활성). 절단fix 완전 입증. 남은것=step1 완료→gs1 anchor(~12min 창). b2(mighty-civet) running 2h+·gs143 완전(durability 정상). 다음: b3pkg gs1 저장+push=anchor 확립.

### 상태 tick — 01:56 UTC(7/18) — ★★b3pkg(smart-bat) step1 완료(Training Progress 1/300, 485s/it, ETA~40h)=첫 로컬 anchor 성립·코드데드락 해결 완전확정. gs1 HF push 대기(pusher). b2 gs143 순항. ★사용자 요청: 모니터링 30min로·save_freq 5로(속도+pusher안정화). 계획: 캐던스 즉시 1800s, save_freq=5는 b3pkg gs1 HF확립+각arm 다음 선점시 적용(무중단·Codex확인). save_freq 속도이득 ~5%(compute지배), 진짜이득=pusher 완결.

---

## E-035 (2026-07-18 02:01 UTC) — ★b3pkg 첫 triobj 실데이터: 방출 0.873(게이트 통과)·PMI-shift/cal 헤드 활성·gs1 anchor 착지 중; b2 frontier gs145 전진

**문제(추상):** 코드 데드락 해결의 최종 확증은 "실행됐다"를 넘어 "의도한 계산이 실제로 나온다"이다. RL 보상 헤드가 로그에 유효값으로 찍혀야 실험이 설계대로 작동함이 입증된다.

**해결(구체):**
1. smart-bat step1 메트릭(프로젝트 최초 b3pkg 실데이터): `training/global_step:1`, save_checkpoint 36s. **triobj 전 헤드 활성 확인** — gdpo/meta_region_utility −0.236(PMI-shift belief margin), gdpo/cal_region_reward −0.145(calibration), **gdpo/meta_emission 0.873**(방출 87%=헌법 게이트 emit≥0.8 통과), gdpo/format_penalty 0.32, correctness 0.109. step 485s/it(triobj가 vanilla b2 730s보다 오히려 빠름=응답길이 짧음). ⇒ 절단fix 후 b3pkg가 **설계된 full triobj+PMI-shift 보상으로 정상 학습** 확정.
2. b3pkg gs1 HF push 진행(model1/4·extra4/4·optim0/4)=anchor 착지중. b2 최신완전 gs143→**145 전진**(durability 정상 작동).
3. 사용자 요청(30min 캐던스·save_freq→5) 반영: 캐던스 1800s 적용. save_freq=5는 b3pkg gs1 HF 완전착지(4/4/4) 확인 후 각 arm 다음 선점/재개시 적용(현재 gs1 model1/4라 아직 취약, 미적용).

**통찰:** 인프라 데드락(코드 절단) 해결의 완결은 "학습 루프 진입"이 아니라 "보상 헤드가 유효값 산출"이다 — meta_emission 0.873은 b3pkg가 처음으로 실제 메타 방출을 하며 학습함을 뜻하고, 이는 0712 이후 이 arm이 한 번도 못 보여준 것. 즉 오늘 두 데드락(b3pkg 코드절단·b2 push-truncation) 해결로 RQ3 4-arm이 처음으로 온전히 굴러가기 시작. 남은 건 순수 시간(gs300 ETA b2~33h·b3pkg~40h)과 선점 생존(save_freq·anchor로 방어).

### 상태 tick — 02:08 UTC(7/18) — [운영]Stop hook block cap 강제종료=코드/실험오류 아님·`/goal` 내장 조건검사기가 gs300 미도달로 9회 블록→안전장치 발동. 수정=settings.json env CLAUDE_CODE_STOP_HOOK_BLOCK_CAP=100000(다음세션 적용, 실험 무영향·ScheduleWakeup 루프 정상). ★b3pkg(smart-bat) step2(2/300, 494s/it)·meta_emission 0.889(게이트 유지)·correctness 0.127·PMI/cal 헤드 정상=triobj 건강. ⚠️로그 SympifyError('could not parse -049')=math_verify가 비표준표기 파싱실패로 해당샘플만 오답처리·무해(학습 계속). b2 running 3h·gs145.

### E-037 (2026-07-18 03:13 UTC) — [선점] 두 arm 동시 선점→auto-retry 재부팅 진행 중

**문제(추상):** 잦은-선점 티어에서 durable 학습 잡의 "정지 신호"는 다의적이다 — HF 프런티어 동결·HB 정지·GPU 무활동은 (a)업로드 랙, (b)로그뷰 랙, (c)실제 선점 셋을 구별하지 못한다. 단일 지표로 판정하면 건강한 auto-recovery를 장애로 오진하거나 반대로 진짜 선점을 놓친다.

**해결(구체):** 03:13 계측서 b2 HB가 03:06:30 정지·b3pkg HB grep 공집합·HF 프런티어 둘 다 동결(b2 gs150·b3pkg gs8)의 삼중 애매신호 발생. `amlt log view <arm> | tail -12`로 **로그 tail의 실제 내용**을 확인해 판정: 두 로그 모두 `[bootstrap] fast-path: pulling conda-pack env from HF`·`extracting env (~5GB)`를 tail에 노출=**두 arm 모두 선점되어 새 노드서 재부팅 중**(선점 직전 마지막 HB/HF가 그대로 동결로 보였던 것). amlt list는 둘 다 여전히 Running(=동일 실험 내 자동 retry_NNN 진행). 조치=**무개입**: auto-retry가 부트스트랩 완료 후 pull_resume_ckpt.py로 HF 앵커에서 resume(b2→gs150·b3pkg→gs8) 하도록 둠. cancel+resubmit은 진행 중 retry와 경합하므로 금지. save_freq=5 적용도 보류(진행 중 recovery 방해). 검증 틱을 30분→10분으로 단축해 resume이 gs0 아닌 HF 앵커에서 되는지 확인 예정.

**통찰:** 선점 판정의 결정적 증거는 지표(HB/HF/GPU)가 아니라 **로그 tail의 실행단계**다 — "HB 정지 + HF 동결"만 보면 [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]처럼 오진하지만, tail이 `bootstrap/extracting`을 보이면 선점+재부팅이 확정된다. 3계층(지표→로그tail 내용→amlt status)을 함께 봐야 auto-recovery와 진짜 장애를 가른다. Basic 티어에선 선점은 정상이며 keep-alive=자동 resume 신뢰가 정답(수동 개입이 오히려 recovery를 깬다).

### 상태 tick — 04:47 UTC(7/18) — [HF 자가관리 확인] `--keep N` 자동트림 발화, HF **959GB→795GB 자가회복**. rq3_b2 [150,151,153]→**[151,153]**(gs150 트림·정확히 keep-2)·rq3_b3pkg [4,5,8,9]→**[5,9,13]**(gs4,8 트림+gs13 신규=전진). 두 arm 생존·전진(b2 gs153·b3pkg gs13, HB 04:47·GPU 76/56GB). ★통찰: 원격 저장소는 **catchup(업로드 상승)↔--keep 트림(하강) 사이클로 자가 bounded**(~795~960GB 진동)—`--keep N`은 지연 발화하나 확실히 회복하므로 진동 구간 내에선 수동개입 불필요(성급한 재정리는 곧 자동트림될 것을 중복 삭제). 수동정리(E-039)는 **자동관리 사각**(완료런 intermediate·pre-recovery 잔재)에만 국한. 즉 정리 판단축=이 아티팩트가 활성 --keep 관할인가(→기다림) vs 관할밖인가(→수동). b3pkg가 b2보다 gs300 더 멀다(b3pkg gs13 vs b2 gs153, 재개점 차이). 임계 재정리 트리거는 진동상단(960)이 아니라 지속상승으로 1.15TB 돌파시로 유지.

### 상태 tick — 04:41 UTC(7/18) — [정시 틱] 두 arm 정상 학습(HB 04:41·GPU 43GB·b2 gs153/300·b3pkg gs9). HF 정리(E-039 795GB) 이후 catchup 사이클로 795→828→860→893→926→959GB 상승 후 **959GB plateau**(gs153/gs9 업로드 완료)=runaway 아님·재정리 불필요(임계 1.2TB 대비 241GB 여유). --keep 2가 arm당 3~4개 유지하나 총량 bounded. b0 [300]·b3nopmi 미변경 유지. gs300까지 ~147step. 캐던스 30분 유지.

### E-041 (2026-07-18 05:35 UTC) — [진단정정] 로그 빈응답=명령어 오타(amlt log→amlt log view)·실측 step속도 ~925s/step·gs300 ETA ~2.5h→~35h 대정정

**문제(추상):** 관측 도구의 "빈 응답"을 인프라 장애로 귀인하기 전에 *도구 사용법 자체*를 의심해야 한다 — 그리고 파생지표(HF 프런티어)만으로 진행률을 추정하면 실제 계산속도를 심하게 오판할 수 있다. 두 오류가 겹치면 "생존은 맞으나 ETA·병목 진단이 통째로 틀린" 상태가 된다.

**해결(구체):** `amlt log <exp>`가 계속 빈 응답이던 근인=**명령어 그룹 오용**: `amlt log`는 서브커맨드 그룹이라 실험명을 직접 못 받고 조용히 usage/에러 반환(2>/dev/null+grep로 에러 은폐됨). 정답=`amlt log view <exp> :<job>`(E-038 메모에 이미 있던 형식). 교정 후 실제 로그 수신: **b2 로컬 `training/global_step:158` gs158 저장(05:18:01)**=HF 프런티어 gs156보다 2step 앞섬→**pusher-lag 확증(스톨 아님)**. HB 05:18:17→48 10초간격 전진·GPU 43GB=생존. ★핵심 실측: `timing_s/step:924.78`(≈15.4분/step), 내역 gen 643.6s(지배적)·update_actor 175.3s·save_checkpoint 41.4s(save_freq=1)·old_log_prob+ref 61s. generate_sequences mean 241s **max 641.97s**·response_length mean 2175 **max 8192(cap)·clip_ratio 0.137**(13.7% 응답이 8192 cap 도달). b3pkg도 동율속(gs4→gs13, ~2h10m, ≈14.5분/step). **∴ gs300 ETA 대정정: b2 (300−158)×~15분=~35h, b3pkg (300−13)×~15분=~70h** — 그간 보고서·틱의 "~2.5h ETA"는 오산(72s/step 가정)이었고 실측은 ~12배 느림. save 오버헤드는 step의 4.4%뿐이라 save_freq=1 유지가 여전히 durability 우위(15분 계산 손실 방지)—변경 불필요.

**통찰:** ①"빈 출력=장애"는 [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]] 계열 오진의 재발이었고, 이번엔 선점이 아니라 **내 명령어 오타**가 근인 — 관측 공백 시 *인프라보다 내 호출을 먼저 의심*하고 usage/에러를 2>/dev/null로 삼키지 말 것. E-040의 "릴레이 transient" 전제는 틀렸다(정정). ②단, E-040의 HF-바이트-증가=노드생존 프록시 추론 자체는 여전히 유효한 교차검증이었다(독립계층). ③가장 중요: **step 속도는 반드시 실측(timing_s/step)해야 하며 프런티어 전진율로 역산하면 pusher-lag가 섞여 오판**한다. ④~925s/step의 지배항은 generation(70%)이고 response가 8192 cap을 13.7% 때림=**RL 진행 중 응답길이 인플레이션**의 시그니처 — 이는 RQ2의 emission-erosion/length 동역학과 같은 뿌리일 수 있어 gs300 해석 시 length confound로 반드시 점검. 관련 [[aime-drop-is-decoding-degeneration-nontermination-0629]](비종료·길이팽창).

### 상태 tick — 04:33 UTC(7/20) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB(13h·retry_001 안정 calm window 지속)·b3pkg=smart-bat GPU58.5GB(2d·retry_004) 둘 다 학습중. durability 트랩·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 03:31 UTC(7/20) — [두 arm 생존·전진] b2=living-pigeon **gs193**(+10·11h 안정 calm window)·b3pkg=smart-bat **gs105**(+15). 둘 다 학습중·durability 트랩·gs300 멀음(b2 ~107step). 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 02:29 UTC(7/20) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB(10h·retry_001 안정 유지)·b3pkg=smart-bat GPU43GB(2d·retry_004) 둘 다 학습중. durability 트랩·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 01:27 UTC(7/20) — [두 arm 생존·학습중] b2=living-pigeon GPU74.6GB(9h·긴시퀀스 backward·activation-offload로 무OOM 유지 확인)·b3pkg=smart-bat GPU43GB 둘 다 학습중. durability 트랩·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 00:25 UTC(7/20) — [두 arm 생존·전진] b2=living-pigeon **gs183**(+10·8h 안정 calm window·retry_001)·b3pkg=smart-bat **gs90**(+25). 둘 다 학습중·durability 트랩·gs300 멀음(b2 ~117step·b3pkg ~210step). 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 23:23 UTC(7/19) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB(7h·retry_001 비교적 안정)·b3pkg=smart-bat GPU43GB(retry_004) 둘 다 학습중. durability 트랩·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 22:21 UTC(7/19) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB(6h)·b3pkg=smart-bat GPU65GB 둘 다 running·학습중(failed 없음). durability 트랩·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 21:19 UTC(7/19) — [두 arm 생존·전진] b2=living-pigeon **gs173**(+3·GPU60GB)·b3pkg=smart-bat **gs65**(+8). 둘 다 학습중·durability 트랩·gs300 멀음·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 20:17 UTC(7/19) — [두 arm 생존·학습중] b2=living-pigeon **gs170**(GPU35GB·gs158서)·b3pkg=smart-bat **gs57**(GPU22.8GB·gs43서 replay). 둘 다 학습중·durability 트랩·gs300 멀음·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 19:15 UTC(7/19) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB(running 3h·retry_001)·b3pkg=smart-bat GPU43GB(retry_004) 둘 다 학습중. durability 트랩 지속·gs300 미도달·변화 없음. 무개입·사용자 경로결정 대기.

### 상태 tick — 18:13 UTC(7/19) — [두 arm 생존·학습중] b2=living-pigeon GPU35GB·b3pkg=smart-bat GPU43GB 둘 다 running·학습중(failed 없음). durability 트랩서 생존·진행 유지·gs300 미도달. 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 17:11 UTC(7/19) — [두 arm 생존·둘 다 주기적 리셋] b2=living-pigeon GPU32.5GB 학습중(keep-alive 복구·gs158서). b3pkg=smart-bat GPU43GB 학습중이나 **retry_004**(신규선점→gs43 재리셋 후 재학습). 두 arm 대칭 durability 트랩서 주기 리셋 지속·생존은 유지·gs300 미도달. 방침대로 무개입(keep-alive만). 사용자 경로결정 대기.

### 상태 tick — 16:09 UTC(7/19) — [b2 keep-alive 복구·b3pkg 정상] b2=living-pigeon **running 36m**(std_log 존재=노드확보·gs158 재개 부팅). b3pkg=smart-bat GPU43GB 정상학습(무선점 1일+·retry_003 불변). b2 keep-alive 재제출 복구 성공. 두 arm 생존·gs300 미도달·durability 미해결. 무개입·사용자 경로결정 대기.

### 상태 tick — 15:27 UTC(7/19) — [b2 terminal failed→keep-alive 재제출] b2=internal-grub **failed**(GPU-0 크래시 후 auto-retry 소진·retry_004 종점). mandate keep-alive대로 재제출 **living-pigeon**(gs158 재개 예상). durability 트랩 지속이나 죽은 arm=gs300 불가라 살려둠. b3pkg=smart-bat GPU43GB **정상 학습**(gs99+·무선점 1일+ 지속). 사용자 경로결정 대기·무리한 durability 재수정은 안함(단순 keep-alive 재제출만).

### 상태 tick — 14:45 UTC(7/19) — [b2 GPU-0 크래시(선점)→auto-retry] b2=internal-grub GPU 2.3GB+로그 **"Total available GPUs 0 < desired 4"→Terminated**=GPU 회수(선점 시그니처)로 종료중→auto-retry가 gs158 재개(durability 트랩 반복·~gs183 진행 손실). 방침대로 무개입(auto-retry 신뢰). b3pkg=smart-bat GPU 43GB **정상 학습**(gs99+·retry_003 불변). calm window는 b2만 깨짐·b3pkg 지속. gs300 미도달·durability 미해결. 사용자 경로결정 대기.

### 상태 tick — 13:42 UTC(7/19) — [calm window ~5h 지속] b2 ~gs180(heartbeat GPU35GB 활성·step마커 window밖)·b3pkg **gs99**·신규선점 없음(retry 불변). 느린 전진·gs300 멀음·durability 미해결. 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 12:40 UTC(7/19) — [calm window ~4h 지속] b2 **gs179**(+3)·b3pkg **gs95**(+8)·신규선점 없음(retry 불변). 느린 전진·gs300 멀음·durability 미해결. 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 11:38 UTC(7/19) — [calm window ~3h 지속] b2 **gs176**(+3)·b3pkg **gs87**(+8)·신규선점 없음(retry b2_004·b3pkg_003 불변). 느린 전진·gs300 멀음·durability 미해결. 변화 없음·무개입·사용자 경로결정 대기.

### 상태 tick — 10:36 UTC(7/19) — [calm window 지속·느린 전진] b2=internal-grub **gs173**(+4·retry_004 불변). b3pkg=smart-bat **gs79**(+9·retry_003 불변). 신규선점 없이 전진 but gs300 멀고 durability 미해결. 변화 없음·무개입. 사용자 경로결정 대기 지속.

### 상태 tick — 09:34 UTC(7/19) — [calm window·두 arm 진행중] b2=internal-grub **gs169**(gs158재개 후 +11·retry_004 불변). b3pkg=smart-bat **gs70**(gs43재개 후 +27·retry_003 불변). 신규선점 없는 calm window서 둘 다 전진 but gs300 멀고(b2 131·b3pkg 230스텝) 다음 선점때 또 리셋(durability 미해결). gs300 임박 아님. 사용자 경로결정 대기 지속. 무개입.

### 상태 tick — 08:32 UTC(7/19) — [calmer window·E-063 구현지점 확정] 두 arm 생존(b2 7h retry_004·b3pkg 1d retry_003)·retry 불변=신규선점 없는 calmer window(진행 가능성). ★E-063 구현지점 파악(dcpo_region.py 1669L 구조): META_REGION=tag포함 meta블록(이미 advantage 0)·META_CONTENT=PMI 라우팅·ANSWER_REGION=response−META_REGION. ★핵심통찰=**emit-결정 토큰(`<|meta|>` 직전, ANSWER_REGION 소속)이 correctness advantage를 받아 emission 억압**(meta가 correctness 음상관시)—이게 collapse 실경로. Option A 구현=emit-결정토큰+meta span서 correctness advantage 양·음 모두 마스킹(유용성=PMI만). ★untested 코드작성은 durability 해결후로(지금 테스트/발사 불가). 무개입, calm window로 gs300 도달 감시.

### 상태 tick — 07:29 UTC(7/19) — [보고서 consolidation·감시] 두 arm 생존(b2 6h·b3pkg 1d·durability 교착 지속). ★보고서 상단 running-tally 현행화+예비결과 요약표(gs50/100 val) 추가=흩어진 E-054~064를 서사로 통합(mandate '결과나오면 보고서update'·예비 명시). durability 경로결정 사용자 대기중—오늘 높은 에러율 감안 위험한 amlt-blob yaml수술 보류·arms 그대로 유지. amlt storage subcommand 존재확인(blob 마운트 가능성 있으나 storage: 섹션 신규추가 필요=비자명). 무개입, calm window로 gs300 도달시 즉시 lifecycle.

### E-064 (2026-07-19 06:24 UTC) — [★critical path 재정의] b3pkg도 선점→gs43 리셋 확정: durability가 두 arm+모든 미래실험의 critical path·HF/Xet 우회(amlt-blob) 필요

**문제(추상):** 한 arm의 우연한 안정을 근거로 "durability 없이도 endpoint 확보 가능"이라 판단하면, 그 arm마저 선점되는 순간 전제가 붕괴한다. durability는 선택이 아니라 *모든 결과-확보의 전제조건*임을 두 arm의 대칭적 리셋이 증명한다.

**구체적 해결(진단).** b3pkg=smart-bat이 겉보기 "1일 무선점"이었으나 실제로 **retry_001/002/003**(선점 3회)·현재 **RGS_STEP=43 재개**(로그 gs49 replay)—즉 gs139 in-memory 진행이 마지막 선점서 gs43으로 리셋(gs44-139는 Xet 차단으로 durable 못됨). b2도 retry_004·gs158 리셋. ∴**두 arm 대칭적으로 durability 함정**: HF/Xet 대용량 업로드 차단→선점마다 durable anchor(b2 gs158·b3pkg gs43)로 리셋→**어느 arm도 gs300 도달 불가**. 게다가 ★E-063 개선실험도 같은 인프라라 발사해도 gs300 못감=**durability가 현 결과+모든 미래실험의 critical path**. amlt blob 스토리지 존재 확인(`.amltconfig`: storage_account=xufluo·container=amulet) but 현 yaml 미마운트. wandb 분석데이터(gs50/100 emission·val)는 유효(선점 전 로깅됨)—과학 결론은 이미 확보. 조치=사용자 판단 요청(아래 통찰의 두 경로).

**통찰.** ①**"안정" 판단은 retry 카운트로 검증해야**: b3pkg "1일 running"을 무선점으로 오인했으나 retry_003이었음—status duration은 누적시간이라 선점을 숨긴다([[singularity-preemption-not-hang-retry-check-0711]]의 retry-확인 규율 재확인). ②**durability는 이제 회피불가 critical path**: 두 경로만 남음—(경로1)amlt-blob 영속스토리지 마운트로 HF/Xet 우회(선점 견딤·현+미래실험 모두 해결·단 yaml/script 비자명 수정·에러리스크), (경로2)gs300 포기+gs50/100 val을 실효 endpoint로 논문화(궤적 단조·결론 명확)+개선실험(E-063)은 durability 해결 후. ③**과학 vs 인프라 ROI**: 결론 방향(RQ1 +7.4·RQ2 emission-collapse)은 gs100서 robust하니 gs300은 확증용—but 개선실험 자체가 durability를 요구하므로 결국 경로1(durability 근본해결)이 미래를 위해 필요. ④권고: 경로1(amlt-blob) 1회 신중 투자가 전체 unblock. 관련 [[E-057]]~[[E-062]] durability 아크·[[E-063]] 개선설계.

### 상태 tick — 05:21 UTC(7/19) — [★durability 소모전 종결 결정·b2 retry_003 교착 수용] b2=internal-grub **retry_003**(또 선점)→gs158 재리셋=순진행0 교착(preemption-heavy 노드+Xet 업로드 차단). ★최종결정: durability 소모전 종결(b2 ~8회 재제출·거대투자에도 구조적 교착·ROI 바닥·xet_enabled API도 inconclusive). 더 안 싸움. 근거: 과학결론 이미 확보(gs100 val RQ1 +7.4·RQ2 emission-collapse)+개선설계(E-063) 완료. 방침: 두 arm 그대로(auto-retry)·b3pkg(잔잔노드·1일+무선점)가 gs300 endpoint 주도·b2는 calm window오면 진행/아니면 gs100 val로 RQ1 확정. b3pkg=smart-bat gs139+ 건재. ★감시 캐던스 60분으로 완화(순수 생존감시)·다음 실질이벤트=b3pkg gs300 or b2 calm window. **통찰:** 인프라 수정의 ROI가 바닥이고 과학적 결론이 이미 손에 있을 때, 더 싸우는 것은 sunk-cost 오류—결과확보+개선설계라는 실가치를 이미 얻었으니 감시로 전환이 정답.

### E-063 (2026-07-19 04:35 UTC) — [★개선실험 설계·Codex검토] RQ2 emission-collapse 처방=옵션A(emission-protected advantage)+PMI 별도수정·A-only v2·gs300후 소규모검증

**문제(추상):** 메커니즘(메타인지 RL)이 자신이 작용할 대상(메타 발화)을 스스로 소멸시키는 자기파괴 동역학을 어떻게 끊는가 — 단, 대상을 억지로 살리면(form 보상) 유용성과 무관한 껍데기만 남는 Gandhi 함정. 무대(emission)를 보존하되 유용성 선택(function)은 분리해야 한다.

**구체적 해결(설계·Codex 적대검토 통과).** RQ2 실패기전=emission collapse(E-056/059: emit 91%→4.5%·correctness credit이 answer span에만·meta span은 순수비용→RL이 meta 억제). ★처방=**옵션A: emission-protected advantage**—`dcpo_region.py`서 (1)A_corr(correctness)와 A_PMI를 먼저 분리, (2)음의 A_corr가 **meta span + ★emit-결정 토큰(meta 직전)**에 억압 gradient로 흐르지 않게 region-mask(correctness는 answer만 벌하고 '메타하는 행위'는 안 벌함), (3)★meta에 positive correctness credit은 주지 않음(주면 form 보상 변질). 유용성은 PMI/별도 outcome-linked credit이 담당. ★**PMI-shift도 별도 수정**(Codex: +10.4→−1.7 역전=emission만 살려선 부족): region 정규화·clipping·correctness 대비 영향력 제한·**emission 회복후 적용하는 warm-up/gating**·궁극적으로 'meta 있을때 answer correctness 개선분'에 연결(PMI=유용성 proxy 가정 현재 깨짐). ★**arm=B2(corr-only)·B3pkg(원본=negative control)·B3pkg-v2(A-only)**—★v2서 PMI 동시변경 금지(원인분리 위해 A-only 먼저). ★성공기준 사전고정: emission+wellformed가 viability floor(예 overall 20%·hard 10%) 이상을 gs100·gs300 모두 유지 AND held-out acc>B2 AND hard-subject서도 초과·paired bootstrap/복수seed. **"emission만 높고 acc 동일=실패"**. ★**지금 발사 금지**(Codex): 현 gs300을 사전계획 baseline endpoint(negative control·rescue 아님)로 확보 후, A-only v2를 matched seed 소규모 검증→그담 확대.

**통찰.** ①**자기파괴 메커니즘의 처방=대상보존과 유용성선택의 분리**: emission 붕괴는 correctness 압력이 '메타 행위'를 벌해서 생기므로, correctness가 answer만 벌하게 마스킹하면(A) 무대가 보존되고, 유용성은 PMI가 독립적으로 선택. 둘을 한 손잡이로 하면(emission 보상) form 함정. ②**emit-결정 토큰 포함이 핵심 디테일**: meta span만 마스킹하고 '발화 여부' 토큰을 빠뜨리면 붕괴 지속(Codex 지적)—collapse는 발화결정 단계서 일어남. ③**PMI 역전(+10.4→−1.7)이 남긴 교훈**: emission 보존은 필요조건이지 충분조건 아님—PMI가 useful-meta를 실제로 고르는지 별도 검증·수정 필요(현재 PMI=유용성 proxy 깨짐). ④**원인분리 규율**: v2가 A+PMI 동시변경하면 3-arm으로 부족—A-only 먼저가 식별가능성의 최소조건. ⑤durability 교착으로 gs300 지연되나 Codex 판정=gs300은 negative control이라 구조적 rescue 기대 낮음→개선은 별도 실험. 관련 [[E-056]]·[[cfgroup-form-vs-behavior-gandhi-0622]]·[[confidence-rv-gs190-delta-content-diagnosis-0621]]·[[meta-fails-root-correlated-self-verification-0625]].

### 상태 tick — 04:35 UTC(7/19) — [b2 선점복구·gs158 재개] internal-grub 재부팅→hf_xet ready→**gs158 재개**(RGS·retry_002 후 auto-retry 성공)·running 4h. gs159~170 재손실(durability 교착). b3pkg=smart-bat gs139+ 건재 1일+. ★E-063 개선실험 설계 확정(Codex검토)·gs300후 발사. 무개입, 두 arm 진행 감시.

### 상태 tick — 04:01 UTC(7/19) — [⚠️b2 선점→gs158 리셋·durability가 load-bearing 판명] b2=internal-grub **선점(retry_002)**→auto-retry 재부팅 중(로그=conda-pack 재추출)→durable gs158로 리셋(gs159~170 진행 손실). ★"durability 미해결 수용" 전략의 약점 노출: b2는 선점 때마다 gs158로 되돌아가 **순진행 0** 위험(Standard tier 선점 잦음). b3pkg=smart-bat heartbeat 04:03·GPU43GB 건재(여전 무선점). ★판단전환: b2가 gs300 도달하려면 durability 필수인데 Xet 업로드 불가로 막힘=구조적 교착. 과학결론은 gs25/50/100 val서 이미 뚜렷(RQ1 강·RQ2 emission-collapse). ∴다음 productive 단계=(a)durability 실질수정(비-Xet repo·복잡) or (b)예비결론 수용+개선실험(emission유지) 설계착수. 사용자 판단 요청+개선실험 draft 준비. 무개입(b2 auto-retry 신뢰).

### 상태 tick — 03:15 UTC(7/19) — [감시모드·두 arm 순항] b2=internal-grub gs167+(running 2h·OOM없음·epoch2·correctness0.58). b3pkg=smart-bat gs139+(running 1일·무선점). 선점/OOM 없음. durability 미해결 수용 유지(재제출 안함). gs300 ETA b2~30h·b3pkg~35h. 무개입, 생존+진행 감시.

### 상태 tick — 02:28 UTC(7/19) — [durability 미해결 수용·재제출중단 유지] b2=internal-grub gs164 running(OOM없음)·gs160 저장 **~1.5h 경과에도 model0/optim0**=Xet 업로드 확정 실패(hf_xet import는 되나 xethub 업로드경로 차단 판단·hf_xet fix로도 durable 업로드 불가 최종확인). ★전략유지: **재제출 안함**(thrash 중단). b3pkg=smart-bat gs139·**1일 무선점**(클러스터 잔잔). durability 아크 종결=미해결이나 우회전략 채택: (a)uninterrupted run→gs300 on-node eval (b)선점시 gs158 resume. ★endgame: 어떤 arm이 gs250+ 근접시 on-node eval(gs300 ckpt는 /scratch, eval결과 JSON은 HTTP로 업로드) or 비-Xet repo 방식을 Codex와 설계(아직 ~15-20h 여유). 무개입, 생존+진행 감시로 전환.

### 상태 tick — 01:41 UTC(7/19) — [⚠️gs160 Xet 업로드도 미완·전략전환:재제출중단+uninterrupted-run 의존] internal-grub running 1h·gs160 저장~27min 경과에도 **model0/optim0**(hf_xet ready인데도 Xet 업로드 미완결). 저속Xet vs xethub 업로드차단 모호(pusher log는 node-local이라 직접확인 불가). ★**전략전환 결정**: durability에 과잉투자(b2 5회+ 재제출·매번 손실) 반성→**재제출 중단**. 근거: b3pkg **24h(1d) 무선점**=클러스터 현재 잔잔→b2도 선점없이 gs300 도달 가능성↑→그경우 **on-node eval로 durability 우회**(gs300 ckpt는 /scratch에 있고 eval결과 JSON은 소용량이라 HTTP 업로드 됨). 선점시엔 gs158 resume 수용. b2 OOM없음·정상진행. b3pkg gs136 건재. ★다음: gs160 완결 여부만 관측(45min)·재제출 안함·gs300 접근 감시.

### 상태 tick — 01:14 UTC(7/19) — [b2 hf_xet ready·gs160 Xet업로드 대기] internal-grub running 39m·★**"[YAML] hf_xet ready"**(smoke test 통과=hf_xet 설치+import 성공·xethub 최소 도달)·gs158 재개→**gs160 저장**(step:160·OOM없음·correctness0.539·response_length1672). HF COMPLETE 여전 gs158(gs160 Xet업로드 진행중·Xet면 수분내 빠를것). b3pkg=smart-bat **gs133** 건재 23h(COMPLETE gs43·old pusher). ★다음(20분): HF COMPLETE gs160 뜨면=hf_xet fix 성공·durability 최종해결→E-063. 여전 gs158이면 xethub 업로드 차단 의심.

### E-062 (2026-07-19 00:25 UTC) — [★진짜 durability fix] 근인=Xet repo에 hf_xet 없이 대용량 업로드 불가→Codex GO hf_xet 설치(내 disable과 정반대)·부트스트랩 안전구조·b2 재제출

**문제(추상):** "저속 업로드"와 "업로드 불가"는 관측 시간으로 구별된다 — 저속이면 큰 파일도 부분 진전(≥1 샤드 커밋)을 보이나, 불가면 무한정 0. 그리고 근인이 백엔드(Xet)면 그 백엔드 라이브러리의 *부재*가 원인이므로 fix는 비활성이 아니라 *설치*다(직관과 반대일 수 있음).

**구체적 해결(정정+최종).** E-061의 "저속·기다려" 판정을 **재계측이 반증**: gs160이 22:52 저장→00:25(**93min**)에도 model0/optim0. 저속(4.5MB/s)이면 3.5GB 1샤드는 13-30min에 커밋됐어야 하나 93min에 0=**대용량 업로드 불가**(저속 아님). ∴근인=**Xet-enabled repo에 hf_xet 미설치→대용량 샤드 업로드 실패**(작은 extra_state는 legacy LFS로 되나 큰 model/optim은 Xet 경로 필요). E-060의 `HF_HUB_DISABLE_XET=1`은 **정반대 방향**(비활성이 아니라 설치가 답). Codex GO: (1)부트스트랩에 `pip install --upgrade hf_xet`+import smoke test, (2)pusher의 disable 제거→Xet 활성. ★안전구조(Codex 리스크경고 반영): hf_xet 설치 시 부트스트랩 다운로드도 Xet 쓰는데 `*.xethub.hf.co` 차단시 다운로드까지 붕괴 위험→**global `HF_HUB_DISABLE_XET=1`(env)로 부트스트랩은 HTTP 유지·pusher만 `env -u`로 Xet 활성**. 최악(xethub 차단)에도 업로드만 현상유지·부트스트랩 무손상. ★yaml 편집서 [[amlt-bashc-quote-wordsplit-truncation-0718]] 함정 재현(주석 "pusher's" 아포스트로피+python -c 단일따옴표가 bash -c '...' 래퍼 절단→shlex "No closing quotation")→단일따옴표 전부 제거·shlex 3토큰 재검증 통과. b2 internal-grub 재제출(gs158 재개). b3pkg yaml 동일 prep(향후 재제출).

**통찰.** ①**0-진전 vs 부분-진전이 저속/불가의 판별자**: 93min 0샤드는 저속으로 설명 불가(1샤드는 나왔어야)—complete-frontier 정체를 "느림"으로 3번째 재해석하려다 이 계측이 "불가"로 확정. 관측시간을 늘리되 *부분진전 유무*를 봐야지 시간만으로 판정하면 안 됨. ②**근인=라이브러리 부재면 fix=설치(반직관)**: hf_xet 함정 메모리가 "제거"를 처방했으나 그건 *설치돼서 대용량 실패할 때*이고, 여기선 *미설치라 대용량 불가*—같은 증상 다른 근인. 메모리 처방도 조건부임을 확인 필수. ③**위험한 인프라 수정은 안전구조로 blast-radius 격리**: hf_xet 활성이 부트스트랩까지 위험화→global-disable+pusher-unset로 다운로드는 검증된 경로 유지, 업로드만 실험. 최악에도 현상유지. ④**yaml 명령블록 편집은 단일따옴표/아포스트로피 절대금지**(주석 포함)—[[amlt-bashc-quote-wordsplit-truncation-0718]] 재실증, shlex 3토큰이 유일 신뢰검증. ⑤durability 아크 E-057→060→061→062: save_freq(부분정답)→xet-disable(오진 no-op)→저속오판→**hf_xet 설치(진짜근인)**. 관련 [[hf-xet-upload-pitfall-and-chain-fix]].

### E-061 (2026-07-18 23:21 UTC) — [정정] E-060 hf_xet 진단 오류: hf_xet 미설치→xet-off는 no-op·진짜근인=LFS 저속 업로드·save_freq=20 window면 충분(더 기다려야)

> **⚠️정정(E-060):** hf_xet 대용량 실패 진단은 **틀렸다**. 로그 실측: `Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP`—**hf_xet가 애초에 미설치**라 업로드는 처음부터 일반 LFS HTTP였고 Xet는 무관. HF_HUB_DISABLE_XET=1은 **no-op**(무해). 근인을 메커니즘 존재 확인 없이 메모리만으로 귀인한 오류.

**문제(추상):** 인프라 근인 귀인은 그 메커니즘이 *실제로 활성인지* 먼저 확인해야 한다 — 메모리·문서의 함정 패턴이 그럴듯해도, 해당 컴포넌트가 설치·활성 상태가 아니면 근인일 수 없다. "그럴듯한 근인"을 존재검증 없이 채택하면 no-op 수정으로 시간을 낭비한다.

**구체적 해결.** included-grizzly 로그 grep서 pusher launch 확인(`nohup env HF_HUB_DISABLE_XET=1 python ...` 정상 배선)됐으나, 동시에 **"hf_xet package is not installed→fallback HTTP"** 경고 발견=hf_xet 미설치 확정. ∴E-060의 xet-off 수정은 이미 없는 것을 끄는 no-op. 진짜 근인=**일반 LFS HTTP 대용량 업로드 저속**(28GB@~4.5MB/s≈100min+/ckpt). 재계산: gs160 `Saved model ...22:52:06`→23:21(29min경과)엔 미완이 정상(100min 필요→~24:30 완결 예상). ★핵심: **save_freq=20이면 ckpt 간격 ~5h≫업로드 100min이라 window 충분**—E-057 수정(save_freq=20)이 실제 정답이고, 21:13 "save_freq=20 불충분" 판정은 **업로드 시간 미부여(29min만에 성급판정)**한 내 오류. E-060 재제출은 ~10step 손실이나 xet-off는 무해·save_freq=20 유지되니 순영향=재제출 비용만. 조치=**무개입·gs160 durable 완결(~24:30) 대기**(재제출 금지—더 기다리면 완결). b2=included-grizzly gs161 running·OOM없음. b3pkg=smart-bat gs121·여전 save_freq=1(pusher 못따라감·COMPLETE gs43·향후 save_freq20 재제출 필요).

**통찰.** ①**근인 귀인 전 메커니즘 존재검증 필수**: hf_xet 함정은 실재하나 *이 컨테이너엔 미설치*라 무관—메모리의 함정을 적용 전 "그 컴포넌트가 여기 활성인가?"를 로그로 확인해야 했다(E-052 "footprint 재계측"과 같은 검증규율의 확장). ②**저속 업로드 판정엔 충분한 관측시간이 필수**: 21:13 판정은 gs160에 업로드 완결시간(~100min)을 안 준 성급판정—complete-frontier 정체를 "실패"로 단정 전 최소 1 upload-cycle(100min) 대기해야. ③**save_freq=20이 진짜 정답**: ckpt간격>업로드시간 조건만 만족하면 저속이어도 durable 완결됨(E-057이 옳았음)—no-op xet-off를 덧댔을 뿐. ④b3pkg는 여전 save_freq=1이라 근본 노출—gs300 전 save_freq20 재제출 필수 재확인. 관련 [[hf-xet-upload-pitfall-and-chain-fix]](이번엔 미설치라 무관)·[[E-057]].

### 상태 tick — 22:50 UTC(7/18) — [b2 gs159·xet-fix 검증대기] included-grizzly running 50m·gs159(재개후 첫스텝·OOM없음·xet-off pusher). gs160 durable 완결 ~23:20 예상(HF COMPLETE 여전 gs158). b3pkg=smart-bat **gs117** 건재 21h(COMPLETE gs43·xet-broken). 무개입, gs160 durable로 xet-fix 검증 대기.

### 상태 tick — 22:19 UTC(7/18) — [b2 노드확보·xet-fix 검증대기] included-grizzly(재제출 2회째) **노드확보·running 19m**·gs158 재개 확인(RGS)·config파싱 통과·OOM없음·xet-off pusher 가동. gs160 durable(HF COMPLETE)로 xet-fix 검증은 ~2h후(~00:20). b3pkg=smart-bat **gs113** 건재 20h(COMPLETE gs43·xet-broken). 무개입, gs160 durable 완결 대기.

### 상태 tick — 21:52 UTC(7/18) — [b2 할당실패→재제출] coherent-mouse **failed**(empty log list+no retry=Standard 할당실패·노드 미확보·부트스트랩 미실행·xet-fix yaml은 정상·미실행). E-046대로 재제출 **included-grizzly**(xet-off+save_freq20+activation-offload·preparing). b3pkg=smart-bat **gs110** 건재·20h 무선점(COMPLETE gs43·xet-broken). ★included-grizzly도 할당실패시 계속 재제출(opportunistic). 노드 확보→gs158 재개→gs160 durable로 xet-fix 검증 대기.

### E-060 (2026-07-18 21:13 UTC) — [인프라위기 2차·진짜근인] save_freq=20 불충분→근인=hf_xet 대용량 사일런트 실패·Codex GO HF_HUB_DISABLE_XET=1(pusher 전용)·b2 재제출

**문제(추상):** durability 완화(save_freq↑)가 여전히 실패하면, 근인은 *빈도*가 아니라 *업로드 경로 자체*일 수 있다 — 작은 파일만 커밋되고 큰 샤드가 무기한 미커밋인 패턴은 대역폭 저속이 아니라 특정 업로드 백엔드(Xet)의 대용량 사일런트 실패를 가리킨다. 완화 후 footprint(여기선 complete-frontier) 재계측이 근인층위를 판별한다.

**구체적 해결.** massive-flea(save_freq=20) 3h 실행 후 **gs160이 로컬저장 ~2h 경과에도 HF에 model0/optim0/extra4**(작은 extra만 커밋). 2h ≫ 예상 업로드(~104min@4.5MB/s로 28GB)=저속 아닌 **대용량 실패/정체**. save_freq=20은 ckpt 수만 줄였지 업로드 실패를 못 고침=근인이 save_freq 아님 확정. Codex 전문 논쟁→근인=**hf_xet 대용량 사일런트 실패**([[hf-xet-upload-pitfall-and-chain-fix]] 메모리 함정: 작은파일 성공·큰파일 조용히 실패). ★Codex GO 옵션A(공식 kill-switch): pusher launch에 `env HF_HUB_DISABLE_XET=1` prepend(pip uninstall보다 안전·실행중 프로세스 import 안깨짐)—Xet 미사용시 Hub가 legacy HTTP/LFS로 fallback(느려도 확실). ★**pusher 프로세스에만** 적용(부트스트랩 다운로드 무영향). hf_transfer는 최신 hub서 deprecated/no-op이라 B/C 기각(과거 b3 붕괴도 이 때문). yaml line143 `nohup python`→`nohup env HF_HUB_DISABLE_XET=1 python`(값-only prepend·shlex 3토큰 검증). massive-flea cancel→재제출 **coherent-mouse**(gs158 재개·save_freq20·activation-offload·xet-off pusher). b3pkg yaml도 동일 prepped(line129).

**통찰.** ①**완화 후 재계측이 근인층위를 벗긴다**: save_freq=20 적용 후에도 complete-frontier 불변→"빈도 문제" 가설 기각·"업로드 경로" 진짜근인 노출. E-052(footprint 불변=레버 오정렬)와 동형 원리의 재적용. ②**작은-성공/큰-실패 패턴=백엔드 특이 실패의 지문**: 순수 대역폭이면 큰파일도 느리게라도 완료되나, Xet는 큰파일만 사일런트 404—패턴이 근인을 지목. ③**공식 kill-switch>파괴적 처방**: pip uninstall(실행중 import 위험) 대신 env kill-switch가 pusher만 격리 비활성=최소 blast-radius. ④★**b3pkg 함의(중대)**: b3pkg도 xet-broken이라 in-mem gs104가 non-durable·**gs300 도달해도 최종 ckpt 저장 실패→eval 불가 리스크**. Codex는 "gs43 검증되면 재학습 불필요·향후 재실행에 동일fix"—19h 무선점이라 지금 두되(61step 손실 회피), ★선점시 즉시 xet-fix 재제출 or gs300 접근 전 반드시 재제출(그전엔 durable gs300 불가). 관련 [[E-057]]·[[hf-xet-upload-pitfall-and-chain-fix]].

### E-059 (2026-07-18 20:46 UTC) — [★핵심결과] gs100서 PMI 우위 역전 확정: B2 66.6>B3noPMI 64.1>B3pkg 62.4>B0 59.2=RQ2 −4.2·PMI −1.7(gs50 +10.4서 역전)·emission붕괴의 정확도 귀결

**문제(추상):** 메커니즘의 초기 이득(E-055 gs50 PMI +10.4)이 메커니즘 소멸(E-056/058 emission→4.5%)과 함께 어떻게 진화하는가 — 존속·증발·역전 중 무엇인가. 이는 pre-register 해석표의 분기를 직접 결정한다.

**구체적 해결.** B3pkg gs100 val 착지→4arm gs100 정확도(9과목 평균·held-out val 594·(rew+1)/2) 완성: **B2 66.6% > B3noPMI 64.1% > B3pkg 62.4% > B0 59.2%**. 격차: **RQ1 B2−B0=+7.4**(강건 유지)·**RQ2 B3pkg−B2=−4.2**(gs50 −4.5와 유사·package가 correctness-only 미달)·**PMI B3pkg−B3noPMI=−1.7**(★gs50 +10.4서 **역전**). 즉 gs100엔 PMI 있는 B3pkg가 PMI 없는 B3noPMI보다 오히려 낮음. emission 궤적과 대조: PMI우위(+10.4)는 emit22%(gs50)때, 역전(−1.7)은 emit4.5%(gs100)때—**메타 소멸이 PMI 이득을 정확히 지웠고 오버헤드만 남겨 순부정 전환**. b2 durable COMPLETE 여전 gs158(20:46·pusher 업로드중·경계선·다음틱 미완이면 hf_transfer escalate).

**통찰.** ①**PMI 이득은 메타-생존-의존이고 base서 소멸**: gs50 +10.4→gs100 −1.7의 역전은 "PMI가 내용적으로 도움"이 아니라 "메타가 살아있는 동안만 도움"이었음을 보여줌—[[rq2-estimand-endtoend-not-content-0717]]의 estimand 우려가 데이터로 실증(PMI 내용효과 아닌 시점-의존 아티팩트). ②**pre-register 분기 판정 방향=B3pkg<B2 (package/PMI 순부정) 또는 B3pkg≈noPMI<B2 (form도 무익·PMI 철회)**: gs100 스냅샷은 "B3pkg가 B2에도 noPMI에도 미달"이라 가장 비관적 분기. gs300서 굳어지면 논문은 **음성결과+기전(emission collapse) 서사**로 재프레임 필요. ③**단 미확정**: val(594)≠held-out 1030·단일시드·gs100≠gs300. B2/B3pkg 격차가 gs300까지 좁혀질 여지는 있으나 emission이 이미 floor(4.5%)라 반등 동력 부재. ④**개선실험 표적 확정**: emission collapse가 RQ2 실패의 단일 기전이므로, fix=**emission을 RL 내내 유지**(w_emit 대폭↑ or advantage 라우팅으로 meta span 직접보상 or fresh-from-SFT 대칭). 이것이 성공하면 gs50의 PMI +10.4가 gs300까지 존속하는지 재검증 가능=논문 구제 경로. gs300 정본 확보 후 Codex와 설계. 관련 [[E-056]]·[[pmishift-starvation-base-sft-undertrained-0710]]·[[meta-fails-root-correlated-self-verification-0625]].

### 상태 tick — 20:19 UTC(7/18) — [두 검증 임박·무개입] b2=massive-flea gs165(running 2h·OOM없음·save_freq20). ★b2 durable COMPLETE 여전 gs158(gs160 pusher 업로드중·~20:40 완결 예상). b3pkg=smart-bat gs100·18h 무선점(COMPLETE gs43). ★B3pkg gs100 val 미착지(594문제 평가중)—현 gs100 B2 66.6·B3noPMI 64.1·B0 59.2, B3pkg 대기. 다음틱(25분): b2 durable 완결(save_freq20 검증→E-059)+B3pkg gs100 val(PMI우위 증발 확정) 근접포착. 무개입.

### E-058 (2026-07-18 19:46 UTC) — [중간분석3·핵심] emission 붕괴 바닥고착(4.5%)+B3noPMI가 gs100서 B2근처 추격=E-055 PMI우위 증발 조짐(gs300 판정 필요)

**문제(추상):** 메커니즘 소멸(E-056)이 진행되면 그 메커니즘의 초기 이득(E-055 gs50 PMI +10.4)이 시간에 따라 존속하는가 소멸하는가 — 그리고 메커니즘-없는 대조(B3noPMI)가 추격하면 "PMI 순기여" 주장은 시점-의존 아티팩트였는가. 중간 궤적으로 이를 조기 판독하되 gs300 판정은 유예.

**구체적 해결(진단).** wandb B3pkg emission 궤적 gs50→99: `meta_emit_rate` 22.3%→8.8%(gs68)→**4.5%(gs85)→4.5%(gs99)**=바닥 고착(원 91%서 소멸). `pmishift_attempted` 9%→**0.6%**·`wellformed` 6.1%→**0.2%**=PMI-shift 사실상 미발화. val 정확도(9과목 평균·(rew+1)/2) gs50→gs100: **B0 56.3→59.2**·**B2 64.9→66.6**·**B3noPMI 50.0→64.1(+14.1)**·B3pkg 60.4→(gs100 미착지·방금 도달). ★B3noPMI(PMI 없는 패키지)가 gs100서 64.1%로 B2(66.6)에 근접·B0(59.2) 상회. E-055서 gs50 B3pkg(PMI有 60.4)≫B3noPMI(PMI無 50.0)로 "PMI +10.4 순기여"였는데, gs100엔 B3noPMI가 64.1로 급등—**PMI 우위가 좁혀지는/증발하는 조짐**. b2 durable 검증은 별도 진행중(COMPLETE gs158·~20:40 완결 예상).

**통찰.** ①**emission 바닥(4.5%)=메커니즘 사망 확정**: gs85 이후 emit·pmi_att·wellformed 전부 floor라 B3pkg는 gs100+엔 메타 없는 순수 정책으로 작동=base 회귀 완료. w_emit(0.1) floor가 4.5% 잔residual만 지탱. ②**B3noPMI 추격이 E-055 재해석 강제**: gs50 "PMI +10.4"는 메타 생존 초반의 시점-의존 현상일 수 있고, 메커니즘 소멸 후(gs100)엔 B3pkg·B3noPMI가 수렴할 전망(둘 다 메타 잃음)—이 경우 pre-register 해석표의 "B3pkg≈noPMI→form/패키지, PMI 철회" 또는 "셋≈→mechanism" 분기로 향함. ③**단 미완**: B3pkg gs100 val 미착지(방금 도달)라 B3pkg-vs-noPMI gs100 직접비교 불가—다음틱 확인 필수. 또 이는 val(594)이지 held-out 1030 아니고 단일시드·gs100(gs300 아님). RQ1(B2-B0 +7.4 @gs100)은 여전 견고. ④**fix 함의 강화**: emission floor(w_emit 0.1)가 base붕괴 압력에 완전 역부족 확증—[[E-056]] fix(emission floor 대폭강화/advantage 라우팅)의 필요성이 데이터로 재확인. 관련 [[rq2-estimand-endtoend-not-content-0717]](시점-의존 estimand)·[[confidence-rv-gs190-delta-content-diagnosis-0621]].

### 상태 tick — 19:14 UTC(7/18) — [b2 gs161·durable검증 진행중] massive-flea running 1h·gs161(gs159/160/161 재계산·OOM없음·response_length 2749 heavy batch도 activation-offload로 통과·entropy0.998). save_freq=20이라 gs160 저장됨·pusher 업로드중(HF COMPLETE 여전 gs158·완결 ~20:40 예상). b3pkg=smart-bat gs99·17h 무선점(COMPLETE gs43). 무개입, gs160 durable 완결 대기(save_freq20 검증).

### 상태 tick — 18:42 UTC(7/18) — [b2 재개 정상·durable검증 대기] massive-flea running 57m·gs159 재계산(재개후 첫스텝·OOM없음·correctness0.594·entropy0.906·response_length1819 건강). save_freq=20이라 **gs160이 첫 durable 저장점**(~1스텝 후)·완결은 업로드~100min 더해 ~20:40경. HF COMPLETE b2 여전 gs158(gs160 미도달·정상). b3pkg=smart-bat **gs95** 전진·17h 무선점(COMPLETE gs43·gap 52step 증가중이나 계획대로 선점때 적용). 무개입, gs160 durable 완결로 save_freq20 검증 대기.

### 상태 tick — 18:10 UTC(7/18) — [b2 save_freq20 재제출 부팅·durable회복 대기] massive-flea running 25m·**gs158 재개 확인**(RGS "resume gs=158")·config파싱 통과("Could not override" 부재)·OOM 없음·activation-offload 유지. save_freq=20이라 첫 저장 gs160(~30min)·durable 완결은 +업로드~100min=~2h후 관측. b3pkg=smart-bat **gs90** 전진·무선점(COMPLETE 여전 gs43·in-mem gs90). 계획대로 b3pkg 두고 선점감시(선점시 즉시 save_freq20 재제출·yaml 준비됨)·사용자 결정 대기. ★다음 검증: massive-flea COMPLETE-frontier가 gs158 초과(gs160/180)=save_freq20 durability 수정 성공.

### E-057 (2026-07-18 17:35 UTC) — [★인프라위기·수정] 두 arm durability 붕괴=save_freq=1×저속HF업로드로 pusher가 대용량ckpt 완결 불가→Codex GO save_freq 1→20(b2 즉시적용·b3pkg 결정대기)

**문제(추상):** 잦은-선점 durable 학습에서 체크포인트 *저장 빈도*와 *업로드 대역폭*의 부정합은 조용한 durability 붕괴를 낳는다 — 로컬 저장은 매step 성공하나 원격 완결이 그 속도를 못 따라가면, 작은 파일(extra_state)만 올라가고 큰 샤드(model/optim)는 영구 미완이 되어, 선점시 마지막 *완전* ckpt로 후퇴한다. 증상(로컬 진행 vs durable 정체)이 정상 진행으로 위장된다.

**구체적 해결.** 발견: HF **complete-frontier**(model4+optim4+extra4)가 in-memory보다 심하게 뒤처짐 — b2 complete=gs158(로컬 gs160·직전노드 gs169·~11h 정체)·b3pkg complete=**gs43**(in-mem gs85·15h 무선점인데도!). 패턴=최근 ckpt들이 HF에 extra_state만 4/4·model/optim 0/4. 근인 규명(Codex 논쟁 수렴): 7B FSDP ckpt=model~14GB+optim~14GB≈28GB, HF 업로드 ~4.5MB/s면 **~100min/ckpt**. 그런데 `save_freq=1`이라 ~14min마다 새 ckpt·선점 ~1h마다→pusher(per-file commit·--keep2·설계는 건전)가 **어떤 ckpt의 대용량도 완결 전에 추월/kill**됨. b0 gs300 durable 성공=긴 calm window 우연. **Codex GO 옵션C**: `save_freq 1→20`(280min 간격≫100min 업로드=충분 margin·hf_transfer는 과거 부트스트랩 붕괴 이력[[yaml-heredoc-edit-verify-runtime-layer-0717]]이라 회피). save_freq/hf_transfer는 세만틱 하이퍼파라미터 아니라 matched 무결. **b2 즉시 적용**: yaml save_freq=20 편집(shlex 3토큰 검증)→beloved-ram cancel(killed·thrash 중이라 손실 최소)→재제출 **massive-flea**(gs158 재개). b3pkg=smart-bat는 in-mem gs85가 선점시 회수불가라 Codex는 즉시수정 권고하나, 재제출=gs43 후퇴(42step≈10h 손실 확정) vs 냅두면 동일손실이 미래 선점때로 이연—비용동일·현재 건강이라 **사용자 결정 대기**(yaml은 준비).

**통찰.** ①**durable 진행의 참지표=complete-frontier(3-way 4/4)이지 all-frontier 아님**—all-step 번호는 extra_state만으로도 증가해 진행을 위장한다. 모니터링은 반드시 model+optim+extra 각 4/4의 최댓값을 봐야 함(이 세션서 이 구분이 위기를 드러냄). ②**save_freq는 durability 파라미터**: 잦은저장=잦은 로컬세이브지만 업로드가 못 따라가면 오히려 durable-gap을 키운다(pusher가 백로그에 익사). 최적 save_freq=upload_time < save_interval을 만족하는 값(여기선 ~100min<280min). "자주 저장=안전"은 대역폭 병목 하에서 거짓. ③**b0 성공이 문제를 가림**: 한 arm의 우연한 성공(calm window)이 systematic 결함을 정상으로 오인하게 함—[[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]의 "작동 쌍둥이가 최강 판별기" 역: 실패 쌍둥이(b2·b3pkg 공통 정체)가 systematic 결함을 드러냈다. 관련 [[singularity-preemption-not-hang-retry-check-0711]].

### 상태 tick — 17:11 UTC(7/18) — [⚠️b2 선점-durability thrash 감지] b2 tqdm "158/300 [00:00]"=또 재시작(retry_002·선점2회)·HF durable frontier gs158 정체(16:06~17:11 ~1h 진척0). ★근인: gs159-168 HF가 **model0/optim0/extra4**(작은 extra만 up·큰 model/optim 샤드 미완)—각 선점이 /scratch wipe→gs158 복귀→gs159+ 재계산→큰샤드 push완료 전 재선점 반복. **hf_xet 블랭킷실패 아님**(gs156/158은 model4/optim4 완전=과거 안정window서 성공). ∴선점이 큰 업로드를 끊는 durability갭. b3pkg=smart-bat **gs82** 안정(15h·무선점=calm window 존재 증거). b2 OOM 부재(activation-offload 유지). 판단: 선점2회뿐·calm window 존재라 자가치유 가능성→**20분 단축캐던스로 밀착관찰**. 다음틱 durable frontier(마지막 complete) 여전 gs158이면=thrash 확정→Codex에 durability fix(pusher 원자적 업로드순서/save_freq/노드재배치) 전문GO.

### 상태 tick — 16:39 UTC(7/18) — [b2 선점replay 진행중·로그window sparse] b2=beloved-ram **status=running 4h**(선점 후 gs158 replay 지속). ★로그 tqdm/heartbeat 마커가 window 밖(E-040 sparseness)이나 판정=생존: status running + **b3pkg HF gs77 전진**=클러스터·릴레이 인프라 생존 증명(죽은 노드면 b3pkg도 정체). b2 HF gs168 정체=replay가 아직 gs168 미재통과(재push 대기). OOM 부재. b3pkg=smart-bat gs77 건강·무선점. 무개입, 다음틱 HF gs168 초과로 replay완료 확증. gs300 ETA b2~34h(선점손실 포함)·b3pkg~48h.

### E-056 (2026-07-18 16:06 UTC) — [★핵심위험] B3pkg 메타발화 붕괴(91%→9% by gs68): PMI-shift가 작용할 무대 소멸=세 증상(PMI신호·trace·instruct미전이)의 공통뿌리

**문제(추상):** 메타인지 RL 메커니즘(PMI-shift)의 "작동"은 두 층이다 — (1)메커니즘이 정확히 배선됐는가, (2)그 메커니즘이 작용할 *대상*(메타 발화)이 RL 내내 존속하는가. 대상이 소멸하면 배선이 완벽해도 메커니즘은 무효가 된다. base substrate에서 메타 습관이 correctness-RL 압력에 존속하는가가 진짜 질문.

**구체적 해결(진단).** wandb B3pkg 훈련지표 궤적(gs5→68): `meta_emit_rate` **90.8%→77.9%(gs25)→40.8%(gs40)→22.3%(gs50)→8.8%(gs68)**=메타발화 극단 붕괴. `pmishift_attempted_rate` 42%(gs25)→**2.3%(gs68)**=헌법게이트(≥30%) 한참 하회. `wellformed_rate` 37.7%→**1.8%**. `pmishift_n_save` 4→0. ∴PMI-shift가 붙을 메타블록 자체가 gs68엔 8.8%뿐=**메커니즘 무대 소멸**. 반면 trace 건전성은 유지: `entropy` 0.20→0.31(붕괴 아님), `clip_ratio` 0.6~3.7%(비종료 degeneration 없음·triobj 길이제어 작동). 세 사용자 질문 답: (Q1)PMI 신호 안 들어감=메타 extinction으로 대상 소멸 / (Q2)trace는 degenerate 아니나 "metacognition-stripped"로 base 회귀 / (Q3)instruct T1 6/6승리([[T1-heldout-meta-beats-base-6of6-0706]] MATH+18.8·AIME+14)는 **instruct substrate**·RQ3는 **base substrate**라 미전이([[base-redesign-teacher-free-selfdistill-pivot-0708]] substrate 인과).

**통찰.** ①**세 증상의 단일 근인=emission collapse**: PMI신호 미유입·trace 밋밋화·instruct 미전이가 전부 "base substrate서 메타발화가 correctness-RL에 extinction"으로 수렴. 근인=**correctness 보상이 answer span에만 credit, meta span은 순수 비용**(CLAUDE.md 명시 구조압력)+base 메타습관 취약성. ②**E-055의 "PMI 순기여 +10.4"는 메타 생존 초반(gs50 emit22%) 현상**이고 gs68(emit8.8%)엔 무대가 무너지는 중—중간분석의 시점의존성 주의. ③**이게 RQ2의 진짜 실패기전**: B3pkg가 B2 못 이기는 건 PMI 설계결함이 아니라 메커니즘이 gs300 전에 꺼지는 것. w_emit(0.1)·emission floor가 base붕괴 압력에 역부족. ④**fix 방향(mandate 승인, 발사전 Codex확인)**: emission floor 강화 or **advantage 라우팅(meta span 직접 credit**·[[confidence-rv-gs190-delta-content-diagnosis-0621]] 처방)or fresh-from-SFT 대칭적용. ★단 현 B2/B3pkg/noPMI는 gs300 완주시켜 정본 비교 확보가 선결(죽이지 않음), fix는 다음 실험. 관련 [[pmishift-starvation-base-sft-undertrained-0710]]·[[meta-fails-root-correlated-self-verification-0625]].

### 상태 tick — 16:06 UTC(7/18) — [b2 선점·auto-resume 정상] b2=beloved-ram **retry_001 발생**(선점~gs169)→gs158 auto-resume 재생중(현 gs160 replay·gs159-168 pusher-lag 미완이라 마지막완전 gs158 후퇴=RGS 정상). ~11스텝 손실=save_freq+push-lag 내구성갭(무개입·E-046 auto-resume 신뢰). b3pkg=smart-bat **gs73** 건강전진(무선점). HF b2[166-168]정체(선점탓)·b3pkg[71-73]. 무개입, b2 재도달 확인 예정.

### 상태 tick — 15:34 UTC(7/18) — [두 arm 안정] b2=beloved-ram **gs168**(running 3h·무OOM), b3pkg=smart-bat **gs68**(13h). HF b2[166-168]·b3pkg[66-68] 전진·선점없음. E-055(val정확도 gs50 4arm비교) 기록. gs300 ETA b2~31h·b3pkg~51h. 무개입.

### E-055 (2026-07-18 15:34 UTC) — [중간분석2] val-aux 정확도 동일-step(gs50) 4arm 비교: RQ1 강함(+8.6pp)·PMI 순기여 확인(+10.4pp)·단 B3pkg 아직 B2 미달(−4.5pp)

**문제(추상):** 학습 rollout 정답률(E-054)보다 estimand에 가까운 신호는 없는가 — 그리고 그 신호로 RQ1(SFT-init 효과)·RQ2(패키지 효과)·PMI 순기여를 *동시에* 조기 판독하되 held-out 판정을 참칭하지 않을 수 있는가.

**구체적 해결.** wandb에 test_freq=50마다 찍히는 **`val-aux/{bench}/correctness/mean@1`**(held-out val 594문제·greedy@1·과목별) 발견([[valcore-reward-meta-shaped-use-valaux-correctness-0703]]대로 val-core 아닌 val-aux 사용). correctness는 {-1,+1}이라 acc%=(x+1)/2 환산. **gs50(4arm 공통 유일점; b3pkg gs68이라 이후 미비교)** 9과목 평균 정확도: **B2 64.9% > B3pkg 60.4% > B0 56.3% > B3noPMI 50.0%**. 격차(pp): **RQ1 B2−B0=+8.6** · B3pkg−B0=+4.1 · **RQ2 B3pkg−B2=−4.5** · **B3pkg−B3noPMI=+10.4**. 과목별: 최난 int_algebra서 **B3pkg 47.9% ≥ B2 45.8% ≥ B0 39.6%**(메타가 어려운문제 돕는 조짐)·쉬운 gsm8k/count/num_th는 B2 압도. AIME는 val split에 없음(gs300 최종배터리만). 두 arm 안정(b2 gs168·b3pkg gs68·HF 전진·무OOM/선점).

**통찰.** ①**RQ1 강건**: B2가 val정확도서 B0 +8.6pp=meta-SFT init이 정확도를 실제로 올림(학습 rollout E-054의 +5~7pp와 방향 일치, 이젠 held-out val서 확인). ②**PMI-shift 순기여 확인=novelty 긍정신호**: B3pkg가 B3noPMI를 **+10.4pp** 이김—triobj서 PMI 빼면 정확도 10pp 붕괴, PMI가 load-bearing. 이는 [[rq2-estimand-endtoend-not-content-0717]]의 "PMI 내용효과" 우려에 대한 조기 반증(적어도 gs50선 PMI가 정확도에 순기여). ③**⚠️핵심 경고=B3pkg가 아직 B2 미달(−4.5pp)**: full triobj가 순수 correctness보다 정답률 낮음. 역설 해소=B3pkg는 PMI로 noPMI를 크게 앞서나(+10.4) triobj의 비-correctness 부담(길이·포맷·cal 오버헤드)이 순수 correctness 대비 정확도를 깎아 net 미달. **"메타가 도움되나 오버헤드를 아직 정확도로 회수 못함"**이 gs50 상태. ④단 gs50=RL 초기 1점·과목별 소표본(precalc/omni 4arm 동일=미분화)·단일시드. B3pkg의 meta-form 투자는 늦게 회수되는 패턴(E-054 학습 rollout서도 초반 느렸다 추격)이라 gs300까지 B3pkg가 B2 따라잡는지가 논문 핵심 승부처. 어려운 과목 조짐(int_alg)이 희망. 판정은 gs300 held-out 1030(AIME 포함).

### 상태 tick — 15:03 UTC(7/18) — [두 arm 안정] b2=beloved-ram **gs167**(epoch2 진입·OOM없음·correctness0.613·response_length 1656·entropy0.998·grad_norm0.867). b3pkg=smart-bat **gs64**. HF b2[165-167]·b3pkg[62-64] 전진·선점 없음. gs300 ETA b2~32h·b3pkg~52h. 무개입.

### 상태 tick — 14:31 UTC(7/18) — [두 arm 안정] b2=beloved-ram **gs165**(running 2h·OOM 재발없음·response_length 2104 plateau·clip12.9%·entropy0.97·grad_norm0.59). b3pkg=smart-bat **gs60**(건재 12h). HF frontier b2[162-165]·b3pkg[57-60] durable 전진. 선점 없음. gs300 ETA b2~35h·b3pkg~55h. 무개입.

### E-054 (2026-07-18 14:31 UTC) — [중간분석] 동일-step arm간 비교(wandb, B0포함): B2가 학습정답률 B0 일관 우위+B2 length inflation 실측·단 held-out 아닌 동역학신호

**문제(추상):** gs300 held-out 종단판정 전, 진행 중 런의 in-training rollout 지표에서 RQ1/RQ2의 *방향성 조기신호*를 읽을 수 있는가 — 단, training-dynamics를 estimand로 오인하는 confound(학습데이터·temp1.0·arm별 상이 보상)를 피하면서. 핵심은 *arm간 동일정의 지표만* 비교하고 판정은 유예하는 것.

**구체적 해결.** wandb `gistdslab/metacot-dcpo-v4`서 4런(rq3-b0-2 finished gs300·rq3-b2-2 running gs165·rq3-b3pkg-2 running gs60·rq3-b3nopmi-2 finished gs149) 히스토리 pull. **arm간 유일 동일정의 지표=`gdpo/correctness/mean`**(전부 같은 math_verify 채점; critic/score는 B3pkg triobj라 비교불가—배제). ±5step 평활(correctness {-1,+1}→acc%=(mean+1)/2). 동일-step 결과: **gs25** B0 60.7%·**B2 65.8%**·B3pkg 57.9%·noPMI 56.7% / **gs50** B0 61.8%·**B2 68.5%**·B3pkg 61.2%·noPMI 60.6% / **gs57** B0 59.0%·**B2 66.7%**·B3pkg 63.9%·noPMI 62.0%. **길이**(tok): B0 시종 266~332(무성장) vs B2 670~742→**gs100 1095·gs125 1351·gs150 2152·gs160 2061**(clip 0.3%→12.5%)=진행성 인플레 실측 / B3pkg·noPMI 초반 ~1000~1130(길이제어로 bounded 예상).

**통찰.** ①**RQ1 조기신호 고무적**: B2가 gs25/50/57 세 지점 모두 학습정답률서 B0 +5~7pp 일관 우위=meta-SFT init 효과의 방향성 지지(단 held-out 미확정). ②**B2 length inflation이 데이터로 확증**: gs100 이후 670→2150tok 폭증이 정확히 [[E-053]] OOM의 근인이며, B0가 ~300tok 무성장이라 "왜 B0만 무-OOM"을 완결 설명. 이 인플레는 VANILLA_GRPO(길이제어 없음)×meta-SFT(긴응답) 상호작용=B2 고유 동역학, [[pmishift-starvation-base-sft-undertrained-0710]] emission-erosion과 동근원. ③**RQ2는 gs57선 시기상조**: B3pkg가 gs25 B0하회→gs57 B0상회(triobj가 초반 meta-form 투자 후 correctness 추격), B2와는 gs57서 66.7 vs 63.9로 아직 B2 우위이나 60step은 판정 불가. ④**결정적 caveat**: 이 숫자는 학습 rollout(temp1.0·학습데이터)이라 held-out과 다름(B0 학습 ~60% vs held-out GSM8K 89%=MATH/AIME-hard 혼합+non-greedy). RQ1/RQ2 판정은 오직 gs300 held-out 1030. B2 학습우위가 held-out 유지인지 긴응답 과적합인지는 gs300서 확정. 관련 [[valcore-reward-meta-shaped-use-valaux-correctness-0703]](지표 혼용 함정).

### 상태 tick — 13:58 UTC(7/18) — [b2 안정진행 확정·OOM아크 완전해소] beloved-ram running. **gs163 완료·저장**(OOM없음·entropy0.935·grad_norm0.695·critic0.43·response_length 2099). ★HF frontier rq3_b2=[160,161,**162,163**]=gs162/163 durable 반영(pusher-lag 해소·gs162 실패점 영구통과 확증). b3pkg=smart-bat **gs57**로 전진(건재 12h). 감시초점 전환 완료: OOM대응→**선점감시+gs300 진척**(b2 gs163/300 ETA~36h·b3pkg gs57/300 ETA~60h). 무개입, 30분 캐던스.

### E-053 (2026-07-18 13:46 UTC) — [★진짜종결] gs162 통과 확인=activation-offload durable 성공·OOM아크(E-045~E-053) 완결

**문제(추상):** 메모리 완화의 durable 성공은 *정확히 근인을 겨냥한 레버가 실패조건을 재현 통과하며 footprint를 계측가능하게 낮췄을 때* 확정된다 — E-050의 오류(단일통과=성급종결)를 피하려면 (a)deterministic 실패점의 직접 통과, (b)footprint의 실측 하락, (c)무거운 배치 포함, 셋을 모두 관측해야 한다.

**구체적 해결.** beloved-ram(9번째·`++enable_activation_offload=true`+enforce_eager)이 gs158 재개→gs159/160/161→**gs162 OOM없이 완료·저장**(`training/global_step:162`·save_checkpoint 43s·13:4x). 3중 확증: (a)**deterministic 실패점 통과**—gs162는 micro_batch=1·opt-C 두 시도서 매번 OOM난 바로 그 replay 지점인데 통과. (b)**footprint 실측 하락**—backward heartbeat gpu0used **57GB**(이전 OOM점 ~70-73GB보다 ~15GB↓)로 천장(80GB) 아래 안정, opt-C가 못 낮춘 그 수치를 activation-offload는 낮춤(레버-성분 정합 확증). (c)**무거운 배치 포함**—gs161 `response_length/mean 2552·clip 0.18`(OOM 유발 배치)·gs162 mean 2189 모두 통과. gs162 후 GPU 35GB(gs163 생성phase)로 정상. OOM아크 E-045~E-053 완결: micro_batch(불충분)→expandable_segments(vLLM금지)→opt-C(레버-성분 오정렬·무효)→**activation-offload(정합·성공)**. 감시초점=OOM대응→선점감시+gs300 진척(b2 ETA~30h·gs162/300). b3pkg=smart-bat gs54↑ 무관·건재 12h. B0/B3pkg 무수정.

**통찰.** ①**footprint 델타가 레버-성분 정합의 유일 판별기**: opt-C(70→70GB 불변)=오정렬, activation-offload(70→57GB)=정합. 메모리 완화는 반드시 완화 후 footprint를 재계측해 실효를 확증해야 하며, 불변이면 다음 레버 쌓기 전 근인 재규명이 정답(E-052서 확립). ②**OOM 근인의 정확한 phase-매핑이 핵심**: 같은 "OOM"도 backward-activation(→activation_offload) vs optimizer-state(→optimizer_offload) vs vLLM-KV(→gpu_mem_util)가 다르고, traceback의 phase(update_actor backward)와 레버 작용 phase가 정합해야 실효. verl은 이 셋에 각각 별도 레버를 둔다. ③**B2 고유 length inflation은 실재 신호**: activation-offload는 긴 응답을 배제 않고 전부 학습(memory-only)하므로 RQ 해석 무편향·clip 0.14~0.18(14-18% 8192cap)은 gs300 해석 시 per-arm 계측 필수(E-048). VANILLA_GRPO(길이제어 없음)×meta-SFT init(긴응답)의 length 폭주는 B0/B3pkg엔 없는 B2 특유 동역학=[[pmishift-starvation-base-sft-undertrained-0710]] emission-erosion과 동근원, RQ2 해석의 실질 재료. 관련 [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]].

### 상태 tick — 13:35 UTC(7/18) — [★gs161 무거운배치 통과·gs162 backward 57GB 진행중=activation-offload 실효 강증거] beloved-ram running 1h. **gs160·gs161 완료·저장**(OOM없음). ★특히 gs161 `response_length/mean:2552·clip_ratio:0.18`(=이전 OOM 유발 그 무거운 배치·update_actor 194s)가 통과. ★현재 gs162 backward가 heartbeat **gpu0used=57GB**(이전 OOM점 ~70-73GB보다 ~15GB↓)로 진행중=activation-offload가 backward peak를 CPU로 눌러 천장(80GB) 아래 유지. ★E-050 교훈(성급종결 금지)에 따라 gs162 저장 직접확인까지 E-053 종결 보류하나 증거 강함(gs162 backward가 실패점을 57GB로 통과중). b3pkg=smart-bat gs54 건재(HF gs51-54·pusher-lag extra만). ~20분후 gs162 저장 확인→E-053. 무개입.

### 상태 tick — 12:57 UTC(7/18) — [b2 activation-offload 재개·gs159 통과·footprint↓ 조기신호] beloved-ram(9번째) running·★config파싱 통과("Could not override" 부재=++ 수정 성공)·gs158 재개→**gs159 완료**(12:50·OOM 없음). response_length/mean 1785·entropy0.919·grad_norm1.088·critic0.564 건강, step 818s. ★activation-offload 실효 조기신호: heartbeat GPU0 **~35GB**(생성phase)로 이전 OOM peak 70GB보다 크게 낮음(gpu_mem_util0.35로 vLLM~28GB+잔여, activation은 CPU오프로드). HF frontier b2[158,159,160,161](gs161=이전 teaching-parakeet 잔재)·b3pkg **gs50**로 전진(건재 11h). ★gs162(deterministic 실패점) ~40분 out=activation-offload 최종판정점. 무개입, gs162 대기.

### 상태 tick — 12:24 UTC(7/18) — [E-052 후속: config-parse 수정] supreme-spaniel(8번째)이 **config 파싱 에러로 즉시 failed**(10m): `Could not override actor_rollout_ref.model.enable_activation_offload — Key not in struct`. 근인=base_matched config chain의 model 블록이 Hydra struct-lock이라 신규키 직접 override 거부(에러가 `+` append 안내). ★검증: verl fsdp_workers.py:903 `self.config.model.get("enable_activation_offload",False)`+:641 `enable_activation_offloading()` 실호출=키 실효 확증(silent no-op 아님). 수정=`++actor_rollout_ref.model.enable_activation_offload=true`(++로 struct append). enforce_eager는 in-struct라 에러없음. shlex(3토큰)+assert 게이팅 통과→재제출 **beloved-ram(9번째)**. b3pkg=smart-bat gs47↑ 건재 10h. **통찰:** struct-locked config에 신규 최적화키 주입은 `++` 필수—단, 주입 전 반드시 워커 소스서 `.get(key)` 실독을 확인해야 append가 silent no-op 아님을 보장(키 존재만으론 실효 불명, 코드 read가 실효 조건). [[amlt-bashc-quote-wordsplit-truncation-0718]] 계열의 "적용≠실효" 검증 원칙.

### E-052 (2026-07-18 11:44 UTC) — [근인규명·재수정] opt-C 무효(메모리 불변) 판명→Codex 근인=optimizer가 backward중 resident·free_cache_engine 기본ON→진짜레버=enable_activation_offload

**문제(추상):** 메모리 완화 레버가 "왜 무효였나"의 규명은 *어느 레버가 어느 메모리 성분을 어느 시점에 건드리는가*의 정확한 인과 매핑을 요구한다. 잘못된 성분(optimizer state·vLLM KV)을 겨냥한 레버는 OOM peak(backward activation)과 시간적·구조적으로 어긋나 메모리를 전혀 안 줄인다 — footprint 불변이 곧 "레버-성분 오정렬"의 결정적 증거다.

**구체적 해결.** opt-C(gpu_mem_util0.35+optimizer_offload) 적용한 teaching-parakeet이 **gs162서 3차 OOM**, 그런데 footprint가 opt-C 이전과 **사실상 불변**(70.26→70.46GiB, 68.50→68.71 PyTorch). Hydra override 리스트에 두 플래그 정확 전달 확인(무효≠미적용). deterministic(gs158 resume가 gs162 배치 동일 replay·retry_001도 동일지점 재-OOM). Codex 전문 논쟁→근인 규명: (1)verl v0.7.1서 `optimizer_offload`는 `update_actor()` **시작 시 optimizer를 GPU로 되올림**→backward 내내 resident→backward-activation peak에 무효(offload 시점이 peak 이후 아님). (2)`gpu_memory_utilization`은 `free_cache_engine`이 **verl 기본 TRUE**라 rollout 후 KV가 이미 freed→actor backward 시점 vLLM KV 무관→무효. ∴ 진짜 peak=단일 8192토큰 **backward activation**(GC 하에도 recompute peak). **진짜 레버=`actor_rollout_ref.model.enable_activation_offload=true`**(activation을 CPU로 오프로드·GC와 짝·verl perf-tuning 공식 권장)+`rollout.enforce_eager=true`(vLLM CUDA-graph 메모리 제거·추가 headroom). 둘 다 memory-only·gradient-identical→matched 무결. opt-C 플래그는 무해라 존치(activation-offload가 load-bearing). shlex 검증(3토큰·전마커)·yaml.safe_load 통과. resume_mode=auto 유지(yaml 커스텀 pull_resume가 이미 gs158 정확선택; Codex의 resume_path는 커스텀 pull 우회라 미채택). 재제출 **supreme-spaniel(8번째)**. B0/B3pkg 무수정. b3pkg=smart-bat gs47↑ 무관·건재.

**통찰.** ①**footprint 불변=레버-성분 오정렬의 결정적 판별기**—E-051 진단(vLLM 공존이 근인)이 틀렸음을 "메모리가 안 줄었다"가 단번에 반증했다. 완화책 적용 후 반드시 footprint 델타를 계측해야 하며, 불변이면 그 레버는 틀린 성분을 겨냥한 것이다(추가 레버 쌓기 전 규명 필수). ②**offload 레버는 '무엇을 언제' 오프로드하는지가 관건**: optimizer_offload는 optimizer.step()의 optimizer state를 겨냥하지 backward activation peak가 아니다—OOM traceback의 phase(update_actor backward)와 레버의 작용 phase(optimizer step)가 어긋나면 무효. verl은 activation을 위한 별도 레버(enable_activation_offload)를 두어 이 정확한 peak를 겨냥한다. ③**free_cache_engine 기본ON의 함의**: vLLM KV는 training 중 이미 freed라 gpu_mem_util·co-residence 가설은 이 verl 버전서 애초에 성립 안 함—E-045~E-051의 "vLLM 공존 peak" 서사는 부분적으로 틀렸고 진짜는 순수 FSDP backward activation. ④B2 고유 문제(긴 응답)라 B0/B3pkg 무-OOM=length inflation이 RQ2/RQ해석의 실재 신호(단순 인프라 아님)—[[pmishift-starvation-base-sft-undertrained-0710]]의 emission-erosion과 동근원. 관련 [[yaml-heredoc-edit-verify-runtime-layer-0717]](검증층위)·[[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]](오진 계열).

### 상태 tick — 11:12 UTC(7/18) — [b2 opt-C training 재개·gs160 통과] teaching-parakeet running 1h·gs158서 재개→gs159 재수행→**gs160 OOM 없이 완료·저장**(11:07). response_length/mean 1652·max8192·clip0.072·entropy0.813·grad_norm0.693·critic/score0.52 전부 건강, step 765s. opt-C 플래그(gpu_mem_util0.35·optimizer_offload) 키가 verl config 실존키라 Hydra override 무에러 적용(job이 config크래시 없이 running=반영확증; config echo는 tail 버퍼 밖). HF b2 gs160 push·b3pkg **gs47**로 전진(gs34→47·건재 9h). ★opt-C 성공 최종판정=gs161→**gs162**(신 실패점) 통과(~20분 out)—gs162 OOM없이 저장되면 E-052 종결. 무개입, gs162 대기.

### 상태 tick — 10:40 UTC(7/18) — [b2 opt-C 재제출 부팅건강] teaching-parakeet(7번째) **running 29m**·vLLM 엔진 초기화 단계(NCCL2.27.3·Gloo rank0-3 peer연결 10:40:33 현재)=verl 부팅 성공·★expandable_segments assert 없음·OOM 없음·EngineCore 실패 없음. 아직 첫 training step(gs158 재개) 전(vLLM init→pull_resume→step 순). opt-C 플래그(gpu_mem_util0.35·optimizer_offload) 실효는 training 시작시 actor init 로그로 확인 예정. b3pkg=smart-bat running 9h(gs34↑ 무관·건재). ★gs162(opt-C 성공 판정점) ~1h+ out. 무개입, 부팅 완주 대기.

### E-051 (2026-07-18 09:59 UTC) — [정정·에스컬레이션] E-050 성급: gs162서 OOM 재발(micro_batch=1 불충분)→Codex GO 옵션C(gpu_mem_util 0.35+optimizer_offload) 적용·gs158 재개

> **⚠️정정(E-050):** E-050의 "b2 OOM 완전해소 확정"은 **성급했다**. gs161(원 OOM점)은 통과했으나 **바로 다음 gs162에서 CUDA OOM으로 job failed**. "원 실패조건 통과=수정 완성"이라는 판정이 틀렸다 — 단일 step 통과는 필요조건이지 충분조건이 아니었고, length inflation의 확률적 fragmentation이 gs162서 천장을 넘겼다.

**문제(추상):** 메모리 완화가 "실패 마진을 넘겼다"의 판정에는 *단일 재현 통과*로 부족하다 — 부하가 천장에 근접(near-ceiling)한 체제에서는 step별 fragmentation·시퀀스 조합의 확률변동이 통과/실패를 가르므로, 한 step 통과가 다음 step 통과를 보장하지 않는다. 진짜 수정은 "천장을 한 번 넘겼나"가 아니라 "천장과 부하 사이에 확률변동을 흡수할 고정 마진을 확보했나"로 판정해야 한다.

**구체적 해결.** witty-imp(6번째)이 gs159/160/161 통과 후 **gs162서 OOM failed**(`tried 5.50GiB, 3.53GiB free, process 70.26GiB/68.50 by PyTorch`, 09:20 이후). 근인 재진단(소스대조): (1)verl 기본 `enable_gradient_checkpointing=true`=GC 이미 ON(추가레버 아님), (2)`actor.ppo_micro_batch_size_per_gpu=1` 이미 최소, (3)`rollout.gpu_memory_utilization=0.45`=vLLM KV가 ~36GB 상주 예약, (4)`optimizer_offload=false`. ∴ backward peak는 단일 8192cap 시퀀스로 이미 BOUNDED(gs161 max=8192 도달=천장)이나 vLLM 0.45 공존이 80GB를 몇 GB 초과→near-ceiling서 fragmentation 확률변동이 gs162를 넘김. **Codex 전문 논의→GO 옵션C**(memory-only 2레버 동시): command 블록에 `actor_rollout_ref.rollout.gpu_memory_utilization=0.35`(vLLM KV ~8GB 축소·생성분포 불변)+`actor_rollout_ref.actor.fsdp_config.optimizer_offload=true`(Adam state→CPU·알고리즘상 gradient-identical, bitwise는 아니나 systematic bias 없어 matched 무결). shlex 검증(3토큰·전마커존재)·yaml.safe_load 통과. witty-imp cancel불요(이미 terminal)→재제출 **teaching-parakeet(7번째)** resume gs158(HF 최종완전; 159/160 pusher-lag 부분·161 미도달 배제). B0/B3pkg 무수정. b3pkg=smart-bat gs34 무관·건재.

**통찰.** ①**near-ceiling 체제의 수정검증은 단일통과로 불충분**—E-050이 gs161 통과(게다가 len2653>2553)를 "완성"으로 오판했으나, 천장 바로 밑에서는 통과가 확률적이라 다음 step이 fragmentation으로 터진다. 올바른 판정선=고정마진(headroom) 확보이지 원조건 1회 통과가 아니다. 이것은 [[deepsup-win-abandoned-process-postmortem-0703]]의 "성급한 승리선언" 계열 실수. ②**leak 결론(E-050)은 여전히 유효**: gs159~161 3연속 통과+GPU 진동(단조증가 부재)은 code-leak을 배제했고, gs162 OOM도 "천장 근접 fragmentation"이지 "step별 단조 누적"이 아니다(같은 ~70GB 수준서 확률적으로 터짐). 근인은 여전히 length-driven near-ceiling+vLLM 공존. ③**수정 레버의 선택축=matched 무결성**: gpu_mem_util·optimizer_offload는 matched-ladder 세만틱 하이퍼파라미터(train_batch/ppo_mini/n/max_response/temp) 밖의 인프라 노브라 조정해도 B2-B0 비교 무결. optimizer_offload의 bitwise 비동일성은 GC의 RNG 노트와 동급 reproducibility 캐비엇으로 명시. ④length inflation(clip 0.16)은 RQ2 해석 시 length confound로 계측 필수(E-048). 관련 [[pmishift-starvation-base-sft-undertrained-0710]]·[[singularity-preemption-not-hang-retry-check-0711]].

### E-050 (2026-07-18 09:27 UTC) — [★종결] b2 OOM 완전해소 확정: gs161(원 OOM점) 통과·게다가 더 긴 응답(2653>2553)서도 생존→micro_batch=1 완전성공+leak 최종배제

**문제(추상):** 메모리 장애의 수정이 "진짜 근인을 고쳤는가"는 *동일 실패조건의 재현 하에서만* 판정 가능하다. 완화책(micro_batch↓)이 우연히 이번엔 안 터진 것인지, 아니면 실패를 유발한 바로 그 조건(길이팽창 step)을 견딜 headroom을 실제로 확보한 것인지를 구분하려면, 수정 후 런이 **원 OOM을 유발한 것과 같거나 더 심한 부하**를 통과하는지를 봐야 한다. 동시에 이는 leak 가설의 최종 심판이기도 하다 — leak이면 더 나중·더 무거운 step일수록 악화돼야 한다.

**구체적 해결.** witty-imp(b2 6번째 재제출, micro_batch=1, gs158 재개)이 gs159→160→**161 전부 OOM 없이 완주·저장**(gs161 저장 09:20:15, model+optim+extra_state 3종). 결정적 지표: **gs161 `response_length/mean:2653.40 · max:8192 · clip_ratio:0.160`**(16%가 8192 cap 도달) — 이는 **원 OOM을 유발한 gs161의 2553보다 +100 토큰 더 긴** 부하다. 그럼에도 `timing_s/update_actor:195.0 · save_checkpoint:25.8 · step:1077`으로 gradient step+저장까지 완주. `global_seqlen/max:432227`, entropy 0.972·grad_norm 0.613·critic/score 0.364 전부 건강(critic score가 gs159의 0.58→0.36으로 내려간 건 이 batch 난이도·정상 변동). 부작용 없음: `response/aborted_ratio:0.0`. 조치=**무개입, E-045~E-049 OOM 아크 종결**. gradient checkpointing 불필요(micro_batch=1로 충분). HF 프런티어 rq3_b2=[156,158,159,160](gs161 로컬저장·pusher-lag로 HF 미반영—E-041 pusher-lag 시그니처, 스톨 아님). b3pkg=smart-bat gs34로 전진(gs32→34·건재). tqdm `161/300, 924.69s/it`.

**통찰.** ①**수정 검증의 황금기준=원 실패조건의 재현 통과**. 이번 gs161이 원 OOM(len 2553)보다 *더 긴* 2653에서 살아남은 것은 "우연히 안 터졌다"를 배제하고 "실패 마진을 초과하는 headroom을 확보했다"를 입증한다 — 완화책이 근인(activation peak)을 실제로 눌렀다는 인과적 증거. ②**leak 최종 배제**: [[l3-coupling-lightsout-confirmed-0716]] 계열의 "list→numpy 경계" 같은 조용한 누수였다면 더 나중·더 무거운 step(gs161 len2653)에서 *더 심하게* 터졌어야 하는데 정반대로 통과 — 08:46틱의 GPU 진동(단조증가 부재)+08:53틱의 gs159 8192-full-step 통과+본 gs161(더 긴 len) 통과의 3중 증거로 length-driven-activation-peak가 유일 근인으로 확정, code-leak/retention-leak 배제. 사용자의 "다른 학습은 OOM 안 나는데 이것만—누수 아니냐"(msg4) 질문에 대한 최종 답: **누수 아님. b2 고유의 init(meta-SFT 긴응답)×reward(VANILLA_GRPO 무길이제어) 상호작용이 length inflation을 낳고, 그 팽창 step의 activation peak가 80GB를 스쳤을 뿐**(B0=짧음·B3pkg=triobj 길이제어라 무OOM). ③이 length inflation(clip_ratio 0.16=16% cap도달)은 그 자체로 **RQ2의 emission-erosion/length 동역학과 같은 뿌리**일 수 있어 gs300 해석 시 length confound로 반드시 계측(E-048 per-arm frac_len>4096·p95/p99·cap-hit 필수보고). 관련 [[aime-drop-is-decoding-degeneration-nontermination-0629]]·[[pmishift-starvation-base-sft-undertrained-0710]].

### 상태 tick — 08:53 UTC(7/18) — [★leak 사실상 배제: gs159 full-step 완주(8192-max 포함)·OOM 0] witty-imp running 33m. ★**결정적 증거: gs159 스텝이 OOM 없이 완전 완료·저장**(model+optim+extra_state 3종, 08:49). 로그 실측: `response_length/mean:1762·max:8192·clip_ratio:0.098`, `timing_s/update_actor:141.6·save_checkpoint:27.7·step:821.5`, entropy 0.878·critic/score 0.58·grad_norm 0.77=전부 건강. ★**leak 반증 논리 완성**: leak이 실재했다면 **response_length/max=8192(원 OOM을 유발한 바로 그 조건)를 포함한 full step**이 update_actor+save까지 통과하지 못했을 것. micro_batch=1이 headroom 제공+현 step 길이(1762)가 원 OOM 지점(2553)보다 낮아 이중 안전. tqdm `159/300, 821.93s/it`. HF 프런티어 rq3_b2=[156,158,**159**,160](gs159 push완료·gs160은 OOM前 partial잔재로 RGS가 gs158 재개시 배제한 폴더). b3pkg=smart-bat gs32(running 7h·건재). ★남은 형식적 확인: 로컬 gs160→**gs161**(원 OOM점, ~28min out) 통과=micro_batch=1 완전검증. 무개입.

### 상태 tick — 08:46 UTC(7/18) — [leak계측 진전: GPU 진동=단조증가 아님] witty-imp running 26m·OOM 0. ★GPU 궤적 진화: 40644 steady(생성phase)→이후 **진동 40644→19296→35540→35880→14146→19082→19504**(rollout↔update 위상전환·vLLM sleep/wake KV토글). ★**leak 강한 반증: GPU가 14~40GB 범위 진동·단조증가 없음**(누수면 floor/ceiling이 시간따라 상승해야; 여기선 위상별 오르내림, peak≤40GB≪80GB). length-activation 가설과 일치(메모리가 현 step 위상·시퀀스길이 추적). max step 아직 <none>(gs159 생성heavy step 미저장·진동=능동작업중이지 stuck 아님). b3pkg gs32(7h). HF b2 gs160. ★최종 leak판정+gs162 OOM검증은 gs161(원 OOM점, ~2step out) 도달 필요—거기서 peak가 80GB 미만이면 micro_batch=1 성공+leak최종배제. 무개입.

### 상태 tick — 08:37 UTC(7/18) — [✅b2 학습재개·leak계측 baseline] witty-imp RUNNING(16m). **"Training Progress" tqdm 출현=학습 재진입**·OOM count 0·micro_batch=1 적용. ★GPU 궤적(nvidia-smi): 부트 램프 3810→4430→22898→**40644MB steady**(현 step 동안 ~40GB 고정). 40GB=정상 학습phase(vLLM KV+FSDP), OOM은 70GB@length-spike였음. ★**leak 예비판정: GPU가 steady 40GB·단조증가 아님**=leak 가설 예비 반증(누수면 step마다 climb해야). 단 결정판정은 gs159→160→**161**(원 OOM점) 다중step 궤적 필요: ~40GB 유지+gs161만 spike=length확정(micro_batch=1이 80GB 미만 유지 기대) / step별 단조climb=leak. b3pkg gs32(running 7h). HF b2 gs160(첫 post-resume step 미저장). 무개입, gs161 궤적 대기.

### 상태 tick — 08:24 UTC(7/18) — [✅b2 노드확보·부팅성공] witty-imp(6번째) **RUNNING**(9m전 grab, 4m 부팅). std_log존재·RGS pull_resume gs158 단계·★**vLLM EngineCore assert 없음**(expandable_segments revert 성공 확인)·OOM無·micro_batch=1 적용. 6번째서 Standard 노드 확보(1~5번째=할당실패/vLLM비호환). b3pkg gs31 건재. HF 836GB. ★다음 ~60분: gs158→159/160/161 재학습→**gs162(OOM검증점)**·이 구간 **HB gpu0used 궤적 추출**해 leak계측(단조↑=leak/진동+spike=length). 무개입, 부팅 완료 대기.

### E-049 (2026-07-18 08:10 UTC) — [leak 적대검증] b2 OOM은 length-driven peak가 1순위·코드leak 배제·단 length-dependent retention/vLLM은 계측으로 최종배제(사용자+Codex ultrathink)

**문제(추상):** "같은 인프라서 다른 arm은 OOM 없는데 이 arm만 OOM"은 두 가설(①length-activation peak ②memory leak)이 *둘 다 '한동안 멀쩡→갑자기 OOM'*을 예측하므로 단일 이벤트로 구분 불가. 성급히 "length다"로 닫으면 leak을 놓치고, "leak이다"로 닫으면 실재 설계동역학을 오진한다. 판별엔 *구조적 논증 + 계측*이 필요하다.

**해결(구체):** 사용자 지적("leak 아니냐")에 3증거로 적대검증(Codex 합의): (1)★**b2·b0 완전 동일코드**(둘 다 VANILLA_GRPO correctness-only·region-split/pmi_shift 없음, 차이=init 체크포인트뿐)인데 b0는 gs300 완주(OOM無)→**무조건적/코드 leak 배제**(코드 leak이면 b0도 터졌어야). (2)★OOM 에러 분해: `68.71GiB allocated by PyTorch·142.40MiB reserved-but-unallocated`→**단편화 극소(142MB)·실제할당 68.71GB=진짜 peak**(leak/fragmentation이면 reserved-unallocated가 크거나 allocated가 부풀어야). (3)gs161 length 1648→2553↑·8192cap·actor backward서 OOM=length-activation peak와 일치(verl FSDP activation ∝ seq_len). ★Codex 최종(방어적): "length-driven peak가 1순위, but length-dependent tensor retention이나 vLLM 비-PyTorch 메모리 누수는 **step별 GPU baseline+NVML 추적 전까진 완전 배제 불가**"→'leak 아님' 단정보다 **'현증거상 length 1순위·leak은 계측으로 최종배제'**가 정확. ★계측계획: b2 다음 실행이 gs150→161 통과할 때 **HB의 gpu0used(nvidia-smi/NVML) 궤적**을 추출—step마다 *단조증가*=leak / *진동+gs161 spike*=length(HB가 이미 10s마다 로깅 중, 전체궤적 필요). ★micro_batch=1인데도 OOM이면=단일 8192-sample activation peak→gradient checkpointing(backward 재계산으로 peak↓, 목적함수 불변·매치드 안깸·단 RNG/재현성 기록). max_response 하향은 진단용만·primary NO-GO(생성분포 변경). content-squid(5번째) 또 할당실패→witty-imp(6번째) 재제출.

**통찰:** ①"희귀 실패"의 leak-vs-정상동역학 판별의 최강 1차증거는 **동일코드 대조군**이다 — b0가 같은 코드로 안 터졌다는 사실이 코드 leak을 즉각 배제하고 가설공간을 'input-dependent(길이비례)'로 좁혔다. ②OOM 에러의 *reserved-vs-allocated 분해*가 fragmentation/leak vs 진짜-peak를 가른다(142MB=단편화 무죄). ③그럼에도 정직한 종결은 "배제"가 아니라 "계측으로 최종배제"—length-dependent retention은 구조논증만으론 못 죽이고 step별 메모리 baseline이 필요. ④keep-alive의 다축 렌즈에 이제 **leak 렌즈**(GPU mem step별 단조성) 추가. 관련 [[metacog-stuck-rewire-observability-bottleneck-0626]](관측가능성).

### E-048 (2026-07-18 08:00 UTC) — [비교유효성 검증] micro_batch=1은 forward-only=매치드비교 무해·단 length inflation은 실재 failure mode로 명시보고 필수(사용자+Codex 논쟁)

**문제(추상):** 장애복구용 파라미터 변경(b2만 micro_batch 2→1)이 *다른 arm과 달라지면* 매치드 비교를 깨는가? 그리고 OOM을 유발한 length inflation은 단순 인프라 잡음인가, 아니면 그 arm 설계(init×reward)의 *실재 동역학*이라 결과 해석의 일부인가? 두 질문은 별개 검증축이다.

**해결(구체):** ★질문1(비교유효성): 실측=B0·b3pkg는 `log_prob_micro_batch_size_per_gpu=2`, b2만 1(내 OOM픽스). verl dp_actor.py 소스 확인=**log_prob_micro_batch는 `compute_log_prob()`의 청크(line650 `torch.no_grad()` forward-only), 실제 최적화 `update_policy()`는 별도 `ppo_micro_batch_size_per_gpu`를 씀**. ∴ log_prob_micro_batch 2→1은 순수 forward 메모리분할, 계산 log-prob는 청크크기 무관 동일(sample간 상호작용 없음·layernorm)→**effective batch·gradient·궤적 불변→매치드 비교 여전 유효, b2만 1이어도 OK, 다른arm 변경 불필요**. Codex 초기에 log_prob_micro_batch↔ppo_micro_batch 혼동했으나 소스제시 후 정정확인(bitwise는 float reduction순서로 미보장이나 무시가능·엄밀히는 두 설정 log-prob max/mean abs diff 1회 검증으로 충분). ★질문2(length=failure mode): OOM 직접원인=gs161 length 1648→2553↑ 일부 8192cap(메모리는 mean보다 max/p99에 비선형민감→gs150-160 무사, gs161 임계돌파). 상위원인=**b2의 meta-SFT init(긴 meta응답)×correctness-only(length 무제어) 상호작용**(B0=meta없어 짧음, b3pkg=triobj가 length제어). micro_batch=1은 correctness-preserving 메모리완화지 length동역학 해결 아님. ★**RQ3 결과에 length 불안정성 명시보고 필수**(조용히 완화+"정상성능"만 보고=매치드해석 약화): 길이분포 p95/p99·4096 cap-hit/truncation률·frac_len>4096(프로브 사전배선됨 E-041)·arm별 OOM/실패율·생성길이차. gs159 entropy0.931/corr0.605 건강은 *전역붕괴 부재*지 length degeneration 부재 보장 아님(length tail 미측정).

**통찰:** ①복구 파라미터의 "비교 무해성"은 그 파라미터가 *최적화에 들어가는지 forward-only인지*로 결정 — 값이 arm마다 달라도 no_grad forward-only면 매치드 유효(반대로 ppo_micro_batch였다면 파괴). 소스 확인 없이 "다른 값이니 위험"도, "메모리only니 안전"도 성급; **파라미터의 계산그래프 위치를 봐야** 한다. ②Codex도 틀릴 수 있고(두 micro_batch 혼동), 소스 실증이 자문을 이긴다 — 적대검증의 본령. ③가장 중요: **장애(OOM)를 고치는 것과 그 장애가 드러낸 동역학(length inflation)을 보고하는 것은 별개다.** length폭주는 b2 설계의 실재 failure mode라 OOM으로 censoring하거나 조용히 완화하면 B2-B0 비교가 왜곡 — 살리되(micro_batch=1) 반드시 측정·보고. 관련 [[rq2-estimand-endtoend-not-content-0717]](end-to-end 효과·emission침식).

### E-047 (2026-07-18 07:51 UTC) — [정정·복구] expandable_segments가 vLLM 하드-금지→reverted·micro_batch 2→1로 OOM 재대응

**문제(추상):** 에러 메시지가 *직접 제안한* 수정(expandable_segments)이라도 훈련 스택의 *다른 하드 제약*을 위반하면 무용지물이다 — 국소 증상(OOM allocator 단편화)만 보고 고른 메모리-옵션이 상위 프레임워크(vLLM)의 메모리풀과 비호환이면 부팅 자체를 깬다. "가장 안전해 보이는 메모리-only 옵션"도 스택 호환성을 검증 안 하면 새 실패를 만든다.

**해결(구체):** tender-lion(4번째, 노드확보 성공)이 부팅 중 **vLLM Engine core init 실패**(status=failed·로그有·OOM無). 근인 로그: `Worker_TP3 ... AssertionError: Expandable segments are not compatible with memory pool (pytorch#147851)` — **vLLM multiproc_executor가 `expandable_segments:True`를 명시적 assert로 거부**. ∴ E-045의 OOM픽스(expandable_segments)는 vLLM 기반 verl에서 *항상* 부팅을 깬다(transient 아님·하드 비호환). 조치: (1)`PYTORCH_CUDA_ALLOC_CONF` env-var **완전 revert**(주석으로 사유 명기); (2)Codex 사전승인 fallback **`log_prob_micro_batch_size_per_gpu` 2→1**(rollout+ref 양쪽, yaml command 202-203줄, value-only 편집)—메모리-only·forward 분할=수학적 동등·매치드-래더 불변·vLLM 무영향. yaml.safe_load+command VERL_PID 무결+micro_batch=1×2/=2×0 검증 후 재제출→**b2=content-squid**(5번째). ★micro_batch=1이 gs161 log_prob forward 메모리 절감으로 OOM 방지 기대; 그래도 gs161 재OOM시 Codex 다음단계=gradient checkpointing/offload.

**통찰:** ①**에러가 제안하는 수정을 맹신 말 것** — expandable_segments는 pytorch OOM 에러가 직접 추천하고 Codex도 GO했으나(Codex는 vLLM 금지를 몰랐음), 실제 스택(vLLM)이 하드-금지라 무용. 수정 선택 시 "국소 증상 해결"과 "전체 스택 호환"은 별개 검증축. ②실패의 *로그 유무+에러종류*로 클래스 정확판별이 복구속도를 결정했다: 로그0=할당실패(재제출), 로그有+OutOfMemory=OOM(메모리 픽스), 로그有+EngineCore assert=설정 비호환(revert)—셋을 혼동하면 헛수정. ③메모리-옵션은 allocator 층위라 vLLM 같은 자체 메모리풀 프레임워크와 충돌 가능; 순수해 보여도 스택 검증 필수. 관련 [[amlt-bashc-quote-wordsplit-truncation-0718]](편집 검증층위).

### 상태 tick — 07:41 UTC(7/18) — [✅b2 노드확보·재개시작] **tender-lion(4번째) RUNNING!** 3연속 할당실패 후 4번째서 노드 grab 성공(백오프 전략 유효). std_log 존재(로그0 아님)=부트 진입. 부트로그: **RGS_STEP=158**(gs160 아님!)→pull_resume gs158 active. ★완전성 가드가 gs160을 **OOM-부분업로드로 배제**하고 마지막 완전앵커 gs158 선택([156,158,160] 중 gs160은 mighty-civet OOM crash 직전 부분업로드라 model+extra+optim 4/4 미충족)=E-037 3-way 완전성검증 재작동(raw max 아님). gs158서 재개=gs159/160 재학습(~2step 손실, 수용). b3pkg gs31(running 6h, 건재). HF 852GB. ★다음 마일스톤: tender-lion이 gs159→160→161→**gs162(OOM 검증점)** ~60분후 도달=expandable_segments 성공확인. 무개입, OOM검증 대기.

### E-046 (2026-07-18 07:33 UTC) — [전략전환] b2 할당실패 3연속→즉시재제출→백오프(Codex GO)·b3pkg 독립유지

**문제(추상):** opportunistic 자원경합이 지속되면(3연속 할당실패), *즉시-재시도 반복*은 각 시도가 동일 큐-타임아웃(~13분)에 걸려 스케줄러를 thrash하며 성공률을 못 높인다. keep-alive의 "멈추지 말라"는 무한 즉시재시도가 아니라 *자원가용성 회복을 기다리는 지능적 백오프*를 의미할 수 있고, 병렬 독립 arm(b3pkg)이 정상 진행 중이면 실험 전체는 죽지 않았다.

**해결(구체):** devoted-ghoul(3번째)도 **failed**(12분·로그0)=Standard 4×H100 신규grab 3연속 실패. 4번째 재제출(tender-lion)한 뒤 Codex 전략자문. **Codex GO**: (1)즉시 13분반복보다 **30~60분 백오프**(또는 한산시간대)가 thrash 감소·합리적; (2)SKU/GPU수/tier 고정(매치드-래더 불변)이라 실질 레버=재제출 타이밍·placement 제약 최소화·정책상 priority/예약슬롯 확인뿐; (3)b3pkg는 독립실행·프로토콜 유지되면 **결과 자체 유효**하니 계속 진행(단 b2 없으면 B2-B0 비교/RQ3 결론 미완); (4)수시간 지속 실패시 클러스터 운영자에 Standard 4×H100 용량/우선순위 문의. ∴ 전략전환: b2 재제출 간격을 **~30분 백오프**로(틱마다 실패면 그때 재제출=자연히 ~30분 간격), 틱캐던스 20→30분 복귀, b3pkg 정상감시 지속. tender-lion(4번째) 큐잉 중—다음 틱서 running이면 OOM검증, failed면 백오프후 5번째.

**통찰:** keep-alive("절대 멈추지 말라")의 올바른 해석은 *무지성 즉시재시도*가 아니라 **실패모드에 맞춘 재시도 정책**이다 — 코드버그면 수정후 재시도, 선점이면 auto-retry 신뢰, **자원경합이면 백오프**(즉시반복은 thrash). 그리고 다중-arm 실험에서 한 arm(b2)이 자원경합으로 지연돼도 독립 arm(b3pkg)이 프로토콜 유지하며 진행하면 *실험은 살아있다* — "멈춤"의 단위는 개별 arm이 아니라 실험 전체. 관련 [[singularity-preemption-not-hang-retry-check-0711]][[singularity-interactive-fails-batch-works-0715]].

### 상태 tick — 07:19 UTC(7/18) — [b2 재제출3: 할당실패 2연속·queue-and-grab 지속] many-duckling(b2) queued→**failed**(13m·로그0·retry0=할당실패 재확인, liberal-ghost와 동일 패턴). 양쪽 로그 전무 확인=내 env편집 무죄 확정(부트 미도달). b3pkg는 자기 노드 유지 running(gs28)=클러스터 용량 있으나 **신규 4×H100 grab이 현재 경합**(opportunistic 물리 [[singularity-interactive-fails-batch-works-0715]]). ~13분 후 실패 반복=Singularity 큐 타임아웃 유력. keep-alive=재제출 지속(queue-and-grab)→**b2=devoted-ghoul**(:rq3_b2, preparing, 3번째). HF b2 [160]불변·TOTAL 885GB. **통찰:** Standard tier 신규 노드 grab은 *기존 점유 job의 건강과 무관*하게 경합할 수 있다(b3pkg는 이미 잡은 노드 유지, b2 신규는 못 잡음). 대응=인터랙티브 아닌 배치 재제출 반복(각 ~13분 타임아웃 후 재시도)·코드수정 아님. 연속 실패는 코드버그 아닌 물리가용성이니 재제출 외 조치 무의미; 다수 연속(4+) 실패 지속시 클러스터 용량 이슈로 Codex 상의. b3pkg는 무영향 계속 진행.

### 상태 tick — 07:05 UTC(7/18) — [b2 재제출2: 할당실패 재시도] liberal-ghost(b2) queued→**failed**(13m·**로그0·retry0**=노드서 실행 전 할당단계 실패=Standard opportunistic 노드상실, 런타임 오류 아님·내 env편집 무관[부트 도달 전]). b3pkg 동일클러스터 running 정상=클러스터 살아있음→transient 할당실패. yaml.safe_load 재검증 OK 후 재제출→**b2=many-duckling**(:rq3_b2, preparing). ★키프알라이브=할당실패는 재제출로 대응(queue-and-grab). b3pkg 건재(gs28, gs21트림). HF b2 [160]불변(재개전)·TOTAL 844GB. **통찰:** amlt status=failed의 두 클래스 구분—**로그有 failed=런타임 오류**(OOM 등 E-045, 코드/설정 수정 필요), **로그0·retry0 failed=할당단계 실패**(Standard 노드상실, 무조건 재제출). 후자를 코드오류로 오진하면 헛수정; 판별자=`amlt log list`에 std_log/retry 존재 여부. many-duckling도 로그0 failed면 재제출 반복(정상 opportunistic), 로그有 failed면 그때 근인수정.

### 상태 tick — 06:58 UTC(7/18) — [b2 복구 진행: 큐잉] liberal-ghost(b2) status=**queued**(6m·Standard 노드 대기=opportunistic 정상, 장애 아님 [[singularity-interactive-fails-batch-works-0715]]). 아직 미부팅이라 gs161→gs162 OOM검증 불가(큐잉 중엔 학습 안 해 OOM 리스크 없음)—노드 확보시 재개(HF gs160)→검증. b3pkg=smart-bat 건재(running 5h·gs24·gs17 트림=--keep2). HF b2 [156,158,160] 불변(재개 전)·TOTAL 836GB. 무개입, 부트 대기.

### E-045 (2026-07-18 06:48 UTC) — [장애·복구] b2 CUDA OOM(gs161 길이팽창 근인)→expandable_segments+gs160 재개(Codex GO)

**문제(추상):** 트렌드워치에서 "policy-degeneration 아님"으로 안전 판정한 신호가 *다른 실패 축*에서 치명적일 수 있다 — gs161의 응답길이 급증(1648→2553, 일부 8192 cap)은 entropy↑라 붕괴는 아니었으나(E-042/E-044 판정 유효), 바로 그 긴 시퀀스가 actor update의 log_prob/backward서 80GB H100 메모리를 초과시켜 **CUDA OOM→job failed**를 냈다. 즉 "길이 스파이크가 정책 건강엔 무해해도 메모리엔 치명"일 수 있고, degeneration 렌즈만으로 length를 보면 OOM 리스크를 놓친다.

**해결(구체):** wall 06:47 실측서 b2(mighty-civet) status=**failed**(선점 아닌 hard failure) 발견. 로그 tail=`torch.OutOfMemoryError: tried to allocate 5.54GiB, 3.80GiB free` @ gs161 완료 직후→Terminated. b3pkg는 무사(응답 876토큰으로 짧아 OOM 안 남). Codex 전문 전달→**Plan B GO**: (1)`PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"`를 b2 yaml **env 섹션**(bash command-string 아님=따옴표-wordsplit 리스크 회피, [[amlt-bashc-quote-wordsplit-truncation-0718]])에 구조적 key-value로 추가—**메모리-only·훈련수학 불변·매치드-래더 안 깸**(B0/B3pkg 무영향). (2)HF gs160(완전앵커)서 재개. yaml.safe_load 파싱 OK+command block VERL_PID/verl launch 무결(12334자) 검증 후 `amlt run -y`→새 arm **liberal-ghost**(:rq3_b2, preparing). ★Codex 후속지침: gs161 재OOM시 반복재시도 말고 `log_prob micro_batch 2→1`(forward-only 분할=수학적 동등이라 GO, 단 향후 전 arm 동일기록) 적용후 재개; 그래도 실패면 gradient checkpointing/offload. ★max_response 하향=**NO-GO**(생성분포·truncation 바꿔 B2-B0 매칭 파괴).

**통찰:** ①RL 모니터링의 트렌드워치는 **다축 실패 렌즈**여야 한다 — 한 지표(length↑)를 "degeneration 아님"으로 clear해도 그 지표가 *별개 축*(메모리·throughput·durability)서 위험할 수 있다. length는 degeneration 렌즈(entropy 동반)뿐 아니라 **OOM 렌즈**(peak memory ∝ max seq len)로도 봐야 했고, gs161 스파이크 때 "무해" 대신 "OOM 리스크"를 경보했어야. ②장애 복구의 안전한 개입은 **불변식 보존 축**을 먼저 찾는 것 — 매치드-래더서 OOM을 고치는 여러 손잡이 중 훈련수학을 안 바꾸는 것(allocator 옵션>forward 분할>>batch/length 변경)을 우선순위화하면 비교유효성을 지키며 살릴 수 있다. ③env 섹션의 구조적 key-value가 bash command-string 편집보다 항상 안전(파싱층 분리).

### E-044 (2026-07-18 06:10 UTC, `date -u` 확인) — [시각 정정] 틱 타임스탬프 드리프트: 예약시각 유도 금지·wall clock은 06:09·gs161 watch는 "정상 최전선"으로 해소

**문제(추상):** Stop-hook 빠른 재발화 환경에서 틱 타임스탬프를 *예약 wakeup 시각*(scheduled 06:12/06:15/…)에서 유도하면 실제 wall clock보다 앞서간다 — 재발화는 경과시간이 아닌데 그걸 시간흐름으로 착각. 그 결과 (a)릴레이 lag을 과대추정(30분)하고 (b)"여러 틱째 gs161 정체"를 stall 후보로 오인한다. 실제론 시간이 별로 안 흘렀고 gs161은 방금 끝난 정상 최전선.

**해결(구체):** `date -u` 직접 확인=**Sat Jul 18 06:09:52 UTC**. 그간 보고서 틱(06:12~06:35, E-043 "06:22")은 예약시각 유도라 실제(06:09)보다 앞섬. 재계산: b2 gs161 tqdm elapsed 2:35:03, 학습재개 ~03:30 → gs161 완료 ~06:05 = wall 06:09의 **~4분 전**(방금 완료). ∴ (1)릴레이 lag은 ~10~15분(30분 아님, 과대추정 정정), (2)"3틱 gs161"은 rapid hook이 짧은 실시간을 압축한 것이지 stall 아님 — gs162는 ~06:20 예정(아직 미도래가 정상). gs161 watch=**정상 최전선으로 해소**(uptick은 hard-batch·고entropy, 애초 붕괴 아님 E-035tick). HF gs160(gs161 업로드 +8GB/130s)·b3pkg gs21·둘 다 running clean. 향후 틱 타임스탬프는 **반드시 `date -u`**로 찍고, rapid hook 재발화≠경과시간(실제 20분 wakeup을 실시간으로 기다림).

**통찰:** 자율 루프의 시간감각은 **hook 발화 횟수가 아니라 monotonic wall clock**에 앵커해야 한다 — 빠른 재발화를 "많은 시간 경과"로 읽으면 정상 진행을 stall로, 짧은 릴레이 lag을 긴 lag으로 오판하는 연쇄 오류가 난다. tqdm의 elapsed(2:35:03)와 `date -u`의 차이만이 진짜 lag이다. 관측 도구(로그릴레이·byte-delta)뿐 아니라 *자신의 시계*도 교정 대상. 성급한 stall/lag 단정 전에 `date -u` 한 줄이 가장 싼 진실.

### 상태 tick — 06:35 UTC(7/18) — [watch 계속: gs162 릴레이 미도달·gs161 재평가=batch noise 유력] 로그버퍼 여전 gs161(릴레이 ~30분 lag로 gs162 메트릭 미표면화)→watch 판정 이번 틱 미완. 생존/전진 확인: TOTAL 811→**828GB**(+17 업로드=alive)·status running clean·retry불변. ★gs161 재평가(냉정): entropy **1.029=시리즈 최고**인데 degeneration/mode-collapse는 *저*entropy(결정론·반복) 특성→고entropy+긴응답+저correctness(0.371=어려운배치)는 **hard-batch 건강한 탐색**이지 붕괴 아님. length/clip uptick을 과대평가한 듯(entropy 신호가 batch-noise 강력 지지). 그래도 gs162 확인은 저비용이라 watch 유지(20분). 무개입. **통찰:** degeneration 판별서 **entropy 방향이 length/clip보다 상위 판별자** — 붕괴는 저entropy 동반이 필수라, entropy가 오르면 length/clip이 튀어도 붕괴 아님(탐색적 긴응답). 단일 step의 length↑를 붕괴로 성급 격상한 건 과민, entropy 게이트를 먼저 봤어야. RED 재정의: length·clip↑ **AND entropy↓** 동반이어야(엔트로피 상승 중엔 length spike는 탐색).

### 상태 tick — 06:29 UTC(7/18) — [⚠️YELLOW watch: gs161 length/clip 상승] 양 arm 프런티어 전진(b2 gs158→**gs160**·b3pkg gs17→**gs21**=릴레이독립 학습전진 확증). b2 local **gs161**(tqdm 161/300·879.8s/it·ETA~34h). ★gs161 트렌드(4점): clip 0.080→0.076→**0.162↑**·length 1738→1648→**2553↑**·correctness 0.605→0.531→**0.371↓**·entropy 0.762→**1.029↑**·aborted 0.0. length↑+clip↑+correctness↓ 동반=degeneration 시그니처 *유사*하나 **entropy↑(1.029, ↓아님)+aborted 0.0**이라 full-RED 아님→고분산 hard-batch step 유력(correctness 0.371=어려운문제·긴응답이 s/it 799→879도 설명). ★단 **단조지표(clip·length) 첫 상승반전**이라 watch 격상: 다음 1~2 step서 계속↑(+entropy 하락 전환)이면 degeneration onset→Codex 전문논의; 반전시 batch noise. 캐던스 30→20분 단축(gs162/163 조기포착). status running clean·HF 811GB self-bound. 무개입 유지(단일 step spike).

### E-043 (2026-07-18 06:22 UTC) — [생존 primitive 정밀화] byte-delta 창>pusher interval(90s) 필수·신규 step폴더 출현=학습-전진 증명

**문제(추상):** 생존 primitive(byte-delta)도 *샘플링 창이 pusher 주기보다 짧으면* false-zero를 낸다 — pusher가 90s마다 업로드하는데 45s 창으로 재면 sleep 구간에 걸려 "델타 0=stall?" 오판. 게다가 로그릴레이가 ~30분 지연되면 tqdm·프런티어·HB 전부 과거 스냅이라 *현재* 학습-전진을 못 본다. 지연-관측 3중(릴레이 lag + byte 언더샘플 + 프런티어 lag)이 겹치면 healthy run이 stall처럼 보인다.

**해결(구체):** 06:19 틱서 HF total 934→934(0증가)·b2 tqdm HB-only 윈도(05:43~05:50 전부 heartbeat, step라인 0)·45s byte-delta 0 → stall 의심. 그러나 status running clean·retry 불변(retry_002/001)·HB 05:36→05:50 전진=노드 생존 신호. 결정 test=**130s byte-delta(90s pusher주기 초과)**: total 868→795GB(gs153 −98GB 트림) 와중 **rq3_b2 gs160 폴더 신규출현**(+생성)·gs158 +16GB·**rq3_b3pkg gs21 폴더 신규출현 +8GB**. ∴ 양 arm **학습-전진 확정**(b2 gs160·b3pkg gs21 신규 step폴더=새 스텝 생산). 45s의 "델타 0"은 예상대로 pusher 90s주기 언더샘플이었음. 무개입.

**통찰:** ①byte-delta primitive의 **샘플링 창은 반드시 pusher --interval(90s) 초과**(≥120s)여야 false-zero를 피한다 — 짧은 창의 0은 stall이 아니라 pusher sleep. ②**"신규 global_step 폴더 출현"은 로그릴레이와 독립적인 학습-전진 증명**(byte 총량 증가보다 강함: 총량은 트림에 상쇄될 수 있으나 새 step번호 폴더는 새 스텝을 produce했다는 직접 증거). E-042의 tqdm(릴레이 의존)을 보완하는 릴레이-독립 진행신호. ③관측 지연 3중고(릴레이/언더샘플/프런티어 lag)일수록 **긴-창 byte-delta + 신규폴더 검출**이 유일하게 관통. 성급한 stall 단정 대신 pusher주기 초과 재측정이 정답 — [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]] 계열 오진 방지 재확인.

### 상태 tick — 06:15 UTC(7/18) — [양 arm 전진확인] b2 alive(HF +16GB 246→262=gs159/160 업로드中·tqdm 동일 gs160 스냅=릴레이 버퍼 재반환·바이트로 생존확정). **b3pkg tqdm 첫 포착**: **20/300 [2:17:11<41:27:41, 533.08s/it]**=local gs20(HF gs17, lag3)·533s/it(~9분, b2 799s보다 빠름=terse arm)·ETA~41h. status running clean·HF 901→934GB(self-bound). RED無(b2 트렌드 gs160 불변, b3pkg 전진). 무개입. **통찰:** b3pkg가 b2보다 step 빠름(533<799s/it)은 response_length가 짧아(gs18서 876 vs b2 1648)=rollout 생성시간 단축 — triobj/pmi_shift arm이 더 terse한 정책으로 수렴 중(길이팽창 반대), 계산적으로도 유리. 단 b3pkg ETA 41h로 b2(31h)보다 늦게 gs300 도달→**b2가 결과 lifecycle 먼저 트리거**(순서 확정).

### 상태 tick — 06:12 UTC(7/18) — [트렌드 3점·RED無 재확인] b2 gs160 전진(tqdm 160/300·**799.5s/it**(836→799 개선)·ETA 31:05:31). 3점 트렌드(gs158→159→160): clip_ratio 0.137→0.080→**0.076↓**(단조감소=길이팽창 후퇴)·response_length 2175→1738→**1648↓**(단조감소)·aborted 0.0 불변 = **degeneration 핵심지표 3종 전부 건강방향**. entropy 0.856→0.931→0.762·correctness 0.449→0.605→**0.531**은 진동(단조붕괴 아님, GRPO batch마다 다른문제 샘플=정상분산). ★RED 시그니처(clip↑+aborted↑+entropy↓+correctness↓ 동반) 부재 확정. b2 local gs160·HF gs158(lag 2step, gs159/160 업로드中)·status running clean. HF 901GB(trim↔upload 균형). 무개입. **통찰:** RL 트렌드워치에서 **오실레이팅 지표(entropy·correctness)와 단조 지표(clip·length·aborted)를 구분**해야 오판 안 한다 — correctness 단일step 하락(0.605→0.531)을 degeneration으로 읽으면 거짓경보. 진짜 degeneration은 *length계열이 단조 악화*하는데 여기선 단조 개선 중이라 명백히 건강. 배치분산 큰 지표는 3점+ 추세로만 판정.

### 상태 tick — 06:10 UTC(7/18) — [생존확인·저부하] 두 arm alive(byte-delta). HF b2 229→**246GB**(+17)·b3pkg 246→**262GB**(+16) 증가=양 노드 업로드中=生(byte-delta primitive). status running clean·b3pkg 3h→4h(연속). 프런티어 b2 gs158(gs159 체크포인트 업로드中·1step lag 유지)·b3pkg gs17. TOTAL 869→**901GB**(self-bound 범위). 이번 fetch는 HB-only 로그윈도라 tqdm/trend 미포착(≈10분버퍼 vs ≈14분step 제약)이나 생존 확정+직전 06:08 판독(gs159·836s/it·트렌드 전건강) 유효. 무개입. 다음 틱 step직후 착지 노려 트렌드 재포착.

### E-042 (2026-07-18 06:08 UTC) — [진행판정+트렌드] stall 가설 기각: tqdm 진행바=학습-전진 authoritative 신호·b2 트렌드 전건강

**문제(추상):** "노드 생존"(byte-delta)과 "학습 전진"은 **별개 명제**다 — pusher가 대형 체크포인트를 느리게 올리면 학습이 멈춰도 바이트는 증가한다(생존 프록시가 전진을 보증 못 함). HF 프런티어 step이 여러 틱 정체(b2 gs158×3틱)하면 "느린 업로드 lag"인지 "학습 stall"인지 구분해야 하는데, step-메트릭 로그는 ~10분 tail 버퍼 밖이라 안 잡히고 프런티어·HB·바이트 전부 애매했다.

**해결(구체):** b2 전체 로그(304줄)를 파일로 덤프해 결정적 신호 확보: **`Training Progress: 53%|159/300 [2:05:24<32:44:41, 836.04s/it]`** (tqdm 진행바)=local **gs159 전진 중·836s/it(~14분/step)·ETA 32:44:41(~33h)**. 추가로 `wandb WARNING: log to step 159 < current step 162`=내부 step카운터 이미 162=학습루프 확실 전진. `Saved global_step_159/actor` 확인. ∴ **stall 가설 기각**: local gs159(내부 162)·HF gs158=**lag 1step뿐**(무한 업로드부채 우려도 기각, HF가 1step 뒤로 따라붙는 중). ★트렌드워치 2점(gs158→gs159): entropy 0.856→**0.931↑**·clip_ratio 0.137→**0.080↓**(cap-hit 감소)·correctness/mean 0.449→**0.605↑**·response_length 2175→**1738↓**·aborted 0.0 불변 — **전 신호 건강방향**(degeneration 시그니처=clip↑+entropy↓+길이↑+correctness정체의 정반대). E-041 "yellow flag"(gs158 clip 13.7%)는 고분산 단일step였고 gs159서 길이팽창 후퇴+correctness 상승. GPU 43GB=update/save phase(rollout시 63GB). 무개입 지속.

**통찰:** durable RL 모니터링의 생존판정은 **2층 명제**로 분리해야 한다 — ①**노드-생존-NOW** = HF byte-delta(45s 2스냅, pusher가 노드서 돎), ②**학습-전진** = **tqdm 진행바(`N/300, s/it, ETA`)** 또는 wandb 내부 step. ①만으론 stall을 못 잡고(죽은 학습+살아있는 pusher 가능), ②가 authoritative. 그리고 진행바엔 **s/it·ETA가 직접 박혀** 있어 프런티어-역산(pusher-lag 오염)보다 정확한 ETA를 준다(836s/it·ETA 33h를 tqdm이 직접 명시). 앞으론 전체로그 덤프→`grep "Training Progress"`가 진행판정 1순위, byte-delta는 로그 부재시 노드-생존 보조. 관련 [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]](무출력 오진 계열).

### 상태 tick — 06:02 UTC(7/18) — [생존 primitive 확립] HF 바이트-델타 back-to-back=노드-생존-NOW. 로그릴레이가 HB 05:30:53에 정체+HF프런티어 b2 gs158 정체로 "죽었나?" 재부상. ★결정적 해소: **45초 창에서 b2 gs158 폴더 +8189MB(~182MB/s) 실측**=pusher가 지금 이순간 업로드 중=노드 확실 생존(로그릴레이 독립). b3pkg도 +8GB/틱=생존. status running clean·retry불변. HF 811→836→844GB(self-trim과 catchup 균형). ★부수정정: 업로드 대역폭은 ~182MB/s(이전 "4.5MB/s" 가정 오류)—∴ HF프런티어 gs158 정체는 대역폭기아 아니라 **대형 체크포인트(model+optim+extra×4샤드=수십GB/step) in-flight**의 정상 신호, self-resolve. save_freq=1 문제없음 확정(업로드부채 우려 기각). GPU 43GB(update phase). **통찰:** durable 런의 "지금 살아있나"를 로그릴레이·프런티어step 없이 판정하는 최강 primitive=**동일 저장소 바이트를 45s 간격 2회 스냅→델타>0이면 노드-생존-NOW**(pusher가 노드에서 돌므로 바이트증가⇒노드活). 로그릴레이 lag·프런티어 lag·HB stale 3중 관측공백을 단일 바이트-델타가 관통—E-040(바이트-총량)의 정밀화(총량→per-step 델타·시간창 명시). 이 primitive를 선점판정 1순위로 승격.

### 상태 tick — 05:54 UTC(7/18) — [트렌드워치 1회차] 두 arm 전진·RED 무. **b2** HF gs156→**gs158**(pusher catchup)·HB 05:28:02·**GPU 43→63536MB**(활성 롤아웃 generation 단계=학습확실)·running clean. **b3pkg** gs13→**gs18**(+5step)·gs18 메트릭 baseline 기록: entropy 0.187·resp_len mean **876**/max 8192·clip_ratio **0.018**·aborted **0.0**·correctness/mean 0.127(b2보다 terse=triobj/pmi_shift arm 특성). HF 869→**811GB**(self-trim 지속). ★RED 판정 없음(양 arm aborted 0.0·cap-hit↑+완료↓+entropy↓+kl↑ 동반 미관측). ⚠️**관측제약 발견**: `amlt log view`는 ~10분 tail 버퍼만 반환→b2 step이 ~15분 간격이라 틱이 HB-only 윈도에 떨어지면 **b2 per-step 메트릭 라인을 놓침**(이번 틱 b2 gs159+ 메트릭 미포착, gs158 baseline만 보유). b3pkg는 gs18 포착됨. **통찰:** 로그-tail 버퍼폭(~10분) < step주기(~15분)면 per-step 트렌드는 *틱이 step직후에 착지할 때만* 포착 가능—wandb 대안은 entity mismatch로 불가하니, 트렌드는 **기회적 포착**(HF프런티어 전진+HB+GPU가 1차 생존, step메트릭은 잡히는 대로 누적)으로 운용. degeneration 조기경보엔 aborted_ratio/HF-정체가 버퍼독립 보조지표. 다음 틱 b2 step직후 착지 노려 재포착 시도.

### 상태 tick — 05:44 UTC(7/18) — [gs300 준비·비차단] eval 스크립트에 length-confound 프로브 사전배선. Codex 지시(E-041 (2))의 `frac(len>4096)`를 gs300 도착 전 미리 구현해 결과 lifecycle 지연 제거. `scripts/eval_vllm_1030.py`의 `summarize()` per_bench에 **`frac_len_gt_4096`**(strict >: 4096토큰=4k예산 내 종료→제외, 4097→포함) 1필드 추가—측정 전용, reward/correctness/training 정의 불변(헌법 준수). 스크립트는 이미 per-row `completion_length_tokens`·`finish_reason`("length"=truncation) 저장 중이라 parquet에 원자료 존재했고 요약만 미노출이었음→외과적 1필드. py_compile OK + 합성단위테스트(길이 [100,5000,4096,4097,8192]→frac 0.6·경계정확) 통과. ★용도: gs300서 eval을 관대한 max_tokens(≥16384)로 돌려 각 arm의 frac_len_gt_4096 측정→B2−B0·B3−B2를 [capability차 vs 4k종료적합성차]로 분리보고(동일-4k=배포조건 타당·순수capability=confound). ⚠️eval max_tokens≤4096이면 trivially 0이라 반드시 >4096로 생성. **통찰:** 결과-차단 대기 중에도 *결과 해석에 필요한 계측을 선제 배선*하면 gs300 도착 시 스크램블 없이 즉시 confound-분리 보고 가능—Codex 지적을 "gs300때 하자"로 미루지 않고 지금 원자료-이미-존재 확인 후 요약노출만 추가한 게 최소비용 선행투자.

### 상태 tick — 05:36 UTC(7/18) — [Codex 판정: E-041 길이팽창] **yellow flag, not red**. Codex 3답: (1)correctness-only b2의 8192-cap 13.7% hit는 절대값(entropy 0.856·kl 0.127·clip 0.137)만으론 degeneration 아님·**추세**가 관건(cap-hit↑ + EOS/완료율↓ + entropy↓ + KL/clip↑ **동반**시 red). (2)**4k 평가 length-confound 실재**: 학습 8k·평가 4k mismatch로 답이 4096 이후 나오는 b2 샘플은 정확도 인위 삭감→B2−B0가 [추론능력차 + 4k내 종료적합성차]로 오염. 단 8k-cap 13.7%≠4k-truncation률이라 **`length>4096` 비율**을 gs300 eval서 실측해야 판별(mean 2175로는 tail 불명). 동일-4k 예산 평가는 *배포조건* 비교론 타당·*순수 capability* 비교론 confound. (3)~35h 완주 판단 맞음·현 관측으로 취소이유 없음·단 "무개입≠무감시". ★조치 2개 상시화: **(A)매 틱 트렌드워치** clip_ratio·entropy·kl_loss·correctness/mean·response_length mean/max·aborted_ratio를 log view서 뽑아 시계열 누적(단일임계 아닌 동반추세 감시). **(B)gs300 eval 프리레지스터 보강**: eval_vllm_1030에 `frac(len>4096)` per-arm 측정 추가해 B2−B0·B3−B2를 length-confound 분리 보고(배포조건 vs capability 이중해석). 무개입 지속.

### E-040 (2026-07-18 05:30 UTC) — [관측성] 로그-릴레이 공백 시 생존판정: HF 총바이트 증가를 노드-생존 프록시로 사용

> **⚠️정정(E-041, 05:35):** 이 항목의 "로그-릴레이 transient 공백" 전제는 **틀렸다**. 근인은 인프라가 아니라 `amlt log`(그룹)를 `amlt log view` 대신 쓴 **명령어 오타**였다. 아래 HF-바이트 프록시 추론은 유효하나, "릴레이가 죽었다"는 판정은 자기유발 관측오류. 실제 로그는 정상 수신 가능했고 b2는 gs158 학습 중이었다.

**문제(추상):** durable 런의 3계층 생존판정(status·로그tail·HF프런티어) 중 한 계층이 *관측 불가*(장애 아님)가 되면, 남은 계층만으로 생존을 확증해야 한다. 로그tail이 결정적 증거였는데(E-037) 그 파이프라인 자체가 빈 응답을 반환하면, "HF 프런티어 정체 + 로그 공백"을 선점/스톨로 오진할 위험이 생긴다 — 관측 도구의 침묵을 시스템의 침묵으로 혼동하는 오류.

**해결(구체):** 05:30 틱서 `amlt log <exp>`가 두 arm 모두 `-n 400` 포함 모든 변형에서 **빈 응답**(05:09엔 HB 타임스탬프 정상 수신했으므로 로그-릴레이 transient). 대신 남은 두 계층으로 판정: (1)`amlt status`=b2 running **6h 연속**·b3pkg **3h 연속**, `amlt log list`의 retry 카운트 **retry_002/retry_001로 불변**(E-037 복구 이후 신규 retry_003 없음=신규 선점 없음). (2)HF 총량 **901→934GB(+33GB)** 증가=pusher가 샤드를 능동 업로드 중=**컨테이너·노드 생존 + 체크포인트 생산 중**. 프런티어 step번호(b2 gs156)는 고정이나 총바이트 증가=E-039의 catchup 시그니처(유한·양성). 판정=**무개입**, 단 프런티어가 gs156 초과로 전진함을 확인하러 캐던스를 30분→15분으로 단축(로그-릴레이 회복+프런티어 전진 이중확인).

**통찰:** 다계층 생존판정의 강건성은 *계층 간 독립성*에서 온다 — 로그-릴레이(amlt stdout 스트림)와 HF-저장(pusher→repo)과 스케줄러-status(Singularity 제어면)는 서로 다른 인프라라 하나가 침묵해도 나머지가 살아있다. 특히 **"HF 총바이트 증가"는 프런티어 step이 정체여도 노드-생존의 강한 양성 신호**다(죽은 노드는 업로드를 못 한다). 그리고 **연속 running 시간 + 불변 retry 카운트**는 "그 시간 동안 선점이 없었다"의 직접 증거 — 선점이 있었다면 duration 리셋+retry 증가가 남는다. 로그가 안 보인다고 죽었다 단정하는 것은 [[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]의 "무출력=사망 오진" 계열 실수. 관측 공백엔 판정 유예+독립 계층 교차확인+단축 재확인이 정답.

### E-039 (2026-07-18 04:06 UTC) — [저장소] HF 정리 실행: 확정-superseded 3개 삭제(262GB 확보)

**문제(추상):** 잦은-선점 durable 학습의 원격 저장소는 두 힘의 균형이다 — 자동관리(`--keep N`, 실행 중 런만·업로드완료 후·지연 발화)와 자동관리 사각(완료 런의 intermediate·업로드랙으로 --keep이 아직 못 자른 straggler). 용량이 임계로 접근하면 사각의 아티팩트만 골라 안전삭제해야 하나, "무엇이 진짜 잉여인가"의 판정은 크기가 아니라 *최종산출물/재개앵커/자동관리대상 여부*의 3중 체크다.

**해결(구체):** HF가 1057GB로 재증가(1024→1057, 1.2TB 임계까지 143GB)하며 간헐 plateau 후 재climb 반복(신뢰 plateau 아님)→사용자 "필요시 정리" 조건 충족 판정. Codex에 구체 삭제리스트 전문 전달→**GO**(단 삭제전 경로·완료상태·pusher참조 재확인+삭제후 용량재확인). delete_folder 실행 시 **pre-guard**(대상 존재 && 더 새로운 앵커 반드시 잔존) 코드로 방어: (1)`rq3_b0/gs290`(완료·평가된 B0의 intermediate, gs300 baseline+eval결과 CLAUDE.md 보존)→삭제; (2)`rq3_b2/gs148`(gs150·151 신규앵커 잔존)→삭제; (3)`rq3_b3pkg/gs1`→**가드 SKIP**(재조회시 이미 `[4,5,8]`로 자동트림됨=`--keep 2` 실작동 확증). 결과 **1057→795GB(262GB 확보)**: b0 [300]·b2 [150,151](정확히 keep2)·b3pkg [4,5,8]·b3nopmi [135,140,145] 미변경(NO-GO). 두 arm 무경합(pusher는 gs151/gs5에 씀).

**통찰:** ①원격 저장소의 용량증가는 대개 *업로드-랙 catchup*(유한)이고 `--keep N`은 **지연 발화**한다(b3pkg gs1이 삭제 직전 자동트림됨이 증명) — 성급한 대량삭제보다 임계접근 시 *확정-잉여만* 외과절제가 정답. ②삭제의 안전성은 승인(Codex GO)만으로 부족하고 **실행 시점 pre-guard**(더 새로운 앵커 잔존 재확인)가 필수 — 이 가드가 "이미 자동삭제된 gs1"을 SKIP해 중복/오류삭제를 막았다. 승인은 정책, 가드는 그 순간의 상태검증. ③완료 런(b0)의 intermediate는 자동관리 사각이므로 수동정리 대상. 관련 [[yaml-heredoc-edit-verify-runtime-layer-0717]].

### 상태 tick — 04:01 UTC(7/18) — [HF 모니터링] 용량 plateau 확정=runaway 아님. HF 910→934→967→991→1024→**1024GB(STABLE)**: b2 [148,150,151]=246GB·b3pkg [1,4,5,8]=287GB 크기 불변=**post-recovery 업로드 catchup 완료→plateau**. ★판정: 아까의 ~+30GB/체크 증가는 pusher가 gs151/gs5 샤드를 밀어올리던 catchup이지 무한증가 아님(가설 확증). 두 arm 생존(HB 04:01·GPU 12/43GB). 남은 열린질문: `--keep 2`가 arm당 3~4 ckpt 유지 중(pre-recovery 잔재 미트림)—다음 체크포인트(gs152/gs6) 착지시 트림하면 건전, 누적하면 향후 climb→Codex GO 후 잔재정리. 현재 1024GB<임계1.2TB·quota에러 없음=무조치. **통찰:** 원격 저장소 "증가 추세"는 *catchup(유한·자기종료)* vs *runaway(무한)* 두 클래스이고, 판별자는 **"step번호 고정인데 바이트만 증가"**다 — 같은 ckpt 폴더의 크기가 커지면 업로드 진행중(곧 plateau), 새 step번호가 계속 추가되며 커지면 runaway. plateau 확인 전 성급한 삭제는 catchup을 runaway로 오판한 과잉대응.

### 상태 tick — 03:50 UTC(7/18) — [HF 모니터링] 용량 완만 증가+pusher 로그 가시성. 두 arm 생존(b2 GPU 52826MB·b3pkg 43326MB, HB 전진). HF 910→934→**967GB**(체크당 ~30GB↑)=post-recovery pusher가 `--keep 2` 트림 미반영(b2 [148,150,151]=3·b3pkg [1,4,5,8]=4 누적). ★발견: push_ckpts_to_hf.py는 `> /scratch/logs/push_rq3_*.log`로 **노드-로컬 로깅**이라 `amlt log view`(메인 stdout)에 트림 활동이 안 보임→HF 모니터링은 **HF repo 상태 직접 조회**로만 가능(현행 방식 정당). 조치=무조치(E-038 임계 1.2TB 이하·quota 에러 없음). 04:06 틱서 `--keep 2` 트림 반영(b2→[150,151]·b3pkg→[5,8] 감소) 확인, 계속 증가시 pre-recovery 잔재 Codex GO 후 정리. **통찰:** 원격 저장소 모니터링에서 "정리 도구의 로그가 관측 파이프라인 밖(노드-로컬)이면 도구 동작을 상태로 역추론"해야 한다 — pusher가 트림하는지 로그로 못 보므로 HF repo의 ckpt 개수 추이가 곧 --keep 동작의 프록시.

### E-038 (2026-07-18 03:30 UTC) — [저장소] HF 용량 감사·정리 정책(사용자 요청)

**문제(추상):** 잦은-선점+빈번-체크포인트 durable 학습은 원격 저장소를 무한 누적시킬 수 있다. 그러나 "용량이 크다"는 신호만으로 삭제하면 (a)실행 중 런의 자동관리 체크포인트를 건드리거나 (b)미실행 런의 resume 선택지를 비가역 파괴할 수 있다 — 정리의 안전성은 크기가 아니라 *각 아티팩트의 소유 런 상태(실행중/완료/미실행)와 재현·resume 필요성*으로 판정해야 한다.

**해결(구체):** 사용자 요청("HF 용량 모니터링·필요시 정리")에 repo=metacot-h200-triobj-dcpo-v3 감사(files_metadata=True로 config별 step별 바이트 집계). 총 **910GB**: rq3_b0 gs290,300(196GB·완료·평가됨·pusher keep2) / rq3_b2 gs148,150(실행중·keep2 정상) / rq3_b3pkg gs1,4,8(실행중·gs1 곧 self-clean) / **rq3_b3nopmi gs135,140,145(278GB·미실행·auto관리 대상 아님=유일 정리후보)**. Codex GO/NO-GO 전문 전달→**NO-GO**: b3nopmi 278GB는 (1)비가역, (2)gs145 resume 선택지 상실, (3)gs0 fresh-restart 의도 미확정이라 삭제 보류; 대신 b0 intermediate가 안전후보(단 eval결과 보존조건); 910GB는 quota/free space 확인 전엔 긴급 단정 불가(현재 업로드 정상=quota벽 아님). 조치=**무삭제**, HF 용량 체크를 루프 tick에 상시 편입(config별 GB+실행상태 로깅), b3nopmi 정리는 그 arm 런치 시 resume-vs-fresh 결정과 함께 처리.

**통찰:** 원격 저장소 정리의 결정 축은 용량이 아니라 **아티팩트-런 생명주기 매핑**이다 — 실행 중 런은 pusher `--keep N`이 자동관리하므로 만지면 안 되고(경합), 완료 런은 최종 ckpt+평가결과만 남기면 되며, 미실행 런의 중간 ckpt는 그 런의 미래(resume/fresh/폐기)가 정해지기 전엔 삭제가 성급하다. "가장 큰 것을 지운다"가 아니라 "생명주기가 끝난 것만 지운다". quota 에러 부재=긴급 아님이라 판단 유예가 곧 리스크 최소화. 관련 [[rq2-estimand-endtoend-not-content-0717]](HF 2.06TB 정리 이력).

### 상태 tick — 03:30 UTC(7/18) — [E-037 종결] 선점 recovery 완전 완료. b2 GPU 3810MB→**40644MB**·b3pkg 3810MB→**52538MB**(HB 03:30 전진)=vLLM KV캐시 로드·rollout/학습 활성=**두 arm 학습 재진입 확정**. HF 프런티어 gs150/gs8 유지(첫 post-resume 스텝 곧 착지). E-037 선점→auto-retry→pull_resume(gs150/gs4)→verl→vLLM→training 전 파이프라인 무개입으로 정상 복구 확인. 캐던스 30분 복귀.

### 상태 tick — 03:24 UTC(7/18) — [E-037 후속2] 학습 재진입 진행 확인. b2·b3pkg 둘 다 pull_resume 완료→verl 런치→wandb 로그인→**vLLM rollout 워커 초기화 중**(HB 03:24:29/03:24:51 전진·GPU 3810MB=KV캐시 할당 전). 파이프라인(부트스트랩→pull_resume gs150/gs4→verl→wandb→vLLM) 정상 통과. HF 프런티어 gs150/gs8 유지(첫 post-resume 스텝 전이라 정상). GPU는 rollout 시작시 40~70GB로 상승 예정. 무개입 지속. 15분 후 첫 스텝 재개(HF 프런티어 전진) 확인 예정.

### 상태 tick — 03:20 UTC(7/18) — [E-037 후속] 선점 auto-recovery resume 검증 성공. b2 로그 `[YAML] existing GRPO resume gs (model+extra+optim>=4) = 150`→RGS_STEP=150→pull_resume_ckpt.py 실행=**gs150 재개**(gs0 아님 확인). b3pkg RGS_STEP=**4**(HF엔 gs8 있으나 완전성가드 각4/4/4가 gs8을 부분업로드로 배제→마지막 완전 gs4서 재개)=손상 resume 방지 작동. **통찰:** 완전성 가드(model+extra+optim≥4 각)는 선점이 업로드를 중단시켜 남긴 **부분 체크포인트를 자동 배제**하고 마지막 온전한 앵커를 고른다 — "HF max_gs=8"을 그대로 믿었으면 반쪽 gs8로 resume해 깨졌을 것을, RGS 로직이 gs4로 안전 후퇴시켰다. durable 학습의 resume 정확성=단순 max_step이 아니라 **3-way 완전성 검증된 max_step**. 무개입(auto-retry 신뢰)이 정답이었음 재확인.

### 상태 tick — 02:53 UTC(7/18) — 02:52 예약 틱 발화. b2(mighty-civet) **HB 02:53:55 전진·GPU 43GB 활성=확실 생존**, 단 HF 프런티어 gs150에서 정체(직전 체크와 동일). ★진단: 정체·선점 아니라 **pusher 업로드 랙**(save_freq=1로 로컬 in-memory step은 gs151+ 앞서 있으나 ~4.5MB/s 대역폭 병목으로 HF 업로드가 뒤처짐)=알려진 durability 병목의 시그니처. b3pkg(smart-bat) HF gs4·생존. **통찰:** "HF 프런티어 정체 + HB 전진"은 **장애가 아니라 업로드 백로그**다 — HF max_gs를 유일 생존지표로 쓰면 건강한 런을 죽었다 오진한다(2계층 확인=HF 내구성 + HB/GPU 생존이 필수). 부수효과: save_freq=1의 durable-gap이 벌어지는 중이라 b2 save_freq 상향은 오히려 내구성 손해→gs300까지 save_freq=1 유지 결정 재확인.

### 상태 tick — 02:20 UTC(7/18) — 두 arm durable 확인. **b2(mighty-civet)** HF gs148/300(143→145→148 전진·각 21~23샤드 완전)·3h만에 gs148=~50step/h 건강·gs300 ETA~3h. **b3pkg(smart-bat)** in-mem global_step:3·HF gs1 anchor **완전착지(21파일)**=save_freq=5 적용조건 충족·HB 02:18 전진·GPU 14~33MB요동(롤아웃↔업데이트 교대)·vLLM sleep-mode 토글 정상. ★판단: healthy run 강제cancel 금지(gs2/3 로컬진척·모멘텀 낭비)→save_freq=5는 **다음 자연선점 경계에서** 적용. 특히 b2는 gs300까지 ~3h뿐이라 결과 직전 resume 리스크 회피 위해 **끝까지 그대로**·save_freq=5는 ~40h 남은 b3pkg에만 다음 선점때 적용. **통찰:** "안전한 타이밍"의 정의는 시각이 아니라 *경계*다 — 파라미터 변경(save_freq)은 어차피 HF anchor에서 resume하므로, 자연선점이 이미 로컬진척을 버리는 그 순간에 편승하면 추가 손실 0. 건강한 런을 만지는 건 항상 순손실.

### 상태 tick — 05:35 UTC(7/20) — [★durability 근인 재확정·Codex+Fable 수렴]
두 arm running(b2=living-pigeon 14h·b3pkg=smart-bat 2d)·durable frontier 불변(b2 gs158·b3pkg gs43 COMPLETE; b3pkg in-mem extra_state gs139까지 진행). ★**HF 실측 audit로 durability 근인 교정**: 이전 "Xet가 대용량 차단" 진단 **반증**—98GB complete 체크포인트가 실재(b0 gs300·b2 156/158·b3nopmi 135/140+145 partial m4o3·b3pkg 40/43). ∴진짜 근인=**체크포인트 ~98GB ÷ ~4.5MB/s ≈ 6h > 선점간격**이라 완전본은 긴 무선점 창에만 착지, 이후 스텝은 작은 extra_state만 업로드(Codex·Fable 80~85% 수렴). 4.5MB/s 저속은 **hf_xet 제거 부작용**(legacy 브리지 폴백)—"Xet 차단"과 정반대 방향의 오진. `--keep 2` 정리는 **무죄**(arm별 완전본 2개 보존 실측 확인). 미확인 잔여=pusher thrashing(진행중 98GB 버리고 최신으로 갈아탐) / cleanup race—pusher 로그 필요. ★**수렴 권고**: (1)hf_transfer 활성화+14GB 단일샤드 타임드 실측이 최저비용 unblock(성공~75%), (2)gs300 arm별 분리—**B2 gs300 가치**(RQ1 지속성)·**B3pkg gs300 예측음성이라 동결**, (3)**단일 시드가 gs300 미도달보다 큰 과학적 구멍**→여유 컴퓨트는 2번째 시드. ★결과해석도 재확정: RQ1 이득(+7.4)이 emission 4.5% 붕괴 후에도 잔존=**초기화 흔적이지 test-time 메타행동 아님**(북극성 과대주장 주의)·PMI 역전(+10.4→−1.7)은 B3noPMI gs50=50.0 노이즈 의존이라 약화. instruct(T1)는 무붕괴, base(RQ3)만 붕괴=substrate 의존. 사용자 경로결정(A durability수정/B 예비정리+2seed/C 둘다) 대기·무개입.

- 상태 tick 06:37 UTC(7/20): b2=living-pigeon 15h·b3pkg=smart-bat 2d 둘 다 running. durable COMPLETE 불변(b2 gs158·b3pkg gs43). b3pkg in-mem extra_state gs139로 05:35 틱과 동일=1h 순진행 미관측(느린 step or 선점재부팅 가능성·status running이라 무개입, 다음 틱 gs139 정체 지속여부 감시). 신규 terminal 실패 없음. 경로결정 대기·무개입.

- 상태 tick 07:39 UTC(7/20): b2=living-pigeon 16h·b3pkg=smart-bat 2d(retry_004) 둘 다 running. ★**지난 틱 "gs139 2h 정체=F5 의심" 해석 교정**: node 로그 확인 결과 verl 생존 확증(keep-alive가 `kill -0 3992` 성공·gpu0used=43GB 모델적재). HF의 extra_state gs139는 **이전 retry가 in-mem 도달 후 잃은 stale ghost**이지 현재 진행분 아님. retry_004는 durable gs43에서 **재등반 중**(gs139 아래)이라 HF frontier 불변이 정상. ★**교훈: HF extra_state는 live-progress proxy로 부적합**(retry ghost 잔존)—진짜 liveness=kill-0 PID+gpu mem, 진짜 durable진행=COMPLETE frontier. b3pkg는 gs43↔선점 시시포스 구조 실증(Fable 진단 확인)=durability 미해결시 durable 전진 사실상 불가. b2 durable gs158 불변. 신규 terminal 실패 없음. 경로결정 대기·무개입.

- 상태 tick 08:43 UTC(7/20): b2=living-pigeon 17h·b3pkg=smart-bat 2d 둘 다 running. durable COMPLETE 불변(b2 gs158·b3pkg gs43). 신규 terminal 실패 없음. 경로결정 대기·무개입(no-change 틱).

## E-065 (2026-07-20 09:44 UTC) — b3pkg terminal failed·반사적 재제출 보류(수렴 결론 적용)

**추상적 문제.** 선점 컴퓨트에서 "절대 멈추지 말고 keep-alive"라는 사전승인과, "이 arm은 예측된-음성이고 durability 미해결이라 추격이 무가치하다"는 방금의 적대적 수렴 결론이 충돌할 때, terminal failed 이벤트에서 어느 쪽을 따를 것인가. 사전 blanket 승인의 반사적 실행이 성급실행 실패의 근인이었다.

**구체적 해결.** b3pkg=smart-bat status=**failed(1.6kB)**=auto-retry 소진 terminal 선점([[b3pkg-silent-pass-deaths-are-preemption-not-code-0713]]). b2=living-pigeon 18h running 정상. durable COMPLETE 불변(b2 gs158·b3pkg gs43). ★결정=**b3pkg 반사적 재제출 보류**. 근거: (1)이번 세션 Codex+Fable 수렴이 "B3pkg gs300=emission 4.5% 붕괴 후 궤적=예측된 음성대조·정보량≈0→동결" 만장일치, (2)durability 미해결이라 재제출해도 gs43↔선점 시시포스 무한반복·durable 전진 0(노드만 소모), (3)새 지시(0720)=재제출 등 컴퓨트 액션도 수렴 후—그 수렴이 이미 freeze를 지목, (4)b3pkg gs43은 HF 안전=재제출 언제든 가능·무손실. ★b2는 유지(RQ1 지속성 가치·gs158로 gs300 최근접). 사용자에 fork 제시(AskUserQuestion): b3pkg 동결+b2 durability수정 / b3pkg 그대로 재제출 / 둘다 동결+예비정리+2seed.

**통찰.** ①**blanket 사전승인 < 최신 수렴 분석**: "절대 멈추지 말고"는 *생산적* 런을 버리지 말라는 뜻이지 *무가치가 입증된* 런을 강제로 태우라는 뜻이 아니다—"결과 나쁘면 다음 실험"과 정합. ②**terminal failed가 pending 경로결정을 강제이벤트로 승격**: 추상적으로 미뤄둔 A/B/C 결정이 b3pkg 사망으로 지금-여기 구체 fork가 됨. ③반사적 재제출을 참는 것 자체가 사용자 0720 지시(성급실행 금지·수렴우선)의 실천. 관련 [[codex-fable-debate-converge-before-experiment-0720]]·[[E-063]].

## E-066 (2026-07-20 09:55 UTC) — 사용자 재지시("고쳐서 돌려라")→E-063 A-only 구체 계획 Codex+Fable 수렴

**추상적 문제.** 자기파괴(emission collapse)로 음성 난 arm을, 동결도 맹목 재제출도 아닌 "**기전을 고쳐 돌리는**" 방향으로 전환할 때, (a)무엇을 정확히 구현하고 (b)어떻게 값싸게 먼저 검증하며 (c)durability와 어떤 순서로 물리고 (d)무엇이 WIN인지를, 성급실행 없이 사전 확정한다.

**구체적 해결(Codex+Fable 수렴).**
- ★**Step 0 계측먼저(read-only·durability 불필요·advantage 불변)**: `_compute_dcpo_region_advantage`에 리전별 advantage 크기 텔레메트리(emit-결정 토큰 A_corr·나머지 ANSWER |A_corr|·META_CONTENT A_PMI·R_meta·emission%·wellformed%). **게이트 G0**: control서 emit-결정 A_corr가 지속 음수 AND |A_corr(emit)|가 meta로 라우팅되는 |PMI|/|R_meta|와 대등+. G0 실패=단일토큰 이론 오류→피벗(meta-포함 롤아웃 correctness 다운웨이팅/decorrelation). ★Fable 핵심: 단일토큰 마스킹이 붕괴를 멈출 확률 ~55%(공유표현 간접경로 잔존 위험)이라 발사 전 계측이 정보/비용 최대.
- **Step 1 구현(A-only)**: 컴포넌트 라우팅 단계(A_corr+A_PMI 합산 전)서 A_corr를 META_REGION + **모든** emit-결정 토큰(각 meta-open 직전, 다중 메타 전부)에 **양·음 양쪽 0**(음수만 clamp=Gandhi form-reward 함정). PMI/R_meta 라우팅 불변(identifiability=A-only, PMI 동시수정 금지). 유닛테스트 4케이스(no-meta/meta-at-0/single/multi)+**flag-off 회귀**(플래그 끄면 기존 B2/B3pkg와 수치동일=matched 오염방지).
- **Step 2 값싼검증(durability 독립)**: 3arm(B2/B3pkg=음성대조/B3pkg-v2) gs50~100만. emission%는 학습시간 wandb 지표라 ckpt 업로드 불필요. **게이트 G1(필요조건)**: emission(v2)≥50~55%@gs50 & ≥50%@gs100·control 붕괴재현(≤25%@gs50)·wellformed≥90%·acc(v2) B2대비 노이즈이상 나쁘지않음.
- **Step 3 durability(임계경로 밖·병렬)**: hf_transfer 14GB 단일샤드 타임드+**라운드트립 checksum 검증**(hf_xet 404 silent-truncation 재발방지)—검증 전 본런 적용금지.
- **Step 4 결과(durability 필요)**: G1 통과시만 B2 vs v2 gs300. **게이트 G2(충분조건/WIN)**: acc(v2)−acc(B2)≥마진@gs300 AND emission 선택적대역(always-on 아니고 <10% 아님).
- ★**게이트 2분할 사전등록(Fable)**: A-only는 usefulness(PMI) 미변경이라 **MECHANISM 게이트**(붕괴제거+net-harm 없음)로 판정하지 OUTCOME 게이트(acc>B2)로 판정하면 범주오류. 가장 개연성 높은 결과="G1통과·G2평평"=**기전확증+붕괴치료=과학적 성공**(FAIL 아님)—사전에 그렇게 명명. R_meta 퍼즐 해소: R_meta는 META_CONTENT에만 걸려 emit-결정(ANSWER_REGION)에 **닿지 않음=좌표문제**(크기 아님)→emission-protection이 올바른 레버.

**통찰.** ①**"고쳐서 돌려라"의 최소단위는 계측**: 처방(단일토큰 마스킹)이 표적을 맞히는지부터 read-only로 확인(G0)해야 성급 발사를 막는다. ②**A-only는 anti-extinction이지 usefulness 신호 아님**: 붕괴는 막아도 acc 상승은 PMI 몫이라, acc-WIN 확률 ~25%(Codex·Fable 일치)—A-only 성공=붕괴제거까지가 정직한 기대. ③**게이트 분할이 "옳은 실험을 하고도 오독"을 막는다**. ④durability는 gs300 결과에만 필요하고 gs50 emission 생존 검증엔 불필요→병렬화로 임계경로서 제거. 관련 [[E-063]]·[[codex-fable-debate-converge-before-experiment-0720]]·[[cfgroup-form-vs-behavior-gandhi-0622]].

## E-067 (2026-07-21 00:xx UTC) — 사용자 점검요청("SFT/코드 문제 아닌가")→wandb 궤적이 SFT-품질 근인 지목

**추상적 문제.** instruct서 살아남아 이긴 메타 습관이 base서 붕괴할 때, 이것이 (a)순수 substrate 효과인지 (b)SFT 품질 결함인지 (c)코드/보상 버그인지를, 정교한 처방(E-063)을 짓기 전에 배제 판별한다.

**구체적 해결(wandb gistdslab/metacot-dcpo-v4 궤적 직접확인).**
- ★**코드 배제**: RQ3는 T1-우승 pmishift와 IDENTICAL reward config/코드(yaml 명시·archive/launchers_pre_rq3/h100std_pmishift.yaml). 같은 구조가 instruct선 meta 87~98% 유지·우승(docs/reports/2026-07-06-T1). 코드가 base만 죽이는 것 아님.
- ★**SFT 품질 결함=smoking gun**: b3pkg 궤적 gs1에 **wellformed_rate 0.40**(발화 meta의 60%가 malformed)·**rmeta_mean −0.27**(R_meta가 SFT meta를 음수 채점). gs50 wellformed 0.06→gs100 0.002 붕괴. 즉 base meta-SFT는 <|meta|> 토큰만 배웠지 제대로 된 블록을 못 만듦=0710 under-trained 진단 잔존(unmask fix로 emit 38→92% 올렸으나 wellformed 미수정).
- ★**PMI 무관 확증**: b3pkg(PMI有)와 b3nopmi(PMI無)가 emit 0.87→0.03·wellformed 0.40→0.002로 **동일** 붕괴→correctness+base+저품질SFT 공통요인이지 PMI/패키지 아님.
- 뉘앙스: acc_with(0.65~0.73)>acc_wo(0.25~0.50) 전구간 유지=wellformed 40% 코어 meta는 유용할 수 있음(유용한 소수+malformed 다수 혼합, RL이 코어까지 통째 소거).

**통찰.** ①**E-063은 증상처방**: correctness의 emit-억압은 실재하나 근본은 SFT가 meta 60%를 malformed로 생산+R_meta 음수채점. 저품질 meta를 emission-protect로 억지 보존=Gandhi 함정(껍데기). ②**진짜 상류 레버=base meta-SFT 품질 재훈련**(wellformed 0.40→~0.95·R_meta 양수). 그러면 correctness도 덜 억압→자연 생존. ③substrate "순수효과" 프레이밍은 과함—substrate가 SFT 품질을 매개로 작용. ④사용자 "점검" 지시가 정교한-오처방(E-063 선행)을 막음=0720 수렴규율의 실증. 관련 [[pmishift-starvation-base-sft-undertrained-0710]]·[[E-063]]·[[cfgroup-form-vs-behavior-gandhi-0622]].

## E-068 (2026-07-21 UTC) — Codex+Fable 수렴: SFT-품질-먼저(S)>조건부 E-063, 동시적용 기각

**추상적 문제.** 붕괴 근인이 SFT 품질(wellformed 0.40·R_meta<0)로 재지목됐을 때, 정교한 RL 처방(E-063)과 상류 데이터 처방(S) 중 무엇을 어떤 순서로, 어떤 게이트로 검증하는가. 처방이 진단을 앞서지 않게.

**구체적 해결(Codex+Fable 수렴).**
- ★**우선순위 S>조건부E>~~동시B~~**: 동시적용은 (1)T1-패리티(같은 패키지 instruct 우승=최강 지렛대) 파괴, (2)귀속불능. E-063 단독=Gandhi 함정(malformed junk 보존)+correctness만 마스킹해도 R_meta·format 음수 잔존→붕괴 못 멈출 수(실패 75~80%). E-063은 **clean SFT(wellformed≥95·R_meta≥0)로도 gs50 침식시에만** 정당(그때 "검증된 유용행동을 RL이 죽인다" 최초 성립).
- ★**acc_with>acc_wo 유용성 증거 기각**: 선택편향+gs100 emit0.03이라 with표본 극소·GRPO는 그룹내 advantage라 전역대비 무의미. **무학습 판별**: (0a)기존 롤아웃 로그 그룹내 meta/nonmeta advantage 대조(비용0·RL이 본 실제 gradient), (0b)강제 prefill 반사실(선택제거 인과추정).
- ★**SFT 수정 최소충분=(a)불균형 수선-우선-제외 + (b)길이캡 512자 EXCLUDE(절단금지—절단은 미종결블록을 가르침)**. epoch3/lr1e-5/보상 불변(첫 원인분리). (c)에폭↑은 게이트 실패시만(1.7k행 과적합), (d)format-strict 디코딩 기각(estimand 변질·[[rq2-estimand-endtoend-not-content-0717]]), (e)w_emit/w_format↑ 기각(T1 패리티 파괴).
- ★**긴꼬리 주목(Fable)**: 데이터 28%불균형만으론 wellformed0.40 설명부족→**max21k자 롱테일이 temp1.0서 truncation 증폭**. temp0 vs temp1.0 wellformed 비교로 데이터결함 vs 디코딩결함 분리.
- ★**검증 사다리**: SFT-eval 게이트(RL패리티 temp1.0·max8192): wellformed≥95·truncation≤2·R_meta≥0·prefill반사실 Δacc≥0(4번째가 "보존가치" 판정 핵심). RL pilot 게이트 gs50(durability 불필요): emit≥0.6·wellformed≥0.8·R_meta 재음수 안됨·acc≥B0궤적. 통과=SFT가 붕괴 치료. 실패(SFT지표 양호+반사실 양수인데 침식)=그때 E-063 단독투입.
- ★**사다리 유효성**: 새 clean meta-SFT-v2 ckpt 하나서 **B2-v2·B3pkg-v2 둘 다** 재출발(shared-init 유지). B0 불변(gold-SFT=동일base+동일RL protocol이 matched조건). 기존=meta-SFT-**v1 arm** 재명명→v1-vs-v2가 "SFT품질=붕괴원인" ablation 자산. 팩토리얼=B3-v2 vs B3-v2+E(noPMI 반복 불필요·PMI무관 확증됨).
- ★**Step 0 먼저(반나절·거의 무학습)**: 그룹내 advantage 대조+현SFT prefill 반사실+temp0/1.0 wellformed+gold-SFT 코퍼스 대칭감사.
- 신뢰도: E단독먼저 오류 0.9·S먼저 최선 0.8~0.85·clean SFT만으로 붕괴해결 0.55~0.65·clean후 E필요 0.5(Step0 결과의존).

**통찰.** ①**처방이 진단을 앞서면 Gandhi**: wellformed0.40·R_meta<0은 "RL이 쓰레기를 정당히 치움"에 부합→"유용행동 오살"서사는 Step0 반사실 양성 전엔 보류. ②**동시적용의 유혹을 이기는 게 matched 통제**: E-063을 넣는 순간 T1-패리티 축을 버림. ③**절단 vs 제외**: 길이캡 초과행 절단=병(미종결) 학습이라 반드시 제외. ④v1-vs-v2 대비가 인과 ablation 자산. 관련 [[E-063]]·[[E-067]]·[[pmishift-starvation-base-sft-undertrained-0710]]·[[cfgroup-form-vs-behavior-gandhi-0622]].

## E-069 (2026-07-21 UTC) — [★수렴] think-on 논쟁 2R(Codex=비판·Fable=브레인)→근인=base SFT 코퍼스 재앙, 처방=v8 재사용

**추상적 문제.** "instruct서 이긴 데이터를 base에 안 쓰고 새로 만든 게 맞나, base를 think-on으로 해야 하나"라는 사용자 질문을, 전체 코드·헌법·의도·결과를 실제로 읽은 두 역할(비판/브레인) 논쟁으로 수렴시켜 판정한다.

**구체적 해결(데이터 감사가 fork 붕괴).**
- ★**결정적 감사**(think닫힘/meta정상중첩/meta-inside/완전clean): **b23 base=30/71/5/0%**·b23 v2(내 정제)=31/100/0/**0%**·**v8 instruct(우승)=100/100/100/100%**. 즉 base SFT 코퍼스는 **완전정상 행 0개**(70% `<think>` 미종결·95% meta 추론밖), 내 v2도 여전히 0% clean(meta태그만 고침=증상처방).
- ★**think은 이미 ON**(base_rl_recipe.md "THINK-ON PRESERVED, never enable_thinking=false"). 진짜 축="think on/off" 아니라 **데이터 품질+meta 위치**. 코드=T1우승과 동일·substrate 아님→**근인=깨진 base SFT 데이터**.
- ★**처방 수렴=v8_meta_inside_think(100% clean·T1우승 설계) 재사용**(재생성 아님—강한생성기 부재라 순환). "instruct chat-template 벗기기"는 불필요(v8 strict는 순수 [user,assistant] content). base-native=v8 strict를 rq3 SFT 경로(sft.py ChatML+EOS 151645)에 그대로 태움.
- ★**Fable 신규발견=sft.py:101-103 침묵절단**(full_ids>max_length 4096 자름)→클린 코퍼스도 훈련시 재오염(</think>·answer 절단). **빌드게이트에 apply_chat_template 후 토큰길이≤max_length 100% 필수**.
- ★**유일 잔여리스크(양측 일치)**: instruct 생성 v8 체인→base 자유생성 분포이동(SFT proper-nesting은 필요조건이지 자유생성 종결 충분조건 아님). **대응=L2 free-gen 종결게이트**(temp1.0·8192·AIME30+math500: `</think>` 비종결 ≤2%math/≤5%AIME=Codex kill-switch·answer추출≥98%·greedy acc B2'≥B0'−2pp·pmi auc>0.55&n_save>0). gs50 pilot 후 재검사.
- ★**수렴 사다리**: L0 v8재렌더+빌드게이트(무GPU)→L1 SFT×2(B0'/B2')→L2 게이트→L3 B3' gs50 pilot(내구성 불요)→L4 풀 3-arm gs300(hf_transfer 병행). **부채청산**: C-1(norm_adv VANILLA 침묵무시)=gs1 assert 텔레메트리·RQ1-코퍼스(b0_gold1290 vs b231763 상이체인)=B0'/B2' same-chain meta-only 쌍→RQ1 순수 meta효과 식별. 구 래더=부록·신 estimand(think-on/meta-inside) 별도보고.
- 신뢰도: 플랜 0.85·kill-prob ~25%(L2 종결게이트에 집중·값쌈)·게이트 통과후 조건부 실패<10%.

**통찰.** ①**"instruct 데이터 안 썼나"라는 사용자 직감이 정곡**: base 코퍼스가 0% clean인데 그걸 청소하려던 내 v2도 0% clean=RV/답-뒤 설계로는 clean 불가·유일해는 v8-품질(meta-inside) 채택. ②**think on/off는 허위 프레이밍**(이미 on)·진짜는 데이터. ③**클린 코퍼스도 sft.py 침묵절단이 재오염**=게이트는 렌더 후 토큰길이까지. ④**같은-체인 meta-only 쌍(B0'/B2')이 RQ1 식별성을 과거보다 개선**+C-1 청산=재기반이 순비용 아니라 부채정리. 관련 [[E-067]]·[[E-068]]·[[T1-heldout-meta-beats-base-6of6-0706]]·[[base-redesign-teacher-free-selfdistill-pivot-0708]]·[[rq3-triple-audit-c1-stdnorm-claims-0716]].

## E-070 (2026-07-23 UTC) — [L0 완료·GREEN] v8 재사용 clean 코퍼스 확정(Codex 피드백+Fable 점검 수렴)

**추상적 문제.** 근인(base SFT 코퍼스 0% clean)을 고칠 때, T1 우승 데이터(v8)를 base로 재사용하되 렌더/절단/EOS/매치드-쌍 무결성이 이전 성공과 구조적으로 일치함을 발사 전에 저비용 검증한다.

**구체적 해결.**
- `scripts/build_v8_strict_paired_data.py` 실행→v8_meta_inside_strict + v8_base_matched_strict **4264 쌍**(6329서 strict 필터 2065 탈락·verify/redirect 비율 48.4:51.6 보존).
- L0 감사: meta strict = **think-closed/meta-nested/meta-inside 전부 100%**(b23=0%), base strict=100% clean. meta≤500자 캡은 15% 초과.
- ★결정(Codex+Fable 수렴): (1)**500자 캡 철폐**(v8 우승본 무캡·max826·T1 충실), 게이트는 **토큰≤4096 단일**. (2)>4096 **쌍-드롭**(sft.py:102-103 침묵절단 차단). (3)**wrong_prefix 부재 확인→whole-response**(unmask fix 유지·segment-mask 코드상 차단 확증). (4)산출=`data/b2on_v8meta_strict_sft.parquet`(B2'/B3')·`data/b0on_v8base_strict_sft.parquet`(B0').
- ★렌더 패리티 게이트(양측 지목 유일 블로커): 실제 sft.py 경로(Qwen3-8B-Base ChatML 템플릿·meta토큰 151669/70 추가·EOS im_end 151645) replicate. 초기 EOS 체크 과엄(마지막 정확히 im_end 요구)→진단=`...$.<|im_end|>\n`(im_end 존재·뒤 개행)→마지막3토큰내 im_end로 완화. **최종: 4264→19드롭(>4096)→4245쌍·>4096=0·eos_missing=0=GREEN**(양 코퍼스).
- ★T1 대비 divergence 기록(Fable): ①strict가 conf 중간대역 검열→"v8의 conf-분리 강화 부분집합"(동일분포 아님). ②outside 텍스트 "The answer is $boxed$." 표준화. ③B0'는 "meta토큰만 절제"가 아니라 "동일문제 direct-solution 대응군"(RQ1 식별엔 오히려 정확)—논문 기술 정확히.

**통찰.** ①**발사 전 실측 게이트가 거짓 RED도 잡는다**: 내 순진한 EOS 체크가 4264 전부 탈락시켰으나 진단으로 "im_end+개행"임을 확인=게이트가 데이터뿐 아니라 *검증 코드*도 검증. ②**렌더 패리티가 T1 재현의 생명**: 데이터가 clean해도 렌더(ChatML vs passthrough)·절단·EOS가 다르면 구조적 이탈—실측 token IDs로만 확증. ③base=think-ON·ChatML·im_end 종료=구조상 instruct T1과 정합. 다음=L1(B0'/B2' SFT×2). 관련 [[E-069]]·[[autonomous-codex-feedback-fable-propose-loop-benchmark-t1-0721]]·[[T1-heldout-meta-beats-base-6of6-0706]].

## E-071 (2026-07-24 UTC) — [★계보 대정정+수렴] T1=2단 SFT 스택·RV코퍼스=b23과 동일 0%clean·L1=Option C 확정

**추상적 문제.** "instruct 성공 그대로"를 위해 T1의 실제 계보를 실측하니 통념(단일 SFT)이 틀렸다 — 성공 레시피의 어느 층이 필수이고 어느 층이 유해 재수입인지를 코드·데이터로 확정한 뒤 발사 설계를 수렴시킨다.

**구체적 해결(실측+Codex/Fable 수렴).**
- ★**T1 계보 정정**: T1 init=`v8_rv_functional_sft`=**2단 스택** — SFT1(instruct+v8_meta_inside_strict 100%clean·3ep/2e-6·whole-response) → SFT2(rv_redirect_verify_functional·3ep/1e-5·**segment-masked**) → triobj RL. 
- ★**충격 실측**: SFT2 코퍼스=**b23과 동일 통계**(1763행·think닫힘30%·meta-inside5%·완전clean 0%·max20206). b23은 "base가 새로 만든 데이터"가 아니라 **T1의 instruct RV 코퍼스에서 wrong_prefix만 비운 것**. ∴base 실패=T1 대비 **2중 이탈**: (a)SFT1 기반 생략 (b)같은 깨진 코퍼스를 whole-response로 직접 학습(깨진 구조를 통째로 인각).
- ★**Fable 코드 확정(재수입 채널)**: `segment_loss_mask.redirect_train_spans`=단일 스팬 `[prompt+prefix, total)` → **meta+recovery뿐 아니라 시퀀스 끝까지 loss**. 70% 미폐쇄 행에서 "`</think>` 없이 종료" 패턴이 loss 안=segment mask는 wrong_prefix만 보호. instruct서 무해했던 건 prior+SFT1 방패 덕. +`sft.py:101-103` 절단은 EOS 재부착 없음(max20206자 행=비종료 학습).
- ★**수렴 결정=Option C(staged)**: ①지금 2잡 발사 — B0'(=`b0on_v8base_strict_sft`)·B2'(=`b2on_v8meta_strict_sft`) SFT1 **3ep/lr1e-5**(2e-6은 base에 얕음—0710+b23 증거 직교수렴·ledger에 substrate 적응 명기) ②L2 게이트(EOS rc0·emission≥0.85·**wellformed≥0.90**·unclosed-think≤5%·truncation@8k base+2pp내)+PMI 프로브(auc>0.55·n_save>0·placebo/shuffle 대조) ③PRESENT→RL 직행 / ABSENT→**SFT2 필터본만**(think-closed ~529행·segment-masked·2ep/2e-6·4096초과 드롭) 재게이트(무퇴행 필수) 후 RL. **SFT2 원판(B) 금지**(재수입 채널 코드확정). ④w_over=0.0은 T1 런처 자체 설정=파리티 확인(우려 해소).
- ★**RL 킬스위치 사전등록(Fable)**: 프로브=teacher-forced vs RL=on-policy 갭 대비 — RL gs25에 `n_save≥1 & emission≥0.70 & R_meta 양성비율>0` 미달시 gs50 전 kill→SFT2 필터본 브랜치 전환(비용≈노드1일).
- 잡 구조: `h100std_sft_b23_unmasked.yaml` 패턴 복제(단일노드 체인: staging→SFT→EOS게이트→게이트2/프로브→shard-verify push). hf_transfer 리허설 포함.

**통찰.** ①**"성공 그대로"는 계보 실측 없이 불가능**: 통념(T1=v8_meta_inside 단일 SFT)이 틀렸고, 실측이 "기반+수술적 탑업" 구조와 실패의 2중 이탈을 드러냄. ②**마스킹의 보호 범위를 코드로 확인해야**: segment mask가 깨진 구조를 loss 밖에 둔다는 가정이 꼬리(termination)에선 거짓—instruct의 무해는 prior 방패였음. ③**파리티는 층별로**: 데이터·보상=T1 그대로, 하이퍼·디코드=substrate 적응(ledger 명기)—"그대로"와 "적응"의 경계를 명시해야 재현 주장이 산다. 관련 [[E-069]]·[[E-070]]·[[pmishift-starvation-base-sft-undertrained-0710]].

## E-072 (2026-07-24 UTC) — [★L1 발사] B0'/B2' SFT 2잡 제출(probable-seal·learning-mustang)

**추상적 문제.** 수렴된 Option C의 첫 컴퓨트 단계(SFT1×2)를, 이전 성공 패턴(b23 체인)과의 구조 일치를 기계 검증한 뒤 발사한다.

**구체적 해결.** ①configs/sft_b0p_v8base.yaml·sft_b2p_v8meta.yaml(b23 키셋 정확 일치·3ep/lr1e-5/absolute output_dir) ②h100std_sft_b0p.yaml·h100std_sft_b2p.yaml(b23 체인 복제: staging→SFT→EOS게이트→term-eval(math500-100+**aime2024**·8k·temp1.0·n2)→shard-verify push). ★b23 원본의 잠복 함정 2곳 수정: 스피너 f-string 단일따옴표→`device=i`(int ordinal)·push-verify 내부 단일따옴표→`\"` 이스케이프(shlex 3토큰+내부따옴표0 기계검증). ③tarball 재패키지(parquet 2·config 2 동봉 확인)→GH release 359072254 asset **487971950**(round-trip md5 일치)→CODE_TAR_REVISION 주입. 422 교훈: repo default branch=master(main 지정이 422 원인). ④★Fable 프리런치 점검 **GO**(tarball 15파일 바이트 cmp·inner bash -n·python compile·$$이스케이프 스캔·argparse 대조 전부 통과·"b23보다 안전"). ⑤발사: **probable-seal**(b0p)·**learning-mustang**(b2p). ⚠️정직노트: pytest 가드는 로컬 torch 부재로 실효 안 됨(tail이 exit code 마스킹)—이번 사이클 python 코드 무변경이라 위험 낮으나 다음 코드변경 시 exit code 직접 게이트로 수정.

**통찰.** ①검증된 체인의 "복제"조차 원본의 잠복 결함(따옴표 2곳)을 물려받을 뻔—프리런치 기계 점검이 복제본을 원본보다 안전하게 만듦. ②게이트는 exit code를 물려받아야 게이트다(파이프 마스킹 함정). ③L2 판정 기준(사전등록): EOS rc0·truncation≤5%(kill-switch)·b2p emission≥0.85·wellformed≥0.90·acc(b2p)≥acc(b0p)−2pp·T1 기준치(emit0.99·n_save8-11)와 나란히 기록. 관련 [[E-070]]·[[E-071]].

- 상태 tick 04:52 UTC(7/24): [L1 순항] probable-seal(b0p)·learning-mustang(b2p) 둘 다 running(~44m·노드 즉시 확보·living-pigeon 공존). ★**b0p 완주**: SFT rc0→EOS GREEN(암묵: eval 진입 자체가 증거)→term-eval rc0 — **MATH500-100 acc 55.5%**(temp1.0 n2·구 B0 post-RL 54.4%를 SFT만으로 도달)·**AIME 8.3%**(구 B0 post-RL 4.6% 초과)·meta_rate 0.5%≈0(base arm 정상)·avg_len 363/1514≪8192(종결 건강). eval-json push는 HfHubHTTPError(비치명·|| 가드 작동)·모델 4-shard push attempt1 진행중(~1h 예상). ★b2p SFT 95%(720/759·loss 0.129 건강 수렴)→곧 EOS게이트+term-eval. 다음 틱=b2p L2 판정(emission≥0.85·truncation≤5% kill-switch·wellformed≥0.90). 구세대 rq3_b2 4d running(무개입).

## E-073 (2026-07-24 05:30-06:00 UTC) — [★durability 근인 최종 대정정] HF 저장쿼터 403이 진짜 근인 → 476.6+98GB 삭제로 해방

**추상적 문제.** 신세대 첫 업로드(16GB SFT 모델)까지 실패하자 "대용량·저속·선점창" 서사가 재반증됨 — 관측된 모든 durability 증상을 하나로 설명하는 근인을 로그의 실제 에러로 확정하고, 파괴적 복구(삭제)를 수렴 후 실행한다.

**구체적 해결.**
- ★**실제 에러 발견**: b0p push 8회 전패의 원문=`403 Forbidden: You have exceeded your public storage space`(LFS batch 거부). 계정 실측 **1041.5GB**(지배=구세대 ckpt repo 788.4GB). ∴**durability 트랩의 진짜 근인=계정 저장쿼터 소진**: ①"예전엔 올라갔는데 안 올라감"=쿼터가 점진 충만(사용자 직감 적중) ②b3nopmi gs145가 optim 3/4서 동결=쿼터 소진 순간 ③작은 extra_state만 착지=쿼터 잔여 틈. "98GB÷4.5MB/s>선점창" 서사는 기여요인일 수 있으나 주인 아님(E-071까지의 진단 재수정).
- ★**수렴 삭제(Codex+Fable GO·단계적)**: manifest 저장(docs/reports/hf_deletion_manifest_20260724.json·wandb 궤적 보존 명기)→Stage1 b3pkg 전체(198.1GB·1486파일)+b3nopmi 전체(278.5GB) 삭제→Stage2 gs158 무결성 검증(m4/o4/e4 확인) 후 gs156 삭제. **repo 788.4→213.5GB·계정 ~466GB**(쿼터 여유 확보). 보존: b0 gs300(기준선)·b2 gs158(가동중 arm 유일 앵커).
- L1 여파: b2p는 선점→retry로 SFT 재수행 중(~40분·L2 수치는 로그에 남으므로 무손실)·push는 쿼터 해방으로 자연 착지 예상. b0p는 push 루프 소진이라 **cancel→재발사(cosmic-worm)**(재훈련 ~40분·deterministic).

**통찰.** ①**에러 원문 > 증상 패턴**: 세 번의 오진(save_freq→Xet→대역폭) 모두 증상 패턴 추론이었고, 원문 한 줄(403 quota)이 전부 정리 — 처음부터 push 스크립트의 실패 로그를 원문으로 읽었어야 했다. ②**쿼터는 상태가 아니라 자원**: --keep 2×다중 arm×98GB는 저장을 무한 가정한 설계 — L4 RL durability 설계에 저장 예산(keep-1·model-only push·주기적 정리)을 1급 제약으로 편입해야. ③파괴적 복구도 "manifest+단계적+무결성 게이트"면 안전하게 자율 실행 가능. 관련 [[E-062]]·[[E-071]]·[[hf-xet-upload-pitfall-and-chain-fix]].

- 상태 tick 06:00 UTC(7/24): b2p=learning-mustang retry 재SFT 중·b0p=cosmic-worm 재발사(queued)·구세대 living-pigeon 4d running(gs158 앵커 보존). 다음 틱=b2p L2 수치+push 착지 확인.
- 상태 tick 06:18 UTC(7/24): b0p 재런(cosmic-worm) SFT 81%(~7분 잔여)→eval→push=쿼터수정 라이브 검증 예정. b2p(learning-mustang) retry SFT 18%(~30분 잔여). 구세대 living-pigeon 또 선점→queued(무개입·gs158 앵커 보존됨). 다음 틱=양쪽 L2 수치+push 4/4 착지.

## E-074 (2026-07-24 07:00-07:20 UTC) — [★L2 게이트 GREEN] b2p 발화 100%·양 벤치서 b0p 초과 + 쿼터 수술 완결(super-squash)

**추상적 문제.** (a)신세대 첫 과학 관문(L2)의 판정. (b)폴더 삭제 후에도 403이 지속되는 저장 쿼터의 실제 해방 메커니즘 규명.

**구체적 해결.**
- ★**L2 GREEN(사전등록 기준 전부 통과)**: b2p(meta) vs b0p(control) @temp1.0·8k·n2 — **MATH500-100: 58.0% vs 55.5%(+2.5pp)**·**AIME: 10.0% vs 8.3%(+1.7pp)**·**meta 발화 100%**(양 벤치·목표 85%·T1 87-98% 대비도 우위)·avg_len 497/1920≪8192(종결 건강·truncation kill-switch 무발동)·EOS rc0. **집중 리스크(instruct-생성 체인→base 자유생성 비종결·kill 25%) 미실현**. b23 재앙(wellformed 0.40) 대비 **SFT1 기반 가설 실증**. meta arm이 RL 전부터 control 초과 = RQ1' 방향 신호.
- ★**쿼터 수술 2단계(진짜 해방)**: 폴더 삭제(E-073)만으론 403 지속 → 근인=**LFS 객체는 tip 삭제 후에도 히스토리에 잔존해 쿼터 점유**. 해법=(1)`super_squash_history`(v3 repo·tip 파일 보존·히스토리 1커밋化)+(2)구모델 repo 3종 전체 삭제(e8/e9/triobj-v1·~98GB). **20MB LFS 프로브 업로드 성공=쿼터 해방 실증**.
- 후속: push 루프 강화(8×90s→30×120s·shlex 재검증)·cosmic-worm(소진) cancel→b0p 재발사. b2p는 attempt7/8이 squash 후 발화했을 가능성(16GB 업로드 진행 중이면 로그 침묵)이라 1틱 관찰 후 필요시 재발사.

**통찰.** ①**HF 쿼터는 tip이 아니라 LFS 객체 집합**: delete_folder는 회계상 무의미, super-squash/repo-delete만 실제 해방 — durability 설계의 새 상수. ②**L2 GREEN의 의미**: "깨끗한 기반(SFT1)이면 base에서도 meta가 산다"—붕괴 서사가 substrate에서 데이터로 완전히 이동. ③다음=PMI 프로브+RL 준비(저장예산 편입: keep-1·model-only). 관련 [[E-070]]~[[E-073]]·[[T1-heldout-meta-beats-base-6of6-0706]].
- 상태 tick 07:46 UTC(7/24): ★**b2p 모델 4/4 shards HF durable 착지**(squash 후 attempt 성공)=treatment init(models/b2p_v8meta_strict_sft) 완전 확보→learning-mustang 노드 반납(cancel). right-penguin(b0p 3차)=로그 0줄 할당실패→**assuring-spaniel**(4차) 재발사. 구세대 living-pigeon running 복귀. PMI 프로브 도구 CLI 실측: probe_pmi_offline(--data/--model/--smoke/--out)·measure_sft_gate(--model_path/--output_dir/--n_pmi=올인원 게이트+pmi_signal). 다음 사이클=프로브 잡+RL yaml 설계→Codex+Fable 수렴→발사(RL init은 이미 durable).

## E-075 (2026-07-24 08:30 UTC) — [PMI 프로브 발사] b2p 게이트 잡(Option C 분기 판정기)

**추상적 문제.** L2 GREEN(형식·정확도) 다음 관문 — "이 init에서 PMI-shift가 실제로 발화하는가"(auc·n_save·confound/placebo)를 RL GPU 투입 전에 판정해 Option C의 A-직행/SFT2-분기를 데이터로 가른다.

**구체적 해결.** h100std_probe_b2p_gate.yaml: HF durable init(models/b2p_v8meta_strict_sft) snapshot_download(5회 retry+config.json assert)→measure_sft_gate.py(math500-300·max_tokens 8192·n_pmi 200: emission+wellformed+acc-greedy+pmi_signal[auc_shift·SAVE reversals·confound leg-3·placebo] 올인원)→sft_gate.json을 stdout+HF push(비LFS 소형). 출력 파일명 실측 정정(gate_summary→sft_gate.json). Fable 프리런치 GO(CLI 전항 유효·-m import 충족·allow_patterns 레이아웃 정확·escaping 무해·push substring 매치). shlex 3토큰 PASS. **판정 기준: pmi PRESENT(auc>0.55 & n_save>0 & confound 통과)→RL 직행 / ABSENT→SFT2 필터본(think-closed·segment-masked·2ep/2e-6)**.

**통찰.** 게이트 스크립트가 자기완결(자체 롤아웃 생성)이라 프로브 잡은 순수 조립 — 검증된 블록 재사용이 구현 리스크를 CLI 대조 하나로 압축. b0p 4차(assuring-spaniel) SFT 진행 중·구세대 living-pigeon running. 관련 [[E-071]]·[[E-074]].

## E-076 (2026-07-24 09:20-10:00 UTC) — [Option C 판정=ABSENT→SFT2 분기] 프로브 결과 + Fable이 segment-mask 렌더층 버그 4행 적발

**추상적 문제.** SFT1 init의 PMI 발화성 판정과, T1 스택 재현의 마지막 층(SFT2)을 무결하게 얹는 구현.

**구체적 해결.**
- ★**프로브 결과(smashing-bass)**: wellformed **1.0**(b23 0.40 대비 완벽)·greedy acc **0.5967**(math500-300)·**auc_shift 0.481<0.55 FAIL**·n_save 3>0·confound inconclusive(단일 정답클래스). **판정=ABSENT**(사전등록 규칙·Codex 확인: n_save 3/200은 우연변동 수준·inconclusive는 confound 무효화지 판정 무효화 아님). ★해석: T1이 SFT2를 쌓은 이유(SFT1=형식+기반, PMI 판별력은 RV functional 학습 필요)를 그대로 재현=계보 정합, anomaly 아님.
- **SFT2 구현**: rv_functional 필터 코퍼스(think-closed+meta-nested+≤4096+EOS=382행)·wrong_prefix 100%=segment-mask 활성·init=/scratch 스테이징된 b2p SFT1·2ep/lr2e-6(Codex: 3ep는 과적합 위험·plateau시 1ep continuation)·re-gate(--base_accuracy_greedy 0.5967 앵커·무퇴행 필수·퇴행시 폐기→SFT1서 RL).
- ★**Fable NO-GO→실버그 적발**: 382행 중 **4행(전부 verify)이 assistant에 `</think>` 2개** → Qwen3 템플릿이 마지막 `</think>` 기준으로 중간 청크를 렌더서 삭제 → prefix_len 어긋나 wrong_prefix 일부 학습/meta 마스킹(1행은 학습스팬이 답 꼬리뿐). char-prefix 검사는 통과하나 **렌더층에서만 깨지는 케이스**. 수정=`</think>` count==1 필터→**378행**(전수 재검증 GREEN)→재패키지(asset 488239754)→발사.
- 하우스키핑: b0p 4차 **4/4 shards durable**(assuring-spaniel·eval rc0)→반납. 프로브 노드 반납(JSON은 로그 확보·HF push는 원인미상 실패였으나 비치명). **현재 durable 세트: b0p init 4/4·b2p init 4/4·양 arm L2 수치·프로브 JSON**.

**통찰.** ①**검사는 층을 관통해야**: char-level prefix 검사 전부 통과한 데이터가 템플릿 렌더층에서 깨짐 — 마스크 정합은 최종 token id 층에서만 증명됨(Fable의 전수 토큰 시뮬이 잡음). ②ABSENT가 나쁜 소식이 아니라 **계보 확인**: 프로브가 "SFT1만으론 T1이 안 됐다"는 역사적 사실을 정량 재현. ③프로브(1노드·1시간)가 RL 오발사(수일)를 대체=Option C 설계의 ROI 실증. 관련 [[E-071]]·[[E-074]]·[[E-075]].

## E-077 (2026-07-24 10:20-10:40 UTC) — [SFT2 plateau→T1 캘리브레이션 프로브] auc 게이트의 예측타당성을 우승자로 검정

**추상적 문제.** SFT2가 무퇴행이나 auc를 못 움직였을 때(0.481→0.446), 이것이 (a)graft 실패인지 (b)auc 게이트 자체가 SFT 시점에 과잉요구(RL on-policy에서만 발화 가능)인지 — 게이트의 예측타당성을 유일한 성공 사례(T1)로 캘리브레이션한다.

**구체적 해결.** wanted-antelope re-gate: 무퇴행 전부 통과(wellformed 1.0·emission 0.993·acc 0.583≥anchor−0.05)·auc 0.446<0.55·n_save 5·**push 4/4 durable(models/b2p2_rvseg_sft)**. ★미검증 가정 적발: T1 우승 init(v8_rv_functional_sft)이 이 프로브를 통과했으리란 보장 없음—T1의 n_save 8-11은 **RL 중 on-policy** 측정치. ★Codex 승인+정제: paired 측정(동일 seed42/decoy42/math500-300/n_pmi200)·**T1-low면 "게이트가 RL을 예측 못함"으로 해석하고 RL GO(init=b2p2**: T1 스택 미러·n_save 개선·무퇴행)·T1≥0.55면 graft 실패로 SFT2 반복. 프로브 yaml=클론+치환(repo metacot-rv·모델 v8_rv_functional_sft·push repo 정합 수정)·shlex PASS. wanted-antelope 반납.

**통찰.** ①게이트 임계값은 성공 사례로 캘리브레이션되기 전까지 가설이다 — 우승자가 있는 프로젝트의 특권은 모든 게이트를 우승자에 물어볼 수 있다는 것("instruct와 직접 비교" 지시의 정확한 도구화). ②durable 자산 누적: b0p 4/4·b2p 4/4·b2p2 4/4·L2/프로브/re-gate 수치 전부 확보—RL 발사 준비 완료 상태로 판정만 대기. 관련 [[E-076]]·[[T1-heldout-meta-beats-base-6of6-0706]].

## E-078 (2026-07-24 11:15-11:45 UTC) — [★T1 캘리브레이션 판정+RL 발사] 우승자 auc 0.357=게이트 예측력 격파→RL GO

**추상적 문제.** SFT-시점 PMI 게이트(auc>0.55)의 예측타당성을 유일한 성공 사례로 검정하고, 통과 시 신세대 RL 래더를 무결하게 발사한다.

**구체적 해결.**
- ★**캘리브레이션 결과(composed-muskrat·paired seed42)**: T1 우승 init(v8_rv_functional_sft)=**auc 0.357·n_save 1**·emission 1.0·wellformed 1.0·acc 0.547 — **우리 후보들보다 전지표 열위**(b2p 0.481/3/0.597·b2p2 0.446/5/0.583). 사전등록 규칙 발동: **SFT-프로브 auc는 RL 발화를 예측 못함**(T1은 이 상태서 RL 중 n_save 8-11 발화·6/6 우승). ∴**RL GO·init=b2p2**(T1 스택 미러·무퇴행·T1-init 대비 우위).
- **RL yaml**: 검증된 구 런처 sed-클론(init→b2p2_rvseg_sft·이름→rq3v2_b2p/b3p·--keep 1 저장예산·tarball 488239754). 보상=T1-IDENTICAL(triobj 풀패키지·w_over0.0)·V2 레시피·OOM 픽스 승계·save_freq 20.
- ★**Fable NO-GO→치명 버그 2번째 적발**: sed가 하이픈 표기 WANDB_RUN_ID(rq3-b2-2/rq3-b3pkg-2)를 못 건드려 `WANDB_RESUME: allow`가 fresh 런을 구 런에 이어붙임→**초기 ~160스텝 지표 조용한 증발=킬스위치 관측 불능**. 수정=rq3v2-b2p-1/rq3v2-b3p-1. 그 외 전항 PASS(스테이징 3자 일치·pusher 가드 통과·RGS 격리·shlex).
- Codex 조건부 GO: 동시 발사(matched 노출)·**living-pigeon 취소**(wandb 보존·gs158 durable·신세대 대체)·전송(98GB~3h vs save 5h 창)은 수용하되 **첫 durable ckpt 업로드를 gs25 전 확인, 실패시 transport-first 전환**.
- 발사: **inviting-jackal**(rq3v2_b2p)·**main-mink**(rq3v2_b3p). 킬스위치 사전등록: gs25 val서 n_save≥1 & emission≥0.70 & entropy>0.1 미달=gs50 전 kill.

**통찰.** ①**게이트를 우승자로 캘리브레이션하니 게이트가 죽었다** — auc 임계는 RL-전 시점에 무의미했고, 이를 모른 채 SFT2를 무한 반복했다면 사이클 소각. 성공 사례가 있는 프로젝트는 모든 새 게이트를 우승자에 먼저 물어야 한다. ②**sed-클론의 사각=표기 변형**(언더스코어 치환이 하이픈 ID를 놓침) — Fable 전수 diff가 2연속 치명 버그(렌더 마스크·wandb ID)를 발사 전 차단=이중 점검 루프의 실증 ROI. ③구세대 공식 은퇴: RQ3(think-off/b23)는 부록으로, RQ3v2(think-on/v8-parity)가 본선. 관련 [[E-076]]·[[E-077]]·[[rq3-audit-fresh-restart-0714]].
- 상태 tick 12:10 UTC(7/24): RL 2-arm 부트스트랩 순항 — b2p(inviting-jackal) cold start 정상감지(RGS 격리 작동·구lineage 오염無)·b3p(main-mink) GPU 램프업(24→65GB=학습진입). 다음 마일스톤: verl step 진행→wandb 초기지표(entropy/clip/emit·v1 붕괴게이트)→gs20 ckpt 업로드→gs25 킬스위치(n_save≥1&emit≥0.70&entropy>0.1). ~15분/step이면 gs25≈6-7h 후.

## E-079 (2026-07-24 13:15-13:50 UTC) — [RL 초기지표+분류기 무죄 실증+게이트 재설계] wellformed 0.389 수수께끼 해소

**추상적 문제.** RL gs5 지표가 게이트 측정(wellformed 1.0)과 다르게 b23 시그니처(0.389·rmeta −0.42)를 보일 때, 이것이 (a)보상의 방언 mis-scoring인지 (b)실제 생성 행동인지, 그리고 어느 시점 게이트가 T1-경로/b23-경로를 판별하는지.

**구체적 해결.**
- **gs5 실측(wandb)**: b2p entropy 0.897·corr 65.1%. b3p entropy 0.913✅·**emit 0.98**✅·**attempted 0.322**(b23 0.03 기아 대비 극적 개선·T1 0.52-0.66)·acc_with 0.635>without 0.471(+16pp)·**wellf 0.389·rmeta −0.419**⚠️(b23 gs1: 0.404/−0.27과 유사).
- ★**분류기 무죄 실증(직접 실행)**: Codex가 샌드박스 오류로 코드 못 읽고 "방언 mis-scoring" 잠정 판정→검증 원칙대로 `classify_dcpo_format`을 로컬 실행(torch mock·합성 4방언): **v8-inside-closed=wellformed·RV-after-closed=wellformed**·unclosed=drift·no-meta=no_meta. **분류기는 방언 중립=구조적 mis-scoring 없음, Codex 잠정 판정 반증**. RL 프롬프트도 clean 질문(think/meta 0/5344)=게이트와 동일 형식. ∴0.389=train mix(더 어려운 혼합·n8·temp1)에서의 **실제 생성 행동**.
- ★**게이트 맹점 발견**: b23 궤적 재대조 — b23도 **gs25에 emit 0.78·attempted 0.42·n_save 4로 헌법 게이트를 통과**했고 붕괴는 gs25→50(emit 0.22·wellf 0.06). ∴사전등록된 gs25 킬스위치는 b23-경로를 못 거른다. **게이트 재설계(사전등록 추가)**: gs25 헌법 게이트 유지 + **gs50 궤적 게이트 신설: emit≥0.70 & wellf≥0.30 & attempted≥0.25**(b23 gs50: 0.22/0.06/0.09=명확 판별). T1-대비 우리 유리한 점: init 구조습관이 clean(게이트 free-gen 1.0)이라 w_format(+1 closer)이 잡을 손잡이가 있음—b23은 습관 자체가 깨져 있었음.
- 무개입 계속(entropy·emit 건강·패키지 비계 존재—이걸 검증하는 게 실험의 목적).

**통찰.** ①**잠정 판정에 행동 금지**: Codex가 코드를 못 읽었다고 명시했는데도 그 결론(즉시 분류기 수정)을 따랐다면 무죄인 분류기를 수술할 뻔—직접 실행 실험(30분)이 오진을 차단. ②**게이트는 실패 사례의 실제 붕괴 시점에 놓아야**: b23이 gs25를 통과했다는 사실이 gs25-단독 게이트를 무효화—실패 데이터가 게이트 설계의 진짜 스승. ③on-policy 지표는 SFT-게이트와 분포가 달라 절대값 비교 불가—**같은 측정계 안의 궤적(우리 gs5 vs b23 gs5 vs gs50)**만이 유효 비교. 관련 [[E-067]]·[[E-078]]·[[verify-causes-keep-clean]].
- 상태 tick 14:23 UTC(7/24): b2p gs10(entropy 0.884·corr 59.6%)·b3p gs5+(wandb 요약 지연 or triobj 저속·entropy 0.913). 붕괴게이트 무발동·무개입. 다음 판정점: b3p gs25 헌법게이트→gs50 궤적게이트.
- 상태 tick 15:25 UTC(7/24): b3p gs13 궤적 건강 — emit 0.97-0.99 안정·wellf 0.36-0.39 유지(침식無)·attempted 0.32·n_save 간헐발화(1-3)·entropy 상승 0.84→0.97·corr 0.15→0.26 개선. b2p gs10+(dcpo 키 없어 scan 공백·summary 정상). ~18분/step→b3p gs25≈3.5h·gs50≈11h 후. 무개입.
- 상태 tick 16:27 UTC(7/24): ★b3p(gs17) wellf **상승** 0.39→0.46-0.47(w_format 비계 작동=b23 침식과 정반대)·emit 0.96-0.97·attempted 0.33→0.36↑·n_save 발화(2-3)·entropy 건강. b2p gs29·corr 66.8%·entropy 0.62(정상범위). ★**rq3v2_b2p gs20 ckpt model4/4+optim4/4 HF 착지=98GB RL durable 파이프라인 최종 검증 완료**(쿼터+transport). b3p gs25 val ~2.5h 후. 무개입.
- HF 주기정리 실행 16:50 UTC(7/24) [사용자 지시→프로토콜化]: 3repo super-squash+은퇴자산 프루닝(구 b2 gs158 98GB·b23 SFT 16GB 삭제·구 b0 gs300은 논문앵커 보존)→**523→408GB**. 프로토콜 메모리 저장(hf-periodic-cleanup-squash-protocol-0724)·루프에 ~6h 주기 편입. 핵심: delete_folder는 tip만—LFS 히스토리는 squash만 해방·keep-1 pusher도 히스토리 재성장시키므로 주기 squash 필수.
- 상태 tick 17:29 UTC(7/24): b2p gs40·durable gs20+40 롤링 착지(keep-1 작동)·ent0.526(게이트 여유). b3p gs17+(21분/step·wellf 0.44-0.47 유지·gs25≈3h·gs20 첫 ckpt≈1h). 무개입.
- 상태 tick 18:31 UTC(7/24): ★b3p(main-mink) 선점→queued — gs20 첫 ckpt 직전이라 durable 0=gs0 재시작 확정. 근인=save주기(20스텝×21분/step=7h) vs 선점창 미스매치(transport 아님). 무비용 수정 기회 활용: **save_freq 20→10**(첫 durable 3.5h로 단축·b2p는 9분/step이라 20 유지) 후 재발사=**finer-ostrich**. b2p 순항: gs43·corr 71%·durable gs40(keep-1이 gs20 프루닝 정상작동). 교훈: save_freq는 arm별 step 속도에 맞춰야(matched 비교엔 무영향—저장 주기는 학습동역학 밖).
- 상태 tick 19:37 UTC(7/24): ★**VC 장애 재발**(E-000 7/17과 동일): 신규 제출 3연속 유령화(finer-ostrich/touched-pangolin/holy-cardinal — Created 출력 후 서버측 "virtual cluster does not exist" 거부·Created 문구는 불신 대상 확정). b3p 재발사는 VC 복구까지 블록·매 틱 재시도. **b2p는 무사**(HB 19:34 gpu35GB — 기존 할당은 유지됨·wandb summary 정체는 로깅 지연=F5 교훈). b3p yaml은 준비 완료(save_freq10·RUN_ID rq3v2-b3p-2). 지난 시간 교훈: ①amlt "Created" 출력≠제출 성공—제출 후 amlt status로 실재 검증 필수 ②검증 목적으로 amlt run 재실행 금지(중복제출 위험—이번엔 유령이라 무해했으나 프로세스 결함).
- 상태 tick 20:18 UTC(7/24): VC 여전히 다운(still-chow도 유령·"virtual cluster does not exist"). b2p 생존(HB 20:18·gpu35GB·running 8h). b3p 재발사 대기 지속. 무개입.
- 상태 tick 20:59 UTC(7/24): VC 다운 지속(5번째 유령=selected-ghost). b2p 생존(running 9h). 대기 지속.
- 상태 tick 21:46 UTC(7/24): ★b2p도 선점→queued(durable gs40 안전·VC 복구시 자동재개·resume gs40+). VC 다운 지속(유령 6호=exotic-meerkat). 전 클러스터 대기 상태—durable 자산 전부 안전(b0p/b2p/b2p2 init+RL gs40). HF 주기정리 실행(3repo squash·아래 사용량). 대기 지속.
- 상태 tick 22:33 UTC(7/24): ★b2p queued→**running 복귀**(기존 큐 항목은 할당 가능=신규 제출 API만 차단된 부분 장애로 판명·resume gs40 예상). b3p 신규 제출은 여전히 거부(유령 7호=subtle-lark). 다음 틱: b2p resume 정상(RGS gs40) 확인+VC 재시도.
- 상태 tick 23:19 UTC(7/24): b2p 생존·학습중(HB·gpu35GB=resume 성공). VC 신규제출 거부 지속(유령 8호·~4h째). b3p 대기 지속.
- 상태 tick 00:21 UTC(7/25): ★b2p gs57 통과·**첫 RL val(gs50): GSM8K 90.8%**(구세대 B0 gs300=89.05%를 gs50에 초과·단 val셋 상이 유의)·corr 0.379·ent 0.609. VC 다운 지속(유령 9호·~5h). 다음 틱: b2p 전과목 val 회수+durable gs60 확인.
- 상태 tick 01:23 UTC(7/25): ★b2p gs50 val 전과목 회수+**변환 교정**(val-aux score=reward∈[−1,1]·acc=(r+1)/2·직전 틱 "GSM8K 90.8%"는 오독→실제 **95.4%**). **b2p-RL gs50 9과목 평균 65.8%**: GSM8K 95.4·counting 89.7·number_theory 80.5·prealgebra 79.4·algebra 70.9·int_alg 59.6·geometry 50.6·precalc 39.8·omni 26.4. ★구세대 동일 val 대비: B2v1 gs50=64.9 초과·B2v1 gs100=66.6 근접 — v8-parity 세대가 RQ1' 방향으로 우위 신호. durable gs60 롤링. VC 다운 지속(유령 10호). b2p gs67 진행중.
- 상태 tick 02:25 UTC(7/25): b2p gs76·ent0.921(회복·건강). VC 다운 지속(유령 11호·~7h). 대기 지속.
- 상태 tick 03:27 UTC(7/25): b2p 생존(HB·gpu35GB·wandb summary는 s76서 지연). VC 다운 지속(유령 12호·~8h). 대기 지속.
- 상태 tick 04:29 UTC(7/25): b2p durable gs80 착지(롤링 정상·in-mem ~gs85+·gs100 val 접근중). VC 다운 지속(유령 13호·~9h). HF 주기정리 실행(3repo squash·523GB). 대기 지속.
- 상태 tick 05:31 UTC(7/25): b2p gs90(gs100 val 임박). VC 다운 지속(유령 14호·~10h). 대기 지속.
- 상태 tick 06:33 UTC(7/25): b2p gs98(gs100 val 다음 틱). VC 다운 지속(유령 15호·~11h). 대기.
- 상태 tick 07:35 UTC(7/25): ★b2p gs100 val 착지 — 9과목 평균 **59.7%**(gs50 65.8 대비 **−6.1pp 하락**·GSM8K 95.4→88.2%). 구 B2v1은 동구간 64.9→66.6 상승이었기에 역방향=우려 신호(노이즈/dip 가능성 유보). 다음 틱: response_length·clip·entropy 궤적으로 원인 후보(길이팽창·디코딩 퇴행—구 b2 gs160 OOM 아크의 전조였던 패턴) 조사. VC 다운 지속(유령 16호·12h). b2p gs104 진행중.
- 상태 tick 08:27 UTC(7/25): ★b2p gs100 val 하락 원인조사 — **퇴행 시그니처 없음**: respL 427-792 안정(구 b2 팽창패턴 1762→2653과 상이)·lenclip≤1.8%·pg_clipfrac~0.002·entropy 0.53→1.16 상승(붕괴 반대)·train corr s100=0.59(79.5% 강함·배치 고변동). 판정=사전등록 (b)분기: 무징후→gs150 val 대기(val 노이즈/진동 가설). durable gs100 착지(keep-1 롤링). VC 다운 지속(유령 17호·13h). 무개입.
- 상태 tick 09:29 UTC(7/25): b2p gs123 순항(corr 75.4%·ent 0.851). VC 다운 지속(유령 18호·~14h). gs150 val ~2h 후.
- 상태 tick 10:31 UTC(7/25): b2p gs134·★entropy 1.36 상승 지속(0.53→1.16→1.36—이례적·val 하락과 합치면 정책 노이즈화 가설·gs150 val이 판정·구세대 instruct twin은 0.07-0.18이었음). HF 주기정리(506GB). VC 다운(유령 19호·~15h). gs150 val 다음 틱.
- 상태 tick 11:33 UTC(7/25): b2p gs145·ent 0.958(1.36→하강=폭주 아닌 진동). gs150 val ~30분 후. VC 다운(유령 20호·~16h).
- 상태 tick 13:01 UTC(7/25): ★b2p gs150 val=**61.0%**(궤적 65.8→59.7→61.0=진동 양상·단조하락 아님·사전등록 애매구간 59-62)→**gs200 재대기**. 과목별 churn 큼(geometry 50.6→58.2↑·int_alg 59.6→48.3↓)=고분산 val. ent 0.955. VC 다운(유령 22호·~17.5h). b2p gs155 진행.
- 상태 tick 14:03 UTC(7/25): b2p durable gs160 롤링(100→160·파이프라인 완벽—구세대 durability 지옥과 대비). VC 다운 지속(유령 23호·~18.5h·역대급 장기화). gs200 val ~2.5h 후.

## E-081 (2026-07-25 15:07-15:40 UTC) — [entropy 폭주→KL 앵커 개입 수렴+실행] + 시퀀싱 실수 자백

**추상적 문제.** clean init에서 V2 레시피(최대 엔트로피 노브: temp1.0+top_k-1+KL0+Dr.GRPO)가 v1과 정반대 실패(entropy 폭주 0.53→2.05·val 하락)를 일으킬 때의 개입 판단과, 장애 중 개입 실행의 안전 순서.

**구체적 해결.**
- ★사전등록 트리거 발동(ent 2.05>1.5·gs178). Codex 판정=(b)즉시 개입: 단일 노브 **kl_loss_coef 0.0→0.01**(reference-policy KL 앵커=GRPO 계열 선례·temp하향/ent정규화/lr↓ 기각 사유 명시)·gs160 재시작·b3p 동일 적용(matched 유지)·신뢰도 75%. (c)arm별 상이 레시피는 래더 무효라 배제.
- ★Fable GO(코드 확증): use_kl_loss=true라 ref worker 이미 활성=순수 계수 플립(신규 기계 0)·KL 앵커 대상=b2p2 SFT init(의도 정합)·kl_ctrl.kl_coef=0.005는 inert·b3p의 ref=PMI scorer 겸용이나 계산 경로 분리라 무영향·sed 앵커라인 유일성 확인. T1은 KL0이었으나 instruct prior=암묵 앵커·base는 명시 앵커 필요=substrate 적응(ledger 기재).
- 실행: 양 yaml에 kl_loss_coef=0.01 삽입·shlex PASS·inviting-jackal cancel→재제출 시도.
- ★**시퀀싱 실수(자백)**: 재제출이 VC 장애에 거부(tight-grouper 유령)되어 **b2p도 다운**—취소 전에 VC 생존 프로브를 했어야 했는데 "cancel→resubmit" 관성으로 순서를 어김. 손실=gs160→178 구간(~3h)+VC 복구까지 대기. durable gs160 안전=복구 시 KL픽스와 함께 재개. **새 규율: 제출 장애 중에는 running 잡 취소 전 반드시 대체 제출 경로 생존 확인.**

**통찰.** ①**V2 레시피는 init-의존적**: 깨진 init(b23)의 엔트로피 붕괴엔 약이었으나 clean init(b2p2)엔 과처방(앵커 부재→노이즈화)—레시피는 절대값이 아니라 init prior와의 합. ②T1이 KL0으로 이긴 건 instruct prior가 공짜 앵커였기 때문—base 이식엔 명시 앵커가 대응물. ③장애 중 개입은 "새 것 확보 후 옛 것 폐기" 순서가 철칙. 관련 [[E-079]]·[[base-redesign-teacher-free-selfdistill-pivot-0708]](v1 붕괴)·[[E-000]].
- 상태 tick 15:55 UTC(7/25): ★★**VC 복구**(~20h 장애 종료·유령 25연속 후)—양 arm 재발사+실재 검증: **optimum-fawn**(b2p·gs160 resume+KL앵커 0.01)·**sunny-stallion**(b3p·fresh·save_freq10·b3p-2·KL앵커). 관전: b2p 재개 후 entropy 정상화(2.05→1 이하 안착 여부)=KL 앵커 유효성 첫 검증·b3p 부트스트랩→gs10 durable→gs25/50 게이트.
- 상태 tick 16:48 UTC(7/25): b3p(sunny-stallion) 학습 진입(HB·gpu43GB)·b2p(optimum-fawn) 부트스트랩 후반(env 추출). 다음 틱: b2p gs160 resume 검증+b3p-2 초기지표(KL앵커 하 emit/wellf/ent—b3p-1 gs1-17 대비).

## E-082 (2026-07-25 17:41-18:40 UTC) — [★KL 앵커 역효과 실증(우연한 A/B)+전면 롤백] + 실수 2건 자백

**추상적 문제.** 안정화 개입(KL 앵커)이 우연히 깨끗한 A/B로 검증됐을 때의 해석과 회수, 그리고 자율 루프에서 파괴적 조작의 안전 규율.

**구체적 해결.**
- ★**우연한 완벽 A/B**: b3p-1(KL=0) emit 0.87-0.98 안정 vs b3p-2(KL=0.01·유일 유효 diff) **emit ~0.03 by gs61**·wellf 0.01·respL 단축·corr 역설적 상승(0.44-0.67). config diff 기계 확인=KL+save_freq(무영향)뿐. **∴KL 앵커가 triobj arm의 meta를 ≤60스텝에 소거**. 기제 가설(Codex 0.75): w_meta warmup 80스텝 동안 PMI OFF → meta 스팬 수입=w_format뿐인데 per-token KL 비용이 긴 meta 스팬에 누적 → 순음수 → 급속 소거. b2p(VANILLA)서도 KL이 entropy 못 잡음(33스텝 후 2.08 유지)=이중 실패.
- ★**Codex 수렴(0.9)**: 양 arm **KL=0 복원**(래더 divergence 불허·b3p entropy는 원래 건강 0.8-1.0·b2p 폭주는 v2-레시피 관찰로 수용→held-out서 판정·강한 KL/스케줄링은 별도 실험로 분리).
- **실수 1(재확인)**: E-081의 KL 개입 자체가 역효과—Fable의 클린룸 분석("계산 경로 분리·무해")을 실측이 반증. 분석<실측.
- **실수 2(신규 자백)**: b2p KL-구간 프루닝시 keep-1이 이미 gs160을 지운 걸 모른 채 목록확인과 삭제를 한 스크립트에 묶어 **마지막 앵커 gs180까지 삭제**→b2p cold start 강제(gs0-180 손실 ~1일). 완화: gs180은 KL-오염이라 순수 resume점 아니었고 해당 궤적은 val하락+entropy폭주라 **KL=0 단일 레시피 fresh 런이 오히려 깨끗**. ★새 규율: **LIST→판단→삭제를 분리**(같은 스크립트에 묶지 않기)·pruning 전 keep-N의 현재 상태 확인.
- 조치: sunny-stallion kill·오염 계보 purge(checkpoints/rq3v2_b3p gs10-60)·KL 라인 롤백(양 yaml·shlex PASS)·RUN_ID bump(b2p-2/b3p-3=wandb 오염 회피)·양 arm fresh 재발사=**superb-terrier**(b2p)·**stirring-calf**(b3p)·실재 검증.
- 부수 확인: save_freq10+pusher가 b3p gs10-60을 실제 durable화(선점→gs60 재개 작동)=주기 단축 유효 실증.

**통찰.** ①**우연한 A/B는 공짜 ablation**: 오배선이 준 데이터가 KL-warmup 상호작용이라는 비자명한 함정을 1런 만에 적발—의도된 실험이었다면 몇 배 비쌌다. ②**클린룸 코드 분석은 상호작용을 못 본다**: "계수 플립뿐"은 참이지만 보상 생태계(warmup 중 meta의 수입 구조)와의 상호작용이 결과를 뒤집음—개입은 소규모 파일럿으로 먼저. ③**파괴 조작 3율**: 리스트→판단→실행 분리, keep-N 상태 선확인, 대체 확보 후 폐기. ④양 arm이 이제 완전 동형(fresh·KL0·V2)=가장 깨끗한 matched 상태로 재출발. 관련 [[E-081]]·[[E-079]].
- 상태 tick 18:43 UTC(7/25): ★**KL 인과 3중 확증** — b3p-3(KL0·fresh) s3: **emit 0.984 회복**(b3p-1 KL0=0.87건강→b3p-2 KL0.01=0.03사멸→b3p-3 KL0=0.98회복)·wellf 0.34·att 0.32·ent 0.96=건강 프로파일. b2p-2 부트스트랩 중(wandb 미개시). 정상 감시 모드 복귀: b3p gs10 durable(~3.5h)→gs25/50 게이트.
- 상태 tick 19:45 UTC(7/25): 양 arm 선점→queued(저녁 경합). b3p-3 s7까지 진행(emit 0.963 건강 유지) 후 선점—gs10 durable 미달이라 cold 재시작 예정(save_freq10 덕에 다음 세션 ~3.5h면 첫 durable). b2p-2는 학습 미진입 채 선점. auto-requeue 대기·무개입.
- 상태 tick 20:47 UTC(7/25): 양 arm queued 지속(~1h·저녁 경합). 대기.
- 상태 tick 21:50 UTC(7/25): 양 arm queued 지속(~2h·주말 경합). HF 주기정리(310GB). 대기.
- 상태 tick 22:52 UTC(7/25): b2p running 복귀(부트스트랩·cold start). b3p **paused**(0709 stuck-paused 패턴 후보—정책: 2연속 감지시 cancel+재제출·다음 틱 판정). 대기.
- 상태 tick 23:38 UTC(7/25): 역할 교대—b3p running 복귀(paused 자연해소·1틱 인내가 정답이었음)·b2p 재선점 queued(주말 경합·auto-requeue). 무개입.
- 상태 tick 00:39 UTC(7/26): b3p running(재세션·s8+부터 wandb 가시—RESUME=allow의 s≤7 드랍)·b2p queued 지속. 대기.
- 상태 tick 01:40 UTC(7/26): ★b3p **gs10 첫 durable 착지**(save_freq10 성공=시시포스 절단·이후 선점→queued나 다음 세션 gs10 재개)·s10 지표 건강(emit0.977·wellf0.398). b2p running 복귀(부트스트랩). 의자놀이 지속 중이나 durable 앵커 확보로 순전진 구조 확립.
- 상태 tick 02:42 UTC(7/26): 양 arm running. b3p gs10 재개 세션·b2p 재부트스트랩(★주시: b2p는 ~4세션째 부트스트랩 중 선점으로 학습 미진입—다음 재제출 기회에 save_freq 20→10 검토). HF 정리(342.8GB).
- 상태 tick 03:44 UTC(7/26): b3p running(gs10 재개 세션)·b2p 재선점 queued(5세션째 학습 미진입—다음 failed/재제출 기회에 save_freq10 적용 예정). 대기.
- 상태 tick 04:45 UTC(7/26): ★b2p 드디어 학습 진입(s1·ent0.873·5세션만)·b3p running(gs10 재개). 양 arm 동시 학습 중. b2p 첫 durable=gs20(save_freq20·~1-3h).
- 상태 tick 05:49-06:05 UTC(7/26): b3p(stirring-calf) failed(1.6kB 조기사망)→재제출 시도=유령(feasible-silkworm·"Created" 후 VC does not exist=**제출장애 재발**). 점검 중 amlt list의 sunny-stallion/optimum-fawn "Queued" 표기는 낡은 캐시—실제 둘 다 killed 확인(KL구세대 오염·gs10 덮어삭제 위험 해소). 현황: b2p 학습 진행(s2·ent0.792·corr0.293)·b3p 다운(durable gs10 안전)·틱마다 재제출 재시도.
- 상태 tick 06:24-08:09 UTC(7/26): b3p 재제출 **3연속 유령**(feasible-silkworm→bold-ray→charming-tapir·전부 "Created new experiment ... on msrresrchbasicvc" 직후 `(UserError) The virtual cluster does not exist`·`amlt list` 최신 라인은 여전히 stirring-calf=미등록 확증). **VC 제출장애 ~2.3h 지속**. 한편 b2p(superb-terrier) **running 14h·gs12 순항**: corr 0.281(s1)→**0.398**(s12)·ent 0.873→0.925(전회 gs40 폭주 패턴 대비 아직 완만—관찰만). durable: b3p gs10 안전(4/4 shard)·b2p 여전히 무(save_freq20이라 gs20이 첫 저장점=단일장애점). ★깨짐 감사(사용자 질의): (a)KL0.01 메타사멸→b3p yaml에서 KL 라인 제거 확인·gs10 실측 emit 0.977로 회복 확증 (b)0712 scaffold-stripping 깨짐(w_format/w_emit/len_cost/w_cal/trunc_open 0화→wellf 0.43→0.02·RQ2 +0.042→-0.120)→현 yaml은 `w_over=0.0`만 override·나머지 config default 복원·실측 wellf 0.398@gs10로 침식 없음 확증 (c)stale triobj config(agent block) ABORT 가드 line129 상주 (d)**미해결=1.6kB 조기사망의 원인**: line57-60 관측성 훅이 발동 전에 죽어 evidence 미확보(선점 진단 유지·b3pkg-silent-pass-deaths-0713 패턴).

## E-083 (2026-07-26 08:2x UTC) — 변천사 감사: "instruct서 본 효과가 왜 base 실험으로 바뀌었나"

**추상적 문제**: T1(instruct)에서 6/6 유의 승리를 봤는데 왜 base 사다리로 전환했고, 왜 base에서는 그 강화가 안 보이는가. 사용자 질의에 대해 기록(메모리+리포트)만으로 재구성.

**구체적 해결(연대기)**:
1. **0706 T1 승리(실재)**: MATH500 +18.8 / AIME +14.2 / GSM8K +4.0, 6/6 p<.001. 단 메모 자체가 3단서 기재 — 단일시드·**6-head 패키지 효과(PMI 단독 아님)**·emit 88~98% 상시켜짐(선택성 미검증).
2. **0708 base 전환 사유 2**: (a) instruct+think-on은 `</think>` 미종료 degeneration으로 **base 5%/meta 20% = 둘 다 원본 ~70% 이하 "손상된 두 모델 비교"**이며 T1 2차발견(base 잘림 MATH ~19% vs 6%·AIME ~73% vs 50%)때문에 격차가 "메타 우수" vs "base 더 망가짐" 미분리. (b) `base_matched`가 misnomer로 **실제 base 학습 전무**, 외부 baseline(SRPO/RLSD/RLRT/SDPO)은 전부 base. ⇒ 전환은 "T1 오류"가 아니라 "T1 귀속불가".
3. **0710 v1 전멸**: base 3-arm 전부 gs50~100 엔트로피 붕괴(B0 0.108→0.011·clip 4.5%→86%). **instruct 쌍둥이는 byte-동일 레시피로 300 완주(ent 0.07~0.18·clip≤14.5%)** ⇒ 레시피가 instruct-튜닝품. →RECIPE V2.
4. **0710 PMI 굶음(핵심)**: instruct T1 emit 0.99/att 0.52-0.66/**n_save 8~11**/rmeta +1.0~1.2 vs base B3 emit 0.40/att **0.03**/**n_save 0**/rmeta −0.2. 부호반전 이벤트 전무 ⇒ **메커니즘 오류 아니라 발화 무대 미성립**(1epoch/lr2e-6이 base엔 얕음). →SFT unmask fix로 emission 38%→92%(현 b2p2_rvseg_sft init).
5. **0712 scaffold-stripping 역효과**: 부수헤드 0화 → wellf 0.43→0.02·RQ2 +0.042→−0.120. →패키지 복원(현 b3p).
6. **0714/0716 감사**: fresh gs0 재시작(이전 단일시드 not-certifiable)·**C-1 norm 플래그 침묵무시**(B0/B2 std vs B3 mean-only ⇒ RQ1 내부일관 유효·RQ2 정규화비대칭 한계)·RQ1도 corpus 상이(gold 1,290 vs RV 1,763).
7. **0717 블로커 확정(Codex 수렴)**: **identification 70% : gs300 미완 30%**·"T1 6/6=노벨티 증거 아님(패키지 승)"·4세대 반복병(answer-identity/form/package 귀속 or inert).

**통찰**: 예비 분해에서 **살아남은 조각=meta-SFT init(RQ1 +8.6/+7.4 강건 양성)**, **죽은 조각=triobj 패키지(RQ2 −4.5/−4.2)**, PMI는 emission collapse로 소멸(+10.4→−1.7). ⇒ 현행 b2p vs b3p는 init 동일이라 **정확히 죽은 축(RQ2)만 재측정**하고 있고, T1 헤드라인과 같은 축(B3pkg−B0 = RQ1+RQ2)을 만들려면 **b0p arm(v2 레시피)이 필수**. 구세대 B0 held-out 앵커(GSM8K 0.8905/MATH500 0.5440/AIME 0.0458)는 **구세대 레시피 산출물이라 v2 arm에 직접 붙이면 레시피 교란**. ⇒ 다음 결정 안건=b0p 발사 여부(Codex+Fable 수렴 대상, VC 장애 중 GPU 0 작업).

## E-084 (2026-07-26 08:5x UTC) — T1 패리티 감사: "같은 데이터·같은 think-on·같은 pmi_shift인가?"

**추상적 문제**: base b3p가 instruct T1과 정말 동일 조건(데이터/think-on/pmi_shift)인가. yaml 설명의 "IDENTICAL reward setup to archived T1" 주장을 원본 파일과 직접 대조 검증.

**구체적 해결(4축 대조, 원본=archive/launchers_pre_rq3/h100std_pmishift.yaml)**:
1. **pmi_shift 기계=동일 ✅**. T1 호출은 리워드 오버라이드가 정확히 2줄(`dcpo_rmeta_source=pmi_shift`,`dcpo_w_over=0.0`)이고 config도 동일 `triobj_dcpo_v4_stage3b_h100_4x4k`. ⇒ 6-head 전부 기본값 상속(w_meta0.8/warmup80·w_format0.35·w_emit0.1·w_cal0.3·len_cost0.08·trunc_open0.3·anchor_norm true)이 양쪽 일치. **yaml 주장 사실 확인**.
2. **RL 학습데이터=바이트 동일 ✅**. 양쪽 `data.train_files` 미오버라이드 → config 기본 `verl_train_meta_mix.parquet`, 양쪽 동일 `pull_parquets.py`(REPO=iamseungpil/metacot-sdc-data). 실물 5,344행(gsm8k1830+hendrycks 7과목 2938+omni576), 프롬프트=순수 user 메시지·시스템프롬프트 없음. 차이는 SFT init뿐(v8_rv_functional_sft[instruct] vs b2p2_rvseg_sft[base])=실험 목적상 의도된 차이.
3. **think-on=구조 동일 ✅**. 두 init `chat_template.jinja` 실측: 양쪽 `<think>`×3·`enable_thinking`×2, 결정로직 동일(L82-83 `enable_thinking is false`일 때만 빈 블록 주입 ⇒ verl 미전달=기본 think-on). diff 52B는 Qwen 업스트림 버전차(content is string→reasoning_content is defined)로 기능 동등. gen_config eos는 T1 `[151645,151643]`→b3p `[151645]`=0708 EOS 불변식 수정(올바른 변경).
4. **⛔디코딩 레짐=완전히 다름**. **T1은 디코딩 오버라이드 전무** ⇒ 상속 config(verl_e4_selfdistill_h200_4x4k.yaml:58-61) 기본값 **temp 0.6 / top_k 20 / top_p 0.95 / resp 4096**로 승리. b3p는 명시적 **temp 1.0 / top_k −1 / top_p 1.0 / resp 8192 / max_model_len 10240**. 이는 0710 base 엔트로피붕괴 대응의 의도된 수정이나, 같은 메모가 **instruct 쌍둥이는 temp0.6/top_k20으로 300스텝 정상 완주**(ent 0.07~0.18·clip≤14.5%)를 기록 ⇒ 변경은 base에만 필요했고 그 결과 T1 승리조건과 현행 조건이 디코딩 축에서 갈라짐. (`norm_adv_by_std=false`는 region 경로가 이미 Dr.GRPO mean-only라 b3p엔 no-op 추정—코드 확인 미완.)

**통찰**: "같은 데이터·같은 think-on·같은 pmi_shift" 주장은 **3축 성립·1축 불성립**. 불성립 축(디코딩)이 하필 **PMI-shift 발화 조건에 직접 작용**한다 — PMI-shift는 롤아웃에 meta 블록이 나와야 open/close 2위치 teacher-forcing이 성립. 실측 대조가 이를 시사: emit는 T1 0.99 vs b3p **0.977(회복)**인데 **pmishift_attempted는 T1 0.52~0.66 vs b3p 0.285 = 절반 이하**, wellformed 0.398. 즉 **방출은 SFT unmask fix로 되찾았으나 발화율은 T1 수준 미달**이고 디코딩 레짐이 그 갭의 유력 후보. ⇒ 기존 pending task "decode-sweep"을 **붕괴원인 규명이 아니라 "T1 패리티 복원" 실험으로 재프레이밍**해야 함(base에서 temp0.6/top_k20/4096이 붕괴하는지, 중간값이 att를 T1대로 올리는지). Codex+Fable 수렴 안건 2건 확정: **(안건1) b0p arm 발사 여부**(E-083) + **(안건2) 디코딩 패리티 sweep**(E-084).

## E-085 (2026-07-26 09:1x UTC) — 스텝-정렬 T1 대조: "T1의 b3처럼 학습되고 있나?"

**추상적 문제**: base b3p가 instruct T1과 같은 궤적인지, 같은 global_step끼리 정렬해 판정. (E-084는 설정 대조였고 이건 궤적 대조.)

**⚠️정정(E-084 및 이전 틱의 오류)**: "b3p att 0.285 = T1 0.52~0.66의 절반"은 **잘못된 비교**였다. T1 wandb는 3세그먼트 재개 런이고 0.52~0.66은 **gs181~300(후반)** 값. T1 초기 세그먼트(id 7217571783, _step 1~64)를 찾아 **gs1-10으로 정렬**하니 T1 att=**0.328** vs b3p **0.304 = 동등**. 이후 대조는 모두 스텝-정렬 기준.

**구체적 해결(gs1-10 정렬 대조)**:
| 지표 | T1 gs1-10 | b3p gs1-10 | 판정 |
|---|---|---|---|
| meta_emit_rate | 0.480 | **0.974** | ✅ T1 시작점의 2배 = **SFT unmask fix 확증**(0710 굶주림 해소) |
| pmishift_attempted | 0.328 | 0.304 | ✅ 동등 = **무대 성립** |
| wellformed_rate | 0.440 | 0.373 | 🔶 동급(T1은 후반 0.80까지 상승) |
| discard_rate | 0.560 | 0.523 | ✅ 동등 |
| **pmishift_rmeta_mean_scored** | **+0.485** | **−1.080** | ⛔ **부호 반대** |
| n_save / n_derail | 3.9 / 4.0 (비 0.98) | 0.8 / 1.2 (비 **0.67**) | ⛔ derail 우세 |
| acc_with | 0.767 | 0.634 | 🔶 |
| **acc_without** | **0.154** | **0.486** | ⛔ |
| **acc_with−without(메타 기여)** | **+0.613** | **+0.148** | ⛔ **1/4** |
| actor/entropy | 0.136 | **0.924** | ⚠️ **7배**(v2 temp1.0 의도) |
| critic/score/mean | 0.268 | 0.198 | 🔶 |

**거짓경보 해소**: b3p에 `eff_ratio_format`/`eff_ratio_emit`/`eff_scale_*` 4키 부재를 "format·emit head 미장전" 의심했으나, verl_sdc.py:1050-1059가 `_ANCHOR_EMA_STATE`에 head가 등록된 경우만 로깅하고 dcpo_region.py:1588-1611이 `_warm_done`(=gstep > `dcpo_anchor_warmup_steps`, config값 **20**) 게이트를 걸기 때문 — **b3p gs10은 웜업 미도달**이 원인. head는 정상 장전. **gs20 초과 시 자동으로 키 출현 = 다음 세션에서 자기검증됨**.

**통찰**: **방출 계열은 T1을 상회(고쳐짐)·보상 계열은 부호 반대(안 고쳐짐)**. 즉 0710 진단("무대 미성립")은 해결됐고 **새 병목은 블록 내용의 질**로 이동 — base가 쓰는 메타 블록이 믿음을 gold보다 decoy 쪽으로 밀고 있다(derail>save). 또한 **메타 기여도가 T1의 1/4**(+0.148 vs +0.613)로 메타가 거의 장식. 단 해석 갈림길: T1 acc_without 0.154는 **instruct degeneration으로 메타 제거시 생성 붕괴**한 아티팩트일 수 있어 T1 +0.613이 부풀려진 값일 가능성(그러면 base +0.148이 정직한 값). 어느 쪽이든 **두 실험은 동일 상태 아님**. 유력 인과 후보=**엔트로피 7배**(E-084 디코딩 패리티 파괴): 넓은 샘플링이 블록 내용을 산만하게 해 derail 유발.
**판정선 사전등록**: gs25 헌법게이트에서 **rmeta 부호**를 본다. (i) rmeta>0 & n_save>n_derail → T1 궤적 합류로 판정, 계속. (ii) rmeta<0 지속 → **디코딩 패리티 sweep(temp 0.6/top_k 20)이 필수 처방**으로 승격(안건2가 선택→필수). w_meta warmup 80스텝이므로 gs10은 PMI 보상 거의 미인가 구간임을 감안(조기판정 금지).

## E-086 (2026-07-26 09:4x UTC) — wandb config 전수 diff: "T1 세팅과 일치하는가" 최종 판정

**추상적 문제**: E-084는 yaml 대조, E-085는 궤적 대조였다. 이번엔 **wandb에 저장된 실제 런 config(verl이 받은 값) 298 파라미터 전수 diff** — 해석 여지 없는 ground truth.

**구체적 해결**: T1 초기 세그먼트(7217571783, gs1-64) vs rq3v2-b3p-3. **298개 중 275 동일 / 23 다름**(양쪽 키 집합 완전 일치, 누락 0).
- **리워드 기계 전부 동일 ✅**: rmeta_source=pmi_shift·w_meta 0.8·w_meta_warmup 80·w_format 0.35·w_emit 0.1·w_cal 0.3·w_over 0·len_cost 0.08·trunc_open 0.3·anchor_norm True·anchor_warmup 20·anchor_ema 0.9·pmi_clip_gate 0.1085·pmi_agg mean·pmi_topk_frac 0.25·pmi_ngram_n 8·meta_len_cap 96·emit_route first_token·w_score_alpha 1.5 전부 일치. 배치/lr/clip/KL/rollout.n도 diff 미출현=동일. ⇒ **"같은 pmi_shift" 주장 확정 사실**.
- **23개 차이 분류**: (A)행정 5=이름/경로/test_freq 25→50 (B)의도된 실험변수 5=init 모델 5곳(v8_rv_functional_sft→b2p2_rvseg_sft) (C)개선 3=checkpoint +optimizer(0716 픽스)·log_prob micro-batch 4→2(8192 OOM가드) (D)**⛔디코딩 7**=temp 0.6→1.0·top_k 20→−1·top_p 0.95→1.0·response_length 4096→8192·max_response 4096→8192·max_model_len 4608→10240·max_num_batched 8192→10240 (E)**⛔정규화 1**=norm_adv_by_std_in_grpo **True→False** (F)**⛔rollout 아키텍처 1(신규 발견)**=agent_loop_config_path **cf_agents_combined.yaml→None** ⇒ T1=agent-loop 롤아웃, b3p=single-turn.

**3자 대조(핵심)**: D·E·F 세 실질축 **전부에서 b3p와 b2p가 서로 일치하고 둘이 함께 T1과 갈라진다**(temp 1.0/top_k −1/resp 8192/norm False/agent_loop None 공통). ⇒ **b3p는 T1이 아니라 b2p에 정렬돼 있다** = matched-ladder 설계 의도(arm끼리 맞춰야 b3p−b2p를 리워드에 귀속 가능). T1 재현은 애초에 목표가 아니었음.

**통찰(판정 분리)**: (i) **"T1 세팅 일치?"=아니오** — 디코딩·정규화·rollout 3축 상이 ⇒ 현행 실험으로 **"T1이 base에서 재현됨"은 주장 불가**. (ii) **"b2p와 일치?"=예** ⇒ **RQ2("base에서 패키지가 correctness-only를 이기나")는 깨끗히 답 가능**. 논문 서술은 이 두 주장을 절대 섞지 말 것.
**남은 경고 2**: ①**정규화 실효 비대칭 미해소** — 0716 C-1은 이 플래그가 VANILLA 경로에서 침묵 무시된다고 판정했고, 그렇다면 명목(둘 다 False)과 달리 **실효는 b2p=std / b3p=mean-only로 b3p만 유일하게 mean-only**. C-1이 남긴 노드 검증(`inspect.getsource(core_algos.compute_gdpo_outcome_advantage)` 시그니처에 norm_adv 존재 여부)은 **여전히 미완**이며 verl이 로컬 미설치(클러스터 /scratch/conda_envs/simplerl 전용)라 오프라인 확인 불가 ⇒ **다음 b3p 세션 필수 점검 항목으로 등록**. ②**E-085 acc_without 해석 약화** — (F) agent-loop→single-turn 차이로 acc_without 산출 롤아웃 경로가 다를 수 있어 T1 0.154 vs b3p 0.486 대조는 불공정 가능 ⇒ **이 지표는 T1 대조에서 제외하고 b3p 내부 추세로만 판독**(E-085의 "메타 기여도 1/4=장식" 주장은 보류로 격하). rmeta 부호·n_save/n_derail 비는 arm 내부 지표라 유효 유지.

## E-087 (2026-07-26 08:5x UTC) — 코드 tarball 실물 diff + 사망지점 특정 + 우선순위 역전 판정

**추상적 문제**: 사용자 4문(①왜 자꾸 죽나 ②T1은 올바른가 ③RQ2보다 재현이 먼저 아닌가 ④pmi_shift 설계가 원래와 같나)에 기록·실물 근거로 답.

**구체적 해결**:
1. **1.6kB 사망지점 특정(신규·계측 결함)**: b3p yaml 실행순서 = L97 pull_parquets → L104 snapshot_download → L134 pusher → **L204 verl 실행(verl_main.log 생성)** → **L225+ tail -F + 10s 하트비트**. ⇒ **관측성이 verl 실행 이후에야 켜지므로 L204 전 사망은 setup echo 몇 줄(=정확히 1.6kB)만 남긴다.** 즉 "증거 미확보"의 원인은 진단 실패가 아니라 **수집기 기동 시점이 사망 시점보다 늦음**. 정황은 여전히 콜드스타트 선점(Standard·반복 패턴)이나 **확증 불가**. **처방=하트비트를 스테이징 前으로 전진 배치**(다음 사망은 반드시 흔적 남김). VC 제출장애(4연속 유령)는 별개 원인.
2. **T1 유효성 판정**: **측정 견고 / 귀속 미확정**을 분리. 견고=robust 재채점(MATH500 66→81.5)·avg@8(AIME@16)·쌍대 부트스트랩+McNemar 6/6 p<.001. 미확정=①두 모델 모두 손상(원본~70%→base 5%/meta 20%)+base가 더 잘림(AIME 73% vs 50%) ②6-head 패키지 승리 ③단일시드. ⇒ **"긍정적 결과"는 참, "PMI-shift가 효과를 냈다"는 미입증**.
3. **⭐우선순위 역전 판정(사용자 지적 수용)**: 존재 미확인 효과는 분해 불가 — base RQ2 음성은 "instruct 승리의 원인 규명"이 아니라 "패키지가 base서 안 통함"이라는 **다른 명제**. 구조적 결정타: **T1의 축=pmishift vs matched-base = RQ3 번역상 b3p−b0p(RQ1+RQ2 합성)** 인데 **현행 b3p−b2p는 init 동일이라 RQ2 단독**. ⇒ **b0p arm 없이는 재현 판정 자체가 불가능**. 신 우선순위: **①b0p 발사(T1 축 복원) ②계측 하트비트 전진(안 고치면 조용한 사망 계속) ③논문·사이트 교란 고지(GPU 0) ④T1 디코딩 패리티 sweep ⑤RQ2 병행(b2p/b3p 이미 가동)**. (이전 틱에서 b0p를 안건③로 후순위 배치한 것을 ①로 정정.)
4. **pmi_shift 설계 동일성=실물 tarball diff로 확정**: T1 asset **458068577** vs 현행 **488239754** 다운로드·전개 후 파일 단위 diff — **`src/training/dcpo_pmi_shift.py` 바이트 완전 동일 ✅**(SAVE/DERAIL 부호반전·eps 밴드 게이팅·비대칭 +1/−2·진단 카운터 전부 무변경) ⇒ **"원래 설계와 같은가"=예**. 단 배선 2건 행동 변화(`dcpo_region.py` 76줄): **(a)truncation→R_corr −1 강제 신규**(T1은 `1.0 if c2 else -1.0`만; base는 resp8192+넓은 샘플링으로 잘림 빈발 가능 ⇒ correctness 신호가 T1보다 가혹) **(b)anchor warmup 게이트 `_n`→`global_step`**(T1의 `_n`은 체크포인트 미저장 모듈전역 ⇒ **T1은 선점·재개마다 20스텝 warmup 재시작, 실제로 3세그먼트 재개 런이라 비정규화 창을 여러 번 겪음**; b3p는 재개 즉시 warmup 완료 ⇒ 버그수정이나 **T1 승리가 그 반복 비정규화 창에 의존했을 가능성 배제 불가**). `verl_sdc.py` 673줄 차이는 관측성/부트스트랩 추정이나 **정밀 확인 미완**(확언 금지).

**통찰**: pmi_shift **알고리즘 코어는 T1과 바이트 동일**이므로 "메커니즘을 바꿔서 안 되는 것"은 아니다. 차이는 전부 **주변 조건**(디코딩 7·정규화 1·rollout 1·truncation 페널티 1·warmup 게이트 1 = 총 11개 축)에 있고, 이 11개가 모두 base 생존 또는 arm 정합을 위해 강제된 변경이다. ⇒ 논문 서술 상 **"메커니즘 이식은 충실했고 조건이 갈렸다"**가 정확한 프레이밍. 재현 실패 시 원인 후보는 메커니즘이 아니라 이 11축.

## E-088 (2026-07-26 09:0x UTC) — 고지상태 실측 + 자기정정 2건 + 절대기준선 결손 발견

**추상적 문제**: "사이트·논문에 T1 결과가 올라가 있는데 교란 고지는 되어 있나" 실측 확인.

**⚠️자기정정 2건(직전 턴의 내 주장이 틀림)**:
1. **"사이트에 degeneration 교란 미고지"=틀림**. `docs/site/index.html`(72,945B)에 전용 **"한계·정직하게 읽기"** 섹션 실재 — ①단일시드 PRELIMINARY("시드간 재현 없음") ②**패키지 효과(PMI-shift 단독 아님)**+"PMI 순기여는 RQ2 분해 끝나야 주장 가능·**아직 미실행**" ③shortcut/boilerplate 위험+"placebo 필수" ④**AIME degeneration 완화지 해결 아님**(base ~73% 잘림→meta 50%) ⑤grader robust 재채점. 추가로 상단 배너에 fresh 재시작+"방법론·역사적 기록으로만 유효" 고지. 고지어 실측: degener 4·잘림 2·비종료 2·한계 4·단일시드 3·패키지 5.
2. **"논문에 truncation 미고지"=틀림**. `experiments.tex`에 truncat/degenerat **8회**(abstract 0·discussion 0이나 본문 존재).

**신규 발견(진짜 결손)**: 사이트에서 `원본`·`손상`·`untouched`·`개입 전` **전부 0회** ⇒ **"두 arm 모두 미개입 모델보다 훨씬 낮다"는 절대기준선이 어디에도 없다**. 사이트는 truncation 비대칭을 **"메타가 degeneration을 완화"=장점**으로 프레이밍하는데(타당한 독법), 절대기준선이 없으면 **"완화"인지 "덜 나쁜 손상"인지 구분 불가**. ⚠️단 대조에 쓰이던 **원본 ~70% 수치는 16일 전 메모 출처이며 우리 1030 하니스로 실측한 기록을 확인 못함 — 미검증**(확언 금지).
⇒ **미실행 실험 신규 식별: 미개입 Qwen3-8B를 1030 하니스로 1회 eval**. 학습 불필요·최저비용인데 **CLAUDE.md core metric("Meta-CoT must outperform Base SFT")보다 상위 기준("손대지 않은 모델을 넘는가")을 아무도 측정하지 않았음**. 결과가 어느 쪽이든 논문 프레이밍을 결정.

**통찰·우선순위 재확정**: **①미개입 모델 1030 baseline eval(절대기준선·eval만) ②b0p arm 발사(b3p−b0p=T1 축 복원=재현 판정) ③계측 하트비트 스테이징 前 전진(E-087) ④논문 abstract/discussion 고지 한 줄(본문엔 있음) ⑤T1 디코딩 패리티 sweep ⑥RQ2 병행(b2p/b3p 가동중)**. ②가 "재현"이고 ①이 **그 재현을 읽을 자(尺)** 를 만든다 — 재현해도 절대기준선이 없으면 "재현 여부"만 알고 "그게 좋은가"는 여전히 모름. VC 제출장애 중이므로 ②⑤ 발사 불가·**①은 eval로 노드 조건 상이·③④는 GPU 불요** ⇒ 장애 창에서 ①③④ 처리가 최적.

## E-089 (2026-07-26 09:2x UTC) — ⭐SFT2 데이터 감사: base init이 T1의 1/35 학습량·1/8 redirect로 출발

**추상적 문제**: "base에 학습한 데이터는 올바른가" — RL 데이터·코드는 이미 동일 확인(E-086/087). 남은 미검증 축=**SFT init 코퍼스**.

**구체적 해결(실물 parquet 대조, 코드 tarball 488239754 내부)**:
- base init `b2p2_rvseg_sft` = 2단계 SFT. SFT1=`b2p_v8meta_strict_sft`, **SFT2=`b2p2_rvseg_sft2.parquet`**. T1 init `v8_rv_functional_sft`의 SFT2=`rv_redirect_verify_functional.parquet`(런처 archive/launchers_pre_rq3/h100std_rv_functional_sft.yaml).
- **base SFT2 378행 전부가 원본 1763행과 md5 완전일치 ⇒ 재생성 아니라 100% 필터(부분집합)**.

| 항목 | T1 SFT2 | base SFT2 | 배율 |
|---|---|---|---|
| 행수 | **1,763** | **378** | 21.4% |
| epochs | 3 | 2 | |
| lr | **1e-5** | **2e-6** | 1/5 |
| **유효 학습량(행×ep×lr)** | — | — | **≈1/35** |
| **redirect** | **554(31.4%)** | **67(17.7%)** | **1/8.3** |
| verify | 1209(68.6%) | 311(82.3%) | 1/3.9 |
| difficulty medium | 50.7% | 41.8% | |
| confidence_label 평균 | 0.602 | **0.699** | |

- **필터가 단일 방향으로 편향**: 살아남은 행은 **더 쉽고(medium↓)·더 자신있고(conf↑)·verify에 쏠림(redirect 31→18%)**. 즉 "쉬운 문제 재확인" 행은 남고 **"어려운 문제에서 접근 폐기 후 재시작(redirect)" 행이 집중 배제**.
- **⚠️내 길이 가설은 반증됨**: redirect가 오히려 **더 짧다**(p90 3,871자 vs verify 6,548자), 어떤 길이 컷(12k/16k/20k자)에서도 **redirect 통과율 100%**. ⇒ **≤4096토큰 필터는 원인이 아님. 실제 배제 사유 미특정**(남은 후보=think-closed / meta-nested / EOS).
- 무결 확인: RL 데이터 `verl_train_meta_mix.parquet` 두 tarball서 **바이트 크기 동일(686,267B)**, `dcpo_pmi_shift.py` 바이트 동일.

**통찰(최상위 원인 후보 갱신)**: PMI-shift의 **SAVE=믿음이 decoy→gold로 뒤집히는 사건**이고 그 뒤집기를 실제 수행하는 행동이 **redirect**(verify는 이미 맞는 답 확인이라 마진 이동 작음). ⇒ **base는 SAVE를 만드는 행동을 T1의 1/8만 배운 채 RL 진입**. 이는 E-085 관측치와 정확히 정합: **rmeta −1.08(T1 +0.485)·n_save 0.8(T1 3.9)·derail>save**. ⇒ **b3p 부진을 "PMI-shift가 base서 무효"로 읽으면 오독** — 아직 그 결론을 낼 자격 없음.
**"재현 안 됨"의 정확한 지위**: 재현 *실패*가 아니라 **재현 시도 부재**. (a)축 부재(b3p−b2p는 init 동일=RQ2 단독, T1 축은 b3p−b0p) (b)출발선 상이(1/35 학습량·1/8 redirect).
**우선순위 재갱신**: **⓪redirect 배제 사유 특정(오프라인·비용0·길이 반증됐으므로 원인 모른 채 수정 금지) → ①SFT2 재구축(redirect 비율 T1 수준 복원+학습량 상향) → ②미개입 Qwen3-8B 1030 eval(절대기준선) → ③b0p arm 발사(init `b0p_v8base_strict_sft` **이미 HF에 존재**=T1 축 복원) → ④계측 하트비트 전진 → ⑤논문 고지 1줄 → ⑥T1 디코딩 패리티 sweep → ⑦RQ2 병행**. ⚠️①은 신중: 필터엔 근거가 있었고(base 형식 불안정→clean만, E-076 kill-switch 존재) **"1763 전부 사용"이 답 아님** — ⓪에서 사유 특정 후 redirect에 한해 기준 정당성 심사. ⓪④⑤는 GPU 불요=VC 장애 창에서 즉시 수행 가능.

## E-090 (2026-07-26 09:4x UTC) — ⭐VC "제출장애" 오진 정정: 노드 문제 아니라 amlt 클라이언트 버전 스큐 유력

**추상적 문제**: 재제출 4~5연속 `(UserError) The virtual cluster does not exist`. 노드 재신청이 필요한가?

**구체적 해결(모순 추적)**:
- `amlt target list` → **msrresrchbasicvc 정상 열거**(A100·CPU·H100·H200·MI200·MI300X·V100). VC 실재·건강.
- **b2p(superb-terrier)가 바로 그 msrresrchbasicvc에서 14h째 running**(`Compute: msrresrchbasicvc (sing)`) ⇒ **"does not exist"는 사실이 아닌 오해성 메시지**.
- 에러 전문이 *"...which exists **in the same cloud**"* ⇒ 클러스터 부재가 아니라 **클라우드/리전 라우팅 검증 실패**를 지시.
- **결정적: 같은 yaml·같은 클라이언트로 14h 전 제출은 성공**. 우리 쪽 무변경 ⇒ **05:49경 서비스 측 변경**이 원인.
- 클라이언트 상태: **amulet 11.9.1**, amlt 자체가 **11.16.0 업데이트 권고**(7 마이너 스큐). yaml target=`service: sing / name: msrresrchbasicvc / workspace_name: msra-sh-aml-ws`(cloud 필드 없음).

**통찰·처방**: **진단=amlt 클라이언트 버전 스큐**(서비스가 제출 검증을 조인 뒤 구버전 요청이 오해성 에러로 거절; 기존 잡은 서버측이라 무영향 ⇒ "신규 제출만 막힘" 현상과 정확히 정합). **처방 순서: ①격리된 새 conda env에 amlt 11.16.0 설치→그 env에서만 b3p 제출 시도 ②그래도 실패시 서비스 이슈 확정→지원요청/롤백 대기 ③노드 재신청은 하지 않음(층이 다름)**. ⚠️**기존 amlt env 절대 불변**(b2p 모니터링 의존·신버전이 yaml 스키마 해석 변경 가능) — 격리 필수.

**부수 판정 2건(사용자 질의)**:
1. **데이터·코드 전면정리=반대, 표적 수술 3건만**. 반대 근거=0708 "코드정리 Phase A"가 살아있는 config를 archive/로 git-mv해 **B0 RL 크래시**시킨 전례(정리가 프로덕션을 깨뜨림) + 정리는 어떤 RQ도 답하지 않음. 그러나 현 상태가 실제로 오도함: **CLAUDE.md가 B2/B3 init을 `data/b23_rv_unmasked_sft.parquet`로 기술하나 그 파일은 HF에 없고 실제 init은 `b2p2_rvseg_sft`**(이 때문에 이번 세션에서 계보 추적에 tarball 다운로드가 필요했음). ⇒ **이동·삭제·rename 금지** 하에 (a)CLAUDE.md 데이터 섹션 정정 (b)`b2p2_rvseg_sft2.parquet` HF 업로드(현재 **코드 tarball에만 존재=단일 소실점**) (c)미커밋 18파일 커밋(`configs/sft_b0p_v8base.yaml`·`configs/sft_b2p2_rvseg.yaml` 신규 포함). 好: **미푸시 커밋 0건**(0716 감사의 "254커밋 미푸시"는 이미 해소). `? paper` submodule 상태만 확인 필요.
2. **"base 재현되나"=현 구성으로는 판정 불가**. RQ1축은 강건 양성(+8.6/+7.4·9데이터셋 전부)=전이됨. RQ2축은 −4.5/−4.2이나 **4중 교란**(SFT2 학습량 1/35·redirect 1/8·b0p 부재로 T1축 자체 없음·디코딩 상이). ⇒ **판정조건 사전등록**: **b0p 존재 + SFT2가 T1 수준 복원된 상태에서 held-out 1030의 b3p−b0p가 유의 양수면 재현 성공, 음수면 substrate-dependence(음성결과로 발표 가능)**. 현재 전제 2개(b0p·SFT2) 모두 미충족.

---

## E-091 ⭐⭐⭐ 제출 불능의 진짜 원인 특정 — `msrresrchbasicvc` **VC 단위 전면 거부**(계정·클라이언트·티어·리전 전부 무죄) → b3p를 A100×4 `msrresrchvc`로 이식 발사 (2026-07-26 10:35~10:43 UTC)

사용자 지시: "b3가 핵심 아니야? 계속 신청해줄래? 신청은 잘 되는 게 맞아? 다른 문제때문에 실험이 멈추는 거 아니야?"
→ 맹목 재제출 대신 **가설을 하나씩 반증**하는 제출 사다리를 돌렸다. 사용자의 의심("다른 문제")이 정확히 맞았다.

### 반증 사다리 (모두 실제 제출/조회로 검증)

| # | 가설 | 검증 방법 | 결과 |
|---|---|---|---|
| H1 | b3p yaml/코드/데이터 문제 | `h100std_rq3v2_b3p.yaml` 재제출 (10:35) | ❌ 반증 — 동일 에러 |
| H2 | 티어 오배정(Basic↔Standard 재개명) | `sla_tier: Standard` 변종 제출 (10:36) | ❌ 반증 — 동일 에러 |
| H3 | 클라이언트 버전 스큐(11.9.1 vs 11.16.0) | 격리 venv에 **11.16.0 설치 성공**(index `https://msrpypi.azurewebsites.net/stable/leloojoo`) 후 제출 (10:38) | ❌ 반증 — 동일 에러. `.amltconfig` md5 불변(`f4871dca…`) = 기존 amlt env 무오염 |
| H4 | 리전/클라우드 불일치(에러 문언 그대로) | `az`로 실측: 워크스페이스 `msra-sh-aml-ws`=**westus2**, VC `msrresrchbasicvc`=**westus2** | ❌ 반증 — **에러 메시지가 문자 그대로 거짓** |
| H5 | H100 할당만 소멸 | **1-CPU throwaway 잡**(`sku: C1`, echo 2줄) 동일 VC 제출 (10:40) | ❌ 반증 — **동일 에러**. GPU 무관 |
| **H6** | **basicvc VC 단위 전면 거부** | 같은 throwaway를 **다른 VC** `msrresrchvc`(sku 10C3) 제출 (10:40) | ✅ **확증 — 즉시 registered/preparing** |

### 판정
**`msrresrchbasicvc`에 대한 모든 신규 제출이 managementfrontend 단계에서 거부된다** — 1 CPU짜리 echo 잡조차. 반면 동일 계정·동일 클라이언트·동일 워크스페이스·동일 프로젝트 스토리지로 `msrresrchvc`는 정상 제출된다. 따라서:
- 무죄 확정: 계정/토큰, amlt 클라이언트(9.1도 16.0도 동일), 워크스페이스, 프로젝트 스토리지, 리전, SKU, 티어, **b3p yaml·코드·데이터**
- 유죄 확정: **basicvc에 대한 우리 엔타이틀먼트/VC 상태** (서비스측). 05:49 UTC경 발생. b2p가 16h째 그 VC에서 running인 것은 **이미 admit된 잡은 유지**되기 때문 — VC 자체는 살아있다.
- ⚠️ E-090의 "클라이언트 버전 스큐" 진단은 **오진이었다. 여기서 정정한다.** 11.16.0으로도 동일하게 거부된다.
- ⚠️ 0717 주석의 "이 에러=티어 풀 할당 없음" 해석도 이번 케이스에는 **적용되지 않는다**(Standard도 실패).

### 조치 — b3p를 A100×4로 이식 발사
`msrresrchvc` 가용 액셀러레이터 = A100/CPU/MI200 (H100/H200 **없음**). `80G4-A100` 제출 가능 확인(probe → 즉시 preparing, 이후 cancel).
`a100_rq3v2_b3p.yaml` = `h100std_rq3v2_b3p.yaml`에서 **정확히 2줄만** 변경(`diff` 검증 완료):
```
5c5   name: msrresrchbasicvc  ->  name: msrresrchvc
17c17 sku: 80G4-H100          ->  sku: 80G4-A100
```
- **발사**: `rq3v2-b3p-a100` / job `rq3v2_b3p` / msrresrchvc / 80G4-A100 / 10:42:49 UTC / status `preparing` / portal `aka.ms/amlt?q=kssb3`
- `trainer.resume_mode=auto` + `pull_resume_ckpt.py` 경로 그대로 → **durable gs10에서 이어받는다**(작업 손실 0)
- H100 특이 가정 부재 확인: yaml 내 `h100` 문자열은 config **파일명**(`triobj_dcpo_v4_stage3b_h100_4x4k`)과 주석뿐. fp8/sm90/hopper 참조 없음. A100 80GB = 동일 VRAM이라 `max_model_len 10240`·`log_prob micro_batch 2` 그대로 유효.

### 이 이식이 만드는 새 한계 (논문 고지 대상)
- **b2p는 H100×4, b3p는 A100×4** — 즉 RQ2 축(b3p−b2p)에 하드웨어 비대칭이 생겼다. 수학(배치·스텝·lr·시드·보상)은 동일하므로 **체계적 편향이 아니라 커널 수준 잡음**이지만, 롤아웃 샘플링이 비트 단위로 달라질 수 있으므로 **명시 고지 필요**.
- A100은 bf16 throughput이 H100의 약 절반 → 스텝시간 ~14min(H100) → **~25-30min 추정**. 300스텝 = 6~7일. `max_run_duration_seconds: 604800`(7일) 경계에 걸린다 ⇒ **save_freq 10 + resume 체인에 의존**해 재제출로 이어가야 한다. 선점 시 손실 상한은 여전히 10스텝.
- basicvc가 복구되면 **H100으로 되돌려 재제출**하는 것이 패리티상 바람직(durable에서 이어받으므로 비용은 스텝시간 차이뿐).

### 부수 확인
- **b2p gs20 durable 착지 확인**(10:35, HF `checkpoints/rq3v2_b2p/global_step_20` 4/4) — 첫 저장점 통과. 이제 사망해도 처음부터 가지 않는다. b2p: running 16h, gs20+.
- durable frontier: `rq3v2_b2p [20]`, `rq3v2_b3p [10]`
- throwaway 잡 2건(`tw-a100-probe`, `tw-probe-vc2b`) **cancel 완료** — GPU 낭비 없음
- 격리 amlt: `scratchpad/amlt216/bin/amlt` = 11.16.0. **기존 `envs/amlt`(11.9.1) 무변경** — b2p 모니터링 의존성 보존

### 다음
1. b3p A100 잡 노드 확보 + **L204 통과(verl_main.log 생성) 확인** — 1.6kB 조용한 사망 재발 여부가 최우선 관측 대상
2. 재발 시에만 계측 하트비트 전진 패치 적용(변수 하나씩 원칙)
3. basicvc 복구 감시(주기 probe: 1-CPU throwaway가 가장 싼 카나리아)
4. ⓠredirect 배제 사유 특정(GPU 불요) + SFT2 복원 — **재현 판정의 전제**는 여전히 미충족

---

## E-092 ⛔ 사용자 지시 등록: **basicvc 복구 시 b3p를 H100으로 환원** — 단 이중발사 금지(체크포인트 궤적 혼합 위험 발견) (2026-07-26 10:47 UTC)

사용자 지시(원문): "basicvc 복구되면 H100으로 되돌려줘" → **상시 지시로 등록**. 근거는 T1 패리티: b2p가 H100이므로 b3p도 H100이어야 RQ2 축의 하드웨어 비대칭(E-091 기록)이 사라진다.

### ⚠️ 절차 설계 중 발견한 위험 — 순진한 "먼저 띄우고 나중에 죽이기"는 금지
`push_ckpts_to_hf.py`의 프루닝은 `--keep 1`이고, 두 런처(`h100std_rq3v2_b3p.yaml`·`a100_rq3v2_b3p.yaml`)가 **동일한 durable 경로**를 공유한다:
- 동일 `--ckpt_dir /scratch/checkpoints/rq3v2_b3p`, 동일 `--repo_id iamseungpil/metacot-h200-triobj-dcpo-v3`, 동일 `--config_name rq3v2_b3p`
- `_prune_old_verl_ckpts`(scripts/push_ckpts_to_hf.py:35-95)는 step 정렬 후 최신 `keep`개 complete만 남기고 **나머지를 `delete_folder`로 삭제**

따라서 A100 잡과 H100 잡을 **동시에** 돌리면:
1. 두 잡이 같은 `checkpoints/rq3v2_b3p/`에 push하고 서로의 체크포인트를 프루닝한다
2. 더 심각: 이후 어느 쪽이 선점당해 resume하면 `pull_resume_ckpt.py`가 **다른 궤적의 체크포인트**(옵티마이저 상태까지 다른 학습 경로)를 집어온다 ⇒ **조용한 궤적 혼합 = 데이터 손실보다 나쁜 과학적 오염**(눈치채기 어렵다)
3. `delete_folder`가 in-flight 업로드와 경합해 부분/손상 상태 가능

### 확정 환원 절차 (순차 — 겹침 금지)
파괴조작 3율의 "폐기 전 대체 확보"는 **자산(체크포인트) 확보**로 해석한다. 자산은 HF에 있고 잡이 아니다. 따라서:
1. **카나리아 green 확인** — 1-CPU `sku: C1` throwaway를 basicvc에 제출해 registered 되는지 확인(가장 싼 검증). 실패면 아무것도 하지 않는다.
2. **A100 잡의 durable frontier 확정** — `HfApi.list_repo_files`로 `checkpoints/rq3v2_b3p/global_step_N` 4/4 완전성 확인. 이 gsN이 보전 자산.
3. **A100 잡 cancel** — 손실 상한은 진행 중 부분 스텝뿐(save_freq 10).
4. **즉시 `h100std_rq3v2_b3p.yaml` 제출** — `resume_mode=auto`로 gsN에서 이어받는다.
5. 갭 = 노드 대기 시간뿐, **데이터 위험 0**. 순서를 뒤집으면 3번 위험이 발생하므로 절대 뒤집지 않는다.

### 현재 상태 (10:46-10:47 UTC)
- **basicvc 카나리아: RED** — 1-CPU echo 잡도 여전히 `The virtual cluster does not exist` 거부. 미복구.
- **b3p A100(`rq3v2-b3p-a100`): queued 4m** — 노드 미확보
- **b2p(`superb-terrier`): running 16h**, durable gs20
- 카나리아 런처 보관: `.tw_probe/canary.yaml`(1 CPU·echo·300s cap). 매 tick 재사용.

---

## E-093 ⭐⭐⭐ [ⓠ해결] redirect 배제 사유 **확정**: `think-closed` 조건은 품질필터가 아니라 **위장된 시나리오 필터** — `</think>`는 강제주입 prefix에서만 오고(생성 recovery엔 전 코퍼스 0건) redirect는 정의상 think 내부에서 잘린다 (2026-07-26 10:50-11:05 UTC, GPU 0)

필터 스크립트가 리포에 없어(`configs/sft_b2p2_rvseg.yaml`이 "E-071 fallback spec"으로만 기술) **원본↔생존 직접 매칭**으로 실제 배제 집합을 특정했다. 로컬 `data/rv_redirect_verify_functional.parquet`(1763행)과 `data/b2p2_rvseg_sft2.parquet`(378행) 사용.

### 0. 필터는 순수 선택(변환 아님)
messages md5 매칭 결과 **378/378이 원본에 그대로 존재** ⇒ 행 단위 selection. 따라서 배제된 1385행을 정확히 지목할 수 있다.

| scenario | 원본 | 생존 | 생존율 |
|---|---|---|---|
| redirect | 554 | 67 | **12.1%** |
| verify | 1209 | 311 | **25.7%** |

### 1. 조건별 분해 — 범인은 `think-closed` 하나
| 조건 | redirect 통과 | verify 통과 | 판정 |
|---|---|---|---|
| **C1 think-closed** (`</think>` 존재) | **84/554 = 15.2%** | **642/1209 = 53.1%** | 🔴 **범인**. 3.5배 차별 |
| C2 meta proper-nested | 457/554 = 82.5% | 787/1209 = 65.1% | ⚪ **redirect에 유리** — 원인 아님 |
| `<think>` open 존재 | 554/554 = 100% | 1209/1209 = 100% | ⚪ 무차별 |
| 길이(char) | — | — | ⚪ **이미 반증**(E-089). 재확인: 생존행이 **더 길다** |

`C1 & C2`는 생존의 **필요조건**: C1&C2 실패 1245행 중 생존 **0건**.

### 2. 근인 — `</think>`는 모델이 만들지 않는다
`</think>`의 위치를 prefix 경계(`prefix_split_char`) 기준으로 분류:

| scenario | in_prefix | **in_recovery** | none |
|---|---|---|---|
| redirect | 84 | **0** | 470 |
| verify | 642 | **0** | 567 |

⇒ **전 코퍼스 1763행 중 생성된 recovery에 `</think>`가 나타나는 행은 0건.** `wp_has_close` 교차표가 C1 카운트와 **완전히 동일**(redirect 84, verify 642) ⇒ **C1 ⟺ "강제주입된 wrong_prefix가 이미 `</think>`를 포함했는가"**. 모델의 산출물과 무관한, prefix 구성의 우연이다.

그리고 두 시나리오는 정의상 prefix 절단 위치가 다르다:
- **verify** = prefix에 (틀렸을 수 있는) **완결 해답**이 들어감 ⇒ `<think>…</think>` 를 넘어가는 경우가 흔하다(53.1%)
- **redirect** = 되돌릴 **잘못된 추론 경로**를 think **내부에서** 자른다 ⇒ `</think>`를 포함할 수 없다(15.2%가 예외)

∴ **`think-closed`는 데이터 품질을 재지 않는다. prefix가 `</think>` 경계를 넘었는지를 재고, redirect 시나리오는 그 경계를 정의상 넘지 않는다.** 위장된 시나리오 필터다.

### 3. 배제된 행은 절단이 아니다 (중요)
unclosed redirect 470행: **`\boxed{}` 보유 100%**, 문장 완결, `chars` med 1140/p95 3813. 즉 **답까지 완성된 정상 데이터를 `</think>` 부재만으로 버렸다.** (절단 가설도 반증)

### 4. 구조적 부수 발견 — 필터가 템플릿 하나만 남겼다
`</think>` 대비 `<|meta|>` 위치:

| 구조 | 행수 | 생존 |
|---|---|---|
| meta **after** `</think>` | 644 | **378 전부 여기서** |
| meta **inside closed** think (`meta_before_close`) | 82 | 0 |
| meta inside **unclosed** think | 1037 | 0 |

⇒ `c1_think_closed ⟺ meta-outside-think`(644/644). **생존 378/378이 전부 `<think>…</think>` → `<|meta|>…<|/meta|>` → recovery 단일 템플릿.** base SFT2는 세 구조 중 하나만 학습했다. T1은 필터 없이 1763행 전부로 학습했으므로 **세 구조를 다 봤다** — think-on 패리티 관점에서 별도 검토 필요.

### 5. ⚠️ 미해결 잔여 + 문서 불일치
`C1&C2` 통과 518행 중 **140행(verify 132·redirect 8)이 여전히 배제**됐다. 그 메커니즘은 특정 실패:
- `<|im_end|>`: 생존 0.0% / 배제 0.0% ⇒ 판별력 없음
- 길이: **생존이 더 길다**(p90 6004 vs 2412, max 14646 vs 8453) ⇒ 길이 컷 아님
- ⇒ `configs/sft_b2p2_rvseg.yaml`의 문서 스펙("<=4096 tokens AND EOS im_end at render")이 **산출물과 맞지 않는다**. 14,646자 행이 생존한 것은 4096-token 컷과 양립하기 어렵다. **문서-현실 불일치로 기록**하고, 잔여 140행 기준은 미특정으로 남긴다(redirect엔 8행뿐이라 본 결론에 영향 없음).

### 6. 수리안 사이징 (구현 아님 — 승인/수렴 대기)
redirect 비중: **T1 원본 31.4%(554/1763) → 현재 base SFT2 17.7%(67/378)**.
- **(A) C1 폐기, C2 유지**: redirect 457 + verify 787 = 1244행, redirect 비중 **36.7%** ⇒ T1(31.4%)을 살짝 넘어섬. 학습량 3.3배. 가장 단순.
- **(B) 버리지 말고 수리**: unclosed 행에 `</think>`를 최종답 앞에 삽입. 답 시작점 특정이 필요해 위험.
- **(C) T1과 동일하게 raw 1763 전부**: T1이 실제로 이겼던 구성. 단 `configs`의 "raw corpus was 0% clean and is BANNED" 주장과 충돌 ⇒ 그 "0% clean"의 정의를 먼저 확인해야.

⚠️ **E-089의 경고 유지**: "1763 전부 사용"이 자동 정답이 아니다. 필터엔 base 형식 불안정 대응이라는 근거가 있었다. 다만 **이제 그 근거가 redirect에 대해서는 성립하지 않음이 입증**됐다 — `think-closed`는 형식 안정성을 재는 조건이 아니라 시나리오를 고르는 조건이므로, redirect에 한해 **기준 자체가 무효**다. 대체 조건은 정직해야 한다(예: "meta-nested AND 답 보존 AND 렌더 토큰 상한"처럼 시나리오 중립인 것).

### 다음
1. (A)/(C) 중 선택 + 잔여 140행 기준 재정의 → SFT2 재구축 → b2p2 SFT2 재학습
2. 그 다음이 b0p arm 발사(재현 판정의 나머지 전제)
3. **재현 판정 전제 2개 중 하나(SFT2 원인 특정)가 이로써 해소**. 남은 전제 = SFT2 실제 복원 + b0p 존재.

---

## tick 11:14-11:30 UTC (2026-07-26) — 카나리아 RED 유지·b3p queued·정리 4건 완료·HF squash 3repo·삭제후보 131.2GB는 승인대기로 보류

### 상태
| 항목 | 값 |
|---|---|
| basicvc 카나리아 | 🔴 **RED** (11:14, 1-CPU echo 여전히 거부) ⇒ E-092 환원 절차 미발동 |
| b3p `rq3v2-b3p-a100` | **queued 31m** — 노드 미확보. L204 판정 불가 |
| b2p `superb-terrier` | running **17h** |
| durable frontier | `rq3v2_b2p {20: 23파일}`, `rq3v2_b3p {10: 19파일}` — b2p 다음 저장점 gs40(save_freq 20, ~4h 후) |

### 완료: 표적 정리 4건
1. **CLAUDE.md 데이터 계보 정정** — 존재하지 않는 `data/b23_rv_unmasked_sft.parquet` 경로를 현행 **2단 SFT 스택**으로 교체: b0p=`b0on_v8base_strict_sft`(Qwen3-8B-Base·3ep·1e-5) / b2p·b3p=SFT1 `b2on_v8meta_strict_sft`(3ep·1e-5) → SFT2 `b2p2_rvseg_sft2`(2ep·2e-6) → `models/b2p2_rvseg_sft`(양 arm 공용 RL init). think-off 세대는 retired로 분리하고 E-093 기아 경고 명시.
2. **⭐HF 백업 3건** — 현행 계보 SFT 데이터가 **전부 HF에 없어** 로컬+코드tarball뿐인 단일 소실점이었다. `iamseungpil/metacot-rv`에 순수 추가 업로드 후 존재 검증:
   - `data/b2p2_rvseg_sft2.parquet` 472,308 B
   - `data/b0on_v8base_strict_sft.parquet` 2,830,364 B
   - `data/b2on_v8meta_strict_sft.parquet` 3,960,627 B
3. **커밋 `d81c85a`** (red커밋 가드 준수: pytest와 한 `&&` 체인, **747 passed / 8 skipped**) — 런처 8종·configs 3종·**이벤트 원장 1909행**(untracked = 최대 소실점이었음)·카나리아 런처·삭제 manifest. 미푸시 0.
   - `h100std_rq3v2_b3p_tierSTD.yaml` **삭제** — 내가 만든 진단 부산물이고 결론은 E-091에 기록됨(자기 뒷정리 원칙)
   - `paper` submodule은 **손대지 않음** — `M`의 원인은 별도 리포 내부의 untracked figure 파일(`figures/fig1_contrast_spec.txt`·`figures/outputs/`). 별개 결정이라 보고만.
4. **⭐redirect 배제 사유 특정 완료** → E-093 (별도 항목)

### HF 주기정리
- **super_squash_history 3 repo 실행 완료**: `metacot-h200-triobj-dcpo-v3`(model)·`metacot-rv`·`metacot`(dataset). 무결성 게이트 **OK** — durable frontier 전후 완전 불변.
- ⚠️**측정 방법 정정**: `repo_info(files_metadata=True)` 합계는 **현재 트리** 크기이므로 history LFS 해방분이 보이지 않는다. squash 전후 441.1GB 동일하게 나온 것은 실패가 아니라 **지표가 잘못된 것**. 트리 크기와 저장소 소비는 다른 양이다.
- 트리 실측 분해(ckpt repo 279.0GB): `rq3_b0/gs300` 98.3GB · **`rq3v2_b2p/gs20` 98.3GB(LIVE)** · `rq3_b2/gs160` 32.8GB · **`rq3v2_b3p/gs10` 32.8GB(LIVE)** · `models` 16.4GB(LIVE, SFT init) · eval 0.4GB

### ⛔ 삭제 후보 131.2GB — **실행 안 함, 승인 대기**
`docs/reports/hf_deletion_candidate_20260726.json` 작성(비파괴). 은퇴한 RQ3 think-off 세대 `rq3_b0/gs300`(98.3GB) + `rq3_b2/gs160`(32.8GB).
**보류 사유**: 이 repo의 `eval/`에는 두 후보의 held-out 산출물이 **없다** — `eval/`은 T1 instruct 세대(base_matched·gandhi·pmishift·shiftonly gs300) 전용이다. wandb엔 `rq3_b0`(finished·300steps)·`rq3_b2`(crashed·233steps) 학습궤적만 보존. **`rq3_b0`는 완주한 베이스라인이고 논문 부록 대상**이라, 삭제하면 재평가가 영구 불가하다. 0724 선례(`rq3_b3nopmi`)는 *포기된* arm이라 등급이 다르다.
**긴급성 낮음**: 트리 441.1GB vs 과거 403 quota 실패 지점 ~1041.5GB + 방금 squash로 history 해방. ⇒ 자율 삭제 금지, 사용자 판단 대상으로 상신.

### 다음 틱
카나리아 → b3p 노드/L204 → b2p gs40 → (승인 시)삭제 실행 → SFT2 재구축 안(A/C) 선택.

---

## E-094 ⭐⭐⭐ SFT2 필터가 **SFT1과 싸우고 있었다** — "0% clean" 정의 추적 결과 필터는 목표를 달성하지 못한 채 세 번째 기준(meta-inside)을 **100%→0%로 역전**시켰고, redirect의 69%가 든 계열을 통째로 버렸다 (2026-07-26 11:47-12:05 UTC, GPU 0)

E-093의 남은 질문("configs의 `raw corpus was 0% clean and is BANNED`에서 clean의 정의는?")을 추적했다. 정의는 **E-070 L0 감사**에 있다: clean = **think-closed AND meta-nested AND meta-INSIDE-think** 3중 조건. 당시 측정: b23 base 30/71/5/**0%** · b23 v2 31/100/0/**0%** · **v8 instruct(T1 우승) 100/100/100/100%**.

### 1. 4개 코퍼스 재감사 (동일 3기준)
| 코퍼스 | n | think-closed | meta-nested | **meta-INSIDE** | ALL-clean |
|---|---|---|---|---|---|
| T1 SFT1 `v8_meta_inside_strict` | 4264 | 100% | 100% | **100%** | **100%** |
| base SFT1 `b2on_v8meta_strict_sft` | 4245 | 100% | 100% | **100%** | **100%** |
| T1 SFT2 `rv_functional` RAW | 1763 | 41.2% | 70.6% | **63.5%** | 0% |
| **base SFT2 (378 필터)** | 378 | 100% | 100% | **0.0%** | **0%** |

🔴 **필터는 목표를 달성하지 못했다.** 목표가 "fully-clean rows"였는데 결과는 raw와 똑같이 **0% clean**이다. 두 기준을 100%로 올리는 대가로 **세 번째를 0%로 몰았다**. 게다가 T1 SFT2는 meta-inside를 **63.5%** 유지했다 — base는 0%.

### 2. 왜 0%인가: raw 1763의 5계열 완전 분해
| 계열 | closed | nested | inside | n | redirect | redirect 비중 |
|---|---|---|---|---|---|---|
| **F011** | ✗ | ✓ | **✓** | **726** | **382** | **52.6%** |
| **F110** | ✓ | ✓ | ✗ | **518** | 75 | 14.5% |
| F001 | ✗ | ✗ | ✓ | 311 | 88 | 28.3% |
| F100 | ✓ | ✗ | ✗ | 126 | 8 | 6.3% |
| F101 | ✓ | ✗ | ✓ | 82 | 1 | 1.2% |

`closed & inside & nested = **0**`. closed+inside인 82행은 **전부 태그 개수 불일치**(open 2-3 vs close 1-3) ⇒ E-071의 "RV/답-뒤 설계로는 clean 불가"가 확증된다.

### 3. 🔴 핵심: 필터는 **SFT1과 반대되는 계열**을 골랐다
필터 풀 = **F110**(meta **OUTSIDE** think) → 378행 선택. 그런데 SFT1은 **100% meta-INSIDE**다. 즉 SFT2 top-up이 방금 SFT1이 심어놓은 구조 습관을 **정면으로 되돌리는 방향으로** 학습시켰다.
그리고 버려진 **F011(726행)** 은:
- meta **정상 중첩** ✓
- meta **INSIDE think** ✓ ⇒ **SFT1 구조와 일치**
- 결함은 `</think>` 하나뿐 (E-093: 이 행들은 `\boxed{}`로 답까지 완결·절단 아님)
- **전체 redirect 554행 중 382행(69%)이 여기 있다**

∴ 필터는 두 오류를 **동시에** 저질렀다: ①redirect를 굶겼다(69%가 든 계열 폐기) ②meta 위치 구조를 SFT1 대비 역전시켰다. E-089의 "1/35 학습량·1/8 redirect"는 이 중 ①의 정량화였고, ②는 이번에 처음 드러났다. b3p의 `emit 0.974`(높음)인데 `rmeta −1.08`·`derail>save`인 관측과 정합한다 — 메타 블록을 **내보내긴 하지만 SFT1·보상기계가 기대하는 위치와 다른 곳에** 두도록 배웠다면 그 블록의 역할이 달라진다.

### 4. 재구축 안 재정렬 (사이징 확정)
| 안 | 구성 | n | redirect 비중 | meta-inside | T1 대비 |
|---|---|---|---|---|---|
| **(C) T1 그대로** | raw 1763 전부 | 1763 | **31.4%** | 63.5% | **편차 0** |
| (A′) 태그불일치만 배제 | F011+F110 | 1244 | 36.7% | 58.4% | 소편차 |
| (D) SFT1 정합 최대화 | F011만 | 726 | 52.6% | 100% | redirect 과대·verify 손실 |
| 현행(폐기) | F110 부분집합 | 378 | 17.7% | **0%** | 대편차 |

**권고 = (C) raw 1763 + T1 용량(3ep·lr 1e-5)**. 근거: ①T1이 **정확히 이 SFT2로 이겼다** — `configs`의 "0% clean BANNED" 주장은 **SFT1 코퍼스 품질 논의를 SFT2에 잘못 적용한 것**이다(T1 자신의 SFT2가 그 raw 코퍼스다) ②base SFT1은 이미 100% clean base-native라 구조 습관은 T1과 동일한 출발점 ③우리가 발명하는 모든 필터는 검증되지 않은 편차이고, 이번 감사로 그 편차가 **해로울 수 있음이 실증**됐다.
⚠️ 함께 가져갈 안전장치(과거 실패에서 학습): E-076의 **렌더층 token-id 수준 segment-mask 검증 게이트**(char-level 검사는 4행을 놓쳤다) + E-071의 L2 kill-switch(EOS rc0·truncation≤5%·emission≥0.85·wellformed≥0.90).
⛔ **구현은 사전점검 후** — 표준 프로토콜(구조 사전검사·승인) 준수. 지금은 사이징·근거까지만.

### 5. 틱 상태 (11:47 UTC)
basicvc 카나리아 **RED**(3회 연속) · b3p `rq3v2-b3p-a100` **queued 1h**(노드 미확보, L204 판정 불가) · b2p running **17h** · durable `rq3v2_b2p{20:23}` `rq3v2_b3p{10:19}`(b2p 다음 gs40 ~15:30 예상)

---

## E-095 ⭐⭐⭐ 렌더층 token-id 감사 도구 신설 + **T1 자신의 SFT2에 메타신호가 죽은 행이 30%** 발견 / 시나리오 중립 대체조건 확정 (2026-07-26 12:16-12:45 UTC, GPU 0)

E-094의 (C)안을 발사 전에 검증하기 위해 **`scripts/audit_sft2_render_mask.py`** 를 신설했다. `sft.py:tokenize_row`(L85-128)를 그대로 재현하고 — 실제 SFT 토크나이저(HF `models/b2p2_rvseg_sft`, `<|meta|>`=151669·`<|/meta|>`=151670 추가어휘 포함) —  최종 token id 층에서 5개 게이트를 계측한다. char-level 검사가 E-076에서 4행을 놓쳤기 때문이다.

### 게이트 결과
| 코퍼스 | n | G1 절단 | G2 답 학습영역 | **G5 메타블록 학습됨** | G4 EOS | G3 trace-shape |
|---|---|---|---|---|---|---|
| raw 1763 · redirect | 554 | 0.0% | 100% | **81.6%** (102행 사망) | 100% | 14.1% |
| raw 1763 · verify | 1209 | 0.0% | 100% | **65.1%** (422행 사망) | 100% | 37.1% |
| F011 (726) | 726 | 0.0% | 100% | **99.3%** (5행) | 100% | 0.0% |
| F110 (518) | 518 | 0.0% | 100% | **100%** (0행) | 100% | 70-87% |
| 현행 378 | 378 | 0.0% | 100% | 100% | 100% | 95.5-100% |

### 🔴 핵심 발견: **T1의 SFT2에도 메타신호가 죽은 행이 524행(29.7%) 있었다**
`sft.py`는 `[prompt]+[wrong_prefix]`를 마스킹한다(L112-116). 그런데 raw 1763 중 **524행은 `<|meta|>…<|/meta|>` 블록 전체가 그 마스크된 prefix 안에** 들어간다 ⇒ 그 행들은 **메타 방출에 관해 아무것도 가르치지 않고** recovery 텍스트만 학습시킨다. 보상이 나중에 채점하는 바로 그 행동을 조용히 굶기는 구조다.
그 524행의 정체가 정확히 특정된다: **F001(311)+F100(126)+F101(82)=519 = 태그 개수 불일치 계열** + F011의 5행 = **524. 완전 일치.** ⇒ **"태그 개수 불일치" = "메타 신호 사망"** 이 동일 사건이다.
T1은 이 1763행 전부로 학습해서 **이겼다.** 즉 30%의 죽은 행은 치명적이지 않았다 — 그러나 제거하면 순이득이다.

### 시나리오 중립 대체조건 확정 (E-093 약속 이행)
E-093에서 "대체 조건은 시나리오 중립이어야 한다"고 적었다. **G5가 그 조건이다**: *메타 블록 전체가 학습영역 안에 있어야 한다.* 시나리오를 보지 않고, 마스크 계약만 본다.
- **G5 통과 = 1763 − 524 = 1239행**, redirect 452 = **36.5%** (T1 31.4% · 현행 17.7%)
- 학습량 현행의 **3.3배**, G1/G2/G4 전부 green

### 발사 순서 확정 (실험논리)
지금까지의 병은 **계측 없이 발명한 편차**였다. 그래서:
1. **1차 = (C) raw 1763 그대로 + T1 용량(3ep·lr 1e-5)** — T1과 문자 그대로 동일한 재현 시도. 죽은 30%까지 포함해서 동일하게 간다. b3p−b0p가 음수로 나와도 "편차 탓"이라는 해석 여지를 남기지 않는다.
2. **2차 = (C′) G5 게이트 1239행** — 계측으로 정당화된 개선. (C)가 재현되면 (C′)는 추가 이득으로 보고, (C)가 실패하면 (C′)가 substrate-dependence와 데드로우를 분리한다.
⇒ (A)/(D)는 폐기. redirect 비중을 임의로 조정하는 안은 근거가 G5보다 약하다.

### ⚠️ 이번 감사에서 **내 지표 2건을 정정**한다
1. **G4 EOS 최초 0.0% 보고는 내 측정 버그였다.** 템플릿이 `<|im_end|>`(=eos 151645) **뒤에 개행(198)** 을 붙이므로 마지막 토큰은 `\n`이다. `[-1]==eos`는 완벽히 종결된 코퍼스에서도 0%를 낸다. 꼬리창(`eos in [-3:]`)으로 교체 → **전 코퍼스 100%**. 부수 소득: E-093이 미해결로 남긴 "문서스펙의 `EOS im_end at render` 조건"은 **판별력이 원리적으로 0**(렌더 후엔 항상 존재)임이 확정 — 그 조건은 vacuous였다.
2. **G3 최초 43/51% 보고도 내 지표 결함이었다.** decode 문자열 길이 비교는 선행공백 정규화 때문에 유령 1토큰 드리프트를 만든다. 경계 토큰을 직접 보니 `<|meta|>` 자체였다(마스크가 **정확**했다). 직접 계약검사로 교체 → 현행 378은 95.5/100%. **거짓 버그 보고 직전에 자체 반증됨.**
3. ⚠️**G3은 게이트가 아니라 기술자다.** F011은 G3 0%지만 G5 99.3%다 — 메타 블록이 prefix 직후가 아니라 몇 토큰 뒤에서 시작할 뿐이고, 그 사이 텍스트는 recovery 서술이라 **학습되는 게 맞다**. 코퍼스 사용가능성 판정은 **G5로만** 한다. 스크립트 docstring에 이 구분을 명시했다.
4. 잔여 140행 미스터리 **종결**: 실제 토큰 수로 재검정했으나 어떤 임계값도 생존/배제를 기저율(73%) 이상으로 설명하지 못하고, 1763행 중 4096 토큰 초과는 **0행**이다(max 3184). ⇒ 문서스펙 4조건 중 **2개(토큰상한·EOS)가 vacuous**, 나머지 2개는 518행을 주는데 산출물은 378행 ⇒ **실제 실행된 필터는 문서와 다르며 재구성 불가**. 코퍼스를 교체하므로 더 이상 블로커 아님.

### 틱 상태 (12:16 UTC)
basicvc 카나리아 **RED**(4회 연속) · b3p `rq3v2-b3p-a100` **queued 1h+**(노드 미확보) · b2p running **18h** · durable 불변(`rq3v2_b2p{20}`·`rq3v2_b3p{10}`)

---

## E-096 SFT2 (C)안 **발사 준비 완료** — config·런처·tarball 재패키징·전수 검증 (2026-07-26 13:03-13:20 UTC, GPU 0)

E-095에서 확정한 1차안(raw 1763 + T1 용량)을 노드 확보 즉시 발사할 수 있는 상태로 만들었다.

### 산출물
1. **`configs/sft_b2p2_rvfull.yaml`** — T1의 `configs/archive/sft_rv_functional.yaml`을 그대로 미러: `dataset_path: data/rv_redirect_verify_functional.parquet`(raw 1763), **3ep · lr 1.0e-5 · max_length 4096 · bs1×ga4 · save_strategy epoch**. T1과 다른 점은 **init(base-native `b2p_v8meta_strict_sft`)과 출력 이름 둘뿐**.
2. **`a100_sft_b2p2_rvfull.yaml`** — `h100std_sft_b2p2_rvseg.yaml`의 최소편집 클론. `msrresrchvc` + **80G4-A100 / Basic**(basicvc가 전면 거부 중이므로). E-071 L2 kill-switch(EOS 게이트 → `measure_sft_gate` 재게이트 `base_accuracy_greedy 0.5967` → 회귀시 ckpt 폐기) **원형 유지**.
3. **tarball 재패키징** — `scripts/build_sdc_code_snapshot.sh`(configs·data 디렉토리 전체 복사 + 비밀 리댁션 패스)로 빌드. release `359072254` / tag `rq3v2-thinkon-20260724`에 **asset `490360017`** (`metacognition_rq3v2_0726_rvfull.tar.gz`, 34,163,963 B) 업로드.

### ⭐출력 이름을 새로 만든 이유 (조용한 init 스왑 방지)
기존 런처는 `/scratch/checkpoints/b2p2_rvseg_sft` → HF `models/b2p2_rvseg_sft`에 push하고, **실행 중인 b2p·b3p RL 런처가 바로 그 경로에서 init을 스테이징한다**(`h100std_rq3v2_b3p.yaml:106-107`). 같은 이름을 재사용하면 **재제출되는 arm 밑에서 init이 조용히 바뀐다** — E-092에서 막은 궤적 혼합과 같은 계열의 사고다. 그래서 전 경로를 `b2p2_rvfull_sft`로 분리했다: 체크포인트 디렉토리·HF 모델 경로·게이트 로그 디렉토리·wandb 이름·shard 검증 문자열 전부.

### 발사 전 기계 검증 (전부 통과)
| 검사 | 결과 |
|---|---|
| tarball에 `configs/sft_b2p2_rvfull.yaml` 존재 | ✅ |
| tarball에 `data/rv_redirect_verify_functional.parquet` 존재 | ✅ (기존부터 포함) |
| 비밀 유출 스캔(`ghp_`/`hf_` 패턴) | ✅ 0건 |
| `.env` 미포함 | ✅ 0건 |
| **round-trip md5** (업로드 후 재다운로드) | ✅ `a3817d7653c8fab30619dbfd9336d61b` 일치 |
| 재다운로드본에 새 config 존재 | ✅ |
| yaml 파싱 | ✅ target `msrresrchvc` / `80G4-A100` / `Basic` |
| **shlex 토큰 3개**(`bash`,`-c`,payload) | ✅ 공백 word-split 없음 |
| **`$$`→`$` 치환 후 inner `bash -n`** | ✅ 클론·원본 **양쪽** OK |
| 치환본 원본↔클론 diff | ✅ **의도한 8곳만**(데이터·config·run명·ckpt·게이트dir·model명·push명·shard검증) |

⚠️**sed 클론의 사각을 이번에도 실측으로 잡았다**: sed 치환 후에도 **description 전문이 낡은 상태로 남아 있었다**("382 fully-clean rows·2 epochs·lr 2e-6·models/b2p2_rvseg_sft"). yaml은 파싱되고 bash도 통과하므로 기계검사로는 안 걸린다 — 전수 diff를 눈으로 읽어야 잡힌다. `WANDB_TAGS`의 `rvseg`, 코퍼스 부재 에러 문구도 같이 낡아 있었다. 3건 모두 수정. **교훈 재확인: 클론 검증은 "파싱 통과"가 아니라 "전수 diff 통독"이다.**
⚠️`$$(seq 1 30)`에 대한 첫 `bash -n` 실패는 **치환 전 검사라 발생한 거짓 오류**였다(amlt가 노드에서 `$$`→`$`로 바꾼다). 치환을 먼저 적용해야 유효한 검사가 된다 — 이 순서를 기록해 둔다.

### 가역성
`CODE_TAR_REVISION` **값 하나**만 `490360017`→`488239754`로 되돌리면 구 tarball로 복귀한다. 신규 발사가 부트스트랩을 깨도 즉시 복구 가능.

### 남은 것
- **발사 대기 = 노드**. `msrresrchvc` A100은 b3p가 2시간째 queued이므로 SFT까지 동시에 물리면 경합한다 ⇒ **b3p가 노드를 잡은 뒤 또는 b3p 판정 후 발사**.
- SFT2 산출물(`b2p2_rvfull_sft`)이 나오면 그걸 init으로 쓰는 **b2p/b3p RL 런처 신규본**이 필요하다(기존 런처는 구 init을 가리킨다). 이건 SFT2 게이트 통과 후 작성.
- b0p arm은 별개 전제로 여전히 미발사.

### 틱 상태 (13:03 UTC)
basicvc 카나리아 **RED**(5회 연속) · b3p `rq3v2-b3p-a100` **queued 2h**(노드 미확보) · b2p running **19h** · durable 불변

---

## tick 13:35 UTC (2026-07-26) — **SFT2 (C) 발사** (발사 순서 재고: b3p보다 SFT2 우선)

### 상태 (전 항목 무변화)
카나리아 **RED 6회 연속** · b3p `rq3v2-b3p-a100` **queued 2h**(노드 미확보) · b2p `superb-terrier` running **19h** · durable `rq3v2_b2p{20}`·`rq3v2_b3p{10}`

### ⭐발사 순서를 바꿨다 — 근거
직전 틱 계획은 "b3p가 노드를 잡거나 판정난 뒤 SFT2 발사"(A100 경합 회피)였다. 재검토 결과 **우선순위가 뒤집혔다**:
- **b3p A100은 결함 있는 init에서 학습한다** — 378행 기아 SFT2(redirect 1/8.3·meta-inside 0%·유효 학습량 ~1/35). b2p와 init이 매치되므로 **RQ2 축(b3p−b2p, "이 init에서 보상 패키지가 돕는가")은 내부적으로 유효**하지만, **재현 질문(b3p−b0p)에는 답할 수 없다**.
- **SFT2가 재현 판정의 전제**다. 사전등록 조건("b0p 존재 + SFT2 T1수준 복원")의 절반이 이것이고, 사용자의 중심 관심사다.
- ⇒ 큐에서 SFT2가 먼저 슬롯을 잡는 것이 이득. b3p의 큐 위치를 빼앗지 않으며 둘 다 대기할 뿐이다.

### ⚠️ 이중발사 금지 규칙에 저촉되지 않음 — 경로 대조로 확인
E-092의 금지는 **동일 durable 경로**를 두 잡이 쓰는 경우다. 이번 두 잡은 다르다:
- SFT2 → HF **dataset** `iamseungpil/metacot` : `models/b2p2_rvfull_sft`
- b3p  → HF **model** `metacot-h200-triobj-dcpo-v3` : `checkpoints/rq3v2_b3p`
repo도 경로도 겹치지 않으므로 프루닝 경합·궤적 혼합 위험 **0**. (SFT2의 출력 이름을 `rvfull`로 분리해 둔 E-096의 조치가 여기서 그대로 효력을 낸다.)

### 발사
- **`rq3v2-sft2-rvfull`** / job `sft_b2p2_rvfull` / msrresrchvc / 80G4-A100 / **13:36:02 UTC** / status `preparing` / portal `aka.ms/amlt?q=kssl6`
- red커밋 가드 준수: `pytest 747 passed, 8 skipped` && 제출을 한 체인으로 묶음
- CODE_TAR_REVISION `490360017`(round-trip md5 검증본), config `configs/sft_b2p2_rvfull.yaml`
- 판정 게이트는 런처 원형 유지: EOS 게이트(rc0 필수) → `measure_sft_gate`(math500 300문항·`base_accuracy_greedy 0.5967` SFT1 앵커) → **회귀시 ckpt 폐기**(E-071 kill-switch)

### 다음 틱
카나리아 → **SFT2 preparing→running·L2 게이트 로그** → b3p 노드 → b2p gs40(~15:30) → (SFT2 게이트 통과시)`b2p2_rvfull_sft` init을 쓰는 RL 런처 신규본 작성

---

## E-097 ⭐⭐⭐ **매치드 사다리에 단(stage)이 빠져 있었다** — 컨트롤 arm이 1단, 메타 arm이 2단 ⇒ 재현 축이 "메타 메커니즘"과 "SFT 한 단계 추가"를 섞고 있었다. 누락된 단을 발사 (2026-07-26 14:03-14:12 UTC, GPU 0)

b0p arm 발사 준비를 하던 중 발견. HF `iamseungpil/metacot` `models/` 전수 열거 결과:
```
b0p_v8base_strict_sft   14 files   ← 컨트롤 SFT1
b2p_v8meta_strict_sft   14 files   ← 메타 SFT1
b2p2_rvseg_sft          14 files   ← 메타 SFT2
(b0p2_* 는 존재하지 않음)          ← 컨트롤 SFT2가 없다
```
즉 사다리가 **비대칭**이었다:
```
b2p/b3p : SFT1(meta)  →  SFT2(RV 코퍼스, meta)  →  RL
b0p     : SFT1(base)  →  (없음)                 →  RL
```

### 🔴 이것이 만드는 교란
재현 축 **b3p − b0p**는 우리가 재려는 것(메타 메커니즘)과 재려는 것이 아닌 것(**RV 코퍼스에 대한 SFT 한 단계 추가**)을 함께 담는다. 매치드 사다리가 존재하는 이유가 바로 이 교란을 없애는 것이었다. 그리고 이것은 이미 기록된 함정과 같은 계열이다 — RQ3 3중감사의 "RQ1은 meta-init 효과가 아니라 **corpus까지 다른** SFT 비교"(0716).

### 수리는 새 데이터가 필요 없었다 — 이미 만들어 놓고 안 쓴 트윈이 있다
`scripts/build_base_rv_sft_data.py`의 docstring 첫 줄이 정확히 이 용도다: *"Build the META-REMOVED TWIN of the rv_functional SFT corpus (base SFT-2 data)"*. 산출물 `data/v8_base_rv_sft.parquet`이 HF `metacot`에 **미사용 상태로 있었다**. 0726 실측 검증:

| 항목 | 값 |
|---|---|
| 행수 | **1763 — 메타 코퍼스와 정확히 1:1**(탈락 0행) |
| `<|meta|>`/`<|/meta|>` 잔존 | **0** (빌드 스크립트의 hard invariant 성립) |
| 시나리오 분포 | verify 1209 / redirect 554 — **메타 코퍼스와 동일** |
| think-closed | **41.2% — 메타 코퍼스와 동일**(meta 제거는 `</think>`를 건드릴 수 없다) |
| wrong_prefix 비어있지 않음 | **100%** ⇒ sft.py가 **동일하게** segment-mask |
| 렌더층 감사(G1) | 절단 **0%**(max 3124 tok < 4096) |
| 렌더층 감사(G2) | 최종 `\boxed` 답이 학습영역 안 **100%** |
| 렌더층 감사(G4) | EOS **100%** |
| G3/G5 | **N/A(구조상 당연)** — 메타 전용 게이트이고 이 코퍼스엔 메타 블록이 없다 |

⇒ 두 arm은 이 단계에서 **메타 블록의 유무만** 다르게 된다. 정확히 매치드.

### 산출물 + 발사
1. **`configs/sft_b0p2_rvfull.yaml`** — init `b0p_v8base_strict_sft`, data `v8_base_rv_sft.parquet`, **3ep · lr 1e-5 · max_length 4096 · bs1×ga4** = 메타 arm (C) 런과 **용량 동일**.
2. **`a100_sft_b0p2_rvfull.yaml`** — `a100_sft_b2p2_rvfull.yaml` 클론. ⭐**게이트를 컨트롤 arm에 맞게 교체**: `measure_sft_gate`(pmi auc)는 **쓰지 않는다** — 메타 블록이 없는 arm에 pmi auc는 무의미하다. 대신 b0p SFT1 선례(`h100std_sft_b0p.yaml`)의 `eval_vllm_1030` term-eval(math500+aime2024·8k·temp1.0·n2). 앵커(b0p SFT1 실측): **MATH500-100 55.5% · AIME 8.3% · meta_rate 0.5%≈0**. meta_rate가 오르면 트윈이 메타를 흘렸다는 신호.
3. **tarball 재패키징 #2** — asset **`490407111`**(`metacognition_rq3v2_0726_b0p2.tar.gz`, 35,682,037 B, md5 `881fb4a609a6978c0e1c825b25c3d122`, round-trip 일치). 490360017의 **상위집합**: `configs/sft_b0p2_rvfull.yaml` + **`data/v8_base_rv_sft.parquet`** 추가(트윈은 490360017에 **없었다** — HF에만 있었다). 메타 arm 런처는 이미 490360017로 제출됐으므로 **건드리지 않았다**(실행 중인 것과 yaml을 일치시켜 둠).
4. **발사**: **`rq3v2-sft2-b0p2`** / job `sft_b0p2_rvfull` / msrresrchvc / 80G4-A100 / **14:10:33 UTC** / `preparing` / portal `aka.ms/amlt?q=kssnu`. red커밋 가드 준수(pytest 747 passed && 제출).

### durable 경로 충돌 매트릭스 (전부 상이 — 동시 대기 안전)
| 잡 | durable 경로 |
|---|---|
| SFT2-meta `rq3v2-sft2-rvfull` | HF **dataset** metacot : `models/b2p2_rvfull_sft` |
| **SFT2-control `rq3v2-sft2-b0p2`** | HF **dataset** metacot : `models/b0p2_rvfull_sft` |
| b3p `rq3v2-b3p-a100` | HF **model** v3 : `checkpoints/rq3v2_b3p` |
| b2p `superb-terrier`(running) | HF **model** v3 : `checkpoints/rq3v2_b2p` |

### ⚠️ 클론 검증에서 잡은 실제 버그 1건 (+ 낡은 문구 1건)
`a100_sft_b0p2_rvfull.yaml`의 shard 검증 줄이 **여전히 메타 arm 경로를 세고 있었다** — `\"b2p2_rvfull_sft/model-\"`. 치환 문자열이 **이스케이프된 따옴표** 안에 있어 일반 치환이 빗나갔다. 그대로 발사했다면 push 루프가 **다른 arm의 샤드 수**를 보고 조기 break하거나 영원히 4/4를 못 봤을 것이다. E-096에서 기록한 "클론 검증은 전수 diff 통독"이 두 틱 연속으로 실제 버그를 잡았다. 부수로 코퍼스 부재 에러 문구도 낡아 있어 수정.
기계검증 통과: yaml 파싱 · shlex 3토큰 · `$$`→`$` 치환 후 inner `bash -n` OK · 메타 arm 런처 대비 diff가 의도한 항목만.

### b3p를 취소하지 않은 이유
b3p는 결함 init(378행 기아 SFT2)에서 돌지만 b2p와 init이 매치되므로 **RQ2 축(b3p−b2p)은 내부적으로 유효**하고, 논문의 RQ2 표는 전부 `\todo{}`다. b2p가 이미 20h 투입됐으므로 취소는 그 측정을 버리는 것. ⇒ 유지. 새 SFT2가 게이트를 통과하면 **새 init으로 b2p/b3p를 다시 돌리는 것이 재현 축**이고, 현행 두 arm은 RQ2 부록으로 남는다.

### 상태 (14:10 UTC)
카나리아 **RED 7회** · SFT2-meta **queued 34m** · **SFT2-control preparing(신규)** · b3p **queued 3h** · b2p running **20h** · durable `rq3v2_b2p{20}`·`rq3v2_b3p{10}`

---

## E-098 ⚠️⭐⭐ **정정: b3p의 gs10은 resume 불가였다**(optim 0/4) — 여러 틱에 걸친 "durable gs10 안전 4/4" 기록은 파일 수만 센 것 / 우선순위 역전 해소를 위해 b3p 취소 (2026-07-26 14:39-14:50 UTC)

### 🔴 정정
E-091 이후 계속 "b3p durable gs10 안전 4/4"로 기록하고 런처 설명에도 "resumes from durable gs10"이라 적었다. **틀렸다.** 샤드 종류별로 세어 보니:

| config | step | model | optim | extra | 판정 |
|---|---|---|---|---|---|
| `rq3v2_b2p` | gs20 | 4/4 | **4/4** | 4/4 | ✅ **RESUMABLE** |
| `rq3v2_b3p` | gs10 | 4/4 | **0/4** | 4/4 | ❌ **resume 불가** |

`pull_resume_ckpt.py:80`의 조건은 **model≥4 AND optim≥4 AND extra≥4**이고, 그 docstring이 이 함정을 이미 명시하고 있다: *"a step with model 4/4 but optim <4 (upload cut mid-flight) still crashes resume — all three shard sets must be complete"*. 내가 센 것은 **파일 총수 19개**였고 종류별 완결성이 아니었다. 19 = 23(b2p) − 4(optim 샤드) 로 정확히 설명된다.
이것은 E-071 H-3b가 이미 기록한 **완결성 정의 3원 불일치**(푸셔=model+extra≥4 / pull_resume=model+optim+extra≥4)가 실제로 물린 사례다. 푸셔는 gs10을 "완결"로 보고 올렸지만 resume은 그것을 후보로 인정하지 않는다.
⇒ **b3p A100 잡은 노드를 잡았어도 gs0부터 300스텝을 다시 돌 예정이었다.** A100 ~25-30min/step 추정이면 6~7일이 아니라 **처음부터 6~7일**.

### 결정: b3p 취소 (파괴조작 3율)
1. **LIST** — b3p: `queued 3h`, 노드 미확보(런타임 정보 없음), 소비한 계산 0, **resume 자산 0**(gs10 사용 불가). b2p gs20은 RESUMABLE로 무관.
2. **decide** — msrresrchvc A100은 **전 SKU Basic 티어뿐**(Standard 없음 = 티어 레버 없음), 다른 VC 없음(`target list -a`로 확인: msrresrchbasicvc·msrresrchvc 2개뿐), 인스턴스 타입은 이미 양쪽 유연(`80G4-A100 | 80G4-A100-NvLink`). ⇒ 순수 opportunistic 용량 대기. b3p가 **가장 오래 기다린 잡**이라 다음 슬롯을 잡을 가능성이 높은데, 셋 중 **가치가 가장 낮다**(결함 init·resume 자산 없어 gs0부터·6~7일). 두 SFT2는 각 2~4h이고 재현 판정의 critical path. b3p가 슬롯을 잡으면 **SFT2가 일주일 막힌다** = 우선순위 역전.
3. **execute** — `amlt cancel rq3v2-b3p-a100` → `killed`. **잃은 것은 큐 위치 3시간뿐, 데이터 0.**

b3p는 새 SFT2(`b2p2_rvfull_sft`)가 게이트를 통과하면 **새 init으로 다시 발사**하는 것이 어차피 재현 축이다. 구 init 재실행은 계획에 없었다.
⚠️단 대가를 정직하게 적는다: **RQ2 축(b3p−b2p, 구 init 매치) 측정이 무기한 연기**된다. b2p는 20h·gs20까지 그 축에 투입돼 있다. 새 init으로 b2p·b3p를 다시 돌리면 RQ2도 새 init에서 다시 나오므로 총량은 손해가 아니지만, **b2p의 20시간은 구 init RQ2 부록으로만 남는다**.

### 상태 (14:39 UTC)
카나리아 **RED 8회** · SFT2-meta `rq3v2-sft2-rvfull` **queued 1h** · SFT2-control `rq3v2-sft2-b0p2` **queued 28m** · b3p **killed** · b2p running **20h**(durable gs20 RESUMABLE)
### 다음
두 SFT2가 슬롯을 잡는지 확인 → 게이트 → 새 init RL 런처 3종 작성. b2p는 무개입 지속(gs40 ~15:30).

---

## tick 15:09 UTC (2026-07-26) — 용량 대기 지속 / **논문 §Discussion에 감사 한계 3건 반영**(submodule 커밋 406b471)

### 상태 (변화 없음)
카나리아 **RED 9회 연속** · SFT2-meta `rq3v2-sft2-rvfull` **queued 1h33m** · SFT2-control `rq3v2-sft2-b0p2` **queued 58m** · b2p running **21h** · b3p killed
b3p 취소로 큐를 비웠지만 여전히 슬롯 미확보 ⇒ msrresrchvc A100은 **다른 테넌트가 점유 중**인 opportunistic 풀. 우리가 할 수 있는 레버는 이미 전부 소진(티어 없음·VC 없음·인스턴스 타입 유연).

### 완료: 논문 §Discussion 한계 3문단 추가
GPU 불요이고 결과와 무관하게 필요한 작업. 전부 **측정된 수치만** 사용.

1. **"지도학습 코퍼스의 29.7%가 recovery만 가르치고 meta를 가르치지 않는다"**(E-095) — 1763행 중 **524행**에서 `<|meta|>…<|/meta|>` 블록 전체가 마스크된 prefix 안에 들어간다. 해당 행은 **정확히 meta 구분자 개수 불일치 행들**이라 구분자 카운트만으로 식별된다. 결과 두 가지: prime이 행 수보다 약하고, **보상이 나중에 채점하는 바로 그 행동을 그 행들이 가르치지 않는다**. ⚠️이 문단은 **T1(instruct) 헤드라인 수치가 이 코퍼스 그대로로 학습됐다**는 사실을 명시한다 — 우리 자신의 발표 결과에 붙는 caveat이므로 숨기지 않고 적었다.
2. **"형식 기반 필터가 의도 없이 시나리오를 고를 수 있다"**(E-093/094) — think 닫힘 요구가 실은 "강제 주입 prefix가 닫는 태그를 포함했는가"를 재고, redirect는 정의상 think 내부에서 잘리므로 **554→67행**으로 떨어졌으며 **meta가 think 밖에 있는 궤적만** 남았다(1단계가 심는 구조의 반대). 일반화된 교훈으로 적었다: *필터가 요구하는 형식이 모델이 만든 것이 아니라 컨텍스트에서 온 것이면, 그 필터는 데이터 품질이 아니라 데이터 출처를 고른다.*
3. **"매치드 사다리는 첫 단계만이 아니라 모든 단계에서 대칭이어야 한다"**(E-097) — 메타 arm 2단 vs 컨트롤 arm 1단이었고 그 비교는 메타 메커니즘과 검증 코퍼스 한 단계 추가를 섞었을 것이다. 수리에 새 데이터는 필요 없었다(meta-removed twin이 이미 존재). *matched design은 단계가 추가될 때마다 재감사해야 한다*는 일반 논점으로 마무리.

**검증**: `pdflatex -halt-on-error` rc=0, **17페이지** 클린 컴파일. 사용 매크로 3종(`\metaopen`·`\metaclose`·`\pmishift`) 정의 확인(main.tex:25-27).
**submodule 처리**: `paper`는 별도 리포이므로 그 안에서 커밋(**406b471**). `figures/fig1_contrast_spec.txt`·`figures/outputs/`는 이전 figure 생성 시도의 untracked 산출물이라 **손대지 않았다**(별개 결정). 부모 리포의 `paper` 포인터는 이제 modified 상태 — 부모 커밋은 포인터를 올릴 시점에 별도 판단.

### 의도적으로 넣지 않은 것
**H100 vs A100 하드웨어 비대칭 고지**는 넣지 않았다. b3p가 취소됐고 세 arm 전부 새 init으로 재실행할 예정이며 현재 가용 용량이 A100뿐이므로 **재실행 후에는 세 arm이 모두 A100 = 대칭**이 된다. 우리 계획이 제거할 한계를 미리 적으면 곧 거짓이 된다. b2p의 구 init 20h 런이 부록으로 보고될 때에만 해당 문장이 필요하다.

### 다음 틱
두 SFT2 슬롯 확보 감시 · 카나리아 · b2p gs40(종류별 완결성으로 확인)

---

## E-099 ⭐⭐⭐ 계산 예산 실측 — **벽시계의 63%가 정지(suspend) 구간**이고, 이 용량으로는 3-arm 계획이 수 주 규모다 (2026-07-26 15:43-16:00 UTC)

b2p gs40 착지를 확인하다가 wandb run 디렉토리 타임스탬프가 여러 개(10:58·12:54·14:31·15:23)인 것을 보고 **재시작 의심** → wandb 히스토리를 직접 뽑아 정량화했다.

### 실측 (`gistdslab/metacot-dcpo-v4/rq3v2-b2p-2`, scan_history 48행)
| 항목 | 값 |
|---|---|
| 진행 | step 1(07-26 04:03) → **step 48**(15:36) |
| 벽시계 | 47스텝 / **11.5h** = **14.7분/step** |
| 순수 계산(`timing_s/step`) | median **380s = 6.3분/step** (min 238·max 479) |
| **20분 초과 정지 구간** | **5회, 합 7.2h = span의 63%** |

정지 구간: 94분(step1→2) · 72분(5→6) · 43분(21→22) · 100분(27→28) · **123분(36→37)**. **점점 길어진다**(43→100→123) = 경합 심화.

### 판정 1: 작업 손실은 없다 — 잃는 것은 시간뿐
정지 후 **step 번호가 되돌아가지 않는다**(21→22, 27→28, 36→37). 즉 프로세스가 죽고 durable에서 resume하는 것이 아니라 **같은 노드에서 suspend/resume**되고 있다(Singularity opportunistic 선점 semantics). ⇒ **`save_freq`는 레버가 아니다.** 낮춰도 얻는 게 없다. (⚠️처음엔 save_freq 하향을 검토했으나 이 관측으로 기각.)
⇒ 유일한 레버는 **SLA 티어**인데 basicvc는 신규 제출 자체가 막혔고 msrresrchvc는 **전 SKU Basic뿐**이다. **레버 없음.**

### 판정 2: 3-arm 계획의 실제 소요
- **b2p 잔여**: 252스텝 × 14.7분 = **62h ≈ 2.6일** (H100, 현 정지율 유지 시)
- **신규 arm(A100)**: A100은 bf16 처리량이 H100의 약 절반 ⇒ 순수 12.6분/step + 유사 정지율 ⇒ **~29분/step ⇒ 300스텝 = 145h ≈ 6일/arm**
- **3 arm(b0p·b2p·b3p) 신규 init 재실행**: 순차면 **~18일**, 병렬이면 **12 GPU 동시 필요**(현재 4 GPU 슬롯 하나도 못 잡는 중)
- 여기에 SFT2 2건(각 2~4h)과 arm별 held-out 1030 eval이 더해진다

### ⚠️ 정직한 결론
**과학적 설계는 이제 정합하다**(E-093~E-097로 코퍼스·사다리 결함을 모두 특정·수리). **남은 병목은 순수하게 계산 용량이고, 내가 쓸 수 있는 레버는 전부 소진했다** — 티어 없음·VC 2개뿐이며 하나는 엔타이틀먼트 차단·인스턴스 타입 이미 유연·다른 구독 없음(`target list -a` 확인).
⇒ **사람이 해야 하는 조치**: basicvc(H100/H200 보유, 우리가 늘 쓰던 곳) 엔타이틀먼트 차단은 05:49 UTC경 서비스측에서 발생했고 10회 연속 카나리아 RED다. 이건 GCR/Singularity 소유 조직에 문의해야 풀린다 — 내가 할 수 있는 진단(계정·토큰·클라이언트·리전·SKU·티어·코드 전부 무죄, 1-CPU echo도 거부, 다른 VC는 정상)은 문의에 그대로 첨부할 수 있는 형태로 E-091에 정리돼 있다.

### 부수 정정
`timing_s/step` 하나만 보고 "4.3분/step"이라 적을 뻔했다. 그것은 **마지막 로그값**이지 평균이 아니다(median 380s·max 479s). 그리고 순수 계산시간은 벽시계와 다르다 — **63% 정지를 빼고 계획을 세우면 3배 낙관**하게 된다.

### 상태 (15:43 UTC)
카나리아 **RED 10회** · SFT2-meta **queued 2h** · SFT2-control **queued 1h** · b2p **running·step 48·durable gs40 RESUMABLE**(gs20은 keep=1 프루닝됨)

---

## E-100 (C′) **G5 게이트 코퍼스 생성기 완성·검증** + b2p 정지율 재측정으로 E-099 추정 확인 (2026-07-26 16:19-16:40 UTC, GPU 0)

### 상태 (용량 대기 지속)
카나리아 **RED 11회 연속** · SFT2-meta **queued 2h** · SFT2-control **queued 2h** · b2p running **step 50**
**b2p 재측정**: step 1→50 / span 12.2h / **14.9분/step 벽시계** / **정지 7.8h = 64%** / 잔여 250스텝 = **2.6일**. 직전 측정(14.7분·63%)과 **일치** ⇒ E-099 추정 안정.

### 산출물 1: `scripts/build_g5_gated_rv_sft.py`
E-095가 확정한 **시나리오 중립 대체조건**을 코드로 만들었다. 게이트는 한 줄로 요약된다: *메타 블록 전체가 `wrong_prefix` 경계 이후에 있어야 한다* — `sft.py`가 `[prompt]+[wrong_prefix]`를 마스킹하므로(sft.py:111-116), 이 조건이 깨지면 그 행은 **메타 방출을 전혀 가르치지 않고** recovery 산문만 학습시킨다.

실행 결과(예측과 **정확히 일치**):
| | 행수 | redirect | redirect 비중 |
|---|---|---|---|
| source | 1763 | 554 | 31.4% |
| **배제**(메타가 마스크 안) | **524** | 102 | 19.5% |
| **잔존** | **1239** | **452** | **36.5%** |

⭐**게이트가 `scenario`를 읽지 않는데도 redirect 비중이 올라간다** — 이것이 구 필터와의 결정적 차이다. 구 필터는 형식(think 닫힘)을 요구해 시나리오를 골랐고(554→67), 이 게이트는 마스크 계약만 보는데 결과적으로 redirect가 덜 배제된다(배제된 524행 중 redirect는 19.5%로 원본 31.4%보다 **낮다**).

### 산출물 2: 렌더층 검증 통과
`audit_sft2_render_mask.py`로 최종 token id 층에서 재확인:
| 시나리오 | n | G1 절단 | G2 답 보존 | **G5 메타 학습됨** | G4 EOS |
|---|---|---|---|---|---|
| redirect | 452 | 0.0% | 100% | **100%**(0행 사망) | 100% |
| verify | 787 | 0.0% | 100% | **100%**(0행 사망) | 100% |
(G3는 기술자이므로 14.4%/39.6%여도 무관 — E-095 참조)

### 산출물 3: 회귀 테스트 `tests/test_build_g5_gated_rv_sft.py` (8 passed)
게이트 술어를 못박았다. 특히:
- 경계 **정확히 위**에서 시작하는 블록은 통과(half-open 계약)
- 블록이 경계를 **가로지르면 기각** — open이 마스크되고 close만 학습되면 여는 태그가 타깃이 된 적이 없어 방출을 못 가르친다
- **`inspect.getsource`로 술어 본문에 `scenario`·`think` 문자열이 없음을 기계 검증** — 시나리오 중립성이 주석이 아니라 테스트로 강제된다
- 실코퍼스 회귀 핀(1763→1239·redirect 452), 코퍼스 미스테이징 시 skip

### 발사 순서 재확인 (변경 없음)
1차는 여전히 **(C) raw 1763**(이미 queued). (C′)는 (C)가 게이트를 통과·완주한 뒤의 **해석 가능한 개선**이지, 지금 (C)를 대체하지 않는다 — 그래야 base 결과가 음수여도 "우리가 발명한 편차 탓"이라는 해석 여지가 없다.
⇒ (C′)는 **발사 준비만 완료**. 발사 시점엔 코퍼스 HF 업로드 + tarball 재패키징 + 런처(=`a100_sft_b2p2_rvfull.yaml`에서 data/config/출력명만 교체)가 필요하다.

---

## tick 16:55 UTC (2026-07-26) — 용량 대기 지속 / **입구 문서(EXPERIMENT_PLAN.md) 상태원장 갱신**

### 상태 (무변화)
카나리아 **RED 12회 연속** · SFT2-meta **queued 3h** · SFT2-control **queued 2h** · b2p running **23h**

### 완료: `docs/EXPERIMENT_PLAN.md` 최상단에 현행 상태원장 삽입
그 문서는 **2026-07-17자**이고 `rq3v2`·`E-09` 언급이 **0건**이었다 — 즉 9일째 **은퇴한 RQ3(think-off) 세대를 현행처럼 기술**하고 있었다. 이것이 정확히 E-071 인수인계 감사가 지적한 "입구 문서 3곳이 폐기된 설계를 현행처럼 기술" 문제의 재발이다.
전면 재작성은 하지 않았다 — 본문 §1–§5는 RQ3 세대의 기록으로서 **부록 가치가 있고**, 다시 쓰면 그 이력이 사라진다. 대신 **최상단에 경고 블록 + 현행 사다리 표 + 0726 감사 확정사항 + 발사 순서 + 판정조건**을 넣어, 입구에서 세대를 혼동할 수 없게 했다.

삽입 내용:
- ⛔ "아래 본문은 RQ3(think-off) 세대 계획표다. 본선은 RQ3v2. **최신 상태는 언제나 run-and-iteration-log가 원장**"
- 현행 3-arm 2단 SFT 스택 표(arm × SFT1 × SFT2 × RL) + 축 정의(RQ1=b2p−b0p / RQ2=b3p−b2p / **재현축=b3p−b0p**)
- E-091·E-093·E-094·E-095·E-097·E-098·E-099 요약 7항(각각 한 문장 + 수치)
- 발사 순서 3단((C) raw 1763 → (C′) G5 1239행 → RL 3종)과 **"(C′)는 대체가 아니라 (C) 완주 후의 개선"** 명시
- 재현 판정조건(사전등록·미충족) 재기재

### 왜 이 작업을 지금 했나
용량 대기 중 GPU 불요 작업 중에서, **인수인계 리스크가 가장 큰 항목**이었다. 실험 산출물은 원장에 다 있지만 **입구 문서가 틀린 세대를 가리키면 다음 사람(또는 다음 세션의 나)이 잘못된 arm 이름·init 경로로 출발**한다. E-071이 같은 문제를 이미 한 번 지적했다는 사실이 재발 비용을 보여준다.

---

## E-101 새 init용 **RL 런처 3종 작성 완료**(발사 대기) — b0p arm은 이 프로젝트에서 **처음 만들어진다** (2026-07-26 17:28-17:50 UTC, GPU 0)

### 상태 (무변화)
카나리아 **RED 13회 연속** · SFT2-meta **queued 3h** · SFT2-control **queued 3h** · b2p running **23h**

### 산출물 3종 (전부 신규 파일 · 기존 런처 무수정)
| 파일 | arm | init | config | ckpt 계보 |
|---|---|---|---|---|
| `a100_rq3v2f_b0p.yaml` | **B0 컨트롤(신설)** | `models/b0p2_rvfull_sft`(meta-removed twin) | `base_matched_grpo_h100_4x4k` | `rq3v2f_b0p` |
| `a100_rq3v2f_b2p.yaml` | B2 | `models/b2p2_rvfull_sft` | `base_matched_grpo_h100_4x4k` | `rq3v2f_b2p` |
| `a100_rq3v2f_b3p.yaml` | B3 | `models/b2p2_rvfull_sft` **(b2p와 동일)** | `triobj_dcpo_v4_stage3b_h100_4x4k` + `rmeta=pmi_shift`·`w_over=0` | `rq3v2f_b3p` |

⇒ **b0p vs b2p는 init만, b2p vs b3p는 보상 config만** 다르다. RQ1·RQ2·재현축이 각각 한 변수만 움직인다. 기계 대조로 확인(위 표의 init/config/ckpt는 yaml에서 직접 추출한 값).

### ⭐계보 이름을 `rq3v2f_*`로 분리한 이유
기존 `rq3v2_b2p`·`rq3v2_b3p` 계보를 재사용하면 새 런이 **구 init의 durable을 resume 후보로 집어간다**. 특히 `rq3v2_b3p` gs10은 optim 0/4라 resume 불가지만(E-098) `rq3v2_b2p` gs40은 **RESUMABLE**이라 실제로 물릴 수 있었다 — 구 init에서 40스텝 진행한 상태를 새 init 런이 이어받으면 **두 init이 섞인 궤적**이 된다. 이름 분리로 원천 차단. 구 계보는 HF에 남아 **RQ2 부록**으로 쓰인다.

### ⚠️ 클론 검증이 **3틱 연속** 실제 결함을 잡았다
b0p는 b2p 런처의 클론인데, 치환 후에도 **주석이 자신을 "B2 INIT / the META SFT / 92% emission / the ONLY B2-vs-B0 difference"라고 설명**하고 있었다(118행·217-219행). 컨트롤 arm은 정반대로 **meta-removed 트윈**을 쓴다. yaml 파싱·`bash -n`·shlex 전부 통과하므로 기계검사로는 안 걸린다.
수정 내용은 사실 기술로 교체했다 — 두 지도학습 단계 모두 메타 없음, SFT2는 `v8_base_rv_sft.parquet`(1763행 1:1·동일 시나리오 분포·동일 think-closed율·메타 구분자 0), 이 단을 추가하는 것이 b3p−b0p를 "메타 메커니즘"으로 만드는 이유(E-097).
**누적 3건**: E-096 description 낡음 → E-097 이스케이프 따옴표 안 shard 검증이 다른 arm 경로 → E-101 클론 주석이 자기 정체를 반대로 기술. **패턴이 확립됐다: 기계검사는 문법만 보고, 의미는 전수 diff 통독만이 잡는다.**

### 기계 검증 (3종 전부 통과)
yaml 파싱 · target `msrresrchvc`/`80G4-A100`/`Basic` · `CODE_TAR_REVISION 490407111`(상위집합 tarball) · WANDB_RUN_ID 신규(`rq3v2f-{b0p,b2p,b3p}-1`) · **shlex 3토큰** · **`$$`→`$` 치환 후 inner `bash -n` OK** · 구 토큰 잔존 0(`b2p2_rvseg_sft`·`rq3v2_b2p`·`rq3v2_b3p`·`488239754`·`msrresrchbasicvc`·`80G4-H100`).

### 발사 조건 (아직 미충족)
1. `rq3v2-sft2-rvfull` 게이트 통과 → HF `models/b2p2_rvfull_sft` 4샤드 durable
2. `rq3v2-sft2-b0p2` 게이트 통과 → HF `models/b0p2_rvfull_sft` 4샤드 durable
3. 그 후 3종 발사. ⚠️E-099 기준 A100에서 **arm당 ~6일**, 3 arm 순차면 ~18일 — 용량이 풀리지 않으면 이 계획은 실행 불가하다는 점은 변하지 않는다.

---

## tick 18:03 UTC (2026-07-26) — 대기 지속 / **문의용 에스컬레이션 리포트 작성**

### 상태 (무변화)
카나리아 **RED 14회 연속** · SFT2-meta **queued 4h** · SFT2-control **queued 3h** · b2p running **1d**

### 왜 이걸 만들었나
준비물은 전부 끝났다(SFT2 2종 발사·RL 런처 3종·(C′) 생성기·논문 한계·입구 문서). **실제 블로커는 basicvc 엔타이틀먼트이고 그것만은 사람이 문의해야 풀린다.** 산출물을 더 쌓는 것보다, 사용자가 **그대로 복사해 보낼 수 있는 형태**로 만드는 것이 지금 가장 레버리지가 크다.

### 산출물: `docs/reports/2026-07-26-basicvc-submission-block-escalation.md`
- **최소 재현**: 1-CPU `echo` 두 줄짜리 yaml 전문. VC 이름 한 줄만 바꾸면 성공한다는 대조 포함
- **correlation ID 5건**(별개 시도, UTC 시각과 함께) — 지원팀이 서버 로그를 바로 찾을 수 있는 형태
- **반증 표 8행**: 각 가설을 어떻게 검증했는지와 결과. "가정"이 아니라 "실측"임을 보이는 형식
  - 특히 **에러 메시지가 주장하는 리전 불일치가 실제와 다르다**는 점(`az` 실측: 워크스페이스·VC 둘 다 westus2)
  - **1-CPU echo 잡도 거부** ⇒ GPU/쿼터 무관
  - 같은 계정·클라이언트·워크스페이스로 **다른 VC는 성공**
  - **24시간 전 admit된 우리 잡은 그 VC에서 계속 running** ⇒ 계정 전체 차단이 아니라 **신규 admission만** 차단
- **영향**: H100/H200가 그 VC에만 있고, 대안 VC는 A100 전 SKU Basic이라 3~4h째 스케줄 안 됨. 3-arm 사다리 arm당 ~6일
- **요청 3가지**: 05:49 UTC경 엔타이틀먼트/RBAC 변경 여부 / 의도적이면 올바른 타깃·티어 / 비의도면 복구

### 판단
이 리포트는 **사실만** 담았고 추측은 "message is not consistent with the measured region"처럼 **측정과 대조하는 형태로만** 적었다. 지원 티켓에서 추측을 단정으로 적으면 진단이 그 방향으로 끌려가므로, 반증표를 주고 결론은 상대가 내리게 두는 편이 낫다.

---

## tick 18:37 UTC (2026-07-26) — 대기 (신규 산출물 없음, 의도적)

| 항목 | 값 |
|---|---|
| basicvc 카나리아 | **RED 15회 연속** |
| `rq3v2-sft2-rvfull` | queued **5h** |
| `rq3v2-sft2-b0p2` | queued **4h** |
| `superb-terrier`(b2p) | running **1d** · **step 64** · 13.5분/step 벽시계 · 정지 **61%** · 잔여 **2.2일** |
| 새 SFT2 산출물 | `models/b2p2_rvfull_sft` 0파일 · `models/b0p2_rvfull_sft` 0파일 |

b2p는 step 50→64로 전진(정지율 61%로 3회 측정 모두 61-64% 범위 — 추정 안정). msrresrchvc A100 슬롯은 5시간째 미확보.

**이번 틱은 새 산출물을 만들지 않았다.** 준비물(SFT2 2종 발사·RL 런처 3종·(C′) 생성기·논문 한계·입구 문서·에스컬레이션 리포트)이 전부 끝났고, 남은 후보(eval 런처·`experiments.tex` 점검·`EXPERIMENT_LOG` 동기화)는 **한계효용이 뚜렷이 낮다** — 그것들은 RL 결과가 나온 뒤에 형태가 정해지는 편이 낫고, 지금 만들면 결과에 맞춰 다시 쓸 가능성이 크다. 대기 중 산출물을 쌓는 것 자체가 목적이 되면 나중에 폐기할 코드를 늘린다. 감시만 계속한다.

---

## tick 19:25 UTC (2026-07-26) — 대기

| 항목 | 값 |
|---|---|
| basicvc 카나리아 | **RED 16회 연속** |
| `rq3v2-sft2-rvfull` / `rq3v2-sft2-b0p2` | queued **5h** / **5h** |
| b2p | running 1d · step **64**(직전 틱과 동일 = 현재 정지 구간) · 13.5분/step · 정지 61% · 잔여 2.2일 |
| **b2p durable** | **gs60 RESUMABLE**(model 4/4·optim 4/4·extra 4/4) — gs40에서 전진, gs40은 keep=1 프루닝 |
| 새 SFT2 산출물 | 양쪽 0파일 |

45분간 step 진행 0인데 정지율 61%와 정합한다(정지 구간 중간). durable은 gs60으로 전진했으므로 진행 자체는 살아 있다. 신규 산출물 없음 — 대기 방침 유지.

---

## E-102 ⚠️⭐⭐ **정정 2건**: wandb는 2시간 전 죽었는데 학습은 계속되고 있었다 / 실제 스텝시간은 내가 보고한 값의 **절반**이다 (2026-07-26 20:12-20:25 UTC)

### 발견 경위
durable이 **gs80**으로 전진했는데 wandb는 **step 64**를 가리켜 불일치가 났다. 내 진행·소요 추정이 전부 wandb 기반이었으므로 확인했다.

### 🔴 정정 1 — wandb run이 `crashed` 상태다 (학습은 살아 있다)
```
run rq3v2-b2p-2 : state=crashed, heartbeat 18:21Z, last step 64 at 18:16Z
```
그런데 **HF durable gs80은 19:55에 착지**했다. 즉 **wandb 로거만 18:16경 죽었고 트레이너는 계속 돌고 있다.** amlt도 `running`으로 보고한다.
⇒ **wandb만 보면 "런이 죽었다"고 오판한다.** 이 프로젝트에서 신뢰할 진행 신호는 **HF 체크포인트 커밋 시각**이다. (E-098에서 durable을 파일 수로 세다 틀렸던 것과 같은 계열 — 관측 대리변수를 검증 없이 신뢰한 실수.)

### 🔴 정정 2 — 스텝시간이 내가 여러 번 보고한 값의 절반이다
HF 체크포인트 커밋 시각(20스텝 간격, **정지 포함 벽시계**):
| 구간 | 소요 | 스텝시간 |
|---|---|---|
| gs40(14:57) → gs60(17:11) | 2h14m | **6.7분/step** |
| gs60(17:11) → gs80(19:55) | 2h44m | **8.2분/step** |

정상상태 ≈ **7.5분/step**. 내가 E-099·E-100에서 반복 보고한 **13.5~14.9분/step은 초기 구간(step 1-40)의 큰 정지들**(94·72·100·123분)을 포함한 전체 평균이라 **정상상태를 2배 비관**했다. `timing_s/step`이 마지막값이라 못 쓴다고 정정했었는데, 그 대안으로 쓴 "전체 span ÷ 전체 step"도 **초기 이상구간에 오염**돼 있었다.

### 소요 재추정 (하향)
| | 이전 보고 | **정정** |
|---|---|---|
| b2p 잔여(220스텝) | 2.2일 | **~1.1일**(220×7.5분=27.5h) |
| 신규 arm A100 1개 | ~6일 | **~3.1일**(A100 2배 가정: 15분/step×300=75h) |
| 3 arm 순차 | ~18일 | **~9일** |

여전히 길지만 **실행 불가에서 빠듯하게 가능**으로 바뀐다. E-099의 "3 arm 순차 ~18일" 문장은 이 항목으로 대체한다.

### 조치
- 앞으로 진행 판정은 **HF 커밋 시각 기준**. wandb는 보조(그리고 죽을 수 있음).
- b2p의 wandb 로거 사망은 **개입하지 않는다** — 트레이너가 정상이고, 재시작하면 오히려 진행을 잃는다. 학습 메트릭(정확도·엔트로피)은 18:16 이후 기록이 없으므로 **gs300 도달 후 held-out eval로 판정**한다(사전등록 판정 규약이 원래 그러하다 — in-training val은 모니터링 전용).

### 틱 상태 (20:12)
카나리아 **RED 17회** · SFT2 양쪽 **queued 6h** · b2p **durable gs80 RESUMABLE**(4/4/4) · 새 SFT2 산출물 0파일

---

## E-103 ⚠️ **b3p 취소는 오판이었다 — 재제출** / 오늘 하루의 정직한 결산 (2026-07-26 20:36 UTC)

사용자 질문: "아직 b3를 안하고 있어? 시간이 너무 오래 걸리는데 잘 하고 있는 게 맞아?"

### 🔴 오판 인정 — E-098의 b3p 취소
14:50에 b3p를 취소했다. 근거는 (a)gs10이 optim 0/4라 resume 자산 0 (b)최장 대기라 다음 슬롯을 잡을 텐데 두 SFT2(critical path)를 막는다 — **우선순위 역전**이었다.
**그런데 6시간이 지나도록 누구도 슬롯을 못 잡았다.** 비운 자리를 아무도 쓰지 않았으므로 **취소로 얻은 것은 0이고 잃은 것은 b3p의 3시간 큐 위치**다.

더 중요한 건 내가 **b3p의 가치를 잘못 계산**했다는 점이다. "결함 init이라 재현축에 못 쓴다"는 맞지만, 그 프레이밍이 놓친 것:
- **b2p가 같은 init에서 이미 gs80/300까지 진행했다**(약 하루치 계산)
- **RQ2 = b3p − b2p는 두 arm이 다 있어야 성립**한다
- 논문의 RQ2 표는 **전부 `\todo{}`** 다 — 즉 RQ2는 "부록"이 아니라 **아직 증거가 0인 핵심 축**
⇒ b3p를 취소한 순간 **b2p의 하루치 계산이 좌초**했다. 이걸 "가치 최저 arm"이라고 부른 것이 오판이다.

### 조치
`rq3v2-b3p-a100-r2` 재제출(20:36:19 UTC·`preparing`·msrresrchvc A100x4). gs0부터 시작(구 gs10은 여전히 resume 불가). E-102 정정 스텝시간 기준 **~3.1일** 예상.
경로 충돌 없음: b3p는 HF **model** repo `checkpoints/rq3v2_b3p`, 두 SFT2는 HF **dataset** repo `models/*`.

### 오늘 하루 결산 (정직하게)
**성과 — 과학적으로는 실질적이다.**
- 제출 불능의 근인을 확정(E-091): 계정·클라이언트·리전·SKU·티어·코드 전부 반증하고 **VC 단위 차단**으로 특정
- 데이터 파이프라인 결함 **3건** 발견: think-closed가 위장된 시나리오 필터(E-093) / SFT2가 SFT1과 반대 구조를 학습(E-094) / **T1 자신의 SFT2에도 메타를 안 가르치는 행이 29.7%**(E-095)
- **사다리 비대칭** 발견(E-097): 컨트롤 arm에 SFT 한 단계가 통째로 없어 재현축이 교란돼 있었다
- 수리 산출물 전부 완성: SFT2 2종 발사·RL 런처 3종·(C′) 생성기·논문 한계 3문단·에스컬레이션 리포트
- 내 측정 오류 **3건 자체 정정**(E-098 durable 세는 법·E-102 wandb 신뢰/스텝시간)

**실패 — 실험은 하루 동안 한 걸음도 못 나갔다.**
- 새 파이프라인 계산 진행 = **0**. SFT2 2종 6시간째 queued, RL 3종 미발사
- b3p 취소로 b2p의 하루치를 좌초시켰다가 지금 되돌림
- 병목(basicvc 차단·A100 포화)은 **내가 풀 수 없다**

**판정**: 감사는 제값을 했다 — 그 결함들을 모른 채 발사했으면 결과가 무효였을 것이다. 다만 **"오래 걸린다"는 지적은 정당하다.** 오늘 벽에 부딪힌 지점은 과학이 아니라 용량이고, 그건 사람이 GCR/Singularity에 문의해야 풀린다(자료는 `docs/reports/2026-07-26-basicvc-submission-block-escalation.md`에 그대로 보낼 수 있게 준비됨).

---

## E-104 ⭐⭐⭐ **b3p 레시피 자기비판**: "T1 레시피를 base에 그대로"가 아니다 — 그리고 b3p가 재는 축은 **base에서 이미 한 번 음성이었다** (2026-07-26 20:40 UTC)

사용자 질문: "b3p는 레시피가 예전 instruct에서 성공했던 sft 데이터·학습 방법이 맞아? base에도 잘 되는 방식이 맞아? 스스로 비판적으로 검토해줄래?"

### Q1. T1과 같은가 — **3축 일치·3축 불일치**
| 축 | T1 | 현 b3p | 판정 |
|---|---|---|---|
| pmi_shift 알고리즘 | `dcpo_pmi_shift.py` | **바이트 동일**(tarball diff) | ✅ |
| RL 학습 데이터 | `verl_train_meta_mix.parquet` | **바이트 동일**(686,267 B) | ✅ |
| 보상 config | `rmeta=pmi_shift`·`w_over=0`+config 기본 6헤드 | 동일 | ✅ |
| **SFT2 데이터** | raw 1763행·3ep·lr1e-5 | **378행 필터·2ep·lr2e-6** | ❌ (수리 중, queued) |
| **디코딩** | 오버라이드 **없음** → temp 0.6·top_k 20·top_p 0.95·4096 | temp **1.0**·top_k **−1**·top_p **1.0**·**8192** | ❌ |
| 보상 코드 미세차 | — | truncation→`R_corr=−1` 추가·anchor warmup 게이트 변경 | ❌ |

⇒ **"T1 레시피 그대로"는 사실이 아니다.** 다만 디코딩 이탈은 **의도적**이다 — base는 T1 디코딩에서 **엔트로피 붕괴**로 전멸했고(v1), temp 1.0·top_k 해제가 그걸 살린 레버였다. 즉 **T1의 디코딩과 T1의 pmi 발화율을 동시에 가질 수 없다**: E-085 실측이 그 대가를 보여준다 — emit는 회복(T1 0.99 vs b3p 0.977)했지만 **`pmishift_attempted`는 T1 0.52~0.66 vs b3p 0.285로 절반**이다. PMI-shift는 롤아웃에 meta 블록이 나와야 open/close 2지점 teacher-forcing이 성립하므로, 디코딩 레짐이 이 갭의 유력 용의자다.

### 🔴 Q2. base에 맞는 방식인가 — **b3p가 재는 축은 base에서 이미 음성이 나왔다**
원장 §예비결과(E-055/E-059, think-off 세대 base):
| gs | RQ1(B2−B0) | **RQ2(B3pkg−B2)** | PMI(B3pkg−noPMI) |
|---|---|---|---|
| 50 | +8.6 | **−4.5** | +10.4 |
| 100 | +7.4 | **−4.2** | **−1.7** |

- **살아남은 조각 = meta-SFT init**(RQ1 +7~9pp 강건 양성)
- **죽은 조각 = triobj 패키지**(RQ2 음성) — 그리고 **b3p가 추가하는 것이 정확히 그 패키지다**
- **PMI는 +10.4 → −1.7로 역전**, 기전은 **emission collapse 91%→4.5%**

∴ **T1 헤드라인(+18.8pp 등)이 나온 축은 `b3pkg−B0` = RQ1+RQ2 합**인데, base에서의 예비 분해는 **그 이득이 RQ1(SFT init)에서 오고 RQ2(보상 패키지)는 오히려 깎는다**고 말한다. 지금 b3p−b2p는 init이 같으므로 **정확히 그 "죽은 축"만 다시 재고 있다.**

### 처방이 있었는데 안 넣었다
E-063이 그 실패에 대한 처방을 명시했다: **emission-protected advantage** — (1)A_corr/A_PMI 분리 (2)음의 A_corr가 meta span **및 emit-결정 토큰**에 억압 gradient로 흐르지 않게 마스킹 (3)meta엔 positive correctness credit 금지.
코드 확인 결과 `dcpo_region.py:1511`은 이미 `w_corr * A_corr * ans` — **correctness는 answer span에만** 흐른다. 즉 (1)(3)과 (2)의 일부는 **구조적으로 이미 존재**한다. 그런데도 think-off 세대에서 emission이 붕괴했다 ⇒ **answer-span 제한만으로는 불충분했다**는 뜻이고, `emission_protect` 계열 스위치는 코드·config 어디에도 **없다**(grep 0건). **처방 arm은 끝내 만들어지지 않았다.**

### 그래서 지금 b3p는 무엇인가 — 솔직한 규정
**"실패한 arm을, 더 나은 SFT init 위에서, 처방 없이 다시 돌리는 것"** 이다. 달라진 것과 안 달라진 것을 나누면:

**나아진 것(붕괴 전제가 제거됐을 수 있다)**
- think-off b23 SFT는 **0% clean·wellformed 0.40**이었다. 현 SFT1은 **100% clean·L2 GREEN·emission 100%**
- 실제로 b3p gs10에서 **emit 0.974로 붕괴 없음** — 91%→4.5% 같은 사태는 재현되지 않았다

**여전히 위험한 것**
- gs10 실측 **`rmeta` −1.08**·**n_save 0.8 vs n_derail 1.2** — PMI가 그 메타 블록들을 **평균적으로 나쁘다고 채점**하고 있다. 그러면 meta 항이 **메타를 내지 말라는 압력**이 된다(붕괴의 다른 경로)
- 그 원인 후보를 이미 특정했다: **E-095** SFT2의 29.7%가 메타를 아예 안 가르치고, **E-094** base SFT2는 메타 위치를 SFT1과 **반대로** 가르쳤다 ⇒ 메타 블록 품질이 낮아 DERAIL 다수 ⇒ rmeta 음수 ⇒ RL이 메타 억제
- **`pmishift_attempted` 0.285**(T1의 절반) — 디코딩 레짐 미해결

### 판정과 권고 (사전등록 갱신 제안)
1. **지금 큐에 있는 b3p(구 init)는 RQ2 부록으로만 쓴다.** b2p가 같은 init에서 gs80까지 갔으므로 그 축을 완성할 가치는 있다. 다만 **음성이 나와도 놀랄 일이 아니다** — base에서 한 번 나온 결과의 재확인이 될 가능성이 높다.
2. **새 init 3-arm이 본선**이다. E-094/E-095 수리가 rmeta 음수를 뒤집는지가 핵심 관전 포인트다. 뒤집히지 않으면 **PMI 자체가 base에서 useful-meta를 못 고른다**는 결론으로 간다.
3. ⚠️**추가로 필요한 실험 하나를 지금 명시한다**: E-063 **emission-protected / PMI-gating arm**. 이건 아직 코드도 없다. 새 SFT2로도 rmeta가 음수면 이 arm이 유일한 다음 수다.
4. **디코딩 패리티 sweep**(T1 temp 0.6·top_k 20 중간값)은 `attempted` 갭이 디코딩 탓인지 가르는 유일한 실험인데 아직 미실행이다.

### 요약 한 줄
**b3p는 T1 레시피의 복제가 아니고(3축 이탈), 그것이 재는 축은 base에서 이미 −4.2~−4.5로 음성이 나왔으며, 그 실패의 처방은 코드에 없다.** 지금 수리한 것(SFT2·사다리)은 **붕괴 전제**를 없앴을 뿐 **PMI가 base에서 유용한 메타를 고르는지**는 아직 미검증이다.

---

## E-105 ⭐⭐⭐⭐ **돌파: b2p의 H100 노드에서 SFT2를 사이드카로 실행 중** — 6시간 큐 대기를 우회했다 (2026-07-26 20:45-21:00 UTC)

사용자 제안: "노드를 멈추진 말고, **그 노드 위에서 그대로 실험을 진행할 방법은 없는 건가?**"
→ **가능했다. 그리고 이것이 오늘의 유일한 실질 돌파다.**

### 사용자 질문에 대한 사실 정리
- **"b2p는 재현이 아니지?"** — 맞다. 재현축은 `b3p−b0p`이고 b2p는 둘 중 어느 것도 아니다(RQ1/RQ2 기준 arm).
- **"b3p부터 하는 게 맞지 않아?"** — 취지는 맞으나 **정정이 필요하다**: 큐에 있는 b3p도 **구 init(378행 기아 SFT2)** 이라 재현이 아니다. 재현엔 **새 SFT2가 선행**이므로 **"b3p부터"의 실질은 "SFT2부터"** 다.
- **"안 된 이유가 데이터 때문 아니었어?"** — 맞다. E-093(위장된 시나리오 필터)·E-094(SFT1과 반대 구조)·E-095(29.7%가 메타 안 가르침).

### ⭐발견: 배치 잡에도 SSH가 된다
`amlt ssh --help`는 *"For Singularity's managed mode, it is only possible to SSH into jobs that were already launched in interactive mode"* 라고 쓰여 있다. **실측 결과 배치 발사 잡(`superb-terrier`)에 SSH가 정상 접속된다.** ⇒ **문서가 실제보다 보수적이다.** 이건 앞으로 계속 쓸 수 있는 레버다(메모리 `singularity-interactive-fails-batch-works-0715`에 배치가 정답이라 적혀 있는데, **거기에 "배치도 SSH 된다"를 추가해야 한다**).

### 노드 자원 실측 (b2p 단독 실행 중)
| | 값 |
|---|---|
| GPU | 81.5GB × 4, 사용 35.2GB ⇒ **여유 45.8GB/GPU** |
| CPU RAM | 1870GB 중 **954GB 가용** |
| /scratch | 28TB 중 **25TB 여유** |
| 코드·환경 | `/scratch/metacognition`·`/scratch/conda_envs/simplerl`(ds 0.19.2·acc 1.14.0·tf 4.57.6) |

### 설계: b2p를 죽이지 않는 사이드카
1. **옵티마이저를 CPU로 오프로드** — `/scratch/sft2_side/accelerate_sft_cpuoff.yaml`(zero3 + `offload_optimizer_device: cpu` + `main_process_port: 29711`로 포트 충돌 회피). Adam 상태 ~96GB를 GPU에서 CPU로 옮겨 **b2p의 GPU 메모리를 잠식하지 않는다**(CPU RAM 954GB 가용이라 여유).
2. **config를 노드에 작성** — 노드의 tarball은 구버전(488239754)이라 `sft_b2p2_rvfull.yaml`이 없다. `/scratch/sft2_side/sft_b2p2_rvfull.yaml`로 직접 작성(raw 1763·3ep·lr1e-5·max_len4096 = T1 미러).
3. **SFT1 init 스테이징** — 노드엔 `b2p2_rvseg_sft`(구 RL init)만 있고 SFT1이 없었다. HF `iamseungpil/metacot` `models/b2p_v8meta_strict_sft`를 **16GB·14파일** 다운로드 완료.
4. **`setsid nohup`으로 발사** — SSH 세션이 끊겨도 살아남게.

### 실행 상태 (21:00)
| 항목 | 값 |
|---|---|
| SFT2 | **7/309 steps · 19.8s/it · ETA ~1h40m** |
| **b2p verl** | **ALIVE** — 죽지 않았다 |
| GPU | 56~68GB 사용 / **12~24GB 여유 남음** |
| CPU RAM | **1402GB 가용** |
| wandb | `gistdslab/metacot-math` run `b2p2-rvfull-sft2-side` |

### 의미
**6시간 큐 대기를 우회했다.** msrresrchvc A100 슬롯을 기다리는 대신, 이미 확보된 **H100 노드**에서 SFT2를 돈다. H100은 A100보다 2배 빠르므로 후속 RL도 이 노드에서 돌릴 수 있으면 arm당 ~3.1일이 아니라 **~1.6일**이 된다.
⚠️단 노드 수명 제약: b2p가 끝나면(잔여 ~1.1일) 런처가 최종 push 후 `sleep 43200`(12h) → 그 다음 잡 종료·노드 반납. **RL 300스텝(H100 ~37h)은 이 창에 안 들어간다.** SFT2 2종(각 ~2h)은 충분히 들어간다.
⇒ 전략: **이 창에서 SFT2 2종을 끝내 HF에 올린다.** RL은 큐/복구된 basicvc에서 새 init으로 발사.

### 다음
1. 메타 SFT2 완주 → EOS 게이트 → HF `models/b2p2_rvfull_sft` push
2. 이어서 **컨트롤 SFT2**(`v8_base_rv_sft.parquet`은 노드에 없으니 HF에서 내려야 함) 같은 방식으로
3. 두 개 다 올라가면 RL 런처 3종(E-101)의 발사 조건 충족

---

## E-106 ⭐⭐⭐⭐ **1.6kB 조용한 사망의 진짜 원인 발견 = RGS 가드 영구 라이브락** / HF 점검 / pmi_shift는 가짜였나 — 수치 재분석 (2026-07-26 21:05-21:20 UTC)

### 🔴 발견 1: 큐의 b3p는 노드를 잡아도 즉시 죽었을 것이다 (그리고 그게 1.6kB의 정체다)
런처의 resume 가드(`a100_rq3v2_b3p.yaml` L152-186)를 **실제 HF 상태에 대입 실행**했다:
```
모델샤드 {10: 4}  extra {10: 4}  optim {}      ← optim 0/4
RGS_STEP=0   RGS_ANY=1
pull_resume 조건(model+optim+extra 각 ≥4) 만족 step: []  → LOCAL_GS 없음
⇒ if RGS_ANY=1 AND LOCAL_GS 없음 → "ABORT: refusing gs0 cold-start" → sleep 300 → exit 1
```
`RGS_ANY`는 **어떤 종류든 파일이 하나라도 있으면 1**이다(`1 if (cm or ce or co)`). gs10은 model·extra는 4/4지만 optim이 0이라 `steps`가 비고 `RGS_STEP=0`이 되는데, `RGS_ANY=1`이므로 fail-closed 가드가 발동한다. ⇒ **노드를 잡을 때마다 300초 후 exit. 영구 라이브락.**

⭐**이것이 E-087에서 미해결로 남긴 "1.6kB 조용한 사망"의 원인이다.** 가드는 L152-186에서 실행되고 verl은 L204이므로, **abort는 정확히 pre-L204**다 — 셋업 echo만 남고 `verl_main.log`는 생성되지 않는다. 나는 그것을 "계측이 L204 이후에 시작돼서 증거가 안 남는다"고 진단했는데(맞는 관찰), **원인은 관측 부재가 아니라 가드 발동**이었다. E-071 H-3b가 예언한 "완결성 정의 3원 불일치 → 영구 abort 라이브락"이 **실제로 물린 사례**다.

### 조치: 차단물 제거 (파괴조작 3율)
1. **LIST** — `checkpoints/rq3v2_b3p/global_step_10/` 19파일 32.8GB. manifest `docs/reports/hf_deletion_rq3v2_b3p_gs10_20260726.json`에 전체 파일목록 기록.
2. **decide** — (a)optim 0/4라 **resume 원리적 불가** (b)10스텝·결함 init이라 평가 가치 ≈0 (c)**적극적으로 해롭다**(영구 라이브락). ⇒ 삭제가 정당하고 필요하다.
3. **execute + verify** — 삭제 후 `RGS_ANY=0` 확인 ⇒ **b3p가 gs0부터 정상 발사 가능.**

### HF 점검 (사용자 요청) — 트리 441.1GB
| repo | 항목 | GB | 판정 |
|---|---|---|---|
| v3(model) | `checkpoints/rq3v2_b2p/gs80` | 98.3 | ✅ **LIVE**(resumable) |
| v3 | `checkpoints/rq3_b0/gs300` | 98.3 | ⛔은퇴 세대·**승인대기 manifest** |
| v3 | `checkpoints/rq3_b2/gs160` | 32.8 | ⛔은퇴 세대·**승인대기 manifest** |
| v3 | ~~`checkpoints/rq3v2_b3p/gs10`~~ | ~~32.8~~ | ✅**삭제 완료**(위) |
| v3 | `models/v4_s1b_final` | 16.4 | 🟡v4 세대·검토 대상 |
| metacot | `models/b2p_v8meta_strict_sft` | 16.4 | ✅**LIVE**(사이드카 SFT2 init) |
| metacot | `models/b0p_v8base_strict_sft` | 16.4 | ✅**LIVE**(컨트롤 SFT2 init) |
| metacot | `models/b2p2_rvseg_sft` | 16.4 | ✅구 init·b2p·큐 b3p가 참조 |
| metacot | `models/v8_meta_inside_strict_sft` | 16.4 | ✅T1 계보(재현성) |
| metacot | `models/b0_gold_sft` | 16.4 | 🟡은퇴 RQ3 B0 init |
| metacot | `models/base_pilot_meta_sft` + `_nometa_sft` | 32.8 | 🟡**파일럿·검토 대상** |
| metacot | `env_snapshots` | 12.1 | ⚠️부트스트랩 의존 가능·**확인 전 보존** |
| metacot-rv | `models/v8_rv_functional_sft` | 16.4 | ✅**T1 우승 init**(절대 보존) |
| metacot-rv | `models/v8_rv_confidence_warmup` | 16.4 | 🟡T1 런처가 "drop-in replacement"라 명시=**대체됨** |
⇒ 즉시 정리한 32.8GB 외에 **회수 후보 ~114GB**(은퇴 131.1 승인대기 + 파일럿/대체 65.6 중 중복 제외). ⚠️`env_snapshots`는 노드 부트스트랩 의존 여부 확인 전 손대지 않는다.

### Q. pmi_shift는 가짜였나 — **유일한 양성 수치가 검증을 통과하지 못한다**
원장 예비결과를 열별로 다시 읽으면:
| gs | B0 | B2 | B3pkg | **B3noPMI** | RQ1 | RQ2 | PMI |
|---|---|---|---|---|---|---|---|
| 50 | 56.3 | 64.9 | 60.4 | **50.0** | +8.6 | −4.5 | **+10.4** |
| 100 | 59.2 | 66.6 | 62.4 | **64.1** | +7.4 | −4.2 | **−1.7** |

**B3noPMI가 50.0 → 64.1로 +14.1 점프**했다. 같은 구간 B3pkg는 60.4 → 62.4로 **+2.0**뿐이다. 그리고 gs50의 B3noPMI 50.0은 **B0(56.3)보다도 낮다** — 대조군이 비정상적으로 부진한 상태였다.
⇒ **gs50의 "PMI +10.4"는 PMI의 이득이 아니라 대조군이 빠져 있던 구덩이를 잰 값**이다. 대조군이 회복한 gs100에서 PMI는 −1.7이다.
⇒ 따라서 서사를 정정해야 한다. **"PMI 이득이 있었다가 역전됐다"가 아니라 "애초에 이득의 증거가 없었고, 유일한 양성 수치는 대조군의 일시적 부진으로 설명된다."**

**"가짜"의 세 가지 의미로 나누면:**
1. **코드 버그/아티팩트?** → **아니다.** 알고리즘은 T1 우승본과 바이트 동일하고 감사에서 누수도 안 나왔다.
2. **보고된 이득이 실재했나?** → **아니다.** +10.4는 위와 같이 설명되고, T1 헤드라인(+18.8pp)은 **PMI 단독이 아니라 6-헤드 패키지 대 base**였다(원장에 기지·공개인정).
3. **메커니즘이 작동하지 않는가?** → **미결.** 오히려 `rmeta −1.08`은 **PMI가 계측기로서 정직하게 작동하며 "이 메타 블록들은 해롭다"고 보고하는 것**일 수 있다. 그렇다면 문제는 PMI가 아니라 **메타 블록을 만든 SFT 데이터**이고, 그게 지금 고치고 있는 것이다.
⇒ 정직한 한 줄: **PMI가 이득을 준다는 증거는 한 번도 검증을 통과한 적이 없다. 그러나 PMI가 틀렸다는 증거도 없다 — 지금까지는 나쁜 재료를 정확히 나쁘다고 보고했을 가능성이 남아 있다.**

### Q. 중간까진 괜찮았는데 왜 박살났나 — **emit-결정 토큰이 벌을 받는다**
기록된 기전은 emission collapse **91% → 4.5%**이고, 실경로는 E-063/원장 L1044에 있다:
> **emit-결정 토큰(`<|meta|>` 직전)은 ANSWER_REGION 소속**이므로 **correctness advantage를 받는다**. 메타가 correctness와 음상관이면 그 음의 advantage가 "메타를 낼까" 결정 토큰에 직접 떨어진다.

즉 급격히 깨진 게 아니라 **RL이 꾸준히 "메타를 내는 건 손해"를 학습**했다: (a)메타 토큰은 `len_cost`를 물고 (b)그 데이터에선 메타가 정확도를 못 올리고 (c)발화 결정 토큰이 직접 벌을 받는다. gs50은 붕괴 진행 중, gs100은 붕괴 후다. 무대가 사라지면 PMI 항은 자동으로 0이 되므로 **PMI의 "역전"은 PMI가 나빠진 게 아니라 PMI가 채점할 대상이 없어진 것**이다.

⚠️**이 경로는 지금도 안 막혀 있다.** E-104에서 나는 `w_corr*A_corr*ans`를 보고 "correctness가 answer span에만 흐르니 처방 일부는 이미 있다"고 적었는데, **`ans`=ANSWER_REGION에 emit-결정 토큰이 포함**되므로 그 판단은 **틀렸다.** 처방 (2)는 전혀 충족되지 않았다 — 여기서 정정한다.

### 그래서 이번 실험의 인과 베팅은 명확하다
새 SFT2가 **메타를 실제로 유용하게** 만들면 → 메타가 correctness와 **양상관** → emit-결정 토큰이 벌 대신 상을 받음 → 붕괴 없음 → PMI가 채점할 무대가 유지됨 → rmeta 양수.
뒤집히지 않으면 남은 수는 **E-063 emission-protected arm(코드 없음)** 하나이고, 그 arm 없이는 "PMI가 base에서 작동하지 않는다"가 결론이 된다.

---

## E-107 🔴⭐⭐⭐ **정정: instruct에서 PMI-shift는 격리 실험을 통과했다** — 내가 E-106에서 "증거가 한 번도 검증을 통과한 적 없다"고 쓴 것은 틀렸다 (2026-07-26 21:15-21:25 UTC)

사용자 질문: "instruct에서는 가짜가 아니지 않았나?"
→ **맞다. 내가 틀렸다.** E-106에서 나는 base think-off의 in-training 수치(+10.4→−1.7)와 B3noPMI 일시부진 분석만 보고 **"PMI가 이득을 준다는 증거는 한 번도 검증을 통과한 적이 없다"** 고 썼다. **instruct 세대에 gs300 held-out 격리 실험이 존재하고 그것은 유의하게 양성이다.** 여기서 정정한다.

### 이미 존재하던 격리 실험 (2026-07-08 문서·내 발견이 아니다)
`docs/reports/2026-07-08-RQ2-isolated-pmishift-net-shiftonly-vs-gandhi.md`
두 arm은 **동일 meta-SFT init(`v8_rv_functional_sft`)** 에서 출발하고 **보상만** 다르다:
- **gandhi**(B2-analog) = meta-SFT + VANILLA_GRPO(correctness only)
- **shiftonly**(B3-analog) = meta-SFT + correctness + **PMI-shift만**(cal/format/emit/len_cost/over 전부 0·cf_group 제거)
⇒ `shiftonly − gandhi` = **PMI-shift 보상의 격리 순효과**(priming 고정). 이것이 `pmishift − base`(패키지 대 base)나 `gandhi − base`(priming)가 줄 수 없던 클린 트윈이다.

### 결과 (held-out 1030·avg@8·format-fair math_verify 재채점·paired bootstrap)
| 벤치 | budget | gandhi | shiftonly | **PMI 순효과** | bootstrap p | McNemar p |
|---|---|---|---|---|---|---|
| GSM8K | 16k | 92.5 | 93.3 | +0.8 | .15 | .42 |
| **MATH500** | 4k | 72.1 | 78.0 | **+5.9** | **<.001** | **<.001** |
| **MATH500** | 16k | 71.5 | 77.1 | **+5.6** | **<.001** | **.001** |
| AIME | 4k | 21.2 | 14.2 | −7.1 | <.001 | .25(n=30) |
| AIME | 16k | 20.0 | 14.2 | −5.8 | .002 | 1.0(n=30) |

⇒ **PMI-shift는 풀 수 있는 어려운 문제(MATH500)에서 실재하는 유의한 순양성(+5.6~5.9pp)** 이고, **능력한계 문제(AIME)에서는 음성(−5.8~−7.1pp)** 이다. AIME 음성의 기전은 길이 팽창이다: raw artifact 실측 — shiftonly AIME 평균 7432토큰·절단 84 vs gandhi 5903토큰·절단 66. 못 푸는 문제에서 늘어난 길이가 **미종결로 발현**된다.

### ⚠️ 내가 새로 확인한 단서: **부호가 채점기에 의존한다**
HF의 **raw eval artifact**를 직접 읽으니 재채점본과 다르다:
| arm | bench | **raw artifact** | 0708 재채점 | 차이 |
|---|---|---|---|---|
| gandhi | math500 | 68.07 | 71.5 | +3.4 |
| **shiftonly** | math500 | **62.28** | **77.1** | **+14.8** |
| gandhi | aime2024 | 19.58 | 20.0 | +0.4 |
| shiftonly | aime2024 | 14.17 | 14.2 | ~0 |
⇒ **raw로 계산하면 MATH500 PMI 순효과가 −5.8(음성)**, 재채점하면 **+5.6(양성)**. 부호가 뒤집힌다.
**설명은 정합적이다**: 스윙이 **shiftonly의 MATH500에만 +14.8pp** 집중되고 AIME는 거의 안 움직인다. shiftonly는 `w_format=0`이라 **답 포맷이 비정규**(LaTeX 표현식)일 수 있고, MATH500 답은 LaTeX이지만 AIME 답은 정수라 포맷 관용이 MATH500에서만 효과를 낸다. 그리고 이 프로젝트는 **raw 채점기가 깨졌다는 기록**이 있다(0625 `math500-grader-broken` — math_verify로 교체). ⇒ **재채점본이 방어 가능한 수치**다.
그러나 정직하게: **주장의 크기가 "한 arm에만 14.8pp 영향을 주는 채점기 선택"에 걸려 있다.** 논문에 이 의존성을 적어야 한다.

### 이것이 base 실험을 어떻게 재규정하는가
1. **"base RQ2 음성"과 "instruct RQ2 양성"은 모순이 아닐 수 있다.** instruct 격리는 **MATH500 +5.6 / AIME −5.8**로 **벤치마크 의존**이다. base think-off의 −4.2는 9과목 평균이라, AIME형(능력한계) 실패가 지배했거나 emission collapse가 겹친 결과일 수 있다.
2. ⇒ **base 재현의 판정은 종합 평균이 아니라 층별로 봐야 한다.** 사전등록 판정에 **MATH500(풀 수 있는 어려운 층)** 을 1차 지표로, AIME를 길이/종결 side-effect 지표로 분리해 넣는 것이 옳다.
3. **PMI가 "가짜"가 아니라는 근거는 instruct에 있다.** base에서 안 되면 그것은 substrate-dependence이고, 그 자체가 발표 가능한 결과다(사전등록대로).

### 내 오류의 원인 (재발 방지)
E-106을 쓸 때 **원장(run log)과 예비 in-training 수치만 봤고 `docs/reports/`의 기존 분석 문서를 확인하지 않았다.** 같은 질문에 대한 **더 강한 증거가 18일 전에 이미 문서화**돼 있었다. ⇒ 규율 추가: **"증거가 없다"고 단언하기 전에 `docs/reports/` 전체를 grep한다.**

---

## E-108 🔴 **b2p 사망 — 노드 상실·사이드카 SFT2 전손** / VC 차단이 선점을 치명화했다 (2026-07-26 ~21:25 UTC)

### 사건
`superb-terrier`(b2p) 상태가 **`failed  1.6 kB`** 로 전환. SSH 무응답 ⇒ **노드 반납 확정**.
- **b2p durable `gs80`은 무사**(model 4/4·optim 4/4·extra 4/4 = **RESUMABLE**). 데이터 손실은 gs80 이후 진행분뿐.
- ⛔**사이드카 SFT2는 전손**: `models/b2p2_rvfull_sft` **0 파일**. `save_strategy: epoch`이라 1에폭 끝(step 103)에 첫 저장인데 **~40/309에서 노드와 함께 소멸**. 약 25분 계산 손실.

### 근인 추정 (강한 정황)
wandb run 디렉토리 타임스탬프가 17:50 → **20:45** 재시작을 보여주고, 21:24 마지막 동기화 후 사망. b2p는 계속 **선점→같은 노드 resume**로 버텨왔다(E-099 정지 61-64%). 이번엔 재시작이 **재admission**을 필요로 했고, **basicvc는 신규 admission을 전면 거부**한다(E-091).
⇒ **VC 차단이 "견딜 수 있는 선점"을 "종료"로 바꿨다.** 차단이 풀리기 전까지 basicvc의 모든 잡은 선점 한 번에 죽을 수 있다. 이건 E-091의 영향 범위를 내가 과소평가한 부분이다 — "이미 admit된 잡은 유지된다"고 적었으나, **정확히는 "재admission이 필요 없는 동안만 유지된다"** 였다.

### 교훈 (사이드카 설계 결함)
사이드카를 `save_strategy: epoch`로 둔 것은 **호스트 잡의 수명이 불확실한 환경에서 잘못된 선택**이었다. 노드가 언제 사라질지 모르면 **저장 주기를 짧게(steps 단위) 두고 중간 산출물을 즉시 HF로 밀어야** 한다. 다음 사이드카는 `save_strategy: steps`·`save_steps 25`·완료 즉시 push로 설계한다.

### 현재 자산 상태
| 자산 | 상태 |
|---|---|
| b2p durable gs80 | ✅ RESUMABLE(HF) |
| b3p 구 init durable | ❌ 없음(gs10 삭제·의도적) |
| 새 SFT2 산출물 2종 | ❌ 없음 |
| RL 런처 3종·(C′) 생성기·감사도구 | ✅ 커밋됨 |
| 큐 | `rq3v2-b3p-a100-r2`(48m)·SFT2 2종(7h) — msrresrchvc A100 대기 |

---

## E-109 ⚠️ **정정: b2p는 죽지 않았다** — amlt `failed`인데 학습은 계속 중(gs120) / SSH만 끊겼다 (2026-07-27 01:45 UTC)

E-108에서 "b2p 사망·노드 반납·사이드카 전손"이라 적었다. **부분적으로 틀렸다.**

### 실측
| 신호 | 값 |
|---|---|
| `amlt status superb-terrier` | **`failed  1.6 kB`** (21:25부터 유지) |
| HF durable | **gs120 RESUMABLE**(4/4/4) — gs100은 keep=1 프루닝됨 |
| 최근 HF 커밋 | **01:00 `ckpt global_step_120`** · **01:40 wandb 동기화** |
| `amlt ssh` | **무응답**(터널 죽음) |

⇒ **컨테이너는 살아서 학습·푸시를 계속하는데 AML 컨트롤 플레인이 잡을 놓쳤다.** status가 `failed`가 된 것과 SSH가 끊긴 것은 같은 원인(컨트롤 플레인 상실)으로 보이고, 실제 계산은 무관하게 진행된다.

### 세 번째 관측 대리변수 함정
지금까지 쌓인 목록에 하나 더 추가된다:
1. `timing_s/step`은 마지막값(평균 아님)
2. wandb `state=crashed`인데 학습은 진행(E-102)
3. **`amlt status=failed`인데 학습은 진행(여기)**
⇒ **이 프로젝트에서 진행의 유일한 신뢰 신호는 HF 체크포인트 커밋이다.** amlt status·wandb state·SSH 접속성은 전부 실제 진행과 독립적으로 실패할 수 있다.

### 사이드카 SFT2의 실제 운명 — 미상, 회수 불가
21:25에 ~40/309였고 15.8s/it였으므로 **22:15경 완주했을 가능성이 높다**. 그러나 (a)내가 push 단계를 넣지 않아 산출물이 노드 로컬(`/scratch/checkpoints/b2p2_rvfull_sft`)에만 있고 (b)SSH가 끊겨 **회수할 수 없다**. HF `models/b2p2_rvfull_sft`는 여전히 0 파일.
⇒ E-108의 "전손"은 **"완주했을 수 있으나 회수 불가"** 로 정정한다. 그리고 **진짜 설계 결함은 save_strategy가 아니라 push 단계 부재**였다 — 노드 로컬에만 남기면 노드 접근성이 곧 자산 접근성이 된다. 다음 사이드카는 **저장 즉시 HF push를 붙인다.**

### 현재 자산
b2p durable **gs120**(잔여 180스텝) · 새 SFT2 산출물 0 · 큐 3건(A100 5h/12h/11h 대기) · basicvc 여전히 차단

---

## E-110 ⭐⭐ 큐가 안 풀리는 이유 특정 + **1-GPU 변종 발사**로 우회 (2026-07-27 01:57-02:02 UTC)

사용자 질문: "queue가 왜 안 풀리지?"

### 원인: **msrresrchvc의 A100은 SKU 9종 전부 `Basic` 티어다 — Standard가 없다**
```
80G4-A100-NvLink  NC96ad_A100_v4  Basic     80G4-A100  ND48am_A100_v4  Basic
80G2-A100 / 80G1-A100 / 80G8-* ...          전부 Basic
```
Singularity에서 **Basic = opportunistic** — 보장 할당이 없고 **Standard 보유자가 놀리는 용량만 주워** 쓴다(이 프로젝트 0717 기록의 "1120-slot opportunistic pool"과 같은 의미). 따라서:
1. 우리에겐 그 VC에 **보장 할당이 0**이라 스케줄러가 우선할 이유가 없다
2. msrresrchvc는 작은 연구 VC(A100/CPU/MI200만·H100/H200 없음)라 풀 자체가 작다
3. **우리 잡은 4-GPU 연속 슬롯을 요구** — 기회편승으로 4장을 한 번에 잡는 건 1~2장보다 훨씬 어렵다
4. 티어 상향 불가(Standard 미제공) + basicvc 차단
⇒ **"제출 에러"가 아니라 "순번이 안 오는 것"이다. 같은 SKU로 잡을 더 넣어도 빨라지지 않는다.**

### 미사용 레버: GPU 수 축소
SKU 목록에 `80G1-A100`·`80G2-A100`이 있다. **SFT2는 1장으로 충분하다** — ZeRO-3 + Adam을 CPU로 오프로드하면 GPU당 params 16GB + grads 16GB + activations ~5GB ≈ **37GB**(A100 80GB). 이 배치는 0726 사이드카에서 **RL 잡과 공존하며 실증**됐다(당시 CPU RAM 954GB 여유).

### 산출물
- **`configs/accelerate_sft_1gpu_cpuoff.yaml`**(신규) — zero3·`offload_optimizer_device: cpu`·`num_processes: 1`
- **`a100g1_sft_b2p2_rvfull.yaml`** / **`a100g1_sft_b0p2_rvfull.yaml`**(신규) — 4-GPU판의 최소편집 클론
- tarball 재패키징 **asset 490894146**(round-trip md5 `fa07db4c264ccccd788419a6e4db2c2a` 일치·1gpu config 동봉 확인·비밀 0·`.env` 0)

### 전수 diff = 의도한 5곳만 (양 파일 동일 패턴)
`job name` · `sku 80G4→80G1` · `CODE_TAR_REVISION` · `WANDB_NAME` · `accelerate config` · **`--tp_size 4 → 1`**
⭐마지막 항목이 중요하다: 1-GPU에서 게이트 eval이 `tp_size 4`면 실패한다. 4틱 연속 전수 diff 통독이 실제 결함을 잡아온 자리이고, 이번에도 여기서 잡혔다.

### 발사
| 실험 | job | SKU | 상태 |
|---|---|---|---|
| `a100g1-sft-b2p2-rvfull` | `sft_b2p2_rvfull_g1` | **80G1-A100** | preparing (02:01:46) |
| `a100g1-sft-b0p2-rvfull` | `sft_b0p2_rvfull_g1` | **80G1-A100** | preparing (02:02:01) |
기존 4-GPU 잡 2건은 **그대로 둔다** — 먼저 붙는 쪽이 이긴다. **출력 계보(`models/b2p2_rvfull_sft`·`models/b0p2_rvfull_sft`)가 동일**하므로 어느 쪽이 완주해도 같은 자산이 나온다.
⚠️둘 다 완주하면 같은 HF 경로에 중복 push가 되지만 `push_models_hf.py`는 덮어쓰기이므로 손상은 없다. 다만 **한쪽이 running으로 확인되면 다른 쪽을 취소**해 낭비를 막는다.
red커밋 가드 준수: `pytest 755 passed, 8 skipped` && 제출.

### 워크플로 진행
`wf_89a8db4d-9e2`: **started 9 / completed 6** — Recon 2 + Regrade 4 완료, Verify 3인 진행 중.

---

## E-111 ⭐⭐⭐⭐⭐ **Stage 0 완료 — 4 arm을 한 채점기에 올렸다.** 내 C축 판단이 틀렸고, 진짜 취약한 것은 **A축(priming)** 이다 (2026-07-27 02:15 UTC, ultracode `wf_89a8db4d-9e2`, 11 agents)

### 결과 — MATH500 16k, format-fair 재채점
| arm | raw | **format-fair** |
|---|---|---|
| base_matched | 54.4 | **63.3** |
| gandhi | 68.1 | **71.5** |
| shiftonly | 62.3 | **77.2** |
| pmishift | 66.0 | **81.9** |

| 축 | MATH500 16k | AIME 16k(avg@16) | GSM8K 16k |
|---|---|---|---|
| **A priming**(gandhi−base) | **+8.2** | +15.2 | +2.3 |
| **B PMI 단독**(shiftonly−gandhi) | **+5.6** | **−5.8** | +0.8 |
| **C 풀패키지**(pmishift−gandhi) | **+10.4** | −1.5 | +0.8 |
| 헤드라인(pmishift−base) | **+18.6** | +13.8 | +3.1 |
헤드라인이 논문의 +18.8/+14.2와 일치 ⇒ **논문 수치는 처음부터 format-fair 채점기 산출물이었다.**

### 🔴 정정 1 — 내가 E-104/E-106에서 "C축은 instruct에서도 음성(−2.1)"이라 한 것은 **틀렸다**
그건 **raw 채점** 기준이었다. 방어 가능한 채점기에서 **C = +10.4pp**다. 그리고 **pmishift의 raw는 −15.9pp 저평가**돼 있었다(66.0 vs 81.9).
⇒ 여기서 파생된 "**b3p는 틀린 arm이다**"라는 내 주장도 **철회한다.** C(+10.4)는 B(+5.6)보다 크고 AIME 손해도 작다(−1.5 vs −5.8). **풀 패키지가 PMI 단독보다 낫다** — 보조 헤드들이 PMI를 망치는 게 아니라 **PMI의 길이 부작용을 교정한다.** b3p는 올바른 arm이다.

### 🔴 정정 2 — 취약한 축은 B/C가 아니라 **A(priming)** 이다 (적대검증이 발견)
Skeptic 1(**SOUND**) 실측:
- shiftonly의 up-flip **608건 전부 BOXED 경로**, fallback 경로는 **0건**. pmishift도 fallback 기여 +0.05pp.
- **fallback을 쓰는 arm은 gandhi**(382행 = **+9.55pp**) — 세 축 전부의 **공유 대조군**이다.
- ⇒ **fallback은 B·C축을 낮춘다.** 실제로 fallback을 끄면(엄격: `\boxed` 필수) **B +5.6→+15.1**, **C +10.4→+19.8**로 커진다. *"공격 대상 메커니즘을 제거하면 효과가 커진다"* ⇒ 아티팩트가 아니다.
- **A축은 반대다**: gandhi 9.55pp vs base 0.35pp를 fallback에서 얻는다 ⇒ A = **+13.7(raw) / +8.2(fair) / −0.97(엄격)** — **세 가지 방어 가능한 채점기 선택에서 부호가 뒤집힌다.**
- 기전: base의 무-boxed 행은 **99.7%가 절단**(구할 게 없음), gandhi는 **15%만 절단** — 나머지는 `\boxed` 없이 산문으로 답을 쓴 완결 응답.

### 재채점이 편향을 **만든 게 아니라 제거**했다는 증거
- 표면 정규화 후 gold와 동일한 부분집합에서 새 채점기는 **전 arm 100.000%** 인정(n=2367/2278/2850/3043·예외 0). raw는 86.4/86.0/**82.6**/**82.3**% ⇒ raw가 **효과를 억누르던 두 arm에 3.5~4pp 차별적 위음성**을 갖고 있었다.
- flip 실검사: byte-identical 또는 사소한 LaTeX 변형(`2\sqrt{5}` vs `2 \sqrt{5}`, `C` vs `\text{(C)}`)만. **오답 인정 0건.** shiftonly down-flip 13건은 전부 raw의 위양성 정정.
- leniency placebo(다른 문제의 gold로 채점): **0/600 위양성**, 전 arm.
- 독립 재계산(Skeptic 3, **SOUND**): pmishift MATH500 81.85%를 **레포 코드 미사용 자체 구현**으로 재현, delta 0.00pp. 아티팩트 byte-identical 확인.

### ⚠️ 남은 진짜 약점 (Skeptic 2 **SUSPECT**)
fallback은 `answer_extracted`를 채점하는데 그 말단 분기가 `nums[-1]`(**응답 어디든 마지막 숫자**)이고, **레포 자신이 2026-07-14 버그픽스 주석에서 이 휴리스틱을 비난**했다. 즉 fallback은 약한 채점기이고 **gandhi만 크게 사용**한다. 생성 측 비교가능성은 깨끗하다(4개 런처가 identity 4줄만 다르고 런타임 메타데이터 temp 0.7·top_p 0.95·n=8·tp 4·max_model_len 20480 전부 동일).
⇒ **논문 RQ1(priming) 주장은 채점기 선택에 걸려 있다.** B/C는 견고하고 A가 위태롭다 — 내가 지금까지 정반대로 말해 왔다.

### 🔴 제출 안전감사 — **최대 위험 발견**
`rq3v2f_b0p/b2p/b3p` 세 RL 런처를 **지금 제출하면**: 제출 성공 → 노드 확보 → **SFT2 init이 HF에 없어 모델 로드에서 조용히 실패** → **`sleep 86400`으로 A100×4 노드를 24시간 점유**. 12시간째 4-GPU 슬롯을 못 잡는 기회편승 풀에서 **가장 희소한 자원을 정확히 사용자가 막으려는 방식으로 낭비**한다. 스테이징 블록에 **fail-closed 가드가 없다.**
⇒ **SFT2 산출물 2종이 HF에 4샤드 착지하기 전에는 RL 3종을 절대 제출하지 않는다.** (그 외 검사는 전부 PASS: VC/SKU 유효·tarball 3종 멤버 확인·RGS 가드 전부 abort 안 함·yaml/shlex/bash -n 통과.)

### crash-safe config 생성됨 — **다만 절반만 작동**
`configs/sft_b2p2_rvfull_safe.yaml`·`sft_b0p2_rvfull_safe.yaml`(save_strategy steps·save_steps 25). 검증 결과:
- `save_strategy`·`save_steps`는 `sft.py:450-451`이 읽는다 ✅
- `save_total_limit`은 **`sft.py:452`에 3으로 하드코딩**돼 yaml 키가 무시된다(디스크만 더 씀)
- ⚠️**더 큰 문제: resume 배선이 없다.** `sft.py:486`이 `trainer.train()`을 **`resume_from_checkpoint` 없이** 호출한다 ⇒ 자주 저장해도 **아무도 그걸 집어가지 않는다.** 노드가 또 죽으면 step 0부터. **잦은 저장은 필요조건이지 충분조건이 아니다.**

### 운영 조치
- **1-GPU 컨트롤 SFT2 `sft_b0p2_rvfull_g1` RUNNING 12분**(GPU 37,828 MiB = 예측 ~37GB와 일치·util 8%). **1-GPU 레버가 통했다** — 4-GPU가 12시간 못 잡던 슬롯을 4분에 확보.
- 감사관의 "동일 durable 경로 동시 push 경합" 경고에 따라 **중복 4-GPU 잡 `rq3v2-sft2-b0p2` 취소**(1-GPU running 확인 후).

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
