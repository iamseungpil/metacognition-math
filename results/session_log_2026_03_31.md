# Meta-CoT 세션 기록: 2026-03-31

**작성일**: 2026-03-31  
**세션 목적**: 4개 GRPO 실험(E3/E5/E7/E8) 병렬 실행 + V2 SFT 평가 완료

---

## 1. 세션 시작 배경

### AMLT CLI 사용 불가

AMLT CLI가 Python 3.13 환경에서 다음 오류로 동작 불가:

```
Field.__init__() got an unexpected keyword argument 'missing'
```

이를 우회해 AML REST API + WebSocket SSH proxy로 노드에 직접 접속하는 방식을 사용.

### 완전한 SSH 접속 명령

```bash
AZ_PYTHON="/opt/az/bin/python3"
CONNECTOR="/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py"
PROXY_URL="wss://ssh-<HASH>.westus2.nodes.azureml.ms"   # 노드마다 다름

ssh -o ConnectTimeout=20 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o "ProxyCommand=$AZ_PYTHON $CONNECTOR $PROXY_URL" \
    -i ~/.ssh/id_rsa \
    azureuser@placeholder \
    "COMMAND_HERE"
```

> **주의**: `azureuser@placeholder`에서 hostname은 실제로 무시되고 ProxyCommand가 전부 처리함. `placeholder`는 리터럴 그대로 써야 함.

### 현재 세션 노드 프록시 URL

| 노드 | WSS 프록시 URL | 용도 |
|------|--------------|------|
| metacognition-e8 | `wss://ssh-2etszrmvdrq4cwqdql4al50f32aqiwdcl036benvkg6kmzk8bpc.westus2.nodes.azureml.ms` | E3 cont + after_e3 파이프라인 |
| tops-caiman | `wss://ssh-2etszrmvdrq4cwqdql4al50f32ckgsyoyi2puoyq678vdlx42vc.westus2.nodes.azureml.ms` | E7 / E8 학습 |
| eval-e8 | `wss://ssh-2etszrmvdrq4cwqdql4al50f365fggn0cs41y3ld90c6m331nlc.westus2.nodes.azureml.ms` | V2 SFT eval 완료, 이후 Base SFT eval 예정 |

> **주의**: 이 URL은 AMLT job이 살아있는 동안만 유효함. Job이 종료되거나 노드가 재할당되면 URL이 바뀜.

### 프록시 URL을 모를 때 — AML REST API로 찾는 방법

```bash
# 1. Azure 토큰 발급
TOKEN=$(az account get-access-token \
  --scope https://management.azure.com/.default \
  --query accessToken -o tsv)

# 2. Subscription ID 확인
az account show --query id -o tsv
# → c4c534bc-9978-4974-9c87-551f7c5754ef

# 3. Workspace 목록 확인
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://management.azure.com/subscriptions/c4c534bc-9978-4974-9c87-551f7c5754ef/providers/Microsoft.MachineLearningServices/workspaces?api-version=2023-04-01-preview" \
  | python3 -c "
import json,sys
for w in json.load(sys.stdin).get('value',[]):
    rg = w['id'].split('/resourceGroups/')[1].split('/')[0]
    print(w['name'], rg)
"
# → msra-sh-aml-ws  msra-sh-aml-rg  (westus2)

# 4. 실행 중인 job 목록
SUB="c4c534bc-9978-4974-9c87-551f7c5754ef"
RG="msra-sh-aml-rg"
WS="msra-sh-aml-ws"

curl -s -H "Authorization: Bearer $TOKEN" \
  "https://management.azure.com/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.MachineLearningServices/workspaces/$WS/jobs?api-version=2023-04-01-preview&\$top=20" \
  | python3 -c "
import json,sys
for item in json.load(sys.stdin).get('value',[]):
    status = item['properties'].get('status','')
    name = item['name']
    print(status, name)
"

# 5. 특정 job의 SSH 서비스 endpoint (= WSS 프록시 URL) 조회
JOB_NAME="<job_name>"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://management.azure.com/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.MachineLearningServices/workspaces/$WS/jobs/$JOB_NAME?api-version=2023-04-01-preview" \
  | python3 -c "
import json,sys
p = json.load(sys.stdin)['properties']
for k,v in p.get('services',{}).items():
    ep = v.get('endpoint','')
    if 'ssh' in ep or 'wss' in ep.lower():
        print(k, ep)
    # SSH proxy URL은 jupyterlab endpoint의 hash 부분과 동일
    # wss://ssh-<HASH>.westus2.nodes.azureml.ms 형태로 재구성
"
```

> **팁**: `jupyterlab` endpoint URL의 hash(`jptrl-<nodeIndex>-<HASH>.westus2...`)에서 `<HASH>` 부분을 추출해서  
> `wss://ssh-<HASH>.westus2.nodes.azureml.ms` 로 만들면 SSH 프록시 URL이 됨.

### 직전 세션에서 프록시 URL을 찾은 방법

AMLT 소스코드에서 WebSocket proxy 패턴을 역공학으로 발견:

```
/home/v-seungplee/.azure/cliextensions/ml/azext_mlv2/manual/custom/_ssh_connector.py
```

이 파일이 `wss://<proxy_endpoint>/nbip/v1.0/ws-tcp` 로 연결하는 ProxyCommand 역할을 함.  
AML의 `az ml compute connect-ssh` 내부 구현이 이 스크립트를 쓰는 것을 발견해 직접 활용.

---

## 2. 이번 세션에서 수행한 작업

### 2.1 after_e3 파이프라인 복구

**문제**: after_e3 파이프라인 프로세스가 죽어있었음 (마지막 로그 11:49 UTC, 이후 기록 없음)

**원인**: 이전 세션에서 실행된 `nohup` 프로세스가 종료됨

**조치**: 재시작 (PID 128799 → 이후 135209로 재시작)

```
Tue Mar 31 15:53:14 2026: after_e3 pipeline RESTARTED
Tue Mar 31 16:43:24 2026: after_e3 pipeline RESTARTED (v2) - E5 will start from qwen3_metacot_v2_sft
```

### 2.2 E5 시작점 수정 (핵심 변경)

**기존**: E5가 `checkpoints/grpo_v2_E3_500/final`에서 시작

**변경**: E5가 `checkpoints/qwen3_metacot_v2_sft`에서 시작

**이유**:
- E3 cont 학습 중 `completions/mean_length = 2048`, `clipped_ratio = 1.0` 확인
- E3 모델이 2048 토큰을 꽉 채우는 과도한 verbosity를 학습함
- E5가 이 습관을 그대로 물려받으면 실험 결과가 오염됨
- E5와 E7 모두 동일한 V2 SFT 기준점에서 시작해야 비교가 공정함

**결과**:
```
E5 모델 체인: qwen3_metacot_v2_sft → GRPO E5 (stepwise_trajectory, 500 steps)
E7 모델 체인: qwen3_metacot_v2_sft → GRPO E7 (stepwise_probe, 500 steps)
```

### 2.3 V3 SFT HuggingFace 업로드 완료

- **경로**: `/scratch/metacognition/checkpoints/qwen3_metacot_v3_sft/`
- **HF 위치**: `iamseungpil/metacot` → `models/qwen3_metacot_v3_sft`
- **특이사항**: private LFS quota 초과로 레포 public 전환 후 업로드 성공
- **학습 정보**: 585/585 steps, loss 0.27 → 0.09, 6,566 difficulty-adaptive meta chains

### 2.4 E7 학습 시작 (tops-caiman)

V3 SFT 완료 후 tops-caiman이 유휴 상태임을 확인하고 E7 즉시 시작:

```bash
accelerate launch configs/accelerate_grpo.yaml src/training/grpo_v2.py \
  --mode E7 --max_steps 500 \
  --model_path checkpoints/qwen3_metacot_v2_sft \
  --output_dir checkpoints/grpo_v2_E7
```

- **시작 시각**: 2026-03-31 ~16:43 UTC
- **마지막 확인**: 140/500 steps (28%), GPU 60%, 75GB VRAM 사용 중

### 2.5 V2 SFT 1030-problem eval 완료

eval-e8 노드에서 1030문제 평가 완료. 결과:

| 벤치마크 | 정답/전체 | 정확도 |
|---------|---------|--------|
| GSM8K | 450/500 | **90.00%** |
| MATH-500 | 295/500 | **59.00%** |
| AIME 2024 | 4/30 | **13.33%** |
| **전체** | **749/1030** | **72.72%** |

> **참고**: 이전 실험 계획서(v4)의 V2 SFT 결과(51.1%)는 30문제/max_tokens=1024 기준이었음.  
> 이번 결과는 500문제/max_tokens=2048 기준으로 **더 신뢰할 수 있는 baseline**.

---

## 3. 현재 실험 상태 (2026-03-31 18:36 UTC 기준)

| 실험 | 노드 | 진행 상황 | 예상 완료 |
|------|------|---------|---------|
| **E3 cont** | metacognition-e8 | 276/300 (92%) | ~30분 내 |
| **after_e3 파이프라인** | metacognition-e8 | E3 완료 대기 | E3 완료 후 자동 시작 |
| **Base SFT** | metacognition-e8 | 대기 중 | E3 완료 후 ~2h |
| **E5** | metacognition-e8 | 대기 중 | Base SFT 완료 후 ~12h |
| **E7** | tops-caiman | 140/500 (28%) | ~8-10h |
| **E8** | - | 미시작 | E7 완료 후 시작 예정 |
| **V2 SFT eval** | eval-e8 | ✅ 완료 | — |
| **V3 SFT 학습** | tops-caiman | ✅ 완료 | — |
| **V3 SFT HF 업로드** | — | ✅ 완료 | — |

### after_e3 파이프라인 상세 (PID 135209)

```
[자동 순서]
1. checkpoints/grpo_v2_E3_500/final 생성 확인 (E3 cont 완료 감지)
2. Base SFT 학습: Qwen3-8B → checkpoints/qwen3_base_sft
   - 설정: configs/sft_base.yaml
   - 로그: /scratch/metacognition/base_sft.log
3. E5 GRPO 학습: qwen3_metacot_v2_sft → checkpoints/grpo_v2_E5
   - 모드: stepwise_trajectory
   - rewards: correctness(1.0), format(0.5), meta_quality(0.5), stepwise_trajectory(1.0)
   - 500 steps
   - 로그: /scratch/metacognition/grpo_e5.log
```

---

## 4. 실험 설계 요약

### 4개 GRPO 실험 비교

| 실험 | 시작 모델 | 모드 | 핵심 reward | 목적 |
|------|---------|------|-----------|------|
| **E3 cont** | V1 E3 (200step) | calibration | correctness + meta_quality + calibration | E3 500step 완성 |
| **E5** | V2 SFT | stepwise_trajectory | correctness(1.0) + stepwise_trajectory(1.0) | 단계별 정확성 |
| **E7** | V2 SFT | stepwise_probe | correctness(1.0) + stepwise_probe(1.5) | 내부 probe 활용 |
| **E8** | E7 final | correctness-dominant | correctness(3.0) + correct_meta(0.5) + length_penalty(1.0) | 정확도 우선 + verbosity 방지 |

### 핵심 비교 질문

```
Meta-CoT (E5 / E7 / E8) >= Base SFT?
```

현재 알고 있는 기준선:
- **V2 SFT (Meta-CoT, SFT only)**: 72.72% 전체 (GSM8K 90%, MATH 59%, AIME 13.3%)
- **Base SFT (이전 실험, 30문제 기준)**: 58.9% — 500문제 기준 재측정 필요

---

## 5. 다음 세션 계획

### 즉시 해야 할 것

**[A] E3 cont 완료 확인** (~30분 내)
```bash
# 확인 명령
ssh metacognition-e8: "tail -2 /scratch/metacognition/e3_cont.log"
# 완료 신호: 300/300 또는 after_e3.log에 "E3_500 final checkpoint found!" 출력
```

**[B] Base SFT 학습 모니터링**
- E3 완료 후 자동 시작됨
- 완료 기준: `after_e3.log`에 "Base SFT done" 출력
- 완료 후 `checkpoints/qwen3_base_sft` 존재 확인

**[C] Base SFT eval 시작** (Base SFT 완료 직후)
- eval-e8은 현재 유휴 상태이므로 즉시 사용 가능
```bash
# eval-e8에서 실행
CUDA_VISIBLE_DEVICES=0 python -u src/eval/eval_hf.py \
  --model_path checkpoints/qwen3_base_sft \
  --model_name 1030_base_sft \
  --benchmarks gsm8k math500 aime2024 \
  --max_problems 500 \
  --output_dir results > results/eval_base_sft.log 2>&1 &
```

**[D] E7 완료 후 E8 즉시 시작**
- E7 완료 기준: `grpo_e7.log`에 500/500 출력 + `checkpoints/grpo_v2_E7/final` 존재
```bash
# tops-caiman에서 실행
accelerate launch configs/accelerate_grpo.yaml src/training/grpo_v2.py \
  --mode E8 --max_steps 200 \
  --model_path checkpoints/grpo_v2_E7/final \
  --output_dir checkpoints/grpo_v2_E8
```

### 모든 학습 완료 후 전체 eval

6개 모델 모두 1030-problem eval 실행:

| 모델 | 체크포인트 | 상태 |
|------|---------|------|
| V2 SFT (Meta-CoT) | `checkpoints/qwen3_metacot_v2_sft` | ✅ eval 완료 (72.72%) |
| Base SFT | `checkpoints/qwen3_base_sft` | 학습 후 eval 예정 |
| E3_500 | `checkpoints/grpo_v2_E3_500` | 학습 후 eval 예정 |
| E5 | `checkpoints/grpo_v2_E5` | 학습 후 eval 예정 |
| E7 | `checkpoints/grpo_v2_E7` | 학습 후 eval 예정 |
| E8 | `checkpoints/grpo_v2_E8` | 학습 후 eval 예정 |

### 결과 분석 (autoresearch 판단)

6개 모델 eval 완료 후:

1. **M1 달성 여부**: 최고 Meta-CoT 모델 >= Base SFT?
2. **벤치마크별 분석**: GSM8K / MATH-500 / AIME 각각 어느 모델이 best?
3. **E3 verbosity 영향**: E3_500 vs E5/E7 비교로 verbosity 문제 정량화
4. **GRPO 효과**: V2 SFT → E5/E7/E8 delta 측정

**M1 미달성 시 다음 가설 (experiment_plan_v4.md 기준)**:
- H2: 난이도 적응형 meta (easy 문제에 meta 감소)
- H3: 검증 전용 meta (답 이후 meta-only)
- H4: GPT-5.4 full 모델로 SFT 데이터 재생성

---

## 6. 체크포인트 위치 정리

모든 체크포인트는 `/scratch/metacognition/checkpoints/` 하위에 있음.

| 체크포인트 | 노드 | 설명 |
|---------|------|------|
| `qwen3_metacot_v2_sft` | metacognition-e8, eval-e8 | V2 Meta SFT (난이도 적응형 meta) |
| `qwen3_metacot_v3_sft` | tops-caiman | V3 Meta SFT (difficulty-adaptive, 6566 chains) |
| `grpo_v2_E3_500` | metacognition-e8 | E3 500-step calibration GRPO (진행 중) |
| `qwen3_base_sft` | metacognition-e8 | Base SFT (학습 예정) |
| `grpo_v2_E5` | metacognition-e8 | E5 stepwise_trajectory (학습 예정) |
| `grpo_v2_E7` | tops-caiman | E7 stepwise_probe (학습 중) |
| `grpo_v2_E8` | tops-caiman | E8 correctness-dominant (예정) |

HuggingFace 백업: `iamseungpil/metacot` (dataset repo)
- `models/qwen3_metacot_v2_sft` ✅
- `models/qwen3_metacot_v3_sft` ✅

---

## 7. 주요 판단 근거 메모

### max_completion_length=2048 유지 결정
- 4096으로 늘리면 GPU 메모리 90%+ 초과 → OOM 위험
- 2048이 현재 가능한 최대값

### E5를 E3_500이 아닌 V2 SFT에서 시작하는 이유
- E3 cont: `clipped_ratio=1.0`, `mean_length=2048` → 모델이 verbose behavior 학습
- E5가 E3에서 시작하면 이 verbosity를 물려받음
- E5, E7 모두 V2 SFT에서 시작 → 공정한 비교 가능

### E8를 E7 final에서 시작하는 이유
- E8의 `length_penalty` reward가 E7의 verbosity를 교정하는 구조
- E7 → E8 체인이 의미 있는 실험 설계

### V3 SFT를 HF repo public으로 전환한 이유
- private repo의 LFS 무료 할당량(1GB) 초과
- 학습 데이터/코드가 민감하지 않으므로 public 전환
