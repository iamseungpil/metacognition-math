# probes/ — 일회성 진단과 그 증거

이 디렉터리가 있는 이유. 여기 있던 것들이 **스크래치패드에만** 있었고, 그 사이 대장과
사전등록은 그 결과를 인용하고 있었다. 주장은 커밋돼 있는데 증거는 임시 디렉터리에 있어서,
이 기계가 사라지면 *"그 수가 어디서 나왔나"* 에 답할 수 없는 상태였다.

⛔**여기서 학습하지 않는다.** 정본 학습 경로는 `src/training/` 이다. 여기 있는 것은
전부 읽기·측정·재채점이고, 결과는 CLAIMS 로 올라가기 전의 **탐색(EXP)** 이다.

## 배치

```
countdown/   Countdown 과제 계열 — 생성기·채점기·오라클 사다리·발화율
math_meta/   수학 메타 프로브 계열 — 접합·PMI 신호·격자·경보 탐지기
results/     작은 판정 요약(저장소) + MANIFEST.json(큰 결과의 HF 위치)
```

큰 결과 JSON 26개(36MB)는 HF `iamseungpil/metacot-sdc-data` 의 `probe_results/` 에 있다.
`results/MANIFEST.json` 이 파일명·크기·sha256 앞 16자를 들고 있으므로, 받은 파일이 인용된
그 파일인지 대조할 수 있다.

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("iamseungpil/metacot-sdc-data", "probe_results/cd_oracle.json",
                    repo_type="dataset")
```

## 어느 수가 어느 파일에서 나왔나

주장을 인용할 때 이 표를 근거로 댄다. 여기 없는 수는 출처가 없는 수다.

| 수 | 파일 | 스크립트 |
|---|---|---|
| **Countdown 오라클 사다리** — `acc(N)=0` 인 255문제에서 N 0.0% · N′ 7.1% · R 7.8% · OF 10.2% · O1 14.9% · O3 99.6%. `O1−N′` **+7.84pp** 유의 | `cd_oracle.json` | `countdown/cd_oracle.py` |
| **게이팅** `O1` +1.85pp [+0.96,+2.85] 유의 · 위약 +0.00. 무조건부는 +0.83pp 널(여집합 −2.82pp 가 상쇄) | `cd_oracle.json` | `countdown/cd_oracle.py` |
| **p̂ 분포** p̂=0 63.7% · p̂=1 1.0% ⇒ correctness 기울기가 **64.8%** 에서 0 | `cd_oracle.json` | `countdown/cd_oracle.py` |
| **Countdown 5팔** — N .1104 / N′ .1065 / B .0960 / SH .1015 / R .1073. 주 지표 `R−N′` +1.97pp [−0.99,+4.93] **널** | `cd_main.json` | `countdown/cd_main.py` |
| ★**발화율 58.8%** (P2_format, 형식 준수) — `SFT 없음` 결정의 근거. P0 0.0% / P1 20.8%(형식 0%) / P3 48.8% | `emit_test.json` | `countdown/emit_test.py` |
| **난이도 보정** — 수 5개 확정 | `cd_pilot3.json` | `countdown/cd_pilot.py` |
| ★**C-037 의 근거** — 수학 접합 v2, 500문제·6팔·cap 2048. N .8430 > E .8363 > S .8347 > R .8313 ≈ B′ .8310 > B .8213. `acc(R)−acc(B)｜L4-5` **+0.70pp [−1.40,+2.93]**, 잡음바닥 δ_eq **2.23pp** ⇒ 해상도 아래 | `splice2_4b.json` | `math_meta/splice_probe2.py` |
| **PMI 신호 검정** — 참/헛 경보 AUC 0.457, `meta_len` 0.598 이 이김 | `signal_probe_4b.json` | `math_meta/signal_probe.py` |
| **3축 격자 + 영향력** — 티처포싱 포화(6/6 틀려도 +6.38, AUC 0.539) | `grid_probe_4b.json` | `math_meta/grid_probe.py` |
| **경보 탐지기 14공식 판별력** | `alarm_probe_4b*.json` | `math_meta/alarm_probe*.py` |

## 이 프로브들이 산 교훈 (재현하려는 사람용)

**층화를 처치군·대조군 성적으로 정의하지 마라.** 이 계열에서 `acc_N`(대조군 성적)으로 층을
잘라 회귀 효과를 처치 효과로 읽은 적이 있다 — 같은 층에서 **어절 섞기가 +4.55pp**, 정형문이
+3.64pp 였다. 외생 층(MATH level)으로 바꾸자 천장 표가 전부 철회됐다.

**절단이 결론을 만든다.** cap 700 이 정확도 8pp 를 깎았고, L5 −5.56pp 와 헛경보 분해가
cap 2048 에서 소멸했다. 길이 상한을 결론과 함께 적어라.

**오답(decoy)은 충실해야 한다.** 규칙기반 오답은 티처포싱에서 포화된다(AUC 0.539).
Countdown 의 연산자 교체 오답이 이 문제를 푼 이유가 그것이다 — 단 `swap_op_decoy` 초판은
나누어떨어짐·양수를 안 봐서 37.5% 가 무효였다(`countdown/countdown.py` 주석 참조).
