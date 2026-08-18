# Countdown 6팔 — 다른 서버로 이어받는 법 (2026-08-18)

이 서버가 사라져도 실험을 이어갈 수 있게 필요한 것을 한 곳에 적는다. **모든 산출물은
이미 원격에 있다** — 이 기계에만 있는 것은 없다.

## 무엇을 묻는 실험인가

두 질문이다. ① 메타 보상이 실제로 작동하나 ② 작동하게 할 방법은 무엇인가.

수학에서는 답이 안 나왔다. 메타 블록의 **내용**이 결과를 안 바꿨고(C-037), 이유는 채점이
0/1 한 비트뿐이라 "어느 메타가 도왔나"를 국소화할 수 없어서다. Countdown 은 그 둘을 푼다 —
목표수가 프롬프트에 있어 **gold 없이 자가검증**되고(p̂ 를 그룹에서 직접 센다), 증인식의
연산자 하나만 바꾼 **완전히 짝지어진 오답**이 있어 PMI 를 정의할 수 있다.

## 지금 어디까지 왔나

| | 상태 |
|---|---|
| 실험 | `cd6b0818` (basicvc · 80G1-H100 × 7) |
| 옛 판 | `cd6a0818` — 코드 tar 404 로 7개 전멸. 자산 ID 를 HF SHA 로 잘못 채웠다 |
| 코드 | GitHub `master` `2318439` · 릴리스 `countdown-6arm-0818` 자산 **519191232** |
| 데이터 | HF `iamseungpil/metacot-sdc-data` — `data/countdown_{train,val}.parquet` (8000/500) |
| 체크포인트 | HF `iamseungpil/metacot-h200-triobj-dcpo-v3` · lineage `cd6_<팔>` (이웃과 안 겹침) |

## 팔 여섯

```
A corr   R_corr 만                              기준
B cur    + clip(shift,±2) + reversal            현행 재현
C mul    + [B의 항] × sign(A_corr)              ★곱하기
E gate   + −(2p̂−1) × 1{메타 냈다}               ★게이팅
F full   C + E                                  수정판
G neg    + meta_len/100                         가짜 대조군
```

귀속: `C−B` = 곱하기 · `E−A` = 게이팅 · `F−C−E+A` = 상호작용.
**`G` 가 `B~F` 중 하나라도 이기면 전부 폐기** — `check_negative_control()` 이 판정한다.

정의는 `src/training/countdown_rewards.py` **한 곳뿐**이다. 런처에 보상 가중치가 한 줄도
없어서 두 잡의 diff 는 `name` 과 `ARM` 두 줄이다(G8).

## 이어받는 절차

```bash
git clone https://github.com/iamseungpil/metacognition-math.git && cd metacognition-math
set -a; source .env; set +a          # ⚠토큰은 .env 에만. 셸 GH_TOKEN 은 401 이었다
python -m pytest tests/ src/training/tests/ -q      # 953 passed 여야 한다

# 코드를 고쳤으면 릴리스 자산을 다시 올리고 그 **숫자 id** 를 채운다
#   ⛔CODE_TAR_REVISION 은 GitHub 릴리스 자산 id 다. HF 커밋 SHA 가 아니다(0818 사고).
#   런처와 똑같은 curl 로 반드시 실물 확인:
curl -fsSL -H "Authorization: token $GH_TOKEN" -H "Accept: application/octet-stream" \
     -o /tmp/t.tar.gz https://api.github.com/repos/iamseungpil/metacognition-math/releases/assets/<ID>
tar -xzOf /tmp/t.tar.gz src/training/verl_sdc.py | grep -c 'COUNTDOWN\]\[WIRED\]'   # 1 이어야 한다

amlt run countdown_rl_6arm.yaml <새이름> -y
```

데이터를 다시 만들려면 `countdown_task.build_parquet(8000, 11, ..., variant="new", n_nums=5)`
(val 은 `500, seed 999`). 증인식은 구성으로 풀이가 보장되고, 오답은 연산자 교체다.

## ★먼저 볼 것 — 손실도 정확도도 아니다

```
[COUNTDOWN][WIRED] arm=X step=N n=.. mean=.. distinct=..
```

이 줄이 **이번 판의 전부**다. 두 판이 연속으로 여기서 죽었다. 8팔 판은 `countdown_arm` 을
읽는 코드가 0건이라 전 팔이 `SDC_SHARED` 로 돌 뻔했고, 6팔 첫 판은 스태시를 **읽는 쪽만**
착지해 전 팔이 상수 0 보상으로 돌 뻔했다. 둘 다 로그는 그럴듯했다.

지금은 잡마다 자기 로그에서 이 줄을 240×10초 폴링하고 없으면 스스로 창을 죽인다.
그 다음 확인은 **팔별 `mean` 이 서로 다른가** — 같으면 배선이 또 무효다.

감시기: `watch_cd6.py`(스크래치패드). 상태·배선·보상분리·텔레메트리 15종·중단조건을 읽는다.
지표는 **wandb 로 안 간다** — stdout 에만 찍히므로 로그를 판다. 다음 판에 wandb 로깅을
붙이는 게 맞다.

## 함정 넷 (전부 이 실험에서 실측)

**`amlt status` 의 "running" 은 일하고 있다는 뜻이 아니다.** 0818 에 6개가 "running" 이었고
전부 가드의 `sleep 300` 안에 있었다. 상태 칸이 아니라 로그가 증거다.

**`CODE_TAR_REVISION` 은 GitHub 자산 id(숫자)다.** 변수명이 REVISION 이라 HF SHA 를 넣었고
7개를 잃었다. 자리표 이름은 `..._ASSET_ID` 였다.

**`amlt cancel` 은 접두사 일치다.** 실험명을 `cd6b0818` 처럼 확장 불가능하게 짓는다 —
`cd6` 로 지으면 나중 `cd6b` 를 같이 죽인다.

**`ARM_SPECS` 에 D·H 정의가 남아 있지만 돌지 않는다.** `LAUNCHED_ARMS` 가 실제 발사 목록이고
`SPEC_VERSION` 은 `countdown-6arm-0818` 이다. 정의가 있다는 이유로 "돌았다"고 읽지 마라.

## 중단 조건

```
발화율 < 0.20     메타를 안 내면 잴 것이 없다
정형문 > 0.50     최빈 문장이 과반이면 메타가 아니라 상투구다
답누출 > 0.10     메타가 답을 담으면 메타 보상이 정답 보상의 사본이 된다
G 가 B~F 중 하나를 이김   → 전부 폐기
```

앞의 셋은 10스텝마다 `check_abort()` 가 로그에 찍는다.

## 판정 전 확인

`p̂` 분포를 먼저 본다. 수학에서 `p̂` 가 0/1 로 몰린 문제가 **64.8%** 였고 그 구간에서
correctness 기울기가 0 이었다. Countdown 에서도 몰리면 게이팅(E·F)은 침묵한다 —
그때 `E−A` 의 널은 "게이팅이 안 통한다"가 아니라 "잴 수 없었다"이다.

판정문은 codex-sol 게이트를 지난 뒤 `docs/CLAIMS.md` 에 쌓는다. 원장은
`docs/reports/2026-07-17-rq3-run-and-iteration-log.md`.
