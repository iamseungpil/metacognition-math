# HABIT vs PMI (0919)

스크립트: `docs/analysis/habit_vs_pmi_0919.py`

- 전체 행 n=6400, revised=True n=184
- revised 중 텍스트/zone 매칭 실패(제외) n=0
- 분석 대상(revised & zone 있음) n=184

## 앵커 쌍: gold

| category | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| recheck | 116 | 32.511 [27.975,36.858] | 6.438 | 26.073 | 0.724 | 2.072 |
| switch | 47 | 25.974 [19.596,32.171] | 5.010 | 20.964 | 0.548 | 1.924 |
| doubt | 104 | 29.397 [24.830,34.152] | 5.383 | 24.013 | 0.716 | 2.033 |
| error_naming | 49 | 28.764 [21.754,35.887] | 4.287 | 24.477 | 0.775 | 1.879 |
| reread | 26 | 17.397 [8.270,26.641] | 1.511 | 15.886 | 0.650 | 1.349 |
| constraint | 91 | 32.843 [28.170,37.484] | 6.631 | 26.213 | 0.682 | 2.212 |
| no_marker | 33 | 11.834 [3.982,20.598] | 3.803 | 8.032 | 0.474 | 0.490 |

### 배타적 분할 (switch_only/recheck_only/both/neither)

| split | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| switch_only | 6 | 16.765 [0.256,31.413] | 8.288 | 8.477 | 0.600 | 1.333 |
| recheck_only | 75 | 35.349 [29.676,41.132] | 7.481 | 27.867 | 0.824 | 2.105 |
| both | 41 | 27.321 [20.888,33.613] | 4.530 | 22.791 | 0.541 | 2.011 |
| neither | 49 | 17.701 [10.906,24.756] | 4.097 | 13.604 | 0.625 | 0.878 |

## 앵커 쌍: self

| category | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| recheck | 108 | 42.441 [38.169,46.730] | 6.475 | 35.966 | 0.728 | 2.445 |
| switch | 42 | 35.330 [28.807,41.502] | 5.787 | 29.544 | 0.550 | 2.127 |
| doubt | 97 | 40.624 [36.203,44.991] | 5.951 | 34.673 | 0.720 | 2.379 |
| error_naming | 42 | 43.230 [36.601,49.973] | 4.890 | 38.340 | 0.789 | 2.597 |
| reread | 23 | 33.093 [22.895,42.728] | 2.585 | 30.508 | 0.650 | 1.976 |
| constraint | 84 | 40.815 [36.396,45.084] | 7.362 | 33.452 | 0.687 | 2.513 |
| no_marker | 26 | 20.771 [11.905,29.667] | 1.237 | 19.534 | 0.474 | 1.476 |

### 배타적 분할 (switch_only/recheck_only/both/neither)

| split | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| switch_only | 7 | 19.066 [5.869,31.503] | 6.814 | 12.252 | 0.600 | 1.336 |
| recheck_only | 73 | 44.291 [38.938,49.425] | 6.904 | 37.387 | 0.824 | 2.522 |
| both | 35 | 38.583 [31.730,45.054] | 5.581 | 33.002 | 0.543 | 2.286 |
| neither | 40 | 27.813 [19.657,35.303] | 2.389 | 25.424 | 0.625 | 1.700 |

## 앵커 쌍: goldx

| category | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| recheck | 107 | 38.813 [34.982,42.836] | 7.145 | 31.668 | 0.726 | 2.516 |
| switch | 44 | 30.454 [24.382,36.447] | 5.821 | 24.633 | 0.548 | 2.090 |
| doubt | 96 | 36.761 [32.550,41.094] | 6.461 | 30.300 | 0.719 | 2.467 |
| error_naming | 40 | 43.152 [36.973,49.053] | 6.407 | 36.745 | 0.775 | 2.761 |
| reread | 22 | 26.873 [17.791,36.116] | 2.598 | 24.274 | 0.650 | 1.995 |
| constraint | 86 | 37.655 [33.396,41.804] | 7.906 | 29.749 | 0.686 | 2.516 |
| no_marker | 24 | 20.066 [11.048,29.747] | 2.939 | 17.127 | 0.474 | 1.583 |

### 배타적 분할 (switch_only/recheck_only/both/neither)

| split | n | mean SHIFT_real [95%CI] | mean SHIFT_placebo | mean(real-placebo) | frac w->r | mean reward |
|---|---|---|---|---|---|---|
| switch_only | 7 | 18.453 [3.499,32.645] | 6.628 | 11.825 | 0.600 | 1.143 |
| recheck_only | 70 | 42.031 [37.201,46.635] | 7.926 | 34.105 | 0.826 | 2.646 |
| both | 37 | 32.725 [26.318,39.092] | 5.668 | 27.056 | 0.541 | 2.269 |
| neither | 37 | 28.422 [20.391,36.853] | 4.484 | 23.938 | 0.625 | 1.849 |

## Zone 길이(단어수) per category, 그리고 SHIFT_real과의 상관

| category | n | mean n_words | median n_words |
|---|---|---|---|
| recheck | 119 | 1239.5 | 949.0 |
| switch | 49 | 1628.4 | 1679.0 |
| doubt | 106 | 1331.9 | 1096.0 |
| error_naming | 49 | 1533.5 | 1344.0 |
| reread | 28 | 1467.3 | 836.5 |
| constraint | 92 | 1438.0 | 1344.0 |
| no_marker | 41 | 71.5 | 15.0 |

### Spearman(zone n_words, SHIFT_real) — 앵커별, 전체 revised 행

| pair | rho | n |
|---|---|---|
| gold | 0.312 | 171 |
| self | 0.308 | 155 |
| goldx | 0.241 | 151 |


## 해석 (≤8줄)

1. 세 앵커(gold/self/goldx) 전부 `recheck_only`(재확인만, switch 없음, n=70~75)가 SHIFT_real·reward 최고: gold reward=2.105, self=2.522, goldx=2.646 — `revision_pmi` 크레딧이 가장 후하게 주는 습관은 "재확인/검산" 계열이다.
2. `switch_only`(대안 경로만, switch 있고 recheck 없음, n=6~7)가 모든 앵커에서 가장 낮다(gold reward=1.333, self=1.336, goldx=1.143) — 접근을 바꾸는 것만으로는 신념이 잘 안 움직이고, 크레딧도 적게 준다.
3. `no_marker`(어떤 키워드도 없음, n=24~33)가 최저 그룹 중 하나(gold reward=0.490)로, 마커 존재 자체가 SHIFT와 강하게 결부돼 있다.
4. w→r(오답→정답) 비율로 보면 `recheck_only`가 0.82로 가장 높고 `switch_only`는 0.60 — 즉 recheck 습관은 outcome(오답 교정) 자체도 더 잘 낸다. 그런데 reward 격차(2.6 vs 1.1~1.3)는 outcome 격차(0.82 vs 0.60)보다 훨씬 크므로, 크레딧은 recheck을 "결과 개선분 이상으로" 우대한다.
5. `error_naming`(mistake/wrong 등 명시)도 상위권(goldx reward=2.761)이라, 실수를 명시적으로 지목하는 습관 역시 크게 보상받는다.
6. zone 길이(단어수)는 `no_marker`가 압도적으로 짧고(median 15) 나머지 카테고리는 800~1700 단어대로 비슷하다 — 마커 있음/없음의 차이는 크지만, 마커가 있는 카테고리들 사이의 순위(recheck_only 최고, switch_only 최저)는 길이만으로 설명되지 않는다.
7. SHIFT_real과 zone 길이의 Spearman rho는 0.24~0.31(약한 양의 상관)로, "길게 쓰면 무조건 많이 이동한다"는 아니다 — 길이는 부분적 교란요인이지만 지배적이지 않다.
8. 결론: `revision_pmi` 크레딧은 사실상 "재확인/검산 + 실수 지목" 습관을 "대안 경로 탐색"보다 훨씬 강하게 강화하며, 이 우대는 outcome 개선 폭을 넘어선다.
