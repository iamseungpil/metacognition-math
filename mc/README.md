# mc/ — 약속 첫 답 재시작(PFX) 메타인지 자기 증류

활성 팔은 **`SPONT_PFX` 하나**다(0925 정리: SPONT_SCORE·SPONT_V5·SPONT_PMI2·PMI2/U/Dir/R4 경로 삭제 — 기록은
`docs/RESULTS_cd9.md`·`docs/PREREGISTRATION_cd9_math_judgment.md`). 코드+테스트 5,000줄 이하 유지.

## 한 스텝
1. 학습 행 = 문제 + 모델 자신의 풀이를 **약속한 첫 답 박스까지** 자른 앞부분(`pool.pfx_commit_records`,
   약속 판정 = 강제 탐침 믿음 `shift_check.commit_weight` ∧ 글자 규칙 거부). 표집 무게 = `PFX_WEIGHT_KEY`(`trainer.pfx_sampler`).
2. `trainer.PrefixAgentLoop` 가 채팅 템플릿 + 앞부분 뒤에서 이어쓰기(1턴, 주입 없음).
3. `train_hook.pfx_rewards`: r = 앞부분+이어쓰기 마지막 박스 정오(A0 = 앞부분 **마지막** 박스), GRPO(같은 앞부분 K 개).
4. 선택 손잡이(함께 쓸 수 있다 — 곱 재배분 뒤 가산 크레딧, `trainer.add_span_credit`):
   - `PFX_FORK=ch|hsd` — CH-Fork(수정 28/28b): 섞인 묶음의 행마다 교차 적합 이웃(성공 s⁺·실패 s⁻, 박스 답 가림,
     중립 문구)을 붙인 선생님 문맥에서 동결 ref 토큰별 log p → CH_t = log p(·|s⁺) − log p(·|s⁻)(hsd: s⁺ 대 문맥 없음)
     → 누설 전 구간에만 `credit.fork_weights`(결과 방향, 평균 1) → `trainer.add_span_credit` 가 결과 adv 에 곱함(행 총량 보존).
   - `PFX_FORK=chdir`(수정 45) — ch 대조 그대로, 가중을 `credit.dir_weights` 로: 행 결과 쪽(성공 = 성공 쪽, 실패 = 대조를 뒤집은 실패 쪽)으로 기운 만큼만 몫(반대쪽·|CH|<τ .05 = 0), 말당 평균 4 배 상한, 균등 30% 섞음. 외톨이 행(같은 쪽 형제 없음)은 데이터 `extra_info.succ_bank`·`fail_bank`(base 거르기 이어쓰기)에서 그쪽 이웃을 빌린다(모든 대조 모드, 계기 `fork_bank`).
   - `PFX_TRUNC=mask` — 수정 32/32c: 상한에서 잘린 이어쓰기는 학습 밖(고유 uid · adv 0 · 손실 마스크 0).
   - `PFX_BREAK_W=w`(수정 43) — 맞은 첫 답을 뒤집어 실패한 행의 결과 adv × w(수정 18 «맞은 첫 답: 망침만 벌»; 어느 말에 몰릴지는 CH).
   - `PFX_REP=c`(수정 43, `pfx_rep`) — 같은 답(동치) 박스 3번째(`loop_char`) 뒤 토큰에만 −c 가산, 끝낸 행·잘린 행 모두
     (mask 로 빠진 잘린 행도 그 구간은 손실에 남김 `pfx_keep_from`) · 총량 ≤ REP_CAP 몫. `PFX_REP_HARD=1`(수정 46): 그 구간
     결과 adv ≤ 0(`rep_clamp`, 정답으로 끝난 반복도 꼬리는 칭찬 없음) · 총량 상한 없이 말당 −c · 계보 `_rp<c>h`.
   - `PFX_ADV_CAP=c`(수정 48) — CH·BREAK_W 곱 뒤 말 단위 adv 를 ±c 로 자름(계보 `_ac<c>`, chdir 와 함께 쓰지 않음).
   - `PFX_DISTILL=β`(수정 51, `pfx_distill`) — 틀린 첫 답 행의 이어쓰기 말 [16, 마지막 박스·반복 시작)에 가산 크레딧
     말당 β·clip(log π_ref(y|문제 + `context.FACT_TMPL`(X)) − log π_old(y|문제 + 앞부분), ±2)(수정 52) — 자기 풀이를 못 본
     얼린 자기 자신으로부터의 문맥 증류(정박 해소). 계보 `_ds<β>`, 계기 ds_rows·ds_neg_frac·ds_d_mean·ds_share.
   - 삭제(실패): R2 `PFX_CHECK_*`(27d) · `PFX_FORK=anchor`(28g) · `PFX_TRUNC=zero`(28f)·`loop`(42) · `PFX_STOP`(34, H4c)·`PFX_TAIL0`(37) — 지운 손잡이는 즉사.
5. 중단: `pfx_guard`(맞은 첫 답 파괴율 최근 3스텝 > max(.05, 2×기준선)) → rc 75.

## 파일
| 파일 | 역할 |
|---|---|
| `run.sh` | 발사기 `bash mc/run.sh SPONT_PFX SEED STEPS` — 계보 = 체크포인트 자리(변형마다 접미사), resume·병합·12k 평가 |
| `trainer.py` | verl 진입점, 에이전트 루프, 표집기, advantage 훅, 트리 채점(`tree_score`, per_token) |
| `train_hook.py` | 보상·CH-Fork·뒤집기 벌·반복 벌·로그·중단 |
| `ref_worker.py` | 동결 ref 워커 `mc_tree_score` |
| `credit.py` | `fork_weights` |
| `shift_check.py` | 박스별 믿음 채점 `--boxes [--prior]`, `score_trees`, `leak_point` |
| `pool.py` | PFX 학습·탐침 데이터 생성 |
| `probe.py` | 약속 앞부분 탐침(되짚기·고침·망침·메타 말) |
| `eval.py`·`rollout.py` | 12k 단일 패스 평가·vLLM·FSDP 병합 |
| `grade.py`·`context.py` | 채점(단일 진실 원천)·프롬프트 |

테스트: `CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/mc`(GPU 사용 금지 — `tests/mc/conftest.py`).
