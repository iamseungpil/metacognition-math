# RQ3 Matched-Ladder — Submission & Operations Runbook

How to submit and operate the RQ3 4-arm jobs on Singularity/amlt **without the
token / setup failures we hit on 0714**. Read this before every submit.

---

## 0. THE GOLDEN RULE (this is what bit us)

**Always `source .env` in the SAME shell, immediately before `amlt run`:**

```bash
cd /home/v-seungplee/metacognition-math      # amlt must run from the project dir
export PATH=/home/v-seungplee/miniconda3/envs/amlt/bin:$PATH
set -a; source .env; set +a                  # <-- REQUIRED, do not skip
amlt run h100std_rq3_b0.yaml -y
```

**Why.** The launcher injects tokens by *shell substitution at submit time*
(`env: GH_TOKEN: ${GH_TOKEN}`). amlt reads `${GH_TOKEN}` from **your submit
shell**, not from `.env`. The base login shell carries a **stale/rotated
GH_TOKEN** that differs from `.env`. If you submit without sourcing `.env`, the
node bakes in the stale token and the very first setup step fails:

```
curl: (22) The requested URL returned error: 401
tar (child): /tmp/metacognition.tar.gz: Cannot open: No such file or directory
bash: /scratch/metacognition/scripts/bootstrap_sdc_node.sh: No such file
... /scratch/conda_envs/simplerl/bin/python: No such file ... rc=127  -> job "failed"
```

i.e. the GitHub code-tarball download 401s → `/scratch/metacognition` never
exists → bootstrap can't run → no conda env → verl exits 127. **All four arms
fail identically and fast (~6 min).** The fix is one line: source `.env` first.

Token status (0714): base-shell `GH_TOKEN` ≠ `.env` `GH_TOKEN` (stale); `HF_TOKEN`
matches. Only GH was the problem, but sourcing `.env` fixes both regardless.

---

## 1. Submit all four arms

```bash
cd /home/v-seungplee/metacognition-math
export PATH=/home/v-seungplee/miniconda3/envs/amlt/bin:$PATH
set -a; source .env; set +a
for y in h100std_rq3_b0.yaml h100std_rq3_b2.yaml h100std_rq3_b3.yaml h100std_rq3_b3nopmi.yaml; do
  amlt run "$y" -y
done
```

| Launcher | Arm | config-name | init model | key knob |
|---|---|---|---|---|
| h100std_rq3_b0.yaml | B0 baseline | base_matched_grpo_h100_4x4k | b0_gold_sft | vanilla GRPO |
| h100std_rq3_b2.yaml | B2 meta-SFT | base_matched_grpo_h100_4x4k | b23_rv_unmasked_sft | vanilla GRPO |
| h100std_rq3_b3.yaml | B3pkg | triobj_dcpo_v4_stage3b_h100_4x4k | b23_rv_unmasked_sft | full pkg, w_meta=0.8 |
| h100std_rq3_b3nopmi.yaml | B3-noPMI | triobj_dcpo_v4_stage3b_h100_4x4k | b23_rv_unmasked_sft | full pkg, w_meta=0.0 |

Shared v2 collapse-fixed recipe (all arms): `temperature=1.0 top_k=-1 top_p=1.0
max_response_length=8192 norm_adv_by_std_in_grpo=false`. `save_freq` is
per-arm, NOT shared: **b0/b2 = 10, b3pkg/b3nopmi = 5** (the b3 pair keeps a
shorter cold-start foothold on the preemptible Standard tier).

Each `amlt run` prints a random experiment name (e.g. `enough-wombat`); note the
four names — you monitor by experiment name.

---

## 2. Monitor (run from the project dir, not elsewhere)

```bash
cd /home/v-seungplee/metacognition-math
export PATH=/home/v-seungplee/miniconda3/envs/amlt/bin:$PATH
for e in <exp_b0> <exp_b2> <exp_b3pkg> <exp_b3nopmi>; do
  amlt status "$e" 2>/dev/null | grep ':rq3_'
done
```

Status meanings:
- `queued` / `preparing` — waiting for / acquiring a node (Standard tier can wait). Normal.
- `running` — has a node. Verify setup succeeded (§3).
- `pass` — **usually preemption** (StopUserNode → graceful SIGTERM → exit 0), not success. Resubmit if it has no checkpoint yet; otherwise it auto-resumes.
- `failed` — a real error exit. Inspect the log (§3).

Read a job's log (needs the experiment name AND `:job`):
```bash
amlt log view <exp> :rq3_b0            # print to terminal
amlt log tail <exp> :rq3_b0            # stream
```

Confirm setup actually worked — the healthy early markers:
```bash
amlt log view <exp> :rq3_b0 | grep -E "extracting env|bootstrap.*complete|training/global_step"
```
Red flags: `error: 401`, `No such file or directory`, `rc=127`.

---

## 3. When something dies

| Symptom in log | Cause | Fix |
|---|---|---|
| `curl ... 401` then missing `/scratch/metacognition` | stale GH_TOKEN (didn't source .env) | cancel + resubmit **with `.env` sourced** (§0) |
| `pass` with no output, ~cold-start duration | Standard-tier preemption in cold-start | resubmit (source .env); repeats are normal until a clean window |
| `hf_hub_download failed` in bootstrap | HF Xet / token | check HF_TOKEN; bootstrap now uses the HF python client (not curl) |
| `ModuleNotFoundError: flash_attn` | conda-pack lacks flash-attn | on-demand build in bootstrap; if persists, rebuild env snapshot |
| `Error in memory profiling ... free memory` | vLLM init GPU-memory race (non-deterministic) | just resubmit / relaunch; 2–3 tries usually pass |

Cancel (needs `-y`, no TTY prompt):
```bash
amlt cancel <exp> -y
```

**Cold-start note.** A FRESH (gs0) job must survive bootstrap + reach its first
checkpoint (b0/b2: `save_freq=10` → gs10, ~70 min; b3pkg/b3nopmi:
`save_freq=5` → gs5, ~35 min) without preemption to become
durable. On Standard tier this can take several resubmits. Once the first HF
checkpoint lands, `resume_mode=auto` + `pull_resume_ckpt.py` make every later
resume clean (optimizer state is now saved too). If one arm can't get a clean
window after ~3 tries, temporarily lower that arm's `trainer.save_freq` to 3–5
to grab the first checkpoint sooner, then it's durable.

---

## 4. HF / wandb operations (local)

Use the amlt-env python (the base python3 has no `huggingface_hub`/`wandb`):
```bash
PY=/home/v-seungplee/miniconda3/envs/amlt/bin/python
set -a; source .env; set +a
```
- HF repo (checkpoints): `iamseungpil/metacot-h200-triobj-dcpo-v3`
  (rq3 ckpts live under `checkpoints/rq3_{b0,b2,b3nopmi,b3pkg}/`; leave
  `models/ eval/ reports/ wandb/` alone).
- wandb: project `gistdslab/metacot-dcpo-v4`, run ids
  `rq3-b0-2 / rq3-b2-2 / rq3-b3pkg-2 / rq3-b3nopmi-2` (**-2 접미사** — 0714
  fresh 재시작 세대; fixed id + `WANDB_RESUME=allow`). Delete the old run
  before a fresh gs0 start so history doesn't overlay.
- Code delivery: launcher curls a **GitHub release asset** (id in
  `CODE_TAR_REVISION`) → `/scratch/metacognition`. To ship new code: build a new
  tarball (root dir `metacognition/`), upload as a release asset, update
  `CODE_TAR_REVISION` in all four launchers. The env conda-pack comes from HF
  (`iamseungpil/metacot` `env_snapshots/simplerl_v4.tar.gz`) via the HF client.

---

## 5. Security TODO (do not defer indefinitely)

The launchers run under `set -x`, so `--token $${HF_TOKEN}` prints the **HF token
in plaintext** in job logs. The token is therefore leaked. **Rotate HF_TOKEN**
(and GH_TOKEN, already known-leaked) and update `.env`. Longer-term: pass tokens
via env-only (`os.environ`) in the scripts instead of CLI args, and/or `set +x`
around token-bearing lines.

---

## 6. Fresh-start 절차 (gs0부터 새로 시작할 때)

순서가 중요하다 — **HF 삭제가 먼저다**:

1. **HF `checkpoints/rq3_<arm>/` 를 먼저 삭제**한다. 안 지우면 RGS
   (`resume_mode=auto` + `pull_resume_ckpt.py`)가 옛 lineage의 마지막 ckpt를
   찾아 **자동으로 그 위에 resume**해 버린다 — fresh가 아니게 된다.
2. wandb 옛 run 삭제 (fixed run id + `WANDB_RESUME=allow`라 히스토리가 겹쳐
   쌓인다).
3. 그 다음 제출. RGS가 아무것도 못 찾고 gs0로 시작하는 것이 올바른 동작이다.

기록: 2026-07-14 4-arm 감사(8개 수정) 후 **HF `checkpoints/rq3_*` 전삭제 +
전 arm fresh 재시작**을 실제로 수행했다. 그래서 현행 wandb run id가 -2
접미사다. 이전 단일-시드 런은 not-certifiable 판정
(`docs/redesign/EXPERIMENT_LOG.md` §11).

---

## 7. 진단 추가 규칙

- **선점 vs hang**: `amlt log list <exp>`에 `retry_NNN`이 있고 상태가
  running이면 **선점 사이클 중** — 그대로 둔다. 성급히 cancel하면 잡아둔
  노드가 글로벌 풀로 새어나가 다시 몇 시간 대기다. 진짜 hang 시그니처 =
  stdout 라인 수가 재다운로드 간 불변 + retry 증가 없음 + 다음 HF ckpt 부재
  — 이 세 가지가 모두 맞을 때만 개입.
- **배치 제출만**: `amlt run <yaml> -y`. interactive `-i`는 Standard
  opportunistic 풀에서 노드를 잡지 못한다(배치는 queue-and-grab).
- **완료된 arm 재제출 금지**: gs300 완료 arm(현재 B0)을 재제출하면
  resume→즉시 finish→`sleep 86400`이 노드를 하루 점거한다.

---

## 8. 유지보수 이력

- **2026-07-16**: MSR GCR 전체 정지 — B200(Bonete) 재할당 + NVLINK 펌웨어.
  16:00 UTC 전 잡 강제취소 + 오프라인, 복구 ETA **2026-07-17 00:00 UTC** +
  신규 할당(랩 GPU delegate 확인 대기). 2026-07-15 22:41 UTC부터의
  `amlt run` 에러 **"(UserError) The virtual cluster does not exist ... in
  the same cloud"는 이 유지보수/재할당 컨트롤플레인 문제**이지 로컬 설정
  문제가 아니다(읽기 계열 `amlt target info`는 정상). A100
  Palisades/MSRRESRCHVC 유지보수 메일(7/20–24)은 다른 VC·다른 GPU 건으로
  무관. **GPU/VC 갈아타기 금지** — b0/b2가 msrresrchbasicvc H100 Standard에서
  돌았으므로 b3만 옮기면 매치드 래더가 깨진다.

---

## 9. gs300 후 평가

최종 판정은 held-out 1030(GSM8K 500 + MATH-500 500 + AIME 30)이며 merge →
vLLM eval 레시피는 `experiments/configs/science/eval_1030.yaml` 참조.
**주의**: 그 yaml의 models 목록은 pre-rq3 세대 경로라서 rq3 판정에는
`checkpoints/rq3_b*/global_step_300` 병합 경로로 갈아끼워야 한다. arm 비교는
매치드 gs(또는 gs300)에서만, 채점은 math_verify.
