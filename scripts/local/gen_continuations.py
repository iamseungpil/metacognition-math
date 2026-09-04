#!/usr/bin/env python
r"""CLI — 사이트(prefix) 각각에서 K개 이어쓰기를 세 모드로 뽑아 채점한다.

동기. `src/training/countdown_sites.py` 가 뽑은 "지점"(site) 각각에 대해, 그 지점에서
실제로 롤아웃을 이어 굴렸을 때 (a) 메타가 있는 채로(원 프롬프트 그대로) (b) 메타
요구가 아예 없는 채로(대조군) (c) **남의** 메타를 이식받은 채로 결과가 어떻게
갈리는지를 본다. 이것이 prefix-anchored GRPO 를 태우기 전에 먼저 확인해야 하는
전제("이 지점에서 K개 이어쓰기의 결과가 실제로 갈리는가" — 안 갈리면 advantage 가
0이라 GRPO 신호 자체가 없다)다.

세 모드
    meta    — site 의 `prompt` 컬럼 그대로(시스템 메시지가 메타를 요구한다).
    nometa  — 시스템 메시지만 `countdown_task.PROMPT_VARIANTS["plain"]` 로 교체한
              반사실(메타 요구 없음). 토큰 금지가 아니라 지시문 자체를 뺀다
              (토큰 금지는 우회당한다는 것이 이미 확인됐다 — 과제 지시 참조).
    donor   — `prompt` 그대로 + **다른 문제**에서 뽑은 기증 메타를 프리픽스 바로
              뒤에 이어붙인다(text = prefix + "\n" + donor_meta_raw + "\n").

렌더링. `apply_chat_template(msgs, tokenize=False, continue_final_message=True,
add_generation_prompt=False, enable_thinking=False)` — `scripts/local/build_sites.py`
가 실측 검증한 것과 같은 조합이다. **chat API 가 아니라 렌더링된 문자열**을
`llm.generate` 에 넘긴다(그래야 continue_final_message 렌더링이 모델이 실제로 보는
바이트와 정확히 같다).

채점. `countdown_task.grade(fed_prefix + continuation, nums, target)` — **전체
응답**을 채점한다(\\boxed 가 continuation 에만 있을 수 있어서다). fed_prefix 는 모드별로
모델에 "이미 쓴 것"으로 먹인 텍스트다(meta/nometa 는 site 의 prefix 그대로, donor 는
그 뒤에 기증 메타를 붙인 것 — `build_fed_prefix_text` 참조). `parse_meta` 는
continuation **만** 본다(모델이 새로 무엇을 썼는지가 관심사다). `sc_row` 는 전체
응답(fed_prefix + continuation)에 대해 돌려 novel/followed/checked 를 얻는다 — 이
함수가 내부에서 "첫 <meta> 앞"을 스스로 찾으므로, site 의 저장된 `pairs_pre` 를
다시 파싱해 넣지 않는다(같은 값을 두 곳에서 따로 셀 이유가 없다 — 복제 금지 규약).

메모리. 사이트를 256개씩 청크로 나눠 처리한다(모드 3종 × 사이트 256개 = 최대 768개
프롬프트를 한 번에 `llm.generate` 에 넘긴다).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training import countdown_rewards as cdr  # noqa: E402
from src.training import countdown_selfcontrol as csc  # noqa: E402
from src.training import countdown_task as ct  # noqa: E402

MODES = ("meta", "nometa", "donor")
CHUNK_SIZE = 256

# SC(자기제어) 항의 K_S·CONF_HI — `countdown_rewards.SC_K_STUCK`/`SC_CONF_HI` 와 같은
# 값을 이 스크립트 호출부(task spec)에서 그대로 지정한 것. 상수를 다시 선언하는 이유는
# 이 스크립트가 학습 팔 조립(`arm_reward`)과 무관한 진단 스크립트라 `countdown_rewards`
# 의 학습용 상수에 배선을 걸지 않기 위해서다 — 값만 같게 맞춘다(0904 task spec: "4, 0.8").
SC_K_STUCK = 4
SC_CONF_HI = 0.8


# ══════════════════════════════════════════════════════════════════════════════
# 1. 순수 함수 — 메시지/텍스트 조립 (CPU 테스트 대상)
# ══════════════════════════════════════════════════════════════════════════════

def build_fed_prefix_text(mode: str, prefix: str, donor_meta_raw: Optional[str]) -> str:
    """모델에게 "이미 이렇게 썼다"고 먹일 텍스트. meta/nometa 는 site 의 prefix 그대로,
    donor 는 그 뒤에 기증 메타를 붙인다: `prefix + "\\n" + donor_meta_raw + "\\n"`.
    """
    if mode not in MODES:
        raise ValueError(f"build_fed_prefix_text: unknown mode {mode!r}")
    if mode == "donor":
        if not donor_meta_raw:
            raise ValueError("build_fed_prefix_text: donor 모드인데 donor_meta_raw 가 비었다.")
        return f"{prefix}\n{donor_meta_raw}\n"
    return prefix


def build_messages_for_mode(prompt_msgs: Sequence[Mapping], mode: str, *,
                            prefix: str, donor_meta_raw: Optional[str] = None,
                            plain_system: str = "") -> list[dict]:
    """site 의 `prompt`(system/user/assistant-프리픽스 3행) → 모드별 chat 메시지.

    ★모드 간 차이는 시스템 메시지(nometa)와 마지막 assistant 메시지 내용(donor)뿐이다
    — 나머지(유저 메시지 등)는 원본 그대로 복사한다(원본 리스트·딕셔너리를 변형하지
    않는다 — 호출자가 같은 `prompt_msgs` 를 세 모드에 재사용한다).
    """
    msgs = [dict(m) for m in prompt_msgs]
    if not msgs or msgs[0].get("role") != "system":
        raise ValueError("build_messages_for_mode: 첫 메시지가 system 이 아니다.")
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("build_messages_for_mode: 마지막 메시지가 assistant(프리픽스) 가 아니다.")
    if mode == "nometa":
        msgs[0] = {**msgs[0], "content": plain_system}
    fed = build_fed_prefix_text(mode, prefix, donor_meta_raw)
    msgs[-1] = {**msgs[-1], "content": fed}
    return msgs


# ══════════════════════════════════════════════════════════════════════════════
# 2. 순수 함수 — 기증 메타 풀
# ══════════════════════════════════════════════════════════════════════════════

def parse_donor_pool(rows: Iterable[Mapping]) -> list[dict]:
    """롤아웃 jsonl 행({"text","nums","target",...}) → 기증 메타 풀.

    **완결된**(confidence·decision 둘 다 있는) 메타가 있는 행만 후보로 삼는다
    (`countdown_rewards.parse_meta` 재사용 — 복제하지 않는다).
    """
    pool: list[dict] = []
    for row in rows:
        text = row.get("text") or ""
        m = cdr.parse_meta(text, "new")
        if not m.get("emitted"):
            continue
        nums = tuple(int(v) for v in row["nums"])
        target = int(row["target"])
        pool.append({
            "raw": m["raw"], "decision": m.get("decision"),
            "confidence": m.get("confidence"), "problem": (nums, target),
        })
    return pool


def sample_donor(pool: Sequence[Mapping], exclude_problem: tuple,
                 rng: random.Random) -> tuple[Optional[dict], bool]:
    """`exclude_problem` 과 다른 문제에서 기증 메타 하나를 뽑는다.

    Returns (entry_또는_None, fallback_used) — fallback_used 는 "같은 문제를 뺀 풀이
    비어 전체 풀(같은 문제 포함)에서 뽑았다"는 뜻이고, 호출자가 이 사실을 요약에
    남겨야 한다(조용히 규칙을 어기지 않는다).

    ★"풀과 같은 decision 분포" — 사양이 정한 표집 방식이 아니라 이 스크립트의
    설계 결정(미확인, 문서화): 필터링한 풀에서 **균등** 표집한다. 별도 층화를 하지
    않는 이유는, 풀 자체의 decision 비율이 모집단이므로 여러 site 에 걸쳐 집계하면
    균등 표집이 기댓값에서 그 비율을 그대로 재현하기 때문이다.
    """
    if not pool:
        return None, False
    filtered = [e for e in pool if e["problem"] != exclude_problem]
    if filtered:
        return dict(rng.choice(filtered)), False
    return dict(rng.choice(pool)), True


# ══════════════════════════════════════════════════════════════════════════════
# 3. 순수 함수 — 생성 결과 한 개 → 출력 행
# ══════════════════════════════════════════════════════════════════════════════

def build_record(*, site_id: str, mode: str, policy_tag: str, k_index: int,
                 prefix: str, donor_meta_raw: Optional[str], nums, target,
                 continuation: str, n_tokens: int, truncated: bool) -> dict:
    """생성된 이어쓰기 하나를 채점·파싱해 출력 스키마 한 행으로 조립한다.

    ★채점은 fed_prefix(모드별로 모델에 먹인 텍스트) + continuation 의 **전체**에
    대해 한다 — donor 모드의 fed_prefix 에는 기증 메타가 이미 포함돼 있으므로
    `full_text` 는 "모델이 실제로 완성한 응답 전체"와 바이트가 같다.
    """
    fed_prefix = build_fed_prefix_text(mode, prefix, donor_meta_raw)
    full_text = fed_prefix + continuation
    r_corr = int(ct.grade(full_text, nums, target))
    m = cdr.parse_meta(continuation, "new")
    sc = csc.sc_row(full_text, nums, target, r_corr, SC_K_STUCK, SC_CONF_HI)
    return {
        "site_id": site_id,
        "mode": mode,
        "policy_tag": policy_tag,
        "k_index": int(k_index),
        "continuation": continuation,
        "full_text": full_text,
        "r_corr": r_corr,
        "emitted": int(m.get("emitted") or 0),
        "meta_raw": m.get("raw") or "",
        "decision": m.get("decision"),
        "confidence": m.get("confidence"),
        "meta_start_in_cont": m.get("start"),
        "novel": int(sc.get("novel") or 0),
        "followed": int(sc.get("followed") or 0),
        "checked": int(sc.get("checked") or 0),
        "n_tokens": int(n_tokens),
        "truncated": int(bool(truncated)),
        "donor_meta_raw": donor_meta_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 순수 함수 — summary.json 집계
# ══════════════════════════════════════════════════════════════════════════════

def _fam_key(fam) -> str:
    if fam is None:
        return "none"
    try:
        if isinstance(fam, float) and math.isnan(fam):
            return "none"
    except TypeError:
        pass
    return str(int(fam))


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(records: Sequence[Mapping], site_family_dead: Mapping[str, object]) -> dict:
    """모드별 성공률·발화율, site 별 K 결과 분산(=prefix-anchored GRPO 관문),
    family_dead 별 평균 성공률.
    """
    by_mode: dict[str, list] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    per_mode = {}
    for mode, rows in by_mode.items():
        by_site: dict[str, list] = defaultdict(list)
        for r in rows:
            by_site[r["site_id"]].append(r["r_corr"])
        n_var = sum(1 for outs in by_site.values() if len(set(outs)) > 1)
        variance_frac = (n_var / len(by_site)) if by_site else float("nan")

        by_fam: dict[str, list] = defaultdict(list)
        for r in rows:
            fam = site_family_dead.get(r["site_id"])
            by_fam[_fam_key(fam)].append(r["r_corr"])
        success_by_family = {k: _mean(v) for k, v in sorted(by_fam.items())}

        per_mode[mode] = {
            "n_rows": len(rows),
            "n_sites": len(by_site),
            "success_rate": _mean(r["r_corr"] for r in rows),
            "emit_rate": _mean(r["emitted"] for r in rows),
            "variance_frac": variance_frac,
            "success_by_family_dead": success_by_family,
        }
    return {"per_mode": per_mode, "n_records": len(records)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. I/O — 사이트/풀 로딩, 렌더링, vLLM 생성 (CPU 테스트 대상 아님)
# ══════════════════════════════════════════════════════════════════════════════

def _chunks(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _site_rng(site_id: str, seed: int) -> random.Random:
    return random.Random((hash((site_id, "donor")) ^ seed) & 0xFFFFFFFF)


def render(tokenizer, msgs: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True,
        add_generation_prompt=False, enable_thinking=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", required=True, help="sites_{train,judge}.parquet 경로")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--policy_tag", required=True)
    ap.add_argument("--modes", default="meta,nometa,donor")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor_pool", default=None,
                    help="기증 메타 소스 jsonl (기본: $WORK/eval/gs0_Qwen3-4B_new/s11/texts.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="모델을 로드하지 않고 렌더링된 프롬프트만 찍는다(GPU 없이 검증)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown mode {m!r} (allowed: {MODES})")

    import pandas as pd

    df = pd.read_parquet(args.sites)
    if args.limit:
        df = df.head(args.limit)
    print(f"[gen_continuations] {len(df)} sites x {args.k} continuations x "
          f"{len(modes)} modes ({','.join(modes)})", flush=True)

    donor_pool: list[dict] = []
    if "donor" in modes:
        donor_path = Path(args.donor_pool or
                          (Path(os.environ["WORK"]) / "eval/gs0_Qwen3-4B_new/s11/texts.jsonl"))
        donor_pool = parse_donor_pool(_load_jsonl(donor_path))
        print(f"[gen_continuations] donor pool: {len(donor_pool)} completed metas from {donor_path}",
              flush=True)

    plain_system = ct.PROMPT_VARIANTS["plain"]

    # ── 사이트별 (모드 → 렌더 문자열, 기증 메타) 미리 조립 ────────────────────────
    def build_site_plan(row) -> dict:
        site_id = row["site_id"]
        nums = [int(v) for v in row["nums"]]
        target = int(row["target"])
        prefix = row["prefix"]
        prompt_msgs = list(row["prompt"])
        plan = {"site_id": site_id, "nums": nums, "target": target, "prefix": prefix,
               "family_dead": row["family_dead"], "per_mode": {}}
        donor_entry = None
        donor_fallback = False
        if "donor" in modes:
            rng = _site_rng(site_id, args.seed)
            donor_entry, donor_fallback = sample_donor(
                donor_pool, (tuple(nums), target), rng)
        for mode in modes:
            donor_raw = donor_entry["raw"] if (mode == "donor" and donor_entry) else None
            if mode == "donor" and donor_entry is None:
                continue  # 기증 풀이 아예 비었다 — 이 사이트는 donor 모드에서 스킵
            msgs = build_messages_for_mode(
                prompt_msgs, mode, prefix=prefix, donor_meta_raw=donor_raw,
                plain_system=plain_system)
            plan["per_mode"][mode] = {"messages": msgs, "donor_meta_raw": donor_raw}
        plan["donor_fallback"] = donor_fallback
        return plan

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path)
        n_show = min(len(df), args.limit if args.limit else 2)
        print(f"[gen_continuations] --dry-run: rendering {n_show} sites, no model load", flush=True)
        for i in range(n_show):
            plan = build_site_plan(df.iloc[i])
            print(f"\n=== site {plan['site_id']} nums={plan['nums']} target={plan['target']} "
                  f"family_dead={plan['family_dead']} ===")
            for mode, info in plan["per_mode"].items():
                rendered = render(tok, info["messages"])
                tail = rendered[-400:]
                print(f"--- mode={mode} (donor_meta_raw={'yes' if info['donor_meta_raw'] else 'no'}) "
                      f"rendered tail ---\n{tail}")
        print("\n[gen_continuations] dry-run 완료 — 생성/채점/출력 없음(exit 0).", flush=True)
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, dtype="bfloat16", seed=args.seed,
             gpu_memory_utilization=args.gpu_util, max_model_len=4096,
             enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)

    all_records: list[dict] = []
    site_family_dead: dict[str, object] = {}
    donor_fallback_n = 0
    n_sites_done = 0
    t0 = time.time()

    for chunk_i, chunk_df in enumerate(_chunks(df, CHUNK_SIZE)):
        plans = []
        for _, row in chunk_df.iterrows():
            plan = build_site_plan(row)
            plans.append(plan)
            site_family_dead[plan["site_id"]] = plan["family_dead"]
            if plan.get("donor_fallback"):
                donor_fallback_n += 1

        # 청크 안 (site, mode) 조합 전부를 하나의 generate 호출에 담는다.
        flat_keys: list[tuple] = []   # (site_idx, mode)
        flat_prompts: list[str] = []
        for si, plan in enumerate(plans):
            for mode, info in plan["per_mode"].items():
                flat_keys.append((si, mode))
                flat_prompts.append(render(tok, info["messages"]))

        outs = llm.generate(flat_prompts, sp)

        for (si, mode), out in zip(flat_keys, outs):
            plan = plans[si]
            info = plan["per_mode"][mode]
            for k_index, comp in enumerate(out.outputs):
                truncated = comp.finish_reason == "length"
                rec = build_record(
                    site_id=plan["site_id"], mode=mode, policy_tag=args.policy_tag,
                    k_index=k_index, prefix=plan["prefix"],
                    donor_meta_raw=info["donor_meta_raw"], nums=plan["nums"],
                    target=plan["target"], continuation=comp.text,
                    n_tokens=len(comp.token_ids), truncated=truncated)
                all_records.append(rec)

        n_sites_done += len(plans)
        print(f"[gen_continuations] chunk {chunk_i}: {n_sites_done}/{len(df)} sites done "
              f"({time.time() - t0:.0f}s elapsed, {len(all_records)} rows so far)", flush=True)

    out_df = pd.DataFrame(all_records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary = summarize(all_records, site_family_dead)
    summary["n_sites"] = len(df)
    summary["donor_fallback_n"] = donor_fallback_n
    summary["donor_pool_size"] = len(donor_pool)
    summary["modes"] = modes
    summary["k"] = args.k
    summary["policy_tag"] = args.policy_tag
    summary_path = out_path.parent / f"{out_path.name}.summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[gen_continuations] done. wrote {len(all_records)} rows -> {out_path}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
