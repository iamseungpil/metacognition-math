"""Countdown ↔ self-distill 어댑터 — 「메타를 self distill 해서 응답을 강화」의 실제 구현.

페이블 판정(2026-09-01): 4단계는 보상 조성 GRPO 가 아니라 **self-distill 이어야 한다.**
근거(코드로 확인됨):
  · src/training/self_distill/online.py `_select_best_candidate` 의 `correct_then_meta` 는
    사전식 튜플 `(is_correct, meta_commit_quality, selection_score, −len)` 로 고른다.
    → **어떤 메타 점수도 정답 불리언을 이길 수 없다.**
  · 그 위에 `require_correct_teacher=True` 와 출력 빌더의 재필터가 각각 한 겹씩 더 있다.
  → 적대적 메타는 «이미 정답인» 후보들 사이에서만 이길 수 있고, 거기서의 승리는 무해하다.
    즉 게이밍이 **구조적으로** 막힌다. 자(ruler)를 보상으로 쓰는 방식과 결정적으로 다르다.
  · 반면 보상 조성 GRPO 는 이길 수 없다 — logP 계열 자가 전멸했고(bakeoff), clean 사이트에서
    실제 메타의 평균 유도가치가 음수이며(L2 = −0.024), 역대 팔 효과(≤0.8pp)가 런 잡음 아래다.

이 어댑터가 하는 일은 셋뿐이다. 본체는 손대지 않는다.
  ① 문제 로더   Countdown parquet → OnlineSdpoProblem(question, gold_answer, metadata)
  ② 채점기      judge_completion 을 Countdown 정확 채점(grade)으로 갈아끼운다
                 ★수식 문자열 비교가 아니라 «세 규칙»(각 수 한 번 · 사칙 · 목표 도달) 채점이다.
                   Countdown 은 해가 여럿이라 문자열 정규화 비교는 정답을 오답으로 만든다.
  ③ 프롬프트    Countdown 시스템 프롬프트를 쓰게 한다 (2단계에서 이긴 칸을 넘길 수 있게 인자화)

주의: 본체는 메타 태그로 `<|meta|>`(src/metacot/prompt.META_START)를 쓰고 Countdown 은
      `<meta>` 를 쓴다. meta_commit_quality 가 태그를 못 찾으면 «모든 후보의 메타 점수가 0»
      이 되어 correct_then_meta 가 조용히 correctness_only 로 퇴화한다. 그래서 발사 전
      `--selfcheck` 로 두 모드의 선택이 실제로 갈리는지 확인한다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.training.countdown_task import PROMPT_VARIANTS, grade  # noqa: E402
import src.curriculum.rq3_pipeline as rq3  # noqa: E402
from src.training.self_distill.online import OnlineSdpoProblem  # noqa: E402


# ── ② 채점기 교체 ────────────────────────────────────────────────────────
def countdown_judge(completion: str, gold_answer: str) -> dict:
    """gold_answer 는 "nums|target" 로 인코딩한다 (본체가 문자열 하나만 넘겨주므로).

    Countdown 은 해가 여럿이라 «증인식과 문자열이 같은가»로 채점하면 안 된다.
    grade() 가 세 규칙(각 수 정확히 한 번 · 사칙/괄호 · 목표 도달)을 모두 확인한다.
    """
    try:
        nums_s, tgt_s = str(gold_answer).split("|")
        nums = [int(v) for v in nums_s.split(",") if v != ""]
        tgt = int(tgt_s)
    except Exception:
        return {"is_correct": False, "boxed_answer": None, "parse_error": True}
    ok = bool(grade(completion, nums, tgt))
    return {"is_correct": ok, "boxed_answer": None,
            "normalized_prediction": "", "normalized_gold": gold_answer}


def install_countdown_judge() -> None:
    """본체가 import 해 간 이름까지 함께 갈아끼운다 (from-import 는 사본을 만든다)."""
    rq3.judge_completion = countdown_judge
    import src.training.self_distill.online as on
    on.judge_completion = countdown_judge


def install_countdown_selector() -> None:
    """선택 점수 함수를 통째로 Countdown 판으로 바꾼다.

    본체의 `_score_completion_for_selection` 은 여섯 항목을 쓰는데 여섯 개가 전부 수학용이다
    — correctness_reward 는 문자열 정규화 비교(Countdown 은 해가 여럿이라 오답 처리),
      confidence_revision / redirect_execution / verify_execution / meta_floor 는 `<|meta|>`
      태그와 수학 어휘를 전제한다. 하나만 갈아끼우면 나머지 다섯이 조용히 잡음을 주입한다.
    `_select_best_candidate` 는 `selector_breakdown` 을 «받지 않고» 이 함수를 직접 호출하므로
    (시그니처 확인: candidates, gold_answer, *, selector_mode, weights) 여기가 유일한 주입점이다.
    """
    import src.training.self_distill.online as on

    def _cd_score(completion: str, gold_answer: str, *, weights=None) -> dict:
        try:
            nums_s, tgt_s = str(gold_answer).split("|")
            nums = [int(v) for v in nums_s.split(",") if v != ""]
            tgt = int(tgt_s)
        except Exception:
            nums, tgt = None, None
        corr = 1.0 if (nums is not None and bool(grade(completion, nums, tgt))) else 0.0
        mq = countdown_meta_quality(completion, nums, tgt)
        w = weights or {"correctness": 1.0, "meta_commit_quality": 0.25}
        comp = {"correctness": corr, "meta_commit_quality": mq["total"]}
        for k, v in mq.items():
            comp[f"meta_commit_quality_{k}"] = v
        comp["total"] = sum(float(w.get(k, 0.0)) * float(comp.get(k, 0.0)) for k in w)
        return comp

    on._score_completion_for_selection = _cd_score


def install_countdown_prompt(cell: str) -> None:
    """★네 번째 배선 구멍: 본체의 `_render_chat_prompt` 는 user 메시지 «하나만» 넣는다
    (online.py:140). 시스템 프롬프트가 통째로 빠지므로 모델은 Countdown 규칙도,
    `<meta>` 서식도 안 받는다 — 그대로 돌리면 메타가 한 줄도 안 나오고, 규칙 위반 식이
    쏟아진다. 여기서 시스템 프롬프트를 끼워 넣는다."""
    import src.training.self_distill.online as on
    sys_txt = system_prompt(cell)

    def _render(tokenizer, content: str) -> str:
        msgs = [{"role": "system", "content": sys_txt},
                {"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    on._render_chat_prompt = _render

    # ★여섯 번째 배선 구멍: SFT 데이터의 messages 에도 시스템 프롬프트가 있어야 한다.
    #   builders._build_messages_for_mode 의 naive 모드는 [user, assistant] 두 줄만 만든다.
    #   평가 때는 시스템 프롬프트를 «주는데» 학습 때는 «안 주면» 학습·평가 불일치가 된다.
    import src.training.self_distill.builders as bld
    _orig = bld._build_messages_for_mode

    def _with_sys(*a, **kw):
        msgs, comp, kind, ptext, synth = _orig(*a, **kw)
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": sys_txt}] + list(msgs)
        return msgs, comp, kind, ptext, synth

    bld._build_messages_for_mode = _with_sys


# ── ① 문제 로더 ──────────────────────────────────────────────────────────
def load_countdown_problems(data: str, limit: int, offset: int = 0) -> list[OnlineSdpoProblem]:
    df = pd.read_parquet(data).iloc[offset:offset + limit]
    out = []
    for _, r in df.iterrows():
        p = r["prompt"]
        q = p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p)
        nums = [int(v) for v in r["nums"]]
        out.append(OnlineSdpoProblem(
            question=q.strip(),
            gold_answer=",".join(map(str, nums)) + "|" + str(int(r["target"])),
            benchmark="countdown4",
            metadata={"nums": nums, "target": int(r["target"]),
                      "witness": str(r.get("witness", ""))}))
    return out


# ── ②-b 메타 품질 키 ────────────────────────────────────────────────────
#   진단(실측): 본체의 score_meta_commit_quality 는 `<|meta|>` 태그를 찾는데 Countdown 은
#   `<meta>` 를 쓴다. 그래서 메타가 «있는» 응답이 total −0.1000 (epistemic_outside_meta_penalty
#   가 발화), 메타가 «없는» 응답이 0.0000 을 받았다. 태그를 옮기면 +0.7569 로 정상화된다.
#   → correct_then_meta 가 메타를 «벌하는» 방향으로 돌 뻔했다. 자가점검이 잡았다.
#
#   다만 그 점수는 «형식»만 본다(블록 있나·하나인가·confidence 있나·뒤에 boxed 있나).
#   우리 의도는 「혼잣말이 다음 행동을 실제로 조작하는가」이므로, 게이밍을 막도록 만든
#   Countdown 전용 행동 점수를 곱해 쓴다. 두 축 모두 기계 채점이고 정답을 보지 않는다.
_TAGS = (("<meta>", "<|meta|>"), ("</meta>", "<|/meta|>"))


def _to_module_tags(text: str) -> str:
    for a, b in _TAGS:
        text = text.replace(a, b)
    return text


_META_ANY = None


def _meta_body(completion: str):
    """★두 태그 계열을 모두 받는다 — `<meta>`(Countdown)와 `<|meta|>`(본체).
    한쪽만 보면 SFT 를 한 번 거친 모델에서 조용히 beh=0 으로 퇴화한다(감사 지적)."""
    import re as _re
    global _META_ANY
    if _META_ANY is None:
        _META_ANY = _re.compile(r"<\|?meta\|?>(.*?)<\|?/meta\|?>", _re.S)
    return _META_ANY.search(str(completion or ""))


def countdown_meta_quality(completion: str, nums=None, target=None,
                           use_behavior: bool = False) -> dict:
    """1차 실행의 동점 처리 키. 정답 여부는 보지 않는다(순환 차단).

    ★감사 판정으로 두 곳을 고쳤다.
      ① 실격이 «곱하기 0» 이면 실격이 아니다 — 맨 응답(form 0.0)과 동점이 되고, 동점은
         길이로 갈려 «짧은 유출 응답»이 이긴다. form 은 음수도 될 수 있어(측정 −0.26)
         «실격된 유출»이 깨끗한 응답을 이기기까지 한다. → 덧셈형 큰 음수로 바꾼다.
      ② 행동 항(grounded/next_ok/followed)은 1차 선택에서 «뺀다» — 칸마다 서로 다른
         항을 속인다(P2 는 유출, P3 는 ruled_out 날조). 진단으로만 기록한다.
    """
    from src.training.meta_quality import score_meta_commit_quality
    form = float(score_meta_commit_quality(_to_module_tags(str(completion or "")))["total"])
    detail, leak = {}, 0
    m = _meta_body(completion)
    if m is not None and nums is not None and target is not None:
        import steer_prompts as SP
        detail = SP.score_meta(m.group(1), str(completion)[:m.start()],
                               str(completion)[m.end():], nums, target)
        leak = int(bool(detail.get("leak")) or detail.get("false_ruled_out", 0) > 0)
    if leak:
        total = -1000.0                    # 덧셈형 실격 — 어떤 form 보다도 낮다
    elif use_behavior:
        g = 1.0 if detail.get("grounded") else 0.0
        n = 1.0 if detail.get("next_ok") else 0.0
        f = detail.get("followed")
        total = form + 0.5 * (g + n + (f if f is not None else 0.0)) / 3.0
    else:
        total = form
    return {"total": total, "form": form, "leak": leak,
            **{f"b_{k}": v for k, v in detail.items()}}


# ── ③ 프롬프트 ───────────────────────────────────────────────────────────
def system_prompt(cell: str) -> str:
    if cell in PROMPT_VARIANTS:
        return PROMPT_VARIANTS[cell]
    import steer_prompts as SP           # 2단계에서 이긴 칸 (P0/P0e/P1/P2/P3)
    return SP.VARIANTS[cell]


# ── 발사 전 자가점검 ─────────────────────────────────────────────────────
def selfcheck(cell: str) -> int:
    """메타 태그 불일치로 correct_then_meta 가 correctness_only 로 조용히 퇴화하지 않는가."""
    from src.training.self_distill.online import _select_best_candidate
    sysx = system_prompt(cell)
    print(f"[selfcheck] 프롬프트 칸 {cell} · {len(sysx)}자 · <meta> 요구: "
          f"{'<meta>' in sysx}", flush=True)

    good = ("Try 25*3=75, too big.\n<meta>\nconfidence: 0.3\n"
            "The multiply-first family overshoots; not worth continuing.\n"
            "decision: redirect\n</meta>\n8*7=56, 56+25-3=78.\n\\boxed{(25-8)*(7-3)}")
    bare = "25*3=75.\n8*7=56.\n\\boxed{(25-8)*(7-3)}"
    NUMS, TGT = [25, 3, 7, 8], 68
    sg = countdown_meta_quality(good, NUMS, TGT)["total"]
    sb = countdown_meta_quality(bare, NUMS, TGT)["total"]
    # 태그 계열 양쪽을 다 받는가
    alt = good.replace("<meta>", "<|meta|>").replace("</meta>", "<|/meta|>")
    print(f"[selfcheck] 태그 양립: <meta> {sg:.4f}  <|meta|> "
          f"{countdown_meta_quality(alt, NUMS, TGT)['total']:.4f}", flush=True)
    print(f"[selfcheck] meta_commit_quality  메타있음 {sg:.4f}  메타없음 {sb:.4f}", flush=True)
    if sg <= sb:
        print("[selfcheck] ✗ 메타가 있는 응답이 더 높지 않다 — 태그 불일치일 가능성. "
              "이 상태로 돌리면 correct_then_meta 가 correctness_only 와 같아진다.", flush=True)
        return 1

    # 두 모드가 실제로 다른 후보를 고르는가 (정답 2개 중 메타 있는 쪽을 골라야 한다)
    rows = [{"completion": bare, "judgment": {"is_correct": True},
             "selector_breakdown": {"meta_commit_quality": sb}, "selection_score": 0.9},
            {"completion": good, "judgment": {"is_correct": True},
             "selector_breakdown": {"meta_commit_quality": sg}, "selection_score": 0.1}]
    GOLD = "25,3,7,8|68"
    try:
        a = _select_best_candidate(rows, GOLD, selector_mode="correctness_only")
        b = _select_best_candidate(rows, GOLD, selector_mode="correct_then_meta")
        # 반환 형태는 {"selected": <후보>, "ranked_candidates": [...], ...}
        ta = str(((a or {}).get("selected") or {}).get("completion", ""))[:24]
        tb = str(((b or {}).get("selected") or {}).get("completion", ""))[:24]
        same = ta == tb
        print(f"[selfcheck] correctness_only  → {ta!r}", flush=True)
        print(f"[selfcheck] correct_then_meta → {tb!r}", flush=True)
        print(f"[selfcheck] 두 모드의 선택이 갈리는가: {'아니오 ✗' if same else '예 ✓'}", flush=True)

        # ★구조적 게이밍 차단 검증 — 적대적 메타가 «오답»이면 절대 못 이겨야 한다.
        leak = ("<meta>\nconfidence: 0.99\nruled_out: none\n"
                "next: (25-8)*(7-3) is the answer\ndecision: verify\n</meta>\n"
                "\\boxed{25*3-7}")                       # 정답식을 메타에 흘렸지만 답은 오답
        rows2 = rows + [{"completion": leak, "judgment": {"is_correct": False},
                         "candidate_id": "leak"}]
        c = _select_best_candidate(rows2, GOLD, selector_mode="correct_then_meta")
        tc = str((c.get("selected") or {}).get("candidate_id", ""))
        blocked = tc != "leak"
        print(f"[selfcheck] 오답 적대 메타가 선택되는가: "
              f"{'아니오 ✓ (정답이 사전식 1순위)' if blocked else '예 ✗ 위험'}", flush=True)
        return 0 if (not same and blocked) else 1
    except Exception as e:
        print(f"[selfcheck] 선택기 호출 실패 — 시그니처 확인 필요: {type(e).__name__}: {e}",
              flush=True)
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/hf_data/metacot-sdc-data/countdown_train_4num_new.parquet")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--cell", default="new", help="시스템 프롬프트 칸 (new/P0/P1/P2/P3)")
    ap.add_argument("--selfcheck", action="store_true", help="발사 전 배선 점검만 하고 종료")
    ap.add_argument("--k", type=int, default=8, help="문제당 후보 수")
    ap.add_argument("--selector", default="correct_then_meta",
                    choices=["correctness_only", "correct_then_meta"],
                    help="reward_weighted 는 오답을 «직접» 고를 수 있어 금지(감사 지적)")
    ap.add_argument("--out_dir", default="cd6_work/sd/run1")
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args()

    install_countdown_judge()
    install_countdown_selector()
    install_countdown_prompt(args.cell)
    # 채점기 배선 확인 — 오답/정답/다른 유효해 세 경우
    t = countdown_judge("\\boxed{(25-8)*(7-3)}", "25,3,7,8|68")
    f1 = countdown_judge("\\boxed{25*3-7}", "25,3,7,8|68")          # 수를 다 안 씀 → 오답
    f2 = countdown_judge("no answer here", "25,3,7,8|68")
    print(f"[wire] 채점기  유효해 {t['is_correct']}  수누락 {f1['is_correct']}  "
          f"무응답 {f2['is_correct']}", flush=True)
    assert t["is_correct"] and not f1["is_correct"] and not f2["is_correct"], "채점기 배선 실패"

    probs = load_countdown_problems(args.data, args.limit)
    print(f"[wire] 문제 {len(probs)}  예시 gold='{probs[0].gold_answer}'", flush=True)

    rc = selfcheck(args.cell)
    if args.selfcheck:
        print(f"[selfcheck] 종료 코드 {rc}", flush=True)
        raise SystemExit(rc)
    if rc != 0:
        raise SystemExit("자가점검 실패 — 배선을 고치기 전에는 돌리지 않는다")

    # ★버그 A(감사 지적): 이전 판은 여기서 «끝났다». 실제 구동기는 별도 인터프리터의
    #   scripts/run_online_sdpo_regen.py 라서 이 원숭이패치가 전혀 닿지 않았고, 그대로
    #   돌렸다면 수학 채점기가 "25,3,7,8|68" 을 채점해 전량 오답 처리 → 빈 데이터셋으로
    #   GPU 만 태웠을 것이다. 같은 프로세스에서 굴린다.
    from src.training.self_distill.online import (
        run_online_question_only_best_of_n_rollouts, write_online_sdpo_outputs)
    outdir = Path(args.out_dir)
    assert not outdir.exists() or not any(outdir.iterdir()), \
        "resume 는 이전 판정을 «재채점 없이» 그대로 읽는다 — 채점기를 바꾼 뒤에는 새 폴더를 쓴다"
    outdir.mkdir(parents=True, exist_ok=True)
    for gold in (p.gold_answer for p in probs):
        assert "|" in gold and gold.split("|")[1].isdigit(), f"gold 인코딩 파손: {gold!r}"

    print(f"[run] 문제 {len(probs)} · K={args.k} · 칸 {args.cell} · 선택 {args.selector} "
          f"· 출력 {outdir}", flush=True)
    # 구동기는 llm·tokenizer 를 «받는다» (직접 만들지 않는다) — 여기서 만든다.
    import os as _os
    _os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM
    from transformers import AutoTokenizer
    MODEL = _os.environ.get(
        "PROBE_MODEL",
        "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_model_len=4864, enforce_eager=True)

    res = run_online_question_only_best_of_n_rollouts(
        llm=llm, tokenizer=tok, problems=probs, output_dir=str(outdir),
        num_candidates=args.k, selector_mode=args.selector,
        require_correct_teacher=True, resume=False,
        max_new_tokens=3072, temperature=1.0, top_p=1.0, seed=args.seed)
    write_online_sdpo_outputs(rows=res, output_dir=str(outdir),
                              source_tag=f"countdown_{args.selector}",
                              mode="naive")
    print(f"[run] 완료 → {outdir}", flush=True)


if __name__ == "__main__":
    main()
