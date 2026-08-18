"""Prompt-only meta: no SFT, no added vocabulary, plain-text <meta> tags.

WHY NO SFT
----------
Every accuracy gain the b4 generation produced traced to swapping the SFT2
corpus, not to the reward (b4v 67.40% vs b4p2 68.21% at step 200 -- inside
noise, while the init swap alone moved gs0 from 64.70 to 66.38). As long as
each generation ships a new SFT, the reward can never be the isolated variable.
Dropping SFT and putting the format in the prompt makes the public checkpoint
the shared origin, so two arms differ by exactly one thing: R_meta.

THINK-ANYWHERE (arXiv 2603.29957 §3.2) warns that prompt instructions "often
fail to enforce this behavior reliably", which is why they cold-start with
LoRA on ~5k samples. That is a claim about YIELD, and yield is measurable in
twenty minutes -- hence the pilot below, which is a STOP gate, not a formality.

Plain-text tags (not the 151669/151670 special tokens) keep this runnable on any
checkpoint with no vocabulary surgery. The cost is that a delimiter can come out
malformed; wellformed_rate measures exactly that.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

META_OPEN = "<meta>"
META_CLOSE = "</meta>"

# Measured on Qwen3-1.7B, 200 problems x n4, 43 s per iteration:
#
#   v1  schema as an imperative bullet list
#       -> emission 0.986, but 502/502 blocks COPIED the bullet list verbatim
#          before adding their own sentence (echo 1.00, meta_chars 411)
#   v2  one concrete worked example instead of the list
#       -> echo 0.00, but 91.9% copied THE EXAMPLE ("The 14 in the second
#          step..."), emission collapsed 0.986 -> 0.334, and redirect jumped
#          0.00 -> 0.53 purely because the example happened to say redirect
#
# Both failures are the same thing: at this size the model reproduces whatever
# concrete text the prompt shows it. v3 gives the SHAPE with uppercase slots
# (nothing quotable), two deliberately unlike examples so neither is the obvious
# thing to copy, and keeps v1's imperative framing, which is what drove emission
# to 0.99.
SYSTEM_PROMPT = """You are solving a competition math problem.

Solve it normally. Exactly once, immediately before you commit to the final
answer, you must write one self-check block in this form:

<meta>confidence: NUM | DOUBT | decision: verify</meta>

  NUM       your own probability, 0 to 1, that your current answer is right
  DOUBT     one sentence about a specific number, step, or assumption that
            appears in THIS problem and could be wrong. Write it in your own
            words about this problem -- never reuse the wording above or below.
  decision  verify   the approach is sound and only needs checking
            redirect the approach should be abandoned for a different route

Two blocks from other problems, to show the range -- do not borrow their content:

  <meta>confidence: 0.9 | Both cases give 12, so the only risk is that I \
double-counted the shared arrangement. | decision: verify</meta>
  <meta>confidence: 0.35 | I assumed the sequence is arithmetic, but the third \
term does not fit that gap. | decision: redirect</meta>

Write exactly one block, then finish the solution and give the final answer as
\\boxed{...}."""

# Every prompt version's quotable text, kept so echo stays measurable ACROSS
# versions -- a marker set that only knows the current prompt would report 0.00
# for a run that is copying the previous one.
_ECHO_MARKERS = (
    # v1 bullet list
    "one or two sentences naming what",
    "in terms of THIS problem's own quantities",
    "abandon it and take a different route",
    # v2 worked example
    "The 14 in the second step",
    "I used it as the total",
    # v3 slots and examples
    "confidence: NUM", "| DOUBT |", "DOUBT ",
    "double-counted the shared arrangement",
    "the third term does not fit that gap",
    "do not borrow their content",
)

USER_TEMPLATE = "{problem}"

_CONF_RE = re.compile(r"confidence:\s*([0-9]*\.?[0-9]+)")
_DEC_RE = re.compile(r"decision:\s*(verify|redirect)", re.I)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def parse_meta(text: str) -> dict:
    """One row of structure for a rollout. Every field is a gate input."""
    n_open, n_close = text.count(META_OPEN), text.count(META_CLOSE)
    i = text.find(META_OPEN)
    j = text.find(META_CLOSE, i + len(META_OPEN)) if i >= 0 else -1
    inner = text[i + len(META_OPEN):j] if (i >= 0 and j > i) else None
    conf = _CONF_RE.search(inner) if inner else None
    dec = _DEC_RE.search(inner) if inner else None
    boxed = _BOXED_RE.findall(text)
    echo = bool(inner) and any(m in inner for m in _ECHO_MARKERS)
    return {
        "has_meta": i >= 0,
        "closed": inner is not None,
        "echo": echo,
        # exactly one block, opened before it is closed, the answer AFTER it, and
        # the block is the model's own words -- an instruction echo is a constant
        # string, so it would make every boilerplate metric read 100% and every
        # ablation delta read 0 for a reason that has nothing to do with meta
        # usefulness.
        "wellformed": bool(
            inner is not None and n_open == 1 and n_close == 1
            and conf and dec and boxed
            and text.rfind("\\boxed{") > j
            and not echo
        ),
        "inner": inner,
        "conf": float(conf.group(1)) if conf else None,
        "decision": dec.group(1).lower() if dec else None,
        "answer": boxed[-1] if boxed else None,
        "meta_chars": len(inner) if inner else 0,
    }


def opener(inner: str, n: int = 6) -> str:
    """First n words with the confidence line stripped -- the boilerplate key.

    Recomputed per run: each arm may converge on its OWN template, so a hardcoded
    string would silently measure nothing in an arm that picked a different one.
    """
    w = re.sub(r"confidence:\s*[0-9.]+", "", inner or "")
    w = re.sub(r"decision:\s*\w+", "", w)
    return " ".join(w.split()[:n])


def summarize(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    metas = [r for r in rows if r["wellformed"]]
    heads = Counter(opener(r["inner"]) for r in metas)
    top = heads.most_common(1)[0] if heads else ("", 0)
    dec = Counter(r["decision"] for r in metas)
    return {
        "n": len(rows),
        "emission_rate": sum(r["has_meta"] for r in rows) / n,
        "wellformed_rate": sum(r["wellformed"] for r in rows) / n,
        "echo_rate": sum(r["echo"] for r in rows) / n,
        "boxed_rate": sum(r["answer"] is not None for r in rows) / n,
        "decision_dist": dict(dec),
        # redirect share is a program-level quantity, not a diagnostic: b4p2 ended
        # at 0.2% after 300 RL steps, so where it starts decides whether RL killed
        # it or it was never there.
        "redirect_rate": dec.get("redirect", 0) / max(1, len(metas)),
        "conf_unique": len({r["conf"] for r in metas if r["conf"] is not None}),
        "top_opener": top[0],
        "top_opener_share": top[1] / max(1, len(metas)),
        "meta_chars_mean": sum(r["meta_chars"] for r in metas) / max(1, len(metas)),
    }


# --------------------------------------------------------------- yield pilot

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--problems", default="problems_math500.json",
                    help="JSON list of {problem, gold} -- keeps this runnable in "
                         "the vLLM env, which has no `datasets`")
    ap.add_argument("--n_problems", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(args.problems) as f:
        pool = json.load(f)[:args.n_problems]
    problems = [r["problem"] for r in pool]
    golds = [r["gold"] for r in pool]

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=0.85, max_model_len=4096)
    tok = llm.get_tokenizer()

    def chat(p):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(problem=p)}]
        try:  # Qwen3 exposes a thinking switch; older templates do not
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts = [chat(p) for p in problems]
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                        top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    rows, records = [], []
    for o, prob, ans in zip(outs, problems, golds):
        for c in o.outputs:
            r = parse_meta(c.text)
            rows.append(r)
            records.append({"problem": prob, "gold": ans, "prompt": o.prompt,
                            "text": c.text, **{k: r[k] for k in
                                               ("has_meta", "closed", "wellformed",
                                                "conf", "decision", "answer")}})

    summary = summarize(rows)
    summary["model"] = args.model
    # Pre-registered STOP gate. Below either line the redesign has no substrate
    # and no node gets requested.
    summary["GATE_1_PASS"] = bool(summary["wellformed_rate"] >= 0.50
                                  and summary["boxed_rate"] >= 0.80)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
