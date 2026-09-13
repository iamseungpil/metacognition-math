"""math_site_emission_eval 회귀 시험 (CPU, vllm 모의).

1. «그 자리에서» vs «나중에» 발화 판정(400자 경계).
2. 모의 생성기로 라벨별 집계 — 발화율·정확도·판단 일치율.
3. 강제 프롬프트(math_new) 거부 — vllm 을 올리기 전에 죽는다.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_site_emission_eval as E  # noqa: E402

META_V = "<meta>\nconfidence: 0.8\nThe method is fine.\ndecision: verify\n</meta>\n"
META_R = "<meta>\nconfidence: 0.2\nThis is going nowhere.\ndecision: redirect\n</meta>\n"
META_NODEC = "<meta>\nconfidence: 0.6\nHmm, unsure about the setup.\n</meta>\n"


def test_at_site_vs_later_detection():
    c = E.classify_continuation("so\n" + META_V + "\\boxed{2}")
    assert (c["emitted"], c["at_site"], c["decision"]) == (1, 1, "verify")
    late = "x" * (E.AT_SITE_CHARS + 5) + "\n" + META_R
    c = E.classify_continuation(late)
    assert (c["emitted"], c["at_site"], c["decision"]) == (1, 0, "redirect")
    c = E.classify_continuation(META_NODEC)
    assert (c["emitted"], c["at_site"], c["decision"]) == (1, 1, None)
    c = E.classify_continuation("plain \\boxed{2}")
    assert (c["emitted"], c["at_site"], c["decision"]) == (0, 0, None)
    # 신뢰도 없는 블록은 form="math" 에서 미발화.
    assert E.classify_continuation("<meta>\njust words\n</meta>")["emitted"] == 0


def _site(sid, label, best, gold="2"):
    return {"site_id": sid, "label": label, "best_decision": best, "gold": gold,
            "problem": "P", "prefix": "step\n", "p_nometa": 0.5}


def test_per_label_aggregation_with_mock_generator():
    sites = [_site("s1", "SAVE", "redirect"), _site("s2", "DERAIL", "verify"),
             _site("s3", "NEUTRAL", "tie")]
    late = "y" * (E.AT_SITE_CHARS + 1) + "\n"
    outs = [
        # SAVE: 2 at-site(redirect 일치, verify 불일치), 1 late(nodec), 1 none. 정답 3/4.
        [(META_R + "\\boxed{2}", 0), (META_V + "\\boxed{2}", 0),
         (late + META_NODEC + "\\boxed{2}", 0), ("\\boxed{7}", 1)],
        # DERAIL: 발화 0, 정답 4/4.
        [("\\boxed{2}", 0)] * 4,
        # NEUTRAL(tie): 발화 at-site 4/4, 판단 분모에서 제외. 정답 0/4.
        [(META_V + "\\boxed{9}", 0)] * 4,
    ]
    calls = []

    def gen(prompts):
        calls.append(list(prompts))
        return outs

    res = E.run_eval(sites, gen, "math_opt", k=4)
    assert calls == [["step\n"] * 3], "tok=None 이면 프롬프트는 prefix 그대로(모의 경로)"
    s = res["summary"]
    assert s["n_by_label"] == {"SAVE": 1, "DERAIL": 1, "NEUTRAL": 1}
    assert s["emit_at_site_by_label"] == pytest.approx({"SAVE": 0.5, "DERAIL": 0.0, "NEUTRAL": 1.0})
    assert s["emit_anywhere_by_label"] == pytest.approx({"SAVE": 0.75, "DERAIL": 0.0, "NEUTRAL": 1.0})
    assert s["acc_by_label"] == pytest.approx({"SAVE": 0.75, "DERAIL": 1.0, "NEUTRAL": 0.0})
    assert s["overall_acc"] == pytest.approx(7 / 12)
    # 판단: SAVE 자리의 발화 3개만 분모(DERAIL 발화 없음, NEUTRAL tie 제외).
    #   redirect 일치 1 / verify 불일치 0 / 결정 없음 0 → 1/3 ; 결정 있는 것만 → 1/2.
    assert s["n_judged_conts"] == 3 and s["judgment_match_rate"] == pytest.approx(1 / 3)
    assert s["n_decided_conts"] == 2 and s["judgment_match_rate_decided"] == pytest.approx(0.5)
    assert s["truncated_rate"] == pytest.approx(1 / 12)
    ps = {p["site_id"]: p for p in res["per_site"]}
    assert ps["s1"]["n_redirect"] == 1 and ps["s1"]["n_verify"] == 1
    assert ps["s3"]["judgment_match"] is None and ps["s2"]["judgment_match"] is None
    assert "| SAVE | 1 | 0.500 | 0.750 | 0.750 |" in E.markdown_table(s)
    with pytest.raises(RuntimeError):
        E.run_eval(sites, lambda p: outs[:2], "math_opt", k=4)


def test_gen_request_matches_math_sites_context():
    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking=None):
            return "|".join(m["content"][:8] for m in msgs) + "<gen>"

    import math_sites as MS
    sites = [_site("s1", "SAVE", "redirect")]
    seen = []

    def gen(prompts):
        seen.extend(prompts)
        return [[("\\boxed{2}", 0)]]

    E.run_eval(sites, gen, "math_opt", k=1, tok=Tok())
    assert seen == [MS.gen_request(Tok(), "math_opt", "P", "step\n")]
    assert seen[0].endswith("<gen>step\n"), "앞부분은 생성 프롬프트 뒤에 rstrip 없이 그대로"


def test_mandate_refusal(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="At least once"):
        E.check_variant("math_new")
    with pytest.raises(SystemExit):
        E.check_variant("nope")
    assert E.check_variant("math_opt") == "math_opt"
    with pytest.raises(SystemExit):
        E.run_eval([], lambda p: [], "math_new", k=1)
    # main() 은 vllm 을 올리기 **전에** 거부해야 한다 — 독이 든 vllm 모듈을 심어 확인.
    poison = types.ModuleType("vllm")

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("vllm 이 올라갔다 — 거부가 늦다")
    poison.LLM = _Boom
    poison.SamplingParams = _Boom
    monkeypatch.setitem(sys.modules, "vllm", poison)
    sp = tmp_path / "sites.jsonl"
    sp.write_text(json.dumps(_site("s1", "SAVE", "redirect")) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", "--sites", str(sp), "--model_path", "m",
                                      "--variant", "math_new", "--out_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="At least once"):
        E.main()
