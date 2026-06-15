import pandas as pd

from src.training.verl_gdpo_data import _gold_is_rule_gradable, build_v8_meta_subset


def test_numeric_and_boxed_gold_pass():
    assert _gold_is_rule_gradable("42")
    assert _gold_is_rule_gradable("\\frac{3}{4}")
    assert _gold_is_rule_gradable("7\\sqrt{5}")


def test_prose_gold_rejected():
    assert not _gold_is_rule_gradable("\\text{Yes, it must be a cube.}")
    assert not _gold_is_rule_gradable("Player 0 wins")
    assert not _gold_is_rule_gradable("")


def _toy_corpus(tmp_path):
    rows = []
    specs = [("redirect", "easy", "gsm8k", "5"),
             ("verify", "medium", "hendrycks_math/algebra", "7"),
             ("redirect", "hard", "omni-math", "\\text{Yes}"),  # prose -> dropped
             ("verify", "easy", "gsm8k", "3")]
    for sc, df_, src, gt in specs:
        rows.append({"scenario": sc, "difficulty": df_, "source": src, "trigger": "anomaly",
                     "messages": [{"role": "user", "content": "Q?"},
                                  {"role": "assistant", "content": f"x \\boxed{{{gt}}}"}]})
    meta = pd.DataFrame(rows)
    base = meta.copy()
    mp = tmp_path / "meta.parquet"
    bp = tmp_path / "base.parquet"
    meta.to_parquet(mp)
    base.to_parquet(bp)
    return str(mp), str(bp)


def test_meta_subset_widens_scenarios_difficulties_and_drops_prose(tmp_path):
    mp, bp = _toy_corpus(tmp_path)
    out = build_v8_meta_subset(mp, bp, scenarios=("redirect", "verify"),
                               allowed_difficulties=("easy", "medium", "hard"),
                               require_gradable_gold=True, val_ratio=0.25, seed=0)
    rows = out["meta_train"] + out["meta_val"]
    scns = {r["split_tags"]["scenario"] for r in rows}
    diffs = {r["split_tags"]["difficulty"] for r in rows}
    assert scns == {"redirect", "verify"}        # both scenarios kept
    assert "easy" in diffs                        # easy restored
    assert len(rows) == 3                          # prose-gold row dropped (4 -> 3)
