from src.training.meta_template import rebuild_meta_block


def test_rebuild_extracts_labels_and_caps():
    raw = ("confidence: 0.22\nThe current route is weak because ...\n"
           "assessment: boundary tracing needed\naction: switch to boundary view\n"
           "study_need: composite regions\n")
    out = rebuild_meta_block(raw, max_chars=200)
    # fixed order, only known labels, capped
    assert out.startswith("confidence: 0.22")
    assert "assessment:" in out and "action:" in out
    assert "study_need:" not in out   # not in the fixed 3-line template
    assert len(out) <= 200


def test_missing_label_skipped():
    out = rebuild_meta_block("confidence: 0.5\naction: verify the boundary\n", max_chars=200)
    assert out.startswith("confidence: 0.5")
    assert "action:" in out
    assert "assessment:" not in out   # absent -> skipped, not fabricated
