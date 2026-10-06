"""The question-improvement loop: improves on dev, confirms on test, rations test openings."""

import pytest

from engine.text import grade, read
from engine.text import improve as text_improve
from engine.text import questions as tq


def test_loop_improves_on_dev_confirms_on_test_and_rations_test_opens(sets_and_q, tmp_path):
    _, sets, q = sets_and_q
    cfg = text_improve.LoopConfig(max_iter=4, n_boot=50, out_dir=tmp_path)
    res = text_improve.run_loop(sets, q, read.KeywordReader(), text_improve.KeywordProposer(), cfg)
    assert res["accepted"] >= 1 and (tmp_path / "loop_log.csv").exists()
    assert res["best"]["summary"]["S"] > grade.score(sets, q, read.KeywordReader())["summary"]["S"]
    changed = [x for x in res["best_qsets"]["release"] if x.version > 1]
    assert changed  # accepted wording is a new version
    c = text_improve.confirm(sets, q, res["best_qsets"], read.KeywordReader(), cfg, "test 1")
    assert c["pass"]
    for i in (2, 3):
        text_improve.confirm(sets, q, res["best_qsets"], read.KeywordReader(), cfg, f"test {i}")
    with pytest.raises(PermissionError):
        text_improve.confirm(sets, q, res["best_qsets"], read.KeywordReader(), cfg, "a 4th look")
    h = text_improve.freeze(res["best_qsets"], tmp_path / "frozen.json")
    assert len(h) == 64


def test_invalid_changes_are_rejected_before_any_reading():
    q = tq.Question("x", "yes_no", "COMPANY_A raised guidance and cut costs or sold assets.")
    assert "one condition" in text_improve.validate(
        text_improve.Change("rewrite", "release", [q]), set()
    )
    q2 = tq.Question("y", "yes_no", "Calculate the percentage of revenue from China.")
    assert "arithmetic" in text_improve.validate(text_improve.Change("add", "release", [q2]), set())
    ch = text_improve.Change("add", "release", [tq.Question("z", "yes_no", "The text says Z.")])
    assert text_improve.validate(ch, {ch.signature()}) == "duplicate of a logged variant"
