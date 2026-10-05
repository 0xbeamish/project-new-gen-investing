import pytest

from engine import config
from engine.text import masking
from engine.text import questions as tq


def test_question_sets_load_and_keep_judgment_out_of_the_key():
    single = tq.load(config.path("markets/questions/synthetic_release.yaml"))["release"]
    assert {q.id for q in single.reading()} == {
        "guidance_action",
        "demand",
        "one_off_charge",
        "buyback",
    }
    assert {q.id for q in single.judgment()} == {"overall_tone", "leak_probe"}
    bundle = tq.load(config.path("markets/questions/sec_8k_card.yaml"))
    assert "direction" in bundle["earnings"].ids() and "direction" not in [
        q.id for q in bundle["earnings"].reading()
    ]
    assert bundle["headline"].ids()[:2] == [
        "relation",
        "is_subject",
    ]  # block.field picks


def test_questions_are_typed_and_versioned():
    with pytest.raises(ValueError):
        tq.Question(
            "x", "scale", "How?", levels=("a", "b")
        )  # a scale needs five anchors
    with pytest.raises(ValueError):
        tq.Question("x", "choice", "Which?", options=("only",))
    with pytest.raises(ValueError):
        tq.Question("x", "yes_no", "Is it?", tag="opinion")
    q = tq.Question("x", "yes_no", "The text says X.")
    q2 = tq.bump(q, prompt="The text says Y.")
    assert q2.version == 2 and q2.key == "x@v2"
    qs = tq.QuestionSet("d", [q])
    assert qs.fingerprint() != qs.with_question(q2).fingerprint()


def test_masking_hides_names_first_words_and_tickers():
    words = {"cypress", "first", "national", "the"}
    text = "Sucampo Pharmaceuticals, Inc. (SCMP) said Sucampo grew. Cypress Semiconductor beat; the cypress tree stood."
    m = masking.mask(
        text, ["Sucampo Pharmaceuticals Inc"], "SCMP", extras=(), words=words
    )
    assert "Sucampo" not in m and "SCMP" not in m
    m2 = masking.mask(
        text, ["Cypress Semiconductor Corp"], None, extras=(), words=words
    )
    assert (
        "Cypress" not in m2 and "cypress tree" in m2
    )  # a dictionary-word name only when capitalized
    m3 = masking.mask(
        "First National Bank of X: First quarter results",
        ["First National Bank"],
        None,
        extras=(),
        words=words,
    )
    assert "First quarter" in m3  # generic first words stay
    assert masking.residual_names(m, ["Sucampo Pharmaceuticals Inc"]) == []
    assert masking.alternative("COMPANY_A grew") == "the Company grew"


def test_masking_extras_hide_drugs_codes_and_trials():
    t = "Results for semaglutide (ABC-123) in the STEP-1 trial; FDA review in FY2016."
    m = masking.mask(t, ["Nobody Corp"], None, words=set())
    assert "semaglutide" not in m and "ABC-123" not in m and "STEP-1" not in m
    assert "FDA" in m and "FY2016" in m
