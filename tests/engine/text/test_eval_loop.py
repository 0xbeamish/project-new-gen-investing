import json

import pytest

from engine import config
from engine.markets import synthetic_text as stm
from engine.text import evaluate as ev
from engine.text import harness, readers, textloop
from engine.text import questions as tq


@pytest.fixture(scope="module")
def sets_and_q(tmp_path_factory):
    from tests.engine.text.conftest import text_cfg

    st = config.from_dict(
        "synthetic_text", text_cfg(tmp_path_factory.mktemp("e"), entities=120)
    )
    return (
        st,
        stm.eval_sets(st.market, end="2015-01-01", n_gold=200),
        tq.load(config.path("markets/questions/synthetic_release.yaml")),
    )


def test_harness_scores_reading_fields_only_and_gates(sets_and_q):
    _, sets, q = sets_and_q
    r = harness.score(sets, q, readers.KeywordReader(), "dev", efficacy=True)
    assert set(r["per_field"]) == {
        "guidance_action",
        "demand",
        "one_off_charge",
        "buyback",
    }  # no judgment
    assert 0 < r["summary"]["S"] < 1 and r["summary"]["E"] == 1.0
    assert r["gates"]["leak"]["pass"]  # the keyword reader can't know outcomes
    assert (
        r["reaction"]["gain_vs_prior"]["IC_react"] > 0
    )  # the card beats its history prior
    eff = r["efficacy"].set_index("question")
    assert eff.loc["buyback", "recommend"] == "drop"  # it moves nothing
    assert eff.loc["guidance_action", "recommend"] == "keep"


def test_agreement_spotcheck_and_identification(sets_and_q, tmp_path):
    _, sets, q = sets_and_q
    a = ev.agreement(sets.gold, q)
    assert 0.8 < a["overall"] < 1.0  # the second labeler errs ~10% of the time
    texts = dict(zip(sets.docs["doc_id"], sets.docs["text"]))
    out = ev.spotcheck_export(sets.gold, texts, q, tmp_path / "spot.json", 5, 3)
    assert len(out) == 8 and json.loads((tmp_path / "spot.json").read_text())
    for d in out:
        for f in d["fields"]:
            f["correct"] = f["field"] != "demand"
            f["correction"] = 0 if f["field"] == "demand" else None
    gold, corr, wrong = ev.apply_spotcheck(sets.gold, out)
    assert corr and wrong == pytest.approx(0.25) and gold[0] is not sets.gold[0]
    # masked-name identification: pick the entity from 5 candidates. Masked: chance; unmasked: found
    ents = list(sets.names)
    docs = sets.docs[sets.docs["doc_id"].str.startswith("r")].head(60)
    unmasked = {d: t for d, t in zip(sets.docs["doc_id"], sets.docs["text"])}
    guesses, leaked, truth = [], [], []
    for i, (e, t) in enumerate(zip(docs["entity_id"], docs["text"])):
        cands = [sets.names[e].split()[0]] + [
            sets.names[x].split()[0] for x in ents[i + 1 : i + 5]
        ]
        qid = ev.identification_question(cands)
        guesses.append(readers.KeywordReader().answer(qid, t)["value"])
        raw = (
            unmasked[f"unmasked_{docs['doc_id'].iloc[i]}"]
            if f"unmasked_{docs['doc_id'].iloc[i]}" in unmasked
            else t.replace("COMPANY_A", sets.names[e])
        )
        leaked.append(readers.KeywordReader().answer(qid, raw)["value"])
        truth.append(cands[0])
    assert ev.identification_gate(guesses, truth)["pass"]
    assert not ev.identification_gate(leaked, truth)[
        "pass"
    ]  # the test can catch a leak


def test_loop_improves_on_dev_confirms_on_test_and_rations_test_opens(
    sets_and_q, tmp_path
):
    _, sets, q = sets_and_q
    cfg = textloop.LoopConfig(max_iter=4, n_boot=50, out_dir=tmp_path)
    res = textloop.run(
        sets, q, readers.KeywordReader(), textloop.KeywordProposer(), cfg
    )
    assert res["accepted"] >= 1 and (tmp_path / "loop_log.csv").exists()
    assert (
        res["best"]["summary"]["S"]
        > harness.score(sets, q, readers.KeywordReader())["summary"]["S"]
    )
    changed = [x for x in res["best_qsets"]["release"] if x.version > 1]
    assert changed  # accepted wording is a new version
    c = textloop.confirm(
        sets, q, res["best_qsets"], readers.KeywordReader(), cfg, "test 1"
    )
    assert c["pass"]
    for i in (2, 3):
        textloop.confirm(
            sets, q, res["best_qsets"], readers.KeywordReader(), cfg, f"test {i}"
        )
    with pytest.raises(PermissionError):
        textloop.confirm(
            sets, q, res["best_qsets"], readers.KeywordReader(), cfg, "a 4th look"
        )
    h = textloop.freeze(res["best_qsets"], tmp_path / "frozen.json")
    assert len(h) == 64


def test_invalid_changes_are_rejected_before_any_reading():
    q = tq.Question(
        "x", "yes_no", "COMPANY_A raised guidance and cut costs or sold assets."
    )
    assert "one condition" in textloop.validate(
        textloop.Change("rewrite", "release", [q]), set()
    )
    q2 = tq.Question("y", "yes_no", "Calculate the percentage of revenue from China.")
    assert "arithmetic" in textloop.validate(
        textloop.Change("add", "release", [q2]), set()
    )
    ch = textloop.Change(
        "add", "release", [tq.Question("z", "yes_no", "The text says Z.")]
    )
    assert textloop.validate(ch, {ch.signature()}) == "duplicate of a logged variant"
