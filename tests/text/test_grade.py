"""The eval harness: reading fields only, gates, efficacy, agreement, spot-check, identification."""

import json

import pytest

from engine.text import grade, read


def test_harness_scores_reading_fields_only_and_gates(sets_and_q):
    _, sets, q = sets_and_q
    r = grade.score(sets, q, read.KeywordReader(), "dev", efficacy=True)
    assert set(r["per_field"]) == {
        "guidance_action",
        "demand",
        "one_off_charge",
        "buyback",
    }  # no judgment
    assert 0 < r["summary"]["S"] < 1 and r["summary"]["E"] == 1.0
    assert r["gates"]["leak"]["pass"]  # the keyword reader can't know outcomes
    assert r["reaction"]["gain_vs_prior"]["IC_react"] > 0  # the card beats its history prior
    eff = r["efficacy"].set_index("question")
    assert eff.loc["buyback", "recommend"] == "drop"  # it moves nothing
    assert eff.loc["guidance_action", "recommend"] == "keep"


def test_agreement_spotcheck_and_identification(sets_and_q, tmp_path):
    _, sets, q = sets_and_q
    a = grade.agreement(sets.gold, q)
    assert 0.8 < a["overall"] < 1.0  # the second labeler errs ~10% of the time
    texts = dict(zip(sets.docs["doc_id"], sets.docs["text"]))
    out = grade.spotcheck_export(sets.gold, texts, q, tmp_path / "spot.json", 5, 3)
    assert len(out) == 8 and json.loads((tmp_path / "spot.json").read_text())
    for d in out:
        for f in d["fields"]:
            f["correct"] = f["field"] != "demand"
            f["correction"] = 0 if f["field"] == "demand" else None
    gold, corr, wrong = grade.apply_spotcheck(sets.gold, out)
    assert corr and wrong == pytest.approx(0.25) and gold[0] is not sets.gold[0]
    # masked-name identification: pick the entity from 5 candidates. Masked: chance; unmasked: found
    ents = list(sets.names)
    docs = sets.docs[sets.docs["doc_id"].str.startswith("r")].head(60)
    unmasked = {d: t for d, t in zip(sets.docs["doc_id"], sets.docs["text"])}
    guesses, leaked, truth = [], [], []
    for i, (e, t) in enumerate(zip(docs["entity_id"], docs["text"])):
        cands = [sets.names[e].split()[0]] + [sets.names[x].split()[0] for x in ents[i + 1 : i + 5]]
        qid = grade.identification_question(cands)
        guesses.append(read.KeywordReader().answer(qid, t)["value"])
        raw = (
            unmasked[f"unmasked_{docs['doc_id'].iloc[i]}"]
            if f"unmasked_{docs['doc_id'].iloc[i]}" in unmasked
            else t.replace("COMPANY_A", sets.names[e])
        )
        leaked.append(read.KeywordReader().answer(qid, raw)["value"])
        truth.append(cands[0])
    assert grade.identification_gate(guesses, truth)["pass"]
    assert not grade.identification_gate(leaked, truth)["pass"]  # the test can catch a leak
