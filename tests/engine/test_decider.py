import pandas as pd
import pytest

from engine import decider, models, pipeline
from engine.spend import BudgetExceeded, Ledger


def test_no_decider_is_the_model_and_cards_explain_the_score(study, tuning_rows):
    wf = pipeline.walk_forward(study)
    feats = ["planted", "noise_1"]
    scored, w = wf.run(tuning_rows, feats, study.market.calendar)
    scored = decider.batches(scored, size=10)
    keyed = tuning_rows.set_index(["entity_id", "decision_time"])
    raw = keyed.loc[
        list(zip(scored["entity_id"], scored["decision_time"])), feats
    ].reset_index(drop=True)
    ranked = models.rank_features(tuning_rows, feats).set_index(
        pd.MultiIndex.from_frame(tuning_rows[["entity_id", "decision_time"]])
    )
    x = ranked.loc[list(zip(scored["entity_id"], scored["decision_time"]))].reset_index(
        drop=True
    )
    contrib = x * w.reindex(scored["block"]).to_numpy()
    jobs = decider.build_jobs(
        scored, contrib, raw, w, {"planted": "the planted signal"}
    )
    assert jobs and "the planted signal" in jobs[0]["state"]["cards"]
    dec = decider.run(decider.NoDecider(), jobs)
    g = decider.grade(dec)
    assert g["override_rate"] == 0 and g["decider_vs_model"] == 0
    assert g["model_vs_random"] > 0  # the planted signal shows up in pick-1-of-10 too


def test_paid_deciders_refuse_past_their_cap(tmp_path):
    led = Ledger(
        tmp_path / "costs.csv",
        step_caps={"decider": 0.01},
        funds={"typesafe": 1.0},
        prices={"jev-1.13.0": {"provider": "typesafe", "input": 0.042}},
    )
    led.record("decider", "jev-1.13.0", 200_000)
    with pytest.raises(BudgetExceeded):
        led.guard("decider", "jev-1.13.0", 0.01)
    assert decider.make(None).name == "none"
    with pytest.raises(ValueError):
        decider.make({"kind": "jev"})
