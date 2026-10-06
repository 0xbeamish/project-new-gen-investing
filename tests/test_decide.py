"""The AI decider layer: the free decider is the model, paid ones refuse past their caps, and the
feedback note reports closed data only and changes nothing."""

import numpy as np
import pandas as pd
import pytest

from engine import decide, model, run
from engine.spend import BudgetExceeded, Ledger


def test_no_decider_is_the_model_and_cards_explain_the_score(study, tuning_rows):
    feats = ["planted", "noise_1"]
    scored, w = run.walk_forward(study).run(tuning_rows, feats, study.market.calendar)
    scored = decide.batches(scored, size=10).reset_index(drop=True)
    scored["rt_cost"] = scored["rt_cost"].fillna(0.0)
    idx = list(zip(scored["entity_id"], scored["decision_time"]))
    raw = (
        tuning_rows.set_index(["entity_id", "decision_time"]).loc[idx, feats].reset_index(drop=True)
    )
    contrib = (
        model.rank_features(tuning_rows, feats)
        .set_index(pd.MultiIndex.from_frame(tuning_rows[["entity_id", "decision_time"]]))
        .loc[idx]
        .reset_index(drop=True)
        * w.reindex(scored["block"]).to_numpy()
    )
    jobs = []
    out = decide.run_batches(
        decide.NoDecider(),
        scored,
        tuning_rows,
        contrib,
        raw,
        feats,
        {"planted": "the planted signal"},
        {},
        feedback=False,
        on_batch=jobs.append,
    )
    assert jobs and "the planted signal" in jobs[0]["state"]["cards"]
    g = decide.grade_decisions(out)
    assert g["override_rate"] == 0 and g["decider_minus_model_gross"] == 0
    assert g["model_vs_batch_gross"] > 0  # the planted signal shows up in pick-1-of-10 too


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
    assert decide.make_decider(None).name == "none"
    with pytest.raises(ValueError):
        decide.make_decider({"kind": "jev"})


def test_feedback_note_reports_closed_data_only_and_changes_nothing(small_text):
    st, _rows = small_text
    prep = decide.prepare_cards(st, "numbers_text")
    w_before = prep["contrib"].copy()
    notes = []
    out = decide.run_batches(
        decide.NoDecider(),
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
        on_batch=lambda j: notes.append((j["decision_time"], j["state"]["reliability_note"])),
    )
    assert not out["override"].any()
    first_t = min(t for t, _ in notes)
    assert "little track record" in next(n for t, n in notes if t == first_t)
    kept = next(ln for ln in notes[-1][1].splitlines() if ln.startswith("Questions with a track"))
    assert "demand" in kept.split(";")[0]  # U-shaped: ~0 linear IC, but its levels have a record
    pd.testing.assert_frame_equal(w_before, prep["contrib"])  # reporting only
    tab, ends = decide.ic_table(prep["rows"], prep["features"])
    t = sorted(prep["scored"]["decision_time"].unique())[20]
    rec = decide.input_record(tab, ends, t)
    assert rec["periods"].max() <= (ends < t).sum()  # only periods whose labels had all ended
    g = decide.grade_decisions(out)
    assert g["decider_minus_model_net"] == 0 and (
        np.isnan(g["decider_minus_model_net_t"]) or g["decider_minus_model_net_t"] == 0
    )


def test_a_logged_decider_run_is_one_registry_row(small_text):
    st, _ = small_text
    res = decide.run_decider(st, "none", "numbers_text", log=True, note="demo")
    assert res["grade"]["override_rate"] == 0
    assert res["registry"]["name"] == "decider:none+feedback vs model top pick"
    assert len(st.registry.own()) == 1
    assert st.registry.n_judged() == 0  # the model vs itself has no t: logged, not judged
