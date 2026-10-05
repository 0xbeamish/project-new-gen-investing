import pandas as pd
import pytest

from engine import discovery
from engine.discovery import Candidate


def test_transforms_use_only_the_entitys_own_earlier_rows():
    f = pd.DataFrame(
        {
            "entity_id": ["a", "b", "a", "b", "a"],
            "decision_time": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29", "2020-03-31"],
                utc=True,
            ),
            "x": [1.0, 10.0, 3.0, 20.0, 6.0],
        }
    )
    chg = discovery.transform(f, "x", "chg1")
    assert chg.isna().tolist() == [True, True, False, False, False]
    assert chg.dropna().tolist() == [2.0, 10.0, 3.0]
    assert discovery.transform(f, "x", "pct1").iloc[4] == pytest.approx(1.0)


def test_known_positive_is_kept_after_a_counted_check(study, tuning_rows):
    row = discovery.test_candidate(
        study, tuning_rows, ["noise_1"], Candidate("planted")
    )
    assert row["t_tune"] >= row["bar_tune"]
    assert row["check_used"] is True and row["t_check"] >= row["bar_check"]
    assert row["kept"] is True
    assert study.registry.check_uses() == 1 and study.registry.n_judged() == 1


def test_known_negative_noise_never_clears_the_bar_or_opens_the_check(
    study, tuning_rows
):
    rows = [
        discovery.test_candidate(study, tuning_rows, ["noise_1"], Candidate(f, t))
        for f in ("noise_2", "noise_3")
        for t in ("level", "chg1")
    ]
    assert all(r["t_tune"] < r["bar_tune"] for r in rows)
    assert study.registry.check_uses() == 0
    assert [r["bar_tune"] for r in rows] == sorted(
        r["bar_tune"] for r in rows
    )  # the bar rises
    assert study.registry.n_judged() == 4


def test_loop_stops_at_the_first_kept_candidate_and_skips_tried_ones(
    study, tuning_rows
):
    study.cfg["discovery"]["candidates"] = [
        {"feature": "noise_2"},
        {"feature": "planted"},
        {"feature": "noise_3"},
    ]
    out = discovery.run(
        study, tuning_rows, ["planted", "noise_1", "noise_2", "noise_3"]
    )
    assert [r["name"] for r in out] == ["noise_2", "planted"]
    again = discovery.ConfigProposer().propose(study, study.registry.tested(), [])
    assert [c.name for c in again] == ["noise_3"]
