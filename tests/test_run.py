"""Candidates (test, discover), the cohort test's plumbing, and the report."""

import numpy as np
import pandas as pd
import pytest

from engine import data, run
from engine.run import Candidate


def test_transforms_use_only_the_entitys_own_earlier_rows():
    f = pd.DataFrame(
        {
            "entity_id": ["a", "b", "a", "b", "a"],
            "decision_time": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29", "2020-03-31"], utc=True
            ),
            "x": [1.0, 10.0, 3.0, 20.0, 6.0],
        }
    )
    chg = run.transform(f, "x", "chg1")
    assert chg.isna().tolist() == [True, True, False, False, False]
    assert chg.dropna().tolist() == [2.0, 10.0, 3.0]
    assert run.transform(f, "x", "pct1").iloc[4] == pytest.approx(1.0)


def test_known_positive_is_kept_after_a_counted_check(study, tuning_rows):
    row = run.test_candidate(study, tuning_rows, ["noise_1"], Candidate("planted"))
    assert row["t_tune"] >= row["bar_tune"]
    assert row["check_used"] is True and row["t_check"] >= row["bar_check"]
    assert row["kept"] is True
    assert study.registry.check_uses() == 1 and study.registry.n_judged() == 1


def test_known_negative_noise_never_clears_the_bar_or_opens_the_check(study, tuning_rows):
    rows = [
        run.test_candidate(study, tuning_rows, ["noise_1"], Candidate(f, t))
        for f in ("noise_2", "noise_3")
        for t in ("level", "chg1")
    ]
    assert all(r["t_tune"] < r["bar_tune"] for r in rows)
    assert study.registry.check_uses() == 0
    assert [r["bar_tune"] for r in rows] == sorted(r["bar_tune"] for r in rows)  # the bar rises
    assert study.registry.n_judged() == 4


def test_an_empty_baseline_judges_the_candidates_own_rank_ic(study, tuning_rows):
    wf = run.walk_forward(study)
    planted = run.judge(study, tuning_rows, [], Candidate("planted"), wf)
    noise = run.judge(study, tuning_rows, [], Candidate("noise_2"), wf)
    assert planted["t_tune"] > 5 and abs(noise["t_tune"]) < 1.96


def test_loop_stops_at_the_first_kept_candidate_and_skips_tried_ones(study, tuning_rows):
    study.cfg["discovery"]["candidates"] = [
        {"feature": "noise_2"},
        {"feature": "planted"},
        {"feature": "noise_3"},
    ]
    out = run.discover(study, tuning_rows, ["planted", "noise_1", "noise_2", "noise_3"])
    assert [r["name"] for r in out] == ["noise_2", "planted"]
    again = run.ConfigProposer().propose(study, study.registry.tested(), [])
    assert [c.name for c in again] == ["noise_3"]


def test_report_is_descriptive_and_never_logs(study):
    rep = run.report(study, "all")
    assert rep["tuning"]["ic_t"] > 5 and rep["registry"]["judged_tests"] == 0
    assert not study.registry.path.exists()
    study.cfg["features"]["empty"] = []
    with pytest.raises(SystemExit):
        run.report(study, "empty")


# ---------------------------------------------------------------- the cohort test's plumbing
def test_composite_ranks_within_each_formation_and_fills_missing_with_the_median():
    f = pd.DataFrame(
        {
            "decision_time": ["t1"] * 4 + ["t2"] * 2,
            "entity_id": list("abcd") + list("ab"),
            "x": [1.0, 2.0, 3.0, 4.0, 10.0, 0.0],
            "y": [4.0, np.nan, 2.0, 1.0, 1.0, 2.0],
            "req": [1, 1, 1, np.nan, 1, 1],
        }
    )
    s = run.composite(f, ["x", "y"], ["req"], 0.5)
    assert np.isnan(s[3])  # d lacks the required input: not ranked
    assert s[0] == pytest.approx((1 / 3 + 1.0) / 2)  # ranked among a, b, c only
    assert s[1] == pytest.approx((2 / 3 + 0.5) / 2)  # y missing -> median rank
    assert s[4] == pytest.approx((1.0 + 0.5) / 2)  # t2 ranks on its own


def test_pick_takes_a_fraction_or_a_count():
    f = pd.DataFrame({"decision_time": "t", "entity_id": list("abcdefghij"), "s": range(10)})
    assert run.pick(f, "s", 0.2) == {"t": ["j", "i"]}
    assert run.pick(f, "s", 3, bottom=True) == {"t": ["a", "b", "c"]}


class _Market:
    calendar = data.TradingCalendar()

    def __init__(self, rets):
        self.rets = rets

    def daily_returns(self, codes):
        return self.rets, pd.Series(dtype="datetime64[ns]")

    def cost_bps(self, rows):
        return pd.Series(20.0, index=rows.index)  # 20 bp round trip


def test_cohort_portfolio_enters_after_formation_and_rolls_hold_cohorts():
    days = pd.bdate_range("2015-01-01", "2016-12-31")
    rets = pd.DataFrame({"a": 0.001, "b": 0.0}, index=days)
    cal = data.TradingCalendar()
    times = cal.decision_times("2015-01-01", "2016-01-01", "BQE")  # Mar 31 .. Dec 31 2015
    ct = object.__new__(run.CohortTest)
    ct.market, ct.cal, ct.times = _Market(rets), cal, times
    ct._rets = (rets, pd.Series(dtype="datetime64[ns]"))
    ct.study = type("S", (), {"cfg": {"schedule": "BQE"}})()
    m = ct.portfolio({t: ["a"] for t in times}, hold=2)
    assert m.index.min() == pd.Period("2015-04")  # bought at Apr 1's close, after formation
    assert m.loc[pd.Period("2015-05"), "cohorts"] == 1
    assert m.loc[pd.Period("2015-08"), "cohorts"] == 2
    assert m.loc[pd.Period("2015-10"), "cohorts"] == 2  # the first cohort sold as the third bought
    assert m["cohorts"].max() == 2  # never more than `hold` at once
    # 10 bp in, 10 bp out per cohort, net only
    gap = m.loc[pd.Period("2015-04"), "gross"] - m.loc[pd.Period("2015-04"), "net"]
    assert gap == pytest.approx(0.001, rel=0.05)
