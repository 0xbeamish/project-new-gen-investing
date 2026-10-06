"""Overlapping-cohort portfolio math, the Newey-West t and the cohort study's plumbing, on
synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from engine import calendar as calmod
from engine import cohort_study, cohorts, scoring


def _rets(cols: dict, start="2015-01-01", n=10) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {k: v if np.ndim(v) else [v] * n for k, v in cols.items()}, index=days
    )


def test_buy_and_hold_drifts_from_equal_weight():
    r = _rets({"a": 0.10, "b": 0.0}, n=3)
    p = cohorts.cohort_returns(r, ["a", "b"], r.index[0])
    # day 1: (1.1 + 1) / 2 - 1 = 5%; day 2: a is now 1.1 / 2.1 of the cohort
    assert p["gross"].iloc[0] == pytest.approx(0.05)
    assert p["gross"].iloc[1] == pytest.approx(0.10 * 1.1 / 2.1)
    assert list(p.index) == list(
        r.index[1:]
    )  # bought at entry_day's close: returns start after


def test_entry_and_exit_costs_are_half_round_trips():
    r = _rets({"a": 0.0, "b": 0.0}, n=4)
    half = pd.Series({"a": 0.002, "b": 0.004})
    p = cohorts.cohort_returns(r, ["a", "b"], r.index[0], r.index[3], half, half)
    assert p["net"].iloc[0] == pytest.approx(-0.003)  # mean entry cost
    assert p["net"].iloc[1] == pytest.approx(0.0)
    assert p["net"].iloc[-1] == pytest.approx(-0.003, rel=1e-3)  # exit, value-weighted
    assert p["gross"].abs().max() == 0.0
    assert len(p) == 3  # held entry < d <= exit


def test_no_exit_cost_when_the_exit_is_past_the_data():
    r = _rets({"a": 0.0}, n=3)
    p = cohorts.cohort_returns(
        r, ["a"], r.index[0], pd.Timestamp("2030-01-01"), None, pd.Series({"a": 0.01})
    )
    assert p["net"].abs().max() == 0.0


def test_a_delisted_member_is_sold_and_its_value_spread_over_the_rest():
    a = [0.0, -0.5, np.nan, np.nan]  # day 2 carries the delisting return; then no bars
    r = _rets({"a": a, "b": [0.0, 0.0, 0.10, 0.10]}, n=4)
    ended = pd.Series({"a": r.index[1]})
    p = cohorts.cohort_returns(r, ["a", "b"], r.index[0], ended=ended)
    assert p["gross"].iloc[0] == pytest.approx(-0.25)  # a's last day counts
    assert p["gross"].iloc[1] == pytest.approx(0.10)  # then the cohort is all b
    assert p["gross"].iloc[2] == pytest.approx(0.10)


def test_a_member_dead_before_entry_is_never_bought():
    r = _rets({"a": [0.0] + [np.nan] * 4, "b": 0.02}, n=5)
    p = cohorts.cohort_returns(
        r, ["a", "b"], r.index[1], ended=pd.Series({"a": r.index[0]})
    )
    assert p["gross"].iloc[0] == pytest.approx(0.02)


def test_a_missing_bar_earns_zero_but_keeps_its_weight():
    r = _rets({"a": [0.0, np.nan, 0.2], "b": [0.0, 0.0, 0.0]}, n=3)
    p = cohorts.cohort_returns(r, ["a", "b"], r.index[0])
    assert p["gross"].iloc[0] == 0.0 and p["gross"].iloc[1] == pytest.approx(0.1)


def test_overlapping_averages_the_cohorts_held_each_day_and_counts_them():
    r = _rets({"a": 0.01, "b": 0.03}, n=6)
    c1 = cohorts.cohort_returns(r, ["a"], r.index[0], r.index[3])
    c2 = cohorts.cohort_returns(r, ["b"], r.index[2])
    d = cohorts.overlapping([c1, c2])
    assert d.loc[r.index[1], "gross"] == pytest.approx(0.01)
    assert d.loc[r.index[3], "gross"] == pytest.approx(0.02)  # both held
    assert d.loc[r.index[4], "gross"] == pytest.approx(
        0.03
    )  # c1 sold at index 3's close
    assert d["cohorts"].tolist() == [1, 1, 2, 1, 1]


def test_monthly_compounds_daily_returns():
    days = pd.bdate_range("2015-01-26", "2015-02-06")
    d = pd.DataFrame({"gross": 0.01, "net": 0.01, "cohorts": 2}, index=days)
    m = cohorts.monthly(d)
    assert m.loc[pd.Period("2015-01"), "gross"] == pytest.approx(1.01**5 - 1)
    assert m["cohorts"].tolist() == [2, 2]


def test_newey_west_t_matches_iid_without_lags_and_shrinks_with_autocorrelation():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.01, 0.02, 200))
    n = len(x)
    assert scoring.newey_west_t(x, 0) == pytest.approx(
        scoring.per_period_t(x) * np.sqrt(n / (n - 1))
    )
    smooth = (
        x.rolling(12).mean().dropna()
    )  # overlapping windows: strong positive autocorrelation
    assert scoring.newey_west_t(smooth, 12) < 0.6 * scoring.per_period_t(smooth)


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
    s = cohort_study.composite(f, ["x", "y"], ["req"], 0.5)
    assert np.isnan(s[3])  # d lacks the required input: not ranked
    assert s[0] == pytest.approx((1 / 3 + 1.0) / 2)  # ranked among a, b, c only
    assert s[1] == pytest.approx((2 / 3 + 0.5) / 2)  # y missing -> median rank
    assert s[4] == pytest.approx((1.0 + 0.5) / 2)  # t2 ranks on its own


def test_pick_takes_a_fraction_or_a_count():
    f = pd.DataFrame(
        {"decision_time": "t", "entity_id": list("abcdefghij"), "s": range(10)}
    )
    assert cohort_study.pick(f, "s", 0.2) == {"t": ["j", "i"]}
    assert cohort_study.pick(f, "s", 3, bottom=True) == {"t": ["a", "b", "c"]}


class _Market:
    calendar = calmod.TradingCalendar()

    def __init__(self, rets):
        self.rets = rets

    def daily_returns(self, codes):
        return self.rets, pd.Series(dtype="datetime64[ns]")

    def cost_bps(self, rows):
        return pd.Series(20.0, index=rows.index)  # 20 bp round trip


def test_study_portfolio_enters_after_formation_and_rolls_hold_cohorts():
    days = pd.bdate_range("2015-01-01", "2016-12-31")
    rets = pd.DataFrame({"a": 0.001, "b": 0.0}, index=days)
    cal = calmod.TradingCalendar()
    times = cal.decision_times(
        "2015-01-01", "2016-01-01", "BQE"
    )  # Mar 31 .. Dec 31 2015
    st = object.__new__(cohort_study.Study)
    st.market, st.cal, st.times = _Market(rets), cal, times
    st._rets = (rets, pd.Series(dtype="datetime64[ns]"))
    st.study = type("S", (), {"cfg": {"schedule": "BQE"}})()
    m = st.portfolio({t: ["a"] for t in times}, hold=2)
    assert m.index.min() == pd.Period(
        "2015-04"
    )  # bought at Apr 1's close, after formation
    assert m.loc[pd.Period("2015-05"), "cohorts"] == 1
    assert m.loc[pd.Period("2015-08"), "cohorts"] == 2
    assert (
        m.loc[pd.Period("2015-10"), "cohorts"] == 2
    )  # the first cohort sold as the third bought
    assert m["cohorts"].max() == 2  # never more than `hold` at once
    # 10 bp in, 10 bp out per cohort, net only
    assert m.loc[pd.Period("2015-04"), "gross"] - m.loc[
        pd.Period("2015-04"), "net"
    ] == pytest.approx(0.001, rel=0.05)
