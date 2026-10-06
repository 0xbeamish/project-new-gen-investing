"""The scorer: t-statistics, rank IC, spreads, portfolio rules, overlapping cohorts, and the
planted-signal / pure-noise checks every component must pass."""

import numpy as np
import pandas as pd
import pytest

from engine import run, score


def _period(t, scores, rets, group="g", cost=0.01):
    return pd.DataFrame(
        {
            "entity_id": [f"s{i}" for i in range(len(scores))],
            "decision_time": pd.Timestamp(t, tz="UTC"),
            "group": group,
            "score": scores,
            "fwd_return": rets,
            "rt_cost": cost,
        }
    )


# ---------------------------------------------------------------- statistics and books
def test_per_period_t_and_overlap_deflation():
    s = pd.Series([0.01, 0.02, 0.03, 0.02])
    t = s.mean() / (s.std(ddof=1) / 2)
    assert score.per_period_t(s) == pytest.approx(t)
    assert score.per_period_t(s, overlap=4) == pytest.approx(t / 2)
    assert np.isnan(score.per_period_t(pd.Series([0.1, 0.2])))
    assert score.overlap(21, 21) == 1 and score.overlap(30, 7) == 5


def test_rank_ic_is_one_for_a_perfect_ranking_and_uses_group_adjusted_returns():
    f = pd.concat(
        [
            # adjusted: a = -.03 -.01 .01 .03, b = -.06 -.02 .02 .06; scores follow that order
            _period("2020-01-31", [2, 4, 5, 7], [0.00, 0.02, 0.04, 0.06], "a"),
            _period("2020-01-31", [1, 3, 6, 8], [0.50, 0.54, 0.58, 0.62], "b").assign(
                entity_id=lambda d: "b" + d["entity_id"]
            ),
        ]
    )
    assert score.rank_ic(f).iloc[0] == pytest.approx(1.0)
    raw = f["score"].corr(f["fwd_return"], method="spearman")
    assert raw < 0.8  # on raw returns every b would beat every a


def test_quantile_spreads_gross_net_and_turnover():
    m1 = _period("2020-01-31", list(range(10)), [0.0] * 9 + [0.05])
    m2 = _period("2020-02-29", list(range(10)), [0.0] * 9 + [0.05])
    r = score.quantile_spreads(pd.concat([m1, m2]), q=0.1, min_names=10)
    assert r["gross"].tolist() == pytest.approx([0.05, 0.05])
    assert r["net"].tolist() == pytest.approx([0.05 - 0.02, 0.05])  # both legs new in month 1 only
    assert r["turnover"].tolist() == pytest.approx([1.0, 0.0])


def test_band_holds_until_out_of_top_30_and_charges_half_trips():
    # 4 names: percentiles 0.25 / 0.5 / 0.75 / 1.0 by score
    m1 = _period("2015-01-31", [4, 3, 2, 1], [0.10, 0.0, 0.0, 0.0])  # s0 top: bought
    m2 = _period("2015-02-28", [3, 4, 2, 1], [0.00, 0.10, 0.0, 0.0])  # s0 at 75th: kept; s1 bought
    m3 = _period("2015-03-31", [2, 4, 3, 1], [0.00, 0.00, 0.0, 0.0])  # s0 at 50th: sold
    r = score.simulate_portfolio(pd.concat([m1, m2, m3]), score.Band(0.9, 0.7))
    g, n = r["gross"], r["net"]
    assert g.iloc[0] == pytest.approx(0.10 - 0.025)
    assert n.iloc[0] == pytest.approx(g.iloc[0] - 0.005)  # one buy, half a round trip, whole book
    assert g.iloc[1] == pytest.approx(0.05 - 0.025)  # s0 + s1 held
    assert n.iloc[1] == pytest.approx(g.iloc[1] - 0.005 / 2)  # one buy over a book of two
    assert n.iloc[2] == pytest.approx(g.iloc[2] - 0.005)  # one sell, book of one
    assert r["avg_holding_periods"] == pytest.approx(2.0)  # s0 held Jan + Feb


def test_topk_without_a_band_turns_over_every_change():
    m1 = _period("2015-01-31", [4, 3, 2, 1], [0.0] * 4)
    m2 = _period("2015-02-28", [3, 4, 2, 1], [0.0] * 4)
    r = score.simulate_portfolio(pd.concat([m1, m2]), score.TopK(0.9))
    assert r["turnover"].tolist() == pytest.approx([0.5, 1.0])  # month 2: sell s0, buy s1


def test_missing_name_earns_the_universe_mean():
    m1 = _period("2015-01-31", [4, 3, 2, 1], [0.0] * 4)
    m2 = _period("2015-02-28", [3, 2, 1], [0.03, 0.0, 0.0])  # s3 left the universe; s0 still top
    r = score.simulate_portfolio(pd.concat([m1, m2]), score.Band(0.9, 0.7))
    assert r["gross"].iloc[1] == pytest.approx(0.03 - 0.01)


# ---------------------------------------------------------------- planted signal and pure noise
def test_known_positive_planted_signal_is_found(study, tuning_rows):
    scored, w = run.walk_forward(study).run(
        tuning_rows, ["planted", "noise_1"], study.market.calendar
    )
    ev = run.evaluate(study, scored)
    assert ev["ic"] > 0.15 and ev["ic_t"] > 5
    assert ev["spread"]["gross_t"] > 3
    assert (w["planted"] > w["noise_1"].abs()).all()


def test_known_negative_pure_noise_is_not_found(study, tuning_rows):
    scored, _ = run.walk_forward(study).run(
        tuning_rows, ["noise_1", "noise_2", "noise_3"], study.market.calendar
    )
    ev = run.evaluate(study, scored)
    assert abs(ev["ic_t"]) < 1.96
    assert abs(ev["ic"]) < 0.05


# ---------------------------------------------------------------- overlapping cohorts
def _rets(cols: dict, start="2015-01-01", n=10) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=n)
    return pd.DataFrame({k: v if np.ndim(v) else [v] * n for k, v in cols.items()}, index=days)


def test_buy_and_hold_drifts_from_equal_weight():
    r = _rets({"a": 0.10, "b": 0.0}, n=3)
    p = score.cohort_returns(r, ["a", "b"], r.index[0])
    # day 1: (1.1 + 1) / 2 - 1 = 5%; day 2: a is now 1.1 / 2.1 of the cohort
    assert p["gross"].iloc[0] == pytest.approx(0.05)
    assert p["gross"].iloc[1] == pytest.approx(0.10 * 1.1 / 2.1)
    assert list(p.index) == list(r.index[1:])  # bought at entry_day's close: returns start after


def test_entry_and_exit_costs_are_half_round_trips():
    r = _rets({"a": 0.0, "b": 0.0}, n=4)
    half = pd.Series({"a": 0.002, "b": 0.004})
    p = score.cohort_returns(r, ["a", "b"], r.index[0], r.index[3], half, half)
    assert p["net"].iloc[0] == pytest.approx(-0.003)  # mean entry cost
    assert p["net"].iloc[1] == pytest.approx(0.0)
    assert p["net"].iloc[-1] == pytest.approx(-0.003, rel=1e-3)  # exit, value-weighted
    assert p["gross"].abs().max() == 0.0
    assert len(p) == 3  # held entry < d <= exit


def test_no_exit_cost_when_the_exit_is_past_the_data():
    r = _rets({"a": 0.0}, n=3)
    p = score.cohort_returns(
        r, ["a"], r.index[0], pd.Timestamp("2030-01-01"), None, pd.Series({"a": 0.01})
    )
    assert p["net"].abs().max() == 0.0


def test_a_delisted_member_is_sold_and_its_value_spread_over_the_rest():
    a = [0.0, -0.5, np.nan, np.nan]  # day 2 carries the delisting return; then no bars
    r = _rets({"a": a, "b": [0.0, 0.0, 0.10, 0.10]}, n=4)
    ended = pd.Series({"a": r.index[1]})
    p = score.cohort_returns(r, ["a", "b"], r.index[0], ended=ended)
    assert p["gross"].iloc[0] == pytest.approx(-0.25)  # a's last day counts
    assert p["gross"].iloc[1] == pytest.approx(0.10)  # then the cohort is all b
    assert p["gross"].iloc[2] == pytest.approx(0.10)


def test_a_member_dead_before_entry_is_never_bought():
    r = _rets({"a": [0.0] + [np.nan] * 4, "b": 0.02}, n=5)
    p = score.cohort_returns(r, ["a", "b"], r.index[1], ended=pd.Series({"a": r.index[0]}))
    assert p["gross"].iloc[0] == pytest.approx(0.02)


def test_a_missing_bar_earns_zero_but_keeps_its_weight():
    r = _rets({"a": [0.0, np.nan, 0.2], "b": [0.0, 0.0, 0.0]}, n=3)
    p = score.cohort_returns(r, ["a", "b"], r.index[0])
    assert p["gross"].iloc[0] == 0.0 and p["gross"].iloc[1] == pytest.approx(0.1)


def test_overlapping_averages_the_cohorts_held_each_day_and_counts_them():
    r = _rets({"a": 0.01, "b": 0.03}, n=6)
    c1 = score.cohort_returns(r, ["a"], r.index[0], r.index[3])
    c2 = score.cohort_returns(r, ["b"], r.index[2])
    d = score.overlapping_cohorts([c1, c2])
    assert d.loc[r.index[1], "gross"] == pytest.approx(0.01)
    assert d.loc[r.index[3], "gross"] == pytest.approx(0.02)  # both held
    assert d.loc[r.index[4], "gross"] == pytest.approx(0.03)  # c1 sold at index 3's close
    assert d["cohorts"].tolist() == [1, 1, 2, 1, 1]


def test_monthly_compounds_daily_returns():
    days = pd.bdate_range("2015-01-26", "2015-02-06")
    d = pd.DataFrame({"gross": 0.01, "net": 0.01, "cohorts": 2}, index=days)
    m = score.monthly_returns(d)
    assert m.loc[pd.Period("2015-01"), "gross"] == pytest.approx(1.01**5 - 1)
    assert m["cohorts"].tolist() == [2, 2]


def test_newey_west_t_matches_iid_without_lags_and_shrinks_with_autocorrelation():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.01, 0.02, 200))
    n = len(x)
    assert score.newey_west_t(x, 0) == pytest.approx(score.per_period_t(x) * np.sqrt(n / (n - 1)))
    smooth = x.rolling(12).mean().dropna()  # overlapping windows: strong positive autocorrelation
    assert score.newey_west_t(smooth, 12) < 0.6 * score.per_period_t(smooth)


# ---------------------------------------------------------------- minimum detectable effect
def test_mde_reproduces_the_large_cap_hand_calculation():
    """docs/FINDINGS.md, us_largecap: 99 months, standard error about 0.11%/month, bar 3.43 ->
    detects about 0.4%/month half the time and about 0.5%/month 80% of the time."""
    se, n, bar = 0.0011, 99, 3.434
    sd = se * np.sqrt(n)  # per-period noise that gives that standard error
    assert score.mde(sd, n, bar, 0.5) == pytest.approx(0.004, abs=0.0005)
    assert score.mde(sd, n, bar, 0.8) == pytest.approx(0.005, abs=0.0005)
    assert score.mde(sd, n, bar, 0.5) == pytest.approx(bar * se)  # 50% power: bar x SE
    assert np.isnan(score.mde(sd, 2, bar, 0.8))


def test_mde_is_what_the_test_detects_with_that_power():
    """Simulate the t-test at a true effect equal to the MDE: it clears the bar ~80% / ~50% of the
    time."""
    rng = np.random.default_rng(0)
    n, sd, bar = 60, 0.1, 2.5
    for power in (0.5, 0.8):
        effect = score.mde(sd, n, bar, power)
        hits = [
            score.per_period_t(pd.Series(rng.normal(effect, sd, n))) >= bar for _ in range(4000)
        ]
        assert np.mean(hits) == pytest.approx(power, abs=0.04)
