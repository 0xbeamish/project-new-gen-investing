"""The trend book (pre-registration 7) on synthetic bars: no network."""

import numpy as np
import pandas as pd
import pytest

from engine import score
from engine.markets.crypto import trend
from tests.test_crypto import bars, market


def path(start, n, daily, vol=1e6):
    b = bars(start, n, vol=vol)
    b["close"] = 100 * np.exp(np.cumsum(np.full(n, daily)))
    return b


def three():
    return market(
        {
            "UP": path("2021-01-01", 400, 0.002, vol=3e6),
            "DOWN": path("2021-01-01", 400, -0.002, vol=2e6),
            "YOUNG": path("2021-10-01", 300, 0.002, vol=9e6),  # < 225 bars for a while
        }
    )


def test_exposure_counts_rising_lookbacks_and_needs_history():
    m = three()
    m.top = 5
    rows = trend.weekly_rows(m, "2021-11-01", "2022-01-01")
    first = rows[rows["decision_time"] == rows["decision_time"].min()].set_index("entity_id")
    assert first.loc["UP", "exposure"] == 1.0
    assert first.loc["DOWN", "exposure"] == 0.0
    assert "YOUNG" not in first.index  # 225 bars not reached
    assert (rows["n"] == 2).all()


def test_weights_use_nothing_after_the_sunday_and_the_week_runs_sunday_to_sunday():
    m = three()
    c = trend.closes(m)
    t = pd.Timestamp("2021-12-06", tz="UTC")
    rows = trend.weekly_rows(m, "2021-12-01", "2021-12-10", c=c)
    rows = rows[rows["decision_time"] == t].set_index("entity_id")
    sun = pd.Timestamp("2021-12-05")
    want = c.loc[sun + pd.Timedelta(days=7), "UP"] / c.loc[sun, "UP"] - 1
    assert rows.loc["UP", "ret"] == pytest.approx(want)
    assert (rows["decision_time"] == t).all()
    later = c.copy()
    later.loc[sun + pd.Timedelta(days=1) :, "UP"] *= 0.01  # a crash after the decision
    again = trend.weekly_rows(m, "2021-12-01", "2021-12-10", c=later)
    again = again[again["decision_time"] == t].set_index("entity_id")
    for col in ("exposure", "vol", "scale"):
        assert again.loc["UP", col] == pytest.approx(rows.loc["UP", col])
    assert again.loc["UP", "ret"] < -0.9


def test_a_coin_that_stops_exits_at_its_last_close():
    dead = path("2021-01-01", 340, 0.002)
    m = market({"UP": path("2021-01-01", 400, 0.002), "DEAD": dead})
    c = trend.closes(m)
    last = dead["time"].iloc[-1]  # 2021-12-06, mid-week
    rows = trend.weekly_rows(m, "2021-12-01", "2021-12-10", c=c)
    rows = rows[rows["decision_time"] == pd.Timestamp("2021-12-06", tz="UTC")]
    rows = rows.set_index("entity_id")
    sun = pd.Timestamp("2021-12-05")
    assert rows.loc["DEAD", "ret"] == pytest.approx(c.loc[last, "DEAD"] / c.loc[sun, "DEAD"] - 1)


def test_book_weights_costs_turnover_and_cash():
    t1, t2 = pd.Timestamp("2022-01-03", tz="UTC"), pd.Timestamp("2022-01-10", tz="UTC")
    rows = pd.DataFrame(
        {
            "decision_time": [t1, t1, t2, t2],
            "entity_id": ["A", "B", "A", "B"],
            "exposure": [1.0, 0.0, 1.0, 1.0],
            "scale": [0.5, 1.0, 0.5, 1.0],
            "ret": [0.10, 0.50, 0.0, 0.0],
            "n": [2, 2, 2, 2],
        }
    )
    side = pd.Series(0.001, index=rows.index)
    b = trend.book(rows, side)
    # week 1: A = 1 x 0.5 / 2 = 0.25, B = 0; gross = 0.025; buying 0.25 costs 0.00025
    assert b.loc[t1, "gross"] == pytest.approx(0.025)
    assert b.loc[t1, "cost"] == pytest.approx(0.00025)
    assert b.loc[t1, "gross_weight"] == pytest.approx(0.25)
    # week 2: A drifted to 0.275 / 1.025; target 0.25; B 0 -> 0.5
    drift = 0.275 / 1.025
    assert b.loc[t2, "turnover"] == pytest.approx(abs(0.25 - drift) + 0.5)
    fixed = trend.book(rows, side, divisor="fixed")
    assert fixed.loc[t1, "gross"] == pytest.approx(0.025 * 2 / trend.TOP)


def test_robust_alpha_t_is_the_smaller_of_hc3_and_newey_west():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, 500)
    y = 0.002 + 0.5 * x + rng.normal(0, 0.02, 500)
    fit = score.ols_alpha_robust(y, x)
    assert fit["alpha"] == pytest.approx(0.002, abs=0.002)
    assert fit["t_alpha"] == min(fit["t_hc3"], fit["t_nw"], key=abs)
    assert fit["t_nw"] == pytest.approx(fit["t_hc3"], rel=0.15)  # iid: about the same
    smooth = np.convolve(rng.normal(0.002, 0.02, 520), np.ones(20) / 20, "valid")[:500]
    s = score.ols_alpha_robust(smooth + 0.5 * x, x)
    assert abs(s["t_nw"]) < 0.6 * abs(s["t_hc3"])
