import pandas as pd
import pytest

from engine import scoring


def _frame(t, scores, rets, cost=0.01):
    return pd.DataFrame(
        {
            "entity_id": [f"s{i}" for i in range(len(scores))],
            "decision_time": pd.Timestamp(t, tz="UTC"),
            "group": "tech",
            "score": scores,
            "fwd_return": rets,
            "rt_cost": cost,
        }
    )


def test_band_holds_until_out_of_top_30_and_charges_half_trips():
    # 4 names: percentiles 0.25 / 0.5 / 0.75 / 1.0 by score
    m1 = _frame("2015-01-31", [4, 3, 2, 1], [0.10, 0.0, 0.0, 0.0])  # s0 top: bought
    m2 = _frame(
        "2015-02-28", [3, 4, 2, 1], [0.00, 0.10, 0.0, 0.0]
    )  # s0 at 75th pct: kept; s1 bought
    m3 = _frame(
        "2015-03-31", [2, 4, 3, 1], [0.00, 0.00, 0.0, 0.0]
    )  # s0 at 50th pct: sold
    r = scoring.simulate(pd.concat([m1, m2, m3]), scoring.Band(0.9, 0.7))
    g, n = r["gross"], r["net"]
    assert g.iloc[0] == pytest.approx(0.10 - 0.025)
    assert n.iloc[0] == pytest.approx(
        g.iloc[0] - 0.005
    )  # one buy, half a round trip, whole book
    assert g.iloc[1] == pytest.approx(0.05 - 0.025)  # s0 + s1 held
    assert n.iloc[1] == pytest.approx(
        g.iloc[1] - 0.005 / 2
    )  # one buy over a book of two
    assert n.iloc[2] == pytest.approx(g.iloc[2] - 0.005)  # one sell, book of one
    assert r["avg_holding_periods"] == pytest.approx(2.0)  # s0 held Jan + Feb


def test_topk_without_a_band_turns_over_every_change():
    m1 = _frame("2015-01-31", [4, 3, 2, 1], [0.0] * 4)
    m2 = _frame("2015-02-28", [3, 4, 2, 1], [0.0] * 4)
    r = scoring.simulate(pd.concat([m1, m2]), scoring.TopK(0.9))
    assert r["turnover"].tolist() == pytest.approx(
        [0.5, 1.0]
    )  # month 2: sell s0, buy s1


def test_missing_name_earns_the_universe_mean():
    m1 = _frame("2015-01-31", [4, 3, 2, 1], [0.0] * 4)
    m2 = _frame(
        "2015-02-28", [3, 2, 1], [0.03, 0.0, 0.0]
    )  # s3 left the universe; s0 still top
    r = scoring.simulate(pd.concat([m1, m2]), scoring.Band(0.9, 0.7))
    assert r["gross"].iloc[1] == pytest.approx(0.03 - 0.01)
