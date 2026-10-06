"""The large-cap universe rule (universe.large), offline: synthetic snapshots, SEC lookups stubbed."""

import numpy as np
import pandas as pd
import pytest

from engine.markets.us_smallcap import universe

Q = [pd.Timestamp(d) for d in ("2014-03-31", "2014-06-30", "2014-09-30")]


def _snap(code, cik, closes, shares, dvol=1e8):
    return pd.DataFrame(
        {
            "as_of": Q,
            "cik": cik,
            "code": code,
            "name": code,
            "close": closes,
            "mcap": np.array(closes) * np.array(shares),
            "dollar_vol": dvol,
        }
    )


@pytest.fixture
def stubs(monkeypatch):
    forms = {
        6: pd.DataFrame({"form": ["20-F"], "filed": [pd.Timestamp("2013-04-01")]}),
        7: pd.DataFrame({"form": ["6-K"], "filed": [pd.Timestamp("2014-01-05")]}),
    }
    floats = {  # (measured, filed, USD): about the true value for most names
        1: 50e9,
        2: 20e9,
        3: 10e9,
        4: 5e9,
        5: 30e9,
    }
    monkeypatch.setattr(
        universe,
        "annual_forms",
        lambda cik: forms.get(
            cik, pd.DataFrame({"form": ["10-K"], "filed": [pd.Timestamp("2013-02-01")]})
        ),
    )
    monkeypatch.setattr(
        universe,
        "public_floats",
        lambda cik: (
            pd.DataFrame(
                {
                    "end": [pd.Timestamp("2013-06-28")],
                    "filed": [pd.Timestamp("2013-02-01")],
                    "val": [floats[cik]],
                }
            )
            if cik in floats
            else pd.DataFrame(columns=["end", "filed", "val"])
        ),
    )
    flat = pd.DataFrame(
        {"adj_close": 1.0}, index=pd.bdate_range("2013-01-01", "2014-12-31")
    )
    monkeypatch.setattr(universe, "load_prices", lambda code: flat)


def _snaps():
    s = pd.concat(
        [
            _snap("BIG", 1, [50.0] * 3, [1e9] * 3),  # $50B, consistent with its float
            _snap(
                "UNIT", 2, [20.0] * 3, [1e9, 1e12, 1e9]
            ),  # a 1,000x share error mid-year
            _snap(
                "JUNK", 3, [5000.0] * 3, [1e9] * 3
            ),  # raw close 500x: float says $10B
            _snap("MID", 4, [6.0] * 3, [1e9] * 3),  # $6B
            _snap("C-WS-A", 5, [30.0] * 3, [1e9] * 3),  # a warrant code
            _snap("ADR", 6, [100.0] * 3, [1e9] * 3),  # a 20-F filer
            _snap(
                "NEWF", 7, [100.0] * 3, [1e9] * 3
            ),  # a 6-K filer with no annual report yet
            _snap(
                "IPO", 8, [40.0] * 3, [1e9] * 3, dvol=2e8
            ),  # no float yet, trades 0.5% a day
            _snap(
                "THIN", 9, [90.0] * 3, [1e9] * 3, dvol=1e6
            ),  # no float, trades 0.001% a day
        ],
        ignore_index=True,
    )
    s["mcap_rank"] = s.groupby("as_of")["mcap"].rank(ascending=False)
    return s


def test_large_ranks_a_checked_value(stubs):
    out = universe.large(_snaps(), n=4)
    q2 = out[out["as_of"] == Q[2]].set_index("code")
    # ADR and NEWF (foreign), C-WS-A (warrant), THIN (unchecked, no trading) are out;
    # UNIT's share error is smoothed by the median; JUNK ranks at its float, below the cut
    assert list(q2.index) == ["BIG", "IPO", "UNIT", "JUNK"]
    assert q2.loc["UNIT", "mcap"] == pytest.approx(20e9)
    assert (
        q2.loc["UNIT", "value_source"] == "close_x_shares"
    )  # fixed by the median, not the float
    assert q2.loc["JUNK", "value_source"] == "float" and q2.loc["JUNK", "mcap"] == 10e9
    assert q2.loc["IPO", "value_source"] == "close_x_shares_unchecked"
    assert q2.loc["BIG", "value_source"] == "close_x_shares"
    assert (out.groupby("as_of").size() == 4).all()


def test_checked_value_rules():
    assert universe.checked_value(40e9, 20e9, 0.01, 50.0) == (40e9, "close_x_shares")
    assert universe.checked_value(400e9, 20e9, 0.01, 50.0) == (
        20e9,
        "float",
    )  # 20x float
    assert universe.checked_value(5e9, 20e9, 0.01, 50.0) == (
        20e9,
        "float",
    )  # below its own float
    assert universe.checked_value(2e12, np.nan, 0.01, 50.0)[1] == "dropped"  # > $1.5T
    assert (
        universe.checked_value(9e9, np.nan, 0.01, 900.0)[1] == "dropped"
    )  # first-year, $900
    assert universe.checked_value(9e9, np.nan, 1e-4, 50.0)[1] == "dropped"  # no trading


def test_foreign_only_until_a_10k():
    f = pd.DataFrame(
        {
            "form": ["6-K", "10-K"],
            "filed": [pd.Timestamp("2012-01-01"), pd.Timestamp("2013-01-01")],
        }
    )
    assert universe.is_foreign(f, pd.Timestamp("2012-06-30"))
    assert not universe.is_foreign(f, pd.Timestamp("2013-06-30"))
    assert not universe.is_foreign(f.iloc[:0], pd.Timestamp("2013-06-30"))


def _facts(rnd_late_q: str | None = None) -> dict:
    """Synthetic companyfacts: 9 quarters of 3-month revenue / operating income / R&D."""
    ends = pd.date_range("2010-03-31", periods=9, freq="QE")
    rev = [100, 100, 100, 100, 110, 120, 130, 140, 165]  # growth 10% .. 65%
    rows = lambda vals, late=None: [
        {
            "start": str((e - pd.offsets.QuarterBegin(startingMonth=1)).date()),
            "end": str(e.date()),
            "val": v,
            "form": "10-Q",
            "filed": str(
                (e + pd.Timedelta(days=400 if late == str(e.date()) else 35)).date()
            ),
        }
        for e, v in zip(ends, vals)
    ]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": rows(rev)}},
                "OperatingIncomeLoss": {
                    "units": {"USD": rows([10] * 4 + [11, 12, 13, 14, 33])}
                },
                "ResearchAndDevelopmentExpense": {
                    "units": {"USD": rows([5] * 9, rnd_late_q)}
                },
            }
        }
    }


def test_extended_quarterly_inputs_and_their_dates(monkeypatch):
    from engine.markets.us_smallcap import fundamentals, sec

    monkeypatch.setattr(sec, "_get_json", lambda url, name: _facts("2012-03-31"))
    q = fundamentals.quarterly_features("X", 1, extended=True).set_index("filed")
    last = q.iloc[-1]  # quarter ending 2012-03-31
    assert last["q_rev_growth_yoy"] == pytest.approx(165 / 110 - 1)
    assert last["q_rev_growth_yoy_chg4"] == pytest.approx(
        (165 / 110 - 1) - (110 / 100 - 1)
    )
    assert last["q_op_margin_chg_yoy"] == pytest.approx(33 / 165 - 11 / 110)
    assert np.isnan(
        last["q_rnd_intensity"]
    )  # its R&D was first filed long after the quarter
    assert q.iloc[-2]["q_rnd_intensity_chg_yoy"] == pytest.approx(5 / 140 - 5 / 100)
    base = fundamentals.quarterly_features("X", 1)
    assert not set(fundamentals.EXTENDED) & set(
        base.columns
    )  # default output unchanged
    cut = fundamentals.quarterly_features("X", 1, True, filed_until="2011-12-31")
    assert pd.to_datetime(cut["filed"]).max() <= pd.Timestamp("2011-12-31")


def test_data_end_hides_later_bars_and_only_earlier_stops_are_delistings(
    tmp_path, monkeypatch
):
    from engine.markets import us_largecap
    from engine.markets.us_smallcap import prices

    uni = tmp_path / "u.csv"
    pd.DataFrame(
        {
            "as_of": ["2018-12-31"],
            "cik": [1],
            "code": ["A"],
            "name": ["A"],
            "mcap": [1e10],
            "sector": ["tech"],
        }
    ).to_csv(uni, index=False)
    bars = {
        "LIVE": pd.DataFrame(
            {"adj_close": 1.0}, index=pd.bdate_range("2019-01-01", "2021-06-30")
        ),
        "GONE": pd.DataFrame(
            {"adj_close": 1.0}, index=pd.bdate_range("2019-01-01", "2019-06-28")
        ),
    }
    monkeypatch.setattr(prices, "load_prices", lambda code: bars[code])
    mkt, sources = us_largecap.build(
        {"universe": {"file": str(uni)}, "data_end": "2019-12-31"}
    )
    live = mkt.bars("LIVE")
    assert mkt.calendar.local_date(live.index[-1:]).iloc[0] == pd.Timestamp(
        "2019-12-31"
    )
    assert not mkt._ended(live) and mkt._ended(mkt.bars("GONE"))
    assert {"price", "sec_annual", "sec_quarterly"} <= set(
        sources
    )  # the small-cap sources, reused
