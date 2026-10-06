"""Market plug-ins: the label convention, the demo market, the CSV market, and the US stock
plug-in's helpers offline (no cache, no network)."""

import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from engine import data, panel, run
from engine.market import forward_returns
from engine.markets import csv as csv_market
from engine.markets import us_stocks
from engine.markets.us_stocks import filings, prices, sec, universe


def test_forward_returns_enter_after_the_decision_and_handle_delisting():
    close = pd.date_range("2020-01-01 21:00", periods=10, freq="D", tz="UTC")
    px = np.arange(100.0, 110.0)
    T = pd.DatetimeIndex([close[2] + pd.Timedelta("30min")])
    lab = forward_returns(close, px, T, 3, ended=False)
    assert lab["entry_time"].iloc[0] == close[3] and lab["label_end"].iloc[0] == close[6]
    assert lab["fwd_return"].iloc[0] == pytest.approx(106 / 103 - 1)
    late = pd.DatetimeIndex([close[7] + pd.Timedelta("30min")])
    assert forward_returns(close, px, late, 3, ended=False)["fwd_return"].isna().all()  # unfinished
    dead = forward_returns(close, px, late, 3, ended=True, delist_adjustment=-0.3)
    assert dead["fwd_return"].iloc[0] == pytest.approx((109 / 108) * 0.7 - 1)


# ---------------------------------------------------------------- the demo market
def test_demo_planted_inputs_come_from_their_own_stream(tmp_path):
    """Turning the planted inputs on must not move a single text-market draw (the demo's numbers
    depend on it)."""
    from engine.markets import demo

    cfg = run.read_config("demo")
    base = demo.Demo(cfg)
    cfg["demo"] = cfg["demo"] | {"planted": {"beta": 0.0, "noise_features": 2}}
    more = demo.Demo(cfg)
    assert np.array_equal(base.prices, more.prices) and base.docs.equals(more.docs)
    assert more.noise_values.shape[0] == 2 and base.z is None


# ---------------------------------------------------------------- the CSV market
def _csv_study(tmp_path):
    """markets/csv_example.yaml with a temporary registry and cache."""
    cfg = run.read_config("csv_example")
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["registry"] = {
        "file": str(tmp_path / "registry.csv"),
        "holdout_unlock_log": str(tmp_path / "unlocks.csv"),
    }
    return run.study_from_config("csv_example", cfg)


@pytest.fixture(scope="module")
def csv_rows(tmp_path_factory):
    st = _csv_study(tmp_path_factory.mktemp("csv"))
    return st, panel.model_rows(st, run.build_panel(st, "tuning"))


def test_csv_example_finds_its_planted_signals_and_not_its_noise(csv_rows, tmp_path):
    st, rows = csv_rows
    st.registry.path = tmp_path / "registry.csv"
    flow = run.test_candidate(st, rows, [], run.Candidate("flow"))
    social = run.test_candidate(st, rows, [], run.Candidate("social"))
    assert flow["t_tune"] >= flow["bar_tune"] and flow["kept"]
    assert social["t_tune"] < social["bar_tune"] and not social["check_used"]
    # the posts are read by the free keyword reader: an upgrade helps, an exploit hurts
    wf = run.walk_forward(st)
    for f in ("doc_upgrade_p", "doc_exploit_p"):
        assert run.judge(st, rows, [], run.Candidate(f), wf)["t_tune"] > 3


def test_csv_market_is_point_in_time_on_24_7_bars(csv_rows):
    st, rows = csv_rows
    cal = st.market.calendar
    assert cal.kind == "continuous"
    t = rows["decision_time"]
    assert (t.dt.dayofweek == 0).all() and (t.dt.hour == 0).all()  # Monday 00:00 UTC
    assert (rows["entry_time"] > t).all()  # entered at the first close after the decision
    # every decision has the week's flow (published Sunday 12:00) and nothing older than 6 days
    assert rows["flow"].notna().mean() > 0.99
    # dead tokens leave the universe; their last window is a delisting
    late = rows["decision_time"] > pd.Timestamp("2022-12-01", tz="UTC")
    assert rows.loc[late, "entity_id"].nunique() == 27
    assert set(rows.loc[rows["delisted"], "entity_id"]) == {"tok28", "tok29", "tok30"}


def test_csv_timestamps_never_guess_earlier():
    cal = data.make_calendar({"kind": "trading", "tz": "America/New_York"})
    got = csv_market.parse_available_at(
        pd.Series(["2021-03-05T12:00:00Z", "2021-03-05", "2021-03-05 12:00"]), cal
    )
    assert got[0] == pd.Timestamp("2021-03-05 12:00", tz="UTC")
    assert got[1] == pd.Timestamp("2021-03-06", tz="America/New_York")  # date only: next midnight
    assert got[2] == pd.Timestamp("2021-03-05 12:00", tz="America/New_York")  # naive: local time
    with pytest.raises(ValueError):
        csv_market.parse_available_at(pd.Series(["soon"]), cal)


def test_csv_market_runs_from_the_cli_on_your_own_files(tmp_path, monkeypatch):
    """Copy the example, as a user would, and run `engine test` on it."""
    for f in ("prices.csv", "signals.csv"):
        shutil.copy(run.path(f"examples/csv/{f}"), tmp_path / f)
    cfg = run.read_config("csv_example")
    cfg["csv"] = {"prices": str(tmp_path / "prices.csv"), "signals": str(tmp_path / "signals.csv")}
    cfg["sources"] = {"signals": {"max_age_days": 6}}
    cfg["features"] = {"baseline": [], "all": ["flow", "social"]}
    cfg["registry"] = {
        "file": str(tmp_path / "reg.csv"),
        "holdout_unlock_log": str(tmp_path / "u.csv"),
    }
    cfg["cache_dir"] = str(tmp_path / "cache")
    import yaml

    (tmp_path / "mine.yaml").write_text(yaml.safe_dump(cfg))
    from engine import cli

    monkeypatch.chdir(tmp_path)
    cli.main(
        ["test", "--market", "mine", "--config", str(tmp_path / "mine.yaml"), "--feature", "flow"]
    )
    reg = pd.read_csv(tmp_path / "reg.csv")
    assert reg["name"].tolist() == ["flow"] and bool(reg["kept"].iloc[0])


# ---------------------------------------------------------------- US stocks, offline
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
    flat = pd.DataFrame({"adj_close": 1.0}, index=pd.bdate_range("2013-01-01", "2014-12-31"))
    monkeypatch.setattr(universe, "load_prices", lambda code: flat)


def _snaps():
    s = pd.concat(
        [
            _snap("BIG", 1, [50.0] * 3, [1e9] * 3),  # $50B, consistent with its float
            _snap("UNIT", 2, [20.0] * 3, [1e9, 1e12, 1e9]),  # a 1,000x share error mid-year
            _snap("JUNK", 3, [5000.0] * 3, [1e9] * 3),  # raw close 500x: float says $10B
            _snap("MID", 4, [6.0] * 3, [1e9] * 3),  # $6B
            _snap("C-WS-A", 5, [30.0] * 3, [1e9] * 3),  # a warrant code
            _snap("ADR", 6, [100.0] * 3, [1e9] * 3),  # a 20-F filer
            _snap("NEWF", 7, [100.0] * 3, [1e9] * 3),  # a 6-K filer with no annual report yet
            _snap("IPO", 8, [40.0] * 3, [1e9] * 3, dvol=2e8),  # no float yet, trades 0.5% a day
            _snap("THIN", 9, [90.0] * 3, [1e9] * 3, dvol=1e6),  # no float, trades 0.001% a day
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
    assert q2.loc["UNIT", "value_source"] == "close_x_shares"  # fixed by the median, not the float
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
    assert universe.checked_value(9e9, np.nan, 0.01, 900.0)[1] == "dropped"  # first-year, $900
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
            "filed": str((e + pd.Timedelta(days=400 if late == str(e.date()) else 35)).date()),
        }
        for e, v in zip(ends, vals)
    ]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": rows(rev)}},
                "OperatingIncomeLoss": {"units": {"USD": rows([10] * 4 + [11, 12, 13, 14, 33])}},
                "ResearchAndDevelopmentExpense": {"units": {"USD": rows([5] * 9, rnd_late_q)}},
            }
        }
    }


def test_extended_quarterly_inputs_and_their_dates(monkeypatch):
    monkeypatch.setattr(sec, "_get_json", lambda url, name: _facts("2012-03-31"))
    q = sec.quarterly_features("X", 1, extended=True).set_index("filed")
    last = q.iloc[-1]  # quarter ending 2012-03-31
    assert last["q_rev_growth_yoy"] == pytest.approx(165 / 110 - 1)
    assert last["q_rev_growth_yoy_chg4"] == pytest.approx((165 / 110 - 1) - (110 / 100 - 1))
    assert last["q_op_margin_chg_yoy"] == pytest.approx(33 / 165 - 11 / 110)
    assert np.isnan(last["q_rnd_intensity"])  # its R&D was first filed long after the quarter
    assert q.iloc[-2]["q_rnd_intensity_chg_yoy"] == pytest.approx(5 / 140 - 5 / 100)
    base = sec.quarterly_features("X", 1)
    assert not set(sec.EXTENDED) & set(base.columns)  # default output unchanged
    cut = sec.quarterly_features("X", 1, True, filed_until="2011-12-31")
    assert pd.to_datetime(cut["filed"]).max() <= pd.Timestamp("2011-12-31")


def test_data_end_hides_later_bars_and_only_earlier_stops_are_delistings(tmp_path, monkeypatch):
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
        "LIVE": pd.DataFrame({"adj_close": 1.0}, index=pd.bdate_range("2019-01-01", "2021-06-30")),
        "GONE": pd.DataFrame({"adj_close": 1.0}, index=pd.bdate_range("2019-01-01", "2019-06-28")),
    }
    monkeypatch.setattr(prices, "load_prices", lambda code: bars[code])
    mkt, sources = us_stocks.build({"universe": {"file": str(uni)}, "data_end": "2019-12-31"})
    live = mkt.bars("LIVE")
    assert mkt.calendar.local_date(live.index[-1:]).iloc[0] == pd.Timestamp("2019-12-31")
    assert not mkt._ended(live) and mkt._ended(mkt.bars("GONE"))
    assert {"price", "sec_annual", "sec_quarterly"} <= set(sources)  # the small-cap sources, reused


def test_spread_estimators_on_known_bars():
    h = np.log(np.array([10.2, 10.3, 10.1, 10.4]))
    lo = np.log(np.array([9.8, 9.9, 9.7, 10.0]))
    c = np.log(np.array([10.0, 10.1, 9.9, 10.2]))
    assert prices.abdi_ranaldo(h, lo, c) >= 0
    assert 0 <= prices.corwin_schultz(h, lo) < 0.1
    flat = np.log(np.full(5, 10.0))
    assert prices.abdi_ranaldo(flat, flat, flat) == 0.0  # no range, no spread
    assert np.isnan(prices.abdi_ranaldo(h[:2], lo[:2], c[:2]))  # too few bars


def test_quarter_snapshot_uses_the_last_bar_and_rejects_stale_ones():
    idx = pd.bdate_range("2015-10-01", "2015-12-31")
    px = pd.DataFrame({"close": np.arange(len(idx), dtype=float) + 1, "volume": 1000.0}, index=idx)
    close, dvol = prices.quarter_snapshot(px, pd.Timestamp("2015-12-31"))
    assert close == px["close"].iloc[-1] and dvol > 0
    assert prices.quarter_snapshot(px, pd.Timestamp("2016-01-31")) is None  # > 7 days old


def test_eight_k_items_and_names():
    assert sec.EIGHT_K_ITEMS["2.02"] == "results" and sec.EIGHT_K_ITEMS["8.01"] == "other"
    assert sec.sic_to_sector(2834) == "health_care" and sec.sic_to_sector(None) == "other"
    assert universe._norm("Aetna Inc /PA/") == universe._norm("AETNA INC")


def test_earnings_releases_are_masked_and_read_from_the_cache_only(tmp_path, monkeypatch):
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000115000001/ex99.htm"
    monkeypatch.setattr(filings, "TEXT_CACHE", tmp_path / "8k_text")
    path = filings._cache_path(url)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        gzip.compress(
            b"Zorblax Widgets Inc. (ZRBX) reported record revenue. Zorblax raised guidance."
        )
    )
    monkeypatch.setattr(sec, "CACHE", tmp_path / "sec")
    (tmp_path / "sec").mkdir()
    (tmp_path / "sec" / "submissions_1.json").write_text(
        json.dumps({"name": "Zorblax Widgets Inc", "formerNames": []})
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "cik": 1,
                "accession": "0000000001-15-000001",
                "primary": "x.htm",
                "accepted_et": "2015-02-03 16:05:00",
                "url": url,
                "error": None,
            }
        ]
    ).to_csv(manifest, index=False)

    class Mkt:
        calendar = data.TradingCalendar()

        def entities(self):
            return pd.DataFrame({"entity_id": ["ZRBX"], "cik": [1]})

    import requests

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )
    d = filings.SecEarningsReleases(Mkt(), {"manifest": str(manifest)}).documents(
        None, "2016-01-01"
    )
    assert len(d) == 1 and d["doc_id"].iloc[0] == "0000000001-15-000001|2.02"
    t = d["text"].iloc[0]
    assert "Zorblax" not in t and "ZRBX" not in t and "COMPANY_A" in t
    # usable from the next New York midnight after acceptance
    assert d["available_at"].iloc[0] == pd.Timestamp("2015-02-04 05:00", tz="UTC")
