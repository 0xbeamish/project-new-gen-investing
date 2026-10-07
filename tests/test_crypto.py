"""The crypto plug-in on synthetic bars: no network. Every source must ignore data stamped at or
after the decision."""

import numpy as np
import pandas as pd
import pytest

from engine.markets import crypto
from engine.markets.crypto import binance, llama

CFG = {
    "calendar": {"kind": "continuous", "close": "23:59", "decide_after_close": "1min"},
    "schedule": "weekly",
    "data_end": "2021-12-31",
    "labels": {"horizon": 7, "ended_if_no_bar_for_days": 10, "delisted_return": -0.30},
    "universe": {
        "top": 2,
        "volume_days": 30,
        "min_history_bars": 90,
        "exclude": ["USDC"],
        "renamed": ["OLD"],
        "gap_days": 10,
    },
    "costs": {"taker_fee_bps": 10, "trade_usd": 10000, "impact_days": 30},
}


def bars(start, n, close=100.0, vol=1e6, seed=0):
    rng = np.random.default_rng(seed)
    c = close * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="D"),
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "quote_volume": np.full(n, vol),
        }
    )


def market(series: dict) -> crypto.Crypto:
    m = crypto.Crypto(CFG)
    m._series = {}
    for e, s in series.items():
        s = s.copy()
        s["close_at"] = m.calendar.bar_close(s["time"])
        m._series[e] = s
    end = m.data_end
    m.ended = {
        e: bool(s["time"].iloc[-1] < end - pd.Timedelta(days=10)) for e, s in m._series.items()
    }
    m._volume_table()
    return m


def at(day: str) -> pd.Timestamp:
    """The decision just after `day`'s bar closes."""
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)


def test_split_at_gaps_makes_a_reused_ticker_a_new_entity():
    a, b = bars("2021-01-01", 50), bars("2021-04-01", 30)
    parts = crypto.split_at_gaps(pd.concat([a, b], ignore_index=True), "LUNA", 10)
    assert list(parts) == ["LUNA", "LUNA~2021-04-01"]
    assert len(parts["LUNA"]) == 50 and len(parts["LUNA~2021-04-01"]) == 30


def test_keep_drops_stablecoins_and_leveraged_tokens_only():
    m = crypto.Crypto(CFG)
    bases = {"BTC", "ETH", "USDC", "BTCUP", "ETHBEAR", "JUP", "SUPER"}
    kept = {b for b in bases if m.keep(b, bases)}
    assert kept == {"BTC", "ETH", "JUP", "SUPER"}


def test_universe_ranks_by_volume_before_the_decision_and_needs_history():
    big = bars("2021-01-01", 200, vol=5e6)
    small = bars("2021-01-01", 200, vol=1e6)
    young = bars("2021-05-01", 60, vol=9e9)  # huge volume, too little history
    spike = bars("2021-01-01", 200, vol=1e5)
    m = market({"BIG": big, "SMALL": small, "YOUNG": young, "SPIKE": spike})
    t = at("2021-06-29")
    assert set(m.universe(t)["entity_id"]) == {"BIG", "SMALL"}
    # a volume spike on the decision day's own bar (closing after t) changes nothing
    s = m._series["SPIKE"]
    s.loc[s["time"] == pd.Timestamp("2021-06-30"), "quote_volume"] = 1e15
    m._volume_table()
    assert set(m.universe(t)["entity_id"]) == {"BIG", "SMALL"}


def test_price_features_use_only_bars_closed_before_the_decision():
    m = market({"BTC": bars("2021-01-01", 200)})
    rows = pd.DataFrame({"entity_id": ["BTC"], "decision_time": [at("2021-06-29")]})
    src = crypto.PriceFeatures(m)
    before = src.observations_at(rows).set_index("feature")["value"]
    assert (src.observations_at(rows)["available_at"] < rows["decision_time"].iloc[0]).all()
    s = m._series["BTC"]
    s.loc[s["time"] >= pd.Timestamp("2021-06-30"), "close"] *= 10  # later bars
    after = src.observations_at(rows).set_index("feature")["value"]
    pd.testing.assert_series_equal(before, after)


def test_labels_enter_after_the_decision_and_penalise_delisting_unless_renamed():
    dead = bars("2021-01-01", 100, seed=1)
    old = bars("2021-01-01", 100, seed=1)
    m = market({"DEAD": dead, "OLD": old})
    t = at("2021-04-07")  # last bar is 2021-04-10: the window is cut short
    rows = pd.DataFrame({"entity_id": ["DEAD", "OLD"], "decision_time": [t, t]})
    lab = m.labels(rows, 7).set_index("entity_id")
    assert lab["delisted"].all()
    assert (lab["entry_time"] > t).all()
    c = dead.set_index("time")["close"]
    raw = c.iloc[-1] / c[pd.Timestamp("2021-04-08")] - 1
    assert lab.loc["OLD", "fwd_return"] == pytest.approx(raw)
    assert lab.loc["DEAD", "fwd_return"] == pytest.approx((1 + raw) * 0.7 - 1)


def test_cost_is_two_fees_plus_square_root_impact():
    b = bars("2021-01-01", 200)
    m = market({"BTC": b})
    t = at("2021-06-29")
    rows = pd.DataFrame({"entity_id": ["BTC"], "decision_time": [t]})
    hist = b[b["time"] <= pd.Timestamp("2021-06-29")].tail(31)
    sd = np.std(np.diff(np.log(hist["close"])), ddof=1)
    want = 2 * (10 + 1e4 * sd * np.sqrt(1e4 / 1e6))
    assert m.cost_bps(rows).iloc[0] == pytest.approx(want)
    s = m._series["BTC"]
    s.loc[s["time"] >= pd.Timestamp("2021-06-30"), "quote_volume"] = 1.0  # later bars: no effect
    assert m.cost_bps(rows).iloc[0] == pytest.approx(want)


def test_funding_ignores_settlements_at_or_after_the_decision(monkeypatch, tmp_path):
    m = market({"BTC": bars("2021-01-01", 200)})
    t = at("2021-06-29")
    times = pd.date_range(t - pd.Timedelta(days=10), t + pd.Timedelta(days=2), freq="8h")
    f = pd.DataFrame({"time": times, "rate": np.where(times < t, 0.0001, 0.05)})
    monkeypatch.setattr(binance, "CACHE", tmp_path)
    (tmp_path / "perp_symbols.txt").write_text("BTCUSDT")
    monkeypatch.setattr(binance, "load", lambda kind, sym: f)
    rows = pd.DataFrame({"entity_id": ["BTC"], "decision_time": [t]})
    obs = crypto.Funding(m).observations_at(rows)
    assert obs["value"].iloc[0] == pytest.approx(0.0001)
    assert (obs["available_at"] < t).all()


def test_fees_are_usable_only_from_the_end_of_the_next_day(monkeypatch):
    m = market({"UNI": bars("2021-01-01", 300)})
    days = pd.date_range("2021-01-01", "2021-10-20", freq="D")
    fees = pd.Series(1000.0, index=days)
    monkeypatch.setattr(llama, "token_fee_slugs", lambda bases, **k: {"UNI": "uniswap"})
    monkeypatch.setattr(llama, "daily_fees", lambda slug: fees.copy())
    t = at("2021-09-30")
    rows = pd.DataFrame({"entity_id": ["UNI"], "decision_time": [t]})
    src = crypto.Fees(m)
    base = src.observations_at(rows).set_index("feature")["value"]
    assert base["fee_growth_90d"] == pytest.approx(0.0)
    fees[pd.Timestamp("2021-10-01")] = 1e9  # the decision day's fees
    fees[pd.Timestamp("2021-09-30")] = 1e9  # the last closed day's: usable at the end of 10-01
    obs = src.observations_at(rows)
    assert obs.set_index("feature")["value"]["fee_growth_90d"] == pytest.approx(0.0)
    assert (obs["available_at"] < t).all()
    fees[pd.Timestamp("2021-09-29")] = 1e9  # usable at the end of 09-30, before the decision
    assert src.observations_at(rows).set_index("feature")["value"]["fee_growth_90d"] > 1


def test_a_hack_counts_from_the_end_of_the_day_after_it(monkeypatch):
    m = market({"AAVE": bars("2021-01-01", 300)})
    h = pd.DataFrame({"symbol": ["AAVE"], "date": [pd.Timestamp("2021-06-10")], "usd": [1e6]})
    monkeypatch.setattr(llama, "hacks", lambda symbols, **k: h)
    src = crypto.Hacks(m)

    def value(day):
        rows = pd.DataFrame({"entity_id": ["AAVE"], "decision_time": [at(day)]})
        obs = src.observations_at(rows)
        assert (obs["available_at"] < at(day)).all()
        return obs["value"].iloc[0]

    assert value("2021-06-10") == 0.0  # decided right after the hack day: not yet
    assert value("2021-06-11") == pytest.approx(np.log1p(1e6))
    assert value("2021-07-12") == 0.0  # more than 30 days later


def test_a_chain_token_maps_to_its_chain_and_true_ties_to_nothing(monkeypatch):
    e = pd.DataFrame(
        {
            "slug": ["near", "near-protocol", "a", "b"],
            "symbol": ["NEAR", "NEAR", "ALT", "ALT"],
            "all_time": [1.0, 50.0, 5.0, 4.0],
            "chain": [True, False, False, False],
        }
    )
    monkeypatch.setattr(llama, "fee_entities", lambda max_age_s=None: e)
    assert llama.token_fee_slugs(["NEAR", "ALT"]) == {"NEAR": "near"}


def test_growth_uses_volume_before_the_decision_and_dex_volume_from_the_next_days_close(
    monkeypatch,
):
    b = bars("2021-01-01", 300)
    m = market({"ETH": b})
    days = pd.date_range("2021-01-01", "2021-10-31", freq="D")
    dex = pd.Series(1e6, index=days)
    monkeypatch.setattr(crypto.indicators, "series", lambda token: {"chain_dex_volume": dex})
    t = at("2021-09-30")
    rows = pd.DataFrame({"entity_id": ["ETH"], "decision_time": [t]})
    src = crypto.Growth(m)
    obs = src.observations_at(rows)
    assert (obs["available_at"] < t).all()
    base = obs.set_index("feature")["value"]
    assert base["dex_volume_growth_90d"] == pytest.approx(0.0)
    assert base["px_volume_growth_90d"] == pytest.approx(0.0)
    dex[pd.Timestamp("2021-09-30")] = 1e12  # usable only at the end of 10-01: after t
    s = m._series["ETH"]
    s.loc[s["time"] >= pd.Timestamp("2021-10-01"), "quote_volume"] = 1e15  # after t
    after = src.observations_at(rows).set_index("feature")["value"]
    pd.testing.assert_series_equal(base, after)
    dex[pd.Timestamp("2021-09-29")] = 1e12  # usable at the end of 09-30: before t
    assert src.observations_at(rows).set_index("feature")["value"]["dex_volume_growth_90d"] > 1
