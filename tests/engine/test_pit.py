import numpy as np
import pandas as pd
import pytest

from engine import calendar, config, observations, panel, pipeline, pit
from engine.market import forward_returns

from .conftest import synthetic_cfg

UTC = lambda s: pd.Timestamp(s, tz="UTC")


def test_check_cells_rejects_data_at_or_after_the_decision():
    T = UTC("2020-01-31 21:30")
    ok = pd.DataFrame(
        {
            "entity_id": ["a"],
            "decision_time": [T],
            "feature": ["x"],
            "available_at": [T - pd.Timedelta("1s")],
        }
    )
    assert pit.check_cells(ok) == 1
    for at in (T, T + pd.Timedelta("1s")):
        with pytest.raises(pit.PointInTimeError):
            pit.check_cells(ok.assign(available_at=[at]))


def test_check_labels_rejects_entry_at_or_before_the_decision():
    T = UTC("2020-01-31 21:30")
    f = pd.DataFrame({"entity_id": ["a"], "decision_time": [T], "entry_time": [T]})
    with pytest.raises(pit.PointInTimeError):
        pit.check_labels(f)
    assert pit.check_labels(f.assign(entry_time=[T + pd.Timedelta("1D")])) == 1


class _Market:
    calendar = calendar.TradingCalendar(tz="UTC", close="21:00")

    def universe(self, as_of):
        return pd.DataFrame({"entity_id": ["a"], "group": ["g"]})

    def labels(self, rows, horizon):
        return rows.assign(
            entry_time=rows["decision_time"] + pd.Timedelta("1D"),
            label_end=pd.NaT,
            fwd_return=0.0,
        )


class _Source:
    name = "s"

    def __init__(self, obs):
        self.obs = obs

    def fetch(self, start, end):
        pass

    def observations(self, start, end):
        return self.obs.assign(source="s")


def _build(obs, max_age=None):
    spec = panel.PanelSpec(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-02-01"),
        sources=[panel.SourceSpec(_Source(obs), ["x"], max_age)],
    )
    return panel.build(_Market(), spec)


def test_panel_uses_only_strictly_earlier_observations():
    T = _Market.calendar.decision_times("2020-01-01", "2020-01-31", "monthly")[0]
    obs = pd.DataFrame(
        {
            "entity_id": "a",
            "available_at": [T - pd.Timedelta("1D"), T, T + pd.Timedelta("1h")],
            "feature": "x",
            "value": [1.0, 2.0, 3.0],
        }
    )
    p = _build(obs)
    assert p.frame["x"].tolist() == [
        1.0
    ]  # the value stamped exactly at T is not usable yet
    assert pit.check_panel(p)["cells_checked"] == 1


def test_latest_wins_ties_go_to_the_later_row_and_nan_masks_older_values():
    T = _Market.calendar.decision_times("2020-01-01", "2020-01-31", "monthly")[0]
    day = pd.Timedelta("1D")
    obs = pd.DataFrame(
        {
            "entity_id": "a",
            "available_at": [T - 3 * day, T - 2 * day, T - 2 * day],
            "feature": "x",
            "value": [1.0, 2.0, 5.0],
        }
    )
    assert _build(obs).frame["x"].tolist() == [5.0]
    masked = obs.assign(value=[1.0, 2.0, np.nan])
    assert (
        _build(masked).frame["x"].isna().all()
    )  # the latest report had no value: not the old one


def test_stale_values_are_blanked():
    T = _Market.calendar.decision_times("2020-01-01", "2020-01-31", "monthly")[0]
    obs = pd.DataFrame(
        {
            "entity_id": "a",
            "available_at": [T - pd.Timedelta("40D")],
            "feature": "x",
            "value": [1.0],
        }
    )
    assert _build(obs, max_age=60).frame["x"].tolist() == [1.0]
    assert _build(obs, max_age=30).frame["x"].isna().all()


def test_rolling_window_matches_brute_force():
    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        {
            "entity_id": rng.choice(["a", "b"], 60),
            "available_at": UTC("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 24, 60), unit="h"),
            "w": rng.random(60),
        }
    )
    window = pd.Timedelta("30D")
    agg = lambda e: [float(len(e)), float(e["w"].sum())]
    obs = observations.rolling_window(
        events, window, agg, ["n", "wsum"], "s", ["a", "b", "c"]
    )
    for T in (
        UTC("2020-01-01")
        + pd.to_timedelta(rng.integers(0, 400 * 24, 40), unit="h")
        + pd.Timedelta("30min")
    ):
        for e in ("a", "b", "c"):
            inside = events[
                (events["entity_id"] == e)
                & (events["available_at"] < T)
                & (events["available_at"] > T - window)
            ]
            o = obs[(obs["entity_id"] == e) & (obs["available_at"] < T)]
            latest = (
                o.sort_values("available_at", kind="stable")
                .groupby("feature")["value"]
                .last()
            )
            assert latest["n"] == len(inside)
            assert latest["wsum"] == pytest.approx(inside["w"].sum())


def test_forward_returns_enter_after_the_decision_and_handle_delisting():
    close = pd.date_range("2020-01-01 21:00", periods=10, freq="D", tz="UTC")
    px = np.arange(100.0, 110.0)
    T = pd.DatetimeIndex([close[2] + pd.Timedelta("30min")])
    lab = forward_returns(close, px, T, 3, ended=False)
    assert (
        lab["entry_time"].iloc[0] == close[3] and lab["label_end"].iloc[0] == close[6]
    )
    assert lab["fwd_return"].iloc[0] == pytest.approx(106 / 103 - 1)
    late = pd.DatetimeIndex([close[7] + pd.Timedelta("30min")])
    assert (
        forward_returns(close, px, late, 3, ended=False)["fwd_return"].isna().all()
    )  # unfinished
    dead = forward_returns(close, px, late, 3, ended=True, delist_adjustment=-0.3)
    assert dead["fwd_return"].iloc[0] == pytest.approx((109 / 108) * 0.7 - 1)


def test_a_planted_future_feature_never_reaches_its_own_row(tmp_path):
    """`leaky` is each row's realised forward return, published when the window closes. A correct
    builder only ever shows an EARLIER decision's (already closed) value."""
    study = config.from_dict("synthetic", synthetic_cfg(tmp_path, leak=True))
    study.sources["signals"].features = None
    p = pipeline.build_panel(study, "tuning")
    f = p.frame.dropna(subset=["fwd_return", "leaky"])
    assert len(f) > 1000
    assert not np.isclose(f["leaky"], f["fwd_return"], rtol=0, atol=1e-15).any()
