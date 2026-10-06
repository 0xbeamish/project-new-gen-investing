"""The input contracts, calendars and the point-in-time checker."""

import numpy as np
import pandas as pd
import pytest

from engine import data

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
    assert data.check_cells(ok) == 1
    for at in (T, T + pd.Timedelta("1s")):
        with pytest.raises(data.PointInTimeError):
            data.check_cells(ok.assign(available_at=[at]))


def test_check_labels_rejects_entry_at_or_before_the_decision():
    T = UTC("2020-01-31 21:30")
    f = pd.DataFrame({"entity_id": ["a"], "decision_time": [T], "entry_time": [T]})
    with pytest.raises(data.PointInTimeError):
        data.check_labels(f)
    assert data.check_labels(f.assign(entry_time=[T + pd.Timedelta("1D")])) == 1


def test_observations_reject_naive_times_and_keep_nan_values():
    obs = pd.DataFrame(
        {
            "entity_id": ["a", "a"],
            "available_at": [UTC("2020-01-01"), UTC("2020-01-02")],
            "source": "s",
            "feature": "x",
            "value": [1.0, None],
        }
    )
    clean = data.validate_observations(obs, "s")
    assert clean["value"].isna().tolist() == [False, True]
    with pytest.raises(TypeError):
        data.validate_observations(obs.assign(available_at=pd.to_datetime(["2020-01-01"] * 2)))
    assert data.validate_observations(data.empty_observations()).empty


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

    def agg(e):
        return [float(len(e)), float(e["w"].sum())]

    obs = data.rolling_window(events, window, agg, ["n", "wsum"], "s", ["a", "b", "c"])
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
            latest = o.sort_values("available_at", kind="stable").groupby("feature")["value"].last()
            assert latest["n"] == len(inside)
            assert latest["wsum"] == pytest.approx(inside["w"].sum())


def test_calendars_trading_and_continuous():
    ny = data.TradingCalendar()
    t = ny.decision_times("2020-01-01", "2020-03-31", "monthly")
    assert list(ny.local_date(pd.Series(t)).dt.day) == [31, 28, 31]  # business month ends
    assert t[0] == pd.Timestamp("2020-01-31 16:30", tz="America/New_York")
    # a filing dated by day only is usable from the next New York midnight
    assert ny.date_available(["2020-01-31"]).iloc[0] == pd.Timestamp(
        "2020-02-01", tz="America/New_York"
    )
    crypto = data.make_calendar(
        {"kind": "continuous", "close": "23:59", "decide_after_close": "1min"}
    )
    w = crypto.decision_times("2021-01-01", "2021-01-31", "weekly")
    assert all(x.dayofweek == 0 and x.hour == 0 for x in w)  # Monday 00:00 UTC, after Sunday's bar


def test_documents_need_unique_ids_and_sort_by_publication():
    d = pd.DataFrame(
        {
            "entity_id": ["a", "a"],
            "available_at": [UTC("2020-02-01"), UTC("2020-01-01")],
            "doc_type": "note",
            "doc_id": ["2", "1"],
            "text": ["later", None],
        }
    )
    clean = data.validate_documents(d)
    assert clean["doc_id"].tolist() == ["1", "2"] and clean["text"].iloc[0] == ""
    with pytest.raises(ValueError):
        data.validate_documents(d.assign(doc_id="1"))
    src = data.FrameDocuments("x", d)
    assert src.documents(None, "2020-01-15")["doc_id"].tolist() == ["1"]
