"""The panel builder: latest value strictly before the decision, staleness, and a planted leak."""

import numpy as np
import pandas as pd

from engine import data, panel, run

from .conftest import planted_cfg


class _Market:
    calendar = data.TradingCalendar(tz="UTC", close="21:00")

    def universe(self, as_of):
        return pd.DataFrame({"entity_id": ["a"], "group": ["g"]})

    def labels(self, rows, horizon):
        return rows.assign(
            entry_time=rows["decision_time"] + pd.Timedelta("1D"), label_end=pd.NaT, fwd_return=0.0
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


def _decision():
    return _Market.calendar.decision_times("2020-01-01", "2020-01-31", "monthly")[0]


def test_panel_uses_only_strictly_earlier_observations():
    T = _decision()
    obs = pd.DataFrame(
        {
            "entity_id": "a",
            "available_at": [T - pd.Timedelta("1D"), T, T + pd.Timedelta("1h")],
            "feature": "x",
            "value": [1.0, 2.0, 3.0],
        }
    )
    p = _build(obs)
    assert p.frame["x"].tolist() == [1.0]  # the value stamped exactly at T is not usable yet
    assert data.check_panel(p)["cells_checked"] == 1


def test_latest_wins_ties_go_to_the_later_row_and_nan_masks_older_values():
    T = _decision()
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
    assert _build(masked).frame["x"].isna().all()  # the latest report had no value: not the old one


def test_stale_values_are_blanked():
    T = _decision()
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


def test_a_source_with_nothing_to_say_leaves_its_features_empty():
    p = _build(data.empty_observations().drop(columns="source"))
    assert p.frame["x"].isna().all()


def test_a_planted_future_feature_never_reaches_its_own_row(tmp_path):
    """`leaky` is each row's realised forward return, published when the window closes. A correct
    builder only ever shows an EARLIER decision's (already closed) value."""
    study = run.study_from_config("demo", planted_cfg(tmp_path, leak=True))
    study.sources["numbers"].features = None
    p = run.build_panel(study, "tuning")
    f = p.frame.dropna(subset=["fwd_return", "leaky"])
    assert len(f) > 1000
    assert not np.isclose(f["leaky"], f["fwd_return"], rtol=0, atol=1e-15).any()


def test_model_rows_rank_the_label_within_group_and_attach_costs(tuning_rows):
    r = tuning_rows
    assert r["fwd_return"].notna().all()
    assert r["fwd_rank"].between(-0.5, 0.5).all()
    assert r.groupby(["decision_time", "group"])["fwd_rank"].mean().abs().max() < 0.05
    assert (r["rt_cost"] == 0.002).all()  # the demo's 20 bp round trip, as a fraction
