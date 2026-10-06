"""The walk-forward: purge, embargo, ranks, and the yearly recency rule."""

import itertools

import numpy as np
import pandas as pd

from engine import model, panel, run

from .conftest import text_cfg


def _spy_runs(wf, rows, feats, calendar):
    """(train index, test index) of every refit."""
    seen = []

    class Spy:
        def __init__(self):
            self.m = model.ridge()()

        def fit(self, X, y, sample_weight=None):
            seen.append([X.index, None])
            self.m.fit(X, y)
            return self

        def predict(self, X):
            seen[-1][1] = X.index
            return self.m.predict(X)

    wf.model = Spy
    wf.run(rows, feats, calendar)
    return seen


def test_training_labels_never_overlap_the_test_block(study, tuning_rows):
    seen = _spy_runs(run.walk_forward(study), tuning_rows, ["planted"], study.market.calendar)
    assert seen
    for tr, te in seen:
        first_test = tuning_rows.loc[te, "decision_time"].min()
        assert (tuning_rows.loc[tr, "label_end"] < first_test).all()


def test_an_embargo_only_moves_the_training_cut(study, tuning_rows):
    cal = study.market.calendar
    base = [len(tr) for tr, _ in _spy_runs(run.walk_forward(study), tuning_rows, ["planted"], cal)]
    emb = run.walk_forward(study)
    emb.embargo = pd.Timedelta(days=40)
    with_emb = [len(tr) for tr, _ in _spy_runs(emb, tuning_rows, ["planted"], cal)]
    assert all(e < b for e, b in zip(with_emb, base))  # an embargo drops the last month


def test_rank_features_are_per_period_and_centred():
    f = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2020-01-31"] * 3 + ["2020-02-29"] * 2, utc=True),
            "x": [1.0, 2.0, np.nan, 100.0, 50.0],
        }
    )
    assert model.rank_features(f, ["x"])["x"].tolist() == [0.0, 0.5, 0.0, 0.5, 0.0]


def test_auto_recency_is_chosen_yearly_moves_one_step_and_follows_a_regime_change(tmp_path):
    """A planted numeric effect flips sign in 2015. The yearly rule holds one value a year, moves
    at most one grid step a year, stays put while the regime is stable, and shortens within 2 years
    of the flip."""
    cfg = text_cfg(
        tmp_path,
        entities=300,
        end="2018-12-31",
        regime_change="2015-01-01",
        effects={"guidance": 0.01, "x": 0.03, "x_after": -0.03},
    )
    cfg["periods"] = {
        "tuning": ["2010-01-01", "2019-01-01"],
        "check": ["2019-01-01", "2019-06-01"],
        "holdout_start": "2019-06-01",
    }
    st = run.study_from_config("demo", cfg)
    rows = panel.model_rows(st, run.build_panel(st, "tuning"))
    wf = model.WalkForward(
        model=model.ridge(10.0),
        block="M",
        min_train_periods=12,
        keep=("group", "fwd_return", "rt_cost"),
        half_life="auto",
    )
    wf.run(rows, st.feature_set("numbers_text"), st.market.calendar)
    ch = pd.Series({p.start_time: v for p, v in wf.chosen_.items()}).sort_index().dropna()
    new = ch.groupby(ch.index.year).agg(lambda s: list(dict.fromkeys(s)))
    assert all(len(v) == 1 for v in new)  # one choice a year
    steps = [6, 12, 24, 36]
    yearly = {y: v[0] for y, v in new.items()}
    years = sorted(yearly)
    assert all(
        abs(steps.index(yearly[a]) - steps.index(yearly[b])) <= 1
        for a, b in itertools.pairwise(years)
    )
    assert yearly[2013] == yearly[2014] == yearly[2015]  # stable regime: no movement
    assert min(yearly[2016], yearly[2017]) < yearly[2015]  # the flip shortens it within 2 years
