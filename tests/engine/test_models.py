import numpy as np
import pandas as pd
import pytest

from engine import models, pipeline, sample


def test_training_labels_never_overlap_the_test_block(study, tuning_rows):
    wf = pipeline.walk_forward(study)
    seen = {}

    class Spy:
        def __init__(self):
            self.m = models.ridge()()

        def fit(self, X, y):
            seen.setdefault("train", []).append(X.index)
            self.m.fit(X, y)
            return self

        def predict(self, X):
            seen.setdefault("test", []).append(X.index)
            return self.m.predict(X)

    wf.model = Spy
    wf.run(tuning_rows, ["planted"], study.market.calendar)
    for tr, te in zip(seen["train"], seen["test"]):
        first_test = tuning_rows.loc[te, "decision_time"].min()
        assert (tuning_rows.loc[tr, "label_end"] < first_test).all()


def test_embargo_and_legacy_purge_change_only_the_training_cut(study, tuning_rows):
    def n_train(wf):
        sizes = []

        class Count:
            def __init__(self):
                self.m = models.ridge()()

            def fit(self, X, y):
                sizes.append(len(X))
                self.m.fit(X, y)
                return self

            def predict(self, X):
                return self.m.predict(X)

        wf.model = Count
        wf.run(tuning_rows, ["planted"], study.market.calendar)
        return sizes

    base = n_train(pipeline.walk_forward(study))
    emb = pipeline.walk_forward(study)
    emb.embargo = pd.Timedelta(days=40)
    assert all(
        e < b for e, b in zip(n_train(emb), base)
    )  # an embargo drops the last month
    leg = pipeline.walk_forward(study)
    leg.legacy_purge_days = 31
    assert len(n_train(leg)) == len(base)


def test_rank_features_are_per_period_and_centred():
    f = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2020-01-31"] * 3 + ["2020-02-29"] * 2, utc=True
            ),
            "x": [1.0, 2.0, np.nan, 100.0, 50.0],
        }
    )
    r = models.rank_features(f, ["x"])["x"].tolist()
    assert r == [0.0, 0.5, 0.0, 0.5, 0.0]


def test_make_batches_matches_the_pilot():
    jev_model = pytest.importorskip(
        "jev.model"
    )  # the pilot code: present in the research repo only

    rng = np.random.default_rng(1)
    n = 137
    f = pd.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(n)],
            "decision_time": pd.to_datetime(
                rng.choice(["2015-01-30", "2015-02-27"], n), utc=True
            ),
            "group": rng.choice(["a", "b", "c"], n),
        }
    )
    ours = sample.make_batches(f, 10, "group", 0)
    legacy = jev_model.make_batches(
        f.rename(columns={"decision_time": "as_of"}).assign(sector=f["group"]),
        10,
        0,
        by="sector",
        period="M",
    )
    key = lambda d: sorted(map(frozenset, d.groupby("batch")["entity_id"].apply(set)))
    assert key(ours) == key(legacy)
