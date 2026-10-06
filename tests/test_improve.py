"""Tracking, the adjustment ladder, the loop coordinator and the replay on planted markets."""

import numpy as np
import pandas as pd
import pytest

from engine import improve
from engine.text import tracking

from .conftest import study_and_rows

CUTS = pd.date_range("2012-04-01", "2015-01-01", freq="QS", tz="UTC")
FEATS = [
    "x_value",
    "x_noise",
    "txt_guidance_action_raised",
    "txt_guidance_action_lowered",
    "txt_demand_level",
    "txt_one_off_charge_p",
    "txt_buyback_p",
    "txt_overall_tone_level",
]


def coordinator(st, rows, **cfg):
    meta = st.sources["releases_text"].source.feature_meta()
    return improve.Coordinator(
        st, FEATS, meta, lambda v: rows, improve.LoopsConfig(**cfg), log_tests=False
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """one_off stops mattering in 2012 (the model keeps its old weight: over-weighted);
    demand pays at both extremes (a U: non-linear)."""
    return study_and_rows(
        tmp_path_factory.mktemp("planted"),
        entities=1000,
        cost_bps=5,
        regime_change="2012-01-01",
        effects={
            "guidance": 0.02,
            "demand_u": 0.02,
            "one_off": -0.06,
            "one_off_after": 0.0,
            "x": 0.01,
        },
    )


@pytest.fixture(scope="module")
def correlated(tmp_path_factory):
    """The true error is OUTER: numeric x stops mattering in 2012, and text demand correlates
    with x (rho 0.8) while its own small effect never changes."""
    return study_and_rows(
        tmp_path_factory.mktemp("corr"),
        entities=1000,
        cost_bps=5,
        regime_change="2012-01-01",
        rho_x_demand=0.8,
        effects={"guidance": 0.02, "demand_lin": 0.006, "x": 0.06, "x_after": 0.0},
    )


def flags(co, c, cutoff):
    sc, w, enc, cols = co.scores(c)
    w = w[[co._block_start(b) < cutoff for b in w.index]]
    meta = {k: co.meta[k] for k in cols if k in co.meta}
    return tracking.report(enc, sc, w, cols, meta, co.cal, cutoff=cutoff)["flags"].set_index(
        "field"
    )


def test_tracker_flags_the_over_weighted_and_the_non_linear_field(planted):
    st, rows = planted
    co = coordinator(st, rows)
    f = flags(co, improve.ModelConfig(), CUTS[-1])
    assert f.loc["txt_one_off_charge_p", "status"] == "over-weighted"
    assert f.loc["txt_demand_level", "non_linear"]
    assert f.loc["txt_demand_level", "status"] == "ok"  # a U has no linear mis-weight to see
    assert f.loc["txt_buyback_p", "status"] == "ok" and not f.loc["txt_buyback_p", "non_linear"]


def test_the_ladder_fixes_both(planted):
    st, rows = planted
    co = coordinator(st, rows)
    log = co.run(CUTS)
    acc = log[log["accepted"] & (log["loop"] != "brake")]
    assert set(acc["rung"]) >= {"weights", "encoding"}
    assert "txt_demand_level" in dict(co.cur.encodings)
    assert co.cur.half_life is not None
    assert (acc.groupby("cycle").size() <= 1).all()  # one structural change per cycle
    frozen, final = (
        flags(co, improve.ModelConfig(), CUTS[-1]),
        flags(co, co.cur, CUTS[-1]),
    )
    assert abs(final.loc["txt_one_off_charge_p", "t_joint"]) < abs(
        frozen.loc["txt_one_off_charge_p", "t_joint"]
    )
    assert not final["non_linear"].any() or "txt_demand_level" not in final.index
    on = co.paired(co.cur, improve.ModelConfig(), CUTS[-1], accept=True)
    assert on["gain"] > 0 and on["t"] > 2  # end to end, net of costs, on held-out entities


def test_inner_loop_leaves_text_alone_when_the_error_is_outer(correlated):
    st, rows = correlated
    co = coordinator(st, rows)
    f = flags(co, improve.ModelConfig(), CUTS[-1])
    # the trap is real: one at a time, demand looks badly over-weighted ...
    assert f.loc["txt_demand_level", "t_one_at_a_time"] < -2
    # ... jointly the error sits on x, where it belongs
    assert (
        f.loc["x_value", "status"] == "over-weighted"
        and f.loc["txt_demand_level", "status"] == "ok"
    )
    log = co.run(CUTS)
    inner = log[(log["loop"] == "inner") & (log["field"] == "txt_demand_level")]
    assert inner.empty  # no encoding, rewrite or drop of the demand question
    assert ((log["rung"] == "weights") & log["accepted"]).any()  # the outer refit fixed it
    after = flags(co, co.cur, CUTS[-1])
    assert abs(after.loc["x_value", "t_joint"]) < abs(f.loc["x_value", "t_joint"])


def test_question_change_rescores_then_blocks_a_cycle_and_rests_the_field(planted):
    """The mechanism, not the statistics: the judge is stubbed to accept the rewrite."""
    st, rows = planted

    class Rewrite:
        def propose_field(self, f, meta, cutoff, coord):
            if f != "txt_one_off_charge_p":
                return None
            return (
                "v2",
                "COMPANY_A says a one-time charge lowered this period's results (v2).",
            )

    meta = st.sources["releases_text"].source.feature_meta()
    seen = []

    def rows_for(v):
        seen.append(v)
        return rows.assign(txt_one_off_charge_p=0.5) if v == "v2" else rows

    co = improve.Coordinator(
        st,
        FEATS,
        meta,
        rows_for,
        improve.LoopsConfig(half_lives=(), encodings=(), persist_k=2, oscillation_flips=99),
        proposer=Rewrite(),
        log_tests=False,
    )
    real = co.paired

    def paired(cand, cur, cutoff, accept, since=None):
        if cand.questions == "v2" and cur.questions == "v1":
            return {"gain": 0.01, "t": 9.0, "periods": 30}
        return real(cand, cur, cutoff, accept, since)

    co.paired = paired
    co._rollback = lambda k, cutoff: False
    log = co.run(CUTS)
    q = log[(log["rung"] == "question") & log["accepted"]]
    assert len(q) == 1 and "v2" in seen  # history re-read whole with the new version
    k = int(q["cycle"].iloc[0])
    assert log[(log["cycle"] == k + 1) & (log["loop"] != "brake")].empty  # outer refit first
    assert co.rest_until["txt_one_off_charge_p"] >= k + co.cfg.cooldown
    assert co.cur.questions == "v2"


def test_oscillation_alarms_freeze_the_field():
    class Stub(improve.Coordinator):
        def __init__(self):
            self.cfg = improve.LoopsConfig(oscillation_window=6)
            self.weight_signs, self.frozen_fields, self.events, self.question_texts = (
                {},
                set(),
                [],
                {},
            )

    co = Stub()
    for i, s in enumerate([1, -1, 1, 1]):
        co._oscillation(pd.DataFrame([{"a": s * 0.1, "b": 0.1}]), i, pd.Timestamp("2020-01-01"))
    assert "a" in co.frozen_fields and "b" not in co.frozen_fields
    t = pd.Timestamp("2020-01-01")
    assert not co.question_changed("q", "The text says the outlook was raised.", 0, t)
    assert not co.question_changed("q", "COMPANY_A says it raised its outlook for next year.", 1, t)
    assert co.question_changed("q", "The text says the outlook was raised!", 2, t)  # back toward v1
    assert "q" in co.frozen_fields


def test_replay_runs_on_vs_frozen_month_by_month(small_text):
    st, rows = small_text
    meta = st.sources["releases_text"].source.feature_meta()
    cfg = improve.ReplayConfig(
        years=(2013, 2014),
        loops=improve.LoopsConfig(block="Q", half_lives=(), persist_k=2),
    )
    res = improve.replay(st, FEATS, meta, lambda v: rows, cfg)
    r = res["result"]
    assert r["months"] == 24 and set(r) >= {
        "diff_net",
        "diff_net_t",
        "diff_ic",
        "diff_ic_t",
    }
    assert res["final_config"].startswith(
        "hl=auto"
    )  # the ON outer loop's recency, fixed in advance


def test_tracking_report_is_empty_before_anything_closes(small_text):
    st, rows = small_text
    co = coordinator(st, rows)
    sc, w, enc, cols = co.scores(improve.ModelConfig())
    rep = tracking.report(enc, sc, w, cols, {}, co.cal, cutoff=sc["decision_time"].min())
    assert rep["periods"] == 0 and rep["flags"].empty


def test_oscillation_alarm_ignores_noise_near_zero():
    """A: fought over by the loops (large weight driven up and down). B: pure noise near zero.
    C, D: steady inputs that set the typical weight. The alarm fires on A only."""

    class Stub(improve.Coordinator):
        def __init__(self, min_frac):
            self.cfg = improve.LoopsConfig(oscillation_window=12, oscillation_min=min_frac)
            self.weight_signs, self.frozen_fields, self.events, self.question_texts = (
                {},
                set(),
                [],
                {},
            )

    rng = np.random.default_rng(0)
    rows = [
        {
            "A": 0.15 * (1 if i % 4 < 2 else -1),
            "B": 0.002 * rng.choice([-1, 1]),
            "C": 0.08,
            "D": -0.06,
        }
        for i in range(12)
    ]
    for min_frac, want in (
        (0.5, {"A"}),
        (0.0, {"A", "B"}),
    ):  # 0 = the old rule: B froze too
        co = Stub(min_frac)
        for i, r in enumerate(rows):
            co._oscillation(pd.DataFrame([r]), i, pd.Timestamp("2020-01-01"))
        assert co.frozen_fields == want
    assert improve.count_flips([1, 0, 0, 1, 0, -1]) == 1  # near-zero refits are skipped, not flips


def test_alarm_and_persistence_count_refits_not_cycles(small_text):
    st, rows = small_text
    co = coordinator(st, rows, block="year")
    cuts = pd.date_range("2012-04-01", "2015-01-01", freq="QS", tz="UTC")  # 12 quarterly cycles
    co.run(cuts)
    refits = len({c.year for c in cuts})  # the latest yearly refit changes once a year
    assert max(len(v) for v in co.weight_signs.values()) <= refits
    assert all(sum(v.values()) <= 3 * refits for v in co.streak.values())
