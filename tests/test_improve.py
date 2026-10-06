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
    co.rollback_if_worse = lambda k, cutoff: False
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
        co.oscillation_alarm(
            pd.DataFrame([{"a": s * 0.1, "b": 0.1}]), i, pd.Timestamp("2020-01-01")
        )
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
    assert r["periods"] == 24 and set(r) >= {
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
            co.oscillation_alarm(pd.DataFrame([r]), i, pd.Timestamp("2020-01-01"))
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


# ---------------------------------------------------------------- the feedback loop (engine loop)
@pytest.fixture(scope="module")
def regime_loop(tmp_path_factory):
    """The demo's scenario A: x_value starts to matter in 2012, a one-time charge stops."""
    from engine import model, panel, run
    from engine.markets import demo

    tmp = tmp_path_factory.mktemp("regime")
    st = run.study_from_config("demo", demo.loop_scenario("regime", tmp))
    res = improve.run_loop(st, out_dir=tmp / "results")
    rows = panel.model_rows(st, run.build_panel(st, "tuning"))
    new = rows[rows["decision_time"] >= pd.Timestamp("2012-02-01", tz="UTC")]
    new = new[new["label_end"] < pd.Timestamp("2015-01-01", tz="UTC")]
    feats = st.feature_set("numbers_text")
    y = new["fwd_rank"] - new.groupby(["decision_time", "group"])["fwd_rank"].transform("mean")
    true = dict(zip(feats, model.ridge(10.0)().fit(model.rank_features(new, feats), y).coef_))
    return res, true


def test_the_loop_learns_a_new_weight_and_unlearns_a_stale_one(regime_loop):
    res, true = regime_loop
    w = res["_weights"].pivot(index="period", columns="input", values="weight")
    early = w[w.index < pd.Timestamp("2012-01-01", tz="UTC")]
    late = w[w.index >= pd.Timestamp("2014-07-01", tz="UTC")]
    assert early["x_value"].abs().max() < 0.03  # nothing to learn before 2012
    assert late["x_value"].mean() > 0.8 * true["x_value"] > 0.2  # converged to the new regime's
    assert early["txt_one_off_charge_p"].max() < -0.1  # the charge mattered...
    assert late["txt_one_off_charge_p"].abs().max() < 0.05  # ...and the outer loop let it go
    ch = res["_changes"]
    acc = ch[ch["accepted"] & (ch["loop"] == "inner")]
    assert (acc["rung"] == "encoding").any()  # the U-shaped demand field was re-encoded
    assert res["result"]["diff_net_t"] > 2  # loop ON beats FROZEN, net of costs
    assert res["inner_judged"] >= 1 and res["registry_internal"]["count"] == res["inner_judged"]


def test_loop_files_have_one_row_per_period_and_input(regime_loop):
    res, _ = regime_loop
    w = pd.read_csv(res["files"]["weights"])
    assert list(w.columns) == ["period", "input", "weight", "config"]
    assert not w.duplicated(["period", "input"]).any()
    per_period = w.groupby("period").size()
    expect = (
        w.drop_duplicates("period")
        .set_index("period")["config"]
        .map(
            lambda key: len(res["_weights"].loc[res["_weights"]["config"] == key, "input"].unique())
        )
    )
    assert (per_period == expect.reindex(per_period.index)).all()  # every input of its config
    r = pd.read_csv(res["files"]["returns"])
    assert {"period", "on_net", "frozen_net", "pick_net", "model_net"} <= set(r.columns)
    assert len(r) == r["period"].nunique()
    c = pd.read_csv(res["files"]["changes"])
    assert {"period", "loop", "rung", "field", "trigger", "t", "bar", "accepted"} <= set(c.columns)


def test_the_loop_leaves_the_text_alone_when_the_error_is_outer(tmp_path):
    from engine import run
    from engine.markets import demo

    st = run.study_from_config("demo", demo.loop_scenario("trap", tmp_path))
    res = improve.run_loop(st, out_dir=tmp_path / "results")
    ch = res["_changes"]
    assert ch.empty or ch[(ch["loop"] == "inner") & (ch["field"] == "txt_demand_level")].empty
    w = res["_weights"].pivot(index="period", columns="input", values="weight")
    assert w["x_value"].iloc[-1] < 0.25 * w["x_value"].iloc[:6].mean()  # the outer loop fixed x


def test_loop_log_writes_exactly_one_registry_row_and_only_once(tmp_path):
    import yaml

    from engine import cli, run
    from engine.markets import demo

    cfg = demo.loop_scenario("regime", tmp_path, entities=120)
    cfg["loop"] |= {"years": [2012, 2013]}
    path = tmp_path / "loop.yaml"
    path.write_text(yaml.safe_dump(cfg))
    args = ["loop", "--market", "demo", "--config", str(path), "--log"]
    with pytest.raises(SystemExit):  # not pre-registered: no loop.name
        cli.main(args)
    cfg["loop"]["name"] = "loop: demo regime, pre-registered"
    path.write_text(yaml.safe_dump(cfg))
    cli.main(args)
    reg = run.study_from_config("demo", cfg).registry
    tests = reg.own()[reg.own()["kind"] == "test"]
    assert len(tests) == 1 and tests["name"].iloc[0] == cfg["loop"]["name"]  # ONE test
    assert tests["mde_80"].iloc[0] > tests["mde_50"].iloc[0] > 0  # its MDE, from FROZEN's noise
    internal = reg.own()[reg.own()["kind"] == "loop_internal"]
    assert len(internal) <= 1  # plus, if it judged any, one row counting its internal judgments
    with pytest.raises(SystemExit):  # one look only
        cli.main(args)
    assert len(reg.own()) == len(tests) + len(internal)


def test_the_question_rung_splits_a_question_and_reads_it_point_in_time(small_text):
    """The free phrase splitter: a 3-gram from the documents behind the largest leave-text-out
    residuals becomes a yes/no question, re-read for every document, attached as a new version."""
    from engine import decide
    from engine.text.improve import PhraseSplitter

    st, rows = small_text
    meta = {c: m for c, m in decide.text_feature_meta(st).items() if c in FEATS}
    sp = PhraseSplitter(st, rows, meta, max_splits=1, min_lift=0.0)
    co = improve.Coordinator(st, FEATS, meta, sp.rows_for, improve.LoopsConfig(), log_tests=False)
    f = "txt_one_off_charge_p"
    version, text, cols = sp.propose_field(f, meta[f], CUTS[-1], co)
    new = sp.rows_for(version)
    assert version == "v2" and text.startswith("The text says:") and cols[0] in new
    assert new[cols[0]].notna().mean() > 0.5 and cols[0] in co.meta  # read and registered
    assert new.drop(columns=cols).equals(rows)  # the old columns are untouched
    assert sp.propose_field(f, meta[f], CUTS[-1], co) is None  # max_splits reached


def test_question_splits_are_not_rewrites_and_never_trip_the_rewrite_alarm(planted):
    """Regression: three split proposals for one field (new questions, similar wording) were read
    as rewrites of the original question, so the third tripped "rewritten back toward an earlier
    version" and froze it although no question had changed."""
    st, rows = planted
    prompts = iter(
        [
            "The text says: 'upgrade shipped on'.",
            "The text says: 'the mainnet upgrade'.",
            "The text says: 'shipped on schedule'.",
        ]
    )

    class Splits:
        def propose_field(self, f, meta, cutoff, coord):
            if f != "txt_one_off_charge_p":
                return None
            text = next(prompts, None)
            return None if text is None else ("v1", text, ["txt_buyback_p"])

    meta = st.sources["releases_text"].source.feature_meta()
    co = improve.Coordinator(
        st,
        FEATS,
        meta,
        lambda v: rows,
        improve.LoopsConfig(half_lives=(), encodings=(), persist_k=2, oscillation_flips=99),
        proposer=Splits(),
        log_tests=False,
    )
    co.run(CUTS)
    assert next(prompts, None) is None  # all three splits were proposed
    assert not any(e.rung == "oscillation" and e.loop == "brake" for e in co.events)
    assert "one_off_charge" not in co.frozen_fields
