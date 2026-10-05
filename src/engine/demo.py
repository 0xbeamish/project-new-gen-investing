"""`engine demo`: the whole system on a synthetic market, with no keys and no downloads.

1. numbers vs numbers + text: the walk-forward model and the scorer
2. text scoring: the eval harness (answer key, probes, reaction and drift beyond a history prior,
   leak gate), per-question efficacy, with the FREE keyword reader
3. the question-improvement loop (free proposer), confirmed once on the test split
4. tracking + the loop coordinator: what is mis-weighted or mis-shaped, and what got fixed
5. the decider layer with its feedback note (the free model-pick decider)
State goes to a temporary directory, so the demo can run any number of times.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from engine import config, decider, feedback, loops, pipeline
from engine import decide as dec_mod
from engine.markets import synthetic_text as stm
from engine.text import harness, readers, textloop
from engine.text import questions as tq


def _line(s: str = "") -> None:
    print(s, flush=True)


def run(quick: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="engine_demo_"))
    cfg = config.read("synthetic_text")
    cfg["cache_dir"] = str(tmp / "cache")
    cfg["registry"] = {
        "file": str(tmp / "registry.csv"),
        "holdout_unlock_log": str(tmp / "unlocks.csv"),
    }
    st = config.from_dict("synthetic_text", cfg)
    out = {}

    _line(
        "1. Model: numbers vs numbers + text (tuning periods, rank IC and net long-short)"
    )
    rows = pipeline.rows(st, pipeline.build_panel(st, "tuning"))
    for fs in ("numbers", "numbers_text"):
        sc, _ = pipeline.walk_forward(st).run(
            rows, st.feature_set(fs), st.market.calendar
        )
        ev = pipeline.evaluate(st, sc)
        out[fs] = ev
        _line(
            f"   {fs:13s} IC {ev['ic']:+.3f} (t {ev['ic_t']:.1f})   decile spread net {ev['spread']['net']:+.2%}/month (t {ev['spread']['net_t']:.1f})"
        )

    _line("\n2. Text scoring with the free keyword reader (dev split)")
    sets = stm.eval_sets(st.market, end=st.periods.tuning[1])
    qsets = tq.load(config.path(cfg["text"]["questions"]))
    reader = readers.KeywordReader()
    res = harness.score(sets, qsets, reader, "dev", efficacy=True)
    s = res["summary"]
    out["text"] = s
    _line(
        f"   S {s['S']:.3f} = 0.30 A {s['A']:.2f} + 0.25 B {s['B']:.2f} + 0.30 C {s['C']:.2f} + 0.10 D {s['D']:.2f} + 0.05 E {s['E']:.2f}"
    )
    r = res["reaction"]
    _line(
        f"   reaction IC: prior {r['prior']['IC_react']:.3f} -> card {r['card']['IC_react']:.3f}; drift IC gain {r['gain_vs_prior']['IC_drift']:+.3f}; leak gate {'pass' if res['gates']['leak']['pass'] else 'FAIL'}"
    )
    eff = res["efficacy"][
        ["question", "tag", "gold_skill", "gain_IC_react", "recommend"]
    ]
    for row in eff.itertuples():
        skill = (
            "  -  "
            if row.gold_skill is None or row.gold_skill != row.gold_skill
            else f"{row.gold_skill:.2f}"
        )
        _line(
            f"   {row.question:16s} {row.tag:9s} gold skill {skill}  reaction gain {row.gain_IC_react:+.3f}  -> {row.recommend}"
        )

    _line(
        "\n3. Question-improvement loop (free proposer; judged on dev, confirmed once on test)"
    )
    lc = textloop.LoopConfig(
        max_iter=3 if quick else 6,
        n_boot=50 if quick else 200,
        out_dir=tmp / "text_loop",
    )
    lp = textloop.run(sets, qsets, reader, textloop.KeywordProposer(), lc)
    for row in lp["log"].itertuples():
        _line(
            f"   {row.iteration}. {row.hypothesis[:60]:60s} dS {row.dS:+.3f} (low {row.lo:+.3f}) {'accepted' if row.accepted else 'rejected'}"
        )
    conf = textloop.confirm(sets, qsets, lp["best_qsets"], reader, lc, "demo")
    out["loop"] = {
        "accepted": lp["accepted"],
        "S_dev": lp["best"]["summary"]["S"],
        "confirm": conf,
    }
    _line(
        f"   stop: {lp['stop']}; test split: S {conf['S_frozen']:.3f} -> {conf['S_best']:.3f} ({'pass' if conf['pass'] else 'fail'})"
    )

    _line(
        "\n4. Tracking and the loop coordinator (quarterly cycles; acceptance on held-out entities)"
    )
    feats = st.feature_set("numbers_text")
    meta = st.sources["releases_text"].source.feature_meta()
    co = loops.Coordinator(
        st, feats, meta, lambda v: rows, loops.LoopsConfig(), registry=st.registry
    )
    cuts = pd.date_range("2012-04-01", st.periods.tuning[1], freq="QS", tz="UTC")
    if quick:
        cuts = cuts[:6]
    log = co.run(cuts)
    first = co.reports[0]["flags"]
    flagged = first[(first["status"] != "ok") | first["non_linear"]]
    for row in flagged.itertuples():
        what = "non-linear by level" if row.non_linear else row.status
        _line(
            f"   first report: {row.field} {what} (joint t {row.t_joint:+.1f}, one-at-a-time t {row.t_one_at_a_time:+.1f})"
        )
    for e in log.itertuples():
        if e.loop != "brake":
            _line(
                f"   cycle {e.cycle}: {e.loop} {e.rung} {e.change[:50]} t {e.t:.2f} vs bar {e.bar:.2f} -> {'accepted' if e.accepted else 'rejected'}"
            )
        else:
            _line(f"   cycle {e.cycle}: brake ({e.rung}) {e.field}: {e.trigger}")
    out["coordinator"] = {"config": co.cur.key(), "judged": co.judged}
    _line(f"   final model config: {co.cur.key()}")

    _line(
        "\n5. Decider with the feedback note (free model-pick decider; Jev or Claude with keys)"
    )
    prep = dec_mod.prepare(st, "numbers_text")
    notes = []
    d = feedback.run(
        decider.NoDecider(),
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
        on_batch=lambda j: notes.append(j["state"]["reliability_note"]),
    )
    g = feedback.grade(d)
    out["decider"] = g
    _line(
        f"   {g['batches']} batches over {g['periods']} periods; model pick vs batch, net of costs: t {g['model_vs_batch_net_t']:.1f}"
    )
    _line("   last feedback note:")
    for ln in notes[-1].splitlines():
        _line("     " + ln)
    reg = st.registry.summary()
    _line(
        f"\nRegistry (demo, temporary): {reg['judged_tests']} judged tests; next bar t >= {reg['next_bar']}. State in {tmp}"
    )
    out["registry"] = reg
    return out
