"""The `engine` command: seven commands, and `engine demo`, the whole system on a synthetic market.

  demo        everything on the demo market (numbers + documents with planted signals), ending with
              the feedback loop at work: no keys, no downloads, about 2.5 minutes
  loop        THE FEEDBACK LOOP over the tuning years, period by period, as if live: weights from
              closed returns (outer loop), cards and the decider's pick, the period's returns, the
              re-weighting, and text fixes when a field stays mis-weighted (inner loop); against a
              FROZEN twin. Writes results/<market>/loop_weights.csv, loop_changes.csv,
              loop_returns.csv. Descriptive; --log records ONE test under the YAML's loop.name
  build       the tuning-period panel, point-in-time checked; prints coverage per feature.
              --fetch first lets each source fill its cache (the only step that uses the network)
  report      a feature set's model on tuning periods: rank IC, spreads, portfolios net of costs,
              plus the registry (tests so far, next bar, check uses, holdout looks). Descriptive,
              never logged. --decider runs an AI decider over the model's cards (descriptive);
              --final opens the holdout ONCE, asks for a reason and logs the look first
  test        ONE judged test, logged whatever the result: --feature (+ --transform, --scope)
              against the baseline; --decider (decider vs the model's own pick); with neither, the
              YAML's pre-registered cohort_test. The check period opens only if tuning clears the bar
  discover    the discovery loop: candidates from the YAML (or every feature x transform), each a
              logged test, until one is kept or a stop rule fires
  grade-text  the text eval harness on the market's eval sets: S = 0.30 A + 0.25 B + 0.30 C +
              0.10 D + 0.05 E, the leak gate, and a keep / drop per question

Paid steps (--decider jev | claude, paid readers in the YAML) print a cost estimate first and stop
at their spend cap; --estimate prints only the estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine import decide, improve, markets, model, panel, run
from engine.text import grade, read
from engine.text import improve as text_improve
from engine.text import questions as tq

COMMANDS = ["demo", "loop", "build", "report", "test", "discover", "grade-text"]


def load_env(path: Path | None = None) -> None:
    """KEY=VALUE lines from the repo's .env into the environment (never printed, never committed)."""
    path = path or run.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print(obj) -> None:
    print(
        json.dumps(
            obj, indent=1, default=lambda x: round(float(x), 5) if isinstance(x, float) else str(x)
        )
    )


def cmd_build(study, args) -> None:
    """Panel + point-in-time check + coverage (after fetching, with --fetch)."""
    if args.fetch:
        run.fetch(study)
    p = run.build_panel(study, "tuning")
    f = p.frame
    _print(
        {
            "rows": len(f),
            "decision_times": int(f["decision_time"].nunique()),
            "entities": int(f["entity_id"].nunique()),
            "labelled": int(f["fwd_return"].notna().sum()),
            "coverage": {c: round(float(f[c].notna().mean()), 3) for c in p.features},
        }
    )


def cmd_report(study, args) -> None:
    """Descriptive report; the holdout only with --final."""
    if args.final:
        reason = input("Reason for opening the holdout (logged): ").strip()
        _print(run.report_holdout(study, args.features, reason))
    elif args.decider:
        _print(decide.run_decider(study, args.decider, args.features, estimate_only=args.estimate))
    else:
        _print(run.report(study, args.features))


def cmd_test(study, args) -> None:
    """One logged test: a feature, a decider, or the YAML's cohort test."""
    note = args.note or ""
    if args.decider:
        _print(
            decide.run_decider(
                study,
                args.decider,
                args.features,
                estimate_only=args.estimate,
                log=not args.estimate,
                note=note,
            )
        )
    elif args.feature:
        rows = panel.model_rows(study, run.build_panel(study, "tuning"))
        c = run.Candidate(args.feature, args.transform, args.scope)
        _print(run.test_candidate(study, rows, study.feature_set("baseline"), c, note=note))
    elif study.cfg.get("cohort_test"):
        if run.cohort_test_logged(study):
            raise SystemExit("the cohort test is already in the registry: one look only")
        res = run.CohortTest(study).run()
        res.pop("_monthly")
        res["registry"] = run.log_cohort_test(study, res, note)
        _print(res)
    else:
        raise SystemExit("test needs --feature or --decider (or a cohort_test block in the YAML)")


def cmd_loop(study, args) -> None:
    """The feedback loop, period by period; descriptive unless --log."""
    res = improve.run_loop(
        study, args.decider, log=args.log, note=args.note or "", estimate_only=args.estimate
    )
    if "result" not in res:  # --estimate
        _print(res)
        return
    print(loop_summary(res))
    if "registry" in res:
        _print(res["registry"])


def cmd_discover(study, args) -> None:
    """The discovery loop; every candidate tried is a logged test."""
    p = run.build_panel(study, "tuning")
    rows = panel.model_rows(study, p)
    _print(run.discover(study, rows, p.features, max_tests=args.max_tests))


def cmd_grade_text(study, args) -> None:
    """The text eval harness on the plug-in's eval sets, with the YAML's question set and reader."""
    plugin = markets.load(study.cfg.get("plugin", study.name))
    if not hasattr(plugin, "eval_sets"):
        raise SystemExit(f"{study.name}: the plug-in has no eval_sets()")
    t = study.cfg["text"]
    sets = plugin.eval_sets(study.market, end=study.periods.tuning[1])
    qsets = tq.load(run.path(t["questions"]))
    res = grade.score(sets, qsets, read.make_reader(t.get("reader")), args.split, efficacy=True)
    _print({k: v for k, v in res.items() if not k.startswith("_") and k != "efficacy"})
    print(res["efficacy"].to_string(index=False))


def loop_summary(res: dict, top: int = 5, rows: int = 8) -> str:
    """The `engine loop` printout: how the top inputs' weights moved, the changes, ON vs FROZEN."""
    w, ch, r, s = res["_weights"], res["_changes"], res["result"], res["settings"]
    ret = res["_returns"]
    days = [pd.Timestamp(t).date() for t in ret.index[[0, -1]]] if len(ret) else ["-"] * 2
    logged = (
        "logged as ONE test" if "registry" in res else "descriptive, not logged (--log: one test)"
    )
    head = (
        f"The feedback loop on {res['market']}: {len(ret)} periods, {days[0]} to {days[1]} "
        f"({r['periods']} with both ON and FROZEN returns)"
    )
    lines = [f"{head}; refit {s['refit']}, recency {s['recency']}; {logged}"]
    if len(w):
        last = w[w["period"] == w["period"].max()].set_index("input")["weight"]
        inputs = last.abs().sort_values(ascending=False).index[:top].tolist()
        tab = w[w["input"].isin(inputs)].pivot(index="period", columns="input", values="weight")
        tab = tab.reindex(columns=inputs)
        pick = np.unique(np.linspace(0, len(tab) - 1, min(rows, len(tab))).round().astype(int))
        lines.append("\nWeights of the top inputs (x100), period by period:")
        lines.append(f"{'period':10s}" + "".join(f"{c[-20:]:>21s}" for c in inputs))
        for t, row in tab.iloc[pick].iterrows():
            vals = "".join(f"{v * 100:>21.2f}" if pd.notna(v) else f"{'-':>21s}" for v in row)
            lines.append(f"{pd.Timestamp(t).date()!s:10s}{vals}")
    judged = ch[ch["loop"] != "brake"] if len(ch) else ch
    brakes = ch[ch["loop"] == "brake"] if len(ch) else ch
    n_acc = int(judged["accepted"].sum()) if len(judged) else 0
    lines.append(
        f"\nChanges judged end to end: {len(judged)}, accepted: {n_acc}; brakes: {len(brakes)}"
    )
    for e in judged.itertuples() if len(judged) else []:
        lines.append(
            f"  {pd.Timestamp(e.period).date()!s} {e.loop} {e.rung} {e.change[:48]}: t {e.t:.2f} vs bar "
            f"{e.bar:.2f} -> {'accepted' if e.accepted else 'rejected'}"
        )
    for e in brakes.itertuples() if len(brakes) else []:
        lines.append(
            f"  {pd.Timestamp(e.period).date()!s} brake ({e.rung}) on {e.field}: {e.trigger}"
        )
    splits = res.get("splits", [])
    if splits:
        tried = ", ".join(x["prompt"].removeprefix("The text says: ").rstrip(".") for x in splits)
        n_q = int((judged["rung"] == "question").sum()) if len(judged) else 0
        lines.append(
            f"  question splits proposed: {len(splits)} ({tried}); {n_q} helped enough on the "
            "diagnosis entities to be judged"
        )
    lines.append(
        f"\nON vs FROZEN, V1 portfolio net of costs: {r['diff_net']:+.3%}/period (t {r['diff_net_t']:.2f});"
        f" rank IC {r['ic_on']:.4f} vs {r['ic_frozen']:.4f} (diff t {r['diff_ic_t']:.2f})"
    )
    if "decider" in res:
        d = res["decider"]
        lines.append(
            f"Decider ({d['kind']}): pick minus the model's pick, net {d['decider_minus_model_net']:+.3%}"
            f"/period; overrides {d['override_rate']:.0%}"
        )
    files = [Path(f) for f in res["files"].values()]
    shown = [str(f.relative_to(run.ROOT)) if f.is_relative_to(run.ROOT) else str(f) for f in files]
    lines.append("Files: " + ", ".join(shown))
    return "\n".join(lines)


# ---------------------------------------------------------------- engine demo
def _line(s: str = "") -> None:
    print(s, flush=True)


def run_demo(quick: bool = False) -> dict:
    """The whole system on the demo market; state goes to a temporary directory, so it can run
    any number of times. Returns the numbers it prints.

    1. numbers vs numbers + text: the walk-forward model and the scorer
    2. text scoring with the FREE keyword reader: answer key, probes, reaction and drift beyond a
       history prior, the leak gate, per-question efficacy
    3. the question-improvement loop (free proposer), confirmed once on the test split
    4. tracking + the loop coordinator: what is mis-weighted or mis-shaped, and what got fixed
    5. the decider layer with its feedback note (the free model-pick decider)
    6. the feedback loop, period by period: a planted weight rising, a stale one corrected, the trap
    """
    from engine.markets import demo as demo_market

    tmp = Path(tempfile.mkdtemp(prefix="engine_demo_"))
    cfg = run.read_config("demo")
    cfg["cache_dir"] = str(tmp / "cache")
    cfg["registry"] = {
        "file": str(tmp / "registry.csv"),
        "holdout_unlock_log": str(tmp / "unlocks.csv"),
    }
    st = run.study_from_config("demo", cfg)
    out = {}

    _line("1. Model: numbers vs numbers + text (tuning periods, rank IC and net long-short)")
    rows = panel.model_rows(st, run.build_panel(st, "tuning"))
    for fs in ("numbers", "numbers_text"):
        sc, _ = run.walk_forward(st).run(rows, st.feature_set(fs), st.market.calendar)
        ev = run.evaluate(st, sc)
        out[fs] = ev
        _line(
            f"   {fs:13s} IC {ev['ic']:+.3f} (t {ev['ic_t']:.1f})   decile spread net {ev['spread']['net']:+.2%}/month (t {ev['spread']['net_t']:.1f})"
        )

    _line("\n2. Text grader: is each question worth asking? (free keyword reader, dev split)")
    sets = demo_market.eval_sets(st.market, end=st.periods.tuning[1])
    qsets = tq.load(run.path(cfg["text"]["questions"]))
    reader = read.KeywordReader()
    res = grade.score(sets, qsets, reader, "dev", efficacy=True)
    s = res["summary"]
    out["text"] = s
    _line(
        f"   S {s['S']:.3f} = 0.30 A {s['A']:.2f} + 0.25 B {s['B']:.2f} + 0.30 C {s['C']:.2f} + 0.10 D {s['D']:.2f} + 0.05 E {s['E']:.2f}"
    )
    r = res["reaction"]
    _line(
        f"   reaction IC: prior {r['prior']['IC_react']:.3f} -> card {r['card']['IC_react']:.3f}; drift IC gain {r['gain_vs_prior']['IC_drift']:+.3f}; leak gate {'pass' if res['gates']['leak']['pass'] else 'FAIL'}"
    )
    eff = res["efficacy"][["question", "tag", "gold_skill", "gain_IC_react", "recommend"]]
    for row in eff.itertuples():
        skill = (
            "  -  "
            if row.gold_skill is None or row.gold_skill != row.gold_skill
            else f"{row.gold_skill:.2f}"
        )
        _line(
            f"   {row.question:16s} {row.tag:9s} gold skill {skill}  reaction gain {row.gain_IC_react:+.3f}  -> grader: {row.recommend}"
        )

    _line("\n3. Question-improvement loop (free proposer; judged on dev, confirmed once on test)")
    lc = text_improve.LoopConfig(
        max_iter=3 if quick else 6, n_boot=50 if quick else 200, out_dir=tmp / "text_loop"
    )
    lp = text_improve.run_loop(sets, qsets, reader, text_improve.KeywordProposer(), lc)
    for row in lp["log"].itertuples():
        _line(
            f"   {row.iteration}. {row.hypothesis:62s} dS {row.dS:+.3f} (low {row.lo:+.3f}) {'accepted' if row.accepted else 'rejected'}"
        )
    conf = text_improve.confirm(sets, qsets, lp["best_qsets"], reader, lc, "demo")
    out["loop"] = {"accepted": lp["accepted"], "S_dev": lp["best"]["summary"]["S"], "confirm": conf}
    _line(
        f"   stop: {lp['stop']}; test split: S {conf['S_frozen']:.3f} -> {conf['S_best']:.3f} ({'pass' if conf['pass'] else 'fail'})"
    )

    _line(
        "\n4. Tracking and the loop coordinator (quarterly cycles; acceptance on held-out entities)"
    )
    feats = st.feature_set("numbers_text")
    meta = st.sources["releases_text"].source.feature_meta()
    co = improve.Coordinator(
        st, feats, meta, lambda v: rows, improve.LoopsConfig(), registry=st.registry
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

    _line("\n5. Decider with the feedback note (free model-pick decider; Jev or Claude with keys)")
    prep = decide.prepare_cards(st, "numbers_text")
    notes = []
    d = decide.run_batches(
        decide.NoDecider(),
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
        on_batch=lambda j: notes.append(j["state"]["reliability_note"]),
    )
    g = decide.grade_decisions(d)
    out["decider"] = g
    _line(
        f"   {g['batches']} batches over {g['periods']} periods; model pick vs batch, net of costs: t {g['model_vs_batch_net_t']:.1f}"
    )
    _line(
        "   last feedback note (the decider's view: which inputs and questions have a track record"
        " so far; separate from the grader's keep / drop in section 2):"
    )
    for ln in notes[-1].splitlines():
        _line("     " + ln)
    out["feedback_loop"] = demo_feedback_loop(tmp, entities=150 if quick else 300)
    reg = st.registry.summary()
    _line(
        f"\nRegistry (demo, temporary): {reg['judged_tests']} judged tests; next bar t >= {reg['next_bar']}. State in {tmp}"
    )
    out["registry"] = reg
    return out


def main(argv=None) -> None:
    """Parse the command line and run one command."""
    ap = argparse.ArgumentParser(
        prog="engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=COMMANDS)
    ap.add_argument("--market", help="a YAML in markets/ (demo, csv_example, us_smallcap, ...)")
    ap.add_argument("--config", help="a YAML path instead of markets/<market>.yaml")
    ap.add_argument("--fetch", action="store_true", help="build: fill the caches first (network)")
    ap.add_argument("--features", default="baseline", help="report, test --decider: feature set")
    ap.add_argument("--final", action="store_true", help="report: open the holdout once (logged)")
    ap.add_argument("--feature", help="test: the candidate feature")
    ap.add_argument("--transform", default="level", help="test: level | chg<k> | pct<k> | log")
    ap.add_argument("--scope", default="universal", help="test: universal or one group")
    ap.add_argument("--decider", help="loop, report, test: none | jev | claude")
    ap.add_argument("--log", action="store_true", help="loop: record it as ONE registry test")
    ap.add_argument("--estimate", action="store_true", help="paid steps: print the cost only")
    ap.add_argument("--note", help="test, loop --log: a note stored with the registry row")
    ap.add_argument("--max-tests", type=int, help="discover: at most this many tests")
    ap.add_argument("--split", default="dev", help="grade-text: dev | test")
    ap.add_argument("--quick", action="store_true", help="demo: fewer iterations")
    args = ap.parse_args(argv)
    os.chdir(run.ROOT)  # plug-ins read paths relative to the repo root
    if args.cmd == "demo":
        run_demo(quick=args.quick)
        return
    if not args.market:
        ap.error("--market is required")
    load_env()
    study = run.load_study(args.market, args.config)
    {
        "loop": cmd_loop,
        "build": cmd_build,
        "report": cmd_report,
        "test": cmd_test,
        "discover": cmd_discover,
        "grade-text": cmd_grade_text,
    }[args.cmd](study, args)


def _weight_table(w: pd.DataFrame, cols: dict, every: int = 6) -> pd.DataFrame:
    tab = w.pivot(index="period", columns="input", values="weight").reindex(columns=list(cols))
    return tab.iloc[::every].rename(columns=cols)


def demo_feedback_loop(tmp: Path, entities: int = 300) -> dict:
    """Section 6 of the demo: `engine loop` on two planted markets, 2011-2014, monthly refits."""
    from engine.markets import demo as demo_market

    out = {}
    _line(
        f"\n6. The feedback loop, period by period ({entities} entities, monthly refits, 2011-2014)"
    )
    _line("   A. x_value starts to matter in 2012; a one-time charge stops mattering")
    st = run.study_from_config(
        "demo", demo_market.loop_scenario("regime", tmp / "regime", entities)
    )
    res = improve.run_loop(st, out_dir=tmp / "regime" / "results")
    cols = {"x_value": "x_value", "txt_one_off_charge_p": "one-time charge"}
    tab = _weight_table(res["_weights"], cols)
    rows = panel.model_rows(st, run.build_panel(st, "tuning"))
    new = rows[(rows["decision_time"] >= pd.Timestamp("2012-02-01", tz="UTC"))]
    new = new[new["label_end"] < pd.Timestamp(st.periods.tuning[1], tz="UTC")]
    feats = st.feature_set("numbers_text")
    y = new["fwd_rank"] - new.groupby(["decision_time", "group"])["fwd_rank"].transform("mean")
    fit = model.ridge(10.0)().fit(model.rank_features(new, feats), y)
    true = dict(zip(feats, fit.coef_))
    _line(f"   {'period':12s}" + "".join(f"{c:>17s}" for c in tab.columns) + "   (weights x100)")
    for t, r in tab.iterrows():
        _line(f"   {pd.Timestamp(t).date()!s:12s}" + "".join(f"{v * 100:>17.1f}" for v in r))
    true_row = "".join(f"{true[c] * 100:>17.1f}" for c in cols)
    _line(f"   {'true, 2012+':12s}{true_row}   (one fit on the new regime alone)")
    acc = res["_changes"]
    for e in acc[(acc["loop"] != "brake") & acc["accepted"]].itertuples() if len(acc) else []:
        _line(
            f"   inner loop, {pd.Timestamp(e.period).date()!s}: {e.change} accepted "
            f"(t {e.t:.2f} vs bar {e.bar:.2f}): demand pays at both extremes"
        )
    r = res["result"]
    _line(f"   loop ON vs FROZEN, V1 net: {r['diff_net']:+.2%}/month (t {r['diff_net_t']:.1f})")
    out["regime"] = {"weights": tab, "true": {c: true[c] for c in cols}, "result": r}

    _line("   B. The trap: x stops mattering in 2012; text demand is 0.8 correlated with x")
    st = run.study_from_config("demo", demo_market.loop_scenario("trap", tmp / "trap", entities))
    res = improve.run_loop(st, out_dir=tmp / "trap" / "results")
    cols = {"x_value": "x_value", "txt_demand_level": "demand (text)"}
    tab = _weight_table(res["_weights"], cols)
    _line(f"   {'period':12s}" + "".join(f"{c:>17s}" for c in tab.columns) + "   (weights x100)")
    for t, r in tab.iterrows():
        _line(f"   {pd.Timestamp(t).date()!s:12s}" + "".join(f"{v * 100:>17.1f}" for v in r))
    ch = res["_changes"]
    inner = ch[(ch["loop"] == "inner") & (ch["field"] == "txt_demand_level")] if len(ch) else ch
    _line(
        f"   inner-loop changes to the demand question: {len(inner) or 'none'}; the outer refits "
        "moved x_value instead"
    )
    r = res["result"]
    _line(f"   loop ON vs FROZEN, V1 net: {r['diff_net']:+.2%}/month (t {r['diff_net_t']:.1f})")
    out["trap"] = {"weights": tab, "inner_on_demand": len(inner), "result": r}
    return out


if __name__ == "__main__":
    main()
