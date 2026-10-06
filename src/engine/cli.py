"""engine demo | fetch|build|report|test|discover|parity|decide|text-eval|spend --market <name>

demo      the whole system on a synthetic market: no keys, no downloads

fetch     each source fills its own cache (the only step that may use the network)
build     the tuning-period panel, point-in-time checked; prints coverage per feature
report    the baseline model on tuning periods: rank IC, spreads, portfolios (descriptive, not
          logged), plus the registry: tests so far, next bars, check uses, holdout looks.
          --final opens the holdout ONCE and logs the look first
test      one pre-registered candidate (--feature, --transform, --scope), judged and logged
discover  the discovery loop over the config's candidates (or every feature x transform)
parity    reproduce the pilot's recorded numbers (us_smallcap; never logged)
decide    a decider over the model's cards with the feedback note (--estimate first; --log)
text-eval the text eval harness on the market's eval sets (needs the plug-in's eval_sets)
spend     the spend ledger: spent vs cap per step
replay    loops ON vs FROZEN month by month over tuning years (--estimate first; --log = ONE test)
cohort    the YAML's cohort_test (overlapping long-horizon cohorts): --coverage (no returns), the
          design (descriptive), --log (ONE test), or variants --hold / --top / --input --spread /
          --names (descriptive, never logged)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from engine import config, discovery, parity, pipeline


def _print(obj) -> None:
    print(
        json.dumps(
            obj,
            indent=1,
            default=lambda x: round(float(x), 5) if isinstance(x, float) else str(x),
        )
    )


def cmd_fetch(study, args) -> None:
    lo, hi = study.periods.tuning[0], study.periods.check[1]
    for name, s in study.sources.items():
        print(f"fetch {name}", file=sys.stderr)
        s.source.fetch(lo - pd.Timedelta(days=s.lookback_days), hi)


def cmd_build(study, args) -> None:
    p = pipeline.build_panel(study, "tuning")
    f = p.frame
    cover = {c: round(float(f[c].notna().mean()), 3) for c in p.features}
    _print(
        {
            "rows": len(f),
            "decision_times": int(f["decision_time"].nunique()),
            "entities": int(f["entity_id"].nunique()),
            "labelled": int(f["fwd_return"].notna().sum()),
            "coverage": cover,
        }
    )


def _scored(study, features: list[str], legacy: bool):
    p = parity.legacy_panel(study) if legacy else pipeline.build_panel(study, "tuning")
    rows = pipeline.rows(study, p, legacy=legacy)
    wf = pipeline.walk_forward(study, legacy=legacy)
    return wf.run(rows, features, study.market.calendar)


def cmd_report(study, args) -> None:
    feats = study.feature_set(args.features)
    if args.final:
        reason = input("Reason for opening the holdout (logged): ").strip()
        study.holdout.unlock(study.name, reason)
        p = pipeline.build_panel(study, "holdout", final=True)
        rows = pipeline.rows(study, p)
        scored, _ = pipeline.walk_forward(study).run(rows, feats, study.market.calendar)
        _print({"holdout": pipeline.evaluate(study, scored, "holdout")})
        return
    scored, weights = _scored(study, feats, args.legacy)
    ev = pipeline.evaluate(study, scored, "tuning")
    avg = weights.mean().sort_values() if len(weights) else pd.Series(dtype=float)
    _print(
        {
            "features": args.features,
            "tuning": ev,
            "average_weights_x100": (pd.concat([avg.head(5), avg.tail(5)]) * 100)
            .round(2)
            .to_dict(),
            "registry": study.registry.summary(),
            "holdout_looks": study.holdout.looks(),
        }
    )


def cmd_test(study, args) -> None:
    p = pipeline.build_panel(study, "tuning")
    rows = pipeline.rows(study, p)
    c = discovery.Candidate(args.feature, args.transform, args.scope)
    _print(
        discovery.test_candidate(
            study, rows, study.feature_set("baseline"), c, note=args.note or ""
        )
    )


def cmd_discover(study, args) -> None:
    p = pipeline.build_panel(study, "tuning")
    rows = pipeline.rows(study, p)
    _print(discovery.run(study, rows, p.features, max_tests=args.max_tests))


def cmd_parity(study, args) -> None:
    res = parity.run(study, config.path(f"data/engine/{study.name}/parity.json"))
    print(parity.table(res))


def cmd_decide(study, args) -> None:
    from engine import decide

    decide.main_decide(study, args)


def cmd_text_eval(study, args) -> None:
    from engine import markets
    from engine.text import harness, readers
    from engine.text import questions as tq

    plugin = markets.load(study.cfg.get("plugin", study.name))
    if not hasattr(plugin, "eval_sets"):
        raise SystemExit(f"{study.name}: the plug-in has no eval_sets()")
    t = study.cfg["text"]
    sets = plugin.eval_sets(study.market, end=study.periods.tuning[1])
    qsets = tq.load(config.path(t["questions"]))
    res = harness.score(
        sets, qsets, readers.make(t.get("reader")), args.split, efficacy=True
    )
    _print({k: v for k, v in res.items() if not k.startswith("_") and k != "efficacy"})
    print(res["efficacy"].to_string(index=False))


def cmd_replay(study, args) -> None:
    from engine import markets

    plugin = markets.load(study.cfg.get("plugin", study.name))
    try:
        from importlib import import_module

        mod = import_module(plugin.__name__ + ".replay_text")
    except ImportError:
        raise SystemExit(
            f"{study.name}: no replay adapter (engine/markets/<m>/replay_text.py)"
        )
    mod.main(study, args)


def cmd_cohort(study, args) -> None:
    from engine import cohort_study

    variant = any(
        v is not None for v in (args.hold, args.top, args.input, args.names)
    ) or bool(args.spread)
    if args.log and (variant or args.coverage):
        raise SystemExit("--log runs the pre-registered design only: no variants")
    cs = cohort_study.Study(study)
    if args.coverage:
        print(cs.coverage().round(3).to_string())
        return
    if args.names:
        y = [int(x) for x in (args.years or "2013-2016").split("-")]
        print(cs.ranks_of(args.names.split(","), (y[0], y[-1])).to_string(index=False))
        return
    res = cs.run(
        top=args.top,
        hold=args.hold,
        inputs=[args.input] if args.input else None,
        spread=args.spread,
    )
    m = res.pop("_monthly")
    if args.log:
        res["registry"] = cohort_study.log(study, res, args.note or "")
    _print(res)
    if args.monthly_out:
        m.to_csv(args.monthly_out)


def cmd_spend(study, args) -> None:
    from engine.spend import Ledger

    print(Ledger.from_config(study.cfg.get("spend"), config.ROOT).report().to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="engine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "cmd",
        choices=[
            "demo",
            "fetch",
            "build",
            "report",
            "test",
            "discover",
            "parity",
            "decide",
            "text-eval",
            "spend",
            "replay",
            "cohort",
        ],
    )
    ap.add_argument("--market")
    ap.add_argument("--config", help="YAML path (default markets/<market>.yaml)")
    ap.add_argument(
        "--features", default="baseline", help="named feature set (report, decide)"
    )
    ap.add_argument(
        "--legacy", action="store_true", help="the pilot's sampling quirks (report)"
    )
    ap.add_argument(
        "--final", action="store_true", help="open the holdout once (report; logged)"
    )
    ap.add_argument("--feature")
    ap.add_argument("--transform", default="level")
    ap.add_argument("--scope", default="universal")
    ap.add_argument("--note")
    ap.add_argument("--max-tests", type=int)
    ap.add_argument(
        "--decider", help="decide: none | jev | claude (default: the YAML's)"
    )
    ap.add_argument("--years", help="decide: decision years, e.g. 2013-2019")
    ap.add_argument("--estimate", action="store_true", help="decide: cost only")
    ap.add_argument(
        "--log", action="store_true", help="decide: log the result as ONE test"
    )
    ap.add_argument(
        "--no-feedback", action="store_true", help="decide: without the note"
    )
    ap.add_argument("--step", help="decide: spend-ledger step")
    ap.add_argument("--split", default="dev", help="text-eval: dev | test")
    ap.add_argument("--quick", action="store_true", help="demo: fewer iterations")
    ap.add_argument("--refit", help="replay: M (monthly) | year")
    ap.add_argument("--recency", help="replay: auto | none")
    ap.add_argument("--coverage", action="store_true", help="cohort: inputs only")
    ap.add_argument("--hold", type=int, help="cohort: quarters held (variant)")
    ap.add_argument("--top", type=float, help="cohort: fraction or count (variant)")
    ap.add_argument("--input", help="cohort: one input instead of the composite")
    ap.add_argument("--spread", action="store_true", help="cohort: top minus bottom")
    ap.add_argument("--names", help="cohort: codes whose ranks to show, e.g. NVDA,MU")
    ap.add_argument("--monthly-out", help="cohort: write the monthly series here")
    ap.add_argument(
        "--min-funds-left",
        type=float,
        help="replay: stop if TypeSafe funds would fall below",
    )
    args = ap.parse_args(argv)
    os.chdir(config.ROOT)  # plug-ins read paths relative to the repo root
    if args.cmd == "demo":
        from engine import demo

        demo.run(quick=args.quick)
        return
    if not args.market:
        ap.error("--market is required")
    if args.market in ("us_smallcap", "us_largecap"):
        from engine.markets.us_smallcap.env import load_env

        load_env()
    study = config.load(args.market, args.config)
    {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "report": cmd_report,
        "test": cmd_test,
        "discover": cmd_discover,
        "parity": cmd_parity,
        "decide": cmd_decide,
        "text-eval": cmd_text_eval,
        "spend": cmd_spend,
        "replay": cmd_replay,
        "cohort": cmd_cohort,
    }[args.cmd](study, args)


if __name__ == "__main__":
    main()
