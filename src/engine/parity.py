"""Parity: reproduce the pilot's recorded tuning-year numbers through the engine.

Not a test: nothing here is written to the registry. Recorded values are read from the files that
logged them (data/discover_log.csv, data/eval/lowturn_v1.json), not retyped.

Two engine runs over the same point-in-time panel:
  legacy settings   the pilot's sampling quirks switched on (random batches of 10 with leftovers
                    dropped, target demeaned within batch, 31-calendar-day purge): should match
  engine defaults   every row kept, target demeaned within sector-month, purge on the real label end
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from engine import pipeline
from engine.config import path

NUMBERS_ROW = "MODEL(numbers), clean panel, fwd_rank"
TEXT_ROW = "MODEL(numbers + earnings text), clean panel, fwd_rank"
V1 = "V1 small/mid text, top-decile buy / top-30% hold, monthly"


def recorded() -> dict:
    log = pd.read_csv(path("data/discover_log.csv"))
    out = {}
    for key, name in (("numbers", NUMBERS_ROW), ("numbers_text", TEXT_ROW)):
        row = log[log["indicator"] == name].iloc[0]
        out[key] = {
            "ic": float(re.search(r"IC ([0-9.]+)", row["note"]).group(1)),
            "ic_t": float(row["t_tune"]),
        }
    v1 = json.loads(path("data/eval/lowturn_v1.json").read_text())[V1]
    out["v1"] = {k: v1[k] for k in ("gross_monthly", "gross_t", "net_monthly", "net_t")}
    return out


def legacy_panel(study):
    """The tuning panel with model.legacy.source_params applied (e.g. the pilot's tie order)."""
    over = study.cfg["model"]["legacy"].get("source_params", {})
    saved = {n: dict(study.sources[n].source.params) for n in over}
    try:
        for n, params in over.items():
            study.sources[n].source.params.update(params)
        return pipeline.build_panel(study, "tuning")
    finally:
        for n, params in saved.items():
            study.sources[n].source.params = params


def run(study, out_file: Path | None = None) -> dict:
    res = {"recorded": recorded()}
    months = (
        None  # the decision times the pilot scored; every mode is compared on these
    )
    modes = (
        ("legacy_settings", True, True),
        ("engine_defaults", False, True),
        ("engine_defaults_all_months", False, False),
    )
    for mode, legacy, same_months in modes:
        p = legacy_panel(study) if legacy else pipeline.build_panel(study, "tuning")
        rows = pipeline.rows(study, p, legacy=legacy)
        wf = pipeline.walk_forward(study, legacy=legacy)
        r = {"rows": len(rows)}
        for fs in ("numbers", "numbers_text"):
            scored, _ = wf.run(rows, study.feature_set(fs), study.market.calendar)
            if legacy and months is None:
                months = set(scored["decision_time"])
            if same_months:
                scored = scored[scored["decision_time"].isin(months)]
            ev = pipeline.evaluate(study, scored, "tuning")
            r[fs] = {"ic": ev["ic"], "ic_t": ev["ic_t"], "periods": ev["periods"]}
            if fs == "numbers_text":
                b = ev["band"]
                r["v1"] = {
                    "gross_monthly": b["gross"],
                    "gross_t": b["gross_t"],
                    "net_monthly": b["net"],
                    "net_t": b["net_t"],
                    "annual_turnover": b["annual_turnover"],
                    "avg_holding_months": b["avg_holding_periods"],
                }
        res[mode] = r
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(res, indent=1, default=float))
    return res


def table(res: dict) -> str:
    cols = [
        "recorded",
        "legacy_settings",
        "engine_defaults",
        "engine_defaults_all_months",
    ]
    heads = ["recorded", "legacy set.", "engine def.", "eng. all mo."]
    metrics = [
        ("numbers + text: rank IC", "numbers_text", "ic", "{:.4f}"),
        ("numbers + text: IC t", "numbers_text", "ic_t", "{:.2f}"),
        ("numbers only: rank IC", "numbers", "ic", "{:.4f}"),
        ("numbers only: IC t", "numbers", "ic_t", "{:.2f}"),
        ("V1 gross / month", "v1", "gross_monthly", "{:+.2%}"),
        ("V1 gross t", "v1", "gross_t", "{:.2f}"),
        ("V1 net / month", "v1", "net_monthly", "{:+.2%}"),
        ("V1 net t", "v1", "net_t", "{:.2f}"),
    ]
    lines = [f"{'':26}" + "".join(f"{h:>14}" for h in heads)]
    for name, group, key, fmt in metrics:
        vals = [fmt.format(res[c][group][key]) for c in cols]
        lines.append(f"{name:26}" + "".join(f"{v:>14}" for v in vals))
    months = [str(res[c]["numbers"].get("periods", "")) for c in cols]
    lines.append(f"{'months scored':26}" + "".join(f"{m:>14}" for m in months))
    return "\n".join(lines)
