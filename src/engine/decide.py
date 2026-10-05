"""Run a decider over the model's cards on tuning periods, with the feedback note (on by default).

    uv run engine decide --market us_smallcap --estimate     # cost first; no calls
    uv run engine decide --market us_smallcap --decider jev --log   # the paid run; logs ONE test

Batches of `size` same-group entities per decision time; the decider picks one per batch. Graded
per decision period against the model's own top pick on the same batches, gross and net of the
market's measured round-trip costs (each pick pays its own cost; the batch average pays the
average), t from the period means.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from engine import decider as dmod
from engine import feedback, models, pipeline
from engine.spend import Ledger

NOTE_ALLOWANCE = (
    800  # chars a grown override record may add to a note (estimate is worst case)
)


def text_meta(study) -> dict:
    out = {}
    for spec in study.sources.values():
        if hasattr(spec.source, "feature_meta"):
            out |= spec.source.feature_meta()
    return out


def labels(study, meta: dict) -> dict:
    lab = {}
    for spec in study.sources.values():
        qsets = getattr(spec.source, "qsets", {})
        for c, m in meta.items():
            qs = qsets.get(m["doc_type"])
            if qs is not None and m["question"] in qs.ids():
                q = qs.get(m["question"])
                part = "" if m["part"] in ("p", "level") else f" [{m['part']}]"
                enc = {
                    "change": " (change vs its previous documents)",
                    "surprise": " (vs history base rate)",
                }.get(m["encoding"], "")
                lab[c] = f"text: {q.prompt}{part}{enc}"
    return lab | study.cfg.get("feature_labels", {})


def prepare(
    study,
    feature_set: str = "baseline",
    years: tuple[int, int] | None = None,
    size: int = 10,
    seed: int = 0,
) -> dict:
    p = pipeline.build_panel(study, "tuning")
    rows = pipeline.rows(study, p)
    feats = study.feature_set(feature_set)
    wf = pipeline.walk_forward(study)
    scored, w = wf.run(rows, feats, study.market.calendar)
    scored = pipeline.in_stage(study, scored, "tuning")
    if years:
        y = study.market.calendar.local_date(scored["decision_time"]).dt.year
        scored = scored[(y >= years[0]) & (y <= years[1])].reset_index(drop=True)
    b = dmod.batches(scored, size, seed)
    keyed = rows.set_index(["entity_id", "decision_time"])
    idx = list(zip(b["entity_id"], b["decision_time"]))
    raw = keyed.loc[idx, feats].reset_index(drop=True)
    contrib_rows = models.contributions(rows, feats, w, study.market.calendar)
    contrib = (
        contrib_rows.set_index(
            pd.MultiIndex.from_frame(
                rows.loc[contrib_rows.index, ["entity_id", "decision_time"]]
            )
        )
        .loc[idx]
        .reset_index(drop=True)
    )
    b = b.reset_index(drop=True)
    b["rt_cost"] = b["rt_cost"].fillna(b["rt_cost"].median())
    meta = {c: m for c, m in text_meta(study).items() if c in feats}
    return {
        "rows": rows,
        "scored": b,
        "raw": raw,
        "contrib": contrib,
        "features": feats,
        "meta": meta,
        "labels": labels(study, meta),
    }


class _Sizer:
    """A free stand-in decider that records how big each prompt would be."""

    name = "sizer"

    def __init__(self):
        self.chars = []

    def pick(self, job: dict) -> dict:
        self.chars.append(
            len(json.dumps(job["state"])) + len(dmod.QUESTION) + NOTE_ALLOWANCE
        )
        return {job["model_pick"]: 1.0}


def estimate(prep: dict, ledger: Ledger, model: str = "jev-1.13.0") -> dict:
    sizer = _Sizer()
    feedback.run(
        sizer,
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
    )
    usd = ledger.price_chars(model, float(np.sum(sizer.chars)))
    return {
        "batches": len(sizer.chars),
        "chars": int(np.sum(sizer.chars)),
        "projected_usd": usd,
    }


def log_test(study, g: dict, name: str, note: str) -> dict:
    reg = study.registry
    bar = reg.next_bar()
    return reg.record(
        {
            "kind": "test",
            "name": name,
            "features": "baseline",
            "scope": "universal",
            "metric": "pick-1-of-10 monthly excess net of measured costs, decider minus model top pick",
            "t_tune": round(g["decider_minus_model_net_t"], 3),
            "bar_tune": round(bar, 3),
            "gain_tune": round(g["decider_minus_model_net"], 6),
            "check_used": False,
            "kept": bool(
                g["decider_minus_model_net_t"] >= bar
                and g["decider_minus_model_net"] > 0
            ),
            "note": note,
        }
    )


def main_decide(study, args) -> None:
    cfg = study.cfg.get("decider") or {"kind": "none"}
    kind = args.decider or cfg.get("kind", "none")
    years = tuple(int(x) for x in args.years.split("-")) if args.years else None
    prep = prepare(study, args.features, years)
    ledger = Ledger.from_config(study.cfg.get("spend"))
    step = args.step or cfg.get("step", "decider")
    model = (
        cfg.get("model", "jev-1.13.0")
        if kind == "jev"
        else cfg.get("model", "claude-sonnet-5-5")
    )
    if kind != "none":
        est = estimate(prep, ledger, model)
        est |= {
            "step": step,
            "cap": ledger.step_caps.get(step),
            "already_spent": ledger.spent(step),
        }
        print(json.dumps(est, indent=1), file=sys.stderr)
        if args.estimate:
            return
        if est["projected_usd"] > ledger.remaining(step):
            raise SystemExit(
                f"estimate ${est['projected_usd']:.2f} is over what's left in {step}: not running"
            )
    dec = dmod.make(
        {
            "kind": kind,
            **{k: v for k, v in cfg.items() if k not in ("kind", "feedback", "step")},
            **({"step": step} if kind != "none" else {}),
        },
        ledger if kind != "none" else None,
    )
    fb = bool(cfg.get("feedback", True)) and not args.no_feedback
    out = feedback.run(
        dec,
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
        feedback=fb,
    )
    g = feedback.grade(out)
    path = study.cache_dir / f"decisions_{kind}{'_fb' if fb else ''}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    res = {"decider": kind, "feedback": fb, "grade": g, "decisions": str(path)}
    if args.log:
        res["registry"] = log_test(
            study,
            g,
            f"decider:{kind}{'+feedback' if fb else ''} vs model top pick",
            args.note or "",
        )
    print(json.dumps(res, indent=1, default=str))
