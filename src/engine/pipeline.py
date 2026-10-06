"""The standard run, wired from a Study: panel -> rows -> walk-forward scores -> the scorer.

Every period is gated here, not by convention:
  tuning   always open
  check    needs a CheckGrant from the registry (counted, limited)
  holdout  needs final=True, which logs the look in the unlock log first
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine import models, panel, pit, sample, scoring
from engine.registry import HoldoutLocked


@dataclass(frozen=True)
class CheckGrant:
    """Proof that the registry opened the check period for one test (registry.open_check)."""

    bar: float


def stage_end(study, stage: str, grant: CheckGrant | None = None, final: bool = False):
    p = study.periods
    if stage == "tuning":
        return p.tuning[1]
    if stage == "check":
        if not isinstance(grant, CheckGrant):
            raise PermissionError(
                "the check period opens only through registry.open_check"
            )
        return p.check[1]
    if stage == "holdout":
        if not final:
            raise HoldoutLocked(
                "the holdout is locked; a person runs it once with --final"
            )
        return pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    raise ValueError(stage)


def build_panel(
    study,
    stage: str = "tuning",
    grant=None,
    final=False,
    sources=None,
    with_labels=True,
):
    end = stage_end(study, stage, grant, final)
    spec = study.panel_spec(study.periods.tuning[0], end, sources)
    p = panel.build(
        study.market, spec, cache_dir=study.cache_dir, with_labels=with_labels
    )
    pit.check_panel(p)  # a leak stops the run here
    return p


def rows(study, p: panel.Panel, legacy: bool = False) -> pd.DataFrame:
    """Labelled rows ready for a model: winsorized label, fwd_rank target, round-trip costs."""
    lab = study.cfg["labels"]
    f = sample.usable_rows(p.frame)
    if lab.get("winsorize"):
        f = sample.winsorize(f, "fwd_return", tuple(lab["winsorize"]))
    if legacy:
        b = study.cfg["model"]["legacy"]["batches"]
        f = sample.make_batches(
            f, int(b["size"]), b.get("by", "group"), int(b.get("seed", 0))
        )
    within = study.cfg["model"].get("target", {}).get("within", "group")
    f = sample.add_rank_target(f, within)
    f["rt_cost"] = study.market.cost_bps(f) / 1e4
    return f


def walk_forward(study, legacy: bool = False, model=None) -> models.WalkForward:
    m = study.cfg["model"]
    factory = model or (
        models.ridge(float(m.get("alpha", 10.0)))
        if m.get("name", "ridge") == "ridge"
        else models.MODELS[m["name"]]()
    )
    return models.WalkForward(
        target="fwd_rank",
        demean_by=("batch",) if legacy else ("decision_time", "group"),
        model=factory,
        block=m.get("block", "year"),
        min_train_periods=int(m.get("min_train_periods", 12)),
        embargo=pd.Timedelta(days=int(m.get("embargo_days", 0))),
        legacy_purge_days=int(m["legacy"]["purge_days"]) if legacy else None,
        keep=("group", "fwd_return", "rt_cost"),
        half_life=m.get("half_life"),
    )


def in_stage(study, scored: pd.DataFrame, stage: str) -> pd.DataFrame:
    local = study.market.calendar.local_date(scored["decision_time"])
    lo, hi = study.periods.tuning if stage == "tuning" else study.periods.check
    if stage == "holdout":
        lo, hi = study.periods.holdout_start, pd.Timestamp.max
    return scored[((local >= lo) & (local < hi)).to_numpy()].reset_index(drop=True)


def portfolio_rules(study) -> dict:
    rules = {}
    for name, r in (study.cfg.get("portfolio") or {}).items():
        if name == "band":
            rules[name] = scoring.Band(**r)
        elif name == "topk":
            rules[name] = scoring.TopK(**r)
        elif name == "periodic":
            rules[name] = scoring.Periodic(**r)
    return rules


def evaluate(study, scored: pd.DataFrame, stage: str = "tuning") -> dict:
    s = scoring.fill_costs(in_stage(study, scored, stage))
    return scoring.evaluate(s, portfolio_rules(study), overlap_n=overlap(study))


def overlap(study) -> int:
    per = {"monthly": 21, "weekly": 5, "daily": 1}.get(study.cfg.get("schedule"), 1)
    if getattr(study.market.calendar, "kind", "trading") == "continuous":
        per = {"monthly": 30, "weekly": 7, "daily": 1}.get(study.cfg.get("schedule"), 1)
    return scoring.overlap(int(study.cfg["labels"]["horizon"]), per)
