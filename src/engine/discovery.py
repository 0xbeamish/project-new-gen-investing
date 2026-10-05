"""Discovery: propose candidate signals, judge each against the market's baseline, log every one.

A candidate = one source feature + one transform (+ an optional group scope):
  level        the value as observed
  chg<k>       change vs the same entity's value k decision periods earlier (e.g. chg3, chg12)
  pct<k>       percent change over k periods
  log          sign(x) * log(1 + |x|)
Transforms read only the panel's own earlier rows, so they inherit its point-in-time guarantee.

Judge (ported from the pilot's contrast judge, scored on rank IC instead of pick-1 returns): the same
walk-forward with and without the candidate; the statistic is the per-period difference in rank IC
(candidate minus baseline), t over tuning periods. A group scope adds a group-only copy of the
candidate so the model can give that group its own weight, and is measured on that group only.

Bars: t_tune must reach the registry's next bar (Bonferroni over every judged test so far). Only
then is the check period opened (counted, limited); kept = t_check >= the check bar and the
check-period gain is positive. The loop never sees check numbers, only pass/fail.

Proposers are plug-ins: anything with propose(study, tried) -> list[Candidate]. The default reads
the config's candidate list, or crosses every non-baseline feature with every transform. An LLM
proposer (jev.discover's prompting) can come back as one of these.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import models, pipeline, scoring


@dataclass(frozen=True)
class Candidate:
    feature: str
    transform: str = "level"
    scope: str = "universal"

    @property
    def name(self) -> str:
        return (
            self.feature
            if self.transform == "level"
            else f"{self.feature}|{self.transform}"
        )


def transform(frame: pd.DataFrame, feature: str, how: str) -> pd.Series:
    x = frame[feature].astype(float)
    if how == "level":
        return x
    if how == "log":
        return np.sign(x) * np.log1p(x.abs())
    m = re.fullmatch(r"(chg|pct)(\d+)", how)
    if not m:
        raise ValueError(f"unknown transform {how!r}")
    k = int(m.group(2))
    order = frame.sort_values("decision_time", kind="stable").index
    prev = (
        x.loc[order]
        .groupby(frame.loc[order, "entity_id"])
        .shift(k)
        .reindex(frame.index)
    )
    return x - prev if m.group(1) == "chg" else x / prev.where(prev != 0) - 1


def with_candidate(frame: pd.DataFrame, c: Candidate) -> tuple[pd.DataFrame, list[str]]:
    col = f"cand__{c.name}"
    f = frame.assign(**{col: transform(frame, c.feature, c.transform)})
    cols = [col]
    if (
        c.scope != "universal"
    ):  # a group-only copy: the group's own deviation from the shared weight
        scoped = f"{col}__{c.scope}"
        f[scoped] = f[col].where(f["group"] == c.scope)
        cols.append(scoped)
    return f, cols


def ic_gain(base: pd.DataFrame, cand: pd.DataFrame, scope: str) -> pd.Series:
    if scope != "universal":
        base, cand = base[base["group"] == scope], cand[cand["group"] == scope]
    return (scoring.rank_ic(cand) - scoring.rank_ic(base)).dropna()


def judge(
    study,
    frame: pd.DataFrame,
    baseline: list[str],
    c: Candidate,
    wf: models.WalkForward,
) -> dict:
    """Tuning-period statistics for one candidate (no bars, no logging)."""
    f, cols = with_candidate(frame, c)
    cal = study.market.calendar
    base, _ = wf.run(f, baseline, cal)
    cand, _ = wf.run(f, baseline + cols, cal)
    gain = ic_gain(
        pipeline.in_stage(study, base, "tuning"),
        pipeline.in_stage(study, cand, "tuning"),
        c.scope,
    )
    return {
        "t_tune": scoring.per_period_t(gain, pipeline.overlap(study)),
        "gain_tune": float(gain.mean()),
        "periods": len(gain),
    }


def check(
    study, baseline: list[str], c: Candidate, wf, grant: pipeline.CheckGrant
) -> dict:
    p = pipeline.build_panel(study, "check", grant=grant)
    f, cols = with_candidate(pipeline.rows(study, p), c)
    cal = study.market.calendar
    base, _ = wf.run(f, baseline, cal)
    cand, _ = wf.run(f, baseline + cols, cal)
    gain = ic_gain(
        pipeline.in_stage(study, base, "check"),
        pipeline.in_stage(study, cand, "check"),
        c.scope,
    )
    return {
        "t_check": scoring.per_period_t(gain, pipeline.overlap(study)),
        "gain_check": float(gain.mean()),
    }


def test_candidate(
    study,
    frame: pd.DataFrame,
    baseline: list[str],
    c: Candidate,
    wf: models.WalkForward | None = None,
    note: str = "",
) -> dict:
    """Judge, apply both bars, and log the result: the only way a candidate becomes a test."""
    wf = wf or pipeline.walk_forward(study)
    reg = study.registry
    bar = reg.next_bar()
    res = judge(study, frame, baseline, c, wf)
    row = {
        "kind": "test",
        "name": c.name,
        "features": c.feature,
        "scope": c.scope,
        "metric": "rank_ic_gain_vs_baseline",
        "t_tune": round(res["t_tune"], 3),
        "bar_tune": round(bar, 3),
        "gain_tune": round(res["gain_tune"], 5),
        "check_used": False,
        "kept": False,
        "note": (note + f" {res['periods']} periods").strip(),
    }
    if res["t_tune"] >= bar:  # only now is the check period looked at
        grant = pipeline.CheckGrant(reg.open_check())
        chk = check(study, baseline, c, wf, grant)
        row |= {
            "check_used": True,
            "t_check": round(chk["t_check"], 3),
            "bar_check": round(grant.bar, 3),
            "gain_check": round(chk["gain_check"], 5),
            "kept": bool(chk["t_check"] >= grant.bar and chk["gain_check"] > 0),
        }
    return reg.record(row)


class ConfigProposer:
    """The config's candidate list; if empty, every panel feature x every transform, minus the
    baseline's own levels (already in the model)."""

    def propose(self, study, tried: set, features: list[str]) -> list[Candidate]:
        d = study.cfg.get("discovery", {})
        listed = [Candidate(**c) for c in d.get("candidates") or []]
        if not listed:
            base = set(study.feature_set("baseline"))
            listed = [
                Candidate(f, t)
                for f in features
                for t in d.get("transforms", ["level"])
                if not (t == "level" and f in base)
            ]
        return [c for c in listed if (c.name, c.scope) not in tried]


def run(
    study,
    frame: pd.DataFrame,
    features: list[str],
    proposer=None,
    max_tests: int | None = None,
) -> list[dict]:
    """Test candidates until one is kept, the limit is hit, the check budget is spent, or
    `no_progress` tests in a row miss the tuning bar."""
    d = study.cfg.get("discovery", {})
    max_tests = max_tests or int(d.get("max_tests", 3))
    no_progress = int(d.get("no_progress", 10))
    baseline = study.feature_set("baseline")
    proposer = proposer or ConfigProposer()
    wf = pipeline.walk_forward(study)
    out, misses = [], 0
    for c in proposer.propose(study, study.registry.tested(), features)[:max_tests]:
        if c.feature not in frame:
            print(f"skip {c.name}: not in the panel", file=sys.stderr)
            continue
        row = test_candidate(study, frame, baseline, c, wf)
        out.append(row)
        print(
            f"{c.name} [{c.scope}]: t_tune {row['t_tune']} (bar {row['bar_tune']}), "
            f"check {'used' if row['check_used'] else 'not reached'}, kept {row['kept']}",
            file=sys.stderr,
        )
        if row["kept"]:
            break
        misses = 0 if row["t_tune"] >= row["bar_tune"] else misses + 1
        if (
            misses >= no_progress
            or study.registry.check_uses() >= study.registry.check_limit
        ):
            break
    return out
