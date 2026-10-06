"""Load a market YAML into a Study, then build, report, test and discover, with every period gated
here: tuning is always open, check needs a grant from the registry, holdout needs final=True.

Market YAML keys (markets/us_smallcap.yaml is a full example, markets/csv_example.yaml a small one):
  plugin       module under engine.markets (or a dotted path) exposing build(cfg)
  calendar     engine.data calendar settings (kind: trading | continuous)
  labels       horizon (bars), winsorize [lo, hi], plug-in label rules
  schedule     monthly | weekly | daily (or a pandas frequency, e.g. BQE)
  sources      name -> {features, max_age_days, lookback_days, params}; type: text for documents
  features     named feature sets; `baseline` is what every new candidate must beat
  model        name (ridge | trees), alpha, target {within}, block, min_train_periods, embargo_days,
               half_life
  portfolio    long-only rules for the scorer (band / topk / periodic)
  periods      tuning [start, end), check [start, end), holdout_start
  registry     file, inherit [other registries], check_limit, holdout_unlock_log
  discovery    transforms, candidates, max_tests, no_progress
  cohort_test  optional: a pre-registered long-horizon test (see CohortTest)
  spend        ledger path, caps {step: usd}, funds {provider: usd} (engine.spend)
  decider      kind (none | jev | claude) + options (engine.decide)

A candidate (test, discover) = one feature + a transform + a scope:
  level | chg<k> (change vs k decision periods earlier) | pct<k> | log;  universal | one group
Judge: the same walk-forward with and without the candidate; the statistic is the per-period
difference in rank IC, t over tuning periods (with an empty baseline: the candidate's own rank IC).
It must reach the registry's next bar; only then is the check period opened (counted), and kept
needs the check bar and a positive check-period gain.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import data, markets, model, panel, score
from engine.registry import CheckGrant, HoldoutLock, HoldoutLocked, Periods, Registry

ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- the study
@dataclass
class Study:
    """Everything a market YAML defines, built: market, sources, periods, registry, holdout."""

    name: str
    cfg: dict
    market: object
    sources: dict  # name -> panel.SourceSpec
    periods: Periods
    registry: Registry
    holdout: HoldoutLock
    cache_dir: Path

    def panel_spec(self, start, end, source_names: list[str] | None = None) -> panel.PanelSpec:
        """A panel spec over [start, end) with the YAML's schedule, horizon and sources."""
        names = source_names or list(self.sources)
        return panel.PanelSpec(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            schedule=self.cfg.get("schedule", "monthly"),
            horizon=int(self.cfg["labels"]["horizon"]),
            sources=[self.sources[n] for n in names],
        )

    def feature_set(self, name: str) -> list[str]:
        """A named feature set; "@other" includes another set."""
        out: list[str] = []
        for item in self.cfg.get("features", {})[name] or []:
            out += self.feature_set(item[1:]) if item.startswith("@") else [item]
        return list(dict.fromkeys(out))


def path(p: str | Path) -> Path:
    """Paths in configs are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_config(market: str, cfg_path: str | Path | None = None) -> dict:
    """markets/<market>.yaml (or cfg_path) as a dict."""
    return yaml.safe_load(path(cfg_path or f"markets/{market}.yaml").read_text())


def load_study(market: str, cfg_path: str | Path | None = None) -> Study:
    """Read the market's YAML and build its Study."""
    return study_from_config(market, read_config(market, cfg_path))


def study_from_config(market: str, cfg: dict) -> Study:
    """Market, sources, registry and periods from a config dict."""
    plugin = markets.load(cfg.get("plugin", market))
    mkt, factories = plugin.build(cfg)
    cache_dir = path(cfg.get("cache_dir", f".engine_cache/{market}"))
    specs = {}
    for name, s in (cfg.get("sources") or {}).items():
        s = s or {}
        text = s.get("type") == "text"
        if text:  # documents + questions + a reader
            from engine.spend import Ledger
            from engine.text.source import TextSource

            src = TextSource(
                mkt,
                s.get("params"),
                plugin=plugin,
                ledger=Ledger.from_config(cfg.get("spend"), ROOT),
                cache_dir=cache_dir,
                root=ROOT,
                name=name,
            )
        else:
            src = factories[s.get("type", name)](mkt, s.get("params"))
        features = s.get("features")
        if text and not features:  # listed under params: the columns exist even if nothing was read
            features = (s.get("params") or {}).get("features")
        specs[name] = panel.SourceSpec(
            source=src,
            features=features,
            max_age_days=s.get("max_age_days"),
            lookback_days=int(s.get("lookback_days", 3650)),
        )
    reg = cfg.get("registry", {})
    registry = Registry(
        path(reg.get("file", f"data/engine/{market}/registry.csv")),
        market,
        [path(p) for p in reg.get("inherit", [])],
        int(reg.get("check_limit", 30)),
    )
    lock = HoldoutLock(
        path(reg.get("holdout_unlock_log", f"data/engine/{market}/holdout_unlocks.csv"))
    )
    return Study(
        name=market,
        cfg=cfg,
        market=mkt,
        sources=specs,
        periods=Periods.from_config(cfg["periods"]),
        registry=registry,
        holdout=lock,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------- the gated standard run
def stage_end(study: Study, stage: str, grant: CheckGrant | None = None, final: bool = False):
    """The end of the data a stage may read; check needs a grant, holdout needs final=True."""
    p = study.periods
    if stage == "tuning":
        return p.tuning[1]
    if stage == "check":
        if not isinstance(grant, CheckGrant):
            raise PermissionError("the check period opens only through registry.open_check")
        return p.check[1]
    if stage == "holdout":
        if not final:
            raise HoldoutLocked("the holdout is locked; a person runs it once with --final")
        return pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    raise ValueError(stage)


def build_panel(
    study: Study, stage="tuning", grant=None, final=False, sources=None, with_labels=True
):
    """The panel from the first tuning date to the stage's end, point-in-time checked."""
    end = stage_end(study, stage, grant, final)
    spec = study.panel_spec(study.periods.tuning[0], end, sources)
    p = panel.build(study.market, spec, cache_dir=study.cache_dir, with_labels=with_labels)
    data.check_panel(p)  # a leak stops the run here
    return p


def walk_forward(study: Study, factory=None) -> model.WalkForward:
    """The YAML's model and walk-forward settings."""
    m = study.cfg["model"]
    factory = factory or (
        model.ridge(float(m.get("alpha", 10.0)))
        if m.get("name", "ridge") == "ridge"
        else model.MODELS[m["name"]]()
    )
    return model.WalkForward(
        target="fwd_rank",
        demean_by=("decision_time", "group"),
        model=factory,
        block=m.get("block", "year"),
        min_train_periods=int(m.get("min_train_periods", 12)),
        embargo=pd.Timedelta(days=int(m.get("embargo_days", 0))),
        keep=("group", "fwd_return", "rt_cost"),
        half_life=m.get("half_life"),
    )


def in_stage(study: Study, scored: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Scored rows whose local decision date falls in the stage."""
    local = study.market.calendar.local_date(scored["decision_time"])
    lo, hi = study.periods.tuning if stage == "tuning" else study.periods.check
    if stage == "holdout":
        lo, hi = study.periods.holdout_start, pd.Timestamp.max
    return scored[((local >= lo) & (local < hi)).to_numpy()].reset_index(drop=True)


def portfolio_rules(study: Study) -> dict:
    """The YAML's long-only portfolio rules."""
    kinds = {"band": score.Band, "topk": score.TopK, "periodic": score.Periodic}
    return {n: kinds[n](**r) for n, r in (study.cfg.get("portfolio") or {}).items() if n in kinds}


def label_overlap(study: Study) -> int:
    """How many decisions' label windows overlap (deflates every t)."""
    per = {"monthly": 21, "weekly": 5, "daily": 1}.get(study.cfg.get("schedule"), 1)
    if getattr(study.market.calendar, "kind", "trading") == "continuous":
        per = {"monthly": 30, "weekly": 7, "daily": 1}.get(study.cfg.get("schedule"), 1)
    return score.overlap(int(study.cfg["labels"]["horizon"]), per)


def periods_per_year(study: Study) -> int:
    """Decisions a year, for annual turnover (pandas-frequency schedules count as monthly)."""
    continuous = getattr(study.market.calendar, "kind", "trading") == "continuous"
    per = {"weekly": 52, "daily": 365 if continuous else 252}
    return per.get(study.cfg.get("schedule"), 12)


def evaluate(study: Study, scored: pd.DataFrame, stage: str = "tuning") -> dict:
    """The scorer on one stage, costs filled, t deflated by the label overlap."""
    s = score.fill_costs(in_stage(study, scored, stage))
    return score.evaluate(
        s,
        portfolio_rules(study),
        periods_per_year=periods_per_year(study),
        overlap_n=label_overlap(study),
    )


def fetch(study: Study) -> None:
    """Each source fills its own cache through the end of the check period (network, your keys)."""
    lo, hi = study.periods.tuning[0], study.periods.check[1]
    for name, s in study.sources.items():
        _log(f"fetch {name}")
        s.source.fetch(lo - pd.Timedelta(days=s.lookback_days), hi)


def report(study: Study, features: str = "baseline") -> dict:
    """The feature set's model on tuning periods (descriptive, never logged) plus the registry."""
    feats = study.feature_set(features)
    if not feats:
        raise SystemExit(f"feature set {features!r} is empty: pass --features <a named set>")
    rows = panel.model_rows(study, build_panel(study, "tuning"))
    scored, weights = walk_forward(study).run(rows, feats, study.market.calendar)
    avg = weights.mean().sort_values() if len(weights) else pd.Series(dtype=float)
    return {
        "features": features,
        "tuning": evaluate(study, scored, "tuning"),
        "average_weights_x100": (pd.concat([avg.head(5), avg.tail(5)]) * 100).round(2).to_dict(),
        "registry": study.registry.summary(),
        "holdout_looks": study.holdout.looks(),
    }


def report_holdout(study: Study, features: str, reason: str) -> dict:
    """Open the holdout ONCE: the look is logged with the reason before a single row is read."""
    study.holdout.unlock(study.name, reason)
    rows = panel.model_rows(study, build_panel(study, "holdout", final=True))
    scored, _ = walk_forward(study).run(rows, study.feature_set(features), study.market.calendar)
    return {"holdout": evaluate(study, scored, "holdout")}


# ---------------------------------------------------------------- candidates: test and discover
@dataclass(frozen=True)
class Candidate:
    """A feature + a transform + a scope: what one test judges."""

    feature: str
    transform: str = "level"
    scope: str = "universal"

    @property
    def name(self) -> str:
        """feature, or feature|transform."""
        return self.feature if self.transform == "level" else f"{self.feature}|{self.transform}"


def transform(frame: pd.DataFrame, feature: str, how: str) -> pd.Series:
    """level | log | chg<k> | pct<k>. Reads only the entity's own earlier rows (point-in-time)."""
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
    prev = x.loc[order].groupby(frame.loc[order, "entity_id"]).shift(k).reindex(frame.index)
    return x - prev if m.group(1) == "chg" else x / prev.where(prev != 0) - 1


def with_candidate(frame: pd.DataFrame, c: Candidate) -> tuple[pd.DataFrame, list[str]]:
    """The rows plus the candidate's column(s); a group scope adds a group-only copy, so that group
    can carry its own weight."""
    col = f"cand__{c.name}"
    f = frame.assign(**{col: transform(frame, c.feature, c.transform)})
    cols = [col]
    if c.scope != "universal":
        scoped = f"{col}__{c.scope}"
        f[scoped] = f[col].where(f["group"] == c.scope)
        cols.append(scoped)
    return f, cols


def ic_gain(base: pd.DataFrame | None, cand: pd.DataFrame, scope: str) -> pd.Series:
    """Per-period rank IC of the candidate model minus the baseline's (no baseline: minus 0)."""
    if scope != "universal":
        cand = cand[cand["group"] == scope]
        base = None if base is None else base[base["group"] == scope]
    ic = score.rank_ic(cand)
    return (ic if base is None else ic - score.rank_ic(base)).dropna()


def _gain(study, frame, baseline, c, wf, stage) -> pd.Series:
    f, cols = with_candidate(frame, c)
    cal = study.market.calendar
    base = in_stage(study, wf.run(f, baseline, cal)[0], stage) if baseline else None
    cand = in_stage(study, wf.run(f, baseline + cols, cal)[0], stage)
    return ic_gain(base, cand, c.scope)


def judge(study, frame, baseline: list[str], c: Candidate, wf: model.WalkForward) -> dict:
    """Tuning-period statistics for one candidate (no bars, no logging)."""
    gain = _gain(study, frame, baseline, c, wf, "tuning")
    return {
        "t_tune": score.per_period_t(gain, label_overlap(study)),
        "gain_tune": float(gain.mean()),
        "periods": len(gain),
    }


def check(study, baseline: list[str], c: Candidate, wf, grant: CheckGrant) -> dict:
    """The same statistic on the check period; needs the registry's grant."""
    rows = panel.model_rows(study, build_panel(study, "check", grant=grant))
    gain = _gain(study, rows, baseline, c, wf, "check")
    return {
        "t_check": score.per_period_t(gain, label_overlap(study)),
        "gain_check": float(gain.mean()),
    }


def test_candidate(
    study, frame, baseline: list[str], c: Candidate, wf=None, note: str = ""
) -> dict:
    """Judge, apply both bars, and log the result: the only way a candidate becomes a test."""
    wf = wf or walk_forward(study)
    reg = study.registry
    bar = reg.next_bar()
    res = judge(study, frame, baseline, c, wf)
    row = {
        "kind": "test",
        "name": c.name,
        "features": c.feature,
        "scope": c.scope,
        "metric": "rank_ic_gain_vs_baseline" if baseline else "rank_ic (empty baseline)",
        "t_tune": round(res["t_tune"], 3),
        "bar_tune": round(bar, 3),
        "gain_tune": round(res["gain_tune"], 5),
        "check_used": False,
        "kept": False,
        "note": (note + f" {res['periods']} periods").strip(),
    }
    if res["t_tune"] >= bar:  # only now is the check period looked at
        grant = reg.open_check()
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
    baseline's own levels (already in the model). Any object with propose() can replace it."""

    def propose(self, study, tried: set, features: list[str]) -> list[Candidate]:
        """Untried candidates, in order."""
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


def discover(
    study, frame, features: list[str], proposer=None, max_tests: int | None = None
) -> list[dict]:
    """Test candidates until one is kept, the limit is hit, the check budget is spent, or
    `no_progress` tests in a row miss the tuning bar."""
    d = study.cfg.get("discovery", {})
    max_tests = max_tests or int(d.get("max_tests", 3))
    no_progress = int(d.get("no_progress", 10))
    baseline = study.feature_set("baseline")
    proposer = proposer or ConfigProposer()
    wf = walk_forward(study)
    out, misses = [], 0
    for c in proposer.propose(study, study.registry.tested(), features)[:max_tests]:
        if c.feature not in frame:
            _log(f"skip {c.name}: not in the panel")
            continue
        row = test_candidate(study, frame, baseline, c, wf)
        out.append(row)
        _log(
            f"{c.name} [{c.scope}]: t_tune {row['t_tune']} (bar {row['bar_tune']}), "
            f"check {'used' if row['check_used'] else 'not reached'}, kept {row['kept']}"
        )
        if row["kept"]:
            break
        misses = 0 if row["t_tune"] >= row["bar_tune"] else misses + 1
        if misses >= no_progress or study.registry.check_uses() >= study.registry.check_limit:
            break
    return out


# ---------------------------------------------------------------- the long-horizon cohort test
def composite(
    frame: pd.DataFrame, inputs: list[str], require: list[str], missing_rank: float
) -> pd.Series:
    """Equal-weight mean of per-formation percentile ranks; NaN where a required input is missing."""
    ok = frame[require].notna().all(axis=1)
    ranks = (
        frame.loc[ok, inputs]
        .groupby(frame.loc[ok, "decision_time"])
        .rank(pct=True)
        .fillna(missing_rank)
    )
    return ranks.mean(axis=1).reindex(frame.index)


def pick(frame: pd.DataFrame, col: str, top: float, bottom: bool = False) -> dict:
    """decision_time -> entity ids: the top (or bottom) `top` by col (a fraction, or a count >= 1)."""
    out = {}
    for t, g in frame.dropna(subset=[col]).groupby("decision_time"):
        k = int(top) if top >= 1 else max(1, round(top * len(g)))
        g = g.sort_values(col, ascending=bottom, kind="stable")
        out[t] = list(g["entity_id"].head(k))
    return out


class CohortTest:
    """The YAML's `cohort_test:` block: a fixed composite (no fitting), overlapping cohorts.

    At each formation, every input is ranked across the names that have the `require`d inputs; a
    missing input gets `missing_rank`; the score is the mean rank. The top `top` (fraction or count)
    are bought equal weight at the close of the first trading day after formation and held
    `hold_quarters` formations; each day the portfolio is the mean of the cohorts held. Benchmark:
    the same construction over the whole universe. Both net of measured costs (half a round trip in,
    half out). Statistic: monthly portfolio minus benchmark, net, Newey-West t. Needs a market with
    daily_returns(codes). run() with no arguments is the pre-registered design; its arguments
    (top, hold, inputs, spread) are descriptive variants that are never logged.
    """

    def __init__(self, study: Study):
        self.study = study
        self.cfg = study.cfg["cohort_test"]
        self.market = study.market
        self.cal = self.market.calendar
        first = pd.Timestamp(self.cfg["first_formation"])
        last = pd.Timestamp(self.cfg["last_formation"])
        self.panel = build_panel(
            study, "tuning", sources=self.cfg.get("sources"), with_labels=False
        )  # point-in-time checked inside
        f = self.panel.frame
        local = self.cal.local_date(f["decision_time"])
        self.frame = f[((local >= first) & (local <= last)).to_numpy()].reset_index(drop=True)
        self.times = pd.DatetimeIndex(sorted(self.frame["decision_time"].unique()))
        self._rets = None

    def coverage(self) -> pd.DataFrame:
        """Input coverage per formation (no returns): what a pre-registration looks at."""
        f, cols = self.frame, self.cfg["inputs"]
        cov = f.groupby("decision_time")[cols].apply(lambda g: g.notna().mean())
        cov["universe"] = f.groupby("decision_time").size()
        cov["ranked"] = f.dropna(subset=self.cfg["require"]).groupby("decision_time").size()
        cov.index = self.cal.local_date(pd.Series(cov.index)).dt.date.to_numpy()
        return cov

    def scored(self, inputs: list[str] | None = None) -> pd.DataFrame:
        """The panel with the composite score."""
        c = self.cfg
        s = composite(
            self.frame, inputs or c["inputs"], c["require"], float(c.get("missing_rank", 0.5))
        )
        return self.frame.assign(score=s)

    def returns(self):
        """(daily returns, ended dates) for every name in the panel, loaded once."""
        if self._rets is None:
            codes = sorted(self.frame["entity_id"].unique())
            _log(f"cohort: daily returns for {len(codes):,} codes")
            self._rets = self.market.daily_returns(codes)
        return self._rets

    def _formation_dates(self, n_after: int) -> pd.DatetimeIndex:
        """Formation decision times, extended n_after quarters past the last (for exit dates)."""
        end = self.times[-1] + pd.DateOffset(months=3 * n_after + 3)
        more = self.cal.decision_times(self.times[0], end, self.study.cfg.get("schedule", "BQE"))
        return pd.DatetimeIndex(sorted(set(self.times) | set(more)))

    def _cost(self, ids, when) -> pd.Series:
        """Half a round trip per id at decision time `when`; unknown -> that date's median."""
        rows = pd.DataFrame({"entity_id": list(ids), "decision_time": when})
        bps = pd.Series(self.market.cost_bps(rows).to_numpy(), index=list(ids))
        return 0.5 * bps.fillna(bps.median()).fillna(0.0) / 1e4

    def portfolio(self, members: dict, hold: int) -> pd.DataFrame:
        """Monthly gross and net returns of overlapping cohorts {formation time: ids}."""
        rets, ended = self.returns()
        days = rets.index
        sched = self._formation_dates(hold)
        paths = []
        for t, ids in members.items():
            local = self.cal.local_date(pd.Series([t])).iloc[0]
            after = days[days > local]
            if not len(after):
                continue  # entry after the data end
            entry = after[0]
            exit_t = sched[sched.get_loc(t) + hold]
            exit_local = self.cal.local_date(pd.Series([exit_t])).iloc[0]
            later = days[days > exit_local]
            exit_day = later[0] if len(later) else None
            assert entry > local  # bought strictly after the formation date
            paths.append(
                score.cohort_returns(
                    rets,
                    ids,
                    entry,
                    exit_day,
                    self._cost(ids, t),
                    self._cost(ids, exit_t) if exit_day is not None else None,
                    ended,
                )
            )
        return score.monthly_returns(score.overlapping_cohorts(paths))

    def run(self, top=None, hold=None, inputs=None, spread=False) -> dict:
        """Monthly excess over the benchmark, net of costs; no arguments = the pre-registered design."""
        c = self.cfg
        top = float(top if top is not None else c["top"])
        hold = int(hold if hold is not None else c["hold_quarters"])
        s = self.scored(inputs)
        long = pick(s, "score", top)
        other = (
            pick(s, "score", top, bottom=True)
            if spread
            else {t: list(g["entity_id"]) for t, g in s.groupby("decision_time")}
        )  # benchmark: the whole universe, ranked or not
        p, b = self.portfolio(long, hold), self.portfolio(other, hold)
        m = p[["gross", "net"]].join(b[["gross", "net"]], rsuffix="_bench", how="inner")
        m["excess_gross"] = m["gross"] - m["gross_bench"]
        m["excess_net"] = m["net"] - m["net_bench"]
        lags = int(c.get("nw_lags", 12))
        full = m[p["cohorts"].reindex(m.index) >= hold]
        return {
            "months": len(m),
            "first_month": str(m.index.min()),
            "last_month": str(m.index.max()),
            "names_per_cohort": float(np.mean([len(v) for v in long.values()])),
            "excess_net": float(m["excess_net"].mean()),
            "excess_net_t_nw": score.newey_west_t(m["excess_net"], lags),
            "excess_net_t_iid": score.per_period_t(m["excess_net"]),
            "excess_gross": float(m["excess_gross"].mean()),
            "excess_gross_t_nw": score.newey_west_t(m["excess_gross"], lags),
            "port_net": float(m["net"].mean()),
            "bench_net": float(m["net_bench"].mean()),
            "cost_drag_port": float((m["gross"] - m["net"]).mean()),
            "tracking_error_monthly": float(m["excess_net"].std(ddof=1)),
            "full_ramp_months": len(full),
            "full_ramp_excess_net": float(full["excess_net"].mean()) if len(full) else None,
            "full_ramp_t_nw": score.newey_west_t(full["excess_net"], lags)
            if len(full) > 2
            else None,
            "_monthly": m,
        }


def cohort_test_logged(study: Study) -> bool:
    """Whether the pre-registered cohort test already has its one look in the registry."""
    return bool((study.registry.own()["name"] == study.cfg["cohort_test"]["name"]).any())


def log_cohort_test(study: Study, res: dict, note: str) -> dict:
    """Log the pre-registered cohort test as ONE judged test (refused if already logged)."""
    c = study.cfg["cohort_test"]
    if cohort_test_logged(study):
        raise SystemExit(f"{c['name']!r} is already in the registry: one look only")
    bar = study.registry.next_bar()
    t = res["excess_net_t_nw"]
    return study.registry.record(
        {
            "kind": "test",
            "name": c["name"],
            "features": ",".join(c["inputs"]),
            "scope": "universal",
            "metric": "monthly excess vs equal-weight universe, overlapping cohorts, net of "
            f"measured costs, Newey-West t (lag {c.get('nw_lags', 12)})",
            "t_tune": round(t, 3),
            "bar_tune": round(bar, 3),
            "check_used": False,
            "gain_tune": round(res["excess_net"], 6),
            "kept": bool(t >= bar and res["excess_net"] > 0),
            "note": note,
        }
    )
