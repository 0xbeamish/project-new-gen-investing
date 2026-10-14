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
  registry     file, inherit [other registries], check_limit, holdout_unlock_log, planned (the
               number of pre-registered tests: from the first, each faces the bar of the last)
  discovery    transforms, candidates, max_tests, no_progress
  cohort_test  optional: a pre-registered long-horizon test (see CohortTest)
  spend        ledger path, caps {step: usd}, funds {provider: usd} (engine.spend)
  decider      kind (none | jev | claude) + options (engine.decide)
  loop         `engine loop` settings: name (to --log it), features, years, refit, recency, band,
               persist_k, cooldown, question_splits (engine.improve.loop_settings)
  results_dir  where `engine loop` writes its CSVs (default results/<market>)

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
        int(reg.get("planned", 1)),
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
        keep=("group", "fwd_return", "fwd_excess", "rt_cost"),
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


def t_lags(study: Study) -> int | str:
    """The market's Newey-West lags for every t (0 = off; "auto" = floor(4 (n/100)^(2/9)))."""
    return study.cfg.get("t_stat_lags", 0)


def period_t(study: Study, series, overlap: bool = True) -> float:
    """t of a per-period series under the market's rules: label overlap and Newey-West lags."""
    return score.per_period_t(series, label_overlap(study) if overlap else 1, t_lags(study))


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
        lags=t_lags(study),
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


# ---------------------------------------------------------------- power: what a test can see
def power_check(
    units: str, noise_sd: float, periods: int, bar: float, overlap: int, noise_from: str, what=""
) -> dict:
    """The minimum detectable effect (MDE) at the test's bar with 50% and 80% power, from noise
    measured before the result; printed before the test runs and stored with its registry row."""
    m50, m80 = (score.mde(noise_sd, periods, bar, p, overlap) for p in (0.5, 0.8))
    out = {
        "units": units,
        "periods": int(periods),
        "noise_sd": float(noise_sd),
        "noise_from": noise_from,
        "bar": round(bar, 3),
        "mde_50": m50,
        "mde_80": m80,
        "underpowered": bool(m80 > score.PLAUSIBLE[units]),  # False for NaN
    }
    verdict = (
        f"UNDERPOWERED: effects below {m80:.4g} are likely missed, and that is above "
        f"{score.PLAUSIBLE[units]:g}, more than any edge this research has measured; a miss can only "
        "say 'couldn't tell'"
        if out["underpowered"]
        else "a miss says the effect, if any, is smaller than this"
    )
    _log(
        f"Minimum detectable effect ({what or units}), {periods} periods, noise sd {noise_sd:.4g} "
        f"({noise_from}), bar t >= {bar:.2f}: {m80:.4g} with 80% power, {m50:.4g} with 50%. {verdict}"
    )
    return out


def _ic_noise(study, frame, base, scope: str, wf) -> tuple[float, int, str]:
    """Per-period noise of a rank IC on tuning periods, from the BASELINE model (never the
    candidate): the sd of its per-period IC, or, with no baseline, the null sd 1/sqrt(N - 1) of a
    rank correlation over N names."""
    if base is not None:
        b = base if scope == "universal" else base[base["group"] == scope]
        ic = score.rank_ic(b).dropna()
        return score.long_run_sd(ic, t_lags(study)), len(ic), "the baseline's per-period rank IC"
    f = frame.assign(_flat=0.0)
    scored = in_stage(study, wf.run(f, ["_flat"], study.market.calendar)[0], "tuning")
    if scope != "universal":
        scored = scored[scored["group"] == scope]
    n = scored.groupby("decision_time").size()
    n = n[n > 2]
    return float(np.sqrt((1 / (n - 1)).mean())), len(n), "null sd of a rank IC, 1/sqrt(N-1)"


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


def _base(study, frame, baseline, wf, stage):
    """The baseline model's scores on the stage (None for an empty baseline)."""
    return (
        in_stage(study, wf.run(frame, baseline, study.market.calendar)[0], stage)
        if baseline
        else None
    )


def _gain(study, frame, baseline, c, wf, stage, base=None) -> pd.Series:
    f, cols = with_candidate(frame, c)
    base = base if base is not None else _base(study, f, baseline, wf, stage)
    cand = in_stage(study, wf.run(f, baseline + cols, study.market.calendar)[0], stage)
    return ic_gain(base, cand, c.scope)


def judge(
    study, frame, baseline: list[str], c: Candidate, wf: model.WalkForward, base=None
) -> dict:
    """Tuning-period statistics for one candidate (no bars, no logging)."""
    gain = _gain(study, frame, baseline, c, wf, "tuning", base)
    return {
        "t_tune": period_t(study, gain),
        "gain_tune": float(gain.mean()),
        "periods": len(gain),
    }


def check(study, baseline: list[str], c: Candidate, wf, grant: CheckGrant) -> dict:
    """The same statistic on the check period; needs the registry's grant."""
    rows = panel.model_rows(study, build_panel(study, "check", grant=grant))
    gain = _gain(study, rows, baseline, c, wf, "check")
    return {
        "t_check": period_t(study, gain),
        "gain_check": float(gain.mean()),
    }


def residualize(f: pd.DataFrame, col: str, net: list[str]) -> pd.Series:
    """Per decision time: the rank of `col` minus its least-squares fit on the ranks of `net`
    (missing -> the middle), i.e. what `col` says that those features don't."""
    out = pd.Series(np.nan, index=f.index)
    for _, g in f.groupby("decision_time"):
        g = g.dropna(subset=[col])
        if len(g) <= len(net) + 2:
            continue
        x = [np.ones(len(g))] + [g[k].rank(pct=True).fillna(0.5).to_numpy() for k in net]
        X, y = np.column_stack(x), g[col].rank(pct=True).to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[g.index] = y - X @ b
    return out


def _confirm_noise(study, c: Candidate, sign: int, net: list[str] | None) -> float:
    """Per-period noise of the candidate's own rank IC on the TUNING periods (open data, before the
    check is read): coins move together, so it runs well above the null sd 1/sqrt(N - 1)."""
    rows = panel.model_rows(study, build_panel(study, "tuning"))
    f = in_stage(study, rows.assign(score=sign * transform(rows, c.feature, c.transform)), "tuning")
    f = f.dropna(subset=["score", "fwd_return"])
    if c.scope != "universal":
        f = f[f["group"] == c.scope]
    if net:
        f = f.assign(score=residualize(f, "score", net)).dropna(subset=["score"])
    return score.long_run_sd(score.rank_ic(f).dropna(), t_lags(study))


def confirm(study, c: Candidate, sign: int, note: str = "", net: list[str] | None = None) -> dict:
    """ONE logged out-of-sample test of a candidate found by exploring the tuning years: the
    candidate's own per-period rank IC (no model), times its pre-registered sign, on the check
    period ALONE, within its scope; with `net`, the IC of what the candidate says beyond those
    features (residualize). Uses one check-period look; the bar is the registry's next bar (it
    counts as a judged test). Kept if t reaches the bar."""
    if sign not in (1, -1):
        raise ValueError("sign is the pre-registered direction: 1 or -1")
    reg = study.registry
    bar = reg.next_bar()
    grant = reg.open_check()
    rows = panel.model_rows(study, build_panel(study, "check", grant=grant))
    f = in_stage(study, rows.assign(score=sign * transform(rows, c.feature, c.transform)), "check")
    f = f.dropna(subset=["score", "fwd_return"])
    if c.scope != "universal":
        f = f[f["group"] == c.scope]
    if net:
        f = f.assign(score=residualize(f, "score", net)).dropna(subset=["score"])
    n = f.groupby("decision_time").size()
    n = n[n > 2]
    null = float(np.sqrt((1 / (n - 1)).mean())) if len(n) else float("nan")
    tuned = _confirm_noise(study, c, sign, net)
    noise, src = (
        (tuned, "the candidate's per-period rank IC sd on tuning periods")
        if tuned > null
        else (null, "null sd of a rank IC, 1/sqrt(N-1)")
    )
    power = power_check(
        "rank IC", noise, len(n), bar, label_overlap(study), src, "rank IC on the check period"
    )
    ic = score.rank_ic(f[f["decision_time"].isin(n.index)]).dropna()
    t = period_t(study, ic)
    row = {
        "kind": "confirm",
        "name": c.name,
        "features": c.feature,
        "scope": c.scope,
        "metric": f"rank_ic x {sign:+d} on check (confirm)"
        + (f", net of {'+'.join(net)}" if net else ""),
        "bar_tune": round(bar, 3),
        "check_used": True,
        "t_check": round(t, 3),
        "bar_check": round(bar, 3),
        "gain_check": round(float(ic.mean()), 5),
        "kept": bool(t >= bar),
        "note": (
            note + f" {len(ic)} periods, median {int(n.median()) if len(n) else 0} names"
        ).strip(),
        "mde_50": round(power["mde_50"], 5),
        "mde_80": round(power["mde_80"], 5),
    }
    return reg.record(row)


def pooled_strategy_test(
    study,
    feature: str,
    sign: int,
    enter: float,
    exit: float,
    pre: tuple[str, str] | None,
    holdout_reason: str | None,
    note: str = "",
    holdout_study=None,
    sources: list[str] | None = None,
) -> dict:
    """ONE logged test of a rule-based strategy (score = sign x feature; no model, so no training
    data) on every period it was not explored on: `pre` (before the tuning years), the check
    period, and the holdout (opened once, logged with `holdout_reason`; `holdout_study` is the same
    market without its data_end). The book: long-only, equal weight, band within group (enter above
    `enter`, hold while above `exit`), net of the market's costs. Statistic: the weekly net excess
    return over the equal-weight universe, pooled over the periods (each starts from an empty
    book), t = mean / (sd / sqrt(periods)). Kept if t reaches the registry's next bar and the mean
    is positive. Uses one check look. `sources`: the sources to build (default all)."""
    if sign not in (1, -1):
        raise ValueError("sign is the pre-registered direction: 1 or -1")
    reg = study.registry
    bar = reg.next_bar()
    grant = reg.open_check()
    frames = {}
    if pre:
        p = panel.build(
            study.market, study.panel_spec(pre[0], pre[1], sources), cache_dir=study.cache_dir
        )
        data.check_panel(p)
        frames["pre"] = panel.model_rows(study, p)
    frames["check"] = in_stage(
        study,
        panel.model_rows(study, build_panel(study, "check", grant, sources=sources)),
        "check",
    )
    if holdout_reason:
        hs = holdout_study or study
        hs.holdout.unlock(hs.name, holdout_reason)
        frames["holdout"] = in_stage(
            hs,
            panel.model_rows(hs, build_panel(hs, "holdout", final=True, sources=sources)),
            "holdout",
        )
    nets, parts = [], {}
    for name, f in frames.items():
        f = f.assign(score=sign * f[feature]).dropna(subset=["score", "fwd_return"])
        net = score.simulate_portfolio(score.fill_costs(f), score.Band(enter=enter, exit=exit))[
            "net"
        ]
        nets.append(net)
        parts[name] = {
            "weeks": len(net),
            "net_bp": round(float(net.mean()) * 1e4, 2),
            "t": round(period_t(study, net, overlap=False), 3),
        }
    pooled = pd.concat(nets)
    t = period_t(study, pooled, overlap=False)
    row = {
        "kind": "confirm",
        "name": f"strategy {'+' if sign > 0 else '-'}{feature} band {enter}/{exit}",
        "features": feature,
        "scope": "universal",
        "metric": "pooled net excess vs equal-weight universe per period: " + "+".join(frames),
        "bar_tune": round(bar, 3),
        "check_used": True,
        "t_check": round(t, 3),
        "bar_check": round(bar, 3),
        "gain_check": round(float(pooled.mean()), 6),
        "kept": bool(t >= bar and pooled.mean() > 0),
        "note": (note + " " + str(parts)).strip(),
    }
    return reg.record(row) | {"periods": parts}


def pooled_long_short_test(
    study,
    feature: str,
    sign: int,
    enter: float,
    exit: float,
    attach,
    benchmark,
    min_names: int = 5,
    pre: tuple[str, str] | None = None,
    holdout_reason: str | None = None,
    holdout_study=None,
    sources: list[str] | None = None,
    note: str = "",
    report=print,
) -> dict:
    """ONE logged test of a dollar-neutral long-short band book (score = sign x feature, ranked
    across all tradable names; score.band_long_short) on `pre`, tuning, check and holdout periods,
    each starting from an empty book. `attach(rows)` adds listed, ret, funding, side_cost (e.g. perp
    legs); `benchmark(rows)` gives the benchmark's return per decision time. Judged statistic: alpha
    of the pooled weekly net return regressed on the benchmark (full-sample beta: a test statistic,
    not a tradable hedge), t from HC3 errors; kept if t reaches the registry's next bar and alpha > 0.
    Coverage is reported before any return; then power, the primary, secondaries and risk lines.
    Uses one check look; opens the holdout (logged) only with `holdout_reason`."""
    if sign not in (1, -1):
        raise ValueError("sign is the pre-registered direction: 1 or -1")
    reg = study.registry
    bar = reg.next_bar()
    grant = reg.open_check()
    raw = {}
    if pre:
        p = panel.build(
            study.market, study.panel_spec(pre[0], pre[1], sources), cache_dir=study.cache_dir
        )
        data.check_panel(p)
        raw["pre"] = panel.model_rows(study, p)
    tune = panel.model_rows(study, build_panel(study, "tuning", sources=sources))
    raw["tuning"] = in_stage(study, tune, "tuning")
    chk = panel.model_rows(study, build_panel(study, "check", grant, sources=sources))
    raw["check"] = in_stage(study, chk, "check")
    if holdout_reason:
        hs = holdout_study or study
        hs.holdout.unlock(hs.name, holdout_reason)
        hold = panel.model_rows(hs, build_panel(hs, "holdout", final=True, sources=sources))
        raw["holdout"] = in_stage(hs, hold, "holdout")
    frames, bench = {}, {}
    for name, f in raw.items():
        f = score.fill_costs(f.assign(score=sign * f[feature]).dropna(subset=["score"]))
        f = attach(f)
        bench[name] = benchmark(f)
        frames[name] = f[f["listed"] & f["ret"].notna()]
    cov = pd.concat(
        [
            f.assign(year=f["decision_time"].dt.year)
            .groupby(["year", "decision_time"])["score"]
            .agg(names="size")
            .reset_index()
            .groupby("year")["names"]
            .agg(weeks="size", median_names="median", min_names="min")
            for f in frames.values()
        ]
    )
    report(f"coverage (tradable names per decision, before any return):\n{cov.to_string()}")
    empty = [n for n, f in frames.items() if f.empty]
    if empty:  # stop before any return: a period with no tradable names is a data gap, not a result
        raise ValueError(f"no tradable names in {empty}: fill the data first")
    books = {n: score.band_long_short(f, enter, exit, min_names) for n, f in frames.items()}
    dropped = pd.concat([b["dropped"] for b in books.values()])
    report(
        f"dropped (< {min_names} names in a leg) by year: "
        f"{dropped.groupby(dropped.index.year).sum().astype(int).to_dict()}"
    )
    kept_w = {
        n: b[~b["dropped"]].join(bench[n].rename("bench"), how="inner") for n, b in books.items()
    }
    pooled = pd.concat(kept_w.values()).dropna(subset=["net", "bench"])
    fit = score.ols_alpha_hc3(pooled["net"], pooled["bench"])
    m50, m80 = (score.mde(fit["resid_sd"], fit["n"], bar, q) for q in (0.5, 0.8))
    report(
        f"power: residual sd {fit['resid_sd'] * 1e4:.0f} bp/period over {fit['n']} periods, bar "
        f"{bar:.3f}: MDE {m50 * 1e4:.0f} bp at 50% power, {m80 * 1e4:.0f} bp at 80%"
    )
    kept = bool(fit["t_alpha"] >= bar and fit["alpha"] > 0)
    report(
        f"PRIMARY alpha vs benchmark: {fit['alpha'] * 1e4:+.1f} bp/period, HC3 t "
        f"{fit['t_alpha']:.2f} (bar {bar:.3f}), beta {fit['beta']:.2f} -> "
        f"{'KEPT' if kept else 'not kept'}"
    )
    ex = score.ols_alpha_hc3(pooled["net"] - pooled["funding"], pooled["bench"])
    per = {}
    for n, w in kept_w.items():
        w = w.dropna(subset=["net", "bench"])
        a = score.ols_alpha_hc3(w["net"], w["bench"]) if len(w) > 3 else {}
        per[n] = {
            "periods": len(w),
            "alpha_bp": round(a.get("alpha", np.nan) * 1e4, 1),
            "t_hc3": round(a.get("t_alpha", np.nan), 2),
            "beta": round(a.get("beta", np.nan), 2),
            "dollar_neutral_bp": round(float(w["net"].mean()) * 1e4, 1),
            "dollar_neutral_t": round(period_t(study, w["net"], overlap=False), 2),
        }
    wealth = (1 + pooled["net"]).cumprod()
    secondary = {
        "dollar_neutral_bp": round(float(pooled["net"].mean()) * 1e4, 1),
        "dollar_neutral_t": round(period_t(study, pooled["net"], overlap=False), 2),
        "alpha_ex_funding_bp": round(ex["alpha"] * 1e4, 1),
        "alpha_ex_funding_t": round(ex["t_alpha"], 2),
        "carry": bool(ex["alpha"] < fit["alpha"] / 2),
        "gross_bp": round(float(pooled["gross"].mean()) * 1e4, 1),
        "funding_bp": round(float(pooled["funding"].mean()) * 1e4, 1),
        "cost_bp": round(float(pooled["cost"].mean()) * 1e4, 1),
        "worst_period_bp": round(float(pooled["net"].min()) * 1e4, 1),
        "max_drawdown": round(float((wealth / wealth.cummax() - 1).min()), 3),
        "short_leg_over_30pct": int((pooled["short_ret"] > 0.30).sum()),
    }
    report(f"secondary: {secondary}")
    report("per period:\n" + pd.DataFrame(per).T.to_string())
    report("not modelled: squeezes, liquidations, margin calls, position limits, perp-spot basis")
    row = {
        "kind": "confirm",
        "name": f"long-short {'+' if sign > 0 else '-'}{feature} band {enter}/{exit}",
        "features": feature,
        "scope": "universal",
        "metric": "pooled long-short alpha vs benchmark (HC3): " + "+".join(frames),
        "bar_tune": round(bar, 3),
        "check_used": True,
        "t_check": round(fit["t_alpha"], 3),
        "bar_check": round(bar, 3),
        "gain_check": round(fit["alpha"], 6),
        "kept": kept,
        "note": (note + " " + str({"beta": round(fit["beta"], 3), **secondary})).strip(),
        "mde_50": round(m50, 5),
        "mde_80": round(m80, 5),
    }
    return reg.record(row) | {
        "periods": per,
        "secondary": secondary,
        "coverage": cov,
        "books": pooled,  # the weekly book behind every number, for later descriptive lines
    }


def test_candidate(
    study, frame, baseline: list[str], c: Candidate, wf=None, note: str = ""
) -> dict:
    """Judge, apply both bars, and log the result: the only way a candidate becomes a test."""
    wf = wf or walk_forward(study)
    reg = study.registry
    bar = reg.next_bar()
    base = _base(study, frame, baseline, wf, "tuning")  # the baseline needs no candidate column
    noise, n, src = _ic_noise(study, frame, base, c.scope, wf)
    what = "rank IC gain over the baseline" if baseline else "rank IC"
    power = power_check("rank IC", noise, n, bar, label_overlap(study), src, what)
    res = judge(study, frame, baseline, c, wf, base)
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
        "mde_50": round(power["mde_50"], 5),
        "mde_80": round(power["mde_80"], 5),
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


def cohort_power(study: Study) -> dict | None:
    """The cohort test's MDE before any return is computed: months from the first entry to the
    data's end, noise = the tracking error assumed in the YAML (`cohort_test.noise_monthly`)."""
    c = study.cfg["cohort_test"]
    if not c.get("noise_monthly"):
        _log(
            "cohort test: set cohort_test.noise_monthly (the assumed tracking error) to see its MDE"
        )
        return None
    first = pd.Period(pd.Timestamp(c["first_formation"]), "M") + 1
    last = pd.Period(pd.Timestamp(c["last_formation"]), "M") + 3 * int(c["hold_quarters"])
    if study.cfg.get("data_end"):
        last = min(last, pd.Period(pd.Timestamp(study.cfg["data_end"]), "M"))
    months = (last - first).n + 1
    return power_check(
        "excess return per month",
        float(c["noise_monthly"]),
        months,
        study.registry.next_bar(),
        1,
        "the tracking error assumed in the YAML",
        "monthly net excess over the benchmark",
    )


def cohort_test_logged(study: Study) -> bool:
    """Whether the pre-registered cohort test already has its one look in the registry."""
    return bool((study.registry.own()["name"] == study.cfg["cohort_test"]["name"]).any())


def log_cohort_test(study: Study, res: dict, note: str, power: dict | None = None) -> dict:
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
        | (
            {"mde_50": round(power["mde_50"], 6), "mde_80": round(power["mde_80"], 6)}
            if power
            else {}
        )
    )
