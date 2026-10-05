"""One YAML per market (markets/<name>.yaml) -> a ready Study: market, sources, specs, registry.

Keys (see markets/us_smallcap.yaml for a full example):
  plugin       module under engine.markets (or a dotted path) exposing build(cfg)
  calendar     engine.calendar settings
  universe     plug-in specific universe rules
  labels       horizon (bars), winsorize [lo, hi], plug-in label rules
  schedule     monthly | weekly | daily
  sources      name -> {features, max_age_days, lookback_days, params}
  features     named feature sets: baseline (the model new candidates must beat) and others
  model        name (ridge | trees), alpha, target {within}, demean_by, block, min_train_periods,
               embargo_days, legacy_purge_days, batches {size, by, seed}
  portfolio    rules for the scorer (band / topk / periodic)
  periods      tuning [start, end), check [start, end), holdout_start
  registry     file, inherit [legacy logs], check_limit, holdout_unlock_log
  discovery    transforms, candidates, max_tests, no_progress
  spend        ledger path, caps {step: usd}, funds {provider: usd} (engine.spend)
  decider      kind (none | jev | claude) + options; feedback note on by default when set
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from engine import markets
from engine.panel import PanelSpec, SourceSpec
from engine.registry import HoldoutLock, Periods, Registry

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Study:
    name: str
    cfg: dict
    market: object
    sources: dict  # name -> SourceSpec
    periods: Periods
    registry: Registry
    holdout: HoldoutLock
    cache_dir: Path

    def panel_spec(
        self, start, end, source_names: list[str] | None = None
    ) -> PanelSpec:
        names = source_names or list(self.sources)
        return PanelSpec(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            schedule=self.cfg.get("schedule", "monthly"),
            horizon=int(self.cfg["labels"]["horizon"]),
            sources=[self.sources[n] for n in names],
        )

    def feature_set(self, name: str) -> list[str]:
        sets = self.cfg.get("features", {})
        out: list[str] = []
        for item in sets[name]:
            # "@other_set" includes another named set
            out += self.feature_set(item[1:]) if item.startswith("@") else [item]
        return list(dict.fromkeys(out))


def path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read(market: str, cfg_path: str | Path | None = None) -> dict:
    return yaml.safe_load(path(cfg_path or f"markets/{market}.yaml").read_text())


def load(market: str, cfg_path: str | Path | None = None) -> Study:
    return from_dict(market, read(market, cfg_path))


def from_dict(market: str, cfg: dict) -> Study:
    plugin = markets.load(cfg.get("plugin", market))
    mkt, factories = plugin.build(cfg)
    specs = {}
    for name, s in (cfg.get("sources") or {}).items():
        s = s or {}
        if (
            s.get("type") == "text"
        ):  # a generic text source: documents + questions + reader
            from engine.spend import Ledger
            from engine.text.source import TextSource

            src = TextSource(
                mkt,
                s.get("params"),
                plugin=plugin,
                ledger=Ledger.from_config(cfg.get("spend"), ROOT),
                cache_dir=path(cfg.get("cache_dir", f".engine_cache/{market}")),
                root=ROOT,
                name=name,
            )
        else:
            src = factories[s.get("type", name)](mkt, s.get("params"))
        specs[name] = SourceSpec(
            source=src,
            features=s.get("features"),
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
        cache_dir=path(cfg.get("cache_dir", f".engine_cache/{market}")),
    )
