"""Models and the walk-forward that keeps them honest.

Inputs are per-period percentile ranks (centred at 0, missing -> 0), so scales and outliers don't
matter and a model only ever compares entities known at the same decision time. The target is
demeaned within a comparison group, so the model learns WHICH entities beat their peers, not where
the market went.

Walk-forward: the decision times are cut into blocks (calendar years by default). Each block's model
trains only on rows whose label had ENDED (plus an optional embargo) before the block's first
decision. That's the purge: no training label overlaps the test period.

A model is anything with fit(X, y) -> self and predict(X); `coef_` is read if present (cards, weight
reports). Ridge is the default; trees or an ensemble of many weak signals plug in the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class Model(Protocol):
    def fit(self, X, y, sample_weight=None): ...

    def predict(self, X): ...


def ridge(alpha: float = 10.0) -> Callable[[], Model]:
    return lambda: Ridge(alpha=alpha)


def trees() -> Callable[[], Model]:
    """Gradient-boosted trees, settings fixed in advance (jev.model.trees): one try, no tuning."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return lambda: HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=5.0,
        max_features=0.5,
        early_stopping=False,
        random_state=0,
    )


class MeanEnsemble:
    """Average of several models' predictions (e.g. one weak model per signal family)."""

    def __init__(self, factories: list[Callable[[], Model]]):
        self.models = [f() for f in factories]

    def fit(self, X, y, sample_weight=None):
        for m in self.models:
            m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


MODELS = {"ridge": ridge, "trees": trees}


def rank_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile rank within each decision time, centred at 0; missing -> 0 (the middle)."""
    g = frame.groupby("decision_time")
    return pd.DataFrame(
        {c: g[c].rank(pct=True).fillna(0.5) - 0.5 for c in cols}, index=frame.index
    )


@dataclass
class WalkForward:
    target: str = "fwd_rank"
    demean_by: tuple[str, ...] = (
        "decision_time",
        "group",
    )  # ("batch",) for legacy batches
    model: Callable[[], Model] = field(default_factory=ridge)
    block: str = "year"  # refit once per calendar year of decision dates
    min_train_periods: int = 12
    embargo: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))
    # Legacy only: purge on decision_time + N calendar days instead of the real label end. jev used
    # 31 days; at a few year boundaries that let in training labels that ended up to 2 days after
    # the test year's first decision. Kept so the old numbers can be reproduced exactly.
    legacy_purge_days: int | None = None
    keep: tuple[str, ...] = ("group", "fwd_return")
    # Optional recency weighting: a training row's weight halves every `half_life` decision periods
    # back from the block being scored (None = equal weights, the default). "auto" picks it by rank
    # IC on the last `auto_valid` closed periods before the refit (a nested walk-forward: only data
    # whose labels had ended). auto_every="year": chosen once a year, at the year's first refit,
    # from `auto_steps`, moving at most `auto_max_step` grid steps from last year's choice, so the
    # choice can't jump month to month. auto_every="block": the old rule (re-chosen at every refit
    # from `auto_grid`), kept to reproduce earlier runs.
    half_life: float | str | None = None
    auto_every: str = "year"
    auto_steps: tuple = (6, 12, 24, 36)
    auto_max_step: int = 1
    auto_grid: tuple = (None, 36, 24, 12, 6)
    auto_valid: int = 12

    def run(
        self, frame: pd.DataFrame, features: list[str], calendar, last_block=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Out-of-sample score for every row of every block that has enough history.

        Returns (scored rows, weights per block if the model has coef_)."""
        self.chosen_: dict = {}
        self.year_choice_: dict = {}
        X = rank_features(frame, features)
        keys = list(self.demean_by)
        y = frame[self.target] - frame.groupby(keys)[self.target].transform("mean")
        local = calendar.local_date(frame["decision_time"])
        blocks = (
            local.dt.year if self.block == "year" else local.dt.to_period(self.block)
        )
        if self.legacy_purge_days is not None:
            ends = frame["decision_time"] + pd.Timedelta(days=self.legacy_purge_days)
        else:
            ends = frame["label_end"]
        ends = ends + self.embargo
        parts, weights = [], []
        for b in sorted(blocks.unique()):
            if last_block is not None and b > last_block:
                continue
            test = (blocks == b).to_numpy()
            first = frame.loc[test, "decision_time"].min()
            train = (ends < first).to_numpy() & y.notna().to_numpy()
            if frame.loc[train, "decision_time"].nunique() < self.min_train_periods:
                continue
            periods = sorted(frame["decision_time"].unique())
            pos = {t: i for i, t in enumerate(periods)}
            hl = self.half_life
            if hl == "auto":
                if self.auto_every == "block":
                    hl = self._choose(frame, X, y, ends, train, pos, self.auto_grid)
                else:
                    hl = self._yearly(calendar, frame, X, y, ends, train, pos, first)
                self.chosen_[b] = hl
            m = self._fit(frame, X, y, train, first, pos, hl)
            keep = ["entity_id", "decision_time", *[c for c in self.keep if c in frame]]
            parts.append(
                frame.loc[test, keep].assign(score=m.predict(X[test]), block=b)
            )
            if hasattr(m, "coef_"):
                weights.append(pd.Series(m.coef_, index=features, name=b))
        scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return scored, pd.DataFrame(weights)

    def _fit(self, frame, X, y, train, first, pos, hl):
        if not hl:
            return self.model().fit(X[train], y[train])
        age = pos[first] - frame.loc[train, "decision_time"].map(pos).to_numpy()
        return self.model().fit(
            X[train], y[train], sample_weight=0.5 ** (age / float(hl))
        )

    def _yearly(self, calendar, frame, X, y, ends, train, pos, first):
        year = int(calendar.local_date(pd.Series([first])).dt.year.iloc[0])
        if year not in self.year_choice_:
            steps = list(self.auto_steps)
            prev = [
                v
                for k, v in sorted(self.year_choice_.items())
                if k < year and v is not None
            ]
            if prev:
                i = steps.index(prev[-1])
                lo, hi = max(0, i - self.auto_max_step), i + self.auto_max_step + 1
                steps = steps[lo:hi]
            self.year_choice_[year] = self._choose_rolling(
                frame, X, y, ends, train, pos, tuple(steps)
            )
        return self.year_choice_[year]

    def _choose_rolling(self, frame, X, y, ends, train, pos, grid):
        """Each candidate refit month by month through the last `auto_valid` closed periods (each
        fit uses only labels ended before that month) and scored on that month: a half-life that
        adapts to a regime change can only show it if the validation lets it adapt."""
        times = sorted(frame.loc[train, "decision_time"].unique())
        if len(times) < self.min_train_periods + self.auto_valid:
            return None
        dt = frame["decision_time"].to_numpy()
        ok = y.notna().to_numpy()
        score = {}
        for hl in grid:
            ics = []
            for t in times[-self.auto_valid :]:
                inner = (ends < t).to_numpy() & ok
                if frame.loc[inner, "decision_time"].nunique() < self.min_train_periods:
                    continue
                m = self._fit(frame, X, y, inner, t, pos, hl)
                at = train & (dt == t)
                p = pd.Series(m.predict(X[at]))
                ics.append(p.corr(pd.Series(y[at].to_numpy()), method="spearman"))
            score[hl] = float(np.nanmean(ics)) if ics else -np.inf
        return max(
            grid, key=lambda h: (score[h], -grid.index(h))
        )  # ties: the earlier step

    def _choose(self, frame, X, y, ends, train, pos, grid):
        times = sorted(frame.loc[train, "decision_time"].unique())
        if len(times) < self.min_train_periods + self.auto_valid:
            return None
        v0 = times[-self.auto_valid]
        inner = (ends < v0).to_numpy() & y.notna().to_numpy()
        valid = train & (frame["decision_time"] >= v0).to_numpy()
        best, best_ic = None, -np.inf
        for hl in grid:
            m = self._fit(frame, X, y, inner, v0, pos, hl)
            g = pd.DataFrame(
                {
                    "t": frame.loc[valid, "decision_time"],
                    "p": m.predict(X[valid]),
                    "y": y[valid],
                }
            )
            ic = (
                g.groupby("t")
                .apply(
                    lambda d: d["p"].corr(d["y"], method="spearman"),
                    include_groups=False,
                )
                .mean()
            )
            if ic > best_ic + 1e-12:
                best, best_ic = hl, ic
        return best


def contributions(
    frame: pd.DataFrame, features: list[str], weights: pd.DataFrame, calendar
) -> pd.DataFrame:
    """weight x ranked input per row (the block's weights): what pushed each score up or down."""
    X = rank_features(frame, features)
    if "block" in frame:
        block = frame["block"]
    elif len(weights) and isinstance(weights.index[0], pd.Period):
        block = calendar.local_date(frame["decision_time"]).dt.to_period(
            weights.index[0].freqstr
        )
    else:
        block = calendar.local_date(frame["decision_time"]).dt.year
    ok = block.isin(weights.index).to_numpy()
    w = weights.reindex(block[ok]).to_numpy()
    return pd.DataFrame(X[ok].to_numpy() * w, columns=features, index=frame.index[ok])
