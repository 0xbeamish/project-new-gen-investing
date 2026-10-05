"""Observations: the one shape every data source emits.

Long format, one row per (entity, moment, feature):

  entity_id     the market's id for the thing being ranked (a stock code, a token symbol)
  available_at  UTC timestamp from which a decision may use the value. The publication time, or
                LATER if the source adds a conservative lag (e.g. "usable from the next midnight").
                Never earlier. This is the single most important column in the engine.
  source        the source's name
  feature       the feature's name (unique across sources)
  value         float. NaN is allowed and meaningful: "the latest report had no value for this",
                so it masks older values instead of letting a stale one show through.

Ties (same entity, feature and available_at): the row emitted LAST wins, so emit in publication order.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

OBS_COLUMNS = ["entity_id", "available_at", "source", "feature", "value"]


@runtime_checkable
class Source(Protocol):
    """A data source plug-in.

    fetch(start, end)        download whatever is missing into the source's own cache; idempotent
    observations(start, end) from the cache only: every observation with available_at < end. `start`
                             says how far back the caller needs; older rows are allowed (a 10-K
                             filed before `start` may still be the latest one)

    Optional, for values that are cheapest to compute only where they're needed (bar-derived
    features such as momentum): observations_at(rows) with rows = entity_id, decision_time. The
    rows it returns must still carry the true available_at of the data they were computed from.
    """

    name: str

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> None: ...

    def observations(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def utc(ts) -> pd.Series | pd.Timestamp | pd.DatetimeIndex:
    """Coerce to tz-aware UTC nanoseconds. Naive input is rejected: guessing a zone is how leaks start."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            raise ValueError(f"naive timestamp {ts}: give it a time zone")
        return ts.tz_convert("UTC").as_unit("ns")
    if isinstance(ts, pd.DatetimeIndex):
        if ts.tz is None:
            raise ValueError("naive DatetimeIndex: give it a time zone")
        return ts.tz_convert("UTC").as_unit("ns")
    s = pd.Series(ts)
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        raise TypeError("available_at must be tz-aware")
    return s.dt.tz_convert("UTC").dt.as_unit("ns")


def validate(obs: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Check the contract and return a clean copy (UTC ns, float values, stable column order)."""
    missing = [c for c in OBS_COLUMNS if c not in obs.columns]
    if missing:
        raise ValueError(f"observations missing columns {missing}")
    out = obs[OBS_COLUMNS].copy()
    out["available_at"] = utc(out["available_at"]).to_numpy()
    if out["available_at"].isna().any():
        raise ValueError("observations with no available_at")
    if source is not None and not (out["source"] == source).all():
        raise ValueError(f"rows not labelled with source {source!r}")
    out["entity_id"] = out["entity_id"].astype(str)
    out["feature"] = out["feature"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    return out.reset_index(drop=True)


def from_wide(
    wide: pd.DataFrame,
    source: str,
    entity_col: str = "entity_id",
    at_col: str = "available_at",
    features: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (entity, moment) with one column per feature -> observations (NaN cells kept)."""
    feats = features or [c for c in wide.columns if c not in (entity_col, at_col)]
    long = wide[[entity_col, at_col, *feats]].melt(
        id_vars=[entity_col, at_col], var_name="feature", value_name="value"
    )
    # melt stacks feature by feature; restore emission order within each feature
    long["_order"] = np.tile(np.arange(len(wide)), len(feats))
    long = long.sort_values(["feature", "_order"], kind="stable").drop(columns="_order")
    return long.rename(
        columns={entity_col: "entity_id", at_col: "available_at"}
    ).assign(source=source)[OBS_COLUMNS]


def rolling_window(
    events: pd.DataFrame,
    window: pd.Timedelta,
    aggregate,
    feature_names: list[str],
    source: str,
    baseline_entities: list[str] | None = None,
    baseline_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Window aggregates ("insiders buying in the last 182 days") as exact point-in-time observations.

    events: entity_id, available_at, plus whatever `aggregate` reads. The value at decision time T is
    aggregate(events with T - window < available_at < T). That value only changes when an event
    enters (at available_at) or leaves (at available_at + window), so one observation is emitted at
    each of those moments; the panel's "latest value before T" rule then reproduces the window exactly.
    `aggregate(frame) -> sequence of floats` must accept an empty frame (that's the baseline value,
    emitted at baseline_at for every entity in baseline_entities).
    """
    ents, ats, vals = [], [], []
    empty = list(aggregate(events.iloc[0:0]))
    base = (
        utc(baseline_at or pd.Timestamp("1990-01-01", tz="UTC"))
        .tz_localize(None)
        .to_datetime64()
    )
    for entity in baseline_entities or []:
        ents.append(str(entity))
        ats.append(base)
        vals.append(empty)
    ev_all = events.assign(
        available_at=utc(events["available_at"]).to_numpy()
    ).sort_values("available_at", kind="stable")
    w = window.to_timedelta64()
    for entity, ev in ev_all.groupby("entity_id", sort=False):
        at = ev["available_at"].dt.tz_localize(None).to_numpy()
        for p in np.unique(np.concatenate([at, at + w])):
            # value in force just after p: events with p - window < t <= p
            lo = np.searchsorted(at, p - w, side="right")
            hi = np.searchsorted(at, p, side="right")
            ents.append(str(entity))
            ats.append(p)
            vals.append(list(aggregate(ev.iloc[lo:hi])))
    wide = pd.DataFrame(vals, columns=feature_names, dtype=float)
    wide.insert(
        0,
        "available_at",
        pd.to_datetime(np.array(ats, dtype="datetime64[ns]")).tz_localize("UTC"),
    )
    wide.insert(0, "entity_id", ents)
    return from_wide(wide, source, features=feature_names)
