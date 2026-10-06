"""The input contracts: observations (numbers) and documents (text), the calendar that turns local
dates into UTC decision times, and the point-in-time checker every panel goes through.

Observations, long format, one row per (entity, moment, feature):
  entity_id     the market's id for the thing being ranked (a stock code, a token symbol)
  available_at  UTC time from which a decision may use the value: the publication time, or LATER if
                the source adds a conservative lag ("usable from the next midnight"). Never earlier
  source        the source's name
  feature       the feature's name (unique across sources)
  value         float. NaN is meaningful: "the latest report had no value", so it masks older values
Ties (same entity, feature and available_at): the row emitted LAST wins, so emit in publication order.

Documents, one row per (document, entity): entity_id, available_at (same rule), doc_type (question
sets are chosen by it; "the previous document" is the previous one of the same type), doc_id
(unique per document), text (already masked), metadata (optional dict; keys can be history keys).

Point in time: a panel cell is legal only if the observation behind it has available_at strictly
before the decision time; a label only if its entry bar closes strictly after it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

OBS_COLUMNS = ["entity_id", "available_at", "source", "feature", "value"]
DOC_COLUMNS = ["entity_id", "available_at", "doc_type", "doc_id", "text"]


# ---------------------------------------------------------------- observations
@runtime_checkable
class Source(Protocol):
    """A numeric data source.

    fetch(start, end)        download whatever is missing into the source's own cache; idempotent;
                             the only step that may use the network
    observations(start, end) from the cache only: every observation with available_at < end. Older
                             rows are allowed (a 10-K filed before `start` may still be the latest)
    Optional observations_at(rows), rows = entity_id, decision_time: for values cheapest to compute
    only where needed (momentum from bars). Its rows still carry the true available_at.
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


def validate_observations(obs: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Check the observation contract; return a clean copy (UTC ns, float values, column order)."""
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


def empty_observations() -> pd.DataFrame:
    """No observations, with the contract's dtypes (a source with nothing to say)."""
    return pd.DataFrame(
        {
            "entity_id": pd.Series(dtype=str),
            "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "source": pd.Series(dtype=str),
            "feature": pd.Series(dtype=str),
            "value": pd.Series(dtype=float),
        }
    )


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
    return long.rename(columns={entity_col: "entity_id", at_col: "available_at"}).assign(
        source=source
    )[OBS_COLUMNS]


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
    aggregate(events with T - window < available_at < T). It only changes when an event enters (at
    available_at) or leaves (at available_at + window), so one observation is emitted at each of
    those moments and the panel's "latest value before T" rule reproduces the window exactly.
    `aggregate(frame) -> sequence of floats` must accept an empty frame: that baseline value is
    emitted at baseline_at for every entity in baseline_entities.
    """
    ents, ats, vals = [], [], []
    empty = list(aggregate(events.iloc[0:0]))
    base = (
        utc(baseline_at or pd.Timestamp("1990-01-01", tz="UTC")).tz_localize(None).to_datetime64()
    )
    for entity in baseline_entities or []:
        ents.append(str(entity))
        ats.append(base)
        vals.append(empty)
    ev_all = events.assign(available_at=utc(events["available_at"]).to_numpy()).sort_values(
        "available_at", kind="stable"
    )
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


# ---------------------------------------------------------------- documents
@runtime_checkable
class DocumentSource(Protocol):
    """The text twin of Source: fetch(start, end) fills its cache; documents(start, end) reads it
    and returns every document with available_at < end."""

    name: str

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> None: ...

    def documents(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def validate_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """Check the document contract; return a clean copy sorted by available_at (publication order)."""
    missing = [c for c in DOC_COLUMNS if c not in docs.columns]
    if missing:
        raise ValueError(f"documents missing columns {missing}")
    out = docs.copy()
    out["available_at"] = utc(out["available_at"]).to_numpy()
    if out["available_at"].isna().any():
        raise ValueError("documents with no available_at")
    out["entity_id"] = out["entity_id"].astype(str)
    out["doc_id"] = out["doc_id"].astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    if "metadata" not in out:
        out["metadata"] = [{} for _ in range(len(out))]
    if out.duplicated(["doc_id", "entity_id"]).any():
        raise ValueError("duplicate (doc_id, entity_id) rows")
    return out.sort_values("available_at", kind="stable").reset_index(drop=True)


def content_hash(*parts: str) -> str:
    """Cache key for a reading: same text + same questions + same reader -> same answers."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


class FrameDocuments:
    """A DocumentSource over an in-memory frame (tests, small corpora, the CSV market)."""

    def __init__(self, name: str, frame: pd.DataFrame):
        self.name = name
        self.frame = validate_documents(frame)

    def fetch(self, start, end) -> None:
        """Nothing to download."""

    def documents(self, start, end) -> pd.DataFrame:
        """Every document available before `end`."""
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        return self.frame[self.frame["available_at"] < utc(end)].reset_index(drop=True)


# ---------------------------------------------------------------- calendar
FREQS = {
    "trading": {"monthly": "BME", "weekly": "W-FRI", "daily": "B"},
    "continuous": {"monthly": "ME", "weekly": "W-SUN", "daily": "D"},
}


@dataclass(frozen=True)
class TradingCalendar:
    """Stock-style: business days with a session close (e.g. 16:00 New York); decisions a little after
    the close, so that day's bar is known but the next one isn't."""

    tz: str = "America/New_York"
    close: str = "16:00"
    decide_after_close: str = "30min"
    kind: str = "trading"

    def at(self, dates, time: str) -> pd.DatetimeIndex:
        """Local calendar dates + a local clock time -> UTC instants."""
        d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        return (
            (d + pd.Timedelta(time + ":00" if time.count(":") == 1 else time))
            .tz_localize(self.tz)
            .tz_convert("UTC")
            .as_unit("ns")
        )

    def bar_close(self, dates) -> pd.DatetimeIndex:
        """When each local date's bar closes, in UTC."""
        return self.at(dates, self.close)

    def decision_times(self, start, end, freq: str) -> pd.DatetimeIndex:
        """Decision instants (UTC) on the schedule: monthly | weekly | daily, or a pandas frequency."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        dates = pd.date_range(
            start.tz_localize(None).normalize() if start.tzinfo else start,
            end.tz_localize(None) if end.tzinfo else end,
            freq=FREQS[self.kind].get(freq, freq),
        )
        return self.bar_close(dates) + pd.Timedelta(self.decide_after_close)

    def local_date(self, ts) -> pd.Series:
        """UTC instants -> local calendar dates (midnight, naive)."""
        s = pd.Series(ts)
        return s.dt.tz_convert(self.tz).dt.tz_localize(None).dt.normalize()

    def next_midnight(self, ts) -> pd.Series:
        """The first local midnight strictly after each instant, in UTC: the conservative stamp for
        anything dated by day only, or published during a day treated as a unit."""
        s = pd.Series(ts)
        if not isinstance(s.dtype, pd.DatetimeTZDtype):
            raise TypeError("next_midnight needs tz-aware instants")
        local = s.dt.tz_convert(self.tz).dt.tz_localize(None)
        nxt = local.dt.floor("D") + pd.Timedelta(days=1)
        return nxt.dt.tz_localize(self.tz).dt.tz_convert("UTC").dt.as_unit("ns")

    def date_available(self, dates) -> pd.Series:
        """Dates with no time of day (e.g. an SEC filing date): usable from the next local midnight."""
        d = pd.Series(pd.to_datetime(dates)).dt.normalize()
        return self.next_midnight(d.dt.tz_localize(self.tz))


@dataclass(frozen=True)
class ContinuousCalendar(TradingCalendar):
    """24/7 markets (crypto): every day is a trading day; decisions at a fixed UTC time."""

    tz: str = "UTC"
    close: str = "00:00"
    decide_after_close: str = "0min"
    kind: str = "continuous"


def make_calendar(cfg: dict) -> TradingCalendar:
    """A calendar from the market YAML's `calendar:` block (kind: trading | continuous)."""
    kind = cfg.get("kind", "trading")
    cls = ContinuousCalendar if kind == "continuous" else TradingCalendar
    return cls(**{k: v for k, v in cfg.items() if k != "kind"}, kind=kind)


# ---------------------------------------------------------------- point-in-time checks
class PointInTimeError(AssertionError):
    pass


def check_cells(provenance: pd.DataFrame, limit: int = 5) -> int:
    """provenance: entity_id, decision_time, feature, available_at (NaT = cell left empty).
    Returns the number of filled cells checked; raises PointInTimeError on any leak."""
    used = provenance.dropna(subset=["available_at"])
    bad = used[used["available_at"] >= used["decision_time"]]
    if len(bad):
        raise PointInTimeError(
            f"{len(bad):,} panel cells use data not yet available at decision time, e.g.\n"
            + bad.head(limit).to_string(index=False)
        )
    return len(used)


def check_labels(panel: pd.DataFrame, limit: int = 5) -> int:
    """Every label must start after its decision: entry_time > decision_time."""
    has = panel.dropna(subset=["entry_time"])
    bad = has[has["entry_time"] <= has["decision_time"]]
    if len(bad):
        raise PointInTimeError(
            f"{len(bad):,} labels start at or before their decision time, e.g.\n"
            + bad[["entity_id", "decision_time", "entry_time"]].head(limit).to_string(index=False)
        )
    return len(has)


def check_panel(panel) -> dict:
    """Both checks on an engine.panel.Panel (it records the available_at behind every filled cell,
    so the check re-reads what was used instead of trusting the builder); counts for the log."""
    return {
        "cells_checked": check_cells(panel.provenance()),
        "labels_checked": check_labels(panel.frame),
    }
