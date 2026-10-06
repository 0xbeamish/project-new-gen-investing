"""Panel builder: decision schedule x universe -> one row per (decision time, entity).

For each row:
  features   the LATEST observation of each feature with available_at < decision_time, blanked if
             older than the feature's max_age_days (calendar days, local dates, from the day it
             became usable to the decision date)
  label      the market's forward return from the first bar closing after the decision
  provenance the available_at behind every filled cell, so engine.pit can re-check the panel

Generalizes jev.smallpanel (which hand-joined each source with its own merge_asof rule).
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine import observations as obsmod


@dataclass
class SourceSpec:
    source: object  # an engine.observations.Source
    features: list[str] | None = None  # None = every feature the source emits
    max_age_days: int | None = None  # None = never stale
    lookback_days: int = 3650  # how far before the first decision to read observations


@dataclass
class PanelSpec:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    schedule: str = "monthly"
    horizon: int = 21  # bars
    sources: list[SourceSpec] = field(default_factory=list)


@dataclass
class Panel:
    frame: pd.DataFrame  # entity_id, decision_time, group, attrs..., labels, features
    available_at: dict[
        str, pd.Series
    ]  # feature -> available_at per frame row (NaT = empty)
    features: list[str]

    def provenance(self) -> pd.DataFrame:
        parts = [
            pd.DataFrame(
                {
                    "entity_id": self.frame["entity_id"].to_numpy(),
                    "decision_time": self.frame["decision_time"].to_numpy(),
                    "feature": f,
                    "available_at": at.to_numpy(),
                }
            )
            for f, at in self.available_at.items()
        ]
        if not parts:
            return pd.DataFrame(
                columns=["entity_id", "decision_time", "feature", "available_at"]
            )
        out = pd.concat(parts, ignore_index=True)
        for c in ("decision_time", "available_at"):
            out[c] = pd.to_datetime(out[c], utc=True)
        return out

    def subset(self, mask) -> Panel:
        mask = np.asarray(mask, dtype=bool)
        return Panel(
            self.frame[mask].reset_index(drop=True),
            {f: s[mask].reset_index(drop=True) for f, s in self.available_at.items()},
            list(self.features),
        )


def as_utc(x) -> pd.Timestamp:
    """A bare date means midnight UTC; a tz-aware stamp is converted."""
    t = pd.Timestamp(x)
    return obsmod.utc(t if t.tzinfo else t.tz_localize("UTC"))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cached(cache_dir: Path | None, key: str, fn):
    if cache_dir is None:
        return fn()
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def _fingerprint(*parts) -> str:
    return hashlib.sha1(pickle.dumps(parts)).hexdigest()[:12]


def universe_rows(market, times: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for t in times:
        u = market.universe(t)
        if len(u):
            frames.append(u.assign(decision_time=t))
    rows = pd.concat(frames, ignore_index=True)
    if "group" not in rows:
        rows["group"] = "all"
    rows["entity_id"] = rows["entity_id"].astype(str)
    rows["decision_time"] = pd.to_datetime(rows["decision_time"], utc=True)
    return rows


def attach(
    rows: pd.DataFrame,
    obs: pd.DataFrame,
    calendar,
    max_age_days: int | None,
) -> tuple[pd.Series, pd.Series]:
    """Latest obs.value with available_at < decision_time per row; returns (value, available_at)."""
    left = pd.DataFrame(
        {
            "_pos": np.arange(len(rows)),
            "entity_id": rows["entity_id"].to_numpy(),
            "decision_time": rows["decision_time"].to_numpy(),
        }
    ).sort_values("decision_time", kind="stable")
    right = obs[["entity_id", "available_at", "value"]].sort_values(
        "available_at", kind="stable"
    )
    m = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="available_at",
        by="entity_id",
        direction="backward",
        allow_exact_matches=False,  # strictly before the decision
    ).sort_values("_pos")
    value = pd.Series(m["value"].to_numpy(dtype=float), index=rows.index)
    at = pd.Series(
        pd.to_datetime(m["available_at"].to_numpy(), utc=True), index=rows.index
    )
    if max_age_days is not None:
        age = (
            calendar.local_date(m["decision_time"]).to_numpy()
            - calendar.local_date(m["available_at"]).to_numpy()
        ) / np.timedelta64(1, "D")
        stale = np.asarray(age > max_age_days)
        value[stale] = np.nan
        at[stale] = pd.NaT
    return value, at  # a NaN observation still counts as used: it masks older values


def build(
    market,
    spec: PanelSpec,
    cache_dir: Path | None = None,
    times: pd.DatetimeIndex | None = None,
    with_labels: bool = True,
) -> Panel:
    """with_labels=False: features only, no forward return is computed (e.g. a coverage check
    before a pre-registration, or a study that prices its own holdings)."""
    if times is None:
        times = market.calendar.decision_times(spec.start, spec.end, spec.schedule)
    times = times[(times >= as_utc(spec.start)) & (times < as_utc(spec.end))]
    rows = universe_rows(market, times)
    _log(f"panel: {len(times)} decision times, {len(rows):,} universe rows")
    if with_labels:
        labels = market.labels(rows[["entity_id", "decision_time"]], spec.horizon)
        labels["decision_time"] = pd.to_datetime(labels["decision_time"], utc=True)
        rows = rows.merge(labels, on=["entity_id", "decision_time"], how="left")
    else:
        nat = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
        rows = rows.assign(
            entry_time=nat, label_end=nat, fwd_return=np.nan, delisted=False
        )
    feats: dict[str, pd.Series] = {}
    ats: dict[str, pd.Series] = {}
    # generated markets change their data with their settings: key the cache on them too
    market_key = [market.fingerprint] if hasattr(market, "fingerprint") else []
    for s in spec.sources:
        src = s.source
        if hasattr(src, "observations_at"):
            key = _fingerprint(
                src.name,
                getattr(src, "cache_key", getattr(src, "params", None)),
                rows[["entity_id", "decision_time"]].to_numpy().tolist(),
                *market_key,
            )
            obs = _cached(
                cache_dir,
                f"obs_{src.name}_{key}",
                lambda src=src: obsmod.validate(
                    src.observations_at(rows[["entity_id", "decision_time"]]), src.name
                ),
            )
        else:
            start = times.min() - pd.Timedelta(days=s.lookback_days)
            key = _fingerprint(
                src.name,
                getattr(src, "cache_key", getattr(src, "params", None)),
                str(start),
                str(times.max()),
                *market_key,
            )
            obs = _cached(
                cache_dir,
                f"obs_{src.name}_{key}",
                lambda src=src, start=start: obsmod.validate(
                    src.observations(start, times.max()), src.name
                ),
            )
        wanted = s.features or list(dict.fromkeys(obs["feature"]))
        by_feature = dict(tuple(obs.groupby("feature", sort=False)))
        for f in wanted:
            if f in feats:
                raise ValueError(f"feature {f!r} emitted by two sources")
            o = by_feature.get(f)
            if o is None:
                feats[f] = pd.Series(np.nan, index=rows.index)
                ats[f] = pd.Series(
                    pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]"
                )
                continue
            feats[f], ats[f] = attach(rows, o, market.calendar, s.max_age_days)
        _log(f"  {src.name}: {len(obs):,} observations -> {len(wanted)} features")
    frame = pd.concat([rows, pd.DataFrame(feats, index=rows.index)], axis=1)
    return Panel(frame, ats, list(feats))
