"""Bring your own data, no code: a market built from CSV files named in the YAML's `csv:` block.

  prices     entity, date, close [, volume] [, group]. One row per entity per bar; `date` is the
             local date whose bar closes at the calendar's `close` time. A series that stops for
             good (no bar in the last `ended_if_no_bar_for_days` of the file) is a delisting: its
             last window gets `delisted_return` added. Entities stay in the universe only while
             they trade (last bar within `max_stale_days` of the decision)
  signals    optional: entity, available_at, feature, value. One row per published value
  documents  optional: entity, available_at, doc_type, text [, doc_id]. Read by a question set
             through a `type: text` source with `documents: csv`

available_at, for signals and documents:
  with a time zone ("2021-03-07T12:00:00Z")  used as is
  a date only ("2021-03-07")                 usable from the next local midnight (the safe guess)
  a naive time ("2021-03-07 12:00")          read in the calendar's time zone
Costs are a constant round trip (`cost_bps`). The calendar is `trading` (business days) or
`continuous` (24/7, crypto); the schedule, horizon and periods come from the YAML as for any market.
See markets/csv_example.yaml and examples/csv/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine import data
from engine.market import forward_returns
from engine.run import path as repo_path


def _read(path: str | Path) -> pd.DataFrame:
    f = pd.read_csv(
        path, dtype={"entity": str, "entity_id": str}, keep_default_na=False, na_values=[""]
    )
    return f.rename(columns={"entity_id": "entity"})


def parse_available_at(col: pd.Series, calendar) -> pd.Series:
    """Timestamps -> UTC by the rules in the module docstring (never guessing earlier)."""
    raw = col.astype(str).str.strip()
    has_tz = raw.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True)
    date_only = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns, UTC]")
    if has_tz.any():
        out[has_tz] = pd.to_datetime(raw[has_tz], utc=True, format="ISO8601")
    if date_only.any():
        out[date_only] = calendar.date_available(raw[date_only]).to_numpy()
    rest = ~has_tz & ~date_only
    if rest.any():
        local = pd.to_datetime(raw[rest], format="ISO8601")
        later = np.zeros(len(local), dtype=bool)  # a DST-ambiguous hour reads as the later instant
        out[rest] = local.dt.tz_localize(
            calendar.tz, ambiguous=later, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    if out.isna().any():
        raise ValueError(f"unreadable available_at, e.g. {raw[out.isna()].iloc[0]!r}")
    return out


class CsvMarket:
    """The market contract over prices.csv; costs constant; groups optional."""

    name = "csv"

    def __init__(self, cfg: dict):
        c = cfg["csv"]
        files = {k: repo_path(c[k]) for k in ("prices", "signals", "documents") if c.get(k)}
        self.cfg, self.files = cfg, files
        self.calendar = data.make_calendar(cfg.get("calendar", {}))
        self.cost = float(c.get("cost_bps", 20.0))
        self.delisted_return = float(c.get("delisted_return", 0.0))
        self.max_stale = pd.Timedelta(days=float(c.get("max_stale_days", 5)))
        self.min_dollar_volume = c.get("min_dollar_volume")
        labels = cfg.get("labels", {})
        self.min_history = int(labels.get("min_history_bars", 0))
        self.drop_flat = bool(labels.get("drop_flat", False))
        ended_days = int(c.get("ended_if_no_bar_for_days", 10))
        # edited files must not hit a stale panel cache: key it on what the files are now
        self.fingerprint = str(
            [(k, p.stat().st_size, p.stat().st_mtime_ns) for k, p in sorted(files.items())]
        )
        px = _read(files["prices"])
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
        px = px.sort_values(["entity", "date"], kind="stable").reset_index(drop=True)
        px["close_at"] = self.calendar.bar_close(px["date"])
        self.px = px
        self.series = {e: g.reset_index(drop=True) for e, g in px.groupby("entity", sort=True)}
        end = px["close_at"].max()
        self.ended = {
            e: bool(g["close_at"].iloc[-1] < end - pd.Timedelta(days=ended_days))
            for e, g in self.series.items()
        }

    def entities(self) -> list[str]:
        """Every entity in prices.csv."""
        return list(self.series)

    def universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Entities with a bar closed before as_of, recent enough (and liquid enough, if set)."""
        rows = []
        for e, g in self.series.items():
            i = int(g["close_at"].searchsorted(as_of, side="left"))  # bars closed before as_of
            if i == 0 or as_of - g["close_at"].iloc[i - 1] > self.max_stale:
                continue
            if self.min_dollar_volume is not None and "volume" in g:
                last = g.iloc[max(0, i - 21) : i]
                if (last["close"] * last["volume"]).mean() < float(self.min_dollar_volume):
                    continue
            row = {"entity_id": e}
            if "group" in g:
                row["group"] = str(g["group"].iloc[i - 1])
            rows.append(row)
        return pd.DataFrame(rows)

    def labels(self, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forward returns: enter at the first close after the decision, exit `horizon` bars later."""
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            s = self.series[e]
            lab = forward_returns(
                pd.DatetimeIndex(s["close_at"]),
                s["close"].astype(float).to_numpy(),
                pd.DatetimeIndex(g["decision_time"]),
                horizon,
                self.ended[e],
                self.delisted_return,
                self.min_history,
                self.drop_flat,
            )
            out.append(lab.assign(entity_id=e))
        lab = pd.concat(out, ignore_index=True)
        for col in ("decision_time", "entry_time", "label_end"):
            lab[col] = pd.to_datetime(lab[col], utc=True)
        lab["delisted"] = lab["delisted"].fillna(False).astype(bool)
        return lab

    def cost_bps(self, rows: pd.DataFrame) -> pd.Series:
        """The constant round trip."""
        return pd.Series(self.cost, index=rows.index)

    def daily_returns(self, codes) -> tuple[pd.DataFrame, pd.Series]:
        """Daily simple returns (local dates x entities) and the last date of series that ended;
        an ended series' last return carries `delisted_return` (for the cohort test)."""
        cols, ended = {}, {}
        for e in codes:
            s = self.series[e]
            r = s["close"].astype(float).pct_change()
            if self.ended[e]:
                r.iloc[-1] = (1 + r.iloc[-1]) * (1 + self.delisted_return) - 1
                ended[e] = s["date"].iloc[-1]
            r.index = s["date"].to_numpy()
            cols[e] = r
        return pd.DataFrame(cols).sort_index(), pd.Series(ended, dtype="datetime64[ns]")


class Signals:
    """signals.csv as observations."""

    name = "signals"

    def __init__(self, market: CsvMarket, params: dict | None = None):
        self.market, self.params = market, params or {}

    def fetch(self, start, end) -> None:
        """Nothing to download: the file is the cache."""

    def observations(self, start, end) -> pd.DataFrame:
        """Every row of signals.csv, in file order (the last of a tie wins)."""
        path = self.market.files.get("signals")
        if not path:
            return data.empty_observations()
        s = _read(path)
        return pd.DataFrame(
            {
                "entity_id": s["entity"].astype(str),
                "available_at": parse_available_at(s["available_at"], self.market.calendar),
                "source": self.name,
                "feature": s["feature"].astype(str),
                "value": pd.to_numeric(s["value"], errors="coerce"),
            }
        )


def _documents(market: CsvMarket, params: dict | None = None) -> data.FrameDocuments:
    d = _read(market.files["documents"])
    frame = pd.DataFrame(
        {
            "entity_id": d["entity"].astype(str),
            "available_at": parse_available_at(d["available_at"], market.calendar),
            "doc_type": d["doc_type"].astype(str),
            "doc_id": d["doc_id"].astype(str)
            if "doc_id" in d
            else [f"{e}|{i}" for i, e in enumerate(d["entity"])],
            "text": d["text"].fillna("").astype(str),
        }
    )
    return data.FrameDocuments("csv", frame)


DOCUMENT_SOURCES = {"csv": _documents}


def build(cfg: dict):
    """The plug-in entry point."""
    return CsvMarket(cfg), {"signals": Signals}
