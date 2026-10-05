"""Calendars: when decisions happen, and how local dates map to UTC moments.

A decision time is a UTC instant. Everything with available_at strictly before it may be used;
the position is entered at the first bar that CLOSES after it.

  TradingCalendar   stock-style: business days, a session close (e.g. 16:00 New York), decisions a
                    little after the close so that day's bar is known but the next one isn't
  ContinuousCalendar  24/7 markets (crypto): decisions at a fixed UTC time of day
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FREQS = {
    "trading": {"monthly": "BME", "weekly": "W-FRI", "daily": "B"},
    "continuous": {"monthly": "ME", "weekly": "W-SUN", "daily": "D"},
}


@dataclass(frozen=True)
class TradingCalendar:
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
        return self.at(dates, self.close)

    def decision_times(self, start, end, freq: str) -> pd.DatetimeIndex:
        dates = pd.date_range(
            pd.Timestamp(start).tz_localize(None).normalize()
            if pd.Timestamp(start).tzinfo
            else pd.Timestamp(start),
            pd.Timestamp(end).tz_localize(None)
            if pd.Timestamp(end).tzinfo
            else pd.Timestamp(end),
            freq=FREQS[self.kind].get(freq, freq),
        )
        return self.bar_close(dates) + pd.Timedelta(self.decide_after_close)

    def local_date(self, ts) -> pd.Series:
        """UTC instants -> local calendar dates (midnight, naive)."""
        s = pd.Series(ts)
        return s.dt.tz_convert(self.tz).dt.tz_localize(None).dt.normalize()

    def next_midnight(self, ts) -> pd.Series:
        """The first local midnight strictly after each instant, in UTC.

        The conservative stamp for anything dated by day only, or published during a day the
        decision rules treat as a unit: usable from the start of the following local day."""
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
    tz: str = "UTC"
    close: str = "00:00"
    decide_after_close: str = "0min"
    kind: str = "continuous"


def make(cfg: dict) -> TradingCalendar:
    kind = cfg.get("kind", "trading")
    cls = ContinuousCalendar if kind == "continuous" else TradingCalendar
    return cls(**{k: v for k, v in cfg.items() if k != "kind"}, kind=kind)
