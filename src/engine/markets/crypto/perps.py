"""Perp legs for long-short tests: each row's USD-margined perp return, funding and one-way cost.

A row is tradable on perps when the token's perp was listed at the decision (first funding record
<= decision time <= last) and has a bar on the entry date. Timing matches the spot labels: close on
the entry date to close on the label_end date. A perp that stops trading inside the window closes at
its last close (never dropped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.markets.crypto import binance

PERP_FEE = 0.0005  # Binance USD-M taker, one way
SPOT_FEE = 0.0010  # what the market's rt_cost assumes per side


def perp_symbol(token: str, listed: set[str]) -> str | None:
    """The token's USDT perp (some trade in thousands: 1000PEPEUSDT)."""
    return next((s for s in (f"{token}USDT", f"1000{token}USDT") if s in listed), None)


def _day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(None).dt.normalize()


def attach(
    frame: pd.DataFrame, bars=None, funding=None, listed: set[str] | None = None
) -> pd.DataFrame:
    """Rows -> rows with listed, ret, funding, side_cost. bars/funding: symbol -> frame (default: the
    Binance cache). side_cost = perp fee + the market's square-root impact (rt_cost / 2 - spot fee)."""
    listed = binance.perp_symbols(max_age_s=float("inf")) if listed is None else listed
    bars = bars or (lambda s: binance.load("perp", s))
    funding = funding or (lambda s: binance.load("funding", s))
    out = frame.assign(listed=False, ret=np.nan, funding=np.nan)
    for e, g in frame.groupby("entity_id", sort=False):
        sym = perp_symbol(e.split("~")[0], listed)
        if sym is None:
            continue
        b, f = bars(sym), funding(sym)
        if b.empty or f.empty:
            continue
        ft = pd.DatetimeIndex(pd.to_datetime(f["time"], utc=True))
        on = (g["decision_time"] >= ft.min()) & (g["decision_time"] <= ft.max())
        bt = pd.DatetimeIndex(pd.to_datetime(b["time"]))
        close = b["close"].astype(float).to_numpy()
        entry, end = _day(g["entry_time"]), _day(g["label_end"])
        i = bt.searchsorted(entry)  # bar on the entry date
        has = (i < len(bt)) & (bt[np.minimum(i, len(bt) - 1)] == entry.to_numpy())
        j = bt.searchsorted(end, side="right") - 1  # last bar on or before label_end: forced close
        ret = np.where(
            has & (j >= i), close[np.maximum(j, 0)] / close[np.minimum(i, len(bt) - 1)] - 1, np.nan
        )
        rate = f["rate"].astype(float).to_numpy()
        a = ft.searchsorted(pd.to_datetime(g["entry_time"], utc=True), side="right")
        z = ft.searchsorted(pd.to_datetime(g["label_end"], utc=True), side="right")
        fund = np.array([np.nansum(rate[x:y]) for x, y in zip(a, z)])
        out.loc[g.index, "listed"] = (on & has).to_numpy()
        out.loc[g.index, "ret"] = ret
        out.loc[g.index, "funding"] = fund
    impact = (out["rt_cost"] / 2 - SPOT_FEE).clip(lower=0)
    return out.assign(side_cost=PERP_FEE + impact)


def benchmark(rows: pd.DataFrame, token: str = "BTC") -> pd.Series:
    """The token's own perp return each decision time (from its attached rows)."""
    r = rows[rows["entity_id"].str.split("~").str[0] == token]
    return r.set_index("decision_time")["ret"]
