"""Overlapping-cohort portfolios (Jegadeesh-Titman): buy a list at each formation date, hold it for
a fixed number of formations, and hold the equal-weight average of every cohort still held.

Plain frames in, so any market with daily bars can use it:
  returns   daily simple returns, local trading dates x entities; NaN = no bar that day
  ended     entity -> last date, for series that stopped for good (delisting); the market's
            delisting return is already in that day's return

cohort_returns  one cohort, buy and hold. Bought at the close of `entry_day` (the first trading day
                after formation), equal weight; weights then drift. Held days: entry_day < d <=
                exit_day, sold at exit_day's close. A member with no bar on a day earns 0 that day;
                a member whose series ended is gone after its last day and its value is spread over
                the cohort's live members pro rata (all gone: cash, 0). Costs: half a round trip
                per member at entry (on the first day held) and half at exit (on exit_day, on the
                value then held; none for members that already ended: a delisting has its own rule)
overlapping     each day, the mean return of the cohorts held that day (daily equal weight across
                cohorts); the count of cohorts held
monthly         compounded calendar-month returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _path(
    r: np.ndarray, last: np.ndarray, c_in: np.ndarray, c_out: np.ndarray, sells: bool
) -> np.ndarray:
    """Daily returns of one buy-and-hold cohort. r: days x members (0 where no bar); last[j]: index
    of member j's final day (len(r) if it doesn't end inside)."""
    n_days, n = r.shape
    v = (1 - c_in) / n  # value per member after paying entry costs out of $1
    alive = np.ones(n, dtype=bool)
    cash, prev = 0.0, 1.0
    out = np.empty(n_days)
    for i in range(n_days):
        v = v * (1 + r[i])
        total = v.sum() + cash
        if sells and i == n_days - 1:
            total -= (v * c_out).sum()  # dead members hold 0 value
        out[i] = total / prev - 1
        prev = total
        dying = alive & (last == i)
        if dying.any():
            freed = v[dying].sum()
            alive &= ~dying
            v = np.where(alive, v, 0.0)
            live = v.sum()
            if live > 0:
                v = v * (live + freed) / live
            else:
                cash += freed
    return out


def cohort_returns(
    returns: pd.DataFrame,
    members,
    entry_day,
    exit_day=None,
    cost_in: pd.Series | None = None,
    cost_out: pd.Series | None = None,
    ended: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily gross and net returns of one cohort (index = held days). cost_in / cost_out: half a
    round trip per member as a fraction (missing -> 0). exit_day None: held to the data's end."""
    ended = ended if ended is not None else pd.Series(dtype="datetime64[ns]")
    entry_day = pd.Timestamp(entry_day)
    days = returns.index[returns.index > entry_day]
    sells = exit_day is not None and pd.Timestamp(exit_day) in set(days)
    if exit_day is not None:
        days = days[days <= pd.Timestamp(exit_day)]
    names = [
        m
        for m in dict.fromkeys(members)
        if m in returns.columns and not (m in ended.index and ended[m] <= entry_day)
    ]
    if not names or not len(days):
        return pd.DataFrame(columns=["gross", "net"], dtype=float)
    r = returns.loc[days, names].fillna(0.0).to_numpy(dtype=float)
    pos = {d: i for i, d in enumerate(days)}
    last = np.array(
        [pos.get(ended[m], len(days)) if m in ended.index else len(days) for m in names]
    )
    zero = np.zeros(len(names))

    def costs(c):
        return zero if c is None else c.reindex(names).fillna(0.0).to_numpy(dtype=float)

    c_in, c_out = costs(cost_in), costs(cost_out)
    return pd.DataFrame(
        {
            "gross": _path(r, last, zero, zero, sells),
            "net": _path(r, last, c_in, c_out, sells),
        },
        index=days,
    )


def overlapping(paths: list[pd.DataFrame]) -> pd.DataFrame:
    """Daily mean over the cohorts held each day, plus `cohorts` = how many were held."""
    paths = [p for p in paths if len(p)]
    stacked = pd.concat(paths, keys=range(len(paths)))
    daily = stacked.groupby(level=1).mean()
    daily["cohorts"] = stacked.groupby(level=1).size()
    return daily.sort_index()


def monthly(daily: pd.DataFrame, cols=("gross", "net")) -> pd.DataFrame:
    """Compounded calendar-month returns; `cohorts` = the fewest cohorts held on any day."""
    month = pd.DatetimeIndex(daily.index).to_period("M")
    out = (1 + daily[list(cols)]).groupby(month).prod() - 1
    if "cohorts" in daily:
        out["cohorts"] = daily["cohorts"].groupby(month).min()
    return out
