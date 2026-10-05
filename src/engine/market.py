"""Market plug-in: what an asset class must provide to be researched.

A Market answers five questions, all point-in-time:

  universe(as_of)       who is tradable at this decision time: entity_id, group (sector, chain...),
                        plus any attributes (name, market value). Survivors-only universes lie.
  labels(rows, horizon) the forward return of each (entity_id, decision_time) row. Convention: enter
                        at the close of the FIRST bar closing after the decision time, exit `horizon`
                        bars later. If the bars stop early because the asset delisted, the return
                        runs to the last bar plus the market's delisting adjustment; if they stop
                        because the window simply hasn't finished, the label is NaN. Returns
                        entry_time and label_end too (training rows are purged on label_end).
  cost_bps(rows)        round-trip trading cost per row in basis points (NaN = unknown)
  calendar              decision times and local-date handling (engine.calendar)
  group                 the universe's `group` column; optional (all one group if absent)

forward_returns() below implements the label convention once, from a bar series, so markets
only have to supply bars and their delisting rule.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "entity_id",
    "decision_time",
    "entry_time",
    "label_end",
    "fwd_return",
    "delisted",
]


@runtime_checkable
class Market(Protocol):
    name: str
    calendar: object

    def universe(self, as_of: pd.Timestamp) -> pd.DataFrame: ...

    def labels(self, rows: pd.DataFrame, horizon: int) -> pd.DataFrame: ...

    def cost_bps(self, rows: pd.DataFrame) -> pd.Series: ...


def forward_returns(
    close_times: pd.DatetimeIndex,
    prices: np.ndarray,
    decision_times: pd.DatetimeIndex,
    horizon: int,
    ended: bool,
    delist_adjustment: float = 0.0,
    min_history: int = 0,
    drop_flat: bool = False,
) -> pd.DataFrame:
    """Labels for one entity at many decision times.

    close_times   UTC close of each bar (sorted), prices = total-return price (e.g. adjusted close)
    ended         True if the series stopped for good (delisted); then a short window is a real
                  outcome, otherwise it's an unfinished window and the label is NaN
    delist_adjustment  added (compounded) to a window cut short by delisting, e.g. -0.30
    min_history   bars required at or before the decision (0 = none)
    drop_flat     NaN when the window has < 3 bars or zero variance (stale quotes, not a price)
    """
    n = len(prices)
    pos = close_times.searchsorted(
        decision_times, side="right"
    )  # first bar closing after T
    out = []
    for t, i in zip(decision_times, pos):
        row = {
            "decision_time": t,
            "entry_time": pd.NaT,
            "label_end": pd.NaT,
            "fwd_return": np.nan,
            "delisted": False,
        }
        if i >= n or i < min_history:
            out.append(row)
            continue
        exit_i = min(i + horizon, n - 1)
        complete = exit_i - i == horizon
        if not (complete or ended):
            out.append(row)
            continue
        window = prices[i : exit_i + 1]
        fwd = window[-1] / window[0] - 1
        delisted = not complete
        if delisted:
            fwd = (1 + fwd) * (1 + delist_adjustment) - 1
        if drop_flat:
            r = np.diff(window) / window[:-1]
            if len(window) <= 2 or not np.nanstd(r, ddof=1) > 0:
                fwd = np.nan
        row |= {
            "entry_time": close_times[i],
            "label_end": close_times[exit_i],
            "fwd_return": fwd,
            "delisted": delisted,
        }
        out.append(row)
    return pd.DataFrame(out)
