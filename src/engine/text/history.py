"""History priors: how did EARLIER events like this one turn out? Strictly earlier, shrunk by backoff.

Pricing an event from its own text is a guess; the record of similar past events isn't. Two uses:
  base_rates()     outcome priors per event (mean, median, hit rate of e.g. the 3-day reaction),
                   the prior a text card must beat in the eval (ported from the pilot's history)
  running_means()  the base rate of each ANSWER among earlier similar documents, so a document's
                   answer can be read as a surprise: answer - base rate (engine.text.features)

Keys go coarse to fine (e.g. [], [doc_type], [doc_type, size], [doc_type, detail, size]); a level
is skipped for a row whose key holds a missing value ("-" or None). Each level's mean is shrunk
toward its parent: mean_L = (n * mean_raw + K * mean_parent) / (n + K), so three examples barely
move the prior and three hundred dominate it. Medians come from the finest level with >= MIN_N.

Point in time: an outcome observed `lag` after its event is used only by targets whose time is
strictly after event time + lag. leak_check() proves that on the output.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

K, MIN_MEDIAN_N = 20, 20
MISSING = ("-", None, "", "nan")


class _Median:
    """Running median over a growing set (two heaps)."""

    def __init__(self):
        self.lo, self.hi = [], []

    def add(self, x: float) -> None:
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def get(self) -> float:
        if not self.lo:
            return np.nan
        return (
            -self.lo[0]
            if len(self.lo) > len(self.hi)
            else (-self.lo[0] + self.hi[0]) / 2
        )


def level_keys(row: dict, levels: list[list[str]]) -> list[str | None]:
    out = []
    for cols in levels:
        vals = [row[c] for c in cols]
        if any((v in MISSING) or (isinstance(v, float) and np.isnan(v)) for v in vals):
            out.append(None)
        else:
            out.append("|".join(["L"] + [f"{c}={v}" for c, v in zip(cols, vals)]))
    return out


def base_rates(
    targets: pd.DataFrame,
    pool: pd.DataFrame,
    levels: list[list[str]],
    outcomes: dict[str, pd.Timedelta],
    at: str = "event_day",
    k: float = K,
    min_median_n: int = MIN_MEDIAN_N,
    medians: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per target row: hist_n (finest level's n for the first outcome) and, per outcome o,
    hist_<o>_mean (shrunk), hist_<o>_hit (P(o > 0), shrunk toward 0.5 at the root),
    hist_<o>_med (medians for `medians`, default the first outcome), plus hist_<o>_last (the latest
    event time used, for leak_check)."""
    medians = medians if medians is not None else tuple(outcomes)[:1]
    res = pd.DataFrame(index=range(len(targets)))
    first = True
    for o, lag in outcomes.items():
        p = pool.dropna(subset=[o]).copy()
        p["_known"] = p[at] + lag
        p = p.sort_values("_known", kind="stable")
        rows = p.to_dict("records")
        stats: dict = {}  # key -> [n, sum, hits, median, last]
        order = targets.assign(_pos=range(len(targets))).sort_values(at, kind="stable")
        out = {}
        i = 0
        for t in order.to_dict("records"):
            when = t[at]
            while i < len(rows) and rows[i]["_known"] < when:
                r = rows[i]
                for key in level_keys(r, levels):
                    if key is None:
                        continue
                    s = stats.setdefault(key, [0, 0.0, 0, _Median(), pd.NaT])
                    s[0] += 1
                    s[1] += r[o]
                    s[2] += r[o] > 0
                    if o in medians:
                        s[3].add(r[o])
                    s[4] = r[at] if pd.isna(s[4]) else max(s[4], r[at])
                i += 1
            mean = hit = 0.0
            med, n_fine, last = np.nan, 0, pd.NaT
            for li, key in enumerate(level_keys(t, levels)):
                s = stats.get(key) if key else None
                if s is None or not s[0]:
                    continue
                n = s[0]
                mean = (s[1] + k * mean) / (n + k)
                hit = (s[2] + k * (hit if li else 0.5)) / (n + k)
                if o in medians and n >= min_median_n:
                    med = s[3].get()
                n_fine = n
                last = s[4] if pd.isna(last) else max(last, s[4])
            out[t["_pos"]] = (n_fine, mean, med, hit, last)
        cols = pd.DataFrame.from_dict(
            out,
            orient="index",
            columns=["n", "mean", "med", "hit", "last"],
        ).sort_index()
        if first:
            res["hist_n"] = cols["n"].to_numpy()
            first = False
        res[f"hist_{o}_mean"] = cols["mean"].to_numpy()
        if o in medians:
            res[f"hist_{o}_med"] = cols["med"].to_numpy()
        res[f"hist_{o}_hit"] = cols["hit"].to_numpy()
        res[f"hist_{o}_last"] = cols["last"].to_numpy()
    return pd.concat([targets.reset_index(drop=True), res], axis=1)


def leak_check(
    rates: pd.DataFrame, outcomes: dict[str, pd.Timedelta], at: str = "event_day"
) -> dict:
    """No base rate may use an outcome that wasn't fully observed before the target's time."""
    out, ok = {"events": len(rates)}, True
    for o, lag in outcomes.items():
        last = rates[f"hist_{o}_last"]
        bad = int((last.notna() & (last + lag >= rates[at])).sum())
        out[f"{o}_stats_using_unfinished_events"] = bad
        out[f"{o}_min_gap"] = str((rates[at] - last).min())
        ok &= bad == 0
    out["pass"] = bool(ok)
    return out


def running_means(
    frame: pd.DataFrame,
    cols: list[str],
    levels: list[list[str]],
    at: str = "available_at",
    k: float = K,
) -> pd.DataFrame:
    """For each row, the backoff-shrunk mean of each column over STRICTLY earlier rows (same key
    hierarchy as base_rates). NaN values don't count. Returns a frame aligned to `frame`."""
    f = frame.reset_index(drop=True)
    vals = f[cols].to_numpy(dtype=float)
    keys = [level_keys(r, levels) for r in f.to_dict("records")]
    times = f[at].to_numpy()
    order = np.argsort(times, kind="stable")
    stats: dict = {}  # key -> [n vector, sum vector]
    out = np.full_like(vals, np.nan)
    j = 0
    m = len(cols)
    while j < len(order):
        jj = j
        while jj < len(order) and times[order[jj]] == times[order[j]]:
            jj += 1
        block = order[j:jj]
        for r in block:  # read before adding: strictly earlier only
            mean = np.zeros(m)
            seen = np.zeros(m, dtype=bool)
            for key in keys[r]:
                s = stats.get(key) if key else None
                if s is None:
                    continue
                n, tot = s
                upd = n > 0
                mean = np.where(upd, (tot + k * mean) / np.maximum(n + k, 1e-12), mean)
                seen |= upd
            out[r] = np.where(seen, mean, np.nan)
        for r in block:
            v = vals[r]
            ok = ~np.isnan(v)
            for key in keys[r]:
                if key is None:
                    continue
                s = stats.setdefault(key, [np.zeros(m), np.zeros(m)])
                s[0] += ok
                s[1] += np.where(ok, v, 0.0)
        j = jj
    return pd.DataFrame(out, columns=cols, index=frame.index)
