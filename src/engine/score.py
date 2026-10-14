"""The one scorer every test and report goes through, plus overlapping-cohort portfolios for
long holds.

Input: scored rows (entity_id, decision_time, group, fwd_return, score, and rt_cost = round-trip
cost as a fraction). One value per decision time, then a t-statistic over decision times:
  rank_ic             Spearman of score with the group-adjusted forward return (net of each name's
                      market beta when the rows carry fwd_excess)
  quantile_spreads    within each group, top q by score minus bottom q, equal weight, averaged over
                      groups with >= min_names names; net = minus the round trips of names entering
                      each leg (long-short, group-neutral)
  rank_weighted       a long-short book over every name, weight ~ centred score rank
  band_long_short     a dollar-neutral band book ranked across all names, with funding and forced
                      closes (perps); ols_alpha_hc3 judges it net of a benchmark's beta
  simulate_portfolio  a long-only rule (top-k, hysteresis band, periodic) against the equal-weight
                      universe; half a round trip per buy and per sell
  per_period_t        mean / (sd / sqrt(n)), deflated by sqrt(overlap) when label windows overlap;
                      with lags, never above the Newey-West t (the market's `t_stat_lags`)
  newey_west_t        mean / Newey-West (Bartlett) standard error, for autocorrelated series

Overlapping cohorts (Jegadeesh-Titman), plain frames in, so any market with daily bars can use them:
  returns   daily simple returns, local trading dates x entities; NaN = no bar that day
  ended     entity -> last date, for series that stopped for good (the market's delisting return
            is already in that day's return)
cohort_returns: one cohort bought at the close of `entry_day` (the first trading day after
formation), equal weight, then drifting; held entry_day < d <= exit_day. A member with no bar
earns 0 that day; a member whose series ended is gone after its last day and its value is spread
over the live members (all gone: cash). Costs: half a round trip per member at entry and at exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- statistics
def per_period_t(series: pd.Series, overlap: int = 1, lags: int | str = 0) -> float:
    """t of the mean over periods, deflated by sqrt(overlap). With lags (an int, or "auto" for
    floor(4 (n/100)^(2/9))), the smaller of that and the Newey-West t: streaky periods (crypto's
    bull and bear runs) can only lower it."""
    s = pd.Series(series).dropna()
    sd = s.std(ddof=1)
    if len(s) < 3 or not sd > 0:
        return float("nan")
    t = float(s.mean() / (sd / np.sqrt(len(s))) / np.sqrt(overlap))
    k = nw_lags(len(s), lags)
    return min(t, newey_west_t(s, k), key=abs) if k else t


def nw_lags(n: int, lags: int | str) -> int:
    """Lags for Newey-West: an int as given, "auto" = floor(4 (n/100)^(2/9)), at least 1."""
    if lags == "auto":
        return max(1, int(4 * (n / 100) ** (2 / 9)))
    return int(lags or 0)


def newey_west_t(series, lags: int) -> float:
    """t of the mean with a Newey-West (Bartlett kernel) long-run variance, `lags` lags."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    var = _nw_var(x, lags)
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def _nw_var(x: np.ndarray, lags: int) -> float:
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(var)


def long_run_sd(series, lags: int | str = 0) -> float:
    """Per-period noise for a power calculation: the sd, raised to the Newey-West long-run sd when
    the series is streaky (never lowered), so the MDE matches the t that will judge the test."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(x) < 3:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    k = nw_lags(len(x), lags)
    return max(sd, float(np.sqrt(max(_nw_var(x, k), 0.0)))) if k else sd


# The largest edge this research has measured is a rank IC near 0.035 and a net excess near 0.2% a
# month; a test that can't see effects well above these can only say "couldn't tell".
PLAUSIBLE = {"rank IC": 0.05, "excess return per month": 0.01, "net return per period": 0.01}


def mde(noise_sd: float, periods: int, bar: float, power: float, overlap: int = 1) -> float:
    """Minimum detectable effect: the smallest true mean a t-test over `periods` periods, with this
    per-period noise, clears `bar` with probability `power`: (bar + z_power) x sd x sqrt(overlap)
    / sqrt(periods). At 50% power that is bar x the standard error."""
    from scipy.stats import norm

    if periods < 3 or not noise_sd > 0:
        return float("nan")
    return float((bar + norm.ppf(power)) * noise_sd * np.sqrt(overlap) / np.sqrt(periods))


def overlap(horizon_bars: int, bars_between_decisions: int) -> int:
    """How many decisions' label windows overlap one another (1 = none)."""
    return max(1, int(np.ceil(horizon_bars / max(1, bars_between_decisions))))


def group_adjusted(scored: pd.DataFrame) -> pd.Series:
    """Forward return (beta-adjusted fwd_excess when the market has it) minus its (decision time,
    group) mean."""
    col = "fwd_excess" if "fwd_excess" in scored else "fwd_return"
    return scored[col] - scored.groupby(["decision_time", "group"])[col].transform("mean")


def rank_ic(scored: pd.DataFrame) -> pd.Series:
    """Per decision time: Spearman of score with the group-adjusted forward return."""
    s = scored.assign(adj=group_adjusted(scored))
    return s.groupby("decision_time").apply(
        lambda g: g["score"].corr(g["adj"], method="spearman"), include_groups=False
    )


def fill_costs(scored: pd.DataFrame, col: str = "rt_cost") -> pd.DataFrame:
    """Unknown costs -> the median known cost of the scored rows (stated, not hidden)."""
    if col not in scored:
        raise ValueError(f"no {col} column: attach the market's costs first")
    return scored.assign(**{col: scored[col].fillna(scored[col].median())})


# ---------------------------------------------------------------- long-short books
def quantile_spreads(
    scored: pd.DataFrame, q: float = 0.1, min_names: int = 10, cost_col: str = "rt_cost"
) -> dict:
    """Top minus bottom q within each group, gross and net of the round trips of names entering."""
    s = scored
    pct = s.groupby(["decision_time", "group"])["score"].rank(pct=True)
    n = s.groupby(["decision_time", "group"])["score"].transform("size")
    s = s.assign(top=(pct > 1 - q) & (n >= min_names), bot=(pct <= q) & (n >= min_names))
    gross, net, turn = {}, {}, {}
    prev = {"top": set(), "bot": set()}
    for t, g in s.groupby("decision_time"):
        legs = g[g["top"] | g["bot"]]
        if legs.empty:
            continue
        by_group = legs.groupby("group").apply(
            lambda x: x.loc[x["top"], "fwd_return"].mean() - x.loc[x["bot"], "fwd_return"].mean(),
            include_groups=False,
        )
        gross[t] = by_group.mean()
        cost, tv = 0.0, []
        for leg in ("top", "bot"):
            now = g[g[leg]]
            names = set(now["entity_id"])
            new = now[~now["entity_id"].isin(prev[leg])]
            share = len(new) / max(1, len(names))
            tv.append(share)
            cost += share * new[cost_col].mean() if len(new) else 0.0
            prev[leg] = names
        turn[t] = float(np.mean(tv))
        net[t] = gross[t] - cost
    return {"gross": pd.Series(gross), "net": pd.Series(net), "turnover": pd.Series(turn)}


def rank_weighted(
    scored: pd.DataFrame, cost_col: str = "rt_cost", within_group: bool = True
) -> dict:
    """A long-short book holding EVERY name, weight proportional to its centred score rank (within
    group), scaled to $1 long and $1 short. Uses the whole cross-section, so it is a far less noisy
    end-to-end statistic than a decile spread; net = minus half a round trip per unit traded."""
    keys = ["decision_time", "group"] if within_group else ["decision_time"]
    s = scored.assign(pct=scored.groupby(keys)["score"].rank(pct=True))
    s["w"] = s["pct"] - s.groupby(keys)["pct"].transform("mean")
    gross_w = s.groupby("decision_time")["w"].transform(lambda w: w.abs().sum() / 2)
    s["w"] = s["w"] / gross_w.replace(0, np.nan)
    gross, net, turn = {}, {}, {}
    prev = pd.Series(dtype=float)
    for t, g in s.groupby("decision_time"):
        w = g.set_index("entity_id")["w"].fillna(0.0)
        cost = g.set_index("entity_id")[cost_col]
        traded = w.sub(prev, fill_value=0.0).abs()
        c = float((traded * cost.reindex(traded.index).fillna(cost.median())).sum() / 2)
        gross[t] = float((w * g.set_index("entity_id")["fwd_return"]).sum())
        net[t] = gross[t] - c
        turn[t] = float(traded.sum() / 2)
        prev = w
    return {"gross": pd.Series(gross), "net": pd.Series(net), "turnover": pd.Series(turn)}


def band_long_short(
    rows: pd.DataFrame, enter: float = 0.8, exit: float = 0.4, min_names: int = 5
) -> pd.DataFrame:
    """A dollar-neutral band book ranked across all rows each period (no groups): long the names above
    `enter` (score percentile), held while above `exit`; short the mirror (below 1 - enter, held while
    below 1 - exit). Each leg equal weight, half the capital.

    rows: decision_time, entity_id, score, ret (the position's return over the period, forced closes
    included), funding (sum of funding rates over the hold), side_cost (one-way cost, a fraction).
    Funding: each leg's P&L includes -position x rate (long +1, short -1), sign as recorded.
    A period with fewer than `min_names` in either leg is dropped (flagged, no return) and the book
    restarts empty. Costs: half capital / leg size x side_cost for every name entering or leaving."""
    d = rows.assign(
        pl=rows.groupby("decision_time")["score"].rank(pct=True),
        ps=(-rows["score"]).groupby(rows["decision_time"]).rank(pct=True),
    )
    L, S, nL0, nS0 = set(), set(), 1, 1
    out = []
    for t, g in d.groupby("decision_time"):
        g = g.set_index("entity_id")
        newL = {e for e in L if e in g.index and g.at[e, "pl"] > exit} | set(
            g.index[g["pl"] > enter]
        )
        newS = {e for e in S if e in g.index and g.at[e, "ps"] > exit} | set(
            g.index[g["ps"] > enter]
        )
        newS -= newL
        row = {"decision_time": t, "n_long": len(newL), "n_short": len(newS)}
        if min(len(newL), len(newS)) < min_names:
            out.append(row | {"dropped": True})
            L, S = set(), set()
            continue
        nL, nS = len(newL), len(newS)
        med = g["side_cost"].median()
        cost = sum(0.5 / nL * g["side_cost"].get(e, med) for e in newL - L)
        cost += sum(0.5 / nL0 * g["side_cost"].get(e, med) for e in L - newL)
        cost += sum(0.5 / nS * g["side_cost"].get(e, med) for e in newS - S)
        cost += sum(0.5 / nS0 * g["side_cost"].get(e, med) for e in S - newS)
        lr, sr = g.loc[list(newL), "ret"].mean(), g.loc[list(newS), "ret"].mean()
        fund = 0.5 * -g.loc[list(newL), "funding"].fillna(0).mean()
        fund += 0.5 * g.loc[list(newS), "funding"].fillna(0).mean()
        gross = 0.5 * (lr - sr)
        turn_l = 0.5 * (len(newL - L) / nL + len(L - newL) / nL0)  # one-way, share of the leg
        turn_s = 0.5 * (len(newS - S) / nS + len(S - newS) / nS0)
        out.append(
            row
            | {
                "dropped": False,
                "long_ret": lr,
                "short_ret": sr,
                "gross": gross,
                "funding": fund,
                "cost": cost,
                "turn_long": turn_l,
                "turn_short": turn_s,
                "net": gross + fund - cost,
            }
        )
        L, S, nL0, nS0 = newL, newS, nL, nS
    return pd.DataFrame(out).set_index("decision_time")


def ols_alpha_hc3(y, x) -> dict:
    """y = alpha + beta x + e by OLS; alpha's t from HC3 (heteroskedasticity-robust) errors."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    cov = XtXi @ (X.T * (e / (1 - h)) ** 2) @ X @ XtXi
    return {
        "alpha": float(b[0]),
        "beta": float(b[1]),
        "t_alpha": float(b[0] / np.sqrt(cov[0, 0])),
        "resid_sd": float(np.std(e, ddof=2)),
        "n": len(y),
    }


def ols_alpha_robust(y, x, lags: int | str = "auto") -> dict:
    """ols_alpha_hc3 plus a Newey-West (Bartlett) t of alpha; t_alpha is the smaller of the two in
    size, so neither heteroskedastic nor streaky weeks can inflate it."""
    out = ols_alpha_hc3(y, x)
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    e = y - X @ (XtXi @ X.T @ y)
    u = X * e[:, None]
    k = nw_lags(len(y), lags)
    S = u.T @ u
    for lag in range(1, min(k, len(y) - 1) + 1):
        g = u[lag:].T @ u[:-lag]
        S += (1 - lag / (k + 1)) * (g + g.T)
    cov = XtXi @ S @ XtXi
    t_nw = float(out["alpha"] / np.sqrt(cov[0, 0]))
    return out | {
        "t_hc3": out["t_alpha"],
        "t_nw": t_nw,
        "nw_lags": k,
        "t_alpha": min(out["t_alpha"], t_nw, key=abs),
    }


# ---------------------------------------------------------------- long-only portfolio rules
# A rule: choose(frame of one decision time indexed by entity, with `pct`, held set, t) -> set


@dataclass
class Band:
    """Buy names above `enter` percentile (within group); hold until they fall to `exit` or below."""

    enter: float = 0.9
    exit: float = 0.7

    def __call__(self, g: pd.DataFrame, held: set, t) -> set:
        keep = {e for e in held if e in g.index and g.loc[e, "pct"] > self.exit}
        return keep | set(g.index[g["pct"] > self.enter])


@dataclass
class TopK:
    """Hold the names above `enter` percentile (within group), replaced every period (no band)."""

    enter: float = 0.9

    def __call__(self, g: pd.DataFrame, held: set, t) -> set:
        return set(g.index[g["pct"] > self.enter])


class Periodic:
    """Replace the book with the top names only in the given months; hold between."""

    def __init__(self, enter: float = 0.9, months: tuple[int, ...] = (3, 6, 9, 12)):
        self.enter, self.months, self.book = enter, months, set()

    def __call__(self, g: pd.DataFrame, held: set, t) -> set:
        if pd.Timestamp(t).month in self.months or not self.book:
            self.book = set(g.index[g["pct"] > self.enter])
        return set(self.book)


def simulate_portfolio(
    scored: pd.DataFrame, rule, cost_col: str = "rt_cost", within_group: bool = True
) -> dict:
    """Long-only, equal weight, vs the equal-weight universe of scored names each period.
    A held name with no row this period (left the universe) earns the universe mean: neutral,
    stated. Exit cost uses the name's last known cost."""
    keys = ["decision_time", "group"] if within_group else ["decision_time"]
    s = scored.assign(pct=scored.groupby(keys)["score"].rank(pct=True))
    held: set = set()
    start: dict = {}
    spells = []
    gross, net, turn = {}, {}, {}
    prev_cost: dict = {}
    times = sorted(s["decision_time"].unique())
    for t in times:
        g = s[s["decision_time"] == t].set_index("entity_id")
        new = rule(g, held, t)
        adds, drops = new - held, held - new
        n = max(1, len(new))
        exit_cost = sum(0.5 * prev_cost.get(e, g[cost_col].median()) for e in drops)
        entry_cost = sum(0.5 * g.loc[e, cost_col] for e in adds)
        for e in drops:
            spells.append(len([x for x in times if start[e] <= x < t]))
        for e in adds:
            start[e] = t
        held = new
        if not held:
            continue
        ret = g["fwd_return"].reindex(list(held)).fillna(g["fwd_return"].mean())
        gross[t] = ret.mean() - g["fwd_return"].mean()
        net[t] = gross[t] - (entry_cost + exit_cost) / n
        turn[t] = 0.5 * (len(adds) + len(drops)) / n
        prev_cost = g[cost_col].to_dict() | {e: prev_cost[e] for e in prev_cost if e not in g.index}
    return {
        "gross": pd.Series(gross),
        "net": pd.Series(net),
        "turnover": pd.Series(turn),
        "avg_holding_periods": float(np.mean(spells)) if spells else float("nan"),
    }


def summarize(
    series: dict, periods_per_year: int = 12, overlap_n: int = 1, lags: int | str = 0
) -> dict:
    """Means, t-statistics, turnover (and holding / cost share for portfolios) of a book's series."""
    g, n = series["gross"], series["net"]
    out = {
        "periods": len(g),
        "gross": float(g.mean()),
        "gross_t": per_period_t(g, overlap_n, lags),
        "net": float(n.mean()),
        "net_t": per_period_t(n, overlap_n, lags),
        "annual_turnover": float(periods_per_year * series["turnover"].mean()),
    }
    if "avg_holding_periods" in series:
        out["avg_holding_periods"] = series["avg_holding_periods"]
        out["share_of_gross_lost_to_costs"] = (
            float((g.mean() - n.mean()) / g.mean()) if g.mean() else float("nan")
        )
    return out


def evaluate(
    scored: pd.DataFrame,
    rules: dict | None = None,
    q: float = 0.1,
    min_names: int = 10,
    periods_per_year: int = 12,
    overlap_n: int = 1,
    lags: int | str = 0,
) -> dict:
    """Everything the scorer reports for one model's scores: rank IC, spreads, each portfolio rule."""
    ic = rank_ic(scored)
    out = {
        "periods": int(ic.notna().sum()),
        "ic": float(ic.mean()),
        "ic_t": per_period_t(ic, overlap_n, lags),
        "spread": summarize(
            quantile_spreads(scored, q, min_names), periods_per_year, overlap_n, lags
        ),
    }
    for name, rule in (rules or {}).items():
        out[name] = summarize(simulate_portfolio(scored, rule), periods_per_year, overlap_n, lags)
    return out


# ---------------------------------------------------------------- overlapping cohorts
def _cohort_path(
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
            "gross": _cohort_path(r, last, zero, zero, sells),
            "net": _cohort_path(r, last, c_in, c_out, sells),
        },
        index=days,
    )


def overlapping_cohorts(paths: list[pd.DataFrame]) -> pd.DataFrame:
    """Daily mean over the cohorts held each day, plus `cohorts` = how many were held."""
    paths = [p for p in paths if len(p)]
    stacked = pd.concat(paths, keys=range(len(paths)))
    daily = stacked.groupby(level=1).mean()
    daily["cohorts"] = stacked.groupby(level=1).size()
    return daily.sort_index()


def monthly_returns(daily: pd.DataFrame, cols=("gross", "net")) -> pd.DataFrame:
    """Compounded calendar-month returns; `cohorts` = the fewest cohorts held on any day."""
    month = pd.DatetimeIndex(daily.index).to_period("M")
    out = (1 + daily[list(cols)]).groupby(month).prod() - 1
    if "cohorts" in daily:
        out["cohorts"] = daily["cohorts"].groupby(month).min()
    return out
