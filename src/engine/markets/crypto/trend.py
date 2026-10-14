"""Time-series trend book (pre-registration 7): hold each liquid coin in proportion to how many of
its lookbacks rose, sized to a target volatility, the rest in cash; judged against holding BTC.

Weekly, decisions Monday 00:00 UTC; prices are Sunday's close (the bar that closes at the decision),
so entry has no dead day. Everything a week's weights use comes from bars dated on or before that
Sunday; the week's return is Sunday close to the next Sunday close. A coin that stops trading exits
at its last close (no penalty: the crash is in its bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (28, 56, 112, 224)  # days
VOL_DAYS = 60
TARGET_VOL = 0.50  # per coin, annualized
TOP = 20
MIN_BARS = 225


def closes(market) -> pd.DataFrame:
    """Daily closes, date x entity (NaN = no bar that day)."""
    c = pd.DataFrame(
        {e: s.set_index("time")["close"].astype(float) for e, s in market.series.items()}
    ).sort_index()
    return c.asfreq("D")


def weekly_rows(
    market,
    start,
    end,
    top: int = TOP,
    min_bars: int = MIN_BARS,
    only: tuple[str, ...] | None = None,
    c: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (decision, coin held in the set): exposure, vol, scale, ret (the week ahead), n
    (coins in the set that week). The set: the top `top` coins by 30-day mean USDT volume among
    those with >= min_bars bars and a bar on the Sunday (`only`: just these base tokens)."""
    c = closes(market) if c is None else c
    vol, hist = market.vol.reindex(c.index), market.hist.reindex(c.index)
    lr = np.log(c).diff()
    times = market.calendar.decision_times(pd.Timestamp(start), pd.Timestamp(end), "weekly")
    out = []
    for t in times:
        d = (pd.Timestamp(t).tz_convert(None) - pd.Timedelta(days=1)).normalize()  # the Sunday
        if d not in c.index or d + pd.Timedelta(days=7) > c.index[-1]:
            continue
        i = c.index.get_loc(d)
        ok = (hist.iloc[i] >= min_bars) & c.iloc[i].notna() & vol.iloc[i].notna()
        if only is not None:
            ok &= pd.Index(c.columns).str.split("~").str[0].isin(only)
        pick = vol.iloc[i][ok].sort_values(ascending=False).head(top).index
        if not len(pick):
            continue
        hist_c = c.iloc[: i + 1][pick].ffill(limit=3)
        now = hist_c.iloc[-1]
        ups = pd.DataFrame({L: now > hist_c.iloc[-1 - L] for L in LOOKBACKS if i - L >= 0})
        valid = pd.DataFrame({L: hist_c.iloc[-1 - L].notna() for L in LOOKBACKS if i - L >= 0})
        exposure = (ups & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
        r = lr.iloc[i - VOL_DAYS + 1 : i + 1][pick]
        sd = r.std(ddof=1).where(r.notna().sum() >= VOL_DAYS // 2) * np.sqrt(365)
        week = c.iloc[i : i + 8][pick]
        last = week.ffill().iloc[-1]  # a coin with no later bar exits at its last close
        ret = last / now - 1
        out.append(
            pd.DataFrame(
                {
                    "decision_time": t,
                    "entity_id": pick.astype(str),
                    "exposure": exposure.fillna(0.0).to_numpy(),
                    "vol": sd.to_numpy(),
                    "ret": ret.to_numpy(),
                    "n": len(pick),
                }
            )
        )
    f = pd.concat(out, ignore_index=True)
    f["scale"] = np.minimum(1.0, TARGET_VOL / f["vol"]).fillna(0.0)
    return f


def book(rows: pd.DataFrame, side_cost: pd.Series, divisor: str = "eligible", scaled=True):
    """Weekly net return of the book. weight = exposure x scale / D, D = min(TOP, coins in the set)
    ('eligible') or TOP ('fixed'); turnover is against last week's weights after they drifted;
    each unit traded pays that coin's side cost. Returns a frame per decision_time: gross, cost,
    net, gross_weight (the share invested), turnover."""
    f = rows.assign(side_cost=side_cost.reindex(rows.index).to_numpy())
    D = f["n"].clip(upper=TOP) if divisor == "eligible" else TOP
    f["w"] = f["exposure"] * (f["scale"] if scaled else 1.0) / D
    prev, prev_cost, out = pd.Series(dtype=float), pd.Series(dtype=float), {}
    for t, g in f.groupby("decision_time", sort=True):
        g = g.set_index("entity_id")
        w, r = g["w"], g["ret"].fillna(0.0)
        cost_now = g["side_cost"].fillna(g["side_cost"].median())
        names = w.index.union(prev.index)
        dw = (w.reindex(names, fill_value=0.0) - prev.reindex(names, fill_value=0.0)).abs()
        sc = cost_now.reindex(names).fillna(prev_cost.reindex(names)).fillna(cost_now.median())
        cost = float((dw * sc).sum())
        gross = float((w * r).sum())
        out[t] = {
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
            "gross_weight": float(w.sum()),
            "turnover": float(dw.sum()),
        }
        grown = w * (1 + r)
        prev = grown / (1 + gross)  # cash earns 0: the book's value grew by 1 + gross
        prev_cost = cost_now
    return pd.DataFrame.from_dict(out, orient="index").sort_index()


def btc_weekly(rows_or_times, c: pd.DataFrame, token: str = "BTC") -> pd.Series:
    """The token's Sunday-close to Sunday-close return for each decision time."""
    times = pd.DatetimeIndex(sorted(pd.Series(rows_or_times).unique()))
    s = c[token]
    out = {}
    for t in times:
        d = (t.tz_convert(None) - pd.Timedelta(days=1)).normalize()
        out[t] = s.get(d + pd.Timedelta(days=7), np.nan) / s.get(d, np.nan) - 1
    return pd.Series(out)


def equal_weight_hold(rows: pd.DataFrame) -> pd.Series:
    """Equal weight of the week's set, fully invested, gross (a comparison line, not a strategy)."""
    return rows.groupby("decision_time")["ret"].mean()


def stats(weekly: pd.Series) -> dict:
    """CAGR, annual vol, Sharpe (cash = 0) and max drawdown of a weekly return series."""
    w = weekly.dropna()
    if len(w) < 2:
        return {}
    wealth = (1 + w).cumprod()
    years = len(w) / 52.0
    return {
        "weeks": len(w),
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
        "vol": float(w.std(ddof=1) * np.sqrt(52)),
        "sharpe": float(w.mean() / w.std(ddof=1) * np.sqrt(52)) if w.std() > 0 else np.nan,
        "max_dd": float((wealth / wealth.cummax() - 1).min()),
    }


PERIODS = {
    "2018-20": ("2018-01-01", "2021-01-01"),
    "2021-23": ("2021-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025-26": ("2025-01-01", "2026-10-01"),
}


def _by_period(weekly: pd.Series) -> dict:
    idx = pd.DatetimeIndex(weekly.index).tz_convert(None)
    return {k: stats(weekly[(idx >= lo) & (idx < hi)]) for k, (lo, hi) in PERIODS.items()}


def full_market(study):
    """The study's market without data_end (reads the holdout: unlock first)."""
    import copy

    from engine import run

    cfg = copy.deepcopy(study.cfg)
    cfg["data_end"] = None
    m = run.study_from_config(study.name, cfg).market
    _ = m.series
    return m


def run_book(m, start, end, divisor="eligible", scaled=True, only=None, c=None, report=print):
    """Rows, side costs and the weekly book for one variant; coverage is reported before returns."""
    c = closes(m) if c is None else c
    rows = weekly_rows(m, start, end, only=only, c=c)
    by_year = rows.groupby(rows["decision_time"].dt.year)["n"].agg(["min", "median", "max"])
    report(f"coverage (coins in the set per week, before any return):\n{by_year.to_string()}")
    side = m.cost_bps(rows[["entity_id", "decision_time"]]) / 2 / 1e4
    return rows, side, book(rows, side, divisor, scaled), c


def trend_test(study, start, end, holdout_reason: str, divisor="eligible", note="", report=print):
    """ONE logged test (pre-registration 7, test 15): the trend book's weekly net return regressed
    on BTC's; t of alpha = the smaller of HC3 and Newey-West; kept if t >= the registry's next bar
    and alpha > 0. Uses one check look (2024 is inside) and opens the holdout (logged) first."""
    from engine import score

    reg = study.registry
    bar = reg.next_bar()
    reg.open_check()  # 2024 is inside: counts a check look
    study.holdout.unlock(study.name, holdout_reason)
    m = full_market(study)
    rows, side, b, c = run_book(m, start, end, divisor, report=report)
    btc = btc_weekly(b.index, c)
    years = len(b) / 52
    report(
        f"power: {len(b)} weeks ({years:.2f} years), bar {bar:.3f}: alpha information ratio "
        f"{bar / np.sqrt(years):.2f}/yr for 50% power, {(bar + 0.8416) / np.sqrt(years):.2f} for 80%"
    )
    j = pd.concat({"net": b["net"], "btc": btc}, axis=1).dropna()
    fit = score.ols_alpha_robust(j["net"], j["btc"])
    kept = bool(fit["t_alpha"] >= bar and fit["alpha"] > 0)
    report(
        f"PRIMARY alpha vs BTC: {fit['alpha'] * 1e4:+.1f} bp/wk ({fit['alpha'] * 52:+.1%}/yr), t "
        f"{fit['t_alpha']:.2f} (HC3 {fit['t_hc3']:.2f}, Newey-West {fit['t_nw']:.2f}, "
        f"{fit['nw_lags']} lags), beta {fit['beta']:.2f}, bar {bar:.3f} -> "
        f"{'KEPT' if kept else 'not kept'}"
    )
    ew = equal_weight_hold(rows)
    secondary = {
        "book": _by_period(b["net"]) | {"all": stats(b["net"])},
        "btc": _by_period(btc) | {"all": stats(btc)},
        "equal_weight_hold": _by_period(ew) | {"all": stats(ew)},
        "avg_invested": float(b["gross_weight"].mean()),
        "annual_turnover": float(b["turnover"].mean() * 52),
        "cost_drag_per_year": float(b["cost"].mean() * 52),
    }
    variants = {
        "no_vol_scaling": book(rows, side, divisor, scaled=False)["net"],
        "btc_eth_only": run_book(
            m, start, end, divisor, only=("BTC", "ETH"), c=c, report=lambda s: None
        )[2]["net"],
    }
    for k, v in variants.items():
        jv = pd.concat({"net": v, "btc": btc}, axis=1).dropna()
        secondary[k] = stats(v) | {
            "alpha_vs_btc_t": round(score.ols_alpha_robust(jv["net"], jv["btc"])["t_alpha"], 2)
        }
    row = {
        "kind": "confirm",
        "name": "trend book top 20 vs BTC",
        "features": "trend 28/56/112/224d, vol target 50%",
        "scope": "universal",
        "metric": "alpha of weekly net return on BTC, t = min(HC3, Newey-West)",
        "bar_tune": round(bar, 3),
        "check_used": True,
        "t_check": round(fit["t_alpha"], 3),
        "bar_check": round(bar, 3),
        "gain_check": round(fit["alpha"], 6),
        "kept": kept,
        "note": (
            f"{note} {len(j)} weeks {start}..{end}, divisor {divisor}; beta {fit['beta']:.2f}; "
            f"HC3 t {fit['t_hc3']:.2f}, NW t {fit['t_nw']:.2f}"
        ).strip(),
    }
    return {
        "row": reg.record(row),
        "fit": fit,
        "secondary": secondary,
        "weekly": j.assign(gross_weight=b["gross_weight"], cost=b["cost"]),
        "rows": rows.assign(side_cost=side.to_numpy()),
    }


def jev_test(
    study, rows: pd.DataFrame, jev_exposure: pd.Series, divisor="eligible", note="", report=print
):
    """ONE logged test (pre-registration 7, test 16): the book with Jev's exposure in place of the
    rule's, minus the rule's book, weekly net; t = the smaller of iid and Newey-West; kept if t >=
    the registry's next bar and the mean > 0. `rows` and their side_cost come from trend_test."""
    from engine import score

    reg = study.registry
    bar = reg.next_bar()
    reg.open_check()  # 2024 and the holdout are inside: counts a check look
    side = rows["side_cost"]
    rule = book(rows, side, divisor)["net"]
    jev = book(rows.assign(exposure=jev_exposure.reindex(rows.index).to_numpy()), side, divisor)[
        "net"
    ]
    d = (jev - rule).dropna()
    t = score.per_period_t(d, 1, "auto")
    kept = bool(t >= bar and d.mean() > 0)
    report(
        f"PRIMARY Jev book minus rule book: {d.mean() * 1e4:+.1f} bp/wk over {len(d)} weeks, t "
        f"{t:.2f} (iid {score.per_period_t(d):.2f}), bar {bar:.3f} -> {'KEPT' if kept else 'not kept'}"
    )
    diff = (jev_exposure.reindex(rows.index) - rows["exposure"]).abs()
    big = rows[diff >= 0.5]
    secondary = {
        "rule_avg_exposure": float(rows["exposure"].mean()),
        "jev_avg_exposure": float(jev_exposure.reindex(rows.index).mean()),
        "overrides_ge_50pts": len(big),
        "override_weeks_mean_ret": float(big["ret"].mean()) if len(big) else np.nan,
        "jev_book": _by_period(jev) | {"all": stats(jev)},
        "rule_book": _by_period(rule) | {"all": stats(rule)},
    }
    row = {
        "kind": "confirm",
        "name": "jev judges the trend book",
        "features": "jev-1.13.0 score over the trend card",
        "scope": "universal",
        "metric": "Jev book minus rule book, weekly net, t = min(iid, Newey-West)",
        "bar_tune": round(bar, 3),
        "check_used": True,
        "t_check": round(t, 3),
        "bar_check": round(bar, 3),
        "gain_check": round(float(d.mean()), 6),
        "kept": kept,
        "note": (f"{note} {len(d)} weeks; overrides >= 50 pts: {len(big)}").strip(),
    }
    return {"row": reg.record(row), "t": t, "diff": d, "secondary": secondary}
