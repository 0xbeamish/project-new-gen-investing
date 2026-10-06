"""Measured trading costs: effective bid-ask spreads from daily high/low/close (clean OHLC).

    uv run python -m engine.markets.us_smallcap.spreads build   # per stock-month -> .cache/spreads.pkl

Two estimators, both giving the effective spread s as a fraction of price. Buying at the ask and
selling at the bid costs s, so s IS the round-trip cost.
  Abdi-Ranaldo (2017), close-high-low: s^2 = 4 * mean[(c_t - eta_t)(c_t - eta_{t+1})], where
      c = log close and eta = (log high + log low) / 2. One estimate per stock-month from the
      trailing 63 trading days (negatives -> 0); the 1-month version is kept as `ar_1m`.
  Corwin-Schultz (2012), high-low: from 2-day vs 1-day high/low ranges, daily estimates with
      negatives set to 0, averaged over the month. Known to run high in quiet, low-priced names.
The primary number is Abdi-Ranaldo; Corwin-Schultz is reported as a check.

Market impact for ~$10k positions (square-root law, impact = Y * sigma_daily * sqrt(Q / ADV), Y ~ 1):
reported separately as a sensitivity, not folded into the main cost. At $10k against the
universe's >= $1M a day it's ~1% of volume (impact ~ 0.1 * daily vol, similar to the spread); in
micro caps ($100k a day) it can be 10% of volume, where the formula is least reliable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.markets.us_smallcap import prices, universe

OUT = Path(".cache") / "spreads.pkl"
POSITION_USD = 10_000
WINDOW_DAYS = 63  # one month of bars gives AR a zero in ~45% of liquid stock-months; 3 months is steadier
IMPACT_Y = 1.0


def abdi_ranaldo(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    """One spread for a window of daily bars (log prices in)."""
    eta = (h + l) / 2
    if len(c) < 3:
        return np.nan
    v = 4 * np.mean((c[:-1] - eta[:-1]) * (c[:-1] - eta[1:]))
    return float(np.sqrt(v)) if v > 0 else 0.0


def corwin_schultz(h: np.ndarray, l: np.ndarray) -> float:
    """Mean of daily 2-day estimates over the window (log high/low in)."""
    if len(h) < 3:
        return np.nan
    beta = (h[:-1] - l[:-1]) ** 2 + (h[1:] - l[1:]) ** 2
    gamma = (np.maximum(h[:-1], h[1:]) - np.minimum(l[:-1], l[1:])) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.mean(np.clip(s, 0, None)))


def stock_months(code: str, until=None) -> pd.DataFrame:
    """Per calendar month: AR and CS spreads, daily volatility, average dollar volume.
    until: ignore bars after this date."""
    px = prices.load_prices(code)
    if until is not None:
        px = px[px.index <= pd.Timestamp(until)]
    px = px.dropna(subset=["open", "high", "low", "close"])
    px = px[(px["high"] >= px["low"]) & (px["high"] > 0)]
    if len(px) < 30:
        return pd.DataFrame()
    lh, ll, lc = (
        np.log(px["high"].to_numpy()),
        np.log(px["low"].to_numpy()),
        np.log(px["adj_close"].to_numpy()),
    )
    dvol = (px["close"] * px["volume"]).to_numpy()  # raw close x shares traded
    month = px.index.to_period("M")
    rows = []
    pos = pd.Series(range(len(px)), index=month)
    for m, idx in pos.groupby(level=0):
        i = idx.to_numpy()
        if len(i) < 10:
            continue
        w = np.arange(
            max(0, i[-1] - WINDOW_DAYS + 1), i[-1] + 1
        )  # trailing ~3 months ending this month
        r = np.diff(lc[i])
        rows.append(
            {
                "code": code,
                "month": m,
                "ar": abdi_ranaldo(lh[w], ll[w], lc[w]),
                "ar_1m": abdi_ranaldo(lh[i], ll[i], lc[i]),
                "cs": corwin_schultz(lh[w], ll[w]),
                "vol_d": float(np.std(r)) if len(r) > 2 else np.nan,
                "adv": float(np.mean(dvol[i])),
            }
        )
    return pd.DataFrame(rows)


def impact(
    vol_d: pd.Series, adv: pd.Series, position: float = POSITION_USD
) -> pd.Series:
    """Square-root impact, one way; x2 for a round trip."""
    return 2 * IMPACT_Y * vol_d * np.sqrt(position / adv.clip(lower=1.0))


def build(lists=None, out_path=OUT, until=None) -> pd.DataFrame:
    """Spreads for every code in the universe lists (default: small/mid + micro) -> out_path."""
    codes = set()
    for f in lists or (universe.OUT, universe.MICRO_OUT):
        if not Path(f).exists():
            continue
        codes |= set(
            pd.read_csv(f, dtype={"code": str}, keep_default_na=False, na_values=[""])[
                "code"
            ]
        )
    parts = []
    for i, c in enumerate(sorted(codes), 1):
        parts.append(stock_months(c, until))
        if i % 1000 == 0:
            print(f"  {i:,}/{len(codes):,} codes", file=sys.stderr)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(out_path)
    return out


def load(path=None) -> pd.DataFrame:
    return pd.read_pickle(path or OUT)


def attach(rows: pd.DataFrame, month_col: str = "as_of") -> pd.DataFrame:
    """rows: code + a date. Adds the spread measured over that calendar month (rt_cost_ar/cs) and the
    impact for a $10k round trip. Uses the month of the trade, as a cost estimate (not an input)."""
    sp = load()
    m = rows.assign(month=pd.to_datetime(rows[month_col]).dt.to_period("M"))
    m = m.merge(sp, on=["code", "month"], how="left")
    m["rt_cost_ar"], m["rt_cost_cs"] = m["ar"], m["cs"]
    m["rt_impact"] = impact(m["vol_d"], m["adv"])
    return m.drop(columns=["month"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="build", choices=["build"])
    ap.add_argument(
        "--lists", nargs="*", help="universe CSVs (default small/mid + micro)"
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--until", help="ignore bars after this date")
    a = ap.parse_args()
    out = build(a.lists, a.out, a.until)
    print(f"{len(out):,} stock-months -> {a.out}")


if __name__ == "__main__":
    main()
