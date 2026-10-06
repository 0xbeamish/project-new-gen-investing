"""Daily bars from EODHD, cleaned before anything uses them, and the trading costs measured from
them.

    fetch_prices(code)  close-only bars (date, close, adj_close, volume) -> .cache/eodhd/px/
    fetch_ohlc(code)    open/high/low/close/volume                       -> .cache/eodhd/ohlc/
    load_prices(code)   cleaned bars: OHLC when fetched, else the close-only cache cleaned the same way
Cleaning, in order: drop zero or negative prices; drop zero-volume days (carried prices, and the
padding after a delisting); repair adjusted-close jumps the raw close doesn't share; drop bad prints
(an extreme move that mostly reverses within 3 sessions); put open/high/low on the adjusted scale.
Unadjusted-split lookalikes are only counted, not changed. Fetching needs EODHD_API_KEY; reading
the cache needs nothing.

Measured costs: effective bid-ask spreads per stock-month, built once from the clean OHLC cache:
    uv run python -m engine.markets.us_stocks.prices spreads [--lists ...] [--out ...] [--until ...]
Two estimators, both the effective spread s as a fraction of price (buying at the ask and selling
at the bid costs s, so s IS the round trip):
  Abdi-Ranaldo (2017), close-high-low: s^2 = 4 * mean[(c_t - eta_t)(c_t - eta_{t+1})], c = log
      close, eta = (log high + log low) / 2; one estimate per stock-month from the trailing 63
      trading days (negatives -> 0); the 1-month version is kept as `ar_1m`
  Corwin-Schultz (2012), high-low: from 2-day vs 1-day ranges, daily estimates (negatives -> 0)
      averaged over the month. Runs high in quiet, low-priced names
The market's cost is the mean of the two.
"""

import functools
import gzip
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(".cache") / "eodhd"
PX_DIR = CACHE / "px"
OHLC_DIR = CACHE / "ohlc"
PX_START = "2009-06-01"
EXCHANGES = {"NASDAQ", "NYSE", "NYSE MKT", "AMEX", "BATS", "NYSE ARCA"}
# A delisted code's label is its LAST venue: a company that fell from NASDAQ to the pink sheets is
# labelled OTC. Those are kept, but only up to their exchange delisting (see `listing_evidence`).
OTC_EXCHANGES = {
    "PINK",
    "OTCGREY",
    "OTCQB",
    "OTCQX",
    "OTCMKTS",
    "OTCBB",
    "OTCCE",
    "NMFQS",
    "OTC",
}


def _key() -> str:
    k = os.environ.get("EODHD_API_KEY")
    if not k:
        raise RuntimeError("EODHD_API_KEY is not set")
    return k


def symbols(otc_delisted: bool = True) -> pd.DataFrame:
    """US common stocks on EODHD: listed (active + delisted), plus delisted codes whose final venue
    was OTC (kept for their exchange-listed years only). Columns: code, name, active, final_exchange."""
    frames = []
    for name, extra in (("active", {}), ("delisted", {"delisted": 1})):
        path = CACHE / f"symbols_{name}.json"
        if not path.exists():
            r = requests.get(
                "https://eodhd.com/api/exchange-symbol-list/US",
                params={"api_token": _key(), "fmt": "json", **extra},
                timeout=120,
            )
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
        d = pd.DataFrame(json.loads(path.read_text()))
        venues = EXCHANGES | (OTC_EXCHANGES if name == "delisted" and otc_delisted else set())
        d = d[(d["Type"] == "Common Stock") & d["Exchange"].isin(venues)]
        frames.append(d.assign(active=name == "active"))
    out = pd.concat(frames, ignore_index=True)
    code = out["Code"].str.replace(r"_old\d*$", "", regex=True)
    derivative = (code.str.len() == 5) & code.str[-1].isin(
        ["W", "U", "R"]
    )  # warrants, units, rights
    out = out[~derivative & ~code.str.contains(r"-(?:UN|U|WT|WS|W|RT|R)$", regex=True)]
    return out.rename(columns={"Code": "code", "Name": "name", "Exchange": "final_exchange"})[
        ["code", "name", "active", "final_exchange"]
    ].dropna(subset=["name"])


_LOCK = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 60 / 900  # EODHD allows ~1,000 requests a minute; stay under it


def _get(url: str, params: dict) -> requests.Response:
    """GET with a shared rate limit, retrying 429s and network hiccups with backoff."""
    for attempt in range(6):
        with _LOCK:
            wait = _last[0] + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if r.status_code != 429:
            return r
        time.sleep(min(60, 5 * 2**attempt))
    r.raise_for_status()
    return r


def _px_path(code: str) -> Path:
    return PX_DIR / f"{code}.csv.gz"


def fetch_prices(code: str) -> str:
    """Daily bars for one code, cached compactly: date, close (raw), adj_close, volume."""
    path = _px_path(code)
    if path.exists():
        return "cached"
    r = _get(
        f"https://eodhd.com/api/eod/{code}.US",
        {"api_token": _key(), "fmt": "json", "from": PX_START},
    )
    if r.status_code == 404:
        rows = []
    else:
        r.raise_for_status()
        rows = r.json()
    df = pd.DataFrame(rows, columns=["date", "close", "adjusted_close", "volume"])
    PX_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        df.rename(columns={"adjusted_close": "adj_close"}).to_csv(f, index=False)
    return "fetched" if len(df) else "empty"


def _load_close_only(code: str) -> pd.DataFrame:
    path = _px_path(code)
    if not path.exists():
        return pd.DataFrame(columns=["close", "adj_close", "volume"])
    with gzip.open(path, "rt") as f:
        return pd.read_csv(io.StringIO(f.read()), parse_dates=["date"], index_col="date")


def quarter_snapshot(px: pd.DataFrame, qend: pd.Timestamp) -> tuple[float, float] | None:
    """(raw close on the last trading day <= qend, 63-day average dollar volume), or None if stale."""
    p = px[px.index <= qend]
    if p.empty or (qend - p.index[-1]).days > 7:
        return None
    last = p.tail(63)
    return float(p["close"].iloc[-1]), float((last["close"] * last["volume"]).mean())


# ---------- cleaning
COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="date"), dtype=float)


BAD_UP, BAD_DOWN, REVERSE_SHARE, REVERSE_DAYS = 3.0, -0.8, 0.5, 3
ADJ_JUMP_UP, ADJ_JUMP_DOWN, RAW_CALM = 1.0, -0.5, 0.15
SPLIT_RATIOS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100])


def _path(code: str) -> Path:
    return OHLC_DIR / f"{code}.csv.gz"


def fetch_ohlc(code: str) -> str:
    """Daily OHLC bars for one code, cached."""
    path = _path(code)
    if path.exists():
        return "cached"
    r = _get(
        f"https://eodhd.com/api/eod/{code}.US",
        {"api_token": _key(), "fmt": "json", "from": PX_START},
    )
    rows = [] if r.status_code == 404 else (r.raise_for_status() or r.json())
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "adjusted_close", "volume"],
    )
    OHLC_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        df.rename(columns={"adjusted_close": "adj_close"}).to_csv(f, index=False)
    return "fetched" if len(df) else "empty"


def load_raw(code: str) -> pd.DataFrame:
    """OHLC bars if re-fetched; otherwise the close-only cache, with open/high/low left empty."""
    path = _path(code)
    if not path.exists():
        return _load_close_only(code).reindex(columns=COLS)
    with gzip.open(path, "rt") as f:
        return pd.read_csv(io.StringIO(f.read()), parse_dates=["date"], index_col="date")


def clean(px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleaned bars (adjusted OHLC) and a report of what was removed or repaired."""
    rep = {"rows_in": len(px)}
    if px.empty:
        return _empty(), rep | {"rows_out": 0}
    px = px.astype(float).sort_index()
    rep["has_ohlc"] = bool(px["open"].notna().any())
    bad_price = (px[["close", "adj_close"]] <= 0).any(axis=1) | px["close"].isna()
    bad_price |= (px[["open", "high", "low"]] <= 0).any(axis=1)  # NaN (close-only cache) passes
    rep["zero_or_negative_price"] = int(bad_price.sum())
    px = px[~bad_price]
    no_trade = px["volume"].fillna(0) <= 0
    last_trade = px.index[~no_trade].max() if (~no_trade).any() else None
    rep["trailing_padding"] = (
        int((px.index > last_trade).sum()) if last_trade is not None else len(px)
    )
    rep["zero_volume_days"] = int(no_trade.sum())
    px = px[~no_trade]
    if px.empty:
        return _empty(), rep | {"rows_out": 0}

    # adjusted-close errors: keep the adjusted series chained through the raw day's move
    r_adj, r_raw = px["adj_close"].pct_change(), px["close"].pct_change()
    adj_err = ((r_adj > ADJ_JUMP_UP) | (r_adj < ADJ_JUMP_DOWN)) & (r_raw.abs() < RAW_CALM)
    rep["adj_close_errors"] = int(adj_err.sum())
    if adj_err.any():
        r = r_adj.where(~adj_err, r_raw).fillna(0)
        px["adj_close"] = px["adj_close"].iloc[0] * (1 + r).cumprod()

    # bad prints: an extreme move that mostly reverses within a few sessions
    adj = px["adj_close"].to_numpy()
    drop = np.zeros(len(px), bool)
    i = 1
    while i < len(adj):
        move = adj[i] / adj[i - 1] - 1
        if move > BAD_UP or move < BAD_DOWN:
            for k in range(1, REVERSE_DAYS + 1):
                if i + k < len(adj) and abs(adj[i + k] / adj[i - 1] - 1) < REVERSE_SHARE * abs(
                    move
                ):
                    drop[i : i + k] = True
                    i += k
                    break
        i += 1
    rep["bad_prints"] = int(drop.sum())
    px = px[~drop]

    # flag only: both series jump by about a split ratio on the same day
    r_adj, r_raw = px["adj_close"].pct_change(), px["close"].pct_change()
    ratio = np.maximum(1 + r_raw, 1 / (1 + r_raw).clip(lower=1e-9))
    near_split = np.isclose(ratio.to_numpy()[:, None], SPLIT_RATIOS, rtol=0.03).any(axis=1)
    rep["possible_unadjusted_splits"] = int(
        ((r_adj - r_raw).abs() < 0.02).to_numpy().__and__(near_split).sum()
    )

    factor = px["adj_close"] / px["close"]
    for c in ("open", "high", "low"):
        px[c] = px[c] * factor
    rep["rows_out"] = len(px)
    return px[COLS], rep


@functools.lru_cache(maxsize=4096)
def load_prices(code: str) -> pd.DataFrame:
    """Cleaned bars: OHLC once fetched, else the close-only cache, cleaned the same way."""
    return clean(load_raw(code))[0]


# ---------------------------------------------------------------- measured spreads
SPREADS_OUT = Path(".cache") / "spreads.pkl"
WINDOW_DAYS = (
    63  # one month of bars gives AR a zero in ~45% of liquid stock-months; 3 months is steadier
)


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
    px = load_prices(code)
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


def build_spreads(lists=None, out_path=SPREADS_OUT, until=None) -> pd.DataFrame:
    """Spreads for every code in the universe lists (default: small/mid + micro) -> out_path."""
    from engine.markets.us_stocks import universe

    codes = set()
    for f in lists or (universe.OUT, universe.MICRO_OUT):
        if not Path(f).exists():
            continue
        codes |= set(
            pd.read_csv(f, dtype={"code": str}, keep_default_na=False, na_values=[""])["code"]
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


def load_spreads(path=None) -> pd.DataFrame:
    """The measured spread table (code, month, ar, cs, ...)."""
    return pd.read_pickle(path or SPREADS_OUT)


def main() -> None:
    """python -m engine.markets.us_stocks.prices spreads [--lists ...] [--out ...] [--until ...]"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="spreads", choices=["spreads"])
    ap.add_argument("--lists", nargs="*", help="universe CSVs (default small/mid + micro)")
    ap.add_argument("--out", default=str(SPREADS_OUT))
    ap.add_argument("--until", help="ignore bars after this date")
    a = ap.parse_args()
    out = build_spreads(a.lists, a.out, a.until)
    print(f"{len(out):,} stock-months -> {a.out}")


if __name__ == "__main__":
    main()
