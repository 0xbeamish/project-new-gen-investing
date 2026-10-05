"""Daily bars from EODHD, cleaned before anything uses them (ported from the pilot).

    fetch_prices(code)  close-only bars (date, close, adj_close, volume) -> .cache/eodhd/px/
    fetch_ohlc(code)    open/high/low/close/volume                       -> .cache/eodhd/ohlc/
    load_prices(code)   cleaned bars: OHLC when fetched, else the close-only cache cleaned the same way

Cleaning, in order: drop zero or negative prices; drop zero-volume days (carried prices, and the
padding after a delisting); repair adjusted-close jumps the raw close doesn't share; drop bad
prints (an extreme move that mostly reverses within 3 sessions); put open/high/low on the
adjusted scale. Unadjusted-split lookalikes are only counted, not changed.
Needs EODHD_API_KEY to fetch; reading the cache needs nothing.
"""

import functools
import gzip
import io
import json
import os
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
        venues = EXCHANGES | (
            OTC_EXCHANGES if name == "delisted" and otc_delisted else set()
        )
        d = d[(d["Type"] == "Common Stock") & d["Exchange"].isin(venues)]
        frames.append(d.assign(active=name == "active"))
    out = pd.concat(frames, ignore_index=True)
    code = out["Code"].str.replace(r"_old\d*$", "", regex=True)
    derivative = (code.str.len() == 5) & code.str[-1].isin(
        ["W", "U", "R"]
    )  # warrants, units, rights
    out = out[~derivative & ~code.str.contains(r"-(?:UN|U|WT|WS|W|RT|R)$", regex=True)]
    return out.rename(
        columns={"Code": "code", "Name": "name", "Exchange": "final_exchange"}
    )[["code", "name", "active", "final_exchange"]].dropna(subset=["name"])


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


def load_prices(code: str) -> pd.DataFrame:
    """Cleaned bars: OHLC once fetched, else the close-only cache, cleaned the same way."""
    return load_clean(code)


def load_prices_v1(code: str) -> pd.DataFrame:
    path = _px_path(code)
    if not path.exists():
        return pd.DataFrame(columns=["close", "adj_close", "volume"])
    with gzip.open(path, "rt") as f:
        return pd.read_csv(
            io.StringIO(f.read()), parse_dates=["date"], index_col="date"
        )


def quarter_snapshot(
    px: pd.DataFrame, qend: pd.Timestamp
) -> tuple[float, float] | None:
    """(raw close on the last trading day <= qend, 63-day average dollar volume), or None if stale."""
    p = px[px.index <= qend]
    if p.empty or (qend - p.index[-1]).days > 7:
        return None
    last = p.tail(63)
    return float(p["close"].iloc[-1]), float((last["close"] * last["volume"]).mean())


# ---------- cleaning
COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=COLS, index=pd.DatetimeIndex([], name="date"), dtype=float
    )


BAD_UP, BAD_DOWN, REVERSE_SHARE, REVERSE_DAYS = 3.0, -0.8, 0.5, 3
ADJ_JUMP_UP, ADJ_JUMP_DOWN, RAW_CALM = 1.0, -0.5, 0.15
SPLIT_RATIOS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100])


def _path(code: str) -> Path:
    return OHLC_DIR / f"{code}.csv.gz"


def fetch_ohlc(code: str) -> str:
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
        return load_prices_v1(code).reindex(columns=COLS)
    with gzip.open(path, "rt") as f:
        return pd.read_csv(
            io.StringIO(f.read()), parse_dates=["date"], index_col="date"
        )


def clean(px: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleaned bars (adjusted OHLC) and a report of what was removed or repaired."""
    rep = {"rows_in": len(px)}
    if px.empty:
        return _empty(), rep | {"rows_out": 0}
    px = px.astype(float).sort_index()
    rep["has_ohlc"] = bool(px["open"].notna().any())
    bad_price = (px[["close", "adj_close"]] <= 0).any(axis=1) | px["close"].isna()
    bad_price |= (px[["open", "high", "low"]] <= 0).any(
        axis=1
    )  # NaN (close-only cache) passes
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
    adj_err = ((r_adj > ADJ_JUMP_UP) | (r_adj < ADJ_JUMP_DOWN)) & (
        r_raw.abs() < RAW_CALM
    )
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
                if i + k < len(adj) and abs(
                    adj[i + k] / adj[i - 1] - 1
                ) < REVERSE_SHARE * abs(move):
                    drop[i : i + k] = True
                    i += k
                    break
        i += 1
    rep["bad_prints"] = int(drop.sum())
    px = px[~drop]

    # flag only: both series jump by about a split ratio on the same day
    r_adj, r_raw = px["adj_close"].pct_change(), px["close"].pct_change()
    ratio = np.maximum(1 + r_raw, 1 / (1 + r_raw).clip(lower=1e-9))
    near_split = np.isclose(ratio.to_numpy()[:, None], SPLIT_RATIOS, rtol=0.03).any(
        axis=1
    )
    rep["possible_unadjusted_splits"] = int(
        ((r_adj - r_raw).abs() < 0.02).to_numpy().__and__(near_split).sum()
    )

    factor = px["adj_close"] / px["close"]
    for c in ("open", "high", "low"):
        px[c] = px[c] * factor
    rep["rows_out"] = len(px)
    return px[COLS], rep


@functools.lru_cache(maxsize=4096)
def load_clean(code: str) -> pd.DataFrame:
    return clean(load_raw(code))[0]
