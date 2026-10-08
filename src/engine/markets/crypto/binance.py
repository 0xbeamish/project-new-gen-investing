"""Binance's public archive (data.binance.vision): daily spot bars and perp funding, no key.

The archive keeps delisted pairs, so dead tokens stay in. Monthly zips are cached as one pickle
per symbol under .cache/crypto/. Timestamps are milliseconds (microseconds from 2025 on spot).
"""

from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from engine.run import path as repo_path

BASE = "https://data.binance.vision"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = repo_path(".cache/crypto")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _get(url: str, **kw) -> requests.Response:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"GET failed: {url}")


def list_keys(prefix: str, folders: bool) -> list[str]:
    """Every key (or sub-folder, if folders) under an archive prefix, following pagination."""
    out, marker = [], ""
    tag = "Prefix" if folders else "Key"
    while True:
        params = {"prefix": prefix, "marker": marker}
        if folders:
            params["delimiter"] = "/"
        xml = _get(LIST, params=params).text
        found = re.findall(rf"<{tag}>([^<]+)</{tag}>", xml)
        found = [f for f in found if f != prefix]
        out += found
        if "<IsTruncated>true" not in xml or not found:
            return out
        marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = marker[0] if marker else found[-1]


def spot_symbols(quote: str = "USDT", max_age_s: float = 86400) -> list[str]:
    """Every spot pair ever archived with this quote currency (delisted included); the cached list
    is refreshed when older than max_age_s (a build passes inf: no network)."""
    f = CACHE / f"spot_symbols_{quote}.txt"
    if f.exists() and time.time() - f.stat().st_mtime < max_age_s:
        return f.read_text().split()
    folders = list_keys("data/spot/monthly/klines/", folders=True)
    syms = sorted({p.rstrip("/").split("/")[-1] for p in folders})
    syms = [s for s in syms if s.endswith(quote) and len(s) > len(quote)]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(syms))
    return syms


def perp_symbols(max_age_s: float = 86400) -> set[str]:
    """Every USD-margined perp with archived funding rates."""
    f = CACHE / "perp_symbols.txt"
    if f.exists() and time.time() - f.stat().st_mtime < max_age_s:
        return set(f.read_text().split())
    folders = list_keys("data/futures/um/monthly/fundingRate/", folders=True)
    syms = sorted({p.rstrip("/").split("/")[-1] for p in folders})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(syms))
    return set(syms)


def _read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if re.search(r"[a-z]", first) else None  # newer files carry a header row
    return pd.read_csv(io.StringIO(raw), header=header)


def _ms(ts: pd.Series) -> pd.Series:
    """Epoch ms or us -> UTC timestamps."""
    v = pd.to_numeric(ts, errors="coerce")
    v = v.where(v < 1e14, v // 1000)  # microseconds -> ms
    return pd.to_datetime(v, unit="ms", utc=True)


def _months(kind_prefix: str, symbol: str, until: pd.Timestamp, since: str) -> list[str]:
    keys = list_keys(f"{kind_prefix}{symbol}/", folders=False)
    keys = [k for k in keys if k.endswith(".zip")]
    last = until.strftime("%Y-%m")
    return [k for k in keys if since <= re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) <= last]


def _fetch(kind: str, symbol: str, until: pd.Timestamp, since: str) -> str:
    """All monthly zips for one symbol up to `until` -> one pickle. Skips if already complete."""
    out = CACHE / kind / f"{symbol}.pkl"
    stamp = out.with_suffix(".until")  # how far the last fetch looked (delisted pairs stop early)
    if out.exists() and stamp.exists() and stamp.read_text() >= until.strftime("%Y-%m"):
        return "cached"
    prefix = (
        "data/spot/monthly/klines/" if kind == "spot" else "data/futures/um/monthly/fundingRate/"
    )
    keys = _months(prefix, symbol, until, since)
    if kind == "spot":
        keys = [k for k in keys if "/1d/" in k]
    parts = []
    for k in keys:
        r = _get(f"{BASE}/{k}")
        if r.status_code != 200:
            continue
        parts.append(_parse(kind, _read_zip(r.content)))
    frame = pd.concat(parts, ignore_index=True) if parts else _parse(kind, pd.DataFrame())
    frame = frame.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(out)
    stamp.write_text(until.strftime("%Y-%m"))
    return f"{len(keys)} months"


def _parse(kind: str, z: pd.DataFrame) -> pd.DataFrame:
    if kind == "spot":
        if z.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "quote_volume"])
        z = z.iloc[:, : len(KLINE_COLS)]
        z.columns = KLINE_COLS
        f = pd.DataFrame(
            {"time": _ms(z["open_time"]).dt.tz_localize(None).dt.normalize()}
        )  # UTC date
        for c in ("open", "high", "low", "close", "quote_volume"):
            f[c] = pd.to_numeric(z[c], errors="coerce")
        return f
    if z.empty:
        return pd.DataFrame(columns=["time", "rate"])
    cols = [str(c) for c in z.columns]
    if "calc_time" in cols:
        return pd.DataFrame(
            {"time": _ms(z["calc_time"]), "rate": pd.to_numeric(z["last_funding_rate"], "coerce")}
        )
    return pd.DataFrame({"time": _ms(z.iloc[:, 0]), "rate": pd.to_numeric(z.iloc[:, -1], "coerce")})


def fetch_all(kind: str, symbols, until: pd.Timestamp, since: str, workers: int = 16) -> None:
    """Fill the cache for many symbols in parallel (kind: spot | funding), months since..until."""
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(lambda s: _fetch(kind, s, until, since), symbols), 1):
            if i % 100 == 0 or i == len(symbols):
                _log(f"  binance {kind}: {i:,}/{len(symbols):,}")


def load(kind: str, symbol: str) -> pd.DataFrame:
    """The cached frame (empty if never fetched)."""
    f = CACHE / kind / f"{symbol}.pkl"
    return pd.read_pickle(f) if f.exists() else _parse(kind, pd.DataFrame())
