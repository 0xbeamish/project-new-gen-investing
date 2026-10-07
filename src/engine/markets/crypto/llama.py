"""DeFiLlama's free API: daily fees per protocol or chain, and the hacks list, mapped to token symbols.

A token's fees are its parent protocol's (Uniswap = every Uniswap version), or its chain's (ETH =
Ethereum's transaction fees). A symbol claimed by several fee entries goes to the chain if exactly
one is a chain, else to the one with the larger all-time fees if it is 10x the runner-up, else to
none. Cached under .cache/crypto/llama/.
DeFiLlama back-fills new adapters, so a past value is what is known now, not what was known then.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from engine.markets.crypto.binance import CACHE

API = "https://api.llama.fi"
DIR = CACHE / "llama"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _json(url: str, name: str, max_age_s: float):
    """GET url as JSON, cached in DIR/name for max_age_s (inf = never refresh)."""
    f = DIR / name
    if f.exists() and time.time() - f.stat().st_mtime < max_age_s:
        return json.loads(f.read_text())
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(r.text)
                return r.json()
            if r.status_code in (400, 402, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"GET failed: {url}")


def _sym(s) -> str | None:
    s = str(s or "").strip().upper()
    return s if s and s != "-" and s != "NONE" else None


def fee_entities(max_age_s: float = float("inf")) -> pd.DataFrame:
    """One row per fee-reporting entity: slug to query, token symbol, all-time fees."""
    ov = _json(
        f"{API}/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true",
        "overview_fees.json",
        max_age_s,
    )["protocols"]
    lite = _json(f"{API}/lite/protocols2", "protocols2.json", max_age_s)
    chains = _json(f"{API}/v2/chains", "chains.json", max_age_s)
    proto_sym = {str(p.get("defillamaId")): _sym(p.get("symbol")) for p in lite["protocols"]}
    parent_sym = {p["id"]: _sym(p.get("symbol")) for p in lite["parentProtocols"]}
    chain_sym = {c["name"]: _sym(c.get("tokenSymbol")) for c in chains}
    rows = {}
    for p in ov:
        if p.get("parentProtocol"):
            slug, sym = p["parentProtocol"].split("#", 1)[1], parent_sym.get(p["parentProtocol"])
        elif p.get("category") == "Chain":
            slug, sym = p["slug"], chain_sym.get(p["name"]) or chain_sym.get(p.get("displayName"))
        else:
            slug, sym = p["slug"], proto_sym.get(str(p.get("defillamaId")))
        if not sym:
            continue
        chain = p.get("category") == "Chain" and not p.get("parentProtocol")
        r = rows.setdefault(slug, {"slug": slug, "symbol": sym, "all_time": 0.0, "chain": chain})
        r["all_time"] += float(p.get("totalAllTime") or 0)
    return pd.DataFrame(list(rows.values()), columns=["slug", "symbol", "all_time", "chain"])


def token_fee_slugs(symbols, max_age_s: float = float("inf")) -> dict[str, str]:
    """Token symbol -> the one fee slug it maps to (ambiguous symbols left out). Reads the cached
    lists unless they are older than max_age_s (a fetch passes a day)."""
    e = fee_entities(max_age_s)
    e = e[e["symbol"].isin(set(symbols))].sort_values("all_time", ascending=False)
    out = {}
    for sym, g in e.groupby("symbol"):
        top = g["all_time"].to_numpy()
        if g["chain"].sum() == 1:  # a chain's own token (NEAR, OP): its chain's fees
            out[sym] = g.loc[g["chain"], "slug"].iloc[0]
        elif len(top) == 1 or top[0] >= 10 * top[1]:
            out[sym] = g["slug"].iloc[0]
        else:
            _log(f"  llama: {sym} is ambiguous ({', '.join(g['slug'].head(3))}): left out")
    return out


def fetch_fees(slugs, workers: int = 8) -> None:
    """Daily fee history for each slug (refreshed if a day old)."""

    def one(slug):
        _json(f"{API}/summary/fees/{slug}?dataType=dailyFees", f"fees/{slug}.json", 86400)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, sorted(set(slugs))))


def daily_fees(slug: str) -> pd.Series:
    """USD fees by UTC date (empty if none)."""
    f = DIR / "fees" / f"{slug}.json"
    d = json.loads(f.read_text()) if f.exists() else None
    chart = (d or {}).get("totalDataChart") or []
    if not chart:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series(
        [float(v or 0) for _, v in chart],
        index=pd.to_datetime([t for t, _ in chart], unit="s").normalize(),
    )
    return s.groupby(level=0).sum().sort_index()


def hacks(symbols, max_age_s: float = float("inf")) -> pd.DataFrame:
    """Hacks of a protocol whose token is in symbols: symbol, date (UTC day), usd."""
    h = _json(f"{API}/hacks", "hacks.json", max_age_s) or []
    lite = _json(f"{API}/lite/protocols2", "protocols2.json", max_age_s)
    ov = _json(
        f"{API}/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true",
        "overview_fees.json",
        max_age_s,
    )["protocols"]
    sym = {str(p.get("defillamaId")): _sym(p.get("symbol")) for p in lite["protocols"]}
    parent_sym = {p["id"]: _sym(p.get("symbol")) for p in lite["parentProtocols"]}
    parent_of = {str(p.get("defillamaId")): p.get("parentProtocol") for p in ov}
    rows = []
    for x in h:
        pid = str(x.get("defillamaId") or "")
        s = sym.get(pid) or parent_sym.get(parent_of.get(pid))
        if s in symbols and x.get("amount"):
            rows.append(
                {
                    "symbol": s,
                    "date": pd.to_datetime(x["date"], unit="s").normalize(),
                    "usd": float(x["amount"]),
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "date", "usd"])
