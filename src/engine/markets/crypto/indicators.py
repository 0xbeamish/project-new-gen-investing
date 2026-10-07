"""Daily fundamental series per token, free: DeFiLlama (fees, revenue, holders revenue, TVL, borrows,
DEX volume, active users; chain TVL and stablecoins) and CoinMetrics Community (market cap, 135
older assets). A protocol token maps to its parent protocol (UNI = every Uniswap version), a chain's
token to its chain. Cached under .cache/crypto/llama/ and .cache/crypto/coinmetrics/.

    python -m engine.markets.crypto.indicators fetch     # fills the caches for markets/crypto_sectors.csv
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from engine.markets.crypto import llama
from engine.run import path as repo_path

API, SC = "https://api.llama.fi", "https://stablecoins.llama.fi"
CM = "https://community-api.coinmetrics.io/v4"
CM_DIR = llama.CACHE / "coinmetrics"
SECTORS = repo_path("markets/crypto_sectors.csv")
DAY = 86400.0
INF = float("inf")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def sectors() -> pd.DataFrame:
    """token, sector (hand-labelled today: a mild look-ahead for tokens whose story changed)."""
    return pd.read_csv(SECTORS, keep_default_na=False)


# ---------- which DeFiLlama names a token maps to ----------
def protocol_slugs(symbols, max_age_s: float = INF) -> dict[str, str]:
    """Token -> parent-or-own protocol slug, by largest current TVL (10x the runner-up, or alone)."""
    ps = llama._json(f"{API}/protocols", "protocols.json", max_age_s)
    tvl: dict[str, dict[str, float]] = {}
    for p in ps:
        sym = llama._sym(p.get("symbol"))
        if sym not in symbols or p.get("category") == "Chain":
            continue
        slug = p.get("parentProtocolSlug") or p["slug"]
        tvl.setdefault(sym, {}).setdefault(slug, 0.0)
        tvl[sym][slug] += float(p.get("tvl") or 0) + 1.0
    out = {}
    for sym, d in tvl.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])
        if len(top) == 1 or top[0][1] >= 10 * top[1][1]:
            out[sym] = top[0][0]
    return out


def chain_names(symbols, max_age_s: float = INF) -> dict[str, tuple[str, str | None]]:
    """Token -> (chain name for TVL / stablecoins, chain fee slug or None)."""
    chains = llama._json(f"{API}/v2/chains", "chains.json", max_age_s)
    fe = llama.fee_entities(max_age_s)
    fee_slug = dict(zip(fe.loc[fe["chain"], "symbol"], fe.loc[fe["chain"], "slug"]))
    out = {}
    for c in sorted(chains, key=lambda c: -float(c.get("tvl") or 0)):
        sym = llama._sym(c.get("tokenSymbol"))
        if sym in symbols and sym not in out:
            out[sym] = (c["name"], fee_slug.get(sym))
    return out


# ---------- fetch ----------
def _urls(token: str, slug: str | None, chain: tuple[str, str | None] | None) -> dict[str, str]:
    u = {}
    if slug:
        for dt in ("dailyFees", "dailyRevenue", "dailyHoldersRevenue"):
            u[f"p_{dt}"] = f"{API}/summary/fees/{slug}?dataType={dt}"
        u["p_dex"] = f"{API}/summary/dexs/{slug}"
        u["p_users"] = f"{API}/summary/active-users/{slug}"
        u["p_protocol"] = f"{API}/protocol/{slug}"
    if chain:
        name, fslug = chain
        u["c_tvl"] = f"{API}/v2/historicalChainTvl/{name}"
        u["c_stables"] = f"{SC}/stablecoincharts/{name}"
        u["c_dex"] = f"{API}/overview/dexs/{name}?excludeTotalDataChartBreakdown=true"
        if fslug:
            u["c_dailyFees"] = f"{API}/summary/fees/{fslug}?dataType=dailyFees"
            u["c_users"] = f"{API}/summary/active-users/{fslug}"
    return u


def _file(token: str, kind: str) -> str:
    return f"ind/{token}/{kind}.json"


def mapping(tokens, max_age_s: float = INF) -> dict[str, dict[str, str]]:
    """Token -> {series kind: url} for every token with a DeFiLlama protocol or chain."""
    tokens = set(tokens)
    slugs, chains = protocol_slugs(tokens, max_age_s), chain_names(tokens, max_age_s)
    return {t: _urls(t, slugs.get(t), chains.get(t)) for t in sorted(tokens)}


def fetch(tokens=None, workers: int = 8) -> None:
    """Every series for every token (refreshed if a week old), then CoinMetrics market caps."""
    tokens = list(tokens if tokens is not None else sectors()["token"])
    jobs = [(t, k, u) for t, d in mapping(tokens, 7 * DAY).items() for k, u in d.items()]
    _log(f"  indicators: {len(jobs):,} series for {len(tokens):,} tokens")

    def one(job) -> bool:
        try:
            return llama._json(job[2], _file(job[0], job[1]), 7 * DAY) is not None
        except RuntimeError:  # an endpoint that keeps failing (5xx): that series is missing
            return False

    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(one, jobs))
    _log(f"  indicators: {sum(ok):,} fetched, {len(ok) - sum(ok):,} missing or failed")
    fetch_market_caps(tokens)


def fetch_market_caps(tokens) -> None:
    """CoinMetrics Community daily CapMrktCurUSD for the tokens it covers, 2020 on."""
    cat = requests.get(
        f"{CM}/catalog-v2/asset-metrics?metrics=CapMrktCurUSD&page_size=10000", timeout=120
    )
    have = {d["asset"] for d in cat.json()["data"]}
    CM_DIR.mkdir(parents=True, exist_ok=True)
    for t in tokens:
        a = t.lower()
        if a not in have or (CM_DIR / f"{t}.json").exists():
            continue
        rows, url = (
            [],
            (
                f"{CM}/timeseries/asset-metrics?assets={a}&metrics=CapMrktCurUSD&frequency=1d"
                "&start_time=2020-01-01&page_size=10000"
            ),
        )
        while url:
            r = requests.get(url, timeout=120).json()
            rows += r.get("data", [])
            url = r.get("next_page_url")
        (CM_DIR / f"{t}.json").write_text(json.dumps(rows))


# ---------- load ----------
def _read(token: str, kind: str):
    f = llama.DIR / _file(token, kind)
    return json.loads(f.read_text()) if f.exists() else None


def _chart(pairs) -> pd.Series:
    if not pairs:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series(
        [float(v or 0) for _, v in pairs],
        index=pd.to_datetime([int(t) for t, _ in pairs], unit="s").normalize(),
    )
    return s.groupby(level=0).last().sort_index()


def series(token: str) -> dict[str, pd.Series]:
    """Every daily series cached for this token, by name (UTC dates, USD unless a count)."""
    out = {}
    for kind, name in (
        ("p_dailyFees", "fees"),
        ("p_dailyRevenue", "revenue"),
        ("p_dailyHoldersRevenue", "holders_revenue"),
        ("p_dex", "dex_volume"),
        ("p_users", "users"),
        ("c_dailyFees", "chain_fees"),
        ("c_users", "chain_users"),
        ("c_dex", "chain_dex_volume"),
    ):
        d = _read(token, kind)
        if d and d.get("totalDataChart"):
            out[name] = _chart(d["totalDataChart"])
    p = _read(token, "p_protocol")
    if p and p.get("tvl"):
        out["tvl"] = _chart([(x["date"], x["totalLiquidityUSD"]) for x in p["tvl"]])
        b = (p.get("chainTvls") or {}).get("borrowed", {}).get("tvl")
        if b:
            out["borrowed"] = _chart([(x["date"], x["totalLiquidityUSD"]) for x in b])
    c = _read(token, "c_tvl")
    if c:
        out["chain_tvl"] = _chart([(x["date"], x["tvl"]) for x in c])
    st = _read(token, "c_stables")
    if st:
        out["chain_stables"] = _chart(
            [(x["date"], (x.get("totalCirculatingUSD") or {}).get("peggedUSD", 0)) for x in st]
        )
    f = CM_DIR / f"{token}.json"
    if f.exists():
        rows = json.loads(f.read_text())
        if rows:
            out["market_cap"] = pd.Series(
                [float(r["CapMrktCurUSD"]) for r in rows],
                index=pd.to_datetime([r["time"] for r in rows]).tz_localize(None).normalize(),
            )
    return {k: v for k, v in out.items() if len(v)}


def main() -> None:
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
