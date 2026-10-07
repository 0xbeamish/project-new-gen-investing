"""Free daily series for chain tokens (sectors L1, L2, Payments & privacy), beyond DeFiLlama:

  coinmetrics  Community API: every free daily metric per asset (active and funded addresses,
               transactions, fees, issuance, supply, market cap, MVRV, exchange flows, hash rate)
  growthepie   Ethereum and its L2s: active addresses, transactions, fees, profit, data cost,
               stablecoins, market cap, FDV, TVL, throughput, app revenue
  l2beat       rollups: activity (transactions, user operations) and total value secured
  binance      USD-margined perp metrics: open interest and long/short ratios (daily files)
  wikipedia    English Wikipedia page views of the chain's article (titles in WIKI_TITLES)

Everything is fetched only through FETCH_END (2025+ stays unseen) and cached under .cache/crypto/.
    python -m engine.markets.crypto.chains fetch [coinmetrics growthepie l2beat binance wikipedia]
    python -m engine.markets.crypto.chains fetch binance --shard 0/4   # one of 4 parallel processes
    python -m engine.markets.crypto.chains fetch binance coinmetrics --all-tokens   # every sector
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from engine.markets.crypto import binance, indicators

FETCH_START, FETCH_END = "2020-01-01", "2024-12-31"
CHAIN_SECTORS = ("L1", "L2", "Payments & privacy")
CM = "https://community-api.coinmetrics.io/v4"
GTP = "https://api.growthepie.com/v1"
L2B = "https://l2beat.com/api/scaling"
WIKI = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
)
UA = {"User-Agent": "personal-crypto-research/0.1 (non-commercial) python-requests"}
DIR = binance.CACHE
SKIP_CM = ("ReferenceRate", "PriceBTC", "AssetCompletionTime", "AssetEODCompletionTime", "ROI")
GROWTHEPIE = {
    "ETH": "ethereum",
    "ARB": "arbitrum",
    "OP": "optimism",
    "MATIC": "polygon_pos",
    "CELO": "celo",
    "LSK": "lisk",
}
L2BEAT = {
    "ARB": "arbitrum",
    "OP": "op-mainnet",
    "IMX": "immutablex",
    "LRC": "loopring",
    "CELO": "celo",
    "LSK": "lisk",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def chain_tokens() -> list[str]:
    s = indicators.sectors()
    return s.loc[s["sector"].isin(CHAIN_SECTORS), "token"].tolist()


def _get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code in (200, 400, 403, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))  # rate limited (429) or a server error: back off
    return None


# ---------- coinmetrics ----------
def fetch_coinmetrics(tokens) -> None:
    cat = _get(f"{CM}/catalog-v2/asset-metrics?page_size=10000").json()["data"]
    free = {
        a["asset"]: [
            m["metric"]
            for m in a["metrics"]
            if not m["metric"].startswith(SKIP_CM)
            and any(f.get("community") and f["frequency"] == "1d" for f in m["frequencies"])
        ]
        for a in cat
    }
    out = DIR / "coinmetrics" / "onchain"
    out.mkdir(parents=True, exist_ok=True)

    def one(t):
        metrics = free.get(t.lower())
        f = out / f"{t}.json"
        if not metrics or f.exists():
            return
        rows = []
        for i in range(0, len(metrics), 10):  # a few metrics per request
            url = (
                f"{CM}/timeseries/asset-metrics?assets={t.lower()}&metrics={','.join(metrics[i : i + 10])}"
                f"&frequency=1d&start_time={FETCH_START}&end_time={FETCH_END}&page_size=10000"
                "&ignore_unsupported_errors=true"
            )
            while url:
                r = _get(url)
                d = r.json() if r is not None and r.ok else {}
                rows += d.get("data", [])
                url = d.get("next_page_url")
        f.write_text(json.dumps(rows))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, tokens))
    _log(f"  coinmetrics: {len(list(out.glob('*.json')))} assets")


def coinmetrics(token: str) -> dict[str, pd.Series]:
    f = DIR / "coinmetrics" / "onchain" / f"{token}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d.pop("time")).dt.tz_localize(None).dt.normalize()
    d = d.drop(columns="asset").apply(pd.to_numeric, errors="coerce").groupby(level=0).last()
    return {f"cm_{c}": d[c].dropna() for c in d.columns if d[c].notna().any()}


# ---------- growthepie ----------
def fetch_growthepie() -> None:
    out = DIR / "growthepie"
    out.mkdir(parents=True, exist_ok=True)
    metrics = list(_get(f"{GTP}/master.json").json()["metrics"])
    for m in metrics:
        r = _get(f"{GTP}/export/{m}.json")
        if r is not None and r.ok:
            rows = [x for x in r.json() if x["date"] <= FETCH_END]
            (out / f"{m}.json").write_text(json.dumps(rows))
    _log(f"  growthepie: {len(metrics)} metrics")


def growthepie(token: str) -> dict[str, pd.Series]:
    key = GROWTHEPIE.get(token)
    out = {}
    for f in sorted((DIR / "growthepie").glob("*.json")) if key else []:
        rows = [x for x in json.loads(f.read_text()) if x["origin_key"] == key]
        if rows:
            d = pd.DataFrame(rows)
            for (metric,), g in d.groupby(["metric_key"]):
                s = pd.Series(g["value"].astype(float).to_numpy(), index=pd.to_datetime(g["date"]))
                out[f"gtp_{metric}"] = s.groupby(level=0).last().sort_index()
    return out


# ---------- l2beat ----------
def fetch_l2beat() -> None:
    out = DIR / "l2beat"
    out.mkdir(parents=True, exist_ok=True)
    for token, slug in L2BEAT.items():
        for kind in ("activity", "tvs"):
            r = _get(f"{L2B}/{kind}/{slug}?range=max")
            if r is not None and r.ok and r.json().get("success"):
                (out / f"{token}_{kind}.json").write_text(r.text)
    _log(f"  l2beat: {len(list(out.glob('*.json')))} series")


def l2beat(token: str) -> dict[str, pd.Series]:
    out = {}
    for kind in ("activity", "tvs"):
        f = DIR / "l2beat" / f"{token}_{kind}.json"
        if not f.exists():
            continue
        chart = json.loads(f.read_text())["data"]["chart"]
        d = pd.DataFrame(chart["data"], columns=chart["types"])
        d.index = pd.to_datetime(d.pop("timestamp"), unit="s").dt.normalize()
        d = d[d.index <= FETCH_END]
        for c in d.columns:
            out[f"l2b_{kind}_{c}"] = pd.to_numeric(d[c], errors="coerce").groupby(level=0).last()
    return out


# ---------- binance perp metrics ----------
def _perp(token: str, perps: set[str]) -> str | None:
    for p in (f"{token}USDT", f"1000{token}USDT"):
        if p in perps:
            return p
    return None


def _metrics_day(key: str) -> pd.DataFrame | None:
    r = _get(f"{binance.BASE}/{key}")
    if r is None or not r.ok:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        d = pd.read_csv(io.StringIO(z.read(z.namelist()[0]).decode()))
    if d.empty:
        return None
    d["create_time"] = pd.to_datetime(d["create_time"])
    day = d["create_time"].dt.normalize().iloc[0]
    num = d.drop(columns=["create_time", "symbol"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    last = num.iloc[-1]  # end-of-day open interest, mean of the ratios
    row = {
        "date": day,
        "oi": last.get("sum_open_interest"),
        "oi_usd": last.get("sum_open_interest_value"),
    }
    for c in (
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ):
        if c in num:
            row[c] = num[c].mean()
    return pd.DataFrame([row])


def fetch_binance_metrics(tokens, workers: int = 16, shard: tuple[int, int] = (0, 1)) -> None:
    """Daily files per perp; shard (i, n) takes every n-th perp from the i-th, so n processes can
    split the list."""
    perps = binance.perp_symbols()
    out = DIR / "perp_metrics"
    out.mkdir(parents=True, exist_ok=True)
    syms = sorted({p for t in tokens if (p := _perp(t, perps))})[shard[0] :: shard[1]]
    _log(f"  binance metrics: {len(syms)} perps")
    for n, sym in enumerate(syms, 1):
        f = out / f"{sym}.pkl"
        if f.exists():
            continue
        keys = [
            k
            for k in binance.list_keys(f"data/futures/um/daily/metrics/{sym}/", folders=False)
            if k.endswith(".zip") and FETCH_START <= k[-14:-4] <= FETCH_END
        ]
        with ThreadPoolExecutor(workers) as pool:
            parts = [p for p in pool.map(_metrics_day, keys) if p is not None]
        frame = pd.concat(parts, ignore_index=True).sort_values("date") if parts else pd.DataFrame()
        frame.to_pickle(f)
        _log(f"  binance metrics: {n}/{len(syms)} {sym} {len(frame)} days")


def binance_metrics(token: str) -> dict[str, pd.Series]:
    for p in (f"{token}USDT", f"1000{token}USDT"):
        f = DIR / "perp_metrics" / f"{p}.pkl"
        if f.exists():
            d = pd.read_pickle(f)
            if d.empty:
                return {}
            d = d.set_index("date").sort_index()
            return {f"perp_{c}": d[c].dropna() for c in d.columns if d[c].notna().any()}
    return {}


# ---------- wikipedia ----------
def wiki_titles() -> dict[str, str]:
    f = indicators.SECTORS.with_name("crypto_wikipedia.csv")
    if not f.exists():
        return {}
    d = pd.read_csv(f, keep_default_na=False)
    return dict(zip(d["token"], d["title"]))


def fetch_wikipedia() -> None:
    out = DIR / "wikipedia"
    out.mkdir(parents=True, exist_ok=True)
    for token, title in wiki_titles().items():
        if not title:
            continue
        a, b = FETCH_START.replace("-", ""), FETCH_END.replace("-", "")
        if (out / f"{token}.json").exists():
            continue
        r = _get(f"{WIKI}/{title.replace(' ', '_')}/daily/{a}00/{b}00", headers=UA)
        if r is not None and r.ok:
            (out / f"{token}.json").write_text(r.text)
        time.sleep(1)  # Wikimedia asks for gentle clients
    _log(f"  wikipedia: {len(list(out.glob('*.json')))} articles")


def wikipedia(token: str) -> dict[str, pd.Series]:
    f = DIR / "wikipedia" / f"{token}.json"
    if not f.exists():
        return {}
    items = json.loads(f.read_text()).get("items", [])
    s = pd.Series(
        [float(x["views"]) for x in items],
        index=pd.to_datetime([x["timestamp"][:8] for x in items]),
    )
    return {"wiki_views": s} if len(s) else {}


def series(token: str) -> dict[str, pd.Series]:
    """Every chain series cached for this token."""
    out = {}
    for f in (coinmetrics, growthepie, l2beat, binance_metrics, wikipedia):
        out |= f(token)
    return {k: v for k, v in out.items() if len(v)}


def fetch(
    which=("coinmetrics", "growthepie", "l2beat", "wikipedia", "binance"),
    shard=(0, 1),
    every: bool = False,
) -> None:
    tokens = indicators.sectors()["token"].tolist() if every else chain_tokens()
    steps = {
        "coinmetrics": lambda: fetch_coinmetrics(tokens),
        "growthepie": fetch_growthepie,
        "l2beat": fetch_l2beat,
        "wikipedia": fetch_wikipedia,
        "binance": lambda: fetch_binance_metrics(tokens, shard=shard),
    }
    for w in which:
        _log(f"fetch {w}")
        steps[w]()


def main() -> None:
    if sys.argv[1:2] == ["fetch"]:
        args, shard = sys.argv[2:], (0, 1)
        if "--shard" in args:  # --shard i/n: this process's share of the perps
            i = args.index("--shard")
            shard = tuple(int(x) for x in args[i + 1].split("/"))
            args = args[:i] + args[i + 2 :]
        every = "--all-tokens" in args  # every sector, not only chains (perp metrics, market caps)
        args = [a for a in args if a != "--all-tokens"]
        fetch(
            tuple(args) or ("coinmetrics", "growthepie", "l2beat", "wikipedia", "binance"),
            shard,
            every,
        )
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
