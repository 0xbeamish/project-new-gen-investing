"""The point-in-time universe lists (small/mid, large, micro), including companies later delisted
or acquired.

    uv run python -m engine.markets.us_stocks.universe build     # downloads; EODHD_API_KEY, SEC_USER_AGENT
    uv run python -m engine.markets.us_stocks.universe rebuild   # from the cached map and prices
    uv run python -m engine.markets.us_stocks.universe large --until 2019-12-31
    uv run python -m engine.markets.us_stocks.universe micro

Who existed and how big, each quarter:
  shares outstanding  SEC XBRL frames (dei:EntityCommonStockSharesOutstanding, cover pages), from the
                      PREVIOUS calendar quarter so every value was public
  price, volume       EODHD daily bars (raw close for market value; adjusted close for returns)
SEC ID -> ticker:     current tickers from the SEC's list; delisted ones by EXACT company name against
                      EODHD's delisted listings. Warrants, units, rights and blank-check shells dropped.
Small/mid at each quarter end: drop the 500 largest, keep market value >= $300M, price >= $5 and
>= $1M traded a day (63-day average). Membership is re-set every quarter.
Large (`large --until DATE`, for us_largecap): the 500 largest US filers, same price and volume
floors and SPAC filter; lists dated after --until are never written. The top of a size ranking is
where data errors land: XBRL share counts off by 1,000x, and vendor raw closes 10-20x the traded
price for years, or back-adjusted for a LATER reverse split (which would pull future collapses into
the list). So the large list ranks a checked market value, every step point-in-time:
  shares     the median of the company's last 5 quarterly share counts (unit errors are one-offs)
  foreign    filers whose latest annual report before the list date is a 20-F / 40-F, or with no
             annual report yet but a 6-K / F-1 filed, are dropped (ADRs and foreign listings: not
             US common stock, no 10-Q data, share counts in ordinary shares)
  warrants   codes like C-WS-A dropped
  check      A = raw close x shares; B = the last public float reported before the list date
             (dei:EntityPublicFloat, 10-K cover, USD) carried to the list date by the stock's
             adjusted return since its measurement date (immune to raw-close errors). Use A when
             0.5 <= A/B <= 4 (a real company sits at 1 / non-affiliate share); else B (a float-
             adjusted value, as index providers use). No usable B (a first-year filer): A, if it
             trades >= 0.1% of itself a day at a close <= $500. Anything above $1.5T (more than
             any US company before 2020) is dropped"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from engine.markets.us_stocks import sec
from engine.markets.us_stocks.prices import (
    OTC_EXCHANGES,
    fetch_ohlc,
    fetch_prices,
    load_prices,
    quarter_snapshot,
    symbols,
)
from engine.run import ROOT

OUT = ROOT / "data" / "smallcap_universe.csv"
MAP_OUT = ROOT / "data" / "smallcap_tickers.csv"
BULK_SUBMISSIONS = Path(".cache") / "bulk" / "submissions.zip"
NAMES_CACHE = Path(".cache") / "sec_names_and_delistings.csv"
OTC_GRACE_DAYS = 30  # trading after the delisting-notice 8-K is assumed to be off-exchange
FIRST_Q = "2010Q1"
TOP_EXCLUDED = 500
MIN_MCAP, MIN_PRICE, MIN_DOLLAR_VOL = 300e6, 5.0, 1e6
BLANK_CHECK_SIC = 6770  # SPACs: cash shells, not operating companies
SNAPSHOTS = Path(".cache") / "priced_snapshots.pkl"
MICRO_OUT = ROOT / "data" / "microcap_universe.csv"
LARGE_OUT = ROOT / "data" / "largecap_universe.csv"
MICRO_MIN_MCAP, MICRO_MIN_PRICE, MICRO_MIN_DOLLAR_VOL = 30e6, 1.0, 1e5


def _norm(name: str) -> str:
    name = name.upper().replace("\xa0", " ").replace("&", " AND ")
    name = re.sub(r"/[A-Z]{2,3}/", " ", name)  # EDGAR state tags like "AETNA INC /PA/"
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(
        r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|HOLDINGS|GROUP|THE|NEW|DE)\b",
        " ",
        name,
    )
    return " ".join(name.split())


def read_map() -> pd.DataFrame:
    """The cik -> code map. Codes stay text: a ticker literally named "NA" isn't missing."""
    return pd.read_csv(
        MAP_OUT,
        dtype={"code": str},
        keep_default_na=False,
        na_values={"listed_until": [""]},
        parse_dates=["listed_until"],
    )


def sec_names_and_delistings(ciks: set[int]) -> pd.DataFrame:
    """Per filer: every name it has filed under (current + former), and its last 8-K item 3.01
    (notice of delisting) date. From the SEC bulk submissions zip; cached."""
    import zipfile

    if NAMES_CACHE.exists():
        return pd.read_csv(NAMES_CACHE, parse_dates=["last_delisting_8k"])
    rows = []
    with zipfile.ZipFile(BULK_SUBMISSIONS) as z:
        for name in z.namelist():
            if not (name.startswith("CIK") and name[3:13].isdigit()) or "-submissions-" in name:
                continue
            cik = int(name[3:13])
            if cik not in ciks:
                continue
            d = json.loads(z.read(name))
            names = [d.get("name")] + [f.get("name") for f in d.get("formerNames", [])]
            r = d.get("filings", {}).get("recent", {})
            dates = [
                f
                for f, items, form in zip(
                    r.get("filingDate", []), r.get("items", []), r.get("form", [])
                )
                if form == "8-K" and "3.01" in (items or "")
            ]
            rows += [
                {
                    "cik": cik,
                    "name": n,
                    "last_delisting_8k": max(dates) if dates else None,
                }
                for n in names
                if n
            ]
    out = pd.DataFrame(rows)
    out["last_delisting_8k"] = pd.to_datetime(out["last_delisting_8k"])
    out.to_csv(NAMES_CACHE, index=False)
    return out


def shares_frames(last_q: str) -> pd.DataFrame:
    """Shares outstanding per filer per calendar quarter: cik, name, quarter, shares."""
    rows = []
    for q in pd.period_range(FIRST_Q, last_q, freq="Q"):
        tag = f"CY{q.year}Q{q.quarter}I"
        try:
            d = sec._get_json(
                f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{tag}.json",
                f"frames_shares_{q}.json",
            )
        except requests.HTTPError:  # the current quarter isn't published yet
            continue
        f = pd.DataFrame(d["data"])
        rows.append(f.assign(quarter=str(q))[["cik", "entityName", "quarter", "val"]])
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"entityName": "name", "val": "shares"})


def ticker_map(filers: pd.DataFrame, syms: pd.DataFrame) -> pd.DataFrame:
    """cik -> candidate EODHD codes, with `listed_until` for codes whose final venue was OTC.

    Active: SEC's ticker list. Delisted: exact company-name match, against every name the filer
    has used (the frames carry only the CURRENT name: Aqua America files as Essential Utilities).
    An OTC-ending code counts only up to OTC_GRACE_DAYS after the company's last delisting-notice
    8-K (item 3.01); with no such 8-K there is no evidence it was ever exchange-listed, so it's dropped.
    """
    current = sec.ticker_to_cik()
    active = syms[syms["active"]]
    rows = [
        {"cik": current[c], "code": c, "how": "sec_ticker"} for c in active["code"] if c in current
    ]
    history = sec_names_and_delistings(set(filers["cik"].astype(int)))
    names = pd.concat([filers[["cik", "name"]], history[["cik", "name"]]]).drop_duplicates(
        ["cik", "name"]
    )
    names = names.assign(norm=lambda d: d["name"].map(_norm))
    by_norm = names.groupby("norm")["cik"].unique()
    unique_norm = {n: int(c[0]) for n, c in by_norm.items() if len(c) == 1 and n}
    for code, name in syms.loc[~syms["active"], ["code", "name"]].itertuples(index=False):
        n = _norm(name)
        if n in unique_norm:  # exact only: close matches paired MTC with FMC Technologies
            rows.append({"cik": unique_norm[n], "code": code, "how": "name_exact"})
    out = pd.DataFrame(rows).drop_duplicates(["cik", "code"])
    venue = syms.drop_duplicates("code").set_index("code")["final_exchange"]
    otc = out["code"].map(venue).isin(OTC_EXCHANGES) & (out["how"] != "sec_ticker")
    last_301 = history.drop_duplicates("cik").set_index("cik")["last_delisting_8k"]
    out["listed_until"] = pd.NaT
    out.loc[otc, "listed_until"] = out.loc[otc, "cik"].map(last_301) + pd.Timedelta(
        days=OTC_GRACE_DAYS
    )
    return out[~(otc & out["listed_until"].isna())].reset_index(drop=True)


def build_universe(
    shares: pd.DataFrame, tmap: pd.DataFrame, keep_all: bool = False
) -> pd.DataFrame:
    """One row per (quarter end, company) in the small/mid universe for the NEXT quarter."""
    codes = tmap.groupby("cik")["code"].apply(list)
    until = (
        tmap.dropna(subset=["listed_until"])
        .set_index("code")["listed_until"]
        .pipe(pd.to_datetime)
        .to_dict()
        if "listed_until" in tmap
        else {}
    )
    px_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for q in sorted(shares["quarter"].unique()):
        qend = (
            pd.Period(q, "Q") + 1
        ).end_time.normalize()  # shares from q are public by end of q+1
        if qend > pd.Timestamp.now():
            continue
        recent = shares[  # latest of the last 2 frames: a quarter's frame misses many filers
            shares["quarter"].isin([q, str(pd.Period(q, "Q") - 1)])
        ].sort_values("quarter")
        recent = recent.drop_duplicates("cik", keep="last")
        for cik, name, sh in recent[["cik", "name", "shares"]].itertuples(index=False):
            best = None
            for code in codes.get(cik, []):
                if code in until and qend > until[code]:  # off-exchange by then
                    continue
                if code not in px_cache:
                    px_cache[code] = load_prices(code)
                snap = quarter_snapshot(px_cache[code], qend)
                if snap and (best is None or snap[1] > best[2]):
                    best = (
                        code,
                        snap[0],
                        snap[1],
                    )  # several codes trading: the most liquid
            if best is None or not sh or sh <= 0:
                continue
            code, close, dvol = best
            rows.append(
                {
                    "as_of": qend,
                    "cik": int(cik),
                    "code": code,
                    "name": name,
                    "close": close,
                    "mcap": close * sh,
                    "dollar_vol": dvol,
                }
            )
        print(
            f"{qend.date()}: {sum(r['as_of'] == qend for r in rows):,} priced companies",
            file=sys.stderr,
        )
    df = pd.DataFrame(rows)
    df["mcap_rank"] = df.groupby("as_of")["mcap"].rank(ascending=False)
    return df if keep_all else small_mid(df)


def small_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 500 largest; keep >= $300M market value, price >= $5, >= $1M traded a day."""
    keep = (
        (df["mcap_rank"] > TOP_EXCLUDED)
        & (df["mcap"] >= MIN_MCAP)
        & (df["close"] >= MIN_PRICE)
        & (df["dollar_vol"] >= MIN_DOLLAR_VOL)
    )
    return df[keep].reset_index(drop=True)


LARGE_N = 500
LARGE_MAX_VALUE = 1.5e12
LARGE_CONSISTENT = (0.5, 4.0)  # A / B band in which the raw-close value is trusted
# no float to check against: daily dollar volume / value floor, and a close above which a
# first-year filer's quote is a vendor scaling error
LARGE_MIN_TURNOVER = 1e-3
LARGE_MAX_UNCHECKED_CLOSE = 500.0
SHARES_MEDIAN_QUARTERS = 5
FOREIGN_FORMS = {"20-F", "40-F"}
FOREIGN_ONLY_FORMS = {"6-K", "F-1", "F-3", "F-4"}  # foreign private issuers only
WARRANT_CODE = r"-(?:WS|WT|W|U|UN|R|RT)(?:-[A-Z])?$"


def annual_forms(cik: int) -> pd.DataFrame:
    """Every annual report (10-K / 20-F / 40-F) and foreign-issuer form the filer filed: form, filed."""
    d = sec._get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions_{cik}.json",
    )
    blocks = [d["filings"]["recent"]] + [
        sec._get_json(f"https://data.sec.gov/submissions/{f['name']}", f["name"])
        for f in d["filings"].get("files", [])
    ]
    rows = [
        (f, pd.Timestamp(day))
        for b in blocks
        for f, day in zip(b["form"], b["filingDate"])
        if f in {"10-K", "10-K405"} | FOREIGN_FORMS | FOREIGN_ONLY_FORMS
    ]
    return pd.DataFrame(rows, columns=["form", "filed"])


def public_floats(cik: int) -> pd.DataFrame:
    """dei:EntityPublicFloat as reported: measured (end), filed, val (USD)."""
    try:
        d = sec._get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            f"facts_{cik}.json",
        )
    except Exception:  # noqa: BLE001 -- no XBRL facts: no float check
        d = {}
    f = d.get("facts", {}).get("dei", {}).get("EntityPublicFloat")
    rows = [
        (pd.Timestamp(r["end"]), pd.Timestamp(r["filed"]), float(r["val"]))
        for u in (f or {}).get("units", {}).values()
        for r in u
    ]
    return pd.DataFrame(rows, columns=["end", "filed", "val"])


def _latest_before(t: pd.DataFrame, when: pd.Timestamp):
    t = t[t["filed"] <= when]
    return t.sort_values("filed", kind="stable").iloc[-1] if len(t) else None


def is_foreign(forms: pd.DataFrame, when: pd.Timestamp) -> bool:
    """A 20-F / 40-F filer at `when`, or a foreign-issuer form with no annual report yet."""
    f = forms[forms["filed"] <= when]
    annual = f[~f["form"].isin(FOREIGN_ONLY_FORMS)]
    if len(annual):
        return _latest_before(annual, when)["form"] in FOREIGN_FORMS
    return bool(len(f))  # no annual report yet, but a foreign-issuer form


def checked_value(a: float, b: float, turnover: float, close: float) -> tuple[float, str]:
    """(market value to rank on, which one) from A = close x shares and B = carried float."""
    b_ok = not np.isnan(b) and 0 < b <= LARGE_MAX_VALUE
    if b_ok and a <= LARGE_MAX_VALUE and LARGE_CONSISTENT[0] <= a / b <= LARGE_CONSISTENT[1]:
        return a, "close_x_shares"
    if b_ok:
        return b, "float"
    if (
        a <= LARGE_MAX_VALUE
        and turnover >= LARGE_MIN_TURNOVER
        and close <= LARGE_MAX_UNCHECKED_CLOSE
    ):
        return a, "close_x_shares_unchecked"
    return float("nan"), "dropped"


def large(df: pd.DataFrame, n: int = LARGE_N) -> pd.DataFrame:
    """The n largest by checked market value (see the module docstring), with small_mid's price
    and volume floors. df: snapshots (as_of, cik, code, close, mcap, dollar_vol)."""
    df = df.sort_values(["cik", "as_of"]).copy()
    shares = df["mcap"] / df["close"]
    df["shares_med"] = shares.groupby(df["cik"]).transform(
        lambda x: x.rolling(SHARES_MEDIAN_QUARTERS, min_periods=1).median()
    )
    df["mcap_raw"] = df["mcap"]
    df["mcap_a"] = df["close"] * df["shares_med"]
    df = df[~df["code"].str.contains(WARRANT_CODE, regex=True)]
    # only the top of each quarter can reach the list: check those (by A or by raw value)
    top = (df.groupby("as_of")["mcap_a"].rank(ascending=False) <= 3 * n) | (
        df["mcap_rank"] <= 3 * n
    )
    cand = df[top].copy()
    forms = {c: annual_forms(int(c)) for c in cand["cik"].unique()}
    floats = {c: public_floats(int(c)) for c in cand["cik"].unique()}
    foreign, carried = [], []
    for r in cand.itertuples(index=False):
        foreign.append(is_foreign(forms[r.cik], r.as_of))
        fl = _latest_before(floats[r.cik], r.as_of)
        b = float("nan")
        if fl is not None and fl["val"] > 0:
            adj = load_prices(r.code)["adj_close"]
            at_list, at_float = adj[adj.index <= r.as_of], adj[adj.index <= fl["end"]]
            if len(at_list) and len(at_float):
                b = fl["val"] * at_list.iloc[-1] / at_float.iloc[-1]
        carried.append(b)
    cand["foreign"], cand["float_carried"] = foreign, carried
    checked = [
        checked_value(a, b, dv / a if a > 0 else 0.0, c)
        for a, b, dv, c in zip(
            cand["mcap_a"], cand["float_carried"], cand["dollar_vol"], cand["close"]
        )
    ]
    cand["mcap"] = [v for v, _ in checked]
    cand["value_source"] = [w for _, w in checked]
    kept = cand[~cand["foreign"] & cand["mcap"].notna()].copy()
    kept["mcap_rank"] = kept.groupby("as_of")["mcap"].rank(ascending=False)
    keep = (
        (kept["mcap_rank"] <= n)
        & (kept["close"] >= MIN_PRICE)
        & (kept["dollar_vol"] >= MIN_DOLLAR_VOL)
    )
    out = kept[keep].sort_values(["as_of", "mcap_rank"]).reset_index(drop=True)
    print(
        f"large: checked {len(cand):,} rows; dropped {int(cand['foreign'].sum()):,} foreign-filer rows, "
        f"{int((cand['value_source'] == 'dropped').sum()):,} implausible; listed rows by value source "
        f"{out['value_source'].value_counts().to_dict()}",
        file=sys.stderr,
    )
    return out


def build_large(until: str | None) -> None:
    """Large-cap list from the cached snapshots (run `build` first); nothing dated after `until`."""
    every = pd.read_pickle(SNAPSHOTS)
    if until:
        every = every[every["as_of"] <= pd.Timestamp(until)]
    uni = with_sectors(large(every))
    uni.to_csv(LARGE_OUT, index=False)
    per_q = uni.groupby("as_of").size()
    print(
        f"large: {len(uni):,} rows, {uni['cik'].nunique():,} companies; per quarter {per_q.min()}-{per_q.max()} -> {LARGE_OUT}"
    )


def micro(df: pd.DataFrame) -> pd.DataFrame:
    """Below where most funds can trade: $30M-$300M market value, price >= $1, >= $100k a day."""
    keep = (
        (df["mcap"] >= MICRO_MIN_MCAP)
        & (df["mcap"] < MIN_MCAP)
        & (df["close"] >= MICRO_MIN_PRICE)
        & (df["dollar_vol"] >= MICRO_MIN_DOLLAR_VOL)
    )
    return df[keep].reset_index(drop=True)


def with_sectors(uni: pd.DataFrame) -> pd.DataFrame:
    """Adds SIC, sector and the cyclical flag; drops blank-check shells (SPACs)."""
    sic = {c: sec.company_sic(int(c)) for c in uni["cik"].unique()}
    uni = uni.assign(sic=uni["cik"].map(sic))
    uni = uni[uni["sic"] != BLANK_CHECK_SIC]
    return uni.assign(
        sector=uni["sic"].map(sec.sic_to_sector),
        cyclical=uni["sic"].map(sec.is_cyclical),
    )


def build_micro() -> None:
    """Micro-cap list from cached prices and share counts (run `build` first for the cache)."""
    if SNAPSHOTS.exists():
        every = pd.read_pickle(SNAPSHOTS)
    else:
        shares = shares_frames(str(pd.Timestamp.now().to_period("Q")))
        every = build_universe(shares, read_map(), keep_all=True)
        every.to_pickle(SNAPSHOTS)
    uni = with_sectors(micro(every))
    uni.to_csv(MICRO_OUT, index=False)
    per_q = uni.groupby("as_of").size()
    print(
        f"micro universe: {len(uni):,} rows, {uni['cik'].nunique():,} companies; per quarter median {int(per_q.median())} -> {MICRO_OUT}"
    )


def rebuild() -> None:
    """Small/mid and micro lists from the cached map and prices (no downloads)."""
    shares = shares_frames(str(pd.Timestamp.now().to_period("Q")))
    every = build_universe(shares, read_map(), keep_all=True)
    every.to_pickle(SNAPSHOTS)
    for name, uni, out in (
        ("small/mid", small_mid(every), OUT),
        ("micro", micro(every), MICRO_OUT),
    ):
        uni = with_sectors(uni)
        uni.to_csv(out, index=False)
        per_q = uni.groupby("as_of").size()
        print(
            f"{name}: {len(uni):,} rows, {uni['cik'].nunique():,} companies; median {int(per_q.median())}/quarter -> {out}"
        )


def main() -> None:
    """build | rebuild | large --until DATE | micro."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "micro", "large", "rebuild"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--until", help="large: last list date written")
    args = ap.parse_args()
    from engine.cli import load_env

    load_env()
    if args.cmd == "large":
        build_large(args.until)
        return
    if args.cmd == "micro":
        build_micro()
        return
    if args.cmd == "rebuild":  # from the cached map and prices, no downloads (cleaned prices)
        rebuild()
        return
    syms = symbols()
    shares = shares_frames(str(pd.Timestamp.now().to_period("Q")))
    print(
        f"{len(syms):,} listed common stocks on EODHD; {shares['cik'].nunique():,} SEC filers with shares",
        file=sys.stderr,
    )
    tmap = ticker_map(shares, syms)
    tmap.to_csv(MAP_OUT, index=False)
    print(
        f"ticker map: {tmap['cik'].nunique():,} filers -> {len(tmap):,} codes; {tmap['how'].value_counts().to_dict()}",
        file=sys.stderr,
    )
    todo = sorted(set(tmap["code"]))
    for fetch in (
        fetch_prices,
        fetch_ohlc,
    ):  # close-only first, then OHLC (spreads, cleaning)
        with ThreadPoolExecutor(args.workers) as pool:
            for i, _status in enumerate(pool.map(fetch, todo), 1):
                if i % 1000 == 0:
                    print(f"{fetch.__name__} {i:,}/{len(todo):,}", file=sys.stderr)
    every = build_universe(shares, tmap, keep_all=True)
    every.to_pickle(SNAPSHOTS)  # every priced filer, before size filters (micro caps reuse it)
    uni = small_mid(every)
    sic = {c: sec.company_sic(int(c)) for c in uni["cik"].unique()}
    uni["sic"] = uni["cik"].map(sic)
    uni = uni[uni["sic"] != BLANK_CHECK_SIC]
    uni["sector"] = uni["sic"].map(sec.sic_to_sector)
    uni["cyclical"] = uni["sic"].map(sec.is_cyclical)
    uni.to_csv(OUT, index=False)
    per_q = uni.groupby("as_of").size()
    print(
        f"universe: {len(uni):,} rows, {uni['cik'].nunique():,} companies; per quarter {per_q.min()}-{per_q.max()} (median {int(per_q.median())}) -> {OUT}"
    )


if __name__ == "__main__":
    main()
