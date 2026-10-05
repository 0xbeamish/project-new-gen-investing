"""Insider buying and selling (Form 4), from the SEC's free quarterly insider data sets
(ported from the pilot).

    uv run python -m engine.markets.us_smallcap.insiders   # .cache/insiders/<quarter>.csv, 2009Q1 on

Each quarterly zip (sec.gov/dera/data/form-345) holds every Form 3/4/5. We keep only open-market
trades: code P (purchase) and S (sale), by issuers in our universe. Option exercises, grants and tax
withholding are routine pay, not a view on the stock, so they're dropped.

Inputs per (entity, as_of), over the 182 days of filings before as_of (filing date = public date):
  ins_buyers          distinct insiders who bought on the open market
  ins_sellers         distinct insiders who sold
  ins_officer_buyers  distinct officers (not just directors) who bought
  ins_net_usd_share   (bought $ - sold $) / (bought $ + sold $); 0 when nobody traded
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from engine.markets.us_smallcap import sec

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
CACHE = Path(".cache") / "insiders"
OUT = Path(".cache") / "insiders.csv"
FIRST_QUARTER = "2009Q1"
WINDOW_DAYS = 182


def _quarter_trades(q: str) -> pd.DataFrame:
    """Open-market trades in one quarterly file, cached as a small CSV (the zip is discarded)."""
    path = CACHE / f"{q}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"cik": int})
    sec.throttle()
    r = requests.get(URL.format(q=q.lower()), headers=sec._headers(), timeout=300)
    if r.status_code == 404:  # not published yet
        return pd.DataFrame()
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    read = lambda name, cols: pd.read_csv(
        z.open(name), sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip"
    )
    subs = read(
        "SUBMISSION.tsv",
        ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"],
    )
    subs = subs[subs["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    trans = read(
        "NONDERIV_TRANS.tsv",
        ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"],
    )
    trans = trans[trans["TRANS_CODE"].isin(["P", "S"])]
    owners = read(
        "REPORTINGOWNER.tsv",
        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"],
    ).drop_duplicates("ACCESSION_NUMBER")  # joint filers: count the filing once
    df = trans.merge(subs, on="ACCESSION_NUMBER").merge(owners, on="ACCESSION_NUMBER")
    out = pd.DataFrame(
        {
            "cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
            "filed": pd.to_datetime(
                df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"
            ),
            "owner": df["RPTOWNERCIK"],
            "officer": df["RPTOWNER_RELATIONSHIP"].fillna("").str.contains("Officer"),
            "buy": df["TRANS_CODE"] == "P",
            "usd": pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
            * pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        }
    ).dropna(subset=["cik", "filed"])
    out["cik"] = out["cik"].astype(int)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def download(ciks: set[int] | None = None, out: Path = OUT) -> pd.DataFrame:
    quarters = pd.period_range(
        FIRST_QUARTER, pd.Timestamp.now().to_period("Q"), freq="Q"
    )
    frames = []
    for q in quarters:
        t = _quarter_trades(str(q))
        if t.empty:
            continue
        frames.append(t[t["cik"].isin(ciks)] if ciks else t)
        print(f"{q}: {len(t):,} open-market trades", file=sys.stderr)
    trades = pd.concat(frames, ignore_index=True)
    trades["filed"] = pd.to_datetime(trades["filed"])  # cached files come back as text
    trades.to_csv(out, index=False)
    return trades


def load() -> pd.DataFrame:
    return pd.read_csv(OUT, parse_dates=["filed"]) if OUT.exists() else pd.DataFrame()


def insider_features(rows: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """rows: entity, cik, as_of. Returns entity, as_of + the ins_* inputs (point-in-time)."""
    by_cik = dict(tuple(trades.groupby("cik"))) if len(trades) else {}
    out = []
    for entity, cik, as_of in rows[["entity", "cik", "as_of"]].itertuples(index=False):
        t = by_cik.get(int(cik))
        if t is not None:
            t = t[
                (t["filed"] < as_of)
                & (t["filed"] >= as_of - pd.Timedelta(days=WINDOW_DAYS))
            ]
        if t is None or t.empty:
            out.append((entity, as_of, 0, 0, 0, 0.0))
            continue
        b, s = t[t["buy"]], t[~t["buy"]]
        bought, sold = b["usd"].sum(), s["usd"].sum()
        out.append(
            (
                entity,
                as_of,
                b["owner"].nunique(),
                s["owner"].nunique(),
                b.loc[b["officer"], "owner"].nunique(),
                (bought - sold) / (bought + sold) if bought + sold > 0 else 0.0,
            )
        )
    return pd.DataFrame(
        out,
        columns=[
            "entity",
            "as_of",
            "ins_buyers",
            "ins_sellers",
            "ins_officer_buyers",
            "ins_net_usd_share",
        ],
    ).astype({"ins_net_usd_share": float})


def main() -> None:
    from engine.markets.us_smallcap.env import load_env

    load_env()
    trades = download()
    print(
        f"{len(trades):,} open-market insider trades -> {CACHE}/<quarter>.csv and {OUT}"
    )


if __name__ == "__main__":
    main()
