"""SEC data for the us_stocks plug-in: XBRL company facts, 10-K and 10-Q fundamentals, 8-K items,
and insider trades. Everything is cached under .cache/ and needs SEC_USER_AGENT
("Your Name you@example.com") to fetch; the SEC rejects requests without a contact.

Fundamentals: each value is taken from the ORIGINAL filing that first reported it, never a later
restatement, and dated by that filing, so a backtest only sees what an investor could have known.

Insiders (Form 4, from the SEC's free quarterly data sets): only open-market trades, code P
(purchase) and S (sale); option exercises, grants and tax withholding are routine pay, not a view.
Inputs over the last `window_days` of filings: distinct buyers, sellers and officer buyers, and
(bought $ - sold $) / (bought $ + sold $), 0 when nobody traded.
"""

import io
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(os.environ.get("ENGINE_SEC_CACHE", ".cache/sec"))


# SEC requires "name email" in the User-Agent (403 without it) and allows 10 requests/sec.
def _headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise RuntimeError(
            'Set SEC_USER_AGENT="Your Name you@example.com" -- the SEC rejects requests without a contact email.'
        )
    return {"User-Agent": ua}


# Companies use different XBRL tags for the same concept (and switch tags over time); all are merged.
CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}


_RATE_LOCK = threading.Lock()
_last_request = [0.0]
MIN_INTERVAL = 1 / 8  # 8 requests/sec across all threads; the SEC allows 10


def throttle() -> None:
    """Block until this thread may send the next SEC request."""
    with _RATE_LOCK:
        wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def _get_json(url: str, cache_name: str) -> dict:
    path = CACHE / cache_name
    if path.exists():
        return json.loads(path.read_text())
    throttle()
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    return resp.json()


def ticker_to_cik() -> dict[str, int]:
    """Current tickers -> SEC company id (listed companies only)."""
    # Current listings only: delisted companies are missing (survivorship bias, see README).
    data = _get_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")
    return {row["ticker"]: row["cik_str"] for row in data.values()}


def annual_facts(cik: int, filed_until=None) -> pd.DataFrame:
    """One row per (fiscal_year_end, concept): the value as first reported in a 10-K.
    filed_until: ignore filings after this date (a market that must not read past its data end)."""
    data = _get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        f"facts_{cik}.json",
    )
    rows = []
    for concept, tags in CONCEPTS.items():
        for tag in tags:
            fact = data["facts"].get("us-gaap", {}).get(tag)
            if not fact:
                continue
            for unit_rows in fact["units"].values():
                for r in unit_rows:
                    if r.get("form") != "10-K":
                        continue
                    if "start" in r:  # flow item: keep full-year durations only
                        days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
                        if not 350 <= days <= 380:
                            continue
                    rows.append(
                        {
                            "concept": concept,
                            "end": r["end"],
                            "filed": r["filed"],
                            "val": r["val"],
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["concept", "end", "filed", "val"])
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if filed_until is not None:
        df = df[df["filed"] <= pd.Timestamp(filed_until)]
    # Earliest filing for each period = the original number, not a restatement.
    return df.sort_values("filed").drop_duplicates(["concept", "end"], keep="first")


# Quarterly concepts: the cycle shows up in revenue, margins and inventory a year before the 10-K.
QUARTERLY_CONCEPTS = {
    "revenue": CONCEPTS["revenue"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "inventory": ["InventoryNet"],
}
FLOW = {"revenue", "gross_profit", "cost_of_revenue", "operating_income"}
# opt-in flow items (quarterly_features(extended=True))
EXTRA_QUARTERLY = {"rnd": CONCEPTS["rnd"]}


def _first_filed(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df.sort_values("filed")


def quarterly_facts(cik: int, extra=(), filed_until=None) -> pd.DataFrame:
    """One row per (quarter_end, concept): the 3-month value as first reported, and when it was filed.

    Flow items: 3-month durations from 10-Qs and 10-Ks. Most companies never tag a 3-month Q4, so
    Q4 = full year minus the 9-month year-to-date, dated by the later of the two filings.
    Balance-sheet items (inventory): the value at quarter end.
    extra: names in EXTRA_QUARTERLY to add (flow items); filed_until: ignore later filings.
    """
    data = _get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        f"facts_{cik}.json",
    )
    rows = []
    concepts = QUARTERLY_CONCEPTS | {k: EXTRA_QUARTERLY[k] for k in extra}
    flow_set = FLOW | set(extra)
    for concept, tags in concepts.items():
        for tag in tags:
            fact = data["facts"].get("us-gaap", {}).get(tag)
            if not fact:
                continue
            for unit_rows in fact["units"].values():
                for r in unit_rows:
                    if r.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                        continue
                    rows.append(
                        {
                            "concept": concept,
                            "start": r.get("start"),
                            "end": r["end"],
                            "filed": r["filed"],
                            "val": r["val"],
                        }
                    )
    cols = ["concept", "end", "filed", "val"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = _first_filed(rows)
    if filed_until is not None:
        df = df[df["filed"] <= pd.Timestamp(filed_until)]
    stock = df[~df["concept"].isin(flow_set)].drop_duplicates(["concept", "end"])[cols]
    flow = df[df["concept"].isin(flow_set)].dropna(subset=["start"])
    flow = flow.assign(days=(flow["end"] - flow["start"]).dt.days)
    q = flow[flow["days"].between(80, 100)].drop_duplicates(["concept", "end"])
    fy = flow[flow["days"].between(350, 380)].drop_duplicates(["concept", "end"])
    ytd9 = flow[flow["days"].between(260, 285)].drop_duplicates(["concept", "start", "end"])
    derived = []
    for r in fy.itertuples(index=False):
        if ((q["concept"] == r.concept) & ((q["end"] - r.end).abs().dt.days <= 7)).any():
            continue  # a 3-month Q4 was tagged directly
        m = ytd9[(ytd9["concept"] == r.concept) & ((ytd9["start"] - r.start).abs().dt.days <= 7)]
        if len(m):
            m = m.iloc[0]
            derived.append(
                {
                    "concept": r.concept,
                    "end": r.end,
                    "filed": max(r.filed, m["filed"]),
                    "val": r.val - m["val"],
                }
            )
    out = pd.concat([q[cols], pd.DataFrame(derived, columns=cols), stock])
    return out.sort_values("filed").drop_duplicates(["concept", "end"]).reset_index(drop=True)


# Boom-and-bust industries: revenue and margins swing with an industry-wide price or capacity cycle,
# and the stock tends to turn before the reported numbers do. SIC ranges, inclusive.
CYCLICAL_SIC = [
    (1000, 1499),  # mining, oil and gas extraction
    (1520, 1531),  # homebuilders
    (2600, 2699),  # paper and packaging
    (2800, 2829),  # basic chemicals, plastics (not pharma 2830-2836 or soaps 2840-2844)
    (2860, 2899),  # industrial chemicals, fertilizers
    (2900, 2999),  # refining
    (3300, 3399),  # steel and metals
    (3531, 3533),  # construction, mining and oilfield machinery
    (3559, 3559),  # semiconductor equipment
    (3674, 3674),  # semiconductors (memory included)
    (3711, 3716),  # vehicles and parts
]


def company_sic(cik: int) -> int | None:
    """The company's current SIC code (not point-in-time)."""
    data = _get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions_{cik}.json",
    )
    return int(data["sic"]) if data.get("sic") else None


def is_cyclical(sic: int | None) -> bool:
    """True for boom-and-bust industries (CYCLICAL_SIC)."""
    return sic is not None and any(lo <= sic <= hi for lo, hi in CYCLICAL_SIC)


# SIC code ranges -> ~11 GICS-like sectors. First match wins, so narrow ranges come first.
# Approximate: SIC is an old government code, and some companies (e.g. Amazon = retail) land oddly.
SIC_SECTORS = [
    ((1300, 1399), "energy"),
    ((2900, 2999), "energy"),
    ((2830, 2836), "health_care"),
    ((3841, 3851), "health_care"),
    ((8000, 8099), "health_care"),
    ((3570, 3579), "tech"),
    ((3600, 3699), "tech"),
    ((3800, 3840), "tech"),
    ((7370, 7379), "tech"),
    ((3711, 3711), "consumer_discretionary"),
    ((4800, 4899), "communication"),
    ((7800, 7999), "communication"),
    ((4900, 4999), "utilities"),
    ((6500, 6599), "real_estate"),
    ((6798, 6798), "real_estate"),
    ((6000, 6799), "financials"),
    ((2000, 2199), "consumer_staples"),
    ((5400, 5499), "consumer_staples"),
    ((5912, 5912), "consumer_staples"),
    ((2840, 2844), "consumer_staples"),
    ((1000, 1499), "materials"),
    ((2400, 2899), "materials"),
    ((3000, 3399), "materials"),
    ((2200, 2399), "consumer_discretionary"),
    ((3900, 3999), "consumer_discretionary"),
    ((5200, 5999), "consumer_discretionary"),
    ((7000, 7299), "consumer_discretionary"),
    ((1500, 1799), "industrials"),
    ((3400, 3799), "industrials"),
    ((4000, 4799), "industrials"),
    ((5000, 5199), "industrials"),
    ((7300, 7399), "industrials"),
    ((8700, 8799), "industrials"),
]


def sic_to_sector(sic: int | None) -> str:
    """One of ~11 GICS-like sectors, or "other"."""
    for (lo, hi), sector in SIC_SECTORS:
        if sic is not None and lo <= sic <= hi:
            return sector
    return "other"


def localize(naive: pd.Series, tz: str) -> pd.Series:
    """Naive local clock times -> tz-aware. DST-ambiguous hours read as daylight time; only the
    next midnight is used, so the choice can't move a value across a decision."""
    return naive.dt.tz_localize(
        tz, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward"
    )


def eight_k_events(cik: int, entity: str) -> pd.DataFrame:
    """Every 8-K item the company filed, as events. Category = SEC item code (no LLM involved).

    Direction is 0 (unknown) until an LLM reads the text; the severity table then learns each
    item's average signed reaction (e.g. restatements are negative on average).
    """
    data = _get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions_{cik}.json",
    )
    blocks = [data["filings"]["recent"]]
    for f in data["filings"].get("files", []):  # older filings live in extra pages
        blocks.append(_get_json(f"https://data.sec.gov/submissions/{f['name']}", f["name"]))
    rows = []
    for b in blocks:
        for form, items, accepted in zip(b["form"], b["items"], b["acceptanceDateTime"]):
            if form != "8-K" or not items:
                continue
            # acceptanceDateTime is UTC; convert to US Eastern so the 4 pm close rule works.
            ts = pd.Timestamp(accepted).tz_convert("America/New_York").tz_localize(None)
            for code in items.split(","):
                category = EIGHT_K_ITEMS.get(code.strip())
                if category and category != "other":
                    rows.append(
                        {
                            "entity": entity,
                            "published_at": ts,
                            "source": "8k",
                            "category": category,
                            "relation": "self",
                            "direction": 0.0,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "entity",
            "published_at",
            "source",
            "category",
            "relation",
            "direction",
        ],
    )


# SEC 8-K item codes -> event category (8.01 is a catch-all, so it isn't counted)
EIGHT_K_ITEMS = {
    "1.01": "material_agreement",
    "1.02": "agreement_terminated",
    "1.03": "bankruptcy",
    "1.05": "cybersecurity_incident",
    "2.01": "acquisition_or_disposal",
    "2.02": "results",
    "2.03": "new_debt",
    "2.04": "debt_acceleration",
    "2.05": "restructuring",
    "2.06": "impairment",
    "3.01": "delisting_notice",
    "3.02": "unregistered_equity_sale",
    "4.01": "auditor_change",
    "4.02": "restatement",
    "5.01": "change_in_control",
    "5.02": "executive_or_director_change",
    "5.07": "shareholder_vote",
    "7.01": "reg_fd_disclosure",
    "8.01": "other",
}


# ---------------------------------------------------------------- fundamentals
FEATURE_TABLE_COLUMNS = ["entity", "as_of", "feature", "value", "source"]


def _safe_div(a, b):
    return a / b if b not in (0, None) and pd.notna(a) and pd.notna(b) else np.nan


def stock_features(ticker: str, cik: int, filed_until=None) -> pd.DataFrame:
    """Turn a company's 10-K facts into ratio features, dated by the 10-K filing date."""
    facts = annual_facts(cik, filed_until)
    if facts.empty:
        return pd.DataFrame(columns=FEATURE_TABLE_COLUMNS)
    wide = (
        facts.pivot_table(index="end", columns="concept", values="val", aggfunc="first")
        .reindex(columns=list(CONCEPTS))
        .sort_index()
    )  # missing concept -> NaN, not KeyError
    filed = facts.groupby("end")["filed"].max()  # date every number for this year was public
    prev = wide.shift(1)  # prior fiscal year, for growth features

    rows = []
    for end, cur in wide.iterrows():
        p = prev.loc[end]
        g = cur.get
        feats = {
            "revenue_growth": _safe_div(
                g("revenue") - p.get("revenue"), abs(p.get("revenue", np.nan))
            ),
            "gross_margin": _safe_div(g("gross_profit"), g("revenue")),
            "operating_margin": _safe_div(g("operating_income"), g("revenue")),
            "net_margin": _safe_div(g("net_income"), g("revenue")),
            "roa": _safe_div(g("net_income"), g("total_assets")),
            "liabilities_to_assets": _safe_div(g("total_liabilities"), g("total_assets")),
            "cash_to_assets": _safe_div(g("cash"), g("total_assets")),
            "rnd_intensity": _safe_div(g("rnd"), g("revenue")),
            "capex_intensity": _safe_div(g("capex"), g("revenue")),
            "share_dilution": _safe_div(g("shares") - p.get("shares"), p.get("shares")),
        }
        for name, value in feats.items():
            rows.append(
                {
                    "entity": ticker,
                    "as_of": filed[end],
                    "feature": name,
                    "value": value,
                    "source": "sec_xbrl",
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_TABLE_COLUMNS)


def to_wide(features: pd.DataFrame) -> pd.DataFrame:
    """Long feature table -> one row per (entity, as_of), one column per feature."""
    return features.pivot_table(
        index=["entity", "as_of"], columns="feature", values="value"
    ).reset_index()


EXTENDED = [
    "q_rev_growth_yoy_chg4",
    "q_op_margin_chg_yoy",
    "q_rnd_intensity",
    "q_rnd_intensity_chg_yoy",
]


def quarterly_features(
    ticker: str, cik: int, extended: bool = False, filed_until=None
) -> pd.DataFrame:
    """Per filed quarter: growth, margins and inventory, dated by when that quarter was public.

    Columns: entity, filed, q_rev_growth_yoy, q_rev_growth_qoq, q_gross_margin,
    q_gross_margin_chg_yoy, q_op_margin, q_inventory_days, q_inventory_days_chg_yoy.
    extended adds EXTENDED: revenue-growth acceleration (YoY growth minus the YoY growth four
    quarters earlier), operating-margin and R&D-intensity change vs a year ago. A quarter's date
    stays the base filing date; an R&D value first filed after it is left out (NaN).
    """
    extra = ("rnd",) if extended else ()
    facts = quarterly_facts(cik, extra, filed_until)
    if facts.empty:
        return pd.DataFrame()
    w = (
        facts.pivot_table(index="end", columns="concept", values="val", aggfunc="first")
        .reindex(columns=list(QUARTERLY_CONCEPTS) + list(extra))
        .sort_index()
    )
    filed = facts[facts["concept"].isin(FLOW)].groupby("end")["filed"].max()
    w = w[w.index.isin(filed.index)]
    if extended:  # R&D only where it was public by the quarter's own date
        rnd_filed = facts[facts["concept"] == "rnd"].set_index("end")["filed"]
        late = rnd_filed.reindex(w.index) > filed.reindex(w.index)
        w.loc[late.to_numpy(), "rnd"] = np.nan
    gross = w["gross_profit"].fillna(w["revenue"] - w["cost_of_revenue"])
    cogs = w["cost_of_revenue"].fillna(w["revenue"] - w["gross_profit"])
    out = pd.DataFrame(index=w.index)
    out["q_gross_margin"] = gross / w["revenue"].where(w["revenue"] > 0)
    out["q_op_margin"] = w["operating_income"] / w["revenue"].where(w["revenue"] > 0)
    out["q_inventory_days"] = w["inventory"] / cogs.where(cogs > 0) * 91

    def ago(days: int) -> pd.DataFrame:
        """The row for the quarter ending `days` earlier (within 20 days), else NaN."""
        idx = [
            w.index[np.argmin(np.abs((w.index - (e - pd.Timedelta(days=days))).days))]
            if len(w)
            else None
            for e in w.index
        ]
        ok = [
            i is not None and abs((e - pd.Timedelta(days=days) - i).days) <= 20
            for e, i in zip(w.index, idx)
        ]
        src = pd.concat([w["revenue"], out], axis=1)
        prev = src.reindex(idx).set_axis(w.index)
        return prev.where(pd.Series(ok, index=w.index), axis=0)

    yr, qtr = ago(364), ago(91)
    out["q_rev_growth_yoy"] = w["revenue"] / yr["revenue"].where(yr["revenue"] > 0) - 1
    out["q_rev_growth_qoq"] = w["revenue"] / qtr["revenue"].where(qtr["revenue"] > 0) - 1
    out["q_gross_margin_chg_yoy"] = out["q_gross_margin"] - yr["q_gross_margin"]
    out["q_inventory_days_chg_yoy"] = out["q_inventory_days"] - yr["q_inventory_days"]
    if extended:
        out["q_rnd_intensity"] = w["rnd"] / w["revenue"].where(w["revenue"] > 0)
        yr = ago(364)  # again: now carries this quarter's growth and R&D columns
        out["q_rev_growth_yoy_chg4"] = out["q_rev_growth_yoy"] - yr["q_rev_growth_yoy"]
        out["q_op_margin_chg_yoy"] = out["q_op_margin"] - yr["q_op_margin"]
        out["q_rnd_intensity_chg_yoy"] = out["q_rnd_intensity"] - yr["q_rnd_intensity"]
    return out.assign(entity=ticker, filed=filed.reindex(out.index).to_numpy()).reset_index(
        drop=True
    )


# ---------------------------------------------------------------- insider trades
INSIDERS_URL = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
)
INSIDERS_CACHE = Path(".cache") / "insiders"
FIRST_QUARTER = "2009Q1"


def _quarter_trades(q: str) -> pd.DataFrame:
    """Open-market trades in one quarterly file, cached as a small CSV (the zip is discarded)."""
    path = INSIDERS_CACHE / f"{q}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"cik": int})
    throttle()
    r = requests.get(INSIDERS_URL.format(q=q.lower()), headers=_headers(), timeout=300)
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
            "filed": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
            "owner": df["RPTOWNERCIK"],
            "officer": df["RPTOWNER_RELATIONSHIP"].fillna("").str.contains("Officer"),
            "buy": df["TRANS_CODE"] == "P",
            "usd": pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
            * pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce"),
        }
    ).dropna(subset=["cik", "filed"])
    out["cik"] = out["cik"].astype(int)
    INSIDERS_CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def download_insiders() -> None:
    """Every quarterly Form 4 data set since FIRST_QUARTER, cached as small per-quarter CSVs."""
    for q in pd.period_range(FIRST_QUARTER, pd.Timestamp.now().to_period("Q"), freq="Q"):
        t = _quarter_trades(str(q))
        print(f"{q}: {len(t):,} open-market trades", file=sys.stderr)
