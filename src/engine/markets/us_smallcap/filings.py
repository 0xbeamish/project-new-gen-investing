"""8-K texts from EDGAR (ported from the pilot) and the earnings-release DocumentSource.

The main 8-K document carries most items (executive changes, agreements, impairments). For
results (2.02) and Reg FD (7.01) the substance is in the attached press release (EX-99.x), so
that exhibit is used instead.
"""

import gzip
import json
import re
from pathlib import Path

import lxml.html
import numpy as np
import pandas as pd
import requests

from engine.markets.us_smallcap import sec
from engine.text import masking

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EXHIBIT_ITEMS = {"2.02", "7.01"}  # substance lives in the press release exhibit
MAX_CHARS = 12_000  # keeps Jev well inside its 32k-token state limit and focused (its docs: long inputs hurt)
SUFFIXES = r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|HOLDINGS|GROUP|THE|N\.?V|S\.?A)\b\.?"


def filings(cik: int) -> list[dict]:
    """Every 8-K in the company's submissions history: accession, primary doc, items, dates."""
    data = sec._get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions_{cik}.json",
    )
    blocks = [data["filings"]["recent"]]
    blocks += [
        sec._get_json(f"https://data.sec.gov/submissions/{f['name']}", f["name"])
        for f in data["filings"].get("files", [])
    ]
    out = []
    for b in blocks:
        for form, acc, doc, items, filed, accepted in zip(
            b["form"],
            b["accessionNumber"],
            b["primaryDocument"],
            b["items"],
            b["filingDate"],
            b["acceptanceDateTime"],
        ):
            if form == "8-K" and items:
                out.append(
                    {
                        "accession": acc,
                        "primary": doc,
                        "items": items.split(","),
                        "filed": filed,
                        # UTC on EDGAR; US Eastern here so the 4 pm close rule applies
                        "accepted_et": pd.Timestamp(accepted)
                        .tz_convert("America/New_York")
                        .tz_localize(None),
                    }
                )
    return out


def _get_text(url: str) -> str:
    sec.throttle()
    resp = requests.get(url, headers=sec._headers(), timeout=30)
    resp.raise_for_status()
    if url.lower().endswith((".htm", ".html")):
        text = lxml.html.fromstring(resp.content).text_content()
    else:
        text = resp.text
    return re.sub(r"\s+", " ", text).strip()


class NoPressRelease(RuntimeError):
    """An earnings/Reg FD 8-K without an attached release: tagging its cover page would read as neutral."""


def _exhibit_name(cik: int, acc_nodash: str, primary: str) -> str | None:
    """The press release file: EX-99 by name ("ex991", "exhibit991"), else the largest other document."""
    sec.throttle()
    resp = requests.get(
        f"{ARCHIVES}/{cik}/{acc_nodash}/index.json", headers=sec._headers(), timeout=30
    )
    resp.raise_for_status()
    items = resp.json()["directory"]["item"]
    docs = [
        i
        for i in items
        if i["name"].lower().endswith((".htm", ".html", ".txt"))
        and i["name"] != primary
    ]
    docs = [
        i
        for i in docs
        if "index" not in i["name"].lower()
        and not re.fullmatch(r"[\d-]+\.txt", i["name"])
    ]
    ex99 = [
        i["name"]
        for i in docs
        if re.search(r"ex(hibit)?[-_]?99", i["name"], re.IGNORECASE)
    ]
    if ex99:
        return min(ex99)  # 99.1 sorts before 99.2
    sized = [i for i in docs if str(i.get("size", "")).isdigit()]
    return max(sized, key=lambda i: int(i["size"]))["name"] if sized else None


TEXT_CACHE = Path(".cache") / "8k_text"


def _cache_path(url: str) -> Path:
    return TEXT_CACHE / (
        re.sub(r"[^A-Za-z0-9.]+", "_", url.split("/data/")[-1]) + ".gz"
    )


def _cached_text(url: str) -> str:
    """Extracted text, saved compressed so new questions or masking fixes never need a re-download."""
    path = _cache_path(url)
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode()
    text = _get_text(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode()))
    return text


def document_text(cik: int, filing: dict, item: str) -> tuple[str, str]:
    """(trimmed text, source URL) for one item of one 8-K."""
    acc = filing["accession"].replace("-", "")
    name = None
    if item in EXHIBIT_ITEMS:
        name = _exhibit_name(cik, acc, filing["primary"])
        if name is None:
            raise NoPressRelease(f"{filing['accession']}: no press release attached")
    url = f"{ARCHIVES}/{cik}/{acc}/{name or filing['primary']}"
    text = _cached_text(url)
    if name is None:  # main 8-K: skip the cover page, stop at the signature block
        start = re.search(r"Item\s+\d\.\d\d", text, re.IGNORECASE)
        end = re.search(r"\bSIGNATURES?\b", text[start.start() if start else 0 :])
        text = text[start.start() if start else 0 :][: end.start() if end else None]
    return text[:MAX_CHARS], url


def text_for(url: str, item: str) -> str:
    """Trimmed text for a known document URL (cached): press releases whole, main 8-Ks without cover/signatures."""
    if url.startswith(
        "file://"
    ):  # locally built documents, e.g. new risk-factor sentences
        return Path(url[len("file://") :]).read_text()[:MAX_CHARS]
    text = _cached_text(url)
    if item not in EXHIBIT_ITEMS:
        start = re.search(r"Item\s+\d\.\d\d", text, re.IGNORECASE)
        end = re.search(r"\bSIGNATURES?\b", text[start.start() if start else 0 :])
        text = text[start.start() if start else 0 :][: end.start() if end else None]
    return text[:MAX_CHARS]


def company_names(cik: int) -> list[str]:
    """Current and former SEC names (Howmet's 2017 filings say "Arconic")."""
    data = sec._get_json(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        f"submissions_{cik}.json",
    )
    return [data["name"]] + [
        f["name"] for f in data.get("formerNames", []) if f.get("name")
    ]


def cached_names(cik: int) -> list[str] | None:
    """company_names from the submissions cache only (None if not cached): no network."""
    path = sec.CACHE / f"submissions_{cik}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [data["name"]] + [
        f["name"] for f in data.get("formerNames", []) if f.get("name")
    ]


def _localize(naive: pd.Series, tz: str) -> pd.Series:
    return naive.dt.tz_localize(
        tz, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward"
    )


MANIFEST_COLUMNS = ["cik", "accession", "primary", "accepted_et", "url", "error"]


class SecEarningsReleases:
    """Earnings press releases (8-K item 2.02, exhibit 99) of every company the universe ever held.

    fetch(start, end)      finds each company's 2.02 filings accepted in [start, end), downloads the
                           release text (SEC_USER_AGENT needed) and appends to the manifest CSV
    documents(start, end)  reads ONLY the manifest and the text cache: one row per (release, code),
                           available_at = the next New York midnight after acceptance (the pilot's
                           rule), text masked (company names, first words, ticker, v2 extras)
    params: manifest (default .cache/earnings_releases.csv), doc_type, max_chars, workers
    """

    name = "sec_earnings_releases"
    ITEM = "2.02"

    def __init__(self, market, params: dict | None = None):
        p = params or {}
        self.market = market
        self.manifest = Path(p.get("manifest", ".cache/earnings_releases.csv"))
        self.doc_type = p.get("doc_type", "earnings_release")
        self.max_chars = int(p.get("max_chars", MAX_CHARS))
        self.workers = int(p.get("workers", 6))

    def _read_manifest(self) -> pd.DataFrame:
        if not self.manifest.exists():
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        return pd.read_csv(self.manifest, dtype={"accession": str})

    def fetch(self, start, end) -> None:
        from concurrent.futures import ThreadPoolExecutor

        done = set(self._read_manifest()["accession"])
        lo, hi = (
            pd.Timestamp(start).tz_localize(None),
            pd.Timestamp(end).tz_localize(None),
        )
        todo = []
        for cik in sorted(self.market.entities()["cik"].astype(int).unique()):
            for f in filings(int(cik)):
                wanted = self.ITEM in [i.strip() for i in f["items"]]
                if (
                    wanted
                    and lo <= f["accepted_et"] < hi
                    and f["accession"] not in done
                ):
                    keep = ("accession", "primary", "accepted_et")
                    todo.append({"cik": int(cik), **{k: f[k] for k in keep}})

        def one(row):
            try:
                _, url = document_text(row["cik"], row, self.ITEM)
                return row | {"url": url, "error": None}
            except Exception as e:  # noqa: BLE001 -- a missing release is skipped, not fatal
                return row | {
                    "url": None,
                    "error": f"{type(e).__name__}: {str(e)[:120]}",
                }

        for k in range(0, len(todo), 500):
            with ThreadPoolExecutor(self.workers) as pool:  # the SEC throttle is shared
                got = pd.DataFrame(
                    list(pool.map(one, todo[k : k + 500])), columns=MANIFEST_COLUMNS
                )
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            got.to_csv(
                self.manifest, mode="a", header=not self.manifest.exists(), index=False
            )

    def documents(self, start, end) -> pd.DataFrame:
        m = self._read_manifest()
        m = m[m["error"].isna() & m["url"].notna()].drop_duplicates("accession")
        cal = self.market.calendar
        accepted = pd.to_datetime(m["accepted_et"])
        m = m.assign(
            available_at=cal.next_midnight(
                _localize(accepted.reset_index(drop=True), cal.tz)
            ).to_numpy()
        )
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        m = m[m["available_at"] < end].sort_values("accepted_et", kind="stable")
        codes = self.market.entities().groupby("cik")["entity_id"].apply(list).to_dict()
        rows, names = [], {}
        for r in m.itertuples(index=False):
            path = _cache_path(r.url)
            cik = int(r.cik)
            if not path.exists() or cik not in codes:
                continue  # documents() never touches the network
            raw = text_for(r.url, self.ITEM)[: self.max_chars]
            if cik not in names:
                names[cik] = cached_names(cik) or []
            for code in codes[cik]:
                rows.append(
                    {
                        "entity_id": code,
                        "available_at": r.available_at,
                        "doc_type": self.doc_type,
                        "doc_id": f"{r.accession}|{self.ITEM}",
                        "text": masking.mask(raw, names[cik], code),
                        "metadata": {"cik": cik, "url": r.url},
                    }
                )
        cols = ["entity_id", "available_at", "doc_type", "doc_id", "text", "metadata"]
        return pd.DataFrame(rows, columns=cols)
