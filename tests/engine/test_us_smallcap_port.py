"""The us_smallcap plug-in's own helpers (ported from the pilot), offline: no cache, no network."""

import gzip
import json

import numpy as np
import pandas as pd

from engine import calendar as calmod
from engine.markets.us_smallcap import filings, prices, sec, spreads, universe


def test_spread_estimators_on_known_bars():
    h = np.log(np.array([10.2, 10.3, 10.1, 10.4]))
    lo = np.log(np.array([9.8, 9.9, 9.7, 10.0]))
    c = np.log(np.array([10.0, 10.1, 9.9, 10.2]))
    assert spreads.abdi_ranaldo(h, lo, c) >= 0
    assert 0 <= spreads.corwin_schultz(h, lo) < 0.1
    flat = np.log(np.full(5, 10.0))
    assert spreads.abdi_ranaldo(flat, flat, flat) == 0.0  # no range, no spread
    assert np.isnan(spreads.abdi_ranaldo(h[:2], lo[:2], c[:2]))  # too few bars


def test_quarter_snapshot_uses_the_last_bar_and_rejects_stale_ones():
    idx = pd.bdate_range("2015-10-01", "2015-12-31")
    px = pd.DataFrame(
        {"close": np.arange(len(idx), dtype=float) + 1, "volume": 1000.0}, index=idx
    )
    close, dvol = prices.quarter_snapshot(px, pd.Timestamp("2015-12-31"))
    assert close == px["close"].iloc[-1] and dvol > 0
    assert (
        prices.quarter_snapshot(px, pd.Timestamp("2016-01-31")) is None
    )  # > 7 days old


def test_eight_k_items_and_names():
    assert (
        sec.EIGHT_K_ITEMS["2.02"] == "results" and sec.EIGHT_K_ITEMS["8.01"] == "other"
    )
    assert (
        sec.sic_to_sector(2834) == "health_care" and sec.sic_to_sector(None) == "other"
    )
    assert universe._norm("Aetna Inc /PA/") == universe._norm("AETNA INC")


def test_earnings_releases_are_masked_and_read_from_the_cache_only(
    tmp_path, monkeypatch
):
    url = "https://www.sec.gov/Archives/edgar/data/1/000000000115000001/ex99.htm"
    monkeypatch.setattr(filings, "TEXT_CACHE", tmp_path / "8k_text")
    path = filings._cache_path(url)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        gzip.compress(
            b"Zorblax Widgets Inc. (ZRBX) reported record revenue. Zorblax raised guidance."
        )
    )
    monkeypatch.setattr(sec, "CACHE", tmp_path / "sec")
    (tmp_path / "sec").mkdir()
    (tmp_path / "sec" / "submissions_1.json").write_text(
        json.dumps({"name": "Zorblax Widgets Inc", "formerNames": []})
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "cik": 1,
                "accession": "0000000001-15-000001",
                "primary": "x.htm",
                "accepted_et": "2015-02-03 16:05:00",
                "url": url,
                "error": None,
            }
        ]
    ).to_csv(manifest, index=False)

    class Mkt:
        calendar = calmod.TradingCalendar()

        def entities(self):
            return pd.DataFrame({"entity_id": ["ZRBX"], "cik": [1]})

    import requests

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")),
    )
    d = filings.SecEarningsReleases(Mkt(), {"manifest": str(manifest)}).documents(
        None, "2016-01-01"
    )
    assert len(d) == 1 and d["doc_id"].iloc[0] == "0000000001-15-000001|2.02"
    t = d["text"].iloc[0]
    assert "Zorblax" not in t and "ZRBX" not in t and "COMPANY_A" in t
    # usable from the next New York midnight after acceptance
    assert d["available_at"].iloc[0] == pd.Timestamp("2015-02-04 05:00", tz="UTC")
