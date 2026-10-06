"""US stocks, small and large caps from one plug-in (markets/us_smallcap.yaml, us_largecap.yaml):
SEC filings, EODHD prices, Form 4 insiders, measured trading costs, earnings-release text.

Run from the repo root; caches live in ./.cache. `engine build --fetch --market <m>` fills them with
your keys (EODHD_API_KEY for prices, SEC_USER_AGENT for EDGAR). The universe lists are built once
with `python -m engine.markets.us_stocks.universe build` (and `large`), the cost tables with
`python -m engine.markets.us_stocks.prices spreads`. Modules: universe, prices (bars, cleaning,
spreads), sec (XBRL facts, fundamentals, 8-K items, insiders), filings (8-K texts, the earnings
releases).

Market
  universe   `universe.file`: a point-in-time quarterly list (dead companies included), usable from
             the list date's close. entity_id = EODHD code, group = sector
  labels     adjusted close; enter at the first close after the decision, hold `horizon` bars.
             A series that stops early is a delisting: return to the last trade, then -30%
             (Shumway 1997) unless a merger 8-K sits near the last trade (a buyout pays the deal)
  cost_bps   measured round trip for that stock-month: mean of Abdi-Ranaldo and Corwin-Schultz
             effective spreads (`costs.file` picks the table)
  data_end   optional: bars, filings and 8-Ks after this date are never read (a study that must
             not see later years even through a label); a series is "ended" only if it stopped
             trading more than ended_if_no_bar_for_days before data_end

Sources (SEC filings are dated by day or accepted at a known time; either way they are usable from
the next midnight New York time)
  universe_size    px_log_mcap from the quarterly list
  price            momentum, volatility, 52-week high, volume trend: computed at decision times from
                   bars up to the last close before the decision
  sec_annual       10-K ratios (first-reported values, never restatements)
  sec_quarterly    10-Q growth, margins, inventory days (params.extended: growth acceleration,
                   operating-margin and R&D-intensity changes)
  insiders         Form 4 open-market buys / sells, last `window_days`
  eightk_counts    8-K filings per SEC item category, last `window_days`
  documents        sec_earnings_releases: earnings releases (8-K item 2.02), masked, for a
                   `type: text` source
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine import data
from engine.market import forward_returns
from engine.markets.us_stocks import prices, sec

DELIST_PENALTY = -0.30
BUYOUT_ITEMS = {"acquisition_or_disposal", "change_in_control"}  # 8-K 2.01 / 5.01
K8_CATEGORIES = [
    "results",
    "executive_or_director_change",
    "material_agreement",
    "acquisition_or_disposal",
    "new_debt",
    "restructuring",
    "impairment",
    "auditor_change",
    "restatement",
    "delisting_notice",
    "unregistered_equity_sale",
    "agreement_terminated",
]
PRICE_FEATURES = [
    "px_mom_1m",
    "px_mom_12_1",
    "px_vol_3m",
    "px_off_high",
    "px_volume_trend",
]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def read_universe(path: str | Path) -> pd.DataFrame:
    """Codes stay text: a ticker literally named "NA" isn't missing."""
    return pd.read_csv(
        path,
        parse_dates=["as_of"],
        dtype={"code": str},
        keep_default_na=False,
        na_values=[""],
    ).rename(columns={"as_of": "list_date"})


class UsStocks:
    """The market contract for US stocks; the YAML picks the universe list, costs and data end."""

    name = "us_stocks"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        u = cfg["universe"]
        self.calendar = data.make_calendar(cfg.get("calendar", {}))
        self.lists = read_universe(u["file"])
        self.lists["_available"] = self.calendar.bar_close(self.lists["list_date"])
        self.exclude_name = u.get("exclude_name_regex")
        self.group_col = u.get("group", "sector")
        self.min_history = int(cfg.get("labels", {}).get("min_history_bars", 0))
        self.drop_flat = bool(cfg.get("labels", {}).get("drop_flat", False))
        self.ended_if_older_than = pd.Timedelta(
            days=int(cfg.get("labels", {}).get("ended_if_no_bar_for_days", 10))
        )
        self._list_dates = pd.DatetimeIndex(self.lists["_available"].unique()).sort_values()
        self.data_end = pd.Timestamp(cfg["data_end"]) if cfg.get("data_end") else None
        # panel caches key on it; absent, the small-cap cache keys are unchanged
        if self.data_end is not None:
            self.fingerprint = f"data_end={self.data_end.date()}"
        self.cost_file = (cfg.get("costs") or {}).get("file")

    # ---------- universe ----------
    def entities(self) -> pd.DataFrame:
        """Every (code, cik) pair the universe ever held: how cik-keyed sources reach codes."""
        return self.lists[["code", "cik"]].drop_duplicates().rename(columns={"code": "entity_id"})

    def universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """The latest list usable before the decision (SPAC shells by name excluded)."""
        i = self._list_dates.searchsorted(as_of, side="left")
        if i == 0:
            return pd.DataFrame()
        latest = self._list_dates[i - 1]  # the latest list usable before the decision
        u = self.lists[self.lists["_available"] == latest]
        if self.exclude_name:
            u = u[~u["name"].str.upper().str.contains(self.exclude_name, regex=True, na=False)]
        return (
            u.drop(columns="_available")
            .rename(columns={"code": "entity_id", self.group_col: "group"})
            .assign(sector=u[self.group_col].to_numpy())
        )

    # ---------- bars and labels ----------
    def bars(self, code: str) -> pd.DataFrame:
        """Cleaned daily bars indexed by UTC close, cut at data_end."""
        px = prices.load_prices(code)
        if self.data_end is not None:
            px = px[px.index <= self.data_end]
        if len(px):
            px = px.copy()
            px.index = self.calendar.bar_close(px.index)
        return px

    def _now(self) -> pd.Timestamp:
        if self.data_end is None:
            return pd.Timestamp.now(tz="UTC")
        return self.calendar.bar_close([self.data_end])[0]

    def _ended(self, px: pd.DataFrame) -> bool:
        return bool(len(px) and px.index[-1] < self._now() - self.ended_if_older_than)

    def _buyout(self, cik: int, last_bar: pd.Timestamp) -> bool:
        ev = sec.eight_k_events(int(cik), str(cik))
        if self.data_end is not None:
            ev = ev[ev["published_at"] < self.data_end + pd.Timedelta(days=1)]
        last = self.calendar.local_date(pd.Series([last_bar])).iloc[0]
        near = ev[
            (ev["published_at"] >= last - pd.Timedelta(days=90))
            & (ev["published_at"] <= last + pd.Timedelta(days=30))
        ]
        return bool(near["category"].isin(BUYOUT_ITEMS).any())

    def labels(self, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forward returns with the delisting rule."""
        cik_of = self.entities().drop_duplicates("entity_id").set_index("entity_id")["cik"]
        out = []
        for code, g in rows.groupby("entity_id", sort=False):
            px = self.bars(code)
            times = pd.DatetimeIndex(g["decision_time"])
            if px.empty:
                out.append(pd.DataFrame({"decision_time": times}).assign(entity_id=code))
                continue
            ended = self._ended(px)
            adj = 0.0
            if ended and not self._buyout(cik_of[code], px.index[-1]):
                adj = DELIST_PENALTY
            lab = forward_returns(
                px.index,
                px["adj_close"].astype(float).to_numpy(),
                times,
                horizon,
                ended,
                adj,
                self.min_history,
                self.drop_flat,
            )
            out.append(lab.assign(entity_id=code))
        lab = pd.concat(out, ignore_index=True)
        for c in ("decision_time", "entry_time", "label_end"):
            lab[c] = pd.to_datetime(lab[c], utc=True)
        lab["delisted"] = lab["delisted"].fillna(False).astype(bool)
        return lab

    def daily_returns(self, codes) -> tuple[pd.DataFrame, pd.Series]:
        """Daily simple returns (local trading dates x codes, NaN = no bar) and, for series that
        ended, their last local date. An ended series' last return carries the delisting rule
        (-30% unless a merger 8-K sits near the last trade), as in labels()."""
        cik_of = self.entities().drop_duplicates("entity_id").set_index("entity_id")["cik"]
        cols, ended = {}, {}
        for code in codes:
            px = self.bars(code)
            if px.empty:
                continue
            r = px["adj_close"].astype(float).pct_change()
            if self._ended(px):
                if not self._buyout(cik_of[code], px.index[-1]):
                    r.iloc[-1] = (1 + r.iloc[-1]) * (1 + DELIST_PENALTY) - 1
                ended[code] = self.calendar.local_date(px.index[-1:]).iloc[0]
            r.index = self.calendar.local_date(pd.Series(px.index)).to_numpy()
            cols[code] = r
        return pd.DataFrame(cols).sort_index(), pd.Series(ended, dtype="datetime64[ns]")

    # ---------- costs ----------
    def cost_bps(self, rows: pd.DataFrame) -> pd.Series:
        """Measured round trip for the stock-month (NaN when unmeasured)."""
        sp = prices.load_spreads(self.cost_file)[["code", "month", "ar", "cs"]]
        month = self.calendar.local_date(rows["decision_time"]).dt.to_period("M")
        m = pd.DataFrame({"code": rows["entity_id"].to_numpy(), "month": month.to_numpy()}).merge(
            sp, on=["code", "month"], how="left"
        )
        return pd.Series(m[["ar", "cs"]].mean(axis=1).to_numpy() * 1e4, index=rows.index)


# ---------- sources ----------


class _Base:
    def __init__(self, market: UsStocks, params: dict | None = None):
        self.market = market
        self.cal = market.calendar
        self.params = params or {}

    def fetch(self, start, end) -> None:
        """Default: the per-company SEC submissions JSON (8-K items, names, SIC), cached."""
        for cik in sorted(self._codes_by_cik()):
            try:
                sec._get_json(
                    f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json",
                    f"submissions_{int(cik)}.json",
                )
            except Exception as e:  # noqa: BLE001 -- a filer the SEC no longer serves
                _log(f"  {self.name}: cik {cik}: {type(e).__name__}")

    def _codes_by_cik(self) -> dict[int, list[str]]:
        e = self.market.entities()
        return e.groupby("cik")["entity_id"].apply(list).to_dict()


class UniverseSize(_Base):
    name = "universe_size"

    def fetch(self, start, end) -> None:
        _log(
            "universe_size: the list is built by "
            "`python -m engine.markets.us_stocks.universe build` (EODHD + SEC)"
        )

    def observations(self, start, end) -> pd.DataFrame:
        """log market value from each list, usable from the list date's close."""
        u = self.market.lists
        return pd.DataFrame(
            {
                "entity_id": u["code"],
                "available_at": u["_available"],
                "source": self.name,
                "feature": "px_log_mcap",
                "value": np.log(u["mcap"].astype(float)),
            }
        )


class PriceFeatures(_Base):
    """Bar-derived features, computed only at the decision times asked for."""

    name = "price"

    def fetch(self, start, end) -> None:
        """Bars for every code the universe ever held (close-only, then OHLC), cached."""
        from concurrent.futures import ThreadPoolExecutor

        codes = sorted(self.market.entities()["entity_id"].unique())
        for fetch in (prices.fetch_prices, prices.fetch_ohlc):
            with ThreadPoolExecutor(int(self.params.get("workers", 8))) as pool:
                for i, _ in enumerate(pool.map(fetch, codes), 1):
                    if i % 1000 == 0:
                        _log(f"  {fetch.__name__}: {i:,}/{len(codes):,}")

    def observations(self, start, end) -> pd.DataFrame:
        """Not used: see observations_at."""
        raise NotImplementedError("price features are computed at decision times: observations_at")

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Momentum, volatility, 52-week high and volume trend from bars closed before each decision."""
        need = int(self.params.get("min_history_bars", 253))
        out = []
        for code, g in rows.groupby("entity_id", sort=False):
            px = self.market.bars(code)
            if px.empty:
                continue
            adj = px["adj_close"].astype(float)
            dvol = px["close"].astype(float) * px["volume"].astype(float)
            times = pd.DatetimeIndex(g["decision_time"].unique())
            for t, i in zip(times, px.index.searchsorted(times, side="right")):
                if i < need:  # bars 0..i-1 closed before the decision
                    continue
                hist = adj.iloc[:i]
                vol_base = dvol.iloc[i - 126 : i].mean()
                vals = {
                    "px_mom_1m": hist.iloc[-1] / hist.iloc[-22] - 1,
                    "px_mom_12_1": hist.iloc[-22] / hist.iloc[-253] - 1,
                    "px_vol_3m": hist.pct_change().iloc[-63:].std(),
                    "px_off_high": hist.iloc[-1] / hist.iloc[-253:].max() - 1,
                    "px_volume_trend": dvol.iloc[i - 21 : i].mean() / vol_base
                    if vol_base > 0
                    else np.nan,
                }
                at = adj.index[i - 1]  # the last close used
                out += [(code, at, f, v) for f, v in vals.items()]
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


class _PerCik(_Base):
    """Sources whose data is keyed by SEC company ID, emitted once per code of that company."""

    def fetch(self, start, end) -> None:
        """XBRL company facts per company, cached (both 10-K and 10-Q features read them)."""
        for cik in sorted(self._codes_by_cik()):
            try:
                sec._get_json(
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
                    f"facts_{int(cik)}.json",
                )
            except Exception as e:  # noqa: BLE001 -- filers with no XBRL facts
                _log(f"  {self.name}: cik {cik}: {type(e).__name__}")

    def _frames(self, cik: int) -> pd.DataFrame:
        """entity-free frame: available_at + one column per feature, in publication order."""
        raise NotImplementedError

    def observations(self, start, end) -> pd.DataFrame:
        """Each company's rows, emitted once per code it traded under."""
        parts = []
        by_cik = self._codes_by_cik()
        for n, (cik, codes) in enumerate(by_cik.items(), 1):
            w = self._frames(int(cik))
            if w is None or w.empty:
                continue
            for code in codes:
                parts.append(w.assign(entity_id=code))
            if n % 1000 == 0:
                _log(f"  {self.name}: {n:,}/{len(by_cik):,} companies")
        if not parts:
            return data.empty_observations()
        wide = pd.concat(parts, ignore_index=True)
        feats = [c for c in wide.columns if c not in ("entity_id", "available_at")]
        return data.from_wide(wide, self.name, features=feats)


class SecAnnual(_PerCik):
    name = "sec_annual"

    def _frames(self, cik: int) -> pd.DataFrame | None:
        w = sec.to_wide(sec.stock_features(str(cik), cik, self.market.data_end))
        if w.empty:
            return None
        w = w.drop(columns="entity").sort_values("as_of", kind="stable")
        w["available_at"] = self.cal.date_available(w.pop("as_of")).to_numpy()
        return w


class SecQuarterly(_PerCik):
    """One filing can report two quarter ends (e.g. a 10-K with the fourth quarter and a restated
    one) on the same date: the latest quarter end wins (rows arrive sorted by quarter end; the sort
    by filing date is stable). params.extended: also emit sec.EXTENDED (growth acceleration,
    margin and R&D changes)."""

    name = "sec_quarterly"

    def _frames(self, cik: int) -> pd.DataFrame | None:
        q = sec.quarterly_features(
            str(cik), cik, bool(self.params.get("extended")), self.market.data_end
        )
        if q.empty:
            return None
        q = q.dropna(subset=["filed"]).sort_values("filed", kind="stable")
        return self._stamp(q)

    def _stamp(self, q: pd.DataFrame) -> pd.DataFrame:
        q = q.copy()
        q["available_at"] = self.cal.date_available(q.pop("filed")).to_numpy()
        return q.drop(columns="entity").astype({c: float for c in q.columns if c.startswith("q_")})


class EightKCounts(_Base):
    name = "eightk_counts"

    def observations(self, start, end) -> pd.DataFrame:
        """8-K counts per item category over the window (exact rolling-window observations)."""
        days = int(self.params.get("window_days", 90))
        events = []
        for cik, codes in self._codes_by_cik().items():
            ev = sec.eight_k_events(int(cik), str(cik))
            if ev.empty:
                continue
            at = self.cal.next_midnight(sec.localize(ev["published_at"], self.cal.tz))
            e = pd.DataFrame(
                {"available_at": at.to_numpy(), "category": ev["category"].to_numpy()}
            ).dropna()
            events += [e.assign(entity_id=c) for c in codes]
        events = pd.concat(events, ignore_index=True)
        names = [f"k8_{c}" for c in K8_CATEGORIES]

        def counts(e: pd.DataFrame):
            vc = e["category"].value_counts()
            return [float(vc.get(c, 0)) for c in K8_CATEGORIES]

        return data.rolling_window(
            events,
            pd.Timedelta(days=days),
            counts,
            names,
            self.name,
            baseline_entities=list(self.market.entities()["entity_id"].unique()),
        )


class InsiderTrades(_Base):
    name = "insiders"

    def fetch(self, start, end) -> None:
        """The SEC's quarterly Form 3/4/5 data sets -> open-market trades per quarter, cached."""
        sec.download_insiders()

    def _trades(self) -> pd.DataFrame:
        files = sorted(Path(self.params.get("dir", ".cache/insiders")).glob("*.csv"))
        ciks = set(self.market.entities()["cik"].astype(int))
        frames = []
        for f in files:
            t = pd.read_csv(f, dtype={"cik": int})
            frames.append(t[t["cik"].isin(ciks)])
        t = pd.concat(frames, ignore_index=True)
        t["filed"] = pd.to_datetime(t["filed"])
        return t

    def observations(self, start, end) -> pd.DataFrame:
        """Insider buyers, sellers, officer buyers and net $ share over the window."""
        days = int(self.params.get("window_days", 182))
        t = self._trades()
        t["available_at"] = self.cal.date_available(t["filed"]).to_numpy()
        codes = self._codes_by_cik()
        events = pd.concat(
            [g.assign(entity_id=c) for cik, g in t.groupby("cik") for c in codes.get(int(cik), [])],
            ignore_index=True,
        )
        names = ["ins_buyers", "ins_sellers", "ins_officer_buyers", "ins_net_usd_share"]

        def agg(e: pd.DataFrame):
            if e.empty:
                return [0.0, 0.0, 0.0, 0.0]
            b, s = e[e["buy"]], e[~e["buy"]]
            bought, sold = b["usd"].sum(), s["usd"].sum()
            return [
                float(b["owner"].nunique()),
                float(s["owner"].nunique()),
                float(b.loc[b["officer"], "owner"].nunique()),
                float((bought - sold) / (bought + sold)) if bought + sold > 0 else 0.0,
            ]

        return data.rolling_window(
            events,
            pd.Timedelta(days=days),
            agg,
            names,
            self.name,
            baseline_entities=list(self.market.entities()["entity_id"].unique()),
        )


SOURCES = {
    s.name: s
    for s in (
        UniverseSize,
        PriceFeatures,
        SecAnnual,
        SecQuarterly,
        EightKCounts,
        InsiderTrades,
    )
}


def _sec_releases(market, params):
    from engine.markets.us_stocks.filings import SecEarningsReleases

    return SecEarningsReleases(market, params)


# document sources for engine.text.source.TextSource (sources: {type: text, params: {documents: ...}})
DOCUMENT_SOURCES = {"sec_earnings_releases": _sec_releases}


def build(cfg: dict):
    """The plug-in entry point."""
    return UsStocks(cfg), SOURCES
