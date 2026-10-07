"""Crypto (markets/crypto.yaml): the most-traded Binance USDT pairs, with DeFiLlama fees, perp funding
and hacks. Free data, no keys: `engine build --fetch --market crypto` fills .cache/crypto/.
Modules: binance (spot bars, funding), llama (fees, hacks, token mapping).

Market
  entities   one per Binance USDT pair, entity_id = the base token (BTC). Stablecoins, fiat, gold,
             wrapped copies (`universe.exclude`) and leveraged tokens (BTCUP, ETHBEAR) are left out.
             A pair with no bar for `gap_days` is split: the part after the gap is a new entity
             (LUNA~2022-05-31), since Binance reuses tickers
  universe   the `top` entities by mean daily USDT volume over the last `volume_days` bars closed
             before the decision, listed at least `min_history_bars`, with a bar in the last 3 days.
             group = the token's sector from `sectors` (a CSV), so returns are sector-adjusted
  labels     enter at the first daily close after the decision, hold `horizon` bars. A series that
             stops for good is a delisting: `delisted_return` is added, except for `renamed` tokens
  cost_bps   round trip: 2 x (`taker_fee_bps` + square-root impact of a `trade_usd` trade: daily
             return sd x sqrt(trade_usd / mean daily USDT volume), over the last `impact_days`
             bars before the decision). Spread estimators from daily high/low fail on 24/7 bars
  data_end   bars, fees, funding and hacks after it are never read

Sources (all computed at the decision times asked for, from data usable strictly before them)
  price      px_mom_1w / 4w / 12w (7, 28, 84 bars), px_vol_4w (daily return sd), px_log_dvol
             (log mean daily USDT volume, 28 bars)
  funding    funding_7d: mean funding rate of the token's USDT perp over the last 7 days
  fees       fee_growth_90d: log(fees over the latest 30 usable days / the 30 days ending 90 days
             earlier); fee_vs_price_90d: that minus the log price change over the same 90 days.
             A day's fees are usable from the end of the next day
  hacks      hack_usd_30d: log(1 + USD lost to hacks of the token's protocol, usable in the last
             30 days); a hack is usable from the end of the day after it. 0 when none
  growth     px_volume_growth_90d, dex_volume_growth_90d: log(30-day sum / the 30 days ending 90
             days earlier) of USDT volume, and of DEX volume (protocol, else chain; DeFiLlama)
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from engine import data
from engine.market import forward_returns
from engine.markets.crypto import binance, indicators, llama
from engine.run import path as repo_path

LEVERAGED = re.compile(r"^(.{2,})(UP|DOWN|BULL|BEAR)$")
FETCH_SINCE = "2020-01"  # warm-up for 2021: 90 bars of history + 90-day features


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def split_at_gaps(bars: pd.DataFrame, base: str, gap_days: int) -> dict[str, pd.DataFrame]:
    """One entity per run of bars without a gap of gap_days or more."""
    if bars.empty:
        return {}
    gap = bars["time"].diff() >= pd.Timedelta(days=gap_days)
    seg = gap.cumsum()
    out = {}
    for k, g in bars.groupby(seg, sort=True):
        name = base if k == 0 else f"{base}~{g['time'].iloc[0].date()}"
        out[name] = g.reset_index(drop=True)
    return out


class Crypto:
    """The market contract over Binance daily bars (loaded on first use, after any fetch)."""

    name = "crypto"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        u, lab, c = cfg.get("universe", {}), cfg.get("labels", {}), cfg.get("costs", {})
        self.calendar = data.make_calendar(cfg.get("calendar", {}))
        self.top = int(u.get("top", 100))
        self.volume_days = int(u.get("volume_days", 30))
        self.min_history = int(u.get("min_history_bars", 90))
        self.exclude = {s.upper() for s in u.get("exclude", [])}
        self.renamed = {s.upper() for s in u.get("renamed", [])}
        self.gap_days = int(u.get("gap_days", 10))
        self.ended_days = int(lab.get("ended_if_no_bar_for_days", 10))
        self.delisted_return = float(lab.get("delisted_return", -0.30))
        self.fee_bps = float(c.get("taker_fee_bps", 10))
        self.trade_usd = float(c.get("trade_usd", 10000))
        self.impact_days = int(c.get("impact_days", 30))
        self.data_end = pd.Timestamp(cfg["data_end"]) if cfg.get("data_end") else None
        sectors = u.get("sectors")  # token -> group (sector); returns are then sector-adjusted
        self.sector = (
            pd.read_csv(repo_path(sectors), keep_default_na=False).set_index("token")["sector"]
            if sectors
            else None
        )
        self._series = None

    @property
    def fingerprint(self) -> str:
        """Panel caches key on it: the data end and when the caches last changed."""
        stamps = [f.stat().st_mtime_ns for f in binance.CACHE.rglob("*") if f.is_file()]
        return f"crypto|data_end={self.data_end}|{max(stamps, default=0)}"

    # ---------- entities and bars ----------
    def keep(self, base: str, all_bases: set[str]) -> bool:
        """Not a stablecoin / fiat / wrapped copy, and not a leveraged token of a listed base."""
        if base in self.exclude:
            return False
        m = LEVERAGED.match(base)
        return not (m and m.group(1) in all_bases)

    def pairs(self) -> list[str]:
        """The USDT pairs to download (from the archive listing)."""
        syms = binance.spot_symbols()
        bases = {s[:-4] for s in syms}
        return [s for s in syms if self.keep(s[:-4], bases)]

    def reload(self) -> None:
        """Forget loaded bars (after a fetch)."""
        self._series = None

    @property
    def series(self) -> dict[str, pd.DataFrame]:
        if self._series is None:
            self._load()
        return self._series

    def _load(self) -> None:
        series = {}
        listing = binance.CACHE / "spot_symbols_USDT.txt"
        syms = self.pairs() if listing.exists() else []
        for sym in syms:
            b = binance.load("spot", sym)
            if self.data_end is not None and len(b):
                b = b[b["time"] <= self.data_end]
            b = b.dropna(subset=["close"])
            b = b[b["close"] > 0]
            series |= split_at_gaps(b, sym[:-4], self.gap_days)
        end = self.data_end or max((s["time"].iloc[-1] for s in series.values()), default=None)
        for s in series.values():
            s["close_at"] = self.calendar.bar_close(s["time"])
        self.ended = {
            e: bool(s["time"].iloc[-1] < end - pd.Timedelta(days=self.ended_days))
            for e, s in series.items()
        }
        self._series = series
        self._volume_table()
        _log(f"crypto: {len(series):,} entities from {len(syms):,} pairs")

    def _volume_table(self) -> None:
        """Mean USDT volume over the last volume_days bars and bars listed, by date x entity."""
        if not self._series:
            self.vol, self.hist = pd.DataFrame(), pd.DataFrame()
            return
        qv = pd.DataFrame(
            {e: s.set_index("time")["quote_volume"] for e, s in self._series.items()}
        ).sort_index()
        qv = qv.asfreq("D")
        has = qv.notna()
        self.vol = qv.rolling(self.volume_days, min_periods=self.volume_days // 2).mean()
        self.hist = has.cumsum()
        pos = np.where(has.to_numpy(), np.arange(len(has))[:, None], np.nan)
        self.last_bar = pd.DataFrame(pos, index=has.index, columns=has.columns).ffill()
        self.bar_closes = self.calendar.bar_close(qv.index)

    # ---------- the contract ----------
    def universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Top entities by recent USDT volume, from bars closed before as_of."""
        s = self.series
        if not s:
            return pd.DataFrame(columns=["entity_id"])
        i = int(self.bar_closes.searchsorted(as_of, side="left")) - 1  # last day closed before
        if i < 0:
            return pd.DataFrame(columns=["entity_id"])
        vol, hist, last = self.vol.iloc[i], self.hist.iloc[i], self.last_bar.iloc[i]
        ok = (hist >= self.min_history) & (i - last <= 3) & vol.notna()
        pick = vol[ok].sort_values(ascending=False).head(self.top)
        out = pd.DataFrame({"entity_id": pick.index.astype(str)})
        if self.sector is not None:
            out["group"] = out["entity_id"].str.split("~").str[0].map(self.sector).fillna("Other")
        return out

    def labels(self, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forward returns: enter at the first close after the decision, exit `horizon` bars later."""
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            s = self.series[e]
            adj = 0.0 if e.split("~")[0] in self.renamed else self.delisted_return
            lab = forward_returns(
                pd.DatetimeIndex(s["close_at"]),
                s["close"].astype(float).to_numpy(),
                pd.DatetimeIndex(g["decision_time"]),
                horizon,
                self.ended[e],
                adj,
            )
            out.append(lab.assign(entity_id=e))
        lab = pd.concat(out, ignore_index=True)
        for col in ("decision_time", "entry_time", "label_end"):
            lab[col] = pd.to_datetime(lab[col], utc=True)
        lab["delisted"] = lab["delisted"].fillna(False).astype(bool)
        return lab

    def cost_bps(self, rows: pd.DataFrame) -> pd.Series:
        """2 x (taker fee + square-root impact), from the last impact_days bars before the decision."""
        out = pd.Series(np.nan, index=rows.index)
        n = self.impact_days
        for e, g in rows.groupby("entity_id", sort=False):
            s = self.series[e]
            lc = np.log(s["close"].astype(float).to_numpy())
            qv = s["quote_volume"].astype(float).to_numpy()
            pos = pd.DatetimeIndex(s["close_at"]).searchsorted(
                pd.DatetimeIndex(g["decision_time"]), side="left"
            )
            for idx, i in zip(g.index, pos):
                if i < n + 1:
                    continue
                sd, adv = np.std(np.diff(lc[i - n - 1 : i]), ddof=1), np.nanmean(qv[i - n : i])
                if adv > 0:
                    out[idx] = 2 * (self.fee_bps + 1e4 * sd * np.sqrt(self.trade_usd / adv))
        return out

    def ever_in_universe(self, start, end, schedule: str = "weekly") -> list[str]:
        """Every entity the universe holds at some decision in [start, end)."""
        times = self.calendar.decision_times(start, end, schedule)
        return sorted({e for t in times for e in self.universe(t)["entity_id"]})


# ---------- sources ----------
class _Base:
    def __init__(self, market: Crypto, params: dict | None = None):
        self.market, self.params = market, params or {}

    def observations(self, start, end) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} is computed at decision times: observations_at")

    def _bases(self, start, end) -> list[str]:
        ents = self.market.ever_in_universe(start, end, self.market.cfg.get("schedule", "weekly"))
        return sorted({e.split("~")[0] for e in ents})

    def _until(self, end) -> pd.Timestamp:
        end = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
        de = self.market.data_end
        return min(end, de) if de is not None else end


class PriceFeatures(_Base):
    """Momentum, volatility and dollar volume from bars closed before each decision."""

    name = "price"

    def fetch(self, start, end) -> None:
        """Daily bars of every kept USDT pair, delisted included, from FETCH_SINCE."""
        pairs = self.market.pairs()
        _log(f"  binance spot: {len(pairs):,} USDT pairs")
        binance.fetch_all("spot", pairs, self._until(end), since=FETCH_SINCE)
        self.market.reload()

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            s = self.market.series[e]
            c = s["close"].astype(float).to_numpy()
            qv = s["quote_volume"].astype(float).to_numpy()
            closes = pd.DatetimeIndex(s["close_at"])
            times = pd.DatetimeIndex(g["decision_time"].unique())
            for t, i in zip(times, closes.searchsorted(times, side="left")):
                if i < 85:  # bars 0..i-1 closed before the decision
                    continue
                r = np.diff(np.log(c[i - 29 : i]))
                vals = {
                    "px_mom_1w": c[i - 1] / c[i - 8] - 1,
                    "px_mom_4w": c[i - 1] / c[i - 29] - 1,
                    "px_mom_12w": c[i - 1] / c[i - 85] - 1,
                    "px_vol_4w": float(np.std(r, ddof=1)),
                    "px_log_dvol": float(np.log1p(np.nanmean(qv[i - 28 : i]))),
                }
                out += [(e, closes[i - 1], f, v) for f, v in vals.items()]
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


class Funding(_Base):
    """funding_7d from the token's USDT perp (or its 1000x contract, e.g. 1000SHIBUSDT)."""

    name = "funding"

    def perp(self, base: str, perps: set[str]) -> str | None:
        for p in (f"{base}USDT", f"1000{base}USDT"):
            if p in perps:
                return p
        return None

    def fetch(self, start, end) -> None:
        perps = binance.perp_symbols()
        syms = sorted({p for b in self._bases(start, end) if (p := self.perp(b, perps))})
        _log(f"  binance funding: {len(syms):,} perps")
        binance.fetch_all("funding", syms, self._until(end), since=FETCH_SINCE)

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        listing = binance.CACHE / "perp_symbols.txt"
        perps = set(listing.read_text().split()) if listing.exists() else set()
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            p = self.perp(e.split("~")[0], perps)
            f = binance.load("funding", p) if p else None
            if f is None or f.empty:
                continue
            if self.market.data_end is not None:
                f = f[f["time"] < self.market.data_end.tz_localize("UTC") + pd.Timedelta(days=1)]
            ft = pd.DatetimeIndex(f["time"])
            rate = f["rate"].to_numpy(float)
            for t in pd.DatetimeIndex(g["decision_time"].unique()):
                lo, hi = ft.searchsorted(t - pd.Timedelta(days=7)), ft.searchsorted(t, side="left")
                if hi - lo < 7:  # at least ~a third of the week's settlements
                    continue
                out.append((e, ft[hi - 1], "funding_7d", float(np.nanmean(rate[lo:hi]))))
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


class Fees(_Base):
    """Fee growth vs price growth over 90 days, from DeFiLlama daily fees."""

    name = "fees"

    def fetch(self, start, end) -> None:
        slugs = llama.token_fee_slugs(self._bases(start, end), max_age_s=86400)
        _log(f"  llama fees: {len(slugs):,} tokens mapped")
        llama.fetch_fees(slugs.values())

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        bases = sorted({e.split("~")[0] for e in rows["entity_id"]})
        slugs = llama.token_fee_slugs(bases)
        cal, out = self.market.calendar, []
        for e, g in rows.groupby("entity_id", sort=False):
            slug = slugs.get(e.split("~")[0])
            if not slug:
                continue
            fees = llama.daily_fees(slug)
            if self.market.data_end is not None:
                fees = fees[fees.index <= self.market.data_end]
            if fees.empty:
                continue
            fees = fees.asfreq("D", fill_value=0.0)
            usable = cal.bar_close(fees.index + pd.Timedelta(days=1))  # end of the next day
            f30 = fees.rolling(30, min_periods=30).sum().to_numpy()
            s = self.market.series[e]
            px = pd.Series(s["close"].astype(float).to_numpy(), index=s["time"])
            px = px.asfreq("D").ffill()
            for t in pd.DatetimeIndex(g["decision_time"].unique()):
                j = int(usable.searchsorted(t, side="left")) - 1  # latest usable fee day
                if j < 120:
                    continue
                now, then = f30[j], f30[j - 90]
                day, day0 = fees.index[j], fees.index[j - 90]
                if not (now > 0 and then > 0) or day not in px.index or day0 not in px.index:
                    continue
                growth = float(np.log(now / then))
                dprice = float(np.log(px[day] / px[day0]))
                at = usable[j]  # the fee day's close is later than its price close
                out += [
                    (e, at, "fee_growth_90d", growth),
                    (e, at, "fee_vs_price_90d", growth - dprice),
                ]
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


class Hacks(_Base):
    """hack_usd_30d from DeFiLlama's hacks list."""

    name = "hacks"

    def fetch(self, start, end) -> None:
        h = llama.hacks(set(self._bases(start, end)), max_age_s=86400)
        _log(f"  llama hacks: {len(h):,} hacks of universe tokens")

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        bases = {e.split("~")[0] for e in rows["entity_id"]}
        h = llama.hacks(bases)
        if self.market.data_end is not None:
            h = h[h["date"] <= self.market.data_end]
        h = h.assign(at=self.market.calendar.bar_close(h["date"] + pd.Timedelta(days=1)))
        by = {b: g for b, g in h.groupby("symbol")}
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            hb = by.get(e.split("~")[0])
            for t in pd.DatetimeIndex(g["decision_time"].unique()):
                usd, at = 0.0, t - pd.Timedelta(days=1)  # nothing known: stamped a day before
                if hb is not None:
                    w = hb[(hb["at"] < t) & (hb["at"] >= t - pd.Timedelta(days=30))]
                    if len(w):
                        usd, at = float(w["usd"].sum()), w["at"].max()
                out.append((e, at, "hack_usd_30d", float(np.log1p(usd))))
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


class Growth(_Base):
    """90-day growth of 30-day activity sums: USDT trading volume (from the bars) and DEX volume
    (the token's protocol, else its chain, from DeFiLlama; a day usable from the end of the next)."""

    name = "growth"

    def fetch(self, start, end) -> None:
        indicators.fetch(self._bases(start, end))

    @staticmethod
    def _growth(s: pd.Series, last: pd.Timestamp, floor: float) -> float:
        def window(d):
            w = s[(s.index > d - pd.Timedelta(days=30)) & (s.index <= d)]
            return w.sum() if len(w) >= 18 else np.nan  # at least 60% of the days

        a, b = window(last - pd.Timedelta(days=90)), window(last)
        return float(np.log(b / a)) if a > floor and b > floor else np.nan

    def observations_at(self, rows: pd.DataFrame) -> pd.DataFrame:
        cal, out = self.market.calendar, []
        for e, g in rows.groupby("entity_id", sort=False):
            b = self.market.series[e]
            qv = pd.Series(b["quote_volume"].astype(float).to_numpy(), index=b["time"])
            closes = pd.DatetimeIndex(b["close_at"])
            ind = indicators.series(e.split("~")[0])
            dex = ind.get("dex_volume", ind.get("chain_dex_volume"))
            if dex is not None and self.market.data_end is not None:
                dex = dex[dex.index <= self.market.data_end]
            for t in pd.DatetimeIndex(g["decision_time"].unique()):
                i = int(closes.searchsorted(t, side="left"))  # bars 0..i-1 closed before t
                if i > 120:
                    v = self._growth(qv.iloc[:i], b["time"].iloc[i - 1], 1e4)
                    if np.isfinite(v):
                        out.append((e, closes[i - 1], "px_volume_growth_90d", v))
                if dex is not None and len(dex):
                    usable = cal.bar_close(dex.index + pd.Timedelta(days=1))
                    j = int(usable.searchsorted(t, side="left"))
                    if j > 0:
                        v = self._growth(dex.iloc[:j], dex.index[j - 1], 1e3)
                        if np.isfinite(v):
                            out.append((e, usable[j - 1], "dex_volume_growth_90d", v))
        df = pd.DataFrame(out, columns=["entity_id", "available_at", "feature", "value"])
        return df.assign(source=self.name)[data.OBS_COLUMNS]


def build(cfg: dict):
    """The plug-in entry point."""
    return Crypto(cfg), {
        "price": PriceFeatures,
        "funding": Funding,
        "fees": Fees,
        "hacks": Hacks,
        "growth": Growth,
    }
