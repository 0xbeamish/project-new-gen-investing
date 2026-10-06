"""A long-horizon cohort test from the market YAML's `cohort_test:` block (engine cohort).

Signal: a fixed composite, no fitting. At each formation date, every input is ranked across the
names that have the `require`d inputs (percentile, ties averaged); a missing input gets
`missing_rank`; the score is the equal-weight mean of the ranks. Names without the required
inputs are not ranked but stay in the benchmark.

Portfolio (engine.cohorts): the top `top` of the ranked names (a fraction, or a count if >= 1),
equal weight, bought at the close of the first trading day after formation and held
`hold_quarters` formations; each day the portfolio is the mean of the cohorts held. Benchmark: the
same construction over the whole universe at each formation. Both net of measured costs: half a
round trip at entry (that month's spread) and half at exit (the exit month's; unknown -> that
date's median, stated). Statistic: monthly portfolio minus benchmark, net, Newey-West t.

  engine cohort --market us_largecap --coverage   input coverage per formation; no returns
  engine cohort --market us_largecap              the pre-registered design, descriptive (not logged)
  engine cohort --market us_largecap --log        the same, logged as ONE registry test
  engine cohort --market us_largecap --hold 8 | --top 10 | --input X --spread | --names A,B
                                                  descriptive variants (never logged)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from engine import cohorts, pipeline, scoring


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def composite(
    frame: pd.DataFrame, inputs: list[str], require: list[str], missing_rank: float
) -> pd.Series:
    """Equal-weight mean of per-formation percentile ranks; NaN where a required input is missing."""
    ok = frame[require].notna().all(axis=1)
    ranks = (
        frame.loc[ok, inputs]
        .groupby(frame.loc[ok, "decision_time"])
        .rank(pct=True)
        .fillna(missing_rank)
    )
    return ranks.mean(axis=1).reindex(frame.index)


def pick(frame: pd.DataFrame, score: str, top: float, bottom: bool = False) -> dict:
    """decision_time -> entity ids: the top (or bottom) `top` by score (fraction, or count if >= 1)."""
    out = {}
    for t, g in frame.dropna(subset=[score]).groupby("decision_time"):
        k = int(top) if top >= 1 else max(1, round(top * len(g)))
        g = g.sort_values(score, ascending=bottom, kind="stable")
        out[t] = list(g["entity_id"].head(k))
    return out


class Study:
    """Panel, returns and costs for one market's cohort test, loaded once."""

    def __init__(self, study):
        self.study = study
        self.cfg = study.cfg["cohort_test"]
        self.market = study.market
        self.cal = self.market.calendar
        first = pd.Timestamp(self.cfg["first_formation"])
        last = pd.Timestamp(self.cfg["last_formation"])
        self.panel = pipeline.build_panel(
            study, "tuning", sources=self.cfg.get("sources"), with_labels=False
        )  # point-in-time checked inside
        f = self.panel.frame
        local = self.cal.local_date(f["decision_time"])
        self.frame = f[((local >= first) & (local <= last)).to_numpy()].reset_index(
            drop=True
        )
        self.times = pd.DatetimeIndex(sorted(self.frame["decision_time"].unique()))
        self._rets = None

    def coverage(self) -> pd.DataFrame:
        f = self.frame
        cols = self.cfg["inputs"]
        cov = f.groupby("decision_time")[cols].apply(lambda g: g.notna().mean())
        cov["universe"] = f.groupby("decision_time").size()
        cov["ranked"] = (
            f.dropna(subset=self.cfg["require"]).groupby("decision_time").size()
        )
        cov.index = self.cal.local_date(pd.Series(cov.index)).dt.date.to_numpy()
        return cov

    def scored(self, inputs: list[str] | None = None) -> pd.DataFrame:
        c = self.cfg
        inputs = inputs or c["inputs"]
        return self.frame.assign(
            score=composite(
                self.frame, inputs, c["require"], float(c.get("missing_rank", 0.5))
            )
        )

    # ---------- returns and costs ----------
    def returns(self):
        if self._rets is None:
            codes = sorted(self.frame["entity_id"].unique())
            _log(f"cohort: daily returns for {len(codes):,} codes")
            self._rets = self.market.daily_returns(codes)
        return self._rets

    def _formation_dates(self, n_after: int) -> pd.DatetimeIndex:
        """Formation decision times, extended n_after quarters past the last (for exit dates)."""
        end = self.times[-1] + pd.DateOffset(months=3 * n_after + 3)
        sched = self.study.cfg.get("schedule", "BQE")
        more = self.cal.decision_times(self.times[0], end, sched)
        return pd.DatetimeIndex(sorted(set(self.times) | set(more)))

    def _cost(self, ids, when) -> pd.Series:
        """Half a round trip per id at decision time `when`; unknown -> that date's median."""
        rows = pd.DataFrame({"entity_id": list(ids), "decision_time": when})
        bps = pd.Series(self.market.cost_bps(rows).to_numpy(), index=list(ids))
        return 0.5 * bps.fillna(bps.median()).fillna(0.0) / 1e4

    def portfolio(self, members: dict, hold: int) -> pd.DataFrame:
        rets, ended = self.returns()
        days = rets.index
        sched = self._formation_dates(hold)
        paths = []
        for t, ids in members.items():
            local = self.cal.local_date(pd.Series([t])).iloc[0]
            after = days[days > local]
            if not len(after):
                continue  # entry after the data end
            entry = after[0]
            k = sched.get_loc(t) + hold
            exit_t = sched[k]
            exit_local = self.cal.local_date(pd.Series([exit_t])).iloc[0]
            later = days[days > exit_local]
            exit_day = later[0] if len(later) else None
            assert entry > local  # bought strictly after the formation date
            paths.append(
                cohorts.cohort_returns(
                    rets,
                    ids,
                    entry,
                    exit_day,
                    self._cost(ids, t),
                    self._cost(ids, exit_t) if exit_day is not None else None,
                    ended,
                )
            )
        return cohorts.monthly(cohorts.overlapping(paths))

    # ---------- the test ----------
    def run(self, top=None, hold=None, inputs=None, spread=False) -> dict:
        c = self.cfg
        top = float(top if top is not None else c["top"])
        hold = int(hold if hold is not None else c["hold_quarters"])
        s = self.scored(inputs)
        long = pick(s, "score", top)
        other = (
            pick(s, "score", top, bottom=True)
            if spread
            else {t: list(g["entity_id"]) for t, g in s.groupby("decision_time")}
        )  # benchmark: the whole universe, ranked or not
        p, b = self.portfolio(long, hold), self.portfolio(other, hold)
        m = p[["gross", "net"]].join(b[["gross", "net"]], rsuffix="_bench", how="inner")
        m["excess_gross"] = m["gross"] - m["gross_bench"]
        m["excess_net"] = m["net"] - m["net_bench"]
        lags = int(c.get("nw_lags", 12))
        full = m[p["cohorts"].reindex(m.index) >= hold]
        return {
            "months": len(m),
            "first_month": str(m.index.min()),
            "last_month": str(m.index.max()),
            "names_per_cohort": float(np.mean([len(v) for v in long.values()])),
            "excess_net": float(m["excess_net"].mean()),
            "excess_net_t_nw": scoring.newey_west_t(m["excess_net"], lags),
            "excess_net_t_iid": scoring.per_period_t(m["excess_net"]),
            "excess_gross": float(m["excess_gross"].mean()),
            "excess_gross_t_nw": scoring.newey_west_t(m["excess_gross"], lags),
            "port_net": float(m["net"].mean()),
            "bench_net": float(m["net_bench"].mean()),
            "cost_drag_port": float((m["gross"] - m["net"]).mean()),
            "tracking_error_monthly": float(m["excess_net"].std(ddof=1)),
            "full_ramp_months": len(full),
            "full_ramp_excess_net": float(full["excess_net"].mean())
            if len(full)
            else None,
            "full_ramp_t_nw": scoring.newey_west_t(full["excess_net"], lags)
            if len(full) > 2
            else None,
            "_monthly": m,
        }

    def ranks_of(self, codes: list[str], years: tuple[int, int]) -> pd.DataFrame:
        """Composite rank (1 = best) of the given codes at each formation in the years given."""
        s = self.scored()
        s = s.dropna(subset=["score"]).copy()
        s["rank"] = s.groupby("decision_time")["score"].rank(
            ascending=False, method="min"
        )
        s["ranked"] = s.groupby("decision_time")["score"].transform("size")
        top = float(self.cfg["top"])
        s["top_k"] = (top * s["ranked"]).round().clip(lower=1)
        local = self.cal.local_date(s["decision_time"])
        s = s[
            (local.dt.year >= years[0]).to_numpy()
            & (local.dt.year <= years[1]).to_numpy()
        ]
        out = []
        for code in codes:
            for t in self.times[
                (
                    self.cal.local_date(pd.Series(self.times)).dt.year >= years[0]
                ).to_numpy()
                & (
                    self.cal.local_date(pd.Series(self.times)).dt.year <= years[1]
                ).to_numpy()
            ]:
                row = s[(s["entity_id"] == code) & (s["decision_time"] == t)]
                inu = (
                    (self.frame["entity_id"] == code)
                    & (self.frame["decision_time"] == t)
                ).any()
                out.append(
                    {
                        "code": code,
                        "formation": self.cal.local_date(pd.Series([t])).iloc[0].date(),
                        "in_universe": bool(inu),
                        "rank": int(row["rank"].iloc[0]) if len(row) else None,
                        "of": int(row["ranked"].iloc[0]) if len(row) else None,
                        "top_decile": bool(
                            len(row) and row["rank"].iloc[0] <= row["top_k"].iloc[0]
                        ),
                        "score": round(float(row["score"].iloc[0]), 3)
                        if len(row)
                        else None,
                    }
                )
        return pd.DataFrame(out)


def log(study, res: dict, note: str) -> dict:
    c = study.cfg["cohort_test"]
    reg = study.registry
    if (reg.own()["name"] == c["name"]).any():
        raise SystemExit(f"{c['name']!r} is already in the registry: one look only")
    bar = reg.next_bar()
    t = res["excess_net_t_nw"]
    return reg.record(
        {
            "kind": "test",
            "name": c["name"],
            "features": ",".join(c["inputs"]),
            "scope": "universal",
            "metric": "monthly excess vs equal-weight universe, overlapping cohorts, net of "
            f"measured costs, Newey-West t (lag {c.get('nw_lags', 12)})",
            "t_tune": round(t, 3),
            "bar_tune": round(bar, 3),
            "check_used": False,
            "gain_tune": round(res["excess_net"], 6),
            "kept": bool(t >= bar and res["excess_net"] > 0),
            "note": note,
        }
    )
