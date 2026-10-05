"""A toy market with a planted signal: runs anywhere, needs no data, and knows the right answer.

Every entity gets a fresh `planted` value z ~ N(0, 1) at each month-end decision. Over the next
month its daily returns drift by beta * z in total, on top of noise, so a correct pipeline must find
`planted` (positive rank IC, positive spreads) and must NOT find the `noise_*` features. Each
value is published at the decision day's close, i.e. just before the decision.

Optional `leak: true` adds `leaky`: the realised forward return, stamped AFTER the decision. A
correct panel builder never uses it (its cells stay empty); tests rely on that.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import calendar as calmod
from engine.market import forward_returns


class Synthetic:
    name = "synthetic"

    def __init__(self, cfg: dict):
        s = cfg.get("synthetic", {})
        self.fingerprint = json.dumps(s, sort_keys=True, default=str)
        self.cfg = cfg
        self.calendar = calmod.make(
            cfg.get("calendar", {"tz": "UTC", "close": "21:00"})
        )
        self.n = int(s.get("entities", 120))
        self.groups = int(s.get("groups", 4))
        self.beta = float(s.get("beta", 0.03))
        self.vol = float(s.get("daily_vol", 0.02))
        self.n_noise = int(s.get("noise_features", 3))
        self.cost = float(s.get("cost_bps", 20.0))
        self.leak = bool(s.get("leak", False))
        rng = np.random.default_rng(int(s.get("seed", 7)))
        days = pd.bdate_range(s.get("start", "2010-01-01"), s.get("end", "2016-12-31"))
        self.close = self.calendar.bar_close(days)
        self.decisions = self.calendar.decision_times(days[0], days[-1], "monthly")
        self.ids = [f"E{i:03d}" for i in range(self.n)]
        self.z = rng.standard_normal((len(self.decisions), self.n))
        noise = rng.standard_normal((len(days), self.n)) * self.vol
        # bar d belongs to the window after the latest decision before it
        k = self.decisions.searchsorted(self.close, side="left") - 1
        drift = np.zeros_like(noise)
        for m in range(len(self.decisions)):
            in_m = k == m
            if in_m.any():
                drift[in_m] = self.beta * self.z[m] / in_m.sum()
        self.prices = 100 * np.cumprod(1 + noise + drift, axis=0)
        self.noise_values = rng.standard_normal(
            (self.n_noise, len(self.decisions), self.n)
        )

    def universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "entity_id": self.ids,
                "group": [f"g{i % self.groups}" for i in range(self.n)],
            }
        )

    def labels(self, rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
        out = []
        for e, g in rows.groupby("entity_id", sort=False):
            j = self.ids.index(e)
            lab = forward_returns(
                self.close,
                self.prices[:, j],
                pd.DatetimeIndex(g["decision_time"]),
                horizon,
                False,
            )
            out.append(lab.assign(entity_id=e))
        return pd.concat(out, ignore_index=True)

    def cost_bps(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series(self.cost, index=rows.index)


class Signals:
    name = "signals"

    def __init__(self, market: Synthetic, params: dict | None = None):
        self.market, self.params = market, params or {}

    def fetch(self, start, end) -> None:
        pass  # generated in memory

    def observations(self, start, end) -> pd.DataFrame:
        m = self.market
        at = m.decisions - pd.Timedelta(
            m.calendar.decide_after_close
        )  # the decision day's close
        frames = []
        values = {"planted": m.z} | {
            f"noise_{i + 1}": m.noise_values[i] for i in range(m.n_noise)
        }
        for f, v in values.items():
            frames.append(
                pd.DataFrame(
                    {
                        "entity_id": np.tile(m.ids, len(at)),
                        "available_at": np.repeat(at, m.n),
                        "feature": f,
                        "value": v.ravel(),
                    }
                )
            )
        if m.leak:  # the future, stamped when it becomes known: after the decision
            rows = pd.DataFrame(
                [(e, t) for t in m.decisions for e in m.ids],
                columns=["entity_id", "decision_time"],
            )
            lab = m.labels(
                rows, int(m.cfg.get("labels", {}).get("horizon", 21))
            ).dropna(subset=["label_end"])
            frames.append(
                pd.DataFrame(
                    {
                        "entity_id": lab["entity_id"],
                        "available_at": lab["label_end"],
                        "feature": "leaky",
                        "value": lab["fwd_return"],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True).assign(source=self.name)


def build(cfg: dict):
    return Synthetic(cfg), {"signals": Signals}
