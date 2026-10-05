"""A toy market whose returns depend on what its documents SAY: text scoring end to end, no data.

Each entity publishes a quarterly "release" (a templated text, masked) carrying latent facts:
  guidance   raised | maintained | lowered | none
  demand     a 0-4 level ("weak" ... "record")
  one_off    a one-time charge mentioned or not
  buyback    a buyback announced or not (noise: moves nothing)
plus a numeric input `x_value` per entity-month, correlated with demand by `rho_x_demand`.

Next month's return drifts by (per standard unit, `effects` in the YAML):
  guidance    +b * (raised) - b * (lowered)
  demand      linear b_lin * (d - 2) / 2  and/or  U-shaped b_u * (|d - 2| - 1)  (non-linear)
  one_off     b_one * one_off, switched to `one_off_after` from `regime_change` (a decaying field)
  x_value     b_x * x, switched to `x_after` from `regime_change` (a decaying NUMERIC input)
and each release moves the price on its first bar by `react` * (its good news) + noise.

Everything the eval harness needs is generated with it: an answer key (truth, a second labeler
with `label_noise` errors), probes (paraphrase = other templates with the same facts, alternative
mask, unmasked, sentence-order shuffles, one-fact counterfactual edits) and a market set (3-day
reaction in sigma units, 20-day drift). Some phrasings are deliberately missing from the keyword
reader's patterns, so the free reader is imperfect and the loops have something to fix.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from engine import calendar as calmod
from engine.market import forward_returns
from engine.text import masking

GUIDANCE = ["raised", "maintained", "lowered", "none"]
G_TEXT = {
    "raised": [
        "COMPANY_A raised its full-year guidance.",
        "The company increased its outlook for the year.",
        "Management now expects results above the prior outlook range.",  # missed by the reader
    ],
    "maintained": [
        "COMPANY_A reaffirmed its full-year guidance.",
        "The company maintained its outlook for the year.",
    ],
    "lowered": [
        "COMPANY_A lowered its full-year guidance.",
        "The company reduced its outlook for the year.",
        "Management now expects results below the prior outlook range.",  # missed by the reader
    ],
    "none": [
        "The company does not provide guidance.",
        "No outlook was given for the year.",
    ],
}
D_TEXT = {
    0: ["Demand was weak and declining.", "Orders fell sharply across regions."],
    1: ["Demand softened during the quarter.", "Customers delayed some orders."],
    2: ["Demand was stable.", "Order levels were in line with last year."],
    3: ["Demand grew during the quarter.", "Orders increased across regions."],
    4: [
        "Demand reached a record level.",
        "Orders were the strongest in company history.",
    ],
}
ONE_OFF = [
    "Results include a one-time charge for a legal settlement.",
    "A non-recurring expense lowered net income.",
]
BUYBACK = [
    "The board approved a new share repurchase program.",
    "COMPANY_A announced a stock buyback.",
]
FILLER = [
    "Revenue was reported in the attached tables.",
    "The company will host a conference call.",
    "Forward-looking statements involve risks.",
    "Employees continued to execute on the plan.",
]
NAMES = [
    "Acme",
    "Borealis",
    "Cindral",
    "Dunmore",
    "Elsworth",
    "Farrago",
    "Glimmer",
    "Halvard",
]
SUFFIX = ["Widgets Inc", "Systems Corp", "Holdings", "Industries Inc", "Labs Corp"]


def _name(i: int) -> str:
    return f"{NAMES[i % len(NAMES)]}{i // len(NAMES)} {SUFFIX[i % len(SUFFIX)]}"


def render(
    facts: dict, rng: np.random.Generator, name: str, order: list[int] | None = None
) -> str:
    """One release from its facts. Template choices come from rng, so another rng = a paraphrase."""
    lines = [
        f"{name} today reported results for the quarter.",
        G_TEXT[facts["guidance"]][rng.integers(len(G_TEXT[facts["guidance"]]))],
        D_TEXT[facts["demand"]][rng.integers(2)],
    ]
    if facts["one_off"]:
        lines.append(ONE_OFF[rng.integers(2)])
    if facts["buyback"]:
        lines.append(BUYBACK[rng.integers(2)].replace("COMPANY_A", name))
    lines.append(FILLER[rng.integers(len(FILLER))])
    if order is not None:
        head, rest = lines[:1], lines[1:]
        lines = head + [rest[i] for i in order if i < len(rest)]
    return " ".join(lines).replace("COMPANY_A", name)


def mask(text: str, name: str) -> str:
    return masking.mask(text, [name], None, extras=(), words=set())


class SyntheticText:
    name = "synthetic_text"

    def __init__(self, cfg: dict):
        s = cfg.get("synthetic_text", {})
        self.fingerprint = json.dumps(s, sort_keys=True, default=str)
        e = s.get("effects", {})
        self.cfg = cfg
        self.calendar = calmod.make(
            cfg.get("calendar", {"tz": "UTC", "close": "21:00"})
        )
        self.n = int(s.get("entities", 200))
        self.groups = int(s.get("groups", 4))
        self.vol = float(s.get("daily_vol", 0.02))
        self.cost = float(s.get("cost_bps", 20.0))
        self.label_noise = float(s.get("label_noise", 0.1))
        rng = np.random.default_rng(int(s.get("seed", 11)))
        days = pd.bdate_range(s.get("start", "2010-01-01"), s.get("end", "2016-12-31"))
        self.days = days
        self.close = self.calendar.bar_close(days)
        self.decisions = self.calendar.decision_times(days[0], days[-1], "monthly")
        self.ids = [f"T{i:03d}" for i in range(self.n)]
        self.names = {e_: _name(i) for i, e_ in enumerate(self.ids)}
        regime = pd.Timestamp(s.get("regime_change", "2099-01-01"), tz="UTC")

        # releases: one per entity per quarter, on a random business day of the quarter's 2nd month
        rows = []
        for i, ent in enumerate(self.ids):
            for q in pd.period_range(days[0], days[-1], freq="Q"):
                month = q.asfreq("M", "s") + 1
                bd = pd.bdate_range(month.start_time, month.end_time)
                day = bd[rng.integers(len(bd))]
                rows.append((ent, i, day))
        docs = pd.DataFrame(rows, columns=["entity_id", "idx", "day"])
        n = len(docs)
        x_ent = rng.standard_normal(self.n)  # persistent numeric input per entity...
        rho = float(s.get("rho_x_demand", 0.0))
        dz = rho * x_ent[docs["idx"]] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        docs["demand"] = np.digitize(dz, [-1.28, -0.52, 0.52, 1.28])  # 10/20/40/20/10 %
        docs["guidance"] = rng.choice(GUIDANCE, n, p=[0.25, 0.3, 0.2, 0.25])
        docs["one_off"] = rng.random(n) < 0.3
        docs["buyback"] = rng.random(n) < 0.3
        docs["doc_id"] = [f"r{k:06d}" for k in range(n)]
        docs["available_at"] = self.calendar.bar_close(docs["day"]) + pd.Timedelta(
            minutes=5
        )
        docs["template_seed"] = rng.integers(0, 2**31, n)
        self.docs = docs

        # numeric input: entity level + monthly noise
        m = len(self.decisions)
        self.x = x_ent[None, :] + 0.5 * rng.standard_normal((m, self.n))
        self.x_noise = rng.standard_normal((m, self.n))

        # each decision's drift from the latest release before it
        latest = np.full((m, self.n), -1)
        for j, ent in enumerate(self.ids):
            at = docs.loc[docs["entity_id"] == ent, "available_at"].to_numpy()
            pos = np.searchsorted(at, self.decisions.to_numpy(), side="left") - 1
            ids = docs.index[docs["entity_id"] == ent].to_numpy()
            latest[:, j] = np.where(pos >= 0, ids[np.clip(pos, 0, None)], -1)
        g = docs["guidance"].map({"raised": 1, "lowered": -1}).fillna(0).to_numpy()
        d = docs["demand"].to_numpy()
        o = docs["one_off"].to_numpy().astype(float)
        after = np.asarray(self.decisions >= regime)[:, None]
        b = {
            k: float(e.get(k, 0.0))
            for k in ("guidance", "demand_lin", "demand_u", "one_off", "x")
        }
        b_one = np.where(
            after, float(e.get("one_off_after", b["one_off"])), b["one_off"]
        )
        b_x = np.where(after, float(e.get("x_after", b["x"])), b["x"])
        has = latest >= 0
        li = np.clip(latest, 0, None)
        text_mu = (
            b["guidance"] * g[li]
            + b["demand_lin"] * (d[li] - 2) / 2
            + b["demand_u"] * (np.abs(d[li] - 2) - 1)
            + b_one * o[li]
        )
        self.mu = np.where(has, text_mu, 0.0) + b_x * self.x

        noise = rng.standard_normal((len(days), self.n)) * self.vol
        k = self.decisions.searchsorted(self.close, side="left") - 1
        drift = np.zeros_like(noise)
        for mm in range(m):
            in_m = k == mm
            if in_m.any():
                drift[in_m] = self.mu[mm] / in_m.sum()
        # the release's own price reaction on its first bar
        react = float(s.get("react", 0.03))
        good = g + (d - 2) / 2 - o
        bar = self.close.searchsorted(docs["available_at"], side="right")
        self.jump = react * good + 0.01 * rng.standard_normal(n)
        for r, (bi, j) in enumerate(zip(bar, docs["idx"])):
            if bi < len(days):
                noise[bi, j] += self.jump[r]
        self.prices = 100 * np.cumprod(1 + noise + drift, axis=0)
        self._doc_frame = None

    # ---------- market contract ----------
    def universe(self, as_of):
        return pd.DataFrame(
            {
                "entity_id": self.ids,
                "group": [f"g{i % self.groups}" for i in range(self.n)],
            }
        )

    def labels(self, rows, horizon):
        out = []
        for e_, grp in rows.groupby("entity_id", sort=False):
            j = self.ids.index(e_)
            lab = forward_returns(
                self.close,
                self.prices[:, j],
                pd.DatetimeIndex(grp["decision_time"]),
                horizon,
                False,
            )
            out.append(lab.assign(entity_id=e_))
        return pd.concat(out, ignore_index=True)

    def cost_bps(self, rows):
        return pd.Series(self.cost, index=rows.index)

    # ---------- documents ----------
    def facts(self, r) -> dict:
        return {k: r[k] for k in ("guidance", "demand", "one_off", "buyback")}

    def text(self, r, seed=None, masked=True, order=None, facts=None) -> str:
        name = self.names[r["entity_id"]]
        t = render(
            facts or self.facts(r),
            np.random.default_rng(seed if seed is not None else r["template_seed"]),
            name,
            order,
        )
        return mask(t, name) if masked else t

    def documents_frame(self) -> pd.DataFrame:
        if self._doc_frame is None:
            d = self.docs
            self._doc_frame = pd.DataFrame(
                {
                    "entity_id": d["entity_id"],
                    "available_at": d["available_at"],
                    "doc_type": "release",
                    "doc_id": d["doc_id"],
                    "text": [self.text(r) for r in d.to_dict("records")],
                    "metadata": [{"group": f"g{i % self.groups}"} for i in d["idx"]],
                }
            )
        return self._doc_frame

    # ---------- eval sets ----------
    def split(self, entity_id: str, test_share: float = 0.4) -> str:
        return (
            "test" if zlib.crc32(entity_id.encode()) % 100 < test_share * 100 else "dev"
        )

    def gold(self, n: int = 300, seed: int = 0, end=None) -> list[dict]:
        """Answer key: truth as labeler 1, truth with `label_noise` flips as labeler 2."""
        rng = np.random.default_rng(seed)
        d = (
            self.docs
            if end is None
            else self.docs[self.docs["available_at"] < pd.Timestamp(end, tz="UTC")]
        )
        pick = d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]
        out = []
        for r in pick.to_dict("records"):
            lab = {
                "guidance_action": r["guidance"],
                "demand": int(r["demand"]),
                "one_off_charge": bool(r["one_off"]),
                "buyback": bool(r["buyback"]),
            }
            second = dict(lab)
            if rng.random() < self.label_noise:
                second["guidance_action"] = GUIDANCE[rng.integers(4)]
            if rng.random() < self.label_noise:
                second["demand"] = int(
                    np.clip(lab["demand"] + rng.choice([-1, 1]), 0, 4)
                )
            out.append(
                {
                    "doc_id": r["doc_id"],
                    "doc_type": "release",
                    "entity_id": r["entity_id"],
                    "split": self.split(r["entity_id"]),
                    "labels": lab,
                    "second": second,
                }
            )
        return out

    def probes(
        self, gold: list[dict], seed: int = 0
    ) -> tuple[list[dict], pd.DataFrame]:
        """Probe records + their documents (masked texts), built by construction from the facts."""
        rng = np.random.default_rng(seed)
        by = self.docs.set_index("doc_id")
        out, texts = [], []
        edits = [
            (
                "guidance_action",
                {"guidance": "raised"},
                ("to", "raised"),
                lambda f: f["guidance"] == "none",
            ),
            ("demand", {"demand": 3}, ("up", 0.5), lambda f: f["demand"] == 1),
            (
                "one_off_charge",
                {"one_off": True},
                ("up", 0.1),
                lambda f: not f["one_off"],
            ),
        ]
        for g in gold:
            r = by.loc[g["doc_id"]].to_dict() | {"doc_id": g["doc_id"]}
            f = self.facts(r)
            base = {"orig_id": g["doc_id"], "family": "release"}
            name = self.names[r["entity_id"]]
            masked = self.text(r)
            variants = [
                ("paraphrase", self.text(r, seed=int(rng.integers(2**31)))),
                ("mask_alt", masking.alternative(masked)),
                ("unmasked", self.text(r, masked=False)),
                ("order", self.text(r, order=list(rng.permutation(6)))),
            ]
            for kind, t in variants:
                pid = f"{kind}_{g['doc_id']}"
                out.append(
                    base
                    | {"probe": kind, "probe_id": pid}
                    | ({"shuffle_questions": True} if kind == "order" else {})
                )
                texts.append((pid, r["entity_id"], t))
            for field, change, want, when in edits:
                if when(f):
                    pid = f"cf_{field}_{g['doc_id']}"
                    t = self.text(r, facts=f | change)
                    out.append(
                        base
                        | {
                            "probe": "counterfactual",
                            "probe_id": pid,
                            "target_field": field,
                            "want": list(want),
                            "may_move": [],
                        }
                    )
                    texts.append((pid, r["entity_id"], t))
            del name
        docs = pd.DataFrame(texts, columns=["doc_id", "entity_id", "text"]).assign(
            doc_type="release", available_at=pd.Timestamp("2000-01-01", tz="UTC")
        )
        return out, docs

    def market_set(self, end=None) -> pd.DataFrame:
        """Per release: 3-day reaction (car3, z3 in sigma units), abnormal volume, drift days 3-22."""
        d = (
            self.docs
            if end is None
            else self.docs[self.docs["available_at"] < pd.Timestamp(end, tz="UTC")]
        )
        rets = self.prices[1:] / self.prices[:-1] - 1
        rows = []
        bar = self.close.searchsorted(d["available_at"], side="right")
        for r, bi in zip(d.to_dict("records"), bar):
            j = r["idx"]
            if bi + 22 >= len(rets) or bi < 61:
                continue
            mkt = rets[bi - 1 : bi + 22].mean(axis=1)
            car3 = float((rets[bi - 1 : bi + 2, j] - mkt[:3]).sum())
            sigma = float(rets[bi - 61 : bi - 1, j].std())
            drift = float((rets[bi + 2 : bi + 22, j] - mkt[3:23]).sum())
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "entity_id": r["entity_id"],
                    "year": r["day"].year,
                    "split": self.split(r["entity_id"]),
                    "group": f"g{j % self.groups}",
                    "car3": car3,
                    "z3": car3 / (sigma * np.sqrt(3)),
                    "abvol": abs(car3) / sigma
                    + 0.3 * np.random.default_rng(j).standard_normal(),
                    "drift_22": drift,
                    "event_day": r["day"],
                }
            )
        return pd.DataFrame(rows)


class Numbers:
    """Numeric inputs: x_value (may carry a planted, decaying effect) and x_noise (nothing)."""

    name = "numbers"

    def __init__(self, market: SyntheticText, params: dict | None = None):
        self.market, self.params = market, params or {}

    def fetch(self, start, end) -> None:
        pass

    def observations(self, start, end) -> pd.DataFrame:
        m = self.market
        at = m.decisions - pd.Timedelta(m.calendar.decide_after_close)
        frames = []
        for f, v in {"x_value": m.x, "x_noise": m.x_noise}.items():
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
        return pd.concat(frames, ignore_index=True).assign(source=self.name)


class Releases:
    """The DocumentSource: masked release texts, generated in memory."""

    name = "releases"

    def __init__(self, market: SyntheticText, params: dict | None = None):
        self.market = market

    def fetch(self, start, end) -> None:
        pass

    def documents(self, start, end) -> pd.DataFrame:
        f = self.market.documents_frame()
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        return f[f["available_at"] < end].reset_index(drop=True)


DOCUMENT_SOURCES = {"releases": Releases}


def build(cfg: dict):
    return SyntheticText(cfg), {"numbers": Numbers}


def eval_sets(market: SyntheticText, end=None, n_gold: int = 300, seed: int = 0):
    """The text eval sets for this market (engine.text.harness.EvalSets), built by construction."""
    from engine.text import history
    from engine.text.harness import EvalSets

    gold = market.gold(n_gold, seed=seed, end=end)
    probes, pdocs = market.probes(gold, seed=seed)
    m = market.market_set(end=end).assign(doc_type="release")
    m["event_day"] = pd.to_datetime(m["event_day"])
    rates = history.base_rates(
        m,
        m,
        [[], ["group"]],
        {"car3": pd.Timedelta(days=5), "drift_22": pd.Timedelta(days=35)},
    )
    prior = ["hist_car3_mean", "hist_car3_hit", "hist_drift_22_mean"]
    m = pd.concat([m, rates[prior].fillna(0.0)], axis=1)
    base = pd.get_dummies(m["group"], prefix="grp").astype(float)
    m = pd.concat([m, base], axis=1)
    docs = market.documents_frame()[
        ["doc_id", "entity_id", "doc_type", "text", "available_at"]
    ]
    docs = pd.concat([docs, pdocs[docs.columns]], ignore_index=True)
    return EvalSets(
        docs=docs,
        gold=gold,
        probes=probes,
        market=m,
        base_cols=list(base.columns),
        prior_cols=prior,
        leak_question="leak_probe",
        direction_question="overall_tone",
        names=dict(market.names),
    )
