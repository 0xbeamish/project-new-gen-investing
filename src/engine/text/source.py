"""TextSource: documents + a question set + a reader -> observations, like any other source; and
the history base rates behind "surprise".

Market YAML:
    sources:
      earnings_text:
        type: text
        max_age_days: 400
        params:
          documents: sec_earnings_releases      # a name in the plug-in's DOCUMENT_SOURCES
          documents_params: {...}
          questions: markets/questions/earnings_release.yaml
          reader: {kind: keyword}               # keyword (free) | jev | claude
          step: text_read_jev                   # spend-ledger step for a paid reader
          prefix: earn
          change: {k: 1, min_prior: 1}          # or false
          surprise: {keys: [group]}             # or false
          features: [...]                       # optional subset
The plug-in exposes DOCUMENT_SOURCES = {name: factory(market, params)} and optionally
READERS = {kind: factory(market, params)} for readers only it can build.

Answers -> columns, per document and question (named <prefix>_<question>_<part>):
  level     scale: the expected level sum(k * P(k)); `spread` its standard deviation
  p         yes_no / probability: P(true)
  <option>  choice: P(option)
then per (entity, doc_type), in publication order:
  _chg      the value minus the mean of the entity's previous `k` documents of the same type
  _surp     the value minus its history base rate among STRICTLY earlier documents of the same
            type (and finer metadata keys, e.g. sector), shrunk by backoff
Every value is stamped with its document's available_at, so the panel's "latest value strictly
before the decision" rule applies unchanged. An unread document emits nothing.

History base rates: keys go coarse to fine ([], [doc_type], [doc_type, group] ...); a level is
skipped for a row whose key holds a missing value. Each level's mean is shrunk toward its parent,
mean_L = (n * mean_raw + K * mean_parent) / (n + K), so three examples barely move the prior and
three hundred dominate it. base_rates() gives outcome priors per event (the prior a text card must
beat in engine.text.grade); an outcome observed `lag` after its event is used only by targets
strictly after event time + lag, and leak_check() proves it.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np
import pandas as pd

from engine import data
from engine.text import questions as tq
from engine.text import read


# ---------------------------------------------------------------- answers -> observations
def answer_parts(q) -> list[str]:
    """The columns one question becomes."""
    return {"scale": ["level", "spread"], "choice": list(q.options)}.get(q.kind, ["p"])


def encode_answer(q, answer: dict | None) -> dict[str, float]:
    """One answer -> its numeric columns (NaN when unanswered)."""
    v = None if answer is None else answer.get("value")
    if q.kind in ("yes_no", "probability"):
        return {"p": float(v) if v is not None else np.nan}
    if q.kind == "scale":
        if not v:
            return {"level": np.nan, "spread": np.nan}
        probs = {int(k): float(p) for k, p in v.items()}
        tot = sum(probs.values()) or 1.0
        mean = sum(k * p for k, p in probs.items()) / tot
        spread = math.sqrt(sum(p * (k - mean) ** 2 for k, p in probs.items()) / tot)
        return {"level": mean, "spread": spread}
    if not v:
        return {o: np.nan for o in q.options}
    return {o: float(v.get(o, 0.0)) for o in q.options}


def answers_frame(
    docs: pd.DataFrame, answers: dict, qsets: dict[str, tq.QuestionSet], prefix: str
) -> pd.DataFrame:
    """One row per (doc_id, entity_id): entity_id, available_at, doc_type, doc_id, metadata keys,
    and every encoded answer column. Documents of a type with no question set are skipped."""
    rows = []
    for d in docs.to_dict("records"):
        qs = qsets.get(d["doc_type"])
        if qs is None:
            continue
        ans = answers.get((d["doc_id"], d["entity_id"]), {})
        if not ans:
            continue  # an unread document must not mask the previous one with NaNs
        row = {k: d[k] for k in ("entity_id", "available_at", "doc_type", "doc_id")}
        row |= {k: v for k, v in (d.get("metadata") or {}).items() if np.isscalar(v)}
        for q in qs:
            for part, val in encode_answer(q, ans.get(q.id)).items():
                row[f"{prefix}_{q.id}_{part}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _derived(col: str, tag: str) -> str:
    return col[: -len("_level")] + f"_{tag}" if col.endswith("_level") else f"{col}_{tag}"


def add_change(wide: pd.DataFrame, cols: list[str], k: int = 1, min_prior: int = 1) -> pd.DataFrame:
    """<col>_chg: the value minus the mean of the entity's previous k documents of the same type."""
    w = wide.sort_values("available_at", kind="stable")
    g = w.groupby(["entity_id", "doc_type"], sort=False)
    for c in cols:
        prior = g[c].transform(lambda s: s.shift(1).rolling(k, min_periods=min_prior).mean())
        w[_derived(c, "chg")] = w[c] - prior
    return w.sort_index()


def add_surprise(
    wide: pd.DataFrame, cols: list[str], keys: list[str] | None = None, k: float = 20
) -> pd.DataFrame:
    """<col>_surp: the value minus its backoff-shrunk base rate among strictly earlier documents.
    keys: metadata columns for finer levels, e.g. ["group"]: [] -> [doc_type] -> [doc_type, group]."""
    levels = [[], ["doc_type"]]
    for i in range(len(keys or [])):
        levels.append(["doc_type", *keys[: i + 1]])
    base = running_means(wide, cols, levels, at="available_at", k=k)
    w = wide.copy()
    for c in cols:
        w[_derived(c, "surp")] = w[c] - base[c]
    return w


def to_observations(
    wide: pd.DataFrame, source: str, features: list[str] | None = None
) -> pd.DataFrame:
    """The answer columns as observations stamped with each document's available_at."""
    if wide.empty:  # nothing read: no observations, not a crash
        return data.empty_observations()
    w = wide.sort_values("available_at", kind="stable")
    feats = features or [
        c
        for c in w.columns
        if c not in ("entity_id", "available_at", "doc_type", "doc_id")
        and pd.api.types.is_float_dtype(w[c])
    ]
    return data.from_wide(w, source, features=[f for f in feats if f in w])


def feature_meta(
    qsets: dict[str, tq.QuestionSet], prefix: str, change: bool, surprise: bool
) -> dict:
    """feature -> {doc_type, question, kind, tag, part, encoding}: how a column maps back to a question."""
    out = {}
    for dt, qs in qsets.items():
        for q in qs:
            for part in answer_parts(q):
                col = f"{prefix}_{q.id}_{part}"
                base = {
                    "doc_type": dt,
                    "question": q.id,
                    "kind": q.kind,
                    "tag": q.tag,
                    "part": part,
                    "version": q.version,
                }
                out[col] = base | {"encoding": "level"}
                if part in ("level", "p"):
                    if change:
                        out[_derived(col, "chg")] = base | {"encoding": "change"}
                    if surprise:
                        out[_derived(col, "surp")] = base | {"encoding": "surprise"}
    return out


def numeric_cols(qsets: dict[str, tq.QuestionSet], prefix: str) -> list[str]:
    """The columns change and surprise are computed for: scale levels and yes_no/probability p."""
    out = []
    for qs in qsets.values():
        for q in qs:
            if q.kind != "choice":
                out.append(f"{prefix}_{q.id}_{'level' if q.kind == 'scale' else 'p'}")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- history base rates
K, MIN_MEDIAN_N = 20, 20
MISSING = ("-", None, "", "nan")


class _Median:
    """Running median over a growing set (two heaps)."""

    def __init__(self):
        self.lo, self.hi = [], []

    def add(self, x: float) -> None:
        if not self.lo or x <= -self.lo[0]:
            heapq.heappush(self.lo, -x)
        else:
            heapq.heappush(self.hi, x)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def get(self) -> float:
        if not self.lo:
            return np.nan
        return -self.lo[0] if len(self.lo) > len(self.hi) else (-self.lo[0] + self.hi[0]) / 2


def level_keys(row: dict, levels: list[list[str]]) -> list[str | None]:
    """One key per level for a row (None where the row's key holds a missing value)."""
    out = []
    for cols in levels:
        vals = [row[c] for c in cols]
        if any((v in MISSING) or (isinstance(v, float) and np.isnan(v)) for v in vals):
            out.append(None)
        else:
            out.append("|".join(["L"] + [f"{c}={v}" for c, v in zip(cols, vals)]))
    return out


def base_rates(
    targets: pd.DataFrame,
    pool: pd.DataFrame,
    levels: list[list[str]],
    outcomes: dict[str, pd.Timedelta],
    at: str = "event_day",
    k: float = K,
    min_median_n: int = MIN_MEDIAN_N,
    medians: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Per target row: hist_n (finest level's n for the first outcome) and, per outcome o,
    hist_<o>_mean (shrunk), hist_<o>_hit (P(o > 0), shrunk toward 0.5 at the root),
    hist_<o>_med (medians for `medians`, default the first outcome), plus hist_<o>_last (the latest
    event time used, for leak_check)."""
    medians = medians if medians is not None else tuple(outcomes)[:1]
    res = pd.DataFrame(index=range(len(targets)))
    first = True
    for o, lag in outcomes.items():
        p = pool.dropna(subset=[o]).copy()
        p["_known"] = p[at] + lag
        p = p.sort_values("_known", kind="stable")
        rows = p.to_dict("records")
        stats: dict = {}  # key -> [n, sum, hits, median, last]
        order = targets.assign(_pos=range(len(targets))).sort_values(at, kind="stable")
        out = {}
        i = 0
        for t in order.to_dict("records"):
            when = t[at]
            while i < len(rows) and rows[i]["_known"] < when:
                r = rows[i]
                for key in level_keys(r, levels):
                    if key is None:
                        continue
                    s = stats.setdefault(key, [0, 0.0, 0, _Median(), pd.NaT])
                    s[0] += 1
                    s[1] += r[o]
                    s[2] += r[o] > 0
                    if o in medians:
                        s[3].add(r[o])
                    s[4] = r[at] if pd.isna(s[4]) else max(s[4], r[at])
                i += 1
            mean = hit = 0.0
            med, n_fine, last = np.nan, 0, pd.NaT
            for li, key in enumerate(level_keys(t, levels)):
                s = stats.get(key) if key else None
                if s is None or not s[0]:
                    continue
                n = s[0]
                mean = (s[1] + k * mean) / (n + k)
                hit = (s[2] + k * (hit if li else 0.5)) / (n + k)
                if o in medians and n >= min_median_n:
                    med = s[3].get()
                n_fine = n
                last = s[4] if pd.isna(last) else max(last, s[4])
            out[t["_pos"]] = (n_fine, mean, med, hit, last)
        cols = pd.DataFrame.from_dict(
            out,
            orient="index",
            columns=["n", "mean", "med", "hit", "last"],
        ).sort_index()
        if first:
            res["hist_n"] = cols["n"].to_numpy()
            first = False
        res[f"hist_{o}_mean"] = cols["mean"].to_numpy()
        if o in medians:
            res[f"hist_{o}_med"] = cols["med"].to_numpy()
        res[f"hist_{o}_hit"] = cols["hit"].to_numpy()
        res[f"hist_{o}_last"] = cols["last"].to_numpy()
    return pd.concat([targets.reset_index(drop=True), res], axis=1)


def leak_check(
    rates: pd.DataFrame, outcomes: dict[str, pd.Timedelta], at: str = "event_day"
) -> dict:
    """No base rate may use an outcome that wasn't fully observed before the target's time."""
    out, ok = {"events": len(rates)}, True
    for o, lag in outcomes.items():
        last = rates[f"hist_{o}_last"]
        bad = int((last.notna() & (last + lag >= rates[at])).sum())
        out[f"{o}_stats_using_unfinished_events"] = bad
        out[f"{o}_min_gap"] = str((rates[at] - last).min())
        ok &= bad == 0
    out["pass"] = bool(ok)
    return out


def running_means(
    frame: pd.DataFrame,
    cols: list[str],
    levels: list[list[str]],
    at: str = "available_at",
    k: float = K,
) -> pd.DataFrame:
    """For each row, the backoff-shrunk mean of each column over STRICTLY earlier rows (same key
    hierarchy as base_rates). NaN values don't count. Returns a frame aligned to `frame`."""
    f = frame.reset_index(drop=True)
    vals = f[cols].to_numpy(dtype=float)
    keys = [level_keys(r, levels) for r in f.to_dict("records")]
    times = f[at].to_numpy()
    order = np.argsort(times, kind="stable")
    stats: dict = {}  # key -> [n vector, sum vector]
    out = np.full_like(vals, np.nan)
    j = 0
    m = len(cols)
    while j < len(order):
        jj = j
        while jj < len(order) and times[order[jj]] == times[order[j]]:
            jj += 1
        block = order[j:jj]
        for r in block:  # read before adding: strictly earlier only
            mean = np.zeros(m)
            seen = np.zeros(m, dtype=bool)
            for key in keys[r]:
                s = stats.get(key) if key else None
                if s is None:
                    continue
                n, tot = s
                upd = n > 0
                mean = np.where(upd, (tot + k * mean) / np.maximum(n + k, 1e-12), mean)
                seen |= upd
            out[r] = np.where(seen, mean, np.nan)
        for r in block:
            v = vals[r]
            ok = ~np.isnan(v)
            for key in keys[r]:
                if key is None:
                    continue
                s = stats.setdefault(key, [np.zeros(m), np.zeros(m)])
                s[0] += ok
                s[1] += np.where(ok, v, 0.0)
        j = jj
    return pd.DataFrame(out, columns=cols, index=frame.index)


# ---------------------------------------------------------------- the source
class TextSource:
    """A Source over documents: reads them (cache, estimate, cap) and emits the answer columns."""

    def __init__(
        self,
        market,
        params: dict,
        plugin=None,
        ledger=None,
        cache_dir: Path | None = None,
        root: Path | None = None,
        name: str = "text",
    ):
        self.market, self.params, self.ledger = market, params or {}, ledger
        self.name = name
        p = self.params
        root = root or Path(".")
        docs = p["documents"]
        if isinstance(docs, str):
            factory = getattr(plugin, "DOCUMENT_SOURCES", {})[docs]
            self.documents = factory(market, p.get("documents_params", {}))
        else:
            self.documents = docs
        qpath = Path(p["questions"])
        self.qsets = tq.load(qpath if qpath.is_absolute() else root / qpath)
        rcfg = p.get("reader", {"kind": "keyword"})
        rcfg = {"kind": rcfg} if isinstance(rcfg, str) else rcfg
        plugin_readers = getattr(plugin, "READERS", {})
        self.reader = (
            plugin_readers[rcfg["kind"]](market, rcfg)
            if rcfg["kind"] in plugin_readers
            else read.make_reader(rcfg)
        )
        self.step = p.get("step", f"text_read_{self.reader.name}")
        self.prefix = p.get("prefix", "txt")
        self.change = p.get("change", {"k": 1, "min_prior": 1})
        self.surprise = p.get("surprise", {"keys": []})
        self.cache_path = (
            Path(p["cache"])
            if p.get("cache")
            else (cache_dir / f"text_answers_{self.name}.sqlite" if cache_dir else None)
        )
        self.answers_: dict = {}
        self.docs_: pd.DataFrame | None = None

    @property
    def cache_key(self) -> dict:
        """What the panel cache keys on: params plus question wording and the reader."""
        return self.params | {
            "_questions": {dt: qs.fingerprint() for dt, qs in self.qsets.items()},
            "_reader": [self.reader.name, getattr(self.reader, "model", "")],
        }

    def fetch(self, start, end) -> None:
        """The document source fills its cache."""
        self.documents.fetch(start, end)

    def estimate(self, start, end) -> dict:
        """What reading [start, end) would cost (only what the cache doesn't hold)."""
        docs = self.documents.documents(start, end)
        cache = read.AnswerCache(self.cache_path)
        return read.estimate(self.reader, docs, self.qsets, self.ledger, cache)

    def wide(self, start, end) -> pd.DataFrame:
        """Per-document answer columns (+ change and surprise), before the observation melt."""
        docs = self.documents.documents(start, end)
        self.docs_ = docs
        self.answers_ = read.read_all(
            self.reader, docs, self.qsets, self.ledger, self.step, self.cache_path
        )
        w = answers_frame(docs, self.answers_, self.qsets, self.prefix)
        if w.empty:
            return w
        nums = [c for c in numeric_cols(self.qsets, self.prefix) if c in w]
        if self.change:
            w = add_change(
                w,
                nums,
                int(self.change.get("k", 1)),
                int(self.change.get("min_prior", 1)),
            )
        if self.surprise:
            w = add_surprise(w, nums, self.surprise.get("keys") or [])
        return w

    def observations(self, start, end) -> pd.DataFrame:
        """Answer columns as observations."""
        w = self.wide(start, end)
        return to_observations(w, self.name, self.params.get("features"))

    def feature_meta(self) -> dict:
        """Column -> question metadata (what tracking, the loops and the cards read)."""
        return feature_meta(self.qsets, self.prefix, bool(self.change), bool(self.surprise))
