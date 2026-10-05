"""TextSource: documents + a question set + a reader -> observations, like any other Source.

Market YAML:
    sources:
      earnings_text:
        type: text
        max_age_days: 400
        params:
          documents: sec_earnings_releases      # a DocumentSource the plug-in registers
          documents_params: {...}
          questions: markets/questions/earnings_release.yaml
          reader: {kind: keyword}               # keyword (free) | jev | claude | a plug-in reader
          step: text_read_jev                   # spend-ledger step for a paid reader
          prefix: earn
          change: {k: 1, min_prior: 1}          # or false
          surprise: {keys: [group]}             # or false
          features: [...]                       # optional subset

The plug-in module exposes DOCUMENT_SOURCES = {name: factory(market, params)} and optionally
READERS = {kind: factory(market, params)} for readers only it can build (e.g. stored answers).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.text import features as tf
from engine.text import questions as tq
from engine.text import readers


class TextSource:
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
            else readers.make(rcfg)
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
        self.documents.fetch(start, end)

    def estimate(self, start, end) -> dict:
        docs = self.documents.documents(start, end)
        cache = readers.AnswerCache(self.cache_path)
        return readers.estimate(self.reader, docs, self.qsets, self.ledger, cache)

    def wide(self, start, end) -> pd.DataFrame:
        """Per-document answer columns (+ change and surprise), before the observation melt."""
        docs = self.documents.documents(start, end)
        self.docs_ = docs
        self.answers_ = readers.read_all(
            self.reader, docs, self.qsets, self.ledger, self.step, self.cache_path
        )
        w = tf.answers_frame(docs, self.answers_, self.qsets, self.prefix)
        if w.empty:
            return w
        nums = [c for c in tf.numeric_cols(self.qsets, self.prefix) if c in w]
        if self.change:
            w = tf.add_change(
                w,
                nums,
                int(self.change.get("k", 1)),
                int(self.change.get("min_prior", 1)),
            )
        if self.surprise:
            w = tf.add_surprise(w, nums, self.surprise.get("keys") or [])
        return w

    def observations(self, start, end) -> pd.DataFrame:
        w = self.wide(start, end)
        return tf.to_observations(w, self.name, self.params.get("features"))

    def feature_meta(self) -> dict:
        return tf.feature_meta(
            self.qsets, self.prefix, bool(self.change), bool(self.surprise)
        )
