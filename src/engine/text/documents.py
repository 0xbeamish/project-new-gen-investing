"""Documents: the one shape every text source emits, and the DocumentSource protocol.

Long format, one row per (document, entity):

  entity_id     the market's id for the entity the document is about
  available_at  UTC time from which a decision may use what the document says. The publication time
                or later (e.g. "usable from the next local midnight"). Never earlier
  doc_type      what kind of document (earnings_release, 8k, news, forum_post, ...). Question sets
                are chosen by doc_type, and "the previous document" means the previous one of the
                same type for the same entity
  doc_id        stable id, unique per document (one document may concern several entities)
  text          the text a reader sees. Mask it before it gets here if the source can identify
                the entity (engine.text.masking)
  metadata      optional dict (url, items, group, size bucket ...); keys used as history-prior keys

A DocumentSource is the text twin of engine.observations.Source:
  fetch(start, end)      fill the source's own cache (the only step that may use the network)
  documents(start, end)  read the cache: every document with available_at < end
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import pandas as pd

from engine.observations import utc

DOC_COLUMNS = ["entity_id", "available_at", "doc_type", "doc_id", "text"]


@runtime_checkable
class DocumentSource(Protocol):
    name: str

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> None: ...

    def documents(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def validate(docs: pd.DataFrame) -> pd.DataFrame:
    """Check the contract; returns a clean copy sorted by available_at (publication order)."""
    missing = [c for c in DOC_COLUMNS if c not in docs.columns]
    if missing:
        raise ValueError(f"documents missing columns {missing}")
    out = docs.copy()
    out["available_at"] = utc(out["available_at"]).to_numpy()
    if out["available_at"].isna().any():
        raise ValueError("documents with no available_at")
    out["entity_id"] = out["entity_id"].astype(str)
    out["doc_id"] = out["doc_id"].astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    if "metadata" not in out:
        out["metadata"] = [{} for _ in range(len(out))]
    if out.duplicated(["doc_id", "entity_id"]).any():
        raise ValueError("duplicate (doc_id, entity_id) rows")
    return out.sort_values("available_at", kind="stable").reset_index(drop=True)


def content_hash(*parts: str) -> str:
    """Cache key for a reading: same text + same questions + same reader -> same answers."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


class FrameDocuments:
    """A DocumentSource over an in-memory frame (tests, the demo, pre-built corpora)."""

    def __init__(self, name: str, frame: pd.DataFrame):
        self.name = name
        self.frame = validate(frame)

    def fetch(self, start, end) -> None:
        pass

    def documents(self, start, end) -> pd.DataFrame:
        return self.frame[
            self.frame["available_at"]
            < utc(
                pd.Timestamp(end, tz="UTC")
                if pd.Timestamp(end).tzinfo is None
                else pd.Timestamp(end)
            )
        ].reset_index(drop=True)
