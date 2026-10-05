"""Answers -> numeric columns -> observations for the normal panel.

Per document and question (columns named <prefix>_<question>_<part>):
  level     scale: the expected level sum(k * P(k)); `spread` its standard deviation
  p         yes_no / probability: P(true)
  <option>  choice: P(option)
Then, per (entity, doc_type), in publication order:
  change    `_chg`: the value minus the mean of the entity's previous `k` documents of the same type
            (k = 1: "vs the previous release"; the pilot used 4 with >= 2 required). Apple scores
            ~4.0 on "demand" every quarter; a 3.2 is a weak Apple quarter even though it reads
            "growing", and only the change says so
  surprise  `_surp`: the value minus its history base rate among STRICTLY earlier documents of the
            same type (and finer keys from metadata, e.g. sector), shrunk by backoff
            (engine.text.history.running_means)
Every value is stamped with its document's available_at, so the panel's "latest value strictly
before the decision" rule applies unchanged; change and surprise read only earlier documents.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine import observations as obsmod
from engine.text.history import running_means
from engine.text.questions import QuestionSet


def parts(q) -> list[str]:
    return {"scale": ["level", "spread"], "choice": list(q.options)}.get(q.kind, ["p"])


def encode(q, answer: dict | None) -> dict[str, float]:
    if (
        answer is not None and answer.get("encoded") is not None
    ):  # stored, already encoded
        return {p: float(answer["encoded"].get(p, np.nan)) for p in parts(q)}
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
    docs: pd.DataFrame, answers: dict, qsets: dict[str, QuestionSet], prefix: str
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
            for part, val in encode(q, ans.get(q.id)).items():
                row[f"{prefix}_{q.id}_{part}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _derived(col: str, tag: str) -> str:
    return (
        col[: -len("_level")] + f"_{tag}" if col.endswith("_level") else f"{col}_{tag}"
    )


def add_change(
    wide: pd.DataFrame, cols: list[str], k: int = 1, min_prior: int = 1
) -> pd.DataFrame:
    w = wide.sort_values("available_at", kind="stable")
    g = w.groupby(["entity_id", "doc_type"], sort=False)
    for c in cols:
        prior = g[c].transform(
            lambda s: s.shift(1).rolling(k, min_periods=min_prior).mean()
        )
        w[_derived(c, "chg")] = w[c] - prior
    return w.sort_index()


def add_surprise(
    wide: pd.DataFrame, cols: list[str], keys: list[str] | None = None, k: float = 20
) -> pd.DataFrame:
    """keys: metadata columns for finer levels, e.g. ["group"]: [] -> [doc_type] -> [doc_type, group]."""
    levels = [[], ["doc_type"]]
    for i in range(len(keys or [])):
        levels.append(["doc_type", *keys[: i + 1]])
    base = running_means(wide, cols, levels, at="available_at", k=k)
    w = wide.copy()
    for c in cols:
        w[_derived(c, "surp")] = w[c] - base[c]
    return w


def text_columns(wide: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in wide.columns if c.startswith(prefix + "_")]


def to_observations(
    wide: pd.DataFrame, source: str, features: list[str] | None = None
) -> pd.DataFrame:
    w = wide.sort_values("available_at", kind="stable")
    feats = features or [
        c
        for c in w.columns
        if c not in ("entity_id", "available_at", "doc_type", "doc_id")
        and pd.api.types.is_float_dtype(w[c])
    ]
    return obsmod.from_wide(w, source, features=[f for f in feats if f in w])


def feature_meta(
    qsets: dict[str, QuestionSet], prefix: str, change: bool, surprise: bool
) -> dict:
    """feature -> {doc_type, question, kind, tag, part, encoding}: how a column maps back to a question."""
    out = {}
    for dt, qs in qsets.items():
        for q in qs:
            for part in parts(q):
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


def numeric_cols(qsets: dict[str, QuestionSet], prefix: str) -> list[str]:
    """The columns change and surprise are computed for: scale levels and yes_no/probability p."""
    out = []
    for qs in qsets.values():
        for q in qs:
            if q.kind != "choice":
                out.append(f"{prefix}_{q.id}_{'level' if q.kind == 'scale' else 'p'}")
    return list(dict.fromkeys(out))
