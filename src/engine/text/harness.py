"""Score one (question set, reader) on the eval sets, end to end: read, then A-E, gates, S.

A plug-in builds EvalSets once (documents to read, answer key, probes, market set with its base
and prior columns); score() reads what it needs through the reading service (cache + spend cap)
and returns the metric suite. bootstrap_delta() compares two scores the way the loop must: paired,
resampling ENTITIES (all their documents and events together), out-of-fold predictions held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.text import evaluate as ev
from engine.text import features as tf
from engine.text import readers


@dataclass
class EvalSets:
    docs: (
        pd.DataFrame
    )  # doc_id, entity_id, doc_type, text, available_at: everything to read
    gold: list[dict]  # doc_id, doc_type, entity_id, split, labels, second
    probes: list[dict]  # probe, probe_id, orig_id, (target_field, want, may_move)
    market: (
        pd.DataFrame
    )  # doc_id, entity_id, split, year, z3, car3, abvol, drift_22, base + prior cols
    base_cols: list[str]
    prior_cols: list[str] = field(default_factory=list)
    leak_question: str | None = (
        None  # the outcome probe ("the stock rose sharply after this")
    )
    direction_question: str | None = (
        None  # the reader's own good/bad reading, for the leak gate
    )
    names: dict = field(
        default_factory=dict
    )  # entity -> name, for the identification test


def _split(sets: EvalSets, split: str):
    gold = [g for g in sets.gold if g["split"] == split]
    ids = {g["doc_id"] for g in gold}
    probes = [p for p in sets.probes if p["orig_id"] in ids]
    market = sets.market[sets.market["split"] == split].reset_index(drop=True)
    need = ids | {p["probe_id"] for p in probes} | set(market["doc_id"])
    docs = sets.docs[sets.docs["doc_id"].isin(need)]
    return gold, probes, market, docs


def market_features(
    market: pd.DataFrame, answers: dict, qsets: dict, skip: set[str]
) -> pd.DataFrame:
    """Encoded answers per market event (all questions, reading AND judgment: markets grade both)."""
    rows = []
    for d, e, dt in market[["doc_id", "entity_id", "doc_type"]].itertuples(index=False):
        a = answers.get((d, e), {})
        row = {}
        for q in qsets[dt]:
            if q.id in skip:
                continue
            for part, v in tf.encode(q, a.get(q.id)).items():
                row[f"{q.id}_{part}"] = v
        rows.append(row)
    return pd.DataFrame(rows, index=market.index).fillna(0.0)


def score(
    sets: EvalSets,
    qsets: dict,
    reader,
    split: str = "dev",
    ledger=None,
    step: str | None = None,
    cache_path=None,
    efficacy: bool = False,
    keep: bool = True,
) -> dict:
    gold, probes, market, docs = _split(sets, split)
    answers = readers.read_all(reader, docs, qsets, ledger, step, cache_path)
    by_doc = {}
    for (d, _e), a in answers.items():
        by_doc[d] = a
    preds = {
        d: {q: a["value"] for q, a in qa.items() if a.get("value") is not None}
        for d, qa in by_doc.items()
    }
    A, per_field = ev.gold_accuracy(gold, preds, qsets)
    judgment = {q.id for qs in qsets.values() for q in qs.judgment()}
    B = ev.consistency(probes, preds, judgment)
    E = ev.coverage(gold, preds, qsets, subject_field=None)
    if "doc_type" not in market:
        market = market.merge(
            docs[["doc_id", "doc_type"]].drop_duplicates("doc_id"),
            on="doc_id",
            how="left",
        )
    skip = {sets.leak_question} if sets.leak_question else set()
    feats = market_features(market, answers, qsets, skip)
    mk = market[
        ["z3", "car3", "abvol", "drift_22", "year", *sets.base_cols, *sets.prior_cols]
    ]
    C = ev.reaction_power(mk, feats, sets.base_cols, sets.prior_cols, keep_resid=True)
    out = {
        "summary": ev.summary(A, B["B"], C, E),
        "per_field": per_field,
        "consistency": {k: v for k, v in B.items() if k != "stability_by_field"},
        "stability_by_field": B["stability_by_field"],
        "reaction": {k: v for k, v in C.items() if not k.startswith("_")},
        "n": {
            "gold": len(gold),
            "probes": len(probes),
            "market": len(market),
            "docs": len(docs),
        },
        "cost_usd_per_1k_docs": 1000
        * readers.estimate(reader, docs, qsets, ledger)["usd"]
        / max(1, len(docs))
        if reader.paid
        else 0.0,
    }
    if sets.leak_question and sets.direction_question:
        dq = sets.direction_question
        lv = np.array(
            [
                tf.encode(qsets[t].get(dq), answers.get((d, e), {}).get(dq))["level"]
                for d, e, t in market[["doc_id", "entity_id", "doc_type"]].itertuples(
                    index=False
                )
            ]
        )
        lp = np.array(
            [
                tf.encode(
                    qsets[t].get(sets.leak_question),
                    answers.get((d, e), {}).get(sets.leak_question),
                )["p"]
                for d, e, t in market[["doc_id", "entity_id", "doc_type"]].itertuples(
                    index=False
                )
            ]
        )
        ok = ~(np.isnan(lv) | np.isnan(lp))
        up = (market["car3"].to_numpy() > 0).astype(int)
        out["gates"] = {"leak": ev.leak_gate(lv[ok], lp[ok], up[ok])}
    if efficacy:
        qs_all = {q.id: q for qs in qsets.values() for q in qs if q.id not in skip}
        out["efficacy"] = ev.efficacy(
            mk,
            feats,
            list(qs_all.values()),
            sets.base_cols,
            sets.prior_cols,
            per_field,
            B["stability_by_field"],
        )
    if keep:
        out["_keep"] = {
            "gold": gold,
            "probes": probes,
            "preds": preds,
            "qsets": qsets,
            "market": market,
            "C": C,
            "judgment": judgment,
        }
    return out


def _resample_score(k: dict, ents: list[str]) -> dict:
    """S recomputed on one entity resample: A, B, E re-evaluated; C, D from the fixed OOF fits."""
    count = pd.Series(ents).value_counts()
    gold = [g for g in k["gold"] for _ in range(int(count.get(g["entity_id"], 0)))]
    ent_of = {g["doc_id"]: g["entity_id"] for g in k["gold"]}
    probes = [
        p
        for p in k["probes"]
        for _ in range(int(count.get(ent_of.get(p["orig_id"]), 0)))
    ]
    A, _ = ev.gold_accuracy(gold, k["preds"], k["qsets"]) if gold else (0.0, {})
    B = ev.consistency(probes, k["preds"], k["judgment"])["B"] if probes else 0.0
    E = ev.coverage(gold, k["preds"], k["qsets"], subject_field=None) if gold else 0.0
    m = k["market"]
    idx = (
        np.concatenate([np.flatnonzero(m["entity_id"].to_numpy() == e) for e in ents])
        if ents
        else np.array([], int)
    )
    y = k["C"]["_y"]
    st = {}
    for model in ("prior", "card"):
        mag, zhat = k["C"]["_preds"][model]
        st[model] = ev.stats_from_preds(
            y["absz"][idx],
            y["z"][idx],
            y["react"][idx],
            y["drift"][idx],
            mag[idx],
            zhat[idx],
        )
    C = {
        "gain_vs_prior": {
            x: st["card"][x] - st["prior"][x]
            for x in ("R2_mag", "IC_react", "IC_drift")
        }
    }
    return ev.summary(A, B, C, E)


def bootstrap_delta(new: dict, cur: dict, n: int = 200, seed: int = 0) -> dict:
    """Paired bootstrap of S(new) - S(cur), resampling entities (blocked by entity)."""
    rng = np.random.default_rng(seed)
    ents = sorted(
        {g["entity_id"] for g in new["_keep"]["gold"]}
        | set(new["_keep"]["market"]["entity_id"])
    )
    deltas = []
    for _ in range(n):
        pick = list(rng.choice(ents, len(ents), replace=True))
        deltas.append(
            _resample_score(new["_keep"], pick)["S"]
            - _resample_score(cur["_keep"], pick)["S"]
        )
    d = np.array(deltas)
    return {
        "delta": new["summary"]["S"] - cur["summary"]["S"],
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "n": n,
    }
