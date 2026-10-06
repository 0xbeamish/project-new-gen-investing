"""Is each question worth asking? Score one (question set, reader) on fixed eval sets, end to end.

Three sets, all on tuning-period documents, split dev / test by ENTITY (no entity on both sides):
  answer key (gold)  double-labeled documents: does each READING field mean what its name says?
                     Judgment fields are never in the key. agreement() reports labeler agreement;
                     spotcheck_export() writes a JSON for a person's yes/no check; apply_spotcheck()
                     lets the person override
  probes             paraphrase, alternative mask, order shuffles (text and question order), and
                     one-change counterfactual edits: answers must hold still, except the edited
                     field, which must move the right way
  market set         events with their reaction (3-day abnormal return in sigma units) and later
                     drift. reaction_power() fits out-of-fold models: base (event type + size),
                     prior (base + history base rates), card (prior + the answers), and reports the
                     card's GAIN over the prior

Summary (weights fixed before any loop runs):
  S = 0.30 A + 0.25 B + 0.30 C + 0.10 D + 0.05 E
  A gold accuracy, READING fields only (Brier skill for yes/no and choice, tolerance skill for
    scales), floored at 0 per field
  B consistency: mean of paraphrase, mask and order invariance and the counterfactual pass rate
  C reaction gain over the prior: mean(min(1, dR2_mag / 0.05), min(1, dIC_react / 0.10))
  D drift gain over the prior: min(1, dIC_drift / 0.05)
  E coverage: share of gold documents with every reading field answered
Gates (fail = reject, whatever S): leak (the outcome probe's AUC beats the card's own good/bad AUC
by > 0.05), name identification above chance, cost per 1,000 documents.

A plug-in builds EvalSets once; score() reads what it needs through the reading service and
returns the suite; question_efficacy() adds a keep / drop per question; bootstrap_delta() compares two scores
the way the question loop must: paired, resampling ENTITIES, out-of-fold predictions held fixed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from engine.text import questions as tq
from engine.text import read
from engine.text.questions import QuestionSet
from engine.text.source import encode_answer

WEIGHTS = {"A": 0.30, "B": 0.25, "C": 0.30, "D": 0.10, "E": 0.05}
GAIN_CAPS = {"R2_mag": 0.05, "IC_react": 0.10, "IC_drift": 0.05}
LEAK_MAX_EXCESS = 0.05


# ---------------------------------------------------------------- A: the answer key
def _bss(p: np.ndarray, y: np.ndarray) -> float:
    bs = np.mean((p - y) ** 2)
    base = np.mean((y.mean() - y) ** 2)
    return 1 - bs / base if base > 0 else float(bs == 0)


def gold_accuracy(
    gold: list[dict], preds: dict, qsets: dict[str, QuestionSet]
) -> tuple[float, dict]:
    """gold: [{doc_id, doc_type, labels}]; preds: doc_id -> {qid: value}. READING fields only."""
    fields: dict[str, list] = {}
    for g in gold:
        qs = qsets[g["doc_type"]]
        reading = {q.id: q for q in qs.reading()}
        for f, y in g["labels"].items():
            if y is None or f not in reading:
                continue
            fields.setdefault(f, []).append((reading[f], y, preds.get(g["doc_id"], {}).get(f)))
    per = {}
    for f, rows in fields.items():
        q = rows[0][0]
        if any(r[2] is None for r in rows):
            per[f] = {"skill": 0.0, "covered": False, "n": len(rows)}
            continue
        if q.gold_kind == "bool":
            y = np.array([float(r[1]) for r in rows])
            p = np.array([float(r[2]) for r in rows])
            per[f] = {
                "skill": max(0.0, _bss(p, y)),
                "brier": float(np.mean((p - y) ** 2)),
                "covered": True,
                "n": len(rows),
            }
        elif q.gold_kind == "choice":
            opts = list(q.options)
            freq = pd.Series([r[1] for r in rows]).value_counts(normalize=True)
            bs = np.mean([sum((r[2].get(o, 0) - (o == r[1])) ** 2 for o in opts) for r in rows])
            base = np.mean([sum((freq.get(o, 0) - (o == r[1])) ** 2 for o in opts) for r in rows])
            per[f] = {
                "skill": max(0.0, 1 - bs / base) if base > 0 else 0.0,
                "brier": float(bs),
                "covered": True,
                "n": len(rows),
            }
        else:
            tol = q.tolerance
            pred = [max(r[2], key=r[2].get) for r in rows]
            y = [int(r[1]) for r in rows]
            acc = np.mean([abs(a - b) <= tol for a, b in zip(pred, y)])
            mode = pd.Series(y).mode().iloc[0]
            base = np.mean([abs(mode - b) <= tol for b in y])
            per[f] = {
                "skill": max(0.0, (acc - base) / (1 - base)) if base < 1 else 0.0,
                "acc": float(acc),
                "mae": float(np.mean([abs(a - b) for a, b in zip(pred, y)])),
                "covered": True,
                "n": len(rows),
            }
    return (float(np.mean([v["skill"] for v in per.values()])) if per else 0.0), per


def agreement(gold: list[dict], qsets: dict[str, QuestionSet]) -> dict:
    """Exact agreement between the two labelers per reading field (gold records carry `second`)."""
    hits: dict[str, list] = {}
    for g in gold:
        reading = {q.id for q in qsets[g["doc_type"]].reading()}
        for f, y in g["labels"].items():
            if f in reading and f in g.get("second", {}) and y is not None:
                hits.setdefault(f, []).append(g["second"][f] == y)
    per = {f: float(np.mean(v)) for f, v in hits.items()}
    allv = [x for v in hits.values() for x in v]
    return {"overall": float(np.mean(allv)) if allv else float("nan"), "per_field": per}


def spotcheck_export(
    gold: list[dict],
    docs: dict[str, str],
    qsets: dict[str, QuestionSet],
    path: Path,
    n_random: int = 20,
    n_flagged: int = 10,
    seed: int = 0,
) -> list[dict]:
    """A JSON a person answers yes/no per field: n random documents + up to n flagged disagreements."""
    rng = random.Random(seed)
    flagged = [
        g
        for g in gold
        if any(g.get("second", {}).get(f) not in (None, y) for f, y in g["labels"].items())
    ]
    rest = [g for g in gold if g not in flagged]
    pick = rng.sample(rest, min(n_random, len(rest))) + rng.sample(
        flagged, min(n_flagged, len(flagged))
    )
    out = []
    for g in pick:
        reading = {q.id: q for q in qsets[g["doc_type"]].reading()}
        out.append(
            {
                "doc_id": g["doc_id"],
                "doc_type": g["doc_type"],
                "text": docs.get(g["doc_id"], ""),
                "fields": [
                    {
                        "field": f,
                        "question": reading[f].text(),
                        "label": y,
                        "second": g.get("second", {}).get(f),
                        "evidence": g.get("evidence", {}).get(f),
                        "correct": None,  # the person fills true / false
                        "correction": None,  # optional: the right value
                    }
                    for f, y in g["labels"].items()
                    if f in reading
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


def apply_spotcheck(gold: list[dict], checks: list[dict]) -> tuple[list[dict], list[dict], float]:
    """The person's answers override the labels. Returns (gold, corrections, share judged wrong).
    Corrections feed the reader-fix rung of the ladder as few-shot examples."""
    by = {c["doc_id"]: c for c in checks}
    corrections, judged, wrong = [], 0, 0
    out = []
    for g in gold:
        g = json.loads(json.dumps(g))
        c = by.get(g["doc_id"])
        if c:
            for f in c["fields"]:
                if f["correct"] is None:
                    continue
                judged += 1
                if not f["correct"]:
                    wrong += 1
                    if f.get("correction") is not None:
                        g["labels"][f["field"]] = f["correction"]
                        corrections.append(
                            {
                                "doc_id": g["doc_id"],
                                "field": f["field"],
                                "was": f["label"],
                                "now": f["correction"],
                                "evidence": f.get("evidence"),
                            }
                        )
            g.setdefault("labeler", {})["user_checked"] = True
        out.append(g)
    return out, corrections, (wrong / judged if judged else float("nan"))


def coverage(
    gold: list[dict],
    preds: dict,
    qsets: dict[str, QuestionSet],
    subject_field: str | None = "is_subject",
) -> float:
    """E: share of gold documents with every reading field answered (and about the entity)."""
    ok = []
    for g in gold:
        p = preds.get(g["doc_id"], {})
        reading = [q.id for q in qsets[g["doc_type"]].reading()]
        subj = p.get(subject_field, 1.0) if subject_field else 1.0
        ok.append(
            all(f in p for f in reading) and (subj if isinstance(subj, float) else 1.0) >= 0.5
        )
    return float(np.mean(ok)) if ok else 0.0


# ---------------------------------------------------------------- B: probes
def _stable(a, b) -> bool:
    if isinstance(a, float):
        return abs(a - b) < 0.1
    if all(isinstance(k, int) for k in a):
        ea, eb = sum(k * p for k, p in a.items()), sum(k * p for k, p in b.items())
        return abs(ea - eb) < 0.5
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) < 0.1


def invariance(pairs: list[tuple[dict, dict]], fields: set[str] | None = None) -> float:
    """Share of (original, probe) answers that held still."""
    ok = [
        _stable(a[f], b[f])
        for a, b in pairs
        for f in a
        if f in b and (fields is None or f in fields)
    ]
    return float(np.mean(ok)) if ok else 0.0


def stability_by_field(pairs: list[tuple[dict, dict]]) -> dict[str, float]:
    """Invariance per field."""
    per: dict[str, list] = {}
    for a, b in pairs:
        for f in a:
            if f in b:
                per.setdefault(f, []).append(_stable(a[f], b[f]))
    return {f: float(np.mean(v)) for f, v in per.items()}


def _moved(a, b, want) -> bool:
    """want: ("up"/"down", threshold) for yes/no and scales, or ("to", option) for a choice."""
    kind, target = want
    if kind == "to":
        return max(b, key=b.get) == target and b.get(target, 0) > a.get(target, 0)
    sign = -1 if kind == "down" else 1
    if isinstance(a, float):
        return sign * (b - a) >= 0.1
    ea, eb = sum(k * p for k, p in a.items()), sum(k * p for k, p in b.items())
    return sign * (eb - ea) >= 0.5


def counterfactual(edits: list[dict], preds: dict, skip_fields: set[str] = frozenset()) -> float:
    """Target field moves the right way AND the other fields stay put (>= 90% of them)."""
    passed = []
    for e in edits:
        f = e["target_field"]
        if f in skip_fields:
            continue  # judgment fields aren't read off the text, so edits can't grade them
        a, b = preds.get(e["orig_id"], {}), preds.get(e["probe_id"], {})
        if f not in a or f not in b:
            passed.append(False)
            continue
        others = [
            _stable(a[g], b[g]) for g in a if g != f and g in b and g not in e.get("may_move", [])
        ]
        passed.append(
            _moved(a[f], b[f], tuple(e["want"])) and (np.mean(others) >= 0.9 if others else True)
        )
    return float(np.mean(passed)) if passed else 0.0


def consistency(probes: list[dict], preds: dict, skip_fields: set[str] = frozenset()) -> dict:
    """B: paraphrase, mask and order invariance and the counterfactual pass rate."""

    def pairs(kind):
        return [
            (preds[p["orig_id"]], preds[p["probe_id"]])
            for p in probes
            if p["probe"] == kind and p["orig_id"] in preds and p["probe_id"] in preds
        ]

    inv = {k: invariance(pairs(k)) for k in ("paraphrase", "mask_alt", "order")}
    cf = counterfactual([p for p in probes if p["probe"] == "counterfactual"], preds, skip_fields)
    by_field = stability_by_field(pairs("paraphrase") + pairs("mask_alt") + pairs("order"))
    return {
        "B": float(np.mean([inv["paraphrase"], inv["mask_alt"], inv["order"], cf])),
        "invariance": inv,
        "counterfactual_pass": cf,
        "unmasked_vs_masked_stability": invariance(pairs("unmasked")),
        "stability_by_field": by_field,
    }


# ---------------------------------------------------------------- C, D: market reaction and drift
def _oof(m: pd.DataFrame, cols: list[str], y: np.ndarray, fold: str) -> np.ndarray:
    """Out-of-fold Ridge predictions, folds = `fold` values (calendar years): a fair fit for every variant."""
    if not cols:
        return np.zeros(len(m))
    folds = m[fold].to_numpy()
    pred = np.full(len(m), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        pred[te] = Ridge(alpha=10.0).fit(m.loc[tr, cols], y[tr]).predict(m.loc[te, cols])
    return pred


def fit_stats(
    m: pd.DataFrame,
    cols: list[str],
    z: str,
    react: str,
    volume: str | None,
    drift: str,
    fold: str,
    drift_cost: float,
) -> dict:
    """Out-of-fold fit of one column set: R2 of |reaction|, reaction IC, drift IC of the residual."""
    absz = m[z].abs().clip(upper=10).to_numpy()
    zz = m[z].clip(-10, 10).to_numpy()
    mag, zhat = _oof(m, cols, absz, fold), _oof(m, cols, zz, fold)
    r2 = 1 - np.mean((absz - mag) ** 2) / np.var(absz)
    resid = zz - zhat
    d = m[drift].clip(*m[drift].quantile([0.01, 0.99]))  # spread in % needs outliers capped
    dec = pd.qcut(pd.Series(-resid).rank(method="first"), 10, labels=False).to_numpy()
    return {
        "R2_mag": float(r2),
        "IC_react": float(spearmanr(zhat, m[react]).statistic) if cols else 0.0,
        "IC_vol": float(spearmanr(mag, m[volume], nan_policy="omit").statistic)
        if cols and volume
        else 0.0,
        "IC_drift": float(spearmanr(-resid, m[drift]).statistic),
        "drift_spread_net": float(
            d.to_numpy()[dec == 9].mean() - d.to_numpy()[dec == 0].mean() - drift_cost
        ),
        "_resid": resid,
        "_mag": mag,
        "_zhat": zhat,
    }


def stats_from_preds(absz, z, react, drift, mag, zhat) -> dict:
    """R2_mag, IC_react, IC_drift from fixed out-of-fold predictions (for bootstrap resamples)."""
    r2 = 1 - np.mean((absz - mag) ** 2) / np.var(absz) if np.var(absz) > 0 else 0.0
    ic = spearmanr(zhat, react).statistic if np.std(zhat) > 0 else 0.0
    return {
        "R2_mag": float(r2),
        "IC_react": float(ic),
        "IC_drift": float(spearmanr(-(z - zhat), drift).statistic),
    }


def reaction_power(
    market: pd.DataFrame,
    feats: pd.DataFrame,
    base_cols: list[str],
    prior_cols: list[str] | None = None,
    z: str = "z3",
    react: str = "car3",
    volume: str | None = "abvol",
    drift: str = "drift_22",
    fold: str = "year",
    drift_cost: float = 0.012,
    keep_resid: bool = False,
) -> dict:
    """C and D: does the card explain the reaction, and leave drift, BEYOND its history prior?
    base = base_cols; prior = base + prior_cols (history base rates + context); card = prior + feats."""
    m = market.join(feats, how="left").fillna({c: 0.0 for c in feats.columns})
    prior = base_cols + (prior_cols or [])
    card = prior + list(feats.columns)
    args = (z, react, volume, drift, fold, drift_cost)
    out = {
        "base": fit_stats(m, base_cols, *args),
        "prior": fit_stats(m, prior, *args),
        "card": fit_stats(m, card, *args),
    }
    resid = {k: v.pop("_resid") for k, v in out.items()}
    preds = {k: (v.pop("_mag"), v.pop("_zhat")) for k, v in list(out.items())}
    out["gain_vs_prior"] = {
        k: out["card"][k] - out["prior"][k] for k in ("R2_mag", "IC_react", "IC_drift")
    }
    out["prior_vs_base"] = {
        k: out["prior"][k] - out["base"][k] for k in ("R2_mag", "IC_react", "IC_drift")
    }
    out["IC_drift_se"] = float(1 / np.sqrt(len(m) - 3))
    out["n_events"] = len(m)
    if keep_resid:
        out["_resid"] = resid
        out["_preds"] = preds
        out["_y"] = {
            "absz": m[z].abs().clip(upper=10).to_numpy(),
            "z": m[z].clip(-10, 10).to_numpy(),
            "react": m[react].to_numpy(),
            "drift": m[drift].to_numpy(),
        }
    return out


def summary(A: float, B: float, C: dict, E: float) -> dict:
    """S and its parts from A, B, the reaction fits and E (weights fixed in advance)."""
    g = C["gain_vs_prior"]
    c = (
        min(1, max(0, g["R2_mag"]) / GAIN_CAPS["R2_mag"])
        + min(1, max(0, g["IC_react"]) / GAIN_CAPS["IC_react"])
    ) / 2
    d = min(1, max(0, g["IC_drift"]) / GAIN_CAPS["IC_drift"])
    S = WEIGHTS["A"] * A + WEIGHTS["B"] * B + WEIGHTS["C"] * c + WEIGHTS["D"] * d + WEIGHTS["E"] * E
    return {"S": S, "A": A, "B": B, "C": c, "D": d, "E": E}


# ---------------------------------------------------------------- gates
def leak_gate(direction_level: np.ndarray, leak_p: np.ndarray, up: np.ndarray) -> dict:
    """The outcome probe ("the stock rose sharply after this") may not beat the reader's own
    good/bad reading by more than LEAK_MAX_EXCESS AUC: if it does, the reader knows the outcome."""
    a_dir, a_leak = roc_auc_score(up, direction_level), roc_auc_score(up, leak_p)
    return {
        "auc_direction": float(a_dir),
        "auc_leak_probe": float(a_leak),
        "excess": float(a_leak - a_dir),
        "pass": bool(a_leak - a_dir <= LEAK_MAX_EXCESS),
    }


def identification_gate(
    guess_probs: list[dict], truth: list[str], max_excess: float = 0.05
) -> dict:
    """Masked-name identification: the reader picks the masked entity from k candidates (truth +
    decoys). Accuracy above chance by more than `max_excess` means the masking leaks."""

    def hit(g, t):  # a tie at the top is a guess: credit 1 / (number tied)
        if not g:
            return 0.0
        top = max(g.values())
        tied = [k for k, v in g.items() if v == top]
        return (t in tied) / len(tied)

    hits = [hit(g, t) for g, t in zip(guess_probs, truth)]
    chance = float(np.mean([1 / max(1, len(g)) for g in guess_probs])) if guess_probs else 0.0
    acc = float(np.mean(hits)) if hits else 0.0
    se = float(np.sqrt(chance * (1 - chance) / max(1, len(hits))))
    return {
        "accuracy": acc,
        "chance": chance,
        "excess": acc - chance,
        "z": (acc - chance) / se if se else float("nan"),
        "pass": bool(acc - chance <= max_excess),
    }


def identification_question(candidates: list[str]):
    """A choice question: which of these candidates is COMPANY_A (for identification_gate)."""
    return tq.Question(
        id="identify_company",
        kind="choice",
        prompt="Which company is COMPANY_A in this text?",
        options=tuple(candidates),
        tag="judgment",
        evidence="none",
        keywords={c: [rf"\b{c}\b"] for c in candidates},
    )


# ---------------------------------------------------------------- per-question efficacy
def question_columns(feats: pd.DataFrame, qid: str) -> list[str]:
    """The feature columns one question produced."""
    return [c for c in feats.columns if c == qid or c.startswith(qid + "_")]


def question_efficacy(
    market: pd.DataFrame,
    feats: pd.DataFrame,
    questions: list,
    base_cols: list[str],
    prior_cols: list[str],
    per_field_gold: dict,
    stability: dict,
    thresholds: dict | None = None,
    **kw,
) -> pd.DataFrame:
    """Per question: reading skill (gold; reading fields only), probe stability, and the card's
    reaction / drift gain with vs without the question (leave-one-question-out, out-of-fold).
    Recommends keep / drop with thresholds fixed in advance."""
    th = {"react": 0.005, "drift": 0.005, "r2": 0.002, "stable": 0.7} | (thresholds or {})
    full = reaction_power(market, feats, base_cols, prior_cols, **kw)["card"]
    rows = []
    for q in questions:
        cols = question_columns(feats, q.id)
        if not cols:
            continue
        without = reaction_power(market, feats.drop(columns=cols), base_cols, prior_cols, **kw)[
            "card"
        ]
        g = {k: full[k] - without[k] for k in ("R2_mag", "IC_react", "IC_drift")}
        gold = per_field_gold.get(q.id, {})
        informative = (
            g["IC_react"] > th["react"] or g["IC_drift"] > th["drift"] or g["R2_mag"] > th["r2"]
        )
        reads = q.tag == "judgment" or not gold.get("covered") or gold.get("skill", 0) > 0
        stab = stability.get(q.id, float("nan"))
        rec = "keep" if informative and reads else "drop"
        why = []
        if not informative:
            why.append("adds nothing to reaction or drift")
        if not reads:
            why.append("misreads (gold skill 0)")
        if not np.isnan(stab) and stab < th["stable"]:
            why.append(f"unstable under probes ({stab:.2f})")
        rows.append(
            {
                "question": q.id,
                "tag": q.tag,
                "gold_skill": gold.get("skill") if q.tag == "reading" else None,
                "stability": stab,
                "gain_R2_mag": g["R2_mag"],
                "gain_IC_react": g["IC_react"],
                "gain_IC_drift": g["IC_drift"],
                "recommend": rec,
                "why": "; ".join(why) or "informative and read correctly",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the harness
@dataclass
class EvalSets:
    """Everything the harness reads, built once by a market plug-in's eval_sets()."""

    docs: pd.DataFrame  # doc_id, entity_id, doc_type, text, available_at: everything to read
    gold: list[dict]  # doc_id, doc_type, entity_id, split, labels, second
    probes: list[dict]  # probe, probe_id, orig_id, (target_field, want, may_move)
    market: (
        pd.DataFrame
    )  # doc_id, entity_id, split, year, z3, car3, abvol, drift_22, base + prior cols
    base_cols: list[str]
    prior_cols: list[str] = field(default_factory=list)
    leak_question: str | None = None  # the outcome probe ("the stock rose sharply after this")
    direction_question: str | None = None  # the reader's own good/bad reading, for the leak gate
    names: dict = field(default_factory=dict)  # entity -> name, for the identification test


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
            for part, v in encode_answer(q, a.get(q.id)).items():
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
    """Read the split's documents, then A-E, gates and S (and per-question efficacy if asked)."""
    gold, probes, market, docs = _split(sets, split)
    answers = read.read_all(reader, docs, qsets, ledger, step, cache_path)
    by_doc = {}
    for (d, _e), a in answers.items():
        by_doc[d] = a
    preds = {
        d: {q: a["value"] for q, a in qa.items() if a.get("value") is not None}
        for d, qa in by_doc.items()
    }
    A, per_field = gold_accuracy(gold, preds, qsets)
    judgment = {q.id for qs in qsets.values() for q in qs.judgment()}
    B = consistency(probes, preds, judgment)
    E = coverage(gold, preds, qsets, subject_field=None)
    if "doc_type" not in market:
        market = market.merge(
            docs[["doc_id", "doc_type"]].drop_duplicates("doc_id"),
            on="doc_id",
            how="left",
        )
    skip = {sets.leak_question} if sets.leak_question else set()
    feats = market_features(market, answers, qsets, skip)
    mk = market[["z3", "car3", "abvol", "drift_22", "year", *sets.base_cols, *sets.prior_cols]]
    C = reaction_power(mk, feats, sets.base_cols, sets.prior_cols, keep_resid=True)
    out = {
        "summary": summary(A, B["B"], C, E),
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
        * read.estimate(reader, docs, qsets, ledger)["usd"]
        / max(1, len(docs))
        if reader.paid
        else 0.0,
    }
    if sets.leak_question and sets.direction_question:
        dq = sets.direction_question
        lv = np.array(
            [
                encode_answer(qsets[t].get(dq), answers.get((d, e), {}).get(dq))["level"]
                for d, e, t in market[["doc_id", "entity_id", "doc_type"]].itertuples(index=False)
            ]
        )
        lp = np.array(
            [
                encode_answer(
                    qsets[t].get(sets.leak_question),
                    answers.get((d, e), {}).get(sets.leak_question),
                )["p"]
                for d, e, t in market[["doc_id", "entity_id", "doc_type"]].itertuples(index=False)
            ]
        )
        ok = ~(np.isnan(lv) | np.isnan(lp))
        up = (market["car3"].to_numpy() > 0).astype(int)
        out["gates"] = {"leak": leak_gate(lv[ok], lp[ok], up[ok])}
    if efficacy:
        qs_all = {q.id: q for qs in qsets.values() for q in qs if q.id not in skip}
        out["efficacy"] = question_efficacy(
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
    probes = [p for p in k["probes"] for _ in range(int(count.get(ent_of.get(p["orig_id"]), 0)))]
    A, _ = gold_accuracy(gold, k["preds"], k["qsets"]) if gold else (0.0, {})
    B = consistency(probes, k["preds"], k["judgment"])["B"] if probes else 0.0
    E = coverage(gold, k["preds"], k["qsets"], subject_field=None) if gold else 0.0
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
        st[model] = stats_from_preds(
            y["absz"][idx],
            y["z"][idx],
            y["react"][idx],
            y["drift"][idx],
            mag[idx],
            zhat[idx],
        )
    C = {
        "gain_vs_prior": {
            x: st["card"][x] - st["prior"][x] for x in ("R2_mag", "IC_react", "IC_drift")
        }
    }
    return summary(A, B, C, E)


def bootstrap_delta(new: dict, cur: dict, n: int = 200, seed: int = 0) -> dict:
    """Paired bootstrap of S(new) - S(cur), resampling entities (blocked by entity)."""
    rng = np.random.default_rng(seed)
    ents = sorted(
        {g["entity_id"] for g in new["_keep"]["gold"]} | set(new["_keep"]["market"]["entity_id"])
    )
    deltas = []
    for _ in range(n):
        pick = list(rng.choice(ents, len(ents), replace=True))
        deltas.append(
            _resample_score(new["_keep"], pick)["S"] - _resample_score(cur["_keep"], pick)["S"]
        )
    d = np.array(deltas)
    return {
        "delta": new["summary"]["S"] - cur["summary"]["S"],
        "lo": float(np.quantile(d, 0.025)),
        "hi": float(np.quantile(d, 0.975)),
        "n": n,
    }
