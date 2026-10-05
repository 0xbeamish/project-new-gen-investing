"""Text evaluation harness: is a question set (read by a given reader) worth its cost?

Three sets with three jobs, all on tuning-period documents, split dev / test by ENTITY (no entity
on both sides, so a loop can't learn one company's quirks):

  answer key (gold)  double-labeled documents: does each READING field mean what its name says?
                     Judgment fields are never in the key. agreement() reports labeler agreement;
                     spotcheck_export() writes a JSON for a human yes/no check; apply_spotcheck()
                     lets the human override
  probes             paraphrase, alternative mask, order shuffles (text and question order), and
                     one-change counterfactual edits: answers must hold still, except the edited
                     field, which must move the right way
  market set         events with their reaction (e.g. 3-day abnormal return, in sigma units) and
                     later drift. reaction_power() fits out-of-fold models: base (event type +
                     size), prior (base + history base rates + looked-up context), card (prior +
                     the answers), and reports the card's GAIN over the prior

Summary (weights fixed before any loop runs):
  S = 0.30 A + 0.25 B + 0.30 C + 0.10 D + 0.05 E
  A gold accuracy, READING fields only (Brier skill for yes/no and choice, tolerance skill for
    scales), floored at 0 per field
  B consistency: mean of paraphrase, mask and order invariance and the counterfactual pass rate
  C reaction gain over the prior: mean(min(1, dR2_mag / 0.05), min(1, dIC_react / 0.10))
  D drift gain over the prior: min(1, dIC_drift / 0.05)
  E coverage: share of gold documents with every reading field answered (and about the entity)
Gates (fail = reject, whatever S): leak (outcome probe AUC beats the card's own good/bad AUC by >
0.05), name identification above chance, cost per 1,000 documents.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from engine.text.questions import QuestionSet

WEIGHTS = {"A": 0.30, "B": 0.25, "C": 0.30, "D": 0.10, "E": 0.05}
GAIN_CAPS = {"R2_mag": 0.05, "IC_react": 0.10, "IC_drift": 0.05}
LEAK_MAX_EXCESS = 0.05


def values(answers: dict) -> dict:
    """{key: {qid: answer dict}} -> {key: {qid: value}} (what the metrics compare)."""
    return {
        k: {q: a["value"] for q, a in qa.items() if a.get("value") is not None}
        for k, qa in answers.items()
    }


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
            fields.setdefault(f, []).append(
                (reading[f], y, preds.get(g["doc_id"], {}).get(f))
            )
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
            bs = np.mean(
                [sum((r[2].get(o, 0) - (o == r[1])) ** 2 for o in opts) for r in rows]
            )
            base = np.mean(
                [sum((freq.get(o, 0) - (o == r[1])) ** 2 for o in opts) for r in rows]
            )
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
        if any(
            g.get("second", {}).get(f) not in (None, y) for f, y in g["labels"].items()
        )
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


def apply_spotcheck(
    gold: list[dict], checks: list[dict]
) -> tuple[list[dict], list[dict], float]:
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
    ok = []
    for g in gold:
        p = preds.get(g["doc_id"], {})
        reading = [q.id for q in qsets[g["doc_type"]].reading()]
        subj = p.get(subject_field, 1.0) if subject_field else 1.0
        ok.append(
            all(f in p for f in reading)
            and (subj if isinstance(subj, float) else 1.0) >= 0.5
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
    ok = [
        _stable(a[f], b[f])
        for a, b in pairs
        for f in a
        if f in b and (fields is None or f in fields)
    ]
    return float(np.mean(ok)) if ok else 0.0


def stability_by_field(pairs: list[tuple[dict, dict]]) -> dict[str, float]:
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


def counterfactual(
    edits: list[dict], preds: dict, skip_fields: set[str] = frozenset()
) -> float:
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
            _stable(a[g], b[g])
            for g in a
            if g != f and g in b and g not in e.get("may_move", [])
        ]
        passed.append(
            _moved(a[f], b[f], tuple(e["want"]))
            and (np.mean(others) >= 0.9 if others else True)
        )
    return float(np.mean(passed)) if passed else 0.0


def consistency(
    probes: list[dict], preds: dict, skip_fields: set[str] = frozenset()
) -> dict:
    def pairs(kind):
        return [
            (preds[p["orig_id"]], preds[p["probe_id"]])
            for p in probes
            if p["probe"] == kind and p["orig_id"] in preds and p["probe_id"] in preds
        ]

    inv = {k: invariance(pairs(k)) for k in ("paraphrase", "mask_alt", "order")}
    cf = counterfactual(
        [p for p in probes if p["probe"] == "counterfactual"], preds, skip_fields
    )
    by_field = stability_by_field(
        pairs("paraphrase") + pairs("mask_alt") + pairs("order")
    )
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
        pred[te] = (
            Ridge(alpha=10.0).fit(m.loc[tr, cols], y[tr]).predict(m.loc[te, cols])
        )
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
    absz = m[z].abs().clip(upper=10).to_numpy()
    zz = m[z].clip(-10, 10).to_numpy()
    mag, zhat = _oof(m, cols, absz, fold), _oof(m, cols, zz, fold)
    r2 = 1 - np.mean((absz - mag) ** 2) / np.var(absz)
    resid = zz - zhat
    d = m[drift].clip(
        *m[drift].quantile([0.01, 0.99])
    )  # spread in % needs outliers capped
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
    g = C["gain_vs_prior"]
    c = (
        min(1, max(0, g["R2_mag"]) / GAIN_CAPS["R2_mag"])
        + min(1, max(0, g["IC_react"]) / GAIN_CAPS["IC_react"])
    ) / 2
    d = min(1, max(0, g["IC_drift"]) / GAIN_CAPS["IC_drift"])
    S = (
        WEIGHTS["A"] * A
        + WEIGHTS["B"] * B
        + WEIGHTS["C"] * c
        + WEIGHTS["D"] * d
        + WEIGHTS["E"] * E
    )
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
    chance = (
        float(np.mean([1 / max(1, len(g)) for g in guess_probs]))
        if guess_probs
        else 0.0
    )
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
    from engine.text.questions import Question

    return Question(
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
    return [c for c in feats.columns if c == qid or c.startswith(qid + "_")]


def efficacy(
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
    th = {"react": 0.005, "drift": 0.005, "r2": 0.002, "stable": 0.7} | (
        thresholds or {}
    )
    full = reaction_power(market, feats, base_cols, prior_cols, **kw)["card"]
    rows = []
    for q in questions:
        cols = question_columns(feats, q.id)
        if not cols:
            continue
        without = reaction_power(
            market, feats.drop(columns=cols), base_cols, prior_cols, **kw
        )["card"]
        g = {k: full[k] - without[k] for k in ("R2_mag", "IC_react", "IC_drift")}
        gold = per_field_gold.get(q.id, {})
        informative = (
            g["IC_react"] > th["react"]
            or g["IC_drift"] > th["drift"]
            or g["R2_mag"] > th["r2"]
        )
        reads = (
            q.tag == "judgment" or not gold.get("covered") or gold.get("skill", 0) > 0
        )
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
