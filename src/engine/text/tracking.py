"""Tracking: is a text field under-weighted, over-weighted, mis-shaped, decaying or misread?

A free report (no LLM) the loops write after every closed period. Everything here reads only
CLOSED periods (labels ended before the cutoff) of the DIAGNOSIS entities; the frozen test split
is kept for accepting fixes (engine.improve).

1. residual attribution  each period, rank the model's residual (realized - predicted target) and
                         regress it on the ranked inputs, one at a time AND jointly. A slope with
                         the same sign as the field's weight = under-weighted; the opposite sign =
                         over-weighted. The JOINT slope decides: a numeric input that is mis-weighted
                         and correlated with a text field shows up on the text field one-at-a-time,
                         but not jointly. Mean, t, and rolling 12 / 24-period windows
2. calibration by level  per answer level: the realized partial residual (residual + the field's
                         own contribution) vs the model's implied effect (its contribution). A
                         linear weight on a 0-4 scale can be right on average and wrong at every
                         level; the de-linearized gaps catch that shape (e.g. a U)
3. rolling IC + decay    per-field rank IC by period, rolling 12 / 24, and the last 24 periods vs
                         the earlier ones (a field that stopped working)
4. reading quality       gold skill, spot-check corrections and probe stability per question:
                         separates "misread" from "mis-weighted"
5. decider overrides     the decider's departures from the model, split by which text fields
                         were the strongest lines on the card it picked (engine.decide.override_split,
                         passed in as `overrides`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine import model, score

FLAG_T = 2.0
CALIB_T = 2.5
DECAY_T = -2.0


def _crank(s: pd.Series, by: pd.Series) -> pd.Series:
    return s.groupby(by).rank(pct=True) - 0.5


def residual_frame(
    rows: pd.DataFrame, scored: pd.DataFrame, cutoff=None, entities=None
) -> pd.DataFrame:
    """rows (target, inputs, label_end) joined to the walk-forward scores; closed periods only."""
    keep = [c for c in rows.columns if c not in ("score", "block")]
    f = scored[["entity_id", "decision_time", "score", "block"]].merge(
        rows[keep], on=["entity_id", "decision_time"], how="inner"
    )
    f["y"] = f["fwd_rank"] - f.groupby(["decision_time", "group"])["fwd_rank"].transform("mean")
    f["resid"] = f["y"] - f["score"]
    if cutoff is not None:
        f = f[f["label_end"] < pd.Timestamp(cutoff)]
    if entities is not None:
        f = f[f["entity_id"].isin(entities)]
    return f.reset_index(drop=True)


def attribution(
    f: pd.DataFrame, features: list[str], weights: pd.DataFrame, overlap: int = 1
) -> pd.DataFrame:
    """Residual slopes per field, one at a time and jointly; status from the joint t vs the weight."""
    X = model.rank_features(f, features)
    r = _crank(f["resid"], f["decision_time"])
    uni, joint = {c: {} for c in features}, {c: {} for c in features}
    for t, idx in f.groupby("decision_time").groups.items():
        x = X.loc[idx].to_numpy()
        y = r.loc[idx].to_numpy()
        if len(y) < len(features) + 5:
            continue
        for j, c in enumerate(features):
            v = x[:, j].var()
            uni[c][t] = float(np.cov(x[:, j], y, bias=True)[0, 1] / v) if v > 0 else np.nan
        A = np.column_stack([np.ones(len(y)), x])
        coef = np.linalg.lstsq(A, y, rcond=None)[0][1:]
        for j, c in enumerate(features):
            joint[c][t] = float(coef[j])
    last_w = weights.iloc[-1] if len(weights) else pd.Series(0.0, index=features)
    rows = []
    for c in features:
        su, sj = pd.Series(uni[c]).sort_index(), pd.Series(joint[c]).sort_index()
        w = float(last_w.get(c, 0.0))
        tj = score.per_period_t(sj, overlap)
        status = "ok"
        if abs(tj) >= FLAG_T:
            if abs(w) < 1e-9:
                status = "missing"
            else:
                status = "under-weighted" if np.sign(sj.mean()) == np.sign(w) else "over-weighted"
        rows.append(
            {
                "field": c,
                "weight": w,
                "slope_one_at_a_time": float(su.mean()),
                "t_one_at_a_time": score.per_period_t(su, overlap),
                "slope_joint": float(sj.mean()),
                "t_joint": tj,
                "t_joint_12": score.per_period_t(sj.tail(12), overlap),
                "t_joint_24": score.per_period_t(sj.tail(24), overlap),
                "periods": len(sj),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def levels_of(x: pd.Series, kind: str) -> pd.Series:
    """Answer levels: 0-4 for a scale, 0/1 for a probability."""
    if kind == "scale":
        return x.round().clip(0, 4)
    return (x >= 0.5).astype(float).where(x.notna())


def calibration(
    f: pd.DataFrame,
    field: str,
    kind: str,
    features: list[str],
    weights: pd.DataFrame,
    calendar,
    overlap: int = 1,
) -> pd.DataFrame:
    """Per answer level: realized partial residual vs the model's implied effect, de-linearized."""
    contrib = model.contributions(f, features, weights, calendar)
    c = contrib[field].reindex(f.index)
    # take out what the OTHER inputs' mis-weights explain (joint, per period), so a mis-weighted
    # correlated input can't paint a shape on this field
    others = [x for x in features if x != field]
    X = model.rank_features(f, others)
    clean = f["resid"].copy()
    for idx in f.groupby("decision_time").groups.values():
        A = np.column_stack([np.ones(len(idx)), X.loc[idx].to_numpy()])
        b = np.linalg.lstsq(A, f.loc[idx, "resid"].to_numpy(), rcond=None)[0]
        clean.loc[idx] = f.loc[idx, "resid"].to_numpy() - A[:, 1:] @ b[1:]
    part = clean + c
    lev = levels_of(f[field], kind)
    g = pd.DataFrame(
        {
            "t": f["decision_time"],
            "lev": lev,
            "gap": part - c,
            "realized": part,
            "implied": c,
        }
    ).dropna()
    per = g.groupby(["t", "lev"])["gap"].mean().unstack("lev")
    # remove each period's linear trend across levels: what's left is shape, not scale
    dev = per.copy()
    for t, row in per.iterrows():
        ok = row.notna()
        if ok.sum() >= 3:
            k = np.array(row.index[ok], dtype=float)
            b = np.polyfit(k, row[ok].to_numpy(), 1)
            dev.loc[t, ok] = row[ok] - np.polyval(b, k)
        else:
            dev.loc[t] = np.nan
    out = []
    for k in per.columns:
        out.append(
            {
                "field": field,
                "level": k,
                "n": int((g["lev"] == k).sum()),
                "realized": float(g.loc[g["lev"] == k, "realized"].mean()),
                "implied": float(g.loc[g["lev"] == k, "implied"].mean()),
                "gap": float(per[k].mean()),
                "gap_t": score.per_period_t(per[k], overlap),
                "shape_dev": float(dev[k].mean()),
                "shape_t": score.per_period_t(dev[k], overlap),
            }
        )
    return pd.DataFrame(out)


def rolling_ic(f: pd.DataFrame, features: list[str], overlap: int = 1) -> pd.DataFrame:
    """Per-field rank IC, rolling 12 / 24, and decay: the last 24 periods vs the earlier ones."""
    rows = []
    for c in features:
        ic = (
            f.groupby("decision_time")
            .apply(
                lambda g, c=c: (
                    spearmanr(g[c], g["y"], nan_policy="omit").statistic
                    if g[c].nunique() > 1
                    else np.nan
                ),
                include_groups=False,
            )
            .dropna()
        )
        early, recent = ic.iloc[:-24], ic.tail(24)
        if len(early) >= 6 and len(recent) >= 6:
            se = np.sqrt(early.var() / len(early) + recent.var() / len(recent))
            # signed by the early IC: negative = the field's effect is shrinking toward zero
            dt = (
                float((recent.mean() - early.mean()) / se * np.sign(early.mean()))
                if se > 0
                else np.nan
            )
        else:
            dt = np.nan
        rows.append(
            {
                "field": c,
                "ic": float(ic.mean()),
                "ic_t": score.per_period_t(ic, overlap),
                "ic_12": float(ic.tail(12).mean()),
                "ic_24": float(ic.tail(24).mean()),
                "early": float(early.mean()) if len(early) else np.nan,
                "recent": float(recent.mean()),
                "decay_t": dt,
                "decaying": bool(not np.isnan(dt) and dt <= DECAY_T),
            }
        )
    return pd.DataFrame(rows)


def reading_quality(
    per_field_gold: dict,
    stability: dict,
    spotcheck_wrong: dict | None = None,
    skill_min: float = 0.3,
    stable_min: float = 0.8,
    wrong_max: float = 0.15,
) -> pd.DataFrame:
    """Per question: misread if gold skill, probe stability or spot-check errors are past their
    thresholds. Pass the result as `reading` to report() / Coordinator to separate "misread" from
    "mis-weighted"."""
    qids = sorted(set(per_field_gold) | set(stability) | set(spotcheck_wrong or {}))
    rows = []
    for q in qids:
        g = per_field_gold.get(q, {})
        skill = g.get("skill") if g.get("covered", True) else 0.0
        st = stability.get(q, np.nan)
        wr = (spotcheck_wrong or {}).get(q, np.nan)
        mis = (
            (skill is not None and skill < skill_min)
            or (not np.isnan(st) and st < stable_min)
            or (not np.isnan(wr) and wr > wrong_max)
        )
        rows.append(
            {
                "question": q,
                "gold_skill": skill,
                "stability": st,
                "spotcheck_wrong": wr,
                "misread": bool(mis),
            }
        )
    return pd.DataFrame(rows)


def report(
    rows: pd.DataFrame,
    scored: pd.DataFrame,
    weights: pd.DataFrame,
    features: list[str],
    text_meta: dict,
    calendar,
    cutoff=None,
    entities=None,
    reading: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
    overlap: int = 1,
) -> dict:
    """The full tracking report for one closed period. text_meta: feature -> {question, kind, ...}."""
    f = residual_frame(rows, scored, cutoff, entities)
    if f["decision_time"].nunique() < 3:  # nothing closed yet (the first out-of-sample months)
        cols = [
            "field",
            "status",
            "t_joint",
            "t_one_at_a_time",
            "decaying",
            "decay_t",
            "max_shape_t",
            "non_linear",
            "text",
            "question",
            "misread",
        ]
        empty = pd.DataFrame(columns=["field"])
        return {
            "periods": int(f["decision_time"].nunique()),
            "attribution": empty,
            "calibration": empty,
            "ic": empty,
            "reading": reading,
            "overrides": overrides,
            "flags": pd.DataFrame(columns=cols),
        }
    att = attribution(f, features, weights, overlap)
    ric = rolling_ic(f, features, overlap)
    cal = []
    for c in features:
        meta = text_meta.get(c)
        if (
            meta
            and meta.get("encoding") == "level"
            and meta.get("kind") in ("scale", "yes_no", "probability")
        ):
            cal.append(calibration(f, c, meta["kind"], features, weights, calendar, overlap))
    cal = pd.concat(cal, ignore_index=True) if cal else pd.DataFrame()
    shape = (
        cal.groupby("field")["shape_t"]
        .apply(lambda s: float(s.abs().max()))
        .rename("max_shape_t")
        .reset_index()
        if len(cal)
        else pd.DataFrame(columns=["field", "max_shape_t"])
    )
    flags = (
        att[["field", "status", "t_joint", "t_one_at_a_time"]]
        .merge(ric[["field", "decaying", "decay_t"]], on="field")
        .merge(shape, on="field", how="left")
    )
    flags["non_linear"] = flags["max_shape_t"].fillna(0) >= CALIB_T
    flags["text"] = flags["field"].isin(list(text_meta))
    flags["question"] = flags["field"].map(lambda c: (text_meta.get(c) or {}).get("question"))
    if reading is not None and len(reading):
        mis = dict(zip(reading["question"], reading["misread"]))
        flags["misread"] = flags["question"].map(lambda q: bool(mis.get(q, False)))
    else:
        flags["misread"] = False
    return {
        "periods": int(f["decision_time"].nunique()),
        "attribution": att,
        "calibration": cal,
        "ic": ric,
        "reading": reading,
        "overrides": overrides,
        "flags": flags,
    }
