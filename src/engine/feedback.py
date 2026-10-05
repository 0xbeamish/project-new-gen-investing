"""The decider's feedback note: what closed periods say about the model, its inputs, the text
questions and the decider's own overrides. On by default whenever a decider is configured.

Built fresh at every decision time from CLOSED data only (labels ended before the decision):
  model record     how the model's top pick did vs its batch average
  input record     per input, its rank IC with the group-adjusted return (steadiest first), so
                   the decider knows which lines on a card have earned trust
  text questions   per question: incremental value (the full model with vs without its fields)
                   and the keep / drop status that follows; dropped questions vanish from cards
  override record  the decider's own past departures from the model's top pick, and how they did,
                   split by which text fields led the card it picked

The note REPORTS. It never changes weights or questions (loop coordination rule 7): weights come
from the walk-forward, questions from the ladder.
"""

from __future__ import annotations

import string
import zlib

import numpy as np
import pandas as pd

from engine import decider as dmod
from engine import scoring

LETTERS = string.ascii_uppercase


def ic_table(rows: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Per decision period and input: rank IC with the group-adjusted return; plus each period's
    last label end (a period is closed for a decision only once all its labels have ended)."""
    adj = scoring.group_adjusted(rows)
    ranked = rows[features].groupby(rows["decision_time"]).rank(pct=True)
    target = adj.groupby(rows["decision_time"]).rank(pct=True)
    tab = {}
    for t, idx in rows.groupby("decision_time").groups.items():
        x, y = ranked.loc[idx], target.loc[idx]
        tab[t] = x.corrwith(y)  # Pearson on ranks = Spearman (ties averaged)
    return pd.DataFrame(tab).T.sort_index(), rows.groupby("decision_time")[
        "label_end"
    ].max()


def input_record(
    tab: pd.DataFrame, ends: pd.Series, t, min_periods: int = 12
) -> pd.DataFrame:
    """Per input, from periods closed before t: mean rank IC and its t."""
    closed = tab[ends.reindex(tab.index).to_numpy() < t]
    rows = []
    for c in closed.columns:
        ic = closed[c].dropna()
        if len(ic) >= min_periods:
            rows.append(
                {
                    "input": c,
                    "ic": float(ic.mean()),
                    "t": scoring.per_period_t(ic),
                    "periods": len(ic),
                }
            )
    return pd.DataFrame(rows, columns=["input", "ic", "t", "periods"])


def question_status(
    rec: pd.DataFrame, text_meta: dict, keep_t: float = 1.0, min_periods: int = 24
) -> pd.DataFrame:
    """Keep / drop per question from its fields' closed-period record (the strongest field counts).
    A question with < min_periods of history is kept (not enough evidence to drop it)."""
    rows = []
    by_q: dict[str, list] = {}
    for _, r in rec.iterrows():
        m = text_meta.get(r["input"])
        if m:
            by_q.setdefault(m["question"], []).append(r)
    for q, rs in by_q.items():
        best = max(rs, key=lambda r: abs(r["t"]))
        enough = best["periods"] >= min_periods
        rows.append(
            {
                "question": q,
                "best_field": best["input"],
                "ic": best["ic"],
                "t": best["t"],
                "status": "keep"
                if (not enough or abs(best["t"]) >= keep_t)
                else "drop",
            }
        )
    return pd.DataFrame(rows, columns=["question", "best_field", "ic", "t", "status"])


def note(
    model_hist: pd.Series,
    rec: pd.DataFrame,
    qstat: pd.DataFrame,
    past: pd.DataFrame,
    labels: dict,
    drivers: dict | None = None,
) -> str:
    lines = ["Track record (closed periods only):"]
    if len(model_hist) >= 12:
        lines.append(
            f"- The model's top pick beat its batch average by {model_hist.mean():+.2%} per period over "
            f"{len(model_hist)} periods ({(model_hist > 0).mean():.0%} of them)."
        )
    else:
        lines.append(
            "- The model has little track record yet; treat its score as weak."
        )
    if len(rec):
        strong = (
            rec[rec["t"].abs() >= 2].sort_values("t", key=abs, ascending=False).head(5)
        )
        for r in strong.itertuples():
            lines.append(
                f"- {labels.get(r.input, r.input)}: {('better' if r.ic > 0 else 'worse') + ' when it holds' if '==' in r.input else ('higher is better' if r.ic > 0 else 'higher is worse')} (IC {r.ic:+.3f}, t {r.t:.1f})"
            )
        weak = rec[rec["t"].abs() < 1]["input"].head(4).tolist()
        if weak:
            lines.append(
                "- No reliable signal so far from: "
                + "; ".join(labels.get(w, w) for w in weak)
            )
    if len(qstat):
        kept = qstat[qstat["status"] == "keep"]["question"].tolist()
        dropped = qstat[qstat["status"] == "drop"]["question"].tolist()
        lines.append(
            f"Text questions kept: {', '.join(kept) or 'none'}"
            + (f"; dropped for no record: {', '.join(dropped)}" if dropped else "")
        )
    if len(past):
        o = past[past["override"]]
        if len(o) >= 30:
            gain = (o["pick_excess"] - o["model_excess"]).mean()
            lines.append(
                f"Your past departures from the model's top pick: {len(o):,}; your pick did {abs(gain):.2%}/period "
                f"{'better' if gain > 0 else 'worse'} than the model's ({(o['pick_excess'] > o['model_excess']).mean():.0%} of the time)."
                + (" Depart only for the strongest lines above." if gain <= 0 else "")
            )
            if drivers:
                from engine.text.tracking import override_split

                sp = override_split(
                    past.assign(decision_time=past["decision_time"]), drivers
                )
                for r in sp[sp["overrides"] >= 20].sort_values("t").itertuples():
                    lines.append(
                        f"  - when '{labels.get(r.field, r.field)}' led the card you picked: {r.gain_vs_model:+.2%} vs the model ({r.overrides} times)"
                    )
    return "\n".join(lines)


def run(
    dec,
    scored: pd.DataFrame,
    rows: pd.DataFrame,
    contrib: pd.DataFrame,
    raw: pd.DataFrame,
    features: list[str],
    labels: dict,
    text_meta: dict,
    size: int = 10,
    feedback: bool = True,
    times: list | None = None,
    on_batch=None,
) -> pd.DataFrame:
    """Sequential over decision times (the override record must close before it is shown).

    scored: batched rows (entity_id, decision_time, group, fwd_return, score, batch, rt_cost)
    rows: model rows with inputs and label_end (for the closed-period record)
    contrib, raw: indexed like scored. Returns one row per batch: picks, gross and net excess."""
    out = []
    end_of = rows.set_index(["entity_id", "decision_time"])["label_end"]
    scored = scored.assign(
        label_end=[
            end_of.get((e, t))
            for e, t in zip(scored["entity_id"], scored["decision_time"])
        ]
    )
    excess = scored["fwd_return"] - scored.groupby("batch")["fwd_return"].transform(
        "mean"
    )
    top = scored.loc[scored.groupby("batch")["score"].idxmax()]
    if feedback:
        # a scale question can pay at its extremes with ~0 linear IC: score its levels too, so a
        # U-shaped question isn't dropped for "no record"
        levels = {}
        for c, m in text_meta.items():
            if c in rows and m.get("kind") == "scale" and m.get("encoding") == "level":
                lv = rows[c].round().clip(0, 4)
                for k in range(5):
                    levels[f"{c}=={k}"] = (lv == k).astype(float).where(rows[c].notna())
                    text_meta = text_meta | {
                        f"{c}=={k}": m | {"encoding": f"level {k}"}
                    }
                    labels = labels | {f"{c}=={k}": f"{labels.get(c, c)} = {k}"}
        tab, ends = ic_table(rows.assign(**levels), [*features, *levels])
    else:
        tab, ends = None, None
    model_hist_all = pd.DataFrame(
        {
            "t": top["decision_time"],
            "x": excess.loc[top.index].to_numpy(),
            "end": top["label_end"].to_numpy(),
        }
    )
    for t in sorted(times if times is not None else scored["decision_time"].unique()):
        g_t = scored[scored["decision_time"] == t]
        if g_t.empty:
            continue
        txt = ""
        drivers_past: dict = {}
        if feedback:
            mh = model_hist_all[model_hist_all["end"] < t].groupby("t")["x"].mean()
            rec = input_record(tab, ends, t)
            qstat = question_status(rec, text_meta)
            past = pd.DataFrame(out)
            if len(past):
                past = past[past["label_end"] < t]
                drivers_past = {b: d for b, d in zip(past["batch"], past["drivers"])}
            txt = note(mh, rec, qstat, past, labels, drivers_past)
            dropped_q = set(qstat.loc[qstat["status"] == "drop", "question"])
        else:
            dropped_q = set()
        hidden = {c for c, m in text_meta.items() if m.get("question") in dropped_q}
        jobs = []
        for b, g in g_t.groupby("batch"):
            if len(g) != size:
                continue
            g = g.sample(
                frac=1, random_state=zlib.crc32(str(b).encode())
            )  # letters don't follow the score
            ranks = g["score"].rank(ascending=False, method="first").astype(int)
            letters = LETTERS[: len(g)]
            cards = []
            for L, i in zip(letters, g.index):
                c = contrib.loc[i].drop(
                    labels=[x for x in hidden if x in contrib.columns]
                )
                cards.append(dmod.card(L, c, raw.loc[i], ranks[i], len(g), labels))
            job = {
                "batch": b,
                "decision_time": t,
                "letters": dict(zip(letters, g["entity_id"])),
                "returns": dict(zip(letters, g["fwd_return"])),
                "costs": dict(zip(letters, g["rt_cost"])),
                "model_pick": letters[list(ranks).index(1)],
                "state": {
                    "document_type": "ranking cards",
                    "group": str(g["group"].iloc[0]),
                    "reliability_note": txt,
                    "cards": "\n\n".join(cards),
                },
            }
            jobs.append((job, g, letters))
        if not jobs:
            continue
        if on_batch is not None:
            for job, _, _ in jobs:
                on_batch(job)
        if hasattr(
            dec, "pick_many"
        ):  # paid deciders: one guarded, recorded chunk per period
            all_probs = dec.pick_many([j for j, _, _ in jobs])
        else:
            all_probs = [dec.pick(j) for j, _, _ in jobs]
        for (job, g, letters), probs in zip(jobs, all_probs):
            if not probs:
                continue  # a failed call: the batch is skipped, not guessed
            b = job["batch"]
            pick = max(probs, key=probs.get)
            avg = float(np.mean(list(job["returns"].values())))
            avg_cost = float(np.mean(list(job["costs"].values())))
            i_pick = g.index[letters.index(pick)]
            top_text = [
                x
                for x in contrib.loc[i_pick]
                .abs()
                .sort_values(ascending=False)
                .index[:5]
                if x in text_meta
            ]
            out.append(
                {
                    "batch": b,
                    "decision_time": t,
                    "label_end": g["label_end"].max(),
                    "pick": job["letters"][pick],
                    "model_pick": job["letters"][job["model_pick"]],
                    "pick_excess": job["returns"][pick] - avg,
                    "model_excess": job["returns"][job["model_pick"]] - avg,
                    "pick_net": job["returns"][pick]
                    - job["costs"][pick]
                    - (avg - avg_cost),
                    "model_net": job["returns"][job["model_pick"]]
                    - job["costs"][job["model_pick"]]
                    - (avg - avg_cost),
                    "override": pick != job["model_pick"],
                    "drivers": tuple(top_text),
                    "note_chars": len(txt),
                }
            )
    return pd.DataFrame(out)


def grade(dec: pd.DataFrame) -> dict:
    """Monthly means, t from monthly returns: decider vs model (gross and net of measured costs)."""
    by = dec.groupby("decision_time")
    d, m = by["pick_excess"].mean(), by["model_excess"].mean()
    dn, mn = by["pick_net"].mean(), by["model_net"].mean()
    return {
        "periods": len(d),
        "batches": len(dec),
        "override_rate": float(dec["override"].mean()),
        "decider_vs_batch_gross": float(d.mean()),
        "model_vs_batch_gross": float(m.mean()),
        "decider_minus_model_gross": float((d - m).mean()),
        "decider_minus_model_gross_t": scoring.per_period_t(d - m),
        "decider_minus_model_net": float((dn - mn).mean()),
        "decider_minus_model_net_t": scoring.per_period_t(dn - mn),
        "decider_vs_batch_net_t": scoring.per_period_t(dn),
        "model_vs_batch_net_t": scoring.per_period_t(mn),
    }
