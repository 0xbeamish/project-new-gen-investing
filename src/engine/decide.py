"""Optional AI decider: an LLM picks one entity per batch of model cards, with a feedback note built
from closed periods. Off by default; the free NoDecider (the model's own pick) runs without keys.

Per batch (`size` entities of one group at one decision time):
  card        the model's score rank plus the inputs pushing the score most, each as
              contribution = block weight x the input's percentile rank (centred)
  note        rebuilt at every decision from CLOSED data only (labels ended before it): the model's
              pick record, each input's rank IC (a 0-4 scale is also scored level by level), each
              text question's keep / drop status (dropped questions vanish from the cards) and the
              decider's own override record split by the text fields that led the card it picked.
              The note REPORTS; it never changes weights or questions
  grade       per period, the decider's pick minus the model's own top pick on the same batches,
              gross and net of measured costs (each pick pays its own cost), t from period means

Deciders: anything with name and pick(job) -> {letter: probability} (pick_many(jobs) optional):
  NoDecider (free), JevDecider (TypeSafe; typesafe-sdk + TYPESAFE_API_KEY), ClaudeDecider
  (ANTHROPIC_API_KEY). Paid deciders print an estimate first and go through engine.spend.Ledger.
"""

from __future__ import annotations

import json
import string
import sys
import zlib

import numpy as np
import pandas as pd

from engine import model, panel, run, score
from engine.spend import Ledger

LETTERS = string.ascii_uppercase
QUESTION = (
    "These are companies from the same group at the same date. Which one will perform best over "
    "the next period relative to the others? Use the model's score and the inputs behind it, "
    "weighted by how reliable the note says each has been; depart from the model only when a card "
    "shows something its score clearly misses."
)
NOTE_ALLOWANCE = 800  # chars a grown override record may add to a note (estimate is worst case)


# ---------------------------------------------------------------- cards and deciders
def card(
    letter: str, contrib: pd.Series, raw: pd.Series, rank: int, n: int, labels: dict, top: int = 5
) -> str:
    """One entity's card: its score rank and the `top` inputs pushing the score most."""
    lines = [f"Company {letter}: model score rank {rank} of {n}"]
    for k in contrib.abs().sort_values(ascending=False).index[:top]:
        v = raw.get(k)
        shown = "n/a" if pd.isna(v) else f"{v:.3g}"
        lines.append(
            f"  - {labels.get(k, k)}: {shown} (pushes score "
            f"{'up' if contrib[k] > 0 else 'down'}, {abs(contrib[k]) * 100:.1f})"
        )
    return "\n".join(lines)


def batches(scored: pd.DataFrame, size: int = 10, seed: int = 0) -> pd.DataFrame:
    """Random same-group batches of `size` per decision time (leftovers skipped)."""
    rng = np.random.default_rng(seed)
    out = []
    for (t, g), grp in scored.groupby(["decision_time", "group"]):
        grp = grp.iloc[rng.permutation(len(grp))]
        for i in range(0, len(grp) // size * size, size):
            out.append(grp.iloc[i : i + size].assign(batch=f"{t}-{g}-{i // size}"))
    return pd.concat(out, ignore_index=True) if out else scored.iloc[0:0]


class NoDecider:
    """The model's own top pick: free, and the baseline every decider must beat."""

    name = "none"

    def pick(self, job: dict) -> dict:
        """Always the model's pick."""
        return {job["model_pick"]: 1.0}


class JevDecider:
    """TypeSafe's Jev answering a choice question per batch."""

    name = "jev"

    def __init__(
        self,
        ledger,
        model: str = "jev-1.13.0",
        step: str = "decider",
        chars_per_token: float = 2.0,  # see engine.spend.CHARS_PER_TOKEN_FLOOR
    ):
        self.ledger, self.model, self.step, self.cpt = ledger, model, step, chars_per_token

    def _ask(self, job: dict):
        from typesafe_sdk import Choice, TypeSafeClient  # optional dependency

        letters = list(job["letters"])
        return TypeSafeClient().system_one(
            state=job["state"],
            questions={
                "pick": Choice(instructions=QUESTION, criteria={L: f"Company {L}" for L in letters})
            },
            model=self.model,
        )

    def pick_many(self, jobs: list[dict], workers: int = 8) -> list[dict]:
        """One guarded chunk: project the worst case, ask in parallel, record the tokens once."""
        from concurrent.futures import ThreadPoolExecutor

        chars = sum(len(json.dumps(j["state"])) + len(QUESTION) for j in jobs)
        self.ledger.guard(self.step, self.model, self.ledger.price(self.model, chars / self.cpt))

        def one(j):
            try:
                r = self._ask(j)
                p = {str(k): float(v) for k, v in dict(r.answers["pick"].probabilities).items()}
                return p, r.usage.input_tokens or 0
            except Exception:  # noqa: BLE001 -- a failed batch is skipped and counted
                return {}, 0

        with ThreadPoolExecutor(workers) as pool:
            res = list(pool.map(one, jobs))
        tokens = sum(t for _, t in res)
        errors = sum(1 for p, _ in res if not p)
        self.ledger.record(
            self.step, self.model, tokens, note=f"decider {len(jobs)} batches ({errors} errors)"
        )
        return [p for p, _ in res]

    def pick(self, job: dict) -> dict:
        """One batch (a guarded chunk of one)."""
        return self.pick_many([job], workers=1)[0]


class ClaudeDecider:
    """An Anthropic model answering with a JSON-schema pick."""

    name = "claude"

    def __init__(
        self,
        ledger,
        model: str = "claude-sonnet-5-5",
        step: str = "decider",
        max_tokens: int = 2000,
    ):
        self.ledger, self.model, self.step, self.max_tokens = ledger, model, step, max_tokens

    def pick(self, job: dict) -> dict:
        """Count tokens, guard the worst case, ask, record."""
        import anthropic  # optional dependency

        letters = list(job["letters"])
        schema = {
            "type": "object",
            "properties": {
                "pick": {"type": "string", "enum": letters},
                "reason": {"type": "string"},
            },
            "required": ["pick", "reason"],
            "additionalProperties": False,
        }
        msg = [{"role": "user", "content": json.dumps(job["state"]) + "\n\n" + QUESTION}]
        client = anthropic.Anthropic()
        n_in = client.messages.count_tokens(model=self.model, messages=msg).input_tokens
        self.ledger.guard(
            self.step, self.model, self.ledger.price(self.model, n_in, self.max_tokens)
        )
        r = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=msg,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self.ledger.record(
            self.step,
            self.model,
            r.usage.input_tokens,
            r.usage.output_tokens,
            note=f"decider {job['batch']}",
        )
        ans = json.loads(next(b.text for b in r.content if b.type == "text"))
        return {ans["pick"]: 1.0}


def make_decider(cfg: dict | None, ledger=None):
    """none (default) | jev | claude; a paid decider needs a spend ledger."""
    kind = (cfg or {}).get("kind", "none")
    if kind == "none":
        return NoDecider()
    if ledger is None:
        raise ValueError("a paid decider needs a spend ledger")
    opts = {k: v for k, v in cfg.items() if k != "kind"}
    return {"jev": JevDecider, "claude": ClaudeDecider}[kind](ledger, **opts)


# ---------------------------------------------------------------- the feedback note
def ic_table(rows: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Per decision period and input: rank IC with the group-adjusted return; plus each period's
    last label end (a period is closed for a decision only once all its labels have ended)."""
    adj = score.group_adjusted(rows)
    ranked = rows[features].groupby(rows["decision_time"]).rank(pct=True)
    target = adj.groupby(rows["decision_time"]).rank(pct=True)
    tab = {}
    for t, idx in rows.groupby("decision_time").groups.items():
        x, y = ranked.loc[idx], target.loc[idx]
        tab[t] = x.corrwith(y)  # Pearson on ranks = Spearman (ties averaged)
    return pd.DataFrame(tab).T.sort_index(), rows.groupby("decision_time")["label_end"].max()


def input_record(tab: pd.DataFrame, ends: pd.Series, t, min_periods: int = 12) -> pd.DataFrame:
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
                    "t": score.per_period_t(ic),
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
                "status": "keep" if (not enough or abs(best["t"]) >= keep_t) else "drop",
            }
        )
    return pd.DataFrame(rows, columns=["question", "best_field", "ic", "t", "status"])


def override_split(decisions: pd.DataFrame, drivers: dict) -> pd.DataFrame:
    """decisions: one row per batch (batch, decision_time, override, pick_excess, model_excess);
    drivers: batch -> text fields that led the picked card. Per field: overrides it drove, mean
    pick-minus-model excess, t over periods."""
    o = decisions[decisions["override"]].copy()
    rows = []
    for fld in sorted({f for v in drivers.values() for f in v}):
        m = o[o["batch"].map(lambda b, fld=fld: fld in drivers.get(b, ()))]
        if m.empty:
            continue
        gain = (m["pick_excess"] - m["model_excess"]).groupby(m["decision_time"]).mean()
        rows.append(
            {
                "field": fld,
                "overrides": len(m),
                "gain_vs_model": float(gain.mean()),
                "t": score.per_period_t(gain),
                "won_share": float((m["pick_excess"] > m["model_excess"]).mean()),
            }
        )
    return pd.DataFrame(rows, columns=["field", "overrides", "gain_vs_model", "t", "won_share"])


def feedback_note(
    model_hist: pd.Series,
    rec: pd.DataFrame,
    qstat: pd.DataFrame,
    past: pd.DataFrame,
    labels: dict,
    drivers: dict | None = None,
) -> str:
    """The note's text, from closed-period records only."""
    lines = ["Track record (closed periods only):"]
    if len(model_hist) >= 12:
        lines.append(
            f"- The model's top pick beat its batch average by {model_hist.mean():+.2%} per period over "
            f"{len(model_hist)} periods ({(model_hist > 0).mean():.0%} of them)."
        )
    else:
        lines.append("- The model has little track record yet; treat its score as weak.")
    if len(rec):
        strong = rec[rec["t"].abs() >= 2].sort_values("t", key=abs, ascending=False).head(5)
        for r in strong.itertuples():
            if "==" in r.input:
                how = ("better" if r.ic > 0 else "worse") + " when it holds"
            else:
                how = "higher is better" if r.ic > 0 else "higher is worse"
            lines.append(f"- {labels.get(r.input, r.input)}: {how} (IC {r.ic:+.3f}, t {r.t:.1f})")
        weak = rec[rec["t"].abs() < 1]["input"].head(4).tolist()
        if weak:
            lines.append(
                "- No reliable signal so far from: " + "; ".join(labels.get(w, w) for w in weak)
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
                sp = override_split(past, drivers)
                for r in sp[sp["overrides"] >= 20].sort_values("t").itertuples():
                    lines.append(
                        f"  - when '{labels.get(r.field, r.field)}' led the card you picked: {r.gain_vs_model:+.2%} vs the model ({r.overrides} times)"
                    )
    return "\n".join(lines)


def run_batches(
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
    """The decider over every batch, sequential over decision times (the override record must close
    before it is shown). scored: batched rows (entity_id, decision_time, group, fwd_return, score,
    batch, rt_cost); rows: model rows (inputs, label_end); contrib, raw: indexed like scored.
    Returns one row per batch: picks, gross and net excess."""
    out = []
    end_of = rows.set_index(["entity_id", "decision_time"])["label_end"]
    scored = scored.assign(
        label_end=[end_of.get((e, t)) for e, t in zip(scored["entity_id"], scored["decision_time"])]
    )
    excess = scored["fwd_return"] - scored.groupby("batch")["fwd_return"].transform("mean")
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
                    text_meta = text_meta | {f"{c}=={k}": m | {"encoding": f"level {k}"}}
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
        dropped_q: set = set()
        if feedback:
            mh = model_hist_all[model_hist_all["end"] < t].groupby("t")["x"].mean()
            rec = input_record(tab, ends, t)
            qstat = question_status(rec, text_meta)
            past = pd.DataFrame(out)
            if len(past):
                past = past[past["label_end"] < t]
                drivers_past = dict(zip(past["batch"], past["drivers"]))
            txt = feedback_note(mh, rec, qstat, past, labels, drivers_past)
            dropped_q = set(qstat.loc[qstat["status"] == "drop", "question"])
        hidden = {c for c, m in text_meta.items() if m.get("question") in dropped_q}
        jobs = []
        for b, g in g_t.groupby("batch"):
            if len(g) != size:
                continue
            g = g.sample(frac=1, random_state=zlib.crc32(str(b).encode()))  # letters != score order
            ranks = g["score"].rank(ascending=False, method="first").astype(int)
            letters = LETTERS[: len(g)]
            cards = []
            for L, i in zip(letters, g.index):
                c = contrib.loc[i].drop(labels=[x for x in hidden if x in contrib.columns])
                cards.append(card(L, c, raw.loc[i], ranks[i], len(g), labels))
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
        if hasattr(dec, "pick_many"):  # paid deciders: one guarded, recorded chunk per period
            all_probs = dec.pick_many([j for j, _, _ in jobs])
        else:
            all_probs = [dec.pick(j) for j, _, _ in jobs]
        for (job, g, letters), probs in zip(jobs, all_probs):
            if not probs:
                continue  # a failed call: the batch is skipped, not guessed
            pick = max(probs, key=probs.get)
            avg = float(np.mean(list(job["returns"].values())))
            avg_cost = float(np.mean(list(job["costs"].values())))
            i_pick = g.index[letters.index(pick)]
            lead = contrib.loc[i_pick].abs().sort_values(ascending=False).index[:5]
            mp = job["model_pick"]
            out.append(
                {
                    "batch": job["batch"],
                    "decision_time": t,
                    "label_end": g["label_end"].max(),
                    "pick": job["letters"][pick],
                    "model_pick": job["letters"][mp],
                    "pick_excess": job["returns"][pick] - avg,
                    "model_excess": job["returns"][mp] - avg,
                    "pick_net": job["returns"][pick] - job["costs"][pick] - (avg - avg_cost),
                    "model_net": job["returns"][mp] - job["costs"][mp] - (avg - avg_cost),
                    "override": pick != mp,
                    "drivers": tuple(x for x in lead if x in text_meta),
                    "note_chars": len(txt),
                }
            )
    return pd.DataFrame(out)


def grade_decisions(dec: pd.DataFrame) -> dict:
    """Period means, t from period returns: decider vs the model's own pick, gross and net."""
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
        "decider_minus_model_gross_t": score.per_period_t(d - m),
        "decider_minus_model_net": float((dn - mn).mean()),
        "decider_minus_model_net_t": score.per_period_t(dn - mn),
        "decider_vs_batch_net_t": score.per_period_t(dn),
        "model_vs_batch_net_t": score.per_period_t(mn),
    }


# ---------------------------------------------------------------- a decider run on a study
def text_feature_meta(study) -> dict:
    """feature -> {question, kind, tag, encoding, ...} for every text source's columns."""
    out = {}
    for spec in study.sources.values():
        if hasattr(spec.source, "feature_meta"):
            out |= spec.source.feature_meta()
    return out


def feature_labels(study, meta: dict) -> dict:
    """Plain-language names for cards: question wording for text columns, then the YAML's
    feature_labels."""
    lab = {}
    for spec in study.sources.values():
        qsets = getattr(spec.source, "qsets", {})
        for c, m in meta.items():
            qs = qsets.get(m["doc_type"])
            if qs is not None and m["question"] in qs.ids():
                q = qs.get(m["question"])
                part = "" if m["part"] in ("p", "level") else f" [{m['part']}]"
                enc = {
                    "change": " (change vs its previous documents)",
                    "surprise": " (vs history base rate)",
                }.get(m["encoding"], "")
                lab[c] = f"text: {q.prompt}{part}{enc}"
    return lab | study.cfg.get("feature_labels", {})


def prepare_cards(study, feature_set: str = "baseline", size: int = 10, seed: int = 0) -> dict:
    """Model scores on tuning periods, batched, with each row's contributions and raw inputs."""
    rows = panel.model_rows(study, run.build_panel(study, "tuning"))
    feats = study.feature_set(feature_set)
    scored, w = run.walk_forward(study).run(rows, feats, study.market.calendar)
    scored = run.in_stage(study, scored, "tuning")
    b = batches(scored, size, seed)
    keyed = rows.set_index(["entity_id", "decision_time"])
    idx = list(zip(b["entity_id"], b["decision_time"]))
    raw = keyed.loc[idx, feats].reset_index(drop=True)
    contrib_rows = model.contributions(rows, feats, w, study.market.calendar)
    contrib = (
        contrib_rows.set_index(
            pd.MultiIndex.from_frame(rows.loc[contrib_rows.index, ["entity_id", "decision_time"]])
        )
        .loc[idx]
        .reset_index(drop=True)
    )
    b = b.reset_index(drop=True)
    b["rt_cost"] = b["rt_cost"].fillna(b["rt_cost"].median())
    meta = {c: m for c, m in text_feature_meta(study).items() if c in feats}
    return {
        "rows": rows,
        "scored": b,
        "raw": raw,
        "contrib": contrib,
        "features": feats,
        "meta": meta,
        "labels": feature_labels(study, meta),
    }


def _run_prepared(dec, prep: dict, **kw) -> pd.DataFrame:
    return run_batches(
        dec,
        prep["scored"],
        prep["rows"],
        prep["contrib"],
        prep["raw"],
        prep["features"],
        prep["labels"],
        prep["meta"],
        **kw,
    )


class _Sizer:
    """A free stand-in decider that records how big each prompt would be."""

    name = "sizer"

    def __init__(self):
        self.chars = []

    def pick(self, job: dict) -> dict:
        self.chars.append(len(json.dumps(job["state"])) + len(QUESTION) + NOTE_ALLOWANCE)
        return {job["model_pick"]: 1.0}


def estimate(prep: dict, ledger: Ledger, model_name: str = "jev-1.13.0") -> dict:
    """Projected cost of a paid decider run: every prompt sized by a free dry run (worst case)."""
    sizer = _Sizer()
    _run_prepared(sizer, prep)
    usd = ledger.price_chars(model_name, float(np.sum(sizer.chars)))
    return {"batches": len(sizer.chars), "chars": int(np.sum(sizer.chars)), "projected_usd": usd}


def log_decider_test(study, g: dict, name: str, note: str) -> dict:
    """Record a decider run as ONE judged test: decider minus model top pick, net of costs."""
    reg = study.registry
    bar = reg.next_bar()
    return reg.record(
        {
            "kind": "test",
            "name": name,
            "features": "baseline",
            "scope": "universal",
            "metric": "pick-1-of-10 monthly excess net of measured costs, decider minus model top pick",
            "t_tune": round(g["decider_minus_model_net_t"], 3),
            "bar_tune": round(bar, 3),
            "gain_tune": round(g["decider_minus_model_net"], 6),
            "check_used": False,
            "kept": bool(
                g["decider_minus_model_net_t"] >= bar and g["decider_minus_model_net"] > 0
            ),
            "note": note,
        }
    )


def run_decider(
    study,
    kind: str | None = None,
    features: str = "baseline",
    estimate_only: bool = False,
    log: bool = False,
    note: str = "",
) -> dict:
    """A decider over the model's cards on tuning periods, with the feedback note (YAML decider.
    feedback, on by default). Paid kinds print their estimate first and refuse past the step's cap;
    log=True records the result as ONE test."""
    cfg = study.cfg.get("decider") or {"kind": "none"}
    kind = kind or cfg.get("kind", "none")
    prep = prepare_cards(study, features)
    ledger = Ledger.from_config(study.cfg.get("spend"))
    step = cfg.get("step", "decider")
    default_model = "jev-1.13.0" if kind == "jev" else "claude-sonnet-5-5"
    if kind != "none":
        est = estimate(prep, ledger, cfg.get("model", default_model))
        est |= {
            "step": step,
            "cap": ledger.step_caps.get(step),
            "already_spent": ledger.spent(step),
        }
        print(json.dumps(est, indent=1), file=sys.stderr)
        if estimate_only:
            return {"decider": kind, "estimate": est}
        if est["projected_usd"] > ledger.remaining(step):
            raise SystemExit(
                f"estimate ${est['projected_usd']:.2f} is over what's left in {step}: not running"
            )
    opts = {k: v for k, v in cfg.items() if k not in ("kind", "feedback", "step")}
    dec = make_decider(
        {"kind": kind, **opts, **({"step": step} if kind != "none" else {})},
        ledger if kind != "none" else None,
    )
    fb = bool(cfg.get("feedback", True))
    out = _run_prepared(dec, prep, feedback=fb)
    g = grade_decisions(out)
    path = study.cache_dir / f"decisions_{kind}{'_fb' if fb else ''}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    res = {"decider": kind, "feedback": fb, "grade": g, "decisions": str(path)}
    if log:
        name = f"decider:{kind}{'+feedback' if fb else ''} vs model top pick"
        res["registry"] = log_decider_test(study, g, name, note)
    return res
