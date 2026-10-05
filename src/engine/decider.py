"""Optional decider layer: an LLM picks one entity per batch from model cards. Off by default.

Ported from the pilot's decider. Per batch (N entities of one group at one decision time):
  card              the model's score rank plus the inputs pushing the score most, each as
                    contribution = block weight x the input's percentile rank (centred)
  reliability note  what was known BEFORE the decision's block: the model's track record and its
                    steadiest inputs (and, if given, an extra feedback note)
  pick              the decider's choice; graded per period as pick return minus batch average,
                    next to the model's own top pick and random (0 by construction)

A decider is anything with name and pick(job) -> {letter: probability}. Implementations:
  NoDecider      the model's top pick (default; free; the engine runs with no paid API)
  JevDecider     TypeSafe's Jev (needs typesafe-sdk and TYPESAFE access)
  ClaudeDecider  an Anthropic model with a JSON-schema answer (needs ANTHROPIC_API_KEY)
Paid deciders go through engine.spend.Ledger: a projected cost past the cap refuses the call.
Every layer must beat the version without it: grade() reports decider vs model, paired per period.
"""

from __future__ import annotations

import json
import string
import zlib

import numpy as np
import pandas as pd

from engine import scoring

LETTERS = string.ascii_uppercase
QUESTION = (
    "These are companies from the same group at the same date. Which one will perform best over "
    "the next period relative to the others? Use the model's score and the inputs behind it, "
    "weighted by how reliable the note says each has been; depart from the model only when a card "
    "shows something its score clearly misses."
)


def card(
    letter: str,
    contrib: pd.Series,
    raw: pd.Series,
    rank: int,
    n: int,
    labels: dict,
    top: int = 5,
) -> str:
    lines = [f"Company {letter}: model score rank {rank} of {n}"]
    for k in contrib.abs().sort_values(ascending=False).index[:top]:
        v = raw.get(k)
        shown = "n/a" if pd.isna(v) else f"{v:.3g}"
        lines.append(
            f"  - {labels.get(k, k)}: {shown} (pushes score "
            f"{'up' if contrib[k] > 0 else 'down'}, {abs(contrib[k]) * 100:.1f})"
        )
    return "\n".join(lines)


def reliability_note(
    weights: pd.DataFrame, history: pd.Series, block, labels: dict
) -> str:
    """Only what was known before `block`: past per-period excess of the model's pick, steady weights."""
    past = history[[b < block for b in history.index.map(lambda t: t.year)]]
    w = weights[weights.index < block]
    lines = ["Track record (earlier periods only):"]
    if len(past) >= 12:
        lines.append(
            f"- The model's top pick beat its batch average by {past.mean():+.2%} per period on "
            f"average over {len(past)} periods, and in {(past > 0).mean():.0%} of them."
        )
    else:
        lines.append(
            "- The model has little track record yet; treat its score as weak."
        )
    if len(w) >= 2:
        steady = (
            (w.mean() / w.std().replace(0, np.nan))
            .abs()
            .dropna()
            .sort_values(ascending=False)
        )
        for col in steady.index[:4]:
            lines.append(
                f"- {labels.get(col, col)}: consistently "
                f"{'helps' if w[col].mean() > 0 else 'hurts'} (higher value)"
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


def build_jobs(
    scored: pd.DataFrame,
    contrib: pd.DataFrame,
    raw: pd.DataFrame,
    weights: pd.DataFrame,
    labels: dict,
    size: int = 10,
    extra_note: str = "",
) -> list[dict]:
    """scored: batched rows (entity_id, decision_time, group, fwd_return, score, block, batch);
    contrib / raw: indexed like scored (contributions and raw input values)."""
    excess = scored["fwd_return"] - scored.groupby("batch")["fwd_return"].transform(
        "mean"
    )
    top = scored.loc[scored.groupby("batch")["score"].idxmax()]
    history = (excess.loc[top.index]).groupby(top["decision_time"]).mean()
    jobs = []
    for b, g in scored.groupby("batch"):
        if len(g) != size:
            continue
        g = g.sample(
            frac=1, random_state=zlib.crc32(str(b).encode())
        )  # letters don't follow the score
        ranks = g["score"].rank(ascending=False, method="first").astype(int)
        letters = LETTERS[: len(g)]
        cards = [
            card(L, contrib.loc[i], raw.loc[i], ranks[i], len(g), labels)
            for L, i in zip(letters, g.index)
        ]
        block = g["block"].iloc[0]
        jobs.append(
            {
                "batch": b,
                "decision_time": g["decision_time"].iloc[0],
                "letters": dict(zip(letters, g["entity_id"])),
                "returns": dict(zip(letters, g["fwd_return"])),
                "model_pick": letters[list(ranks).index(1)],
                "state": {
                    "document_type": "ranking cards",
                    "group": str(g["group"].iloc[0]),
                    "reliability_note": reliability_note(
                        weights, history, block, labels
                    )
                    + extra_note,
                    "cards": "\n\n".join(cards),
                },
            }
        )
    return jobs


class NoDecider:
    name = "none"

    def pick(self, job: dict) -> dict:
        return {job["model_pick"]: 1.0}


class JevDecider:
    name = "jev"

    def __init__(
        self,
        ledger,
        model: str = "jev-1.13.0",
        step: str = "decider",
        chars_per_token: float = 2.0,  # see engine.spend.CHARS_PER_TOKEN_FLOOR
    ):
        self.ledger, self.model, self.step, self.cpt = (
            ledger,
            model,
            step,
            chars_per_token,
        )

    def _ask(self, job: dict):
        from typesafe_sdk import Choice, TypeSafeClient  # optional dependency

        letters = list(job["letters"])
        return TypeSafeClient().system_one(
            state=job["state"],
            questions={
                "pick": Choice(
                    instructions=QUESTION, criteria={L: f"Company {L}" for L in letters}
                )
            },
            model=self.model,
        )

    def pick_many(self, jobs: list[dict], workers: int = 8) -> list[dict]:
        """One guarded chunk: project the worst case, ask in parallel, record the tokens once."""
        from concurrent.futures import ThreadPoolExecutor

        chars = sum(len(json.dumps(j["state"])) + len(QUESTION) for j in jobs)
        self.ledger.guard(
            self.step, self.model, self.ledger.price(self.model, chars / self.cpt)
        )

        def one(j):
            try:
                r = self._ask(j)
                p = {
                    str(k): float(v)
                    for k, v in dict(r.answers["pick"].probabilities).items()
                }
                return p, r.usage.input_tokens or 0
            except Exception:  # noqa: BLE001 -- a failed batch is skipped and counted
                return {}, 0

        with ThreadPoolExecutor(workers) as pool:
            res = list(pool.map(one, jobs))
        tokens = sum(t for _, t in res)
        errors = sum(1 for p, _ in res if not p)
        self.ledger.record(
            self.step,
            self.model,
            tokens,
            note=f"decider {len(jobs)} batches ({errors} errors)",
        )
        return [p for p, _ in res]

    def pick(self, job: dict) -> dict:
        from typesafe_sdk import Choice, TypeSafeClient  # optional dependency

        chars = len(json.dumps(job["state"])) + len(QUESTION)
        self.ledger.guard(
            self.step, self.model, self.ledger.price(self.model, chars / self.cpt)
        )
        letters = list(job["letters"])
        r = TypeSafeClient().system_one(
            state=job["state"],
            questions={
                "pick": Choice(
                    instructions=QUESTION, criteria={L: f"Company {L}" for L in letters}
                )
            },
            model=self.model,
        )
        self.ledger.record(
            self.step,
            self.model,
            r.usage.input_tokens or 0,
            note=f"decider {job['batch']}",
        )
        return {
            str(k): float(v) for k, v in dict(r.answers["pick"].probabilities).items()
        }


class ClaudeDecider:
    name = "claude"

    def __init__(
        self,
        ledger,
        model: str = "claude-sonnet-5-5",
        step: str = "decider",
        max_tokens: int = 2000,
    ):
        self.ledger, self.model, self.step, self.max_tokens = (
            ledger,
            model,
            step,
            max_tokens,
        )

    def pick(self, job: dict) -> dict:
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
        msg = [
            {"role": "user", "content": json.dumps(job["state"]) + "\n\n" + QUESTION}
        ]
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


def run(decider, jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for j in jobs:
        probs = decider.pick(j)
        pick = max(probs, key=probs.get)
        avg = float(np.mean(list(j["returns"].values())))
        rows.append(
            {
                "batch": j["batch"],
                "decision_time": j["decision_time"],
                "pick": j["letters"][pick],
                "model_pick": j["letters"][j["model_pick"]],
                "pick_excess": j["returns"][pick] - avg,
                "model_excess": j["returns"][j["model_pick"]] - avg,
                "override": pick != j["model_pick"],
            }
        )
    return pd.DataFrame(rows)


def grade(dec: pd.DataFrame) -> dict:
    """Per-period means, then t over periods: decider vs random, model vs random, decider vs model."""
    by = dec.groupby("decision_time")
    d, m = by["pick_excess"].mean(), by["model_excess"].mean()
    return {
        "periods": len(d),
        "decider_vs_random": float(d.mean()),
        "decider_t": scoring.per_period_t(d),
        "model_vs_random": float(m.mean()),
        "model_t": scoring.per_period_t(m),
        "decider_vs_model": float((d - m).mean()),
        "decider_vs_model_t": scoring.per_period_t(d - m),
        "override_rate": float(dec["override"].mean()),
    }


def make(cfg: dict | None, ledger=None):
    kind = (cfg or {}).get("kind", "none")
    if kind == "none":
        return NoDecider()
    if ledger is None:
        raise ValueError("a paid decider needs a spend ledger")
    opts = {k: v for k, v in cfg.items() if k != "kind"}
    return {"jev": JevDecider, "claude": ClaudeDecider}[kind](ledger, **opts)
