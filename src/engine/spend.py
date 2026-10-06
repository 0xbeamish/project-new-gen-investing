"""AI spend caps: one ledger, with a cap per step, for every paid model call the engine makes
(text readers, the question-loop proposer, the optional deciders).

  estimate first   every paid run prints its projected cost before the first call (worst case:
                   Jev by characters / 2, Claude by counted input tokens + full max_tokens)
  guard            refuse a call whose projection would push its step past its cap, or the provider
                   past the funds loaded in that account
  record           append what the API actually reports to the ledger CSV

Caps live in config (market YAML `spend:`), defaults below. `Ledger.from_config(cfg).report()`
prints spent vs cap per step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

FIELDS = [
    "timestamp",
    "step",
    "model",
    "batch",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "usd",
    "note",
]
# USD per 1M tokens, list prices as of 2026-09 (check them when a paid step starts).
PRICES = {
    "jev-1.13.0": {"provider": "typesafe", "input": 0.042, "output": 0.0},
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.0, "output": 5.0},
    "claude-sonnet-5-5": {"provider": "anthropic", "input": 2.0, "output": 10.0},
    "claude-opus-5-5": {"provider": "anthropic", "input": 4.0, "output": 20.0},
}
# Every paid step needs a cap; an unknown step is refused. Raise a cap deliberately, in a commit.
DEFAULT_CAPS = {
    "text_read_jev": 5.0,  # Jev reading documents (question batteries)
    "text_read_claude": 5.0,  # Claude reading documents (gold labels, second labeler)
    "text_probes_claude": 5.0,  # paraphrases and counterfactual edits for the probe sets
    "text_loop_jev": 14.0,  # question-improvement loop: re-reading dev documents
    "text_loop_claude": 6.0,  # question-improvement loop and ladder: the proposer
    "decider": 5.0,
    "jev_engine_loop": 6.0,  # the pre-registered Jev decider re-test with the feedback note
    "jev_replay": 15.0,  # history replay: re-reading releases when a question version changes
    "claude_replay": 3.0,  # history replay: the question-rewrite proposer, only if needed
}
BATCH_DISCOUNT = 0.5
# Jev has no token counter, so estimates divide characters by a floor. 3 under-shot decider cards
# (measured 2.3 chars/token: numbers and symbols tokenize densely); 2 keeps estimates worst-case.
CHARS_PER_TOKEN_FLOOR = 2.0


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    path: Path
    step_caps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    funds: dict[str, float] = field(default_factory=dict)  # provider -> USD loaded
    prices: dict[str, dict] = field(default_factory=lambda: dict(PRICES))

    @classmethod
    def from_config(cls, cfg: dict | None, root: Path | None = None) -> Ledger:
        """cfg = {ledger: path, caps: {step: usd}, funds: {provider: usd}, prices: {...}}."""
        cfg = cfg or {}
        p = Path(cfg.get("ledger", "data/costs.csv"))
        if root and not p.is_absolute():
            p = root / p
        return cls(
            p,
            DEFAULT_CAPS | cfg.get("caps", {}),
            dict(cfg.get("funds", {})),
            PRICES | cfg.get("prices", {}),
        )

    def price(
        self,
        model: str,
        input_tokens: float = 0,
        output_tokens: float = 0,
        batch: bool = False,
    ) -> float:
        """USD for a token count at list price (half with the batch discount)."""
        p = self.prices[model]
        usd = (input_tokens * p["input"] + output_tokens * p.get("output", 0.0)) / 1e6
        return usd * (BATCH_DISCOUNT if batch else 1.0)

    def price_chars(self, model: str, chars: float) -> float:
        """Worst-case input cost from a character count (for readers with no token counter)."""
        return self.price(model, chars / CHARS_PER_TOKEN_FLOOR)

    def frame(self) -> pd.DataFrame:
        """Every recorded call."""
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=FIELDS)

    def spent(self, step: str | None = None, provider: str | None = None) -> float:
        """USD recorded so far, for one step or one provider (or all)."""
        df = self.frame()
        if step:
            df = df[df["step"] == step]
        if provider:
            df = df[df["model"].map(lambda m: self.prices.get(m, {}).get("provider")) == provider]
        return float(df["usd"].sum())

    def remaining(self, step: str) -> float:
        """USD left under a step's cap."""
        return self.step_caps.get(step, 0.0) - self.spent(step)

    def guard(self, step: str, model: str, projected_usd: float) -> None:
        """Raise BudgetExceeded if the projection would pass the step's cap or the loaded funds."""
        if step not in self.step_caps:
            raise BudgetExceeded(f"unknown step {step!r}: give it a cap first")
        provider = self.prices[model]["provider"]
        if self.spent(step) + projected_usd > self.step_caps[step]:
            raise BudgetExceeded(
                f"{step}: ${self.spent(step) + projected_usd:.2f} projected > "
                f"${self.step_caps[step]:.2f} cap"
            )
        if provider not in self.funds:
            raise BudgetExceeded(
                f"{provider}: no funds configured (spend.funds.{provider} in the YAML)"
            )
        if self.spent(provider=provider) + projected_usd > self.funds[provider]:
            raise BudgetExceeded(
                f"{provider}: projected past the ${self.funds[provider]:.2f} loaded"
            )

    def record(
        self,
        step: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        note: str = "",
        batch: bool = False,
    ) -> float:
        """Append what the API actually reported; returns its USD."""
        usd = self.price(model, input_tokens, output_tokens, batch)
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "step": step,
                    "model": model,
                    "batch": batch,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "usd": round(usd, 6),
                    "note": note,
                }
            )
        return usd

    def report(self) -> pd.DataFrame:
        """Spent vs cap per step."""
        df = self.frame()
        spent = df.groupby("step")["usd"].sum() if len(df) else pd.Series(dtype=float)
        steps = sorted(set(self.step_caps) | set(spent.index))
        return pd.DataFrame(
            {
                "cap": [self.step_caps.get(s, float("nan")) for s in steps],
                "spent": [float(spent.get(s, 0.0)) for s in steps],
            },
            index=steps,
        )
