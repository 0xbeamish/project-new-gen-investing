"""Test registry, the multiple-testing bar, and the three periods (tuning / check / holdout).

Every judged test is a row in the market's registry CSV, kept or not. Trying more ideas raises the
bar for the next one (Bonferroni: a two-sided test at alpha / n), so the best of many lucky tries
doesn't pass. Other registries (earlier research, other markets) are INHERITED read-only, so the
count continues instead of restarting at zero.

Periods, declared per market:
  tuning    everything is designed and judged here
  check     opened only for a candidate that clears the tuning bar; every opening is counted, each
            raises the check bar, and there's a hard limit. The loop sees pass/fail only
  holdout   locked. Unlocking needs an explicit reason and is appended to the unlock log with the
            commit, because every look makes it less of a holdout
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from scipy.stats import norm

COLUMNS = [
    "test_id",
    "market",
    "timestamp",
    "kind",  # test (judged when it has a t) | parity, decision ... (never judged)
    "name",
    "features",
    "scope",
    "metric",
    "t_tune",
    "bar_tune",
    "check_used",
    "t_check",
    "bar_check",
    "gain_tune",
    "gain_check",
    "kept",
    "note",
    "origin",  # engine | the inherited log's name
]


def required_t(n_tries: int, alpha: float = 0.05) -> float:
    """The Bonferroni bar: 1 try -> 1.96, 10 -> 2.81, 50 -> 3.29, 100 -> 3.48."""
    return float(norm.ppf(1 - alpha / 2 / max(1, n_tries)))


@dataclass(frozen=True)
class Periods:
    tuning: tuple[pd.Timestamp, pd.Timestamp]  # [start, end)
    check: tuple[pd.Timestamp, pd.Timestamp]
    holdout_start: pd.Timestamp

    @classmethod
    def from_config(cls, cfg: dict) -> Periods:
        """From the YAML's periods: {tuning: [start, end], check: [start, end], holdout_start}."""

        def ts(x):
            return pd.Timestamp(str(x))

        return cls(
            (ts(cfg["tuning"][0]), ts(cfg["tuning"][1])),
            (ts(cfg["check"][0]), ts(cfg["check"][1])),
            ts(cfg["holdout_start"]),
        )


@dataclass(frozen=True)
class CheckGrant:
    """Proof that the registry opened the check period for one test (Registry.open_check)."""

    bar: float


class CheckLimitReached(RuntimeError):
    pass


class HoldoutLocked(RuntimeError):
    pass


def git_commit() -> str:
    """The short commit hash, or "unknown"."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class HoldoutLock:
    unlock_log: Path

    def unlock(self, market: str, reason: str) -> None:
        """Record a look at the holdout. Call only from a human-run --final command."""
        if not reason.strip():
            raise HoldoutLocked("an unlock needs a reason")
        new = not self.unlock_log.exists()
        row = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
                    "commit": git_commit(),
                    "args": f"engine {market}: {reason}",
                }
            ]
        )
        self.unlock_log.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(self.unlock_log, mode="a", header=new, index=False)

    def looks(self) -> int:
        """How many times the holdout has been opened."""
        return len(pd.read_csv(self.unlock_log)) if self.unlock_log.exists() else 0


@dataclass
class Registry:
    path: Path
    market: str
    inherit: list[Path] = field(default_factory=list)
    check_limit: int = 30

    # ---------- reading ----------
    def own(self) -> pd.DataFrame:
        """This market's own rows."""
        if self.path.exists():
            return pd.read_csv(self.path)
        return pd.DataFrame(columns=COLUMNS)

    def inherited(self) -> pd.DataFrame:
        """Rows of the inherited registries (same columns), read-only."""
        frames = [pd.read_csv(p).reindex(columns=COLUMNS) for p in self.inherit if p.exists()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)

    def rows(self) -> pd.DataFrame:
        """Inherited rows, then this market's."""
        return pd.concat([self.inherited(), self.own()], ignore_index=True)

    @staticmethod
    def judged(rows: pd.DataFrame) -> pd.Series:
        """A row counts toward the bar if it is a test with a t-statistic."""
        return (rows["kind"] == "test") & pd.to_numeric(rows["t_tune"], errors="coerce").notna()

    def n_judged(self) -> int:
        """Judged tests so far, inherited included."""
        return int(self.judged(self.rows()).sum())

    def check_uses(self) -> int:
        """Check-period looks so far, inherited included."""
        used = self.rows()["check_used"]
        return int(used.map(lambda v: str(v).strip().lower() == "true").sum())

    def next_bar(self) -> float:
        """The t the next judged test must reach on tuning periods."""
        return required_t(self.n_judged() + 1)

    def next_check_bar(self) -> float:
        """The t the next check-period look must reach."""
        return required_t(self.check_uses() + 1)

    def tested(self) -> set[tuple[str, str]]:
        """(name, scope) of everything already tried."""
        r = self.rows()
        return set(zip(r["name"].astype(str), r["scope"].fillna("").astype(str)))

    # ---------- writing ----------
    def record(self, row: dict) -> dict:
        """Append one row (kept or not) and return it with its id and timestamp."""
        own = self.own()
        row = {c: row.get(c) for c in COLUMNS} | {
            "market": self.market,
            "origin": "engine",
            "test_id": f"{self.market}-{len(own) + 1:04d}",
            "timestamp": row.get("timestamp")
            or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        }
        out = pd.concat([own, pd.DataFrame([row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out[COLUMNS].to_csv(self.path, index=False)
        return row

    def open_check(self) -> CheckGrant:
        """Grant one check-period look. The caller must record it (check_used=True)."""
        if self.check_uses() >= self.check_limit:
            raise CheckLimitReached(f"all {self.check_limit} check-period looks used")
        return CheckGrant(self.next_check_bar())

    def summary(self) -> dict:
        """Tests so far, next bars, check uses: what `engine report` prints."""
        r = self.rows()
        return {
            "judged_tests": self.n_judged(),
            "next_bar": round(self.next_bar(), 3),
            "check_uses": self.check_uses(),
            "check_limit": self.check_limit,
            "next_check_bar": round(self.next_check_bar(), 3),
            "kept": int(r["kept"].map(lambda v: str(v).strip().lower() == "true").sum()),
            "engine_rows": len(self.own()),
        }
