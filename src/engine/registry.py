"""Test registry, the multiple-testing bar, and the three periods (tuning / check / holdout).

Every judged test is a row in the market's registry CSV, kept or not. Trying more ideas raises the
bar for the next one (Bonferroni: a two-sided test at alpha / n), so the best of many lucky tries
doesn't pass. Rows from earlier logs (jev's discover_log.csv and experiments.csv) are INHERITED at
load time, read-only, so the count continues instead of restarting at zero.

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
    "kind",  # test | parity (never judged) | decision
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
    "origin",  # engine | <inherited file name>
]


def required_t(n_tries: int, alpha: float = 0.05) -> float:
    """1 try -> 1.96, 10 -> 2.81, 50 -> 3.29, 100 -> 3.48 (same rule as jev.model.required_t)."""
    return float(norm.ppf(1 - alpha / 2 / max(1, n_tries)))


@dataclass(frozen=True)
class Periods:
    tuning: tuple[pd.Timestamp, pd.Timestamp]  # [start, end)
    check: tuple[pd.Timestamp, pd.Timestamp]
    holdout_start: pd.Timestamp

    @classmethod
    def from_config(cls, cfg: dict) -> Periods:
        ts = lambda x: pd.Timestamp(str(x))
        return cls(
            (ts(cfg["tuning"][0]), ts(cfg["tuning"][1])),
            (ts(cfg["check"][0]), ts(cfg["check"][1])),
            ts(cfg["holdout_start"]),
        )

    def years(self, stage: str) -> range:
        lo, hi = self.tuning if stage == "tuning" else self.check
        return range(lo.year, hi.year)

    def stage_of(self, local_date: pd.Timestamp) -> str:
        if local_date >= self.holdout_start:
            return "holdout"
        if self.check[0] <= local_date < self.check[1]:
            return "check"
        return "tuning"


class CheckLimitReached(RuntimeError):
    pass


class HoldoutLocked(RuntimeError):
    pass


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
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
                    "commit": _commit(),
                    "args": f"engine {market}: {reason}",
                }
            ]
        )
        self.unlock_log.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(self.unlock_log, mode="a", header=new, index=False)

    def looks(self) -> int:
        return len(pd.read_csv(self.unlock_log)) if self.unlock_log.exists() else 0


@dataclass
class Registry:
    path: Path
    market: str
    inherit: list[Path] = field(default_factory=list)
    check_limit: int = 30

    # ---------- reading ----------
    def own(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path)
        return pd.DataFrame(columns=COLUMNS)

    def inherited(self) -> pd.DataFrame:
        """Earlier logs mapped onto the registry's columns, read-only."""
        frames = []
        for p in self.inherit:
            if not p.exists():
                continue
            d = pd.read_csv(p)
            if (
                "origin" in d and "kind" in d
            ):  # an aggregate already in registry columns
                frames.append(d.reindex(columns=COLUMNS))
                continue
            if "action" in d:  # the pilot's discover_log.csv
                frames.append(
                    pd.DataFrame(
                        {
                            "kind": d["action"],
                            "name": d["indicator"],
                            "scope": d["scope"],
                            "t_tune": d["t_tune"],
                            "bar_tune": d["bar_tune"],
                            "check_used": d["check_used"],
                            "t_check": d["t_check"],
                            "bar_check": d["bar_check"],
                            "gain_tune": d["gain_tune"],
                            "gain_check": d["gain_check"],
                            "kept": d["kept"],
                            "note": d["note"],
                            "origin": p.name,
                        }
                    )
                )
            else:  # the pilot's experiments.csv: every row was a judged test
                frames.append(
                    pd.DataFrame(
                        {
                            "kind": "test",
                            "name": d["inputs"],
                            "t_tune": d["t_tune"],
                            "bar_tune": d["bar"],
                            "check_used": False,  # its check uses are logged as historical_check rows
                            "kept": d["kept"],
                            "note": d["note"],
                            "origin": p.name,
                        }
                    )
                )
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=COLUMNS)
        )

    def rows(self) -> pd.DataFrame:
        return pd.concat([self.inherited(), self.own()], ignore_index=True)

    @staticmethod
    def _judged(rows: pd.DataFrame) -> pd.Series:
        from_experiments = rows["origin"].astype(str).str.startswith("experiments")
        judged_test = (rows["kind"] == "test") & pd.to_numeric(
            rows["t_tune"], errors="coerce"
        ).notna()
        return from_experiments | judged_test

    def n_judged(self) -> int:
        return int(self._judged(self.rows()).sum())

    def check_uses(self) -> int:
        used = self.rows()["check_used"]
        return int(used.map(lambda v: str(v).strip().lower() == "true").sum())

    def next_bar(self) -> float:
        return required_t(self.n_judged() + 1)

    def next_check_bar(self) -> float:
        return required_t(self.check_uses() + 1)

    def tested(self) -> set[tuple[str, str]]:
        r = self.rows()
        return set(zip(r["name"].astype(str), r["scope"].fillna("").astype(str)))

    # ---------- writing ----------
    def record(self, row: dict) -> dict:
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

    def open_check(self) -> float:
        """Grant one check-period look; returns its bar. The caller must record it (check_used=True)."""
        if self.check_uses() >= self.check_limit:
            raise CheckLimitReached(f"all {self.check_limit} check-period looks used")
        return self.next_check_bar()

    def summary(self) -> dict:
        r = self.rows()
        own = self.own()
        return {
            "judged_tests": self.n_judged(),
            "next_bar": round(self.next_bar(), 3),
            "check_uses": self.check_uses(),
            "check_limit": self.check_limit,
            "next_check_bar": round(self.next_check_bar(), 3),
            "kept": int(
                r["kept"].map(lambda v: str(v).strip().lower() == "true").sum()
            ),
            "engine_rows": len(own),
        }
