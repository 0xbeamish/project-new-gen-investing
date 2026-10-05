"""Point-in-time checks. Run on every panel the engine builds; a failure stops the run.

A panel cell (entity, decision time, feature) is legal only if the observation behind it has
available_at strictly before the decision time. A label is legal only if its entry bar closes
strictly after the decision time. The panel builder records the available_at of every cell it
fills (its provenance), so the check doesn't trust the builder: it re-reads what was used.
"""

from __future__ import annotations

import pandas as pd


class PointInTimeError(AssertionError):
    pass


def check_cells(provenance: pd.DataFrame, limit: int = 5) -> int:
    """provenance: entity_id, decision_time, feature, available_at (NaT = cell left empty).

    Returns the number of filled cells checked; raises PointInTimeError on any leak."""
    used = provenance.dropna(subset=["available_at"])
    bad = used[used["available_at"] >= used["decision_time"]]
    if len(bad):
        raise PointInTimeError(
            f"{len(bad):,} panel cells use data not yet available at decision time, e.g.\n"
            + bad.head(limit).to_string(index=False)
        )
    return len(used)


def check_labels(panel: pd.DataFrame, limit: int = 5) -> int:
    """Every label must start after its decision: entry_time > decision_time."""
    has = panel.dropna(subset=["entry_time"])
    bad = has[has["entry_time"] <= has["decision_time"]]
    if len(bad):
        raise PointInTimeError(
            f"{len(bad):,} labels start at or before their decision time, e.g.\n"
            + bad[["entity_id", "decision_time", "entry_time"]]
            .head(limit)
            .to_string(index=False)
        )
    return len(has)


def check_panel(panel) -> dict:
    """Both checks on an engine.panel.Panel; returns counts for the run log."""
    return {
        "cells_checked": check_cells(panel.provenance()),
        "labels_checked": check_labels(panel.frame),
    }
