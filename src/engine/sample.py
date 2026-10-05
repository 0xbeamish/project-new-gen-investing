"""Turning a built panel into the rows a model is trained and judged on.

usable_rows     drop rows with no label
winsorize       clip the label at each decision time's quantiles, so one bad print (a shell going
                $1 -> $141 on no volume) can't dominate an average
add_rank_target fwd_rank: percentile of the forward return within (decision time, group), centred
                at 0. Scale-free; the optional "group-relative" target. within=None ranks across
                the whole decision time instead
make_batches    random batches of N per (decision time, group), leftovers pooled across groups
                and the remainder DROPPED. Legacy: the pick-1-of-10 design of jev.model; kept so
                the old numbers reproduce. New markets should leave `batches` off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def usable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=["fwd_return"]).reset_index(drop=True)


def winsorize(
    frame: pd.DataFrame, col: str, q: tuple[float, float], by: str = "decision_time"
) -> pd.DataFrame:
    g = frame.groupby(by)[col]
    lo, hi = g.transform("quantile", q[0]), g.transform("quantile", q[1])
    return frame.assign(**{col: frame[col].clip(lo, hi)})


def add_rank_target(
    frame: pd.DataFrame, within: str | None = "group", col: str = "fwd_return"
) -> pd.DataFrame:
    keys = ["decision_time"] + ([within] if within else [])
    return frame.assign(fwd_rank=frame.groupby(keys)[col].rank(pct=True) - 0.5)


def make_batches(
    frame: pd.DataFrame, size: int = 10, by: str = "group", seed: int = 0
) -> pd.DataFrame:
    """Port of jev.model.make_batches (period = decision time); same RNG order, so same batches."""
    rng = np.random.default_rng(seed)
    out = []

    def chunk(grp: pd.DataFrame, label: str) -> pd.DataFrame:
        grp = grp.iloc[rng.permutation(len(grp))]
        n_full = len(grp) // size * size
        for i in range(0, n_full, size):
            out.append(grp.iloc[i : i + size].assign(batch=f"{label}-{i // size}"))
        return grp.iloc[n_full:]

    for t, tgrp in frame.groupby("decision_time"):
        leftovers = [chunk(g, f"{t}-{k}") for k, g in tgrp.groupby(by)]
        chunk(pd.concat(leftovers), f"{t}-mixed")
    return pd.concat(out, ignore_index=True)
