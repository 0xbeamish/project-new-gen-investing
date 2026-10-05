"""History replay: run the loops month by month over tuning years, as if live, against a frozen twin.

Each month t, in order:
  1. both loops see only closed data (labels ended before t)
  2. the outer loop refits the weights (every month, recency option fixed in advance)
  3. the inner loop may accept at most one structural change under the coordination rules
     (engine.loops), re-reading history with a new question version when one changes
  4. the month's decisions use the configuration in force at t
FROZEN: the starting questions and plain yearly-refit weights, on the same months and universe.

Primary statistic (fixed in advance): the V1 low-turnover portfolio (buy the group's top 10%, hold
until out of its top 30%), net of measured costs, per month: ON minus FROZEN, paired t from the
monthly differences. Secondary: the rank-IC difference. One registry test for the whole replay;
the changes judged inside it are counted in the replay's own log (they raise the bar the replay's
inner judges use, not the main registry).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from engine import loops, models, pipeline, scoring


@dataclass
class ReplayConfig:
    years: tuple[int, int] = (2013, 2019)
    recency: float | str | None = (
        "auto"  # the ON outer loop's half-life, fixed in advance
    )
    band: tuple[float, float] = (
        0.9,
        0.7,
    )  # V1: enter top 10%, exit below top 30% (within group)
    loops: loops.LoopsConfig = field(
        default_factory=lambda: loops.LoopsConfig(
            block="M", half_lives=(), persist_k=3, cooldown=6
        )
    )


def months(study, rows: pd.DataFrame, years: tuple[int, int]) -> list:
    t = pd.Series(sorted(rows["decision_time"].unique()))
    y = study.market.calendar.local_date(t).dt.year
    return list(t[(y >= years[0]) & (y <= years[1])])


def frozen_scores(study, rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    m = study.cfg["model"]
    wf = models.WalkForward(
        target="fwd_rank",
        model=models.ridge(float(m.get("alpha", 10.0))),
        block="year",
        min_train_periods=int(m.get("min_train_periods", 12)),
        keep=("group", "fwd_return", "rt_cost"),
    )
    return wf.run(rows, features, study.market.calendar)[0]


def compare(
    on: pd.DataFrame,
    frozen: pd.DataFrame,
    times: list,
    band: tuple[float, float],
    overlap: int = 1,
) -> dict:
    on = scoring.fill_costs(on[on["decision_time"].isin(times)])
    fr = scoring.fill_costs(frozen[frozen["decision_time"].isin(times)])
    rule = scoring.Band(*band)
    p_on, p_fr = scoring.simulate(on, rule), scoring.simulate(fr, scoring.Band(*band))
    d = (p_on["net"] - p_fr["net"]).dropna()
    ic = (scoring.rank_ic(on) - scoring.rank_ic(fr)).dropna()
    return {
        "months": len(d),
        "on_net": float(p_on["net"].mean()),
        "on_net_t": scoring.per_period_t(p_on["net"], overlap),
        "frozen_net": float(p_fr["net"].mean()),
        "frozen_net_t": scoring.per_period_t(p_fr["net"], overlap),
        "diff_net": float(d.mean()),
        "diff_net_t": scoring.per_period_t(d, overlap),
        "on_turnover": float(12 * p_on["turnover"].mean()),
        "frozen_turnover": float(12 * p_fr["turnover"].mean()),
        "ic_on": float(scoring.rank_ic(on).mean()),
        "ic_frozen": float(scoring.rank_ic(fr).mean()),
        "diff_ic": float(ic.mean()),
        "diff_ic_t": scoring.per_period_t(ic, overlap),
    }


def run(
    study,
    features: list[str],
    meta: dict,
    rows_for,
    cfg: ReplayConfig,
    proposer=None,
    bar_offset: int = 0,
    out_dir: Path | None = None,
    progress=None,
) -> dict:
    rows = rows_for("v1")
    times = months(study, rows, cfg.years)
    fr = frozen_scores(study, rows, features)
    co = loops.Coordinator(
        study,
        features,
        meta,
        rows_for,
        cfg.loops,
        proposer=proposer,
        log_tests=False,
        bar_offset=bar_offset,
    )
    co.cur = loops.ModelConfig(
        half_life=cfg.recency, alpha=float(study.cfg["model"].get("alpha", 10.0))
    )
    co.timeline = [(pd.Timestamp.min.tz_localize("UTC"), co.cur)]
    for k, t in enumerate(times):
        co.cycle(k, t)
        if progress:
            progress(k, t, co)
    on = co.stitched()
    res = {
        "config": {**asdict(cfg), "loops": asdict(cfg.loops)},
        "result": compare(
            on,
            fr,
            times,
            cfg.band,
            pipeline.overlap(study) if hasattr(study, "cfg") else 1,
        ),
    }
    res["changes"] = (
        co.log()
        .drop(columns=["cutoff"])
        .assign(cutoff=[str(e.cutoff) for e in co.events])
        .to_dict("records")
        if co.events
        else []
    )
    res["inner_judged"] = co.judged
    res["final_config"] = co.cur.key()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay.json").write_text(json.dumps(res, indent=1, default=str))
        on.to_pickle(out_dir / "on_scores.pkl")
    return res
