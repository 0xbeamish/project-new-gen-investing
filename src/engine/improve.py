"""The feedback loop: weight every input, act, observe the returns, re-weight, try again.

One period, as if live (run_loop, `engine loop`):
  1 weights   the OUTER loop refits one weight per input (numbers and each text answer) from closed
              returns only
  2 cards     each candidate's card shows weight x input; a feedback note says what has worked
  3 act       the decider picks (Jev / Claude, or the model's own pick)
  4 observe   the period's returns close
  5 track     which inputs were over- or under-weighted, mis-shaped or decaying?
  6 fix       the next refit re-weights everything; a TEXT field that stays mis-weighted after K
              refits goes to the INNER loop: re-encode its answer, then rewrite or split its question
Both loops can "fix" the same symptom (a text field looks over-weighted because a numeric input it
correlates with is). The coordinator keeps them from fighting:
  1 the inner loop works on its own job: reading quality and INCREMENTAL information measured on the
    full model (joint attribution), never a field's raw correlation with returns
  2 weights first: an inner change is eligible only if the signal persisted through persist_k refits
  3 at most one structural change per period, in one loop; after a question changes, history is
    re-read with the new version and a full refit runs before anything else; the field then rests
  4 one judge: a change must improve the END-TO-END book, net of costs, on acceptance entities the
    tracking never looked at, at the registry's bar
  5 brakes: rejected candidates wait; automatic rollback if the book falls after an acceptance; an
    oscillation alarm freezes a field whose weight keeps flipping sign or whose question is
    rewritten back toward an earlier version
  6 the decider's note reports only: nothing in the loops reads it

The file reads in that order: run_loop and replay (the period loop and its FROZEN twin), then the
Coordinator's cycle (track, outer step, inner step), its judge, its brakes, and the settings.
"""

from __future__ import annotations

import difflib
import itertools
import json
import sys
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from engine import model, run, score
from engine.registry import required_t
from engine.text import tracking

MIN_TIME = pd.Timestamp.min.tz_localize("UTC")
MAX_TIME = pd.Timestamp.max.tz_localize("UTC")


# ---------------------------------------------------------------- the loop, period by period
def loop_settings(study) -> dict:
    """The YAML's `loop:` block with its defaults: what a logged loop pre-registers."""
    c = dict(study.cfg.get("loop") or {})
    last_year = (study.periods.tuning[1] - pd.Timedelta(days=1)).year
    band = (study.cfg.get("portfolio") or {}).get("band", {"enter": 0.9, "exit": 0.7})
    recency = c.get("recency", "auto")
    return {
        "name": c.get("name"),  # the registry name of the pre-registered test (--log)
        "features": c.get("features", "baseline"),
        "years": tuple(c.get("years", (study.periods.tuning[0].year, last_year))),
        "refit": c.get("refit", "M"),  # outer-loop refit every month (or Q, year)
        "recency": None if recency in (None, "none") else recency,
        "band": tuple(c.get("band", (band["enter"], band["exit"]))),
        "persist_k": int(c.get("persist_k", 3)),
        "cooldown": int(c.get("cooldown", 6)),
        "question_splits": int(c.get("question_splits", 3)),  # inner rung 3 proposals at most
    }


def run_loop(
    study,
    decider: str | None = None,
    out_dir: Path | None = None,
    log: bool = False,
    note: str = "",
    estimate_only: bool = False,
    progress=None,
) -> dict:
    """The feedback loop over the tuning years, period by period, against a FROZEN twin; writes
    loop_weights.csv, loop_changes.csv and loop_returns.csv. Descriptive unless log=True, which
    records ONE registry test under the YAML's pre-registered `loop.name`."""
    from engine import decide, panel
    from engine.spend import Ledger
    from engine.text.improve import PhraseSplitter

    s = loop_settings(study)
    if log:
        if not s["name"]:
            raise SystemExit("--log needs loop.name in the YAML, committed before the run")
        if (study.registry.own()["name"] == s["name"]).any():
            raise SystemExit(f"{s['name']!r} is already in the registry: one look only")
    feats = study.feature_set(s["features"])
    if not feats:
        raise SystemExit(f"feature set {s['features']!r} is empty: set loop.features in the YAML")
    kind = decider or (study.cfg.get("decider") or {}).get("kind", "none")
    dcfg = study.cfg.get("decider") or {}
    ledger = Ledger.from_config(study.cfg.get("spend"))
    if kind != "none":  # paid: price it on the starting model's cards before anything runs
        default_model = "jev-1.13.0" if kind == "jev" else "claude-sonnet-5-5"
        est = decide.estimate(
            decide.prepare_cards(study, s["features"]), ledger, dcfg.get("model", default_model)
        )
        print(json.dumps(est, indent=1), file=sys.stderr)
        if estimate_only:
            return {"decider": kind, "estimate": est}
        if est["projected_usd"] > ledger.remaining(dcfg.get("step", "decider")):
            raise SystemExit(f"estimate ${est['projected_usd']:.2f} is over the decider's cap")
    elif estimate_only:
        return {"decider": kind, "estimate": {"projected_usd": 0.0}}

    rows = panel.model_rows(study, run.build_panel(study, "tuning"))
    meta = {c: m for c, m in decide.text_feature_meta(study).items() if c in feats}
    splitter = (
        PhraseSplitter(study, rows, meta, max_splits=s["question_splits"])
        if meta and s["question_splits"] > 0
        else None
    )
    cfg = ReplayConfig(
        years=s["years"],
        recency=s["recency"],
        band=s["band"],
        loops=LoopsConfig(
            block=s["refit"], half_lives=(), persist_k=s["persist_k"], cooldown=s["cooldown"]
        ),
    )
    bar = study.registry.next_bar()
    res = replay(
        study,
        feats,
        meta,
        splitter.rows_for if splitter else (lambda version: rows),
        cfg,
        proposer=splitter,
        bar_offset=study.registry.n_judged(),
        progress=progress,
    )
    co, times = res.pop("_coordinator"), res.pop("_times")
    weights = weights_in_force(co, times)
    changes = co.log().rename(columns={"cutoff": "period"}) if co.events else pd.DataFrame()
    returns = res.pop("_by_period")
    dec = decide.make_decider(
        {"kind": kind, **{k: v for k, v in dcfg.items() if k not in ("kind", "feedback")}},
        ledger if kind != "none" else None,
    )
    picks = decide.over_loop(study, co, times, meta, dec, feedback=bool(dcfg.get("feedback", True)))
    if len(picks):
        per = picks.groupby("decision_time")[["pick_net", "model_net", "override"]].mean()
        returns = returns.join(per.rename(columns={"override": "override_rate"}))
        res["decider"] = {"kind": kind} | decide.grade_decisions(picks)
    out_dir = Path(out_dir or run.path(study.cfg.get("results_dir", f"results/{study.name}")))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "weights": out_dir / "loop_weights.csv",
        "changes": out_dir / "loop_changes.csv",
        "returns": out_dir / "loop_returns.csv",
    }
    weights.to_csv(files["weights"], index=False)
    changes.to_csv(files["changes"], index=False)
    returns.rename_axis("period").reset_index().to_csv(files["returns"], index=False)
    res |= {"market": study.name, "settings": s, "files": {k: str(p) for k, p in files.items()}}
    res |= {"_weights": weights, "_changes": changes, "_returns": returns}
    if log:
        r = res["result"]
        accepted = (
            int(changes["accepted"].sum() - (changes["loop"] == "brake").sum())
            if len(changes)
            else 0
        )
        res["registry"] = study.registry.record(
            {
                "kind": "test",
                "name": s["name"],
                "features": s["features"],
                "scope": "universal",
                "metric": "V1 portfolio net of measured costs, per period, loop ON minus FROZEN",
                "t_tune": round(r["diff_net_t"], 3),
                "bar_tune": round(bar, 3),
                "gain_tune": round(r["diff_net"], 6),
                "check_used": False,
                "kept": bool(r["diff_net_t"] >= bar and r["diff_net"] > 0),
                "note": (
                    f"{note} {r['months']} periods {s['years'][0]}-{s['years'][1]}; rank IC diff "
                    f"{r['diff_ic']:+.4f} (t {r['diff_ic_t']:.2f}); {res['inner_judged']} changes "
                    f"judged inside, {accepted} accepted; final {res['final_config']}"
                ).strip(),
            }
        )
    return res


def replay(
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
    """The loops ON, period by period over cfg.years, as if live, against a FROZEN twin (the starting
    inputs and plain yearly refits). Each period both see only closed data, the outer loop refits,
    the inner loop may accept one change, and that period's decisions use the configuration in
    force. Primary statistic, fixed in advance: the V1 portfolio (buy the group's top 10%, hold
    until out of its top 30%) net of costs, ON minus FROZEN, paired t over periods. rows_for(version)
    -> model rows. Nothing inside writes to the registry; the caller logs the result as ONE test."""
    rows = rows_for("v1")
    times = replay_months(study, rows, cfg.years)
    fr = frozen_scores(study, rows, features)
    co = Coordinator(
        study,
        features,
        meta,
        rows_for,
        cfg.loops,
        proposer=proposer,
        log_tests=False,
        bar_offset=bar_offset,
    )
    co.cur = ModelConfig(half_life=cfg.recency, alpha=float(study.cfg["model"].get("alpha", 10.0)))
    co.timeline = [(MIN_TIME, co.cur)]
    for k, t in enumerate(times):
        co.cycle(k, t)
        if progress:
            progress(k, t, co)
    on = co.stitched()
    cmp = compare_on_frozen(
        on, fr, times, cfg.band, run.label_overlap(study) if hasattr(study, "cfg") else 1
    )
    res = {
        "config": {**asdict(cfg), "loops": asdict(cfg.loops)},
        "result": {k: v for k, v in cmp.items() if not k.startswith("_")},
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
    return res | {"_coordinator": co, "_times": times, "_by_period": cmp["_by_period"]}


def replay_months(study, rows: pd.DataFrame, years: tuple[int, int]) -> list:
    """The decision times replayed."""
    t = pd.Series(sorted(rows["decision_time"].unique()))
    y = study.market.calendar.local_date(t).dt.year
    return list(t[(y >= years[0]) & (y <= years[1])])


def frozen_scores(study, rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The FROZEN twin: the starting inputs, plain yearly refits."""
    m = study.cfg["model"]
    wf = model.WalkForward(
        target="fwd_rank",
        model=model.ridge(float(m.get("alpha", 10.0))),
        block="year",
        min_train_periods=int(m.get("min_train_periods", 12)),
        keep=("group", "fwd_return", "rt_cost"),
    )
    return wf.run(rows, features, study.market.calendar)[0]


def compare_on_frozen(
    on: pd.DataFrame,
    frozen: pd.DataFrame,
    times: list,
    band: tuple[float, float],
    overlap: int = 1,
) -> dict:
    """V1 portfolio net of costs and rank IC: ON, FROZEN and ON minus FROZEN, paired by period
    (`_by_period`: the per-period series)."""
    on = score.fill_costs(on[on["decision_time"].isin(times)])
    fr = score.fill_costs(frozen[frozen["decision_time"].isin(times)])
    rule = score.Band(*band)
    p_on, p_fr = score.simulate_portfolio(on, rule), score.simulate_portfolio(fr, score.Band(*band))
    d = (p_on["net"] - p_fr["net"]).dropna()
    ic_on, ic_fr = score.rank_ic(on), score.rank_ic(fr)
    ic = (ic_on - ic_fr).dropna()
    by_period = pd.DataFrame(
        {
            "on_gross": p_on["gross"],
            "on_net": p_on["net"],
            "frozen_gross": p_fr["gross"],
            "frozen_net": p_fr["net"],
            "on_ic": ic_on,
            "frozen_ic": ic_fr,
        }
    ).sort_index()
    return {
        "months": len(d),
        "on_net": float(p_on["net"].mean()),
        "on_net_t": score.per_period_t(p_on["net"], overlap),
        "frozen_net": float(p_fr["net"].mean()),
        "frozen_net_t": score.per_period_t(p_fr["net"], overlap),
        "diff_net": float(d.mean()),
        "diff_net_t": score.per_period_t(d, overlap),
        "on_turnover": float(12 * p_on["turnover"].mean()),
        "frozen_turnover": float(12 * p_fr["turnover"].mean()),
        "ic_on": float(ic_on.mean()),
        "ic_frozen": float(ic_fr.mean()),
        "diff_ic": float(ic.mean()),
        "diff_ic_t": score.per_period_t(ic, overlap),
        "_by_period": by_period,
    }


def config_at(co: Coordinator, t) -> ModelConfig:
    """The configuration in force for decisions at t."""
    tl = co.timeline or [(MIN_TIME, co.cur)]
    return [c for t0, c in tl if t0 <= t][-1]


def weights_in_force(co: Coordinator, times: list) -> pd.DataFrame:
    """One row per (period, input): the weight that period's decisions used, and its config."""
    out = []
    for t in times:
        c = config_at(co, t)
        scored, w = co.scores(c)[:2]
        b = scored.loc[scored["decision_time"] == t, "block"]
        if b.empty or b.iloc[0] not in w.index:
            continue
        out += [
            {"period": t, "input": i, "weight": float(v), "config": c.key()}
            for i, v in w.loc[b.iloc[0]].items()
        ]
    return pd.DataFrame(out, columns=["period", "input", "weight", "config"])


# ---------------------------------------------------------------- one cycle: track, outer, inner
class Coordinator:
    """Runs one cycle per closed period under the coordination rules: track, then the outer step
    (weights), then the inner step (encoding, question, drop), each change judged end to end.

    rows_for(question_version) -> model rows (re-reads history for a new version; never splices)
    proposer: optional, propose_field(field, meta, cutoff, coordinator) -> (version, question text
              [, columns it adds]) | None, where rows_for knows how to build that version (the
              question rung; without one it is skipped)
    """

    def __init__(
        self,
        study,
        features: list[str],
        text_meta: dict,
        rows_for: Callable[[str], pd.DataFrame],
        cfg: LoopsConfig | None = None,
        registry=None,
        proposer=None,
        reading: pd.DataFrame | None = None,
        bar_offset: int = 0,
        log_tests: bool = True,
    ):
        self.study, self.features, self.meta = study, list(features), text_meta
        self.rows_for, self.cfg = rows_for, cfg or LoopsConfig()
        self.registry, self.proposer, self.reading = registry, proposer, reading
        self.bar_offset, self.log_tests = bar_offset, log_tests
        self.cal = study.market.calendar
        self.cur = ModelConfig(alpha=float(study.cfg["model"].get("alpha", 10.0)))
        self.timeline: list[tuple[pd.Timestamp, ModelConfig]] = []
        self.events: list[Event] = []
        self.judged = 0
        self.streak: dict[str, dict[str, int]] = {}
        self.rest_until: dict[str, int] = {}
        self.frozen_fields: set[str] = set()
        self.blocked_until_refit = False
        self.last_accept: tuple[int, pd.Timestamp, ModelConfig, ModelConfig] | None = None
        self.weight_signs: dict[str, list[float]] = {}
        self._last_block = None
        self.question_texts: dict[str, list[str]] = {}
        self._cache: dict = {}
        self.reports: list[dict] = []
        self.judged_at: dict[str, int] = {}  # candidate key -> cycle it was last judged
        self._accepted = self._judged_now = 0  # this cycle's changes (rules 3 and 6)

    def run(self, cutoffs) -> pd.DataFrame:
        """One cycle per cutoff; returns the event log."""
        for k, t in enumerate(cutoffs):
            self.cycle(k, t)
        return self.log()

    def cycle(self, k: int, cutoff: pd.Timestamp) -> dict:
        """One closed period: track, brakes, the outer step, the inner step; returns the report."""
        cutoff = pd.Timestamp(cutoff)
        if not self.timeline:
            self.timeline.append((MIN_TIME, self.cur))
        rep, cols = self.track(k, cutoff)
        if self.rollback_if_worse(k, cutoff):
            return rep
        if self.blocked_until_refit:  # rule 3: a full outer refit with the new version comes first
            self.blocked_until_refit = False
            return rep
        self._accepted = self._judged_now = 0
        fl = rep["flags"].set_index("field")
        self.outer_step(k, cutoff, fl)
        self.inner_step(k, cutoff, fl, rep, cols)
        return rep

    def track(self, k: int, cutoff) -> tuple[dict, list[str]]:
        """The weights in force, the oscillation alarm, the tracking report on diagnosis entities,
        and the persistence streaks (counted per REFIT, not per cycle)."""
        scored, w, enc, cols = self.scores(self.cur)
        w_now = w[[self._block_start(b) < cutoff for b in w.index]]
        # with yearly refits a monthly cycle sees the same weights 12 times: that is one observation
        new_refit = bool(len(w_now)) and w_now.index[-1] != self._last_block
        if new_refit:
            self._last_block = w_now.index[-1]
            self.oscillation_alarm(w_now, k, cutoff)
        diag = [
            e for e in enc["entity_id"].unique() if not acceptance_entity(e, self.cfg.test_share)
        ]
        meta = {c: self.meta[c] for c in cols if c in self.meta}
        rep = tracking.report(
            enc,
            scored,
            w_now,
            cols,
            meta,
            self.cal,
            cutoff=cutoff,
            entities=diag,
            reading=self.reading,
        )
        self.reports.append({"cycle": k, "cutoff": cutoff, "flags": rep["flags"]})
        fl = rep["flags"].set_index("field")
        for f in fl.index if new_refit else []:
            s = self.streak.setdefault(f, {"misweight": 0, "shape": 0, "misread": 0})
            s["misweight"] = s["misweight"] + 1 if fl.loc[f, "status"] != "ok" else 0
            s["shape"] = s["shape"] + 1 if fl.loc[f, "non_linear"] else 0
            s["misread"] = s["misread"] + 1 if fl.loc[f, "misread"] else 0
        return rep, cols

    def outer_step(self, k: int, cutoff, fl: pd.DataFrame) -> None:
        """Rung 1, the outer loop: a new recency half-life or shrinkage for ALL weights, whenever
        anything is mis-weighted or decaying. (The refit itself happens every period regardless.)"""
        flagged = fl[(fl["status"] != "ok") | fl["decaying"]]
        if not (self._room() and len(flagged)):
            return
        cands = [
            replace(self.cur, half_life=h, alpha=a)
            for h in self.cfg.half_lives
            for a in self.cfg.alphas
            if (h, a) != (self.cur.half_life, self.cur.alpha)
        ]
        pick = self.screen(cands, cutoff)
        if pick:
            c, _r = pick
            self._judged_now += 1
            trig = "; ".join(
                f"{f} {fl.loc[f, 'status']} (t {fl.loc[f, 't_joint']:.1f})"
                for f in flagged.index[:3]
            )
            change = f"half_life={c.half_life}, alpha={c.alpha}"
            self._accepted += self.judge_change(
                k, cutoff, "outer", "weights", "all", c, trig, change
            )

    def inner_step(self, k: int, cutoff, fl: pd.DataFrame, rep: dict, cols: list[str]) -> None:
        """Rungs 2-4, the inner loop, on text fields that stayed wrong through persist_k refits:
        re-encode a mis-shaped answer, rewrite or split a mis-weighted question, drop a useless one."""
        self.reencode(k, cutoff, fl)
        if self.proposer is not None:
            self.rewrite_question(k, cutoff, fl)
        if self._room():
            self.drop_useless(k, cutoff, rep, cols)

    def eligible(self, f: str, kind: str, k: int) -> bool:
        """Rule 2: a text field whose `kind` flag persisted through persist_k refits, rested, not
        frozen."""
        base = f.split("__")[0]
        return (
            (self.meta.get(base) or {}).get("tag") is not None
            and self.streak.get(f, {}).get(kind, 0) >= self.cfg.persist_k
            and self.rest_until.get(base, -1) < k
            and base not in self.frozen_fields
            and (self.meta.get(base) or {}).get("question") not in self.frozen_fields
        )

    def reencode(self, k: int, cutoff, fl: pd.DataFrame) -> None:
        """Rung 2: per-level effects (one-hot or monotone steps) for a field that is non-linear."""
        for f in [
            f
            for f in fl.index
            if self.eligible(f, "shape", k) and f not in dict(self.cur.encodings)
        ]:
            if not self._room():
                break
            cands = [
                replace(
                    self.cur, encodings=tuple(sorted((dict(self.cur.encodings) | {f: e}).items()))
                )
                for e in self.cfg.encodings
            ]
            pick = self.screen(cands, cutoff)
            if pick:
                c, _r = pick
                self._judged_now += 1
                ok = self.judge_change(
                    k,
                    cutoff,
                    "inner",
                    "encoding",
                    f,
                    c,
                    f"non-linear by level (max shape t {fl.loc[f, 'max_shape_t']:.1f}) for {self.streak[f]['shape']} refits",
                    f"{f} -> {dict(c.encodings)[f]}",
                )
                self._accepted += ok
                if ok:
                    self.rest_until[f] = k + self.cfg.cooldown

    def rewrite_question(self, k: int, cutoff, fl: pd.DataFrame) -> None:
        """Rung 3: a mis-weight the weights didn't fix -> the proposer's new question version."""
        for f in [f for f in fl.index if self.eligible(f, "misweight", k)]:
            if not self._room():
                break
            prop = self.proposer.propose_field(f, self.meta.get(f), cutoff, self)
            if prop is None:
                continue
            version, text, *more = prop  # optional third item: the columns it adds
            if self.question_changed(self.meta[f]["question"], text, k, cutoff):
                continue
            added = tuple(sorted(set(self.cur.added) | set(more[0] if more else ())))
            c = replace(self.cur, questions=version, added=added)
            if self.screen([c], cutoff) is None:
                continue
            self._judged_now += 1
            ok = self.judge_change(
                k,
                cutoff,
                "inner",
                "question",
                f,
                c,
                f"{fl.loc[f, 'status']} (joint t {fl.loc[f, 't_joint']:.1f}) for {self.streak[f]['misweight']} refits",
                text,
            )
            self._accepted += ok
            if ok:
                self.blocked_until_refit = True
                self.rest_until[f] = k + self.cfg.cooldown

    def drop_useless(self, k: int, cutoff, rep: dict, cols: list[str]) -> None:
        """Rung 4: drop a text field whose incremental value stayed ~0 (or harmful) over a long
        window, for persist_k cycles."""
        for f in [
            c
            for c in cols
            if c in self.meta and self.rest_until.get(c, -1) < k and c not in self.frozen_fields
        ]:
            if not self._room():
                break
            ic = rep["ic"].set_index("field")
            if f not in ic.index or rep["periods"] < self.cfg.drop_window:
                continue
            inc = self.incremental(f, cutoff)
            st = self.streak.setdefault(f, {"misweight": 0, "shape": 0, "misread": 0})
            useless = inc["t"] < 1.0 and inc["periods"] >= self.cfg.drop_window
            st["useless"] = st.get("useless", 0) + 1 if useless else 0
            if st["useless"] < self.cfg.persist_k:
                continue
            c = replace(self.cur, dropped=tuple(sorted(set(self.cur.dropped) | {f})))
            if self.screen([c], cutoff):
                self._judged_now += 1
                ok = self.judge_change(
                    k,
                    cutoff,
                    "inner",
                    "add_drop",
                    f,
                    c,
                    f"incremental IC {inc['gain']:+.4f} (t {inc['t']:.2f}) over {inc['periods']} periods",
                    f"drop {f}",
                )
                self._accepted += ok
                if ok:
                    self.rest_until[f] = k + self.cfg.cooldown

    def _room(self) -> bool:
        return (
            self._accepted < self.cfg.max_changes_per_cycle
            and self._judged_now < self.cfg.max_judged_per_cycle
        )

    # ---------- the judge: one end-to-end test on acceptance entities
    def scores(self, c: ModelConfig):
        """(scored, weights, encoded rows, inputs) for a config; the walk-forward is causal, so one
        run serves every cutoff."""
        if c.key() not in self._cache:
            rows = self.rows_for(c.questions)
            enc, cols = encode_fields(rows, self.features, c, self.meta)
            m = self.study.cfg["model"]
            wf = model.WalkForward(
                target="fwd_rank",
                model=model.ridge(c.alpha),
                block=self.cfg.block,
                min_train_periods=int(m.get("min_train_periods", 12)),
                keep=("group", "fwd_return", "rt_cost"),
                half_life=c.half_life,
            )
            scored, w = wf.run(enc, cols, self.cal)
            self._cache[c.key()] = (scored, w, enc, cols)
        return self._cache[c.key()]

    def paired(self, cand: ModelConfig, cur: ModelConfig, cutoff, accept: bool, since=None) -> dict:
        """Per-period net book, candidate minus current, closed periods, one entity split."""
        out = []
        for c in (cand, cur):
            sc = self.scores(c)[0]
            rows = self.rows_for(c.questions)[["entity_id", "decision_time", "label_end"]]
            sc = sc.merge(rows, on=["entity_id", "decision_time"], how="left")
            sc = sc[sc["label_end"] < cutoff]
            if since is not None:
                sc = sc[sc["decision_time"] >= since]
            out.append(self._net(self._split(sc, accept)))
        d = (out[0] - out[1]).dropna()
        return {
            "gain": float(d.mean()) if len(d) else np.nan,
            "t": score.per_period_t(d),
            "periods": len(d),
        }

    def screen(self, cands: list[ModelConfig], cutoff) -> tuple[ModelConfig, dict] | None:
        """Choose among candidates on DIAGNOSIS entities (not a test, not logged); a candidate
        judged recently waits rejudge_after cycles."""
        best = None
        k = len(self.reports) - 1
        cands = [
            c for c in cands if k - self.judged_at.get(c.key(), -(10**9)) >= self.cfg.rejudge_after
        ]
        for c in cands:
            r = self.paired(c, self.cur, cutoff, accept=False)
            if (
                r["t"] == r["t"]
                and r["t"] >= self.cfg.diag_t
                and (best is None or r["t"] > best[1]["t"])
            ):
                best = (c, r)
        return best

    def bar(self) -> float:
        """The registry's next bar (plus the changes this coordinator judged without logging)."""
        n = (self.registry.n_judged() if self.registry is not None else 0) + self.bar_offset + 1
        return required_t(n)

    def judge_change(
        self, cycle, cutoff, loop, rung, fld, cand: ModelConfig, trigger: str, change: str
    ) -> bool:
        """Rule 4, one judge: the candidate must beat the current system end to end, net of costs,
        on acceptance entities, at the bar. Every judged change counts, accepted or not."""
        self.judged_at[cand.key()] = cycle
        bar = self.bar()
        r = self.paired(cand, self.cur, cutoff, accept=True)
        ok = bool(r["t"] >= bar and r["gain"] > 0)
        self.judged += 1
        if not self.log_tests:
            self.bar_offset += 1
        elif self.registry is not None:
            self.registry.record(
                {
                    "kind": "test",
                    "name": f"loop:{rung}:{fld}",
                    "features": change,
                    "scope": "universal",
                    "metric": f"end_to_end_{self.cfg.metric}_net_gain (acceptance split)",
                    "t_tune": round(r["t"], 3) if r["t"] == r["t"] else None,
                    "bar_tune": round(bar, 3),
                    "gain_tune": round(r["gain"], 6) if r["gain"] == r["gain"] else None,
                    "check_used": False,
                    "kept": ok,
                    "note": f"{loop} loop, cycle {cycle}, {r['periods']} periods; trigger: {trigger}",
                }
            )
        self.events.append(
            Event(cycle, cutoff, loop, rung, fld, change, trigger, ok, r["t"], bar, r["gain"])
        )
        if ok:
            self.last_accept = (cycle, cutoff, self.cur, cand)
            self.cur = cand
            self.timeline.append((cutoff, cand))
        return ok

    def incremental(self, f: str, cutoff) -> dict:
        """A field's value = the full model's rank IC with it minus without it, per period, on
        diagnosis entities (never its raw correlation with returns)."""
        sc_with = self.scores(self.cur)[0]
        sc_without = self.scores(
            replace(self.cur, dropped=tuple(sorted(set(self.cur.dropped) | {f})))
        )[0]
        rows = self.rows_for(self.cur.questions)[["entity_id", "decision_time", "label_end"]]
        a = sc_with.merge(rows, on=["entity_id", "decision_time"])
        b = sc_without.merge(rows, on=["entity_id", "decision_time"])
        a, b = a[a["label_end"] < cutoff], b[b["label_end"] < cutoff]
        g = (score.rank_ic(self._split(a, False)) - score.rank_ic(self._split(b, False))).dropna()
        return {
            "gain": float(g.mean()) if len(g) else np.nan,
            "t": score.per_period_t(g),
            "periods": len(g),
        }

    def _split(self, scored: pd.DataFrame, accept: bool) -> pd.DataFrame:
        a = scored["entity_id"].map(lambda e: acceptance_entity(e, self.cfg.test_share))
        return scored[a == accept]

    def _net(self, scored: pd.DataFrame) -> pd.Series:
        s = score.fill_costs(scored)
        if self.cfg.metric == "rank_weighted":
            return score.rank_weighted(s)["net"]
        return score.quantile_spreads(s, self.cfg.spread_q, self.cfg.min_names)["net"]

    # ---------- brakes
    def oscillation_alarm(self, weights: pd.DataFrame, cycle: int, cutoff) -> None:
        """Freeze a field whose meaningful weight flipped sign oscillation_flips times in a window."""
        if not len(weights):
            return
        signs = meaningful_signs(weights.iloc[-1], self.cfg.oscillation_min)
        for f, sign in signs.items():  # per model column: an encoded field's levels may differ
            self.weight_signs.setdefault(f, []).append(sign)
            flips = count_flips(self.weight_signs[f][-self.cfg.oscillation_window :])
            if flips >= self.cfg.oscillation_flips and f not in self.frozen_fields:
                self.frozen_fields.add(f)
                self.events.append(
                    Event(
                        cycle,
                        cutoff,
                        "brake",
                        "oscillation",
                        f,
                        "",
                        f"weight sign flipped {flips} times in {self.cfg.oscillation_window} refits",
                        False,
                        note="field frozen for structural changes",
                    )
                )

    def question_changed(self, qid: str, new_text: str, cycle: int, cutoff) -> bool:
        """Record a question rewrite; alarm (and freeze) if it moves back toward an earlier version."""
        hist = self.question_texts.setdefault(qid, [])
        alarm = False
        if len(hist) >= 2:
            prev, older = hist[-1], hist[:-1]
            r_prev = difflib.SequenceMatcher(None, new_text, prev).ratio()
            if any(difflib.SequenceMatcher(None, new_text, o).ratio() > r_prev for o in older):
                alarm = True
                self.frozen_fields.add(qid)
                self.events.append(
                    Event(
                        cycle,
                        cutoff,
                        "brake",
                        "oscillation",
                        qid,
                        "",
                        "question rewritten back toward an earlier version",
                        False,
                        note="question frozen",
                    )
                )
        hist.append(new_text)
        return alarm

    def rollback_if_worse(self, cycle: int, cutoff) -> bool:
        """Undo the last acceptance if the book fell after it (t <= rollback_t over >= 3 periods)."""
        if self.last_accept is None:
            return False
        _c0, t0, prev, new = self.last_accept
        r = self.paired(new, prev, cutoff, accept=True, since=t0)
        if r["periods"] >= 3 and r["t"] == r["t"] and r["t"] <= self.cfg.rollback_t:
            self.cur = prev
            self.timeline.append((cutoff, prev))
            self.events.append(
                Event(
                    cycle,
                    cutoff,
                    "brake",
                    "rollback",
                    "-",
                    new.key(),
                    f"end-to-end fell after acceptance (t {r['t']:.2f} over {r['periods']} periods)",
                    True,
                    r["t"],
                    np.nan,
                    r["gain"],
                )
            )
            self.last_accept = None
            return True
        if r["periods"] >= self.cfg.rollback_m:
            self.last_accept = None  # watched long enough
        return False

    # ---------- the record
    def log(self) -> pd.DataFrame:
        """Every judged change and brake, in order."""
        return pd.DataFrame([e.__dict__ for e in self.events])

    def stitched(self) -> pd.DataFrame:
        """The ON system's scores: each decision uses the config in force at that time."""
        parts = []
        tl = self.timeline or [(MIN_TIME, self.cur)]
        for i, (t0, c) in enumerate(tl):
            t1 = tl[i + 1][0] if i + 1 < len(tl) else MAX_TIME
            sc = self.scores(c)[0]
            parts.append(sc[(sc["decision_time"] >= t0) & (sc["decision_time"] < t1)])
        return pd.concat(parts, ignore_index=True)

    def _block_start(self, b) -> pd.Timestamp:
        if isinstance(b, pd.Period):
            return pd.Timestamp(b.start_time, tz="UTC")
        return pd.Timestamp(f"{int(b)}-01-01", tz="UTC")


# ---------------------------------------------------------------- settings and helpers
@dataclass(frozen=True)
class ModelConfig:
    """One configuration of the full system: recency, shrinkage, encodings, drops, question version."""

    half_life: float | None = None
    alpha: float = 10.0
    encodings: tuple = ()  # ((feature, onehot | monotone | surprise), ...)
    dropped: tuple = ()  # features removed
    questions: str = "v1"  # which question-set version the text features come from
    added: tuple = ()  # features a question split / add brought in

    def key(self) -> str:
        """A readable id, also the score cache key."""
        return f"hl={self.half_life}|a={self.alpha}|enc={dict(self.encodings)}|drop={list(self.dropped)}|q={self.questions}|add={list(self.added)}"


@dataclass
class LoopsConfig:
    """The ladder's grids and the coordination rules' thresholds, fixed in advance."""

    half_lives: tuple = ("auto",)  # the outer alternative; "auto" = chosen inside the refits
    alphas: tuple = (10.0,)
    encodings: tuple = ("onehot", "monotone")
    persist_k: int = 3
    cooldown: int = 6  # cycles a field rests after a change
    max_changes_per_cycle: int = 1
    max_judged_per_cycle: int = 1
    rollback_m: int = 6  # periods watched after an acceptance
    rollback_t: float = -1.0
    oscillation_window: int = 12
    oscillation_flips: int = 2  # sign flips within the window that freeze a field
    # A flip counts only between refits where the weight was meaningfully non-zero: |w| above this
    # fraction of that refit's median |w| (inputs are centred ranks, so weights share one scale).
    # 0 = every sign change counts.
    oscillation_min: float = 0.5
    test_share: float = 0.4  # acceptance entities (by hash): the frozen test split
    diag_t: float = 1.0  # a candidate must show this much on diagnosis entities to be judged
    metric: str = "rank_weighted"  # end-to-end judge: rank_weighted | quantile (net either way)
    spread_q: float = 0.2
    min_names: int = 5
    drop_window: int = 24
    block: str = "Q"
    rejudge_after: int = 6  # cycles a rejected candidate waits before it can be judged again


@dataclass
class ReplayConfig:
    """Settings fixed in advance: the years replayed, the ON outer loop's recency, the V1 band and
    the loops' settings (monthly cycles)."""

    years: tuple[int, int] = (2013, 2019)
    recency: float | str | None = "auto"  # the ON outer loop's half-life, fixed in advance
    band: tuple[float, float] = (0.9, 0.7)  # V1: enter top 10%, exit below top 30% (in group)
    loops: LoopsConfig = field(
        default_factory=lambda: LoopsConfig(block="M", half_lives=(), persist_k=3, cooldown=6)
    )


@dataclass
class Event:
    """One judged change or brake, for the log."""

    cycle: int
    cutoff: pd.Timestamp
    loop: str  # outer | inner | brake
    rung: str
    field: str
    change: str
    trigger: str
    accepted: bool
    t: float = np.nan
    bar: float = np.nan
    gain: float = np.nan
    note: str = ""


def meaningful_signs(w: pd.Series, min_frac: float) -> pd.Series:
    """Sign of each weight, or 0 when |w| <= min_frac x the refit's median |w| (near zero)."""
    floor = min_frac * float(w.abs().median())
    return pd.Series(np.where(w.abs() > floor, np.sign(w), 0.0), index=w.index)


def count_flips(signs: list[float]) -> int:
    """Sign changes between consecutive meaningful (non-zero) entries: near-zero refits are
    skipped, so noise wobbling around 0 never counts as a flip."""
    s = [x for x in signs if x != 0]
    return sum(1 for a, b in itertools.pairwise(s) if a != b)


def acceptance_entity(entity_id: str, share: float) -> bool:
    """A fixed hash split: True for the acceptance (frozen test) entities."""
    return zlib.crc32(("accept|" + str(entity_id)).encode()) % 1000 < share * 1000


def encode_fields(
    rows: pd.DataFrame, features: list[str], cfg: ModelConfig, meta: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the config's encodings and drops; returns (rows with new columns, model inputs)."""
    enc = dict(cfg.encodings)
    out, cols, new = rows, [], {}
    for f in features:
        if f in cfg.dropped:
            continue
        how = enc.get(f)
        if how is None:
            cols.append(f)
        elif how == "onehot":
            lv = rows[f].round().clip(0, 4)
            for k in range(5):
                new[f"{f}__eq{k}"] = (lv == k).astype(float).where(rows[f].notna())
                cols.append(f"{f}__eq{k}")
        elif how == "monotone":
            for k in range(1, 5):
                new[f"{f}__ge{k}"] = (rows[f] >= k - 0.5).astype(float).where(rows[f].notna())
                cols.append(f"{f}__ge{k}")
        elif how == "surprise":
            s = f[: -len("_level")] + "_surp" if f.endswith("_level") else f + "_surp"
            if s not in rows:
                raise ValueError(
                    f"surprise encoding needs {s} in the panel (text source surprise: on)"
                )
            cols.append(s)
    cols += [c for c in cfg.added if c in rows and c not in cols and c not in cfg.dropped]
    if new:
        out = rows.assign(**new)
    return out, cols
