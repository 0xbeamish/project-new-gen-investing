"""The adjustment ladder and the coordinator that keeps the outer and inner loops from fighting.

Two loops can fix a mis-priced text field:
  outer  the weights over ALL inputs (walk-forward refit, recency half-life, shrinkage). Every period
  inner  the text side: encoding of an answer, the question's wording, add / drop, the reader
A field can look wrong because ITS reading is wrong, or because a numeric input it correlates with
is mis-weighted. The rules below make the inner loop fix only its own errors.

Ladder (cheapest first; each rung's candidate is chosen on DIAGNOSIS entities, then judged once on
the frozen ACCEPTANCE split; every judged change is a registry test, accepted or not)
  1 weights    recency half-life / shrinkage from a pre-set grid (outer)
  2 encoding   per-level effects for a scale field: one-hot or monotone steps; or surprise vs its
               history base rate (inner)
  3 question   rewrite or split from residual mining: the documents behind the largest
               LEAVE-TEXT-OUT residuals go to a proposer (free stub by default; paid off) (inner)
  4 add / drop drop a field whose incremental value (full model with vs without it) is ~0 over a
               long window; add a question when large-residual documents share an uncovered phrase
  5 reader     misread fields (gold skill, spot-check corrections, probe stability) go to the
               question-improvement loop (engine.text.textloop) or a reader change

Coordination rules (enforced here)
  1 inner objective = its own job: reading quality, and INCREMENTAL information measured on the full
    outer model (joint residual attribution, leave-out gains), never raw correlation with returns
  2 weights first: the outer refit runs every period; an inner structural change is eligible only
    if the field's signal persisted through >= persist_k consecutive outer refits
  3 never in the same cycle: at most one structural change per cycle, in one loop; after a question
    changes, history is re-read with the new version (versioned answers, never spliced) and a full
    outer refit runs before any further change; a per-field cooldown follows every change
  4 one judge: any change, inner or outer, must improve the END-TO-END system (full model, net of
    costs) on the acceptance split, at the registry's bar; inner metrics are necessary, not enough
  5 diagnosis and acceptance use different entities
  6 brakes: max changes per cycle; automatic rollback if the end-to-end score degrades over the
    next M periods; an oscillation alarm (a weight flipping sign >= 2 times in a window, or a
    question rewritten back toward an earlier version) freezes that field
  7 the decider's feedback note reports only: nothing here reads it
"""

from __future__ import annotations

import difflib
import itertools
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from engine import models, scoring
from engine.text import tracking

RUNGS = ("weights", "encoding", "question", "add_drop", "reader")


@dataclass(frozen=True)
class ModelConfig:
    half_life: float | None = None
    alpha: float = 10.0
    encodings: tuple = ()  # ((feature, onehot | monotone | surprise), ...)
    dropped: tuple = ()  # features removed
    questions: str = "v1"  # which question-set version the text features come from
    added: tuple = ()  # features a question split / add brought in

    def key(self) -> str:
        return f"hl={self.half_life}|a={self.alpha}|enc={dict(self.encodings)}|drop={list(self.dropped)}|q={self.questions}|add={list(self.added)}"


@dataclass
class LoopsConfig:
    half_lives: tuple = (
        "auto",
    )  # the outer alternative; "auto" picks a half-life inside every refit
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
    test_share: float = 0.4  # acceptance entities (by hash): the frozen test split
    diag_t: float = (
        1.0  # a candidate must show this much on diagnosis entities to be judged
    )
    metric: str = "rank_weighted"  # end-to-end judge: rank_weighted | quantile (net of costs either way)
    spread_q: float = 0.2
    min_names: int = 5
    drop_window: int = 24
    block: str = "Q"
    rejudge_after: int = (
        6  # a rejected candidate waits this many cycles before it can be judged again
    )


def acceptance_entity(entity_id: str, share: float) -> bool:
    return zlib.crc32(("accept|" + str(entity_id)).encode()) % 1000 < share * 1000


def encode(
    rows: pd.DataFrame, features: list[str], cfg: ModelConfig, meta: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the config's encodings; returns (rows with new columns, model inputs)."""
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
                new[f"{f}__ge{k}"] = (
                    (rows[f] >= k - 0.5).astype(float).where(rows[f].notna())
                )
                cols.append(f"{f}__ge{k}")
        elif how == "surprise":
            s = f[: -len("_level")] + "_surp" if f.endswith("_level") else f + "_surp"
            if s not in rows:
                raise ValueError(
                    f"surprise encoding needs {s} in the panel (text source surprise: on)"
                )
            cols.append(s)
    cols += [
        c for c in cfg.added if c in rows and c not in cols and c not in cfg.dropped
    ]
    if new:
        out = rows.assign(**new)
    return out, cols


@dataclass
class Event:
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


class Coordinator:
    """Runs the ladder under the coordination rules, one cycle per closed period.

    rows_for(question_version) -> model rows (re-reads history for a new version; never splices)
    proposer: optional, propose_field(field, meta, docs) -> (new_version_name, description) | None,
              where the caller's rows_for knows how to build that version
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
        self.last_accept: tuple[int, pd.Timestamp, ModelConfig, ModelConfig] | None = (
            None
        )
        self.weight_signs: dict[str, list[float]] = {}
        self.question_texts: dict[str, list[str]] = {}
        self._cache: dict = {}
        self.reports: list[dict] = []
        self.judged_at: dict[str, int] = {}  # candidate key -> cycle it was last judged

    # ---------- scores per config (the walk-forward is causal, so one run serves every cutoff)
    def scores(self, c: ModelConfig):
        if c.key() not in self._cache:
            rows = self.rows_for(c.questions)
            enc, cols = encode(rows, self.features, c, self.meta)
            m = self.study.cfg["model"]
            wf = models.WalkForward(
                target="fwd_rank",
                model=models.ridge(c.alpha),
                block=self.cfg.block,
                min_train_periods=int(m.get("min_train_periods", 12)),
                keep=("group", "fwd_return", "rt_cost"),
                half_life=c.half_life,
            )
            scored, w = wf.run(enc, cols, self.cal)
            self._cache[c.key()] = (scored, w, enc, cols)
        return self._cache[c.key()]

    def _split(self, scored: pd.DataFrame, accept: bool) -> pd.DataFrame:
        a = scored["entity_id"].map(lambda e: acceptance_entity(e, self.cfg.test_share))
        return scored[a == accept]

    def _net(self, scored: pd.DataFrame) -> pd.Series:
        s = scoring.fill_costs(scored)
        if self.cfg.metric == "rank_weighted":
            return scoring.rank_weighted(s)["net"]
        return scoring.quantile_spreads(s, self.cfg.spread_q, self.cfg.min_names)["net"]

    def paired(
        self, cand: ModelConfig, cur: ModelConfig, cutoff, accept: bool, since=None
    ) -> dict:
        """Per-period net spread, candidate minus current, closed periods, one entity split."""
        out = []
        for c in (cand, cur):
            sc = self.scores(c)[0]
            rows = self.rows_for(c.questions)[
                ["entity_id", "decision_time", "label_end"]
            ]
            sc = sc.merge(rows, on=["entity_id", "decision_time"], how="left")
            sc = sc[sc["label_end"] < cutoff]
            if since is not None:
                sc = sc[sc["decision_time"] >= since]
            out.append(self._net(self._split(sc, accept)))
        d = (out[0] - out[1]).dropna()
        return {
            "gain": float(d.mean()) if len(d) else np.nan,
            "t": scoring.per_period_t(d),
            "periods": len(d),
        }

    # ---------- the cycle
    def bar(self) -> float:
        from engine.registry import required_t

        n = (
            (self.registry.n_judged() if self.registry is not None else 0)
            + self.bar_offset
            + 1
        )
        return required_t(n)

    def _judge(
        self,
        cycle,
        cutoff,
        loop,
        rung,
        fld,
        cand: ModelConfig,
        trigger: str,
        change: str,
    ) -> bool:
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
                    "gain_tune": round(r["gain"], 6)
                    if r["gain"] == r["gain"]
                    else None,
                    "check_used": False,
                    "kept": ok,
                    "note": f"{loop} loop, cycle {cycle}, {r['periods']} periods; trigger: {trigger}",
                }
            )
        self.events.append(
            Event(
                cycle,
                cutoff,
                loop,
                rung,
                fld,
                change,
                trigger,
                ok,
                r["t"],
                bar,
                r["gain"],
            )
        )
        if ok:
            self.last_accept = (cycle, cutoff, self.cur, cand)
            self.cur = cand
            self.timeline.append((cutoff, cand))
        return ok

    def _pick(
        self, cands: list[ModelConfig], cutoff
    ) -> tuple[ModelConfig, dict] | None:
        """Choose among candidates on DIAGNOSIS entities (not a test, not logged)."""
        best = None
        k = len(self.reports) - 1
        cands = [
            c
            for c in cands
            if k - self.judged_at.get(c.key(), -(10**9)) >= self.cfg.rejudge_after
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

    def _oscillation(self, weights: pd.DataFrame, cycle: int, cutoff) -> None:
        if not len(weights):
            return
        w = weights.iloc[-1]
        for f, v in w.items():
            base = f  # per model column: an encoded field's level effects may differ in sign by design
            self.weight_signs.setdefault(base, []).append(float(np.sign(v)))
            s = [
                x
                for x in self.weight_signs[base][-self.cfg.oscillation_window :]
                if x != 0
            ]
            flips = sum(1 for a, b in itertools.pairwise(s) if a != b)
            if flips >= self.cfg.oscillation_flips and base not in self.frozen_fields:
                self.frozen_fields.add(base)
                self.events.append(
                    Event(
                        cycle,
                        cutoff,
                        "brake",
                        "oscillation",
                        base,
                        "",
                        f"weight sign flipped {flips} times in {self.cfg.oscillation_window} refits",
                        False,
                        note="field frozen for structural changes",
                    )
                )

    def question_changed(self, qid: str, new_text: str, cycle: int, cutoff) -> bool:
        """Record a question rewrite; alarm if it moves back toward an earlier version."""
        hist = self.question_texts.setdefault(qid, [])
        alarm = False
        if len(hist) >= 2:
            prev, older = hist[-1], hist[:-1]
            r_prev = difflib.SequenceMatcher(None, new_text, prev).ratio()
            if any(
                difflib.SequenceMatcher(None, new_text, o).ratio() > r_prev
                for o in older
            ):
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

    def _rollback(self, cycle: int, cutoff) -> bool:
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

    def cycle(self, k: int, cutoff: pd.Timestamp) -> dict:
        cutoff = pd.Timestamp(cutoff)
        if not self.timeline:
            self.timeline.append((pd.Timestamp.min.tz_localize("UTC"), self.cur))
        scored, w, enc, cols = self.scores(self.cur)
        w_now = w[[self._block_start(b) < cutoff for b in w.index]]
        self._oscillation(w_now, k, cutoff)
        diag = [
            e
            for e in enc["entity_id"].unique()
            if not acceptance_entity(e, self.cfg.test_share)
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
        for f in fl.index:
            s = self.streak.setdefault(f, {"misweight": 0, "shape": 0, "misread": 0})
            s["misweight"] = s["misweight"] + 1 if fl.loc[f, "status"] != "ok" else 0
            s["shape"] = s["shape"] + 1 if fl.loc[f, "non_linear"] else 0
            s["misread"] = s["misread"] + 1 if fl.loc[f, "misread"] else 0
        if self._rollback(k, cutoff):
            return rep
        if (
            self.blocked_until_refit
        ):  # rule 3: a full outer refit with the new version comes first
            self.blocked_until_refit = False
            return rep
        accepted, judged = 0, 0

        def room():
            return (
                accepted < self.cfg.max_changes_per_cycle
                and judged < self.cfg.max_judged_per_cycle
            )

        # rung 1: weights (outer), whenever anything is mis-weighted or decaying
        flagged = fl[(fl["status"] != "ok") | fl["decaying"]]
        if room() and len(flagged):
            cands = [
                replace(self.cur, half_life=h, alpha=a)
                for h in self.cfg.half_lives
                for a in self.cfg.alphas
                if (h, a) != (self.cur.half_life, self.cur.alpha)
            ]
            pick = self._pick(cands, cutoff)
            if pick:
                c, _r = pick
                judged += 1
                trig = "; ".join(
                    f"{f} {fl.loc[f, 'status']} (t {fl.loc[f, 't_joint']:.1f})"
                    for f in flagged.index[:3]
                )
                accepted += self._judge(
                    k,
                    cutoff,
                    "outer",
                    "weights",
                    "all",
                    c,
                    trig,
                    f"half_life={c.half_life}, alpha={c.alpha}",
                )

        # inner rungs: text fields only, persistent through >= persist_k refits, rested, not frozen
        def eligible(f, kind):
            base = f.split("__")[0]
            return (
                (self.meta.get(base) or {}).get("tag") is not None
                and self.streak.get(f, {}).get(kind, 0) >= self.cfg.persist_k
                and self.rest_until.get(base, -1) < k
                and base not in self.frozen_fields
                and (self.meta.get(base) or {}).get("question")
                not in self.frozen_fields
            )

        # rung 2: encoding (shape problems)
        for f in [
            f
            for f in fl.index
            if eligible(f, "shape") and f not in dict(self.cur.encodings)
        ]:
            if not room():
                break
            cands = [
                replace(
                    self.cur,
                    encodings=tuple(
                        sorted((dict(self.cur.encodings) | {f: e}).items())
                    ),
                )
                for e in self.cfg.encodings
            ]
            pick = self._pick(cands, cutoff)
            if pick:
                c, _r = pick
                judged += 1
                ok = self._judge(
                    k,
                    cutoff,
                    "inner",
                    "encoding",
                    f,
                    c,
                    f"non-linear by level (max shape t {fl.loc[f, 'max_shape_t']:.1f}) for {self.streak[f]['shape']} refits",
                    f"{f} -> {dict(c.encodings)[f]}",
                )
                accepted += ok
                if ok:
                    self.rest_until[f] = k + self.cfg.cooldown
        # rung 3/4: question rewrite / split / add (persistent mis-weight that weights didn't fix)
        if self.proposer is not None:
            for f in [f for f in fl.index if eligible(f, "misweight")]:
                if not room():
                    break
                prop = self.proposer.propose_field(f, self.meta.get(f), cutoff, self)
                if prop is None:
                    continue
                version, text, *more = prop  # optional third item: the columns it adds
                if self.question_changed(self.meta[f]["question"], text, k, cutoff):
                    continue
                added = tuple(
                    sorted(set(self.cur.added) | set(more[0] if more else ()))
                )
                c = replace(self.cur, questions=version, added=added)
                if (
                    self._pick([c], cutoff) is None
                ):  # screened on diagnosis entities first
                    continue
                judged += 1
                ok = self._judge(
                    k,
                    cutoff,
                    "inner",
                    "question",
                    f,
                    c,
                    f"{fl.loc[f, 'status']} (joint t {fl.loc[f, 't_joint']:.1f}) for {self.streak[f]['misweight']} refits",
                    text,
                )
                accepted += ok
                if ok:
                    self.blocked_until_refit = True
                    self.rest_until[f] = k + self.cfg.cooldown
        # rung 4: drop text fields with ~0 incremental value over a long window
        if room():
            for f in [
                c
                for c in cols
                if c in self.meta
                and self.rest_until.get(c, -1) < k
                and c not in self.frozen_fields
            ]:
                if not room():
                    break
                ic = rep["ic"].set_index("field")
                if f not in ic.index or rep["periods"] < self.cfg.drop_window:
                    continue
                inc = self._incremental(f, cutoff)
                st = self.streak.setdefault(
                    f, {"misweight": 0, "shape": 0, "misread": 0}
                )
                useless = (
                    inc["t"] < 1.0 and inc["periods"] >= self.cfg.drop_window
                )  # ~0 or harmful
                st["useless"] = st.get("useless", 0) + 1 if useless else 0
                if st["useless"] >= self.cfg.persist_k:  # rule 2 applies to drops too
                    c = replace(
                        self.cur, dropped=tuple(sorted(set(self.cur.dropped) | {f}))
                    )
                    pick = self._pick([c], cutoff)
                    if pick:
                        judged += 1
                        ok = self._judge(
                            k,
                            cutoff,
                            "inner",
                            "add_drop",
                            f,
                            c,
                            f"incremental IC {inc['gain']:+.4f} (t {inc['t']:.2f}) over {inc['periods']} periods",
                            f"drop {f}",
                        )
                        accepted += ok
                        if ok:
                            self.rest_until[f] = k + self.cfg.cooldown
        return rep

    def _incremental(self, f: str, cutoff) -> dict:
        sc_with = self.scores(self.cur)[0]
        sc_without = self.scores(
            replace(self.cur, dropped=tuple(sorted(set(self.cur.dropped) | {f})))
        )[0]
        rows = self.rows_for(self.cur.questions)[
            ["entity_id", "decision_time", "label_end"]
        ]
        a = sc_with.merge(rows, on=["entity_id", "decision_time"])
        b = sc_without.merge(rows, on=["entity_id", "decision_time"])
        a, b = a[a["label_end"] < cutoff], b[b["label_end"] < cutoff]
        diag = lambda s: s[
            ~s["entity_id"].map(lambda e: acceptance_entity(e, self.cfg.test_share))
        ]
        g = (scoring.rank_ic(diag(a)) - scoring.rank_ic(diag(b))).dropna()
        return {
            "gain": float(g.mean()) if len(g) else np.nan,
            "t": scoring.per_period_t(g),
            "periods": len(g),
        }

    def _block_start(self, b) -> pd.Timestamp:
        if isinstance(b, pd.Period):
            return pd.Timestamp(b.start_time, tz="UTC")
        return pd.Timestamp(f"{int(b)}-01-01", tz="UTC")

    def run(self, cutoffs) -> pd.DataFrame:
        for k, t in enumerate(cutoffs):
            self.cycle(k, t)
        return self.log()

    def log(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.events])

    def stitched(self) -> pd.DataFrame:
        """The ON system's scores: each decision uses the config in force at that time."""
        parts = []
        tl = self.timeline or [(pd.Timestamp.min.tz_localize("UTC"), self.cur)]
        for i, (t0, c) in enumerate(tl):
            t1 = (
                tl[i + 1][0] if i + 1 < len(tl) else pd.Timestamp.max.tz_localize("UTC")
            )
            sc = self.scores(c)[0]
            parts.append(sc[(sc["decision_time"] >= t0) & (sc["decision_time"] < t1)])
        return pd.concat(parts, ignore_index=True)


@dataclass
class ResidualPhraseProposer:
    """Free stub for rung 3: among documents behind the largest leave-text-out residuals, the word
    3-gram most specific to them becomes a new yes/no question (a split of what the field reads).
    `make_version(field, question)` registers the new version with the caller's rows_for."""

    docs_for: Callable  # (field, cutoff, coordinator) -> (top_docs_texts, other_texts)
    make_version: (
        Callable  # (field, question_dict) -> (version name, columns it adds) | None
    )
    min_lift: float = 0.3
    max_proposals: int | None = None  # each proposal may cost a paid re-read of history
    tried: set = field(default_factory=set)
    proposals: int = 0

    def propose_field(self, f, meta, cutoff, coord):
        from engine.text.textloop import _ngrams, _sentences

        if self.max_proposals is not None and self.proposals >= self.max_proposals:
            return None
        top, rest = self.docs_for(f, cutoff, coord)
        if not top:
            return None
        from collections import Counter

        a = Counter(g for t in top for s in _sentences(t) for g in set(_ngrams(s)))
        b = Counter(g for t in rest for s in _sentences(t) for g in set(_ngrams(s)))
        best = sorted(
            (
                (c / len(top) - b.get(g, 0) / max(1, len(rest)), g)
                for g, c in a.items()
                if "company_a" not in g
            ),
            reverse=True,
        )
        for lift, g in best:
            if lift < self.min_lift:
                return None
            if (f, g) in self.tried:
                continue
            self.tried.add((f, g))
            self.proposals += 1
            q = {
                "id": f"{meta['question']}_phrase{self.proposals}",
                "kind": "yes_no",
                "prompt": f"The text says: '{g}'.",
                "tag": "reading",
                "keywords": [r"\b" + r"\s+".join(g.split()) + r"\b"],
            }
            made = self.make_version(f, q)
            if made is None:  # e.g. the re-read would pass its spend cap
                return None
            version, added = made
            return version, q["prompt"], added
        return None
