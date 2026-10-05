"""The history replay on US small caps: the loops ON vs FROZEN, month by month (engine.replay).

    uv run engine replay --market us_smallcap --estimate     # what a question change would cost
    uv run engine replay --market us_smallcap --log          # the run; ONE registry test

ON: monthly outer refit with the recency half-life fixed in advance ("auto"); the inner loop under
the coordination rules (encodings, question splits from residual mining with the free phrase
proposer, drops). A question split re-reads every earnings release in the manifest with Jev for
the new question (step `jev_replay`); the number of splits the replay may propose is fixed up front
from the estimate so the whole run fits the cap. FROZEN: the v1 questions, yearly refit.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from engine import loops, panel, pipeline, replay
from engine.decide import text_meta
from engine.spend import Ledger
from engine.text import features as tf
from engine.text import questions as tq
from engine.text import readers, tracking


class ReplayText:
    def __init__(
        self,
        study,
        ledger: Ledger,
        step: str = "jev_replay",
        manifest: str = ".cache/earnings_releases.csv",
    ):
        self.study, self.ledger, self.step = study, ledger, step
        self.manifest = manifest
        self.rows_v1 = pipeline.rows(study, pipeline.build_panel(study, "tuning"))
        self.versions = {"v1": self.rows_v1}
        self.meta = {
            c: m
            for c, m in text_meta(study).items()
            if c in study.feature_set("baseline")
        }
        self._docs: pd.DataFrame | None = None
        self.reads: list[dict] = []
        self.cache = study.cache_dir / "replay_answers.sqlite"

    def rows_for(self, version: str) -> pd.DataFrame:
        return self.versions[version]

    def docs(self) -> pd.DataFrame:
        if self._docs is None:
            from engine.markets.us_smallcap.filings import SecEarningsReleases

            src = SecEarningsReleases(self.study.market, {"manifest": self.manifest})
            self._docs = src.documents(None, self.study.periods.tuning[1])
            print(
                f"replay: {len(self._docs):,} release rows with cached text",
                file=sys.stderr,
            )
        return self._docs

    def estimate(self, q: tq.Question) -> float:
        qs = {"earnings_release": tq.QuestionSet("earnings_release", [q])}
        cache = readers.AnswerCache(self.cache)
        return readers.estimate(
            readers.JevReader(), self.docs(), qs, self.ledger, cache
        )["usd"]

    def make_version(self, field: str, qd: dict):
        """Re-read every release for the new question (paid), attach it point in time, register."""
        q = tq.question(qd)
        qs = {"earnings_release": tq.QuestionSet("earnings_release", [q])}
        usd = self.estimate(q)
        if usd > self.ledger.remaining(self.step):
            print(
                f"replay: re-reading for {q.id} would cost ${usd:.2f} > ${self.ledger.remaining(self.step):.2f} left; skipped",
                file=sys.stderr,
            )
            return None
        ans = readers.read_all(
            readers.JevReader(), self.docs(), qs, self.ledger, self.step, self.cache
        )
        w = tf.answers_frame(self.docs(), ans, qs, "earn")
        col = f"earn_{q.id}_p"
        obs = tf.to_observations(w, "earnings_text_split", [col])
        base = self.rows_v1
        value, _ = panel.attach(base, obs, self.study.market.calendar, 400)
        version = f"v{len(self.versions) + 1}"
        self.versions[version] = base.assign(**{col: value.to_numpy()})
        self.meta[col] = {
            "doc_type": "earnings_release",
            "question": q.id,
            "kind": "yes_no",
            "tag": "reading",
            "part": "p",
            "encoding": "level",
            "version": q.version,
        }
        self.reads.append(
            {
                "field": field,
                "question": q.id,
                "prompt": q.prompt,
                "version": version,
                "estimate_usd": usd,
                "coverage": float(value.notna().mean()),
            }
        )
        return version, [col]

    def docs_for(self, field: str, cutoff, coord):
        """Release texts behind the largest LEAVE-TEXT-OUT residuals among rows where the field
        fires (top quintile), vs the other firing rows. Diagnosis entities, closed periods only."""
        text_cols = [c for c in coord.features if c in coord.meta]
        lto = replace(
            coord.cur, dropped=tuple(sorted(set(coord.cur.dropped) | set(text_cols)))
        )
        sc = coord.scores(lto)[0]
        diag = [
            e
            for e in self.rows_v1["entity_id"].unique()
            if not loops.acceptance_entity(e, coord.cfg.test_share)
        ]
        f = tracking.frame(self.rows_for(coord.cur.questions), sc, cutoff, diag)
        f = f[f[field].notna()]
        if f.empty:
            return [], []
        firing = f[f[field] >= f[field].quantile(0.8)]
        # the side the joint residual slope points to: where the model misses most on this field
        tj = (
            coord.reports[-1]["flags"].set_index("field").loc[field, "t_joint"]
            if coord.reports
            else 1.0
        )
        sign = 1 if tj >= 0 else -1
        firing = firing.assign(_r=sign * firing["resid"]).sort_values(
            "_r", ascending=False
        )
        top, rest = (
            firing.head(150),
            firing.iloc[150:].sample(
                min(600, max(0, len(firing) - 150)), random_state=0
            ),
        )
        d = self.docs().sort_values("available_at")
        texts = {}
        for name, part in (("top", top), ("rest", rest)):
            m = pd.merge_asof(
                part[["entity_id", "decision_time"]].sort_values("decision_time"),
                d[["entity_id", "available_at", "text"]]
                .rename(columns={"available_at": "decision_time_doc"})
                .sort_values("decision_time_doc"),
                left_on="decision_time",
                right_on="decision_time_doc",
                by="entity_id",
                direction="backward",
                allow_exact_matches=False,
            )
            texts[name] = m["text"].dropna().tolist()
        return texts["top"], texts["rest"]


def main(study, args) -> None:
    ledger = Ledger.from_config(study.cfg.get("spend"))
    rt = ReplayText(
        study,
        ledger,
        manifest=study.cfg.get("replay", {}).get(
            "manifest", ".cache/earnings_releases.csv"
        ),
    )
    probe = tq.Question(
        "probe_phrase", "yes_no", "The text says: 'raised its full year'."
    )
    per_split = rt.estimate(probe)
    cap_left = ledger.remaining("jev_replay")
    splits = math.floor(cap_left / per_split) if per_split > 0 else 0
    est = {
        "one_question_reread_usd": round(per_split, 2),
        "jev_replay_left": round(cap_left, 2),
        "question_splits_allowed": splits,
        "claude_replay": "not used (free phrase proposer)",
    }
    print(json.dumps(est, indent=1))
    if args.estimate:
        return
    proposer = loops.ResidualPhraseProposer(
        rt.docs_for, rt.make_version, max_proposals=splits
    )
    cfg = replay.ReplayConfig()
    bar = study.registry.next_bar()
    n0 = study.registry.n_judged()
    out_dir = Path(study.cache_dir) / "replay"

    def progress(k, t, co):
        acc = sum(1 for e in co.events if e.accepted and e.loop != "brake")
        print(
            f"{pd.Timestamp(t).date()} cycle {k}: judged {co.judged}, accepted {acc}, config {co.cur.key()}",
            file=sys.stderr,
            flush=True,
        )

    res = replay.run(
        study,
        study.feature_set("baseline"),
        rt.meta,
        rt.rows_for,
        cfg,
        proposer,
        bar_offset=n0,
        out_dir=out_dir,
        progress=progress,
    )
    res["reads"] = rt.reads
    res["estimate"] = est
    r = res["result"]
    if args.log:
        res["registry"] = study.registry.record(
            {
                "kind": "test",
                "name": "replay: loops ON vs FROZEN",
                "features": "baseline",
                "scope": "universal",
                "metric": "V1 low-turnover portfolio net of measured costs, monthly ON minus FROZEN",
                "t_tune": round(r["diff_net_t"], 3),
                "bar_tune": round(bar, 3),
                "gain_tune": round(r["diff_net"], 6),
                "check_used": False,
                "kept": bool(r["diff_net_t"] >= bar and r["diff_net"] > 0),
                "note": f"{r['months']} months 2013-2019; rank IC diff {r['diff_ic']:+.4f} (t {r['diff_ic_t']:.2f}); {res['inner_judged']} changes judged inside, {sum(1 for c in res['changes'] if c['accepted'] and c['loop'] != 'brake')} accepted; final {res['final_config']}; pre-registered (see commit)",
            }
        )
    (out_dir / "replay.json").write_text(json.dumps(res, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "config"}, indent=1, default=str
        )
    )
