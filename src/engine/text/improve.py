"""The question-improvement loop: one change at a time, judged on fixed eval sets by a rule written
before the first run.

  propose   a Proposer reads a briefing (dev metrics, per-field skill, up to 10 dev error examples,
            the log of tried variants) and returns ONE change: a reworded / re-keyworded question,
            a new question, a split, or a drop (at most 3 questions, so a gain can be attributed)
  validate  one condition per yes/no statement, complete options and anchors, no arithmetic or
            dates asked of the reader, not a duplicate of a logged variant
  run       re-read the DEV documents (only the changed questions miss the answer cache)
  judge     accept only if dS > 0 with the lower bound of a paired 95% bootstrap (resampling
            entities) above 0, every gate passes, and the set grows < 25% in length unless dS >=
            0.02. Ties go to the shorter, older set
  log       every attempt, accepted or not (loop_log.csv)
  stop      30 iterations; 8 rejects in a row; 5 accepted changes in a row each adding < 0.005; the
            spend cap; a leak-gate failure on the current best (fix the masker, not the questions)
  confirm   on the TEST split, at most 3 openings in total (each logged); a pass freezes the set
            (JSON + SHA-256), a fail keeps the last confirmed set

Proposers: KeywordProposer is free (it mines phrases from misread dev documents and adds them to a
question's keyword patterns, which only the keyword reader uses); ClaudeProposer rewrites wording
for the paid readers and is OFF unless configured, behind the spend ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine.registry import git_commit
from engine.text import grade
from engine.text import questions as tq

GATES = {
    "leak_excess": 0.05,
    "counterfactual_drop": 0.02,
    "brier_worse": 0.02,
    "cost_ratio": 1.5,
}


@dataclass
class Change:
    """One proposed change to a question set: rewrite | add | drop | split."""

    kind: str  # rewrite | add | drop | split
    doc_type: str
    questions: list[tq.Question] = field(default_factory=list)  # new / reworded questions
    drop: list[str] = field(default_factory=list)  # question ids removed (drop, split)
    hypothesis: str = ""

    def signature(self) -> str:
        """A content hash: the same change is never tried twice."""
        body = json.dumps(
            [
                self.kind,
                self.doc_type,
                [q.to_dict() | {"version": 0} for q in self.questions],
                self.drop,
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def apply(self, qsets: dict) -> dict:
        """The question sets with this change made (changed questions get a new version)."""
        qs = qsets[self.doc_type]
        for qid in self.drop:
            qs = qs.without(qid)
        for q in self.questions:
            qs = qs.with_question(q)
        return qsets | {self.doc_type: qs}


def validate(change: Change, tried: set[str]) -> str | None:
    """None if valid, else why not."""
    if change.signature() in tried:
        return "duplicate of a logged variant"
    if len(change.questions) > 3:
        return "more than 3 questions in one change"
    for q in change.questions:
        if q.kind == "yes_no" and re.search(r"\b(?:and|or)\b[^.]*\b(?:and|or)\b", q.prompt):
            return f"{q.id}: a yes/no statement must state one condition"
        if re.search(
            r"\b(?:calculate|compute|percentage of|how many days|what date)\b",
            q.prompt,
            re.IGNORECASE,
        ):
            return f"{q.id}: arithmetic and dates belong to code, not the reader"
    return None


# ---------------------------------------------------------------- proposers
def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _ngrams(s: str, n: int = 3) -> list[str]:
    w = re.findall(r"[a-z][a-z\-]+", s.lower())
    return [" ".join(w[i : i + n]) for i in range(len(w) - n + 1)]


class KeywordProposer:
    """Free. For the worst-read reading field: the phrase (word 3-gram) most specific to the dev
    documents it misread toward their true answer, added to that answer's keyword patterns."""

    name, paid = "keyword", False

    def propose(self, briefing: dict) -> Change | None:
        """One keyword change for the worst-read field, or None when nothing is left."""
        for fld in briefing["worst_fields"]:
            q = briefing["qsets"][fld["doc_type"]].get(fld["field"])
            errs = [e for e in briefing["errors"] if e["field"] == q.id]
            if not errs:
                continue
            target = Counter(e["label"] for e in errs).most_common(1)[0][0]
            mis = [e["text"] for e in errs if e["label"] == target]
            others = [t for t, lab in briefing["labelled_texts"].get(q.id, []) if lab != target]
            cand = Counter(g for t in mis for s in _sentences(t) for g in set(_ngrams(s)))
            bad = Counter(g for t in others for s in _sentences(t) for g in set(_ngrams(s)))
            scored = [
                (c / len(mis) - bad.get(g, 0) / max(1, len(others)), g)
                for g, c in cand.items()
                if "company_a" not in g
            ]
            scored = [x for x in scored if x[0] > 0.3]
            for _, gram in sorted(scored, reverse=True):
                pat = r"\b" + re.escape(gram).replace(r"\ ", r"\s+") + r"\b"
                kw = q.keywords
                if q.kind in ("yes_no", "probability"):
                    if not target:
                        continue
                    new_kw = list(kw or []) + [pat]
                else:
                    kw = dict(kw or {})
                    key = target if target in kw or q.kind == "choice" else int(target)
                    new_kw = kw | {key: list(kw.get(key, [])) + [pat]}
                newq = tq.bump(q, keywords=new_kw)
                ch = Change(
                    "rewrite",
                    fld["doc_type"],
                    [newq],
                    hypothesis=f"'{gram}' signals {q.id} = {target}",
                )
                if ch.signature() not in briefing["tried"]:
                    return ch
        return None


class ClaudeProposer:
    """Paid, off by default: Claude rewrites one question from the briefing (strict JSON)."""

    name, paid = "claude", True

    def __init__(
        self,
        ledger,
        model: str = "claude-opus-5-5",
        step: str = "text_loop_claude",
        max_tokens: int = 2000,
    ):
        self.ledger, self.model, self.step, self.max_tokens = (
            ledger,
            model,
            step,
            max_tokens,
        )

    def propose(self, briefing: dict) -> Change | None:
        """One rewritten or new question from Claude (guarded by the spend ledger)."""
        import anthropic  # optional dependency

        brief = {
            "scores": briefing["summary"],
            "worst_fields": briefing["worst_fields"],
            "errors": [
                {k: e[k] for k in ("field", "label", "pred")} | {"text": e["text"][:1500]}
                for e in briefing["errors"][:10]
            ],
            "tried": briefing["log_tail"],
            "questions": {dt: [q.to_dict() for q in qs] for dt, qs in briefing["qsets"].items()},
            "rules": "One condition per yes/no statement; no arithmetic or dates; complete options; never ask about returns or what happened later; no company names.",
        }
        schema = {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string"},
                "id": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(tq.KINDS)},
                "prompt": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "levels": {"type": "array", "items": {"type": "string"}},
                "hypothesis": {"type": "string"},
            },
            "required": [
                "doc_type",
                "id",
                "kind",
                "prompt",
                "options",
                "levels",
                "hypothesis",
            ],
            "additionalProperties": False,
        }
        msgs = [
            {
                "role": "user",
                "content": "Propose ONE question change.\n" + json.dumps(brief, default=str),
            }
        ]
        client = anthropic.Anthropic()
        n_in = client.messages.count_tokens(model=self.model, messages=msgs).input_tokens
        self.ledger.guard(
            self.step, self.model, self.ledger.price(self.model, n_in, self.max_tokens)
        )
        r = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=msgs,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        self.ledger.record(
            self.step,
            self.model,
            r.usage.input_tokens,
            r.usage.output_tokens,
            note="text loop proposal",
        )
        a = json.loads(next(b.text for b in r.content if b.type == "text"))
        qs = briefing["qsets"][a["doc_type"]]
        old = qs.get(a["id"]) if a["id"] in qs.ids() else None
        body = {
            "id": a["id"],
            "kind": a["kind"],
            "prompt": a["prompt"],
            "tag": old.tag if old else "reading",
        }
        if a["kind"] == "choice":
            body["options"] = tuple(a["options"])
        if a["kind"] == "scale":
            body["levels"] = tuple(a["levels"])
        q = tq.Question(**body, version=(old.version + 1) if old else 1)
        return Change("rewrite" if old else "add", a["doc_type"], [q], hypothesis=a["hypothesis"])


# ---------------------------------------------------------------- briefing, judge, loop
def briefing(res: dict, qsets: dict, sets: grade.EvalSets, tried: set, log: list) -> dict:
    """What a proposer sees: scores, the worst fields, dev error examples, what was tried."""
    k = res["_keep"]
    texts = dict(zip(sets.docs["doc_id"], sets.docs["text"]))
    worst = sorted(
        (
            {"field": f, "skill": v["skill"]}
            for f, v in res["per_field"].items()
            if v.get("covered")
        ),
        key=lambda x: x["skill"],
    )
    dt_of = {q.id: dt for dt, qs in qsets.items() for q in qs}
    for w in worst:
        w["doc_type"] = dt_of[w["field"]]
    errors, labelled = [], {}
    for g in k["gold"]:
        p = k["preds"].get(g["doc_id"], {})
        for f, y in g["labels"].items():
            if f not in p:
                continue
            v = p[f]
            pred = (v >= 0.5) if isinstance(v, float) else max(v, key=v.get)
            labelled.setdefault(f, []).append((texts.get(g["doc_id"], ""), y))
            if pred != y:
                errors.append(
                    {
                        "field": f,
                        "label": y,
                        "pred": pred,
                        "doc_id": g["doc_id"],
                        "text": texts.get(g["doc_id"], ""),
                    }
                )
    return {
        "summary": res["summary"],
        "worst_fields": worst,
        "errors": errors,
        "labelled_texts": labelled,
        "qsets": qsets,
        "tried": tried,
        "log_tail": log[-20:],
    }


def length(qsets: dict) -> int:
    """Total wording length of the question sets."""
    return sum(len(q.text()) for qs in qsets.values() for q in qs)


def judge(new: dict, cur: dict, new_q: dict, cur_q: dict, n_boot: int = 200, seed: int = 0) -> dict:
    """Accept only if the bootstrap low of dS > 0, every gate passes and the set barely grows."""
    b = grade.bootstrap_delta(new, cur, n_boot, seed)
    why = []
    lk = new.get("gates", {}).get("leak")
    if lk and lk["excess"] > GATES["leak_excess"]:
        why.append("leak gate")
    if (
        new["consistency"]["counterfactual_pass"]
        < cur["consistency"]["counterfactual_pass"] - GATES["counterfactual_drop"]
    ):
        why.append("counterfactual pass rate fell")
    for f, v in cur["per_field"].items():
        nb = new["per_field"].get(f, {}).get("brier")
        if v.get("brier") is not None and nb is not None and nb > v["brier"] + GATES["brier_worse"]:
            why.append(f"{f} Brier worse")
    r_new, r_cur = new["reaction"], cur["reaction"]
    lb = lambda r: r["card"]["IC_drift"] - 2 * r["IC_drift_se"]
    if lb(r_new) < lb(r_cur) - 1e-12:
        why.append("drift lower bound fell")
    if (
        cur["cost_usd_per_1k_docs"] > 0
        and new["cost_usd_per_1k_docs"] > GATES["cost_ratio"] * cur["cost_usd_per_1k_docs"]
    ):
        why.append("cost")
    if length(new_q) > 1.25 * length(cur_q) and b["delta"] < 0.02:
        why.append("25% longer for < 0.02")
    if not (b["delta"] > 0 and b["lo"] > 0):
        why.append(f"dS {b['delta']:+.4f}, bootstrap low {b['lo']:+.4f}")
    return {"accept": not why, "why": "; ".join(why) or "accepted", **b}


@dataclass
class LoopConfig:
    """Stop rules, bootstrap size, test-split openings, where the log and cache go."""

    max_iter: int = 30
    max_rejects: int = 8
    small_gain: float = 0.005
    small_streak: int = 5
    n_boot: int = 200
    test_opens: int = 3
    out_dir: Path = Path(".engine_cache/text_loop")
    step: str | None = None  # spend-ledger step for a paid reader / proposer


def run_loop(
    sets: grade.EvalSets, qsets: dict, reader, proposer, cfg: LoopConfig, ledger=None
) -> dict:
    """Propose, validate, re-read dev, judge, log; until a stop rule fires. Returns the best set."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache = cfg.out_dir / "answers.sqlite"
    log_path = cfg.out_dir / "loop_log.csv"
    best_q = qsets
    best = grade.score(sets, best_q, reader, "dev", ledger, cfg.step, cache)
    log, tried = [], set()
    rejects = small = 0
    stop = "max_iter"
    for it in range(1, cfg.max_iter + 1):
        if best.get("gates", {}).get("leak", {}).get("pass") is False:
            stop = "leak gate failed on the current best: fix the masker"
            break
        if (
            ledger is not None
            and cfg.step
            and getattr(proposer, "paid", False)
            and ledger.remaining(cfg.step) <= 0
        ):
            stop = "budget"
            break
        ch = proposer.propose(briefing(best, best_q, sets, tried, log))
        if ch is None:
            stop = "proposer has nothing left"
            break
        bad = validate(ch, tried)
        tried.add(ch.signature())
        row = {
            "iteration": it,
            "kind": ch.kind,
            "questions": ",".join(q.key for q in ch.questions),
            "drop": ",".join(ch.drop),
            "hypothesis": ch.hypothesis,
        }
        if bad:
            log.append(row | {"accepted": False, "why": f"invalid: {bad}"})
            rejects += 1
        else:
            new_q = ch.apply(best_q)
            new = grade.score(sets, new_q, reader, "dev", ledger, cfg.step, cache)
            j = judge(new, best, new_q, best_q, cfg.n_boot, seed=it)
            log.append(
                row
                | {
                    "S": new["summary"]["S"],
                    "dS": j["delta"],
                    "lo": j["lo"],
                    "accepted": j["accept"],
                    "why": j["why"],
                }
            )
            if j["accept"]:
                best, best_q = new, new_q
                rejects = 0
                small = small + 1 if j["delta"] < cfg.small_gain else 0
            else:
                rejects += 1
        pd.DataFrame(log).to_csv(log_path, index=False)
        if rejects >= cfg.max_rejects:
            stop = f"{cfg.max_rejects} rejects in a row"
            break
        if small >= cfg.small_streak:
            stop = f"{cfg.small_streak} small gains in a row"
            break
    return {
        "best_qsets": best_q,
        "best": best,
        "log": pd.DataFrame(log),
        "stop": stop,
        "accepted": int(sum(r["accepted"] for r in log)),
    }


def confirm(
    sets: grade.EvalSets,
    frozen_q: dict,
    best_q: dict,
    reader,
    cfg: LoopConfig,
    reason: str,
    ledger=None,
) -> dict:
    """Open the TEST split (counted, at most cfg.test_opens): best vs the frozen baseline."""
    unlocks = cfg.out_dir / "test_unlocks.csv"
    n = len(pd.read_csv(unlocks)) if unlocks.exists() else 0
    if n >= cfg.test_opens:
        raise PermissionError(
            f"the test split was opened {n} times already (limit {cfg.test_opens})"
        )
    row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
                "commit": git_commit(),
                "reason": reason,
            }
        ]
    )
    row.to_csv(unlocks, mode="a", header=not unlocks.exists(), index=False)
    cache = cfg.out_dir / "answers.sqlite"
    base = grade.score(sets, frozen_q, reader, "test", ledger, cfg.step, cache)
    best = grade.score(sets, best_q, reader, "test", ledger, cfg.step, cache)
    b = grade.bootstrap_delta(best, base, cfg.n_boot, seed=99)
    return {
        "pass": bool(b["delta"] > 0 and b["lo"] > 0),
        "S_frozen": base["summary"]["S"],
        "S_best": best["summary"]["S"],
        **b,
    }


def freeze(qsets: dict, path: Path, mask_version: int = 2) -> str:
    """Write the frozen question sets with the commit and return the file's SHA-256."""
    body = {
        "questions": {dt: [q.to_dict() for q in qs] for dt, qs in qsets.items()},
        "mask_version": mask_version,
        "commit": git_commit(),
        "frozen_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
    }
    text = json.dumps(body, indent=1, sort_keys=True, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
