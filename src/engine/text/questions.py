"""Question sets as config: one YAML per doc_type (or a bundle of doc_types sharing blocks),
typed, tagged reading / judgment, and versioned.

Every question is typed and tagged:

  kind      choice       one option from `options`               answer: {option: probability}
            scale        a 0-4 level with five `levels` anchors   answer: {0..4: probability}
            yes_no       a statement that is true or false        answer: P(true)
            probability  a likelihood the text states or implies  answer: a probability
  tag       reading      answerable from the text alone. Graded against the gold answer key
            judgment     needs outside context ("how good is this for shareholders?"). NEVER in the
                         answer key: graded only by markets (reaction, drift)
  evidence  required (default) | none. A reader that can quote must return a verbatim span that
            code checks; a reader that can't (Jev) returns none and its answers are marked unquoted
  tolerance scale only: +-levels accepted as correct in gold accuracy (0 = exact)
  keywords  for the free keyword reader: option -> [regex] (choice / scale level), or [regex] for a
            yes_no / probability statement
  version   bump on any wording change; answers stay attached to the version that produced them

Single doc_type file:
    doc_type: earnings_release
    version: 1
    questions: [{id: ..., kind: ..., prompt: ..., ...}, ...]

Bundle (several doc_types sharing blocks; `block.field` picks one question of a block):
    blocks: {core: [...], trial: [...]}
    doc_types: {earnings: [core, earnings], headline: [core.relation, core.is_subject, short]}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

KINDS = {"choice", "scale", "yes_no", "probability"}
TAGS = {"reading", "judgment"}
LEVELS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class Question:
    """One typed, tagged, versioned question."""

    id: str
    kind: str
    prompt: str
    tag: str = "reading"
    options: tuple = ()  # choice: option names
    levels: tuple = ()  # scale: five anchors, level 0..4
    evidence: str = "required"
    tolerance: int = 0
    keywords: dict | list | None = None
    version: int = 1
    notes: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind must be one of {sorted(KINDS)}")
        if self.tag not in TAGS:
            raise ValueError(f"{self.id}: tag must be reading or judgment")
        if self.kind == "choice" and len(self.options) < 2:
            raise ValueError(f"{self.id}: a choice needs >= 2 options")
        if self.kind == "scale" and len(self.levels) != 5:
            raise ValueError(f"{self.id}: a scale needs five level anchors (0-4)")
        if self.evidence not in ("required", "none"):
            raise ValueError(f"{self.id}: evidence is required or none")

    @property
    def key(self) -> str:
        """id@vN: answers stay attached to the version that produced them."""
        return f"{self.id}@v{self.version}"

    @property
    def gold_kind(self) -> str:
        """How the answer key stores it: bool | choice | level."""
        return {"yes_no": "bool", "probability": "bool", "choice": "choice"}.get(self.kind, "level")

    def outcomes(self) -> list:
        """The options of a choice, or the levels 0-4."""
        return list(self.options) if self.kind == "choice" else LEVELS

    def text(self) -> str:
        """The full wording a reader sees (prompt + anchors): what versioning and loops compare."""
        if self.kind == "scale":
            return self.prompt + " " + " · ".join(f"{i} {a}" for i, a in enumerate(self.levels))
        if self.kind == "choice":
            return self.prompt + " Options: " + ", ".join(self.options)
        return self.prompt

    def to_dict(self) -> dict:
        """The YAML form (empty fields left out)."""
        d = asdict(self)
        d["options"], d["levels"] = list(self.options), list(self.levels)
        return {k: v for k, v in d.items() if v not in ((), [], None, "")}


@dataclass
class QuestionSet:
    """The questions asked of one doc_type."""

    doc_type: str
    questions: list[Question]
    version: int = 1
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.doc_type}: duplicate question ids")

    def __iter__(self):
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def get(self, qid: str) -> Question:
        """The question with this id."""
        return next(q for q in self.questions if q.id == qid)

    def ids(self) -> list[str]:
        """Question ids in order."""
        return [q.id for q in self.questions]

    def reading(self) -> list[Question]:
        """The answer key's questions. Judgment questions are never graded against gold."""
        return [q for q in self.questions if q.tag == "reading"]

    def judgment(self) -> list[Question]:
        """Questions only markets can grade."""
        return [q for q in self.questions if q.tag == "judgment"]

    def fingerprint(self) -> str:
        """A hash of the wording (panel caches key on it)."""
        body = json.dumps([q.to_dict() for q in self.questions], sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def with_question(self, q: Question) -> QuestionSet:
        """A new set with q replacing the question of the same id (or appended)."""
        qs = [q if x.id == q.id else x for x in self.questions]
        if q.id not in self.ids():
            qs.append(q)
        return QuestionSet(self.doc_type, qs, self.version + 1, dict(self.meta))

    def without(self, qid: str) -> QuestionSet:
        """A new set without that question."""
        return QuestionSet(
            self.doc_type,
            [q for q in self.questions if q.id != qid],
            self.version + 1,
            dict(self.meta),
        )

    def to_yaml(self) -> str:
        """The set as YAML."""
        return yaml.safe_dump(
            {
                "doc_type": self.doc_type,
                "version": self.version,
                **({"meta": self.meta} if self.meta else {}),
                "questions": [q.to_dict() for q in self.questions],
            },
            sort_keys=False,
            allow_unicode=True,
        )


def question(d: dict) -> Question:
    """A Question from its YAML dict."""
    d = dict(d)
    for k in ("options", "levels"):
        if k in d:
            d[k] = tuple(d[k])
    return Question(**d)


def load(path: str | Path) -> dict[str, QuestionSet]:
    """doc_type -> QuestionSet, from a single-type file or a bundle."""
    cfg = yaml.safe_load(Path(path).read_text())
    if "questions" in cfg:
        qs = QuestionSet(
            cfg["doc_type"],
            [question(q) for q in cfg["questions"]],
            int(cfg.get("version", 1)),
            cfg.get("meta", {}),
        )
        return {qs.doc_type: qs}
    blocks = {name: {q["id"]: question(q) for q in qs} for name, qs in cfg["blocks"].items()}
    out = {}
    for dt, parts in cfg["doc_types"].items():
        picked: list[Question] = []
        for p in parts:
            if "." in p:
                b, qid = p.split(".", 1)
                picked.append(blocks[b][qid])
            else:
                picked += list(blocks[p].values())
        out[dt] = QuestionSet(dt, picked, int(cfg.get("version", 1)), cfg.get("meta", {}))
    return out


def bump(q: Question, **changes) -> Question:
    """A reworded question is a new version: old answers stay attached to the old one."""
    return replace(q, **changes, version=q.version + 1)
