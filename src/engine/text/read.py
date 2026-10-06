"""Reading: mask who a document is about, then answer its questions with a reader, through one
reading service (cache, cost estimate, spend cap).

Masking (version 2). An LLM trained after the period being tested may remember what happened to a
named company. Masking turns "Sucampo Pharmaceuticals reports Phase 3 data" into "COMPANY_A reports
Phase 3 data", in order: every full company name (current and former), longest first, with legal
suffixes; the ticker; distinctive first words ("Sucampo"; not dictionary words, >= 4 letters, any
case); dictionary-word first words ("Cypress") only when capitalized and not generic ("First",
"National" stay); optional extras: brands, drug-name stems, development codes, trial acronyms. The
dictionary is `wordlist` or /usr/share/dict/words; with neither, every capitalized non-generic first
word is masked (over-masks rather than guesses). It is a guard, not a proof: the leak gate and the
masked-vs-unmasked probe in engine.text.grade measure what still leaks.

An answer is {"value": ..., "evidence": str | None}: choice -> {option: probability}; scale ->
{0..4: probability}; yes_no / probability -> P(true). `evidence` is a verbatim span (checked by
code) or None for readers that can't quote.

Readers
  KeywordReader  FREE and deterministic: regexes from each question's `keywords`. Runs everything
                 (tests, the demo, keyless users) and is the baseline a paid reader must beat
  JevReader      TypeSafe's Jev (typed answers, input tokens only, cannot quote). Needs
                 TYPESAFE_API_KEY and typesafe-sdk (uv sync --extra llm)
  ClaudeReader   an Anthropic model with a JSON-schema answer and verified quotes. Needs
                 ANTHROPIC_API_KEY and anthropic (uv sync --extra llm)
  ReplayReader   answers stored by an earlier run; free

read_all() is the only way documents get read: a content-hash cache (the same text, question version
and reader are never paid for twice), a cost estimate printed before any paid call, and the spend
ledger's per-step cap checked before every chunk.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from engine.data import content_hash
from engine.spend import BudgetExceeded, Ledger
from engine.text.questions import LEVELS, Question, QuestionSet

# ---------------------------------------------------------------- masking
VERSION = 2
TOKEN = "COMPANY_A"
SUFFIXES = r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|HOLDINGS|GROUP|THE|N\.?V|S\.?A)\b\.?"
GENERIC_NAME_WORDS = {
    "AMERICA", "AMERICAN", "NATIONAL", "FIRST", "UNITED", "GENERAL", "INTERNATIONAL", "GLOBAL",
    "FEDERAL", "NATURAL", "PACIFIC", "ATLANTIC", "SOUTHERN", "NORTHERN", "WESTERN", "EASTERN",
    "CENTRAL", "CAPITAL", "STANDARD", "UNIVERSAL", "CONTINENTAL", "SECURITY", "COMMUNITY",
    "PEOPLES", "CITIZENS", "HOME", "HEALTH", "ENERGY", "FINANCIAL", "MEDICAL", "PRISON",
    "CORPORATE", "FOODS", "REHABILITATION", "SELECT", "MAIN",
}  # fmt: skip
# International nonproprietary name stems identify drugs, and so companies (-glutide, -mab ...)
DRUG_STEM = re.compile(
    r"\b[A-Za-z]{3,}(?:mab|nib|tinib|glutide|tide|cept|parin|vir|ciclib|lisib|stat|sartan|pril|olol|"
    r"azole|mycin|cillin|floxacin|tant|zumab|ximab|umab|kinra|ercept|plase|gene|tecan|platin|"
    r"rubicin|dronate|lukast|setron|prazole|vastatin|gliptin|gliflozin)\b",
    re.IGNORECASE,
)
CODE_NAME = re.compile(  # ABC-123, XYZ4567; not FY2016, Q3-2016, COVID-19
    r"\b(?!(?:FY|CY|Q|H|EX|COVID|ISO|SARS|NCT)-?\d)[A-Z]{1,5}-?\d{2,6}[A-Z]?\b"
)
BRAND = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,})\s*(?:®|™|\(R\)|\(TM\))")
TRIAL_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9\-]{2,11})\b(?=[^.]{0,60}\b(?:trial|study|studies)\b)")
KEEP_CAPS = {
    "FDA", "EMA", "NDA", "BLA", "SEC", "CEO", "CFO", "COO", "USA", "GAAP", "EPS", "EBITDA",
    TOKEN, "PDUFA", "IND", "CRL", "NASDAQ", "NYSE", "ATM", "PIPE", "LLC", "INC", "LP", "ET",
    "PST", "EST", "QOQ", "YOY",
}  # fmt: skip
EXTRAS = ("brands", "drugs", "codes", "trials")
_WORDS: dict[str, set[str]] = {}


def wordlist(path: str | Path | None = None) -> set[str]:
    """Lower-cased dictionary words (cached); empty if the file is missing."""
    p = Path(path) if path else Path("/usr/share/dict/words")
    key = str(p)
    if key not in _WORDS:
        _WORDS[key] = (
            {w.strip().lower() for w in p.read_text().splitlines()} if p.exists() else set()
        )
    return _WORDS[key]


def _core(name: str) -> list[str]:
    return re.sub(SUFFIXES, " ", name.upper().replace(",", " ")).split()


def mask_names(text: str, names: list[str], ticker: str | None) -> str:
    """Steps 1-2: full names (longest first, so "Arconic Inc" beats "Arconic") and the ticker.
    Letter-only boundaries, because HTML extraction glues words to digits ("Exhibit 99.1Apple")."""
    for name in sorted(names, key=len, reverse=True):
        core = " ".join(_core(name))
        if len(core) < 3:  # too short to mask safely
            continue
        pattern = r"\s+".join(re.escape(w) for w in core.split())
        text = re.sub(
            rf"(?<![A-Za-z]){pattern}(\s*{SUFFIXES})*",
            TOKEN,
            text,
            flags=re.IGNORECASE,
        )
    if ticker:
        text = re.sub(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", TOKEN, text)
    return text


def first_words(names: list[str], words: set[str]) -> tuple[set[str], set[str]]:
    """(distinctive: mask in any case, proper: mask only when capitalized)."""
    distinctive, proper = set(), set()
    for n in names:
        core = _core(n)
        if not core or not core[0].isalpha():
            continue
        w = core[0]
        if words and len(w) >= 4 and w.lower() not in words:
            distinctive.add(w)
        elif len(w) >= 5 and w not in GENERIC_NAME_WORDS and (not words or w.lower() in words):
            proper.add(w)
    return distinctive, proper


def mask(
    text: str,
    names: str | list[str],
    ticker: str | None = None,
    extras: tuple[str, ...] = EXTRAS,
    words: set[str] | None = None,
) -> str:
    """The text with every trace of `names` / `ticker` (and the chosen extras) replaced."""
    names = [names] if isinstance(names, str) else list(names)
    words = wordlist() if words is None else words
    text = mask_names(text, names, ticker)
    distinctive, proper = first_words(names, words)
    for w in distinctive:
        text = re.sub(
            rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])",
            TOKEN,
            text,
            flags=re.IGNORECASE,
        )
    for w in proper:
        text = re.sub(
            rf"(?<![A-Za-z])(?:{re.escape(w.title())}|{re.escape(w)})(?![A-Za-z])",
            TOKEN,
            text,
        )
    if "brands" in extras:
        text = BRAND.sub(lambda m: "PRODUCT_X" + m.group(0)[len(m.group(1)) :], text)
    if "drugs" in extras:
        text = DRUG_STEM.sub("DRUG_X", text)
    if "codes" in extras:
        text = CODE_NAME.sub(lambda m: m.group(0) if m.group(0) in KEEP_CAPS else "DRUG_X", text)
    if "trials" in extras:
        text = TRIAL_ACRONYM.sub(
            lambda m: m.group(1) if m.group(1) in KEEP_CAPS else "TRIAL_X", text
        )
    return text


def alternative_mask(masked: str, token: str = "the Company") -> str:
    """The alternative-mask probe: same text, a different placeholder. Answers must not move."""
    return masked.replace(TOKEN, token)


def residual_names(masked: str, names: list[str]) -> list[str]:
    """Name fragments that survived masking (a leak check you can run on any corpus)."""
    left = []
    for n in names:
        for w in _core(n):
            if (
                len(w) >= 4
                and w not in GENERIC_NAME_WORDS
                and re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", masked, re.IGNORECASE)
            ):
                left.append(w)
    return sorted(set(left))


# ---------------------------------------------------------------- readers
QUESTION_TOKENS = 60  # rough tokens per question in a paid reader's input


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_ok(evidence: str | None, text: str) -> bool:
    """A quote counts only if it is really in the text (whitespace- and case-insensitive)."""
    return bool(evidence) and _norm(evidence) in _norm(text)


def point_mass(q: Question, v) -> object:
    """A hard answer (Claude, a gold label) as a reader-style distribution."""
    if q.kind in ("yes_no", "probability"):
        return float(v)
    return {o: float(o == v) for o in q.outcomes()}


def expected_level(value: dict) -> float:
    """sum(k * P(k)) of a scale answer."""
    return sum(int(k) * p for k, p in value.items())


# ---------------------------------------------------------------- backends
class KeywordReader:
    """Free deterministic reader. Per question, the `keywords` regexes vote:

    yes_no / probability  any match -> `hit` (0.9), else `miss` (0.1)
    choice / scale        the option (level) with the most matches gets `hit`, the rest share the
                          remainder; no match -> the question's `default` option (or level 2,
                          the neutral anchor), else uniform
    Evidence is the matched span, so every answer it gives is quoted."""

    name, model, paid = "keyword", "keyword-v1", False

    def __init__(self, hit: float = 0.9, miss: float = 0.1):
        self.hit, self.miss = hit, miss

    def _matches(self, patterns, text: str) -> list[re.Match]:
        out = []
        for p in patterns or []:
            out += list(re.finditer(p, text, re.IGNORECASE))
        return out

    def answer(self, q: Question, text: str) -> dict:
        """One question's answer from its keyword patterns."""
        kw = q.keywords or ({} if q.kind in ("choice", "scale") else [])
        if q.kind in ("yes_no", "probability"):
            m = self._matches(kw, text)
            return {
                "value": self.hit if m else self.miss,
                "evidence": m[0].group(0) if m else None,
            }
        outcomes = q.outcomes()
        counts, first = {}, {}
        for o in outcomes:
            ms = self._matches(kw.get(o) or kw.get(str(o)), text)
            if ms:
                counts[o], first[o] = len(ms), ms[0]
        if counts:
            win = max(counts, key=lambda o: (counts[o], -first[o].start()))
            rest = [o for o in outcomes if o != win]
            value = {o: (1 - self.hit) / len(rest) for o in rest} | {win: self.hit}
            return {"value": value, "evidence": first[win].group(0)}
        default = (q.keywords or {}).get("default") if isinstance(q.keywords, dict) else None
        if default is None and q.kind == "scale":
            default = 2
        if default is not None:
            rest = [o for o in outcomes if o != default]
            value = {o: 0.4 / len(rest) for o in rest} | {default: 0.6}
        else:
            value = {o: 1 / len(outcomes) for o in outcomes}
        return {"value": value, "evidence": None}

    def read(self, doc: dict, qs: list[Question]) -> tuple[dict, dict]:
        """(answers, usage) for one document."""
        return {q.id: self.answer(q, doc["text"]) for q in qs}, {"input_tokens": 0}

    def estimate_usd(self, doc: dict, qs: list[Question], ledger: Ledger | None) -> float:
        """Free."""
        return 0.0


class JevReader:
    """TypeSafe's Jev: one call answers a document's whole battery. Literal and weak at numbers
    and dates (its documented limits), so questions state one condition each and code does math."""

    name, paid = "jev", True

    def __init__(self, model: str = "jev-1.13.0", doc_label: str | None = None):
        self.model, self.doc_label = model, doc_label
        self._local = threading.local()

    def _client(self):
        if not hasattr(self._local, "client"):
            from typesafe_sdk import TypeSafeClient  # optional dependency

            self._local.client = TypeSafeClient()
        return self._local.client

    @staticmethod
    def sdk(q: Question):
        """The typesafe-sdk question type for q."""
        from typesafe_sdk import Choice, Noul, Score

        if q.kind == "scale":
            return Score(instructions=q.prompt, criteria=list(q.levels))
        if q.kind == "choice":
            return Choice(instructions=q.prompt, criteria={o: None for o in q.options})
        return Noul(instructions=q.prompt)

    def read(self, doc: dict, qs: list[Question]) -> tuple[dict, dict]:
        """One call answers the document's whole battery."""
        r = self._client().system_one(
            state={
                "document_type": self.doc_label or doc["doc_type"],
                "company": "COMPANY_A",
                "text": doc["text"],
            },
            questions={q.id: self.sdk(q) for q in qs},
            model=self.model,
        )
        out = {}
        for q in qs:
            a = r.answers[q.id]
            if q.kind in ("yes_no", "probability"):
                v = float(a.noul)
            else:
                probs = {str(k): float(p) for k, p in dict(a.probabilities).items()}
                v = (
                    {lv: probs.get(str(lv), 0.0) for lv in LEVELS}
                    if q.kind == "scale"
                    else {o: probs.get(o, 0.0) for o in q.options}
                )
            out[q.id] = {"value": v, "evidence": None}
        return out, {"input_tokens": r.usage.input_tokens or 0}

    def estimate_usd(self, doc: dict, qs: list[Question], ledger: Ledger) -> float:
        """Worst-case input cost (characters over the floor, plus the questions)."""
        chars = len(doc["text"]) + sum(len(q.text()) for q in qs)
        return ledger.price_chars(self.model, chars) + ledger.price(
            self.model, QUESTION_TOKENS * len(qs)
        )


class ClaudeReader:
    """An Anthropic model answering every question with a value and a verbatim quote. A quote that
    isn't in the text is retried once, then the answer is dropped (the hallucination guard)."""

    name, paid = "claude", True

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 3000,
        instructions: str = "",
    ):
        self.model, self.max_tokens, self.instructions = model, max_tokens, instructions

    def _schema(self, qs: list[Question]) -> dict:
        props = {}
        for q in qs:
            value = (
                {"type": "boolean"}
                if q.kind == "yes_no"
                else {"type": "number", "minimum": 0, "maximum": 1}
                if q.kind == "probability"
                else {"type": "string", "enum": list(q.options)}
                if q.kind == "choice"
                else {"type": "integer", "enum": LEVELS}
            )
            props[q.id] = {
                "type": "object",
                "properties": {"value": value, "evidence": {"type": "string"}},
                "required": ["value", "evidence"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": props,
            "required": [q.id for q in qs],
            "additionalProperties": False,
        }

    def _system(self, qs: list[Question]) -> str:
        lines = [
            (
                "Answer questions about a document. Judge ONLY from the text; it is about COMPANY_A, "
                "whose name is hidden. Do not try to identify it or use what happened later. For every "
                "answer give an evidence quote copied EXACTLY from the text (shortest supporting span, "
                "<= 200 characters); an empty string only when the answer rests on the text NOT saying "
                "something."
            ),
            self.instructions,
            "Questions:",
        ]
        for q in qs:
            lines.append(f"- {q.id} ({q.kind}): {q.text()}")
        return "\n".join(x for x in lines if x)

    def _call(self, client, doc: dict, qs: list[Question], ledger, step) -> tuple[dict, int, int]:
        msgs = [
            {
                "role": "user",
                "content": f"Document type: {doc['doc_type']}\n\n{doc['text']}",
            }
        ]
        system = self._system(qs)
        n_in = client.messages.count_tokens(
            model=self.model, system=system, messages=msgs
        ).input_tokens
        if ledger is not None:
            ledger.guard(step, self.model, ledger.price(self.model, n_in, self.max_tokens))
        r = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=msgs,
            output_config={"format": {"type": "json_schema", "schema": self._schema(qs)}},
        )
        if ledger is not None:
            ledger.record(
                step,
                self.model,
                r.usage.input_tokens,
                r.usage.output_tokens,
                note=f"read {doc['doc_id']}",
            )
        body = json.loads(next(b.text for b in r.content if b.type == "text"))
        return body, r.usage.input_tokens, r.usage.output_tokens

    def read(
        self, doc: dict, qs: list[Question], ledger=None, step="text_read_claude"
    ) -> tuple[dict, dict]:
        """Every question with a quote; a quote not in the text is retried once, then dropped."""
        import anthropic  # optional dependency

        client = anthropic.Anthropic()
        out, todo, used = (
            {},
            list(qs),
            {"input_tokens": 0, "output_tokens": 0, "recorded": True},
        )
        for _ in range(2):  # one retry for answers whose quote isn't in the text
            if not todo:
                break
            body, i, o = self._call(client, doc, todo, ledger, step)
            used["input_tokens"] += i
            used["output_tokens"] += o
            retry = []
            for q in todo:
                a = body[q.id]
                ev = a["evidence"]
                if ev and not quote_ok(ev, doc["text"]):
                    retry.append(q)
                    continue
                out[q.id] = {"value": point_mass(q, a["value"]), "evidence": ev or None}
            todo = retry
        for q in todo:
            out[q.id] = {
                "value": None,
                "evidence": None,
                "dropped": "quote not in text",
            }
        return out, used

    def estimate_usd(self, doc: dict, qs: list[Question], ledger: Ledger) -> float:
        """Worst case: a retry, each with full max_tokens."""
        chars = len(doc["text"]) + len(self._system(qs))
        return 2 * ledger.price(self.model, chars / 3.0, self.max_tokens)  # worst case: a retry


class ReplayReader:
    """Answers stored by an earlier run, looked up by (doc_id, entity_id) or doc_id."""

    name, model, paid, cacheable = "replay", "replay", False, False

    def __init__(self, answers: dict, convert: Callable | None = None):
        self.answers, self.convert = answers, convert

    @staticmethod
    def content(doc: dict) -> str:
        """Stored answers belong to a document, not to its text: key the cache on the id."""
        return f"{doc['doc_id']}|{doc.get('entity_id')}"

    def read(self, doc: dict, qs: list[Question]) -> tuple[dict, dict]:
        """The stored answers for this document, if any."""
        raw = self.answers.get((doc["doc_id"], doc.get("entity_id"))) or self.answers.get(
            doc["doc_id"]
        )
        if raw is None:
            return {}, {"input_tokens": 0}
        raw = self.convert(doc, raw) if self.convert else raw
        return {q.id: raw[q.id] for q in qs if q.id in raw}, {"input_tokens": 0}

    def estimate_usd(self, doc, qs, ledger) -> float:
        """Free."""
        return 0.0


def make_reader(cfg: dict | str | None):
    """keyword (default) | jev | claude, from a YAML `reader:` value."""
    cfg = {"kind": cfg} if isinstance(cfg, str) else (cfg or {"kind": "keyword"})
    kind = cfg.get("kind", "keyword")
    opts = {k: v for k, v in cfg.items() if k not in ("kind", "step")}
    if kind == "keyword":
        return KeywordReader(**opts)
    if kind == "jev":
        return JevReader(**opts)
    if kind == "claude":
        return ClaudeReader(**opts)
    raise ValueError(f"unknown reader {kind!r} (keyword | jev | claude)")


# ---------------------------------------------------------------- the reading service
class AnswerCache:
    """sqlite: one row per (reader, model, question version + wording, text) content hash."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as con:
                con.execute(
                    "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, payload TEXT)"
                )

    @staticmethod
    def key(reader, q: Question, doc: dict) -> str:
        """Hash of reader, model, question version + wording, and the text (or the doc id)."""
        content = reader.content(doc) if hasattr(reader, "content") else doc["text"]
        return content_hash(reader.name, getattr(reader, "model", ""), q.key, q.text(), content)

    def get_many(self, keys: list[str]) -> dict:
        """The cached answers among `keys`."""
        if self.path is None or not keys:
            return {}
        out = {}
        with sqlite3.connect(self.path) as con:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                rows = con.execute(
                    f"SELECT key, payload FROM answers WHERE key IN ({','.join('?' * len(part))})",
                    part,
                ).fetchall()
                out |= {k: json.loads(p) for k, p in rows}
        return out

    def put_many(self, items: dict) -> None:
        """Store {key: answer}."""
        if self.path is None or not items:
            return
        with self._lock, sqlite3.connect(self.path) as con:
            con.executemany(
                "INSERT OR REPLACE INTO answers VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )


def _jsonable(a: dict) -> dict:
    v = a.get("value")
    if isinstance(v, dict):
        a = a | {"value": {str(k): p for k, p in v.items()}}
    return a


def _restore(q: Question, a: dict) -> dict:
    v = a.get("value")
    if isinstance(v, dict) and q.kind == "scale":
        a = a | {"value": {int(k): p for k, p in v.items()}}
    return a


def estimate(
    reader,
    docs: pd.DataFrame,
    qsets: dict[str, QuestionSet],
    ledger: Ledger | None,
    cache: AnswerCache | None = None,
) -> dict:
    """Projected cost of reading `docs` (only what the cache doesn't already hold)."""
    todo = _todo(reader, docs, qsets, cache or AnswerCache(None))
    usd = sum(reader.estimate_usd(d, qs, ledger) for d, qs in todo) if reader.paid else 0.0
    return {"documents": len(todo), "usd": usd}


def _todo(reader, docs, qsets, cache):
    jobs = []
    keys = []
    for d in docs.to_dict("records"):
        qs = qsets.get(d["doc_type"])
        if qs is None:
            continue
        ks = [AnswerCache.key(reader, q, d) for q in qs]
        jobs.append((d, list(qs), ks))
        keys += ks
    have = cache.get_many(keys)
    return [
        (d, [q for q, k in zip(qs, ks) if k not in have])
        for d, qs, ks in jobs
        if any(k not in have for k in ks)
    ]


def read_all(
    reader,
    docs: pd.DataFrame,
    qsets: dict[str, QuestionSet],
    ledger: Ledger | None = None,
    step: str | None = None,
    cache_path: Path | None = None,
    workers: int = 8,
    chunk: int = 300,
) -> dict:
    """(doc_id, entity_id) -> {question id: answer}. Paid readers: estimate, then guarded chunks."""
    if not getattr(reader, "cacheable", True):
        cache_path = None  # stored answers: caching them again buys nothing
    cache = AnswerCache(cache_path)
    todo = _todo(reader, docs, qsets, cache)
    if reader.paid:
        if ledger is None or step is None:
            raise BudgetExceeded("a paid reader needs a spend ledger and a step")
        usd = sum(reader.estimate_usd(d, qs, ledger) for d, qs in todo)
        print(
            f"{reader.name}: {len(todo):,} documents to read, projected ${usd:.2f} (worst case)",
            file=sys.stderr,
        )
        if usd > ledger.remaining(step):
            raise BudgetExceeded(
                f"{step}: projected ${usd:.2f} > ${ledger.remaining(step):.2f} left"
            )
    fresh: dict = {}  # this run's answers (the only store when there is no cache file)
    for start in range(0, len(todo), chunk):
        part = todo[start : start + chunk]
        if reader.paid:
            ledger.guard(
                step,
                reader.model,
                sum(reader.estimate_usd(d, qs, ledger) for d, qs in part),
            )
        new, tokens, errors = {}, 0, []

        def one(job):
            d, qs = job
            if isinstance(reader, ClaudeReader):
                return d, qs, reader.read(d, qs, ledger, step)
            return d, qs, reader.read(d, qs)

        results = []
        if reader.paid and workers > 1:
            with ThreadPoolExecutor(workers) as pool:
                for f in as_completed([pool.submit(one, j) for j in part]):
                    try:
                        results.append(f.result())
                    except Exception as e:  # noqa: BLE001 -- a failed doc is retried next run
                        errors.append(f"{type(e).__name__}: {str(e)[:120]}")
        else:
            results = [one(j) for j in part]
        for d, qs, (ans, used) in results:
            tokens += used.get("input_tokens", 0)
            for q in qs:
                if q.id in ans:
                    new[AnswerCache.key(reader, q, d)] = _jsonable(ans[q.id])
        if reader.paid and not isinstance(reader, ClaudeReader):
            ledger.record(
                step,
                reader.model,
                tokens,
                note=f"read {len(part)} docs ({len(errors)} errors)",
            )
        cache.put_many(new)
        fresh |= new
        for e in errors[:3]:
            print(f"  error: {e}", file=sys.stderr)
    keys: dict = {}
    for d in docs.to_dict("records"):
        qs = qsets.get(d["doc_type"])
        if qs is None:
            continue
        for q in qs:
            keys.setdefault(AnswerCache.key(reader, q, d), []).append((d, q))
    have = cache.get_many(list(keys)) | fresh
    out: dict = {}
    for k, pairs in keys.items():
        if k in have:
            for d, q in pairs:
                out.setdefault((d["doc_id"], d["entity_id"]), {})[q.id] = _restore(q, have[k])
    return out
