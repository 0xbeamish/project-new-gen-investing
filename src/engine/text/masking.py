"""Masking: hide who a document is about before any LLM reads it.

Why: an LLM trained after the period being tested may remember what happened to a named company
("the stock doubled after this trial readout"). A reading that uses that memory is look-ahead: the
backtest would credit the text with knowledge no reader had at the time. Masking turns "Sucampo
Pharmaceuticals reports positive Phase 3 data" into "COMPANY_A reports positive Phase 3 data", so
the reader can only use what the text says. It is a guard, not a proof: products, people and places
can still identify a company, which is what the leak test (engine.text.evaluate.leak_gate) and the
masked-vs-unmasked probe measure.

Version 2 (ported from the pilot's masking v2) masks, in order:
  1. every full company name (current and former), longest first, with legal suffixes
  2. the ticker
  3. distinctive first words of the names ("Sucampo"): not dictionary words, >= 4 letters, any case
  4. first words that ARE dictionary words ("Cypress"), but only when capitalized (used as a name)
     and not generic ("First", "National", "American" stay)
  5. optional domain extras (`extras`): brands (word + (R)/(TM)), drug-name stems, development
     codes (ABC-123) and trial acronyms. Defaults on: they matter most where hindsight is worst

The dictionary comes from `wordlist` or /usr/share/dict/words. With neither, step 3 is skipped and
step 4 masks every capitalized non-generic first word: safer (over-masks) than guessing.
"""

from __future__ import annotations

import re
from pathlib import Path

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
TRIAL_ACRONYM = re.compile(
    r"\b([A-Z][A-Z0-9\-]{2,11})\b(?=[^.]{0,60}\b(?:trial|study|studies)\b)"
)
KEEP_CAPS = {
    "FDA", "EMA", "NDA", "BLA", "SEC", "CEO", "CFO", "COO", "USA", "GAAP", "EPS", "EBITDA",
    TOKEN, "PDUFA", "IND", "CRL", "NASDAQ", "NYSE", "ATM", "PIPE", "LLC", "INC", "LP", "ET",
    "PST", "EST", "QOQ", "YOY",
}  # fmt: skip
EXTRAS = ("brands", "drugs", "codes", "trials")
_WORDS: dict[str, set[str]] = {}


def wordlist(path: str | Path | None = None) -> set[str]:
    p = Path(path) if path else Path("/usr/share/dict/words")
    key = str(p)
    if key not in _WORDS:
        _WORDS[key] = (
            {w.strip().lower() for w in p.read_text().splitlines()}
            if p.exists()
            else set()
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
        elif (
            len(w) >= 5
            and w not in GENERIC_NAME_WORDS
            and (not words or w.lower() in words)
        ):
            proper.add(w)
    return distinctive, proper


def mask(
    text: str,
    names: str | list[str],
    ticker: str | None = None,
    extras: tuple[str, ...] = EXTRAS,
    words: set[str] | None = None,
) -> str:
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
        text = CODE_NAME.sub(
            lambda m: m.group(0) if m.group(0) in KEEP_CAPS else "DRUG_X", text
        )
    if "trials" in extras:
        text = TRIAL_ACRONYM.sub(
            lambda m: m.group(1) if m.group(1) in KEEP_CAPS else "TRIAL_X", text
        )
    return text


def alternative(masked: str, token: str = "the Company") -> str:
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
                and re.search(
                    rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", masked, re.IGNORECASE
                )
            ):
                left.append(w)
    return sorted(set(left))
