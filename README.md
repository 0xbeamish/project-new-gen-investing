# signal-engine

A research engine for one question: **does this signal, known before the decision, rank the next
period's winners above its losers, after costs, more often than luck allows?** It takes any data
source, numbers or text, turns it into point-in-time inputs, judges every candidate the same way,
logs every attempt, and raises the bar with each one.

It was built on a year of US small-cap research (what was tried and what failed:
[docs/FINDINGS.md](docs/FINDINGS.md)), and is meant to be pointed at new data, especially
proprietary text, where an edge is more likely than in public filings everyone already reads.

```
 Sources ──► observations ──► Panel builder ──► rows ──► Walk-forward model ──► Scorer ──► Registry
 numbers,    entity_id,       latest value per   label,   ranks in, one score   IC, spreads,  every test,
 documents   available_at,    input with          target,  per row, trained on   portfolios,   the bar,
 (+ readers) feature, value   available_at < T    costs    closed labels only    net of costs  locked holdout
```

## Quickstart (no keys, no downloads)

```bash
uv sync
uv run engine demo          # a synthetic market whose returns depend on what its documents say
uv run pytest -v
```

`engine demo` (about 90 seconds) runs everything on generated data:

1. the walk-forward model on numbers vs numbers + text, scored on rank IC and net long-short;
2. text scoring with the free keyword reader: answer-key accuracy (reading fields only), probes,
   reaction and drift beyond a history prior, the leak gate, and a keep / drop per question;
3. the question-improvement loop, judged on dev and confirmed once on a held-out test split;
4. tracking and the loop coordinator: which fields are over-weighted or mis-shaped, and which
   loop fixed them;
5. a decider layer with its feedback note (the free "model's own pick" decider).

## What's inside

| path | what |
|---|---|
| `src/engine/` | the engine: observations, calendar, market, panel, point-in-time checker, models, scorer, registry, discovery, decider + feedback note, loops, spend ledger |
| `src/engine/text/` | text scoring for any document type ([docs/TEXT.md](docs/TEXT.md)) |
| `src/engine/markets/synthetic*.py` | toy markets with planted signals: the tests and the demo |
| `src/engine/markets/us_smallcap/` | US small/mid caps: universe (point-in-time, delisted included), prices, SEC fundamentals, Form 4 insiders, 8-K counts, measured trading costs, earnings-release text |
| `markets/*.yaml`, `markets/questions/*.yaml` | one config per market; question sets |
| `data/registry_aggregate.csv` | every judged test of the original research (aggregate statistics only) |
| `data/engine/us_smallcap/registry.csv` | tests judged through the engine |
| `docs/ENGINE.md` | the contracts, the honesty rules, how to add a source or a market |
| `docs/TEXT.md` | documents, questions, masking, readers, evaluation, the loops |
| `docs/FINDINGS.md` | every experiment and its result |

## Use your own data

**A numeric source**: a class with `name`, `fetch(start, end)` (fill your own cache) and
`observations(start, end)` returning `entity_id, available_at (UTC), source, feature, value`;
add it under `sources:` in a market YAML. `available_at` is the moment the value became public,
never the period it describes. Details: docs/ENGINE.md, "Add a source".

**A text source**: a `DocumentSource` returning `entity_id, available_at, doc_type, doc_id, text`
(masked), a question YAML (typed questions tagged reading or judgment), and a reader. Start with
`reader: {kind: keyword}` (free), measure with the eval harness, then switch to a paid reader:
every paid run prints its cost estimate first and stops at its cap. Details: docs/TEXT.md.

**A new market**: a module with `universe(as_of)`, `labels(rows, horizon)`, `cost_bps(rows)` and a
calendar; see `src/engine/markets/synthetic.py` for the smallest complete one.

## The US small-cap plug-in

It fetches its own data with your keys (copy `.env.example` to `.env`):

| key | used for |
|---|---|
| `SEC_USER_AGENT` | `"Your Name you@example.com"`: SEC filings, XBRL facts, Form 4 data sets, 8-K texts (free; the SEC rejects requests without a contact) |
| `EODHD_API_KEY` | daily prices and delisted listings (paid plan; one month is enough to fill the cache) |
| `TYPESAFE_API_KEY` | optional: Jev as a reader or decider (`uv sync --extra llm`) |
| `ANTHROPIC_API_KEY` | optional: Claude as a reader, labeler, proposer or decider (`uv sync --extra llm`) |

```bash
uv run python -m engine.markets.us_smallcap.universe build   # point-in-time universe list
uv run engine fetch  --market us_smallcap                    # prices, filings, insiders, 8-K texts
uv run engine build  --market us_smallcap                    # panel + point-in-time check
uv run engine report --market us_smallcap                    # descriptive, not logged
```

## Rules the code enforces

Point in time on every panel; dead companies stay in the universe; training labels never overlap
the test period; every judged test is logged and raises the bar (Bonferroni over all of them);
check years are rationed and the holdout is locked; costs are measured and every portfolio number
is net; every layer must beat the version without it. See docs/ENGINE.md, "Honesty rules".

No license file is included yet.
