# project-new-gen-fund

A research engine for one question: **does this signal, known before the decision, rank the next
period's winners above its losers, after costs, more often than luck allows?** It turns any data,
numbers or text, into point-in-time inputs, judges every candidate with the same walk-forward model
and scorer, and logs every attempt in a registry whose bar rises with each try. It was built on a
year of US stock research, where nothing cleared the bar ([docs/FINDINGS.md](docs/FINDINGS.md)), and
is meant to be pointed at new data.

## Quickstart (no keys, no downloads)

```bash
uv sync
uv run engine demo        # ~90 s: the whole system on a synthetic market with planted signals
uv run pytest -q
```

## Try your own data in 3 steps

1. Write `prices.csv` (`entity, date, close`, optional `volume`, `group`) and, for your signal,
   `signals.csv` (`entity, available_at, feature, value`). `available_at` is when the value became
   public, never the period it describes. Documents go in `documents.csv`
   (`entity, available_at, doc_type, text`) with a question set (see `examples/csv/`).
2. Copy `markets/csv_example.yaml` to `markets/mine.yaml`; point `csv:` at your files; set the
   calendar (`trading` or `continuous` for 24/7), schedule, horizon and periods. Pick the periods
   before you look at any result.
3. Run it:

```bash
uv run engine build  --market mine                      # panel, point-in-time check, coverage
uv run engine report --market mine --features all       # descriptive, never logged
uv run engine test   --market mine --feature my_signal  # ONE judged test, logged either way
```

The shipped example (30 tokens trading 24/7, planted signals) runs the same way:
`uv run engine test --market csv_example --feature flow` is kept; `--feature social` (noise) is not.

## What each file does

| file | what it does |
|---|---|
| `src/engine/data.py` | the observation and document contracts, calendars (trading, 24/7), the point-in-time checker |
| `src/engine/panel.py` | decision schedule x universe -> the latest value of every input before each decision, plus labels |
| `src/engine/model.py` | walk-forward models over per-period ranks; training labels always end before the test period |
| `src/engine/score.py` | the one scorer: rank IC, spreads, portfolios net of costs, t-statistics, overlapping cohorts |
| `src/engine/registry.py` | every judged test, the rising bar, the rationed check period, the locked holdout |
| `src/engine/run.py` | market YAML -> Study; build, report, test, discover, the long-horizon cohort test |
| `src/engine/spend.py` | caps and a ledger for every paid AI call |
| `src/engine/market.py` | what a market plug-in must provide |
| `src/engine/decide.py` | optional AI decider over the model's cards, with a feedback note from closed periods |
| `src/engine/improve.py` | the loops that adjust weights and text fields, the coordinator between them, the history replay |
| `src/engine/cli.py` | the `engine` command (demo, build, report, test, discover, grade-text) |
| `src/engine/text/questions.py` | question sets as YAML: typed, tagged reading / judgment, versioned |
| `src/engine/text/read.py` | masking names, readers (keyword free; Jev, Claude paid), the cached, capped reading service |
| `src/engine/text/source.py` | documents -> answers -> level / change / surprise inputs; history base rates |
| `src/engine/text/grade.py` | is each question worth asking: answer key, probes, reaction and drift over a prior, leak gate |
| `src/engine/text/tracking.py` | is a field under- or over-weighted, mis-shaped, decaying or misread |
| `src/engine/text/improve.py` | the question-improvement loop |
| `src/engine/markets/demo.py` | one synthetic market (numbers + documents with planted signals): the demo and the tests |
| `src/engine/markets/csv.py` | bring your own data from CSV files, no code |
| `src/engine/markets/us_stocks/` | US small and large caps: universe, EODHD prices and measured spreads, SEC filings, insiders, 8-K text |
| `markets/*.yaml`, `markets/questions/` | one config per market; question sets |
| `examples/csv/` | the CSV example's data, its question set and the script that generated it |
| `data/registry_aggregate.csv`, `data/engine/*/registry.csv` | every judged test so far (aggregate statistics only) |
| `docs/HOW_IT_WORKS.md` | the data flow, contracts, adding a source or market, text scoring, the loops, readers |
| `docs/FINDINGS.md` | every experiment and its result |
| `AGENTS.md` | the brief for an AI agent working in this repo |

## Where results go

- `engine report`, `test`, `discover` print JSON to stdout; progress goes to stderr.
- Judged tests: the market's registry CSV (`registry.file` in its YAML; `data/engine/<market>/` for
  research markets, `.engine_cache/<market>/` for the demo and CSV example).
- Holdout looks: the YAML's `holdout_unlock_log`. Paid calls: `data/costs.csv`.
- Panel and answer caches: `.engine_cache/` (git-ignored, safe to delete).

## Honesty rules

- **Point in time**: an input may be used only if `available_at` is strictly before the decision;
  labels start strictly after it. Every panel is checked.
- **Every judged test is logged**, kept or not, and raises the bar for the next (Bonferroni over all).
- **Check periods are rationed and the holdout is locked**, by the code, not by memory.
- **Costs are counted**: every portfolio number is net, and dead entities stay in the universe.
- **Every layer must beat the version without it**, and every component is tested on a planted
  signal it must find and on noise it must not.

Optional paid readers and deciders: `uv sync --extra llm` and keys in `.env` (copy `.env.example`).
The US stock plug-in fetches its own data with your keys (`engine build --fetch`). No license file
is included yet.
