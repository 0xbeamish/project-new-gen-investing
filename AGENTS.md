# AGENTS.md

For an AI agent working in this repo. It is a research engine that asks whether a signal, known
before a decision, ranks the next period's winners above its losers after costs, more often than
luck allows. Read README.md first, docs/HOW_IT_WORKS.md for the mechanics, docs/FINDINGS.md for
what has already been tried (and failed).

## If the user says "try it out"

Run these in order (about 10 minutes, no keys, no downloads beyond Python packages), compare with
the expected results, then tell the user what the system does and what each step showed.

| step | command | expected on a fresh clone |
|---|---|---|
| 1. install | `uv sync` | finishes without errors |
| 2. demo | `uv run engine demo` (~2 min) | six numbered sections. In section 6A, `x_value`'s weight climbs from about 0 to about 29 (x100) against a true 30, and the inner loop re-encodes `txt_demand_level` (t 2.48). In 6B (the trap), `x_value` falls from about 50 to about 5 while "inner-loop changes to the demand question: none" |
| 3. a real signal | `uv run engine test --market csv_example --feature flow` | `t_tune` about 11.2 vs `bar_tune` 1.96, `check_used: true`, `kept: true` |
| 4. noise | `uv run engine test --market csv_example --feature social` | `t_tune` about 1.0 vs `bar_tune` about 2.24 (the bar rose because step 3 was logged), `kept: false` |
| 5. the feedback loop | `uv run engine loop --market csv_example` | a weights table where `flow` rises from about 6 to 16–20 and `social` stays near 0; "Changes judged: 0, accepted: 0"; ON vs FROZEN t near 0; three files in `results/csv_example/` |
| 6. tests | `uv run pytest -q` (~3 min) | all pass |

Steps 3–4 write to the toy registry in `.engine_cache/csv_example/`, so a second run shows a higher
bar; delete that folder to start fresh. Then offer the user the next step: their own data through
the CSV market ("Adding data" below).

## Setup

Needs [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or
`brew install uv`). uv installs Python 3.12 itself.

```bash
uv sync                    # Python 3.12, no keys needed
uv sync --extra llm        # only for paid readers / deciders (Jev, Claude)
uv run engine demo         # ~2 min; proves the install and shows the feedback loop
uv run pytest -q           # ~3 min; must stay green
uv run ruff check src tests examples && uv run ruff format src tests examples
```

Keys, only if the user gives them: copy `.env.example` to `.env` (git-ignored). `SEC_USER_AGENT` and
`EODHD_API_KEY` for the US stock plug-in, `TYPESAFE_API_KEY` / `ANTHROPIC_API_KEY` for paid readers.

## The commands (`uv run engine <cmd> --market <name>`)

| command | what it does | what it prints |
|---|---|---|
| `demo` | everything on the synthetic demo market, no `--market` | six sections: model IC, text scores, question loop, loop coordinator, decider note, the feedback loop's weights period by period |
| `loop` | **the feedback loop** over the tuning years, period by period, against a FROZEN twin; descriptive, `--log` = ONE test under the YAML's pre-registered `loop.name` | the top inputs' weights over time, the changes judged and accepted, loop ON vs FROZEN net of costs; writes `results/<market>/loop_weights.csv`, `loop_changes.csv`, `loop_returns.csv` |
| `build` | panel for tuning periods + point-in-time check (`--fetch` downloads first) | JSON: rows, decision times, entities, labelled rows, coverage per feature |
| `report` | a feature set's model on tuning periods (`--features`, default `baseline`); descriptive, never logged | JSON: rank IC and t, decile spread and portfolios gross / net, average weights, registry state, holdout looks |
| `test` | ONE judged test, logged whatever the result: `--feature` (+ `--transform`, `--scope`), or `--decider`, or the YAML's `cohort_test` | JSON: the registry row (t_tune vs bar_tune, check used, kept) |
| `discover` | candidates from the YAML (or every feature x transform), each a logged test, until one is kept | JSON list of registry rows |
| `grade-text` | the text eval harness on the market's eval sets (`--split dev`) | JSON summary (S, A-E, gates) and a keep / drop table per question |

**`engine loop` is the main way to evaluate a new data source over time**: `test` asks whether one
input helps on average; `loop` shows how every weight moves as returns close, whether a text field
stays mis-weighted, and whether re-weighting beats a frozen model. `--decider jev|claude` (loop,
report, test) is paid: `--estimate` prints the cost only.
`report --final` opens the holdout: never run it yourself (see the rules).

## Adding data

1. **CSV first, no code.** `prices.csv` (`entity, date, close`, optional `volume`, `group`),
   `signals.csv` (`entity, available_at, feature, value`), `documents.csv`
   (`entity, available_at, doc_type, text`) + a question YAML. Copy `markets/csv_example.yaml`,
   point `csv:` at the files, set calendar / schedule / horizon / periods, then `build`, `report`,
   `test`, and `loop` to watch the new inputs' weights over time. `examples/csv/` is a working
   24/7 example.
2. **A plug-in** when the data needs fetching or computing: a module under `src/engine/markets/`
   with `build(cfg) -> (market, {source: factory})`; sources implement `fetch` and `observations`
   (or `observations_at`). See docs/HOW_IT_WORKS.md, "Add your data"; `markets/demo.py` and
   `markets/csv.py` are the small examples, `markets/us_stocks/` the full one.
3. Every new source gets a test like `tests/test_panel.py`: a value stamped exactly at the decision
   must not appear.

## Rules you must never break

- **No future data.** Every input's `available_at` is strictly before the decision; when unsure
  when something became public, pick the later time. Never bypass `data.check_panel`.
- **Never open the check period or the holdout without the user.** Do not call
  `registry.open_check()` or `report --final` on your own; `test` opens the check period only
  through the registry when tuning clears the bar.
- **Log every judged test.** Use `engine test` / `discover` / `loop --log` (or `run.test_candidate`);
  never run a judged comparison off the books, and never delete or edit registry rows. A logged
  loop needs its `loop:` block (with `name`) committed before the run, and gets one look.
- **Never tune toward a result.** Decide the periods, the candidate and the bar before looking;
  don't re-run with tweaked settings until something passes. `report` is for description only.
- **Spend caps before paid calls.** Every paid call goes through `engine.spend.Ledger` with a cap;
  print the estimate (`--estimate`) and get the user's yes before spending.
- **Never commit keys or vendor data.** `.env`, `.cache/`, `.engine_cache/`, `data/costs.csv` and
  universe lists are git-ignored; keep it that way. Only aggregate statistics go in `docs/` and the
  registries.

## Where things live

| what | where |
|---|---|
| engine code | `src/engine/` (one module per job; the table in README.md); the feedback loop is `improve.py` |
| text scoring | `src/engine/text/` |
| market plug-ins | `src/engine/markets/` |
| market configs, question sets | `markets/*.yaml`, `markets/questions/` |
| example data | `examples/csv/` |
| registries (judged tests) | `data/registry_aggregate.csv`, `data/engine/<market>/registry.csv`; toy markets in `.engine_cache/<market>/` |
| caches (safe to delete) | `.engine_cache/` (panels, answers), `.cache/` (US stock downloads) |
| tests | `tests/` mirrors `src/engine/`; `tests/text/` mirrors `src/engine/text/` |
| feedback-loop outputs | `results/<market>/loop_*.csv` (git-ignored) |
| results so far | `docs/FINDINGS.md` |
