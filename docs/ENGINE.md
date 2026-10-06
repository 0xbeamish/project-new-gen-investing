# The signal-research engine

`src/engine/` turns any data source into one honest answer to one question: **does this signal,
known before the decision, rank next period's winners above its losers, after costs, more often
than luck allows?** It discovers candidate signals, judges them all the same way, logs every
attempt, and retires what fails.

It grew out of a year of one-off US small-cap experiments (docs/FINDINGS.md). The experiments
were honest; they were also hand-wired to SEC filings and EODHD prices. The engine keeps their
rules and drops the wiring: a stock universe is one plug-in, text of any kind is one more source
(docs/TEXT.md), and a crypto universe is meant to be the next market.

```
 Sources ──► observations ──► Panel builder ──► rows ──► Walk-forward model ──► Scorer ──► Registry
 (any data)  entity_id,       latest value per   label,   ranks in, one score   IC, spreads,  every test,
             available_at,    feature with        target,  per row, trained on   portfolios,   the bar,
             feature, value   available_at < T    costs    closed labels only    t per period  check gate,
                                    ▲                                                          holdout lock
 Market plug-in ────────────────────┘  universe(as_of), labels, cost_bps, calendar, groups
```

Run it with no data at all:

```bash
uv sync
uv run engine demo                                         # everything, on a synthetic market
uv run engine report --market synthetic --features all     # a toy market with a planted signal
uv run pytest -v
```

## 1. Contracts

Five small contracts. Each module's docstring is the authoritative version.

### Observations (`engine/observations.py`)

Every source emits long-format rows:

| column | meaning |
|---|---|
| `entity_id` | the market's id for the thing being ranked (a stock code, a token) |
| `available_at` | **UTC** time from which a decision may use the value: the publication time, or later if the source adds a conservative lag. Never earlier |
| `source` | the source's name |
| `feature` | the feature's name, unique across sources |
| `value` | a float. `NaN` is meaningful: "the latest report had no value", so it masks older values |

Ties (same entity, feature and `available_at`): the row emitted last wins, so emit in publication
order. Naive timestamps are rejected: guessing a time zone is how leaks start.

A **Source** is any object with `name`, `fetch(start, end)` (fill your own cache; the only step
that may touch the network) and `observations(start, end)` (read the cache; return everything
available before `end`). A source whose values are cheapest to compute only where they're needed
(momentum from bars) may implement `observations_at(rows)` instead; its rows must still carry the
true `available_at` of the data they used.

Window features ("insiders buying in the last 182 days") don't need a special path:
`observations.rolling_window()` emits one observation each time an event enters or leaves the
window, so "latest value before T" reproduces the window exactly.

### Market (`engine/market.py`)

| method | returns |
|---|---|
| `universe(as_of)` | who is tradable at that decision time: `entity_id`, `group`, attributes. Point-in-time, dead entities included |
| `labels(rows, horizon)` | forward return per `(entity_id, decision_time)`: enter at the close of the **first bar closing after** the decision, exit `horizon` bars later. A series that stops for good (delisting) returns to the last bar plus the market's delisting adjustment; an unfinished window is `NaN`. Also `entry_time` and `label_end` |
| `cost_bps(rows)` | round-trip cost in basis points per row (`NaN` = unknown; the scorer fills with the median and says so) |
| `calendar` | decision times and local dates (`engine/calendar.py`): trading days with a session close, or 24/7 |

`market.forward_returns()` implements the label convention once from a bar series; a market only
supplies bars and its delisting rule.

### Panel (`engine/panel.py`)

For each decision time `T` on the schedule and each entity in `universe(T)`:
- every feature's **latest observation with `available_at < T`**, blanked if older than its
  `max_age_days` (calendar days from the day it became usable to the decision date);
- the label from the market;
- the provenance: the `available_at` behind every filled cell.

`engine/pit.py` then re-checks the provenance (every used `available_at < T`) and the labels
(every `entry_time > T`). Every panel the pipeline builds goes through it; a failure stops the run.

### Models (`engine/models.py`)

Inputs are per-period percentile ranks centred at 0 (missing = 0). The target `fwd_rank` is the
forward return's percentile within (decision time, group) (`sample.add_rank_target`; set
`target.within: null` to rank across the whole cross-section), demeaned within the comparison
group. A model is anything with `fit(X, y)` / `predict(X)`; Ridge (alpha 10) is the default,
`trees` and `MeanEnsemble` are there, and anything sklearn-shaped plugs in.

`WalkForward` refits once per calendar year. Each year's model trains only on rows whose
**label had ended** (`label_end`, plus an optional embargo) before that year's first decision.

### Scorer (`engine/scoring.py`)

One scorer for everything. Per decision time, then a t-statistic across decision times
(`per_period_t`, deflated by √overlap when label windows overlap):

- **rank IC**: Spearman of score vs group-adjusted forward return;
- **quantile spreads**: top minus bottom decile within each group, gross and net of the round
  trips of names entering each leg;
- **portfolios**: long-only equal weight vs the equal-weight universe, with `Band(enter, exit)`
  (hysteresis), `TopK`, or `Periodic` rebalancing; half a round trip per buy and per sell,
  turnover, average holding period, share of gross lost to costs.

### Registry and periods (`engine/registry.py`)

Every judged test is a row in `data/engine/<market>/registry.csv`, kept or not. The tuning bar
is Bonferroni over **every judged test so far**: `required_t(n + 1)` = 1.96 at the first test,
2.81 at the 10th, 3.29 at the 50th. Earlier logs are inherited read-only: us_smallcap inherits
the pilot's logs (`data/experiments.csv` and `data/discover_log.csv` in the research repo; one
aggregate file, `data/registry_aggregate.csv`, in the shareable export), so its bar starts near
3.42, not 1.96.

Each market declares three periods:

| period | rule |
|---|---|
| tuning | everything is designed and judged here |
| check | opened only for a candidate that clears the tuning bar, through `registry.open_check()` (the pipeline refuses without its grant); each opening is counted and raises the check bar; hard limit (30); the loop sees pass/fail only |
| holdout | locked; `engine report --final` asks for a reason and appends it with the commit to the unlock log before reading a single row. Plan on one look |

### Discovery (`engine/discovery.py`)

A candidate is a feature plus a transform (`level`, `chg<k>`, `pct<k>`, `log`) plus a scope
(`universal` or one group; a group scope adds a group-only copy of the input so that group can
carry its own weight). The judge runs the walk-forward with and without the candidate; the
statistic is the per-period **difference in rank IC**, t over tuning periods. Kept only if it
clears the tuning bar, then the check bar with a positive check-period gain. Proposers are
plug-ins (`propose(study, tried, features)`); the default reads `discovery.candidates` from the
config, or crosses every panel feature with every transform. An LLM proposer (the pilot had one
that read winner / loser filings) can come back as one of these.

### Decider (`engine/decider.py`, `engine/feedback.py`, optional)

An LLM picks one entity per batch of model cards (score rank + the inputs pushing it, as
weight x input). `NoDecider` (the model's own pick) is the default, so the engine runs with no paid
API. `JevDecider` (TypeSafe) and `ClaudeDecider` (Anthropic) go through `engine.spend.Ledger`,
which refuses a call whose projected cost would pass its cap. Whenever a decider is configured,
the **feedback note is on by default**: rebuilt at every decision from closed periods only, it
shows the model's pick record, each input's rank IC, each text question's keep / drop status
(dropped questions vanish from the cards) and the decider's own override record split by the
text fields that drove it. The note reports; it never changes weights or questions.
`engine decide --estimate` prices a run before any call; `feedback.grade()` reports decider vs
the model's own top pick on the same batches, gross and net of measured costs, t from period means.

### Text sources (`engine/text/`) and the loops (`engine/loops.py`)

A text source is documents + a question file + a reader, emitting ordinary observations (the
level, the change vs the entity's previous document, the surprise vs history). The eval harness,
the question-improvement loop, tracking, the adjustment ladder and the coordinator that keeps the
outer (weights) and inner (text) loops from fighting are described in docs/TEXT.md.

### Spend (`engine/spend.py`)

Every paid call (readers, proposers, deciders) has a ledger step with a cap (`spend.caps` in the
market YAML; defaults in code), a cost estimate before the first call, a guard per chunk that also
checks the provider's loaded funds (`spend.funds`), and a ledger row per chunk. `engine spend
--market <m>` prints spent vs cap.

## 2. Add a source

1. Write a class with `name`, `fetch(start, end)` and `observations(start, end)` in your market's
   plug-in module, and add it to the module's source table (`SOURCES` / the dict `build()` returns).
2. Stamp `available_at` honestly. Questions to answer in the docstring:
   - When could a trader first have seen this value? Not "the period it describes": a quarter
     ending March 31 is known when it is filed in May.
   - Is the time zone explicit? Day-only dates are usable from the next local midnight
     (`calendar.date_available`).
   - Does the vendor restate history? Then store snapshots and emit the value as first published.
3. Emit `NaN` rows when a report exists but lacks the field, so stale values don't show through.
4. Add it under `sources:` in the market YAML with `features`, `max_age_days` and any `params`.
5. `uv run engine build --market <m>` prints coverage per feature and runs the point-in-time check.

Test it like `tests/engine/test_pit.py`: a value stamped exactly at the decision must not appear.

## 3. Add a market

1. A module under `engine/markets/` with a class implementing `universe`, `labels`, `cost_bps`
   and `calendar`, and `build(cfg) -> (market, {source name: source class})`.
   `engine/markets/synthetic.py` is the smallest complete example; `us_smallcap.py` is a real one.
   `us_largecap` shows the cheap case: the small-cap plug-in reused with another universe list,
   cost table and `data_end` (a date nothing after is read), in a 20-line module and one YAML.
2. A YAML in `markets/<name>.yaml`: calendar, universe rules, label horizon, sources, feature sets
   (`baseline` is what every candidate must beat), model, portfolio rules, periods, registry.
3. Decide the periods **before** looking at results, and write them in the YAML.
4. `engine build`, then `engine report` (descriptive, not logged), then `engine test` / `engine
   discover` (logged).

Notes for a crypto market (phase 2): `calendar.kind: continuous`, decisions at a fixed UTC time,
dead and delisted tokens in the universe, a minimum-liquidity rule, measured costs per venue,
and point-in-time snapshots for anything a provider restates (supply, TVL, holders).

## 4. Honesty rules

These are enforced by code where possible; the rest are conventions the code assumes.

1. **Point in time.** A panel cell may use only data with `available_at` strictly before the
   decision; labels start strictly after it. Checked on every panel.
2. **Survivors lie.** The universe must include entities that later died; a delisting is an
   outcome (with the market's delisting adjustment), not a missing row.
3. **Training labels never overlap the test period.** Purge on the real `label_end`; add an
   embargo if labels are serially correlated.
4. **Every test is logged, kept or not.** The bar rises with every judged test across the
   market's whole history. Parity and descriptive reports are not tests and don't touch it.
5. **Check years are rationed and the holdout is locked**, by the pipeline, not by memory.
6. **Costs are measured, not assumed**, and every portfolio number is reported net. Name the
   fill rule when a cost is missing.
7. **Green is not evidence.** Every component is tested on a planted signal it must find and on
   noise it must not (`tests/engine`).
8. **Every layer must beat the version without it**: text over numbers, a decider over the model.

## 5. Parity with the pilot (`engine parity --market us_smallcap`)

The port is checked by reproducing numbers the pilot recorded on tuning years (2013-2019), read
from `data/discover_log.csv` and `data/eval/lowturn_v1.json`. Parity runs are not tests: nothing
goes to the registry. Results: `data/engine/us_smallcap/parity.json`.

The panel itself matches the pilot's cell for cell (10-K ratios, insiders, 8-K counts, Jev text
answers, size, labels), except where noted below. The pilot had four quirks the engine doesn't
copy by default; `model.legacy` in the YAML switches them on for reproduction only:

| quirk | pilot | engine default |
|---|---|---|
| sample | random batches of 10 per sector-month; the leftovers that don't fill a batch are dropped (≈0.3% of rows) | every labelled row |
| target demeaned within | the random batch | the sector-month |
| purge | decision date + 31 calendar days | the real label end (exit bar). The 31-day rule let in labels that closed up to 2 days after the next year's first decision (Dec 2016, Dec 2017) and kept out Dec 2012, whose label had closed, so 2013 gets scored now |
| 10-Q filed the same day as another quarter's | whichever row an unstable sort left last (~5% of rows differ) | the latest quarter end |

Tuning years, `engine parity --market us_smallcap` (2026-10-05):

|  | recorded | legacy settings | engine defaults, same 72 months | engine defaults, all 84 months |
|---|---|---|---|---|
| numbers + text: rank IC (t) | 0.0340 (4.66) | 0.0340 (4.66) | 0.0367 (4.94) | 0.0287 (4.13) |
| numbers only: rank IC (t) | 0.0271 (3.50) | 0.0271 (3.50) | 0.0296 (3.73) | 0.0277 (3.87) |
| V1 gross / month (t) | +0.29% (2.57) | +0.29% (2.57) | +0.30% (2.52) | +0.26% (2.35) |
| V1 net / month (t) | +0.17% (1.47) | +0.17% (1.47) | +0.18% (1.51) | +0.14% (1.28) |

Legacy settings reproduce the recorded numbers to floating-point precision. The engine's honest
defaults move them a little in both directions; no conclusion changes (nothing nets a significant
return; the bar is 3.42).

## 6. Layout and commands

```
src/engine/            the engine
  observations.py      the observation contract, from_wide, rolling_window
  calendar.py          trading / continuous calendars
  market.py            the Market contract, forward_returns
  panel.py             the point-in-time panel builder
  pit.py               the point-in-time checker
  sample.py            winsorize, rank target, legacy batches
  models.py            Ridge / trees / ensembles, WalkForward (optional recency half-life, "auto")
  scoring.py           IC, spreads, rank-weighted book, portfolios, t (iid, Newey-West)
  cohorts.py           overlapping-cohort portfolios (buy and hold N formations); cohort_study.py runs one
  registry.py          registry, Bonferroni bar, periods, holdout lock
  pipeline.py          the standard run, with the period gates
  discovery.py         candidates, transforms, the judge, the loop
  decider.py           optional LLM decider (Jev, Claude) over model cards
  feedback.py          the decider's feedback note; decide.py runs and grades a decider
  loops.py             the adjustment ladder and the loop coordinator
  spend.py             spend ledger: caps, estimates, records
  text/                text scoring (docs/TEXT.md)
  parity.py            reproduction of the pilot's numbers (needs the pilot's data)
  config.py, cli.py    YAML -> Study; the `engine` command
  demo.py              `engine demo`
  markets/             synthetic.py, synthetic_text.py, us_smallcap/ (prices, SEC, insiders,
                       spreads, 8-K text, universe build)
markets/*.yaml         one config per market; markets/questions/*.yaml question sets
data/engine/<market>/  registry.csv (committed), parity outputs
.engine_cache/         observation and panel caches (git-ignored)
```

```bash
uv run engine demo                                    # no keys, no downloads
uv run engine fetch    --market us_smallcap           # sources fill their caches (network, your keys)
uv run engine build    --market us_smallcap           # tuning panel + point-in-time check + coverage
uv run engine report   --market us_smallcap [--features numbers] [--legacy]
uv run engine test     --market us_smallcap --feature q_rev_growth_yoy --transform chg3 [--scope tech]
uv run engine discover --market us_smallcap [--max-tests 3]
uv run engine decide   --market us_smallcap --decider jev --years 2013-2019 --estimate
uv run engine text-eval --market synthetic_text
uv run engine spend    --market us_smallcap
uv run engine parity   --market us_smallcap           # only with the pilot's recorded results
uv run engine cohort   --market us_largecap [--coverage | --log]   # long-horizon overlapping cohorts
```

The us_smallcap plug-in keeps its caches in `./.cache` (EODHD prices, SEC JSON, Form 4 data sets,
8-K texts) and must be run from the repo root. `engine fetch` needs `EODHD_API_KEY` (prices,
delisted names) and `SEC_USER_AGENT` ("Your Name you@example.com"); the universe list is built
with `uv run python -m engine.markets.us_smallcap.universe build`. The first panel build takes
~15 minutes (SEC parsing, insider windows); later runs read `.engine_cache/`.
