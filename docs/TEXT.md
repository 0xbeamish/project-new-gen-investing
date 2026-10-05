# Text scoring

`src/engine/text/` turns documents of any type into point-in-time inputs for the engine's panel,
and measures whether each question is worth asking. The edge in text is rarely in "did X happen"
(markets price that within days); it is in reading the same facts more consistently than others
do, and in data others don't have. So the text side is built to be **pointed at proprietary
documents**: a new corpus needs a document source and a question file, nothing else.

```
 DocumentSource ──► masking ──► reader (keyword | Jev | Claude) ──► answers ──► level / change / surprise
 entity_id,          names,      cache + cost estimate + spend cap      │          observations ──► panel
 available_at,       tickers,                                           ▼
 doc_type, doc_id,   products    eval harness: answer key, probes, reaction and drift beyond a
 text                            history prior, leak gates, per-question efficacy ──► question loop
                                 tracking (closed periods) ──► adjustment ladder ──► coordinator
```

Run it with no keys and no downloads: `uv run engine demo`.

## 1. Documents

| column | meaning |
|---|---|
| `entity_id` | the market's id for the entity the document is about |
| `available_at` | UTC time from which a decision may use what it says: publication, or later. Never earlier |
| `doc_type` | earnings_release, 8k, news, forum_post ... Question sets are chosen by doc_type, and "the previous document" means the previous one of the same type |
| `doc_id` | unique per document (one document can concern several entities) |
| `text` | what the reader sees, already masked |
| `metadata` | optional dict; its keys can be history-prior keys (sector, size bucket) |

A `DocumentSource` has `name`, `fetch(start, end)` (fill your own cache; the only step that may use
the network) and `documents(start, end)` (read the cache). It plugs into a market YAML as a text
source:

```yaml
sources:
  earnings_text:
    type: text
    max_age_days: 400
    params:
      documents: sec_earnings_releases     # the plug-in's DOCUMENT_SOURCES name
      questions: markets/questions/earnings_release.yaml
      reader: {kind: keyword}              # keyword (free) | jev | claude
      step: text_read_jev                  # spend-ledger step for a paid reader
      prefix: earn
      change: {k: 4, min_prior: 2}         # vs the mean of the previous 4 documents
      surprise: {keys: [group]}            # vs the history base rate (or false)
```

## 2. Questions

One YAML per doc_type (or a bundle of doc_types sharing blocks: `markets/questions/sec_8k_card.yaml`).

```yaml
doc_type: release
version: 1
questions:
  - id: guidance_action
    kind: choice                      # choice | scale (0-4) | yes_no | probability
    prompt: What does COMPANY_A do with its guidance or outlook for the year?
    options: [raised, maintained, lowered, none]
    tag: reading                      # reading: answerable from the text | judgment: needs context
    evidence: required                # readers that can quote must return a verified span
    keywords: {raised: ['\braised its full-year guidance'], default: none}   # the free reader
```

- **Reading vs judgment.** "Did the company raise guidance?" is answered by the text. "How good is
  this for shareholders?" needs context the excerpt doesn't have, and it is where two careful
  labelers disagreed most in the pilot. Judgment fields are never in the answer key; only markets
  grade them (reaction, drift).
- **Versions.** Any wording change bumps the version; answers stay attached to the version that
  produced them, so old and new answers are never mixed.
- Wording rules for literal readers like Jev: one condition per yes/no statement, boundary cases in
  the options, no arithmetic or dates (code does those).

## 3. Masking, and why

An LLM trained after the period being tested may remember what happened to a named company. If it
reads "Sucampo reports Phase 3 results" it may recall the stock doubled. A reading that uses that
memory is **look-ahead**: the backtest credits the text with knowledge no reader had at the time,
and the edge vanishes live. Masking (`engine.text.masking`, version 2 of the pilot's masker) turns
the text into "COMPANY_A reports Phase 3 results":

1. every full company name (current and former), longest first, with legal suffixes;
2. the ticker;
3. distinctive first words ("Sucampo"), any case;
4. dictionary-word names ("Cypress") only when capitalized; generic words ("First", "National") stay;
5. domain extras: brands, drug-name stems, development codes, trial acronyms.

Masking is a guard, not a proof: products, people and places can still identify a company. Two
gates measure what leaks: the outcome probe (section 6) and masked-name identification (the reader
picks the masked company from five candidates; above chance = a leak). The port reproduces the
pilot's masker exactly (278 of 278 cached 8-Ks identical).

## 4. Readers

| backend | cost | quotes | use |
|---|---|---|---|
| `KeywordReader` | free, deterministic | the matched span | tests, the demo, keyless users, and the baseline a paid reader must beat |
| `JevReader` | input tokens only (jev-1.13.0: $0.042 / 1M) | no | typed answers at scale |
| `ClaudeReader` | input + output | verified, retried once, else dropped | gold labels, second labeler |
| `ReplayReader` / plug-in readers | free | as stored | re-scoring answers already paid for |

Every reading goes through `readers.read_all`: a content-hash cache (same text + question version +
reader = never paid twice), a printed cost estimate before the first paid call, a refusal if the
estimate exceeds what is left under the step's cap, and per-chunk guards that also check the
provider's loaded funds (`engine.spend.Ledger`, caps in the market YAML under `spend:`).

## 5. Answers to observations

Per document and question: the level (`_level` for scales, `_p` for yes/no, one column per option
for choices); per entity and doc_type in publication order, the **change** vs its previous
documents (`_chg`), and the **surprise** vs the history base rate of that answer among strictly
earlier documents of the same type (and finer keys such as sector), shrunk by backoff
(`_surp`; `engine.text.history`). Each value is stamped with its document's `available_at`, so the
panel's "latest value strictly before the decision" rule and the point-in-time checker apply
unchanged. A document whose reading failed emits nothing, so it can't mask the previous answer.

## 6. Evaluation harness

Three sets, all tuning-period, split dev / test **by entity**:

| set | what it answers |
|---|---|
| answer key (gold) | does each reading field mean what its name says? Two labelers (agreement reported); a JSON export for a person's yes/no spot-check, whose corrections override the labels and later become reader few-shot examples |
| probes | paraphrase, alternative mask, unmasked, sentence-order and question-order shuffles (answers must hold still); one-change counterfactual edits (the edited field must move the right way, the rest must not) |
| market set | the reaction (3-day abnormal return in sigma units) and later drift, per event |

Reaction and drift are scored as the card's **gain over a prior** (event type + size + history base
rates of similar earlier events + looked-up context), out-of-fold by year: a card that only knows
"earnings releases move stocks" earns nothing.

`S = 0.30 A + 0.25 B + 0.30 C + 0.10 D + 0.05 E`

- A: answer-key accuracy on **reading fields only** (Brier skill; tolerance skill for scales);
- B: mean of paraphrase, mask and order invariance and the counterfactual pass rate;
- C: mean(min(1, dR2_size / 0.05), min(1, dIC_reaction / 0.10)), gains over the prior;
- D: min(1, dIC_drift / 0.05), gain over the prior;
- E: coverage. Gates: outcome-probe leak (> 0.05 AUC over the reader's own good/bad read),
  name identification above chance, cost.

`evaluate.efficacy` adds a per-question table: gold skill (reading questions), probe stability, and
the reaction / drift gain of the card with vs without that question, with a keep / drop
recommendation from thresholds fixed in advance.

## 7. The question-improvement loop

`engine.text.textloop`: one change at a time (a reworded or new question, a split, a drop; at most 3
questions), validated (one condition per statement, no arithmetic, not a repeat), re-read on dev
(the cache means only changed questions cost anything), and **accepted only if the lower bound of a
95% paired bootstrap of dS (resampling entities) is above 0**, every gate passes, and the set grows
less than 25% unless dS >= 0.02. Every attempt is logged. Stops after 30 iterations, 8 rejects in a
row, 5 tiny gains in a row, the spend cap, or a leak-gate failure. The test split is opened at most
3 times (each logged); a pass freezes the set (JSON + SHA-256). The free `KeywordProposer` mines
phrases from misread dev documents; `ClaudeProposer` rewrites wording for paid readers and is off
unless configured. On the synthetic market the free loop finds the three phrasings the keyword
reader misses (S 0.80 -> 0.98 on dev; 0.81 -> 0.98 on test).

## 8. Is a field under- or over-rated? Tracking

`engine.text.tracking.report` runs after every closed period, costs nothing, and calls no LLM.
Diagnosis entities only; the frozen test split is kept for accepting fixes.

1. **Residual attribution.** Each period, the model's residual (realized minus predicted target) is
   ranked and regressed on the ranked inputs, one at a time and **jointly**. A slope with the same
   sign as the field's weight means under-weighted, the opposite sign over-weighted. Mean, t, and
   rolling 12- and 24-period windows. The joint slope decides (see section 10).
2. **Calibration by level.** Per answer level, the realized partial residual vs the model's implied
   effect, with the linear part removed: what remains is shape. A 0-4 scale can carry the right
   average weight and be wrong at every level.
3. **Rolling IC and decay**: per field, and the last 24 periods vs the earlier ones.
4. **Reading quality** per question: gold skill, spot-check corrections, probe stability. This is
   what separates "misread" (fix the reader or the question) from "mis-weighted" (fix the weights).
5. **Decider overrides**, split by which text fields led the card the decider picked.

## 9. The adjustment ladder

Cheapest first. A rung's candidate is chosen on diagnosis entities, then judged once, end to end,
on the acceptance entities; every judged change is a registry test, accepted or not.

| rung | loop | change | when |
|---|---|---|---|
| 1 weights | outer | recency half-life (`auto` = chosen inside every refit from {none, 36, 24, 12, 6} on the last 12 closed periods) and shrinkage | anything mis-weighted or decaying |
| 2 encoding | inner | per-level effects for a scale (one-hot or monotone steps), or surprise vs its base rate | a shape flag that persisted |
| 3 question | inner | rewrite or split, from the documents behind the largest leave-text-out residuals (free stub proposer; paid off by default) | a mis-weight that persisted after the weights moved |
| 4 add / drop | inner | drop a field whose incremental value is ~0 or harmful over 24+ periods; add a question for an uncovered phrase | the same persistence rule |
| 5 reader | inner | instructions, few-shot from spot-check corrections, model choice; the question loop above | reading quality is the problem |

## 10. How the loops interact

The outer loop (weights over all inputs) and the inner loop (the text side) can both "fix" the same
symptom. Without rules, the inner loop rewrites a good question to compensate for a numeric input
that is mis-weighted. The coordinator (`engine.loops.Coordinator`) enforces:

1. **Inner objective = its own job**: reading accuracy and stability, and incremental information
   measured on the full outer model (joint attribution, with-vs-without gains), never a field's raw
   correlation with returns. Residual mining uses the leave-text-out residual.
2. **Weights first.** The outer refit runs every period. An inner structural change is eligible
   only if the field's signal persisted through >= `persist_k` (3) consecutive outer refits.
3. **Never in the same cycle.** At most one structural change per cycle, in one loop. After a
   question changes, history is re-read with the new version and a full outer refit runs before
   anything else may change; the field then rests for `cooldown` cycles.
4. **One judge.** Every change, inner or outer, must improve the end-to-end system (the full model,
   a rank-weighted long-short book net of costs) on the acceptance entities, at the registry's bar.
   Inner metrics (S, bootstrap) are necessary, not sufficient.
5. **Diagnosis and acceptance use different entities** (a fixed hash split).
6. **Brakes.** A max of one judged change per cycle; a candidate judged and rejected waits 6 cycles;
   automatic rollback when the end-to-end score falls after an acceptance (t <= -1 over >= 3
   periods); an oscillation alarm freezes a field whose weight flips sign twice in 12 refits or whose
   question is rewritten back toward an earlier version.
7. **The decider's note reports only.** Nothing in the loops reads it.

### Worked example (tests/engine/text/test_loops.py; 1,000 synthetic entities, quarterly cycles 2012-2014)

**Planted market**: a one-time-charge field mattered (-6%/month) until 2012, then not at all; demand
pays at both extremes (a U).

| field (frozen model, cutoff 2015-01) | status | joint t | one-at-a-time t | shape t |
|---|---|---|---|---|
| one-time charge | over-weighted | +8.1 | +8.1 | - |
| demand (0-4) | ok (no linear error) | -1.1 | -0.7 | 32.2 (non-linear) |
| buyback (noise) | ok | -0.0 | -0.3 | - |

Calibration for demand, frozen model: the realized partial residual by level is +0.066, +0.019,
-0.048, +0.008, +0.068 (levels 0-4) against an implied effect of about 0 at every level: a U the
linear weight cannot see.

| cycle | loop | change | trigger | t vs bar | result |
|---|---|---|---|---|---|
| 2 | outer | half-life auto | one-time charge over-weighted (t 3.3) | -1.00 vs 1.96 | rejected |
| 3 | inner | demand -> monotone steps | non-linear for 4 refits (shape t 15.8) | 5.79 vs 2.24 | accepted |
| 4 | outer | half-life auto | one-time charge over-weighted (t 4.7) | 2.91 vs 2.39 | accepted |

After both: the shape flag is gone, the one-time charge's joint t falls from 8.1 to 3.4 (half-lives
shrink the stale weight but don't erase it within three years), and the end-to-end book beats the
frozen one by 1.5%/month on held-out entities (t 10.7). Three noise fields tripped the oscillation
alarm and were frozen.

**Correlated market** (the trap): numeric `x` mattered (+6%/month) until 2012; text `demand` is 0.8
correlated with `x` and has a small, constant effect of its own.

| field (frozen model) | one-at-a-time t | joint t | status |
|---|---|---|---|
| x (numeric) | -8.8 | -8.8 | over-weighted |
| demand (text) | **-8.7** | -0.5 | ok |

One at a time, demand looks badly over-weighted, and a naive inner loop would rewrite or drop the
question. Jointly the error sits on `x`. The coordinator made no inner change to demand; the outer
rung (half-life auto) was accepted at cycle 6 (t 2.37 vs bar 2.24) and the end-to-end book improved
(t 4.3).

## 11. The decider's feedback

With a decider configured, the feedback note is on by default (`engine.feedback`). Rebuilt at every
decision from closed data only: the model's pick record, each input's rank IC (strongest shown,
weakest named; a scale question is also scored level by level, so a U-shaped question is not
dropped for a flat linear record), each text question's keep / drop status (dropped questions
vanish from the cards), and the decider's own override record split by the text fields that led the card it picked.

## 12. Worked example: SEC filings (`engine/markets/us_smallcap/`)

- `sec_earnings_releases`: 8-K item 2.02 press-release exhibits, masked; fetched with your own
  `SEC_USER_AGENT`. `markets/questions/earnings_release.yaml`: the pilot's v1 battery (21 questions;
  tone questions tagged judgment).
- `markets/questions/sec_8k_card.yaml`: the scoring-v2 event card, nine doc_types sharing a core block.
- `filings_eval.py`: the event cards through the generic harness.

Parity with the pilot's recorded v1 baseline (same stored answers replayed, no calls; dev split:
186 gold documents, 700 probes, 2,510 market events):

| metric | recorded | engine.text |
|---|---|---|
| S | 0.4457 | 0.4457 |
| A / B / C / D / E | 0.155 / 0.720 / 0.617 / 0.341 / 0.000 | identical |
| reaction IC: base / prior / card | -0.0070 / 0.0194 / 0.1634 | identical |
| drift IC: prior / card | 0.0145 / 0.0316 | identical |
| leak excess AUC | -0.0098 | identical |

The port caught one bug before it could matter: replayed answers have no text, so a cache keyed on
text alone gave every document the same answers (A fell to 0); replay readers now key on the
document id. In the pilot repo the panel's earnings-text features also flow through this path and
match the pre-port observations exactly (1,041,880 of 1,041,880).

## 13. Point it at your own documents

1. Write a `DocumentSource` (fetch into your cache; `documents()` reads it; stamp `available_at`
   honestly; mask names you know).
2. Write a question YAML: reading questions you can label, judgment questions only if markets can
   grade them.
3. Add a `type: text` source to your market YAML, start with `reader: {kind: keyword}`.
4. Label a gold set (two labelers, a spot-check), build probes, and run the harness and the
   question loop before paying for a full read.
5. Only then switch the reader to Jev or Claude: `read_all` prints the estimate first.
