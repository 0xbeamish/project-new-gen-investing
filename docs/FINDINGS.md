# Findings

What a year of honest tests found, before and after the engine existed. Aggregate numbers only:
no vendor values, no company-level results. Every judged test is a row in the registry
(`data/engine/us_smallcap/registry.csv` plus the inherited pilot logs; `data/registry_aggregate.csv`
in the shareable export). Tuning years 2012/2013-2019 unless stated; check years 2020-2023 were
looked at only where logged; 2024+ is a holdout no test has opened.

**The short version.** Nothing has cleared the bar. A small/mid-cap model over numbers plus
LLM-read earnings releases ranks next month's winners above its losers (rank IC 0.034, t 4.7 on
tuning years), but no portfolio built from it earns a significant return after measured trading
costs, and no LLM layer (as reader inputs, as a decider, with or without feedback) beats the
version without it; a month-by-month replay of the self-adjusting loops did not beat the frozen
system either. The market prices "did X happen" within days.

## Honesty rules (enforced by code where possible)

1. **Point in time.** A value may be used only from the moment it was public (`available_at`
   strictly before the decision); labels start strictly after. Checked on every panel.
2. **Survivors lie.** Universes include companies that later died; a delisting is an outcome
   (-30% unless a merger), not a missing row.
3. **No overlap.** Training labels end before the test period begins (purge on the real label end).
4. **Every test is logged, kept or not**, and the bar rises with each one (Bonferroni over every
   judged test: 1.96 at the first, 3.42 around the 80th).
5. **Check years are rationed and the holdout is locked**, by the pipeline, not by memory.
6. **Costs are measured, not assumed**, and every portfolio number is reported net.
7. **Green is not evidence.** Every component is tested on a planted signal it must find and on
   noise it must not.
8. **Every layer must beat the version without it**: text over numbers, a decider over the model.
9. **LLM readers see masked text** (company names hidden) and are never asked about outcomes.
10. **Pre-register** the rule, the sample and the bar before the first return is computed.

## The measured-cost lesson

Assumed costs flattered everything. Effective spreads measured from daily high/low/close
(Abdi-Ranaldo and Corwin-Schultz, averaged) gave round trips of roughly: micro ~110 bp, small ~65
bp, mid ~50 bp, large ~35 bp (medians 2014-2019; the pilot had assumed 150 / 60 / 60 / 30).
Re-netting with them turned every promising result flat or negative:

| strategy (tuning years) | gross / month | net / month (t) |
|---|---|---|
| numbers model, decile long-short | +0.60% | -0.09% (t -0.3) |
| numbers + text, decile long-short | +0.80% | +0.18% (t 0.6) |
| micro caps, numbers, decile long-short | +1.94% | +0.80% (t 1.0) |
| pick 1 of 10 (all three) | | negative |
| V1: small/mid + text, buy top 10%, hold until out of top 30% | +0.29% (t 2.6) | +0.17% (t 1.5); costs take 43% |
| V2: quarterly rebalance | +0.29% (t 2.4) | +0.16% (t 1.3) |
| V3: micro numbers, V1 rule | +0.40% (t 1.8) | +0.18% (t 0.8); costs take 54% |

Where there is an edge, it is about the size of the cost of trading it.

## Experiments, in order

### S&P 500 (survivors), yearly decisions
- 10-K ratios + walk-forward Ridge: plumbing, not evidence (40 survivors, then 622).
- **8-K event layers** (counts, severity from past reactions, Jev's good/bad direction on 60,943
  events, same-company tone change) vs numbers only at 6 / 12 / 24 months: best t +1.94 (severity
  at 12 months); none clears 2. Jev's direction tracks the 3-day reaction (r 0.14; 0.25 on
  earnings) but not later returns.
- **Monthly decisions** (8-Ks days old, not months): best t +0.79. The market prices 8-K news in
  large caps within days.
- **Timing** (wait for an uptrend vs buy now): waiting loses at every horizon (t -4.6 at 3 months,
  -2.3 at 24).
- **Research loop** (Claude proposes, Jev answers; 9 tests): earnings tone changes, abrupt
  executive exits, dividends and buybacks, results drivers, guidance, worry, governance, leading
  indicators, disclosure ambiguity. Best tuning t 1.56; none passed. Its check-year looks were
  logged as spent.
- **Discovery loop** (winners vs losers mined; 18 judged tests, sector scopes): best tuning t 2.27
  against bars of 2.8-3.1. Quarterly 10-Q numbers t -1.95; a separate cyclical-industry weight
  t +1.95. Gradient-boosted trees on all 449 inputs t -0.66.

### Small/mid caps (point-in-time, delisted included), monthly decisions
- Universe: ~1,350-2,000 companies a quarter from 2012; about half of the 2012 members are no
  longer listed.
- **Jev as decider** over model cards, pick 1 of 10, 12,979 batches, 2013-2019, Jev vs model per
  month: numbers-only cards t -0.96; cards + earnings-release facts t +1.19; + a yearly feedback
  note t +0.90 (Jev overrode 44% of the time; its overrides +0.16%/month vs the model's picks).
  None beats the model.
- **Earnings-release answers as model inputs** (47.7k releases read by Jev): t +0.61 vs numbers.
  Releases read section by section: t -1.69. News (GDELT) + Reddit attention: t -0.42.
- **Per-sector weights** (sector-only inputs in five sectors): t -0.25 to +1.05. Partial pooling
  toward the pooled weights: the rule chose "pooled" every year.
- **Micro caps** (numbers): t 1.61 vs random, gross of costs; after measured costs, see above.
- **Target fix**: a risk-adjusted target rewarded low volatility mechanically; switched to the
  within-sector rank of next month's return. Re-scored baselines (rank-IC t): numbers 3.50,
  numbers + text 4.66, micro 10.06. IC clears the bar; net returns don't.
- **Tradable event returns** (2,204 dev events, entry at the first open after the 8-K, 1 / 5 / 20
  days, four models from item code to + v1 card; 24 logged tests): the 3-day reaction edge is gone
  by the next open; 1- and 5-day long-shorts lose money after costs (t as low as -7.8); 20-day net
  +0.3% to +0.6% per event, t < 0.9.
- **Sector rotation**: SPDR sector momentum 1999-2019 t -0.44; SPDR composite t -1.10; French 49
  industries 1970-2019 net +0.39%/month, t 2.62 (bar 3.41).
- **Biotech catalyst run-up** (pre-registered; taken from the pilot's log, logged 2026-10-05): buy
  about 20 trading days before a stated PDUFA / advisory committee / readout date, sell the day
  before; 405 trades, net +1.26% per trade (median -0.67%), monthly +0.17%, t 0.16 (bar 3.42).
  Selling the day before does dodge the binary move (median 2-day move over the date 4.6%).
  `TODO(biocat)`: add any later biocat follow-up tests from its log.

### Scoring v2: text cards that read, history that prices
- Eval sets (2013-2019): 300 masked documents labeled by two Claude models (87.6% exact agreement)
  with a 30-document human spot-check; 700 probes; 4,301 market events.
- Judgment fields (how good, how big, how surprising) were where the labelers disagreed most; they
  are no longer card questions. Pricing comes from history base rates of earlier similar events.
- v1 baseline (dev): **S 0.446**; reading-field gold accuracy 0.16; consistency 0.72
  (counterfactual pass 0.18). Reaction IC: item code + size -0.007, + history prior 0.019, + v1
  card 0.163. The card reads good/bad news; it barely sizes it; it adds nothing significant to
  later drift (IC 0.032 +- 0.020).

### The engine
- **Parity**: the engine reproduces the pilot's recorded numbers to floating point with the
  pilot's settings, and moves them slightly with its stricter defaults (rank IC 0.034 -> 0.037 on
  the same months; 0.029 over all 84 months). The text eval reproduces S 0.4457 and the reaction
  rows exactly.
- **Jev decider re-test through the engine, feedback note on** (pre-registered; one logged test,
  12,874 batches over 84 months, 2013-2019): Jev's pick minus the model's own top pick, net of
  measured costs, -0.015%/month, **t -0.49** (bar 3.42). Both beat the batch average (net t 2.88
  and 2.85). Jev departed from the model in 8.5% of batches; those departures did 0.17%/month
  worse than the model's picks (better in 49%). Cost $1.55 against an estimate of $1.20: the
  cards tokenize at ~2.3 characters per token, denser than the 3 the estimate assumed, so card-style
  prompts need a lower floor.
- **History replay, loops ON vs FROZEN** (pre-registered; one logged test; 84 months 2013-2019):
  ON refit the weights monthly with the recency half-life chosen inside each refit and let the
  inner loop act under the coordination rules; FROZEN kept the v1 questions and yearly refits.
  V1 portfolio net of measured costs, ON minus FROZEN: -0.063%/month, **t -0.79** (bar 3.42);
  rank IC +0.0008 (t 0.16). Monthly refits raised turnover from 2.6x to 3.2x a year, which is
  most of the gap. Inside, 22 changes were judged and none accepted (best: dropping a demand-tone
  change field, t 2.1 against bars of 3.42-3.49); no rollbacks. One question split was proposed
  (a word 3-gram, "the fourth quarter", for guidance withdrawn), re-read over 52,094 releases for
  $7.43, and failed the diagnosis screen. The oscillation alarm (2 weight sign flips in 12
  refits) froze 54 of 59 inputs: calibrated on quarterly refits, it is far too sensitive at
  monthly refits, where weights near zero flip often. So the replay mostly tested monthly
  auto-recency refits against yearly ones; the alarm needs a magnitude floor before the inner loop
  can be judged on real data. Check years were not opened.
- **Replay re-run with the fixed alarm** (pre-registered; one logged test): a flip now counts only
  between refits where |w| > 0.5 x the refit's median |w|. It froze 37 of 59 inputs, not 54, and
  the rest is real instability: the monthly "auto" half-life choice moves between 6 and 36 months
  and the weights swing with it (with no recency weighting the same alarm freezes 4). More fields
  were eligible, and 33 changes were judged (drops and encodings of earnings-text fields; best t
  2.12 against a bar of ~3.43); none was accepted. No question split was possible: the unchanged
  rule allows cap / worst-case re-read estimate = $8 / $12.83 = 0 splits. With nothing accepted,
  ON is the same system as before: ON minus FROZEN -0.063%/month, **t -0.79** (bar 3.43). Cost $0.
  The ON-vs-FROZEN gap is the monthly auto-recency refit itself (more turnover, no better IC).
- **Yearly recency rule, free check only** (no paid replay): the half-life is now chosen once a
  year from {6, 12, 24, 36}, moving at most one step, validated by refitting month by month
  through the last 12 closed months. On the replay's weights it chose 6-24 months (2014-2019), and
  the fixed alarm still froze 29 of 59 inputs (a fixed 36-month half-life: 12; no recency
  weighting: 4). With monthly refits, any recency weighting makes most weights wander, so the
  pre-set gate (<= 8 frozen) failed and the replay was not run.
- **Loops on synthetic markets** (docs/TEXT.md): tracking flags a planted over-weighted field and a
  planted U-shaped field; the ladder fixes both; when the true error is a mis-weighted numeric input
  correlated with a text field, the coordinator leaves the text question alone and the outer refit
  fixes it.

## What would change the conclusion

- Text the market doesn't already read: proprietary documents, and readings that are more
  consistent than the crowd's, scored as surprise against what was expected.
- Lower costs (larger, more liquid names, or slower signals) so a 0.3%/month gross edge survives.
- A forward paper record: the only test no model could have read about in advance.
