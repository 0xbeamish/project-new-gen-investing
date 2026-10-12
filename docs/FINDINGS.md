# Findings

What a year of honest tests found, before and after the engine existed. Aggregate numbers only:
no vendor values, no company-level results. Every judged test is a registry row:
`data/registry_aggregate.csv` for the research before the engine (the "pilot"),
`data/engine/<market>/registry.csv` since. Tuning years 2012/2013-2019 unless stated; check years
2020-2023 were opened only where logged; no test has opened the 2024+ holdout.

**The short version.** Nothing has cleared the bar. A small/mid-cap model over numbers plus
LLM-read earnings releases ranks next month's winners above its losers (rank IC 0.034, t 4.7 on
tuning years), but no portfolio built from it earns a significant return after measured trading
costs, and no LLM layer (as reader inputs, as a decider, with or without feedback) beats the
version without it; a month-by-month replay of the self-adjusting loops did not beat the frozen
system either. The market prices "did X happen" within days. A slow signal in large caps did no
better: a pre-registered test of visibly inflecting fundamentals, held 36 months, earned nothing
over the large-cap universe (t 0.04). On crypto (2021-2023, weekly), perp funding, DeFiLlama fee
growth and hacks each added nothing measurable over price momentum; a trading-volume surge
predicted underperformance out of sample (2024, t 4.2), but it is mostly short-term reversal and
low volatility, which rank well and do not pay after costs. A calm-coin book beat the
equal-weight universe out of sample (2018-2020, 2024, 2025-26 pooled: t 4.4) but trailed simply
holding Bitcoin in every period, and on inspection it rests on the 2025-26 crash (calmer coins fell
less), not on a persistent low-volatility premium. Hedged on perps and judged net of Bitcoin, the same volatility spread
earns +28 bp/week (t 1.66), short of its bar, and only in 2024-2026. Asked to pick the best of 5
anonymized tokens from the same inputs (2000 sets, five variants), Jev did no better than chance.

## Why no edge so far

Reasons, each tied to a logged result below; none is an excuse for a test that wasn't run.

1. **Public information is priced fast.** The v1 card (Jev reading a filing) explains the 3-day
   reaction to it: reaction rank IC 0.163 against 0.019 for the history prior alone. But traded
   from the next open, the edge is gone: 1- and 5-day long-shorts lose money after costs, and the
   card adds nothing significant to later drift (IC 0.032 +- 0.020).
2. **Costs are larger than the edge.** Numbers plus earnings-release text rank next month's winners
   above its losers (rank IC 0.034, t 4.7 on tuning years), yet measured round trips (35-110 bp by
   size) took 43% of the gross return of the low-turnover V1 portfolio and 54% of the micro-cap
   version. The low-turnover rule (buy the top 10%, hold until out of the top 30%) still nets only
   +0.17%/month, t 1.5.
3. **Everyone has the same data.** Every input was public: SEC filings and XBRL facts, Form 4
   insider trades, 8-K texts, vendor prices, GDELT news and Reddit attention. Edges in public data
   usually belong to whoever is faster, bigger, cheaper to trade or holds data others don't.
4. **A strict bar, on purpose.** The bar counts every judged test: 84 logged tests plus the 89
   changes the loop replays judged inside (counted since 2026-10-06), so the next test needs
   t >= 3.63 (it was 3.42-3.44 before the loop's internal tests were counted). Real but modest
   effects can't clear it on this much history: French 49-industry momentum, 1970-2019, net t 2.62.
5. **LLM-read backtests are compromised.** A model trained after 2019 may remember what happened
   to a company. Masking names, tickers, products and trial names reduces this, and the leak gate
   measures what remains, but it can't be proven zero; only a forward paper record can.

The engine's value is therefore the verdict itself: a fast, honest answer on new data,
especially proprietary text that the market hasn't already read.

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

**Calibration on real data (crypto 2021-2023, 2026-10-12; not a judged test).** Placebos are random
inputs (also persistent ones, like a slow real input) with no link to returns; plants add an effect of
known size. Each instrument is the engine's own code.

| instrument | placebo passes at t 1.96 (expect 5%) | placebo passes at the bar (2.9) | plant found at MDE50 / MDE80 (expect 50% / 80%) |
|---|---|---|---|
| input rank IC (tests 1-7) | 4.7% iid, 3.7% persistent (N 300 each) | 0 of 600 | 49% / 80% (N 200) |
| long-short book, alpha vs BTC, HC3 (tests 8-9) | 6.4% (N 1000) | 0 of 1000 | 27% / 50% (N 100); expected 24% / 44% at the alpha that reached the book |
| Jev pick scorer (tests 10-14) | 5.3% (N 2000) | not run | 46% / 81% (N 500) |

The book is slightly liberal at 1.96 (6.4%), harmless at the bars used. A per-token premium reaches
the band book about 25% smaller (it holds names outside the top half), so its MDE in per-token terms is
about a third larger than printed. Not covered: whether the source data match reality, whether costs
are right, and the stock markets.

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
- **Parity**: the engine reproduced the pilot's recorded numbers to floating point with the
  pilot's settings (the reproduction needs the pilot's data and stays with it), and moves them
  slightly with its stricter defaults (rank IC 0.034 -> 0.037 on
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
- **Replay options** (pre-registered together; alarm and persistence now count refits):
  - Option 1, monthly refits, no recency weighting: 4 of 59 frozen, gate passed, run and logged.
    ON minus FROZEN +0.015%/month, **t 0.26** (bar 3.43); rank IC +0.0056 (t 1.75); turnover 2.7x
    vs 2.6x. 34 changes judged inside (drops, encodings; best t 1.15), none accepted; the one
    question split hit the re-read cache. Cost $0.
  - Option 2, yearly refits, recency chosen yearly: 19 of 59 frozen over 7 refits, gate failed,
    not run. Descriptively its outer loop alone lowers rank IC vs FROZEN (-0.009, t -2.15).
  - Descriptive, not logged: option 1's outer loop vs option 2's, rank IC +0.015 (t 2.45), V1 net
    +0.076%/month (t 0.78). Recency weighting hurt on this data; more frequent refits without it
    helped IC a little, not net returns.
- **Registry correction (2026-10-06).** The 22 + 33 + 34 = 89 changes judged inside the three
  replays raised the bar only within each replay and were never counted in the registry, so later
  tests faced too low a bar. They are now `loop_internal` rows (every loop records its own from now
  on): the shared count goes from 84 to 173 and the next bar from 3.44 to 3.63. No earlier verdict
  changes: the rank-IC baselines that cleared their bars (t 3.50, 4.66, 10.06) would also clear 3.63
  and still failed after costs, and nothing else reached t 3.
- **Loops on synthetic markets** (docs/HOW_IT_WORKS.md): tracking flags a planted over-weighted field and a
  planted U-shaped field; the ladder fixes both; when the true error is a mis-weighted numeric input
  correlated with a text field, the coordinator leaves the text question alone and the outer refit
  fixes it.

### Large caps, multi-year holds (us_largecap)
The "catch the business while it inflects, hold for years" bet that retail investors made on a
handful of famous names. Earlier tests ranked small/mid caps monthly and excluded the 500 largest.
- **Market.** The same us_stocks plug-in as us_smallcap (prices, SEC filings, measured spreads,
  the delisting rule), configured differently. It reads nothing after 2019-12-31: not prices, not
  filings, not 8-Ks.
- **Universe.** The 500 largest US filers each quarter, point-in-time, delisted companies
  included: 890 companies over 33 formations, 496-500 a quarter. 156 of them stopped trading
  inside the window; their delisting returns are counted.
- **Universe data errors.** The raw size rank had fake giants: XBRL share counts off by 1,000x,
  and vendor closes 10-1,000x the traded price, some back-adjusted for a *later* reverse split.
  Those would pull future collapses into a large-cap list. The list now ranks a value checked
  against the SEC-reported public float, carried forward by adjusted returns; foreign (20-F/40-F)
  filers are dropped. Estimated residual: 1 row in 17,446.
- **Costs.** Measured round trips: median 33 bp (IQR 22-49).
- **Test** (pre-registered in 65b7966 before any return; one logged test). The signal is an
  equal-weight mean of percentile ranks of five point-in-time 10-Q/10-K inputs: revenue growth
  acceleration, gross margin change, operating margin change, R&D-intensity change, and revenue
  growth.
  - Formation: quarterly, 2011-09 to 2019-09. 2011-09 is the first quarter-end with the
    acceleration input for 90% of ranked names.
  - Portfolio: the top decile, about 39 names. Buy and hold 36 months; overlapping cohorts.
  - Benchmark: the equal-weight universe, same construction. Both net of measured costs.
  - Result: monthly net excess **+0.005%, Newey-West t 0.04** (bar 3.43; 99 months). Gross
    +0.007%. Costs are 0.016%/month: at a 36-month hold they no longer matter, and there is no
    edge for them to eat.
- **Descriptive** (not logged), monthly net excess (t):

  | variant | excess / month (t) |
  |---|---|
  | 24-month hold | -0.10% (-0.75) |
  | 12-month hold | -0.13% (-0.88) |
  | top 10 names, 36 months | -0.08% (-0.38) |
  | composite top minus bottom decile | +0.05% (0.37) |
  | acceleration, decile spread | -0.08% (-0.76) |
  | gross margin change | -0.00% (-0.01) |
  | operating margin change | +0.05% (0.47) |
  | R&D-intensity change | -0.06% (-0.35) |
  | revenue growth | -0.01% (-0.04) |

  A foresight score (each name's realized next-12-month return) through the same harness:
  +1.44%/month, t 5.4. The harness finds an edge when one exists.
- **Power.** 99 months is about 2.75 independent 36-month periods. The realized tracking error,
  1.3%/month, gives a standard error of about 0.11%/month. So the test detects a true excess of
  about 0.4%/month (~5%/year) half the time and about 0.5%/month (~6%/year) 80% of the time.
  It rules out a large, reliable edge for this composite; it cannot rule out a small one.
- **The famous names.** Whether the two names that prompted this were in the top decile in
  2013-2016 is anecdote: choosing them is hindsight. It was looked at descriptively and is
  company-level, so it isn't recorded here.

### Crypto, weekly (crypto)
The same engine on a different asset class, free data only (Binance public archive, DeFiLlama).
- **Market.** The 100 most-traded Binance USDT pairs each week, by mean daily volume over the 30
  days before the decision. Delisted pairs included: 341 tokens over 2021-2023, 72 of which stop
  trading before 2025. Stablecoins, fiat, gold and leveraged tokens excluded. 2025+ was never
  downloaded.
- **Costs.** 2 x (taker fee 10 bp + square-root impact of a $10k trade): median 46 bp round trip
  (IQR 36-60). The stock work's spread estimators failed on 24/7 bars (median 0 and 255 bp).
- **Tests** (pre-registered in 0de36c8 before any data was built; amended in 7f4e38a and 55ba17b
  before any test). Crypto keeps its own registry: the owner chose, before any result, not to share
  the stock tests' count (bar 3.63, which could only see a rank-IC gain of 0.070). The three tests
  form one family at t >= 2.394. Each is the weekly gain in rank IC over a price baseline
  (momentum 1/4/12 weeks, 4-week volatility, log dollar volume), 117 scored weeks of 2021-2023.
  Minimum detectable gain: 0.051 at 80% power, 0.038 at 50%.

  | input | coverage | gain in rank IC | t (bar 2.394) |
  |---|---|---|---|
  | `funding_7d`: mean perp funding rate, last 7 days | 83% | -0.0018 | -1.22 |
  | `fee_vs_price_90d`: 90-day fee growth minus price growth (DeFiLlama) | 26% | -0.0007 | -1.22 |
  | `hack_usd_30d`: USD lost to hacks of the token's protocol, last 30 days | 12 tokens | +0.0002 | +1.27 |

  None kept; the check year (2024) was not opened. The measured gains are near zero, far below
  the MDE: none of the three adds a large edge over price features; a small one can't be ruled out.
- **Exploration, then two confirmations on 2024** (pre-registration 2, 6844b8c). Descriptive
  sector analysis of 2021-2023 (token x quarter, 14 hand-labelled sectors; not logged) found that
  fundamentals grew *with* price within the quarter (active users, DEX volume, fees), but nothing
  measured *before* a quarter predicted it beyond luck. Two leads were confirmed once on 2024 alone
  (`engine confirm`: the feature's own weekly rank IC vs sector-adjusted returns, times the
  pre-registered sign):

  | lead | scope | 2024 rank IC (x sign) | t | bar | kept |
  |---|---|---|---|---|---|
  | `dex_volume_growth_90d`, + | L1 (median 16 names) | -0.028 | -0.83 | 2.498 | no (MDE 0.117: underpowered) |
  | `px_volume_growth_90d`, - | all (median 96) | +0.062 | **4.21** | 2.576 | **yes** |

  The kept one is mostly the price baseline in disguise. On the tuning weeks, a 90-day volume surge
  ranks with 12-week momentum at 0.63, and plain short-term reversal (`px_mom_1w`, -0.057, t 5.2) and
  low volatility (`px_vol_4w`, -0.077, t 6.2) are as strong. Net of momentum, volatility and size,
  volume growth adds only 0.018 rank IC (t 2.0). The price baseline as a model ranks weekly winners
  well (rank IC 0.083, t 6.1, tuning) but does not pay after costs: decile spread net +0.32%/week
  (t 0.28), the long-only band net +0.06%/week (t 0.34), with 72% of its gross lost to costs.
- **Positioning, confirmed on 2024 net of the price baseline** (pre-registration 3, cd8e394). Chain
  exploration found that fundamentals follow price (price leads active addresses, fees and
  stablecoins by a month) and pointed to inputs upstream of price. Two were confirmed once on 2024,
  judged on what they add beyond momentum, volatility and dollar volume (`engine confirm --net`):

  | lead | 2024 rank IC (x sign), net of price | t | bar | kept |
  |---|---|---|---|---|
  | `oi_to_mcap`, -: crowded perp leverage lags (median 66 names) | +0.039 | 1.89 | 2.638 | no |
  | `account_long_short_7d`, -: retail long-heavy lags (median 71) | +0.029 | 1.65 | 2.690 | no |

  Both point the predicted way, but neither clears its bar; MDE ~0.058 (underpowered by the
  engine's rule), so the honest reading is "couldn't tell". 2024 is now spent on them; a fresh test
  needs 2025+ (sealed) or forward paper trading.
- **The low-volatility book, pooled out of sample: KEPT, but it does not beat Bitcoin**
  (pre-registration 4, c41afc8; the holdout opened once, 2026-10-10). Chosen on 2021-2023
  (+30 bp/week net over the equal-weight universe, t 1.44): each week, hold the calmest 20% of the
  100-token universe within sector (keep while in the calmest 60%), spot, long-only, net of costs.
  Judged once on every period it was not chosen on:

  | period | weeks | net excess vs equal-weight universe | t | book, %/yr | BTC, %/yr | equal-weight, %/yr |
  |---|---|---|---|---|---|---|
  | 2018-2020 (before tuning) | 156 | +43.5 bp/wk | 2.64 | -1 | +29 | -22 |
  | 2024 (check) | 52 | +31.8 bp/wk | 1.39 | +44 | +118 | +16 |
  | 2025-01 to 2026-09 (holdout) | 90 | +69.2 bp/wk | 3.56 | -54 | -11 | -69 |
  | **pooled** | 298 | **+49 bp/wk** | **4.41** (bar 2.734) | | | |

  The calm book beat the average altcoin in each period as logged (but see the correction below).
  But the book is mostly the largest coins (BNB, BTC, ETH, TRX, LINK), and it trails simply holding
  BTC in every period (-30, -43, -100 bp/week; t -0.5, -0.5, -1.6). The excess is "the average
  altcoin bleeds"; capturing it needs a short leg (e.g. perps on the equal-weight basket, whose
  costs and funding are not in this number). Diagnostics after the logged run: two stable-value
  tokens slipped past the exclusion list in 2025-26 (U, XAUT); without them the holdout is +73 bp/wk
  (t 3.60), so they did not drive it. Unlabelled tokens fall in sector "Other" (8% of the universe
  in 2018-20, 45% in 2025-26).
- **Correction (2026-10-10; descriptive diagnostics after the logged run; the registry row stands as logged).**
  "Beats the average altcoin in every period, so the effect is real and persistent" overstates it. Test 8
  measures excess over the equal-weight universe and is not beta-adjusted.
  (1) About half the 2018-20 excess came from one-token sectors, which a within-sector rank always holds (a
  lone name ranks 1.0). Without sectors under 5 names, 2018-20 is +26 bp/wk (t 1.06) and 2024 +23 (t 0.87);
  pooled t 2.79 (descriptive, not a judged test), carried by 2025-26.
  (2) Ranking across the universe instead of within sector turns 2018-20 to -8 bp/wk.
  (3) What survives is 2025-26 (+72 bp/wk, t 3.2), a crash in which the equal-weight universe fell 69%/yr.
  Best reading: calmer, likely lower-beta coins fell less in a crash, plus small sectors held regardless of
  volatility in 2018-20; not a persistent low-volatility premium. The book's beta was not measured; alpha
  after BTC beta (pre-registration 5) is the statistic that would test it.
  Design flaw for any within-sector band rule: every sector contributes its top name, so small sectors are
  over-held (a singleton always, a 2-name sector half the time).
  Provenance: singletons.py. 2024 diagnostic cells use the gated frame (panel from 2021-01-01, data_end); an
  earlier 2024-only frame shifted the start week. 2025-26 was re-read outside the gate for this diagnostic
  (recorded in holdout_unlocks.csv).
- **The low-volatility spread on perps, net of Bitcoin: NOT kept** (pre-registration 5, dc5f105,
  amended e3b924b; crypto-0009, 2026-10-11). Every date had been seen before, so this is weaker
  evidence; it asks the question test 8 never asked (excess net of BTC beta). Each Monday, rank the
  perp-listed top-100 by 4-week volatility across the universe; long the calmest 20% (held while in
  the calmest 60%), short the most volatile 20% (mirror), half the capital per leg, Binance USD-M
  perps, net of taker fees, square-root impact and funding. Statistic: alpha of the weekly net return
  regressed on BTC's perp return, HC3 t, pooled 2020-01 to 2026-09 (343 weeks; 7 thin weeks of 2020
  dropped). The first attempt stopped before any return (the funding cache ended 2024-12); both
  holdout unlocks are logged.

  | period | weeks | alpha vs BTC | HC3 t | dollar-neutral net |
  |---|---|---|---|---|
  | 2020 | 45 | +9.7 bp/wk | 0.14 | +26.4 bp/wk |
  | 2021-2023 | 156 | +3.7 bp/wk | 0.13 | -1.3 bp/wk |
  | 2024 | 52 | +57.2 bp/wk | 1.77 | +52.2 bp/wk |
  | 2025-01 to 2026-09 | 90 | +48.4 bp/wk | 2.49 | +49.0 bp/wk |
  | **pooled** | 343 | **+28.1 bp/wk** | **1.66** (bar 2.773) | +23.6 bp/wk (t 1.36) |

  - Beta to BTC is about zero (-0.05): both legs move with BTC almost one for one (long leg 1.03,
    short leg 1.12; their weekly returns correlate 0.78-0.94 by period).
  - Before funding, the volatility spread is +39.6 bp/wk; funding costs the book 11.4 bp/wk, about a
    third. Funding is a real cost of holding it, so the judged number is the one after funding.
  - Costs are small because the band trades little: one-way turnover 7% (long) and 12% (short) of
    each leg a week; costs 4.2 bp/wk, 11% of gross.
  - Risk: worst week -18.1%, maximum drawdown -34%, 5 weeks in which the short leg rose more than
    30%. Squeezes, liquidations, margin calls, position limits and the perp-spot basis are not modelled.
  - The effect appears only in 2024-2026 (t 1.77, 2.49), the same dates on which test 8's excess
    concentrated, and those dates have now been read for test 8, its diagnostics and this test. The data
    cannot separate "a low-volatility premium since 2024" from "a regime".
  - Power: residual sd 319 bp/wk. Telling a true +28 bp/wk from zero at t 2 takes about
    (2 x 319 / 28)^2 = 520 weeks (10 years) of fresh data, and about 19 years at the 2.773 bar. Forward
    collection cannot settle this rule on any useful horizon, so it is not a reason to start collecting.
- **Tests 10-14: Jev picks 1 of 5 tokens (pre-registration 6, 2026-10-12): none kept.** Jev
  (jev-1.13.0) saw 5 anonymized tokens (names, dates and prices hidden) with all 19 inputs and picked
  one; 2000 fixed sets from 2021-07 to 2023-11, the same sets for five variants, one family (bar 2.91).
  Score: pick minus the set's average return, before costs; random = 0, picking the actual winner
  every time = +12.85%. Jev is not trained on customer requests and each call is stateless.

  | variant | gain per set | t | hit rate (random 20%) |
  |---|---|---|---|
  | V0 all inputs, numbers and percentiles | -0.12% | -0.35 | 21.9% |
  | V1 inputs as plain words | -0.38% | -1.20 | 22.1% |
  | V2 plus each input's walk-forward track record | -0.07% | -0.26 | 19.9% |
  | V3 avoid the worst (gain from avoiding) | -0.20% | -0.50 | 21.1% |
  | V4 best over 4 weeks | -1.37% | -1.91 | 20.3% |

  - Jev's confidence means nothing: in V0, picks it gave 90%+ hit 25% of the time, those at 40-70% hit 23%.
  - Free rules on the same sets did no better: the calmest token -0.12% (t -0.42), an IC-weighted score
    from the same track record -0.09% (t -0.36). The calmest token beats the set's median by 1.23%
    and ranks 2.84 of 5 on average, but misses the rare huge winners, so it doesn't beat the average.
  - Power: a 0.96%/set edge (1 week) would have been found 80% of the time; an input as good as the
    engine's best (rank IC 0.083) would earn about 1%. Cost $1.44 for 10,000 calls.
- **Not tested here.** Text (Reddit, X, governance proposals): an AI reader remembers 2021-2024,
  so text needs forward collection and paper trading.

## What would change the conclusion

- Text the market doesn't already read: proprietary documents, and readings that are more
  consistent than the crowd's, scored as surprise against what was expected.
- Lower costs (larger, more liquid names, or slower signals) so a 0.3%/month gross edge survives.
- A forward paper record: the only test no model could have read about in advance.
