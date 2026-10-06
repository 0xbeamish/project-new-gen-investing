"""Signal-research engine: any data source in, one honest evaluation out.

  data      the observation and document contracts, calendars, the point-in-time checker
  panel     decision schedule x universe -> point-in-time panel -> model rows
  model     walk-forward models over per-period ranks
  score     the one scorer (rank IC, spreads, portfolios, t) and overlapping cohorts
  registry  every judged test, the rising bar, gated check years, the locked holdout
  run       a market YAML -> Study; build, report, test, discover, the cohort test
  decide    the optional AI decider and its feedback note
  improve   the tracking-driven ladder, the loop coordinator, the history replay
  spend     caps on every paid call
  text/     documents -> questions -> readers -> inputs, and whether each question is worth it
  markets/  plug-ins: demo (synthetic), csv (your files), us_stocks (SEC + EODHD)

See docs/HOW_IT_WORKS.md.
"""
