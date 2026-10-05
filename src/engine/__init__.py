"""Signal-research engine: any data source in, one honest evaluation out.

Contracts (each module documents its own):
  observations  what a Source emits: entity_id, available_at (UTC), source, feature, value
  market        what an asset class provides: universe, bars/labels, costs, calendar, groups
  panel         decision schedule x universe -> latest point-in-time value per feature + label
  models        walk-forward models over per-period ranks
  scoring       the one scorer: rank IC, quantile spreads, portfolios, per-period t
  registry      every judged test, the Bonferroni bar, gated check years, the locked holdout
  discovery     candidate = feature + transform, judged against the market's baseline
  decider       optional LLM layer over model cards (off by default)

See docs/ENGINE.md.
"""
