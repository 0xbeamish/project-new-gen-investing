"""Text scoring for any document type: documents in, point-in-time observations out, and the
machinery that decides whether the questions are worth asking. See docs/TEXT.md.

  documents   the document contract and the DocumentSource protocol
  questions   question sets as YAML: typed, tagged reading / judgment, versioned
  masking     hide who a document is about (look-ahead guard)
  readers     keyword (free), Jev, Claude, replay; the reading service (cache, estimate, caps)
  features    answers -> level, change vs the previous document, surprise vs history
  history     strictly-earlier base rates with backoff shrinkage
  source      TextSource: plugs documents + questions + a reader into a market as a Source
  evaluate    the metrics: answer key, probes, reaction / drift over a prior, gates, efficacy
  harness     score one (question set, reader) end to end; the entity-blocked bootstrap
  textloop    the question-improvement loop
  tracking    under- / over-weighted, mis-shaped, decaying or misread fields
"""
