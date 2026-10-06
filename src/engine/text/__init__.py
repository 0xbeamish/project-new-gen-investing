"""Text scoring for any document type: documents in, point-in-time inputs out, and the machinery
that decides whether each question is worth asking. See docs/HOW_IT_WORKS.md.

  questions  question sets as YAML: typed, tagged reading / judgment, versioned
  read       masking, readers (keyword free; Jev, Claude paid), the reading service
  source     TextSource: answers -> level / change / surprise observations; history base rates
  grade      the eval harness: answer key, probes, reaction and drift over a prior, gates
  tracking   is a field under- / over-weighted, mis-shaped, decaying or misread?
  improve    the question-improvement loop
"""
