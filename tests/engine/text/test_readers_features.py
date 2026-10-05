import numpy as np
import pandas as pd
import pytest

from engine.spend import BudgetExceeded, Ledger
from engine.text import features as tf
from engine.text import history, readers
from engine.text import questions as tq

QS = tq.QuestionSet(
    "note",
    [
        tq.Question(
            "raised",
            "yes_no",
            "The text says guidance was raised.",
            keywords=[r"raised"],
        ),
        tq.Question(
            "tone",
            "scale",
            "Tone?",
            levels=("a", "b", "c", "d", "e"),
            keywords={4: [r"record"], 0: [r"weak"]},
        ),
        tq.Question(
            "kind",
            "choice",
            "Kind?",
            options=("up", "down", "none"),
            keywords={"up": [r"\bup\b"], "down": [r"\bdown\b"], "default": "none"},
        ),
    ],
)


def docs(texts, entity="E1", start="2020-01-01", doc_type="note"):
    at = pd.date_range(start, periods=len(texts), freq="30D", tz="UTC")
    return pd.DataFrame(
        {
            "entity_id": entity,
            "available_at": at,
            "doc_type": doc_type,
            "doc_id": [f"{entity}-{i}" for i in range(len(texts))],
            "text": texts,
        }
    )


def test_keyword_reader_answers_with_verified_quotes():
    r = readers.KeywordReader()
    ans, _ = r.read(
        {"text": "Guidance raised; demand at a record. Sales up."}, list(QS)
    )
    assert ans["raised"]["value"] == 0.9 and readers.quote_ok(
        ans["raised"]["evidence"], "Guidance raised"
    )
    assert max(ans["tone"]["value"], key=ans["tone"]["value"].get) == 4
    assert max(ans["kind"]["value"], key=ans["kind"]["value"].get) == "up"
    none, _ = r.read({"text": "Nothing here."}, list(QS))
    assert none["raised"]["value"] == 0.1 and none["raised"]["evidence"] is None
    assert max(none["kind"]["value"], key=none["kind"]["value"].get) == "none"
    assert not readers.quote_ok("made up", "Nothing here.")


class CountingPaid:
    """A fake paid reader: counts calls, costs $1 per document."""

    name, model, paid = "fake", "jev-1.13.0", True

    def __init__(self):
        self.calls = 0

    def read(self, doc, qs):
        self.calls += 1
        return {
            q.id: {"value": 0.5 if q.kind == "yes_no" else None, "evidence": None}
            for q in qs
        }, {"input_tokens": 10}

    def estimate_usd(self, doc, qs, ledger):
        return 1.0


def test_reading_service_caches_estimates_and_caps(tmp_path):
    d = docs(["a raised", "b", "c"])
    ledger = Ledger(
        tmp_path / "costs.csv", {"text_read_jev": 10.0}, {"typesafe": 100.0}
    )
    r = CountingPaid()
    with pytest.raises(BudgetExceeded):
        readers.read_all(r, d, {"note": QS})  # a paid reader needs a ledger
    assert readers.estimate(r, d, {"note": QS}, ledger)["usd"] == 3.0
    readers.read_all(
        r, d, {"note": QS}, ledger, "text_read_jev", tmp_path / "cache.sqlite"
    )
    assert r.calls == 3
    readers.read_all(
        r, d, {"note": QS}, ledger, "text_read_jev", tmp_path / "cache.sqlite"
    )
    assert r.calls == 3  # the content-hash cache: never paid twice
    small = Ledger(tmp_path / "costs2.csv", {"text_read_jev": 2.0}, {"typesafe": 100.0})
    with pytest.raises(
        BudgetExceeded
    ):  # projected $3 > $2 cap: refused before any call
        readers.read_all(CountingPaid(), d, {"note": QS}, small, "text_read_jev")
    assert "jev_engine_loop" in Ledger(tmp_path / "x.csv").step_caps


def test_replay_reader_keys_on_the_document_not_its_text():
    d = docs(["", ""])
    stored = {"E1-0": {"raised": {"value": 0.2}}, "E1-1": {"raised": {"value": 0.8}}}
    out = readers.read_all(readers.ReplayReader(stored), d, {"note": QS})
    assert (
        out[("E1-0", "E1")]["raised"]["value"] == 0.2
        and out[("E1-1", "E1")]["raised"]["value"] == 0.8
    )


def test_change_and_surprise_read_only_earlier_documents():
    d = pd.concat(
        [
            docs(["weak", "record", "weak"], "E1"),
            docs(["record", "record"], "E2", "2020-01-15"),
        ],
        ignore_index=True,
    )
    ans = readers.read_all(readers.KeywordReader(), d, {"note": QS})
    w = tf.answers_frame(d, ans, {"note": QS}, "n")
    w = tf.add_change(w, ["n_tone_level"], k=1)
    w = tf.add_surprise(w, ["n_tone_level"])
    e1 = w[w["entity_id"] == "E1"].sort_values("available_at")
    assert np.isnan(e1["n_tone_chg"].iloc[0])  # no earlier document
    assert e1["n_tone_chg"].iloc[1] > 2 and e1["n_tone_chg"].iloc[2] < -2
    first = w.sort_values("available_at").iloc[0]
    assert np.isnan(first["n_tone_surp"])  # nothing earlier to compare with
    obs = tf.to_observations(w, "txt", ["n_tone_level", "n_tone_chg"])
    assert set(obs["available_at"]) <= set(
        d["available_at"]
    )  # stamped with the document's time


def test_unread_documents_do_not_mask_earlier_answers():
    d = docs(["raised", "x"])
    ans = {("E1-0", "E1"): {"raised": {"value": 0.9}}}
    w = tf.answers_frame(d, ans, {"note": QS}, "n")
    assert list(w["doc_id"]) == ["E1-0"]


def test_history_base_rates_are_strictly_earlier_and_shrunk():
    days = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01", "2020-03-01"])
    ev = pd.DataFrame(
        {"event_day": days, "kind": ["a", "a", "a", "b"], "car": [0.1, 0.1, 0.1, -0.2]}
    )
    r = history.base_rates(ev, ev, [[], ["kind"]], {"car": pd.Timedelta(days=5)})
    assert (
        r["hist_n"].iloc[0] == 0 and r["hist_n"].iloc[1] == 0
    )  # day 2 can't use day 1 (lag 5)
    assert r["hist_n"].iloc[2] == 2
    assert 0 < r["hist_car_mean"].iloc[2] < 0.1  # shrunk toward the parent
    assert history.leak_check(r, {"car": pd.Timedelta(days=5)})["pass"]
    m = history.running_means(
        ev.assign(available_at=days), ["car"], [[]], at="available_at", k=0
    )
    assert np.isnan(m["car"].iloc[0]) and m["car"].iloc[3] == pytest.approx(0.1)
