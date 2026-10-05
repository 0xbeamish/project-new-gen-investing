import pytest

from engine import pipeline
from engine.config import path
from engine.registry import (
    CheckLimitReached,
    HoldoutLock,
    HoldoutLocked,
    Registry,
    required_t,
)


def test_bonferroni_bar():
    assert required_t(1) == pytest.approx(1.96, abs=0.005)
    assert required_t(10) == pytest.approx(2.81, abs=0.005)
    assert required_t(50) == pytest.approx(3.29, abs=0.005)


def test_each_judged_test_raises_the_bar_and_unjudged_rows_are_free(tmp_path):
    r = Registry(tmp_path / "reg.csv", "m")
    assert r.next_bar() == pytest.approx(required_t(1))
    r.record({"kind": "test", "name": "a", "t_tune": 1.0})
    r.record(
        {
            "kind": "test",
            "name": "b",
            "t_tune": None,
            "note": "unknown feature; not judged",
        }
    )
    r.record({"kind": "parity", "name": "p", "t_tune": 5.0})
    assert r.n_judged() == 1
    assert r.next_bar() == pytest.approx(required_t(2))
    assert ("a", "") in r.tested()


def test_inherited_pilot_logs_keep_counting_and_reproduce_their_bars():
    """The research repo inherits the pilot's raw logs; the shareable export inherits one
    aggregate CSV with the same rows. Either way the count continues and the bars reproduce."""
    raw = [path("data/experiments.csv"), path("data/discover_log.csv")]
    if all(p.exists() for p in raw):
        sources, discover = raw, "discover_log.csv"
    else:
        sources, discover = [path("data/registry_aggregate.csv")], "pilot discovery log"
    r = Registry(path("data/engine/_nonexistent.csv"), "us_smallcap", sources)
    rows = r.rows()
    judged = Registry._judged(rows)
    assert r.n_judged() == int(judged.sum()) >= 79
    # From the MODEL(...) rows on, each logged bar is required_t(tests before it + 1). (The first
    # discover rounds also counted unjudged rows; that was fixed in the pilot, so those differ.)
    n, checked = 0, 0
    for i, row in rows.iterrows():
        if (
            row["origin"] == discover
            and judged[i]
            and str(row["name"]).startswith(("MODEL", "V1", "A1"))
        ):
            assert float(row["bar_tune"]) == pytest.approx(
                required_t(n + 1), abs=0.0015
            )
            checked += 1
        n += int(judged[i])
    assert checked >= 5
    assert r.check_uses() == 9  # the pilot's logged check-year looks


def test_check_period_is_gated_and_limited(tmp_path, study):
    with pytest.raises(PermissionError):
        pipeline.build_panel(study, "check")
    r = Registry(tmp_path / "reg.csv", "m", check_limit=2)
    for _ in range(2):
        bar = r.open_check()
        r.record(
            {
                "kind": "test",
                "name": "x",
                "t_tune": 9.0,
                "check_used": True,
                "bar_check": bar,
            }
        )
    assert r.next_check_bar() == pytest.approx(required_t(3))
    with pytest.raises(CheckLimitReached):
        r.open_check()


def test_holdout_is_locked_and_unlocks_are_logged(tmp_path, study):
    with pytest.raises(HoldoutLocked):
        pipeline.build_panel(study, "holdout")
    lock = HoldoutLock(tmp_path / "unlocks.csv")
    with pytest.raises(HoldoutLocked):
        lock.unlock("m", "  ")
    lock.unlock("m", "final check of the kept signal")
    assert lock.looks() == 1
    assert "final check" in (tmp_path / "unlocks.csv").read_text()
