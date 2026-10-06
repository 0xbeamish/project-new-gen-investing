"""The registry: the rising bar, inherited history, the gated check period, the locked holdout."""

import pytest

from engine import run
from engine.registry import CheckLimitReached, HoldoutLock, HoldoutLocked, Registry, required_t


def test_bonferroni_bar():
    assert required_t(1) == pytest.approx(1.96, abs=0.005)
    assert required_t(10) == pytest.approx(2.81, abs=0.005)
    assert required_t(50) == pytest.approx(3.29, abs=0.005)


def test_each_judged_test_raises_the_bar_and_unjudged_rows_are_free(tmp_path):
    r = Registry(tmp_path / "reg.csv", "m")
    assert r.next_bar() == pytest.approx(required_t(1))
    r.record({"kind": "test", "name": "a", "t_tune": 1.0})
    r.record({"kind": "test", "name": "b", "t_tune": None, "note": "unknown feature; not judged"})
    r.record({"kind": "parity", "name": "p", "t_tune": 5.0})
    assert r.n_judged() == 1
    assert r.next_bar() == pytest.approx(required_t(2))
    assert ("a", "") in r.tested()


def test_inherited_history_keeps_counting_and_reproduces_its_bars():
    """The earlier research's judged tests (data/registry_aggregate.csv) are inherited: the count
    continues and its logged bars reproduce."""
    r = Registry(
        run.path("data/engine/_nonexistent.csv"),
        "us_smallcap",
        [run.path("data/registry_aggregate.csv")],
    )
    rows = r.rows()
    judged = Registry.judged(rows)
    assert r.n_judged() == int(judged.sum()) >= 79
    # From the MODEL(...) rows on, each logged bar is required_t(tests before it + 1). (The first
    # discovery rounds also counted unjudged rows; that was fixed back then, so those differ.)
    n, checked = 0, 0
    for i, row in rows.iterrows():
        if (
            row["origin"] == "pilot discovery log"
            and judged[i]
            and str(row["name"]).startswith(("MODEL", "V1", "A1"))
        ):
            assert float(row["bar_tune"]) == pytest.approx(required_t(n + 1), abs=0.0015)
            checked += 1
        n += int(judged[i])
    assert checked >= 5
    assert r.check_uses() == 9  # the earlier research's logged check-year looks


def test_check_period_is_gated_and_limited(tmp_path, study):
    with pytest.raises(PermissionError):
        run.build_panel(study, "check")
    r = Registry(tmp_path / "reg.csv", "m", check_limit=2)
    for _ in range(2):
        grant = r.open_check()
        r.record(
            {"kind": "test", "name": "x", "t_tune": 9.0, "check_used": True, "bar_check": grant.bar}
        )
    assert r.next_check_bar() == pytest.approx(required_t(3))
    with pytest.raises(CheckLimitReached):
        r.open_check()


def test_holdout_is_locked_and_unlocks_are_logged(tmp_path, study):
    with pytest.raises(HoldoutLocked):
        run.build_panel(study, "holdout")
    lock = HoldoutLock(tmp_path / "unlocks.csv")
    with pytest.raises(HoldoutLocked):
        lock.unlock("m", "  ")
    lock.unlock("m", "final check of the kept signal")
    assert lock.looks() == 1
    assert "final check" in (tmp_path / "unlocks.csv").read_text()


def test_committed_registries_share_one_count():
    """us_smallcap and us_largecap inherit each other and the earlier research: one bar."""
    small = run.read_config("us_smallcap")["registry"]
    large = run.read_config("us_largecap")["registry"]
    count = lambda reg: Registry(
        run.path(reg["file"]), "m", [run.path(p) for p in reg["inherit"]]
    ).n_judged()
    assert count(small) == count(large)


def _record_many(path, worker, n):
    reg = Registry(path, "m")
    for i in range(n):
        reg.record({"kind": "test", "name": f"w{worker}-{i}", "t_tune": 0.5})


def test_concurrent_writers_lose_no_rows(tmp_path):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_record_many, args=(tmp_path / "reg.csv", w, 15)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    own = Registry(tmp_path / "reg.csv", "m").own()
    assert len(own) == 60 and own["name"].nunique() == 60
    assert own["test_id"].is_unique
