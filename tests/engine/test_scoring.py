import numpy as np
import pandas as pd
import pytest

from engine import pipeline, scoring


def _period(t, scores, rets, group="g", cost=0.01):
    n = len(scores)
    return pd.DataFrame(
        {
            "entity_id": [f"s{i}" for i in range(n)],
            "decision_time": pd.Timestamp(t, tz="UTC"),
            "group": group,
            "score": scores,
            "fwd_return": rets,
            "rt_cost": cost,
        }
    )


def test_per_period_t_and_overlap_deflation():
    s = pd.Series([0.01, 0.02, 0.03, 0.02])
    t = s.mean() / (s.std(ddof=1) / 2)
    assert scoring.per_period_t(s) == pytest.approx(t)
    assert scoring.per_period_t(s, overlap=4) == pytest.approx(t / 2)
    assert np.isnan(scoring.per_period_t(pd.Series([0.1, 0.2])))
    assert scoring.overlap(21, 21) == 1 and scoring.overlap(30, 7) == 5


def test_rank_ic_is_one_for_a_perfect_ranking_and_uses_group_adjusted_returns():
    f = pd.concat(
        [
            # adjusted: a = -.03 -.01 .01 .03, b = -.06 -.02 .02 .06; scores follow that order
            _period("2020-01-31", [2, 4, 5, 7], [0.00, 0.02, 0.04, 0.06], "a"),
            _period("2020-01-31", [1, 3, 6, 8], [0.50, 0.54, 0.58, 0.62], "b").assign(
                entity_id=lambda d: "b" + d["entity_id"]
            ),
        ]
    )
    assert scoring.rank_ic(f).iloc[0] == pytest.approx(1.0)
    raw = f["score"].corr(f["fwd_return"], method="spearman")
    assert raw < 0.8  # on raw returns every b would beat every a


def test_quantile_spreads_gross_net_and_turnover():
    m1 = _period("2020-01-31", list(range(10)), [0.0] * 9 + [0.05])
    m2 = _period("2020-02-29", list(range(10)), [0.0] * 9 + [0.05])
    r = scoring.quantile_spreads(pd.concat([m1, m2]), q=0.1, min_names=10)
    assert r["gross"].tolist() == pytest.approx([0.05, 0.05])
    assert r["net"].tolist() == pytest.approx(
        [0.05 - 0.02, 0.05]
    )  # both legs new in month 1 only
    assert r["turnover"].tolist() == pytest.approx([1.0, 0.0])


def test_known_positive_planted_signal_is_found(study, tuning_rows):
    scored, w = pipeline.walk_forward(study).run(
        tuning_rows, ["planted", "noise_1"], study.market.calendar
    )
    ev = pipeline.evaluate(study, scored)
    assert ev["ic"] > 0.15 and ev["ic_t"] > 5
    assert ev["spread"]["gross_t"] > 3
    assert (w["planted"] > w["noise_1"].abs()).all()


def test_known_negative_pure_noise_is_not_found(study, tuning_rows):
    scored, _ = pipeline.walk_forward(study).run(
        tuning_rows, ["noise_1", "noise_2", "noise_3"], study.market.calendar
    )
    ev = pipeline.evaluate(study, scored)
    assert abs(ev["ic_t"]) < 1.96
    assert abs(ev["ic"]) < 0.05
