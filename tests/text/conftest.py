"""The demo market's text eval sets, built once per module."""

import pytest

from engine import run
from engine.markets import demo
from engine.text import questions as tq
from tests.conftest import text_cfg


@pytest.fixture(scope="module")
def sets_and_q(tmp_path_factory):
    st = run.study_from_config("demo", text_cfg(tmp_path_factory.mktemp("e"), entities=120))
    return (
        st,
        demo.eval_sets(st.market, end="2015-01-01", n_gold=200),
        tq.load(run.path("markets/questions/demo_release.yaml")),
    )
