"""Shared fixtures: the demo market as a pure planted-signal market, and as a text market."""

import pandas as pd
import pytest

from engine import panel, run


def planted_cfg(tmp_path, **demo) -> dict:
    """The demo market with only numbers: `planted` moves next month's return by 3% per standard
    deviation; noise_1..3 move nothing; no text effects. 120 entities, 2010-2016."""
    cfg = run.read_config("demo")
    cfg["demo"] = {
        "entities": 120,
        "groups": 4,
        "daily_vol": 0.02,
        "cost_bps": 20,
        "seed": 7,
        "start": "2010-01-01",
        "end": "2016-12-31",
        "react": 0.0,
        "effects": {},
        "planted": {"beta": 0.03, "noise_features": 3, "seed": 7},
    } | demo
    cfg["sources"] = {"numbers": {"features": ["planted", "noise_1", "noise_2", "noise_3"]}}
    cfg["features"] = {
        "baseline": ["noise_1"],
        "all": ["planted", "noise_1", "noise_2", "noise_3"],
    }
    cfg["periods"] = {
        "tuning": ["2010-01-01", "2014-01-01"],
        "check": ["2014-01-01", "2015-07-01"],
        "holdout_start": "2015-07-01",
    }
    cfg["discovery"] = {"max_tests": 5, "transforms": ["level"]}
    cfg["registry"] = {
        "file": str(tmp_path / "registry.csv"),
        "holdout_unlock_log": str(tmp_path / "unlocks.csv"),
        "check_limit": 30,
    }
    cfg["cache_dir"] = str(tmp_path / "cache")
    return cfg


@pytest.fixture
def study(tmp_path):
    return run.study_from_config("demo", planted_cfg(tmp_path))


@pytest.fixture
def tuning_rows(study) -> pd.DataFrame:
    return panel.model_rows(study, run.build_panel(study, "tuning"))


def text_cfg(tmp_path, **demo) -> dict:
    """markets/demo.yaml with a temporary registry and cache, and `demo` settings overridden."""
    cfg = run.read_config("demo")
    cfg["registry"] = {
        "file": str(tmp_path / "registry.csv"),
        "holdout_unlock_log": str(tmp_path / "unlocks.csv"),
    }
    cfg["cache_dir"] = str(tmp_path / "cache")
    effects = demo.pop("effects", None)
    cfg["demo"] |= demo
    if effects is not None:
        cfg["demo"]["effects"] = effects
    return cfg


def study_and_rows(tmp_path, **demo):
    st = run.study_from_config("demo", text_cfg(tmp_path, **demo))
    return st, panel.model_rows(st, run.build_panel(st, "tuning"))


@pytest.fixture
def small_text(tmp_path):
    return study_and_rows(tmp_path, entities=120)
