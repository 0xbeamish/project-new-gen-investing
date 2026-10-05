import pandas as pd
import pytest

from engine import config, pipeline


def synthetic_cfg(tmp_path, **overrides) -> dict:
    cfg = config.read("synthetic")
    cfg["registry"] = {
        "file": str(tmp_path / "registry.csv"),
        "holdout_unlock_log": str(tmp_path / "unlocks.csv"),
        "check_limit": 30,
    }
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["synthetic"] |= overrides
    return cfg


@pytest.fixture
def study(tmp_path):
    return config.from_dict("synthetic", synthetic_cfg(tmp_path))


@pytest.fixture
def tuning_rows(study) -> pd.DataFrame:
    return pipeline.rows(study, pipeline.build_panel(study, "tuning"))
