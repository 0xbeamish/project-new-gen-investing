import pytest

from engine import config, pipeline


def text_cfg(tmp_path, **synthetic) -> dict:
    cfg = config.read("synthetic_text")
    cfg["registry"] = {
        "file": str(tmp_path / "registry.csv"),
        "holdout_unlock_log": str(tmp_path / "unlocks.csv"),
    }
    cfg["cache_dir"] = str(tmp_path / "cache")
    effects = synthetic.pop("effects", None)
    cfg["synthetic_text"] |= synthetic
    if effects is not None:
        cfg["synthetic_text"]["effects"] = effects
    return cfg


def study_and_rows(tmp_path, **synthetic):
    st = config.from_dict("synthetic_text", text_cfg(tmp_path, **synthetic))
    return st, pipeline.rows(st, pipeline.build_panel(st, "tuning"))


@pytest.fixture
def small_text(tmp_path):
    return study_and_rows(tmp_path, entities=120)
