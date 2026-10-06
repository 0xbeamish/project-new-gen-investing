"""US large caps: the 500 largest US filers each quarter, point-in-time, delisted companies included.

Everything is the us_smallcap plug-in (EODHD bars, SEC filings, measured spreads, the delisting
rule); only markets/us_largecap.yaml differs: the universe list, the cost table, and `data_end`.

    uv run python -m engine.markets.us_smallcap.universe large --until 2019-12-31
    uv run python -m engine.markets.us_smallcap.spreads build --lists data/largecap_universe.csv \\
        --out .cache/spreads_largecap.pkl --until 2019-12-31
"""

from engine.markets.us_smallcap import SOURCES, UsSmallcap


class UsLargecap(UsSmallcap):
    name = "us_largecap"


def build(cfg: dict):
    return UsLargecap(cfg), SOURCES
