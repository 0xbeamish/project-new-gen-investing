"""Market plug-ins. Each module exposes build(config) -> (market, {source name: source factory}).

synthetic    a planted-signal toy market: runs anywhere, no data, used by tests and the docs
us_smallcap  US small/mid caps (SEC filings, EODHD prices, Form 4, Jev's earnings-release reads)
us_largecap  US large caps: the us_smallcap plug-in with another universe, cost table and data end
"""

import importlib


def load(plugin: str):
    return importlib.import_module(
        plugin if "." in plugin else f"engine.markets.{plugin}"
    )
