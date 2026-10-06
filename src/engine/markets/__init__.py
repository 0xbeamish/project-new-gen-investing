"""Market plug-ins. Each module exposes build(cfg) -> (market, {source name: source factory}).

demo       one synthetic market (planted numbers + documents): runs anywhere, no data, no keys
csv        bring your own data: prices.csv (+ signals.csv, documents.csv), no code needed
us_stocks  US small and large caps: SEC filings, EODHD prices, Form 4, measured costs, 8-K text
"""

import importlib


def load(plugin: str):
    """The plug-in module: a name under engine.markets, or a dotted path to your own."""
    return importlib.import_module(plugin if "." in plugin else f"engine.markets.{plugin}")
