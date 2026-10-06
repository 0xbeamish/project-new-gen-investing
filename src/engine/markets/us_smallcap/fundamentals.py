"""10-K ratios and 10-Q growth / margins / inventory from first-reported XBRL facts (ported from
the pilot). Every value is dated by the filing that first made it public."""

import numpy as np
import pandas as pd

from engine.markets.us_smallcap import sec

FEATURE_TABLE_COLUMNS = ["entity", "as_of", "feature", "value", "source"]


def _safe_div(a, b):
    return a / b if b not in (0, None) and pd.notna(a) and pd.notna(b) else np.nan


def stock_features(ticker: str, cik: int, filed_until=None) -> pd.DataFrame:
    """Turn a company's 10-K facts into ratio features, dated by the 10-K filing date."""
    facts = sec.annual_facts(cik, filed_until)
    if facts.empty:
        return pd.DataFrame(columns=FEATURE_TABLE_COLUMNS)
    wide = (
        facts.pivot_table(index="end", columns="concept", values="val", aggfunc="first")
        .reindex(columns=list(sec.CONCEPTS))
        .sort_index()
    )  # missing concept -> NaN, not KeyError
    filed = facts.groupby("end")[
        "filed"
    ].max()  # date every number for this year was public
    prev = wide.shift(1)  # prior fiscal year, for growth features

    rows = []
    for end, cur in wide.iterrows():
        p = prev.loc[end]
        g = cur.get
        feats = {
            "revenue_growth": _safe_div(
                g("revenue") - p.get("revenue"), abs(p.get("revenue", np.nan))
            ),
            "gross_margin": _safe_div(g("gross_profit"), g("revenue")),
            "operating_margin": _safe_div(g("operating_income"), g("revenue")),
            "net_margin": _safe_div(g("net_income"), g("revenue")),
            "roa": _safe_div(g("net_income"), g("total_assets")),
            "liabilities_to_assets": _safe_div(
                g("total_liabilities"), g("total_assets")
            ),
            "cash_to_assets": _safe_div(g("cash"), g("total_assets")),
            "rnd_intensity": _safe_div(g("rnd"), g("revenue")),
            "capex_intensity": _safe_div(g("capex"), g("revenue")),
            "share_dilution": _safe_div(g("shares") - p.get("shares"), p.get("shares")),
        }
        for name, value in feats.items():
            rows.append(
                {
                    "entity": ticker,
                    "as_of": filed[end],
                    "feature": name,
                    "value": value,
                    "source": "sec_xbrl",
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_TABLE_COLUMNS)


def to_wide(features: pd.DataFrame) -> pd.DataFrame:
    """Long feature table -> one row per (entity, as_of), one column per feature."""
    return features.pivot_table(
        index=["entity", "as_of"], columns="feature", values="value"
    ).reset_index()


STALE_QUARTER_DAYS = 200  # past this, the latest quarterly filing is too old to use


EXTENDED = [
    "q_rev_growth_yoy_chg4",
    "q_op_margin_chg_yoy",
    "q_rnd_intensity",
    "q_rnd_intensity_chg_yoy",
]


def quarterly_features(
    ticker: str, cik: int, extended: bool = False, filed_until=None
) -> pd.DataFrame:
    """Per filed quarter: growth, margins and inventory, dated by when that quarter was public.

    Columns: entity, filed, q_rev_growth_yoy, q_rev_growth_qoq, q_gross_margin,
    q_gross_margin_chg_yoy, q_op_margin, q_inventory_days, q_inventory_days_chg_yoy.
    extended adds EXTENDED: revenue-growth acceleration (YoY growth minus the YoY growth four
    quarters earlier), operating-margin and R&D-intensity change vs a year ago. A quarter's date
    stays the base filing date; an R&D value first filed after it is left out (NaN).
    """
    extra = ("rnd",) if extended else ()
    facts = sec.quarterly_facts(cik, extra, filed_until)
    if facts.empty:
        return pd.DataFrame()
    w = (
        facts.pivot_table(index="end", columns="concept", values="val", aggfunc="first")
        .reindex(columns=list(sec.QUARTERLY_CONCEPTS) + list(extra))
        .sort_index()
    )
    filed = facts[facts["concept"].isin(sec.FLOW)].groupby("end")["filed"].max()
    w = w[w.index.isin(filed.index)]
    if extended:  # R&D only where it was public by the quarter's own date
        rnd_filed = facts[facts["concept"] == "rnd"].set_index("end")["filed"]
        late = rnd_filed.reindex(w.index) > filed.reindex(w.index)
        w.loc[late.to_numpy(), "rnd"] = np.nan
    gross = w["gross_profit"].fillna(w["revenue"] - w["cost_of_revenue"])
    cogs = w["cost_of_revenue"].fillna(w["revenue"] - w["gross_profit"])
    out = pd.DataFrame(index=w.index)
    out["q_gross_margin"] = gross / w["revenue"].where(w["revenue"] > 0)
    out["q_op_margin"] = w["operating_income"] / w["revenue"].where(w["revenue"] > 0)
    out["q_inventory_days"] = w["inventory"] / cogs.where(cogs > 0) * 91

    def ago(days: int) -> pd.DataFrame:
        """The row for the quarter ending `days` earlier (within 20 days), else NaN."""
        idx = [
            w.index[np.argmin(np.abs((w.index - (e - pd.Timedelta(days=days))).days))]
            if len(w)
            else None
            for e in w.index
        ]
        ok = [
            i is not None and abs((e - pd.Timedelta(days=days) - i).days) <= 20
            for e, i in zip(w.index, idx)
        ]
        src = pd.concat([w["revenue"], out], axis=1)
        prev = src.reindex(idx).set_axis(w.index)
        return prev.where(pd.Series(ok, index=w.index), axis=0)

    yr, qtr = ago(364), ago(91)
    out["q_rev_growth_yoy"] = w["revenue"] / yr["revenue"].where(yr["revenue"] > 0) - 1
    out["q_rev_growth_qoq"] = (
        w["revenue"] / qtr["revenue"].where(qtr["revenue"] > 0) - 1
    )
    out["q_gross_margin_chg_yoy"] = out["q_gross_margin"] - yr["q_gross_margin"]
    out["q_inventory_days_chg_yoy"] = out["q_inventory_days"] - yr["q_inventory_days"]
    if extended:
        out["q_rnd_intensity"] = w["rnd"] / w["revenue"].where(w["revenue"] > 0)
        yr = ago(364)  # again: now carries this quarter's growth and R&D columns
        out["q_rev_growth_yoy_chg4"] = out["q_rev_growth_yoy"] - yr["q_rev_growth_yoy"]
        out["q_op_margin_chg_yoy"] = out["q_op_margin"] - yr["q_op_margin"]
        out["q_rnd_intensity_chg_yoy"] = out["q_rnd_intensity"] - yr["q_rnd_intensity"]
    return out.assign(
        entity=ticker, filed=filed.reindex(out.index).to_numpy()
    ).reset_index(drop=True)
