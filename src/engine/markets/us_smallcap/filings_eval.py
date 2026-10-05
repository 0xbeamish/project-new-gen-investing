"""Worked example: SEC 8-K event cards through the generic text eval harness (engine.text.evaluate).

    uv run python -m engine.markets.us_smallcap.filings_eval     # parity with the pilot's v1 baseline

Inputs (the pilot's eval sets; tuning years 2013-2019, dev split only):
  data/eval/gold_v1.jsonl, probes_v1.jsonl           answer key and probes
  .cache/eval_market.pkl                             market set (reactions, drift, item codes)
  data/eval/history_rates_v1.csv, context_v1.csv     the prior the card must beat
  .cache/eval_jev_answers.jsonl                      Jev's stored v1 answers (replayed, no calls)
  data/smallcap_universe.csv                         size buckets

v1 = the batteries the pilot asked (earnings releases: the EARNINGS battery; executive changes:
EXEC_CHANGE; other 8-Ks: relation + direction + catalyst questions). Its answers map onto the
event-card questions (markets/questions/sec_8k_card.yaml) where an equivalent exists; card fields
v1 never asks are uncovered (skill 0). Parity is not a test: nothing goes to the registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import ROOT
from engine.text import evaluate as ev
from engine.text import questions as tq
from engine.text import readers

EVAL = ROOT / "data" / "eval"
QUESTIONS = ROOT / "markets" / "questions" / "sec_8k_card.yaml"
ANSWERS = Path(".cache") / "eval_jev_answers.jsonl"
MARKET = Path(".cache") / "eval_market.pkl"
GOLD_DOCS = Path(".cache") / "eval_gold_docs.pkl"
STEP = "eval_loop_jev"  # the pilot ledger step that paid for these answers
HIST_COLS = ["hist_car3_mean", "hist_car3_med", "hist_hit", "hist_drift"]
CTX_COLS = [
    "ctx_ret_1m",
    "ctx_ret_12m",
    "ctx_vol_60d",
    "ctx_days_since_earnings",
    "ctx_n8k_90d",
    "ctx_n502_2y",
    "ctx_press_2y",
    "ctx_cp_matched",
    "ctx_cp_large",
]
JUDGED_PREFIXES = (  # v1 answer columns that are judgments, not readings
    "direction",
    "demand_tone",
    "outlook_tone",
    "margin_tone",
    "hedging_language",
    "surprise",
    "driver_concrete",
)


# ---------------------------------------------------------------- v1 answers -> card fields
def v1_to_card(family: str, raw: dict, qsets: dict) -> dict:
    """Card-field values from v1 answers: yes/no -> p, choice -> {option: p}, scale -> {0..4: p}."""
    rel = raw.get("relation", {})
    card = {
        "relation": {
            o: rel.get(o, 0.0) for o in qsets["earnings"].get("relation").options
        },
        "is_subject": rel.get("self", 0.0),
    }  # direction / demand / surprise are judgment fields: reaction features only
    if family == "earnings":
        up, down, wd = (
            raw["guidance_raised"],
            raw["guidance_lowered"],
            raw["guidance_withdrawn"],
        )
        card["guidance_action"] = {
            "raised": up,
            "lowered": down,
            "withdrawn": wd,
            "maintained": 0.0,
            "introduced": 0.0,
            "none": max(0.0, 1 - up - down - wd),
        }
        card["one_off_hurt"] = max(raw["impairment"], raw["restructuring"])
    elif family == "exec":
        card["role"] = raw["role"]
        card["change_type"] = raw["change_type"]
        for f in (
            "effective_immediately",
            "successor_named",
            "planned_succession",
            "investigation",
        ):
            card[f] = raw[f]
    elif family in ("headline", "reddit"):
        d = (
            {int(k): v for k, v in raw["direction"].items()}
            if "direction" in raw
            else {}
        )
        s = card["is_subject"]
        card["stance"] = {
            "not_about_company_a": 1 - s,
            "negative": s * (d.get(0, 0) + d.get(1, 0)),
            "neutral_or_factual": s * d.get(2, 0),
            "positive": s * (d.get(3, 0) + d.get(4, 0)),
            "hype_or_promotion": 0.0,
        }
    elif family == "trial":
        p = raw["trial_positive"]
        card["primary_endpoint"] = {
            "met_with_significance": p / 2,
            "met_no_significance_stated": p / 2,
            "mixed_or_partly_met": 0.0,
            "not_met": (1 - p) / 2,
            "not_reported": (1 - p) / 2,
        }
        card["reg_outcome"] = {
            o: 0.0 for o in qsets["trial"].get("reg_outcome").options
        } | {
            "approved": raw["approval_received"],
            "no_regulatory_decision": 1 - raw["approval_received"],
        }
        card["next_step_dated"] = raw["dated_catalyst"]
    return card


def numeric(raw: dict) -> dict:
    """Flat numeric v1 answers for the reaction model (C, D)."""
    out = {}
    for qid, v in raw.items():
        if qid == "leak_probe":
            continue
        if isinstance(v, float):
            out[f"{qid}_p"] = v
        elif all(k.isdigit() for k in v):
            out[f"{qid}_level"] = sum(int(k) * p for k, p in v.items())
        else:
            out |= {f"{qid}_{k}": p for k, p in v.items()}
    return out


# ---------------------------------------------------------------- the prior the card must beat
def mcap_bucket(events: pd.DataFrame) -> pd.Series:
    u = pd.read_csv(
        ROOT / "data" / "smallcap_universe.csv",
        dtype={"code": str},
        keep_default_na=False,
        na_values=[""],
        parse_dates=["as_of"],
    )
    m = (
        pd.merge_asof(
            events[["cik", "accepted_et"]]
            .astype({"cik": int})
            .reset_index()
            .astype({"accepted_et": "datetime64[ns]"})
            .sort_values("accepted_et"),
            u[["cik", "as_of", "mcap"]]
            .astype({"as_of": "datetime64[ns]"})
            .sort_values("as_of"),
            left_on="accepted_et",
            right_on="as_of",
            by="cik",
            direction="backward",
        )
        .set_index("index")
        .sort_index()
    )
    return pd.cut(
        m["mcap"], [0, 7e8, 2e9, np.inf], labels=["small", "mid", "large"]
    ).astype(str)


def prior_features(market: pd.DataFrame) -> pd.DataFrame:
    """History base rates (strictly earlier events) + looked-up context, aligned to `market`."""
    rates = pd.read_csv(EVAL / "history_rates_v1.csv").set_index("doc_id")
    ctx = pd.read_csv(EVAL / "context_v1.csv").set_index("doc_id")
    out = pd.DataFrame(index=market.index)
    for c in HIST_COLS:
        out[c] = market["doc_id"].map(rates[c]).to_numpy()
    out["hist_log_n"] = np.log1p(
        market["doc_id"].map(rates["hist_n"]).fillna(0).to_numpy()
    )
    for c in CTX_COLS:
        out[c] = market["doc_id"].map(ctx[c]).to_numpy() if c in ctx else np.nan
    mcap = market["doc_id"].map(ctx["ctx_mcap"])
    out["ctx_log_mcap"] = np.log(
        mcap.where(mcap < 2e12)
    ).to_numpy()  # two bad share counts imply > $1T
    out["ctx_log_cp_mcap"] = (
        np.log(market["doc_id"].map(ctx["ctx_cp_mcap"]).to_numpy())
        if "ctx_cp_mcap" in ctx
        else np.nan
    )
    out = out.join(
        pd.get_dummies(
            market["doc_id"].map(ctx["sector"]).fillna("other"), prefix="ctx_sector"
        ).astype(float)
    )
    missing = out.isna()
    out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
    for c in (
        "ctx_ret_12m",
        "ctx_days_since_earnings",
        "ctx_log_cp_mcap",
    ):  # missingness can be informative
        if c in out:
            out[f"{c}_missing"] = missing[c].astype(float)
    return out


# ---------------------------------------------------------------- the run
def load_sets() -> dict:
    gold = [json.loads(x) for x in (EVAL / "gold_v1.jsonl").read_text().splitlines()]
    for g in gold:
        g["doc_type"] = g["stratum"]["family"]
    probes = [
        json.loads(x) for x in (EVAL / "probes_v1.jsonl").read_text().splitlines()
    ]
    market = pd.read_pickle(MARKET)
    raw = {
        r["key"].split("|", 1)[1]: r["raw"]
        for r in map(json.loads, ANSWERS.read_text().splitlines())
    }
    return {"gold": gold, "probes": probes, "market": market, "raw": raw}


def documents(sets: dict) -> pd.DataFrame:
    """Every dev document the v1 answers cover, as engine.text documents (text not needed: replay)."""
    rows = []
    for g in sets["gold"]:
        if g["split"] == "dev":
            rows.append((g["doc_id"], g["doc_type"]))
    rows += [(p["probe_id"], p["family"]) for p in sets["probes"]]
    m = sets["market"]
    m = m[m["split"] == "dev"]
    rows += list(zip(m["doc_id"], m["family"]))
    d = pd.DataFrame(rows, columns=["doc_id", "doc_type"]).drop_duplicates("doc_id")
    return d.assign(
        entity_id="-", available_at=pd.Timestamp("2020-01-01", tz="UTC"), text=""
    )


def run(out_path: Path | None = None) -> dict:
    sets = load_sets()
    qsets = tq.load(QUESTIONS)
    docs = documents(sets)

    def convert(doc, raw):
        return {
            q: {"value": v} for q, v in v1_to_card(doc["doc_type"], raw, qsets).items()
        }

    reader = readers.ReplayReader(sets["raw"], convert)
    answers = readers.read_all(reader, docs, qsets)
    preds = {k[0]: v for k, v in ev.values(answers).items()}

    dev = [g for g in sets["gold"] if g["split"] == "dev"]
    A, per_field = ev.gold_accuracy(dev, preds, qsets)
    judgment = {q.id for qs in qsets.values() for q in qs.judgment()}
    B = ev.consistency(sets["probes"], preds, judgment)
    E = ev.coverage(dev, preds, qsets)

    market = sets["market"]
    market = market[
        (market["split"] == "dev") & market["doc_id"].isin(sets["raw"])
    ].reset_index(drop=True)
    first_item = market["items"].str.split(",").str[0]
    base = (
        pd.get_dummies(first_item, prefix="item")
        .astype(float)
        .join(pd.get_dummies(mcap_bucket(market), prefix="mcap_b").astype(float))
    )
    feats = pd.DataFrame([numeric(sets["raw"][d]) for d in market["doc_id"]]).fillna(
        0.0
    )
    prior = prior_features(market)
    mk = pd.concat(
        [market[["z3", "car3", "abvol", "drift_22", "year"]], base, prior], axis=1
    )
    C = ev.reaction_power(mk, feats, list(base.columns), list(prior.columns))
    reading_feats = feats[
        [c for c in feats.columns if not c.startswith(JUDGED_PREFIXES)]
    ]
    C_reading = ev.reaction_power(
        mk, reading_feats, list(base.columns), list(prior.columns)
    )

    up = (market["car3"] > 0).astype(int).to_numpy()
    d_lvl = np.array(
        [
            sum(int(k) * p for k, p in sets["raw"][d]["direction"].items())
            for d in market["doc_id"]
        ]
    )
    leak = np.array([sets["raw"][d]["leak_probe"] for d in market["doc_id"]])
    gate = ev.leak_gate(d_lvl, leak, up)
    s = ev.summary(A, B["B"], C, E)
    cost = None
    ledger = ROOT / "data" / "costs.csv"
    if ledger.exists():
        led = pd.read_csv(ledger)
        cost = (
            float(led.loc[led["step"] == STEP, "usd"].sum())
            / max(1, len(sets["raw"]))
            * 1000
        )
    out = {
        "version": "v1 via engine.text",
        "summary": s,
        "components": {
            "A_gold": A,
            "B_consistency": B["B"],
            "invariance": B["invariance"],
            "counterfactual_pass": B["counterfactual_pass"],
            "unmasked_vs_masked_stability": B["unmasked_vs_masked_stability"],
            "C_reaction": C,
            "C_reaction_reading_features_only": C_reading,
            "E_coverage": E,
        },
        "per_field": per_field,
        "gates": {"leak": gate, "cost_usd_per_1k_docs": cost},
        "n": {
            "gold_dev": len(dev),
            "probes": len(sets["probes"]),
            "market_dev": len(market),
            "jev_docs": len(sets["raw"]),
        },
    }
    if out_path:
        out_path.write_text(json.dumps(out, indent=1, default=float))
    return out


def compare(engine: dict, recorded: dict) -> pd.DataFrame:
    rows = []
    for k in ("S", "A", "B", "C", "D", "E"):
        rows.append((f"summary.{k}", recorded["summary"][k], engine["summary"][k]))
    for model in ("base", "prior", "card"):
        for m in ("R2_mag", "IC_react", "IC_drift"):
            rows.append(
                (
                    f"reaction.{model}.{m}",
                    recorded["components"]["C_reaction"][model][m],
                    engine["components"]["C_reaction"][model][m],
                )
            )
    rows.append(
        (
            "leak.excess",
            recorded["gates"]["leak"]["excess"],
            engine["gates"]["leak"]["excess"],
        )
    )
    df = pd.DataFrame(rows, columns=["metric", "recorded", "engine"])
    df["diff"] = df["engine"] - df["recorded"]
    return df


def main() -> None:
    import os

    os.chdir(ROOT)
    out = run(ROOT / "data" / "engine" / "us_smallcap" / "text_parity.json")
    rec = json.loads((EVAL / "baseline_v1.json").read_text())
    print(compare(out, rec).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(out["n"]), file=sys.stderr)


if __name__ == "__main__":
    main()
