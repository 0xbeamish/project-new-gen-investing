"""Regenerate the example CSVs: 30 tokens trading 24/7, 2021-2023, with planted signals.

    uv run python examples/csv/make_example.py

prices.csv     entity, date, close, volume (daily bars, closing 23:59 UTC; three tokens die in 2022,
               two list in 2022)
signals.csv    entity, available_at, feature, value. Every Sunday 12:00 UTC:
                 flow    planted: moves the next week's return by +2% per standard deviation
                 social  pure noise
documents.csv  entity, available_at, doc_type, text. Some Sundays 06:00 UTC, a short post; one that
               says an upgrade shipped adds +4% to the next week, one that reports an exploit -6%
The week a signal or post moves starts at the Monday 00:00 UTC decision (markets/csv_example.yaml).
Seeds 1-8 all give noise |t| < 1.96 on tuning years (seed 1: 1.0) and every planted input t > 4.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED, N, DAILY_VOL, BETA = 1, 30, 0.04, 0.02
POSTS = {
    "upgrade": ("The mainnet upgrade shipped on schedule and fees fell.", 0.04),
    "exploit": ("The team reports a security exploit; withdrawals are paused.", -0.06),
    "neutral": ("The foundation published its monthly community update.", 0.0),
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    days = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    sundays = days[days.dayofweek == 6]
    ids = [f"tok{i:02d}" for i in range(1, N + 1)]
    first = {e: days[0] for e in ids} | {
        "tok26": pd.Timestamp("2022-01-01"),
        "tok27": pd.Timestamp("2022-03-01"),
    }
    last = {e: days[-1] for e in ids} | {
        "tok28": pd.Timestamp("2022-05-15"),
        "tok29": pd.Timestamp("2022-08-20"),
        "tok30": pd.Timestamp("2022-11-10"),
    }
    flow = rng.standard_normal((len(sundays), N))
    social = rng.standard_normal((len(sundays), N))
    kind = rng.choice(list(POSTS), size=(len(sundays), N), p=[0.25, 0.15, 0.6])
    posted = rng.random((len(sundays), N)) < 0.5
    week_drift = BETA * flow + np.where(posted, np.vectorize(lambda k: POSTS[k][1])(kind), 0.0)
    # Sunday w's signals move the 7 bars after Monday's entry bar: Tuesday .. next Monday
    week_of = np.searchsorted(sundays, days - pd.Timedelta(days=2), side="right") - 1
    drift = np.where(week_of[:, None] >= 0, week_drift[np.clip(week_of, 0, None)] / 7, 0.0)
    rets = rng.standard_normal((len(days), N)) * DAILY_VOL + drift
    close = 100 * np.cumprod(1 + rets, axis=0)
    volume = np.exp(rng.normal(13, 0.5, (len(days), N)))
    px, sig, docs = [], [], []
    for j, e in enumerate(ids):
        live = (days >= first[e]) & (days <= last[e])
        px.append(
            pd.DataFrame(
                {
                    "entity": e,
                    "date": days[live].strftime("%Y-%m-%d"),
                    "close": close[live, j].round(3),
                    "volume": volume[live, j].round(0).astype(int),
                }
            )
        )
        for w, d in enumerate(sundays):
            if not first[e] <= d <= last[e]:
                continue
            at = (d + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
            sig += [
                (e, at, "flow", round(flow[w, j], 4)),
                (e, at, "social", round(social[w, j], 4)),
            ]
            if posted[w, j]:
                docs.append(
                    (
                        e,
                        (d + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "post",
                        POSTS[kind[w, j]][0],
                    )
                )
    pd.concat(px).to_csv(OUT / "prices.csv", index=False)
    pd.DataFrame(sig, columns=["entity", "available_at", "feature", "value"]).to_csv(
        OUT / "signals.csv", index=False
    )
    pd.DataFrame(docs, columns=["entity", "available_at", "doc_type", "text"]).to_csv(
        OUT / "documents.csv", index=False
    )


if __name__ == "__main__":
    main()
