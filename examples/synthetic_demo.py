"""Run the public fusion models on a synthetic weekly stock panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal_fusion import (
    HMMMultihorizon,
    HMMMultihorizonConfig,
    RankBlend,
    WalkForwardExpectedReturnFusion,
    WalkForwardLogisticStack,
)


def make_panel(seed: int = 7, n_stocks: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2016-01-08", "2021-12-31", freq="W-FRI")
    rows = []

    state = 0
    for date in dates:
        if rng.rand() < 0.08:
            state = 1 - state
        market = rng.normal(loc=0.5 if state else -0.2, scale=0.7)
        latent = rng.normal(size=n_stocks) + 0.25 * market

        cnn_score = 0.65 * latent + rng.normal(scale=1.0, size=n_stocks)
        rf_score = 0.95 * latent + 0.25 * market + rng.normal(scale=0.75, size=n_stocks)
        p_cnn = 1.0 / (1.0 + np.exp(-cnn_score))
        p_rf = 1.0 / (1.0 + np.exp(-rf_score))

        ret_5d = 0.008 * latent + rng.normal(scale=0.025, size=n_stocks)
        ret_20d = 0.018 * latent + 0.004 * market + rng.normal(scale=0.045, size=n_stocks)

        for i in range(n_stocks):
            rows.append(
                {
                    "Date": date,
                    "StockID": str(10000 + i),
                    "p_cnn": p_cnn[i],
                    "p_rf": p_rf[i],
                    "fwd_ret": ret_5d[i],
                    "y": int(ret_5d[i] > 0),
                    "y_5d": int(ret_5d[i] > 0),
                    "y_20d": int(ret_20d[i] > 0),
                    # Known endpoints of this synthetic weekly process only.
                    "label_end_5d": date + pd.Timedelta(weeks=1),
                    "label_end_20d": date + pd.Timedelta(weeks=4),
                    "ctx_market": market,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    panel = make_panel()

    rank = RankBlend()
    panel["rank_blend"] = rank.predict(panel)

    logistic = WalkForwardLogisticStack(start_year=2017, end_year=2021)
    logistic.fit(panel)
    logistic_rows = panel[panel["Date"].dt.year.between(2017, 2021)].copy()
    logistic_rows["logistic"] = logistic.predict(logistic_rows)

    expected = WalkForwardExpectedReturnFusion(
        start_year=2017,
        end_year=2021,
        min_train_rows=500,
    )
    expected.fit(panel)
    expected_rows = panel.merge(
        expected.predictions_,
        on=["Date", "StockID"],
        how="inner",
    )

    hmm = HMMMultihorizon(
        HMMMultihorizonConfig(
            num_states=2,
            max_iter=4,
            tol=1e-3,
            l2=1.0,
            refit_freq="quarter",
            train_window_years=2,
            label_lag_periods=4,
            min_history_dates=26,
            transition_smoothing=1e-3,
            context_cols=["ctx_market"],
            random_state=7,
            verbose=False,
        )
    )
    hmm.fit(panel)

    print("Synthetic rows:", f"{len(panel):,}")
    print("Rank-blend rows:", f"{panel['rank_blend'].notna().sum():,}")
    print("Logistic rows:", f"{len(logistic.predictions_):,}")
    print("Expected-return rows:", f"{len(expected.predictions_):,}")
    print("HMM rows:", f"{len(hmm.row_pred_):,}")
    print("HMM state rows:", f"{len(hmm.state_probs_by_date_):,}")
    print("Expected-return signal std:", f"{expected_rows['mu_hat'].std():.6f}")


if __name__ == "__main__":
    main()
