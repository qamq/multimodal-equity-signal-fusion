from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.RandomState(123)
    dates = pd.date_range("2017-01-06", "2021-12-31", freq="W-FRI")
    n_stocks = 40
    rows = []
    state = 0

    for date in dates:
        if rng.rand() < 0.07:
            state = 1 - state
        context = rng.normal(loc=0.5 if state else -0.5, scale=0.5)
        latent = rng.normal(size=n_stocks) + 0.2 * context
        p_cnn = 1.0 / (1.0 + np.exp(-(0.6 * latent + rng.normal(scale=0.9, size=n_stocks))))
        p_rf = 1.0 / (1.0 + np.exp(-(0.9 * latent + rng.normal(scale=0.7, size=n_stocks))))
        r5 = 0.01 * latent + rng.normal(scale=0.03, size=n_stocks)
        r20 = 0.02 * latent + 0.004 * context + rng.normal(scale=0.05, size=n_stocks)

        for i in range(n_stocks):
            rows.append(
                {
                    "Date": date,
                    "StockID": str(1000 + i),
                    "p_cnn": p_cnn[i],
                    "p_rf": p_rf[i],
                    "y": int(r5[i] > 0),
                    "fwd_ret": r5[i],
                    "y_5d": int(r5[i] > 0),
                    "y_20d": int(r20[i] > 0),
                    # This synthetic weekly process defines these endpoints.
                    "label_end_5d": date + pd.Timedelta(weeks=1),
                    "label_end_20d": date + pd.Timedelta(weeks=4),
                    "ctx_market": context,
                }
            )

    return pd.DataFrame(rows)
