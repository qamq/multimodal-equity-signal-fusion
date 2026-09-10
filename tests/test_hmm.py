from __future__ import annotations

import numpy as np

from multimodal_fusion import HMMMultihorizon, HMMMultihorizonConfig


def test_hmm_walk_forward_smoke(synthetic_panel):
    model = HMMMultihorizon(
        HMMMultihorizonConfig(
            num_states=2,
            max_iter=2,
            tol=1e-3,
            l2=1.0,
            refit_freq="quarter",
            train_window_years=2,
            label_lag_periods=4,
            min_history_dates=20,
            context_cols=["ctx_market"],
            random_state=3,
            verbose=False,
        )
    )
    model.fit(synthetic_panel)

    assert model.row_pred_ is not None and len(model.row_pred_) > 0
    assert model.state_probs_by_date_ is not None and len(model.state_probs_by_date_) > 0
    assert np.isfinite(model.row_pred_["up_prob"].to_numpy(float)).all()
    assert model.row_pred_["up_prob"].between(0.0, 1.0).all()

    sums = model.state_probs_by_date_.groupby("Date")["prob"].sum().to_numpy(float)
    np.testing.assert_allclose(sums, 1.0, atol=1e-8)
