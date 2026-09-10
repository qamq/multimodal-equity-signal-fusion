from __future__ import annotations

import numpy as np

from multimodal_fusion import RankBlend, WalkForwardExpectedReturnFusion, WalkForwardLogisticStack


def test_rank_blend_is_within_date_score(synthetic_panel):
    model = RankBlend()
    score = model.predict(synthetic_panel)
    assert len(score) == len(synthetic_panel)
    assert score.between(0.0, 1.0).all()

    date = synthetic_panel["Date"].iloc[0]
    mask = synthetic_panel["Date"].eq(date)
    transformed = synthetic_panel.copy()
    transformed.loc[mask, "p_cnn"] = transformed.loc[mask, "p_cnn"] ** 3
    score2 = model.predict(transformed)
    np.testing.assert_allclose(score.loc[mask], score2.loc[mask])


def test_logistic_target_year_does_not_use_target_year_labels(synthetic_panel):
    base = synthetic_panel.copy()
    m1 = WalkForwardLogisticStack(start_year=2019, end_year=2019, label_lag_periods=1)
    m1.fit(base)

    changed = base.copy()
    mask = changed["Date"].dt.year.eq(2019)
    changed.loc[mask, "y"] = 1 - changed.loc[mask, "y"]
    m2 = WalkForwardLogisticStack(start_year=2019, end_year=2019, label_lag_periods=1)
    m2.fit(changed)

    np.testing.assert_allclose(m1.predictions_["up_prob"], m2.predictions_["up_prob"], rtol=0, atol=1e-12)


def test_expected_return_target_year_does_not_use_target_year_returns(synthetic_panel):
    base = synthetic_panel.copy()
    m1 = WalkForwardExpectedReturnFusion(start_year=2019, end_year=2019, min_train_rows=100)
    m1.fit(base)

    changed = base.copy()
    mask = changed["Date"].dt.year.eq(2019)
    changed.loc[mask, "fwd_ret"] = changed.loc[mask, "fwd_ret"] * -100.0
    m2 = WalkForwardExpectedReturnFusion(start_year=2019, end_year=2019, min_train_rows=100)
    m2.fit(changed)

    np.testing.assert_allclose(m1.predictions_["mu_hat"], m2.predictions_["mu_hat"], rtol=0, atol=1e-12)
