from __future__ import annotations

import numpy as np

from multimodal_fusion.experimental.fixed_blend import FixedBlend
from multimodal_fusion.experimental.dynamic_weight import DynamicWeight
from multimodal_fusion.experimental.reliability_hedge import ReliabilityHedge
from multimodal_fusion.experimental.em_responsibility import EMResponsibility, EMResponsibilityConfig
from multimodal_fusion.experimental.moe_gating_date import MoEGatingDate


def _check_probabilities(series, n):
    assert len(series) == n
    assert np.isfinite(series.to_numpy(float)).all()
    assert ((series >= 0.0) & (series <= 1.0)).all()


def test_fixed_blend(synthetic_panel):
    out = FixedBlend().predict(synthetic_panel)
    _check_probabilities(out, len(synthetic_panel))


def test_online_date_level_methods(synthetic_panel):
    for model in [
        DynamicWeight(lookback_periods=13, label_lag=1),
        ReliabilityHedge(label_lag=1),
        EMResponsibility(config=EMResponsibilityConfig(label_lag=1, calibrate=False, max_iters=5)),
        MoEGatingDate(label_lag=1),
    ]:
        model.fit(synthetic_panel)
        out = model.predict(synthetic_panel)
        _check_probabilities(out, len(synthetic_panel))
