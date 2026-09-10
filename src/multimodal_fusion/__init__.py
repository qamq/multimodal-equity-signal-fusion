"""Multimodal signal-fusion models for cross-sectional equity research."""

from .core.rank_blend import RankBlend
from .core.logistic_stack import WalkForwardLogisticStack
from .core.expected_return import WalkForwardExpectedReturnFusion
from .core.hmm_multihorizon import HMMMultihorizon, HMMMultihorizonConfig

__all__ = [
    "RankBlend",
    "WalkForwardLogisticStack",
    "WalkForwardExpectedReturnFusion",
    "HMMMultihorizon",
    "HMMMultihorizonConfig",
]
