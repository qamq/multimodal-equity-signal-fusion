"""
multimodal_fusion.experimental.manager

Selector wrapper for the exploratory ensemble-weighting methods.

You set:
  - method: "fixed_blend", "dynamic_weight", "reliability_hedge", "moe_gating", ...

Then call:
  - fit(df) if supported
  - predict(df) -> Series up_prob

This file exists so higher-level pipelines never import weight classes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .fixed_blend import FixedBlend
from .dynamic_weight import DynamicWeight
from .reliability_hedge import ReliabilityHedge
from .moe_gating import MoEGating
from .moe_gating_date import MoEGatingDate
from .em_responsibility import EMResponsibility, EMResponsibilityConfig
from .contextual_bandit import ContextualBandit, ContextualBanditConfig
from .rl_policy import RLPolicy, RLPolicyConfig


@dataclass
class EnsembleManagerConfig:
    """Configuration for EnsembleManager."""
    method: str = "fixed_blend"
    method_kwargs: Optional[Dict[str, Any]] = None


class EnsembleManager:
    """EnsembleManager delegates fit/predict to the chosen weighting class."""

    # Initialize the manager and instantiate the requested weighting model.
    def __init__(self, cfg: Optional[EnsembleManagerConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else EnsembleManagerConfig()
        self.method = str(self.cfg.method).lower().strip()
        self.kwargs = dict(self.cfg.method_kwargs or {})
        self.model = self._build(self.method, self.kwargs)

    # Fit the underlying weighting model when a fit method is available.
    def fit(self, df: pd.DataFrame) -> "EnsembleManager":
        if hasattr(self.model, "fit"):
            self.model.fit(df)
        return self

    # Generate the final ensemble probability from the selected model.
    def predict(self, df: pd.DataFrame) -> pd.Series:
        out = self.model.predict(df)
        if not isinstance(out, pd.Series):
            raise TypeError("predict(...) must return pd.Series, got %s" % type(out))
        return out.rename(out.name or "up_prob")

    # Return date-level or global weighting diagnostics when supported.
    def get_weights(self) -> Any:
        if hasattr(self.model, "get_weights"):
            return self.model.get_weights()
        if hasattr(self.model, "weights_by_date_"):
            return getattr(self.model, "weights_by_date_")
        if hasattr(self.model, "global_w_cnn_"):
            w = float(getattr(self.model, "global_w_cnn_"))
            return {"w_cnn": w, "w_rf": 1.0 - w}
        return None
    
    # Return row-level diagnostics when the underlying model exposes them.
    def get_row_weights(self) -> Any:
        if hasattr(self.model, "get_row_weights"):
            return self.model.get_row_weights()
        if hasattr(self.model, "row_weights_"):
            return getattr(self.model, "row_weights_")
        return None

    # Construct the concrete weighting model from the normalized method name.
    def _build(self, method: str, kwargs: Dict[str, Any]) -> Any:
        m = self._normalize(method)

        if m == "fixed_blend":
            return FixedBlend(**kwargs)

        if m == "dynamic_weight":
            return DynamicWeight(**kwargs)

        if m == "reliability_hedge":
            return ReliabilityHedge(**kwargs)

        if m == "moe_gating":
            return MoEGating(**kwargs)

        if m == "moe_gating_date":
            return MoEGatingDate(**kwargs)

        if m == "em_responsibility":
            cfg = kwargs.get("config", None)
            if isinstance(cfg, dict):
                cfg = EMResponsibilityConfig(**cfg)
            return EMResponsibility(config=cfg)

        if m == "contextual_bandit":
            cfg = kwargs.get("config", None)
            if isinstance(cfg, dict):
                cfg = ContextualBanditConfig(**cfg)
            return ContextualBandit(config=cfg)

        if m == "rl_policy":
            cfg = kwargs.get("config", None)
            if isinstance(cfg, dict):
                cfg = RLPolicyConfig(**cfg)
            return RLPolicy(config=cfg)

        raise ValueError("Unknown method '%s'" % method)

    # Map common method aliases onto the canonical internal method names.
    def _normalize(self, method: str) -> str:
        aliases = {
            "fixed": "fixed_blend",
            "5050": "fixed_blend",
            "fixed5050": "fixed_blend",
            "fixed_5050": "fixed_blend",
            "dynamic": "dynamic_weight",
            "hedge": "reliability_hedge",
            "reliability": "reliability_hedge",
            "moe": "moe_gating",
            "moe_date": "moe_gating_date",
            "moe_gating_date": "moe_gating_date",
            "em": "em_responsibility",
            "bandit": "contextual_bandit",
            "rl": "rl_policy",
        }
        return aliases.get(method, method)
