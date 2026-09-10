"""
multimodal_fusion.experimental.fixed_blend

Implements a fixed weighted blend:
    up_prob = w_cnn * p_cnn + w_rf * p_rf

Defaults to a 50/50 blend, but weights can be set on class construction
and optionally overridden at predict-time.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class FixedBlend:
    """
    FixedBlend

    Produces a fixed convex combination of CNN and RF probabilities.

    Expected columns on input df:
      - p_cnn
      - p_rf
    """

    w_cnn: float = 0.5
    w_rf: float = 0.5
    eps: float = 1e-6

    def _resolve_weights(
        self,
        w_cnn: float | None = None,
        w_rf: float | None = None,
    ) -> tuple[float, float]:
        wc = self.w_cnn if w_cnn is None else float(w_cnn)
        wr = self.w_rf if w_rf is None else float(w_rf)

        if wc < 0 or wr < 0:
            raise ValueError("w_cnn and w_rf must be non-negative")

        total = wc + wr
        if total <= 0:
            raise ValueError("w_cnn + w_rf must be positive")

        # Normalize so the blend is always convex
        wc /= total
        wr /= total
        return wc, wr

    def predict(
        self,
        df: pd.DataFrame,
        *,
        p_cnn_col: str = "p_cnn",
        p_rf_col: str = "p_rf",
        w_cnn: float | None = None,
        w_rf: float | None = None,
    ) -> pd.Series:
        wc, wr = self._resolve_weights(w_cnn=w_cnn, w_rf=w_rf)

        p1 = pd.to_numeric(df[p_cnn_col], errors="coerce").astype(float).fillna(0.5)
        p2 = pd.to_numeric(df[p_rf_col], errors="coerce").astype(float).fillna(0.5)

        p = wc * p1 + wr * p2
        p = p.clip(self.eps, 1.0 - self.eps)
        return pd.Series(p.to_numpy(), index=df.index, name="up_prob")