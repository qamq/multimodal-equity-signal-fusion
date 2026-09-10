"""Cross-sectional rank blending for two expert probability forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class RankBlend:
    """Blend CNN and RF forecasts after within-date percentile ranking.

    The final score is a ranking signal rather than a calibrated probability.
    """

    w_cnn: float = 0.5
    w_rf: float = 0.5

    def _weights(self) -> tuple[float, float]:
        if self.w_cnn < 0 or self.w_rf < 0:
            raise ValueError("weights must be non-negative")
        total = float(self.w_cnn + self.w_rf)
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        return float(self.w_cnn / total), float(self.w_rf / total)

    def predict(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "Date",
        p_cnn_col: str = "p_cnn",
        p_rf_col: str = "p_rf",
    ) -> pd.Series:
        required = [date_col, p_cnn_col, p_rf_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"missing required columns: {missing}")

        wc, wr = self._weights()
        work = df[[date_col, p_cnn_col, p_rf_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
        work[p_cnn_col] = pd.to_numeric(work[p_cnn_col], errors="coerce")
        work[p_rf_col] = pd.to_numeric(work[p_rf_col], errors="coerce")

        if work[[date_col, p_cnn_col, p_rf_col]].isna().any().any():
            raise ValueError("rank blending received missing or invalid Date/probability values")

        r_cnn = work.groupby(date_col)[p_cnn_col].rank(method="average", pct=True)
        r_rf = work.groupby(date_col)[p_rf_col].rank(method="average", pct=True)
        score = wc * r_cnn + wr * r_rf
        return pd.Series(score.to_numpy(float), index=df.index, name="up_prob")
