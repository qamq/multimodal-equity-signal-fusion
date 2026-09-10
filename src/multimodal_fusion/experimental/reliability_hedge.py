"""
multimodal_fusion.experimental.reliability_hedge

Hedge (multiplicative weights) ensembling.
Maintain weights w_cnn, w_rf and update after each date using losses:

    w_k <- w_k * exp(-eta * loss_k)
    normalize

This is an online version of "trust the model that has been more reliable recently".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ReliabilityHedge:
    """
    ReliabilityHedge

    Requires y in {0,1} for computing losses.

    Walk-forward:
      - Choose weights for date t.
      - Update weights using outcomes from date t after label_lag periods.
    """

    eta: float = 5.0
    label_lag: int = 1
    reward: str = "neg_logloss"  # "neg_logloss" | "neg_brier"
    eps: float = 1e-6

    weights_by_date_: Optional[pd.DataFrame] = None
    global_w_cnn_: float = 0.5

    # Fit per-date hedge weights.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "ReliabilityHedge":
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        dfx = dfx.dropna(subset=[date_col, "p_cnn", "p_rf", y_col]).copy()

        dates = pd.Index(dfx[date_col].unique()).sort_values()
        if len(dates) == 0:
            self.weights_by_date_ = pd.DataFrame(columns=["w_cnn", "w_rf"])
            self.global_w_cnn_ = 0.5
            return self

        w_cnn, w_rf = 0.5, 0.5
        out = []

        for i, dt in enumerate(dates):
            j = i - int(self.label_lag)

            # Update first using the latest label that should be available by date dt
            if j >= 0:
                dt_upd = dates[j]
                sub = dfx[dfx[date_col] == dt_upd]
                if len(sub) > 0:
                    lc = self._loss(sub["p_cnn"].to_numpy(float), sub[y_col].to_numpy(float))
                    lr = self._loss(sub["p_rf"].to_numpy(float), sub[y_col].to_numpy(float))

                    w_cnn = w_cnn * float(np.exp(-self.eta * lc))
                    w_rf = w_rf * float(np.exp(-self.eta * lr))

                    s = w_cnn + w_rf
                    if s <= 0 or not np.isfinite(s):
                        w_cnn, w_rf = 0.5, 0.5
                    else:
                        w_cnn, w_rf = w_cnn / s, w_rf / s

            # Store weights used for current date
            out.append((dt, w_cnn))

        wdf = pd.DataFrame(out, columns=[date_col, "w_cnn"]).set_index(date_col)
        wdf["w_rf"] = 1.0 - wdf["w_cnn"]
        self.weights_by_date_ = wdf
        self.global_w_cnn_ = float(wdf["w_cnn"].iloc[-1])
        return self

    # Predict ensemble probabilities.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        w = self._weights_for_rows(dfx, date_col=date_col)
        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5)
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5)

        p = w * p1 + (1.0 - w) * p2
        p = p.clip(self.eps, 1.0 - self.eps)
        return pd.Series(p.to_numpy(), index=df.index, name="up_prob")

    # Return weights table.
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame({"w_cnn": [self.global_w_cnn_], "w_rf": [1.0 - self.global_w_cnn_]})

    # Compute proper loss.
    def _loss(self, p: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(p.astype(float), self.eps, 1.0 - self.eps)
        y = y.astype(float)

        if self.reward == "neg_brier":
            return float(np.mean((y - p) ** 2))

        return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

    # Map per-date weights to rows.
    def _weights_for_rows(self, dfx: pd.DataFrame, *, date_col: str) -> pd.Series:
        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            return pd.Series(self.global_w_cnn_, index=dfx.index, dtype=float)

        dd = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        w = dd.map(self.weights_by_date_["w_cnn"]).astype(float).fillna(self.global_w_cnn_)
        return pd.Series(w.to_numpy(), index=dfx.index, dtype=float)