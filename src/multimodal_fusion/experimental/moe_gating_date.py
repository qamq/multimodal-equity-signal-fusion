"""
multimodal_fusion.experimental.moe_gating_date

Date-level Mixture-of-Experts (MoE) gating for blending CNN and RF probabilities.

Core output
-----------
For each Date t, produce:
    w_t in [0,1]
    up_prob_{i,t} = w_t * p_cnn_{i,t} + (1 - w_t) * p_rf_{i,t}

Gate model
----------
The gate is a logistic model:
    w_t = sigmoid(x_t theta)

where x_t is a date-level context vector built from:
  - optional aggregated context features (feature_cols)
  - probability-derived date features (if include_prob_context):
        mean(p_cnn), mean(p_rf), mean(logit_cnn), mean(logit_rf),
        mean(abs(p_cnn - p_rf)), std(p_cnn), std(p_rf)

Training objective
------------------
Update theta online using realized outcomes from the most recent available date:
    p_i = w_t * p_cnn_i + (1-w_t) * p_rf_i
    loss_t = mean( -[ y_i log p_i + (1-y_i) log(1-p_i) ] ) + (l2/2) * ||theta[1:]||^2

Walk-forward online update
--------------------------
For each date t:
  - first update theta using realized labels from date t - label_lag
  - then compute x_t and w_t
  - then use w_t to predict on date t

With label_lag=1, week t uses all realized outcomes through week t-1.

Optional tail alignment
-----------------------
If tail_only=True, updates are computed only on top/bottom tail_q by the mean expert probability
for the realized update date. This aligns learning with long-best / short-worst portfolio construction.

Notes
-----
- This class is the general weekly/date-level gate.
- Keep the existing MoEGating class as the row-level stock-week gate.
- Output remains a 5-day probability 'up_prob', compatible with PortfolioManager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MoEGatingDate:
    """MoEGatingDate learns one date-level CNN weight per week via an online logistic gate."""

    # Feature controls.
    feature_cols: Optional[List[str]] = None
    include_prob_context: bool = True

    # Optimizer controls.
    lr: float = 0.05
    l2: float = 1e-3
    forget_gamma: float = 1.0
    grad_clip: float = 5.0
    eps: float = 1e-6

    # Walk-forward controls.
    label_lag: int = 1

    # Optional tail-only updates.
    tail_only: bool = False
    tail_q: float = 0.10
    tail_min_n_per_date: int = 200

    # Reproducibility.
    random_state: int = 7

    # Fitted state.
    _theta: Optional[np.ndarray] = field(default=None, init=False)
    _feat_names: Optional[List[str]] = field(default=None, init=False)
    _x_by_date: Dict[pd.Timestamp, np.ndarray] = field(default_factory=dict, init=False)

    weights_by_date_: Optional[pd.DataFrame] = field(default=None, init=False)
    global_w_cnn_: float = field(default=0.5, init=False)

    # Fit date-level online gate weights.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "MoEGatingDate":
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        dfx = dfx.dropna(subset=[date_col, "p_cnn", "p_rf", y_col]).copy()

        dfx["p_cnn"] = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)
        dfx["p_rf"] = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)
        dfx[y_col] = pd.to_numeric(dfx[y_col], errors="coerce").astype(float)

        self._x_by_date = {}

        if len(dfx) == 0:
            self._theta = None
            self._feat_names = None
            self.weights_by_date_ = pd.DataFrame(columns=["w_cnn", "w_rf"])
            self.global_w_cnn_ = 0.5
            return self

        dates = pd.Index(dfx[date_col].unique()).sort_values()
        first_frame = dfx.loc[dfx[date_col] == pd.Timestamp(dates[0])]
        x0, feat_names = self._build_x(first_frame)
        self._feat_names = feat_names
        self._theta = np.zeros(len(x0), dtype=float)

        out: List[Tuple[pd.Timestamp, float]] = []

        for i, dt in enumerate(dates):
            dt = pd.Timestamp(dt)

            # Update first using the latest realized date that should be available.
            j = i - int(self.label_lag)
            if j >= 0:
                dt_upd = pd.Timestamp(dates[j])
                frame_u = dfx.loc[dfx[date_col] == dt_upd]
                if not frame_u.empty:
                    x_u = self._x_by_date.get(dt_upd)
                    if x_u is None:
                        x_u, _ = self._build_x(frame_u)
                        self._x_by_date[dt_upd] = x_u
                    self._update_from_frame(frame_u, x_u, y_col=y_col)

            # Compute current-date context and weight after the delayed update.
            frame_t = dfx.loc[dfx[date_col] == dt]
            x_t, _ = self._build_x(frame_t)
            self._x_by_date[dt] = x_t

            w_t = self._weight_from_x(x_t)
            out.append((dt, float(w_t)))

        wdf = pd.DataFrame(out, columns=[date_col, "w_cnn"]).set_index(date_col).sort_index()
        wdf["w_rf"] = 1.0 - wdf["w_cnn"]

        self.weights_by_date_ = wdf
        self.global_w_cnn_ = float(wdf["w_cnn"].iloc[-1]) if len(wdf) else 0.5
        return self

    # Predict ensemble probabilities using stored date-level weights.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, self.eps, 1.0 - self.eps)
        p2 = np.clip(p2, self.eps, 1.0 - self.eps)

        w = self.predict_weights(dfx, date_col=date_col).to_numpy(dtype=float)
        p = w * p1 + (1.0 - w) * p2
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return pd.Series(p, index=df.index, name="up_prob")

    # Predict date-level gate weights for each row.
    def predict_weights(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()

        if date_col not in dfx.columns:
            if self._theta is None:
                return pd.Series(0.5, index=df.index, name="w_cnn")
            x, _ = self._build_x(dfx)
            w = self._weight_from_x(x)
            return pd.Series(float(w), index=df.index, name="w_cnn")

        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        out = pd.Series(index=dfx.index, dtype=float, name="w_cnn")

        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            # Fall back to final theta for unseen dates.
            for dt, g in dfx.groupby(date_col, sort=True):
                if self._theta is None:
                    out.loc[g.index] = 0.5
                else:
                    x_t, _ = self._build_x(g)
                    out.loc[g.index] = float(self._weight_from_x(x_t))
            return out.fillna(0.5)

        known = self.weights_by_date_["w_cnn"]
        for dt, g in dfx.groupby(date_col, sort=True):
            dt = pd.Timestamp(dt)
            if dt in known.index:
                out.loc[g.index] = float(known.loc[dt])
            elif self._theta is not None:
                x_t, _ = self._build_x(g)
                out.loc[g.index] = float(self._weight_from_x(x_t))
            else:
                out.loc[g.index] = 0.5

        return out.fillna(self.global_w_cnn_)

    # Return a copy of the learned date-level weights.
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame({"w_cnn": [self.global_w_cnn_], "w_rf": [1.0 - self.global_w_cnn_]})

    # Build the date-level context vector x_t.
    def _build_x(self, frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        feats: Dict[str, float] = {}

        p1 = pd.to_numeric(frame["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(frame["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, self.eps, 1.0 - self.eps)
        p2 = np.clip(p2, self.eps, 1.0 - self.eps)

        feats["intercept"] = 1.0

        for c in (self.feature_cols or []):
            v = pd.to_numeric(frame.get(c, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            feats[c] = float(v.mean()) if len(v) else 0.0

        if self.include_prob_context:
            feats["mean_p_cnn"] = float(np.mean(p1))
            feats["mean_p_rf"] = float(np.mean(p2))
            feats["mean_logit_cnn"] = float(np.mean(self._logit(p1)))
            feats["mean_logit_rf"] = float(np.mean(self._logit(p2)))
            feats["mean_abs_diff"] = float(np.mean(np.abs(p1 - p2)))
            feats["std_p_cnn"] = float(np.std(p1))
            feats["std_p_rf"] = float(np.std(p2))   

        names = list(feats.keys())
        x = np.asarray([feats[k] for k in names], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, names

    # Convert a context vector into the current CNN weight.
    def _weight_from_x(self, x: np.ndarray) -> float:
        if self._theta is None:
            return 0.5
        z = float(np.dot(x, self._theta))
        return float(self._sigmoid(z))

    # Apply one online gradient update from a realized date.
    def _update_from_frame(self, frame_u: pd.DataFrame, x_u: np.ndarray, *, y_col: str) -> None:
        if self._theta is None:
            return

        p1 = pd.to_numeric(frame_u["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(frame_u["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        y = pd.to_numeric(frame_u[y_col], errors="coerce").astype(float).to_numpy()

        p1 = np.clip(p1, self.eps, 1.0 - self.eps)
        p2 = np.clip(p2, self.eps, 1.0 - self.eps)

        mask = np.ones(len(frame_u), dtype=bool)
        if self.tail_only:
            base = 0.5 * (p1 + p2)
            lo = float(np.quantile(base, float(self.tail_q)))
            hi = float(np.quantile(base, 1.0 - float(self.tail_q)))
            mask = (base <= lo) | (base >= hi)
            if int(mask.sum()) < max(int(self.tail_min_n_per_date), 10):
                mask[:] = True

        if int(mask.sum()) < 5:
            return

        p1m = p1[mask]
        p2m = p2[mask]
        ym = y[mask]

        w = self._weight_from_x(x_u)
        p = w * p1m + (1.0 - w) * p2m
        p = np.clip(p, self.eps, 1.0 - self.eps)

        dL_dp = (p - ym) / np.maximum(p * (1.0 - p), 1e-12)
        dL_dw = float(np.mean(dL_dp * (p1m - p2m)))
        dL_dz = dL_dw * w * (1.0 - w)

        grad = dL_dz * x_u

        if float(self.l2) > 0:
            reg = np.zeros_like(self._theta)
            reg[1:] = float(self.l2) * self._theta[1:]
            grad = grad + reg

        if float(self.forget_gamma) < 1.0:
            self._theta *= float(self.forget_gamma)

        gnorm = float(np.linalg.norm(grad))
        clip = float(self.grad_clip)
        if clip > 0 and gnorm > clip:
            grad = (clip / max(gnorm, 1e-12)) * grad

        self._theta = self._theta - float(self.lr) * grad
        self._theta = np.nan_to_num(self._theta, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute the logistic sigmoid safely.
    def _sigmoid(self, z) -> np.ndarray:
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    # Compute the logit safely.
    def _logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p))