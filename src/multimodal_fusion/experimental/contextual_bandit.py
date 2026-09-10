"""
multimodal_fusion.experimental.contextual_bandit

Modern contextual bandit for choosing a date-level ensemble weight between CNN and RF.

Key design choices (used in the research pipeline)
--------------------------------------------
- Action is DATE-LEVEL: choose one w_t per Date, not per stock (keeps it distinct from MoE gating).
- Uses FULL-INFORMATION updates by default: after labels realize, we can score every candidate weight.
- Uses exponential forgetting for drift (forget_gamma).
- Uses tail reward (top/bottom quantiles) to align learning with long-best / short-worst portfolio construction.
- Adds performance-history context: recent realized CNN/RF losses (computed from past labels) as context.

Expected columns
----------------
Required:
  - Date
  - p_cnn
  - p_rf
Optional:
  - y (binary label) for fit()

Public API
----------
- fit(df)
- predict(df) -> Series 'up_prob'
- get_weights() -> DataFrame per-date weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ContextualBanditConfig:
    """Configuration for the contextual bandit ensemble."""
    # Action space.
    weight_grid: Optional[List[float]] = None

    # Learning / feedback timing.
    label_lag: int = 1

    # Reward definition.
    reward: str = "neg_logloss_tail"   # "neg_logloss" | "neg_logloss_tail" | "neg_brier" | "neg_brier_tail"
    tail_q: float = 0.10
    tail_min_n_per_date: int = 200

    # Context features (date-level).
    context_cols: Optional[List[str]] = None
    include_prob_context: bool = True

    # Performance-history context (date-level; uses only past labels).
    include_perf_context: bool = True
    perf_lookback_periods: int = 13
    perf_use_common_tail: bool = True   # define tail mask by mean(p_cnn,p_rf) so CNN/RF perf comparisons are fair

    # Thompson sampling parameters.
    prior_lambda: float = 50.0          # ridge prior precision (larger = more conservative)
    ts_noise: float = 0.03              # exploration scale (smaller = more deterministic)

    # Non-stationarity control.
    forget_gamma: float = 0.985         # exponential forgetting in (0,1]; 1.0 = no forgetting

    # Stability control.
    switch_penalty: float = 0.05        # penalize switching: subtract λ|w_t - w_{t-1}|

    # Update mode.
    full_information: bool = True       # update all arms each period (recommended here)

    # Numerics.
    eps: float = 1e-6
    random_state: int = 7
    verbose: bool = False


class ContextualBandit:
    """
    ContextualBandit

    A date-level contextual bandit that chooses a single mixing weight w_t each date.

    - Actions: discrete weights w in [0,1] (CNN weight).
    - Context: date-level vector x_t from current predictions + past realized CNN/RF performance.
    - Policy: Thompson sampling with a Bayesian linear model per action.
    - Updates: walk-forward with delayed labels, with full-information updates and exponential forgetting.
    """

    # Initialize config and fitted state.
    def __init__(self, **kwargs: Any) -> None:
        # Parse nested config if provided and support simple aliases.
        cfg_dict = kwargs.pop("config", None)
        if cfg_dict is not None and isinstance(cfg_dict, dict):
            kwargs = {**cfg_dict, **kwargs}
        if "seed" in kwargs and "random_state" not in kwargs:
            kwargs["random_state"] = kwargs.pop("seed")
        kwargs.pop("context_mode", None)  # legacy key; ignored

        self.cfg = ContextualBanditConfig(**kwargs)

        # Default weight grid if not provided.
        if self.cfg.weight_grid is None:
            self.cfg.weight_grid = [0.25, 0.5, 0.75]
        else:
            self.cfg.weight_grid = [float(x) for x in list(self.cfg.weight_grid)]

        # Fitted outputs.
        self.weights_by_date_: Optional[pd.DataFrame] = None
        self.global_w_cnn_: float = 0.5

        # Bandit state (per-arm Bayesian linear model).
        self._A: Dict[float, np.ndarray] = {}
        self._b: Dict[float, np.ndarray] = {}
        self._dim: int = 0

        # Cached contexts and chosen actions per date for delayed updates.
        self._x_by_date: Dict[pd.Timestamp, np.ndarray] = {}
        self._chosen_w_by_date: Dict[pd.Timestamp, float] = {}

        # Cached expert performance by date (computed from labels).
        self._perf_by_date: Optional[pd.DataFrame] = None

        # RNG.
        self._rng = np.random.RandomState(int(self.cfg.random_state))

    # Fit the bandit policy and store per-date chosen weights.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "ContextualBandit":
        # Basic cleaning and type coercion.
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        dfx = dfx.dropna(subset=[date_col, "p_cnn", "p_rf"]).copy()

        # Reset cached state for repeated fits.
        self._x_by_date = {}
        self._chosen_w_by_date = {}

        if len(dfx) == 0:
            self.weights_by_date_ = pd.DataFrame(columns=["w_cnn", "w_rf"])
            self.global_w_cnn_ = 0.5
            return self

        dates = pd.Index(dfx[date_col].unique()).sort_values()

        # If labels absent, fall back to fixed 0.5 weights.
        has_labels = (y_col in dfx.columns) and dfx[y_col].notna().any()
        if not has_labels:
            wdf = pd.DataFrame(index=dates)
            wdf["w_cnn"] = 0.5
            wdf["w_rf"] = 0.5
            self.weights_by_date_ = wdf
            self.global_w_cnn_ = 0.5
            return self

        # Precompute per-date expert performance from realized labels (used in context).
        if bool(self.cfg.include_perf_context):
            self._perf_by_date = self._compute_expert_perf_by_date(dfx, date_col=date_col, y_col=y_col)
        else:
            self._perf_by_date = None

        # Initialize bandit state after we know context dimensionality.
        self._init_state(dfx.loc[dfx[date_col] == dates[0]], date=pd.Timestamp(dates[0]))

        out: List[Tuple[pd.Timestamp, float]] = []
        w_prev = 0.5

        # Main walk-forward loop.
        for i, dt in enumerate(dates):
            dt = pd.Timestamp(dt)

            # Update first using realized feedback from the most recent available date.
            j = i - int(self.cfg.label_lag)
            if j >= 0:
                dt_upd = pd.Timestamp(dates[j])
                frame_u = dfx.loc[dfx[date_col] == dt_upd]
                if not frame_u.empty:
                    x_u = self._x_by_date.get(dt_upd)
                    if x_u is None:
                        x_u = self._compute_context(frame_u, date=dt_upd)
                        self._x_by_date[dt_upd] = x_u

                    if bool(self.cfg.full_information):
                        rewards = {w: self._reward_for_weight(frame_u, float(w), y_col=y_col) for w in self.cfg.weight_grid}
                        self._update_all(x_u, rewards)
                    else:
                        w_u = float(self._chosen_w_by_date.get(dt_upd, 0.5))
                        r_u = self._reward_for_weight(frame_u, w_u, y_col=y_col)
                        self._update_one(x_u, w_u, r_u)

            # Compute context for the current date after applying the delayed update.
            frame_t = dfx.loc[dfx[date_col] == dt]
            x_t = self._compute_context(frame_t, date=dt)
            self._x_by_date[dt] = x_t

            # Choose action for this date using the updated state.
            w_t = self._choose_action(x_t, w_prev=w_prev)
            self._chosen_w_by_date[dt] = float(w_t)
            out.append((dt, float(w_t)))
            w_prev = float(w_t)

        # Finalize weights table.
        wdf = pd.DataFrame(out, columns=[date_col, "w_cnn"]).set_index(date_col).sort_index()
        wdf["w_rf"] = 1.0 - wdf["w_cnn"]
        self.weights_by_date_ = wdf
        self.global_w_cnn_ = float(wdf["w_cnn"].iloc[-1]) if len(wdf) else 0.5
        return self

    # Predict blended probabilities using stored per-date weights.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        # Normalize dates and extract base predictions.
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, float(self.cfg.eps), 1.0 - float(self.cfg.eps))
        p2 = np.clip(p2, float(self.cfg.eps), 1.0 - float(self.cfg.eps))

        w = self._weights_for_rows(dfx, date_col=date_col).to_numpy(dtype=float)
        p = w * p1 + (1.0 - w) * p2
        p = np.clip(p, float(self.cfg.eps), 1.0 - float(self.cfg.eps))
        return pd.Series(p, index=df.index, name="up_prob")

    # Return per-date weights for diagnostics.
    def get_weights(self) -> pd.DataFrame:
        # Expose fitted weights table.
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame({"w_cnn": [self.global_w_cnn_], "w_rf": [1.0 - self.global_w_cnn_]})

    # Initialize per-arm Bayesian linear model state.
    def _init_state(self, example_frame: pd.DataFrame, *, date: pd.Timestamp) -> None:
        # Determine context dimensionality.
        x = self._compute_context(example_frame, date=date)
        self._dim = int(x.shape[0])

        # Initialize ridge prior for each arm.
        lam = float(self.cfg.prior_lambda)
        I = np.eye(self._dim, dtype=float)
        self._A = {float(w): lam * I.copy() for w in self.cfg.weight_grid}
        self._b = {float(w): np.zeros(self._dim, dtype=float) for w in self.cfg.weight_grid}

    # Compute date-level context vector x_t.
    def _compute_context(self, frame_t: pd.DataFrame, *, date: pd.Timestamp) -> np.ndarray:
        # Build robust context from cross-sectional predictions plus lagged expert performance.
        eps = float(self.cfg.eps)

        p1 = pd.to_numeric(frame_t["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(frame_t["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, eps, 1.0 - eps)
        p2 = np.clip(p2, eps, 1.0 - eps)

        feats: List[float] = [1.0]

        # Probability-derived context (regime proxies).
        if bool(self.cfg.include_prob_context):
            feats.append(float(np.mean(np.abs(p1 - p2))))
            feats.append(float(np.mean(np.abs(p1 - 0.5))))
            feats.append(float(np.mean(np.abs(p2 - 0.5))))
            feats.append(float(0.5 * (np.mean(p1) + np.mean(p2))))
            feats.append(float(np.std(p1)))
            feats.append(float(np.std(p2)))

        # Optional additional context columns aggregated by mean.
        for c in (self.cfg.context_cols or []):
            v = pd.to_numeric(frame_t.get(c, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            feats.append(float(v.mean()) if len(v) else 0.0)

        # Lagged performance context (uses only realized labels available at decision time).
        if bool(self.cfg.include_perf_context) and self._perf_by_date is not None:
            perf = self._rolling_perf_features(date)
            feats.extend(perf)

        x = np.asarray(feats, dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute rolling expert-performance features available at decision time.
    def _rolling_perf_features(self, date: pd.Timestamp) -> List[float]:
        # Use performance through the most recent realized date available at this decision.
        assert self._perf_by_date is not None
        dates = self._perf_by_date.index.sort_values()
        i = int(dates.searchsorted(pd.Timestamp(date), side="left"))
        latest_idx = i - int(self.cfg.label_lag)
        if latest_idx < 0:
            return [0.0, 0.0, 0.0]

        start_idx = max(0, latest_idx - int(self.cfg.perf_lookback_periods) + 1)
        win = self._perf_by_date.iloc[start_idx:latest_idx + 1]
        if len(win) == 0:
            return [0.0, 0.0, 0.0]

        cnn = float(win["r_cnn"].mean())
        rf = float(win["r_rf"].mean())
        return [cnn, rf, cnn - rf]

    # Compute per-date expert rewards from realized labels for context.
    def _compute_expert_perf_by_date(self, dfx: pd.DataFrame, *, date_col: str, y_col: str) -> pd.DataFrame:
        # Compute per-date negative logloss for CNN and RF, optionally on a common tail mask.
        eps = float(self.cfg.eps)
        q = float(self.cfg.tail_q)
        use_tail = "tail" in str(self.cfg.reward).lower()
        use_common = bool(self.cfg.perf_use_common_tail)

        rows = []
        for dt, g in dfx.groupby(date_col, sort=True):
            p1 = pd.to_numeric(g["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
            p2 = pd.to_numeric(g["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
            y = pd.to_numeric(g[y_col], errors="coerce").astype(float).to_numpy()

            p1 = np.clip(p1, eps, 1.0 - eps)
            p2 = np.clip(p2, eps, 1.0 - eps)

            mask = np.ones(len(g), dtype=bool)
            if use_tail:
                if use_common:
                    m = 0.5 * (p1 + p2)
                else:
                    m = 0.5 * (p1 + p2)
                lo = float(np.quantile(m, q))
                hi = float(np.quantile(m, 1.0 - q))
                mask = (m <= lo) | (m >= hi)

            if mask.sum() < max(int(self.cfg.tail_min_n_per_date), 10):
                mask[:] = True

            r_cnn = -float(np.mean(-(y[mask] * np.log(p1[mask]) + (1.0 - y[mask]) * np.log(1.0 - p1[mask]))))
            r_rf = -float(np.mean(-(y[mask] * np.log(p2[mask]) + (1.0 - y[mask]) * np.log(1.0 - p2[mask]))))
            rows.append((pd.Timestamp(dt), r_cnn, r_rf))

        out = pd.DataFrame(rows, columns=[date_col, "r_cnn", "r_rf"]).set_index(date_col).sort_index()
        return out

    # Choose an action using Thompson sampling with optional switch penalty.
    def _choose_action(self, x: np.ndarray, *, w_prev: float) -> float:
        # Sample theta per arm and pick argmax.
        best_w = float(self.cfg.weight_grid[0])
        best_s = -np.inf

        for w in self.cfg.weight_grid:
            ww = float(w)
            A = self._A[ww]
            b = self._b[ww]

            Ainv = np.linalg.inv(A)
            mu = Ainv @ b

            scale = float(self.cfg.ts_noise)
            z = self._rng.normal(size=self._dim)
            theta = mu + scale * (np.linalg.cholesky(Ainv) @ z)

            score = float(x @ theta)
            if float(self.cfg.switch_penalty) > 0:
                score -= float(self.cfg.switch_penalty) * abs(ww - float(w_prev))

            if (score > best_s) or (abs(score - best_s) < 1e-12 and abs(ww - w_prev) < abs(best_w - w_prev)):
                best_s = score
                best_w = ww

        return best_w

    # Update all arms using full-information rewards with exponential forgetting.
    def _update_all(self, x: np.ndarray, rewards: Dict[float, float]) -> None:
        # Apply discount then Bayesian linear regression update per arm.
        g = float(self.cfg.forget_gamma)
        for w, r in rewards.items():
            ww = float(w)
            if g < 1.0:
                self._A[ww] *= g
                self._b[ww] *= g
            self._A[ww] += np.outer(x, x)
            self._b[ww] += float(r) * x

    # Update only the chosen arm with exponential forgetting.
    def _update_one(self, x: np.ndarray, w: float, r: float) -> None:
        # Apply discount then update one arm.
        g = float(self.cfg.forget_gamma)
        ww = float(w)
        if g < 1.0:
            self._A[ww] *= g
            self._b[ww] *= g
        self._A[ww] += np.outer(x, x)
        self._b[ww] += float(r) * x

    # Compute reward for a given weight on a realized label cross-section.
    def _reward_for_weight(self, frame_u: pd.DataFrame, w: float, *, y_col: str) -> float:
        # Compute blended probability and score via chosen reward.
        eps = float(self.cfg.eps)
        ww = float(w)

        y = pd.to_numeric(frame_u[y_col], errors="coerce").astype(float).to_numpy()
        p1 = pd.to_numeric(frame_u["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(frame_u["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, eps, 1.0 - eps)
        p2 = np.clip(p2, eps, 1.0 - eps)

        p = ww * p1 + (1.0 - ww) * p2
        p = np.clip(p, eps, 1.0 - eps)

        mode = str(self.cfg.reward).lower().strip()
        if "tail" in mode:
            return self._reward_tail(p, y)
        if "brier" in mode:
            return self._reward_brier(p, y)
        return self._reward_logloss(p, y)

    # Compute negative log loss reward (higher is better).
    def _reward_logloss(self, p: np.ndarray, y: np.ndarray) -> float:
        # Reward = -mean log loss.
        eps = float(self.cfg.eps)
        p = np.clip(p.astype(float), eps, 1.0 - eps)
        y = y.astype(float)
        ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(-np.mean(ll))

    # Compute negative Brier score reward (higher is better).
    def _reward_brier(self, p: np.ndarray, y: np.ndarray) -> float:
        # Reward = -mean squared error.
        y = y.astype(float)
        p = p.astype(float)
        return float(-np.mean((p - y) ** 2))

    # Compute tail-only reward based on sorting by p.
    def _reward_tail(self, p: np.ndarray, y: np.ndarray) -> float:
        # Compute reward on top/bottom tail_q by p.
        q = float(self.cfg.tail_q)
        nmin = int(self.cfg.tail_min_n_per_date)
        n = int(len(p))

        if n < max(nmin, 10):
            return self._reward_logloss(p, y) if "logloss" in str(self.cfg.reward) else self._reward_brier(p, y)

        lo = float(np.quantile(p, q))
        hi = float(np.quantile(p, 1.0 - q))
        mask = (p <= lo) | (p >= hi)

        if int(mask.sum()) < max(nmin, 10):
            mask[:] = True

        pt = p[mask]
        yt = y[mask]
        if "brier" in str(self.cfg.reward).lower():
            return self._reward_brier(pt, yt)
        return self._reward_logloss(pt, yt)

    # Map per-date weights to each row with forward-fill.
    def _weights_for_rows(self, dfx: pd.DataFrame, *, date_col: str) -> pd.Series:
        # Align row dates to fitted weight series.
        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            return pd.Series(self.global_w_cnn_, index=dfx.index, dtype=float)

        dd = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        w_series = self.weights_by_date_["w_cnn"].copy().sort_index()
        w = dd.map(w_series).astype(float).fillna(self.global_w_cnn_)
        return pd.Series(w.to_numpy(), index=dfx.index, dtype=float)