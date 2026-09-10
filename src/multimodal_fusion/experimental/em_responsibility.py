"""
multimodal_fusion.experimental.em_responsibility

EM responsibility weighting for two experts (CNN vs RF)

Model (by-date)
---------------
For each rebalance date t, we compute a single mixture weight π_t (the CNN weight) using only
historical data available through t - label_lag:

    P(y=1 | t) = π_t * Bernoulli(y; p_cnn) + (1-π_t) * Bernoulli(y; p_rf)

E-step:
    r_i = P(z_i = CNN | y_i)  (responsibility)

M-step (with optional Beta prior smoothing):
    π <- (a0 - 1 + sum_i r_i) / (a0 + b0 - 2 + N)

Walk-forward fitting
--------------------
By default, this class runs expanding-window batch EM:
  - for date t, fit EM on all realized observations through t - label_lag
  - then use the fitted π_t to predict on date t

If lookback_periods is set to a positive integer, this becomes rolling-window EM instead.

Modern upgrades included
------------------------
(1) Walk-forward by-date EM:
    π_t is estimated using only historical labels available at time t.

(2) Optional temperature calibration (per expert, per date):
    Before EM, transform each expert probability p by:
        p_tilde = sigmoid(logit(p) / T)
    and choose T (separately for CNN and RF) by minimizing log loss on the same historical window.

(3) Optional Beta prior smoothing for π:
    Prevents EM from collapsing to 0/1 in noisy windows; can also add dynamic smoothing
    centered at the previous π.

Optional (tail alignment to long/short)
---------------------------------------
If use_tail=True, the EM fitting window is restricted to cross-sectional tails per date
(top and bottom tail_q by mean probability). This aligns weight learning with your
“long best / short worst” portfolio construction while staying by-date.

Outputs
-------
- predict(df) returns 'up_prob' (5-day probability).
- get_weights() returns per-date weights (and calibration temps if enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class EMResponsibilityConfig:
    """Configuration for walk-forward EM responsibility weights."""
    # Core walk-forward controls.
    by_date: bool = True
    lookback_periods: Optional[int] = None   # None = expanding window; int = rolling window
    label_lag: int = 1

    # EM controls.
    max_iters: int = 50
    tol: float = 1e-6
    eps: float = 1e-6
    weight_floor: float = 1e-3

    # Temperature calibration controls.
    calibrate: bool = True
    temp_grid: Optional[List[float]] = None
    temp_default: float = 1.0

    # Beta prior smoothing for π.
    beta_a: float = 2.0
    beta_b: float = 2.0
    prior_strength: float = 0.0  # if >0, adds dynamic prior centered at the previous π

    # Optional tail-only EM fitting.
    use_tail: bool = False
    tail_q: float = 0.10
    tail_min_n_per_date: int = 200


class EMResponsibility:
    """
    EMResponsibility

    Requires:
      - p_cnn
      - p_rf
      - y in {0,1} for fit()

    By default:
      - learns π_t by date with expanding-window batch EM (leakage-safe)
      - optionally calibrates each expert with temperature scaling inside each window
      - optionally smooths π updates with a Beta prior
    """

    # Initialize EM config and fitted state.
    def __init__(self, config: Optional[EMResponsibilityConfig] = None) -> None:
        self.cfg = config if config is not None else EMResponsibilityConfig()
        self.weights_by_date_: Optional[pd.DataFrame] = None
        self.global_w_cnn_: float = 0.5

    # Fit mixture weight(s) with walk-forward batch EM.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "EMResponsibility":
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        dfx = dfx.dropna(subset=[date_col, "p_cnn", "p_rf", y_col]).copy()

        if len(dfx) == 0:
            self.weights_by_date_ = pd.DataFrame(columns=["w_cnn", "w_rf", "temp_cnn", "temp_rf"])
            self.global_w_cnn_ = 0.5
            return self

        if not self.cfg.by_date:
            w, t1, t2, _ = self._fit_one_window(dfx, y_col=y_col, pi_init=0.5)
            self.global_w_cnn_ = float(w)
            self.weights_by_date_ = pd.DataFrame(
                {"w_cnn": [float(w)], "w_rf": [1.0 - float(w)], "temp_cnn": [float(t1)], "temp_rf": [float(t2)]}
            )
            return self

        dates = pd.Index(dfx[date_col].unique()).sort_values()
        out: List[Tuple[pd.Timestamp, float, float, float]] = []

        pi_prev = 0.5
        tcnn_prev = float(self.cfg.temp_default)
        trf_prev = float(self.cfg.temp_default)

        for i, dt in enumerate(dates):
            j = i - int(self.cfg.label_lag)

            # No historical labels available yet.
            if j < 0:
                out.append((pd.Timestamp(dt), float(pi_prev), float(tcnn_prev), float(trf_prev)))
                continue

            # Build the historical sample available at decision time.
            d0, d1 = self._window_bounds(dates, end_idx=j)
            win = dfx[(dfx[date_col] >= d0) & (dfx[date_col] <= d1)].copy()

            # If the window is empty, carry the previous state forward.
            if len(win) == 0:
                out.append((pd.Timestamp(dt), float(pi_prev), float(tcnn_prev), float(trf_prev)))
                continue

            # Re-fit batch EM from scratch on the available historical sample.
            pi_prev, tcnn_prev, trf_prev, _ = self._fit_one_window(win, y_col=y_col, pi_init=pi_prev)
            out.append((pd.Timestamp(dt), float(pi_prev), float(tcnn_prev), float(trf_prev)))

        wdf = pd.DataFrame(out, columns=[date_col, "w_cnn", "temp_cnn", "temp_rf"]).set_index(date_col)
        wdf["w_rf"] = 1.0 - wdf["w_cnn"]

        self.weights_by_date_ = wdf[["w_cnn", "w_rf", "temp_cnn", "temp_rf"]].copy()
        self.global_w_cnn_ = float(self.weights_by_date_["w_cnn"].iloc[-1])
        return self

    # Predict ensemble probabilities using stored per-date π and temperatures.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        w = self._weights_for_rows(dfx, date_col=date_col)
        t_cnn, t_rf = self._temps_for_rows(dfx, date_col=date_col)

        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()

        p1 = self._temp_scale(p1, t_cnn.to_numpy(dtype=float))
        p2 = self._temp_scale(p2, t_rf.to_numpy(dtype=float))

        p = w.to_numpy(dtype=float) * p1 + (1.0 - w.to_numpy(dtype=float)) * p2
        p = np.clip(p, float(self.cfg.eps), 1.0 - float(self.cfg.eps))
        return pd.Series(p, index=df.index, name="up_prob")

    # Return per-date weights (and temps).
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame(
            {
                "w_cnn": [self.global_w_cnn_],
                "w_rf": [1.0 - self.global_w_cnn_],
                "temp_cnn": [float(self.cfg.temp_default)],
                "temp_rf": [float(self.cfg.temp_default)],
            }
        )

    # Compute the historical window bounds available at the current decision time.
    def _window_bounds(self, dates: pd.Index, *, end_idx: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if self.cfg.lookback_periods is None:
            start_idx = 0
        else:
            lb = int(self.cfg.lookback_periods)
            start_idx = max(0, end_idx - lb + 1)
        return pd.Timestamp(dates[start_idx]), pd.Timestamp(dates[end_idx])

    # Fit one historical window: tail filter (optional) -> temp cal -> batch EM.
    def _fit_one_window(self, win: pd.DataFrame, *, y_col: str, pi_init: float) -> Tuple[float, float, float, float]:
        eps = float(self.cfg.eps)

        w = win.copy()
        w["p_cnn"] = np.clip(pd.to_numeric(w["p_cnn"], errors="coerce").to_numpy(dtype=float), eps, 1.0 - eps)
        w["p_rf"] = np.clip(pd.to_numeric(w["p_rf"], errors="coerce").to_numpy(dtype=float), eps, 1.0 - eps)
        y = pd.to_numeric(w[y_col], errors="coerce").to_numpy(dtype=float)

        # Apply tail-only filtering to align with long/short.
        if bool(self.cfg.use_tail):
            w = self._tail_filter(w, y_col=y_col)
            if len(w) == 0:
                return float(pi_init), float(self.cfg.temp_default), float(self.cfg.temp_default), float("nan")
            y = pd.to_numeric(w[y_col], errors="coerce").to_numpy(dtype=float)

        p1 = w["p_cnn"].to_numpy(dtype=float)
        p2 = w["p_rf"].to_numpy(dtype=float)

        # Calibrate temperatures on this historical window.
        if bool(self.cfg.calibrate):
            t1 = self._fit_temp(p1, y)
            t2 = self._fit_temp(p2, y)
        else:
            t1 = float(self.cfg.temp_default)
            t2 = float(self.cfg.temp_default)

        p1c = self._temp_scale(p1, t1)
        p2c = self._temp_scale(p2, t2)

        # Run genuine batch EM on the historical window.
        pi, ll = self._em_fit(p1c, p2c, y, pi_init)
        return float(pi), float(t1), float(t2), float(ll)

    # Tail filter: keep top/bottom tail_q by mean probability per date.
    def _tail_filter(self, win: pd.DataFrame, *, y_col: str, date_col: str = "Date") -> pd.DataFrame:
        q = float(self.cfg.tail_q)
        kmin = int(self.cfg.tail_min_n_per_date)

        if date_col not in win.columns:
            return win

        w = win.copy()
        mean_p = 0.5 * (pd.to_numeric(w["p_cnn"], errors="coerce") + pd.to_numeric(w["p_rf"], errors="coerce"))
        w["_mean_p"] = mean_p.astype(float)

        parts = []
        for _, g in w.groupby(date_col, sort=False):
            g = g.dropna(subset=["_mean_p", y_col])
            if len(g) < kmin:
                continue
            lo = float(np.quantile(g["_mean_p"].to_numpy(dtype=float), q))
            hi = float(np.quantile(g["_mean_p"].to_numpy(dtype=float), 1.0 - q))
            parts.append(g[(g["_mean_p"] <= lo) | (g["_mean_p"] >= hi)])

        if not parts:
            return w.iloc[0:0].copy()

        out = pd.concat(parts, axis=0).drop(columns=["_mean_p"], errors="ignore")
        return out

    # Temperature scaling: p -> sigmoid(logit(p)/T), with scalar or per-row T.
    def _temp_scale(self, p: np.ndarray, T) -> np.ndarray:
        eps = float(self.cfg.eps)
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)

        Tarr = np.asarray(T, dtype=float)
        if Tarr.ndim == 0:
            Tarr = np.full_like(p, float(Tarr))
        if Tarr.shape[0] != p.shape[0]:
            raise ValueError(f"_temp_scale: T has shape {Tarr.shape} but p has shape {p.shape}")

        z = np.log(p / (1.0 - p))
        z = z / np.maximum(Tarr, 1e-6)
        z = np.clip(z, -50.0, 50.0)

        out = 1.0 / (1.0 + np.exp(-z))
        return np.clip(out, eps, 1.0 - eps)

    # Fit temperature by minimizing log loss on a small grid.
    def _fit_temp(self, p: np.ndarray, y: np.ndarray) -> float:
        grid = self.cfg.temp_grid
        if grid is None:
            grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

        best_T = float(self.cfg.temp_default)
        best_ll = float("inf")

        for T in grid:
            pc = self._temp_scale(p, float(T))
            ll = float(np.mean(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))))
            if ll < best_ll:
                best_ll = ll
                best_T = float(T)

        return best_T

    # Run batch EM for two experts with optional Beta prior smoothing.
    def _em_fit(self, p1: np.ndarray, p2: np.ndarray, y: np.ndarray, pi_init: float) -> Tuple[float, float]:
        eps = float(self.cfg.eps)
        wf = float(self.cfg.weight_floor)

        p1 = np.clip(p1.astype(float), eps, 1.0 - eps)
        p2 = np.clip(p2.astype(float), eps, 1.0 - eps)
        y = y.astype(float)

        logL1 = y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1)
        logL2 = y * np.log(p2) + (1.0 - y) * np.log(1.0 - p2)

        pi = float(np.clip(pi_init, wf, 1.0 - wf))
        ll = -np.inf
        prev = pi

        # Build the optional Beta prior for the M-step.
        a0 = float(self.cfg.beta_a)
        b0 = float(self.cfg.beta_b)
        k = float(self.cfg.prior_strength)
        if k > 0:
            a0 = a0 + k * pi
            b0 = b0 + k * (1.0 - pi)

        for _ in range(int(self.cfg.max_iters)):
            # E-step: compute responsibilities.
            a = np.log(pi) + logL1
            b = np.log(1.0 - pi) + logL2
            m = np.maximum(a, b)
            den = m + np.log(np.exp(a - m) + np.exp(b - m))
            r = np.exp(a - den)

            # M-step: update mixture weight.
            num = (a0 - 1.0) + float(np.sum(r))
            deno = (a0 + b0 - 2.0) + float(len(r))
            pi = float(np.clip(num / max(deno, 1e-12), wf, 1.0 - wf))

            ll = float(np.sum(den))
            if abs(pi - prev) < float(self.cfg.tol):
                break
            prev = pi

        return pi, ll

    # Map per-date weights to rows.
    def _weights_for_rows(self, dfx: pd.DataFrame, *, date_col: str) -> pd.Series:
        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            return pd.Series(self.global_w_cnn_, index=dfx.index, dtype=float)

        dd = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        w = dd.map(self.weights_by_date_["w_cnn"]).astype(float).fillna(self.global_w_cnn_)
        return pd.Series(w.to_numpy(), index=dfx.index, dtype=float)

    # Map per-date temperatures to rows.
    def _temps_for_rows(self, dfx: pd.DataFrame, *, date_col: str) -> Tuple[pd.Series, pd.Series]:
        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            t = float(self.cfg.temp_default)
            return pd.Series(t, index=dfx.index, dtype=float), pd.Series(t, index=dfx.index, dtype=float)

        dd = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        t1 = dd.map(self.weights_by_date_.get("temp_cnn", pd.Series(dtype=float))).astype(float)
        t2 = dd.map(self.weights_by_date_.get("temp_rf", pd.Series(dtype=float))).astype(float)

        tdef = float(self.cfg.temp_default)
        t1 = t1.fillna(tdef)
        t2 = t2.fillna(tdef)

        return pd.Series(t1.to_numpy(), index=dfx.index, dtype=float), pd.Series(t2.to_numpy(), index=dfx.index, dtype=float)