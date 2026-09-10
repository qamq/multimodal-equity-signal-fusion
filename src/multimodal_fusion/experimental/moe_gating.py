
"""
multimodal_fusion.experimental.moe_gating

Mixture-of-Experts (MoE) gating: per-row conditional blending of CNN and RF probabilities.

Core output
-----------
For each row (StockID i, Date t), produce:
    w_{i,t} in [0,1]
    up_prob_{i,t} = w_{i,t} * p_cnn_{i,t} + (1 - w_{i,t}) * p_rf_{i,t}

Gate model
----------
The gate is a logistic model:
    w = sigmoid(X theta)

where X is built from:
  - optional context features (feature_cols)
  - probability-derived features (if include_prob_features):
        p_cnn, p_rf, logit(p_cnn), logit(p_rf), abs(p_cnn - p_rf)

Training objective
------------------
Minimize cross-entropy of the mixture probability:
    p = w * p_cnn + (1-w) * p_rf
    loss = mean( -[ y log p + (1-y) log(1-p) ] ) + (l2/2) * ||theta[1:]||^2

Walk-forward refit (recommended; leakage-safe)
----------------------------------------------
If walk_forward=True:
  - Refit theta quarterly/monthly/yearly on a rolling window (default 2 years),
  - For a period starting at t0, training uses only dates <= (t0 - label_lag steps),
  - Warm-start each refit from the previous period's theta if dimensions match,
  - If a period has insufficient training rows, carry forward the previous model.

Compute controls
----------------
Refitting can be heavy on millions of rows. Use:
  - sample_per_date: cap sampled rows per Date in the training window (default 2000)
  - max_train_rows: optional global cap after per-date sampling

Saved diagnostics
-----------------
After fit(...), the class stores:
  - row_weights_: per-row weights for the fitted dataframe
  - weights_by_date_: per-date summary statistics of those row weights

Notes on integration
--------------------
- fit(...) requires y (binary label) and Date if walk_forward=True.
- predict(...) requires Date if walk_forward=True.
- Output remains a 5-day probability 'up_prob', compatible with PortfolioManager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class MoEGating:
    """
    MoEGating

    Required columns (always):
      - p_cnn
      - p_rf

    Required for training:
      - y  (binary 0/1)

    Required for walk-forward:
      - Date (or date_col argument)
    """

    # Feature controls.
    feature_cols: Optional[List[str]] = None
    include_prob_features: bool = True
    standardize: bool = True

    # Optimizer controls.
    lr: float = 0.05
    l2: float = 1e-3
    max_iters: int = 200
    tol: float = 1e-7
    eps: float = 1e-6

    # Walk-forward controls.
    walk_forward: bool = True
    refit_freq: str = "quarter"      # "quarter" | "month" | "year"
    train_window_years: int = 2
    label_lag: int = 1
    warm_start: bool = True
    min_train_rows: int = 50_000

    # Sampling controls.
    sample_per_date: Optional[int] = 2000
    max_train_rows: Optional[int] = None
    random_state: int = 7

    # Global model (used if walk_forward=False).
    _theta: Optional[np.ndarray] = field(default=None, init=False)
    _mu: Optional[np.ndarray] = field(default=None, init=False)
    _sd: Optional[np.ndarray] = field(default=None, init=False)
    _feat_names: Optional[List[str]] = field(default=None, init=False)

    # Period models (used if walk_forward=True): period -> model dict.
    _models: Dict[pd.Period, Dict[str, Any]] = field(default_factory=dict, init=False)
    _periods_sorted: List[pd.Period] = field(default_factory=list, init=False)

    # Saved diagnostics.
    row_weights_: Optional[pd.DataFrame] = field(default=None, init=False)
    weights_by_date_: Optional[pd.DataFrame] = field(default=None, init=False)
    global_w_cnn_: float = field(default=0.5, init=False)

    # Fit gating parameters (global or walk-forward).
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "MoEGating":
        # Validate required columns.
        if y_col not in df.columns:
            raise KeyError(f"MoEGating.fit requires y_col='{y_col}' in df.")
        if self.walk_forward and date_col not in df.columns:
            raise KeyError(f"MoEGating.fit with walk_forward=True requires date_col='{date_col}' in df.")

        dfx = df.copy()

        # Coerce Date if present.
        if date_col in dfx.columns:
            dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        # Clean required columns.
        req = ["p_cnn", "p_rf", y_col] + ([date_col] if self.walk_forward else [])
        dfx = dfx.dropna(subset=req).copy()

        # Clip probabilities and coerce y.
        dfx["p_cnn"] = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)
        dfx["p_rf"] = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)
        dfx[y_col] = pd.to_numeric(dfx[y_col], errors="coerce").astype(float)

        if len(dfx) == 0:
            self._theta = None
            self._models.clear()
            self._periods_sorted = []
            self.row_weights_ = pd.DataFrame(columns=["row_index", date_col, "w_cnn", "w_rf"])
            self.weights_by_date_ = pd.DataFrame(columns=["mean_w_cnn", "std_w_cnn", "min_w_cnn", "max_w_cnn", "n_rows"])
            self.global_w_cnn_ = 0.5
            return self

        if not self.walk_forward:
            self._fit_global(dfx, y_col=y_col)
        else:
            self._fit_walk_forward(dfx, date_col=date_col, y_col=y_col)

        self._cache_fitted_weights(dfx, date_col=date_col)
        return self

    # Predict ensemble probabilities.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        # Validate.
        if self.walk_forward and date_col not in df.columns:
            raise KeyError(f"MoEGating.predict with walk_forward=True requires date_col='{date_col}' in df.")

        dfx = df.copy()

        # Base expert probabilities.
        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, self.eps, 1.0 - self.eps)
        p2 = np.clip(p2, self.eps, 1.0 - self.eps)

        # Global mode.
        if not self.walk_forward:
            if self._theta is None or self._feat_names is None:
                w = np.full(len(dfx), 0.5, dtype=float)
            else:
                X = self._build_X(dfx, fit_mode=False, mu=self._mu, sd=self._sd, feat_names=self._feat_names)
                w = self._sigmoid(X @ self._theta)
            p = w * p1 + (1.0 - w) * p2
            p = np.clip(p, self.eps, 1.0 - self.eps)
            return pd.Series(p, index=df.index, name="up_prob")

        # Walk-forward mode.
        if not self._periods_sorted:
            w = np.full(len(dfx), 0.5, dtype=float)
            p = w * p1 + (1.0 - w) * p2
            p = np.clip(p, self.eps, 1.0 - self.eps)
            return pd.Series(p, index=df.index, name="up_prob")

        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        per = self._to_period(dfx[date_col], self.refit_freq)

        w_out = np.full(len(dfx), 0.5, dtype=float)

        # Predict period-by-period (few periods, many rows).
        for pk in pd.Index(per).unique():
            idx = np.flatnonzero(per.to_numpy() == pk)
            if idx.size == 0:
                continue

            model = self._get_model_for_period(pk)
            if model is None:
                continue

            X = self._build_X(
                dfx.iloc[idx],
                fit_mode=False,
                mu=model["mu"],
                sd=model["sd"],
                feat_names=model["feat_names"],
            )
            w_out[idx] = self._sigmoid(X @ model["theta"])

        p = w_out * p1 + (1.0 - w_out) * p2
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return pd.Series(p, index=df.index, name="up_prob")

    # Predict gate weights w (diagnostic).
    def predict_weights(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        # Build w in the same pathway as predict().
        if not self.walk_forward:
            if self._theta is None or self._feat_names is None:
                return pd.Series(0.5, index=df.index, name="w_cnn")
            X = self._build_X(df, fit_mode=False, mu=self._mu, sd=self._sd, feat_names=self._feat_names)
            w = self._sigmoid(X @ self._theta)
            return pd.Series(w, index=df.index, name="w_cnn")

        if date_col not in df.columns or not self._periods_sorted:
            return pd.Series(0.5, index=df.index, name="w_cnn")

        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        per = self._to_period(dfx[date_col], self.refit_freq)

        w_out = np.full(len(dfx), 0.5, dtype=float)
        for pk in pd.Index(per).unique():
            idx = np.flatnonzero(per.to_numpy() == pk)
            if idx.size == 0:
                continue
            model = self._get_model_for_period(pk)
            if model is None:
                continue
            X = self._build_X(
                dfx.iloc[idx],
                fit_mode=False,
                mu=model["mu"],
                sd=model["sd"],
                feat_names=model["feat_names"],
            )
            w_out[idx] = self._sigmoid(X @ model["theta"])
        return pd.Series(w_out, index=df.index, name="w_cnn")

    # Return per-date summary weights.
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame(
            {"mean_w_cnn": [self.global_w_cnn_], "std_w_cnn": [0.0], "min_w_cnn": [self.global_w_cnn_], "max_w_cnn": [self.global_w_cnn_], "n_rows": [0]}
        )

    # Return row-level fitted weights.
    def get_row_weights(self) -> pd.DataFrame:
        if self.row_weights_ is not None:
            return self.row_weights_.copy()
        return pd.DataFrame(columns=["row_index", "w_cnn", "w_rf"])

    # Fit global gate on full dataset.
    def _fit_global(self, dfx: pd.DataFrame, *, y_col: str) -> None:
        y = dfx[y_col].to_numpy(dtype=float)
        p1 = dfx["p_cnn"].to_numpy(dtype=float)
        p2 = dfx["p_rf"].to_numpy(dtype=float)

        X, mu, sd, feat_names = self._build_X_fit(dfx)
        theta0 = np.zeros(X.shape[1], dtype=float)
        theta = self._solve_theta(theta0, X, p1, p2, y)

        self._theta = theta
        self._mu = mu
        self._sd = sd
        self._feat_names = feat_names
        self.global_w_cnn_ = 0.5

    # Fit period models with refit schedule, rolling window, and warm start.
    def _fit_walk_forward(self, dfx: pd.DataFrame, *, date_col: str, y_col: str) -> None:
        dfx = dfx.sort_values(date_col).copy()

        # Unique rebalance dates.
        dates = pd.Index(pd.to_datetime(dfx[date_col].unique())).sort_values()

        # Periods covering the dataset.
        periods = self._to_period(dates, self.refit_freq)
        uniq_periods = pd.Index(periods).unique()

        self._models.clear()
        self._periods_sorted = []

        prev_theta: Optional[np.ndarray] = None

        for pk in uniq_periods:
            # Start date of this period.
            mask = (periods == pk)
            if not bool(np.any(mask)):
                continue
            t0 = pd.Timestamp(dates[np.flatnonzero(mask)[0]])

            # Enforce label lag in "number of rebalance dates".
            idx0 = int(dates.get_indexer([t0])[0])
            end_idx = idx0 - int(self.label_lag)
            if end_idx < 0:
                continue
            t_end = pd.Timestamp(dates[end_idx])

            # Rolling training window.
            t_start = t_end - pd.DateOffset(years=int(self.train_window_years))

            df_train = dfx[(dfx[date_col] >= t_start) & (dfx[date_col] <= t_end)].copy()
            df_train = self._subsample_train(df_train, date_col=date_col)

            if len(df_train) < int(self.min_train_rows):
                # Carry forward previous model if available.
                if self._periods_sorted:
                    self._models[pk] = self._models[self._periods_sorted[-1]]
                    self._periods_sorted.append(pk)
                continue

            y = df_train[y_col].to_numpy(dtype=float)
            p1 = df_train["p_cnn"].to_numpy(dtype=float)
            p2 = df_train["p_rf"].to_numpy(dtype=float)

            X, mu, sd, feat_names = self._build_X_fit(df_train)

            if self.warm_start and prev_theta is not None and prev_theta.shape[0] == X.shape[1]:
                theta0 = prev_theta.copy()
            else:
                theta0 = np.zeros(X.shape[1], dtype=float)

            theta = self._solve_theta(theta0, X, p1, p2, y)

            self._models[pk] = {"theta": theta, "mu": mu, "sd": sd, "feat_names": feat_names}
            self._periods_sorted.append(pk)
            prev_theta = theta

        self._periods_sorted = sorted(self._periods_sorted)

    # Subsample training rows to keep refits fast.
    def _subsample_train(self, df_train: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
        rng = np.random.RandomState(int(self.random_state))

        out = df_train

        if self.sample_per_date is not None:
            k = int(self.sample_per_date)
            parts = []
            for _, g in out.groupby(date_col, sort=False):
                if len(g) <= k:
                    parts.append(g)
                else:
                    sel = rng.choice(g.index.to_numpy(), size=k, replace=False)
                    parts.append(g.loc[sel])
            out = pd.concat(parts, axis=0) if parts else out

        if self.max_train_rows is not None and len(out) > int(self.max_train_rows):
            sel = rng.choice(out.index.to_numpy(), size=int(self.max_train_rows), replace=False)
            out = out.loc[sel]

        return out

    # Build X for fit (returns X, mu, sd, feat_names).
    def _build_X_fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        F = self._build_feature_frame(df, feat_names=None, fit_mode=True)
        X = F.to_numpy(dtype=float)

        mu = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1], dtype=float)
        sd_raw = X.std(axis=0) if self.standardize else np.ones(X.shape[1], dtype=float)
        sd = np.where(sd_raw > 1e-12, sd_raw, 1.0)

        if self.standardize:
            X = (X - mu) / sd

        X = self._add_intercept(X)
        return X, mu, sd, list(F.columns)

    # Build X for predict given stored mu/sd and feat_names.
    def _build_X(
        self,
        df: pd.DataFrame,
        *,
        fit_mode: bool,
        mu: Optional[np.ndarray],
        sd: Optional[np.ndarray],
        feat_names: Optional[List[str]],
    ) -> np.ndarray:
        F = self._build_feature_frame(df, feat_names=feat_names, fit_mode=False)
        X = F.to_numpy(dtype=float)

        if self.standardize and (mu is not None) and (sd is not None):
            X = (X - mu) / sd

        X = self._add_intercept(X)
        return X

    # Build the feature DataFrame (no intercept).
    def _build_feature_frame(self, df: pd.DataFrame, *, feat_names: Optional[List[str]], fit_mode: bool) -> pd.DataFrame:
        feats: Dict[str, pd.Series] = {}

        # Optional context features.
        # for c in (self.feature_cols or []):
        #     feats[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0).astype(float)
        missing = [c for c in (self.feature_cols or []) if c not in df.columns]
        if missing:
            raise KeyError(
                f"MoEGating missing feature columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        for c in (self.feature_cols or []):
            feats[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        # Probability-derived features.
        if self.include_prob_features:
            p1 = pd.to_numeric(df["p_cnn"], errors="coerce").fillna(0.5).astype(float).to_numpy()
            p2 = pd.to_numeric(df["p_rf"], errors="coerce").fillna(0.5).astype(float).to_numpy()
            p1 = np.clip(p1, self.eps, 1.0 - self.eps)
            p2 = np.clip(p2, self.eps, 1.0 - self.eps)

            feats["p_cnn"] = pd.Series(p1, index=df.index)
            feats["p_rf"] = pd.Series(p2, index=df.index)
            feats["logit_cnn"] = pd.Series(self._logit(p1), index=df.index)
            feats["logit_rf"] = pd.Series(self._logit(p2), index=df.index)
            feats["abs_diff"] = pd.Series(np.abs(p1 - p2), index=df.index)

        F = pd.DataFrame(feats, index=df.index)

        if fit_mode:
            return F

        # Align predict-time features to training features.
        names = list(feat_names or [])
        return F.reindex(columns=names, fill_value=0.0)

    # Add an intercept column to X.
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.concatenate([ones, X], axis=1)

    # Solve theta by gradient descent on mixture log loss.
    def _solve_theta(self, theta0: np.ndarray, X: np.ndarray, p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta = theta0.astype(float).copy()
        prev = np.inf

        for _ in range(int(self.max_iters)):
            loss, grad = self._loss_grad(theta, X, p1, p2, y)

            if np.isfinite(prev) and prev > 0:
                if abs(prev - loss) / prev < float(self.tol):
                    break
            prev = loss

            theta = theta - float(self.lr) * grad

        return theta

    # Compute loss and gradient for mixture objective.
    def _loss_grad(self, theta: np.ndarray, X: np.ndarray, p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        w = self._sigmoid(X @ theta)
        p = w * p1 + (1.0 - w) * p2
        p = np.clip(p, self.eps, 1.0 - self.eps)

        loss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

        dL_dp = (p - y) / np.maximum(p * (1.0 - p), 1e-12)
        dp_dw = (p1 - p2)
        dw_dz = w * (1.0 - w)
        dL_dz = dL_dp * dp_dw * dw_dz

        grad = (X.T @ dL_dz) / max(X.shape[0], 1)

        if float(self.l2) > 0:
            reg = np.zeros_like(theta)
            reg[1:] = float(self.l2) * theta[1:]
            grad = grad + reg
            loss = loss + 0.5 * float(self.l2) * float(np.sum(theta[1:] ** 2))

        return loss, grad

    # Save fitted row-level and date-level weights for diagnostics.
    def _cache_fitted_weights(self, dfx: pd.DataFrame, *, date_col: str) -> None:
        w = self.predict_weights(dfx, date_col=date_col).rename("w_cnn")
        out = pd.DataFrame(index=dfx.index)
        out["row_index"] = dfx.index
        if date_col in dfx.columns:
            out[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        if "StockID" in dfx.columns:
            out["StockID"] = dfx["StockID"].to_numpy()
        out["w_cnn"] = w.to_numpy(dtype=float)
        out["w_rf"] = 1.0 - out["w_cnn"]

        self.row_weights_ = out.reset_index(drop=True)

        if date_col in out.columns:
            g = out.groupby(date_col)["w_cnn"]
            wdf = pd.DataFrame(
                {
                    "mean_w_cnn": g.mean(),
                    "std_w_cnn": g.std().fillna(0.0),
                    "min_w_cnn": g.min(),
                    "max_w_cnn": g.max(),
                    "n_rows": g.size(),
                }
            ).sort_index()
            self.weights_by_date_ = wdf
            if len(wdf):
                self.global_w_cnn_ = float(wdf["mean_w_cnn"].iloc[-1])
        else:
            self.weights_by_date_ = pd.DataFrame(
                {
                    "mean_w_cnn": [float(out["w_cnn"].mean())],
                    "std_w_cnn": [float(out["w_cnn"].std(ddof=0)) if len(out) > 1 else 0.0],
                    "min_w_cnn": [float(out["w_cnn"].min())],
                    "max_w_cnn": [float(out["w_cnn"].max())],
                    "n_rows": [int(len(out))],
                }
            )
            self.global_w_cnn_ = float(self.weights_by_date_["mean_w_cnn"].iloc[-1])

    # Sigmoid with clipping for stability.
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    # Logit with clipping for stability.
    def _logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(p / (1.0 - p))

    # Convert dates to period keys.
    def _to_period(self, d, freq: str) -> pd.PeriodIndex:
        f = str(freq).lower().strip()
        dt = pd.to_datetime(d, errors="coerce")

        if isinstance(dt, pd.Series):
            if f == "month":
                return dt.dt.to_period("M")
            if f == "year":
                return dt.dt.to_period("Y")
            return dt.dt.to_period("Q")

        dti = pd.DatetimeIndex(dt)
        if f == "month":
            return dti.to_period("M")
        if f == "year":
            return dti.to_period("Y")
        return dti.to_period("Q")

    # Get the latest fitted model for a given period.
    def _get_model_for_period(self, pk: pd.Period) -> Optional[Dict[str, Any]]:
        if pk in self._models:
            return self._models[pk]
        if not self._periods_sorted:
            return None
        prior = [p for p in self._periods_sorted if p <= pk]
        if not prior:
            return None
        return self._models[prior[-1]]
