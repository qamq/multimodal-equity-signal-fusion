"""
multimodal_fusion.experimental.rl_policy

Modern RL-style policy for choosing STOCK-WEEK (row-level) CNN/RF blending weights.

Core idea
---------
For each row (StockID i, Date t), the policy outputs a distinct CNN weight w_{i,t}.

The policy:
- builds a per-row context x_{i,t} from p_cnn, p_rf, and optional columns,
- computes a softmax policy over a discrete weight grid,
- takes the expected weight under that policy,
- blends the experts using that row-level weight.

Leakage-safe historical use
---------------------------
During fit(...), the class walks forward date by date with label lag and stores the
historical fitted row-level weights in row_weights_.

When predict(...) or predict_weights(...) is later called on rows that were already part
of a fitted historical panel, the class first tries to return those stored historical
weights. This prevents historical predictions from being recomputed with the final policy
parameters, which would otherwise leak later information into earlier dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RLPolicyConfig:
    """
    RLPolicyConfig stores the hyperparameters for row-level RL-style blending.

    It defines the action grid, reward type, context construction, learning-rate
    controls, and optional switching-penalty behavior.
    """

    weight_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    label_lag: int = 1

    reward: str = "neg_logloss_tail"
    tail_q: float = 0.10
    tail_common: bool = True
    tail_min_n_per_date: int = 200

    context_mode: str = "auto"
    context_cols: Optional[List[str]] = None
    include_prob_features: bool = True

    policy_temperature: float = 1.0

    lr: float = 0.05
    l2: float = 0.0
    forget_gamma: float = 0.995
    soft_target_scale: float = 5.0
    grad_clip: float = 5.0

    switch_penalty: float = 0.0
    id_col: str = "StockID"

    eps: float = 1e-6
    random_state: int = 7


class RLPolicy:
    """
    RLPolicy learns a row-level softmax policy over a discrete CNN-weight grid.

    The class:
    - updates the policy walk-forward with label lag,
    - saves historical fitted row-level weights during fit(...),
    - returns those saved weights when predicting on rows already seen during fit(...),
    - falls back to the current policy only for genuinely unseen rows.
    """

    # Initialize config, RNG, learned policy state, and fitted-weight caches.
    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        cfg_dict = kwargs.pop("config", None)
        if cfg_dict is not None and isinstance(cfg_dict, dict):
            kwargs = {**cfg_dict, **kwargs}

        if "seed" in kwargs and "random_state" not in kwargs:
            kwargs["random_state"] = kwargs.pop("seed")

        if config is None:
            self.cfg = RLPolicyConfig(**kwargs)
        elif isinstance(config, dict):
            self.cfg = RLPolicyConfig(**{**config, **kwargs})
        else:
            self.cfg = config
            for key, value in kwargs.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)

        self._rng = np.random.RandomState(int(self.cfg.random_state))
        self._Theta: Optional[np.ndarray] = None
        self._feat_names: Optional[List[str]] = None

        self.weights_by_date_: Optional[pd.DataFrame] = None
        self.row_weights_: Optional[pd.DataFrame] = None

        self._prev_w_by_id: Dict[str, float] = {}
        self._row_weight_by_index: Dict[Any, float] = {}
        self._row_weight_by_key: Dict[Tuple[pd.Timestamp, str], float] = {}

    # Fit the policy walk-forward and cache the historical row-level weights it produced.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "RLPolicy":
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        dfx = dfx.dropna(subset=[date_col, "p_cnn", "p_rf", y_col]).copy()

        if len(dfx) == 0:
            self._Theta = None
            self._feat_names = None
            self.weights_by_date_ = pd.DataFrame(columns=["mean_w_cnn", "std_w_cnn", "min_w_cnn", "max_w_cnn", "n_rows"])
            self.row_weights_ = pd.DataFrame(columns=["row_index", date_col, self.cfg.id_col, "w_cnn", "w_rf"])
            self._row_weight_by_index = {}
            self._row_weight_by_key = {}
            return self

        X0, feat_names = self._build_X(dfx.iloc[: min(50_000, len(dfx))], fit_mode=True)
        self._feat_names = feat_names
        self._init_policy(d=int(X0.shape[1]))

        dates = pd.Index(dfx[date_col].unique()).sort_values().to_list()
        diag_rows: List[Tuple[pd.Timestamp, float, float, float, float, int]] = []
        row_parts: List[pd.DataFrame] = []

        for i, dt in enumerate(dates):
            j = i - int(self.cfg.label_lag)
            if j >= 0:
                dt_upd = pd.Timestamp(dates[j])
                frame_u = dfx.loc[dfx[date_col] == dt_upd]
                if not frame_u.empty:
                    self._update_from_frame(frame_u, date=dt_upd, y_col=y_col)

            frame_t = dfx.loc[dfx[date_col] == pd.Timestamp(dt)]
            if frame_t.empty:
                continue

            w_t = self._predict_policy_weights(frame_t).astype(float)

            diag_rows.append(
                (
                    pd.Timestamp(dt),
                    float(np.mean(w_t.to_numpy(dtype=float))),
                    float(np.std(w_t.to_numpy(dtype=float), ddof=0)) if len(w_t) > 1 else 0.0,
                    float(np.min(w_t.to_numpy(dtype=float))),
                    float(np.max(w_t.to_numpy(dtype=float))),
                    int(len(frame_t)),
                )
            )

            part = pd.DataFrame(index=frame_t.index)
            part["row_index"] = frame_t.index
            part[date_col] = pd.Timestamp(dt)
            if self.cfg.id_col in frame_t.columns:
                part[self.cfg.id_col] = frame_t[self.cfg.id_col].astype(str).to_numpy()
            part["w_cnn"] = w_t.to_numpy(dtype=float)
            part["w_rf"] = 1.0 - part["w_cnn"]
            row_parts.append(part)

        if diag_rows:
            wdf = pd.DataFrame(
                diag_rows,
                columns=[date_col, "mean_w_cnn", "std_w_cnn", "min_w_cnn", "max_w_cnn", "n_rows"],
            ).set_index(date_col)
            self.weights_by_date_ = wdf.sort_index()
        else:
            self.weights_by_date_ = pd.DataFrame(columns=["mean_w_cnn", "std_w_cnn", "min_w_cnn", "max_w_cnn", "n_rows"])

        self.row_weights_ = (
            pd.concat(row_parts, axis=0).reset_index(drop=True)
            if row_parts
            else pd.DataFrame(columns=["row_index", date_col, self.cfg.id_col, "w_cnn", "w_rf"])
        )
        self._build_row_weight_cache(date_col=date_col)
        return self

    # Predict blended probabilities using historical fitted weights when available.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        w = self.predict_weights(dfx, date_col=date_col)
        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5)
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5)
        p = w * p1 + (1.0 - w) * p2
        p = p.clip(float(self.cfg.eps), 1.0 - float(self.cfg.eps))
        return pd.Series(p.to_numpy(), index=df.index, name="up_prob")

    # Predict row-level CNN weights, preferring stored historical fitted weights when possible.
    def predict_weights(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        if date_col in dfx.columns:
            dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        out = pd.Series(index=dfx.index, dtype=float, name="w_cnn")

        historical = self._lookup_historical_row_weights(dfx, date_col=date_col)
        if historical is not None:
            out.loc[historical.index] = historical

        missing_idx = out.index[out.isna()]
        if len(missing_idx) == 0:
            return out.fillna(0.5)

        unseen = dfx.loc[missing_idx]
        if self._Theta is None or self._feat_names is None:
            out.loc[missing_idx] = 0.5
            return out.fillna(0.5)

        out.loc[missing_idx] = self._predict_policy_weights(unseen).to_numpy(dtype=float)
        return out.fillna(0.5)

    # Return per-date diagnostics from the fitted historical walk-forward run.
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame(columns=["mean_w_cnn", "std_w_cnn", "min_w_cnn", "max_w_cnn", "n_rows"])

    # Return row-level fitted historical weights from the fitted walk-forward run.
    def get_row_weights(self) -> pd.DataFrame:
        if self.row_weights_ is not None:
            return self.row_weights_.copy()
        return pd.DataFrame(columns=["row_index", "w_cnn", "w_rf"])

    # Initialize the policy parameter matrix Theta.
    def _init_policy(self, d: int) -> None:
        k = len(self.cfg.weight_grid)
        self._Theta = np.zeros((k, d), dtype=float)

    # Build the row-level feature matrix and feature names.
    def _build_X(self, df: pd.DataFrame, *, fit_mode: bool) -> Tuple[np.ndarray, List[str]]:
        feats: Dict[str, np.ndarray] = {}
        eps = float(self.cfg.eps)

        p1 = pd.to_numeric(df["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(df["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p1 = np.clip(p1, eps, 1.0 - eps)
        p2 = np.clip(p2, eps, 1.0 - eps)

        feats["intercept"] = np.ones(len(df), dtype=float)

        if bool(self.cfg.include_prob_features):
            feats["p_cnn"] = p1
            feats["p_rf"] = p2
            feats["logit_cnn"] = np.log(p1 / (1.0 - p1))
            feats["logit_rf"] = np.log(p2 / (1.0 - p2))
            feats["abs_diff"] = np.abs(p1 - p2)

        if str(self.cfg.context_mode).lower().strip() == "columns":
            cols = self.cfg.context_cols or []
            if len(cols) == 0:
                raise ValueError("context_cols must be provided when context_mode='columns'.")
            for col in cols:
                vals = pd.to_numeric(df.get(col, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                feats[col] = vals

        frame = pd.DataFrame(feats, index=df.index)

        if fit_mode:
            feat_names = list(frame.columns)
            self._feat_names = feat_names
        else:
            feat_names = list(self._feat_names or [])
            frame = frame.reindex(columns=feat_names, fill_value=0.0)

        X = frame.to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, feat_names

    # Compute softmax policy probabilities for a batch of feature rows.
    def _policy_batch(self, X: np.ndarray) -> np.ndarray:
        assert self._Theta is not None
        temp = max(float(self.cfg.policy_temperature), 1e-8)

        logits = (X @ self._Theta.T) / temp
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
        norm = np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)
        return exp_logits / norm

    # Predict weights from the current policy only, without using historical caches.
    def _predict_policy_weights(self, df: pd.DataFrame) -> pd.Series:
        if self._Theta is None or self._feat_names is None:
            return pd.Series(0.5, index=df.index, name="w_cnn")
        X, _ = self._build_X(df, fit_mode=False)
        probs = self._policy_batch(X)
        expected_w = probs @ np.asarray(self.cfg.weight_grid, dtype=float)
        return pd.Series(expected_w, index=df.index, name="w_cnn")

    # Update Theta from one realized date using full-information soft targets.
    def _update_from_frame(self, frame_u: pd.DataFrame, *, date: pd.Timestamp, y_col: str) -> None:
        assert self._Theta is not None

        eps = float(self.cfg.eps)
        reward_name = str(self.cfg.reward).lower().strip()
        use_tail = "tail" in reward_name

        p1 = pd.to_numeric(frame_u["p_cnn"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        p2 = pd.to_numeric(frame_u["p_rf"], errors="coerce").astype(float).fillna(0.5).to_numpy()
        y = pd.to_numeric(frame_u[y_col], errors="coerce").astype(float).to_numpy()

        p1 = np.clip(p1, eps, 1.0 - eps)
        p2 = np.clip(p2, eps, 1.0 - eps)

        mask = np.ones(len(frame_u), dtype=bool)
        if use_tail:
            base = 0.5 * (p1 + p2)
            lo = float(np.quantile(base, float(self.cfg.tail_q)))
            hi = float(np.quantile(base, 1.0 - float(self.cfg.tail_q)))
            mask = (base <= lo) | (base >= hi)
            if int(mask.sum()) < max(int(self.cfg.tail_min_n_per_date), 10):
                mask[:] = True

        if int(mask.sum()) < 5:
            return

        X, _ = self._build_X(frame_u.loc[mask], fit_mode=False)
        p1m = p1[mask]
        p2m = p2[mask]
        ym = y[mask]

        weight_grid = np.asarray(self.cfg.weight_grid, dtype=float)[None, :]
        p = p1m[:, None] * weight_grid + p2m[:, None] * (1.0 - weight_grid)
        p = np.clip(p, eps, 1.0 - eps)

        if "brier" in reward_name:
            rewards = -(ym[:, None] - p) ** 2
        else:
            rewards = ym[:, None] * np.log(p) + (1.0 - ym[:, None]) * np.log(1.0 - p)

        if float(self.cfg.switch_penalty) > 0 and (self.cfg.id_col in frame_u.columns):
            ids = frame_u.loc[mask, self.cfg.id_col].astype(str).to_numpy()
            prevw = np.array([self._prev_w_by_id.get(sid, 0.5) for sid in ids], dtype=float)[:, None]
            rewards = rewards - float(self.cfg.switch_penalty) * np.abs(weight_grid - prevw)

        scale = float(self.cfg.soft_target_scale)
        logits = scale * (rewards - np.max(rewards, axis=1, keepdims=True))
        logits = np.clip(logits, -50.0, 50.0)
        Q = np.exp(logits)
        Q = Q / np.maximum(np.sum(Q, axis=1, keepdims=True), 1e-12)

        P = self._policy_batch(X)
        n = float(X.shape[0])
        grad = ((Q - P).T @ X) / max(n, 1.0)

        gamma = float(self.cfg.forget_gamma)
        if gamma < 1.0:
            self._Theta *= gamma
        if float(self.cfg.l2) > 0:
            self._Theta *= (1.0 - float(self.cfg.l2))

        gnorm = float(np.linalg.norm(grad))
        clip = float(self.cfg.grad_clip)
        if clip > 0 and gnorm > clip:
            grad = (clip / max(gnorm, 1e-12)) * grad

        self._Theta = self._Theta + float(self.cfg.lr) * grad
        self._Theta = np.nan_to_num(self._Theta, nan=0.0, posinf=0.0, neginf=0.0)

        if float(self.cfg.switch_penalty) > 0 and (self.cfg.id_col in frame_u.columns):
            w_now = self._predict_policy_weights(frame_u).to_numpy(dtype=float)
            ids_all = frame_u[self.cfg.id_col].astype(str).to_numpy()
            for sid, wv in zip(ids_all, w_now):
                self._prev_w_by_id[sid] = float(wv)

    # Build fast historical row-weight lookup tables after fit(...).
    def _build_row_weight_cache(self, *, date_col: str) -> None:
        self._row_weight_by_index = {}
        self._row_weight_by_key = {}

        if self.row_weights_ is None or len(self.row_weights_) == 0:
            return

        if "row_index" in self.row_weights_.columns:
            self._row_weight_by_index = {
                row_idx: float(w)
                for row_idx, w in zip(self.row_weights_["row_index"], self.row_weights_["w_cnn"])
            }

        if date_col in self.row_weights_.columns and self.cfg.id_col in self.row_weights_.columns:
            dates = pd.to_datetime(self.row_weights_[date_col], errors="coerce").dt.normalize()
            ids = self.row_weights_[self.cfg.id_col].astype(str)
            weights = self.row_weights_["w_cnn"].astype(float)
            self._row_weight_by_key = {
                (pd.Timestamp(dt), sid): float(w)
                for dt, sid, w in zip(dates, ids, weights)
                if pd.notna(dt)
            }

    # Look up stored historical row-level weights for rows that were part of a fitted panel.
    def _lookup_historical_row_weights(self, df: pd.DataFrame, *, date_col: str) -> Optional[pd.Series]:
        if (not self._row_weight_by_index) and (not self._row_weight_by_key):
            return None

        out = pd.Series(index=df.index, dtype=float, name="w_cnn")

        if self._row_weight_by_index:
            out.loc[df.index] = [self._row_weight_by_index.get(idx, np.nan) for idx in df.index]

        if date_col in df.columns and self.cfg.id_col in df.columns and self._row_weight_by_key:
            mask = out.isna()
            if bool(mask.any()):
                dates = pd.to_datetime(df.loc[mask, date_col], errors="coerce").dt.normalize()
                ids = df.loc[mask, self.cfg.id_col].astype(str)
                out.loc[mask] = [
                    self._row_weight_by_key.get((pd.Timestamp(dt), sid), np.nan) if pd.notna(dt) else np.nan
                    for dt, sid in zip(dates, ids)
                ]

        if bool(out.notna().any()):
            return out.dropna()

        return None
