"""
multimodal_fusion.core.hmm_multihorizon

Date-level latent-state multi-horizon ensemble.

This file implements a date-level hidden-state model that combines:
- CNN 5-day probabilities
- RF 20-day probabilities
- date-level context features

The model treats the weekly market regime as a hidden Markov chain and uses
state-specific logistic emission models:
1) a 20-day block for Y_20d
2) a 5-day block for Y_5d conditional on Y_20d

The implementation follows a fit/predict interface and caches leakage-safe
walk-forward predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from multimodal_fusion.core.hmm_labels import mask_unrealized_hmm_labels, prepare_hmm_labels

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# Normalize a date-like series to timezone-naive daily timestamps.
def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


# Clip probabilities into a numerically stable open interval.
def _clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


# Convert probabilities into log-odds after numerical clipping.
def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    q = _clip_prob(np.asarray(p, dtype=float), eps)
    return np.log(q / (1.0 - q))


# Apply a numerically stable sigmoid transformation.
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


# Compute a numerically stable log-sum-exp reduction.
def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    m = np.max(arr, axis=axis, keepdims=True)
    stable = np.exp(arr - m)
    s = np.sum(stable, axis=axis, keepdims=True)
    out = m + np.log(np.clip(s, 1e-300, None))
    if axis is None:
        return np.asarray(out).reshape(()).item()
    return np.squeeze(out, axis=axis)


# Compute row-level Bernoulli log-likelihood contributions.
def _bernoulli_loglik(y: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = _clip_prob(np.asarray(p, dtype=float), eps)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


# Compute a weighted Bernoulli mean with stability clipping.
def _weighted_binary_mean(y: np.ndarray, w: np.ndarray, eps: float) -> float:
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.5
    p = float(np.sum(w * y) / denom)
    return float(np.clip(p, eps, 1.0 - eps))


class _ConstantBernoulliModel:
    """Fallback binary-probability model when weighted logistic regression is ill-posed."""

    # Initialize the constant Bernoulli fallback model.
    def __init__(self, p_one: float) -> None:
        self.p_one = float(np.clip(float(p_one), 1e-6, 1.0 - 1e-6))

    # Return constant class probabilities for every requested row.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = int(len(X))
        p1 = np.full(n, self.p_one, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


# Fit a weighted logistic model or fall back to a constant Bernoulli model.
def _fit_weighted_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, random_state: int, eps: float):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    keep = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if X.ndim == 2:
        keep = keep & np.isfinite(X).all(axis=1)

    Xk = X[keep]
    yk = y[keep].astype(int)
    wk = w[keep]

    if len(yk) == 0:
        return _ConstantBernoulliModel(0.5)

    # Need both classes with positive total weight for logistic regression.
    w0 = float(wk[yk == 0].sum())
    w1 = float(wk[yk == 1].sum())
    if w0 <= 1e-12 or w1 <= 1e-12:
        return _ConstantBernoulliModel(_weighted_binary_mean(yk, wk, eps))

    C = 1e6 if float(l2) <= 0 else float(1.0 / l2)
    model = LogisticRegression(
        C=C,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=500,
        random_state=int(random_state),
    )
    model.fit(Xk, yk, sample_weight=wk)
    return model


@dataclass
class HMMMultihorizonConfig:
    """Configuration for the latent-state multi-horizon ensemble."""
    num_states: int = 2
    max_iter: int = 20
    tol: float = 1e-4
    l2: float = 1.0
    prob_clip: float = 1e-6

    refit_freq: str = "month"          # month / quarter / year
    train_window_years: Optional[int] = 3
    label_lag_periods: int = 4         # additional observation-grid embargo
    min_history_dates: int = 26
    transition_smoothing: float = 1e-3

    context_cols: Optional[List[str]] = None
    random_state: int = 42
    verbose: bool = False

    show_progress: bool = False
    progress_desc: Optional[str] = None
    progress_leave: bool = False
    progress_bar: Any = None


class HMMMultihorizon:
    """
    Date-level latent-state multi-horizon ensemble.

    Required input columns
    ----------------------
    - Date
    - StockID
    - p_cnn
    - p_rf

    Training outcomes
    -----------------
    - y_5d / y_20d (or fwd_ret_5d / fwd_ret_20d, thresholded at zero)
    - label_end_5d / label_end_20d: actual horizon end dates from the source
    Missing outcomes never enter the corresponding emission likelihood.
    Missing horizon endpoints are unavailable for training, too. Each refit
    uses only endpoints strictly before its first prediction date, in addition
    to the configured label lag. Prediction does not require this metadata.
    The conditional 5-day emission requires both observed outcomes.

    Optional
    --------
    - any date-level context columns, typically prefixed with ctx_
    """

    # Initialize the multi-horizon HMM wrapper and its caches.
    def __init__(self, config: Optional[HMMMultihorizonConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = HMMMultihorizonConfig(**kwargs)
        self.config = config
        self.row_pred_: Optional[pd.DataFrame] = None
        self.state_probs_by_date_: Optional[pd.DataFrame] = None
        self.fit_log_: List[Dict[str, Any]] = []
        self.context_cols_: List[str] = []
        self._last_fit_bundle: Optional[Dict[str, Any]] = None
        self._last_q_: Optional[np.ndarray] = None
        self._last_pred_date: Optional[pd.Timestamp] = None

    # Fit the latent-state model on a historical panel and cache walk-forward predictions.
    def fit(self, df: pd.DataFrame) -> "HMMMultihorizon":
        data = self._prepare_panel(df, require_labels=True)
        if len(data) == 0:
            raise ValueError("HMMMultihorizon.fit received an empty panel after preprocessing.")

        dates = pd.Index(sorted(data["Date"].unique()))
        if len(dates) < max(self.config.min_history_dates, self.config.num_states + 2):
            raise ValueError("Not enough unique dates to fit the latent-state model.")

        pred_dates = self._build_refit_dates(dates)
        eligible_blocks = self._collect_eligible_blocks(data, dates, pred_dates)

        pred_rows: List[pd.DataFrame] = []
        state_rows: List[pd.DataFrame] = []

        own_bar = None
        external_bar = self.config.progress_bar
        if external_bar is None and bool(self.config.show_progress) and tqdm is not None:
            own_bar = tqdm(
                total=len(eligible_blocks),
                desc=self.config.progress_desc or "HMM refit blocks",
                leave=bool(self.config.progress_leave),
            )

        try:
            for block in eligible_blocks:
                pred_start = block["pred_start"]
                pred_end = block["pred_end"]
                train_start = block["train_start"]
                hist_end = block["hist_end"]

                train_df = data[(data["Date"] >= train_start) & (data["Date"] <= hist_end)].copy()
                fit_bundle = self._fit_hmm_block(train_df, information_cutoff=pred_start)
                self._last_fit_bundle = fit_bundle

                if external_bar is not None:
                    external_bar.update(1)
                elif own_bar is not None:
                    own_bar.update(1)

                last_train_date = fit_bundle["dates"][-1]
                gap_steps = int(np.sum((dates > last_train_date) & (dates <= pred_start)))
                q = fit_bundle["filtered_last"]
                if gap_steps > 0:
                    q = self._propagate_state(q, fit_bundle["A"], steps=gap_steps)

                block_dates = [d for d in dates if d >= pred_start and d <= pred_end]
                for d in block_dates:
                    rows_d = data[data["Date"] == d].copy()
                    if len(rows_d) == 0:
                        continue

                    if d > pred_start:
                        q = self._propagate_state(q, fit_bundle["A"], steps=1)

                    p = self._predict_rows_given_state_probs(rows_d, q, fit_bundle)
                    out = rows_d[["Date", "StockID"]].copy()
                    out["up_prob"] = p
                    pred_rows.append(out)

                    sp = pd.DataFrame(
                        {
                            "Date": [pd.Timestamp(d)] * self.config.num_states,
                            "state": np.arange(self.config.num_states),
                            "prob": q,
                        }
                    )
                    state_rows.append(sp)

                    self._last_q_ = q.copy()
                    self._last_pred_date = pd.Timestamp(d)
        finally:
            if own_bar is not None:
                own_bar.close()

        if not pred_rows:
            raise ValueError(
                "HMMMultihorizon.fit produced no walk-forward predictions: insufficient "
                "realized training history. Supply actual label_end_5d / label_end_20d "
                "dates; missing endpoints and outcomes are unavailable for training."
            )

        pred = pd.concat(pred_rows, ignore_index=True)
        pred["Date"] = _to_dt(pred["Date"])
        pred["StockID"] = pd.to_numeric(pred["StockID"], errors="coerce").astype("Int64")
        pred = pred.dropna(subset=["Date", "StockID", "up_prob"]).copy()
        pred["StockID"] = pred["StockID"].astype(int).astype(str)
        pred = pred.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)
        self.row_pred_ = pred

        if state_rows:
            sp = pd.concat(state_rows, ignore_index=True)
            sp["Date"] = _to_dt(sp["Date"])
            self.state_probs_by_date_ = sp.sort_values(["Date", "state"]).reset_index(drop=True)

        return self

    # Predict final 5-day probabilities for cached or new stock-date rows.
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return probabilities in input order with exactly ``df.index``.

        Rows with invalid keys or expert inputs, and historical rows without
        a cached walk-forward forecast, receive NaN. Outcome columns
        are neither required nor used. Internal chronological sorting does
        not change positional alignment, including for duplicate index labels.
        """
        panel = self._prepare_panel(df.reset_index(drop=True), require_labels=False)
        def aligned(values: pd.Series) -> pd.Series:
            return values.reindex(pd.RangeIndex(len(df))).set_axis(df.index)

        if len(panel) == 0:
            return pd.Series(np.nan, index=df.index, name="up_prob")

        out = pd.Series(index=panel.index, dtype=float, name="up_prob")

        if self.row_pred_ is not None:
            cached = self.row_pred_.copy()
            cached["key"] = cached["Date"].astype(str) + "||" + cached["StockID"].astype(str)
            cached_map = dict(zip(cached["key"], cached["up_prob"]))
            key = panel["Date"].astype(str) + "||" + panel["StockID"].astype(str)
            mask = key.isin(cached_map)
            if mask.any():
                out.loc[mask] = key.loc[mask].map(cached_map).astype(float).to_numpy()

        if out.notna().all():
            return aligned(out)

        if self._last_fit_bundle is None or self._last_q_ is None:
            raise ValueError("Model has no fitted state bundle available for out-of-sample prediction.")

        # Warm-up rows have no historically fitted forecast. Preserve them as
        # missing instead of applying a model trained later to earlier inputs.
        can_forecast = out.isna()
        if self._last_pred_date is not None:
            can_forecast &= panel["Date"] >= self._last_pred_date
        missing = panel.loc[can_forecast].copy()
        if missing.empty:
            return aligned(out)
        missing_dates = sorted(missing["Date"].unique())
        q = self._last_q_.copy()
        last_seen = self._last_pred_date

        preds_new: List[pd.DataFrame] = []
        state_new: List[pd.DataFrame] = []

        for d in missing_dates:
            if last_seen is not None:
                if d > last_seen:
                    q = self._propagate_state(q, self._last_fit_bundle["A"], steps=1)

            rows_d = missing[missing["Date"] == d].copy()
            p = self._predict_rows_given_state_probs(rows_d, q, self._last_fit_bundle)
            tmp = rows_d[["Date", "StockID"]].copy()
            tmp["up_prob"] = p
            preds_new.append(tmp)

            state_new.append(
                pd.DataFrame({"Date": [pd.Timestamp(d)] * self.config.num_states, "state": np.arange(self.config.num_states), "prob": q})
            )
            last_seen = pd.Timestamp(d)

        if preds_new:
            new_pred = pd.concat(preds_new, ignore_index=True)
            new_pred["key"] = new_pred["Date"].astype(str) + "||" + new_pred["StockID"].astype(str)
            new_map = dict(zip(new_pred["key"], new_pred["up_prob"]))
            key = panel["Date"].astype(str) + "||" + panel["StockID"].astype(str)
            out.loc[out.isna()] = key.loc[out.isna()].map(new_map).astype(float).to_numpy()

            self.row_pred_ = pd.concat([self.row_pred_, new_pred.drop(columns=["key"], errors="ignore")], ignore_index=True)
            self.row_pred_ = self.row_pred_.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)

        if state_new:
            sp = pd.concat(state_new, ignore_index=True)
            self.state_probs_by_date_ = (
                pd.concat([self.state_probs_by_date_, sp], ignore_index=True)
                if self.state_probs_by_date_ is not None
                else sp
            )
            self.state_probs_by_date_ = self.state_probs_by_date_.drop_duplicates(["Date", "state"], keep="last").sort_values(["Date", "state"]).reset_index(drop=True)

        self._last_q_ = q.copy()
        self._last_pred_date = last_seen
        return aligned(out)

    # Return the filtered state probabilities by date.
    def get_weights(self) -> Any:
        return self.state_probs_by_date_

    # Return the cached row-level prediction table.
    def get_row_weights(self) -> Any:
        return self.row_pred_

    # Return the final fitted HMM bundle for diagnostics.
    def get_fit_bundle(self) -> Any:
        return self._last_fit_bundle

    # Return the final context-column order used by the model.
    def get_context_cols(self) -> List[str]:
        return list(self.context_cols_)

    # Validate and standardize the stock-date modeling panel.
    def _prepare_panel(self, df: pd.DataFrame, *, require_labels: bool) -> pd.DataFrame:
        d = df.copy()
        need = ["Date", "StockID", "p_cnn", "p_rf"]
        for c in need:
            if c not in d.columns:
                raise KeyError(f"HMMMultihorizon requires column '{c}'.")

        d["Date"] = _to_dt(d["Date"])
        sid = pd.to_numeric(d["StockID"], errors="coerce")
        d["StockID"] = sid
        d["p_cnn"] = pd.to_numeric(d["p_cnn"], errors="coerce")
        d["p_rf"] = pd.to_numeric(d["p_rf"], errors="coerce")
        d = d.dropna(subset=["Date", "StockID", "p_cnn", "p_rf"]).copy()
        d["StockID"] = d["StockID"].astype(int).astype(str)

        if require_labels:
            d = prepare_hmm_labels(d)

            context_cols = self.config.context_cols
            if context_cols is None:
                context_cols = [c for c in d.columns if str(c).startswith("ctx_")]
            self.context_cols_ = [c for c in context_cols if c in d.columns]

        for c in self.context_cols_:
            if c not in d:
                raise KeyError(f"HMMMultihorizon.predict requires fitted context column '{c}'.")
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.sort_values(["Date", "StockID"], kind="stable")
        if require_labels:
            d = d.drop_duplicates(["Date", "StockID"], keep="last").reset_index(drop=True)
        return d

    # Create the sequence of walk-forward prediction blocks.
    def _build_refit_dates(self, dates: pd.Index) -> List[Dict[str, pd.Timestamp]]:
        freq = str(self.config.refit_freq).lower().strip()
        unique_dates = pd.Index(sorted(pd.to_datetime(dates).normalize().unique()))
        if freq == "quarter":
            grp = pd.Series(unique_dates).dt.to_period("Q")
        elif freq == "year":
            grp = pd.Series(unique_dates).dt.to_period("Y")
        else:
            grp = pd.Series(unique_dates).dt.to_period("M")

        out: List[Dict[str, pd.Timestamp]] = []
        groups = grp.drop_duplicates().tolist()
        for g in groups:
            block_dates = unique_dates[grp == g]
            pred_start = pd.Timestamp(block_dates.min()).normalize()
            pred_end = pd.Timestamp(block_dates.max()).normalize()
            out.append({"pred_start": pred_start, "pred_end": pred_end})
        return out

    # Count and collect the refit blocks that are eligible for one walk-forward fit.
    def _collect_eligible_blocks(
        self,
        data: pd.DataFrame,
        dates: pd.Index,
        pred_dates: List[Dict[str, pd.Timestamp]],
    ) -> List[Dict[str, pd.Timestamp]]:
        eligible: List[Dict[str, pd.Timestamp]] = []

        for block in pred_dates:
            pred_start = block["pred_start"]
            hist_end = self._history_end_date(dates, pred_start)
            if hist_end is None:
                continue

            train_start = self._history_start_date(hist_end)
            train_df = data[(data["Date"] >= train_start) & (data["Date"] <= hist_end)].copy()
            train_df = mask_unrealized_hmm_labels(train_df, pred_start)
            # The conditional 5d block also needs y_20d, so every date with
            # any usable emission has at least one available 20d outcome.
            observed_dates = train_df.loc[train_df["y_20d"].notna(), "Date"].nunique()
            if observed_dates < max(self.config.min_history_dates, self.config.num_states + 2):
                continue

            eligible.append(
                {
                    "pred_start": pd.Timestamp(block["pred_start"]).normalize(),
                    "pred_end": pd.Timestamp(block["pred_end"]).normalize(),
                    "hist_end": pd.Timestamp(hist_end).normalize(),
                    "train_start": pd.Timestamp(train_start).normalize(),
                }
            )

        return eligible

    # Apply the existing observation-grid embargo; realization is checked separately.
    def _history_end_date(self, dates: pd.Index, pred_start: pd.Timestamp) -> Optional[pd.Timestamp]:
        dates = pd.Index(sorted(pd.to_datetime(dates).normalize().unique()))
        lag = int(self.config.label_lag_periods)
        pred_pos = int(np.searchsorted(dates.values, np.datetime64(pred_start), side="left"))
        hist_pos = pred_pos - lag - 1
        if hist_pos < 0:
            return None
        return pd.Timestamp(dates[hist_pos]).normalize()

    # Compute the rolling history start date for one fit window.
    def _history_start_date(self, hist_end: pd.Timestamp) -> pd.Timestamp:
        if self.config.train_window_years is None:
            return pd.Timestamp("1900-01-01").normalize()
        return (pd.Timestamp(hist_end).normalize() - pd.DateOffset(years=int(self.config.train_window_years)) + pd.Timedelta(days=1)).normalize()

    # Fit one latent-state block model on a historical training slice.
    def _fit_hmm_block(self, train_df: pd.DataFrame, *, information_cutoff: pd.Timestamp) -> Dict[str, Any]:
        train_df = mask_unrealized_hmm_labels(train_df, information_cutoff)
        mats = self._build_matrices(train_df, require_labels=True)
        T = mats["num_dates"]
        K = int(self.config.num_states)

        init_states = self._initialize_states(train_df, mats)
        gamma = np.zeros((T, K), dtype=float)
        gamma[np.arange(T), init_states] = 1.0

        pi = self._update_pi(gamma)
        A = self._init_transition(init_states, K)

        models20, models5 = self._fit_state_models(mats, gamma)
        prev_ll = -np.inf
        ll_trace: List[float] = []

        for it in range(int(self.config.max_iter)):
            logB = self._emission_loglik(mats, models20, models5)
            gamma, xi, loglik, filtered = self._forward_backward(logB, pi, A)
            ll_trace.append(float(loglik))

            pi = self._update_pi(gamma)
            A = self._update_A(xi)
            models20, models5 = self._fit_state_models(mats, gamma)

            if self.config.verbose:
                print(f"[HMMMultihorizon] iter={it+1} loglik={loglik:.6f}")

            if it > 0 and abs(float(loglik) - float(prev_ll)) <= float(self.config.tol):
                break
            prev_ll = float(loglik)

        # The loop ends with an M-step: refresh the filter under the returned
        # transition, initial-state, and emission parameters (also for max_iter=0).
        logB = self._emission_loglik(mats, models20, models5)
        _, _, final_loglik, filtered = self._forward_backward(logB, pi, A)
        bundle = {
            "pi": pi,
            "A": A,
            "models20": models20,
            "models5": models5,
            "dates": mats["dates"],
            "context_cols": list(self.context_cols_),
            "filtered_last": filtered[-1].copy(),
            "ll_trace": ll_trace,
            "final_loglik": float(final_loglik),
            "information_cutoff": pd.Timestamp(information_cutoff),
        }
        self.fit_log_.append(
            {
                "train_start": str(train_df["Date"].min().date()),
                "train_end": str(train_df["Date"].max().date()),
                "num_dates": int(T),
                "num_rows": int(len(train_df)),
                "num_observed_20d": int(mats["valid_20d"].sum()),
                "num_observed_5d_conditional": int(mats["valid_5d_conditional"].sum()),
                "information_cutoff": str(pd.Timestamp(information_cutoff).date()),
                "ll_trace": ll_trace,
            }
        )
        return bundle

    # Construct the design matrices used by the emission models.
    def _build_matrices(self, df: pd.DataFrame, *, require_labels: bool) -> Dict[str, Any]:
        d = df.copy()
        dates = pd.Index(sorted(d["Date"].unique()))
        date_to_idx = {pd.Timestamp(dt): i for i, dt in enumerate(dates)}
        date_ix = d["Date"].map(date_to_idx).to_numpy(dtype=int)

        lcnn = _logit(d["p_cnn"].to_numpy(dtype=float), self.config.prob_clip)
        lrf = _logit(d["p_rf"].to_numpy(dtype=float), self.config.prob_clip)

        ctx_cols = list(self.context_cols_)
        if ctx_cols:
            C = d[ctx_cols].to_numpy(dtype=float)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            C = np.zeros((len(d), 0), dtype=float)

        X20 = np.column_stack([lrf, C])
        if require_labels:
            y20 = d["y_20d"].to_numpy(dtype=float, na_value=np.nan)
            y5 = d["y_5d"].to_numpy(dtype=float, na_value=np.nan)
            valid_20d = np.isfinite(y20)
            valid_5d = np.isfinite(y5)
        else:
            y20 = None
            y5 = None
            valid_20d = np.zeros(len(d), dtype=bool)
            valid_5d = np.zeros(len(d), dtype=bool)

        X5_obs = np.column_stack([lcnn, lrf, y20 if require_labels else np.zeros(len(d), dtype=float), C])
        X5_y0 = np.column_stack([lcnn, lrf, np.zeros(len(d), dtype=float), C])
        X5_y1 = np.column_stack([lcnn, lrf, np.ones(len(d), dtype=float), C])

        return {
            "df": d,
            "dates": dates,
            "num_dates": int(len(dates)),
            "date_ix": date_ix,
            "X20": X20,
            "X5_obs": X5_obs,
            "X5_y0": X5_y0,
            "X5_y1": X5_y1,
            "y20": y20,
            "y5": y5,
            "valid_20d": valid_20d,
            "valid_5d": valid_5d,
            "valid_5d_conditional": valid_5d & valid_20d,
        }

    # Initialize the hidden states from date-level clustering features.
    def _initialize_states(self, train_df: pd.DataFrame, mats: Dict[str, Any]) -> np.ndarray:
        dates = mats["dates"]

        # Build the core initialization features directly from expert probabilities.
        date_df = train_df.groupby("Date", as_index=False).agg(
            ctx_mean_p_cnn=("p_cnn", "mean"),
            ctx_mean_p_rf=("p_rf", "mean"),
        )
        date_df["ctx_abs_gap_mean"] = (
            train_df.assign(
                _gap=(
                    pd.to_numeric(train_df["p_cnn"], errors="coerce")
                    - pd.to_numeric(train_df["p_rf"], errors="coerce")
                ).abs()
            )
            .groupby("Date")["_gap"]
            .mean()
            .reindex(dates)
            .to_numpy()
        )

        # Exclude built-in initialization columns from the merged context table to avoid
        # duplicate-name suffixes such as _x / _y after the merge.
        builtin_init_cols = {"ctx_mean_p_cnn", "ctx_mean_p_rf", "ctx_abs_gap_mean"}
        extra_context_cols = [c for c in self.context_cols_ if c not in builtin_init_cols]

        if extra_context_cols:
            date_level = train_df.groupby("Date", as_index=False)[extra_context_cols].mean()
        else:
            date_level = pd.DataFrame({"Date": dates})

        init_df = (
            pd.DataFrame({"Date": dates})
            .merge(date_df, on="Date", how="left")
            .merge(date_level, on="Date", how="left")
        )

        feat_cols = ["ctx_mean_p_cnn", "ctx_mean_p_rf", "ctx_abs_gap_mean"] + extra_context_cols
        X_init = init_df[feat_cols].to_numpy(dtype=float)
        X_init = np.nan_to_num(X_init, nan=0.0, posinf=0.0, neginf=0.0)

        K = int(self.config.num_states)
        if len(X_init) < K:
            return np.arange(len(X_init), dtype=int) % K

        km = KMeans(n_clusters=K, random_state=int(self.config.random_state), n_init=10)
        return km.fit_predict(X_init)

    # Initialize the transition matrix from a hard state path.
    def _init_transition(self, states: np.ndarray, K: int) -> np.ndarray:
        A = np.full((K, K), float(self.config.transition_smoothing), dtype=float)
        for t in range(1, len(states)):
            A[int(states[t - 1]), int(states[t])] += 1.0
        A = A / A.sum(axis=1, keepdims=True)
        return A

    # Update the initial state probabilities from posterior weights.
    def _update_pi(self, gamma: np.ndarray) -> np.ndarray:
        pi = np.asarray(gamma[0], dtype=float) + float(self.config.transition_smoothing)
        pi = pi / np.sum(pi)
        return pi

    # Update the transition matrix from expected transition counts.
    def _update_A(self, xi: np.ndarray) -> np.ndarray:
        A = np.sum(xi, axis=0) + float(self.config.transition_smoothing)
        A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)
        return A

    # Fit the state-specific 20-day and 5-day logistic emission models.
    def _fit_state_models(self, mats: Dict[str, Any], gamma: np.ndarray):
        K = int(self.config.num_states)
        date_ix = mats["date_ix"]
        y20 = mats["y20"]
        y5 = mats["y5"]
        X20 = mats["X20"]
        X5 = mats["X5_obs"]
        valid20 = mats["valid_20d"]
        valid5 = mats["valid_5d_conditional"]

        models20 = []
        models5 = []
        for s in range(K):
            w = gamma[date_ix, s]
            m20 = _fit_weighted_logit(X20[valid20], y20[valid20], w[valid20], self.config.l2, self.config.random_state + s, self.config.prob_clip)
            m5 = _fit_weighted_logit(X5[valid5], y5[valid5], w[valid5], self.config.l2, self.config.random_state + 100 + s, self.config.prob_clip)
            models20.append(m20)
            models5.append(m5)
        return models20, models5

    # Evaluate date-level emission log-likelihoods for every state.
    def _emission_loglik(self, mats: Dict[str, Any], models20: List[Any], models5: List[Any]) -> np.ndarray:
        T = mats["num_dates"]
        K = int(self.config.num_states)
        date_ix = mats["date_ix"]
        y20 = mats["y20"]
        y5 = mats["y5"]
        X20 = mats["X20"]
        X5 = mats["X5_obs"]
        valid20 = mats["valid_20d"]
        valid5 = mats["valid_5d_conditional"]

        logB = np.zeros((T, K), dtype=float)
        for s in range(K):
            ll_row = np.zeros(len(date_ix), dtype=float)
            if valid20.any():
                p20 = models20[s].predict_proba(X20[valid20])[:, 1]
                ll_row[valid20] += _bernoulli_loglik(y20[valid20], p20, self.config.prob_clip)
            if valid5.any():
                p5 = models5[s].predict_proba(X5[valid5])[:, 1]
                ll_row[valid5] += _bernoulli_loglik(y5[valid5], p5, self.config.prob_clip)
            logB[:, s] = np.bincount(date_ix, weights=ll_row, minlength=T)
        return logB

    # Run the forward-backward algorithm for one fitted HMM block.
    def _forward_backward(self, logB: np.ndarray, pi: np.ndarray, A: np.ndarray):
        T, K = logB.shape
        log_pi = np.log(_clip_prob(pi, self.config.prob_clip))
        log_A = np.log(_clip_prob(A, self.config.prob_clip))

        log_alpha = np.zeros((T, K), dtype=float)
        log_alpha[0] = log_pi + logB[0]

        for t in range(1, T):
            for s in range(K):
                log_alpha[t, s] = logB[t, s] + _logsumexp(log_alpha[t - 1] + log_A[:, s])

        log_beta = np.zeros((T, K), dtype=float)
        for t in range(T - 2, -1, -1):
            for r in range(K):
                log_beta[t, r] = _logsumexp(log_A[r, :] + logB[t + 1, :] + log_beta[t + 1, :])

        loglik = float(_logsumexp(log_alpha[-1], axis=0))
        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)
        gamma = gamma / np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)

        xi = np.zeros((T - 1, K, K), dtype=float)
        for t in range(1, T):
            log_xi_t = (
                log_alpha[t - 1][:, None]
                + log_A
                + logB[t][None, :]
                + log_beta[t][None, :]
                - loglik
            )
            xi_t = np.exp(log_xi_t)
            xi_t = xi_t / np.clip(xi_t.sum(), 1e-12, None)
            xi[t - 1] = xi_t

        filtered = np.exp(log_alpha - _logsumexp(log_alpha, axis=1)[:, None])
        return gamma, xi, loglik, filtered

    # Project the filtered state distribution forward through the transition matrix.
    def _propagate_state(self, q: np.ndarray, A: np.ndarray, *, steps: int) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        steps = max(int(steps), 0)
        for _ in range(steps):
            out = out @ A
            out = out / np.clip(out.sum(), 1e-12, None)
        return out

    # Compute final row-level 5-day probabilities given state weights.
    def _predict_rows_given_state_probs(self, rows_d: pd.DataFrame, q: np.ndarray, fit_bundle: Dict[str, Any]) -> np.ndarray:
        mats = self._build_matrices(rows_d, require_labels=False)
        K = int(self.config.num_states)
        p_final = np.zeros(len(rows_d), dtype=float)

        for s in range(K):
            p20 = fit_bundle["models20"][s].predict_proba(mats["X20"])[:, 1]
            p5_y0 = fit_bundle["models5"][s].predict_proba(mats["X5_y0"])[:, 1]
            p5_y1 = fit_bundle["models5"][s].predict_proba(mats["X5_y1"])[:, 1]
            p5 = (1.0 - p20) * p5_y0 + p20 * p5_y1
            p_final += float(q[s]) * p5

        return np.clip(p_final, self.config.prob_clip, 1.0 - self.config.prob_clip)
