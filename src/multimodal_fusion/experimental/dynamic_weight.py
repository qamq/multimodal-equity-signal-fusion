"""
multimodal_fusion.experimental.dynamic_weight

DynamicWeight: online walk-forward weights for blending CNN and RF probabilities.

Weight form
-----------
For each update period t, compute a scalar CNN weight w_t in [0,1] and blend:
    up_prob(i,t) = w_t * p_cnn(i,t) + (1-w_t) * p_rf(i,t)

Learning rule
-------------
We compute per-date (or per-month/per-quarter) model scores, then update weights online.
For period t, we first incorporate realized outcomes from t-label_lag, then convert the
cumulative CNN vs RF score history into w_t using a softmax with temperature temp.

Objectives (score functions)
----------------------------
Base:
  - logloss

Tail-focused (3):
  - tail_quantile_logloss: score only on cross-sectional tails (top/bottom tail_q each date)
  - tail_margin_weighted_logloss: weight by |avg(p)-0.5|^margin_gamma each date
  - focal_logloss: focal loss with focal_gamma

Ranking-based (3):
  - rank_auc: AUC vs y each date
  - rank_spearman_ic: Spearman IC vs fwd_ret each date
  - rank_decile_spread: top-minus-bottom mean fwd_ret (top/bottom spread_q each date)

Update frequency
----------------
update_freq controls when weights are updated:
  - "week": one weight per weekly date
  - "month": one weight per month (applied to all weekly dates in that month)
  - "quarter": one weight per quarter

Notes
-----
- Ranking objectives require 'fwd_ret' to be present (provided by ensemble_pipeline when attach_labels=True).
- With update_freq="week" and label_lag=1, week t uses all realized outcomes through week t-1.
- lookback_periods=None gives a fully cumulative online learner; a positive integer gives a bounded online window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DynamicWeight:
    """DynamicWeight learns online time-varying scalar weights for blending p_cnn and p_rf."""

    lookback_periods: Optional[int] = None
    label_lag: int = 1
    update_freq: str = "week"  # "week" | "month" | "quarter"

    objective: str = "logloss"
    temp: float = 0.10
    eps: float = 1e-6

    tail_q: float = 0.10
    margin_gamma: float = 2.0
    focal_gamma: float = 2.0

    ret_col: str = "fwd_ret"
    spread_q: float = 0.10

    weights_by_date_: Optional[pd.DataFrame] = None
    global_w_cnn_: float = 0.5

    # Fit per-date weights using online cumulative score updates.
    def fit(self, df: pd.DataFrame, *, date_col: str = "Date", y_col: str = "y") -> "DynamicWeight":
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        need = self._required_columns(date_col=date_col, y_col=y_col)
        miss = [c for c in need if c not in dfx.columns]
        if miss:
            raise KeyError("DynamicWeight objective='%s' missing columns: %s" % (self.objective, miss))

        dfx = dfx.dropna(subset=need).copy()
        if len(dfx) == 0:
            self.weights_by_date_ = pd.DataFrame(columns=["w_cnn", "w_rf"])
            self.global_w_cnn_ = 0.5
            return self

        score_by_date = self._scores_by_date(dfx, date_col=date_col, y_col=y_col)
        period_scores = self._collapse_scores(score_by_date)
        w_period = self._online_weights(period_scores)
        w_by_date = self._expand_period_weights(score_by_date.index, w_period)

        wdf = pd.DataFrame({"w_cnn": w_by_date})
        wdf["w_rf"] = 1.0 - wdf["w_cnn"]
        self.weights_by_date_ = wdf
        self.global_w_cnn_ = float(wdf["w_cnn"].iloc[-1]) if len(wdf) else 0.5
        return self

    # Predict ensemble probabilities using learned weights.
    def predict(self, df: pd.DataFrame, *, date_col: str = "Date") -> pd.Series:
        dfx = df.copy()
        dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()

        w = self._weights_for_rows(dfx, date_col=date_col)
        p1 = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).fillna(0.5)
        p2 = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).fillna(0.5)

        p = w * p1 + (1.0 - w) * p2
        p = p.clip(self.eps, 1.0 - self.eps)
        return pd.Series(p.to_numpy(), index=df.index, name="up_prob")

    # Return a copy of the learned weights.
    def get_weights(self) -> pd.DataFrame:
        if self.weights_by_date_ is not None:
            return self.weights_by_date_.copy()
        return pd.DataFrame({"w_cnn": [self.global_w_cnn_], "w_rf": [1.0 - self.global_w_cnn_]})

    # Determine which columns are required for the chosen objective.
    def _required_columns(self, *, date_col: str, y_col: str) -> list:
        obj = str(self.objective).lower().strip()
        base = [date_col, "p_cnn", "p_rf"]

        if obj in ("logloss", "tail_quantile_logloss", "tail_margin_weighted_logloss", "focal_logloss", "rank_auc"):
            return base + [y_col]

        if obj in ("rank_spearman_ic", "rank_decile_spread"):
            return base + [self.ret_col]

        raise ValueError("Unknown objective='%s'" % self.objective)

    # Compute per-date model scores according to the objective.
    def _scores_by_date(self, dfx: pd.DataFrame, *, date_col: str, y_col: str) -> pd.DataFrame:
        obj = str(self.objective).lower().strip()

        dfx["p_cnn"] = pd.to_numeric(dfx["p_cnn"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)
        dfx["p_rf"] = pd.to_numeric(dfx["p_rf"], errors="coerce").astype(float).clip(self.eps, 1.0 - self.eps)

        if obj in ("logloss", "tail_quantile_logloss", "tail_margin_weighted_logloss", "focal_logloss", "rank_auc"):
            dfx[y_col] = pd.to_numeric(dfx[y_col], errors="coerce").astype(float)

        if obj in ("rank_spearman_ic", "rank_decile_spread"):
            dfx[self.ret_col] = pd.to_numeric(dfx[self.ret_col], errors="coerce").astype(float)

        g = dfx.set_index(date_col).groupby(level=0, sort=True)

        if obj == "logloss":
            s_cnn = g.apply(lambda sub: -float(np.mean(self._logloss(sub["p_cnn"].to_numpy(), sub[y_col].to_numpy()))))
            s_rf = g.apply(lambda sub: -float(np.mean(self._logloss(sub["p_rf"].to_numpy(), sub[y_col].to_numpy()))))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "tail_quantile_logloss":
            s_cnn = g.apply(lambda sub: self._tail_quantile_score(sub, pcol="p_cnn", y_col=y_col))
            s_rf = g.apply(lambda sub: self._tail_quantile_score(sub, pcol="p_rf", y_col=y_col))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "tail_margin_weighted_logloss":
            s_cnn = g.apply(lambda sub: self._margin_weighted_score(sub, pcol="p_cnn", y_col=y_col))
            s_rf = g.apply(lambda sub: self._margin_weighted_score(sub, pcol="p_rf", y_col=y_col))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "focal_logloss":
            s_cnn = g.apply(lambda sub: -float(np.mean(self._focal_loss(sub["p_cnn"].to_numpy(), sub[y_col].to_numpy(), self.focal_gamma))))
            s_rf = g.apply(lambda sub: -float(np.mean(self._focal_loss(sub["p_rf"].to_numpy(), sub[y_col].to_numpy(), self.focal_gamma))))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "rank_auc":
            s_cnn = g.apply(lambda sub: self._auc(sub[y_col].to_numpy(), sub["p_cnn"].to_numpy()))
            s_rf = g.apply(lambda sub: self._auc(sub[y_col].to_numpy(), sub["p_rf"].to_numpy()))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "rank_spearman_ic":
            s_cnn = g.apply(lambda sub: self._spearman_ic(sub["p_cnn"].to_numpy(), sub[self.ret_col].to_numpy()))
            s_rf = g.apply(lambda sub: self._spearman_ic(sub["p_rf"].to_numpy(), sub[self.ret_col].to_numpy()))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        if obj == "rank_decile_spread":
            s_cnn = g.apply(lambda sub: self._decile_spread(sub["p_cnn"].to_numpy(), sub[self.ret_col].to_numpy(), self.spread_q))
            s_rf = g.apply(lambda sub: self._decile_spread(sub["p_rf"].to_numpy(), sub[self.ret_col].to_numpy(), self.spread_q))
            return pd.DataFrame({"s_cnn": s_cnn, "s_rf": s_rf})

        raise ValueError("Unknown objective='%s'" % self.objective)

    # Compute logloss per observation.
    def _logloss(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.clip(p.astype(float), self.eps, 1.0 - self.eps)
        y = y.astype(float)
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # Compute focal loss per observation.
    def _focal_loss(self, p: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        p = np.clip(p.astype(float), self.eps, 1.0 - self.eps)
        y = y.astype(float)
        pt = np.where(y >= 0.5, p, 1.0 - p)
        return -((1.0 - pt) ** float(gamma)) * np.log(pt)

    # Score tail quantile logloss on a single date.
    def _tail_quantile_score(self, sub: pd.DataFrame, *, pcol: str, y_col: str) -> float:
        p1 = sub["p_cnn"].to_numpy(dtype=float)
        p2 = sub["p_rf"].to_numpy(dtype=float)
        p_avg = 0.5 * (p1 + p2)

        q = float(self.tail_q)
        lo = float(np.quantile(p_avg, q))
        hi = float(np.quantile(p_avg, 1.0 - q))
        m = (p_avg <= lo) | (p_avg >= hi)
        if not np.any(m):
            return 0.0

        p = sub[pcol].to_numpy(dtype=float)[m]
        y = sub[y_col].to_numpy(dtype=float)[m]
        return -float(np.mean(self._logloss(p, y)))

    # Score margin-weighted logloss on a single date.
    def _margin_weighted_score(self, sub: pd.DataFrame, *, pcol: str, y_col: str) -> float:
        p1 = sub["p_cnn"].to_numpy(dtype=float)
        p2 = sub["p_rf"].to_numpy(dtype=float)
        p_avg = 0.5 * (p1 + p2)
        w = np.abs(p_avg - 0.5) ** float(self.margin_gamma)
        if float(np.sum(w)) <= 0:
            return 0.0

        p = sub[pcol].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        ll = self._logloss(p, y)
        return -float(np.sum(w * ll) / np.sum(w))

    # Compute Spearman IC between two arrays.
    def _spearman_ic(self, a: np.ndarray, r: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        r = np.asarray(r, dtype=float)
        m = np.isfinite(a) & np.isfinite(r)
        a = a[m]
        r = r[m]
        if a.size < 3:
            return 0.0
        ra = pd.Series(a).rank(method="average").to_numpy()
        rr = pd.Series(r).rank(method="average").to_numpy()
        if np.std(ra) < 1e-12 or np.std(rr) < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rr)[0, 1])

    # Compute top-bottom spread based on probability quantiles.
    def _decile_spread(self, p: np.ndarray, r: np.ndarray, q: float) -> float:
        p = np.asarray(p, dtype=float)
        r = np.asarray(r, dtype=float)
        m = np.isfinite(p) & np.isfinite(r)
        p = p[m]
        r = r[m]
        if p.size < 10:
            return 0.0
        lo = float(np.quantile(p, q))
        hi = float(np.quantile(p, 1.0 - q))
        top = r[p >= hi]
        bot = r[p <= lo]
        if top.size == 0 or bot.size == 0:
            return 0.0
        return float(np.mean(top) - np.mean(bot))

    # Compute AUC using rank statistic (Mann-Whitney form).
    def _auc(self, y: np.ndarray, s: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        m = np.isfinite(y) & np.isfinite(s)
        y = y[m]
        s = s[m]
        if y.size < 10:
            return 0.5
        n1 = float(np.sum(y >= 0.5))
        n0 = float(np.sum(y < 0.5))
        if n1 < 1 or n0 < 1:
            return 0.5
        ranks = pd.Series(s).rank(method="average").to_numpy()
        rank_sum_pos = float(np.sum(ranks[y >= 0.5]))
        auc = (rank_sum_pos - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)
        return float(np.clip(auc, 0.0, 1.0))

    # Collapse per-date scores to update periods.
    def _collapse_scores(self, score_by_date: pd.DataFrame) -> pd.DataFrame:
        freq = str(self.update_freq).lower().strip()
        s = score_by_date.copy()
        s.index = pd.to_datetime(s.index)

        if freq == "week":
            return s.sort_index()

        if freq == "month":
            key = s.index.to_period("M")
            return s.groupby(key)[["s_cnn", "s_rf"]].mean()

        if freq == "quarter":
            key = s.index.to_period("Q")
            return s.groupby(key)[["s_cnn", "s_rf"]].mean()

        raise ValueError("update_freq must be one of: 'week','month','quarter'.")

    # Compute online weights from cumulative or bounded score history.
    def _online_weights(self, period_scores: pd.DataFrame) -> pd.Series:
        s = period_scores.sort_index()
        out = []
        w_prev = 0.5

        hist_cnn = deque()
        hist_rf = deque()
        sum_cnn = 0.0
        sum_rf = 0.0

        max_hist = None if self.lookback_periods is None else int(self.lookback_periods)
        if max_hist is not None and max_hist <= 0:
            max_hist = None

        for i, k in enumerate(s.index):
            j = i - int(self.label_lag)

            # Update first using the most recent realized period that should be available.
            if j >= 0:
                sc_new = float(s.iloc[j]["s_cnn"])
                sr_new = float(s.iloc[j]["s_rf"])

                if np.isfinite(sc_new) and np.isfinite(sr_new):
                    hist_cnn.append(sc_new)
                    hist_rf.append(sr_new)
                    sum_cnn += sc_new
                    sum_rf += sr_new

                    if max_hist is not None:
                        while len(hist_cnn) > max_hist:
                            sum_cnn -= hist_cnn.popleft()
                            sum_rf -= hist_rf.popleft()

            # Convert the current score history into the weight used for period k.
            n_hist = len(hist_cnn)
            if n_hist > 0:
                sc = sum_cnn / float(n_hist)
                sr = sum_rf / float(n_hist)
                w_prev = self._softmax_weight(sc, sr)

            out.append((k, w_prev))

        return pd.Series({k: v for k, v in out}, name="w_cnn")

    # Expand period weights back to each Date in the weekly series.
    def _expand_period_weights(self, date_index: pd.Index, w_period: pd.Series) -> pd.Series:
        freq = str(self.update_freq).lower().strip()
        dates = pd.to_datetime(pd.Index(date_index))

        if freq == "week":
            w = dates.map(w_period).astype(float).fillna(self.global_w_cnn_)
            return pd.Series(w.to_numpy(), index=dates, name="w_cnn")

        if freq == "month":
            key = dates.to_period("M")
            w = key.map(w_period).astype(float).fillna(self.global_w_cnn_)
            return pd.Series(w.to_numpy(), index=dates, name="w_cnn")

        if freq == "quarter":
            key = dates.to_period("Q")
            w = key.map(w_period).astype(float).fillna(self.global_w_cnn_)
            return pd.Series(w.to_numpy(), index=dates, name="w_cnn")

        return pd.Series(self.global_w_cnn_, index=dates, name="w_cnn")

    # Convert two scores into a CNN weight using softmax temperature.
    def _softmax_weight(self, score_cnn: float, score_rf: float) -> float:
        t = max(float(self.temp), 1e-8)
        a = np.exp(float(score_cnn) / t)
        b = np.exp(float(score_rf) / t)
        s = float(a + b)
        return 0.5 if (s <= 0 or not np.isfinite(s)) else float(a / s)

    # Map per-date weights onto each row in df.
    def _weights_for_rows(self, dfx: pd.DataFrame, *, date_col: str) -> pd.Series:
        if self.weights_by_date_ is None or len(self.weights_by_date_) == 0:
            return pd.Series(self.global_w_cnn_, index=dfx.index, dtype=float)

        dd = pd.to_datetime(dfx[date_col], errors="coerce").dt.normalize()
        w = dd.map(self.weights_by_date_["w_cnn"]).astype(float).fillna(self.global_w_cnn_)
        return pd.Series(w.to_numpy(), index=dfx.index, dtype=float)