
"""
hmm_diagnostics.py

Diagnostics helpers for the annual multi-horizon HMM workflow.

This module keeps reporting logic separate from model fitting so we can inspect:
- classification quality and calibration,
- hidden-state usage and transition stability,
- emission-model coefficients for interpretation,
- year-level regime profiles,
- and fit-convergence behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd


# Clip probabilities into a numerically stable open interval.
def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


# Compute a numerically stable logit transform.
def _logit(p: float, eps: float = 1e-6) -> float:
    q = float(np.clip(float(p), eps, 1.0 - eps))
    return float(np.log(q / (1.0 - q)))


# Return a scalar mean if finite values exist.
def _safe_mean(x: pd.Series) -> float:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


# Return a scalar quantile if finite values exist.
def _safe_quantile(x: pd.Series, q: float) -> float:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.quantile(float(q))) if len(s) else float("nan")


# Return a JSON-safe finite float or None.
def _json_float(x: Any) -> Optional[float]:
    try:
        val = float(x)
    except Exception:
        return None
    return val if np.isfinite(val) else None


# Normalize Date/StockID fields and de-duplicate.
def _standardize_keys(df: pd.DataFrame, *, prob_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["StockID"] = pd.to_numeric(out["StockID"], errors="coerce")
    keep = ["Date", "StockID"] + ([] if prob_col is None else [prob_col])
    out = out.dropna(subset=["Date", "StockID"] + ([] if prob_col is None else [prob_col])).copy()
    out["StockID"] = out["StockID"].astype(int).astype(str)
    return out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)


# Compute the stationary distribution of a transition matrix.
def _stationary_distribution(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:, idx])
    v = np.clip(v, 0.0, None)
    if float(v.sum()) <= 0.0:
        v = np.ones(A.shape[0], dtype=float)
    v = v / np.clip(v.sum(), 1e-12, None)
    return v.astype(float)


# Build a compact top-feature summary from a coefficient table.
def _top_feature_summary(coef_df: pd.DataFrame, *, top_n: int = 3) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    if coef_df is None or len(coef_df) == 0:
        return {}
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    work = coef_df.loc[coef_df["feature"] != "intercept"].copy()
    if len(work) == 0:
        return out
    for state, sdf in work.groupby("state"):
        pos = sdf.sort_values("coef", ascending=False).head(int(top_n))
        neg = sdf.sort_values("coef", ascending=True).head(int(top_n))
        out[f"state_{int(state)}"] = {
            "top_positive": [
                {"feature": str(r["feature"]), "coef": _json_float(r["coef"]), "odds_ratio": _json_float(r["odds_ratio"])}
                for _, r in pos.iterrows()
            ],
            "top_negative": [
                {"feature": str(r["feature"]), "coef": _json_float(r["coef"]), "odds_ratio": _json_float(r["odds_ratio"])}
                for _, r in neg.iterrows()
            ],
        }
    return out


class HMMDiagnostics:
    """Diagnostics-only helpers for the annual multi-horizon HMM workflow."""

    # Return the model's final fitted bundle when available.
    @staticmethod
    def get_fit_bundle(model: Any) -> Optional[Dict[str, Any]]:
        if hasattr(model, "get_fit_bundle"):
            return model.get_fit_bundle()
        return getattr(model, "_last_fit_bundle", None)

    # Return the model's context columns in final fitted order.
    @staticmethod
    def get_context_cols(model: Any) -> List[str]:
        if hasattr(model, "get_context_cols"):
            return list(model.get_context_cols())
        return list(getattr(model, "context_cols_", []) or [])

    # Merge yearly predictions with yearly truth and modeling inputs.
    @staticmethod
    def merge_pred_truth(pred_df: pd.DataFrame, truth_panel_year: pd.DataFrame) -> pd.DataFrame:
        pred = _standardize_keys(pred_df, prob_col="up_prob")
        pred["up_prob"] = pd.to_numeric(pred["up_prob"], errors="coerce")

        truth = truth_panel_year.copy()
        truth["Date"] = pd.to_datetime(truth["Date"], errors="coerce").dt.normalize()
        truth["StockID"] = pd.to_numeric(truth["StockID"], errors="coerce")
        truth = truth.dropna(subset=["Date", "StockID"]).copy()
        truth["StockID"] = truth["StockID"].astype(int).astype(str)

        keep_cols = [
            c for c in [
                "Date", "StockID", "p_cnn", "p_rf", "y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d", "sector_id", "log_mcap"
            ] if c in truth.columns
        ]
        merged = pred.merge(truth[keep_cols], on=["Date", "StockID"], how="inner")
        merged = merged.dropna(subset=["up_prob"]).copy()
        return merged.sort_values(["Date", "StockID"]).reset_index(drop=True)

    # Compute annual classification and probability diagnostics.
    @staticmethod
    def summarize_prediction_quality(pred_df: pd.DataFrame, truth_panel_year: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score

        merged = HMMDiagnostics.merge_pred_truth(pred_df, truth_panel_year)
        if len(merged) == 0:
            raise ValueError("No overlapping rows between HMM predictions and yearly truth panel.")
        if "y_5d" not in merged.columns:
            raise KeyError("Yearly truth panel must contain y_5d for HMM diagnostics.")

        y_true = pd.to_numeric(merged["y_5d"], errors="coerce").astype(int).to_numpy(dtype=np.int8)
        p_hat = _clip_prob(pd.to_numeric(merged["up_prob"], errors="coerce").to_numpy(dtype=float), 1e-6)
        y_hat = (p_hat >= 0.5).astype(np.int8)

        tp = int(np.sum((y_hat == 1) & (y_true == 1)))
        tn = int(np.sum((y_hat == 0) & (y_true == 0)))
        fp = int(np.sum((y_hat == 1) & (y_true == 0)))
        fn = int(np.sum((y_hat == 0) & (y_true == 1)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        accuracy = float(np.mean(y_hat == y_true))
        mcc = float(matthews_corrcoef(y_true, y_hat)) if len(np.unique(y_true)) > 1 else float("nan")
        loss = float(log_loss(y_true, p_hat, labels=[0, 1]))
        brier = float(np.mean(np.square(p_hat - y_true)))
        auc = float(roc_auc_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else float("nan")

        metrics = {
            "rows": int(len(merged)),
            "n_dates": int(merged["Date"].nunique()),
            "n_stocks": int(merged["StockID"].nunique()),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "mcc": mcc,
            "log_loss": loss,
            "brier": brier,
            "auc": auc,
            "pred_pos_rate": float(np.mean(y_hat)),
            "realized_pos_rate": float(np.mean(y_true)),
            "up_prob_mean": float(np.mean(p_hat)),
            "up_prob_std": float(np.std(p_hat, ddof=1)) if len(p_hat) > 1 else float("nan"),
            "up_prob_p10": float(np.quantile(p_hat, 0.10)),
            "up_prob_p50": float(np.quantile(p_hat, 0.50)),
            "up_prob_p90": float(np.quantile(p_hat, 0.90)),
            "cnn_rf_gap_mean": _safe_mean((pd.to_numeric(merged.get("p_cnn"), errors="coerce") - pd.to_numeric(merged.get("p_rf"), errors="coerce")).abs()) if {"p_cnn", "p_rf"}.issubset(merged.columns) else float("nan"),
        }
        return metrics, merged

    # Build a simple fixed-bin calibration table.
    @staticmethod
    def calibration_table(merged: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
        if len(merged) == 0 or "y_5d" not in merged.columns:
            return pd.DataFrame(columns=["bin", "count", "mean_pred", "realized_rate", "gap"])
        work = merged[["up_prob", "y_5d"]].copy()
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f})" if i < len(edges) - 2 else f"[{edges[i]:.1f}, {edges[i+1]:.1f}]" for i in range(len(edges) - 1)]
        work["bin"] = pd.cut(work["up_prob"], bins=edges, labels=labels, include_lowest=True, right=False)
        cal = (
            work.groupby("bin", observed=False)
            .agg(
                count=("y_5d", "count"),
                mean_pred=("up_prob", "mean"),
                realized_rate=("y_5d", "mean"),
            )
            .reset_index()
        )
        cal["gap"] = cal["realized_rate"] - cal["mean_pred"]
        return cal

    # Extract state transition diagnostics from the final fitted bundle.
    @staticmethod
    def transition_diagnostics(model: Any) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        bundle = HMMDiagnostics.get_fit_bundle(model)
        if not bundle:
            empty = pd.DataFrame()
            return empty, empty, {}

        A = np.asarray(bundle["A"], dtype=float)
        pi = np.asarray(bundle["pi"], dtype=float)
        K = int(A.shape[0])
        stationary = _stationary_distribution(A)

        trans_rows = []
        state_rows = []
        for i in range(K):
            pii = float(A[i, i])
            duration = float("inf") if pii >= 1.0 - 1e-12 else float(1.0 / max(1e-12, 1.0 - pii))
            state_rows.append(
                {
                    "state": int(i),
                    "pi": float(pi[i]),
                    "self_transition": pii,
                    "expected_duration_dates": duration,
                    "stationary_prob": float(stationary[i]),
                }
            )
            for j in range(K):
                trans_rows.append({"from_state": int(i), "to_state": int(j), "prob": float(A[i, j])})

        state_df = pd.DataFrame(state_rows).sort_values("state").reset_index(drop=True)
        trans_df = pd.DataFrame(trans_rows).sort_values(["from_state", "to_state"]).reset_index(drop=True)
        summary = {
            "num_states": K,
            "mean_self_transition": float(np.mean(np.diag(A))),
            "max_self_transition": float(np.max(np.diag(A))),
            "min_self_transition": float(np.min(np.diag(A))),
        }
        return trans_df, state_df, summary

    # Extract coefficient tables from the state-specific emission models.
    @staticmethod
    def emission_coefficient_tables(model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bundle = HMMDiagnostics.get_fit_bundle(model)
        if not bundle:
            return pd.DataFrame(), pd.DataFrame()

        context_cols = list(bundle.get("context_cols", []) or HMMDiagnostics.get_context_cols(model))
        feat20 = ["logit_p_rf"] + context_cols
        feat5 = ["logit_p_cnn", "logit_p_rf", "y20_proxy"] + context_cols

        def _extract_rows(models: List[Any], feature_names: List[str], block_name: str) -> pd.DataFrame:
            rows: List[Dict[str, Any]] = []
            for state, mdl in enumerate(models):
                if hasattr(mdl, "coef_") and hasattr(mdl, "intercept_"):
                    coef = np.asarray(mdl.coef_, dtype=float).reshape(-1)
                    intercept = float(np.asarray(mdl.intercept_, dtype=float).reshape(-1)[0])
                    rows.append(
                        {
                            "state": int(state),
                            "block": block_name,
                            "feature": "intercept",
                            "coef": intercept,
                            "odds_ratio": float(np.exp(intercept)),
                            "model_type": type(mdl).__name__,
                            "constant_prob": np.nan,
                        }
                    )
                    for feat, val in zip(feature_names, coef):
                        rows.append(
                            {
                                "state": int(state),
                                "block": block_name,
                                "feature": str(feat),
                                "coef": float(val),
                                "odds_ratio": float(np.exp(val)),
                                "model_type": type(mdl).__name__,
                                "constant_prob": np.nan,
                            }
                        )
                else:
                    p_one = float(getattr(mdl, "p_one", np.nan))
                    rows.append(
                        {
                            "state": int(state),
                            "block": block_name,
                            "feature": "intercept",
                            "coef": _logit(p_one) if np.isfinite(p_one) else np.nan,
                            "odds_ratio": float(np.exp(_logit(p_one))) if np.isfinite(p_one) else np.nan,
                            "model_type": type(mdl).__name__,
                            "constant_prob": p_one,
                        }
                    )
                    for feat in feature_names:
                        rows.append(
                            {
                                "state": int(state),
                                "block": block_name,
                                "feature": str(feat),
                                "coef": 0.0,
                                "odds_ratio": 1.0,
                                "model_type": type(mdl).__name__,
                                "constant_prob": p_one,
                            }
                        )
            out = pd.DataFrame(rows)
            if len(out):
                out["abs_coef"] = pd.to_numeric(out["coef"], errors="coerce").abs()
                out = out.sort_values(["state", "feature"]).reset_index(drop=True)
            return out

        coef20 = _extract_rows(list(bundle.get("models20", [])), feat20, "20d")
        coef5 = _extract_rows(list(bundle.get("models5", [])), feat5, "5d")
        return coef20, coef5

    # Summarize fit-convergence diagnostics from the final fit block.
    @staticmethod
    def fit_diagnostics(model: Any) -> Dict[str, Any]:
        fit_log = list(getattr(model, "fit_log_", []) or [])
        if not fit_log:
            return {}
        last = fit_log[-1]
        ll_trace = [float(x) for x in last.get("ll_trace", [])]
        final_ll = float(ll_trace[-1]) if ll_trace else float("nan")
        prev_ll = float(ll_trace[-2]) if len(ll_trace) >= 2 else float("nan")
        last_delta = abs(final_ll - prev_ll) if len(ll_trace) >= 2 else float("nan")
        tol = float(getattr(getattr(model, "config", None), "tol", np.nan))
        return {
            "fit_blocks": int(len(fit_log)),
            "last_block_num_dates": int(last.get("num_dates", 0)),
            "last_block_num_rows": int(last.get("num_rows", 0)),
            "last_block_iterations": int(len(ll_trace)),
            "last_block_final_loglik": final_ll,
            "last_block_prev_loglik": prev_ll,
            "last_block_last_delta_loglik": last_delta,
            "last_block_converged": bool(np.isfinite(last_delta) and np.isfinite(tol) and last_delta <= tol),
            "tolerance": tol,
        }

    # Summarize the hidden-state sequence over the predicted year.
    @staticmethod
    def state_sequence_diagnostics(state_probs_by_date: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        if state_probs_by_date is None or len(state_probs_by_date) == 0:
            return pd.DataFrame(), {}, pd.DataFrame()

        sp = state_probs_by_date.copy()
        sp["Date"] = pd.to_datetime(sp["Date"], errors="coerce").dt.normalize()
        sp["state"] = pd.to_numeric(sp["state"], errors="coerce").astype("Int64")
        sp["prob"] = pd.to_numeric(sp["prob"], errors="coerce")
        sp = sp.dropna(subset=["Date", "state", "prob"]).copy()
        if len(sp) == 0:
            return pd.DataFrame(), {}, pd.DataFrame()

        pivot = sp.pivot_table(index="Date", columns="state", values="prob", aggfunc="mean").sort_index()
        if len(pivot) == 0:
            return pd.DataFrame(), {}, pd.DataFrame()

        hard_state = pivot.idxmax(axis=1).astype(int)
        probs = np.clip(pivot.to_numpy(dtype=float), 1e-12, None)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        if pivot.shape[1] > 1:
            entropy = entropy / np.log(float(pivot.shape[1]))
        num_switches = int(np.sum(hard_state.to_numpy()[1:] != hard_state.to_numpy()[:-1])) if len(hard_state) > 1 else 0
        switch_rate = float(num_switches / max(len(hard_state) - 1, 1)) if len(hard_state) > 1 else 0.0

        state_rows = []
        mean_prob = pivot.mean(axis=0)
        hard_counts = hard_state.value_counts().sort_index()
        for state in pivot.columns:
            state_rows.append(
                {
                    "state": int(state),
                    "mean_prob": float(mean_prob.loc[state]),
                    "hard_count_dates": int(hard_counts.get(state, 0)),
                    "hard_share_dates": float(hard_counts.get(state, 0) / len(pivot)),
                }
            )

        state_df = pd.DataFrame(state_rows).sort_values("state").reset_index(drop=True)
        hard_path = pd.DataFrame({"Date": pivot.index, "hard_state": hard_state.to_numpy(dtype=int), "state_entropy": entropy})
        summary = {
            "pred_year_dates": int(len(pivot)),
            "hard_state_switches": num_switches,
            "hard_state_switch_rate": switch_rate,
            "mean_state_entropy": float(np.mean(entropy)) if len(entropy) else float("nan"),
            "max_state_entropy": float(np.max(entropy)) if len(entropy) else float("nan"),
        }
        return state_df, summary, hard_path

    # Build row-level profiles for each dominant hidden state during the predicted year.
    @staticmethod
    def state_profile_table(state_probs_by_date: Optional[pd.DataFrame], pred_df: pd.DataFrame, truth_panel_year: pd.DataFrame) -> pd.DataFrame:
        if state_probs_by_date is None or len(state_probs_by_date) == 0:
            return pd.DataFrame()
        merged = HMMDiagnostics.merge_pred_truth(pred_df, truth_panel_year)
        if len(merged) == 0:
            return pd.DataFrame()

        _, _, hard_path = HMMDiagnostics.state_sequence_diagnostics(state_probs_by_date)
        if len(hard_path) == 0:
            return pd.DataFrame()

        work = merged.merge(hard_path[["Date", "hard_state"]], on="Date", how="left")
        if {"p_cnn", "p_rf"}.issubset(work.columns):
            work["abs_gap"] = (pd.to_numeric(work.get("p_cnn"), errors="coerce") - pd.to_numeric(work.get("p_rf"), errors="coerce")).abs()

        agg = {
            "Date": pd.Series.nunique,
            "StockID": "count",
            "up_prob": "mean",
        }
        for c in ["p_cnn", "p_rf", "abs_gap", "y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d", "log_mcap"]:
            if c in work.columns:
                agg[c] = "mean"

        prof = work.groupby("hard_state").agg(agg).reset_index().rename(columns={"Date": "n_dates", "StockID": "n_rows"})
        if "y_5d" in prof.columns:
            prof = prof.rename(columns={"y_5d": "mean_y_5d", "y_20d": "mean_y_20d"})
        return prof.sort_values("hard_state").reset_index(drop=True)

    # Build one diagnostics payload for a single annual run.
    @staticmethod
    def build_year_diagnostics(model: Any, pred_df: pd.DataFrame, truth_panel_year: pd.DataFrame, *, target_year: int) -> Dict[str, Any]:
        quality_summary, merged = HMMDiagnostics.summarize_prediction_quality(pred_df, truth_panel_year)
        calibration_df = HMMDiagnostics.calibration_table(merged)
        transition_df, transition_state_df, transition_summary = HMMDiagnostics.transition_diagnostics(model)
        coef20_df, coef5_df = HMMDiagnostics.emission_coefficient_tables(model)
        fit_summary = HMMDiagnostics.fit_diagnostics(model)
        state_probs = getattr(model, "state_probs_by_date_", None)
        state_usage_df, state_usage_summary, hard_path_df = HMMDiagnostics.state_sequence_diagnostics(state_probs)
        state_profile_df = HMMDiagnostics.state_profile_table(state_probs, pred_df, truth_panel_year)

        summary = {
            "target_year": int(target_year),
            **quality_summary,
            **fit_summary,
            **transition_summary,
            **state_usage_summary,
            "coef20_top_features": _top_feature_summary(coef20_df),
            "coef5_top_features": _top_feature_summary(coef5_df),
        }
        return {
            "summary": summary,
            "merged_pred_truth": merged,
            "calibration_table": calibration_df,
            "transition_matrix": transition_df,
            "transition_state_summary": transition_state_df,
            "emission_coef_20d": coef20_df,
            "emission_coef_5d": coef5_df,
            "state_usage_summary": state_usage_df,
            "state_hard_path": hard_path_df,
            "state_profile": state_profile_df,
        }

    # Write one annual diagnostics payload to disk and return file paths.
    @staticmethod
    def save_year_diagnostics(year_dir: Path, target_year: int, payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
        year_dir = Path(year_dir)
        year_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Optional[str]] = {}

        summary_fp = year_dir / f"hmm_diagnostics_summary_{int(target_year)}.json"
        summary_fp.write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
        paths["summary_json"] = str(summary_fp)

        file_map = {
            "merged_pred_truth": f"hmm_pred_truth_merge_{int(target_year)}.parquet",
            "calibration_table": f"hmm_calibration_{int(target_year)}.csv",
            "transition_matrix": f"hmm_transition_matrix_{int(target_year)}.csv",
            "transition_state_summary": f"hmm_transition_state_summary_{int(target_year)}.csv",
            "emission_coef_20d": f"hmm_emission_coef_20d_{int(target_year)}.csv",
            "emission_coef_5d": f"hmm_emission_coef_5d_{int(target_year)}.csv",
            "state_usage_summary": f"hmm_state_usage_summary_{int(target_year)}.csv",
            "state_hard_path": f"hmm_state_hard_path_{int(target_year)}.csv",
            "state_profile": f"hmm_state_profile_{int(target_year)}.csv",
        }
        for key, fname in file_map.items():
            df = payload.get(key)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                fp = year_dir / fname
                if str(fp).lower().endswith(".parquet"):
                    df.to_parquet(fp, index=False)
                else:
                    df.to_csv(fp, index=False)
                paths[key] = str(fp)
            else:
                paths[key] = None
        return paths

    # Print a compact console summary for one annual HMM run.
    @staticmethod
    def print_year_diagnostics(summary: Dict[str, Any]) -> None:
        if not summary:
            return
        print(
            f"[HMM Diagnostics | {int(summary['target_year'])}] rows={int(summary.get('rows', 0)):,} "
            f"dates={int(summary.get('n_dates', 0)):,} log_loss={float(summary.get('log_loss', np.nan)):.4f} "
            f"brier={float(summary.get('brier', np.nan)):.4f} auc={float(summary.get('auc', np.nan)):.4f}"
        )
        if "last_block_iterations" in summary:
            print(
                "[HMM Diagnostics] fit "
                f"blocks={int(summary.get('fit_blocks', 0))} | last_iters={int(summary.get('last_block_iterations', 0))} "
                f"| converged={bool(summary.get('last_block_converged', False))} "
                f"| final_loglik={float(summary.get('last_block_final_loglik', np.nan)):.3f}"
            )
        if "mean_self_transition" in summary:
            print(
                "[HMM Diagnostics] state dynamics "
                f"mean_self_transition={float(summary.get('mean_self_transition', np.nan)):.3f} "
                f"| switch_rate={float(summary.get('hard_state_switch_rate', np.nan)):.3f} "
                f"| mean_entropy={float(summary.get('mean_state_entropy', np.nan)):.3f}"
            )
