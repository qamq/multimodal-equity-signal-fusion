"""
hmm_annual_runner.py

Annual expanding-window HMM prediction runner.

Purpose
-------
This file uses the year-specific tuner from `hmm_annual_tuner.py`, then fits the
chosen HMM configuration on all data through the target year and keeps only the
walk-forward predictions that belong to the target year.

It accepts an optional overlap key table for API compatibility, but the annual
runner now writes only the full-year HMM signal because the current HMM panel is
already built on the RF/CNN overlap universe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence
import json

import numpy as np
import pandas as pd


# Import the year-specific HMM tuner from the project layout or the current folder.
def _import_hmm_annual_tuner():
    try:
        from .hmm_annual_tuner import HMMAnnualTuningConfig, _base_version1_cfg, _import_hmm_tuning_stack, _year_dir, tune_hmm_for_year  # type: ignore
        return HMMAnnualTuningConfig, _base_version1_cfg, _import_hmm_tuning_stack, _year_dir, tune_hmm_for_year
    except Exception:
        pass

    try:
        from hmm_annual_tuner import HMMAnnualTuningConfig, _base_version1_cfg, _import_hmm_tuning_stack, _year_dir, tune_hmm_for_year  # type: ignore
        return HMMAnnualTuningConfig, _base_version1_cfg, _import_hmm_tuning_stack, _year_dir, tune_hmm_for_year
    except Exception as exc:
        raise ImportError("Could not import hmm_annual_tuner helpers.") from exc


# Import the overlap restriction helpers.
def _import_overlap_utils():
    try:
        from .overlap_signal_utils import restrict_signal_to_keys, save_signal_frame  # type: ignore
        return restrict_signal_to_keys, save_signal_frame
    except Exception:
        pass

    try:
        from overlap_signal_utils import restrict_signal_to_keys, save_signal_frame  # type: ignore
        return restrict_signal_to_keys, save_signal_frame
    except Exception as exc:
        raise ImportError("Could not import overlap signal helpers.") from exc


# Import the HMM runtime stack from the project.
def _import_hmm_runtime_stack():
    try:
        from multimodal_fusion.core.hmm_multihorizon import HMMMultihorizon  # type: ignore
        from multimodal_fusion.pipeline.hmm_tuner import make_hmm_config, score_signal_with_sharpe  # type: ignore
        return HMMMultihorizon, make_hmm_config, score_signal_with_sharpe
    except Exception:
        pass

    try:
        from hmm_multihorizon import HMMMultihorizon  # type: ignore
        from hmm_tuner import make_hmm_config, score_signal_with_sharpe  # type: ignore
        return HMMMultihorizon, make_hmm_config, score_signal_with_sharpe
    except Exception as exc:
        raise ImportError("Could not import the HMM runtime stack from the project.") from exc


# Import the HMM diagnostics helpers.
def _import_hmm_diagnostics():
    from multimodal_fusion.diagnostics.hmm_diagnostics import HMMDiagnostics
    return HMMDiagnostics


@dataclass
class HMMAnnualRunConfig:
    """Configuration for annual HMM tune-and-predict runs."""

    rf_root: str
    cnn_root: str
    output_dir: str

    rf_pred_path: Optional[str] = None
    cnn_pred_path: Optional[str] = None

    feature_panel_path: Optional[str] = None
    feature_cols: Optional[Sequence[str]] = None

    country: str = "USA"
    freq: str = "week"
    label_threshold: float = 0.0
    verbose: bool = True

    fwd_ret_5d_col: Optional[str] = None
    fwd_ret_20d_col: Optional[str] = None

    start_year: int = 2000
    end_year: int = 2024
    train_start_year: int = 1993
    validation_lookback_years: int = 3

    random_seed: int = 1729
    search_budget: int = 16

    score_weight_type: str = "ew"
    score_cut: int = 10
    score_delay: int = 0

    num_states_grid: Optional[Sequence[int]] = None
    l2_grid: Optional[Sequence[float]] = None
    refit_freq_grid: Optional[Sequence[str]] = None
    train_window_years_grid: Optional[Sequence[int]] = None
    transition_smoothing_grid: Optional[Sequence[float]] = None

    max_iter_fixed: int = 20
    tol_fixed: float = 1e-4
    prob_clip_fixed: float = 1e-6
    label_lag_periods_fixed: int = 4
    min_history_dates_fixed: int = 26
    random_state_fixed: int = 42

    save_state_probs: bool = True
    save_fit_log: bool = True
    save_diagnostics: bool = True
    label_end_5d_col: str = "label_end_5d"
    label_end_20d_col: str = "label_end_20d"


# Compute year-y classification metrics directly from the HMM modeling panel.
def _compute_year_metrics(pred_df, truth_panel_year):
    from sklearn.metrics import log_loss, matthews_corrcoef

    pred = pred_df.copy()
    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.normalize()
    pred["StockID"] = pred["StockID"].astype(str)
    pred["up_prob"] = pd.to_numeric(pred["up_prob"], errors="coerce")
    pred = pred.dropna(subset=["Date", "StockID", "up_prob"]).copy()

    truth = truth_panel_year.copy()
    truth["Date"] = pd.to_datetime(truth["Date"], errors="coerce").dt.normalize()
    truth["StockID"] = truth["StockID"].astype(str)
    truth["y_5d"] = pd.to_numeric(truth["y_5d"], errors="coerce")
    truth = truth.dropna(subset=["Date", "StockID", "y_5d"]).copy()

    merged = pred.merge(truth[["Date", "StockID", "y_5d"]], on=["Date", "StockID"], how="inner")
    if len(merged) == 0:
        # Live forecasts can precede every outcome in the prediction year.
        # Missing evaluation labels must not prevent returning those forecasts.
        return {
            "rows": 0.0, "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0,
            "precision": float("nan"), "recall": float("nan"),
            "accuracy": float("nan"), "mcc": float("nan"), "log_loss": float("nan"),
        }

    y_true = merged["y_5d"].astype(int).to_numpy(dtype=np.int8)
    p_hat = np.clip(merged["up_prob"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
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

    return {
        "rows": float(len(merged)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "mcc": mcc,
        "log_loss": loss,
    }


# Tune and run the HMM for one target year.
def run_hmm_year(cfg, target_year, panel=None, context_cols=None, overlap_keys_df=None):
    _, _base_version1_cfg, _import_hmm_tuning_stack, _year_dir, tune_hmm_for_year = _import_hmm_annual_tuner()
    build_hmm_panel = _import_hmm_tuning_stack()["build_hmm_panel"]
    HMMMultihorizon, make_hmm_config, score_signal_with_sharpe = _import_hmm_runtime_stack()
    _, save_signal_frame = _import_overlap_utils()

    if panel is None:
        panel = build_hmm_panel(_base_version1_cfg(cfg))

    work_panel = panel.copy()
    work_panel["Date"] = pd.to_datetime(work_panel["Date"], errors="coerce").dt.normalize()
    if context_cols is None:
        context_cols = [c for c in work_panel.columns if str(c).startswith("ctx_")]

    tuning_payload = tune_hmm_for_year(cfg, int(target_year), panel=work_panel, context_cols=context_cols)
    best_params = dict(tuning_payload["best_params"])

    run_panel = work_panel[work_panel["Date"].dt.year <= int(target_year)].copy()
    if len(run_panel) == 0:
        raise ValueError("No HMM rows are available through target_year={}.".format(int(target_year)))

    version1_cfg = _base_version1_cfg(cfg)
    hmm_cfg = make_hmm_config(version1_cfg, best_params, context_cols=context_cols)
    model = HMMMultihorizon(config=hmm_cfg)
    model.fit(run_panel)

    pred = model.row_pred_.copy() if model.row_pred_ is not None else pd.DataFrame(columns=["Date", "StockID", "up_prob"])
    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.normalize()
    pred["StockID"] = pred["StockID"].astype(str)
    pred = pred[pred["Date"].dt.year == int(target_year)].copy()
    pred = pred.dropna(subset=["Date", "StockID", "up_prob"]).copy()
    pred = pred.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)

    year_dir = _year_dir(cfg.output_dir, target_year)
    full_paths = save_signal_frame(pred, year_dir, "hmm_preds_{}".format(int(target_year)))

    truth_year = work_panel[work_panel["Date"].dt.year == int(target_year)].copy()
    metrics = _compute_year_metrics(pred, truth_year)
    metrics["hl_sharpe_ew"] = float(
        score_signal_with_sharpe(
            pred,
            freq=str(cfg.freq),
            country=str(cfg.country),
            weight_type=str(cfg.score_weight_type),
            cut=int(cfg.score_cut),
            delay=int(cfg.score_delay),
        )
    )

    metrics_fp = year_dir / "hmm_metrics_{}.json".format(int(target_year))
    metrics_fp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if bool(cfg.save_state_probs) and model.state_probs_by_date_ is not None:
        state_fp = year_dir / "hmm_state_probs_{}.parquet".format(int(target_year))
        model.state_probs_by_date_.to_parquet(state_fp, index=False)
    else:
        state_fp = None

    if bool(cfg.save_fit_log):
        fit_log_fp = year_dir / "hmm_fit_log_{}.json".format(int(target_year))
        fit_log_fp.write_text(json.dumps(model.fit_log_, indent=2), encoding="utf-8")
    else:
        fit_log_fp = None

    diagnostics_paths = None
    diagnostics_summary = None
    if bool(cfg.save_diagnostics) and metrics["rows"] > 0:
        HMMDiagnostics = _import_hmm_diagnostics()
        diagnostics_payload = HMMDiagnostics.build_year_diagnostics(
            model,
            pred,
            truth_year,
            target_year=int(target_year),
        )
        diagnostics_summary = dict(diagnostics_payload.get("summary", {}))
        diagnostics_summary["hl_sharpe_ew"] = float(metrics["hl_sharpe_ew"])
        diagnostics_paths = HMMDiagnostics.save_year_diagnostics(year_dir, int(target_year), diagnostics_payload)
        if bool(cfg.verbose):
            HMMDiagnostics.print_year_diagnostics(diagnostics_summary)

    manifest = {
        "target_year": int(target_year),
        "best_params": best_params,
        "context_cols": list(context_cols),
        "prediction_full": full_paths,
        "metrics_json": str(metrics_fp),
        "state_probs_parquet": None if state_fp is None else str(state_fp),
        "fit_log_json": None if fit_log_fp is None else str(fit_log_fp),
        "diagnostics": diagnostics_paths,
        "tuning_manifest_json": str(tuning_payload["manifest_json"]),
        "config": {
            **asdict(cfg),
            "feature_cols": None if cfg.feature_cols is None else list(cfg.feature_cols),
            "num_states_grid": None if cfg.num_states_grid is None else list(cfg.num_states_grid),
            "l2_grid": None if cfg.l2_grid is None else list(cfg.l2_grid),
            "refit_freq_grid": None if cfg.refit_freq_grid is None else list(cfg.refit_freq_grid),
            "train_window_years_grid": None if cfg.train_window_years_grid is None else list(cfg.train_window_years_grid),
            "transition_smoothing_grid": None if cfg.transition_smoothing_grid is None else list(cfg.transition_smoothing_grid),
        },
    }
    manifest_fp = year_dir / "hmm_year_run_manifest.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "target_year": int(target_year),
        "pred_df": pred,
        "pred_overlap_df": None,
        "metrics": metrics,
        "metrics_overlap": None,
        "diagnostics_summary": diagnostics_summary,
        "manifest_json": str(manifest_fp),
        "tuning_payload": tuning_payload,
    }


# Run the HMM annual tune-and-predict workflow across multiple years.
def run_hmm_years(cfg, years=None, panel=None, context_cols=None, overlap_keys_by_year=None):
    _, _base_version1_cfg, _import_hmm_tuning_stack, _, _ = _import_hmm_annual_tuner()
    build_hmm_panel = _import_hmm_tuning_stack()["build_hmm_panel"]

    if panel is None:
        panel = build_hmm_panel(_base_version1_cfg(cfg))

    run_years = list(range(int(cfg.start_year), int(cfg.end_year) + 1)) if years is None else [int(y) for y in years]
    out = []
    for year in run_years:
        overlap_keys_df = None if overlap_keys_by_year is None else overlap_keys_by_year.get(int(year))
        out.append(run_hmm_year(cfg, int(year), panel=panel, context_cols=context_cols, overlap_keys_df=overlap_keys_df))
    return out
