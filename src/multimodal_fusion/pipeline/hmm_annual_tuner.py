"""
hmm_annual_tuner.py

Annual expanding-window HMM tuner built on top of the existing HMM stack.

Purpose
-------
This file reuses the project's current multi-horizon HMM candidate generation
and scoring helpers, but changes the outer tuning protocol to an annual
expanding-window design.

For prediction year y:
- only data through y-1 may influence hyperparameter choice,
- candidate scores are averaged across the last N completed years ending at y-1,
- the chosen parameters are frozen before the year-y prediction run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
import json

import pandas as pd

from multimodal_fusion.core.hmm_labels import mask_unrealized_hmm_labels


# Import the existing HMM tuning helpers from the project.
def _import_hmm_tuning_stack():
    try:
        from multimodal_fusion.pipeline.hmm_tuner import (  # type: ignore
            HMMVersion1TuningConfig,
            build_hmm_panel,
            evaluate_hmm_candidate,
            ranking_to_frame,
            sample_hmm_candidates,
            summarize_hyperparameter_effects,
        )
        return {
            "HMMVersion1TuningConfig": HMMVersion1TuningConfig,
            "build_hmm_panel": build_hmm_panel,
            "evaluate_hmm_candidate": evaluate_hmm_candidate,
            "ranking_to_frame": ranking_to_frame,
            "sample_hmm_candidates": sample_hmm_candidates,
            "summarize_hyperparameter_effects": summarize_hyperparameter_effects,
        }
    except Exception:
        pass

    try:
        from hmm_tuner import (  # type: ignore
            HMMVersion1TuningConfig,
            build_hmm_panel,
            evaluate_hmm_candidate,
            ranking_to_frame,
            sample_hmm_candidates,
            summarize_hyperparameter_effects,
        )
        return {
            "HMMVersion1TuningConfig": HMMVersion1TuningConfig,
            "build_hmm_panel": build_hmm_panel,
            "evaluate_hmm_candidate": evaluate_hmm_candidate,
            "ranking_to_frame": ranking_to_frame,
            "sample_hmm_candidates": sample_hmm_candidates,
            "summarize_hyperparameter_effects": summarize_hyperparameter_effects,
        }
    except Exception as exc:
        raise ImportError("Could not import the project's HMM tuning stack.") from exc


@dataclass
class HMMAnnualTuningConfig:
    """Configuration for annual expanding-window HMM tuning."""

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
    label_end_5d_col: str = "label_end_5d"
    label_end_20d_col: str = "label_end_20d"


# Create a directory if needed and return it as a Path.
def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Return the year-specific output directory.
def _year_dir(base_output_dir, target_year):
    return _ensure_dir(Path(base_output_dir) / "year_{}".format(int(target_year)))


# Build the base Version-1 HMM config consumed by the existing helper functions.
def _base_version1_cfg(cfg):
    stack = _import_hmm_tuning_stack()
    HMMVersion1TuningConfig = stack["HMMVersion1TuningConfig"]
    return HMMVersion1TuningConfig(
        rf_root=str(cfg.rf_root),
        cnn_root=str(cfg.cnn_root),
        output_dir=str(cfg.output_dir),
        rf_pred_path=cfg.rf_pred_path,
        cnn_pred_path=cfg.cnn_pred_path,
        feature_panel_path=cfg.feature_panel_path,
        feature_cols=None if cfg.feature_cols is None else list(cfg.feature_cols),
        country=str(cfg.country),
        freq=str(cfg.freq),
        label_threshold=float(cfg.label_threshold),
        verbose=bool(cfg.verbose),
        fwd_ret_5d_col=cfg.fwd_ret_5d_col,
        fwd_ret_20d_col=cfg.fwd_ret_20d_col,
        dev_start_year=int(cfg.train_start_year),
        dev_end_year=int(cfg.end_year),
        validation_years=None,
        lockbox_start_year=int(cfg.start_year),
        lockbox_end_year=int(cfg.end_year),
        random_seed=int(cfg.random_seed),
        search_budget=int(cfg.search_budget),
        score_weight_type=str(cfg.score_weight_type),
        score_cut=int(cfg.score_cut),
        score_delay=int(cfg.score_delay),
        num_states_grid=None if cfg.num_states_grid is None else list(cfg.num_states_grid),
        l2_grid=None if cfg.l2_grid is None else list(cfg.l2_grid),
        refit_freq_grid=None if cfg.refit_freq_grid is None else list(cfg.refit_freq_grid),
        train_window_years_grid=None if cfg.train_window_years_grid is None else list(cfg.train_window_years_grid),
        transition_smoothing_grid=None if cfg.transition_smoothing_grid is None else list(cfg.transition_smoothing_grid),
        max_iter_fixed=int(cfg.max_iter_fixed),
        tol_fixed=float(cfg.tol_fixed),
        prob_clip_fixed=float(cfg.prob_clip_fixed),
        label_lag_periods_fixed=int(cfg.label_lag_periods_fixed),
        min_history_dates_fixed=int(cfg.min_history_dates_fixed),
        random_state_fixed=int(cfg.random_state_fixed),
        label_end_5d_col=cfg.label_end_5d_col,
        label_end_20d_col=cfg.label_end_20d_col,
    )


# Return the available panel years in sorted order.
def _available_years(panel):
    years = pd.to_datetime(panel["Date"], errors="coerce").dt.year.dropna().astype(int).unique().tolist()
    return sorted(int(y) for y in years)


# Choose the last N completed validation years ending at y-1.
def select_validation_years(panel, target_year, lookback_years, min_year):
    hist_years = [
        int(y)
        for y in _available_years(panel)
        if int(y) < int(target_year) and int(y) >= int(min_year)
    ]
    if not hist_years:
        return []
    return sorted(hist_years[-int(lookback_years):])


# Build a deterministic default parameter dictionary for early warm-up years.
def _default_hmm_params(cfg):
    return {
        "num_states": int(list(cfg.num_states_grid)[0]) if cfg.num_states_grid is not None else 2,
        "l2": float(list(cfg.l2_grid)[0]) if cfg.l2_grid is not None else 1.0,
        "refit_freq": str(list(cfg.refit_freq_grid)[0]) if cfg.refit_freq_grid is not None else "quarter",
        "train_window_years": int(list(cfg.train_window_years_grid)[0]) if cfg.train_window_years_grid is not None else 3,
        "transition_smoothing": float(list(cfg.transition_smoothing_grid)[0]) if cfg.transition_smoothing_grid is not None else 1e-3,
    }


# Tune HMM hyperparameters for one target year using only history through y-1.
def tune_hmm_for_year(cfg, target_year, panel=None, context_cols=None):
    stack = _import_hmm_tuning_stack()
    build_hmm_panel = stack["build_hmm_panel"]
    evaluate_hmm_candidate = stack["evaluate_hmm_candidate"]
    ranking_to_frame = stack["ranking_to_frame"]
    sample_hmm_candidates = stack["sample_hmm_candidates"]
    summarize_hyperparameter_effects = stack["summarize_hyperparameter_effects"]

    if int(target_year) < int(cfg.start_year) or int(target_year) > int(cfg.end_year):
        raise ValueError("target_year is outside the configured annual tuning range.")

    base_cfg = _base_version1_cfg(cfg)
    if panel is None:
        panel = build_hmm_panel(base_cfg)

    full_panel = panel.copy()
    full_panel["Date"] = pd.to_datetime(full_panel["Date"], errors="coerce").dt.normalize()

    information_cutoff = pd.Timestamp(year=int(target_year), month=1, day=1)
    hist_panel = full_panel[full_panel["Date"] < information_cutoff].copy()
    hist_panel = mask_unrealized_hmm_labels(hist_panel, information_cutoff)

    if context_cols is None:
        source_for_context = hist_panel if len(hist_panel) > 0 else full_panel
        context_cols = [c for c in source_for_context.columns if str(c).startswith("ctx_")]

    if len(hist_panel) == 0:
        validation_years = []
    else:
        validation_years = select_validation_years(
            hist_panel,
            int(target_year),
            int(cfg.validation_lookback_years),
            int(cfg.start_year),
        )

    year_dir = _year_dir(cfg.output_dir, target_year)
    ranking = []

    if validation_years:
        tuned_cfg = _base_version1_cfg(cfg)
        tuned_cfg.output_dir = str(year_dir)
        tuned_cfg.dev_end_year = int(target_year) - 1
        tuned_cfg.validation_years = [int(y) for y in validation_years]
        tuned_cfg.random_seed = int(cfg.random_seed + 1009 * int(target_year))
        tuned_cfg.verbose = bool(cfg.verbose)

        candidates = sample_hmm_candidates(tuned_cfg)
        for idx, params in enumerate(candidates):
            result = evaluate_hmm_candidate(hist_panel, tuned_cfg, params, context_cols=context_cols)
            result["candidate_idx"] = int(idx)
            ranking.append(result)

        if not ranking:
            raise ValueError("All annual HMM tuning candidates failed for target_year={}.".format(int(target_year)))

        ranking.sort(key=lambda item: (float(item["mean_score"]), -float(item["std_score"])), reverse=True)
        best_params = dict(ranking[0]["params"])
        best_mean_score = float(ranking[0]["mean_score"])
    else:
        best_params = _default_hmm_params(cfg)
        best_mean_score = float("nan")

    ranking_df = ranking_to_frame(ranking)
    hyperparam_summary_df = summarize_hyperparameter_effects(ranking_df)

    ranking_fp = year_dir / "hmm_candidate_ranking.csv"
    summary_fp = year_dir / "hmm_hyperparam_summary.csv"
    best_fp = year_dir / "hmm_best_params.json"
    manifest_fp = year_dir / "hmm_annual_tuning_manifest.json"

    ranking_df.to_csv(ranking_fp, index=False)
    hyperparam_summary_df.to_csv(summary_fp, index=False)
    best_fp.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "target_year": int(target_year),
        "information_cutoff": str(information_cutoff.date()),
        "validation_years": [int(y) for y in validation_years],
        "best_params": best_params,
        "best_mean_score": best_mean_score,
        "num_candidates": int(len(ranking)),
        "config": {
            **asdict(cfg),
            "output_dir": str(cfg.output_dir),
            "feature_cols": None if cfg.feature_cols is None else list(cfg.feature_cols),
            "num_states_grid": None if cfg.num_states_grid is None else list(cfg.num_states_grid),
            "l2_grid": None if cfg.l2_grid is None else list(cfg.l2_grid),
            "refit_freq_grid": None if cfg.refit_freq_grid is None else list(cfg.refit_freq_grid),
            "train_window_years_grid": None if cfg.train_window_years_grid is None else list(cfg.train_window_years_grid),
            "transition_smoothing_grid": None if cfg.transition_smoothing_grid is None else list(cfg.transition_smoothing_grid),
        },
    }
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "target_year": int(target_year),
        "validation_years": [int(y) for y in validation_years],
        "best_params": best_params,
        "best_mean_score": best_mean_score,
        "ranking": ranking,
        "ranking_csv": str(ranking_fp),
        "hyperparam_summary_csv": str(summary_fp),
        "best_params_json": str(best_fp),
        "manifest_json": str(manifest_fp),
    }


# Tune HMM hyperparameters year by year across the configured annual range.
def tune_hmm_years(cfg, years=None, panel=None, context_cols=None):
    stack = _import_hmm_tuning_stack()
    build_hmm_panel = stack["build_hmm_panel"]

    if panel is None:
        panel = build_hmm_panel(_base_version1_cfg(cfg))

    run_years = list(range(int(cfg.start_year), int(cfg.end_year) + 1)) if years is None else [int(y) for y in years]
    return [
        tune_hmm_for_year(cfg, int(year), panel=panel, context_cols=context_cols)
        for year in run_years
    ]
