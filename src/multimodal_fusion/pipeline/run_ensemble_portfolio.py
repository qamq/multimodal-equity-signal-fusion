"""
multimodal_fusion.pipeline.run_ensemble_portfolio

This module is the main ensemble-to-portfolio entrypoint. It loads CNN and RF
predictions, optionally attaches labels, optionally runs annual nested random
search to choose ensemble hyperparameters, builds the final `up_prob` signal,
and passes that signal into `PortfolioManager` for portfolio construction.

The file keeps the older DynamicWeight-specific tuner for backward
compatibility, but the recommended path is the method-agnostic annual tuner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from multimodal_fusion.pipeline.annual_tuner import AnnualRandomSearchConfig, run_annual_random_search
from multimodal_fusion.pipeline.ensemble_pipeline import EnsemblePipelineConfig, EnsembleSignalPipeline


@dataclass
class RunEnsemblePortfolioConfig:
    """
    RunEnsemblePortfolioConfig stores the inputs for one ensemble portfolio run.

    It bundles prediction sources, method settings, walk-forward evaluation
    dates, label-attachment controls, portfolio settings, and the optional
    annual random-search configuration.
    """

    rf_root: str
    cnn_root: str
    portfolio_dir: str

    rf_pred_path: Optional[str] = None
    cnn_pred_path: Optional[str] = None

    method: str = "fixed_blend"
    method_kwargs: Optional[Dict[str, Any]] = None

    freq: str = "week"
    country: str = "USA"
    start_year: int = 2016
    end_year: int = 2024
    delay_list: Optional[List[int]] = None

    attach_labels: bool = False
    label_threshold: float = 0.0
    label_ret_col: Optional[str] = None

    cut: int = 10
    delay: int = 0
    verbose: bool = True

    tradability_screens: bool = False
    min_price: float = 1.0
    min_adv_dollar: Optional[float] = 3_000_000
    include_price_adv: bool = False
    warm_period_ret_cache: bool = True

    portfolio_kwargs: Optional[Dict[str, Any]] = None
    portfolio_weight_types: Optional[List[str]] = None

    tune_dynamic: bool = False
    tune_start_year: int = 2016
    tune_end_year: int = 2019
    tune_valid_years: Optional[List[int]] = None
    tune_grid: Optional[List[Dict[str, Any]]] = None

    annual_random_search: bool = False
    annual_first_tuned_year: Optional[int] = None
    annual_search_budget: Optional[int] = None
    annual_random_seed: int = 1729
    annual_score_weight_type: str = "ew"
    annual_score_cut: int = 10
    annual_score_delay: int = 0


# Build the ensemble signal, optionally tune it, and run the requested portfolios.
def run_ensemble_portfolio(cfg: RunEnsemblePortfolioConfig) -> pd.DataFrame:
    from Scripts.Portfolio.portfolio import PortfolioManager

    portfolio_dir = Path(cfg.portfolio_dir)
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    portfolio_kwargs, include_price_adv = _resolve_portfolio_kwargs(cfg)

    if include_price_adv and cfg.warm_period_ret_cache:
        _warm_period_return_cache(freq=cfg.freq, country=cfg.country, verbose=cfg.verbose)

    pipe_cfg = EnsemblePipelineConfig(
        rf_root=cfg.rf_root,
        cnn_root=cfg.cnn_root,
        rf_pred_path=cfg.rf_pred_path,
        cnn_pred_path=cfg.cnn_pred_path,
        attach_labels=cfg.attach_labels or cfg.tune_dynamic or cfg.annual_random_search,
        freq=cfg.freq,
        country=cfg.country,
        label_threshold=cfg.label_threshold,
        label_ret_col=cfg.label_ret_col,
        verbose=cfg.verbose,
        keep_label_cols_in_signal=False,
    )
    pipe = EnsembleSignalPipeline(pipe_cfg)

    if cfg.annual_random_search:
        panel = pipe.build_model_frame()
        annual_cfg = AnnualRandomSearchConfig(
            method=cfg.method,
            base_kwargs=dict(cfg.method_kwargs or {}),
            start_year=int(cfg.start_year),
            end_year=int(cfg.end_year),
            first_tuned_year=cfg.annual_first_tuned_year,
            random_seed=int(cfg.annual_random_seed),
            search_budget=cfg.annual_search_budget,
            score_weight_type=str(cfg.annual_score_weight_type),
            score_cut=int(cfg.annual_score_cut),
            score_delay=int(cfg.annual_score_delay),
            freq=cfg.freq,
            country=cfg.country,
            portfolio_dir=str(portfolio_dir),
            verbose=cfg.verbose,
        )
        signal, _tuning_log = run_annual_random_search(panel, annual_cfg)
        if cfg.verbose:
            print(
                "[run_ensemble_portfolio] annual_random_search complete: method=%s years=%d-%d"
                % (cfg.method, cfg.start_year, cfg.end_year)
            )
    else:
        best_kwargs = dict(cfg.method_kwargs or {})
        if cfg.tune_dynamic and str(cfg.method).lower().strip() == "dynamic_weight":
            best_kwargs = _tune_dynamic_weight(pipe, cfg, base_kwargs=best_kwargs)
        signal = pipe.build_signal_df(method=cfg.method, method_kwargs=best_kwargs).reset_index()

    sig_df = signal.reset_index(drop=True) if not isinstance(signal.index, pd.MultiIndex) else signal.reset_index()

    pm = PortfolioManager(
        signal_df=sig_df,
        freq=cfg.freq,
        portfolio_dir=str(portfolio_dir),
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        country=cfg.country,
        delay_list=cfg.delay_list if cfg.delay_list is not None else [0],
        load_signal=True,
        **portfolio_kwargs,
    )

    pm.generate_portfolio(cut=cfg.cut, delay=cfg.delay, weight_types=cfg.portfolio_weight_types)
    out = sig_df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["StockID"] = out["StockID"].astype(str)
    out = out.set_index(["Date", "StockID"]).sort_index()
    return out


# Resolve PortfolioManager kwargs for tradability and hygiene toggles.
def _resolve_portfolio_kwargs(cfg: RunEnsemblePortfolioConfig) -> Tuple[Dict[str, Any], bool]:
    out = dict(cfg.portfolio_kwargs or {})

    include_price_adv = bool(cfg.include_price_adv)
    tradability_screens = bool(cfg.tradability_screens)

    if tradability_screens and not include_price_adv:
        include_price_adv = True
        if cfg.verbose:
            print("[run_ensemble_portfolio] tradability_screens=True -> forcing include_price_adv=True")

    out.update(
        {
            "tradability_screens": tradability_screens,
            "min_price": float(cfg.min_price),
            "min_adv_dollar": None if cfg.min_adv_dollar is None else float(cfg.min_adv_dollar),
            "include_price_adv": include_price_adv,
        }
    )

    return out, include_price_adv


# Materialize the Price/ADV cache once when tradability-aware runs need it.
def _warm_period_return_cache(*, freq: str, country: str, verbose: bool) -> None:
    from Scripts.Data import equity_data as eqd

    if verbose:
        print(f"[run_ensemble_portfolio] warming {country} {freq} period-return cache with Price/ADV")

    eqd.get_period_ret(
        freq,
        country=country,
        include_price_adv=True,
        require_price_adv=True,
    )


# Tune DynamicWeight on a frozen historical window for backward compatibility.
def _tune_dynamic_weight(
    pipe: EnsembleSignalPipeline,
    cfg: RunEnsemblePortfolioConfig,
    *,
    base_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    from multimodal_fusion.experimental.dynamic_weight import DynamicWeight

    panel = pipe.build_model_frame()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["Date", "p_cnn", "p_rf", "y"]).copy()

    train_mask = (
        (panel["Date"].dt.year >= int(cfg.tune_start_year))
        & (panel["Date"].dt.year <= int(cfg.tune_end_year))
    )
    train_panel = panel.loc[train_mask].copy()

    if len(train_panel) == 0:
        raise ValueError("No rows available in tuning window %d-%d." % (cfg.tune_start_year, cfg.tune_end_year))

    valid_years = cfg.tune_valid_years
    if valid_years is None:
        year_1 = int(cfg.tune_end_year) - 1
        year_2 = int(cfg.tune_end_year)
        valid_years = [year for year in (year_1, year_2) if year >= int(cfg.tune_start_year)]

    grid = cfg.tune_grid if cfg.tune_grid is not None else _default_dynamic_grid(base_kwargs)

    results: List[Tuple[float, Dict[str, Any]]] = []
    for candidate in grid:
        params = dict(base_kwargs)
        params.update(candidate)

        scores: List[float] = []
        for valid_year in valid_years:
            train_end = int(valid_year) - 1
            tr = train_panel[train_panel["Date"].dt.year <= train_end].copy()
            va = train_panel[train_panel["Date"].dt.year == int(valid_year)].copy()

            if len(tr) == 0 or len(va) == 0:
                continue

            dynamic_weight = DynamicWeight(**params)
            dynamic_weight.fit(tr, date_col="Date", y_col="y")
            p_hat = dynamic_weight.predict(va, date_col="Date")
            score = _eval_ensemble_score(va, p_hat, objective=str(params.get("objective", "logloss")))
            scores.append(score)

        if scores:
            results.append((float(np.mean(scores)), params))

    if not results:
        raise ValueError("Tuning produced no valid fold results; check tune window and data coverage.")

    results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_params = results[0]
    _write_best_params(cfg.portfolio_dir, best_score, best_params, results[:25])

    if cfg.verbose:
        print("[DynamicWeightTuner] best_score=%.6f best_params=%s" % (best_score, best_params))

    return best_params


# Provide a small default grid for the legacy DynamicWeight tuner.
def _default_dynamic_grid(base_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    update_freq = str(base_kwargs.get("update_freq", "week"))

    if update_freq == "month":
        lookbacks = [12, 24, 36]
    elif update_freq == "quarter":
        lookbacks = [8, 12, 16]
    else:
        lookbacks = [26, 52, 104]

    temps = [0.05, 0.10, 0.20]

    grid: List[Dict[str, Any]] = []
    for lookback in lookbacks:
        for temp in temps:
            grid.append({"lookback_periods": lookback, "temp": temp, "label_lag": 1})

    if "objective" not in base_kwargs:
        for objective in (
            "logloss",
            "tail_quantile_logloss",
            "tail_margin_weighted_logloss",
            "focal_logloss",
            "rank_auc",
        ):
            for lookback in lookbacks:
                for temp in temps:
                    grid.append({"objective": objective, "lookback_periods": lookback, "temp": temp, "label_lag": 1})

    return grid


# Score ensemble predictions on a validation slice for the legacy DynamicWeight tuner.
def _eval_ensemble_score(df_valid: pd.DataFrame, p_hat: pd.Series, *, objective: str) -> float:
    objective_name = str(objective).lower().strip()
    prob = pd.to_numeric(p_hat, errors="coerce").to_numpy(dtype=float)
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

    if objective_name in ("rank_spearman_ic", "rank_decile_spread"):
        if "fwd_ret" not in df_valid.columns:
            return 0.0
        fwd_ret = pd.to_numeric(df_valid["fwd_ret"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(prob) & np.isfinite(fwd_ret)
        prob = prob[mask]
        fwd_ret = fwd_ret[mask]
        if prob.size < 3:
            return 0.0
        rank_prob = pd.Series(prob).rank().to_numpy()
        rank_ret = pd.Series(fwd_ret).rank().to_numpy()
        if np.std(rank_prob) < 1e-12 or np.std(rank_ret) < 1e-12:
            return 0.0
        return float(np.corrcoef(rank_prob, rank_ret)[0, 1])

    labels = pd.to_numeric(df_valid["y"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(prob) & np.isfinite(labels)
    prob = prob[mask]
    labels = labels[mask]
    if prob.size < 10:
        return 0.0

    log_loss = float(np.mean(-(labels * np.log(prob) + (1.0 - labels) * np.log(1.0 - prob))))
    return -log_loss


# Write the best legacy DynamicWeight parameters to disk as JSON.
def _write_best_params(
    portfolio_dir: str,
    best_score: float,
    best_params: Dict[str, Any],
    top_results: List[Tuple[float, Dict[str, Any]]],
) -> None:
    out = {
        "best_score": float(best_score),
        "best_params": best_params,
        "top_results": [{"score": float(score), "params": params} for score, params in top_results],
    }
    out_dir = Path(portfolio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "dynamic_weight_best_params.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
