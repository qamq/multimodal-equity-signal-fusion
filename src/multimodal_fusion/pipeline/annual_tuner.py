"""
multimodal_fusion.pipeline.annual_tuner

Annual walk-forward random search for ensemble-weight methods.

Purpose
-------
This module runs the yearly ensemble-tuning workflow used by the portfolio runner.

The tuner is intentionally method-agnostic:
- method construction is delegated to EnsembleManager,
- candidate generation is delegated to search_spaces.py,
- validation scoring is delegated to the portfolio engine.

Protocol
--------
For each prediction year t:
1) optionally reuse the previous year's winning hyperparameters as the incumbent,
2) generate a candidate set that includes the incumbent, local perturbations,
   and fresh global random draws,
3) score candidates on the completed calibration year t-1 using a walk-forward
   historical fit on all rows through t-1,
4) freeze the chosen hyperparameters for the full year t,
5) fit the method on all data through year t and keep only the year-t predictions.

Important note
--------------
The final year-t prediction step assumes the underlying weighting class is internally
leakage-safe when fit on a full historical panel and then asked for predictions on rows
that were part of that fitted history. That is true for the date-level and online
classes in this project, and RLPolicy is handled by returning stored historical row-level
weights for fitted rows.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import json

import numpy as np
import pandas as pd

from multimodal_fusion.experimental.search_spaces import default_budget_for_method, sample_candidates
from multimodal_fusion.experimental.manager import EnsembleManager, EnsembleManagerConfig


@dataclass
class AnnualRandomSearchConfig:
    """
    AnnualRandomSearchConfig stores the settings for annual walk-forward tuning.

    It defines the target years, the random-search budget and seed, the calibration-year
    scoring setup, and the portfolio context needed to score each candidate with annualized
    H-L Sharpe.
    """

    method: str
    base_kwargs: Optional[Dict[str, Any]] = None
    start_year: int = 2016
    end_year: int = 2024
    first_tuned_year: Optional[int] = None
    random_seed: int = 1729
    search_budget: Optional[int] = None
    score_weight_type: str = "ew"
    score_cut: int = 10
    score_delay: int = 0
    freq: str = "week"
    country: str = "USA"
    portfolio_dir: str = "."
    verbose: bool = True


# Run annual walk-forward tuning and return the stitched prediction panel plus yearly logs.
def run_annual_random_search(panel: pd.DataFrame, cfg: AnnualRandomSearchConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dfx = panel.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce").dt.normalize()
    dfx = dfx.dropna(subset=["Date", "StockID", "p_cnn", "p_rf"]).copy()
    dfx["StockID"] = dfx["StockID"].astype(str)

    if len(dfx) == 0:
        raise ValueError("Annual tuner received an empty panel.")

    years_avail = sorted(int(y) for y in dfx["Date"].dt.year.dropna().unique().tolist())
    if not years_avail:
        raise ValueError("Annual tuner found no valid years in the panel.")

    budget = int(cfg.search_budget) if cfg.search_budget is not None else int(default_budget_for_method(cfg.method))
    base_kwargs = dict(cfg.base_kwargs or {})

    # Default behavior: use the first completed panel year as the calibration year,
    # so the earliest tuned prediction year is min(years_avail) + 1.
    first_tuned_year = (
        int(cfg.first_tuned_year)
        if cfg.first_tuned_year is not None
        else max(int(cfg.start_year), min(years_avail) + 1)
    )

    yearly_signals: List[pd.DataFrame] = []
    yearly_log: List[Dict[str, Any]] = []
    incumbent_kwargs: Optional[Dict[str, Any]] = None

    for test_year in range(int(cfg.start_year), int(cfg.end_year) + 1):
        year_rows = dfx[dfx["Date"].dt.year == int(test_year)].copy()
        if len(year_rows) == 0:
            continue

        calibration_year: Optional[int] = None
        ranking: List[Dict[str, Any]] = []
        chosen_score: Optional[float] = None

        if int(test_year) < int(first_tuned_year):
            chosen_kwargs = dict(base_kwargs)
            mode = "untuned_warm_start"
        else:
            calibration_year = int(test_year) - 1
            if calibration_year not in years_avail:
                chosen_kwargs = dict(incumbent_kwargs or base_kwargs)
                mode = "no_calibration_year_fallback"
            else:
                candidates = sample_candidates(
                    cfg.method,
                    budget=budget,
                    random_seed=_seed_for_method_year(cfg.random_seed, cfg.method, calibration_year),
                    base_kwargs=base_kwargs,
                    incumbent_kwargs=incumbent_kwargs,
                )
                ranking = _score_candidates_on_calibration_year(
                    panel=dfx,
                    method=cfg.method,
                    candidates=candidates,
                    calibration_year=calibration_year,
                    score_weight_type=cfg.score_weight_type,
                    score_cut=cfg.score_cut,
                    score_delay=cfg.score_delay,
                    freq=cfg.freq,
                    country=cfg.country,
                    verbose=cfg.verbose,
                )
                if ranking:
                    ranking.sort(key=lambda item: item["score"], reverse=True)
                    chosen_score = float(ranking[0]["score"])
                    chosen_kwargs = dict(ranking[0]["params"])
                    mode = "annual_calibration_search"
                else:
                    chosen_kwargs = dict(incumbent_kwargs or base_kwargs)
                    mode = "fallback_to_incumbent_or_base"

        incumbent_kwargs = dict(chosen_kwargs)

        fit_history = dfx[dfx["Date"].dt.year <= int(test_year)].copy()
        target_rows = dfx[dfx["Date"].dt.year == int(test_year)].copy()
        pred_year = _fit_predict_split(fit_history, target_rows, cfg.method, chosen_kwargs)
        yearly_signals.append(pred_year)

        yearly_log.append(
            {
                "test_year": int(test_year),
                "calibration_year": None if calibration_year is None else int(calibration_year),
                "mode": mode,
                "search_budget": int(budget),
                "score_weight_type": str(cfg.score_weight_type),
                "chosen_score": None if chosen_score is None else float(chosen_score),
                "chosen_kwargs": chosen_kwargs,
                "incumbent_kwargs": None if incumbent_kwargs is None else dict(incumbent_kwargs),
                "top_candidates": ranking[:10],
            }
        )

        if cfg.verbose:
            print(
                f"[AnnualTuner] year={test_year} mode={mode} chosen_score={chosen_score} chosen_kwargs={chosen_kwargs}"
            )

    if not yearly_signals:
        raise ValueError("Annual tuner produced no yearly predictions.")

    signal = pd.concat(yearly_signals, axis=0, ignore_index=True)
    signal["Date"] = pd.to_datetime(signal["Date"], errors="coerce").dt.normalize()
    signal["StockID"] = signal["StockID"].astype(str)
    signal = signal.dropna(subset=["Date", "StockID", "up_prob"]).copy()
    signal = signal.drop_duplicates(["Date", "StockID"], keep="last")
    signal = signal.sort_values(["Date", "StockID"]).reset_index(drop=True)

    log_payload = {
        "method": str(cfg.method),
        "random_seed": int(cfg.random_seed),
        "first_tuned_year": int(first_tuned_year),
        "search_budget": int(budget),
        "score_weight_type": str(cfg.score_weight_type),
        "score_cut": int(cfg.score_cut),
        "score_delay": int(cfg.score_delay),
        "years": yearly_log,
    }
    _write_tuning_log(cfg.portfolio_dir, cfg.method, log_payload)
    return signal, log_payload


# Derive a deterministic seed for one method-year pair.
def _seed_for_method_year(master_seed: int, method: str, year: int) -> int:
    out = int(master_seed)
    for ch in f"{method}:{year}":
        out = (out * 131 + ord(ch)) % (2**31 - 1)
    return int(out)


# Fit on one history split and predict on a separate target split.
def _fit_predict_split(
    fit_history: pd.DataFrame,
    target_rows: pd.DataFrame,
    method: str,
    method_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    if len(target_rows) == 0:
        return pd.DataFrame(columns=["Date", "StockID", "up_prob"])

    mgr = EnsembleManager(EnsembleManagerConfig(method=method, method_kwargs=method_kwargs or {}))

    if hasattr(mgr.model, "fit") and ("y" in fit_history.columns):
        mgr.fit(fit_history)

    pred = target_rows.copy()
    pred["up_prob"] = mgr.predict(pred)
    return pred[["Date", "StockID", "up_prob"]].copy()


# Score a list of hyperparameter candidates on one completed calibration year.
def _score_candidates_on_calibration_year(
    *,
    panel: pd.DataFrame,
    method: str,
    candidates: List[Dict[str, Any]],
    calibration_year: int,
    score_weight_type: str,
    score_cut: int,
    score_delay: int,
    freq: str,
    country: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    # Revised rule: treat the completed year itself as a walk-forward calibration year.
    # Each candidate is fit on all rows through that year and then scored only on that
    # year's predictions, relying on the underlying class to remain leakage-safe within
    # the fitted history.
    fit_history = panel[panel["Date"].dt.year <= int(calibration_year)].copy()
    target_rows = panel[panel["Date"].dt.year == int(calibration_year)].copy()

    if len(fit_history) == 0 or len(target_rows) == 0:
        return []

    results: List[Dict[str, Any]] = []
    for idx, params in enumerate(candidates):
        try:
            pred = _fit_predict_split(fit_history, target_rows, method, params)
            score = _score_signal_with_sharpe(
                pred,
                freq=freq,
                country=country,
                weight_type=score_weight_type,
                cut=score_cut,
                delay=score_delay,
            )
            results.append({"candidate_idx": int(idx), "score": float(score), "params": params})
        except Exception as exc:
            if verbose:
                print(f"[AnnualTuner] calibration_year={calibration_year} candidate={idx} failed: {exc}")
    return results


# Score a signal panel with annualized H-L Sharpe.
def _score_signal_with_sharpe(
    signal_df: pd.DataFrame,
    *,
    freq: str,
    country: str,
    weight_type: str,
    cut: int,
    delay: int,
) -> float:
    from Scripts.Portfolio.portfolio import PortfolioManager

    if len(signal_df) == 0:
        return float("-inf")

    sig = signal_df.copy()
    sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce").dt.normalize()
    sig["StockID"] = sig["StockID"].astype(str)

    start_year = int(sig["Date"].dt.year.min())
    end_year = int(sig["Date"].dt.year.max())

    buf = io.StringIO()
    with redirect_stdout(buf):
        pm = PortfolioManager(
            signal_df=sig,
            freq=freq,
            portfolio_dir=".",
            start_year=start_year,
            end_year=end_year,
            country=country,
            delay_list=[int(delay)],
            load_signal=True,
            tradability_screens=False,
            include_price_adv=False,
            verbose=False,
        )
        pf_ret, _ = pm.calculate_portfolio_rets(weight_type=weight_type, cut=int(cut), delay=int(delay))

    if "H-L" not in pf_ret.columns:
        return float("-inf")

    hl = pd.to_numeric(pf_ret["H-L"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(hl) < 3:
        return float("-inf")

    mean_ret = float(hl.mean())
    std_ret = float(hl.std(ddof=1))
    if (not np.isfinite(mean_ret)) or (not np.isfinite(std_ret)) or std_ret <= 0:
        return float("-inf")

    period = 52 if str(freq) == "week" else 12 if str(freq) == "month" else 4
    return float((mean_ret * period) / (std_ret * np.sqrt(period)))


# Write the yearly tuning log to disk as JSON.
def _write_tuning_log(portfolio_dir: str, method: str, payload: Dict[str, Any]) -> None:
    out_dir = Path(portfolio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"annual_tuning_{str(method).lower().strip()}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
