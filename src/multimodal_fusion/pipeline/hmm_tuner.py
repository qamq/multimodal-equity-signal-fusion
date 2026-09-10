
"""
multimodal_fusion.pipeline.hmm_tuner

Purpose
-------
This module implements the version-1 hyperparameter-tuning workflow for the
latent-state multi-horizon HMM ensemble.

Design philosophy
-----------------
- Tune only on the pre-2016 development sample.
- Freeze one final HMM configuration before the 2016+ lockbox period.
- Keep the HMM probability structure fixed.
- Use a small, auditable random-search space over only the HMM knobs that
  materially affect model behavior.
- Avoid forcing the HMM through the older generic ensemble runner, because
  the HMM needs the richer multi-horizon panel built by HMMDataBuilder.

Workflow
--------
1) Build the HMM modeling panel with HMMDataBuilder.
2) Restrict tuning to the development window.
3) Sample a small number of HMM candidates from a bounded random-search space.
4) Score each candidate on rolling pre-2016 validation years using annualized
   H-L Sharpe from the portfolio engine.
5) Save the winning frozen configuration, the full candidate ranking, and
   summary tables for downstream inspection.
6) Optionally run the frozen HMM across the full sample and return 2016+
   lockbox predictions for final evaluation.

Notes
-----
- This file is intentionally standalone. It does not modify the current
  generic annual_tuner.py / run_ensemble_portfolio.py stack.
- The HMM itself is still adaptive inside fit(...): hidden-state inference,
  rolling refits, and state propagation remain active even when the outer
  hyperparameters are frozen.
- For validation years, this tuner mirrors the intended live protocol by
  fitting the HMM on all rows through the target year and then extracting
  that year's internally generated walk-forward predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, Dict, Iterable, List, Optional
import json
import io
import warnings

import numpy as np
import pandas as pd

from multimodal_fusion.pipeline.hmm_data import HMMDataBuilder, HMMDataConfig
from multimodal_fusion.core.hmm_multihorizon import HMMMultihorizon, HMMMultihorizonConfig
from multimodal_fusion.core.hmm_labels import mask_unrealized_hmm_labels, realized_outcome_mask

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class HMMVersion1TuningConfig:
    """
    HMMVersion1TuningConfig

    This configuration stores the file paths, sample splits, search-space
    bounds, and portfolio scoring settings for pre-2016 HMM tuning.
    """

    rf_root: str
    cnn_root: str
    output_dir: str

    rf_pred_path: Optional[str] = None
    cnn_pred_path: Optional[str] = None

    feature_panel_path: Optional[str] = None
    feature_cols: Optional[List[str]] = None

    country: str = "USA"
    freq: str = "week"
    label_threshold: float = 0.0
    verbose: bool = True

    fwd_ret_5d_col: Optional[str] = None
    fwd_ret_20d_col: Optional[str] = None

    dev_start_year: int = 2000
    dev_end_year: int = 2015
    validation_years: Optional[List[int]] = None

    lockbox_start_year: int = 2016
    lockbox_end_year: int = 2024

    random_seed: int = 1729
    search_budget: int = 16

    score_weight_type: str = "ew"
    score_cut: int = 10
    score_delay: int = 0

    num_states_grid: Optional[List[int]] = None
    l2_grid: Optional[List[float]] = None
    refit_freq_grid: Optional[List[str]] = None
    train_window_years_grid: Optional[List[int]] = None
    transition_smoothing_grid: Optional[List[float]] = None

    max_iter_fixed: int = 20
    tol_fixed: float = 1e-4
    prob_clip_fixed: float = 1e-6
    label_lag_periods_fixed: int = 4
    min_history_dates_fixed: int = 26
    random_state_fixed: int = 42
    label_end_5d_col: str = "label_end_5d"
    label_end_20d_col: str = "label_end_20d"


# Wrap an iterable in a progress bar when tqdm is available and verbose output is enabled.
def _progress(
    iterable: Iterable[Any],
    *,
    total: Optional[int],
    desc: str,
    enabled: bool,
    leave: bool = True,
) -> Iterable[Any]:
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=leave)
    return iterable


# Build the richer stock-date panel required by the HMM model.
def build_hmm_panel(cfg: HMMVersion1TuningConfig) -> pd.DataFrame:
    builder = HMMDataBuilder(
        HMMDataConfig(
            rf_root=cfg.rf_root,
            cnn_root=cfg.cnn_root,
            rf_pred_path=cfg.rf_pred_path,
            cnn_pred_path=cfg.cnn_pred_path,
            feature_panel_path=cfg.feature_panel_path,
            feature_cols=cfg.feature_cols,
            freq=cfg.freq,
            country=cfg.country,
            label_threshold=cfg.label_threshold,
            verbose=cfg.verbose,
            fwd_ret_5d_col=cfg.fwd_ret_5d_col,
            fwd_ret_20d_col=cfg.fwd_ret_20d_col,
            label_end_5d_col=cfg.label_end_5d_col,
            label_end_20d_col=cfg.label_end_20d_col,
        )
    )
    panel = builder.build_model_frame()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel["StockID"] = panel["StockID"].astype(str)
    panel = panel.dropna(subset=["Date", "StockID", "p_cnn", "p_rf"]).copy()
    panel = panel.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)
    return panel


# Return the bounded HMM search space used in version 1.
def default_hmm_search_space(cfg: HMMVersion1TuningConfig) -> Dict[str, List[Any]]:
    return {
        "num_states": list(cfg.num_states_grid or [2, 3, 4]),
        "l2": list(cfg.l2_grid or [0.1, 0.3, 1.0, 3.0, 10.0]),
        "refit_freq": list(cfg.refit_freq_grid or ["quarter"]),
        "train_window_years": list(cfg.train_window_years_grid or [2, 3, 4, 5]),
        "transition_smoothing": list(cfg.transition_smoothing_grid or [1e-4, 1e-3, 1e-2]),
    }


# Build one fully specified HMM config from the sampled search-space knobs.
def make_hmm_config(
    cfg: HMMVersion1TuningConfig,
    params: Dict[str, Any],
    *,
    context_cols: Optional[List[str]] = None,
) -> HMMMultihorizonConfig:
    return HMMMultihorizonConfig(
        num_states=int(params["num_states"]),
        max_iter=int(cfg.max_iter_fixed),
        tol=float(cfg.tol_fixed),
        l2=float(params["l2"]),
        prob_clip=float(cfg.prob_clip_fixed),
        refit_freq=str(params["refit_freq"]),
        train_window_years=int(params["train_window_years"]),
        label_lag_periods=int(cfg.label_lag_periods_fixed),
        min_history_dates=int(cfg.min_history_dates_fixed),
        transition_smoothing=float(params["transition_smoothing"]),
        context_cols=None if context_cols is None else list(context_cols),
        random_state=int(cfg.random_state_fixed),
        verbose=False,
        show_progress=False,
        progress_desc=None,
        progress_leave=False,
        progress_bar=None,
    )


# Draw a small deduplicated random candidate set from the version-1 HMM space.
def sample_hmm_candidates(cfg: HMMVersion1TuningConfig) -> List[Dict[str, Any]]:
    space = default_hmm_search_space(cfg)
    budget = int(cfg.search_budget)
    rng = np.random.RandomState(int(cfg.random_seed))

    candidates: List[Dict[str, Any]] = []
    seen = set()

    all_points = [
        (ns, l2, rf, tw, ts)
        for ns in space["num_states"]
        for l2 in space["l2"]
        for rf in space["refit_freq"]
        for tw in space["train_window_years"]
        for ts in space["transition_smoothing"]
    ]

    if budget >= len(all_points):
        chosen_points = all_points
    else:
        idx = rng.choice(len(all_points), size=budget, replace=False)
        chosen_points = [all_points[int(i)] for i in idx]

    for ns, l2, rf, tw, ts in chosen_points:
        cand = {
            "num_states": int(ns),
            "l2": float(l2),
            "refit_freq": str(rf),
            "train_window_years": int(tw),
            "transition_smoothing": float(ts),
        }
        frozen = tuple(sorted(cand.items()))
        if frozen in seen:
            continue
        seen.add(frozen)
        candidates.append(cand)

    return candidates


# Choose default rolling validation years inside the development sample.
def default_validation_years(cfg: HMMVersion1TuningConfig) -> List[int]:
    if cfg.validation_years is not None:
        return [int(y) for y in cfg.validation_years]
    start = max(int(cfg.dev_start_year) + 1, int(cfg.dev_end_year) - 4)
    return list(range(start, int(cfg.dev_end_year) + 1))


# Estimate how many HMM refit blocks will run across the requested validation years.
def estimate_total_hmm_block_fits(
    panel: pd.DataFrame,
    hmm_cfg: HMMMultihorizonConfig,
    *,
    validation_years: List[int],
) -> int:
    total = 0

    for target_year in validation_years:
        fit_df = panel[panel["Date"].dt.year <= int(target_year)].copy()
        if len(fit_df) == 0:
            continue

        model = HMMMultihorizon(config=replace(hmm_cfg, verbose=False, show_progress=False, progress_bar=None))
        data = model._prepare_panel(fit_df, require_labels=True)
        if len(data) == 0:
            continue

        dates = pd.Index(sorted(data["Date"].unique()))
        if len(dates) < max(model.config.min_history_dates, model.config.num_states + 2):
            continue

        pred_dates = model._build_refit_dates(dates)
        eligible_blocks = model._collect_eligible_blocks(data, dates, pred_dates)
        total += int(len(eligible_blocks))

    return int(total)


# Fit the HMM through one validation year and return that year's walk-forward predictions.
def fit_predict_validation_year(
    panel: pd.DataFrame,
    hmm_cfg: HMMMultihorizonConfig,
    *,
    target_year: int,
    progress_bar: Any = None,
) -> pd.DataFrame:
    fit_df = panel[panel["Date"].dt.year <= int(target_year)].copy()
    if len(fit_df) == 0:
        return pd.DataFrame(columns=["Date", "StockID", "up_prob"])

    local_cfg = replace(
        hmm_cfg,
        verbose=False,
        show_progress=False,
        progress_desc=None,
        progress_leave=False,
        progress_bar=progress_bar,
    )

    model = HMMMultihorizon(config=local_cfg)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="KMeans is known to have a memory leak on Windows with MKL.*",
            category=UserWarning,
        )
        model.fit(fit_df)

    pred = model.row_pred_.copy() if model.row_pred_ is not None else pd.DataFrame(columns=["Date", "StockID", "up_prob"])
    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.normalize()
    pred["StockID"] = pred["StockID"].astype(str)
    pred = pred[pred["Date"].dt.year == int(target_year)].copy()
    pred = pred.dropna(subset=["Date", "StockID", "up_prob"]).copy()
    pred = pred.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)
    return pred


# Score one signal panel with annualized H-L Sharpe from the portfolio engine.
def score_signal_with_sharpe(
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
            include_price_adv=True,
            verbose=False,
        )
        pf_ret, _ = pm.calculate_portfolio_rets(weight_type=str(weight_type), cut=int(cut), delay=int(delay))

    if "H-L" not in pf_ret.columns:
        return float("-inf")

    hl = pd.to_numeric(pf_ret["H-L"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(hl) < 3:
        return float("-inf")

    mean_ret = float(hl.mean())
    std_ret = float(hl.std(ddof=1))
    if (not np.isfinite(mean_ret)) or (not np.isfinite(std_ret)) or std_ret <= 0:
        return float("-inf")

    periods_per_year = 52 if str(freq).lower().strip() == "week" else 12 if str(freq).lower().strip() == "month" else 4
    return float((mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year)))


# Evaluate one HMM candidate across all pre-2016 validation years.
def evaluate_hmm_candidate(
    panel: pd.DataFrame,
    cfg: HMMVersion1TuningConfig,
    params: Dict[str, Any],
    *,
    context_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Hyperparameters are chosen before the first day outside development.
    # The annual tuner sets dev_end_year to target_year - 1. This outer
    # cutoff also applies to returns used to score validation forecasts.
    information_cutoff = pd.Timestamp(year=int(cfg.dev_end_year) + 1, month=1, day=1)
    panel = panel[pd.to_datetime(panel["Date"]) < information_cutoff].copy()
    panel = mask_unrealized_hmm_labels(panel, information_cutoff)
    validation_years = default_validation_years(cfg)
    if any(int(y) > int(cfg.dev_end_year) for y in validation_years):
        raise ValueError("Validation years must precede the tuning information cutoff.")
    hmm_cfg = make_hmm_config(cfg, params, context_cols=context_cols)

    per_year: List[Dict[str, Any]] = []
    scores: List[float] = []

    total_blocks = estimate_total_hmm_block_fits(panel, hmm_cfg, validation_years=validation_years)
    block_bar = None
    if bool(cfg.verbose) and tqdm is not None and total_blocks > 0:
        block_bar = tqdm(total=total_blocks, desc="HMM refit blocks", leave=False)

    try:
        for target_year in validation_years:
            if block_bar is not None:
                block_bar.set_postfix_str(f"year={int(target_year)}")

            pred = fit_predict_validation_year(panel, hmm_cfg, target_year=int(target_year), progress_bar=block_bar)
            score_pred = _realized_scoring_signal(pred, panel, cfg, information_cutoff)
            score = score_signal_with_sharpe(
                score_pred,
                freq=cfg.freq,
                country=cfg.country,
                weight_type=cfg.score_weight_type,
                cut=cfg.score_cut,
                delay=cfg.score_delay,
            )
            per_year.append({"target_year": int(target_year), "score": float(score)})
            scores.append(float(score))
    finally:
        if block_bar is not None:
            block_bar.close()

    mean_score = float(np.mean(scores)) if scores else float("-inf")
    std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    return {
        "params": dict(params),
        "mean_score": mean_score,
        "std_score": std_score,
        "per_year": per_year,
        "num_eval_years": int(len(per_year)),
        "information_cutoff": str(information_cutoff.date()),
    }


def _realized_scoring_signal(
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    cfg: HMMVersion1TuningConfig,
    information_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict tuning scores, without restricting forecast coverage.

    The default weekly/monthly, zero-delay scores use the corresponding
    source horizon endpoint. Other holding periods require explicit
    ``score_end`` metadata from the portfolio-return source.
    """
    horizon = {"week": 5, "month": 20}.get(str(cfg.freq).lower().strip())
    if "score_end" in panel:
        end = pd.to_datetime(panel["score_end"], errors="coerce").dt.normalize()
        valid = end.gt(pd.to_datetime(panel["Date"])) & end.lt(information_cutoff)
    elif int(cfg.score_delay) == 0 and horizon is not None:
        valid = realized_outcome_mask(panel, horizon, information_cutoff)
    else:
        raise ValueError("Cutoff-safe tuning for this holding period requires actual score_end dates.")
    keys = panel.loc[valid, ["Date", "StockID"]].drop_duplicates()
    return pred.merge(keys, on=["Date", "StockID"], how="inner", validate="one_to_one")


# Convert one candidate's validation-year scores into a display-friendly DataFrame.
def validation_score_table(result: Dict[str, Any]) -> pd.DataFrame:
    table = pd.DataFrame(result.get("per_year", [])).copy()
    if len(table) == 0:
        return pd.DataFrame(columns=["target_year", "score"])
    return table.sort_values("target_year").reset_index(drop=True)


# Flatten the full candidate ranking into one DataFrame for inspection.
def ranking_to_frame(ranking: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for item in ranking:
        row = {
            "candidate_idx": int(item.get("candidate_idx", -1)),
            "mean_score": float(item.get("mean_score", np.nan)),
            "std_score": float(item.get("std_score", np.nan)),
            "num_eval_years": int(item.get("num_eval_years", 0)),
        }
        params = dict(item.get("params", {}))
        row["num_states"] = params.get("num_states")
        row["l2"] = params.get("l2")
        row["refit_freq"] = params.get("refit_freq")
        row["train_window_years"] = params.get("train_window_years")
        row["transition_smoothing"] = params.get("transition_smoothing")

        per_year = {f"score_{int(x['target_year'])}": float(x["score"]) for x in item.get("per_year", [])}
        row.update(per_year)
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "candidate_idx",
                "mean_score",
                "std_score",
                "num_eval_years",
                "num_states",
                "l2",
                "refit_freq",
                "train_window_years",
                "transition_smoothing",
            ]
        )

    return pd.DataFrame(rows).sort_values(["mean_score", "std_score"], ascending=[False, True]).reset_index(drop=True)


# Build a simple hyperparameter summary table from the candidate ranking.
def summarize_hyperparameter_effects(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if len(ranking_df) == 0:
        return pd.DataFrame(
            columns=[
                "hyperparam",
                "value",
                "n_candidates",
                "mean_of_mean_score",
                "std_of_mean_score",
                "best_mean_score",
                "top25_share",
            ]
        )

    params = ["num_states", "l2", "refit_freq", "train_window_years", "transition_smoothing"]
    q75 = float(ranking_df["mean_score"].quantile(0.75))
    work = ranking_df.copy()
    work["is_top25"] = (work["mean_score"] >= q75).astype(int)

    out_parts: List[pd.DataFrame] = []
    for p in params:
        grp = (
            work.groupby(p, dropna=False)
            .agg(
                n_candidates=("candidate_idx", "count"),
                mean_of_mean_score=("mean_score", "mean"),
                std_of_mean_score=("mean_score", "std"),
                best_mean_score=("mean_score", "max"),
                top25_share=("is_top25", "mean"),
            )
            .reset_index()
            .rename(columns={p: "value"})
        )
        grp["hyperparam"] = p
        out_parts.append(grp)

    out = pd.concat(out_parts, ignore_index=True)
    cols = [
        "hyperparam",
        "value",
        "n_candidates",
        "mean_of_mean_score",
        "std_of_mean_score",
        "best_mean_score",
        "top25_share",
    ]
    return out[cols].sort_values(["hyperparam", "best_mean_score"], ascending=[True, False]).reset_index(drop=True)


# Tune the HMM once on the pre-2016 development sample and save the results.
def tune_hmm_pre2016(
    cfg: HMMVersion1TuningConfig,
    *,
    panel: Optional[pd.DataFrame] = None,
    context_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if panel is None:
        panel = build_hmm_panel(cfg)

    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel = panel[
        (panel["Date"].dt.year >= int(cfg.dev_start_year))
        & (panel["Date"].dt.year <= int(cfg.dev_end_year))
    ].copy()

    if len(panel) == 0:
        raise ValueError("No rows remain in the requested development window.")

    if context_cols is None:
        context_cols = [c for c in panel.columns if str(c).startswith("ctx_")]

    candidates = sample_hmm_candidates(cfg)
    ranking: List[Dict[str, Any]] = []

    iterator = _progress(
        list(enumerate(candidates)),
        total=len(candidates),
        desc="HMM tuning candidates",
        enabled=bool(cfg.verbose),
        leave=True,
    )

    for idx, params in iterator:
        try:
            result = evaluate_hmm_candidate(panel, cfg, params, context_cols=context_cols)
            result["candidate_idx"] = int(idx)
            ranking.append(result)
            if cfg.verbose:
                print(
                    f"[HMMVersion1Tuner] candidate={idx} mean_score={result['mean_score']:.6f} "
                    f"std_score={result['std_score']:.6f} params={result['params']}"
                )
        except Exception as exc:
            if cfg.verbose:
                print(f"[HMMVersion1Tuner] candidate={idx} failed: {exc}")

    if not ranking:
        raise ValueError("All HMM tuning candidates failed.")

    ranking.sort(key=lambda item: (item["mean_score"], -item["std_score"]), reverse=True)
    best = ranking[0]
    best_year_df = validation_score_table(best)
    ranking_df = ranking_to_frame(ranking)
    hyperparam_summary_df = summarize_hyperparameter_effects(ranking_df)

    payload = {
        "tuner": "hmm_version1_pre2016",
        "config": asdict(cfg),
        "context_cols": list(context_cols),
        "num_candidates": int(len(candidates)),
        "validation_years": default_validation_years(cfg),
        "best_params": dict(best["params"]),
        "best_mean_score": float(best["mean_score"]),
        "best_std_score": float(best["std_score"]),
        "best_validation_scores_by_year": best_year_df.to_dict(orient="records"),
        "ranking": ranking,
    }

    best_fp = out_dir / "hmm_best_params.json"
    log_fp = out_dir / "hmm_tuning_log.json"
    summary_fp = out_dir / "hmm_best_validation_sharpes.csv"
    ranking_fp = out_dir / "hmm_candidate_ranking.csv"
    hyperparam_fp = out_dir / "hmm_hyperparam_summary.csv"

    with open(best_fp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": payload["best_params"],
                "best_mean_score": payload["best_mean_score"],
                "best_std_score": payload["best_std_score"],
                "validation_years": payload["validation_years"],
                "context_cols": payload["context_cols"],
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with open(log_fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    best_year_df.to_csv(summary_fp, index=False)
    ranking_df.to_csv(ranking_fp, index=False)
    hyperparam_summary_df.to_csv(hyperparam_fp, index=False)

    if cfg.verbose:
        print("\\n[HMMVersion1Tuner] Best validation-year Sharpe table:")
        print(best_year_df.to_string(index=False))
        print(
            f"[HMMVersion1Tuner] Average validation Sharpe "
            f"({min(payload['validation_years'])}-{max(payload['validation_years'])}): "
            f"{payload['best_mean_score']:.6f}"
        )
        print("\\n[HMMVersion1Tuner] Top candidate ranking preview:")
        print(ranking_df.head(10).to_string(index=False))
        print("\\n[HMMVersion1Tuner] Hyperparameter summary preview:")
        print(hyperparam_summary_df.to_string(index=False))
        print(f"[HMMVersion1Tuner] wrote best params to: {best_fp}")
        print(f"[HMMVersion1Tuner] wrote full tuning log to: {log_fp}")
        print(f"[HMMVersion1Tuner] wrote validation-year Sharpe table to: {summary_fp}")
        print(f"[HMMVersion1Tuner] wrote candidate ranking to: {ranking_fp}")
        print(f"[HMMVersion1Tuner] wrote hyperparameter summary to: {hyperparam_fp}")

    return payload


# Run the frozen HMM across the full sample and keep only 2016+ lockbox predictions.
def run_frozen_hmm_lockbox(
    cfg: HMMVersion1TuningConfig,
    best_params: Dict[str, Any],
    *,
    panel: Optional[pd.DataFrame] = None,
    context_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if panel is None:
        panel = build_hmm_panel(cfg)

    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()

    full_panel = panel[
        (panel["Date"].dt.year >= int(cfg.dev_start_year))
        & (panel["Date"].dt.year <= int(cfg.lockbox_end_year))
    ].copy()

    if context_cols is None:
        context_cols = [c for c in full_panel.columns if str(c).startswith("ctx_")]

    hmm_cfg = make_hmm_config(cfg, best_params, context_cols=context_cols)
    model = HMMMultihorizon(config=hmm_cfg)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="KMeans is known to have a memory leak on Windows with MKL.*",
            category=UserWarning,
        )
        model.fit(full_panel)

    pred = model.row_pred_.copy() if model.row_pred_ is not None else pd.DataFrame(columns=["Date", "StockID", "up_prob"])
    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.normalize()
    pred["StockID"] = pred["StockID"].astype(str)

    pred = pred[
        (pred["Date"].dt.year >= int(cfg.lockbox_start_year))
        & (pred["Date"].dt.year <= int(cfg.lockbox_end_year))
    ].copy()

    pred = pred.dropna(subset=["Date", "StockID", "up_prob"]).copy()
    pred = pred.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)
    return pred


# Load saved best HMM params from disk.
def load_hmm_best_params(output_dir: str) -> Dict[str, Any]:
    fp = Path(output_dir) / "hmm_best_params.json"
    if not fp.exists():
        raise FileNotFoundError(f"HMM best-params file not found: {fp}")
    with open(fp, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload["best_params"])
