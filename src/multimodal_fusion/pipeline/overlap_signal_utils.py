"""
overlap_signal_utils.py

Helpers for building a common stock-date overlap universe for RF, CNN, and HMM
comparisons.

Purpose
-------
This file creates the canonical `(Date, StockID)` overlap subset shared by:
- RF-only portfolio runs,
- CNN-only portfolio runs,
- HMM-only portfolio runs.

Using one explicit overlap key table ensures the three signal families are
compared on the same rows before portfolio construction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Import the existing ensemble signal pipeline from the project.
def _import_ensemble_pipeline():
    try:
        from multimodal_fusion.pipeline.ensemble_pipeline import EnsemblePipelineConfig, EnsembleSignalPipeline  # type: ignore
        return EnsemblePipelineConfig, EnsembleSignalPipeline
    except Exception:
        pass

    try:
        from ensemble_pipeline import EnsemblePipelineConfig, EnsembleSignalPipeline  # type: ignore
        return EnsemblePipelineConfig, EnsembleSignalPipeline
    except Exception as exc:
        raise ImportError("Could not import EnsemblePipelineConfig / EnsembleSignalPipeline.") from exc


# Create a directory if it does not already exist.
def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Standardize a flat signal frame to Date, StockID, and up_prob.
def _standardize_signal_df(df, prob_col):
    out = df.copy()
    if "Date" not in out.columns and "ending_date" in out.columns:
        out = out.rename(columns={"ending_date": "Date"})

    need = ["Date", "StockID", prob_col]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise KeyError("Signal frame is missing required columns: {}".format(missing))

    out = out[["Date", "StockID", prob_col]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["StockID"] = pd.to_numeric(out["StockID"], errors="coerce")
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce")
    out = out.dropna(subset=["Date", "StockID", prob_col]).copy()
    out["StockID"] = out["StockID"].astype(int).astype(str)
    out = out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"])
    return out.reset_index(drop=True)


# Build the RF-CNN overlap panel used as the canonical comparison universe.
def build_overlap_panel(
    rf_root,
    cnn_root,
    rf_pred_path=None,
    cnn_pred_path=None,
    freq="week",
    country="USA",
    verbose=True,
):
    EnsemblePipelineConfig, EnsembleSignalPipeline = _import_ensemble_pipeline()

    pipe_cfg = EnsemblePipelineConfig(
        rf_root=str(rf_root),
        cnn_root=str(cnn_root),
        rf_pred_path=None if rf_pred_path is None else str(rf_pred_path),
        cnn_pred_path=None if cnn_pred_path is None else str(cnn_pred_path),
        attach_labels=False,
        freq=str(freq),
        country=str(country),
        require_overlap=True,
        verbose=bool(verbose),
        keep_label_cols_in_signal=False,
    )
    pipe = EnsembleSignalPipeline(pipe_cfg)
    panel = pipe.build_model_frame()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel["StockID"] = pd.to_numeric(panel["StockID"], errors="coerce")
    panel = panel.dropna(subset=["Date", "StockID", "p_rf", "p_cnn"]).copy()
    panel["StockID"] = panel["StockID"].astype(int).astype(str)
    panel = panel.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"])
    return panel.reset_index(drop=True)


# Select the overlap key table for one target year.
def select_year_overlap_keys(overlap_panel, target_year):
    panel = overlap_panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel["StockID"] = panel["StockID"].astype(str)
    year_keys = panel[panel["Date"].dt.year == int(target_year)][["Date", "StockID"]].copy()
    year_keys = year_keys.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"])
    return year_keys.reset_index(drop=True)


# Build the RF-only signal on the shared overlap universe.
def build_rf_only_overlap_signal(overlap_panel, target_year):
    panel = overlap_panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel["StockID"] = panel["StockID"].astype(str)
    out = panel[panel["Date"].dt.year == int(target_year)][["Date", "StockID", "p_rf"]].copy()
    out = out.rename(columns={"p_rf": "up_prob"})
    return out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)


# Build the CNN-only signal on the shared overlap universe.
def build_cnn_only_overlap_signal(overlap_panel, target_year):
    panel = overlap_panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
    panel["StockID"] = panel["StockID"].astype(str)
    out = panel[panel["Date"].dt.year == int(target_year)][["Date", "StockID", "p_cnn"]].copy()
    out = out.rename(columns={"p_cnn": "up_prob"})
    return out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)


# Restrict an arbitrary signal frame to the supplied overlap keys.
def restrict_signal_to_keys(signal_df, overlap_keys_df):
    sig = _standardize_signal_df(signal_df, "up_prob")
    keys = overlap_keys_df.copy()
    keys["Date"] = pd.to_datetime(keys["Date"], errors="coerce").dt.normalize()
    keys["StockID"] = pd.to_numeric(keys["StockID"], errors="coerce")
    keys = keys.dropna(subset=["Date", "StockID"]).copy()
    keys["StockID"] = keys["StockID"].astype(int).astype(str)
    keys = keys[["Date", "StockID"]].drop_duplicates(["Date", "StockID"], keep="last")
    out = keys.merge(sig, on=["Date", "StockID"], how="inner")
    return out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)


# Save a signal frame to both CSV and parquet for later reuse.
def save_signal_frame(signal_df, output_dir, file_stem):
    out_dir = _ensure_dir(output_dir)
    csv_fp = out_dir / (str(file_stem) + ".csv")
    pq_fp = out_dir / (str(file_stem) + ".parquet")
    signal_df.to_csv(csv_fp, index=False)
    signal_df.to_parquet(pq_fp, index=False)
    return {"csv": str(csv_fp), "parquet": str(pq_fp)}
