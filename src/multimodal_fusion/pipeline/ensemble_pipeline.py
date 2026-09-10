"""
multimodal_fusion.pipeline.ensemble_pipeline

Purpose
-------
This module is the "bridge" between saved model predictions and the portfolio system.

It loads:
  - RF predictions (parquet): Date, StockID, up_prob
  - CNN predictions (csv or parquet): StockID, ending_date/Date, up_prob

It builds:
  - A merged panel with columns: Date, StockID, p_rf, p_cnn

Optional label attachment
-------------------------
If attach_labels=True, it adds:
  - y: binary label = 1{forward return > threshold}
  - fwd_ret: raw forward return used by ranking objectives

Important: When attaching labels, we must NOT trigger the Price/ADV cache upgrade.
We therefore call equity_data.get_period_ret(..., include_price_adv=False, require_price_adv=False).

Public API
----------
- build_signal_df(): returns MultiIndex (Date, StockID) with 'up_prob' (optionally y, fwd_ret)
- build_model_frame(): returns a flat DataFrame with Date, StockID, p_cnn, p_rf, y, fwd_ret
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class EnsemblePipelineConfig:
    """Configuration for loading preds, optional labels, and producing an ensemble-ready panel."""
    rf_root: str
    cnn_root: str

    rf_pred_path: Optional[str] = None
    cnn_pred_path: Optional[str] = None

    rf_prob_col: str = "up_prob"
    cnn_prob_col: str = "up_prob"

    date_col: str = "Date"
    id_col: str = "StockID"

    attach_labels: bool = False
    freq: str = "week"
    country: str = "USA"
    label_threshold: float = 0.0
    label_ret_col: Optional[str] = None

    require_overlap: bool = True
    verbose: bool = True

    # If True, keep y/fwd_ret in build_signal_df output (helpful for diagnostics).
    keep_label_cols_in_signal: bool = False


class EnsembleSignalPipeline:
    """
    EnsembleSignalPipeline

    Builds the merged (p_cnn, p_rf) panel and optionally attaches labels.
    """

    # Initialize pipeline config and lazy-load EnsembleManager only if needed.
    def __init__(self, cfg: EnsemblePipelineConfig) -> None:
        self.cfg = cfg

    # Build MultiIndex (Date, StockID) with up_prob (and optionally y/fwd_ret).
    def build_signal_df(self, *, method: str, method_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        # Lazy import to avoid circular imports during packaging.
        from multimodal_fusion.experimental.manager import EnsembleManager, EnsembleManagerConfig

        merged = self.build_model_frame()

        mgr = EnsembleManager(EnsembleManagerConfig(method=method, method_kwargs=method_kwargs or {}))

        if hasattr(mgr.model, "fit") and ("y" in merged.columns):
            mgr.fit(merged)

        merged["up_prob"] = mgr.predict(merged)

        cols = [self.cfg.date_col, self.cfg.id_col, "up_prob"]
        if self.cfg.keep_label_cols_in_signal:
            for extra in ("y", "fwd_ret"):
                if extra in merged.columns:
                    cols.append(extra)

        out = merged[cols].copy()
        out[self.cfg.date_col] = pd.to_datetime(out[self.cfg.date_col], errors="coerce").dt.normalize()
        out[self.cfg.id_col] = out[self.cfg.id_col].astype(str)
        out["up_prob"] = pd.to_numeric(out["up_prob"], errors="coerce").astype(float)

        out = out.dropna(subset=[self.cfg.date_col, self.cfg.id_col, "up_prob"])
        out = out.drop_duplicates([self.cfg.date_col, self.cfg.id_col], keep="last")
        out = out.set_index([self.cfg.date_col, self.cfg.id_col]).sort_index()
        return out

    # Build the flat model frame used for tuning/training weight models.
    def build_model_frame(self) -> pd.DataFrame:
        rf_fp, cnn_src = self._resolve_inputs()

        rf_raw = self._load_table(rf_fp)
        cnn_raw = self._load_cnn_source(cnn_src)

        rf = self._standardize_preds(rf_raw, model="rf", prob_col=self.cfg.rf_prob_col)
        cnn = self._standardize_preds(cnn_raw, model="cnn", prob_col=self.cfg.cnn_prob_col)

        how = "inner" if self.cfg.require_overlap else "outer"
        merged = rf.merge(cnn, on=[self.cfg.date_col, self.cfg.id_col], how=how)

        if self.cfg.attach_labels:
            merged = self._attach_labels_from_returns(merged)

        if self.cfg.verbose:
            print(
                f"[EnsemblePipeline] rf={len(rf):,} cnn={len(cnn):,} merged={len(merged):,} "
                f"attach_labels={self.cfg.attach_labels}"
            )

        return merged

    # Resolve RF and CNN prediction sources (file or directory).
    def _resolve_inputs(self) -> Tuple[Path, Path]:
        rf_root = Path(self.cfg.rf_root)
        cnn_root = Path(self.cfg.cnn_root)

        rf_fp = Path(self.cfg.rf_pred_path) if self.cfg.rf_pred_path else self._discover_rf_preds(rf_root)
        cnn_src = Path(self.cfg.cnn_pred_path) if self.cfg.cnn_pred_path else self._discover_cnn_source(cnn_root)

        if self.cfg.verbose:
            print(f"[EnsemblePipeline] RF preds: {rf_fp}")
            print(f"[EnsemblePipeline] CNN src:  {cnn_src}")

        return rf_fp, cnn_src

    # Discover RF preds parquet under rf_root.
    def _discover_rf_preds(self, root: Path) -> Path:
        if not root.exists():
            raise FileNotFoundError(f"RF root does not exist: {root}")

        c1 = list(root.rglob("rf_preds_yearly.parquet"))
        if c1:
            return max(c1, key=lambda p: p.stat().st_mtime)

        c2 = list(root.rglob("rf_preds_rolling.parquet"))
        if c2:
            return max(c2, key=lambda p: p.stat().st_mtime)

        c3 = [p for p in root.rglob("*.parquet") if "rf_preds" in p.name.lower()]
        if not c3:
            raise FileNotFoundError(f"No RF preds parquet found under {root}")
        return max(c3, key=lambda p: p.stat().st_mtime)

    # Discover CNN source: ensem_res directory preferred, else latest matching csv.
    def _discover_cnn_source(self, root: Path) -> Path:
        if not root.exists():
            raise FileNotFoundError(f"CNN root does not exist: {root}")

        ensem_dirs = [p for p in root.rglob("ensem_res") if p.is_dir()]
        if ensem_dirs:
            return max(ensem_dirs, key=lambda p: p.stat().st_mtime)

        cands = [p for p in root.rglob("*.csv") if ("ensem" in p.name.lower() and "_res_" in p.name.lower())]
        if not cands:
            raise FileNotFoundError(f"No CNN ensem_res directory or ensem*_res_*.csv found under {root}")
        return max(cands, key=lambda p: p.stat().st_mtime)

    # Load a table from disk (csv/parquet).
    def _load_table(self, fp: Path) -> pd.DataFrame:
        if not fp.exists():
            raise FileNotFoundError(str(fp))
        suf = fp.suffix.lower()
        if suf in {".parquet", ".pq"}:
            return pd.read_parquet(fp)
        if suf == ".csv":
            return pd.read_csv(fp)
        raise ValueError(f"Unsupported file type: {fp}")

    # Load CNN predictions from a directory (concat) or a single file.
    def _load_cnn_source(self, path: Path) -> pd.DataFrame:
        if path.is_dir():
            files = sorted(path.glob("ensem*_res_*.csv"))
            if not files:
                files = sorted(path.glob("*.csv"))
            if not files:
                # Allow parquet sources in directory if needed.
                files_pq = sorted(path.glob("*.parquet"))
                if files_pq:
                    parts = [pd.read_parquet(f) for f in files_pq]
                    return pd.concat(parts, ignore_index=True)
                raise FileNotFoundError(f"No CSV files found in {path}")
            parts = [pd.read_csv(f) for f in files]
            return pd.concat(parts, ignore_index=True)
        return self._load_table(path)

    # Standardize predictions to Date, StockID, and p_{model}.
    def _standardize_preds(self, df: pd.DataFrame, *, model: str, prob_col: str) -> pd.DataFrame:
        dfx = df.copy()

        # Rename CNN ending_date -> Date if needed.
        if self.cfg.date_col not in dfx.columns and "ending_date" in dfx.columns:
            dfx = dfx.rename(columns={"ending_date": self.cfg.date_col})

        # Rename likely StockID aliases if needed.
        if self.cfg.id_col not in dfx.columns:
            for alt in ("permno", "PERMNO", "Permno", "sid", "SID", "code"):
                if alt in dfx.columns:
                    dfx = dfx.rename(columns={alt: self.cfg.id_col})
                    break

        # Recover from index if needed.
        if self.cfg.date_col not in dfx.columns or self.cfg.id_col not in dfx.columns:
            dfx = dfx.reset_index()

        # Second pass for ending_date after reset_index.
        if self.cfg.date_col not in dfx.columns and "ending_date" in dfx.columns:
            dfx = dfx.rename(columns={"ending_date": self.cfg.date_col})

        if self.cfg.date_col not in dfx.columns or self.cfg.id_col not in dfx.columns:
            raise KeyError(f"{model}: missing {self.cfg.date_col}/{self.cfg.id_col}. cols={list(dfx.columns)}")

        if prob_col not in dfx.columns:
            raise KeyError(f"{model}: prob_col='{prob_col}' not in cols={list(dfx.columns)}")

        dfx[self.cfg.date_col] = pd.to_datetime(dfx[self.cfg.date_col], errors="coerce").dt.normalize()

        sid = pd.to_numeric(dfx[self.cfg.id_col], errors="coerce")
        dfx[self.cfg.id_col] = sid
        dfx = dfx.dropna(subset=[self.cfg.date_col, self.cfg.id_col]).copy()
        dfx[self.cfg.id_col] = dfx[self.cfg.id_col].astype(int).astype(str)

        p = pd.to_numeric(dfx[prob_col], errors="coerce").astype(float).clip(1e-6, 1.0 - 1e-6)

        out = dfx[[self.cfg.date_col, self.cfg.id_col]].copy()
        out["p_" + model] = p
        return out.dropna(subset=["p_" + model])

    # Attach y and fwd_ret using forward returns from equity_data.get_period_ret.
    def _attach_labels_from_returns(self, merged: pd.DataFrame) -> pd.DataFrame:
        from Scripts.Data import equity_data as eqd

        ret = eqd.get_period_ret(
            self.cfg.freq,
            country=self.cfg.country,
            include_price_adv=False,
            require_price_adv=False,
        ).copy()

        if isinstance(ret.index, pd.MultiIndex):
            ret = ret.reset_index()
        if "Date" not in ret.columns or "StockID" not in ret.columns:
            ret = ret.reset_index()

        ret["Date"] = pd.to_datetime(ret["Date"], errors="coerce").dt.normalize()
        ret["StockID"] = pd.to_numeric(ret["StockID"], errors="coerce")
        ret = ret.dropna(subset=["Date", "StockID"]).copy()
        ret["StockID"] = ret["StockID"].astype(int).astype(str)

        rcol = self._select_forward_return_column(ret)

        lab = ret[["Date", "StockID", rcol]].copy()
        lab[rcol] = pd.to_numeric(lab[rcol], errors="coerce")
        lab = lab.dropna(subset=[rcol]).copy()

        lab["fwd_ret"] = lab[rcol].astype(float)
        lab["y"] = (lab[rcol].astype(float) > float(self.cfg.label_threshold)).astype(int)

        out = merged.merge(lab[["Date", "StockID", "y", "fwd_ret"]], on=["Date", "StockID"], how="left")
        return out

    # Choose which forward-return column is used for labeling and ranking.
    def _select_forward_return_column(self, ret: pd.DataFrame) -> str:
        if self.cfg.label_ret_col and self.cfg.label_ret_col in ret.columns:
            return self.cfg.label_ret_col

        base = "next_" + str(self.cfg.freq) + "_ret_0delay"
        if base in ret.columns:
            return base

        for c in ("next_" + str(self.cfg.freq) + "_ret", "Ret_" + str(self.cfg.freq), str(self.cfg.freq) + "_ret", "ret", "RET", "Return"):
            if c in ret.columns:
                return c

        raise KeyError(f"Could not find a forward return column in period ret. cols={list(ret.columns)}")