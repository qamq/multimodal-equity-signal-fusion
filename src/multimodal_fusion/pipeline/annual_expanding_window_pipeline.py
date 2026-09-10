
"""
annual_expanding_window_pipeline.py

Helpers for the final annual expanding-window CNN / HMM / portfolio notebook.

Purpose
-------
This module is the notebook-companion helper layer. It centralizes the heavy
logic so the notebook can stay small and mostly act as a runner / audit sheet.

Design
------
- RF uses one fixed prediction source.
- CNN predictions are generated inside the notebook year by year.
- The HMM reads from the exact notebook-generated combined CNN parquet through
  the target year.
- Yearly RF / CNN-overlap / HMM-full portfolios are saved.
- Stitched signals and stitched portfolios can be built after the yearly runs.

The intended workflow is:
1) create CNN history years (for example 1997-1999),
2) run yearly comparison years (for example 2000-2024),
3) run stitched backtests,
4) inspect manifests and diagnostics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _progress(iterable, *, total=None, desc="Progress", enabled=True):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=True)
    return iterable


class AnnualExpandingWindowRunner:
    """Stateful helper used by the final expanding-window notebook."""

    # Initialize the runner with the exact notebook configuration.
    def __init__(
        self,
        *,
        project_root: Path,
        annual_output_root: Path,
        start_year: int,
        first_cnn_output_year: int,
        first_pred_year: int,
        last_pred_year: int,
        country: str,
        freq: str,
        cut: int,
        delay_list: Sequence[int],
        all_weight_types: Optional[Sequence[str]],
        label_threshold: float,
        portfolio_kwargs: Optional[Dict[str, Any]],
        rf_root: Path,
        rf_pred_path: Path,
        cnn_cfg: Any,
        hmm_base_cfg: Any,
        run_rf_portfolios: bool = True,
        run_cnn_portfolios: bool = True,
        run_hmm_overlap_portfolios: bool = True,
        run_hmm_full_portfolios: bool = True,
        run_screened_portfolios: bool = True,
        run_unscreened_portfolios: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.annual_output_root = Path(annual_output_root)
        self.annual_output_root.mkdir(parents=True, exist_ok=True)

        self.combined_sources_dir = self.annual_output_root / "combined_sources"
        self.combined_sources_dir.mkdir(parents=True, exist_ok=True)

        self.stitched_dir = self.annual_output_root / "stitched"
        self.stitched_dir.mkdir(parents=True, exist_ok=True)

        self.start_year = int(start_year)
        self.first_cnn_output_year = int(first_cnn_output_year)
        self.first_pred_year = int(first_pred_year)
        self.last_pred_year = int(last_pred_year)

        self.country = str(country)
        self.freq = str(freq)
        self.cut = int(cut)
        self.delay_list = [int(x) for x in delay_list]
        self.all_weight_types = None if all_weight_types is None else list(all_weight_types)
        self.label_threshold = float(label_threshold)
        self.base_portfolio_kwargs = dict(portfolio_kwargs or {})

        self.rf_root = Path(rf_root)
        self.rf_pred_path = Path(rf_pred_path)
        self.cnn_cfg = cnn_cfg
        self.hmm_base_cfg = hmm_base_cfg

        self.run_rf_portfolios = bool(run_rf_portfolios)
        self.run_cnn_portfolios = bool(run_cnn_portfolios)
        self.run_hmm_overlap_portfolios = bool(run_hmm_overlap_portfolios)
        self.run_hmm_full_portfolios = bool(run_hmm_full_portfolios)
        self.run_screened_portfolios = bool(run_screened_portfolios)
        self.run_unscreened_portfolios = bool(run_unscreened_portfolios)
        self.show_progress = bool(show_progress)

        self.screened_portfolio_kwargs = {
            **self.base_portfolio_kwargs,
            "tradability_screens": True,
            "include_price_adv": True,
        }
        self.unscreened_portfolio_kwargs = {
            **self.base_portfolio_kwargs,
            "tradability_screens": False,
            "include_price_adv": True,
        }

        if not self.rf_root.exists():
            raise FileNotFoundError(f"RF_ROOT not found: {self.rf_root}")
        if not self.rf_pred_path.exists():
            raise FileNotFoundError(f"RF_PRED_PATH not found: {self.rf_pred_path}")

        from Scripts.Experiments.cnn_annual_runner import run_cnn_year
        from Scripts.Portfolio.portfolio_year_runner import run_portfolio_for_year
        from multimodal_fusion.pipeline.hmm_annual_runner import run_hmm_year
        from multimodal_fusion.pipeline.overlap_signal_utils import (
            build_overlap_panel,
            restrict_signal_to_keys,
            save_signal_frame,
            select_year_overlap_keys,
        )

        self._run_cnn_year = run_cnn_year
        self._run_portfolio_for_year = run_portfolio_for_year
        self._run_hmm_year = run_hmm_year
        self._build_overlap_panel = build_overlap_panel
        self._restrict_signal_to_keys = restrict_signal_to_keys
        self._save_signal_frame = save_signal_frame
        self._select_year_overlap_keys = select_year_overlap_keys

        self.rf_signal_all = self.standardize_signal_df(pd.read_parquet(self.rf_pred_path))

    # Normalize a signal frame to Date, StockID, up_prob.
    def standardize_signal_df(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        sig = signal_df.copy()
        if isinstance(sig.index, pd.MultiIndex):
            sig = sig.reset_index()

        sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce").dt.normalize()
        sig["StockID"] = pd.to_numeric(sig["StockID"], errors="coerce")
        sig["up_prob"] = pd.to_numeric(sig["up_prob"], errors="coerce")
        sig = sig.dropna(subset=["Date", "StockID", "up_prob"]).copy()
        sig["StockID"] = sig["StockID"].astype(int).astype(str)
        sig = sig.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"])
        return sig.reset_index(drop=True)

    # Restrict a standardized signal to one target year.
    def filter_signal_to_year(self, signal_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
        sig = self.standardize_signal_df(signal_df)
        sig = sig[sig["Date"].dt.year == int(target_year)].copy()
        return sig.reset_index(drop=True)

    # Return the first matching forward-return column.
    def select_forward_return_column(self, ret: pd.DataFrame) -> str:
        base = f"next_{self.freq}_ret_0delay"
        if base in ret.columns:
            return base
        for c in (
            f"next_{self.freq}_ret",
            f"Ret_{self.freq}",
            f"{self.freq}_ret",
            "ret",
            "RET",
            "Return",
        ):
            if c in ret.columns:
                return c
        raise KeyError(f"Could not find a forward return column in period returns. cols={list(ret.columns)}")

    # Load realized forward-return labels for diagnostics.
    def load_signal_truth(self) -> pd.DataFrame:
        from Scripts.Data import equity_data as eqd

        ret = eqd.get_period_ret(
            self.freq,
            country=self.country,
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

        rcol = self.select_forward_return_column(ret)
        lab = ret[["Date", "StockID", rcol]].copy()
        lab[rcol] = pd.to_numeric(lab[rcol], errors="coerce")
        lab = lab.dropna(subset=[rcol]).copy()
        lab["fwd_ret"] = lab[rcol].astype(float)
        lab["y"] = (lab[rcol].astype(float) > float(self.label_threshold)).astype(int)
        return lab[["Date", "StockID", "fwd_ret", "y"]].copy()

    # Build compact signal diagnostics.
    def build_signal_diagnostics(self, signal_df: pd.DataFrame) -> Dict[str, Any]:
        sig = self.standardize_signal_df(signal_df)

        summary = {
            "rows": int(len(sig)),
            "n_dates": int(sig["Date"].nunique()),
            "n_stocks": int(sig["StockID"].nunique()),
            "up_prob_mean": float(sig["up_prob"].mean()) if len(sig) else float("nan"),
            "up_prob_std": float(sig["up_prob"].std(ddof=1)) if len(sig) > 1 else float("nan"),
            "up_prob_p10": float(sig["up_prob"].quantile(0.10)) if len(sig) else float("nan"),
            "up_prob_p50": float(sig["up_prob"].quantile(0.50)) if len(sig) else float("nan"),
            "up_prob_p90": float(sig["up_prob"].quantile(0.90)) if len(sig) else float("nan"),
        }

        date_summary = (
            sig.groupby("Date", as_index=False)
            .agg(
                rows=("StockID", "size"),
                n_stocks=("StockID", pd.Series.nunique),
                up_prob_mean=("up_prob", "mean"),
                up_prob_std=("up_prob", "std"),
                up_prob_min=("up_prob", "min"),
                up_prob_p10=("up_prob", lambda s: float(pd.Series(s).quantile(0.10))),
                up_prob_p50=("up_prob", lambda s: float(pd.Series(s).quantile(0.50))),
                up_prob_p90=("up_prob", lambda s: float(pd.Series(s).quantile(0.90))),
                up_prob_max=("up_prob", "max"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        merged = pd.DataFrame()
        try:
            truth = self.load_signal_truth()
            merged = sig.merge(truth, on=["Date", "StockID"], how="inner")
        except Exception:
            merged = pd.DataFrame()

        if len(merged) > 0:
            from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score

            p_hat = np.clip(pd.to_numeric(merged["up_prob"], errors="coerce").to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            y_true = pd.to_numeric(merged["y"], errors="coerce").astype(int).to_numpy(dtype=np.int8)
            y_hat = (p_hat >= 0.5).astype(np.int8)

            tp = int(np.sum((y_hat == 1) & (y_true == 1)))
            tn = int(np.sum((y_hat == 0) & (y_true == 0)))
            fp = int(np.sum((y_hat == 1) & (y_true == 0)))
            fn = int(np.sum((y_hat == 0) & (y_true == 1)))

            summary.update(
                {
                    "eval_rows": int(len(merged)),
                    "accuracy": float(np.mean(y_hat == y_true)),
                    "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
                    "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
                    "mcc": float(matthews_corrcoef(y_true, y_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
                    "log_loss": float(log_loss(y_true, p_hat, labels=[0, 1])),
                    "brier": float(np.mean(np.square(p_hat - y_true))),
                    "auc": float(roc_auc_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "realized_pos_rate": float(np.mean(y_true)),
                    "pred_pos_rate": float(np.mean(y_hat)),
                    "mean_fwd_ret": float(pd.to_numeric(merged["fwd_ret"], errors="coerce").mean()),
                    "spearman_up_prob_fwd_ret": float(pd.Series(p_hat).corr(pd.to_numeric(merged["fwd_ret"], errors="coerce"), method="spearman")),
                    "pearson_up_prob_fwd_ret": float(pd.Series(p_hat).corr(pd.to_numeric(merged["fwd_ret"], errors="coerce"), method="pearson")),
                }
            )

            date_eval = (
                merged.groupby("Date", as_index=False)
                .agg(
                    eval_rows=("StockID", "size"),
                    realized_pos_rate=("y", "mean"),
                    up_prob_mean=("up_prob", "mean"),
                    mean_fwd_ret=("fwd_ret", "mean"),
                )
                .sort_values("Date")
                .reset_index(drop=True)
            )
            date_summary = date_summary.merge(date_eval, on="Date", how="left")

        return {
            "summary": summary,
            "date_summary": date_summary,
            "merged_eval": merged,
        }

    # Save one signal plus diagnostics.
    def save_signal_with_diagnostics(self, signal_df: pd.DataFrame, out_dir: Path, *, stem: str) -> Dict[str, Optional[str]]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sig = self.standardize_signal_df(signal_df)
        paths = self._save_signal_frame(sig, out_dir, stem)

        diagnostics = self.build_signal_diagnostics(sig)
        summary_fp = out_dir / f"{stem}_diagnostics_summary.json"
        date_fp = out_dir / f"{stem}_date_diagnostics.csv"
        merge_fp = out_dir / f"{stem}_eval_merge.parquet"

        summary_fp.write_text(json.dumps(diagnostics["summary"], indent=2), encoding="utf-8")
        diagnostics["date_summary"].to_csv(date_fp, index=False)
        if len(diagnostics["merged_eval"]) > 0:
            diagnostics["merged_eval"].to_parquet(merge_fp, index=False)
            merge_path = str(merge_fp)
        else:
            merge_path = None

        return {
            "signal_csv": paths.get("csv"),
            "signal_parquet": paths.get("parquet"),
            "diagnostics_summary_json": str(summary_fp),
            "date_diagnostics_csv": str(date_fp),
            "eval_merge_parquet": merge_path,
        }

    # Collect portfolio artifact paths from one portfolio directory.
    def collect_portfolio_artifacts(self, portfolio_dir: Path) -> Dict[str, Any]:
        out_dir = Path(portfolio_dir)
        pf_data_dir = out_dir / "pf_data"
        return {
            "portfolio_dir": str(out_dir),
            "pf_data_dir": str(pf_data_dir) if pf_data_dir.exists() else None,
            "pf_data_csvs": sorted(str(p) for p in pf_data_dir.glob("pf_data_*.csv")) if pf_data_dir.exists() else [],
            "summary_csvs": sorted(
                str(p) for p in out_dir.glob("*.csv")
                if p.is_file()
                and not p.name.endswith("_portfolio_diagnostics.csv")
                and not p.name.endswith("_tc_diagnostics.csv")
            ),
            "portfolio_diagnostics_csvs": sorted(str(p) for p in out_dir.glob("*_portfolio_diagnostics.csv")),
            "portfolio_diagnostics_summary_jsons": sorted(str(p) for p in out_dir.glob("*_portfolio_diagnostics_summary.json")),
            "tc_diagnostics_csvs": sorted(str(p) for p in out_dir.glob("*_tc_diagnostics.csv")),
            "tc_diagnostics_summary_jsons": sorted(str(p) for p in out_dir.glob("*_tc_diagnostics_summary.json")),
            "latex_txts": sorted(str(p) for p in out_dir.glob("*.txt")),
            "run_metadata_json": str(out_dir / "portfolio_run_metadata.json") if (out_dir / "portfolio_run_metadata.json").exists() else None,
        }


    # Return the configured portfolio variants and their kwargs.
    def portfolio_variants(self) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        if self.run_screened_portfolios:
            out.append(("screened", dict(self.screened_portfolio_kwargs)))
        if self.run_unscreened_portfolios:
            out.append(("unscreened", dict(self.unscreened_portfolio_kwargs)))
        return out

    # Return the first portfolio directory from a variant-result payload.
    def primary_portfolio_dir(self, payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(payload, dict) or not payload:
            return None
        for key in ("screened", "unscreened"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key].get("portfolio_dir")
        first = next(iter(payload.values()), None)
        return None if not isinstance(first, dict) else first.get("portfolio_dir")

    # Run screened and/or unscreened annual portfolios for one saved signal.
    def run_year_portfolio_variants(self, signal_df: pd.DataFrame, signal_name: str, target_year: int) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        year_portfolio_root = self.annual_output_root / f"year_{int(target_year)}" / "portfolio"
        for variant_name, variant_kwargs in self.portfolio_variants():
            pf = self._run_portfolio_for_year(
                signal_df=signal_df,
                portfolio_dir=year_portfolio_root / variant_name / str(signal_name),
                target_year=int(target_year),
                start_year=self.start_year,
                freq=self.freq,
                country=self.country,
                delay_list=self.delay_list,
                cut=self.cut,
                weight_types=self.all_weight_types,
                portfolio_kwargs=variant_kwargs,
            )
            pf["artifacts"] = self.collect_portfolio_artifacts(Path(pf["portfolio_dir"]))
            results[str(variant_name)] = pf
        return results

    # Run screened and/or unscreened stitched portfolios for one stitched signal.
    def run_stitched_portfolio_variants(self, signal_df: pd.DataFrame, signal_name: str, start_year: int, end_year: int) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for variant_name, variant_kwargs in self.portfolio_variants():
            results[str(variant_name)] = self.run_stitched_portfolio(
                signal_df,
                signal_name,
                start_year,
                end_year,
                variant_name=str(variant_name),
                portfolio_kwargs=variant_kwargs,
            )
        return results

    # Run a range of pre-HMM CNN history years with a progress bar.
    def run_cnn_history_range(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> Dict[int, Any]:
        lo = self.first_cnn_output_year if start_year is None else int(start_year)
        hi = (self.first_pred_year - 1) if end_year is None else int(end_year)
        out: Dict[int, Any] = {}
        years = list(range(int(lo), int(hi) + 1))
        for year in _progress(years, total=len(years), desc="CNN history years", enabled=self.show_progress):
            out[int(year)] = self.run_cnn_history_year(int(year))
        return out

    # Run a range of annual prediction years with a progress bar.
    def run_year_range(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> Dict[int, Any]:
        lo = self.first_pred_year if start_year is None else int(start_year)
        hi = self.last_pred_year if end_year is None else int(end_year)
        out: Dict[int, Any] = {}
        years = list(range(int(lo), int(hi) + 1))
        for year in _progress(years, total=len(years), desc="Annual prediction years", enabled=self.show_progress):
            out[int(year)] = self.run_one_year(int(year))
        return out

    # Slice the fixed RF source to one target year.
    def rf_full_signal_for_year(self, target_year: int) -> pd.DataFrame:
        return self.filter_signal_to_year(self.rf_signal_all, int(target_year))

    # Return the saved yearly CNN full signal parquet path.
    def cnn_full_signal_parquet_path(self, target_year: int) -> Path:
        return self.annual_output_root / f"year_{int(target_year)}" / "signals" / "cnn_full" / f"cnn_full_up_prob_{int(target_year)}.parquet"

    # Return the combined CNN parquet path through one target year.
    def cnn_combined_source_path(self, target_year: int) -> Path:
        return self.combined_sources_dir / f"cnn_preds_combined_through_{int(target_year)}.parquet"

    # Build the exact notebook-generated combined CNN source through one target year.
    def build_combined_cnn_source_through_year(self, target_year: int) -> Tuple[Path, pd.DataFrame, str]:
        parts = []
        missing_years = []

        for year in range(int(self.first_cnn_output_year), int(target_year) + 1):
            fp = self.cnn_full_signal_parquet_path(year)
            if not fp.exists():
                missing_years.append(int(year))
                continue
            parts.append(pd.read_parquet(fp))

        if missing_years:
            raise FileNotFoundError(
                "Missing notebook-generated CNN full-year signal parquet(s) for years: "
                f"{missing_years}. Run earlier year cells first so the HMM reads the correct combined CNN source."
            )

        combined = self.standardize_signal_df(pd.concat(parts, ignore_index=True))
        out_fp = self.cnn_combined_source_path(target_year)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_fp, index=False)

        meta_fp = out_fp.with_suffix(".json")
        meta = {
            "target_year": int(target_year),
            "source_years_included": list(range(int(self.first_cnn_output_year), int(target_year) + 1)),
            "rows": int(len(combined)),
            "n_dates": int(combined["Date"].nunique()),
            "n_stocks": int(combined["StockID"].nunique()),
            "parquet": str(out_fp),
        }
        meta_fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_fp, combined, str(meta_fp)

    # Build yearly RF/CNN overlap keys using the exact combined CNN source for that year.
    def build_overlap_keys_for_year(self, target_year: int, combined_cnn_pred_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        overlap_panel = self._build_overlap_panel(
            rf_root=str(self.rf_root),
            cnn_root=str(self.combined_sources_dir),
            rf_pred_path=str(self.rf_pred_path),
            cnn_pred_path=str(combined_cnn_pred_path),
            freq=self.freq,
            country=self.country,
            verbose=True,
        )
        overlap_keys = self._select_year_overlap_keys(overlap_panel, int(target_year))
        return overlap_keys, overlap_panel

    # Refresh the root annual manifest from whatever yearly manifests already exist.
    def refresh_root_manifest(self) -> Path:
        completed = []
        for year in range(int(self.first_pred_year), int(self.last_pred_year) + 1):
            fp = self.annual_output_root / f"year_{year}" / "annual_expanding_window_manifest.json"
            if not fp.exists():
                continue
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                payload = {"target_year": int(year), "manifest_json": str(fp)}
            completed.append(
                {
                    "target_year": int(year),
                    "year_dir": str(self.annual_output_root / f"year_{year}"),
                    "manifest_json": str(fp),
                    "combined_cnn_pred_path_used": payload.get("combined_cnn_pred_path_used"),
                    "rf_full_portfolio_dir": payload.get("rf_full_portfolio_dir"),
                    "cnn_overlap_portfolio_dir": payload.get("cnn_overlap_portfolio_dir"),
                    "hmm_full_portfolio_dir": payload.get("hmm_full_portfolio_dir"),
                    "hmm_manifest_json": payload.get("hmm_manifest_json"),
                }
            )

        root_payload = {
            "project_root": str(self.project_root),
            "annual_output_root": str(self.annual_output_root),
            "combined_sources_dir": str(self.combined_sources_dir),
            "rf_root": str(self.rf_root),
            "rf_pred_path": str(self.rf_pred_path),
            "start_year": int(self.start_year),
            "first_cnn_output_year": int(self.first_cnn_output_year),
            "first_pred_year": int(self.first_pred_year),
            "last_pred_year": int(self.last_pred_year),
            "country": str(self.country),
            "freq": str(self.freq),
            "cut": int(self.cut),
            "delay_list": [int(x) for x in self.delay_list],
            "base_portfolio_kwargs": dict(self.base_portfolio_kwargs),
            "screened_portfolio_kwargs": dict(self.screened_portfolio_kwargs),
            "unscreened_portfolio_kwargs": dict(self.unscreened_portfolio_kwargs),
            "completed_years": completed,
        }

        fp = self.annual_output_root / "annual_expanding_window_run_manifest.json"
        fp.write_text(json.dumps(root_payload, indent=2), encoding="utf-8")
        return fp

    # Run one pre-HMM CNN history year and save its annual output.
    def run_cnn_history_year(self, target_year: int) -> Dict[str, Any]:
        target_year = int(target_year)
        if target_year < self.first_cnn_output_year or target_year >= self.first_pred_year:
            raise ValueError(
                "run_cnn_history_year(target_year) is only for the pre-HMM CNN history years "
                f"{self.first_cnn_output_year}..{self.first_pred_year - 1}."
            )

        year_root = self.annual_output_root / f"year_{target_year}"
        signals_dir = year_root / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)

        cnn_cfg_year = replace(self.cnn_cfg, output_dir=str(year_root / "cnn"))
        cnn_res = self._run_cnn_year(cnn_cfg_year, target_year=target_year)
        if "signal_df" not in cnn_res:
            raise KeyError(
                "run_cnn_year(...) did not return 'signal_df'; update this helper to match the current CNN runner API."
            )

        cnn_full = self.filter_signal_to_year(cnn_res["signal_df"][["Date", "StockID", "up_prob"]].copy(), target_year)
        if len(cnn_full) == 0:
            raise ValueError(f"CNN annual runner returned no signal rows for target_year={target_year}.")

        cnn_full_paths = self.save_signal_with_diagnostics(
            cnn_full,
            signals_dir / "cnn_full",
            stem=f"cnn_full_up_prob_{target_year}",
        )

        combined_cnn_path, _, combined_cnn_meta_json = self.build_combined_cnn_source_through_year(target_year)

        manifest = {
            "target_year": int(target_year),
            "project_root": str(self.project_root),
            "year_root": str(year_root),
            "annual_output_root": str(self.annual_output_root),
            "cnn_runner_output_dir": str(year_root / "cnn"),
            "cnn_full_signal": cnn_full_paths,
            "combined_cnn_pred_path_used": str(combined_cnn_path),
            "combined_cnn_meta_json": str(combined_cnn_meta_json),
            "history_only": True,
        }

        manifest_fp = year_root / "cnn_history_year_manifest.json"
        manifest_fp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        print(f"[CNNHistory] year={target_year}")
        print("  CNN full rows      =", len(cnn_full))
        print("  Combined CNN source=", combined_cnn_path)
        print("  History manifest   =", manifest_fp)
        return manifest

    # Return the saved yearly signal parquet path for a named signal family.
    def saved_signal_parquet_path(self, signal_name: str, year: int) -> Path:
        return self.annual_output_root / f"year_{int(year)}" / "signals" / str(signal_name) / f"{signal_name}_up_prob_{int(year)}.parquet"

    # Return the stitched signal output directory for a named signal family.
    def stitched_signal_output_dir(self, signal_name: str) -> Path:
        out_dir = self.stitched_dir / "signals" / str(signal_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # Return the stitched portfolio output directory for a named signal family.
    def stitched_portfolio_output_dir(self, signal_name: str) -> Path:
        out_dir = self.stitched_dir / "portfolio" / str(signal_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # Stitch yearly saved signals into one continuous signal.
    def stitch_yearly_signal(self, signal_name: str, start_year: int, end_year: int) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], str]:
        parts = []
        missing_years = []
        for year in range(int(start_year), int(end_year) + 1):
            fp = self.saved_signal_parquet_path(signal_name, year)
            if not fp.exists():
                missing_years.append(int(year))
                continue
            parts.append(pd.read_parquet(fp))

        if missing_years:
            raise FileNotFoundError(
                f"Missing yearly saved signal parquet(s) for '{signal_name}' years: {missing_years}"
            )

        stitched = self.standardize_signal_df(pd.concat(parts, ignore_index=True))
        out_dir = self.stitched_signal_output_dir(signal_name)
        stem = f"{signal_name}_stitched_{int(start_year)}_{int(end_year)}"
        paths = self.save_signal_with_diagnostics(stitched, out_dir, stem=stem)

        meta = {
            "signal_name": str(signal_name),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "rows": int(len(stitched)),
            "n_dates": int(stitched["Date"].nunique()),
            "n_stocks": int(stitched["StockID"].nunique()),
            "signal_parquet": paths.get("signal_parquet"),
            "signal_csv": paths.get("signal_csv"),
        }
        meta_fp = out_dir / f"{stem}_manifest.json"
        meta_fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return stitched, paths, str(meta_fp)

    # Run one continuous portfolio on a stitched signal.
    def run_stitched_portfolio(
        self,
        signal_df: pd.DataFrame,
        signal_name: str,
        start_year: int,
        end_year: int,
        *,
        variant_name: str,
        portfolio_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        from Scripts.Portfolio.portfolio import PortfolioManager

        sig = self.standardize_signal_df(signal_df)
        out_dir = self.stitched_dir / "portfolio" / str(variant_name) / str(signal_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        pm = PortfolioManager(
            signal_df=sig,
            freq=self.freq,
            portfolio_dir=str(out_dir),
            start_year=int(start_year),
            end_year=int(end_year),
            country=self.country,
            delay_list=[int(x) for x in self.delay_list],
            load_signal=True,
            **dict(portfolio_kwargs),
        )

        for delay in [int(x) for x in self.delay_list]:
            pm.generate_portfolio(cut=int(self.cut), delay=int(delay), weight_types=self.all_weight_types)

        metadata = {
            "signal_name": str(signal_name),
            "variant_name": str(variant_name),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "freq": str(self.freq),
            "country": str(self.country),
            "delay_list": [int(x) for x in self.delay_list],
            "cut": int(self.cut),
            "weight_types": None if self.all_weight_types is None else list(self.all_weight_types),
            "portfolio_kwargs": dict(portfolio_kwargs),
        }
        meta_fp = out_dir / "stitched_portfolio_run_metadata.json"
        meta_fp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "portfolio_dir": str(out_dir),
            "metadata_path": str(meta_fp),
            "artifacts": self.collect_portfolio_artifacts(out_dir),
        }

    # Run one full comparison year.
    def run_one_year(self, target_year: int) -> Dict[str, Any]:
        target_year = int(target_year)
        if target_year < self.first_pred_year or target_year > self.last_pred_year:
            raise ValueError("target_year is outside the configured prediction range.")

        year_root = self.annual_output_root / f"year_{target_year}"
        overlap_dir = year_root / "overlap"
        signals_dir = year_root / "signals"
        portfolio_root = year_root / "portfolio"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        signals_dir.mkdir(parents=True, exist_ok=True)
        portfolio_root.mkdir(parents=True, exist_ok=True)

        rf_full = self.rf_full_signal_for_year(target_year)
        if len(rf_full) == 0:
            raise ValueError(f"No RF rows found for target_year={target_year} in RF_PRED_PATH.")

        rf_full_paths = self.save_signal_with_diagnostics(
            rf_full,
            signals_dir / "rf_full",
            stem=f"rf_full_up_prob_{target_year}",
        )

        rf_pf = None
        if self.run_rf_portfolios and len(rf_full) > 0:
            rf_pf = self.run_year_portfolio_variants(rf_full, "rf_full", target_year)

        cnn_cfg_year = replace(self.cnn_cfg, output_dir=str(year_root / "cnn"))
        cnn_res = self._run_cnn_year(cnn_cfg_year, target_year=target_year)
        if "signal_df" not in cnn_res:
            raise KeyError(
                "run_cnn_year(...) did not return 'signal_df'; update this helper to match the current CNN runner API."
            )

        cnn_full = self.filter_signal_to_year(cnn_res["signal_df"][["Date", "StockID", "up_prob"]].copy(), target_year)
        if len(cnn_full) == 0:
            raise ValueError(f"CNN annual runner returned no signal rows for target_year={target_year}.")

        cnn_full_paths = self.save_signal_with_diagnostics(
            cnn_full,
            signals_dir / "cnn_full",
            stem=f"cnn_full_up_prob_{target_year}",
        )

        combined_cnn_path, _, combined_cnn_meta_json = self.build_combined_cnn_source_through_year(target_year)

        overlap_keys, overlap_panel_current = self.build_overlap_keys_for_year(target_year, combined_cnn_path)
        overlap_keys_csv = overlap_dir / f"overlap_keys_{target_year}.csv"
        overlap_keys_pq = overlap_dir / f"overlap_keys_{target_year}.parquet"
        overlap_keys.to_csv(overlap_keys_csv, index=False)
        overlap_keys.to_parquet(overlap_keys_pq, index=False)

        overlap_meta = {
            "target_year": int(target_year),
            "rows_in_year_overlap_keys": int(len(overlap_keys)),
            "rows_in_current_overlap_panel": int(len(overlap_panel_current)),
            "combined_cnn_pred_path_used": str(combined_cnn_path),
        }
        overlap_meta_fp = overlap_dir / f"overlap_metadata_{target_year}.json"
        overlap_meta_fp.write_text(json.dumps(overlap_meta, indent=2), encoding="utf-8")

        cnn_overlap = self._restrict_signal_to_keys(cnn_full, overlap_keys)
        cnn_overlap_paths = self.save_signal_with_diagnostics(
            cnn_overlap,
            signals_dir / "cnn_overlap",
            stem=f"cnn_overlap_up_prob_{target_year}",
        )

        cnn_pf = None
        if self.run_cnn_portfolios and len(cnn_overlap) > 0:
            cnn_pf = self.run_year_portfolio_variants(cnn_overlap, "cnn_overlap", target_year)

        hmm_cfg_year = replace(
            self.hmm_base_cfg,
            cnn_root=str(self.combined_sources_dir),
            cnn_pred_path=str(combined_cnn_path),
        )

        hmm_res = self._run_hmm_year(
            hmm_cfg_year,
            target_year=target_year,
            panel=None,
            context_cols=None,
            overlap_keys_df=overlap_keys,
        )

        hmm_full = self.filter_signal_to_year(hmm_res["pred_df"][["Date", "StockID", "up_prob"]].copy(), target_year)
        hmm_full_paths = self.save_signal_with_diagnostics(
            hmm_full,
            signals_dir / "hmm_full",
            stem=f"hmm_full_up_prob_{target_year}",
        )

        hmm_full_pf = None
        if self.run_hmm_full_portfolios and len(hmm_full) > 0:
            hmm_full_pf = self.run_year_portfolio_variants(hmm_full, "hmm_full", target_year)

        manifest = {
            "target_year": int(target_year),
            "project_root": str(self.project_root),
            "year_root": str(year_root),
            "annual_output_root": str(self.annual_output_root),
            "rf_root": str(self.rf_root),
            "rf_pred_path_used": str(self.rf_pred_path),
            "combined_cnn_pred_path_used": str(combined_cnn_path),
            "combined_cnn_meta_json": str(combined_cnn_meta_json),
            "overlap_keys_csv": str(overlap_keys_csv),
            "overlap_keys_parquet": str(overlap_keys_pq),
            "overlap_metadata_json": str(overlap_meta_fp),
            "cnn_runner_output_dir": str(year_root / "cnn"),
            "rf_full_signal": rf_full_paths,
            "cnn_full_signal": cnn_full_paths,
            "cnn_overlap_signal": cnn_overlap_paths,
            "hmm_full_signal": hmm_full_paths,
            "rf_full_portfolio_dir": self.primary_portfolio_dir(rf_pf),
            "cnn_overlap_portfolio_dir": self.primary_portfolio_dir(cnn_pf),
            "hmm_full_portfolio_dir": self.primary_portfolio_dir(hmm_full_pf),
            "rf_portfolio": rf_pf,
            "cnn_overlap_portfolio": cnn_pf,
            "hmm_full_portfolio": hmm_full_pf,
            "hmm_manifest_json": hmm_res.get("manifest_json"),
            "hmm_metrics": hmm_res.get("metrics"),
            "hmm_diagnostics_summary": hmm_res.get("diagnostics_summary"),
        }

        manifest_fp = year_root / "annual_expanding_window_manifest.json"
        manifest_fp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        root_manifest_fp = self.refresh_root_manifest()

        print(f"[AnnualExpandingWindow] year={target_year}")
        print("  RF full rows       =", len(rf_full))
        print("  CNN full rows      =", len(cnn_full))
        print("  CNN overlap rows   =", len(cnn_overlap))
        print("  HMM full rows      =", len(hmm_full))
        print("  Combined CNN source=", combined_cnn_path)
        print("  Year manifest      =", manifest_fp)
        print("  Root manifest      =", root_manifest_fp)

        return manifest

    # Run all stitched backtests.
    def run_all_stitched(self) -> Dict[str, Any]:
        stitched_results: Dict[str, Any] = {}

        rf_full_stitched, rf_full_stitched_paths, rf_full_stitched_manifest = self.stitch_yearly_signal(
            "rf_full", self.first_pred_year, self.last_pred_year
        )
        stitched_results["rf_full"] = {
            "signal_paths": rf_full_stitched_paths,
            "signal_manifest_json": rf_full_stitched_manifest,
            "portfolio": self.run_stitched_portfolio_variants(rf_full_stitched, "rf_full", self.first_pred_year, self.last_pred_year),
        }

        cnn_overlap_stitched, cnn_overlap_stitched_paths, cnn_overlap_stitched_manifest = self.stitch_yearly_signal(
            "cnn_overlap", self.first_pred_year, self.last_pred_year
        )
        stitched_results["cnn_overlap"] = {
            "signal_paths": cnn_overlap_stitched_paths,
            "signal_manifest_json": cnn_overlap_stitched_manifest,
            "portfolio": self.run_stitched_portfolio_variants(cnn_overlap_stitched, "cnn_overlap", self.first_pred_year, self.last_pred_year),
        }

        if self.run_hmm_full_portfolios:
            hmm_full_stitched, hmm_full_stitched_paths, hmm_full_stitched_manifest = self.stitch_yearly_signal(
                "hmm_full", self.first_pred_year, self.last_pred_year
            )
            stitched_results["hmm_full"] = {
                "signal_paths": hmm_full_stitched_paths,
                "signal_manifest_json": hmm_full_stitched_manifest,
                "portfolio": self.run_stitched_portfolio_variants(hmm_full_stitched, "hmm_full", self.first_pred_year, self.last_pred_year),
            }

        stitched_manifest_fp = self.stitched_dir / "stitched_run_manifest.json"
        stitched_manifest_fp.write_text(json.dumps(stitched_results, indent=2, default=str), encoding="utf-8")
        return {
            "stitched_results": stitched_results,
            "stitched_manifest_json": str(stitched_manifest_fp),
        }

    # Summarize completed history-year runs, regular annual runs, and stitched output.
    def final_summary(self) -> Dict[str, Any]:
        manifest_rows = []
        for year in range(self.first_pred_year, self.last_pred_year + 1):
            fp = self.annual_output_root / f"year_{year}" / "annual_expanding_window_manifest.json"
            if not fp.exists():
                continue
            payload = json.loads(fp.read_text(encoding="utf-8"))
            payload["year"] = int(year)
            manifest_rows.append(payload)

        summary_df = pd.DataFrame(manifest_rows).sort_values("year").reset_index(drop=True) if manifest_rows else pd.DataFrame()

        history_rows = []
        for year in range(self.first_cnn_output_year, self.first_pred_year):
            fp = self.annual_output_root / f"year_{year}" / "cnn_history_year_manifest.json"
            if not fp.exists():
                continue
            payload = json.loads(fp.read_text(encoding="utf-8"))
            payload["year"] = int(year)
            history_rows.append(payload)

        history_summary_df = pd.DataFrame(history_rows).sort_values("year").reset_index(drop=True) if history_rows else pd.DataFrame()

        root_manifest_fp = self.refresh_root_manifest()
        stitched_manifest_fp = self.stitched_dir / "stitched_run_manifest.json"

        return {
            "history_summary_df": history_summary_df,
            "year_summary_df": summary_df,
            "root_manifest_json": str(root_manifest_fp),
            "stitched_manifest_json": str(stitched_manifest_fp) if stitched_manifest_fp.exists() else None,
        }
