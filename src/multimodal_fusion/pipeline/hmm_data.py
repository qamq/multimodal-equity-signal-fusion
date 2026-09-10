
"""
multimodal_fusion.pipeline.hmm_data

Data builder for the latent-state multi-horizon ensemble.

Memory-safe label loading
-------------------------
The HMM fallback-label loader reads only compact label-only parquet files from
CACHE_DIR and does not fall back to large feature panels. This keeps panel
construction memory-bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None

from multimodal_fusion.pipeline.ensemble_pipeline import EnsemblePipelineConfig, EnsembleSignalPipeline


@dataclass
class HMMDataConfig:
    """Configuration for building the latent-state multi-horizon training panel."""
    rf_root: str
    cnn_root: str

    rf_pred_path: Optional[str] = None
    cnn_pred_path: Optional[str] = None

    feature_panel_path: Optional[str] = None
    feature_cols: Optional[List[str]] = None

    freq: str = "week"
    country: str = "USA"
    label_threshold: float = 0.0
    verbose: bool = True

    fwd_ret_5d_col: Optional[str] = None
    fwd_ret_20d_col: Optional[str] = None
    label_end_5d_col: str = "label_end_5d"
    label_end_20d_col: str = "label_end_20d"


class HMMDataBuilder:
    """Build the stock-level panel and the date-level context table for the HMM model."""

    # Initialize the HMM data builder and store the configuration.
    def __init__(self, cfg: HMMDataConfig) -> None:
        self.cfg = cfg
        self.context_cols_: List[str] = []

    # Build the stock-level modeling panel used by the latent-state ensemble.
    def build_model_frame(self) -> pd.DataFrame:
        base = self._build_expert_panel()
        labeled = self._attach_multihorizon_labels(base)
        merged = self._merge_optional_stock_features(labeled)
        context = self._build_date_context(merged)
        out = merged.merge(context, on="Date", how="left")

        keep = ["Date", "StockID", "p_cnn", "p_rf", "fwd_ret_5d", "fwd_ret_20d", "y_5d", "y_20d", "label_end_5d", "label_end_20d"] + self.context_cols_
        extra_keep = [c for c in ["sector_id", "log_mcap"] if c in out.columns and c not in keep]
        keep = keep + extra_keep
        out = out[[c for c in keep if c in out.columns]].copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        out["StockID"] = out["StockID"].astype(str)
        out = out.dropna(subset=["Date", "StockID", "p_cnn", "p_rf"]).copy()
        out = out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)

        if self.cfg.verbose:
            print(
                f"[HMMDataBuilder] rows={len(out):,} dates={out['Date'].nunique():,} "
                f"context_cols={len(self.context_cols_)}"
            )
        return out

    # Build only the date-level context table for diagnostics or reuse.
    def build_date_context_frame(self) -> pd.DataFrame:
        base = self._build_expert_panel()
        labeled = self._attach_multihorizon_labels(base)
        merged = self._merge_optional_stock_features(labeled)
        return self._build_date_context(merged)

    # Load and align the CNN and RF expert probability panels.
    def _build_expert_panel(self) -> pd.DataFrame:
        pipe_cfg = EnsemblePipelineConfig(
            rf_root=self.cfg.rf_root,
            cnn_root=self.cfg.cnn_root,
            rf_pred_path=self.cfg.rf_pred_path,
            cnn_pred_path=self.cfg.cnn_pred_path,
            attach_labels=False,
            freq=self.cfg.freq,
            country=self.cfg.country,
            verbose=self.cfg.verbose,
            keep_label_cols_in_signal=False,
        )
        pipe = EnsembleSignalPipeline(pipe_cfg)
        panel = pipe.build_model_frame()
        panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce").dt.normalize()
        panel["StockID"] = panel["StockID"].astype(str)
        panel = panel.drop_duplicates(["Date", "StockID"], keep="last").copy()
        return panel

    # Attach realized 5-day and 20-day returns plus binary labels.
    def _attach_multihorizon_labels(self, panel: pd.DataFrame) -> pd.DataFrame:
        from Scripts.Data import equity_data as eqd

        primary = eqd.get_period_ret(
            self.cfg.freq,
            country=self.cfg.country,
            include_price_adv=False,
            require_price_adv=False,
        ).copy()
        primary = self._normalize_label_source(primary)

        fallback = None

        try:
            col5 = self._select_5d_return_col(primary)
            src5 = primary
        except KeyError:
            fallback = self._load_fallback_multihorizon_label_panel(need_5d=True, need_20d=False)
            col5 = self._select_5d_return_col(fallback)
            src5 = fallback

        try:
            col20 = self._select_20d_return_col(primary)
            src20 = primary
        except KeyError:
            if fallback is None:
                fallback = self._load_fallback_multihorizon_label_panel(need_5d=False, need_20d=True)
            col20 = self._select_20d_return_col(fallback)
            src20 = fallback

        if self.cfg.verbose and (src5 is not primary or src20 is not primary):
            fallback_parts = []
            if src5 is not primary:
                fallback_parts.append("5d")
            if src20 is not primary:
                fallback_parts.append("20d")
            print(
                "[HMMDataBuilder] Using fallback multihorizon label panel for "
                + " and ".join(fallback_parts)
                + " labels."
            )

        lab5 = self._extract_horizon_labels(src5, col5, 5)
        lab20 = self._extract_horizon_labels(src20, col20, 20)

        lab5["fwd_ret_5d"] = pd.to_numeric(lab5["fwd_ret_5d"], errors="coerce")
        lab20["fwd_ret_20d"] = pd.to_numeric(lab20["fwd_ret_20d"], errors="coerce")

        lab = lab5.merge(lab20, on=["Date", "StockID"], how="outer")
        valid_5d = lab["fwd_ret_5d"].notna()
        valid_20d = lab["fwd_ret_20d"].notna()
        lab["y_5d"] = (lab["fwd_ret_5d"] > float(self.cfg.label_threshold)).astype("Int64").where(valid_5d)
        lab["y_20d"] = (lab["fwd_ret_20d"] > float(self.cfg.label_threshold)).astype("Int64").where(valid_20d)

        out = panel.merge(
            lab[["Date", "StockID", "fwd_ret_5d", "fwd_ret_20d", "y_5d", "y_20d", "label_end_5d", "label_end_20d"]],
            on=["Date", "StockID"],
            how="left",
        )
        return out

    def _extract_horizon_labels(self, source: pd.DataFrame, return_col: str, horizon: int) -> pd.DataFrame:
        """Carry the source's actual horizon endpoint; never estimate it."""
        end_col = getattr(self.cfg, f"label_end_{horizon}d_col")
        canonical_end = f"label_end_{horizon}d"
        if end_col not in source and canonical_end in source:
            end_col = canonical_end
        cols = ["Date", "StockID", return_col]
        if end_col in source:
            cols.append(end_col)
        out = source[cols].copy().rename(columns={return_col: f"fwd_ret_{horizon}d", end_col: canonical_end})
        if canonical_end not in out:
            out[canonical_end] = pd.NaT
        out[canonical_end] = pd.to_datetime(out[canonical_end], errors="coerce").dt.normalize()
        return out

    # Normalize a label-source DataFrame onto Date / StockID string keys.
    def _normalize_label_source(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if isinstance(out.index, pd.MultiIndex):
            out = out.reset_index()
        if "Date" not in out.columns or "StockID" not in out.columns:
            out = out.reset_index()

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
        out["StockID"] = pd.to_numeric(out["StockID"], errors="coerce")
        out = out.dropna(subset=["Date", "StockID"]).copy()
        out["StockID"] = out["StockID"].astype(int).astype(str)
        out = out.drop_duplicates(["Date", "StockID"], keep="last").sort_values(["Date", "StockID"]).reset_index(drop=True)
        return out

    # Return candidate CACHE_DIR roots for fallback multihorizon label panels.
    def _candidate_cache_dirs(self) -> List[Path]:
        out: List[Path] = []

        try:
            from Scripts.Data import dgp_config as dcf
            cache_dir = Path(getattr(dcf, "CACHE_DIR", ""))
            if str(cache_dir):
                out.append(cache_dir)
        except Exception:
            pass

        try:
            project_root = Path(__file__).resolve().parents[2]
            out.append(project_root / "CACHE_DIR")
        except Exception:
            pass

        seen = set()
        uniq: List[Path] = []
        for p in out:
            rp = Path(p)
            if str(rp) in seen:
                continue
            seen.add(str(rp))
            uniq.append(rp)
        return uniq

    # Return the preferred compact fallback label filenames.
    def _preferred_small_label_files(self) -> List[str]:
        return [
            "hmm_multihorizon_labels_small.parquet",
            "hmm_multihorizon_labels_small.pq",
            "hmm_labels_small.parquet",
            "hmm_labels_small.pq",
        ]

    # Return the candidate 5d return-column names in preference order.
    def _candidate_5d_return_cols(self) -> List[str]:
        return [c for c in [self.cfg.fwd_ret_5d_col, "fwd_ret_5d", "Ret_5d", "next_week_ret"] if c is not None]

    # Return the candidate 20d return-column names in preference order.
    def _candidate_20d_return_cols(self) -> List[str]:
        return [c for c in [self.cfg.fwd_ret_20d_col, "fwd_ret_20d", "Ret_20d"] if c is not None]

    # Read parquet schema names without materializing the whole file when possible.
    def _parquet_columns(self, fp: Path) -> List[str]:
        if pq is not None:
            try:
                return list(pq.ParquetFile(fp).schema.names)
            except Exception:
                pass

        try:
            head = pd.read_parquet(fp, columns=[])
            return list(head.columns)
        except Exception:
            pass

        return list(pd.read_parquet(fp).columns)

    # Load only the requested fallback label columns from one parquet source.
    def _read_fallback_label_subset(self, fp: Path, *, need_5d: bool, need_20d: bool) -> Optional[pd.DataFrame]:
        cols = self._parquet_columns(fp)
        cols_set = set(cols)

        c5 = None
        c20 = None
        if need_5d:
            for c in self._candidate_5d_return_cols():
                if c in cols_set:
                    c5 = c
                    break
            if c5 is None:
                return None

        if need_20d:
            for c in self._candidate_20d_return_cols():
                if c in cols_set:
                    c20 = c
                    break
            if c20 is None:
                return None

        date_col = "Date" if "Date" in cols_set else None
        stock_col = "StockID" if "StockID" in cols_set else None
        if date_col is None or stock_col is None:
            return None

        read_cols = [date_col, stock_col]
        if c5 is not None and c5 not in read_cols:
            read_cols.append(c5)
        if c20 is not None and c20 not in read_cols:
            read_cols.append(c20)
        for needed, horizon in ((need_5d, 5), (need_20d, 20)):
            if needed:
                for end_col in (getattr(self.cfg, f"label_end_{horizon}d_col"), f"label_end_{horizon}d"):
                    if end_col in cols_set and end_col not in read_cols:
                        read_cols.append(end_col)

        out = pd.read_parquet(fp, columns=read_cols)

        rename_map = {}
        if c5 is not None and c5 != "fwd_ret_5d":
            rename_map[c5] = "fwd_ret_5d"
        if c20 is not None and c20 != "fwd_ret_20d":
            rename_map[c20] = "fwd_ret_20d"
        if rename_map:
            out = out.rename(columns=rename_map)
        return out

    # Load the fallback multihorizon label panel from compact label-only parquet ONLY.
    def _load_fallback_multihorizon_label_panel(self, *, need_5d: bool = True, need_20d: bool = True) -> pd.DataFrame:
        checked: List[str] = []

        for cache_dir in self._candidate_cache_dirs():
            if not cache_dir.exists():
                continue
            for fname in self._preferred_small_label_files():
                fp = cache_dir / fname
                checked.append(str(fp))
                if not fp.exists():
                    continue
                df = self._read_fallback_label_subset(fp, need_5d=need_5d, need_20d=need_20d)
                if df is not None:
                    if self.cfg.verbose:
                        print(f"[HMMDataBuilder] Loaded fallback multihorizon labels from compact file: {fp}")
                    return self._normalize_label_source(df)

        horizon_text = []
        if need_5d:
            horizon_text.append("5d")
        if need_20d:
            horizon_text.append("20d")

        raise FileNotFoundError(
            "Compact HMM fallback label file not found or missing required columns for "
            + " and ".join(horizon_text)
            + " labels. "
            + "Create CACHE_DIR/hmm_multihorizon_labels_small.parquet first. "
            + f"Checked: {checked}"
        )

    # Merge optional stock-level features from an external panel.
    def _merge_optional_stock_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.feature_panel_path:
            return panel

        fp = Path(self.cfg.feature_panel_path)
        if not fp.exists():
            raise FileNotFoundError(f"HMM feature panel not found: {fp}")

        feat = pd.read_parquet(fp)
        feat["Date"] = pd.to_datetime(feat["Date"], errors="coerce").dt.normalize()
        feat["StockID"] = pd.to_numeric(feat["StockID"], errors="coerce")
        feat = feat.dropna(subset=["Date", "StockID"]).copy()
        feat["StockID"] = feat["StockID"].astype(int).astype(str)

        if self.cfg.feature_cols is None:
            default_cols = [
                "days_since_fund_update",
                "log_mcap",
                "abs_log_ret",
                "abs_ret_5d",
                "abs_ret_20d",
                "abs_ret_5d_minus_20d",
                "fresh_fund_7d",
                "fresh_fund_30d",
            ]
            use_cols = [c for c in default_cols if c in feat.columns]
        else:
            use_cols = [c for c in self.cfg.feature_cols if c in feat.columns]

        if not use_cols:
            return panel

        keep = ["Date", "StockID"] + use_cols
        feat = feat[keep].copy()
        return panel.merge(feat, on=["Date", "StockID"], how="left")

    # Aggregate stock-level inputs into one date-level context frame.
    def _build_date_context(self, panel: pd.DataFrame) -> pd.DataFrame:
        d = panel.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()

        d["ctx_mean_p_cnn"] = pd.to_numeric(d["p_cnn"], errors="coerce")
        d["ctx_mean_p_rf"] = pd.to_numeric(d["p_rf"], errors="coerce")
        d["ctx_abs_gap"] = (pd.to_numeric(d["p_cnn"], errors="coerce") - pd.to_numeric(d["p_rf"], errors="coerce")).abs()

        agg_spec = {
            "ctx_mean_p_cnn": "mean",
            "ctx_mean_p_rf": "mean",
            "ctx_abs_gap": ["mean", "std"],
        }

        stock_context_candidates = [
            "days_since_fund_update",
            "log_mcap",
            "abs_log_ret",
            "abs_ret_5d",
            "abs_ret_20d",
            "abs_ret_5d_minus_20d",
            "fresh_fund_7d",
            "fresh_fund_30d",
        ]

        for c in stock_context_candidates:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                agg_spec[c] = "mean"

        ctx = d.groupby("Date").agg(agg_spec)
        ctx.columns = [self._flatten_context_name(col) for col in ctx.columns.to_flat_index()]
        ctx = ctx.reset_index()

        self.context_cols_ = [c for c in ctx.columns if c != "Date"]
        return ctx

    # Choose the forward 5-day return column from the period-return table.
    def _select_5d_return_col(self, ret: pd.DataFrame) -> str:
        candidates = [
            self.cfg.fwd_ret_5d_col,
            "next_week_ret_0delay",
            "next_week_ret",
            "Ret_5d",
            "fwd_ret_5d",
            "week_ret",
        ]
        for c in candidates:
            if c is not None and c in ret.columns:
                return c
        raise KeyError(f"Could not find a 5d forward-return column in period returns. cols={list(ret.columns)}")

    # Choose the forward 20-day return column from the period-return table.
    def _select_20d_return_col(self, ret: pd.DataFrame) -> str:
        candidates = [
            self.cfg.fwd_ret_20d_col,
            "Ret_20d",
            "fwd_ret_20d",
        ]
        for c in candidates:
            if c is not None and c in ret.columns:
                return c
        raise KeyError(f"Could not find a 20d forward-return column in period returns. cols={list(ret.columns)}")

    # Flatten a grouped context aggregation key into one column name.
    def _flatten_context_name(self, key: Tuple[str, str]) -> str:
        left, right = key
        if right == "":
            return str(left)
        return f"{left}_{right}"
