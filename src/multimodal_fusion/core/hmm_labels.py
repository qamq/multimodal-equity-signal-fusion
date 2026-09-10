"""Outcome availability shared by HMM fitting and tuning.

``label_end_5d`` and ``label_end_20d`` are the actual last observation dates
used to construct each return, supplied by its data source. They are not
estimated from calendar days, weekdays, or the stock panel's sampling grid.
An unknown endpoint is unavailable. Endpoints equal to the information cutoff
are excluded: a forecast at date t can use outcomes ending strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_hmm_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve missing labels, deriving absent label columns at threshold zero."""
    out = df.copy()
    for horizon in (5, 20):
        label, ret = f"y_{horizon}d", f"fwd_ret_{horizon}d"
        if label in out:
            out[label] = pd.to_numeric(out[label], errors="coerce").astype(float)
        elif ret in out:
            returns = pd.to_numeric(out[ret], errors="coerce")
            out[label] = (returns > 0).astype(float).where(returns.notna())
        else:
            out[label] = np.nan
        if ret in out:
            # A missing source return also invalidates a stale class-0 label.
            out[label] = out[label].where(pd.to_numeric(out[ret], errors="coerce").notna())
        if not out[label].dropna().isin([0, 1]).all():
            raise ValueError(f"{label} must contain only 0, 1, or missing values.")
    return out


def realized_outcome_mask(df: pd.DataFrame, horizon: int, information_cutoff: pd.Timestamp) -> pd.Series:
    """Whether a nonmissing outcome has a known end strictly before cutoff."""
    label, ret, end = f"y_{horizon}d", f"fwd_ret_{horizon}d", f"label_end_{horizon}d"
    if end not in df:
        return pd.Series(False, index=df.index)
    endpoint = pd.to_datetime(df[end], errors="coerce").dt.normalize()
    date = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    valid = endpoint.gt(date) & endpoint.lt(pd.Timestamp(information_cutoff).normalize())
    if label in df:
        valid &= pd.to_numeric(df[label], errors="coerce").notna()
    elif ret not in df:
        return pd.Series(False, index=df.index)
    if ret in df:
        valid &= pd.to_numeric(df[ret], errors="coerce").notna()
    return valid


def mask_unrealized_hmm_labels(df: pd.DataFrame, information_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Mask each unavailable outcome without removing its prediction inputs."""
    out = prepare_hmm_labels(df)
    for horizon in (5, 20):
        valid = realized_outcome_mask(out, horizon, information_cutoff)
        out[f"y_{horizon}d"] = out[f"y_{horizon}d"].where(valid)
        ret = f"fwd_ret_{horizon}d"
        if ret in out:
            out[ret] = pd.to_numeric(out[ret], errors="coerce").where(valid)
    return out
