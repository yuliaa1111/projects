"""
ret_pred/feature_engineering.py

Feature engineering stage (optional):
Insert between dataloader and preprocess.

First version requirements (see ret_pred/md/tasks.md, ret_pred/md/data.md):
- Input is long-format DataFrame
- Group by stockid, sort by date
- Add rolling mean/std features (default windows: 5,10,20)
- No future leakage (use current and past only)
- Must be config-controlled (enabled switch)
- Output remains long-format with new columns appended
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _infer_numeric_feature_cols(
    df: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    label_col: Optional[str],
) -> List[str]:
    exclude = {date_col, stockid_col}
    if label_col and label_col in df.columns:
        exclude.add(label_col)
    # defensive: if label_postprocess keeps raw label, avoid leaking it into features
    if label_col and f"{label_col}_raw" in df.columns:
        exclude.add(f"{label_col}_raw")

    cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def add_rolling_stats(
    df_long: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    feature_cols: List[str],
    windows: List[int],
    stats: List[str],
    min_periods: int = 1,
    std_ddof: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add rolling stats (mean/std) columns for selected feature columns.

    Naming:
      - {feature}__roll_mean_{window}
      - {feature}__roll_std_{window}
    """
    if df_long.empty or not feature_cols:
        return df_long, []

    windows = [int(w) for w in (windows or []) if int(w) > 0]
    if not windows:
        return df_long, []

    stats = [str(s).lower() for s in (stats or [])]
    allowed = {"mean", "std"}
    stats = [s for s in stats if s in allowed]
    if not stats:
        return df_long, []

    min_periods = int(min_periods)
    if min_periods <= 0:
        raise ValueError(f"min_periods must be >= 1, got: {min_periods}")

    out = df_long.sort_values([stockid_col, date_col]).copy()

    created: List[str] = []
    gb = out.groupby(stockid_col, sort=False)

    for f in feature_cols:
        s = gb[f]
        for w in windows:
            roll = s.rolling(window=w, min_periods=min_periods)

            if "mean" in stats:
                name = f"{f}__roll_mean_{w}"
                out[name] = roll.mean().reset_index(level=0, drop=True)
                created.append(name)

            if "std" in stats:
                name = f"{f}__roll_std_{w}"
                out[name] = roll.std(ddof=int(std_ddof)).reset_index(level=0, drop=True)
                created.append(name)

    return out, created


def run_feature_engineering(
    df_long: pd.DataFrame,
    cfg: Optional[Dict[str, Any]],
    *,
    date_col: str,
    stockid_col: str,
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = "y",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Entry point for feature engineering stage.

    Config schema (first version):
    feature_engineering:
      enabled: false
      rolling_stats:
        enabled: true
        windows: [5, 10, 20]
        stats: ["mean", "std"]
        min_periods: 1
    """
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return df_long, {"enabled": False}

    if df_long.empty:
        return df_long, {"enabled": True, "skipped": "empty_df"}

    rolling_cfg = (cfg.get("rolling_stats", {}) or {})
    if not bool(rolling_cfg.get("enabled", True)):
        return df_long, {"enabled": True, "rolling_stats": {"enabled": False}}

    if feature_cols is None:
        feature_cols = _infer_numeric_feature_cols(
            df_long,
            date_col=date_col,
            stockid_col=stockid_col,
            label_col=label_col,
        )

    windows = rolling_cfg.get("windows", [5, 10, 20])
    stats = rolling_cfg.get("stats", ["mean", "std"])
    min_periods = rolling_cfg.get("min_periods", 1)

    logger.info(
        "feature_engineering start | n_rows=%d | n_base_feats=%d | windows=%s | stats=%s | min_periods=%s",
        int(len(df_long)),
        int(len(feature_cols or [])),
        str(windows),
        str(stats),
        str(min_periods),
    )

    out, created = add_rolling_stats(
        df_long,
        date_col=date_col,
        stockid_col=stockid_col,
        feature_cols=list(feature_cols or []),
        windows=list(windows or []),
        stats=list(stats or []),
        min_periods=int(min_periods),
        std_ddof=int(rolling_cfg.get("std_ddof", 0)),
    )

    fe_state = {
        "enabled": True,
        "rolling_stats": {
            "enabled": True,
            "windows": [int(w) for w in list(windows or [])],
            "stats": [str(s) for s in list(stats or [])],
            "min_periods": int(min_periods),
            "std_ddof": int(rolling_cfg.get("std_ddof", 0)),
            "n_created": int(len(created)),
        },
        "created_cols": list(created),
    }

    logger.info("feature_engineering done | n_created=%d", int(len(created)))
    return out, fe_state

