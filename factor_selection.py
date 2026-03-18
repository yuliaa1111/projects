"""
ret_pred/factor_selection.py

Dynamic factor selection based on historical RankIC stored in `predictions.pkl`.

Per docs (ret_pred/md/tasks.md, ret_pred/md/data.md):
- `predictions.pkl` is a historical record of factor RankIC (NOT model features).
- At a refit node t, selection must only use information available up to t (<= t here by user confirmation).
- Output is the selected factor set for this window; within a window it must stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import logging
import pickle

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankICSelectionCfg:
    enabled: bool = False
    predictions_pkl_path: str = ""
    rankic_threshold: float = 0.03
    aggregate: str = "abs_mean"  # abs_mean | mean_abs
    cutoff_inclusive: bool = True  # True => use dates <= t; False => < t
    min_history_days: int = 1


def load_predictions_pkl(path: str | Path) -> Dict[pd.Timestamp, pd.Series]:
    """
    Load predictions.pkl.

    Expected structure (user confirmed):
    - outer: dict-like mapping date -> pandas.Series
    - series.index: factor names (str)
    - series.values: RankIC (float) for that date & factor
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"predictions.pkl not found: {p}")

    with open(p, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"predictions.pkl must be a dict[date->Series], got: {type(obj)}")

    out: Dict[pd.Timestamp, pd.Series] = {}
    for k, v in obj.items():
        dt = pd.to_datetime(k)
        if not isinstance(v, pd.Series):
            raise TypeError(f"predictions.pkl values must be pandas.Series, got: {type(v)} at key={k}")
        s = v.copy()
        s.index = s.index.astype(str)
        s = pd.to_numeric(s, errors="coerce")
        out[pd.Timestamp(dt)] = s

    if not out:
        raise ValueError(f"predictions.pkl loaded but empty: {p}")

    return out


def to_rankic_frame(pred_map: Dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """
    Convert mapping(date->Series[factor->rankic]) to long DataFrame: [date, factor, rankic].
    """
    rows = []
    for dt, s in pred_map.items():
        if s is None or len(s) == 0:
            continue
        df = s.rename("rankic").reset_index()
        df.columns = ["factor", "rankic"]
        df["date"] = pd.Timestamp(dt)
        rows.append(df[["date", "factor", "rankic"]])

    if not rows:
        return pd.DataFrame(columns=["date", "factor", "rankic"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["factor"] = out["factor"].astype(str)
    out["rankic"] = pd.to_numeric(out["rankic"], errors="coerce")
    return out


def filter_history_by_date(
    df: pd.DataFrame,
    *,
    asof_date: str | pd.Timestamp,
    inclusive: bool,
) -> pd.DataFrame:
    """
    Cut historical records to only use information up to the refit node.
    """
    if df.empty:
        return df
    t = pd.to_datetime(asof_date)
    if inclusive:
        return df[df["date"] <= t].copy()
    return df[df["date"] < t].copy()


def aggregate_factor_rankic(
    hist_df: pd.DataFrame,
    *,
    aggregate: str,
) -> pd.DataFrame:
    """
    Aggregate historical RankIC by factor.

    aggregate:
      - abs_mean: abs(mean(rankic))
      - mean_abs: mean(abs(rankic))
    """
    if hist_df.empty:
        return pd.DataFrame(columns=["factor", "score", "n_days"])

    aggregate = str(aggregate or "abs_mean").lower()
    if aggregate not in ("abs_mean", "mean_abs"):
        raise ValueError(f"aggregate must be 'abs_mean' or 'mean_abs', got: {aggregate}")

    g = hist_df.dropna(subset=["rankic"]).groupby("factor", sort=False)["rankic"]
    if aggregate == "abs_mean":
        score = g.mean().abs()
    else:
        score = g.apply(lambda s: s.abs().mean())
    n_days = g.count()

    out = pd.DataFrame({"factor": score.index.astype(str), "score": score.values, "n_days": n_days.values})
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def select_factors_by_rankic(
    pred_map_or_df: Dict[pd.Timestamp, pd.Series] | pd.DataFrame,
    *,
    asof_date: str | pd.Timestamp,
    threshold: float = 0.03,
    aggregate: str = "abs_mean",
    inclusive: bool = True,
    min_history_days: int = 1,
    candidate_pool: Optional[Iterable[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select factors based on historical RankIC up to `asof_date`.

    Returns:
      selected_factors (sorted by score desc),
      selection_state (for artifact/logging)
    """
    if isinstance(pred_map_or_df, dict):
        df = to_rankic_frame(pred_map_or_df)
    else:
        df = pred_map_or_df.copy()

    if df.empty:
        return [], {"asof_date": str(asof_date), "n_hist_rows": 0, "selected": []}

    hist = filter_history_by_date(df, asof_date=asof_date, inclusive=bool(inclusive))

    if candidate_pool is not None:
        pool = set(str(x) for x in candidate_pool)
        hist = hist[hist["factor"].astype(str).isin(pool)].copy()

    agg_df = aggregate_factor_rankic(hist, aggregate=aggregate)

    threshold = float(threshold)
    min_history_days = int(min_history_days)
    if min_history_days < 1:
        raise ValueError(f"min_history_days must be >= 1, got: {min_history_days}")

    picked = agg_df[(agg_df["score"] >= threshold) & (agg_df["n_days"] >= min_history_days)].copy()
    selected = picked["factor"].astype(str).tolist()

    state: Dict[str, Any] = {
        "asof_date": str(pd.to_datetime(asof_date))[:10],
        "inclusive": bool(inclusive),
        "aggregate": str(aggregate),
        "threshold": float(threshold),
        "min_history_days": int(min_history_days),
        "n_hist_rows": int(len(hist)),
        "n_factors_seen": int(agg_df.shape[0]),
        "n_selected": int(len(selected)),
        "selected": list(selected),
    }
    return selected, state


def select_union_factors_by_rankic(
    pred_map_or_df: Dict[pd.Timestamp, pd.Series] | pd.DataFrame,
    *,
    asof_dates: Iterable[str | pd.Timestamp],
    threshold: float = 0.03,
    aggregate: str = "abs_mean",
    inclusive: bool = True,
    min_history_days: int = 1,
    candidate_pool: Optional[Iterable[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Build a union factor pool across multiple refit asof dates.

    Use case:
    - Pre-plan a compact feature universe for dataloader/preprocess, before trainer starts.
    - Keep strict historical cutoff semantics by reusing select_factors_by_rankic per asof_date.
    """
    dates = sorted(pd.to_datetime(pd.Series(list(asof_dates)).dropna().unique()))
    pool_list = None if candidate_pool is None else [str(x) for x in candidate_pool]
    if len(dates) == 0:
        return [], {"n_asof_dates": 0, "n_union": 0, "union_factors": []}

    union: List[str] = []
    seen = set()
    by_asof: List[Dict[str, Any]] = []

    for dt in dates:
        selected, _ = select_factors_by_rankic(
            pred_map_or_df,
            asof_date=dt,
            threshold=threshold,
            aggregate=aggregate,
            inclusive=inclusive,
            min_history_days=min_history_days,
            candidate_pool=pool_list,
        )
        for f in selected:
            if f not in seen:
                seen.add(f)
                union.append(f)
        by_asof.append({"asof_date": str(pd.Timestamp(dt).date()), "n_selected": int(len(selected))})

    state: Dict[str, Any] = {
        "n_asof_dates": int(len(dates)),
        "asof_start": str(pd.Timestamp(dates[0]).date()),
        "asof_end": str(pd.Timestamp(dates[-1]).date()),
        "threshold": float(threshold),
        "aggregate": str(aggregate),
        "inclusive": bool(inclusive),
        "min_history_days": int(min_history_days),
        "n_union": int(len(union)),
        "union_factors": list(union),
        "by_asof": by_asof,
        "candidate_pool_size": None if pool_list is None else int(len(pool_list)),
    }
    return union, state


def save_selected_factors_json(
    selected_factors: List[str],
    state: Dict[str, Any],
    *,
    out_path: str | Path,
) -> str:
    """
    Save selected factor set artifact for reproducibility.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "selected_factors": list(selected_factors),
        "state": dict(state or {}),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("selected_factors saved | path=%s | n=%d", str(p), int(len(selected_factors)))
    return str(p)
