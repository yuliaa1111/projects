"""
ret_pred/windowing_rankic_refit_roll.py

Windowing helper for trainer strategy `rankic_refit_roll`.

Goals:
- Window definition is configurable (days or ratio) via config.
- Produce a list of WindowDef objects that fully define:
  - train range (expanding by default)
  - test evaluation range
  - future prediction block range (fixed N days)
- Enforce hard constraints:
  - No partial future block: if remaining dates < horizon, STOP.
  - Future blocks should be non-overlapping by default (step = horizon unless configured).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowDef:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    future_start: pd.Timestamp
    future_end: pd.Timestamp
    factor_selection_asof_date: pd.Timestamp  # fixed to train_end by user choice


def _unique_sorted_dates(dates: Sequence[Any]) -> List[pd.Timestamp]:
    if dates is None:
        return []
    s = pd.to_datetime(pd.Series(list(dates)).dropna().unique())
    return sorted(pd.to_datetime(s))


def _resolve_days_from_ratio(n_dates: int, ratio: float, min_days: int) -> int:
    if n_dates <= 0:
        raise ValueError("n_dates must be > 0")
    ratio = float(ratio)
    if ratio <= 0.0:
        raise ValueError(f"ratio must be > 0, got: {ratio}")
    d = int(n_dates * ratio)
    d = max(d, int(min_days))
    return int(d)


def build_rankic_refit_roll_windows(
    dates: Sequence[Any],
    windowing_cfg: Dict[str, Any],
) -> List[WindowDef]:
    """
    Build windows for rankic_refit_roll from trading date index.

    windowing_cfg schema:
      mode: "days" | "ratio"

      # mode=days
      initial_train_days: int
      eval_test_days: int
      prediction_horizon_days: int
      step_days: int | null   (default => horizon)

      # mode=ratio (ratios on the FULL dataset)
      initial_train_ratio: float
      eval_test_ratio: float
      prediction_horizon_ratio: float
      step_ratio: float | null (default => horizon)
      min_initial_train_days: int
      min_eval_test_days: int
      min_prediction_horizon_days: int

      expanding_train: bool (default true)
      max_train_days: int | null (only when expanding_train=false)

    Hard rule:
      If a full future block cannot be formed, STOP (do not output partial).
    """
    dates_sorted = _unique_sorted_dates(dates)
    n = len(dates_sorted)
    if n == 0:
        return []

    mode = str(windowing_cfg.get("mode", "days")).lower()
    if mode not in ("days", "ratio"):
        raise ValueError(f"windowing.mode must be 'days' or 'ratio', got: {mode}")

    expanding_train = bool(windowing_cfg.get("expanding_train", True))
    max_train_days = windowing_cfg.get("max_train_days", None)
    max_train_days = None if max_train_days is None else int(max_train_days)

    if mode == "days":
        initial_train_days = int(windowing_cfg["initial_train_days"])
        eval_test_days = int(windowing_cfg["eval_test_days"])
        horizon_days = int(windowing_cfg["prediction_horizon_days"])
        step_days = windowing_cfg.get("step_days", None)
        step_days = int(step_days) if step_days is not None else int(horizon_days)
    else:
        initial_train_days = _resolve_days_from_ratio(
            n, float(windowing_cfg["initial_train_ratio"]), int(windowing_cfg.get("min_initial_train_days", 1))
        )
        eval_test_days = _resolve_days_from_ratio(
            n, float(windowing_cfg["eval_test_ratio"]), int(windowing_cfg.get("min_eval_test_days", 1))
        )
        horizon_days = _resolve_days_from_ratio(
            n, float(windowing_cfg["prediction_horizon_ratio"]), int(windowing_cfg.get("min_prediction_horizon_days", 1))
        )

        step_ratio = windowing_cfg.get("step_ratio", None)
        if step_ratio is None:
            step_days = int(horizon_days)
        else:
            step_days = _resolve_days_from_ratio(n, float(step_ratio), int(windowing_cfg.get("min_prediction_horizon_days", 1)))

    if initial_train_days <= 0 or eval_test_days <= 0 or horizon_days <= 0 or step_days <= 0:
        raise ValueError("windowing days must be > 0")

    if initial_train_days + eval_test_days >= n:
        raise ValueError(
            f"windowing invalid: initial_train_days+eval_test_days must be < n_dates, got {initial_train_days}+{eval_test_days} >= {n}"
        )

    windows: List[WindowDef] = []

    # The "node" is defined by the end of test evaluation
    test_end_idx = initial_train_days + eval_test_days - 1
    wid = 0

    while True:
        test_start_idx = test_end_idx - eval_test_days + 1
        train_end_idx = test_start_idx - 1

        future_start_idx = test_end_idx + 1
        future_end_idx = test_end_idx + horizon_days

        # stop if cannot build full future block (hard rule)
        if future_end_idx >= n:
            break
        if train_end_idx < 0:
            break

        train_end = dates_sorted[train_end_idx]
        test_start = dates_sorted[test_start_idx]
        test_end = dates_sorted[test_end_idx]
        future_start = dates_sorted[future_start_idx]
        future_end = dates_sorted[future_end_idx]

        if expanding_train:
            train_start = dates_sorted[0]
        else:
            if max_train_days is None or max_train_days <= 0:
                raise ValueError("windowing.expanding_train=false requires max_train_days > 0")
            # use last max_train_days ending at train_end_idx
            train_start_idx = max(0, train_end_idx - max_train_days + 1)
            train_start = dates_sorted[train_start_idx]

        windows.append(
            WindowDef(
                window_id=wid,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                future_start=future_start,
                future_end=future_end,
                factor_selection_asof_date=train_end,  # user confirmed
            )
        )

        wid += 1
        test_end_idx += step_days

    logger.info(
        "windowing built | mode=%s | n_dates=%d | n_windows=%d | initial_train_days=%d | eval_test_days=%d | horizon_days=%d | step_days=%d | expanding_train=%s",
        mode,
        n,
        int(len(windows)),
        int(initial_train_days),
        int(eval_test_days),
        int(horizon_days),
        int(step_days),
        bool(expanding_train),
    )

    return windows

