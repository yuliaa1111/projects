"""
ret_pred/evaluate/factor_eval_bridge.py

Bridge layer to integrate the existing factor evaluation system (run_pred_factor_eval.py / eval_all.pyc)
into this framework's evaluate module.

Hard constraints (see ret_pred/md/tasks.md, ret_pred/md/AGENTS.md):
- The following files are read-only dependencies; do NOT modify them:
  - ret_pred/run_pred_factor_eval.py
  - ret_pred/manage_db_read.pyc
  - ret_pred/db_settings.json
  - ret_pred/eval_all.pyc

Key semantics:
- rankic_refit_roll must use stitched future predictions for factor evaluation (NOT test evaluation preds).
- Other strategies should use their stitched out-of-sample prediction series (typically part='test').
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date as dt_date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import glob
import inspect
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorEvalDispatchResult:
    long_df: pd.DataFrame
    source: str
    meta: Dict[str, Any]


def _require(cfg: Dict[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"factor_eval missing required keys at {where}: {missing}")


def _read_pred_parquets(
    pred_dir: str,
    *,
    pred_glob: str,
    date_col: str,
    stockid_col: str,
    value_col: str,
    parts: Optional[List[str]] = None,
    part_col: str = "part",
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    pattern = os.path.join(str(pred_dir), str(pred_glob))
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files is not None:
        files = files[: int(max_files)]

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {pattern}")

    dfs: List[pd.DataFrame] = []
    skipped = 0
    for fp in files:
        df = pd.read_parquet(fp)

        # For factor eval we need at least date/stock/value (y_pred)
        need_cols = [date_col, stockid_col, value_col]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            skipped += 1
            continue

        tmp = df[need_cols + ([part_col] if part_col in df.columns else [])].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        dfs.append(tmp)

    if not dfs:
        raise ValueError(f"No valid prediction rows with required cols={need_cols} found under: {pattern}")

    out = pd.concat(dfs, ignore_index=True)
    if parts is not None:
        if part_col not in out.columns:
            raise KeyError(f"parts filter requires column '{part_col}' in preds, but it is missing")
        out = out[out[part_col].isin(list(parts))].copy()

    return out


def stitch_long_preds(
    long_df: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    keep: str = "last",
) -> pd.DataFrame:
    """
    Stitch predictions across windows into one long-format table:
    - Sort by (date, stockid)
    - Drop duplicate keys, keep last by default
    """
    if long_df.empty:
        return long_df

    out = long_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([date_col, stockid_col])
    out = out.drop_duplicates([date_col, stockid_col], keep=keep).reset_index(drop=True)
    return out


def build_factor_df(
    stitched_long_df: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Convert stitched long predictions (date, stockid, y_pred) to factor_df:
    - index: date
    - columns: stockid
    - values: y_pred
    """
    if stitched_long_df.empty:
        raise ValueError("build_factor_df: input is empty")

    need = {date_col, stockid_col, value_col}
    miss = [c for c in need if c not in stitched_long_df.columns]
    if miss:
        raise KeyError(f"build_factor_df: missing columns {miss}, got {list(stitched_long_df.columns)}")

    df = stitched_long_df[[date_col, stockid_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    factor_df = (
        df.pivot(index=date_col, columns=stockid_col, values=value_col)
        .sort_index()
    )
    return factor_df


def dispatch_factor_eval_by_strategy(
    *,
    strategy: str,
    input_source: str,
    pred_dir: str,
    eval_dir: str,
    date_col: str,
    stockid_col: str,
    value_col: str,
    pred_glob: str = "**/*.parquet",
    part_col: str = "part",
    part_name: str = "test",
    future_parquet_path: Optional[str] = None,
) -> FactorEvalDispatchResult:
    """
    Decide which prediction source should be used as factor evaluation input.

    input_source options:
    - auto:
        - rankic_refit_roll => future_stitched
        - others => stitched_part (default part='test')
    - future_stitched: use stitched future parquet (rankic_refit_roll)
    - stitched_part: stitch predictions from pred_dir filtered by part_name (default 'test')
    - explicit_parquet: read from future_parquet_path (must be provided)
    """
    strategy = str(strategy or "").lower()
    input_source = str(input_source or "auto").lower()

    if input_source == "auto":
        if strategy == "rankic_refit_roll":
            input_source = "future_stitched"
        else:
            input_source = "stitched_part"

    if input_source == "future_stitched":
        p = Path(future_parquet_path or (Path(eval_dir) / "future" / "future_preds_all.parquet"))
        if not p.exists():
            raise FileNotFoundError(f"future stitched parquet not found: {p}")
        df = pd.read_parquet(p)
        # Only keep necessary columns
        need_cols = [date_col, stockid_col, value_col]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            raise KeyError(f"future stitched parquet missing columns {miss}: {p}")

        long_df = df[need_cols].copy()
        long_df[date_col] = pd.to_datetime(long_df[date_col])
        stitched = stitch_long_preds(long_df, date_col=date_col, stockid_col=stockid_col, keep="last")
        return FactorEvalDispatchResult(
            long_df=stitched,
            source="future_stitched",
            meta={"strategy": strategy, "path": str(p)},
        )

    if input_source == "stitched_part":
        df = _read_pred_parquets(
            pred_dir,
            pred_glob=pred_glob,
            date_col=date_col,
            stockid_col=stockid_col,
            value_col=value_col,
            parts=[part_name],
            part_col=part_col,
        )
        stitched = stitch_long_preds(df, date_col=date_col, stockid_col=stockid_col, keep="last")
        return FactorEvalDispatchResult(
            long_df=stitched[[date_col, stockid_col, value_col]].copy(),
            source=f"stitched_part_{part_name}",
            meta={"strategy": strategy, "pred_dir": str(pred_dir), "pred_glob": str(pred_glob)},
        )

    if input_source == "explicit_parquet":
        if not future_parquet_path:
            raise ValueError("input_source=explicit_parquet requires future_parquet_path")
        p = Path(future_parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"explicit parquet not found: {p}")
        df = pd.read_parquet(p)
        need_cols = [date_col, stockid_col, value_col]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            raise KeyError(f"explicit parquet missing columns {miss}: {p}")
        stitched = stitch_long_preds(df[need_cols].copy(), date_col=date_col, stockid_col=stockid_col, keep="last")
        return FactorEvalDispatchResult(
            long_df=stitched,
            source="explicit_parquet",
            meta={"strategy": strategy, "path": str(p)},
        )

    raise ValueError(f"unknown input_source: {input_source}")


def run_factor_eval_from_preds(
    *,
    # dispatch
    strategy: str,
    input_source: str,
    pred_dir: str,
    eval_dir: str,
    date_col: str,
    stockid_col: str,
    value_col: str = "y_pred",
    pred_glob: str = "**/*.parquet",
    part_name: str = "test",
    future_parquet_path: Optional[str] = None,
    # factor analysis params
    analysis_params: Dict[str, Any],
    # output
    out_dir: str,
    excess_mode: str = "both",  # true | false | both
    excess_output_layout: str = "subdirs",  # subdirs | separate_dirs
    meta_filename: str = "factor_eval_meta.json",
) -> Dict[str, Any]:
    """
    End-to-end runner:
    - dispatch input predictions
    - build factor_df (wide)
    - connect client using existing system
    - call complete_factor_analysis (existing system)
    - save outputs into separate dirs for excess_return True/False if needed

    Note:
    - `analysis_params` MUST be fully provided by config (no implicit defaults here).
    - This function is intended to run on server where clickhouse settings path is available.
    """
    _require(analysis_params, ["factor_name", "save_figures", "return_data", "verbose"], where="analysis_params")

    excess_output_layout = str(excess_output_layout or "subdirs").lower()
    if excess_output_layout not in ("subdirs", "separate_dirs"):
        raise ValueError(f"excess_output_layout must be subdirs|separate_dirs, got: {excess_output_layout}")

    disp = dispatch_factor_eval_by_strategy(
        strategy=strategy,
        input_source=input_source,
        pred_dir=pred_dir,
        eval_dir=eval_dir,
        date_col=date_col,
        stockid_col=stockid_col,
        value_col=value_col,
        pred_glob=pred_glob,
        part_name=part_name,
        future_parquet_path=future_parquet_path,
    )
    factor_df = build_factor_df(disp.long_df, date_col=date_col, stockid_col=stockid_col, value_col=value_col)

    # Lazy imports: server-only dependencies
    from ret_pred.run_pred_factor_eval import connect_client  # read-only file
    from eval_all import complete_factor_analysis  # type: ignore

    client = connect_client()

    excess_mode = str(excess_mode or "both").lower()
    if excess_mode not in ("true", "false", "both"):
        raise ValueError(f"excess_mode must be true|false|both, got: {excess_mode}")

    out_base = Path(out_dir)

    results: Dict[str, Any] = {
        "strategy": str(strategy),
        "input_source": str(disp.source),
        "meta": dict(disp.meta or {}),
        "requested_out_dir": str(out_base),
        "excess_mode": str(excess_mode),
        "excess_output_layout": str(excess_output_layout),
    }

    def _jsonable(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    def _resolve_save_dir(excess_return: bool) -> Path:
        tag = "excess_true" if excess_return else "excess_false"
        if excess_output_layout == "subdirs":
            return out_base / tag

        # separate_dirs: recommended naming: factor_eval_{strategy}_{source}_excess_true/
        name = out_base.name
        if tag in name:
            return out_base
        return out_base.with_name(f"{name}_{tag}")

    def _write_meta(save_dir: Path, *, excess_return: bool) -> None:
        payload = {
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "strategy": str(strategy),
            "requested_input_source": str(input_source),
            "resolved_input_source": str(disp.source),
            "pred_dir": str(pred_dir),
            "pred_glob": str(pred_glob),
            "eval_dir": str(eval_dir),
            "date_col": str(date_col),
            "stockid_col": str(stockid_col),
            "value_col": str(value_col),
            "part_name": str(part_name),
            "future_parquet_path": str(future_parquet_path) if future_parquet_path else None,
            "out_dir": str(save_dir),
            "excess_return": bool(excess_return),
            "analysis_params": _jsonable(dict(analysis_params)),
            "factor_df_shape": [int(factor_df.shape[0]), int(factor_df.shape[1])],
            "dispatch_meta": _jsonable(dict(disp.meta or {})),
        }
        p = save_dir / str(meta_filename)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_one(excess_return: bool) -> str:
        save_dir = _resolve_save_dir(excess_return)
        save_dir.mkdir(parents=True, exist_ok=True)

        params = dict(analysis_params)
        # eval_all.complete_factor_analysis expects datetime-like values
        # for start_date/end_date (it calls `.date()` internally).
        for k in ("start_date", "end_date"):
            if k not in params or params[k] is None:
                continue
            v = params[k]
            if isinstance(v, datetime):
                pass
            elif isinstance(v, dt_date):
                params[k] = datetime(v.year, v.month, v.day)
            else:
                ts = pd.to_datetime(v, errors="raise")
                if isinstance(ts, pd.Timestamp):
                    params[k] = ts.to_pydatetime()
                else:
                    raise TypeError(f"factor_eval analysis_params.{k} must be datetime-like, got: {type(v)}")

        params["input_factor_df"] = factor_df
        params["client"] = client
        params["save_dir"] = str(save_dir)
        params["excess_return"] = bool(excess_return)

        # Compatibility: different server versions of eval_all.complete_factor_analysis
        # may expose different kwargs (e.g., turnover-fee args). Keep only supported ones.
        try:
            sig = inspect.signature(complete_factor_analysis)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw:
                allowed = set(sig.parameters.keys())
                dropped = sorted([k for k in params.keys() if k not in allowed])
                if dropped:
                    logger.warning(
                        "complete_factor_analysis unsupported kwargs dropped: %s",
                        dropped,
                    )
                    params = {k: v for k, v in params.items() if k in allowed}
        except Exception:
            logger.exception("failed to inspect complete_factor_analysis signature; fallback to raw kwargs")

        complete_factor_analysis(**params)
        _write_meta(save_dir, excess_return=excess_return)
        return str(save_dir)

    if excess_mode in ("true", "both"):
        results["save_dir_excess_true"] = _run_one(True)
    if excess_mode in ("false", "both"):
        results["save_dir_excess_false"] = _run_one(False)

    logger.info(
        "factor_eval done | strategy=%s | source=%s | excess_mode=%s | out_dir=%s",
        str(strategy),
        str(disp.source),
        excess_mode,
        str(out_base),
    )
    return results
