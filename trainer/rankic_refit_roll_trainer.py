"""
ret_pred/trainer/rankic_refit_roll_trainer.py

Trainer strategy: rankic_refit_roll

Semantics (see ret_pred/md/tasks.md, ret_pred/md/data.md):
Each round has 4 stages:
1) select factor set for train/test by configured anchor date
2) train -> test evaluation (eval model trained on train split, evaluated on test split)
3) select factor set for future by configured future anchor date, then refit final model on train+test
4) predict future block (N days) AFTER test end; future blocks are stitched for backtest

Hard constraints:
- train/test and future may use different factor sets within one round (per user workflow)
- strict separation between test evaluation and future prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

from .registry import register_trainer
from .plugins import METRIC_FN, build_saver
from .rolling_trainer import RollingTrainer
from ret_pred.cut import datacut_long
from ret_pred.factor_selection import load_predictions_pkl, select_factors_by_rankic, save_selected_factors_json
from ret_pred.windowing_rankic_refit_roll import build_rankic_refit_roll_windows

logger = logging.getLogger(__name__)


def _unique_sorted_dates(dates: Sequence[Any]) -> List[pd.Timestamp]:
    d = pd.to_datetime(pd.Series(list(dates)).dropna().unique())
    return sorted(pd.to_datetime(d))


def _combine_payloads(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine two payloads (train + test) into one for refit.
    Supports tree payload (X) and seq payload (X_seq).
    """
    out = dict(a)

    if "X" in a and "X" in b:
        Xa, Xb = a["X"], b["X"]
        if hasattr(Xa, "iloc") and hasattr(Xb, "iloc"):
            out["X"] = pd.concat([Xa, Xb], axis=0, ignore_index=True)
        else:
            out["X"] = np.concatenate([np.asarray(Xa), np.asarray(Xb)], axis=0)

    if "X_seq" in a and "X_seq" in b:
        out["X_seq"] = np.concatenate([np.asarray(a["X_seq"]), np.asarray(b["X_seq"])], axis=0)

    out["y"] = np.concatenate([np.asarray(a["y"]).reshape(-1), np.asarray(b["y"]).reshape(-1)], axis=0)

    ka, kb = a.get("keys"), b.get("keys")
    if ka is not None and kb is not None:
        out["keys"] = pd.concat([ka, kb], axis=0, ignore_index=True)

    out["feature_cols"] = list(a.get("feature_cols") or b.get("feature_cols") or [])
    return out


def _read_part_from_preprocess(
    preprocess_path: str,
    *,
    date_col: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
    return pd.read_parquet(
        preprocess_path,
        filters=[(date_col, ">=", pd.Timestamp(start_date)), (date_col, "<=", pd.Timestamp(end_date))],
        columns=columns,
    )


def _get_preprocess_columns(preprocess_path: str) -> List[str]:
    """
    Read preprocess parquet schema columns without loading full data.
    """
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(preprocess_path).schema.names)
    except Exception:
        logger.warning("failed to read preprocess schema via pyarrow, fallback pandas | path=%s", preprocess_path)
        return list(pd.read_parquet(preprocess_path).columns)


def _infer_numeric_cols_from_schema(
    preprocess_path: str,
    *,
    exclude: Optional[set] = None,
) -> List[str]:
    """
    Infer numeric columns from parquet schema (pyarrow-first).
    """
    exclude = exclude or set()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(preprocess_path)
        schema = pf.schema_arrow
        out: List[str] = []
        for f in schema:
            if f.name in exclude:
                continue
            t = f.type
            if (
                pa.types.is_integer(t)
                or pa.types.is_floating(t)
                or pa.types.is_decimal(t)
                or pa.types.is_boolean(t)
            ):
                out.append(str(f.name))
        return out
    except Exception:
        logger.warning("failed to infer numeric columns from schema, fallback pandas dtypes | path=%s", preprocess_path)
        df = pd.read_parquet(preprocess_path)
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _resolve_window_anchor_date(w: Any, anchor: Optional[str], *, default: str = "train_start") -> pd.Timestamp:
    """
    Resolve a date anchor from WindowDef by field name.
    """
    name = str(anchor or default)
    if not hasattr(w, name):
        name = default
    return pd.Timestamp(getattr(w, name))


@dataclass
class RankICRefitRollParams:
    # factor selection
    factor_selection: Optional[Dict[str, Any]] = None

    # saver (optional)
    saver: Optional[Dict[str, Any]] = None

    # artifacts
    artifacts_dir: str = ""

    # windowing (required)
    windowing: Optional[Dict[str, Any]] = None


@register_trainer("rankic_refit_roll")
class RankICRefitRollTrainer:
    def __init__(
        self,
        model: Dict[str, Any],
        *,
        preprocess_path: str,
        datacutting_cfg: Dict[str, Any],
        metric: str = "rankic",
        maximize: bool = True,
        task: str = "regression",
        loss: Optional[Dict[str, Any]] = None,
        nn_fit: Optional[Dict[str, Any]] = None,
        run_id: str = "exp001",
        date_col: str = "date",
        stockid_col: str = "stockid",
        label_col: str = "y",
        device: str = "cpu",
        seed: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.model_cfg = dict(model or {})
        self.preprocess_path = str(preprocess_path)
        self.cut_cfg_base = dict(datacutting_cfg or {})

        self.metric_name = str(metric)
        self.maximize = bool(maximize)
        self.task = str(task)
        self.loss_cfg = loss or {"name": "mse", "params": {}}
        self.nn_fit_cfg = nn_fit or {"epochs": 5, "lr": 1e-3, "batch_size": 256}

        self.run_id = str(run_id)
        self.date_col = str(date_col)
        self.stockid_col = str(stockid_col)
        self.label_col = str(label_col)
        self.device = str(device)
        self.seed = int(seed)

        params = params or {}
        try:
            self.p = RankICRefitRollParams(
                factor_selection=(params.get("factor_selection") or None),
                saver=(params.get("saver") or None),
                artifacts_dir=str(params.get("artifacts_dir", "")),
                windowing=(params.get("windowing") or None),
            )
        except KeyError as e:
            raise KeyError(
                "rankic_refit_roll requires trainer.params.windowing"
            ) from e

        if self.metric_name not in METRIC_FN:
            raise KeyError(f"Unknown metric '{self.metric_name}', available: {list(METRIC_FN.keys())}")

        # saver (optional)
        self.save_enabled, self.saver = build_saver(self.p.saver)

        # factor selection (optional)
        self.fs_cfg = dict(self.p.factor_selection or {})
        self.fs_enabled = bool(self.fs_cfg.get("enabled", False))
        self.sample_weighting_cfg = dict(params.get("sample_weighting", {}) or {})
        self._pred_map = None
        if self.fs_enabled:
            pkl_path = str(self.fs_cfg.get("predictions_pkl_path", ""))
            if not pkl_path:
                raise ValueError("factor_selection.enabled=true but factor_selection.predictions_pkl_path is empty")
            self._pred_map = load_predictions_pkl(pkl_path)
            logger.info("factor_selection loaded | path=%s | n_dates=%d", pkl_path, int(len(self._pred_map)))

        # helper trainer (reuse fit/predict/score/save implementations)
        # - tuner/schedule/gate disabled by not using RollingTrainer.run
        self._helper = RollingTrainer(
            model=self.model_cfg,
            metric=self.metric_name,
            maximize=self.maximize,
            task=self.task,
            loss=self.loss_cfg,
            nn_fit=self.nn_fit_cfg,
            schedule=None,
            tuner={"enabled": False},
            update_gate=None,
            saver=self.p.saver,
            model_save={"enabled": False},
            sample_weighting=self.sample_weighting_cfg,
            run_id=self.run_id,
            date_col=self.date_col,
            stockid_col=self.stockid_col,
            device=self.device,
            seed=self.seed,
        )

        logger.info(
            "RankICRefitRollTrainer init | run_id=%s | model=%s | metric=%s | saver=%s | factor_selection=%s",
            self.run_id,
            str(self.model_cfg.get("name", "")),
            self.metric_name,
            bool(self.save_enabled),
            bool(self.fs_enabled),
        )

    def _select_factor_set(self, *, select_date: pd.Timestamp, candidate_pool: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        if not self.fs_enabled or self._pred_map is None:
            return list(candidate_pool), {"enabled": False, "select_date": str(select_date)[:10], "n_selected": int(len(candidate_pool))}

        selection_mode = str(self.fs_cfg.get("selection_mode", "historical_aggregate"))
        selected, state = select_factors_by_rankic(
            self._pred_map,
            asof_date=select_date,
            threshold=float(self.fs_cfg.get("rankic_threshold", self.fs_cfg.get("threshold", 0.03))),
            aggregate=str(self.fs_cfg.get("aggregate", "abs_mean")),
            inclusive=bool(self.fs_cfg.get("cutoff_inclusive", True)),
            min_history_days=int(self.fs_cfg.get("min_history_days", 1)),
            candidate_pool=candidate_pool,
            selection_mode=selection_mode,
        )
        return selected, {"enabled": True, **state}

    def _cut_part(self, part_df: pd.DataFrame, *, feature_cols: List[str], fold: int, part: str) -> Dict[str, Any]:
        cut_cfg = dict(self.cut_cfg_base)
        cut_cfg["label_col"] = self.label_col
        cut_cfg["feature_cols"] = list(feature_cols)
        # mode selection stays in windows/build_streaming_windows for rolling trainer;
        # here we infer from model family for consistency.
        family = str(self.model_cfg.get("family", "tree"))
        cut_cfg["mode"] = "tree" if family == "tree" else "seq"

        payload, _ = datacut_long(part_df, cut_cfg, meta={"fold": fold, "part": part})
        return payload

    def run(self, dates: Sequence[Any]) -> pd.DataFrame:
        """
        Run refit-roll strategy on a date index (unique sorted trading dates).
        """
        dates_sorted = _unique_sorted_dates(dates)
        if len(dates_sorted) == 0:
            raise ValueError("rankic_refit_roll: empty dates")

        win_cfg = dict(self.p.windowing or {})
        windows = build_rankic_refit_roll_windows(dates_sorted, win_cfg)
        if len(windows) == 0:
            raise ValueError("rankic_refit_roll: produced 0 windows; check trainer.params.windowing")

        artifacts_dir = self.p.artifacts_dir or str(Path("./runs") / self.run_id)
        artifacts_dir = str(artifacts_dir).replace("{run_id}", self.run_id)
        factors_dir = str(Path(artifacts_dir) / "selected_factors")

        records: List[Dict[str, Any]] = []
        all_cols = _get_preprocess_columns(self.preprocess_path)
        cols_set = set(all_cols)
        required = {self.date_col, self.stockid_col, self.label_col}
        miss = [c for c in required if c not in cols_set]
        if miss:
            raise KeyError(f"preprocess parquet missing required columns: {miss}")

        # Determine base candidate pool from parquet schema (avoid loading a data slice).
        exclude = {self.date_col, self.stockid_col, self.label_col, f"{self.label_col}_raw"}
        base_pool = _infer_numeric_cols_from_schema(self.preprocess_path, exclude=exclude)
        base_pool = [c for c in base_pool if c in cols_set]

        if len(base_pool) == 0:
            raise ValueError("rankic_refit_roll: candidate_pool is empty after preprocess schema filtering")

        selection_mode = str(self.fs_cfg.get("selection_mode", "historical_aggregate")).lower()
        train_test_anchor = str(self.fs_cfg.get("train_test_anchor", "train_start"))
        future_anchor = str(self.fs_cfg.get("future_anchor", "future_start"))
        future_reselect = bool(self.fs_cfg.get("future_reselect", selection_mode == "single_day_threshold"))

        logger.info(
            "rankic_refit_roll run start | n_windows=%d | candidate_pool=%d | selection_mode=%s | train_test_anchor=%s | future_anchor=%s | future_reselect=%s",
            int(len(windows)),
            int(len(base_pool)),
            selection_mode,
            train_test_anchor,
            future_anchor,
            bool(future_reselect),
        )

        for w in windows:
            fold = int(w.window_id)
            rec_base = {
                "window_id": fold,
                "train_start": str(w.train_start)[:10],
                "train_end": str(w.train_end)[:10],
                "test_start": str(w.test_start)[:10],
                "test_end": str(w.test_end)[:10],
                "future_start": str(w.future_start)[:10],
                "future_end": str(w.future_end)[:10],
            }

            policy = str(self.fs_cfg.get("empty_selection_policy", "fallback_base_pool")).lower()

            def _normalize_selection(raw_selected: List[str], raw_state: Dict[str, Any], *, tag: str) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
                state = dict(raw_state or {})
                selected = [str(x) for x in list(raw_selected or [])]
                selected = [f for f in selected if f in cols_set and f not in required]
                if len(selected) > 0:
                    return selected, state, None
                if policy == "fallback_base_pool":
                    fb = [f for f in base_pool if f in cols_set and f not in required]
                    if len(fb) == 0:
                        return [], state, f"{tag}_fallback_base_pool_empty"
                    state[f"{tag}_empty_selection_fallback"] = "base_pool"
                    return fb, state, None
                return [], state, f"{tag}_empty_selected_factors"

            train_select_date = _resolve_window_anchor_date(w, train_test_anchor, default="train_start")
            train_selected_raw, fs_train_state = self._select_factor_set(select_date=train_select_date, candidate_pool=base_pool)
            train_factors, fs_train_state, skip_reason = _normalize_selection(train_selected_raw, fs_train_state, tag="train_test")
            if skip_reason is not None:
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": 0,
                        "n_selected_factors_train_test": 0,
                        "n_selected_factors_future": 0,
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": None,
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }
                )
                logger.warning("window skipped | id=%d | reason=%s", fold, skip_reason)
                continue

            future_select_date = _resolve_window_anchor_date(w, future_anchor, default="future_start")
            if future_reselect:
                future_selected_raw, fs_future_state = self._select_factor_set(select_date=future_select_date, candidate_pool=base_pool)
            else:
                future_selected_raw = list(train_factors)
                fs_future_state = {
                    "enabled": bool(self.fs_enabled),
                    "selection_mode": "inherit_train_test",
                    "select_date": str(future_select_date)[:10],
                    "n_selected": int(len(future_selected_raw)),
                }
            future_factors, fs_future_state, skip_reason = _normalize_selection(future_selected_raw, fs_future_state, tag="future")
            if skip_reason is not None:
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": 0,
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }
                )
                logger.warning("window skipped | id=%d | reason=%s", fold, skip_reason)
                continue

            # read only needed columns (reduce IO)
            cols = [self.date_col, self.stockid_col, self.label_col] + list(train_factors) + list(future_factors)
            cols = list(dict.fromkeys(cols))

            # train (<= train_end), test (test_start..test_end), refit (<= test_end), future (future_start..future_end)
            df_train = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.train_start,
                end_date=w.train_end,
                columns=cols,
            )
            df_test = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.test_start,
                end_date=w.test_end,
                columns=cols,
            )
            df_future = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.future_start,
                end_date=w.future_end,
                columns=cols,
            )

            if df_train.empty or df_test.empty or df_future.empty:
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": int(len(future_factors)),
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_window_part",
                        "n_train_rows": int(len(df_train)),
                        "n_test_rows": int(len(df_test)),
                        "n_future_rows": int(len(df_future)),
                    }
                )
                logger.warning(
                    "window skipped | id=%d | reason=empty_window_part | n_train=%d n_test=%d n_future=%d",
                    fold, int(len(df_train)), int(len(df_test)), int(len(df_future)),
                )
                continue

            # train/test use train_factors; future can use future_factors (re-selected at future anchor).
            train_pl = self._cut_part(df_train, feature_cols=train_factors, fold=fold, part="train")
            test_pl = self._cut_part(df_test, feature_cols=train_factors, fold=fold, part="test")
            refit_train_pl = self._cut_part(df_train, feature_cols=future_factors, fold=fold, part="train_refit")
            refit_test_pl = self._cut_part(df_test, feature_cols=future_factors, fold=fold, part="test_refit")
            future_pl = self._cut_part(df_future, feature_cols=future_factors, fold=fold, part="future")

            if (
                self._helper._is_empty_payload(train_pl)
                or self._helper._is_empty_payload(test_pl)
                or self._helper._is_empty_payload(refit_train_pl)
                or self._helper._is_empty_payload(refit_test_pl)
                or self._helper._is_empty_payload(future_pl)
            ):
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": int(len(future_factors)),
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_payload_after_cut",
                    }
                )
                logger.warning("window skipped | id=%d | reason=empty_payload_after_cut", fold)
                continue

            meta = {
                "run_id": self.run_id,
                "fold": fold,
                "step_id": fold,
                "train_factor_select_date": str(train_select_date)[:10],
                "future_factor_select_date": str(future_select_date)[:10],
                "n_features_train_test": int(len(train_factors)),
                "n_features_future": int(len(future_factors)),
                "train_end": str(w.train_end)[:10],
                "test_start": str(w.test_start)[:10],
                "test_end": str(w.test_end)[:10],
                "future_start": str(w.future_start)[:10],
                "future_end": str(w.future_end)[:10],
            }

            # ---- 2) eval model: train -> test evaluation ----
            eval_model, _ = self._helper._fit_one_window(train_pl, None, self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {}))
            train_pred = self._helper._predict(eval_model, train_pl)
            test_pred = self._helper._predict(eval_model, test_pl)
            test_score = float(self._helper._score(test_pl, test_pred))

            saved_train = None
            if self.save_enabled and self.saver is not None:
                saved_train = self._helper._save_preds(
                    train_pl,
                    train_pred,
                    meta,
                    part="train",
                    params=(self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {})),
                )

            saved_test = None
            if self.save_enabled and self.saver is not None:
                saved_test = self._helper._save_preds(test_pl, test_pred, meta, part="test", params=(self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {})))

            # ---- 3) refit final model on train+test (future factor set) ----
            refit_pl = _combine_payloads(refit_train_pl, refit_test_pl)
            final_model, _ = self._helper._fit_one_window(refit_pl, None, self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {}))

            # ---- 4) future prediction (separate from test eval) ----
            future_pred = self._helper._predict(final_model, future_pl)
            future_score = float(self._helper._score(future_pl, future_pred))

            saved_future = None
            if self.save_enabled and self.saver is not None:
                saved_future = self._helper._save_preds(future_pl, future_pred, meta, part="future", params=(self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {})))

            # save selected factor set artifact (per window)
            sf_path_train = str(Path(factors_dir) / f"selected_factors_window_{fold:04d}.json")
            save_selected_factors_json(train_factors, fs_train_state, out_path=sf_path_train)
            sf_path_future = str(Path(factors_dir) / f"selected_factors_window_{fold:04d}_future.json")
            save_selected_factors_json(future_factors, fs_future_state, out_path=sf_path_future)

            records.append(
                {
                    **rec_base,
                    "n_selected_factors": int(len(train_factors)),
                    "n_selected_factors_train_test": int(len(train_factors)),
                    "n_selected_factors_future": int(len(future_factors)),
                    "factor_selection": fs_train_state,
                    "factor_selection_train_test": fs_train_state,
                    "factor_selection_future": fs_future_state,
                    "test_score": float(test_score),
                    "future_score": float(future_score),
                    "saved_train": saved_train,
                    "saved_test": saved_test,
                    "saved_future": saved_future,
                    "skipped": False,
                }
            )

            logger.info(
                "window done | id=%d | n_factors_train_test=%d | n_factors_future=%d | test_score=%.6f | future_score=%.6f",
                fold,
                int(len(train_factors)),
                int(len(future_factors)),
                float(test_score),
                float(future_score),
            )

        hist_df = pd.DataFrame(records)
        logger.info("rankic_refit_roll run end | n_windows=%d", int(len(hist_df)))
        return hist_df
