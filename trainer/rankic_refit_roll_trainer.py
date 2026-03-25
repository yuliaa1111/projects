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
import copy
import gc
import logging

import numpy as np
import pandas as pd

from .registry import register_trainer
from .plugins import METRIC_FN, build_saver
from .rolling_trainer import RollingTrainer
from ret_pred.cut import datacut_long
from ret_pred.factor_selection import (
    load_factor_use_flags_csv,
    load_predictions_pkl,
    save_selected_factors_json,
    select_factors_by_rankic,
    select_factors_by_use_flags,
)
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


def _payload_X_as_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    if "X" not in payload:
        raise ValueError("PCA currently supports tree payload with key 'X' only")
    X = payload["X"]
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"PCA expects 2D X, got shape={arr.shape}")
    cols = list(payload.get("feature_cols") or [f"f{i}" for i in range(arr.shape[1])])
    return pd.DataFrame(arr, columns=cols)


def _clone_payload_with_X(payload: Dict[str, Any], X_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    out = dict(payload)
    out["X"] = X_df
    out["feature_cols"] = list(feature_cols)
    return out


def _fit_pca_state(train_payload: Dict[str, Any], pca_cfg: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("pca.enabled=true requires scikit-learn") from e

    X_train = _payload_X_as_dataframe(train_payload)
    if X_train.shape[1] == 0:
        raise ValueError("pca.enabled=true but feature dimension is 0")

    standardize = bool(pca_cfg.get("standardize", True))
    whiten = bool(pca_cfg.get("whiten", False))
    svd_solver = str(pca_cfg.get("svd_solver", "auto"))
    random_state = pca_cfg.get("random_state", None)

    n_components = pca_cfg.get("n_components", None)
    explained_ratio = pca_cfg.get("explained_variance_ratio", None)
    if n_components is not None and explained_ratio is not None:
        raise ValueError("pca config: set either n_components or explained_variance_ratio, not both")

    if explained_ratio is not None:
        n_comp = float(explained_ratio)
        if not (0.0 < n_comp <= 1.0):
            raise ValueError(f"pca.explained_variance_ratio must be in (0,1], got: {n_comp}")
    elif n_components is not None:
        n_comp = int(n_components)
        if n_comp <= 0:
            raise ValueError(f"pca.n_components must be > 0, got: {n_comp}")
        n_comp = min(n_comp, int(X_train.shape[1]))
    else:
        # default keep 95% variance
        n_comp = 0.95

    scaler = None
    X_fit = X_train
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=list(X_train.columns),
            index=X_train.index,
        )

    pca = PCA(
        n_components=n_comp,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    X_pc = pca.fit_transform(X_fit.to_numpy())
    pc_cols = [f"{prefix}_{i:03d}" for i in range(X_pc.shape[1])]

    return {
        "scaler": scaler,
        "pca": pca,
        "pc_cols": pc_cols,
        "n_input_features": int(X_train.shape[1]),
        "n_output_features": int(X_pc.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(getattr(pca, "explained_variance_ratio_", np.array([])))),
    }


def _transform_payload_with_pca(payload: Dict[str, Any], pca_state: Dict[str, Any]) -> Dict[str, Any]:
    X = _payload_X_as_dataframe(payload)
    scaler = pca_state.get("scaler", None)
    pca = pca_state["pca"]
    pc_cols = list(pca_state["pc_cols"])

    X_in: Any = X
    if scaler is not None:
        # Keep feature names to avoid sklearn validation warnings.
        X_in = scaler.transform(X_in)
    if isinstance(X_in, pd.DataFrame):
        X_in = X_in.to_numpy()
    X_pc = pca.transform(X_in)
    X_df = pd.DataFrame(X_pc, columns=pc_cols, index=X.index)
    return _clone_payload_with_X(payload, X_df, pc_cols)


def _pca_state_for_record(pca_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only parquet/json-friendly PCA state fields for history records.
    """
    if not isinstance(pca_state, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in (
        "enabled",
        "n_input_features",
        "n_output_features",
        "explained_variance_ratio_sum",
        "pc_cols",
    ):
        if k not in pca_state:
            continue
        v = pca_state.get(k)
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = v
    return out


def _build_pseudo_train_df(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not bool((cfg or {}).get("enabled", False)):
        return df, {"enabled": False, "n_in": int(len(df)), "n_out": int(len(df))}

    y_source_col = str(cfg.get("y_source_col", f"{label_col}_raw"))
    if y_source_col not in df.columns:
        if bool(cfg.get("allow_fallback_to_label", True)) and label_col in df.columns:
            logger.warning(
                "pseudo_classification y_source_col missing, fallback to label_col | y_source_col=%s label_col=%s",
                y_source_col,
                label_col,
            )
            y_source_col = label_col
        else:
            raise KeyError(
                f"pseudo_classification enabled but y_source_col '{y_source_col}' not in dataframe columns"
            )

    lower_q = float(cfg.get("lower_q", 0.30))
    upper_q = float(cfg.get("upper_q", 0.70))
    if not (0.0 < lower_q < upper_q < 1.0):
        raise ValueError(f"pseudo_classification quantiles invalid: lower_q={lower_q}, upper_q={upper_q}")

    low_label = float(cfg.get("low_label", -1.0))
    high_label = float(cfg.get("high_label", 1.0))
    scope = str(cfg.get("scope", "date")).lower()
    if scope not in ("date", "global"):
        raise ValueError(f"pseudo_classification.scope must be date|global, got: {scope}")

    out = df.copy()
    src = pd.to_numeric(out[y_source_col], errors="coerce")

    if scope == "date":
        lo = out.groupby(date_col, sort=False)[y_source_col].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(lower_q))
        hi = out.groupby(date_col, sort=False)[y_source_col].transform(lambda s: pd.to_numeric(s, errors="coerce").quantile(upper_q))
    else:
        lo_v = float(src.quantile(lower_q))
        hi_v = float(src.quantile(upper_q))
        lo = pd.Series(lo_v, index=out.index)
        hi = pd.Series(hi_v, index=out.index)

    low_mask = src <= lo
    high_mask = src >= hi
    keep = (low_mask | high_mask) & src.notna()

    fit_df = out.loc[keep].copy()
    fit_df[label_col] = np.where(high_mask.loc[fit_df.index], high_label, low_label).astype(np.float32)

    min_samples = int(cfg.get("min_samples", 10))
    if len(fit_df) < min_samples:
        raise ValueError(
            f"pseudo_classification kept too few samples: {len(fit_df)} < min_samples={min_samples}"
        )

    n_low = int(low_mask.loc[fit_df.index].sum())
    n_high = int(high_mask.loc[fit_df.index].sum())
    if n_low == 0 or n_high == 0:
        raise ValueError(
            f"pseudo_classification one class empty after filtering: n_low={n_low}, n_high={n_high}"
        )

    st = {
        "enabled": True,
        "y_source_col": y_source_col,
        "scope": scope,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "low_label": low_label,
        "high_label": high_label,
        "n_in": int(len(out)),
        "n_out": int(len(fit_df)),
        "n_low": int(n_low),
        "n_high": int(n_high),
    }
    return fit_df, st


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
        self.pca_cfg = dict(params.get("pca", {}) or {})
        self.pseudo_cls_cfg = dict(params.get("pseudo_classification", {}) or {})
        self.optuna_cfg = dict(params.get("optuna", {}) or {})
        self.fs_source = str(self.fs_cfg.get("source", "predictions_pkl")).strip().lower()
        self._pred_map = None
        self._flags_map = None
        if self.fs_enabled:
            if self.fs_source in {"predictions_pkl", "pkl", "rankic_pkl"}:
                pkl_path = str(self.fs_cfg.get("predictions_pkl_path", ""))
                if not pkl_path:
                    raise ValueError(
                        "factor_selection.source=predictions_pkl but factor_selection.predictions_pkl_path is empty"
                    )
                self._pred_map = load_predictions_pkl(pkl_path)
                logger.info(
                    "factor_selection loaded | source=predictions_pkl | path=%s | n_dates=%d",
                    pkl_path,
                    int(len(self._pred_map)),
                )
            elif self.fs_source in {"factor_use_flags_csv", "use_flags_csv"}:
                csv_path = str(self.fs_cfg.get("factor_use_flags_csv_path", ""))
                if not csv_path:
                    raise ValueError(
                        "factor_selection.source=factor_use_flags_csv but factor_selection.factor_use_flags_csv_path is empty"
                    )
                self._flags_map = load_factor_use_flags_csv(csv_path, date_col="date")
                logger.info(
                    "factor_selection loaded | source=factor_use_flags_csv | path=%s | n_dates=%d",
                    csv_path,
                    int(len(self._flags_map)),
                )
            else:
                raise ValueError(
                    f"factor_selection.source must be predictions_pkl|factor_use_flags_csv, got: {self.fs_source}"
                )

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

    def _select_factor_set(
        self,
        *,
        select_date: pd.Timestamp,
        candidate_pool: List[str],
        min_selected: int = 0,
        min_selected_mode: str = "none",
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not self.fs_enabled:
            return list(candidate_pool), {"enabled": False, "select_date": str(select_date)[:10], "n_selected": int(len(candidate_pool))}

        if self.fs_source in {"predictions_pkl", "pkl", "rankic_pkl"}:
            if self._pred_map is None:
                raise RuntimeError("factor_selection source=predictions_pkl but map is not loaded")
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
                min_selected=int(min_selected),
                min_selected_mode=str(min_selected_mode),
            )
            return selected, {"enabled": True, "source": "predictions_pkl", **state}

        if self.fs_source in {"factor_use_flags_csv", "use_flags_csv"}:
            if self._flags_map is None:
                raise RuntimeError("factor_selection source=factor_use_flags_csv but map is not loaded")
            selected, state = select_factors_by_use_flags(
                self._flags_map,
                select_date=select_date,
                candidate_pool=candidate_pool,
            )
            return selected, {"enabled": True, "source": "factor_use_flags_csv", **state}

        raise ValueError(f"unknown factor_selection.source: {self.fs_source}")

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

    def _should_run_optuna(
        self,
        *,
        anchor_date: pd.Timestamp,
        last_tune_date: Optional[pd.Timestamp],
        cfg: Dict[str, Any],
    ) -> bool:
        if not bool(cfg.get("enabled", False)):
            return False
        if last_tune_date is None:
            return bool(cfg.get("tune_first_window", True))
        every_n_days = int(cfg.get("every_n_days", 5))
        if every_n_days <= 0:
            return False
        return int((pd.Timestamp(anchor_date) - pd.Timestamp(last_tune_date)).days) >= every_n_days

    def _suggest_optuna_params(self, trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, spec in (search_space or {}).items():
            name = str(k)
            if isinstance(spec, dict):
                typ = str(spec.get("type", "")).lower()
                if typ == "float":
                    out[name] = trial.suggest_float(
                        name,
                        float(spec["low"]),
                        float(spec["high"]),
                        log=bool(spec.get("log", False)),
                        step=spec.get("step", None),
                    )
                elif typ == "int":
                    out[name] = trial.suggest_int(
                        name,
                        int(spec["low"]),
                        int(spec["high"]),
                        step=int(spec.get("step", 1)),
                        log=bool(spec.get("log", False)),
                    )
                elif typ == "categorical":
                    out[name] = trial.suggest_categorical(name, list(spec.get("choices", [])))
                else:
                    raise ValueError(f"optuna.search_space[{name}] unknown type: {typ}")
            elif isinstance(spec, list):
                out[name] = trial.suggest_categorical(name, spec)
            else:
                raise ValueError(
                    f"optuna.search_space[{name}] must be dict/list, got: {type(spec)}"
                )
        return out

    def _run_optuna_search(
        self,
        *,
        train_pl: Dict[str, Any],
        test_pl: Dict[str, Any],
        base_params: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            import optuna
        except Exception as e:
            raise ImportError("optuna.enabled=true but optuna is not available") from e

        search_space = dict(cfg.get("search_space", {}) or {})
        if len(search_space) == 0:
            raise ValueError("optuna.enabled=true requires non-empty optuna.search_space")

        n_trials = int(cfg.get("n_trials", 20))
        if n_trials <= 0:
            raise ValueError(f"optuna.n_trials must be > 0, got: {n_trials}")

        sampler_name = str(cfg.get("sampler", "tpe")).lower()
        seed = int(cfg.get("seed", self.seed))
        if sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"optuna.sampler must be tpe|random, got: {sampler_name}")

        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            delta = self._suggest_optuna_params(trial, search_space)
            trial_params = copy.deepcopy(base_params)
            trial_params.update(delta)
            model, _ = self._helper._fit_one_window(train_pl, None, trial_params)
            pred = self._helper._predict(model, test_pl)
            score = float(self._helper._score(test_pl, pred))
            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params or {})
        state = {
            "enabled": True,
            "n_trials": int(n_trials),
            "best_value": float(study.best_value),
            "best_params": dict(best_params),
        }
        return best_params, state

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
        selection_scope = str(self.fs_cfg.get("selection_scope", "dual_stage")).lower()
        train_test_anchor = str(self.fs_cfg.get("train_test_anchor", "train_start"))
        future_anchor = str(self.fs_cfg.get("future_anchor", "future_start"))
        future_reselect = bool(self.fs_cfg.get("future_reselect", selection_mode == "single_day_threshold"))
        future_single_set = selection_scope in {"future_only_single_set", "future_single_set", "future_only"}
        if future_single_set:
            future_reselect = True
        future_min_selected = int(self.fs_cfg.get("future_min_selected", 0))
        future_min_selected_mode = str(self.fs_cfg.get("future_min_selected_mode", "none"))

        pca_enabled = bool(self.pca_cfg.get("enabled", False))
        pseudo_enabled = bool(self.pseudo_cls_cfg.get("enabled", False))
        optuna_enabled = bool(self.optuna_cfg.get("enabled", False))

        logger.info(
            "rankic_refit_roll run start | n_windows=%d | candidate_pool=%d | selection_mode=%s | selection_scope=%s | "
            "train_test_anchor=%s | future_anchor=%s | future_reselect=%s | future_min_selected=%d | "
            "future_min_selected_mode=%s | pca=%s | pseudo_cls=%s | optuna=%s",
            int(len(windows)),
            int(len(base_pool)),
            selection_mode,
            selection_scope,
            train_test_anchor,
            future_anchor,
            bool(future_reselect),
            int(future_min_selected),
            str(future_min_selected_mode),
            bool(pca_enabled),
            bool(pseudo_enabled),
            bool(optuna_enabled),
        )

        current_model_params = copy.deepcopy(self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {}))
        last_optuna_date: Optional[pd.Timestamp] = None

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

            future_select_date = _resolve_window_anchor_date(w, future_anchor, default="future_start")
            train_select_date = _resolve_window_anchor_date(w, train_test_anchor, default="train_start")

            fs_train_state: Dict[str, Any]
            fs_future_state: Dict[str, Any]
            if future_single_set:
                selected_raw, fs_shared_state = self._select_factor_set(
                    select_date=future_select_date,
                    candidate_pool=base_pool,
                    min_selected=future_min_selected,
                    min_selected_mode=future_min_selected_mode,
                )
                shared_factors, fs_shared_state, skip_reason = _normalize_selection(
                    selected_raw, fs_shared_state, tag="future_shared"
                )
                if skip_reason is not None:
                    records.append(
                        {
                            **rec_base,
                            "n_selected_factors": 0,
                            "n_selected_factors_train_test": 0,
                            "n_selected_factors_future": 0,
                            "factor_selection": dict(fs_shared_state or {}),
                            "factor_selection_train_test": dict(fs_shared_state or {}),
                            "factor_selection_future": dict(fs_shared_state or {}),
                            "train_score": np.nan,
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

                fs_shared_state = dict(fs_shared_state or {})
                fs_shared_state["selection_scope"] = "future_only_single_set"
                fs_shared_state["select_date_train_test"] = str(future_select_date)[:10]
                fs_shared_state["select_date_future"] = str(future_select_date)[:10]
                train_factors = list(shared_factors)
                future_factors = list(shared_factors)
                fs_train_state = dict(fs_shared_state)
                fs_future_state = dict(fs_shared_state)
            else:
                train_selected_raw, fs_train_state = self._select_factor_set(
                    select_date=train_select_date,
                    candidate_pool=base_pool,
                )
                train_factors, fs_train_state, skip_reason = _normalize_selection(
                    train_selected_raw, fs_train_state, tag="train_test"
                )
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
                            "train_score": np.nan,
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

                if future_reselect:
                    future_selected_raw, fs_future_state = self._select_factor_set(
                        select_date=future_select_date,
                        candidate_pool=base_pool,
                        min_selected=future_min_selected,
                        min_selected_mode=future_min_selected_mode,
                    )
                else:
                    future_selected_raw = list(train_factors)
                    fs_future_state = {
                        "enabled": bool(self.fs_enabled),
                        "selection_mode": "inherit_train_test",
                        "select_date": str(future_select_date)[:10],
                        "n_selected": int(len(future_selected_raw)),
                    }
                future_factors, fs_future_state, skip_reason = _normalize_selection(
                    future_selected_raw, fs_future_state, tag="future"
                )
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
                            "train_score": np.nan,
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

            key_cols = [self.date_col, self.stockid_col, self.label_col]
            train_cols = list(dict.fromkeys(key_cols + list(train_factors)))
            future_cols = list(dict.fromkeys(key_cols + list(future_factors)))

            # Pre-check future availability with key columns only, to keep peak memory low.
            df_future_keys = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.future_start,
                end_date=w.future_end,
                columns=key_cols,
            )
            n_future_rows = int(len(df_future_keys))
            del df_future_keys

            # Stage A: read only train/test factor columns for eval model.
            df_train = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.train_start,
                end_date=w.train_end,
                columns=train_cols,
            )
            df_test = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.test_start,
                end_date=w.test_end,
                columns=train_cols,
            )

            if df_train.empty or df_test.empty or n_future_rows == 0:
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": int(len(future_factors)),
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "train_score": np.nan,
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_window_part",
                        "n_train_rows": int(len(df_train)),
                        "n_test_rows": int(len(df_test)),
                        "n_future_rows": int(n_future_rows),
                    }
                )
                logger.warning(
                    "window skipped | id=%d | reason=empty_window_part | n_train=%d n_test=%d n_future=%d",
                    fold, int(len(df_train)), int(len(df_test)), int(n_future_rows),
                )
                continue

            # train/test full-market payloads for scoring
            train_pl = self._cut_part(df_train, feature_cols=train_factors, fold=fold, part="train")
            test_pl = self._cut_part(df_test, feature_cols=train_factors, fold=fold, part="test")
            train_fit_pl = train_pl
            pseudo_train_state: Dict[str, Any] = {"enabled": False}
            if pseudo_enabled:
                df_train_fit, pseudo_train_state = _build_pseudo_train_df(
                    df_train,
                    date_col=self.date_col,
                    label_col=self.label_col,
                    cfg=self.pseudo_cls_cfg,
                )
                train_fit_pl = self._cut_part(df_train_fit, feature_cols=train_factors, fold=fold, part="train_fit")

            if (
                self._helper._is_empty_payload(train_pl)
                or self._helper._is_empty_payload(test_pl)
                or self._helper._is_empty_payload(train_fit_pl)
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
                        "train_score": np.nan,
                        "test_score": np.nan,
                        "future_score": np.nan,
                        "saved_train": None,
                        "saved_test": None,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_payload_after_cut",
                    }
                )
                logger.warning("window skipped | id=%d | reason=empty_train_test_payload_after_cut", fold)
                continue

            # optional PCA for eval stage: fit on train_fit_pl, apply to train_fit/train/test payloads
            pca_eval_state: Dict[str, Any] = {"enabled": False}
            train_fit_pl_model = train_fit_pl
            train_pl_model = train_pl
            test_pl_model = test_pl
            if pca_enabled:
                pca_eval_state = _fit_pca_state(train_fit_pl, self.pca_cfg, prefix=f"pc_eval_w{fold:04d}")
                logger.info(
                    "pca eval fitted | window_id=%d | n_input_features=%d | n_output_features=%d | explained_variance_ratio_sum=%.6f",
                    fold,
                    int(pca_eval_state.get("n_input_features", 0)),
                    int(pca_eval_state.get("n_output_features", 0)),
                    float(pca_eval_state.get("explained_variance_ratio_sum", np.nan)),
                )
                train_fit_pl_model = _transform_payload_with_pca(train_fit_pl, pca_eval_state)
                train_pl_model = _transform_payload_with_pca(train_pl, pca_eval_state)
                test_pl_model = _transform_payload_with_pca(test_pl, pca_eval_state)
                pca_eval_state = dict({"enabled": True}, **pca_eval_state)

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
                "selection_scope": selection_scope,
            }

            tune_state: Dict[str, Any] = {"enabled": False, "tuned": False}
            optuna_anchor = str(self.optuna_cfg.get("anchor", "future_start"))
            tune_anchor_date = _resolve_window_anchor_date(w, optuna_anchor, default="future_start")
            if self._should_run_optuna(
                anchor_date=tune_anchor_date,
                last_tune_date=last_optuna_date,
                cfg=self.optuna_cfg,
            ):
                best_delta, tune_state = self._run_optuna_search(
                    train_pl=train_fit_pl_model,
                    test_pl=test_pl_model,
                    base_params=current_model_params,
                    cfg=self.optuna_cfg,
                )
                if best_delta:
                    current_model_params.update(best_delta)
                    tune_state["updated"] = True
                else:
                    tune_state["updated"] = False
                tune_state["enabled"] = True
                tune_state["tuned"] = True
                tune_state["anchor_date"] = str(pd.Timestamp(tune_anchor_date).date())
                last_optuna_date = pd.Timestamp(tune_anchor_date)

            # ---- 2) eval model: train -> test evaluation ----
            eval_model, _ = self._helper._fit_one_window(train_fit_pl_model, None, current_model_params)
            train_pred = self._helper._predict(eval_model, train_pl_model)
            test_pred = self._helper._predict(eval_model, test_pl_model)
            train_score = float(self._helper._score(train_pl_model, train_pred))
            test_score = float(self._helper._score(test_pl_model, test_pred))

            saved_train = None
            if self.save_enabled and self.saver is not None:
                saved_train = self._helper._save_preds(
                    train_pl_model,
                    train_pred,
                    meta,
                    part="train",
                    params=current_model_params,
                )

            saved_test = None
            if self.save_enabled and self.saver is not None:
                saved_test = self._helper._save_preds(
                    test_pl_model, test_pred, meta, part="test", params=current_model_params
                )

            del train_pred, test_pred
            gc.collect()

            # ---- 3) refit final model on train+test (future factor set) ----
            # If future factor set differs, re-read train/test with future columns only.
            if list(train_factors) == list(future_factors) and not pseudo_enabled:
                refit_pl = _combine_payloads(train_pl, test_pl)
            else:
                # release eval-stage heavy objects before refit-stage read
                if list(train_factors) == list(future_factors):
                    df_train_refit = df_train
                    df_test_refit = df_test
                else:
                    df_train_refit = _read_part_from_preprocess(
                        self.preprocess_path,
                        date_col=self.date_col,
                        start_date=w.train_start,
                        end_date=w.train_end,
                        columns=future_cols,
                    )
                    df_test_refit = _read_part_from_preprocess(
                        self.preprocess_path,
                        date_col=self.date_col,
                        start_date=w.test_start,
                        end_date=w.test_end,
                        columns=future_cols,
                    )
                if df_train_refit.empty or df_test_refit.empty:
                    records.append(
                        {
                            **rec_base,
                            "n_selected_factors": int(len(train_factors)),
                            "n_selected_factors_train_test": int(len(train_factors)),
                            "n_selected_factors_future": int(len(future_factors)),
                            "factor_selection": dict(fs_train_state or {}),
                            "factor_selection_train_test": dict(fs_train_state or {}),
                            "factor_selection_future": dict(fs_future_state or {}),
                            "train_score": float(train_score),
                            "test_score": float(test_score),
                            "future_score": np.nan,
                            "saved_train": saved_train,
                            "saved_test": saved_test,
                            "saved_future": None,
                            "skipped": True,
                            "skip_reason": "empty_refit_train_test_part",
                            "n_train_rows": int(len(df_train_refit)),
                            "n_test_rows": int(len(df_test_refit)),
                            "n_future_rows": int(n_future_rows),
                        }
                    )
                    logger.warning("window skipped | id=%d | reason=empty_refit_train_test_part", fold)
                    continue

                if pseudo_enabled:
                    df_refit_all = pd.concat([df_train_refit, df_test_refit], axis=0, ignore_index=True)
                    df_refit_fit, pseudo_refit_state = _build_pseudo_train_df(
                        df_refit_all,
                        date_col=self.date_col,
                        label_col=self.label_col,
                        cfg=self.pseudo_cls_cfg,
                    )
                    refit_pl = self._cut_part(df_refit_fit, feature_cols=future_factors, fold=fold, part="refit_fit")
                else:
                    refit_train_pl = self._cut_part(
                        df_train_refit, feature_cols=future_factors, fold=fold, part="train_refit"
                    )
                    refit_test_pl = self._cut_part(
                        df_test_refit, feature_cols=future_factors, fold=fold, part="test_refit"
                    )
                    refit_pl = _combine_payloads(refit_train_pl, refit_test_pl)

                if self._helper._is_empty_payload(refit_pl):
                    records.append(
                        {
                            **rec_base,
                            "n_selected_factors": int(len(train_factors)),
                            "n_selected_factors_train_test": int(len(train_factors)),
                            "n_selected_factors_future": int(len(future_factors)),
                            "factor_selection": dict(fs_train_state or {}),
                            "factor_selection_train_test": dict(fs_train_state or {}),
                            "factor_selection_future": dict(fs_future_state or {}),
                            "train_score": float(train_score),
                            "test_score": float(test_score),
                            "future_score": np.nan,
                            "saved_train": saved_train,
                            "saved_test": saved_test,
                            "saved_future": None,
                            "skipped": True,
                            "skip_reason": "empty_refit_payload_after_cut",
                        }
                    )
                    logger.warning("window skipped | id=%d | reason=empty_refit_payload_after_cut", fold)
                    continue

            # Optional PCA for final refit/future stage
            pca_final_state: Dict[str, Any] = {"enabled": False}
            refit_pl_model = refit_pl

            # ---- 4) future prediction (separate from test eval) ----
            df_future = _read_part_from_preprocess(
                self.preprocess_path,
                date_col=self.date_col,
                start_date=w.future_start,
                end_date=w.future_end,
                columns=future_cols,
            )
            if df_future.empty:
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": int(len(future_factors)),
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "train_score": float(train_score),
                        "test_score": float(test_score),
                        "future_score": np.nan,
                        "saved_train": saved_train,
                        "saved_test": saved_test,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_future_part_after_refit",
                        "n_future_rows": 0,
                    }
                )
                logger.warning("window skipped | id=%d | reason=empty_future_part_after_refit", fold)
                continue

            future_pl = self._cut_part(df_future, feature_cols=future_factors, fold=fold, part="future")
            del df_future
            gc.collect()
            if self._helper._is_empty_payload(future_pl):
                records.append(
                    {
                        **rec_base,
                        "n_selected_factors": int(len(train_factors)),
                        "n_selected_factors_train_test": int(len(train_factors)),
                        "n_selected_factors_future": int(len(future_factors)),
                        "factor_selection": dict(fs_train_state or {}),
                        "factor_selection_train_test": dict(fs_train_state or {}),
                        "factor_selection_future": dict(fs_future_state or {}),
                        "train_score": float(train_score),
                        "test_score": float(test_score),
                        "future_score": np.nan,
                        "saved_train": saved_train,
                        "saved_test": saved_test,
                        "saved_future": None,
                        "skipped": True,
                        "skip_reason": "empty_future_payload_after_cut",
                    }
                )
                logger.warning("window skipped | id=%d | reason=empty_future_payload_after_cut", fold)
                continue

            if pca_enabled:
                pca_final_state = _fit_pca_state(refit_pl, self.pca_cfg, prefix=f"pc_final_w{fold:04d}")
                logger.info(
                    "pca final fitted | window_id=%d | n_input_features=%d | n_output_features=%d | explained_variance_ratio_sum=%.6f",
                    fold,
                    int(pca_final_state.get("n_input_features", 0)),
                    int(pca_final_state.get("n_output_features", 0)),
                    float(pca_final_state.get("explained_variance_ratio_sum", np.nan)),
                )
                refit_pl_model = _transform_payload_with_pca(refit_pl, pca_final_state)
                future_pl_model = _transform_payload_with_pca(future_pl, pca_final_state)
                pca_final_state = dict({"enabled": True}, **pca_final_state)
            else:
                future_pl_model = future_pl

            final_model, _ = self._helper._fit_one_window(refit_pl_model, None, current_model_params)
            future_pred = self._helper._predict(final_model, future_pl_model)
            future_score = float(self._helper._score(future_pl, future_pred))

            saved_future = None
            if self.save_enabled and self.saver is not None:
                saved_future = self._helper._save_preds(
                    future_pl_model,
                    future_pred,
                    meta,
                    part="future",
                    params=current_model_params,
                )

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
                    "train_score": float(train_score),
                    "test_score": float(test_score),
                    "future_score": float(future_score),
                    "saved_train": saved_train,
                    "saved_test": saved_test,
                    "saved_future": saved_future,
                    "selection_scope": selection_scope,
                    "model_params": dict(current_model_params),
                    "tune_state": dict(tune_state),
                    "pseudo_state_train": dict(pseudo_train_state),
                    "pca_state_eval": _pca_state_for_record(pca_eval_state),
                    "pca_state_final": _pca_state_for_record(pca_final_state),
                    "skipped": False,
                }
            )

            logger.info(
                "window done | id=%d | n_factors_train_test=%d | n_factors_future=%d | train_score=%.6f | test_score=%.6f | future_score=%.6f",
                fold,
                int(len(train_factors)),
                int(len(future_factors)),
                float(train_score),
                float(test_score),
                float(future_score),
            )

            # explicit per-window cleanup
            try:
                del eval_model, final_model, train_pl, test_pl, train_fit_pl, refit_pl, future_pl
            except Exception:
                pass
            gc.collect()

        hist_df = pd.DataFrame(records)
        logger.info("rankic_refit_roll run end | n_windows=%d", int(len(hist_df)))
        return hist_df
