"""
ret_pred/main.py

项目主入口（thin main）：负责
- 读取配置（YAML）
- 解析命令行覆盖参数（run_id / runs_root）
- resolve_paths：把 runs_root/run_id 等路径占位符渲染为绝对路径
- 初始化日志（只在 main 里初始化一次）
- 设置随机种子
- 根据 task.mode 调度训练或预测流程

用法:
    # 训练
    python -m ret_pred.main --config ret_pred/config_train.yaml
    python -m ret_pred.main --config ret_pred/config_train_sweep.yaml

    # 预测/推理
    python -m ret_pred.main --config ret_pred/config_pred.yaml

运行模式:
- task.mode=train:
    load_cfg
    resolve_paths
    setup_logging
    seed_all
    run_train:
        A) load_long          : dataloader 读取全区间 long 数据
        B) preprocess_long    : 清洗/标准化/可选落 parquet，并返回 preprocess_state（FIT）
        C) datasplit_long     : 只基于 dates 生成 folds（train/valid/test 日期列表）
        D) build_streaming_windows:
             从 preprocess parquet “流式”读取每个 fold 的窗口数据，切分 train/valid/test，并 yield 给 trainer
        E) trainer.run        : 滚动训练 / sweep 训练，保存模型/预测/历史
        F) post_train_predict : （可选）训练结束后立即做一次预测落盘
        G) Evaluator.run      : （可选）对 pred_dir 做评估并输出图表/指标

- task.mode=predict (或 infer):
    load_cfg
    resolve_paths
    setup_logging
    seed_all
    run_predict_only:
        调用 predictor.run_predict 加载 bundle 推理，并保存预测 parquet

设计原则:
- main 尽量薄：不在顶层导入大量模块，避免启动慢/循环依赖；训练细节下放到 run_train。
- 统一路径管理：先 resolve_paths 再 setup_logging，保证日志文件路径正确。
- 可复现：seed_all 固定随机性（Python random + numpy）。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
import copy
import logging

from ret_pred.paths import resolve_paths

logger = logging.getLogger(__name__)


# ======================================================
# Helpers
# ======================================================
def load_cfg(config_path: Path) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    参数:
        config_path (Path): 配置文件路径。

    返回:
        Dict[str, Any]: 解析后的配置 dict。

    异常:
        FileNotFoundError: 配置文件不存在。
        yaml.YAMLError: YAML 格式错误。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    return yaml.safe_load(config_path.read_text())


def seed_all(seed: int) -> None:
    """
    设置随机种子，保证实验可复现（尽量）。

    目前覆盖：
    - Python random
    - numpy random

    说明：
    - 如果引入 torch / lightgbm 的随机性，建议在各自模块里也补充 seed 设置。
    - 这里先保持 main 的职责：只做全局最基础的 seed。

    参数:
        seed (int): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)


def _resolve_long_paths_from_dataloader(dl_cfg: Dict[str, Any]) -> List[Path]:
    """
    Resolve long parquet source paths from dataloader config.
    """
    long_paths = dl_cfg.get("long_paths")
    if isinstance(long_paths, list) and len(long_paths) > 0:
        return [Path(str(p)) for p in long_paths]

    long_path = dl_cfg.get("long_path")
    if long_path:
        return [Path(str(long_path))]

    long_filename = str(dl_cfg.get("long_filename", "long.parquet"))
    if Path(long_filename).is_absolute():
        return [Path(long_filename)]
    return [Path(str(dl_cfg.get("parquet_dir", "./data"))) / long_filename]


def _infer_base_pool_from_dataloader_schema(dl_cfg: Dict[str, Any], *, label_col: str) -> List[str]:
    """
    Infer numeric candidate factor pool from parquet schema.

    Used when feature_mask_csv is null and dataloader.fields is empty.
    """
    import pandas as pd

    date_col = str(dl_cfg.get("date_col", "date"))
    stockid_col = str(dl_cfg.get("stockid_col", "stockid"))
    exclude = {date_col, stockid_col, str(label_col), f"{label_col}_raw"}

    out: List[str] = []
    seen = set()

    for p in _resolve_long_paths_from_dataloader(dl_cfg):
        if not p.exists():
            continue

        cols: List[str] = []
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pq.ParquetFile(p).schema_arrow
            for f in schema:
                t = f.type
                if (
                    pa.types.is_integer(t)
                    or pa.types.is_floating(t)
                    or pa.types.is_decimal(t)
                    or pa.types.is_boolean(t)
                ):
                    cols.append(str(f.name))
        except Exception:
            # Fallback: keep non-key columns if schema typing is unavailable.
            logger.warning("schema numeric inference fallback via pandas columns | path=%s", str(p))
            cols = [str(c) for c in pd.read_parquet(p).columns]

        for c in cols:
            if c in exclude or c in seen:
                continue
            seen.add(c)
            out.append(c)

    return out


def _collect_planning_dates_from_dataloader(dl_cfg: Dict[str, Any]) -> List[Any]:
    """
    Lightweight date scan for rankic_refit_roll pre-planning.

    Reads only date column from one long parquet source within [date_start, date_end].
    """
    import pandas as pd

    date_col = str(dl_cfg.get("date_col", "date"))
    date_start = str(dl_cfg.get("date_start", ""))
    date_end = str(dl_cfg.get("date_end", ""))

    paths = _resolve_long_paths_from_dataloader(dl_cfg)
    if len(paths) == 0:
        return []
    p = paths[0]

    if not p.exists():
        return []

    start_ts = pd.to_datetime(date_start) if date_start else None
    end_ts = pd.to_datetime(date_end) if date_end else None

    try:
        if start_ts is not None and end_ts is not None:
            df = pd.read_parquet(
                p,
                columns=[date_col],
                filters=[(date_col, ">=", start_ts), (date_col, "<=", end_ts)],
            )
        else:
            df = pd.read_parquet(p, columns=[date_col])
    except TypeError:
        df = pd.read_parquet(p, columns=[date_col])

    if date_col not in df.columns:
        return []

    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if start_ts is not None:
        s = s[s >= start_ts]
    if end_ts is not None:
        s = s[s <= end_ts]
    return sorted(pd.to_datetime(s.unique()))


def _resolve_window_anchor_date(w: Any, anchor: str, *, default: str) -> Any:
    """
    Resolve anchor date from WindowDef-like object.
    """
    name = str(anchor or default)
    if not hasattr(w, name):
        name = default
    return getattr(w, name)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out


def _extract_sweep_trainer_override(raw_set: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_set, dict):
        return {}
    if isinstance(raw_set.get("trainer_params"), dict):
        return dict(raw_set["trainer_params"])
    if isinstance(raw_set.get("trainer"), dict):
        tr = raw_set["trainer"]
        if isinstance(tr.get("params"), dict):
            return dict(tr["params"])
        return dict(tr)
    out: Dict[str, Any] = {}
    for k in ("windowing", "factor_selection", "loss", "sample_weighting", "saver"):
        if isinstance(raw_set.get(k), dict):
            out[k] = copy.deepcopy(raw_set[k])
    return out


# ======================================================
# Train / Predict pipeline
# ======================================================
def run_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练模式总流程调度。

    该函数会根据 trainer.name 分派：
    - rolling：单次滚动训练（RollingTrainer）
    - sweep_rolling：参数组 sweep（SweepRollingTrainer），每组参数都走完整 rolling

    参数:
        cfg (Dict[str, Any]): 完整配置（已 resolve_paths，已初始化 logging）。

    返回:
        Dict[str, Any]: 关键产物路径汇总（history、model_dir、pred_dir、eval_dir 等）。

    异常:
        任何阶段失败会抛出异常给上层 main 统一捕获并打印 traceback。
    """
    logger = logging.getLogger(__name__)

    # 延迟 import：避免 main 启动时就加载一堆依赖
    from ret_pred.dataloader import (
        DLRequest,
        DLCfg,
        load_long,
        read_feature_mask_csv,
        filter_fields_by_mask,
    )
    from ret_pred.feature_engineering import run_feature_engineering
    from ret_pred.preprocess import preprocess_long, preprocess_fit_transform_from_parquet
    from ret_pred.split import datasplit_long
    from ret_pred.windows import build_streaming_windows
    from ret_pred.evaluate.evaluator import Evaluator
    from ret_pred.predictor.rolling_predictor import run_predict

    task_cfg = cfg.get("task", {})
    paths = cfg.get("paths", {})
    dl = cfg["dataloader"]

    # -------------------------
    # A) dataloader
    # -------------------------
    # label_col 优先从 task_cfg 指定；否则从 dataloader.label_name 取；最后 fallback 到 "y"
    label_col = task_cfg.get("label_col", dl.get("label_name", "y"))

    # DLRequest：描述本次要取的数据区间与字段
    dl_req = DLRequest(
        date_start=dl["date_start"],
        date_end=dl["date_end"],
        fields=dl.get("fields", []) or [],
        label_name=dl.get("label_name", label_col),
    )

    # DLCfg：描述数据源与字段约定
    dl_lowmem = dict(dl.get("lowmem", {}) or {})
    dl_cfg = DLCfg(
        source=dl.get("source", "parquet_dir"),
        parquet_dir=dl.get("parquet_dir", "./data"),
        long_filename=dl.get("long_filename"),
        long_path=dl.get("long_path"),
        long_paths=dl.get("long_paths"),
        long_merge_how=dl.get("long_merge_how", "outer"),
        date_col=dl.get("date_col", "date"),
        stockid_col=dl.get("stockid_col", "stockid"),
        check_unique_key=dl.get("check_unique_key", True),
        feature_mask_csv=dl.get("feature_mask_csv"),
        feature_col_name=dl.get("feature_col_name", "feature"),
        feature_use_col=dl.get("feature_use_col", "use"),
        # label 可选：由 dataloader 构造（比如 open_to_open / close_to_close）
        build_label=bool(dl.get("build_label", False)),
        label_price_col=dl.get("label_price_col", "open_1d"),
        label_method=dl.get("label_method", "open_to_open"),
        label_log_return=bool(dl.get("label_log_return", False)),
        # label 后处理（optional; default disabled）
        label_postprocess=dl.get("label_postprocess"),
        # low-memory multi-long stream merge (optional)
        lowmem_enabled=bool(dl_lowmem.get("enabled", dl.get("lowmem_enabled", False))),
        lowmem_chunk_days=int(dl_lowmem.get("chunk_days", dl.get("lowmem_chunk_days", 20))),
        lowmem_cast_float32=bool(dl_lowmem.get("cast_float32", dl.get("lowmem_cast_float32", True))),
        lowmem_temp_dir=dl_lowmem.get("temp_dir", dl.get("lowmem_temp_dir")),
        lowmem_output_path=dl_lowmem.get("output_path", dl.get("lowmem_output_path")),
    )

    tr_cfg = cfg.get("trainer", {})
    tr_name = str(tr_cfg.get("name", "rolling"))
    tr_params = dict(tr_cfg.get("params", {}))

    # Stage-2 pre-plan:
    # for rankic_refit_roll + factor_selection, precompute a union factor pool
    # and push it into dataloader fields to reduce load/preprocess width.
    if tr_name in ("rankic_refit_roll", "sweep_rankic_refit_roll"):
        fs_cfg = dict(tr_params.get("factor_selection", {}) or {})
        preplan_union = bool(fs_cfg.get("enabled", False)) and bool(fs_cfg.get("preplan_union_to_dataloader", True))
        if not preplan_union and tr_name == "sweep_rankic_refit_roll":
            sw_cfg = dict(cfg.get("sweep", {}) or {})
            for raw_set in (sw_cfg.get("param_sets") or []):
                tr_over = _extract_sweep_trainer_override(raw_set if isinstance(raw_set, dict) else {})
                tr_i = _deep_merge_dict(tr_params, tr_over)
                fs_i = dict(tr_i.get("factor_selection", {}) or {})
                if bool(fs_i.get("enabled", False)) and bool(fs_i.get("preplan_union_to_dataloader", True)):
                    preplan_union = True
                    break

        if preplan_union:
            base_pool: List[str] = []

            selected_mask = None
            if dl_cfg.feature_mask_csv:
                selected_mask = read_feature_mask_csv(
                    dl_cfg.feature_mask_csv,
                    feature_col_name=dl_cfg.feature_col_name,
                    use_col=dl_cfg.feature_use_col,
                )

            if dl_req.fields:
                base_pool = filter_fields_by_mask(dl_req.fields, selected_mask)
            elif selected_mask is not None:
                base_pool = list(selected_mask)
            else:
                # feature_mask_csv is null: infer candidate pool from parquet schema
                base_pool = _infer_base_pool_from_dataloader_schema(dl, label_col=label_col)

            base_pool = list(dict.fromkeys([str(x) for x in base_pool]))

            if len(base_pool) == 0:
                logger.warning(
                    "rankic preplan skipped: empty base candidate pool. "
                    "Please check long_paths schema or set dataloader.fields explicitly."
                )
            else:
                planning_dates = _collect_planning_dates_from_dataloader(dl)
                if len(planning_dates) == 0:
                    logger.warning("rankic preplan skipped: no planning dates collected from dataloader source.")
                else:
                    from ret_pred.factor_selection import (
                        load_factor_use_flags_csv,
                        load_predictions_pkl,
                        select_union_factors_by_rankic,
                        select_union_factors_by_use_flags,
                    )
                    from ret_pred.windowing_rankic_refit_roll import build_rankic_refit_roll_windows

                    # build preplan specs:
                    # - single rankic run: one spec from trainer.params
                    # - sweep rankic run: one spec per param_set override
                    specs: List[Dict[str, Any]] = []
                    if tr_name == "rankic_refit_roll":
                        specs.append(
                            {
                                "spec_id": "base",
                                "windowing": dict(tr_params.get("windowing", {}) or {}),
                                "factor_selection": dict(fs_cfg),
                            }
                        )
                    else:
                        sw_cfg = dict(cfg.get("sweep", {}) or {})
                        param_sets = sw_cfg.get("param_sets") or []
                        if not isinstance(param_sets, list) or len(param_sets) == 0:
                            raise ValueError("trainer.name=sweep_rankic_refit_roll but cfg.sweep.param_sets is empty")
                        for i, raw_set in enumerate(param_sets, start=1):
                            tr_over = _extract_sweep_trainer_override(raw_set if isinstance(raw_set, dict) else {})
                            tr_i = _deep_merge_dict(tr_params, tr_over)
                            fs_i = dict(tr_i.get("factor_selection", {}) or {})
                            specs.append(
                                {
                                    "spec_id": str((raw_set or {}).get("id") or f"set_{i:03d}") if isinstance(raw_set, dict) else f"set_{i:03d}",
                                    "windowing": dict(tr_i.get("windowing", {}) or {}),
                                    "factor_selection": fs_i,
                                }
                            )

                    union_pool: List[str] = []
                    union_seen = set()
                    spec_states: List[Dict[str, Any]] = []
                    any_no_fs = False
                    total_asof_dates = 0
                    pred_map_cache: Dict[str, Any] = {}
                    flags_map_cache: Dict[str, Any] = {}

                    for sp in specs:
                        fs_i = dict(sp.get("factor_selection", {}) or {})
                        if not bool(fs_i.get("enabled", False)):
                            any_no_fs = True
                            spec_states.append({"spec_id": sp.get("spec_id"), "factor_selection_enabled": False})
                            continue

                        windows = build_rankic_refit_roll_windows(planning_dates, dict(sp.get("windowing", {}) or {}))
                        selection_mode = str(fs_i.get("selection_mode", "historical_aggregate")).lower()
                        selection_scope = str(fs_i.get("selection_scope", "dual_stage")).lower()
                        train_test_anchor = str(fs_i.get("train_test_anchor", "train_start"))
                        future_anchor = str(fs_i.get("future_anchor", "future_start"))
                        future_reselect = bool(fs_i.get("future_reselect", selection_mode == "single_day_threshold"))
                        future_single_set = selection_scope in {"future_only_single_set", "future_single_set", "future_only"}

                        if selection_mode == "single_day_threshold":
                            train_dates = [_resolve_window_anchor_date(w, train_test_anchor, default="train_start") for w in windows]
                            future_dates = [_resolve_window_anchor_date(w, future_anchor, default="future_start") for w in windows]
                            if future_single_set:
                                asof_dates = list(future_dates)
                            else:
                                asof_dates = list(train_dates) + (list(future_dates) if future_reselect else [])
                        else:
                            if future_single_set:
                                asof_dates = [_resolve_window_anchor_date(w, future_anchor, default="future_start") for w in windows]
                            else:
                                asof_dates = [w.factor_selection_asof_date for w in windows]

                        total_asof_dates += int(len(asof_dates))
                        preplan_min_selected = 0
                        preplan_min_selected_mode = "none"
                        if selection_mode == "single_day_threshold" and future_single_set:
                            preplan_min_selected = int(fs_i.get("future_min_selected", 0))
                            preplan_min_selected_mode = str(fs_i.get("future_min_selected_mode", "none"))

                        fs_source = str(fs_i.get("source", "predictions_pkl")).strip().lower()
                        if fs_source in {"predictions_pkl", "pkl", "rankic_pkl"}:
                            pkl_path = str(fs_i.get("predictions_pkl_path", "")).strip()
                            if not pkl_path:
                                raise ValueError(
                                    "factor_selection.source=predictions_pkl with preplan enabled but "
                                    "factor_selection.predictions_pkl_path is empty"
                                )
                            if pkl_path not in pred_map_cache:
                                pred_map_cache[pkl_path] = load_predictions_pkl(pkl_path)
                            pred_map = pred_map_cache[pkl_path]
                            pool_i, state_i = select_union_factors_by_rankic(
                                pred_map,
                                asof_dates=asof_dates,
                                threshold=float(fs_i.get("rankic_threshold", fs_i.get("threshold", 0.03))),
                                aggregate=str(fs_i.get("aggregate", "abs_mean")),
                                inclusive=bool(fs_i.get("cutoff_inclusive", True)),
                                min_history_days=int(fs_i.get("min_history_days", 1)),
                                candidate_pool=base_pool,
                                selection_mode=selection_mode,
                                min_selected=int(preplan_min_selected),
                                min_selected_mode=str(preplan_min_selected_mode),
                            )
                        elif fs_source in {"factor_use_flags_csv", "use_flags_csv"}:
                            flags_path = str(fs_i.get("factor_use_flags_csv_path", "")).strip()
                            if not flags_path:
                                raise ValueError(
                                    "factor_selection.source=factor_use_flags_csv with preplan enabled but "
                                    "factor_selection.factor_use_flags_csv_path is empty"
                                )
                            if flags_path not in flags_map_cache:
                                flags_map_cache[flags_path] = load_factor_use_flags_csv(flags_path, date_col="date")
                            flags_map = flags_map_cache[flags_path]
                            pool_i, state_i = select_union_factors_by_use_flags(
                                flags_map,
                                asof_dates=asof_dates,
                                candidate_pool=base_pool,
                            )
                        else:
                            raise ValueError(
                                f"factor_selection.source must be predictions_pkl|factor_use_flags_csv, got: {fs_source}"
                            )

                        for f in pool_i:
                            if f not in union_seen:
                                union_seen.add(f)
                                union_pool.append(f)

                        spec_states.append(
                            {
                                "spec_id": sp.get("spec_id"),
                                "factor_selection_enabled": True,
                                "selection_mode": selection_mode,
                                "selection_scope": selection_scope,
                                "train_test_anchor": train_test_anchor,
                                "future_anchor": future_anchor,
                                "future_reselect": bool(future_reselect),
                                "future_single_set": bool(future_single_set),
                                "future_min_selected": int(preplan_min_selected),
                                "future_min_selected_mode": str(preplan_min_selected_mode),
                                "source": fs_source,
                                "n_asof_dates": int(len(asof_dates)),
                                "n_union": int(len(pool_i)),
                                "state": state_i,
                            }
                        )

                    plan_state = {
                        "trainer_name": tr_name,
                        "n_specs": int(len(specs)),
                        "n_asof_dates": int(total_asof_dates),
                        "spec_states": spec_states,
                    }
                    if any_no_fs:
                        plan_state["contains_disabled_factor_selection"] = True
                        logger.warning("rankic preplan: some sweep sets disable factor_selection; fallback includes base_pool")
                        for f in base_pool:
                            if f not in union_seen:
                                union_seen.add(f)
                                union_pool.append(f)

                    if len(union_pool) == 0:
                        logger.warning(
                            "rankic preplan produced empty union pool; fallback to base_pool | n_base=%d",
                            int(len(base_pool)),
                        )
                        union_pool = list(base_pool)
                        plan_state["fallback_to_base_pool"] = True
                    else:
                        plan_state["fallback_to_base_pool"] = False

                    dl_req.fields = list(union_pool)

                    out_path = Path(paths.get("run_dir", ".")) / "selected_factors" / "union_factors_preplan.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "mode": "rankic_refit_roll_preplan_union",
                        "base_pool_size": int(len(base_pool)),
                        "union_pool_size": int(len(union_pool)),
                        "union_pool": list(union_pool),
                        "state": plan_state,
                    }
                    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

                    logger.info(
                        "rankic preplan applied | base_pool=%d | union_pool=%d | n_asof_dates=%d | artifact=%s",
                        int(len(base_pool)),
                        int(len(union_pool)),
                        int(plan_state.get("n_asof_dates", 0)),
                        str(out_path),
                    )

    # 读取全区间 long 数据
    long_df, meta = load_long(dl_req, dl_cfg)
    logger.info("dataloader done | shape=%s", long_df.shape)

    # -------------------------
    # B) preprocess (FIT)
    # -------------------------
    # preprocess_long 通常会：清洗缺失/inf，winsorize，zscore，填充策略等
    # 并返回：
    # - clean_df：清洗后的 long df（通常仍在内存里）
    # - pp_state：训练期(FIT)统计量与配置回显，可用于 transform
    pp_cfg = dict(cfg.get("preprocess", {}))
    pp_cfg["label_col"] = label_col
    pp_cfg.setdefault("save_parquet", True)  # 训练默认落盘，以便后续 streaming windows
    fe_cfg = dict(cfg.get("feature_engineering", {}) or {})
    lowmem_source_path = str((meta or {}).get("lowmem_output_path", "")).strip()

    if lowmem_source_path:
        if fe_cfg.get("enabled", False):
            raise ValueError(
                "feature_engineering.enabled=true is not supported with dataloader.lowmem.enabled=true yet"
            )
        clean_df, pp_state = preprocess_fit_transform_from_parquet(
            lowmem_source_path,
            pp_cfg,
            meta=meta,
        )
        long_df = None
    else:
        # -------------------------
        # A2) feature engineering（optional; before preprocess）
        # -------------------------
        if fe_cfg.get("enabled", False):
            long_df, fe_state = run_feature_engineering(
                long_df,
                fe_cfg,
                date_col=dl_cfg.date_col,
                stockid_col=dl_cfg.stockid_col,
                feature_cols=None,  # let feature_engineering infer numeric features
                label_col=label_col,
            )
            meta = dict(meta, feature_engineering=fe_state)
            n_new = int(((fe_state.get("rolling_stats") or {}).get("n_created") or 0)) if isinstance(fe_state, dict) else 0
            logger.info("feature_engineering applied | shape=%s | n_new=%d", long_df.shape, n_new)

        clean_df, pp_state = preprocess_long(long_df, pp_cfg, meta=meta)
        # long_df is no longer needed after preprocess
        long_df = None

    preprocess_path = pp_state.get("saved_path") or pp_state.get("save_path")
    if not preprocess_path:
        raise ValueError("preprocess did not return saved_path; please ensure preprocess.save_parquet=True")

    import pandas as pd
    if isinstance(pp_state.get("all_dates"), list) and len(pp_state.get("all_dates")) > 0:
        all_dates = sorted(pd.to_datetime(pd.Series(pp_state.get("all_dates")), errors="coerce").dropna().unique())
    else:
        all_dates = sorted(pd.to_datetime(clean_df[dl_cfg.date_col], errors="coerce").dropna().unique())
    dates_df = pd.DataFrame({dl_cfg.date_col: all_dates})

    logger.info("preprocess done | shape=%s | n_unique_dates=%d", clean_df.shape, int(len(all_dates)))
    logger.info("preprocess parquet | path=%s", preprocess_path)

    # release large in-memory dataframe as early as possible; downstream trainers stream from parquet
    clean_df = None
    gc.collect()
    logger.info("memory released after preprocess | kept=dates_df/preprocess_path")

    # -------------------------
    # C) datasplit（dates only; only for rolling/sweep_rolling）
    # -------------------------
    folds = None
    if tr_name in ("rolling", "sweep_rolling"):
        # 注意：split 只需要 date 序列，不需要全特征列
        # 这样做可以减少在 split 阶段传递/复制大 df 的成本，并且让 split 逻辑更纯粹。
        sp_cfg = dict(cfg.get("datasplit", {}))
        sp_cfg["return_dfs"] = False
        sp_cfg["save_parquet"] = False

        meta2 = dict(meta, preprocess_state=pp_state, run_id=paths.get("run_id"))

        folds, _ = datasplit_long(dates_df, sp_cfg, meta=meta2)
        logger.info("split done | folds=%d", len(folds))
    else:
        logger.info("datasplit skipped | trainer=%s uses internal windowing", tr_name)

    # -------------------------
    # D) windows (streaming read from preprocess parquet)
    # -------------------------
    model_cfg = dict(cfg.get("model", {}))
    cut_cfg = dict(cfg.get("datacutting", {}))

    # candidates 机制：如果 config 里把不同模型/不同参数写在 candidates 下，
    # 这里根据 model.name 自动挑一组作为当前参数（同时写入 params/model_config）
    if isinstance(model_cfg.get("candidates"), dict):
        name = model_cfg.get("name")
        if name in model_cfg["candidates"]:
            base_params = dict(model_cfg["candidates"][name] or {})
            model_cfg["model_config"] = copy.deepcopy(base_params)
            model_cfg["params"] = copy.deepcopy(base_params)
            logger.info("model candidates resolved | name=%s | n_params=%d", name, len(base_params))

    # -------------------------
    # D2) choose trainer
    # -------------------------

    # =====================================================
    # Trainer: rolling
    # =====================================================
    if tr_name == "rolling":
        from ret_pred.trainer.rolling_trainer import RollingTrainer
        if folds is None:
            raise RuntimeError("internal error: folds is None for trainer=rolling")

        # windows 是 generator：每次 yield 一个 fold 的 train/valid/test payload
        # streaming 的关键：只在需要时读 preprocess parquet 的窗口范围
        windows = build_streaming_windows(
            preprocess_path=str(preprocess_path),
            folds=folds,
            model_family=model_cfg.get("family", "tree"),
            label_col=label_col,
            datacutting_cfg=cut_cfg,
            date_col=dl_cfg.date_col,
            runs_root=str(paths.get("runs_root")),
        )
        logger.info("windows generator built | preprocess_path=%s", preprocess_path)

        trainer = RollingTrainer(
            model=model_cfg,
            metric=tr_params.get("metric", "rankic"),
            maximize=tr_params.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params.get("loss"),
            nn_fit=tr_params.get("nn_fit"),
            schedule=tr_params.get("schedule"),
            tuner=tr_params.get("tuner"),
            update_gate=tr_params.get("update_gate"),
            saver=tr_params.get("saver"),
            model_save=tr_params.get("model_save"),
            sample_weighting=tr_params.get("sample_weighting"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            device=tr_params.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
        )

        # 训练主循环：逐 fold 训练、预测、记录 history
        hist_df = trainer.run(windows)

        # 保存训练历史（用于复盘/画图/调参比较）
        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        hist_df.to_parquet(hist_path, index=False)
        logger.info("train done | history_shape=%s", hist_df.shape)
        logger.info("train_history saved | path=%s", str(hist_path))

        # -------------------------
        # F) post-train predict（可选）
        # -------------------------
        ptp_out = None
        ptp = cfg.get("post_train_predict", {})
        if ptp.get("enabled", False):
            # 复用整份 cfg，但覆盖 predict 段落
            ptp_cfg = dict(cfg)
            ptp_cfg["predict"] = dict(ptp)

            target_date = str(ptp.get("target_date", ""))
            asof_date = str(ptp.get("asof_date", ""))

            # 支持 out_path 模板替换（避免你每次手动写日期）
            if "out_path" in ptp_cfg["predict"]:
                ptp_cfg["predict"]["out_path"] = (
                    ptp_cfg["predict"]["out_path"]
                    .replace("{target_date}", target_date)
                    .replace("{asof_date}", asof_date)
                )

            ptp_cfg["paths"] = cfg["paths"]
            ptp_out = run_predict(ptp_cfg)
            logger.info("post-train predict saved | out=%s", ptp_out)

        # -------------------------
        # G) evaluate（可选/默认）
        # -------------------------
        # evaluator 会从 pred_dir 读取预测 parquet，计算指标并输出图表
        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})
        ev_params.setdefault("pred_dir", cfg["paths"]["pred_dir"])
        ev_params.setdefault("out_dir", cfg["paths"]["eval_dir"])

        evaluator = Evaluator(**ev_params)
        res = evaluator.run()

        logger.info("evaluate done")
        logger.info("metrics:\n%s", res.metrics_df)

        return {
            "train_history": str(hist_path),
            "post_train_pred": ptp_out,
            "eval_dir": cfg["paths"]["eval_dir"],
            "model_dir": cfg["paths"]["model_dir"],
            "pred_dir": cfg["paths"]["pred_dir"],
        }

    # =====================================================
    # Trainer: sweep_rolling
    # =====================================================
    elif tr_name == "sweep_rolling":
        from ret_pred.trainer.sweep_trainer import SweepRollingTrainer
        if folds is None:
            raise RuntimeError("internal error: folds is None for trainer=sweep_rolling")

        sweep_cfg = dict(cfg.get("sweep", {}) or {})
        param_sets = sweep_cfg.get("param_sets") or []
        if not isinstance(param_sets, list) or len(param_sets) == 0:
            raise ValueError("trainer.name=sweep_rolling but cfg.sweep.param_sets is empty")

        # evaluate cfg 传给 sweep trainer，让它对每组参数分别评估
        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})

        # SweepRollingTrainer 内部会对每组 param_set 重新 build windows（必须！）
        # 因为 windows 是 generator，一旦被消费完就没了。
        trainer = SweepRollingTrainer(
            model=model_cfg,
            sweep=sweep_cfg,
            metric=tr_params.get("metric", "rankic"),
            maximize=tr_params.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params.get("loss"),
            nn_fit=tr_params.get("nn_fit"),
            saver=tr_params.get("saver"),
            model_save=tr_params.get("model_save"),
            sample_weighting=tr_params.get("sample_weighting"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            device=tr_params.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
            # windows build args
            preprocess_path=str(preprocess_path),
            folds=folds,
            model_family=model_cfg.get("family", "tree"),
            label_col=label_col,
            datacutting_cfg=cut_cfg,
            runs_root=str(paths.get("runs_root")),
            # evaluate args
            evaluate=ev_params,
            # output
            out_dir=str(Path(paths["run_dir"]) / "sweeps"),
            summary_name=str(tr_params.get("summary_name", "sweep_summary.parquet")),
        )

        summary_df = trainer.run()

        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        summary_df.to_parquet(hist_path, index=False)
        logger.info("sweep done | summary_shape=%s", summary_df.shape)
        logger.info("sweep summary saved | path=%s", str(hist_path))

        # sweep_compare：对 sweep 结果做对比汇总/画图（可选）
        if cfg.get("sweep_compare", {}).get("enabled", False):
            from ret_pred.evaluate.sweep_compare import run_sweep_compare

            run_sweep_compare(cfg)
            logger.info("sweep_compare done | out_dir=%s", cfg["sweep_compare"]["out_dir"])

        return {
            "train_history": str(hist_path),
            "sweep_dir": str(Path(paths["run_dir"]) / "sweeps"),
            "sweep_compare_dir": str(cfg.get("sweep_compare", {}).get("out_dir", "")),
        }

    # =====================================================
    # Trainer: rankic_refit_roll (NEW strategy)
    # =====================================================
    elif tr_name == "rankic_refit_roll":
        from ret_pred.trainer.rankic_refit_roll_trainer import RankICRefitRollTrainer

        # rankic_refit_roll does its own windowing based on date index (no datasplit/windows generator)
        # It still relies on preprocess parquet for data slicing to keep memory bounded.
        dates = list(all_dates)

        # Provide a default artifacts_dir under run_dir for reproducibility
        tr_params2 = dict(tr_params)
        tr_params2.setdefault("artifacts_dir", str(paths.get("run_dir")))

        trainer = RankICRefitRollTrainer(
            model=model_cfg,
            preprocess_path=str(preprocess_path),
            datacutting_cfg=cut_cfg,
            metric=tr_params2.get("metric", "rankic"),
            maximize=tr_params2.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params2.get("loss"),
            nn_fit=tr_params2.get("nn_fit"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            label_col=label_col,
            device=tr_params2.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
            params=tr_params2,
        )

        hist_df = trainer.run(dates)

        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        hist_df.to_parquet(hist_path, index=False)
        logger.info("train done | history_shape=%s", hist_df.shape)
        logger.info("train_history saved | path=%s", str(hist_path))

        # evaluate（可选/默认）：注意 Evaluator 默认 parts=[train,valid,test]，会忽略 future
        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})
        ev_params.setdefault("pred_dir", cfg["paths"]["pred_dir"])
        ev_params.setdefault("out_dir", cfg["paths"]["eval_dir"])

        evaluator = Evaluator(**ev_params)
        res = evaluator.run()

        logger.info("evaluate done")
        logger.info("metrics:\n%s", res.metrics_df)

        return {
            "train_history": str(hist_path),
            "eval_dir": cfg["paths"]["eval_dir"],
            "model_dir": cfg["paths"]["model_dir"],
            "pred_dir": cfg["paths"]["pred_dir"],
        }

    # =====================================================
    # Trainer: sweep_rankic_refit_roll
    # =====================================================
    elif tr_name == "sweep_rankic_refit_roll":
        from ret_pred.trainer.sweep_rankic_refit_roll_trainer import SweepRankICRefitRollTrainer

        dates = list(all_dates)

        sweep_cfg = dict(cfg.get("sweep", {}) or {})
        param_sets = sweep_cfg.get("param_sets") or []
        if not isinstance(param_sets, list) or len(param_sets) == 0:
            raise ValueError("trainer.name=sweep_rankic_refit_roll but cfg.sweep.param_sets is empty")

        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})

        tr_params2 = dict(tr_params)
        tr_params2.setdefault("artifacts_dir", str(paths.get("run_dir")))

        trainer = SweepRankICRefitRollTrainer(
            model=model_cfg,
            sweep=sweep_cfg,
            trainer_params=tr_params2,
            preprocess_path=str(preprocess_path),
            datacutting_cfg=cut_cfg,
            metric=tr_params2.get("metric", "rankic"),
            maximize=tr_params2.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params2.get("loss"),
            nn_fit=tr_params2.get("nn_fit"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            label_col=label_col,
            device=tr_params2.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
            evaluate=ev_params,
            out_dir=str(Path(paths["run_dir"]) / "sweeps_rankic"),
            summary_name=str(tr_params2.get("summary_name", "sweep_rankic_summary.parquet")),
        )

        summary_df = trainer.run(dates)
        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        summary_df.to_parquet(hist_path, index=False)
        logger.info("sweep_rankic done | summary_shape=%s", summary_df.shape)
        logger.info("sweep_rankic summary saved | path=%s", str(hist_path))

        return {
            "train_history": str(hist_path),
            "sweep_rankic_dir": str(Path(paths["run_dir"]) / "sweeps_rankic"),
        }

    else:
        raise KeyError(
            "Unknown trainer.name='{}', expected 'rolling' | 'sweep_rolling' | 'rankic_refit_roll' | 'sweep_rankic_refit_roll'".format(
                tr_name
            )
        )

    # NOTE: rolling / sweep_rolling 分支内已 return


def run_predict_only(cfg: Dict[str, Any]) -> str:
    """
    只跑预测（task.mode=predict/infer）时的入口。

    逻辑：
    - 填充 predict.bundle_dir 默认指向 cfg["paths"]["model_dir"]
    - 填充 predict.out_path 默认落到 pred_dir 下
    - 调用 predictor.run_predict 执行推理并保存 parquet

    参数:
        cfg (Dict[str, Any]): 完整配置（已 resolve_paths，已初始化 logging）。

    返回:
        str: 预测输出 parquet 路径。
    """
    logger = logging.getLogger(__name__)
    from ret_pred.predictor.rolling_predictor import run_predict

    pr = dict(cfg.get("predict", {}) or {})
    pr.setdefault("bundle_dir", cfg["paths"]["model_dir"])
    pr.setdefault("out_path", str(Path(cfg["paths"]["pred_dir"]) / "pred_{target_date}.parquet"))
    cfg["predict"] = pr

    out_path = run_predict(cfg)
    logger.info("predict saved | out=%s", out_path)
    return str(out_path)


# ======================================================
# main
# ======================================================
def main() -> int:
    """
    CLI 入口。

    命令行参数：
    - --config   : YAML 配置路径
    - --run-id   : 覆盖 paths.run_id（常用于快速起不同实验）
    - --runs-root: 覆盖 paths.runs_root（常用于切换输出根目录）

    返回:
        int: 进程退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="ret_pred runner (thin main)")
    parser.add_argument("--config", type=str, default="ret_pred/config.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    args = parser.parse_args()

    # 1) load config
    cfg = load_cfg(Path(args.config))

    # 2) apply CLI overrides before resolve_paths
    #    因为日志路径/产物路径通常依赖 run_id / runs_root
    cfg.setdefault("paths", {})
    if args.run_id is not None:
        cfg["paths"]["run_id"] = str(args.run_id)
    if args.runs_root is not None:
        cfg["paths"]["runs_root"] = str(args.runs_root)

    # 3) resolve all paths placeholders
    cfg = resolve_paths(cfg)

    # 4) init logging (ONLY ONCE)
    #    main 负责初始化 logging，其它模块只需要 logging.getLogger(__name__)
    from ret_pred.utils.logger import setup_logging

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("config loaded | path=%s", args.config)
    logger.info("CWD=%s", os.getcwd())
    logger.info("runs_root=%s", cfg["paths"]["runs_root"])
    logger.info("run_id=%s", cfg["paths"]["run_id"])
    logger.info("run_dir=%s", cfg["paths"]["run_dir"])

    # 5) seed
    seed = int(cfg.get("task", {}).get("seed", 42))
    seed_all(seed)
    logger.info("seed=%d", seed)

    # 6) mode dispatch
    mode = str(cfg.get("task", {}).get("mode", "train")).lower()
    if mode == "infer":
        mode = "predict"

    try:
        if mode == "predict":
            run_predict_only(cfg)
            return 0

        run_train(cfg)
        return 0

    except Exception:
        logger.exception("Unhandled exception in main")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
