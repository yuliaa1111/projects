"""
ret_pred/predict.py

预测入口：加载训练阶段保存的 “bundle”（模型 + 特征列 + preprocess 的 FIT state），
对某一天的横截面数据进行预处理 transform、特征对齐，然后调用 model.predict 输出预测结果 parquet。

支持两种预测模式：

1) daily（默认）
   - 需要 predict.asof_date（比如 "2024-06-10"）
   - 从 dataloader 读取当日横截面 raw long（只读 base_feature_cols）
   - 用训练时保存的 preprocess FIT state 对当日数据做 preprocess_transform
   - 对齐特征列后推理并保存预测

2) post_train
   - 用于“训练跑完立刻做一次预测”，不需要用户额外提供 asof_date
   - 从 preprocess 阶段保存的 parquet（全区间 preprocessed long）中读取最后一个交易日作为输入
   - 仍然使用 preprocess FIT state 做 transform（注意：此时输入已经是 preprocess 后的 last day，
     你目前的实现是“从 preprocess parquet 读最后一天，再走 preprocess_transform 一遍”，
     这在某些配置下可能是重复处理；但如果 preprocess parquet 存的是 preprocess 输出，那这里可行；
     若 preprocess parquet 是 raw，则更合理。项目里要统一语义。）

输出：
- 保存到 predict.out_path（默认 ./runs/{run_id}/preds/pred_{target_date}.parquet）
- 字段包含 [date, stockid, y_pred, asof_date, target_date, run_id, pred_mode, pp_state_path]

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import pickle
import logging

import numpy as np
import pandas as pd

from ret_pred.preprocess import preprocess_transform

logger = logging.getLogger(__name__)


# =========================
# IO helpers
# =========================
def _load_json(p: Path) -> Any:
    """
    从 JSON 文件读取并解析成 Python 对象。

    参数:
        p (Path): json 文件路径。

    返回:
        Any: json 解析结果（dict/list/...）。
    """
    return json.loads(p.read_text(encoding="utf-8"))


def _load_model(model_path: Path) -> Any:
    """
    加载模型文件（优先 joblib，其次 pickle）。

    说明：
    - tree 模型（如 LightGBM/CatBoost/XGBoost）常用 joblib 持久化；
    - 部分场景你也可能直接 pickle dump 了模型对象；
    - 这里做一个“先 joblib 再 pickle”的兜底加载。

    参数:
        model_path (Path): 模型文件路径（通常 bundle_dir/model.pkl）。

    返回:
        Any: 已加载的模型对象（需要具备 .predict 方法）。

    异常:
        任何加载失败都会抛出异常给上层处理。
    """
    # try joblib first
    try:
        import joblib  # type: ignore
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def _pick_preprocess_state_path(bundle_dir: Path) -> Path:
    """
    在 bundle_dir 中选择 preprocess 的 FIT state 文件路径。

    兼容两种命名：
    1) 新版：preprocess_state_fit_*.json（可能每次 fit 都保存一份，按修改时间取最新）
    2) 旧版：preprocess_state.json

    参数:
        bundle_dir (Path): 模型 bundle 目录（例如 ./runs/exp001/model）。

    返回:
        Path: preprocess state json 的路径。

    异常:
        FileNotFoundError: 找不到任何 preprocess state。
    """
    # 新版：可能存在多份 fit state，取最近修改的一份
    fit_states = list(bundle_dir.glob("preprocess_state_fit_*.json"))
    if fit_states:
        fit_states.sort(key=lambda p: p.stat().st_mtime)
        return fit_states[-1]

    # 旧版兼容
    legacy = bundle_dir / "preprocess_state.json"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"No preprocess state found in bundle_dir={bundle_dir}. "
        f"Expected preprocess_state_fit_*.json or preprocess_state.json"
    )


def _base_cols_from_bundle(feature_cols: List[str], state: Dict[str, Any]) -> List[str]:
    """
    推断“daily 读取 raw 横截面”时应该向 dataloader 要哪些字段（base_feature_cols）。

    逻辑优先级：
    1) 优先使用 preprocess FIT state 里记录的 base_feature_cols
       - 这个列表通常不包含你在 preprocess 中派生出来的列（比如 __miss mask）
       - 更适合用于 dataloader 从原始数据源取字段
    2) fallback：如果 state 没记录，则从 feature_cols 里去掉 '__miss' 后缀

    参数:
        feature_cols (List[str]): bundle 里训练时使用的最终特征列（可能包含 __miss）。
        state (Dict[str, Any]): preprocess FIT state。

    返回:
        List[str]: base_feature_cols，用于 load_cross_section(fields=...).
    """
    b = state.get("base_feature_cols")
    if isinstance(b, list) and len(b) > 0:
        return list(b)

    # fallback：从 feature_cols 去掉 __miss
    base = [c for c in feature_cols if not str(c).endswith("__miss")]
    return base


# =========================
# Feature alignment
# =========================
def _align_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    将 df 对齐到训练时的 feature_cols：
    - 如果某些 feature 在 df 中缺失：补一列 NaN（并 warning）
    - 最终按 feature_cols 的顺序取列，保证喂给模型的列顺序稳定

    为什么必须对齐？
    - 训练和推理必须使用相同的特征列集合与顺序，否则预测会错位。
    - 生产环境中经常会遇到“某些特征当天缺失/新增”的情况，需要可控处理。
      你目前选择“缺失补 NaN”是合理兜底，但要注意模型是否能处理 NaN（例如 LGBM 可处理）。

    参数:
        df (pd.DataFrame): preprocess_transform 后的 long df（包含特征列）。
        feature_cols (List[str]): bundle 保存的最终特征列顺序。

    返回:
        pd.DataFrame: 对齐后的 X_df（列顺序与 feature_cols 完全一致）。
    """
    out = df.copy()
    missing = 0
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
            missing += 1

    if missing > 0:
        logger.warning(
            "align_X: missing feature cols filled with NaN | missing=%d | total=%d",
            missing,
            len(feature_cols),
        )

    out = out[feature_cols]
    return out


# =========================
# post-train input helper
# =========================
def _load_last_day_from_preprocess_parquet(
    pp_state: Dict[str, Any],
    date_col: str,
) -> pd.DataFrame:
    """
    post_train 模式下，从 preprocess 阶段保存的 parquet 中读取“最后一个日期”的全量行作为输入。

    依赖：
    - pp_state 中必须包含 saved_path（指向 preprocess parquet 路径）

    参数:
        pp_state (Dict[str, Any]): preprocess FIT state。
        date_col (str): 日期列名。

    返回:
        pd.DataFrame: last_date 的 long df（copy）。

    异常:
        KeyError: pp_state 缺失 saved_path
        FileNotFoundError: parquet 不存在
        ValueError: parquet 为空或 last_date 无数据
        KeyError: parquet 不含 date_col
    """
    saved_path = pp_state.get("saved_path")
    if not saved_path:
        raise KeyError("preprocess_state missing 'saved_path' (needed for post-train predict).")

    p = Path(saved_path)
    if not p.exists():
        raise FileNotFoundError(f"preprocess parquet not found: {p}")

    df_all = pd.read_parquet(p)
    if df_all.empty:
        raise ValueError(f"preprocess parquet is empty: {p}")

    if date_col not in df_all.columns:
        raise KeyError(f"date_col '{date_col}' not found in preprocess parquet columns: {list(df_all.columns)}")

    last_date = df_all[date_col].max()
    df_last = df_all[df_all[date_col] == last_date].copy()

    if df_last.empty:
        raise ValueError(f"preprocess parquet has no rows for last_date={last_date} (unexpected).")

    logger.info(
        "post_train load last day from preprocess parquet | path=%s | last_date=%s | rows=%d",
        str(p),
        str(last_date),
        len(df_last),
    )
    return df_last


# =========================
# Public API
# =========================
def run_predict(cfg: Dict[str, Any]) -> str:
    """
    预测主入口：加载 bundle + 获取输入数据 + preprocess_transform + feature 对齐 + 模型预测 + 保存结果。

    cfg 关键结构（示例）：
    {
      "paths": {"run_id": "exp001"},
      "predict": {
         "mode": "daily" | "post_train",
         "post_train": false,            # 若 true 则强制 mode=post_train
         "asof_date": "2024-06-10",      # daily 模式必填
         "target_date": "2024-06-11",    # 可选；默认等于 asof_date
         "bundle_dir": "./runs/exp001/model",
         "out_path": "./runs/exp001/preds/pred_2024-06-11.parquet"
      },
      "dataloader": {...},              # daily 模式需要，用于 load_cross_section
      "preprocess": {...}               # transform 配置（注意：state 来自 bundle）
    }

    返回:
        str: 预测结果 parquet 的保存路径。

    异常:
        任何阶段失败会 logger.exception 并 re-raise，交给上层 main 处理。
    """
    task = cfg.get("task", {})
    paths = cfg.get("paths", {})

    # run_id 用于默认路径拼装与追踪
    run_id = str(paths.get("run_id") or task.get("run_id") or "exp001")
    pred_cfg = cfg.get("predict", {}) or {}

    # -------------------------
    # 1) mode detection
    # -------------------------
    pred_mode = str(pred_cfg.get("mode", "")).lower()

    # 兼容：如果配置了 post_train=true，则强制 post_train
    if pred_cfg.get("post_train", False):
        pred_mode = "post_train"

    # 允许空字符串（默认 daily）
    if pred_mode not in ("post_train", "daily", ""):
        raise ValueError(f"predict.mode must be 'daily' or 'post_train', got: {pred_mode}")
    if pred_mode == "":
        pred_mode = "daily"

    # daily 模式一般会用 asof_date；post_train 模式允许 asof_date 为空
    asof_date = str(pred_cfg.get("asof_date", ""))
    target_date = str(pred_cfg.get("target_date") or asof_date or "")

    bundle_dir = Path(pred_cfg.get("bundle_dir", f"./runs/{run_id}/model"))
    out_path = Path(pred_cfg.get("out_path", f"./runs/{run_id}/preds/pred_{target_date}.parquet"))

    logger.info(
        "run_predict start | run_id=%s | mode=%s | asof_date=%s | target_date=%s | bundle_dir=%s | out_path=%s",
        run_id,
        pred_mode,
        asof_date,
        target_date,
        str(bundle_dir),
        str(out_path),
    )

    try:
        # -------------------------
        # 2) load bundle (model + feature_cols + preprocess state)
        # -------------------------
        model_path = bundle_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        model = _load_model(model_path)
        logger.info("model loaded | path=%s", str(model_path))

        fc_path = bundle_dir / "feature_cols.json"
        if not fc_path.exists():
            raise FileNotFoundError(f"feature_cols.json not found: {fc_path}")

        raw_fc = _load_json(fc_path)

        # 兼容两种格式：
        # - list: ["f1","f2",...]
        # - dict: {"feature_cols": [...]}
        if isinstance(raw_fc, dict) and "feature_cols" in raw_fc:
            feature_cols: List[str] = list(raw_fc["feature_cols"])
        elif isinstance(raw_fc, list):
            feature_cols = list(raw_fc)
        else:
            raise ValueError(
                f"feature_cols.json format unexpected: type={type(raw_fc)}, "
                f"keys={list(raw_fc.keys()) if isinstance(raw_fc, dict) else None}"
            )

        logger.info("feature_cols loaded | n=%d | path=%s", len(feature_cols), str(fc_path))
        logger.debug("feature_cols (head)=%s", feature_cols[:10])

        # preprocess FIT state（用于 transform）
        pp_state_path = _pick_preprocess_state_path(bundle_dir)
        pp_state: Dict[str, Any] = dict(_load_json(pp_state_path))
        logger.info("preprocess state loaded | path=%s", str(pp_state_path))

        if str(pp_state.get("mode", "fit")) != "fit":
            raise ValueError(
                f"preprocess state must be FIT (mode='fit'), got mode={pp_state.get('mode')} | path={pp_state_path}"
            )

        # -------------------------
        # 3) build dataloader cfg (daily 模式需要)
        # -------------------------
        # 这里延迟 import，避免 predict 脚本加载时引入太多依赖/初始化成本
        from ret_pred.dataloader import DLCfg, load_cross_section

        dl_raw = cfg.get("dataloader", {}) or {}
        dl_cfg = DLCfg(
            source=dl_raw.get("source", "parquet_dir"),
            parquet_dir=dl_raw.get("parquet_dir", "./data"),
            long_filename=dl_raw.get("long_filename"),
            long_path=dl_raw.get("long_path"),
            date_col=dl_raw.get("date_col", "date"),
            stockid_col=dl_raw.get("stockid_col", "stockid"),
            check_unique_key=dl_raw.get("check_unique_key", True),
            feature_mask_csv=dl_raw.get("feature_mask_csv"),
            feature_col_name=dl_raw.get("feature_col_name", "feature"),
            feature_use_col=dl_raw.get("feature_use_col", "use"),
        )

        date_col = dl_cfg.date_col
        stockid_col = dl_cfg.stockid_col

        # daily 模式下只从数据源取“base”特征列（更贴合你 preprocess 的设计）
        base_feature_cols = _base_cols_from_bundle(feature_cols, pp_state)
        logger.info("base_feature_cols resolved | n=%d", len(base_feature_cols))
        logger.debug("base_feature_cols (head)=%s", base_feature_cols[:10])

        # -------------------------
        # 4) fetch input: long_today
        # -------------------------
        if pred_mode == "post_train":
            # 从 preprocess parquet 读最后一天（last_date）
            long_today = _load_last_day_from_preprocess_parquet(pp_state, date_col=date_col)

            # 用 last_date 作为 asof_date2
            asof_date2 = str(pd.to_datetime(long_today[date_col].iloc[0]).date())

            # 如果用户没写 target_date，就默认 target_date = asof_date2
            if not target_date:
                target_date = asof_date2

            meta = {
                "source": "post_train_from_preprocess",
                "asof_date": asof_date2,
                "run_id": run_id,
            }
            asof_date = asof_date2
            logger.info("post_train mode input ready | asof_date=%s | rows=%d", asof_date, len(long_today))

        else:
            # daily：用户必须提供 asof_date
            if not asof_date:
                raise ValueError("daily predict needs predict.asof_date (e.g. '2024-06-10').")

            # 从数据源加载当日横截面：只取 base_feature_cols（减少 IO）
            long_today, meta = load_cross_section(
                asof_date=asof_date,
                fields=base_feature_cols,
                cfg=dl_cfg,
            )
            logger.info("daily mode cross_section loaded | asof_date=%s | rows=%d", asof_date, len(long_today))

        if long_today is None or len(long_today) == 0:
            raise ValueError(
                f"predict input is empty. mode={pred_mode}, asof_date={asof_date}. "
                f"Check your cross-section source or preprocess parquet."
            )

        # -------------------------
        # 5) preprocess transform (using FIT state)
        # -------------------------
        # 注意：transform 的行为由 pre_cfg 决定，但统计量/阈值等依赖 pp_state（FIT）
        pre_cfg = cfg.get("preprocess", {}) or {}
        df_pre, _ = preprocess_transform(long_today, pre_cfg, state=pp_state, meta=meta)

        if df_pre is None or df_pre.empty:
            raise ValueError(
                f"preprocess_transform output is empty. mode={pred_mode}, asof_date={asof_date}. "
                f"Please check preprocess config and input df columns."
            )
        logger.info("preprocess_transform done | shape=%s", df_pre.shape)

        # -------------------------
        # 6) align features to bundle feature_cols
        # -------------------------
        X_df = _align_X(df_pre, feature_cols)
        if X_df is None or X_df.empty:
            raise ValueError(
                f"X_df is empty after align. mode={pred_mode}, asof_date={asof_date}. "
                f"df_pre.shape={df_pre.shape}, feature_cols={len(feature_cols)}"
            )
        logger.info("X aligned | shape=%s | n_features=%d", X_df.shape, X_df.shape[1])

        # -------------------------
        # 7) model predict
        # -------------------------
        y_pred = model.predict(X_df)
        logger.info("model.predict done | n_pred=%d", int(np.asarray(y_pred).reshape(-1).shape[0]))

        # -------------------------
        # 8) build output & save
        # -------------------------
        # 输出至少要带 date/stockid，不然下游无法 join 或落库
        if date_col not in df_pre.columns or stockid_col not in df_pre.columns:
            raise KeyError(
                f"df_pre must contain [{date_col},{stockid_col}] for output, got cols={list(df_pre.columns)}"
            )

        out = df_pre[[date_col, stockid_col]].copy()
        out["y_pred"] = np.asarray(y_pred).reshape(-1)

        # 追踪信息（方便复盘与排错）
        out["asof_date"] = asof_date
        out["target_date"] = target_date
        out["run_id"] = run_id
        out["pred_mode"] = pred_mode
        out["pp_state_path"] = str(pp_state_path)  # optional traceability

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)

        logger.info("predict saved | rows=%d | path=%s", len(out), str(out_path))
        return str(out_path)

    except Exception:
        # 关键：记录上下文，方便定位是哪个 run_id / mode / date 出错
        logger.exception(
            "run_predict failed | run_id=%s | mode=%s | asof_date=%s | target_date=%s | bundle_dir=%s | out_path=%s",
            run_id,
            pred_mode,
            asof_date,
            target_date,
            str(bundle_dir),
            str(out_path),
        )
        raise
