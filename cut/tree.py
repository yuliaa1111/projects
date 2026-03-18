# 把 long-format 数据切成树模型训练要的 payload：
# X = df[feature_cols]
# y = df[label_col]
# keys = df[[date_col, stockid_col]]
# 输入：part_long_df（某一个 part 的长表，比如 train/valid/test）
# 输出：(payload, state)
# payload：给 trainer / model 直接用的训练数据包
# state：记录这次切片的统计信息 +（可选）缓存文件路径

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import os
import numpy as np
import pandas as pd

from .base import infer_feature_cols, ensure_dir, save_json


def cut_tree_long(
    part_long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    将 long-format DataFrame 切分为“树模型（tree-based model）”训练/推理所需的 payload。

    背景与数据约定
    ----------
    在本框架中，dataloader + preprocess 的输出统一为 long-format 表：
        - 一行代表一个样本：(date, stockid)
        - 列通常包含：date, stockid, y(可选), f1..fk（数值特征列）
    树模型（LightGBM / XGBoost / CatBoost 等）通常直接使用“二维特征矩阵”训练：
        X: [n_samples, n_features]
        y: [n_samples]
    因此本函数不构造序列窗口，只做“列选择 + 类型转换 + 可选缓存落盘”。

    核心职责
    ----------
    1) 从输入 df 中确认 label 存在，并提取 y；
    2) 确定 feature_cols（支持从 cfg 显式指定，或自动推断）；
    3) 生成 keys（date, stockid）用于评估/对齐/落盘追踪；
    4) 可选提取样本权重 w（若配置 weight_col 且列存在）；
    5) 返回 payload 给 trainer/model，返回 state 记录统计信息与缓存路径。

    参数:
        part_long_df (pd.DataFrame):
            某一个数据分片（part）的长表，通常是 train / valid / test 的其中之一。
            建议上游保证 (date, stockid) 唯一，否则可能导致训练或评估出现重复样本问题。
        cfg (Dict[str, Any]):
            本次 datacut 的配置字典（来自 YAML）。
            常用字段：
                - date_col (str): 日期列名，默认 "date"
                - stockid_col (str): 股票ID列名，默认 "stockid"
                - label_col (str): 标签列名，默认 "y"
                - weight_col (Optional[str]): 权重列名（可选）
                - feature_cols (Optional[List[str]]): 显式指定特征列；若不提供则自动推断
                - return_dataframe_X (bool):
                    True: payload["X"] 返回 DataFrame（便于调试/看列名）
                    False: payload["X"] 返回 numpy float32 数组（便于模型训练性能）
                - cache (dict):
                    enabled (bool): 是否启用缓存落盘
                    mode (str): "parquet_long" 或 "parquet_xy"
                    dir (str): 缓存目录
                    file_tpl (str): parquet 文件名模板
                    meta_tpl (str): meta json 文件名模板
        meta (Optional[Dict[str, Any]]):
            额外元信息，用于缓存命名与记录，常见字段：
                - fold: 当前滚动窗口编号（默认 0）
                - part: 当前分片名（默认 "train"）
            注意：meta 应仅用于“记录与命名”，不要影响切分逻辑本身。

    返回:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            payload:
                - "X": pd.DataFrame 或 np.ndarray
                - "y": np.ndarray (float32)
                - "keys": pd.DataFrame，含 [date_col, stockid_col]
                - "feature_cols": List[str]
                - "w": np.ndarray (float32)（可选，若权重存在）
            state:
                - 记录样本数、特征数、特征列表、是否有权重
                - 若启用缓存，会包含缓存文件路径（cached_*）

    异常:
        ValueError:
            - label_col 不存在
            - cache.mode 不支持

    维护重点
    - 若存在“未来信息数值列”，需显式排除或在 cfg 指定 feature_cols。
    - dtype 控制：y / w / X numpy 输出统一 float32，减少内存和加速训练。
    - 缓存策略：parquet_long vs parquet_xy 的落盘粒度不同，便于不同调试与复用需求。
    """
    
    # meta 用于记录本次切片的信息（fold/part 等），如果上游没传就用空 dict
    meta = meta or {}

    # 列名配置：尽量通过 cfg 统一管理，避免各模块写死列名导致维护困难
    date_col = cfg.get("date_col", "date")
    stockid_col = cfg.get("stockid_col", "stockid")
    label_col = cfg.get("label_col", "y")
    weight_col = cfg.get("weight_col", None)

    df = part_long_df

    # label 必须存在：树模型训练至少需要 y（除非你做纯推理 cut，但那应该走另一套流程）
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found")

    # feature_cols：优先使用 cfg 显式配置；否则自动推断（infer_feature_cols 函数在base.py中）
    feature_cols = cfg.get("feature_cols", None)
    if feature_cols is None:
        feature_cols = infer_feature_cols(df, date_col, stockid_col, label_col, weight_col)

    # keys：用于追踪每条样本对应的 (date, stockid)，评估/落盘/回测都需要它
    keys = df[[date_col, stockid_col]].copy()

    # y：统一转 float32，copy=False 尽量避免额外内存拷贝
    y = df[label_col].to_numpy(dtype=np.float32, copy=False)

    # w：权重是可选项。只有当 weight_col 配置了且列存在时才提取
    if weight_col and weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=np.float32, copy=False)
    else:
        w = None

    # X：可选择返回 DataFrame（调试）或 numpy array（训练性能更好）
    return_dataframe_X = bool(cfg.get("return_dataframe_X", True))
    X_df = df[feature_cols].copy()
    X = X_df if return_dataframe_X else X_df.to_numpy(dtype=np.float32, copy=False)

    # payload：trainer / model 的直接输入
    payload: Dict[str, Any] = {"X": X, "y": y, "keys": keys, "feature_cols": list(feature_cols)}
    if w is not None:
        payload["w"] = w

    # ========== 可选：缓存落盘（调试/复现实验/避免重复切分开销） ==========
    cache_cfg = cfg.get("cache", {}) or {}
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_mode = cache_cfg.get("mode", "parquet_long")
    cache_dir = cache_cfg.get("dir", "./cache/cut")
    file_tpl = cache_cfg.get("file_tpl", "{model}_{part}_fold{fold}.parquet")
    meta_tpl = cache_cfg.get("meta_tpl", "{model}_{part}_fold{fold}_meta.json")

    # state：记录本次切片的统计信息，便于日志、debug、以及 runs 目录的可追溯性
    state: Dict[str, Any] = {
        "model": "tree",
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "has_weight": bool(w is not None),
    }

    if cache_enabled:
        # fold/part 用于区分不同窗口与数据分片，避免覆盖文件
        fold = meta.get("fold", 0)
        part = meta.get("part", "train")

        parquet_path = os.path.join(cache_dir, file_tpl.format(model="tree", part=part, fold=fold))
        meta_path = os.path.join(cache_dir, meta_tpl.format(model="tree", part=part, fold=fold))

        if cache_mode == "parquet_long":
            # 将“长表子集”落盘：便于复现、也便于直接用 pandas 快速检查数据内容
            cols = [date_col, stockid_col, label_col] + list(feature_cols)
            if w is not None:
                # 注意：这里追加 weight_col 时要保证 weight_col 不为 None
                cols = cols + [weight_col]
            to_save = df[cols].copy()

            ensure_dir(parquet_path)
            to_save.to_parquet(parquet_path, index=False)

            # 缓存 meta：记录列名、特征列表、样本量等，便于后续读取或对齐检查
            save_json(
                {
                    "cache_mode": cache_mode,
                    "date_col": date_col,
                    "stockid_col": stockid_col,
                    "label_col": label_col,
                    "weight_col": weight_col if w is not None else None,
                    "feature_cols": list(feature_cols),
                    "n_samples": int(len(df)),
                    "n_features": int(len(feature_cols)),
                },
                meta_path,
            )

            state["cached_parquet"] = parquet_path
            state["cached_meta"] = meta_path

        elif cache_mode == "parquet_xy":
            # 将 X / y / keys 分开落盘：
            # - X：特征矩阵（DataFrame）
            # - y：标签列（单列 parquet）
            # - keys：索引键（date, stockid）
            # 这种方式对“只想复用特征矩阵 / 只想检查某一块”的场景更方便
            base = os.path.splitext(parquet_path)[0]
            x_path = base + "_X.parquet"
            y_path = base + "_y.parquet"
            k_path = base + "_keys.parquet"

            ensure_dir(x_path)
            X_df.to_parquet(x_path, index=False)
            pd.DataFrame({label_col: y}).to_parquet(y_path, index=False)
            keys.to_parquet(k_path, index=False)

            if w is not None:
                w_path = base + "_w.parquet"
                pd.DataFrame({weight_col: w}).to_parquet(w_path, index=False)
                state["cached_w"] = w_path

            save_json(
                {
                    "cache_mode": cache_mode,
                    "date_col": date_col,
                    "stockid_col": stockid_col,
                    "label_col": label_col,
                    "weight_col": weight_col if w is not None else None,
                    "feature_cols": list(feature_cols),
                    "n_samples": int(len(df)),
                    "n_features": int(len(feature_cols)),
                    "x_path": x_path,
                    "y_path": y_path,
                    "keys_path": k_path,
                },
                meta_path,
            )

            state["cached_X"] = x_path
            state["cached_y"] = y_path
            state["cached_keys"] = k_path
            state["cached_meta"] = meta_path

        else:
            # 这里不要 silent fallback，直接报错能避免“以为缓存了其实没缓存”的坑
            raise ValueError(f"unknown cache.mode for tree: {cache_mode}")

    return payload, state