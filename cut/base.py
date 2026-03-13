from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
import os
import json
import pandas as pd

#存储一些通用函数

def infer_feature_cols(
    df: pd.DataFrame,
    date_col: str,
    stockid_col: str,
    label_col: str,
    weight_col: Optional[str] = None,
) -> List[str]:
    """
    自动推断“特征列（feature columns）”。

    在本项目中，long-format 数据通常包含：
        - date_col: 日期列（非特征）
        - stockid_col: 股票ID列（非特征）
        - label_col: 监督学习目标 y（非特征）
        - weight_col: 可选的样本权重列（非特征）
        - 其余数值列：默认视为特征列（features）

    该函数的设计目标：
        1) 减少配置负担：当配置文件没有显式提供 feature_cols 时自动推断；
        2) 保持一致性：避免 date/stockid/y/weight 等列被误当成特征；
        3) 兼容扩展：后续新增字段，只要不是上述“drop列”，且是数值类型，就会被纳入特征。

    注意事项（非常重要，后期维护常见坑）：
        - 该函数只会选取“数值型列”（numeric dtype）。如果有某些特征是 object/string，
          这里会被自动忽略；需要先在 preprocess 阶段把它们编码成数值型。
        - 若引入了新的“非特征数值列”（例如：分组ID、行业编码、未来信息标记等），
          且它们是数值型，那么它们会被误选为特征。此时应：
            1) 在上游把这些列改成非数值类型，或
            2) 在这里增加额外 drop 逻辑，或
            3) 在 config 中显式指定 feature_cols 覆盖自动推断。

    参数:
        df (pd.DataFrame):
            输入的 long-format 数据表。通常每一行是一个样本：(date, stockid)。
        date_col (str):
            日期列名，例如 "date"。
        stockid_col (str):
            股票ID列名，例如 "stockid"。
        label_col (str):
            标签列名，例如 "y"。
        weight_col (Optional[str]):
            样本权重列名（可选）。若提供，会从候选特征中排除。

    返回:
        List[str]:
            推断得到的特征列名列表（仅包含数值型列，且排除了 date/stockid/y/weight 等列）。
    """
    drop = {date_col, stockid_col, label_col}
    if weight_col:
        drop.add(weight_col)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in numeric_cols if c not in drop]
    return feats


def fmt_date(ts: Union[pd.Timestamp, str]) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> str:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path
