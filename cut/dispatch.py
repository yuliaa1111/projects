from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .tree import cut_tree_long
from .seq import cut_seq_long

"""
    本函数作为“cut 层”的统一入口（dispatch），通过 cfg["mode"] 选择对应的切分逻辑：
        - mode="tree" → cut_tree_long
        - mode="seq"  → cut_seq_long

    设计目的：
        1) 解耦：trainer 只关心 cut 的输出结构，不关心如何从 long_df 构造；
        2) 可扩展：后续新增更多模式（例如 "graph" / "panel_seq" / "multi_target"）时，
           只需要在这里加分支与实现对应 cut_xxx_long，不必侵入 trainer；
        3) 可维护：所有“输入结构转换”的逻辑都集中在 datacut 模块，便于审计数据泄漏、
           对齐问题、窗口构造是否跨 fold 等关键风险点。

    参数:
        part_long_df (pd.DataFrame):
            当前数据分片（例如 train/test 的 long_df）。
            通常由 datasplit 得到某个 part 的日期集合，再从 preprocess parquet 或内存中筛选而来。
        cfg (Dict[str, Any]):
            datacut 配置字典（来自 YAML）。
            关键字段：
                - mode (str): "tree" 或 "seq"，默认 "tree"
            其余字段将原样传递给对应的 cut 函数，例如：
                - date_col / stockid_col / label_col / weight_col
                - feature_cols 或自动推断选项
                - seq_len / stride / padding_policy 等（仅 seq 模式使用）
        meta (Optional[Dict[str, Any]]):
            额外元信息（可选），用于日志/落盘记录/结果追踪。
            例如：数据来源、日期范围、fold 编号、特征列表版本号等。
            注意：meta 只用于记录与追踪，不应影响切分逻辑的正确性（避免“隐式状态”）。

    返回:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            (bundle, meta_out)
            - bundle: 切分后的数据容器，具体结构由 mode 对应的 cut 函数决定；
            - meta_out: 输出的元信息字典（包含输入 meta 的合并/回显，以及切分过程统计等）。

    异常:
        ValueError:
            当 cfg["mode"] 不在支持范围内（目前只支持 "tree" / "seq"）时抛出。
"""

def datacut_long(
    part_long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    mode = cfg.get("mode", "tree")
    if mode == "tree":
        return cut_tree_long(part_long_df, cfg, meta=meta)
    if mode == "seq":
        return cut_seq_long(part_long_df, cfg, meta=meta)
    raise ValueError(f"datacut_long: unknown mode '{mode}'")
