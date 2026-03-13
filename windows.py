"""
ret_pred/windows.py

根据 datasplit 输出的 folds（每个 fold 定义了 window_start/window_end 以及 train/valid/test 的日期列表），
以“流式（streaming）”方式从 preprocess 的 parquet（全区间 long 数据）中 **只读取当前窗口** 的数据，
再在窗口内部切出 train/valid/test，并调用 datacut_long 转成 trainer 需要的 payload，
最后 yield 给 trainer 逐窗口训练。

核心目标：
- **省内存**：不把全区间数据一次性读进来，只读当前 fold 的窗口区间。
- **可复现**：fold 已定义 train/valid/test 的日期列表；窗口内切分只按日期过滤。
- **可扩展**：通过 build_streaming_windows 注入 model_family / label_col / cut_cfg 适配 tree/seq 模型。

"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, List
import gc

import pandas as pd

from ret_pred.cut import datacut_long


# =========================
# Helpers
# =========================
def _slice_by_dates(df: pd.DataFrame, date_col: str, dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    在“窗口 df（df_win）”内部按日期列表切出子集。

    - windows_from_folds 会先用 filters 读出 window_start~window_end 的 df_win；
    - fold 里 train_dates/valid_dates/test_dates 是窗口内的日期列表；
    - 所以我们只需要在 df_win 内部再按日期集合过滤即可。

    参数:
        df (pd.DataFrame): 当前窗口内的 long df（通常已经被 window_start/window_end 过滤过）。
        date_col (str): 日期列名。
        dates (List[pd.Timestamp]): 需要保留的日期列表。

    返回:
        pd.DataFrame: 对应日期的子集（copy）。若 dates 为空则返回空 df（保留列结构）。
    """
    if not dates:
        return df.iloc[0:0].copy()

    # 统一成 Timestamp，避免传入 list[str] / list[datetime] 时出现类型不一致导致 isin 过滤不匹配
    s = set(pd.to_datetime(dates))
    return df[df[date_col].isin(s)].copy()


# =========================
# Core: streaming windows
# =========================
def windows_from_folds(
    *,
    preprocess_path: str,
    folds: Iterable[Dict[str, Any]],
    cut_cfg: Dict[str, Any],
    date_col: str = "date",
    columns: Optional[List[str]] = None,
    do_gc: bool = True,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    从 folds 生成“流式窗口”迭代器：每次只读取一个 fold 的 window 数据，并 yield 给 trainer。

    工作流程（每个 fold）：
    1) 从 fold 取出 window_start/window_end
    2) 只读取 preprocess parquet 在该窗口范围内的数据 df_win
    3) 在 df_win 内按 train_dates/valid_dates/test_dates 切出 train_df/valid_df/test_df
    4) 对每个 part 调用 datacut_long(...) 转成 payload（tree 或 seq 由 cut_cfg["mode"] 决定）
    5) yield (train_pl, valid_pl, test_pl, meta0)
    6) 删除临时变量并可选触发 gc.collect()，避免窗口迭代时内存堆积

    参数:
        preprocess_path (str):
            preprocess 阶段落盘的 parquet 路径（全区间 long 数据）。
        folds (Iterable[Dict[str, Any]]):
            datasplit_long 输出的 folds。
            每个 fold 至少包含：
              - fold
              - window_start/window_end
              - train_dates / test_dates
              - valid_dates（可选）
        cut_cfg (Dict[str, Any]):
            datacut_long 的配置（会被原样传入）。
            通常包含：
              - mode: "tree" 或 "seq"
              - label_col: 标签列名
              - 以及 cut 模块需要的其他参数（例如 feature_cols、seq_len 等）
        date_col (str):
            日期列名，默认 "date"。
        columns (Optional[List[str]]):
            只读取 parquet 的部分列（强烈建议在特征很多时使用，能显著减少 IO 与内存）。
            典型：["date","stockid","y"] + feature_cols
        do_gc (bool):
            每个窗口结束后是否主动 gc.collect()。
            - True：更稳（尤其是大窗口/大特征时）
            - False：更快一点，但可能内存峰值更高

    产出:
        Iterator[Tuple[train_payload, valid_payload, test_payload, meta]]:
            - train_payload/valid_payload/test_payload: datacut_long 输出的 payload dict
            - meta: 当前窗口元信息 dict（fold_id, window_start, window_end）

    """
    for f in folds:
        # fold id：用于日志、模型文件命名、history 归档等
        fid = int(f.get("fold", 0))

        # 当前 fold 的窗口范围（用于 parquet filters）
        window_start = pd.to_datetime(f["window_start"])
        window_end = pd.to_datetime(f["window_end"])

        # 1) 只读当前窗口数据：通过 filters 将读取范围限制在 [window_start, window_end]
        #    这一步是 streaming 的关键：避免把全区间数据都读进来。
        df_win = pd.read_parquet(
            preprocess_path,
            filters=[(date_col, ">=", window_start), (date_col, "<=", window_end)],
            columns=columns,
        )

        # 2) 在窗口内部按“日期列表”切 train/valid/test
        train_df = _slice_by_dates(df_win, date_col, f["train_dates"])
        valid_df = _slice_by_dates(df_win, date_col, f.get("valid_dates", []))
        test_df = _slice_by_dates(df_win, date_col, f["test_dates"])

        # 3) 调用 datacut_long，把每个 part 转成 trainer 需要的 payload
        #    meta 参数可以在 cut 内部做日志、统计、cache key 等用途
        train_pl, _ = datacut_long(train_df, cut_cfg, meta={"fold": fid, "part": "train"})
        valid_pl, _ = datacut_long(valid_df, cut_cfg, meta={"fold": fid, "part": "valid"})
        test_pl, _ = datacut_long(test_df,  cut_cfg, meta={"fold": fid, "part": "test"})

        # 4) 产出给 trainer：trainer 通常会对 train/valid/test 都预测、评估、保存
        meta0 = {"fold": fid, "window_start": window_start, "window_end": window_end}
        yield train_pl, valid_pl, test_pl, meta0

        # 5) 显式释放窗口变量，避免循环中引用残留导致内存无法回收
        del df_win, train_df, valid_df, test_df
        del train_pl, valid_pl, test_pl

        if do_gc:
            gc.collect()


def build_streaming_windows(
    *,
    preprocess_path: str,
    folds: Iterable[Dict[str, Any]],
    model_family: str,
    label_col: str,
    datacutting_cfg: Dict[str, Any],
    date_col: str = "date",
    runs_root: Optional[str] = None,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    构造“流式窗口迭代器”的上层封装：根据 model_family 自动设置 cut_cfg["mode"]，
    并注入 label_col，以及（可选）将 cut 的 cache 目录改写到 runs_root 下，保证路径可控。

    设计目的：
    - 训练入口（main / trainer builder）通常只知道：
        preprocess_path / folds / model_family / label_col / datacutting_cfg
      但 windows_from_folds 需要一个“可直接喂给 datacut_long”的 cut_cfg。
    - 所以这里做一层“适配与补全”。

    参数:
        preprocess_path (str): preprocess 输出 parquet 路径（全区间 long 数据）。
        folds (Iterable[Dict[str, Any]]): datasplit 产出的 folds。
        model_family (str):
            模型家族标记，用于决定 cut 模式：
            - "tree" -> cut_cfg["mode"] = "tree"
            - 其他（如 "seq"/"nn"）-> cut_cfg["mode"] = "seq"
            也可以把它做得更严格（比如只允许 {"tree","seq"}）。
        label_col (str): 标签列名，例如 "y"。
        datacutting_cfg (Dict[str, Any]): 原始 datacutting 配置（来自 yaml）。
        date_col (str): 日期列名。
        runs_root (Optional[str]):
            当前实验 runs_root，用于重定位 cut 的 cache 路径：
            - 如果 cut_cfg["cache"]["dir"] 是相对路径，则拼到 runs_root/cut

    返回:
        Iterator[...]：等价于 windows_from_folds(...) 的迭代器。

    """
    # 复制一份，避免修改原始 datacutting_cfg 影响其他模块
    cut_cfg = dict(datacutting_cfg or {})

    # 注入 label_col：让 cut 知道目标列是哪一列
    cut_cfg["label_col"] = label_col

    # 根据模型类型决定切分模式：
    # - tree：通常输出 {X_df, y, keys, ...}
    # - seq：可能输出序列张量/窗口化样本（由 cut 模块定义）
    cut_cfg["mode"] = "tree" if model_family == "tree" else "seq"

    # 如果 cut_cfg 启用了 cache，并且 cache.dir 是相对路径，则重定位到 runs_root 下
    # 这样可以保证每次实验的 cache 落到 runs/expXXX/cut 或 runs_root/cut，便于清理与复现。
    if runs_root and "cache" in cut_cfg and isinstance(cut_cfg["cache"], dict):
        d = cut_cfg["cache"].get("dir", None)
        if d and not str(d).startswith("/"):
            from pathlib import Path
            cut_cfg["cache"]["dir"] = str(Path(runs_root) / "cut")

    return windows_from_folds(
        preprocess_path=preprocess_path,
        folds=folds,
        cut_cfg=cut_cfg,
        date_col=date_col,
    )
