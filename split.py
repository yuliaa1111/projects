"""
ret_pred/datasplit.py

核心主干流程：_split_step_then_ratio（窗口怎么取、怎么滚、每个窗口怎么切出 train/valid/test）。
函数调用再回看工具函数：比如 _split_ratio_on_dates（窗口内按比例切日期）、_resolve_step_len（步长怎么算）、_slice_by_dates（日期列表→过滤 df）。
最后看 datasplit_long 这个 public API：它负责把“真实输入 df + cfg”串起来，选择策略、决定要不要返回 df/存 parquet、并输出统一的 folds + split_state 给系统其它部分用。

datasplit把 preprocess 输出的 clean_long_df（long-format）按“日期维度”切分为 train/valid/test，
并支持两种切分策略：

1) step_then_ratio（滚动窗口 / walk-forward / rolling）
   - 先在日期序列上切出一个“窗口”（window）
   - 再在窗口内部按 train/valid/test 比例切分
   - 然后窗口按 step 向前滚动，得到多个 fold（每个 fold 一套日期列表）

2) holdout_ratio（一次性 holdout）
   - 在全日期范围上按 train/valid/test 比例一次性切分
   - 只产生 1 个 fold，适合快速检验 baseline 或不需要滚动窗口的实验

核心设计原则：
- **以 date 为切分单位**：同一天的所有 stock 样本一定进入同一个集合，避免信息泄露。
- fold 的核心输出是“日期列表定义”（train_dates/valid_dates/test_dates），可选：
  - 返回对应 df（train_df/valid_df/test_df）
  - 或将各 part 保存为 parquet

fold 的结构是一个 dict，例如：
{
  "fold": 0,
  "window_start": Timestamp,
  "window_end": Timestamp,
  "window_len": int,
  "step_len": int|None,
  "train_dates": List[Timestamp],
  "valid_dates": List[Timestamp],
  "test_dates": List[Timestamp],
  # 可选：train_df/valid_df/test_df
  # 可选：train_path/valid_path/test_path
}
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import os

import pandas as pd


# =========================
# Helpers
# =========================
def _unique_sorted_dates(df: pd.DataFrame, date_col: str) -> List[pd.Timestamp]:
    """
    从 long-format DataFrame 中提取 **唯一且排序后的日期序列**。
    - 我们的切分逻辑是“按 date 切”，所以先得到全量 date 的有序列表。

    参数:
        df (pd.DataFrame): long-format 数据，至少包含 date_col。
        date_col (str): 日期列名。

    返回:
        List[pd.Timestamp]: 排序后的唯一日期列表。
    """
    # dropna：避免 date_col 里存在 NaN 导致 to_datetime 异常或产生 NaT
    dates = pd.to_datetime(df[date_col].dropna().unique())
    return sorted(dates)


def _slice_by_dates(df: pd.DataFrame, date_col: str, dates: List[pd.Timestamp]) -> pd.DataFrame:
    """
    只在public API里用到这个函数，只有当需要 return_dfs 或 save_parquet 时才切片 df
    实现逻辑：把“日期列表”变成“df 子集”
    如果 dates 为空：返回一个空 df（但保留列结构）
    否则：保留 df[date_col] ∈ dates 的行
    这确保了：同一天所有 stockid 行都会一起被切进去。

    参数:
        df (pd.DataFrame): long-format 数据。
        date_col (str): 日期列名。
        dates (List[pd.Timestamp]): 要保留的日期列表。

    返回:
        pd.DataFrame: 过滤后的子集（copy）。
    """
    if not dates:
        # 保持列结构一致，返回 0 行
        return df.iloc[0:0].copy()

    # 用 set 做 membership 加速（但要注意：isin 本身也会做 hash，set 这里更直观）
    s = set(dates)
    return df[df[date_col].isin(s)].copy()


def _fmt_date(ts: pd.Timestamp) -> str:
    """
    将 Timestamp 统一格式化为 YYYY-MM-DD 字符串，用于文件命名与日志。
    """
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _save_parquet(df: pd.DataFrame, path: str) -> str:
    """
    将 DataFrame 保存为 parquet（index=False），并确保父目录存在。

    参数:
        df (pd.DataFrame): 待保存数据。
        path (str): 目标文件路径。

    返回:
        str: 实际保存路径（方便上层记录到 fold 元信息里）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _split_ratio_on_dates(
    dates: List[pd.Timestamp],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, List[pd.Timestamp]]:
    """
    输入是日期列表 + 三个比例；输出是一个 dict，里面放三段日期列表。
    在一个“日期序列”上按比例切分为 train/valid/test 三段（顺序切分）。

    这里的切分方式是：
        train = dates[:n_train]                  前 n_train 个日期给训练集
        valid = dates[n_train:n_train+n_valid]   接下来的 n_valid 个日期给验证集
        test  = rest                             剩下全部日期给测试集

    重要约束：
    - train_ratio + valid_ratio + test_ratio 必须等于 1.0（允许极小浮点误差）。
    - 使用 int(n * ratio) 会向下取整，训练集天数会变少一点；被截掉的那部分日期不会丢，尾差都会落到 test。

    参数:
        dates (List[pd.Timestamp]): 已排序的日期序列。
        train_ratio (float): 训练集比例。
        valid_ratio (float): 验证集比例（可为 0）。
        test_ratio (float): 测试集比例。

    返回:
        Dict[str, List[pd.Timestamp]]:
            {
              "train_dates": [...],
              "valid_dates": [...],
              "test_dates":  [...]
            }
    """
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/valid/test ratios must sum to 1.0, got {total}")

    n = len(dates)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_dates = dates[:n_train]
    valid_dates = dates[n_train : n_train + n_valid]
    test_dates = dates[n_train + n_valid :]

    return {"train_dates": train_dates, "valid_dates": valid_dates, "test_dates": test_dates}


def _resolve_step_len(cfg: Dict[str, Any], n_dates: int) -> int:
    """
    解析 step_then_ratio 策略下的 step_len（窗口每次前移的步长）。

    支持两种配置方式（二选一）：
    - step_days: 直接给步长（按“日期个数”理解，不是自然日）
    - step_ratio: 给比例，step_len = max(int(n_dates * step_ratio), 1)

    参数:
        cfg (Dict[str, Any]): datasplit 配置字典。
        n_dates (int): 全日期长度，用于 step_ratio 的换算。

    返回:
        int: step_len（>=1）

    异常:
        ValueError: 未提供 step_days/step_ratio，或 step_len <= 0。
    """
    step_days = cfg.get("step_days", None)
    step_ratio = cfg.get("step_ratio", None)

    if step_days is not None:      #优先step_days
        step_len = int(step_days)
    elif step_ratio is not None:   #否则按照 step_ratio 计算
        step_len = max(int(n_dates * float(step_ratio)), 1)
    else:
        raise ValueError("datasplit: either 'step_days' or 'step_ratio' must be provided")

    if step_len <= 0:
        raise ValueError(f"datasplit: invalid step_len={step_len}, must be >= 1")

    return step_len


# =========================
# Strategy: step_then_ratio
# =========================
def _split_step_then_ratio(dates: List[pd.Timestamp], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    滚动窗口切分（walk-forward / rolling）策略。

    思路：
    1) 先确定窗口长度 window_len（基于 window_ratio 与 min_window_dates）
    2) 从 start=0 开始，取 window_dates = dates[start : start+window_len]
    3) 在 window_dates 内按 train/valid/test 比例切分
    4) start += step_len，继续生成下一 fold
    5) 直到 start+window_len > n 停止

    参数:
        dates (List[pd.Timestamp]): 全量有序日期列表。
        cfg (Dict[str, Any]): datasplit 配置。关键字段包括：
            - window_ratio (float): 窗口长度占全量 dates 的比例（例如 0.7）
            - min_window_dates (int): 窗口最小日期数（防止样本过少）
            - step_days (int) 或 step_ratio (float): 窗口每次滚动的步长
            - train_ratio/valid_ratio/test_ratio (float): 窗口内部切分比例（sum=1）

    返回:
        List[Dict[str, Any]]: fold 列表，每个元素定义一个窗口及其 train/valid/test 日期列表。

    异常:
        ValueError:
            - dates 为空
            - window_len 太小
            - 0 folds（常见原因：step_len 过大或 window_ratio 太大导致无法形成窗口）
    """
    n = len(dates)
    if n == 0:
        return []

    # --------
    # 1) window length
    # --------
    window_ratio = float(cfg.get("window_ratio", 0.5))
    min_window_dates = int(cfg.get("min_window_dates", 60))  # 优先返回config中设置的min_window_dates,否则就是60；dict.get(x, y) 规则是：字典里有键 x → 返回对应的值（不是返回键名本身）。字典里没有键 x → 返回默认值 y。

    window_len = max(int(n * window_ratio), min_window_dates)
    window_len = min(window_len, n)  # cap 到 n，避免超过范围

    if window_len <= 1:
        raise ValueError(
            f"datasplit: window_len={window_len} too small, adjust window_ratio/min_window_dates"
        )

    # --------
    # 2) ratios inside window
    # --------
    train_ratio = float(cfg.get("train_ratio", 0.7))
    valid_ratio = float(cfg.get("valid_ratio", 0.0))
    test_ratio = float(cfg.get("test_ratio", 0.3))

    # --------
    # 3) step length
    # --------
    step_len = _resolve_step_len(cfg, n_dates=n)

    folds: List[Dict[str, Any]] = []
    start = 0
    fold_id = 0

    # window = [start, start+window_len)
    while start + window_len <= n:   #只要从 start 开始还能取到一个完整窗口（长度 window_len），就继续循环。
        window_dates = dates[start : start + window_len]  

        # 在窗口内部按比例切分
        parts = _split_ratio_on_dates(
            window_dates,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )

        folds.append(
            {
                "fold": fold_id,
                "window_start": window_dates[0],
                "window_end": window_dates[-1],
                "window_len": len(window_dates),
                "step_len": step_len,
                **parts,
            }
        )

        fold_id += 1
        start += step_len  #窗口起点向后移动 step_len，开始下一个窗口

    if len(folds) == 0:
        raise ValueError("datasplit: produced 0 folds; try smaller window_ratio or step_len")

    return folds


# =========================
# Strategy: holdout_ratio (optional)
# =========================
def _split_holdout_ratio(dates: List[pd.Timestamp], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    一次性 holdout 切分：在全日期范围上按比例切 train/valid/test，只产出 1 个 fold。

    适用场景：
    - 快速 baseline
    - 不需要 walk-forward 的实验（例如先验证整体 pipeline）

    参数:
        dates (List[pd.Timestamp]): 全量有序日期列表（必须非空）。
        cfg (Dict[str, Any]): datasplit 配置，关键字段：
            - train_ratio/valid_ratio/test_ratio

    返回:
        List[Dict[str, Any]]: 只包含一个 fold 的列表。
    """
    parts = _split_ratio_on_dates(
        dates,
        train_ratio=float(cfg.get("train_ratio", 0.7)),
        valid_ratio=float(cfg.get("valid_ratio", 0.1)),
        test_ratio=float(cfg.get("test_ratio", 0.2)),
    )
    return [
        {
            "fold": 0,
            "window_start": dates[0],
            "window_end": dates[-1],
            "window_len": len(dates),
            "step_len": None,  # holdout 策略没有滚动步长
            **parts,
        }
    ]


# =========================
# Public API
# =========================
def datasplit_long(
    clean_long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对 preprocess 输出的 clean_long_df 进行日期级切分，返回 folds 定义与 split_state。

    输入数据约定：
    - clean_long_df 是 long-format：一行代表一个样本 (date, stockid)
    - date_col 列必须存在（默认 "date"）
    - 该函数只负责切分定义与可选落盘/返回 df，不做额外清洗

    配置 cfg 常用字段（建议写进 YAML 并在 README 说明）：
        - date_col (str): 日期列名，默认 "date"
        - strategy (str): "step_then_ratio" 或 "holdout_ratio"
        - return_dfs (bool): 是否在 fold 里附带 train_df/valid_df/test_df（内存开销大）
        - save_parquet (bool): 是否将每个 fold 的 train/valid/test 保存为 parquet
        - save_dir (str): 保存目录
        - file_name_tpl (str): 文件命名模板，例如：
            "{strategy}_fold{fold}_{part}_{date_start}_{date_end}.parquet"

    step_then_ratio 额外字段：
        - window_ratio (float)
        - min_window_dates (int)
        - step_days (int) 或 step_ratio (float)
        - train_ratio/valid_ratio/test_ratio (float)

    holdout_ratio 额外字段：
        - train_ratio/valid_ratio/test_ratio (float)

    参数:
        clean_long_df (pd.DataFrame): preprocess 后的干净 long df。
        cfg (Dict[str, Any]): datasplit 配置。
        meta (Optional[Dict[str, Any]]):
            上游 meta（例如 dataloader / preprocess 的 date_start/date_end），用于回显到 split_state。

    返回:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            - folds: List[fold_dict]（每个 fold 包含日期列表定义，可选 df/path）
            - split_state: 汇总信息（fold 数、日期范围等），用于日志与复盘

    异常:
        ValueError: 无有效日期、未知 strategy、或产生 0 folds 等。
    """
    meta = meta or {}

    # --------
    # 读取配置
    # --------
    date_col = cfg.get("date_col", "date")
    strategy = cfg.get("strategy", "step_then_ratio")

    # 是否返回 df（注意：大数据会占用大量内存）
    return_dfs = bool(cfg.get("return_dfs", False))

    # 是否保存 parquet（会产生多个文件：fold * part）
    save_parquet = bool(cfg.get("save_parquet", False))
    save_dir = cfg.get("save_dir", "./cache/splits")

    # 文件命名：date_start/date_end 这里用的是“全数据集范围”，方便定位实验数据区间
    file_name_tpl = cfg.get(
        "file_name_tpl",
        "{strategy}_fold{fold}_{part}_{date_start}_{date_end}.parquet",
    )

    df = clean_long_df

    # --------
    # 1) 提取全量日期序列
    # --------
    dates = _unique_sorted_dates(df, date_col)
    if len(dates) == 0:
        raise ValueError("datasplit_long: no valid dates found")

    # 全数据集日期范围（用于命名与 state）
    dataset_d0 = _fmt_date(dates[0])
    dataset_d1 = _fmt_date(dates[-1])

    # --------
    # 2) 生成 fold 的“日期列表定义”
    # --------
    if strategy == "step_then_ratio":
        raw_folds = _split_step_then_ratio(dates, cfg)
    elif strategy == "holdout_ratio":
        raw_folds = _split_holdout_ratio(dates, cfg)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # --------
    # 3) 可选：对每个 fold 生成 df / 保存 parquet
    # --------
    folds: List[Dict[str, Any]] = []
    for f in raw_folds:
        fold_id = int(f["fold"])
        out = dict(f)  # copy，避免修改 raw_folds

        # 只有当需要 return_dfs 或 save_parquet 时才切片 df
        # （避免不必要的 DataFrame filter 开销）
        if return_dfs or save_parquet:
            train_df = _slice_by_dates(df, date_col, out["train_dates"])
            valid_df = _slice_by_dates(df, date_col, out.get("valid_dates", []))
            test_df = _slice_by_dates(df, date_col, out["test_dates"])

            # ---- attach dfs ----
            if return_dfs:
                out["train_df"] = train_df
                out["valid_df"] = valid_df
                out["test_df"] = test_df

            # ---- save parquet ----
            if save_parquet:
                # 即使 valid_ratio=0，依然会生成一个空 valid parquet（这点你可以根据需求改：跳过空集）
                for part_name, part_df in [
                    ("train", train_df),
                    ("valid", valid_df),
                    ("test", test_df),
                ]:
                    fname = file_name_tpl.format(
                        strategy=strategy,
                        fold=fold_id,
                        part=part_name,
                        date_start=dataset_d0,
                        date_end=dataset_d1,
                    )
                    path = os.path.join(save_dir, fname)
                    _save_parquet(part_df, path)
                    out[f"{part_name}_path"] = path

        folds.append(out)

    # --------
    # 4) split_state：用于日志与复盘的汇总信息
    # --------
    split_state: Dict[str, Any] = {
        "strategy": strategy,
        "n_folds": len(folds),
        "n_dates": len(dates),
        "date_min": dataset_d0,
        "date_max": dataset_d1,
    }

    # 将上游 meta 的关键日期信息透传（可用于追踪“本次切分基于什么数据区间”）
    for k in ("date_start", "date_end"):
        if k in meta:
            split_state[k] = str(meta[k])[:10]

    return folds, split_state
