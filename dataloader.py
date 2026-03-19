# ret_pred/dataloader.py
# 读取原始数据/特征 + 统一格式 + 输出 long-format DataFrame —— 框架数据入口（parquet-only）
#
# 支持两类 parquet 组织方式：
#   1) long.parquet：一张长表，列为 [date, stockid, (optional y), f1..fk]
#   2) wide parquet files：每个因子一个 parquet（index=date, columns=stockid），再转 long
#
# 输出目标（统一标准）：
# - 一条样本 = (date, stockid)
# - 列结构：date, stockid, y(可选), f1..fk（数值特征列）
# - 可选检查 (date, stockid) 唯一性（避免重复样本导致训练/评估异常）
# - 返回 (long_df, meta) —— meta 记录数据来源、区间、字段、label 来源等，用于日志与复盘
#
# 新增功能：
# 1) feature_mask_csv：按每周末 0/1 因子表筛选字段（训练 & 推理共用）
# 2) build_label：用价格列构造 next-day return 作为 label（ret(t+1)）
# 3) load_cross_section：按单日拉横截面（predict 用）

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================
# Config dataclasses
# =========================
@dataclass
class DLRequest:
    """
    一次“批量读取 long 数据”的请求参数。

    这个对象通常由 main / runner 根据 config 解析出来，用于指定：
    - 读取的时间区间
    - 需要的字段（features）
    - label 名称（默认 y）

    Attributes:
        date_start (str): 起始日期（包含），例如 "2024-01-01"
        date_end (str): 结束日期（包含），例如 "2024-06-30"
        fields (List[str]): 需要加载的特征字段名列表
        label_name (Optional[str]): label 列名（默认 "y"），若为 None 表示不读取/不生成 label
    """
    date_start: str
    date_end: str
    fields: List[str]
    label_name: Optional[str] = "y"


@dataclass
class DLCfg:
    """
    DataLoader 的配置。

    目前只支持 parquet_dir（本地 parquet 文件）作为数据源：
    - long 模式：读一张长表 long.parquet（或 long_path 指定的路径）
    - multi-long 模式：读多张 long parquet（long_paths），按 (date, stockid) 合并
    - wide 模式：每个字段一个 parquet（{field}.parquet），index=date, columns=stockid

    注意：
    - feature_mask_csv 用于统一字段选择逻辑（训练 & 推理共用），保证“特征对齐”
    - build_label 会基于 price_col 在 long/wide 读取后构造 next-day return 作为 label

    Attributes:
        source (str): 数据源类型，当前仅支持 "parquet_dir"
        parquet_dir (str): wide 模式 parquet 文件所在目录，或 long_filename 所在目录
        long_filename (str): long 模式默认文件名
        long_path (Optional[str]): long 模式显式路径（优先级高于 parquet_dir/long_filename）
        long_paths (Optional[List[str]]): multi-long 模式下的文件路径列表（非空时优先）
        long_merge_how (str): multi-long 合并方式，支持 inner/left/right/outer（默认 outer）

        date_col (str): 日期列名（long 模式中的列名；wide 模式中会 reset_index 后使用该名）
        stockid_col (str): 股票列名
        check_unique_key (bool): 是否检查 (date, stockid) 唯一性，避免重复样本

        label_src (Optional[str]): 如果 long 文件中 label 列名不是 label_name，可用 label_src 指定原列名以重命名
        label_src_candidates (Optional[List[str]]): 读取 parquet 时尝试额外读入的候选 label 列名（避免列裁剪导致找不到）

        feature_mask_csv (Optional[str]): 0/1 因子表路径（csv），用于筛选 fields
        feature_col_name (str): csv 中“字段名”的列名（Format A）
        feature_use_col (str): csv 中“是否启用(0/1)”的列名（Format A）

        build_label (bool): 是否基于价格列构造 forward return label
        label_price_col (str): 构造 label 使用的价格列名（例如 open_1d / close_1d）
        label_method (str): label 方式标识（目前用于记录 meta；默认 raw label 为 ret(t+1)）
        label_log_return (bool): True=对数收益；False=简单收益
    """
    source: str = "parquet_dir"

    parquet_dir: str = "./data"
    long_filename: str = "long.parquet"

    long_path: Optional[str] = None #这个变量可以是 string 或者 None
    long_paths: Optional[List[str]] = None
    long_merge_how: str = "outer"

    date_col: str = "date"
    stockid_col: str = "stockid"
    check_unique_key: bool = True

    label_src: Optional[str] = None
    label_src_candidates: Optional[List[str]] = None

    feature_mask_csv: Optional[str] = None
    feature_col_name: str = "feature"
    feature_use_col: str = "use"

    build_label: bool = False
    label_price_col: str = "open_1d"
    label_method: str = "open_to_open"
    label_log_return: bool = False

    # label postprocess（optional; applied after raw label is available）
    # expected dict schema: see ret_pred/md/data.md and ret_pred/md/tasks.md
    label_postprocess: Optional[Dict[str, Any]] = None


# =========================
# Label postprocess (NEW)
# =========================
def apply_label_postprocess(
    df_long: pd.DataFrame,
    cfg: Optional[Dict[str, Any]],
    *,
    date_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Apply optional label postprocess on long-format DataFrame.

    Required semantics (per docs):
    - applied AFTER raw label construction
    - does NOT touch any price columns
    - default order fixed: rank_by_date -> zscore_by_date -> winsorize_by_date

    Args:
        df_long: long DataFrame
        cfg: label_postprocess config dict (or None)
        date_col: date column name
        label_col: label column name (e.g., "y")

    Returns:
        DataFrame with processed label column (copy if enabled, otherwise original df).
    """
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return df_long

    if label_col not in df_long.columns:
        logger.warning("label_postprocess enabled but label_col missing | label_col=%s", label_col)
        return df_long
    if date_col not in df_long.columns:
        raise KeyError(f"label_postprocess needs date_col='{date_col}' in df")

    out = df_long.copy()

    keep_y_raw = bool(cfg.get("keep_y_raw", False))
    if keep_y_raw and f"{label_col}_raw" not in out.columns:
        out[f"{label_col}_raw"] = out[label_col]

    # ---- step 1) rank by date (optional) ----
    r_cfg = (cfg.get("rank_by_date", {}) or {})
    if bool(r_cfg.get("enabled", False)):
        ties_method = str(r_cfg.get("ties_method", "average"))
        if ties_method not in {"average", "min", "max", "first", "dense"}:
            raise ValueError(f"label_postprocess rank_by_date.ties_method invalid: {ties_method}")

        na_option = str(r_cfg.get("na_option", "keep"))
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(f"label_postprocess rank_by_date.na_option invalid: {na_option}")

        pct = bool(r_cfg.get("pct", True))
        centered = bool(r_cfg.get("centered", False))

        g = out.groupby(date_col, sort=False)[label_col]
        ranked = g.rank(method=ties_method, na_option=na_option, pct=pct)
        if centered:
            if pct:
                ranked = ranked - 0.5
            else:
                n = g.transform("count")
                ranked = ranked - (n + 1.0) / 2.0

        out[label_col] = ranked.where(out[label_col].notna(), np.nan)

    # ---- step 2) zscore by date (enabled by default within an enabled chain) ----
    z_cfg = (cfg.get("zscore_by_date", {}) or {})
    if bool(z_cfg.get("enabled", True)):
        ddof = int(z_cfg.get("ddof", 0))
        eps = float(z_cfg.get("eps", 1.0e-12))

        g = out.groupby(date_col, sort=False)[label_col]
        mu = g.transform("mean")
        sd = g.transform(lambda s: s.std(ddof=ddof))
        sd = sd.fillna(0.0)
        denom = sd + eps

        z = (out[label_col] - mu) / denom
        # keep NaN as NaN; but if denom collapses, z will be 0 for non-NaN labels
        z = z.where(out[label_col].notna(), np.nan)
        out[label_col] = z

        # optional hard clip right after zscore, before winsorize
        # supported forms:
        #   clip: 5           -> [-5, 5]
        #   clip: [-5, 5]     -> [lo, hi]
        clip_cfg = z_cfg.get("clip", None)
        if clip_cfg is not None:
            lo: Optional[float] = None
            hi: Optional[float] = None
            if isinstance(clip_cfg, (int, float)):
                c = float(clip_cfg)
                if c < 0:
                    raise ValueError(f"label_postprocess zscore_by_date.clip must be >= 0, got: {c}")
                lo, hi = -c, c
            elif isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
                lo, hi = float(clip_cfg[0]), float(clip_cfg[1])
            else:
                raise ValueError(
                    "label_postprocess zscore_by_date.clip must be number or [lower, upper], "
                    f"got: {clip_cfg}"
                )

            if lo is None or hi is None or lo >= hi:
                raise ValueError(
                    "label_postprocess zscore_by_date.clip invalid; require lower < upper, "
                    f"got: {clip_cfg}"
                )
            out[label_col] = out[label_col].clip(lower=lo, upper=hi)

    # ---- step 3) winsorize by date (enabled by default within an enabled chain) ----
    w_cfg = (cfg.get("winsorize_by_date", {}) or {})
    if bool(w_cfg.get("enabled", True)):
        lower_q = float(w_cfg.get("lower_q", 0.05))
        upper_q = float(w_cfg.get("upper_q", 0.95))
        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError(f"label_postprocess winsorize quantiles invalid: lower_q={lower_q}, upper_q={upper_q}")

        g = out.groupby(date_col, sort=False)[label_col]
        lo = g.transform(lambda s: s.quantile(lower_q))
        hi = g.transform(lambda s: s.quantile(upper_q))
        out[label_col] = out[label_col].clip(lo, hi)

    return out


# =========================
# Feature selection
# =========================
def read_feature_mask_csv(
    csv_path: str,
    *,
    feature_col_name: str = "feature",
    use_col: str = "use",
) -> List[str]:
    """
    读取“因子/特征开关表”csv，返回被选中的字段列表（去重且保留顺序）。

    支持两种格式：
    - Format A（推荐）：
        feature,use
        f1,1
        f2,0
        ...
      其中 feature_col_name=feature，use_col=use
    - Format B（宽表）：
        一行 0/1，列名即字段名
        f1,f2,f3
        1,0,1

    Args:
        csv_path: mask 文件路径
        feature_col_name: Format A 中字段名列名
        use_col: Format A 中启用列名

    Returns:
        List[str]: 被选中的字段名列表（dedup, keep order）

    Raises:
        FileNotFoundError: csv 不存在
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"feature_mask_csv not found: {csv_path}")

    logger.info("Reading feature mask csv: %s", str(p))
    m = pd.read_csv(p)
    logger.debug("Feature mask csv loaded: shape=%s cols=%s", m.shape, list(m.columns))

    # ---------- Format A: 两列 feature/use ----------
    if feature_col_name in m.columns and use_col in m.columns:
        out = (
            m.loc[m[use_col].astype(float) == 1, feature_col_name]
            .astype(str)
            .tolist()
        )

        # 去重并保序：防止 mask 文件重复行导致重复字段
        """
        seen = set()：准备一个集合，记录“已经见过的特征名”。set 查找很快（O(1)）。
        out2: List[str] = []：准备输出列表（类型注解表示这是“字符串列表”）。
        for x in out:：按原顺序遍历每个特征名。
        if x not in seen:：如果这是第一次看到这个特征名：
        seen.add(x)：把它记进集合
        out2.append(x)：把它放进输出列表
        结果：out2 会变成 ["f1", "f3", "f2"]
        去重：重复的只保留一次
        保序：保留第一次出现的顺序（不会像 set(out) 那样打乱顺序）
        logger.info(... len(out2))：打日志，告诉你最终选了多少个特征（Format A）。
        return out2：返回去重后的特征列表。
        """

        seen = set()
        out2: List[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                out2.append(x)

        logger.info("Feature mask selected=%d (format A)", len(out2))
        return out2

    # ---------- Format B: 宽表，第一行是 0/1 ----------
    row0 = m.iloc[0]
    selected = [str(c) for c in m.columns if float(row0[c]) == 1.0]
    logger.info("Feature mask selected=%d (format B)", len(selected))
    return selected


def filter_fields_by_mask(req_fields: List[str], selected: Optional[List[str]]) -> List[str]:
    """
    将用户请求的字段 req_fields 按 mask(selected) 进行筛选。

    设计目的：
    - dataloader 的字段选择应保持一致（训练 & 推理共用）
    - 通过 mask 保证“特征对齐”，避免线上推理特征不一致

    Args:
        req_fields: 原始请求字段列表
        selected: mask 选中的字段列表；None 表示不做筛选

    Returns:
        List[str]: 筛选后的字段列表（保持 req_fields 原有顺序）
    """
    if selected is None:
        return list(req_fields)
    sel = set(selected)
    out = [f for f in req_fields if f in sel]
    return out


# =========================
# Label building
# =========================
def build_forward_return_label_long(
    df_long: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    price_col: str,
    label_name: str = "y",
    log_return: bool = False,
) -> pd.DataFrame:
    """
    在 long-format 数据上构造“下一期收益率”label：
      y_t = P_{t+2}/P_{t+1} - 1

    - 对每只股票按日期排序，然后 groupby(stockid) 得到：
      p_t1 = shift(-1), p_t2 = shift(-2)
    - 支持简单收益 / 对数收益

    注意：
    - 本函数不会删除最后一天的样本；最后一天会得到 NaN label（由后续 preprocess 决定如何处理）
    - label_method（open_to_open 等）在本模块中仅作为 meta 记录

    Args:
        df_long: 输入 long DataFrame，至少包含 [date_col, stockid_col, price_col]
        date_col: 日期列名
        stockid_col: 股票 ID 列名
        price_col: 价格列名（例如 open_1d）
        label_name: 生成的 label 列名
        log_return: True 生成 log return；False 生成 simple return

    Returns:
        pd.DataFrame: 返回包含新增 label 列的 DataFrame（copy）

    Raises:
        ValueError: price_col 不存在
    """
    if price_col not in df_long.columns:
        raise ValueError(f"price_col '{price_col}' not found in long df")

    out = df_long.copy()

    # 确保 shift 的“下一期”是按时间递增对齐
    out = out.sort_values([stockid_col, date_col])

    g = out.groupby(stockid_col, sort=False)[price_col]
    p_t1 = g.shift(-1)
    p_t2 = g.shift(-2)
    if log_return:
        out[label_name] = np.log(p_t2) - np.log(p_t1)
    else:
        out[label_name] = p_t2 / p_t1 - 1.0

    return out


# =========================
# Long-format helpers
# =========================
def wide_to_long(
    X_dict: Dict[str, pd.DataFrame],
    y_df: Optional[pd.DataFrame],
    label_name: Optional[str],
    date_col: str,
    stockid_col: str,
) -> pd.DataFrame:
    """
    将 wide-format 的特征表字典转换为 long-format DataFrame。

    wide 格式约定：
    - index = date
    - columns = stockid
    - values = feature values

    转换为 long 后：
    - 每条样本对应 (date, stockid)
    - 每个 feature 变为一列
    - 可选拼接 y_df（同样是 wide 格式）

    Args:
        X_dict: {feature_name: wide_df}
        y_df: 可选，wide 的 label df（index=date, columns=stockid）
        label_name: label 列名；为 None 表示不拼 label
        date_col: long 中日期列名
        stockid_col: long 中股票列名

    Returns:
        pd.DataFrame: long-format DataFrame，列包含 [date_col, stockid_col, features..., (optional label)]
    """
    parts = []
    for k, df in X_dict.items():
        # stack 后 index 变为 MultiIndex(date, stockid)
        # dropna=False：保留 NaN（让后续 preprocess 决定如何处理缺失）
        s = df.stack(dropna=False).rename(k)
        parts.append(s)

    out = pd.concat(parts, axis=1)
    out.index = out.index.set_names([date_col, stockid_col])
    out = out.reset_index()

    # 可选拼 label
    if y_df is not None and label_name:
        y = y_df.stack(dropna=False).rename(label_name)
        y.index = y.index.set_names([date_col, stockid_col])
        out = out.merge(y.reset_index(), on=[date_col, stockid_col], how="left")

    return out


def validate_long(df: pd.DataFrame, date_col: str, stockid_col: str, check_unique_key: bool = True) -> None:
    """
    对输出 long-format DataFrame 做基本合法性校验。

    校验项：
    1) 必须存在 date_col 和 stockid_col
    2) 可选检查 (date, stockid) 唯一性，防止重复样本

    Args:
        df: 待校验 DataFrame
        date_col: 日期列名
        stockid_col: 股票列名
        check_unique_key: 是否检查唯一键

    Raises:
        ValueError: 缺列或存在重复键
    """
    if date_col not in df.columns or stockid_col not in df.columns:
        raise ValueError(f"long df must contain '{date_col}' and '{stockid_col}'")

    if check_unique_key:
        dup = int(df.duplicated([date_col, stockid_col]).sum())
        if dup > 0:
            raise ValueError(f"duplicate (date, stockid) rows: {dup}")


def _rename_label_if_needed(df: pd.DataFrame, label_src: Optional[str], label_name: Optional[str]) -> pd.DataFrame:
    """
    在 long 文件中 label 列名可能不是统一的 "y"：
    - 如果 label_name 已经存在：不动
    - 否则如果 label_src 存在且在 df 中：rename 为 label_name
    - 否则：原样返回（后续可能会 build_label）

    Args:
        df: 输入 DataFrame
        label_src: 原 label 列名（例如 "ret_1d"）
        label_name: 目标 label 列名（例如 "y"）

    Returns:
        pd.DataFrame: 可能发生 rename 的 DataFrame
    """
    if not label_name:
        return df
    if label_name in df.columns:
        return df
    if label_src and label_src in df.columns:
        return df.rename(columns={label_src: label_name})
    return df


def _resolve_long_path(cfg: DLCfg) -> Path:
    """
    解析 long parquet 路径：
    - cfg.long_path 优先（显式指定）
    - 否则使用 parquet_dir / long_filename

    Args:
        cfg: DLCfg

    Returns:
        Path: long parquet 文件路径
    """
    if cfg.long_path:
        return Path(cfg.long_path)
    return Path(cfg.parquet_dir) / cfg.long_filename


def _resolve_long_paths(cfg: DLCfg) -> List[Path]:
    """
    解析可用的 long parquet 路径列表。

    优先级：
    1) cfg.long_paths（multi-long）
    2) cfg.long_path / cfg.long_filename（single-long）
    """
    if cfg.long_paths:
        return [Path(p) for p in cfg.long_paths]
    return [_resolve_long_path(cfg)]


def _get_parquet_columns(path: Path) -> List[str]:
    """
    获取 parquet 字段名（只读 schema，不读全量数据）。
    """
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        # 兜底：极端环境下回退到 pandas（可能更慢）
        logger.warning("failed to inspect parquet schema via pyarrow, fallback pandas | path=%s", str(path))
        return list(pd.read_parquet(path).columns)


def _label_candidate_cols(cfg: DLCfg, label_name: Optional[str]) -> List[str]:
    out: List[str] = []
    if label_name:
        out.append(label_name)
    if cfg.label_src:
        out.append(cfg.label_src)
    if cfg.label_src_candidates:
        out.extend(cfg.label_src_candidates)
    # 去重保序
    return list(dict.fromkeys(out))


def _read_long_part_with_date_filter(
    path: Path,
    *,
    columns: List[str],
    date_col: str,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    从单个 long parquet 读取指定列，并按日期过滤。
    """
    cols = list(dict.fromkeys(columns))
    start_ts = pd.Timestamp(date_start)
    end_ts = pd.Timestamp(date_end)

    try:
        df = pd.read_parquet(
            path,
            columns=cols,
            filters=[(date_col, ">=", start_ts), (date_col, "<=", end_ts)],
        )
    except TypeError:
        # 某些引擎/版本可能不支持 filters，回退到读列后手动过滤
        df = pd.read_parquet(path, columns=cols)

    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' missing in parquet: {path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)].copy()


def _merge_long_parts(
    parts: List[pd.DataFrame],
    *,
    date_col: str,
    stockid_col: str,
    how: str,
) -> pd.DataFrame:
    """
    按 (date, stockid) 合并多个 long 子表。
    """
    if not parts:
        raise ValueError("no long parts to merge")

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"long_merge_how must be one of inner/left/right/outer, got: {how}")

    keys = [date_col, stockid_col]
    out = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        dup_cols = [c for c in part.columns if c in out.columns and c not in keys]
        if dup_cols:
            logger.warning("drop duplicated data columns when merging part=%d: %s", i, dup_cols[:10])
            part = part.drop(columns=dup_cols)
        out = out.merge(part, on=keys, how=how, sort=False)
    return out


def _auto_infer_fields_from_long_df(
    df: pd.DataFrame,
    *,
    date_col: str,
    stockid_col: str,
    label_name: Optional[str],
) -> List[str]:
    """
    当 req.fields 未提供时，从 long df 自动推断“数值特征列”。

    推断规则：
    - 排除 date_col、stockid_col、label_name
    - 只保留 numeric dtype 的列（避免把字符串列/分类列误当特征）

    Args:
        df: long DataFrame
        date_col: 日期列名
        stockid_col: 股票列名
        label_name: label 列名（可为 None）

    Returns:
        List[str]: 推断出的数值特征列名列表
    """
    exclude = {date_col, stockid_col}
    if label_name:
        exclude.add(label_name)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


# =========================
# Public API (train / batch load)
# =========================
def load_long(req: DLRequest, cfg: DLCfg, *, client=None) -> Tuple[pd.DataFrame, dict]:
    """
    批量加载指定区间的 long-format 数据，用于训练/评估。

    总体流程：
    1) （可选）读取 feature_mask_csv，得到 selected_features
    2) 优先尝试 long_file 模式（long_path 存在）：
       - 读取 parquet（可按 columns 裁剪）
       - 日期过滤到 [date_start, date_end]
       - label rename 或 build_label
       - 若 req.fields 缺省：自动推断特征列
       - 最终保留 [date, stockid, (optional y), features...]
    3) 若 long_file 不存在，则进入 wide_files 模式：
       - 对每个字段读取 {field}.parquet
       - wide_to_long 转 long
       - （可选）build_label
    4) validate_long：缺列/重复键检查
    5) 返回 (df_long, meta)

    Args:
        req: DLRequest，包含区间/字段/label 信息
        cfg: DLCfg，包含数据源路径、列名、mask、label 构造设置等
        client: 预留（未来支持 ClickHouse 等在线数据源时使用）；当前未使用

    Returns:
        Tuple[pd.DataFrame, dict]:
            - df_long: long-format DataFrame（已过滤日期与字段）
            - meta: dict，记录本次加载的关键元信息（来源、字段、mask、label 来源等）

    Raises:
        ValueError: source 不支持；wide 模式 fields 缺失等
        FileNotFoundError / Exception: parquet 读取失败等
    """
    if cfg.source != "parquet_dir":
        raise ValueError(f"only 'parquet_dir' is supported, got source={cfg.source}")

    logger.info(
        "load_long start | date=[%s,%s] label_name=%s fields=%d build_label=%s mask=%s",
        req.date_start, req.date_end, req.label_name, len(req.fields or []),
        bool(cfg.build_label), str(cfg.feature_mask_csv),
    )

    meta: Dict[str, Any] = {
        "source": "parquet_dir",
        "date_start": req.date_start,
        "date_end": req.date_end,
        "fields": list(req.fields),
        "label_name": req.label_name,
        "feature_mask_csv": cfg.feature_mask_csv,
    }

    # ---------- feature mask（可选） ----------
    selected = None
    if cfg.feature_mask_csv:
        selected = read_feature_mask_csv(
            cfg.feature_mask_csv,
            feature_col_name=cfg.feature_col_name,
            use_col=cfg.feature_use_col,
        )
        logger.info("feature_mask applied | selected=%d", len(selected))

    long_paths = _resolve_long_paths(cfg)
    explicit_multi_long = bool(cfg.long_paths)

    # =====================================================================
    # 1) multi-long parquet 模式（显式配置 long_paths 时启用）
    # =====================================================================
    if explicit_multi_long:
        if not long_paths:
            raise ValueError("dataloader.long_paths is empty")

        missing = [str(p) for p in long_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"long_paths contains missing files: {missing}")

        meta["input_format"] = "long_files"
        meta["paths"] = [str(p) for p in long_paths]
        meta["long_merge_how"] = cfg.long_merge_how

        fields = filter_fields_by_mask(req.fields, selected) if req.fields else []
        meta["fields_after_mask"] = list(fields)
        meta["fields_filtered_by_mask"] = bool(cfg.feature_mask_csv)

        label_try = _label_candidate_cols(cfg, req.label_name)

        schema_by_path: Dict[Path, List[str]] = {}
        for p in long_paths:
            schema_by_path[p] = _get_parquet_columns(p)

        # req.fields 为空时，基于 schema 推断字段并（可选）过 mask
        if not req.fields:
            exclude = {cfg.date_col, cfg.stockid_col}
            exclude.update(label_try)

            inferred: List[str] = []
            seen = set()
            for p in long_paths:
                for c in schema_by_path[p]:
                    if c in exclude or c in seen:
                        continue
                    seen.add(c)
                    inferred.append(c)

            if selected is not None:
                sel = set(selected)
                inferred = [c for c in inferred if c in sel]

            fields = inferred
            meta["fields_inferred"] = True
            meta["fields_after_mask"] = list(fields)
            logger.info("multi-long fields inferred from schema | n=%d", len(fields))

        need_cols = [cfg.date_col, cfg.stockid_col] + list(fields)
        for c in label_try:
            if c not in need_cols:
                need_cols.append(c)
        if cfg.build_label and cfg.label_price_col not in need_cols:
            need_cols.append(cfg.label_price_col)

        parts: List[pd.DataFrame] = []
        for i, p in enumerate(long_paths):
            available = set(schema_by_path[p])
            cols = [c for c in need_cols if c in available]

            if cfg.date_col not in cols:
                cols.append(cfg.date_col)
            if cfg.stockid_col not in cols:
                cols.append(cfg.stockid_col)

            # 不是第一个文件且只含 key 列时，跳过可减少无效 IO
            if i > 0 and len(cols) == 2:
                continue

            part = _read_long_part_with_date_filter(
                p,
                columns=cols,
                date_col=cfg.date_col,
                date_start=req.date_start,
                date_end=req.date_end,
            )
            parts.append(part)
            logger.info("multi-long part loaded | path=%s | shape=%s | n_cols=%d", str(p), part.shape, len(part.columns))

        df = _merge_long_parts(
            parts,
            date_col=cfg.date_col,
            stockid_col=cfg.stockid_col,
            how=cfg.long_merge_how,
        )
        logger.info("multi-long merged | shape=%s cols=%d", df.shape, len(df.columns))

        # ---------- label rename / build_label ----------
        df = _rename_label_if_needed(df, cfg.label_src, req.label_name)
        if req.label_name and cfg.build_label and req.label_name not in df.columns:
            logger.info(
                "building label | method=%s price_col=%s log_return=%s",
                cfg.label_method, cfg.label_price_col, bool(cfg.label_log_return),
            )
            df = build_forward_return_label_long(
                df,
                date_col=cfg.date_col,
                stockid_col=cfg.stockid_col,
                price_col=cfg.label_price_col,
                label_name=req.label_name,
                log_return=cfg.label_log_return,
            )
            meta["label_src"] = f"computed_{cfg.label_method}_shift(-2,-1)"
            meta["price_col_for_label"] = cfg.label_price_col
        else:
            meta["label_src"] = cfg.label_src
            meta["price_col_for_label"] = (cfg.label_price_col if cfg.build_label else None)

        # ---------- label postprocess（optional; after raw label is available） ----------
        if req.label_name and req.label_name in df.columns:
            before_na = int(df[req.label_name].isna().sum())
            df = apply_label_postprocess(
                df,
                cfg.label_postprocess,
                date_col=cfg.date_col,
                label_col=req.label_name,
            )
            after_na = int(df[req.label_name].isna().sum())
            meta["label_postprocess"] = dict(cfg.label_postprocess or {})
            meta["label_postprocess"]["n_nan_before"] = before_na
            meta["label_postprocess"]["n_nan_after"] = after_na

        # ---------- 最终列裁剪：只保留标准输出列 ----------
        keep = [cfg.date_col, cfg.stockid_col]
        if req.label_name and req.label_name in df.columns:
            keep.append(req.label_name)
        keep += list(fields)
        keep = list(dict.fromkeys([c for c in keep if c in df.columns]))
        df = df[keep].copy()

        validate_long(df, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
        logger.info("load_long done (multi-long) | shape=%s cols=%d", df.shape, len(df.columns))
        return df.reset_index(drop=True), meta

    long_path = _resolve_long_path(cfg)
    logger.debug("resolved long_path=%s", str(long_path))

    # =====================================================================
    # 1) long parquet file 模式（优先）
    # =====================================================================
    if long_path.exists():
        meta["input_format"] = "long_file"
        meta["path"] = str(long_path)

        # 如果 req.fields 提供，则先按 mask 筛选；若不提供，后面会自动推断
        fields = filter_fields_by_mask(req.fields, selected) if req.fields else []
        meta["fields_after_mask"] = list(fields)
        meta["fields_filtered_by_mask"] = bool(cfg.feature_mask_csv)

        logger.info(
            "reading long parquet | path=%s | fields_specified=%s after_mask=%d",
            str(long_path), bool(req.fields), len(fields),
        )

        # ---------- 读取 parquet（支持 columns 裁剪减少 IO） ----------
        # 复杂点：如果 fields 非空，我们只读必要列；但 label 列名可能不确定，需要多尝试读入。
        try:
            if fields:
                need_cols = [cfg.date_col, cfg.stockid_col] + list(fields)

                # label 列的可能来源：
                # - req.label_name：最终希望得到的列名（可能本来就存在于 parquet）
                # - cfg.label_src：原始 label 列名（可能需要 rename）
                # - cfg.label_src_candidates：额外候选
                label_try: List[str] = []
                if req.label_name:
                    label_try.append(req.label_name)
                if cfg.label_src:
                    label_try.append(cfg.label_src)
                if cfg.label_src_candidates:
                    label_try.extend(cfg.label_src_candidates)

                for c in label_try:
                    if c not in need_cols:
                        need_cols.append(c)

                # build_label 需要 price_col，因此即使不在 fields 里也要读入
                if cfg.build_label and cfg.label_price_col not in need_cols:
                    need_cols.append(cfg.label_price_col)

                # dict.fromkeys 用于去重保序
                df = pd.read_parquet(long_path, columns=list(dict.fromkeys(need_cols)))
            else:
                # fields 未指定：读全量，后面再推断数值列
                df = pd.read_parquet(long_path)
        except Exception:
            logger.exception("failed to read parquet: %s", str(long_path))
            raise

        logger.debug("raw loaded shape=%s cols=%d", df.shape, len(df.columns))

        # ---------- 日期过滤（确保只保留请求区间） ----------
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
        s, e = pd.to_datetime(req.date_start), pd.to_datetime(req.date_end)
        before = len(df)
        df = df[(df[cfg.date_col] >= s) & (df[cfg.date_col] <= e)].copy()
        logger.info("date filtered | before=%d after=%d", before, len(df))

        # ---------- label rename / build_label ----------
        df = _rename_label_if_needed(df, cfg.label_src, req.label_name)

        # 若用户要求 label 且 build_label=True 且当前仍没有 label，就用 price_col 计算
        if req.label_name and cfg.build_label and req.label_name not in df.columns:
            logger.info(
                "building label | method=%s price_col=%s log_return=%s",
                cfg.label_method, cfg.label_price_col, bool(cfg.label_log_return),
            )
            df = build_forward_return_label_long(
                df,
                date_col=cfg.date_col,
                stockid_col=cfg.stockid_col,
                price_col=cfg.label_price_col,
                label_name=req.label_name,
                log_return=cfg.label_log_return,
            )
            meta["label_src"] = f"computed_{cfg.label_method}_shift(-2,-1)"
            meta["price_col_for_label"] = cfg.label_price_col
        else:
            # label 不需要计算：记录 meta 方便复盘
            meta["label_src"] = cfg.label_src
            meta["price_col_for_label"] = (cfg.label_price_col if cfg.build_label else None)

        # ---------- label postprocess（optional; after raw label is available） ----------
        if req.label_name and req.label_name in df.columns:
            before_na = int(df[req.label_name].isna().sum())
            df = apply_label_postprocess(
                df,
                cfg.label_postprocess,
                date_col=cfg.date_col,
                label_col=req.label_name,
            )
            after_na = int(df[req.label_name].isna().sum())
            meta["label_postprocess"] = dict(cfg.label_postprocess or {})
            meta["label_postprocess"]["n_nan_before"] = before_na
            meta["label_postprocess"]["n_nan_after"] = after_na

        # ---------- fields 自动推断（当 req.fields 未提供） ----------
        if not req.fields:
            inferred = _auto_infer_fields_from_long_df(
                df,
                date_col=cfg.date_col,
                stockid_col=cfg.stockid_col,
                label_name=req.label_name,
            )
            # 若有 mask，则推断后还要过一遍 mask（保证一致性）
            if selected is not None:
                inferred = [c for c in inferred if c in set(selected)]
            meta["fields_inferred"] = True
            meta["fields_after_mask"] = list(inferred)
            fields = inferred
            logger.info("fields inferred | n=%d", len(fields))
        else:
            fields = filter_fields_by_mask(req.fields, selected) if req.fields else []

        # ---------- 最终列裁剪：只保留标准输出列 ----------
        keep = [cfg.date_col, cfg.stockid_col]
        if req.label_name and req.label_name in df.columns:
            keep.append(req.label_name)
        keep += list(fields)
        keep = list(dict.fromkeys([c for c in keep if c in df.columns]))
        df = df[keep].copy()

        validate_long(df, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
        logger.info("load_long done | shape=%s cols=%d", df.shape, len(df.columns))
        return df.reset_index(drop=True), meta

    # =====================================================================
    # 2) wide parquet files 模式（long 文件不存在时启用）
    # =====================================================================
    if not req.fields:
        raise ValueError(
            "wide parquet mode requires explicit dataloader.fields. "
            "If you want fields optional, please provide a long parquet file (long_path or long_filename)."
        )

    pdir = Path(cfg.parquet_dir)
    fields = filter_fields_by_mask(req.fields, selected)
    meta["input_format"] = "wide_files"
    meta["fields_after_mask"] = list(fields)
    meta["fields_filtered_by_mask"] = bool(cfg.feature_mask_csv)

    logger.info("wide files mode | parquet_dir=%s fields=%d", str(pdir), len(fields))

    X_dict: Dict[str, pd.DataFrame] = {}
    for f in fields:
        fp = pdir / f"{f}.parquet"
        try:
            w = pd.read_parquet(fp)
        except Exception:
            logger.exception("failed to read wide parquet: %s", str(fp))
            raise

        # wide parquet 的 index 应为 date
        w.index = pd.to_datetime(w.index, errors="coerce")
        # 统一排序 + 区间裁剪（避免读进来后还带很长历史）
        w = w.sort_index().loc[pd.to_datetime(req.date_start):pd.to_datetime(req.date_end)]
        X_dict[f] = w
        logger.debug("loaded wide field=%s shape=%s", f, w.shape)

    # 若要 build_label，则必须加载 price_col（不一定在 req.fields 内）
    if cfg.build_label and cfg.label_price_col not in X_dict:
        fp = pdir / f"{cfg.label_price_col}.parquet"
        logger.info("loading price_col for label: %s", cfg.label_price_col)
        w = pd.read_parquet(fp)
        w.index = pd.to_datetime(w.index, errors="coerce")
        w = w.sort_index().loc[pd.to_datetime(req.date_start):pd.to_datetime(req.date_end)]
        X_dict[cfg.label_price_col] = w

    df = wide_to_long(X_dict, None, None, cfg.date_col, cfg.stockid_col)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")

    if req.label_name and cfg.build_label:
        logger.info("building label (wide mode) | method=%s", cfg.label_method)
        df = build_forward_return_label_long(
            df,
            date_col=cfg.date_col,
            stockid_col=cfg.stockid_col,
            price_col=cfg.label_price_col,
            label_name=req.label_name,
            log_return=cfg.label_log_return,
        )
        meta["label_src"] = f"computed_{cfg.label_method}_shift(-2,-1)"
        meta["price_col_for_label"] = cfg.label_price_col

    # label postprocess（optional; after raw label is available）
    if req.label_name and req.label_name in df.columns:
        before_na = int(df[req.label_name].isna().sum())
        df = apply_label_postprocess(
            df,
            cfg.label_postprocess,
            date_col=cfg.date_col,
            label_col=req.label_name,
        )
        after_na = int(df[req.label_name].isna().sum())
        meta["label_postprocess"] = dict(cfg.label_postprocess or {})
        meta["label_postprocess"]["n_nan_before"] = before_na
        meta["label_postprocess"]["n_nan_after"] = after_na

    validate_long(df, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
    logger.info("load_long done (wide mode) | shape=%s cols=%d", df.shape, len(df.columns))
    return df.reset_index(drop=True), meta


# =========================
# Public API (predict) inference: load cross section for a single date
# =========================
def load_cross_section(
    asof_date: str,
    cfg: DLCfg,
    fields: List[str],
    *,
    client=None,
) -> Tuple[pd.DataFrame, dict]:
    """
    加载某一天的横截面数据（predict/infer 用）。

    设计目的：
    - 预测阶段通常只需要单日横截面 X(asof_date)
    - 同样支持 long_file / wide_files 两种存储方式
    - 同样支持 feature_mask_csv，保证与训练阶段字段一致

    Args:
        asof_date: 目标日期（例如 "2024-06-14"）
        cfg: DLCfg
        fields: 想要加载的字段列表（建议与训练一致；也可传入全量再自动推断）
        client: 预留（未来在线数据源）

    Returns:
        Tuple[pd.DataFrame, dict]:
            - out: long-format 横截面 DataFrame，仅包含该日样本
            - meta: 记录来源、字段、mask、path 等信息
    """
    if cfg.source != "parquet_dir":
        raise ValueError("load_cross_section only supports parquet_dir")

    logger.info(
        "load_cross_section start | asof_date=%s fields=%d mask=%s",
        asof_date, len(fields or []), str(cfg.feature_mask_csv),
    )

    meta: Dict[str, Any] = {
        "source": "parquet_dir",
        "asof_date": asof_date,
        "fields": list(fields),
        "feature_mask_csv": cfg.feature_mask_csv,
    }

    # ---------- feature mask（可选） ----------
    selected = None
    if cfg.feature_mask_csv:
        selected = read_feature_mask_csv(
            cfg.feature_mask_csv,
            feature_col_name=cfg.feature_col_name,
            use_col=cfg.feature_use_col,
        )

    fields2 = filter_fields_by_mask(fields, selected) if fields else []
    meta["fields_after_mask"] = list(fields2)

    d = pd.to_datetime(asof_date)
    long_paths = _resolve_long_paths(cfg)
    explicit_multi_long = bool(cfg.long_paths)

    # =====================================================================
    # 1) multi-long 横截面
    # =====================================================================
    if explicit_multi_long:
        if not long_paths:
            raise ValueError("dataloader.long_paths is empty")

        missing = [str(p) for p in long_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"long_paths contains missing files: {missing}")

        schema_by_path: Dict[Path, List[str]] = {p: _get_parquet_columns(p) for p in long_paths}

        if not fields2:
            exclude = {cfg.date_col, cfg.stockid_col}
            inferred: List[str] = []
            seen = set()
            for p in long_paths:
                for c in schema_by_path[p]:
                    if c in exclude or c in seen:
                        continue
                    seen.add(c)
                    inferred.append(c)
            if selected is not None:
                sel = set(selected)
                inferred = [c for c in inferred if c in sel]
            fields2 = inferred
            meta["fields_inferred"] = True
            meta["fields_after_mask"] = list(fields2)
            logger.info("multi-long cross-section fields inferred from schema | n=%d", len(fields2))

        need_cols = [cfg.date_col, cfg.stockid_col] + list(fields2)
        if cfg.label_price_col and cfg.label_price_col not in need_cols:
            need_cols.append(cfg.label_price_col)

        parts: List[pd.DataFrame] = []
        for i, p in enumerate(long_paths):
            available = set(schema_by_path[p])
            cols = [c for c in need_cols if c in available]
            if cfg.date_col not in cols:
                cols.append(cfg.date_col)
            if cfg.stockid_col not in cols:
                cols.append(cfg.stockid_col)

            if i > 0 and len(cols) == 2:
                continue

            part = _read_long_part_with_date_filter(
                p,
                columns=cols,
                date_col=cfg.date_col,
                date_start=str(d.date()),
                date_end=str(d.date()),
            )
            parts.append(part)

        out = _merge_long_parts(
            parts,
            date_col=cfg.date_col,
            stockid_col=cfg.stockid_col,
            how=cfg.long_merge_how,
        )
        out = out[out[cfg.date_col] == d].copy()

        keep = [cfg.date_col, cfg.stockid_col] + list(fields2)
        keep = list(dict.fromkeys([c for c in keep if c in out.columns]))
        out = out[keep].copy()

        validate_long(out, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
        meta["input_format"] = "long_files_cross_section"
        meta["paths"] = [str(p) for p in long_paths]
        meta["long_merge_how"] = cfg.long_merge_how
        logger.info("load_cross_section done (multi-long) | shape=%s", out.shape)
        return out.reset_index(drop=True), meta

    long_path = _resolve_long_path(cfg)

    # =====================================================================
    # 2) long_file 横截面
    # =====================================================================
    if long_path.exists():
        cols = [cfg.date_col, cfg.stockid_col] + list(fields2)

        # 某些时候推理也要读 price_col（例如后续要对齐或做检查）；这里不生成 label，仅“可用就读”
        if cfg.label_price_col and cfg.label_price_col not in cols:
            cols.append(cfg.label_price_col)

        logger.info("reading long parquet for cross-section | path=%s", str(long_path))

        try:
            if fields2:
                df = pd.read_parquet(long_path, columns=list(dict.fromkeys(cols)))
            else:
                # fields2 为空：读全量，再在单日样本上推断数值列
                df = pd.read_parquet(long_path)
        except Exception:
            logger.exception("failed to read parquet: %s", str(long_path))
            raise

        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
        out = df[df[cfg.date_col] == d].copy()

        # fields 未显式给：在该日横截面上推断数值列（并过 mask）
        if not fields2:
            inferred = _auto_infer_fields_from_long_df(
                out,
                date_col=cfg.date_col,
                stockid_col=cfg.stockid_col,
                label_name=None,
            )
            if selected is not None:
                inferred = [c for c in inferred if c in set(selected)]
            fields2 = inferred
            meta["fields_inferred"] = True
            meta["fields_after_mask"] = list(fields2)
            logger.info("cross-section fields inferred | n=%d", len(fields2))

        # 最终裁剪为标准输出列
        keep = [cfg.date_col, cfg.stockid_col] + list(fields2)
        keep = list(dict.fromkeys([c for c in keep if c in out.columns]))
        out = out[keep].copy()

        validate_long(out, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
        meta["input_format"] = "long_file_cross_section"
        meta["path"] = str(long_path)
        logger.info("load_cross_section done | shape=%s", out.shape)
        return out.reset_index(drop=True), meta

    # =====================================================================
    # 3) wide_files 横截面
    # =====================================================================
    if not fields2:
        raise ValueError(
            "wide parquet cross-section requires explicit fields. "
            "If you want fields optional, please provide a long parquet file."
        )

    pdir = Path(cfg.parquet_dir)
    logger.info("wide files cross-section | parquet_dir=%s", str(pdir))

    X_dict: Dict[str, pd.DataFrame] = {}
    for f in fields2:
        fp = pdir / f"{f}.parquet"
        try:
            w = pd.read_parquet(fp)
        except Exception:
            logger.exception("failed to read wide parquet: %s", str(fp))
            raise

        w.index = pd.to_datetime(w.index, errors="coerce")

        # 若该字段缺少 asof_date，当天整行填 NaN（保证输出结构一致，不因缺字段/缺行崩掉）
        if d not in w.index:
            logger.warning("asof_date not found in wide field=%s, filling NaN row", f)
            tmp = pd.DataFrame([np.nan] * w.shape[1], index=w.columns).T
            tmp.index = [d]
            tmp.columns = w.columns
            X_dict[f] = tmp
        else:
            X_dict[f] = w.loc[[d]]

    out = wide_to_long(X_dict, None, None, cfg.date_col, cfg.stockid_col)
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    validate_long(out, cfg.date_col, cfg.stockid_col, cfg.check_unique_key)
    meta["input_format"] = "wide_files_cross_section"
    logger.info("load_cross_section done (wide) | shape=%s", out.shape)
    return out.reset_index(drop=True), meta
