# ret_pred/preprocess.py
"""
Preprocessing utilities for ret_pred.
主流程存储在 def _preprocess_core()里，提供 preprocess_fit_transform 和 preprocess_transform 两个接口。

目标
----
将任意来源的 long-format 数据统一处理为“可训练/可推理”的标准输入：
- 统一 date 类型、清理 inf
- fit: drop 缺失 y(label) 的样本
- 自动识别数值特征列 feature_cols（排除 date/stockid/y 等）
- fit: 按缺失率阈值 drop “坏特征列”和“坏样本行”
- Winsorize 去极值（可 global/date）
- zscore 标准化（可 global/date）
- 按 nan_policy 决定是否/如何填充 NaN（strict 模式要求最终无 NaN）
- 可选为每个特征列添加 missing mask 列（__miss）

接口
----
- preprocess_fit_transform(long_df, cfg, meta) -> (df_out, state)
- preprocess_transform(long_df, cfg, state, meta) -> (df_out, state_out)

设计要点
--------
1) 训练和预测分别设计预处理模式：fit / transform ：
   - fit：记录统计量（如 mean/median、global winsorize 阈值、global zscore 均值方差等），并写入 state
   - transform：尽量复用 fit state，保证线上推理与训练处理一致，避免数据泄露/漂移

2) 输出格式稳定：
   - 输出固定列顺序： [date, stockid, (optional y), base_feature_cols..., mask_cols...]
   - transform 时强制对齐 fit 的 base_feature_cols（缺的补 NaN，后续按 nan_policy 决定是否填充）

3) 解释性强：
   - state 会记录 drop 了哪些列/行、每列 NaN 比例、填充策略、winsorize/zscore 参数与统计量等，
     用于复盘与确保 transform 可复用。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================
# helpers
# =========================
def _render_placeholders(s: str, mapping: Dict[str, str]) -> str:
    """
    将字符串模板中的 {key} 占位符替换为 mapping 对应的值。

    典型用途：
        save_path / state_path 中可能包含 {runs_root}、{run_id}、{date_start} 等占位符，
        在落盘前统一渲染。

    参数:
        s (str): 包含占位符的模板字符串。
        mapping (Dict[str, str]): key->value 映射。

    返回:
        str: 渲染后的字符串。
    """
    out = str(s)
    for k, v in (mapping or {}).items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _infer_feature_cols(df: pd.DataFrame, date_col: str, stockid_col: str, label_col: str) -> List[str]:
    """
    自动推断“数值型特征列”。

    规则：
        - 排除 key 列：date_col, stockid_col
        - 排除 label 列：label_col（如果存在）
        - 仅保留 numeric dtype 的列（float/int/bool 等）

    注意：
        该推断主要用于“长表输入不显式指定 feature_cols”的场景。
        如果你后续需要强制特征集合一致，请在 fit 时记录到 state 并在 transform 时对齐。

    参数:
        df (pd.DataFrame): 输入长表
        date_col (str): 日期列名
        stockid_col (str): 股票ID列名
        label_col (str): 标签列名

    返回:
        List[str]: 推断出的特征列名列表
    """
    exclude = {date_col, stockid_col}
    if label_col in df.columns:
        exclude.add(label_col)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _basic_fix(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    最基础的数据清洗：
    1) 将 date_col 转为 datetime（errors='coerce' 会把非法日期变为 NaT）
    2) 将 +/-inf 替换为 NaN（便于后续缺失处理）

    参数:
        df (pd.DataFrame): 输入长表
        date_col (str): 日期列名

    返回:
        pd.DataFrame: 清洗后的副本
    """
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _drop_y_missing(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, int]:
    """
    在 fit 阶段丢弃 label 缺失的样本（避免训练目标为空/NaN）。

    transform 阶段通常不会做此操作：
        - 线上推理日一般没有 y（或 y 不完整），不应因为缺 y 把样本删掉

    参数:
        df (pd.DataFrame): 输入长表
        label_col (str): 标签列名

    返回:
        (pd.DataFrame, int):
            - out: 丢弃 y 缺失后的 df
            - dropped: 丢弃的行数
    """
    if label_col not in df.columns:
        return df, 0
    before = len(df)
    out = df.dropna(subset=[label_col])
    return out, before - len(out)


def _drop_bad_cols(
    df: pd.DataFrame, feature_cols: List[str], col_drop_threshold: float
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    在 fit 阶段按“列缺失率”剔除坏特征列。

    逻辑：
        nan_ratio_col = mean(isna) over rows
        drop_cols = {c | nan_ratio_col[c] > threshold}
        df.drop(columns=drop_cols)

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 候选特征列
        col_drop_threshold (float): 列缺失率阈值，(0~1)，超过则删除该列

    返回:
        (pd.DataFrame, List[str], pd.Series):
            - out: 删除坏列后的 df
            - drop_cols: 被删除的列名列表
            - nan_ratio_col: 每个特征列的缺失率（用于 state 复盘）
    """
    if not feature_cols:
        return df, [], pd.Series(dtype=float)
    nan_ratio_col = df[feature_cols].isna().mean()
    drop_cols = nan_ratio_col[nan_ratio_col > col_drop_threshold].index.tolist()
    out = df.drop(columns=drop_cols) if drop_cols else df
    return out, drop_cols, nan_ratio_col


def _drop_bad_rows(df: pd.DataFrame, feature_cols: List[str], row_drop_threshold: float) -> Tuple[pd.DataFrame, int]:
    """
    在 fit 阶段按“行缺失率”剔除坏样本行。

    逻辑：
        nan_ratio_row = mean(isna) over features
        drop rows where nan_ratio_row > threshold

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 用于计算行缺失率的特征列集合
        row_drop_threshold (float): 行缺失率阈值，超过则删除该行

    返回:
        (pd.DataFrame, int):
            - out: 删除坏行后的 df
            - dropped: 被删除的行数
    """
    if not feature_cols:
        return df, 0
    nan_ratio_row = df[feature_cols].isna().mean(axis=1)
    bad = nan_ratio_row > row_drop_threshold
    dropped = int(bad.sum())
    out = df.loc[~bad].copy()
    return out, dropped


def _add_missing_mask(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    为每个特征列追加一个 missing indicator（__miss）列。

    用途：
        - 对树模型/线性模型：显式把“缺失信息”作为特征输入
        - 对深度学习模型：也可以让模型知道缺失位置

    约定：
        mask 列名 = f"{feature}__miss"
        mask 值：1 表示缺失，0 表示不缺失（float32）

    注意：
        mask 在 fill 之前生成，因此它记录的是“原始缺失情况”，不会被填充影响。

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 需要生成 mask 的特征列

    返回:
        (pd.DataFrame, List[str]):
            - out: 添加 mask 列后的 df
            - mask_cols: 新增的 mask 列名列表
    """
    if not feature_cols:
        return df, []
    out = df.copy()
    mask_cols: List[str] = []
    for c in feature_cols:
        mc = f"{c}__miss"
        out[mc] = out[c].isna().astype(np.float32)
        mask_cols.append(mc)
    return out, mask_cols


def _fill_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str,
    stockid_col: str,
    *,
    fill_method: str,
    ffill_limit: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    对特征列进行缺失填充。

    支持策略:
        - zero: 全部 NaN -> 0
        - mean: NaN -> 全局均值（会写入 state）
        - median: NaN -> 全局中位数（会写入 state）
        - ffill_then_zero: 按 stockid 分组 forward fill（限制步数）后仍缺 -> 0
        - ffill_all: 按 stockid 分组无限制 forward fill（可能保留开头 NaN）

    说明：
        - 输出会先按 [stockid, date] 排序，以保证 ffill 的时间方向正确。
        - strict 模式不允许最终留 NaN，因此通常不允许 ffill_all。

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 要填充的特征列
        date_col (str): 日期列名（用于排序）
        stockid_col (str): 股票ID列名（用于分组）
        fill_method (str): 填充策略名
        ffill_limit (int): ffill_then_zero 时的最大向前填充步数

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - out: 填充后的 df
            - state: 记录填充步骤与统计量（mean/median 等）
    """
    out = df.sort_values([stockid_col, date_col]).copy()
    state: Dict[str, Any] = {"fill_method": fill_method}

    if not feature_cols:
        state["fill_steps"] = []
        return out, state

    if fill_method == "zero":
        out[feature_cols] = out[feature_cols].fillna(0.0)
        state["fill_steps"] = ["fill0"]
        return out, state

    if fill_method == "mean":
        means = out[feature_cols].mean(numeric_only=True)
        out[feature_cols] = out[feature_cols].fillna(means)
        state["fill_steps"] = ["fill_mean"]
        state["fill_stats"] = {"mean": means.to_dict()}
        return out, state

    if fill_method == "median":
        meds = out[feature_cols].median(numeric_only=True)
        out[feature_cols] = out[feature_cols].fillna(meds)
        state["fill_steps"] = ["fill_median"]
        state["fill_stats"] = {"median": meds.to_dict()}
        return out, state

    if fill_method == "ffill_then_zero":
        out[feature_cols] = out.groupby(stockid_col, sort=False)[feature_cols].ffill(limit=ffill_limit)
        out[feature_cols] = out[feature_cols].fillna(0.0)
        state.update({"ffill_limit": ffill_limit, "fill_steps": ["ffill_by_stock", "fill0"]})
        return out, state

    if fill_method == "ffill_all":
        out[feature_cols] = out.groupby(stockid_col, sort=False)[feature_cols].ffill()
        state["fill_steps"] = ["ffill_by_stock_all"]
        return out, state

    raise ValueError(f"unknown fill_method: {fill_method}")


def _winsorize(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    by: str,
    date_col: str,
    lower_q: float,
    upper_q: float,
    state_stats: Optional[Dict[str, Any]] = None,  # for global reuse
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    对特征列进行 winsorize（分位裁剪去极值）。

    支持两种方式：
        - by="global": 用全样本分位数作为 clip 上下界
            * fit 时计算 lo/hi 并写入 state
            * transform 时可从 state_stats 复用 lo/hi，保证线上一致
        - by="date": 按每个 date 分组计算分位数并裁剪（横截面去极值）
            * 注意：transform 时也会按当日分布重新算（属于“按日自适应”）

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 需要 winsorize 的特征列
        by (str): "global" 或 "date"
        date_col (str): by="date" 时所需的日期列名
        lower_q (float): 下分位点（0~1）
        upper_q (float): 上分位点（0~1）
        state_stats (Optional[Dict[str, Any]]):
            - 仅 by="global" 在 transform 时使用
            - 期望形如 {"global_lo": {...}, "global_hi": {...}}

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - out: 裁剪后的 df
            - st: winsorize 配置与统计量信息（可合并进总 state）
    """
    out = df.copy()
    st: Dict[str, Any] = {
        "winsorize": {"enabled": True, "by": by, "lower_q": float(lower_q), "upper_q": float(upper_q)}
    }
    if not feature_cols:
        return out, st

    lower_q = float(lower_q)
    upper_q = float(upper_q)
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(f"winsorize quantiles invalid: lower_q={lower_q}, upper_q={upper_q}")

    if by == "global":
        # transform 阶段优先复用 fit 阶段统计量，避免线上重新计算导致分布漂移
        if (
            isinstance(state_stats, dict)
            and isinstance(state_stats.get("global_lo"), dict)
            and isinstance(state_stats.get("global_hi"), dict)
        ):
            lo = pd.Series(state_stats["global_lo"]).reindex(feature_cols)
            hi = pd.Series(state_stats["global_hi"]).reindex(feature_cols)
            st["winsorize"]["stats"] = {
                "global_lo": lo.to_dict(),
                "global_hi": hi.to_dict(),
                "from_state": True,
            }
        else:
            lo = out[feature_cols].quantile(lower_q)
            hi = out[feature_cols].quantile(upper_q)
            st["winsorize"]["stats"] = {
                "global_lo": lo.to_dict(),
                "global_hi": hi.to_dict(),
                "from_state": False,
            }

        out[feature_cols] = out[feature_cols].clip(lo, hi, axis=1)
        return out, st

    if by == "date":
        if date_col not in out.columns:
            raise KeyError(f"winsorize.by='date' needs date_col='{date_col}' in df")
        # 逐列按 date 横截面计算分位数并裁剪；写法清晰但大特征数时可能较慢
        for c in feature_cols:
            g = out.groupby(date_col, sort=False)[c]
            lo = g.transform(lambda s: s.quantile(lower_q))
            hi = g.transform(lambda s: s.quantile(upper_q))
            out[c] = out[c].clip(lo, hi)
        return out, st

    raise ValueError(f"winsorize.by must be 'date' or 'global', got: {by}")


def _zscore(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    by: str,
    date_col: str,
    method: str = "standard",
    ddof: int,
    clip: Optional[float],
    robust_cfg: Optional[Dict[str, Any]] = None,
    state_stats: Optional[Dict[str, Any]] = None,  # for global reuse
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    对特征列做 z-score 标准化。

    支持两种方式：
        - by="global": 用全样本均值/标准差标准化
            * fit: 计算 mu/sd 写入 state
            * transform: 优先从 state_stats 复用 mu/sd，保证线上一致
        - by="date": 按每个 date 横截面计算 mu/sd（横截面标准化）

    额外处理：
        - sd=0 时视为无信息（置 NaN），最后 fillna(0) 处理
        - 可选 clip：把 z-score 裁剪到 [-clip, clip]，避免极端值影响模型

    参数:
        df (pd.DataFrame): 输入长表
        feature_cols (List[str]): 需要标准化的特征列
        by (str): "global" 或 "date"
        date_col (str): by="date" 时所需的日期列名
        ddof (int): 标准差的 ddof（0 或 1 常见）
        clip (Optional[float]): 若不为 None，对 z-score 结果做裁剪
        state_stats (Optional[Dict[str, Any]]):
            - 仅 by="global" 在 transform 时使用
            - 期望形如 {"mu": {...}, "sd": {...}}

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - out: 标准化后的 df
            - st: zscore 配置与统计量信息（可合并进总 state）
    """
    out = df.copy()
    method = str(method or "standard").lower()
    if method not in ("standard", "robust"):
        raise ValueError(f"zscore.method must be 'standard' or 'robust', got: {method}")

    robust_cfg = robust_cfg or {}
    robust_eps = float(robust_cfg.get("eps", 1.0e-12))
    robust_use_06745 = bool(robust_cfg.get("use_06745", False))
    scale = 0.6745 if robust_use_06745 else 1.0

    st: Dict[str, Any] = {
        "zscore": {
            "enabled": True,
            "by": by,
            "method": method,
            "ddof": int(ddof),
            "clip": clip,
            "robust": {"eps": robust_eps, "use_06745": robust_use_06745},
        }
    }

    if not feature_cols:
        return out, st

    if by == "global":
        if method == "standard":
            # transform 阶段优先复用 fit 阶段统计量
            if (
                isinstance(state_stats, dict)
                and isinstance(state_stats.get("mu"), dict)
                and isinstance(state_stats.get("sd"), dict)
            ):
                mu = pd.Series(state_stats["mu"]).reindex(feature_cols)
                sd = pd.Series(state_stats["sd"]).reindex(feature_cols)
                st["zscore"]["stats"] = {"mu": mu.to_dict(), "sd": sd.to_dict(), "from_state": True}
            else:
                mu = out[feature_cols].mean(numeric_only=True)
                sd = out[feature_cols].std(ddof=int(ddof), numeric_only=True)
                st["zscore"]["stats"] = {"mu": mu.to_dict(), "sd": sd.to_dict(), "from_state": False}

            sd = sd.replace(0.0, np.nan)
            out[feature_cols] = (out[feature_cols] - mu) / sd
            out[feature_cols] = out[feature_cols].fillna(0.0)

        else:
            # robust: use median + MAD (median absolute deviation)
            if (
                isinstance(state_stats, dict)
                and isinstance(state_stats.get("median"), dict)
                and isinstance(state_stats.get("mad"), dict)
            ):
                med = pd.Series(state_stats["median"]).reindex(feature_cols)
                mad = pd.Series(state_stats["mad"]).reindex(feature_cols)
                st["zscore"]["stats"] = {"median": med.to_dict(), "mad": mad.to_dict(), "from_state": True}
            else:
                med = out[feature_cols].median(numeric_only=True)
                mad = (out[feature_cols] - med).abs().median(numeric_only=True)
                st["zscore"]["stats"] = {"median": med.to_dict(), "mad": mad.to_dict(), "from_state": False}

            denom = mad + robust_eps
            denom = denom.replace(0.0, np.nan)
            out[feature_cols] = ((out[feature_cols] - med) / denom) * scale
            out[feature_cols] = out[feature_cols].fillna(0.0)

    elif by == "date":
        if date_col not in out.columns:
            raise KeyError(f"zscore.by='date' needs date_col='{date_col}' in df")
        if method == "standard":
            # 逐列横截面标准化；同样是清晰优先，特征很大时可考虑向量化优化
            for c in feature_cols:
                g = out.groupby(date_col, sort=False)[c]
                mu = g.transform("mean")
                sd = g.transform(lambda s: s.std(ddof=int(ddof)))
                sd = sd.replace(0.0, np.nan)
                out[c] = (out[c] - mu) / sd
                out[c] = out[c].fillna(0.0)
        else:
            for c in feature_cols:
                g = out.groupby(date_col, sort=False)[c]
                med = g.transform("median")
                mad = g.transform(lambda s: (s - s.median()).abs().median())
                denom = mad + robust_eps
                denom = denom.replace(0.0, np.nan)
                out[c] = ((out[c] - med) / denom) * scale
                out[c] = out[c].fillna(0.0)

    else:
        raise ValueError(f"zscore.by must be 'date' or 'global', got: {by}")

    if clip is not None:
        zc = float(clip)
        out[feature_cols] = out[feature_cols].clip(-zc, zc)

    return out, st


def _validate(df: pd.DataFrame, date_col: str, stockid_col: str, feature_cols: List[str], nan_policy: str) -> None:
    """
    对 preprocess 输出做一致性校验（失败直接抛异常），避免 silent bug。

    校验项：
        1) 必须包含 key 列：date_col / stockid_col
        2) date_col 不允许存在 NaT（通常意味着 date 解析失败）
        3) (date, stockid) 不允许重复（否则训练/评估会发生样本重复）
        4) strict 模式下特征列不允许存在任何 NaN

    参数:
        df (pd.DataFrame): preprocess 输出
        date_col (str): 日期列名
        stockid_col (str): 股票ID列名
        feature_cols (List[str]): 特征列名列表
        nan_policy (str): "strict" / "tree_friendly"
    """
    if date_col not in df.columns or stockid_col not in df.columns:
        raise ValueError(f"preprocess output must contain '{date_col}' and '{stockid_col}'")

    if df[date_col].isna().any():
        bad = int(df[date_col].isna().sum())
        raise ValueError(f"date_col '{date_col}' has {bad} NaT/NaN after to_datetime(coerce)")

    dup = int(df.duplicated([date_col, stockid_col]).sum())
    if dup > 0:
        raise ValueError(f"duplicate (date, stockid) after preprocess: {dup}")

    if nan_policy == "strict" and feature_cols:
        remain = int(df[feature_cols].isna().sum().sum())
        if remain != 0:
            raise ValueError(f"strict nan_policy requires no NaN in features, found {remain}")


def _json_friendly(x: Any) -> Any:
    """
    将 numpy 标量转换为可 JSON 序列化的 python 原生类型。

    用途：
        state 落 JSON 时，避免 np.int64/np.float64 直接 dump 报错。

    参数:
        x (Any): 任意对象

    返回:
        Any: 可 JSON 序列化的对象（尽量转为 int/float/str）
    """
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


# =========================
# public API (fit/transform)
# =========================
def preprocess_fit_transform(
    long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    训练阶段使用：fit + transform 一次完成。

    参数:
        long_df (pd.DataFrame): 输入长表
        cfg (Dict[str, Any]): 预处理配置（见 _preprocess_core 内部读取的键）
        meta (Optional[Dict[str, Any]]): 路径渲染与日志信息（runs_root/run_id/date_start/date_end 等）

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - df_out: 预处理后的输出长表（列顺序固定）
            - state: 预处理统计与可复用统计量（供 transform 复用）
    """
    return _preprocess_core(long_df, cfg, meta=meta, mode="fit")


def preprocess_transform(
    long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    state: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    推理阶段使用：仅 transform（复用 fit 阶段 state）。

    关键点：
        - 会优先对齐 state["base_feature_cols"]：
          * 输入缺的列会补 NaN
          * 输出列顺序固定，保证模型输入维度一致

    参数:
        long_df (pd.DataFrame): 输入长表
        cfg (Dict[str, Any]): 预处理配置（应与 fit 阶段一致）
        state (Dict[str, Any]): fit 阶段产出的 state（包含特征集合与统计量）
        meta (Optional[Dict[str, Any]]): 路径渲染与日志信息

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - df_out: transform 后输出
            - state_out: 当前 transform 的统计（也会带上复用信息）
    """
    return _preprocess_core(long_df, cfg, meta=meta, mode="transform", state=state)


def preprocess_long(
    long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    兼容旧接口：训练使用时等价于 preprocess_fit_transform。

    历史原因：
        早期代码可能只调用 preprocess_long()，这里保留不破坏现有调用链。

    参数/返回 同 preprocess_fit_transform。
    """
    return preprocess_fit_transform(long_df, cfg, meta=meta)


# =========================
# core
# =========================
def _preprocess_core(
    long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]],
    mode: str,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    预处理核心逻辑（fit/transform 共用）。

    流程概览（按执行顺序）：
        0) basic fix: date->datetime, inf->NaN
        1) fit only: drop y missing
        2) infer base_feature_cols
        3) fit only: drop bad cols/rows
           transform: 对齐 fit_base_cols（缺的补 NaN）
        4) optional: add missing masks（在 fill 前）
        5) fill NaN（受 nan_policy 控制）
        6) optional: winsorize + zscore（只处理 base features，不处理 mask）
        7) finalize: 固定列顺序 + 校验
        8) build out_state（包含统计量、可复用统计、cfg_echo）
        9) fit only: 可选保存 parquet / state json

    参数:
        long_df (pd.DataFrame): 输入长表
        cfg (Dict[str, Any]): 配置
        meta (Optional[Dict[str, Any]]): 元信息（用于日志与路径渲染）
        mode (str): "fit" 或 "transform"
        state (Optional[Dict[str, Any]]): transform 时的 fit state

    返回:
        (pd.DataFrame, Dict[str, Any]):
            - df_out: 输出长表
            - out_state: 本次运行统计 +（fit 时）可复用统计量
    """
    meta = meta or {}
    state = state or {}

    # -------------------------
    # A) read config
    # -------------------------
    date_col = cfg.get("date_col", "date")
    stockid_col = cfg.get("stockid_col", "stockid")
    label_col = cfg.get("label_col", "y")

    # 缺失率过滤（fit only）
    col_drop_threshold = float(cfg.get("col_drop_threshold", 0.20))
    row_drop_threshold = float(cfg.get("row_drop_threshold", 0.20))

    # nan_policy 控制：
    #   - strict: 最终特征不得含 NaN（通常用于 NN/线性模型等）
    #   - tree_friendly: 允许保留 NaN（树模型可原生处理 NaN），或按 do_fill_for_tree 决定是否填充
    nan_policy = str(cfg.get("nan_policy", "strict"))
    fill_method = str(cfg.get("fill_method", "ffill_then_zero"))
    ffill_limit = int(cfg.get("ffill_limit", 5))
    do_fill_for_tree = bool(cfg.get("do_fill_for_tree", False))

    add_missing_mask = bool(cfg.get("add_missing_mask", False))

    # winsorize / zscore 配置
    wins_cfg = (cfg.get("winsorize", {}) or {})
    win_enabled = bool(wins_cfg.get("enabled", False))
    win_by = str(wins_cfg.get("by", "date"))
    win_lq = float(wins_cfg.get("lower_q", 0.01))
    win_uq = float(wins_cfg.get("upper_q", 0.99))

    z_cfg = (cfg.get("zscore", {}) or {})
    z_enabled = bool(z_cfg.get("enabled", False))
    z_by = str(z_cfg.get("by", "date"))
    z_method = str(z_cfg.get("method", "standard"))
    z_ddof = int(z_cfg.get("ddof", 0))
    z_clip = z_cfg.get("clip", None)
    z_clip = None if z_clip is None else float(z_clip)
    z_robust = (z_cfg.get("robust", {}) or {})

    # outputs
    save_parquet = bool(cfg.get("save_parquet", False))
    save_path_tpl = str(cfg.get("save_path", "./cache/preprocessed_{label}_{date_start}_{date_end}.parquet"))

    save_state_json = bool(cfg.get("save_state_json", False))
    state_path_tpl = str(cfg.get("state_path", "./cache/preprocess_state_{mode}_{label}_{date_start}_{date_end}.json"))

    logger.info(
        "preprocess start | mode=%s | in_shape=%s | date_col=%s | stockid_col=%s | label_col=%s | nan_policy=%s | fill=%s",
        mode, getattr(long_df, "shape", None), date_col, stockid_col, label_col, nan_policy, fill_method
    )

    # -------------------------
    # 0) basic fix
    # -------------------------
    df0 = _basic_fix(long_df, date_col)

    # -------------------------
    # 1) drop y missing (fit only)
    # -------------------------
    dropped_y = 0
    df1 = df0
    if mode == "fit":
        df1, dropped_y = _drop_y_missing(df0, label_col)

    # -------------------------
    # 2) infer base feature cols
    # -------------------------
    base_feature_cols = _infer_feature_cols(df1, date_col, stockid_col, label_col)

    # -------------------------
    # 3) drop bad cols/rows (fit only)
    #    transform: enforce fit feature columns & column order
    # -------------------------
    drop_cols: List[str] = []
    nan_ratio_col = pd.Series(dtype=float)
    dropped_rows = 0

    df2 = df1
    if mode == "fit":
        # 先删坏列，再重新推断一次（避免 label/key 外的列结构变化）
        df2, drop_cols, nan_ratio_col = _drop_bad_cols(df1, base_feature_cols, col_drop_threshold)
        base_feature_cols = _infer_feature_cols(df2, date_col, stockid_col, label_col)
        df3, dropped_rows = _drop_bad_rows(df2, base_feature_cols, row_drop_threshold)
    else:
        # transform：尽量对齐 fit 阶段的 base_feature_cols（维度一致非常关键）
        df3 = df2.copy()
        fit_base_cols = state.get("base_feature_cols")
        if isinstance(fit_base_cols, list) and len(fit_base_cols) > 0:
            # 推理日可能缺少某些特征列：补 NaN，后续由 fill/nan_policy 决定如何处理
            for c in fit_base_cols:
                if c not in df3.columns:
                    df3[c] = np.nan
            base_feature_cols = list(fit_base_cols)
        else:
            # 如果 state 不可用，就退化为“当前数据推断”
            base_feature_cols = _infer_feature_cols(df3, date_col, stockid_col, label_col)

    # -------------------------
    # 4) missing mask (before fill)
    # -------------------------
    mask_cols: List[str] = []
    df4 = df3
    if add_missing_mask:
        df4, mask_cols = _add_missing_mask(df3, base_feature_cols)

    # -------------------------
    # 5) fill
    # -------------------------
    df5 = df4
    fill_state: Dict[str, Any] = {}

    if nan_policy == "strict":
        # strict 要求最终无 NaN，因此不允许 ffill_all（可能保留开头 NaN）
        if fill_method == "ffill_all":
            raise ValueError("nan_policy='strict' cannot use fill_method='ffill_all' (may keep leading NaN)")

        # transform 时 mean/median 优先复用 fit 的统计量（避免线上重新估计）
        if mode == "transform" and fill_method in ("mean", "median"):
            fill_stats = (state.get("fill_stats") or {})
            stats_key = "mean" if fill_method == "mean" else "median"
            if isinstance(fill_stats, dict) and stats_key in fill_stats:
                vals = fill_stats[stats_key] or {}
                df5 = df4.sort_values([stockid_col, date_col]).copy()
                df5[base_feature_cols] = df5[base_feature_cols].fillna(pd.Series(vals))
                fill_state = {
                    "fill_method": fill_method,
                    "fill_steps": [f"fill_{stats_key}_from_state"],
                    "fill_stats": {stats_key: vals},
                }
            else:
                df5, fill_state = _fill_features(
                    df4, base_feature_cols, date_col, stockid_col, fill_method=fill_method, ffill_limit=ffill_limit
                )
        else:
            df5, fill_state = _fill_features(
                df4, base_feature_cols, date_col, stockid_col, fill_method=fill_method, ffill_limit=ffill_limit
            )

    elif nan_policy == "tree_friendly":
        # tree_friendly 允许保留 NaN 给树模型处理（比如 LightGBM/CatBoost 可原生处理 NaN）
        # 也可以通过 do_fill_for_tree 强制做填充（当你希望线上/线下统一、或模型不接受 NaN 时）
        if do_fill_for_tree:
            df5, fill_state = _fill_features(
                df4, base_feature_cols, date_col, stockid_col, fill_method=fill_method, ffill_limit=ffill_limit
            )
        else:
            df5 = df4
            fill_state = {"fill_method": "none", "fill_steps": []}

    else:
        raise ValueError(f"unknown nan_policy: {nan_policy}")

    # -------------------------
    # 6) winsorize + zscore  (base features only; masks untouched)
    #    常见顺序：winsorize -> zscore
    # -------------------------
    df6 = df5
    trans_state: Dict[str, Any] = {}

    if win_enabled:
        # global 模式下 transform 尽量复用 fit 阶段阈值，避免漂移
        win_state_stats = None
        if mode == "transform" and win_by == "global":
            w = state.get("winsorize")
            if isinstance(w, dict):
                win_state_stats = w.get("stats")
        df6, st = _winsorize(
            df6,
            base_feature_cols,
            by=win_by,
            date_col=date_col,
            lower_q=win_lq,
            upper_q=win_uq,
            state_stats=win_state_stats,
        )
        trans_state.update(st)

    if z_enabled:
        # global 模式下 transform 尽量复用 fit 阶段均值方差
        z_state_stats = None
        if mode == "transform" and z_by == "global":
            z = state.get("zscore")
            if isinstance(z, dict):
                z_state_stats = z.get("stats")
        df6, st = _zscore(
            df6,
            base_feature_cols,
            by=z_by,
            date_col=date_col,
            method=z_method,
            ddof=z_ddof,
            clip=z_clip,
            robust_cfg=z_robust,
            state_stats=z_state_stats,
        )
        trans_state.update(st)

    # -------------------------
    # 7) finalize (固定列顺序；transform 也保持一致)
    # -------------------------
    feature_cols = list(base_feature_cols) + list(mask_cols)

    final_cols = [date_col, stockid_col]
    if label_col in df6.columns:
        final_cols.append(label_col)
    final_cols += list(base_feature_cols) + list(mask_cols)

    # key col 必须存在（否则整个 pipeline 的 join/split 会崩）
    for k in [date_col, stockid_col]:
        if k not in df6.columns:
            raise ValueError(f"preprocess output missing key col: {k}")

    df_out = (
        df6.sort_values([date_col, stockid_col])
        .reset_index(drop=True)
        # 防御式：如果某些列不存在（例如 label 不存在），只取存在的列
        .loc[:, [c for c in final_cols if c in df6.columns]]
    )

    _validate(df_out, date_col, stockid_col, feature_cols, nan_policy)

    # -------------------------
    # 8) build state
    # -------------------------
    # out_state 用于：
    #   - 日志/复盘：本次 preprocess 做了什么
    #   - transform 复用：基于 fit 的统计量与特征集合，保证线上一致
    out_state: Dict[str, Any] = {
        "mode": mode,
        "n_in": int(len(df0)),
        "n_out": int(len(df_out)),
        "dropped_y_rows": int(dropped_y),
        "dropped_rows_by_nan_ratio": int(dropped_rows),
        "dropped_feature_cols": drop_cols,
        "nan_ratio_col": nan_ratio_col.to_dict() if len(nan_ratio_col) else {},
        "nan_policy": nan_policy,
        "base_feature_cols": list(base_feature_cols),
        "mask_cols": list(mask_cols),
        "feature_cols": list(feature_cols),
        **fill_state,
        **trans_state,
        "cfg_echo": {
            "col_drop_threshold": col_drop_threshold,
            "row_drop_threshold": row_drop_threshold,
            "add_missing_mask": add_missing_mask,
            "do_fill_for_tree": do_fill_for_tree,
            "fill_method": fill_method,
            "ffill_limit": ffill_limit,
            "winsorize": dict(wins_cfg),
            "zscore": dict(z_cfg),
        },
    }

    # -------------------------
    # 9) save parquet / state (fit only)
    # -------------------------
    # 落盘路径支持 meta 占位符渲染（通常由 main/runner 注入）
    mapping = {
        "runs_root": str(meta.get("runs_root", "")),
        "run_id": str(meta.get("run_id", meta.get("exp_id", ""))),
        "mode": str(mode),
        "label": str(label_col),
        "date_start": str(meta.get("date_start", ""))[:10],
        "date_end": str(meta.get("date_end", ""))[:10],
    }

    if save_parquet and mode == "fit":
        path = _render_placeholders(save_path_tpl, mapping)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(path, index=False)
        out_state["saved_path"] = str(path)
        logger.info("preprocess parquet saved | path=%s | shape=%s", path, df_out.shape)

    if save_state_json and mode == "fit":
        spath = _render_placeholders(state_path_tpl, mapping)
        Path(spath).parent.mkdir(parents=True, exist_ok=True)
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(out_state, f, ensure_ascii=False, indent=2, default=_json_friendly)
        out_state["saved_state_path"] = str(spath)
        logger.info("preprocess state saved | path=%s", spath)

    logger.info(
        "preprocess end | mode=%s | out_shape=%s | n_base_feats=%d | n_mask_feats=%d",
        mode, df_out.shape, len(base_feature_cols), len(mask_cols)
    )

    return df_out, out_state
