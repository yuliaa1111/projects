# 各种metrics的计算
# 回归任务（收益率预测）常用：mse/rmse/mae/r2 + rankic/icir；分类任务常用：acc/precision/recall/f1/auc/ks
# RankIC 的时间序列；按预测分位做分位均值收益、分位累计收益

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Helpers
# =========================

def _to_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


# =========================
# Regression metrics
# =========================

def mse(y_true, y_pred) -> float:
    yt = _to_1d(y_true); yp = _to_1d(y_pred)
    return float(np.mean((yt - yp) ** 2)) if yt.size else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    yt = _to_1d(y_true); yp = _to_1d(y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else 0.0


def r2(y_true, y_pred) -> float:
    yt = _to_1d(y_true); yp = _to_1d(y_pred)
    if yt.size == 0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def rankic_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    method: str = "spearman",
) -> float:

    if df.empty:
        return 0.0
    if date_col not in df.columns:
        c = df[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        return float(c) if pd.notna(c) else 0.0

    ics: List[float] = []
    for _, g in df.groupby(date_col):
        if len(g) < 2:
            continue
        c = g[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        if pd.notna(c):
            ics.append(float(c))
    return float(np.mean(ics)) if ics else 0.0


def icir_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    method: str = "spearman",
) -> float:

    if df.empty or date_col not in df.columns:
        return 0.0
    ics: List[float] = []
    for _, g in df.groupby(date_col):
        if len(g) < 2:
            continue
        c = g[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        if pd.notna(c):
            ics.append(float(c))
    if not ics:
        return 0.0
    mu = _safe_mean(np.array(ics))
    sd = _safe_std(np.array(ics))
    return float(mu / sd) if sd > 1e-12 else 0.0


def rankic_series_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    method: str = "spearman",
) -> pd.Series:

    if df.empty:
        return pd.Series(dtype=float)
    if date_col not in df.columns:
        # fallback: single value
        c = df[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        return pd.Series([float(c) if pd.notna(c) else 0.0])

    out = {}
    for dt, g in df.groupby(date_col):
        if len(g) < 2:
            continue
        c = g[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        if pd.notna(c):
            out[pd.to_datetime(dt)] = float(c)
    s = pd.Series(out).sort_index()
    s.name = "rankic"
    return s


def assign_pred_quantile(
    df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "y_pred",
    q_bins: int = 10,
    out_col: str = "quantile",
) -> pd.DataFrame:

    if df.empty:
        out = df.copy()
        out[out_col] = np.nan
        return out

    if date_col not in df.columns:
        # if no date, do global quantile
        out = df.copy()
        r = out[pred_col].rank(method="first")
        out[out_col] = pd.qcut(r, q_bins, labels=False, duplicates="drop")
        return out

    def _one_day(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        if len(gg) < q_bins:
            # too few names -> put all in middle-ish
            gg[out_col] = int(q_bins // 2)
            return gg
        r = gg[pred_col].rank(method="first")
        try:
            gg[out_col] = pd.qcut(r, q_bins, labels=False, duplicates="drop").astype(int)
        except ValueError:
            # if qcut fails, fallback to equal-width on rank percentile
            pct = (r - 1) / max(len(r) - 1, 1)
            gg[out_col] = np.minimum((pct * q_bins).astype(int), q_bins - 1)
        return gg

    out = df.groupby(date_col, group_keys=False).apply(_one_day, include_groups=False)
    return out


def quantile_daily_mean_return(
    df: pd.DataFrame,
    date_col: str = "date",
    true_col: str = "y_true",
    q_col: str = "quantile",
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame(columns=[date_col, q_col, "ret"])

    if date_col not in df.columns:
        # single-day style fallback
        tmp = df.groupby(q_col)[true_col].mean().reset_index()
        tmp[date_col] = pd.NaT
        tmp = tmp.rename(columns={true_col: "ret"})
        return tmp[[date_col, q_col, "ret"]]

    daily = (
        df.groupby([date_col, q_col])[true_col]
        .mean()
        .reset_index()
        .rename(columns={true_col: "ret"})
        .sort_values([date_col, q_col])
    )
    daily[date_col] = pd.to_datetime(daily[date_col])
    return daily


def quantile_mean_realized_return(
    df: pd.DataFrame,
    date_col: str = "date",
    true_col: str = "y_true",
    q_bins: int = 10,
    pred_col: str = "y_pred",
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame({"quantile": list(range(q_bins)), "mean_ret": [0.0] * q_bins})

    tmp = assign_pred_quantile(df, date_col=date_col, pred_col=pred_col, q_bins=q_bins, out_col="quantile")
    daily = quantile_daily_mean_return(tmp, date_col=date_col, true_col=true_col, q_col="quantile")

    # average across dates (each date equally weighted)
    mean_df = (
        daily.groupby("quantile")["ret"]
        .mean()
        .reindex(range(q_bins))
        .fillna(0.0)
        .reset_index()
        .rename(columns={"ret": "mean_ret"})
    )
    return mean_df


def quantile_cumulative_return(
    df: pd.DataFrame,
    date_col: str = "date",
    true_col: str = "y_true",
    q_bins: int = 10,
    pred_col: str = "y_pred",
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame({"quantile": list(range(q_bins)), "cum_return": [0.0] * q_bins})

    tmp = assign_pred_quantile(df, date_col=date_col, pred_col=pred_col, q_bins=q_bins, out_col="quantile")
    daily = quantile_daily_mean_return(tmp, date_col=date_col, true_col=true_col, q_col="quantile")

    out_rows = []
    for q, g in daily.groupby("quantile"):
        g = g.sort_values(date_col)
        r = g["ret"].to_numpy()
        cum = float(np.prod(1.0 + r) - 1.0) if len(r) else 0.0
        out_rows.append({"quantile": int(q), "cum_return": cum})

    out = pd.DataFrame(out_rows)
    out = out.set_index("quantile").reindex(range(q_bins)).fillna(0.0).reset_index()
    return out


# =========================
# Classification metrics (binary)
# =========================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _get_prob(df: pd.DataFrame) -> np.ndarray:
    if "y_prob" in df.columns:
        p = _to_1d(df["y_prob"].values)
        return np.clip(p, 0.0, 1.0)

    p = _to_1d(df["y_pred"].values)
    if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
        p = _sigmoid(p)
    return np.clip(p, 0.0, 1.0)


def confusion_counts(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[int, int, int, int]:
    yt = _to_1d(y_true).astype(int)
    yh = _to_1d(y_hat).astype(int)
    tp = int(np.sum((yt == 1) & (yh == 1)))
    tn = int(np.sum((yt == 0) & (yh == 0)))
    fp = int(np.sum((yt == 0) & (yh == 1)))
    fn = int(np.sum((yt == 1) & (yh == 0)))
    return tp, tn, fp, fn


def accuracy(y_true, y_hat) -> float:
    yt = _to_1d(y_true).astype(int)
    yh = _to_1d(y_hat).astype(int)
    return float(np.mean(yt == yh)) if yt.size else 0.0


def precision(y_true, y_hat) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_hat)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_hat) -> float:
    tp, tn, fp, fn = confusion_counts(y_true, y_hat)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_true, y_hat) -> float:
    p = precision(y_true, y_hat)
    r = recall(y_true, y_hat)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def auc_roc(y_true: np.ndarray, prob: np.ndarray) -> float:
    yt = _to_1d(y_true).astype(int)
    pr = _to_1d(prob)
    if yt.size == 0:
        return 0.0
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(pr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, yt.size + 1)

    sum_ranks_pos = float(np.sum(ranks[yt == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def ks_stat(y_true: np.ndarray, prob: np.ndarray) -> float:
    yt = _to_1d(y_true).astype(int)
    pr = _to_1d(prob)
    if yt.size == 0:
        return 0.0
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    idx = np.argsort(pr)
    yt_sorted = yt[idx]
    cum_pos = np.cumsum(yt_sorted == 1) / n_pos
    cum_neg = np.cumsum(yt_sorted == 0) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


# =========================
# Task wrappers
# =========================

@dataclass
class RegressionCfg:
    metrics: List[str]
    ic_method: str = "spearman"


@dataclass
class ClassificationCfg:
    metrics: List[str]
    threshold: float = 0.5


def evaluate_regression(df: pd.DataFrame, date_col: str, cfg: RegressionCfg) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in cfg.metrics:
        if m == "mse":
            out[m] = mse(df["y_true"], df["y_pred"])
        elif m == "rmse":
            out[m] = rmse(df["y_true"], df["y_pred"])
        elif m == "mae":
            out[m] = mae(df["y_true"], df["y_pred"])
        elif m == "r2":
            out[m] = r2(df["y_true"], df["y_pred"])
        elif m == "rankic":
            out[m] = rankic_by_date(df, date_col=date_col, method=cfg.ic_method)
        elif m == "icir":
            out[m] = icir_by_date(df, date_col=date_col, method=cfg.ic_method)
        else:
            raise KeyError(f"Unknown regression metric '{m}'")
    return out


def evaluate_classification(df: pd.DataFrame, cfg: ClassificationCfg) -> Dict[str, float]:
    out: Dict[str, float] = {}
    yt = _to_1d(df["y_true"]).astype(int)
    prob = _get_prob(df)
    y_hat = (prob >= float(cfg.threshold)).astype(int)

    for m in cfg.metrics:
        if m == "acc":
            out[m] = accuracy(yt, y_hat)
        elif m == "precision":
            out[m] = precision(yt, y_hat)
        elif m == "recall":
            out[m] = recall(yt, y_hat)
        elif m == "f1":
            out[m] = f1(yt, y_hat)
        elif m == "auc":
            out[m] = auc_roc(yt, prob)
        elif m == "ks":
            out[m] = ks_stat(yt, prob)
        else:
            raise KeyError(f"Unknown classification metric '{m}'")
    return out
