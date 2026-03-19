"""
ret_pred/trainer/sample_weighting.py

Build configurable per-sample weights for training payloads.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ones(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=np.float32)


def _build_recency_weights(
    *,
    keys: Optional[pd.DataFrame],
    date_col: str,
    cfg: Dict[str, Any],
    n: int,
) -> np.ndarray:
    if not bool(cfg.get("enabled", False)):
        return _ones(n)
    if keys is None or date_col not in keys.columns:
        logger.warning("sample_weighting.recency enabled but keys/date_col missing; fallback to ones")
        return _ones(n)

    d = pd.to_datetime(keys[date_col], errors="coerce")
    if d.notna().sum() == 0:
        return _ones(n)

    ref = d.max()
    age_days = (ref - d).dt.days.astype(float).to_numpy()
    age_days = np.where(np.isfinite(age_days), np.maximum(age_days, 0.0), np.nan)

    mode = str(cfg.get("mode", "exp")).lower()
    if mode == "exp":
        half_life = float(cfg.get("half_life_days", 63.0))
        if half_life <= 0:
            raise ValueError(f"sample_weighting.recency.half_life_days must be > 0, got: {half_life}")
        core = np.exp(-np.log(2.0) * (age_days / half_life))
    elif mode == "linear":
        max_age = np.nanmax(age_days) if np.any(np.isfinite(age_days)) else 0.0
        if max_age <= 0:
            core = np.ones_like(age_days, dtype=float)
        else:
            core = 1.0 - (age_days / max_age)
            core = np.clip(core, 0.0, 1.0)
    else:
        raise ValueError(f"sample_weighting.recency.mode must be exp|linear, got: {mode}")

    min_w = float(cfg.get("min_weight", 0.1))
    max_w = float(cfg.get("max_weight", 1.0))
    if min_w <= 0 or max_w <= 0 or min_w > max_w:
        raise ValueError(f"sample_weighting.recency min/max invalid: min={min_w}, max={max_w}")

    out = min_w + (max_w - min_w) * core
    out = np.where(np.isfinite(out), out, min_w)
    return out.astype(np.float32, copy=False)


def _build_tail_weights(
    *,
    y: np.ndarray,
    keys: Optional[pd.DataFrame],
    date_col: str,
    cfg: Dict[str, Any],
) -> np.ndarray:
    n = int(len(y))
    if not bool(cfg.get("enabled", False)):
        return _ones(n)

    lower_q = float(cfg.get("lower_q", 0.1))
    upper_q = float(cfg.get("upper_q", 0.9))
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError(f"sample_weighting.tail_y quantiles invalid: lower_q={lower_q}, upper_q={upper_q}")

    tail_weight = float(cfg.get("tail_weight", 2.0))
    mid_weight = float(cfg.get("mid_weight", 1.0))
    if tail_weight <= 0 or mid_weight <= 0:
        raise ValueError("sample_weighting.tail_y weights must be > 0")

    side = str(cfg.get("side", "both")).lower()
    if side not in {"both", "upper", "lower"}:
        raise ValueError(f"sample_weighting.tail_y.side must be both|upper|lower, got: {side}")

    scope = str(cfg.get("scope", "global")).lower()
    mask = np.zeros(n, dtype=bool)

    if scope == "global":
        yv = np.asarray(y, dtype=float)
        lq = np.nanquantile(yv, lower_q)
        uq = np.nanquantile(yv, upper_q)
        if side in {"both", "lower"}:
            mask |= (yv <= lq)
        if side in {"both", "upper"}:
            mask |= (yv >= uq)
    elif scope == "date":
        if keys is None or date_col not in keys.columns:
            logger.warning("sample_weighting.tail_y scope=date but keys/date_col missing; fallback to global")
            return _build_tail_weights(y=y, keys=None, date_col=date_col, cfg={**cfg, "scope": "global"})
        tmp = pd.DataFrame(
            {
                "__y": pd.to_numeric(pd.Series(y), errors="coerce"),
                "__d": pd.to_datetime(keys[date_col], errors="coerce"),
            }
        )
        g = tmp.groupby("__d", sort=False)["__y"]
        lq = g.transform(lambda s: s.quantile(lower_q))
        uq = g.transform(lambda s: s.quantile(upper_q))
        if side in {"both", "lower"}:
            mask |= (tmp["__y"].to_numpy() <= lq.to_numpy())
        if side in {"both", "upper"}:
            mask |= (tmp["__y"].to_numpy() >= uq.to_numpy())
    else:
        raise ValueError(f"sample_weighting.tail_y.scope must be global|date, got: {scope}")

    out = np.full(n, mid_weight, dtype=np.float32)
    out[mask] = np.float32(tail_weight)
    return out


def build_sample_weights(
    *,
    payload: Dict[str, Any],
    cfg: Optional[Dict[str, Any]],
    date_col: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build sample weights for one payload.

    Returns:
      (weights_or_none, state)
    """
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None, {"enabled": False}

    y = np.asarray(payload.get("y", []), dtype=float).reshape(-1)
    n = int(len(y))
    if n == 0:
        return None, {"enabled": True, "n": 0}

    keys = payload.get("keys", None)
    if keys is not None and not isinstance(keys, pd.DataFrame):
        keys = None

    recency_cfg = dict(cfg.get("recency", {}) or {})
    tail_cfg = dict(cfg.get("tail_y", {}) or {})

    w_rec = _build_recency_weights(keys=keys, date_col=date_col, cfg=recency_cfg, n=n)
    w_tail = _build_tail_weights(y=y, keys=keys, date_col=date_col, cfg=tail_cfg)

    combine = str(cfg.get("combine", "multiply")).lower()
    if combine == "multiply":
        w = w_rec * w_tail
    elif combine == "add":
        w = w_rec + w_tail
    elif combine == "max":
        w = np.maximum(w_rec, w_tail)
    else:
        raise ValueError(f"sample_weighting.combine must be multiply|add|max, got: {combine}")

    # keep NaN labels at neutral weight to avoid NaN in fit(sample_weight=...)
    w = np.where(np.isfinite(y), w, 1.0)

    normalize = str(cfg.get("normalize", "mean1")).lower()
    if normalize == "mean1":
        m = float(np.nanmean(w))
        if np.isfinite(m) and m > 0:
            w = w / m
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"sample_weighting.normalize must be mean1|none, got: {normalize}")

    w_min = cfg.get("clip_min", None)
    w_max = cfg.get("clip_max", None)
    if w_min is not None or w_max is not None:
        lo = None if w_min is None else float(w_min)
        hi = None if w_max is None else float(w_max)
        w = np.clip(w, lo, hi)

    w = np.asarray(w, dtype=np.float32).reshape(-1)

    st = {
        "enabled": True,
        "n": int(n),
        "recency_enabled": bool(recency_cfg.get("enabled", False)),
        "tail_enabled": bool(tail_cfg.get("enabled", False)),
        "combine": combine,
        "normalize": normalize,
        "w_min": float(np.nanmin(w)),
        "w_max": float(np.nanmax(w)),
        "w_mean": float(np.nanmean(w)),
    }
    return w, st

