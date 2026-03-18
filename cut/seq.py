from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import os
import numpy as np
import pandas as pd

from .base import infer_feature_cols, ensure_dir, save_json


def cut_seq_long(
    part_long_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    meta = meta or {}
    date_col = cfg.get("date_col", "date")
    stockid_col = cfg.get("stockid_col", "stockid")
    label_col = cfg.get("label_col", "y")
    weight_col = cfg.get("weight_col", None)

    seq_len = cfg.get("seq_len", None)
    if seq_len is None:
        raise ValueError("cut_seq_long: cfg['seq_len'] is required")
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("cut_seq_long: seq_len must be > 0")

    # incomplete policy：drop | pad_zero | pad_repeat_first
    incomplete_policy = cfg.get("incomplete_policy", None)

    pad_side = cfg.get("pad_side", "left")  # left | right
    return_mask = bool(cfg.get("return_mask", False))
    mask_dtype = cfg.get("mask_dtype", "bool")  # bool | float32
    min_len_to_keep = int(cfg.get("min_len_to_keep", 1))

    if incomplete_policy not in {"drop", "pad_zero", "pad_repeat_first"}:
        raise ValueError(f"cut_seq_long: unknown incomplete_policy '{incomplete_policy}'")
    if pad_side not in {"left", "right"}:
        raise ValueError(f"cut_seq_long: unknown pad_side '{pad_side}'")
    if mask_dtype not in {"bool", "float32"}:
        raise ValueError(f"cut_seq_long: unknown mask_dtype '{mask_dtype}'")

    df = part_long_df
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found")

    feature_cols = cfg.get("feature_cols", None)
    if feature_cols is None:
        feature_cols = infer_feature_cols(df, date_col, stockid_col, label_col, weight_col)

    if bool(cfg.get("strict_check_nan", False)):
        if df[feature_cols].isna().any().any():
            raise ValueError("cut_seq_long: NaN detected in features; set preprocess nan_policy='strict'")
        if df[label_col].isna().any():
            raise ValueError("cut_seq_long: NaN detected in label")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    w_list: List[float] = []
    key_rows: List[Tuple[pd.Timestamp, Any]] = []
    mask_list: List[np.ndarray] = []

    n_stocks = 0
    n_dropped_short = 0
    n_padded_stocks = 0

    n_features = len(feature_cols)

    def _make_mask(n_real: int) -> np.ndarray:
        m = np.zeros((seq_len,), dtype=np.float32 if mask_dtype == "float32" else np.bool_)
        if pad_side == "left":
            start = seq_len - n_real
            if mask_dtype == "float32":
                m[start:] = 1.0
            else:
                m[start:] = True
        else:
            if mask_dtype == "float32":
                m[:n_real] = 1.0
            else:
                m[:n_real] = True
        return m

    def _pad_window(feat_mat: np.ndarray, n_real: int) -> np.ndarray:
        if n_real <= 0:
            raise ValueError("n_real must be > 0 for padding")

        real = feat_mat[-n_real:]  # last n_real days
        if incomplete_policy == "pad_zero":
            pad_block = np.zeros((seq_len - n_real, n_features), dtype=np.float32)
        elif incomplete_policy == "pad_repeat_first":
            first = real[0:1]
            pad_block = np.repeat(first, repeats=(seq_len - n_real), axis=0).astype(np.float32, copy=False)
        else:
            raise ValueError("internal: _pad_window called but incomplete_policy is drop")

        if pad_side == "left":
            out = np.concatenate([pad_block, real], axis=0)
        else:
            out = np.concatenate([real, pad_block], axis=0)
        return out.astype(np.float32, copy=False)

    for sid, g in df.groupby(stockid_col, sort=False):
        n_stocks += 1
        g = g.sort_values(date_col)
        m = len(g)

        if m < min_len_to_keep:
            n_dropped_short += 1
            continue

        feat_mat = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_vec = g[label_col].to_numpy(dtype=np.float32, copy=False)
        if weight_col and weight_col in g.columns:
            w_vec = g[weight_col].to_numpy(dtype=np.float32, copy=False)
        else:
            w_vec = None
        dates = g[date_col].to_numpy()

        if m < seq_len:
            if incomplete_policy == "drop":
                n_dropped_short += 1
                continue

            X_win = _pad_window(feat_mat, n_real=m)
            X_list.append(X_win)
            y_list.append(float(y_vec[-1]))
            if w_vec is not None:
                w_list.append(float(w_vec[-1]))
            key_rows.append((pd.Timestamp(dates[-1]), sid))
            if return_mask:
                mask_list.append(_make_mask(n_real=m))
            n_padded_stocks += 1
            continue

        for i in range(seq_len - 1, m):
            X_win = feat_mat[i - seq_len + 1: i + 1]
            X_list.append(X_win)
            y_list.append(float(y_vec[i]))
            if w_vec is not None:
                w_list.append(float(w_vec[i]))
            key_rows.append((pd.Timestamp(dates[i]), sid))
            if return_mask:
                if mask_dtype == "float32":
                    mask_list.append(np.ones((seq_len,), dtype=np.float32))
                else:
                    mask_list.append(np.ones((seq_len,), dtype=np.bool_))

    if len(X_list) == 0:
        raise ValueError("cut_seq_long: produced 0 sequence samples; check seq_len and data coverage")

    X_seq = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y_out = np.asarray(y_list, dtype=np.float32)
    keys = pd.DataFrame(key_rows, columns=[date_col, stockid_col])

    payload: Dict[str, Any] = {
        "X_seq": X_seq,
        "y": y_out,
        "keys": keys,
        "feature_cols": list(feature_cols),
        "seq_len": seq_len,
        "incomplete_policy": incomplete_policy,
        "pad_side": pad_side,
    }
    if len(w_list) > 0:
        payload["w"] = np.asarray(w_list, dtype=np.float32)
    if return_mask:
        payload["mask"] = np.stack(mask_list, axis=0)

    state: Dict[str, Any] = {
        "model": "seq",
        "seq_len": seq_len,
        "n_samples": int(X_seq.shape[0]),
        "n_features": int(X_seq.shape[2]),
        "n_stocks_seen": int(n_stocks),
        "n_stocks_dropped_short": int(n_dropped_short),
        "n_stocks_padded": int(n_padded_stocks),
        "has_weight": bool(len(w_list) > 0),
        "has_mask": bool(return_mask),
        "mask_dtype": mask_dtype if return_mask else None,
        "incomplete_policy": incomplete_policy,
        "pad_side": pad_side,
        "min_len_to_keep": int(min_len_to_keep),
        "feature_cols": list(feature_cols),
    }

    # cache (optional) — npz + keys parquet + meta json
    cache_cfg = cfg.get("cache", {}) or {}
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_dir = cache_cfg.get("dir", "./cache/cut")
    npz_tpl = cache_cfg.get("npz_tpl", "{model}_{part}_fold{fold}.npz")
    keys_tpl = cache_cfg.get("keys_tpl", "{model}_{part}_fold{fold}_keys.parquet")
    meta_tpl = cache_cfg.get("meta_tpl", "{model}_{part}_fold{fold}_meta.json")

    if cache_enabled:
        fold = meta.get("fold", 0)
        part = meta.get("part", "train")

        npz_path = os.path.join(cache_dir, npz_tpl.format(model="seq", part=part, fold=fold))
        keys_path = os.path.join(cache_dir, keys_tpl.format(model="seq", part=part, fold=fold))
        meta_path = os.path.join(cache_dir, meta_tpl.format(model="seq", part=part, fold=fold))

        ensure_dir(npz_path)

        npz_kwargs = {"X_seq": X_seq, "y": y_out}
        if "w" in payload:
            npz_kwargs["w"] = payload["w"]
        if "mask" in payload:
            npz_kwargs["mask"] = payload["mask"]
        np.savez_compressed(npz_path, **npz_kwargs)

        ensure_dir(keys_path)
        keys.to_parquet(keys_path, index=False)

        save_json(
            {
                "date_col": date_col,
                "stockid_col": stockid_col,
                "label_col": label_col,
                "weight_col": weight_col if "w" in payload else None,
                "feature_cols": list(feature_cols),
                "seq_len": seq_len,
                "incomplete_policy": incomplete_policy,
                "pad_side": pad_side,
                "return_mask": return_mask,
                "mask_dtype": mask_dtype if return_mask else None,
                "min_len_to_keep": int(min_len_to_keep),
                "n_samples": int(X_seq.shape[0]),
                "n_features": int(X_seq.shape[2]),
                "npz_path": npz_path,
                "keys_path": keys_path,
            },
            meta_path,
        )

        state["cached_npz"] = npz_path
        state["cached_keys"] = keys_path
        state["cached_meta"] = meta_path

    return payload, state
