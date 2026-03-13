from __future__ import annotations
from typing import Any, Dict, Tuple

# 统一 loss 名 → (lgbm objective + extra params)
_LGBM_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "mse": ("regression", {}),
    "l2":  ("regression", {}),
    "mae": ("regression_l1", {}),
    "l1":  ("regression_l1", {}),
    "huber": ("huber", {}),          # 需要 alpha
    "quantile": ("quantile", {}),    # 需要 alpha
}

# 统一 loss 名 → (xgb objective + extra params)
# 注意：xgb 的 huber/quantile 支持与 lgbm 不完全一致，这里给“可用的近似”
_XGB_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "mse": ("reg:squarederror", {}),
    "l2":  ("reg:squarederror", {}),
    "mae": ("reg:absoluteerror", {}),   # xgb 新版本支持；旧版本可能没有
    "l1":  ("reg:absoluteerror", {}),
    "huber": ("reg:pseudohubererror", {}),  # 常用替代
    "quantile": ("reg:quantileerror", {}),  # 新版本支持；alpha/quantile_alpha
}

# 统一 loss 名 → (catboost loss_function string + extra params)
_CAT_MAP: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "mse": ("RMSE", {}),
    "l2":  ("RMSE", {}),
    "mae": ("MAE", {}),
    "l1":  ("MAE", {}),
    # CatBoost 的 Huber 用 delta（有的版本叫 delta 或用 Huber:delta=）
    "huber": ("Huber", {}),           # 需要 delta（或等价参数）
    "quantile": ("Quantile", {}),     # 需要 alpha
}


def apply_tree_objective_from_loss(
    model_name: str,
    model_params: Dict[str, Any],
    loss_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据 loss_cfg 覆盖 tree 模型的 objective/loss_function。
    返回新的 model_params（不原地改）。
    """
    if not loss_cfg:
        return model_params

    loss_name = str(loss_cfg.get("name", "")).lower().strip()
    loss_params = dict(loss_cfg.get("params", {}) or {})

    out = dict(model_params)

    if model_name == "lgbm":
        if loss_name not in _LGBM_MAP:
            return out
        obj, extra = _LGBM_MAP[loss_name]
        out["objective"] = obj
        # 常见：huber/quantile 用 alpha
        if obj in ("huber", "quantile") and "alpha" in loss_params:
            out["alpha"] = loss_params["alpha"]
        out.update(extra)
        return out

    if model_name == "xgb":
        if loss_name not in _XGB_MAP:
            return out
        obj, extra = _XGB_MAP[loss_name]
        out["objective"] = obj
        # xgb quantile 可能用 quantile_alpha
        if obj == "reg:quantileerror" and "alpha" in loss_params:
            out["quantile_alpha"] = loss_params["alpha"]
        out.update(extra)
        return out

    if model_name == "catboost":
        if loss_name not in _CAT_MAP:
            return out
        lf, extra = _CAT_MAP[loss_name]
        # catboost loss_function 有时需要拼参数
        if lf == "Huber" and "delta" in loss_params:
            out["loss_function"] = f"Huber:delta={loss_params['delta']}"
        elif lf == "Quantile" and "alpha" in loss_params:
            out["loss_function"] = f"Quantile:alpha={loss_params['alpha']}"
        else:
            out["loss_function"] = lf
        out.update(extra)
        return out

    return out
