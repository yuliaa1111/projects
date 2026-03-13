"""
ret_pred/trainer/model_bundle.py

训练结束后，将“可用于推理/复现”的模型产物打包保存到一个 bundle 目录中。

bundle 里通常包含：
- model.pkl                : 模型对象（pickle）
- feature_cols.json        : 训练时使用的最终特征列（顺序非常重要）
- preprocess_state.json    : preprocess 的 FIT state（可选；用于推理时 transform）
- meta.json                : 额外元信息（保存时间、模型文件名、实验信息等）

设计目的：
1) 复现：推理时只要给定 bundle_dir，就能加载模型、特征列、以及预处理统计量。
2) 可追溯：meta.json 记录保存时间与一些 bundle 信息，方便排查“这是谁训练出来的”。
3) 兼容性：为了让 meta / state 更容易落盘到 JSON，提供 _to_jsonable 做类型转换。

注意：
- 目前模型保存用 pickle.dump。若未来希望更稳（跨版本/跨环境），可以考虑 joblib 或模型自身的 save_model。
- feature_cols 的顺序必须保持一致：训练与推理对齐特征依赖它。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =========================
# JSON serialization helpers
# =========================
def _to_jsonable(x: Any) -> Any:
    """
    将任意对象尽可能转换为“可 JSON 序列化”的形式。

        设计目的：
    - preprocess_state / meta 里可能包含 Path、Timestamp、numpy 标量、Index、ndarray 等；
    - json.dump 默认无法序列化这些类型，会直接报错；
    - 所以这里做一个递归转换器，把常见对象映射到 JSON 友好的类型：
        - Path -> str
        - datetime/date/Timestamp -> ISO 字符串
        - numpy 标量 -> Python 标量
        - ndarray -> list
        - Series/Index -> dict/list
        - dict/list/tuple/set -> 递归处理元素

    参数:
        x (Any): 待转换对象。

    返回:
        Any: 可 JSON 化的对象（通常是 None/str/int/float/bool/list/dict）。
    """
    # 1) 原生 JSON 可直接支持的类型
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # 2) 常见路径/时间类型
    if isinstance(x, Path):
        return str(x)

    if isinstance(x, (datetime, date)):
        return x.isoformat()

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    # 3) numpy 标量类型（避免 json 序列化失败）
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)

    # 4) numpy / pandas 容器
    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, pd.Series):
        # Series -> dict 再递归（key/value 都可能需要转换）
        return _to_jsonable(x.to_dict())

    if isinstance(x, (pd.Index, pd.DatetimeIndex)):
        # Index -> list
        return [_to_jsonable(v) for v in x.tolist()]

    # 5) 递归处理 dict / iterable
    if isinstance(x, dict):
        # key 强制转 str，避免出现不可序列化的 key（如 Timestamp）
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # 6) 最后兜底：转成字符串（不保证信息可逆，但保证能落盘）
    return str(x)


def _json_dump(obj: Any, path: str) -> None:
    """
    将对象转换为可 JSON 化后写入文件。

    参数:
        obj (Any): 任意对象（内部会先 _to_jsonable）。
        path (str): 输出 json 文件路径。

    返回:
        None
    """
    obj2 = _to_jsonable(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, ensure_ascii=False, indent=2)


# =========================
# Public API
# =========================
def save_model_bundle(
    out_dir: str,
    *,
    model: Any,
    feature_cols: List[str],
    preprocess_state: Optional[Dict[str, Any]] = None,
    bundle_meta: Optional[Dict[str, Any]] = None,
    filename: str = "model.pkl",
) -> Dict[str, str]:
    """
    保存训练产物到一个 bundle 目录，并返回各文件路径。

    参数:
        out_dir (str):
            bundle 输出目录，例如 "./runs/exp001/model"。
        model (Any):
            训练好的模型对象，需要可被 pickle 序列化。
            （如果未来用 joblib 或模型自带 save，你可以替换这里的实现。）
        feature_cols (List[str]):
            训练时使用的最终特征列（顺序重要）。
            推理时会用它对齐特征列，避免错位。
        preprocess_state (Optional[Dict[str, Any]]):
            preprocess 的 FIT state（可选）。
            - 若提供且非空，将保存为 preprocess_state.json
            - 若为空/None，会尝试删除旧的 preprocess_state.json（避免误用旧文件）
        bundle_meta (Optional[Dict[str, Any]]):
            额外元信息（实验参数、数据区间、指标等），将写入 meta.json。
            函数会自动补充：
              - saved_at: 保存时间
              - model_file: 模型文件名
              - out_dir: bundle 输出目录
        filename (str):
            模型文件名，默认 "model.pkl"。
            你可以用它支持多模型版本（例如 "model_fold0.pkl"）。

    返回:
        Dict[str, str]:
            {
              "model_path": ".../model.pkl",
              "meta_path": ".../meta.json",
              "feature_cols_path": ".../feature_cols.json",
              "preprocess_state_path": ".../preprocess_state.json"
            }

    - feature_cols.json 使用 {"feature_cols": [...]} 的结构，兼容 run_predict 的读取逻辑。
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # 约定的 bundle 文件名
    model_path = outp / filename
    meta_path = outp / "meta.json"
    feat_path = outp / "feature_cols.json"
    pp_path = outp / "preprocess_state.json"

    # 1) save model
    # 注意：pickle 对环境/版本较敏感；后续如果你遇到兼容问题，优先考虑 joblib 或模型自带保存格式。
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 2) save feature cols（必须保存，推理对齐用）
    _json_dump({"feature_cols": list(feature_cols)}, str(feat_path))

    # 3) save preprocess state（可选）
    if isinstance(preprocess_state, dict) and len(preprocess_state) > 0:
        _json_dump(preprocess_state, str(pp_path))
    else:
        # 若这次不保存 preprocess_state，尽量删除旧文件，避免推理时误读旧状态
        try:
            if pp_path.exists():
                pp_path.unlink()
        except Exception:
            # 删除失败也不应影响训练主流程
            pass

    # 4) save meta（用于追踪与复盘）
    meta2 = dict(bundle_meta or {})
    meta2.setdefault("saved_at", datetime.now().isoformat())
    meta2.setdefault("model_file", str(model_path.name))
    meta2.setdefault("out_dir", str(outp))
    _json_dump(meta2, str(meta_path))

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "feature_cols_path": str(feat_path),
        "preprocess_state_path": str(pp_path),
    }
