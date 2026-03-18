# 存储训练插件（1-6）
# 每个 step / fold 里，RollingTrainer会做这几件事：
# 0) 用当前参数训练模型，得到 y_true/y_pred/keys（keys 包含 date/stockid 等对齐信息）
# 1) metrics：计算训练/寻参用的指标（mse/mae/rankic）
# 2) history：把每个 step/fold 的分数、参数、元信息记录成表，便于复盘与画训练诊断曲线
# 3) schedule：控制“何时触发 tune（重新寻参）”，如每 k 步触发、warmup 后每 k 步、永不触发
# 4) tuner：控制“怎么寻参”，如手写 candidates 的 grid search、给定 search_space 的 random search
# 5) gate：控制“是否更新参数”，如要求新分数至少比当前高 min_improve 才允许替换（防抖/防过拟合噪声）
# 6) saver：控制“是否保存预测结果”，把 y_true/y_pred/keys/meta 落盘为 parquet，供 evaluate 模块事后统一评估与画图

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
import json
import math
import random
import hashlib

import numpy as np
import pandas as pd

# =========================
# Metrics：计算训练/寻参用的指标（mse/mae/rankic）
# =========================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rankic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keys: pd.DataFrame,
    date_col: str = "date",
    method: str = "spearman",  # "spearman" or "pearson"
) -> float:

    df = pd.DataFrame({
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
    })
    if keys is None:

        s = df[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        return float(s) if pd.notna(s) else 0.0

    if date_col not in keys.columns:
        raise KeyError(f"rankic needs '{date_col}' in keys.columns, got {list(keys.columns)}")

    df[date_col] = keys[date_col].values
    ics: List[float] = []
    for _, g in df.groupby(date_col):
        if len(g) < 2:
            continue
        ic = g[["y_true", "y_pred"]].corr(method=method).iloc[0, 1]
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if len(ics) > 0 else 0.0


METRIC_FN = {
    "mse": mse,
    "mae": mae,
    "rankic": rankic,
}


# =========================
# History：把每个 step/fold 的分数、参数、元信息记录成表，便于复盘与画训练诊断曲线
# =========================

@dataclass
class History:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, **kwargs):
        self.records.append(dict(kwargs))

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)

        # ---- Parquet-safe: object 列统一 stringify ----
        def _to_jsonable(x):
            if x is None:
                return None
            # 常见：dict/list/tuple/set -> JSON
            if isinstance(x, (dict, list, tuple, set)):
                try:
                    return json.dumps(x, ensure_ascii=False, default=str)
                except Exception:
                    return str(x)

            # pandas Timestamp / numpy scalar / 其他对象
            # 尝试 JSON；不行就 str
            try:
                json.dumps(x, ensure_ascii=False, default=str)
                return x  # primitive (str/int/float/bool) 会原样返回
            except Exception:
                return str(x)

        # 只处理 object 列
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].apply(_to_jsonable)

        return df


# =========================
# Schedule：控制“何时触发 tune（重新寻参）”，如每 k 步触发、warmup 后每 k 步、永不触发
# =========================

class BaseSchedule:
    def should_tune(self, step_id: int, history: History, meta: Dict[str, Any]) -> bool:
        raise NotImplementedError


class EveryKSteps(BaseSchedule):
    def __init__(self, k: int = 4, offset: int = 0):
        self.k = int(k)
        self.offset = int(offset)

    def should_tune(self, step_id: int, history: History, meta: Dict[str, Any]) -> bool:
        if self.k <= 0:
            return False
        return (step_id - self.offset) % self.k == 0


class WarmupThenEveryK(BaseSchedule):
    def __init__(self, warmup: int = 10, k: int = 4):
        self.warmup = int(warmup)
        self.k = int(k)

    def should_tune(self, step_id: int, history: History, meta: Dict[str, Any]) -> bool:
        if step_id < self.warmup:
            return False
        if self.k <= 0:
            return False
        return (step_id - self.warmup) % self.k == 0


class NeverTune(BaseSchedule):
    def should_tune(self, step_id: int, history: History, meta: Dict[str, Any]) -> bool:
        return False


SCHEDULE_CLS = {
    "every_k_steps": EveryKSteps,
    "warmup_then_every_k": WarmupThenEveryK,
    "never": NeverTune,
}


def build_schedule(cfg: Optional[Dict[str, Any]]) -> BaseSchedule:
    if not cfg:
        return NeverTune()
    name = cfg.get("name", "never")
    params = cfg.get("params", {}) or {}
    if name not in SCHEDULE_CLS:
        raise KeyError(f"Unknown schedule '{name}', available: {list(SCHEDULE_CLS.keys())}")
    return SCHEDULE_CLS[name](**params)


# =========================
# Tuner：控制“怎么寻参”，如手写 candidates 的 grid search、给定 search_space 的 random search
# =========================

class BaseTuner:
    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        candidates: Optional[List[Dict[str, Any]]] = None,
        search_space: Optional[Dict[str, List[Any]]] = None,
        n_trials: int = 20,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        raise NotImplementedError


class GridSearch(BaseTuner):
    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        candidates: Optional[List[Dict[str, Any]]] = None,
        search_space: Optional[Dict[str, List[Any]]] = None,
        n_trials: int = 20,
        seed: int = 42,
    ):
        if not candidates:
            raise ValueError("GridSearch needs 'candidates' (list of param dicts).")
        best_params, best_score = None, -math.inf
        trial_scores = []
        for i, p in enumerate(candidates):
            s = float(objective_fn(p))
            trial_scores.append((i, s))
            if s > best_score:
                best_score = s
                best_params = p
        return best_params or {}, float(best_score), {"trial_scores": trial_scores}


class RandomSearch(BaseTuner):
    def search(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        candidates: Optional[List[Dict[str, Any]]] = None,
        search_space: Optional[Dict[str, List[Any]]] = None,
        n_trials: int = 20,
        seed: int = 42,
    ):
        if not search_space:
            raise ValueError("RandomSearch needs 'search_space' (dict of name -> list of values).")
        rng = random.Random(seed)
        keys = list(search_space.keys())
        best_params, best_score = None, -math.inf
        trial_scores = []
        for t in range(int(n_trials)):
            p = {k: rng.choice(search_space[k]) for k in keys}
            s = float(objective_fn(p))
            trial_scores.append((t, s))
            if s > best_score:
                best_score = s
                best_params = p
        return best_params or {}, float(best_score), {"trial_scores": trial_scores}


TUNER_CLS = {
    "grid_search": GridSearch,
    "random_search": RandomSearch,
}


def build_tuner(cfg: Optional[Dict[str, Any]]) -> Tuple[bool, BaseTuner, Dict[str, Any]]:
    if not cfg:
        t = RandomSearch()
        t.name = "random_search"
        return False, t, {}

    enabled = bool(cfg.get("enabled", True))
    name = cfg.get("name", "random_search")
    params = cfg.get("params", {}) or {}
    if name not in TUNER_CLS:
        raise KeyError(f"Unknown tuner '{name}', available: {list(TUNER_CLS.keys())}")

    t = TUNER_CLS[name]()
    t.name = name  
    return enabled, t, params


# =========================
# Update gate：控制“是否更新参数”，如要求新分数至少比当前高 min_improve 才允许替换（防抖/防过拟合噪声）
# =========================

class MinImproveGate:
    def __init__(self, min_improve: float = 0.002):
        self.min_improve = float(min_improve)

    def allow_update(self, current_score: float, best_score: float) -> bool:
        return float(best_score) >= float(current_score) + self.min_improve


GATE_CLS = {
    "min_improve": MinImproveGate,
}


def build_gate(cfg: Optional[Dict[str, Any]]) -> MinImproveGate:
    if not cfg:
        return MinImproveGate(min_improve=0.0)
    name = cfg.get("name", "min_improve")
    params = cfg.get("params", {}) or {}
    if name not in GATE_CLS:
        raise KeyError(f"Unknown gate '{name}', available: {list(GATE_CLS.keys())}")
    return GATE_CLS[name](**params)


# =========================
# Saver：控制“是否保存预测结果”，把 y_true/y_pred/keys/meta 落盘为 parquet，供 evaluate 模块事后统一评估与画图
# =========================

def _stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


class ParquetSaver:
    def __init__(
        self,
        dir: str = "./runs/preds",
        save_parts: Optional[List[str]] = None,
        file_tpl: str = "step{step_id}_fold{fold}_{part}.parquet",
    ):
        self.dir = dir
        self.save_parts = save_parts or ["valid", "test"]
        self.file_tpl = file_tpl

    def save(self, df: pd.DataFrame, meta: Dict[str, Any]) -> Optional[str]:
        part = meta.get("part", "test")
        if part not in self.save_parts:
            return None

        step_id = meta.get("step_id", 0)
        fold = meta.get("fold", 0)

        fname = self.file_tpl.format(step_id=step_id, fold=fold, part=part)
        out_dir = os.path.abspath(self.dir)
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.abspath(os.path.join(out_dir, fname))
        df.to_parquet(path, index=False)

        # 可选：调试时打开
        # print(f"[saver] wrote: {path}")

        return path



def build_saver(cfg: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[ParquetSaver]]:
    if not cfg:
        return False, None
    enabled = bool(cfg.get("enabled", True))
    params = cfg.get("params", {}) or {}
    if not enabled:
        return False, None
    return True, ParquetSaver(**params)
