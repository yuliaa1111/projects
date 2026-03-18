from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """
    Rolling 过程中每个 window 训练产生的“候选模型”。

    字段:
        step_id (int): Rolling 的 step 序号。
        fold (int): 当前窗口 fold（如有）。
        valid_score (float): valid 集评分，用于选择 best。
        model (Any): 训练得到的模型对象（可能是 LGBM/CatBoost/XGB/torch model）。
        feature_cols (List[str]): 模型对应的特征列，用于推理阶段对齐特征。
        meta (Dict[str, Any]): window 元信息（如日期范围、preprocess_state 等）。
        params (Dict[str, Any]): 模型参数快照（便于复盘/落盘）。
    """
    step_id: int
    fold: int
    valid_score: float
    model: Any
    feature_cols: List[str]
    meta: Dict[str, Any]
    params: Dict[str, Any]


def _safe_release_model(m: Any) -> None:
    """
    安全释放模型引用（selector 侧）。

    说明:
        - Python 内存释放的关键是“断开引用”。这里做的就是把 selector
          不再需要的 model 引用删掉。
        - 注意：这不会影响 RollingTrainer 里仍然持有的局部引用（model 变量）。
          所以你仍然需要在 RollingTrainer 的 finally 里 del model（这是另一个必须改点）。
        - 这里不做 torch.cuda.empty_cache()，因为那应由 trainer 统一负责。
    """
    try:
        del m
    except Exception:
        pass


class BaseSelector:
    """
    模型选择器基类。

    作用:
        - RollingTrainer 每个 step 训练完后会调用 observe(...) 把候选交给 selector。
        - RollingTrainer 在 run 结束时调用 select() 得到最终要保存的模型。

    内存约定:
        - selector 负责管理它自己持有的模型引用。
        - 当 selector 决定丢弃某个候选时，应及时释放其 model 引用，
          避免 window 数变大导致内存线性上涨。
    """

    def observe(
        self,
        *,
        step_id: int,
        fold: int,
        valid_score: float,
        model: Any,
        feature_cols: List[str],
        meta: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        """
        接收一个候选模型。

        参数:
            step_id (int): 当前 step 编号。
            fold (int): fold 编号。
            valid_score (float): valid 分数。
            model (Any): 模型对象。
            feature_cols (List[str]): 特征列名列表。
            meta (Dict[str, Any]): window 元信息。
            params (Dict[str, Any]): 模型参数。
        """
        raise NotImplementedError

    def select(self) -> Candidate:
        """
        选择最终模型。

        返回:
            Candidate: 最终被选择的候选。
        """
        raise NotImplementedError


class LastStepSelector(BaseSelector):
    """
    选择“最后一个 step”的模型。

    内存策略:
        - 每次 observe 会覆盖 _last。
        - 覆盖前会释放旧 _last.model 引用，避免累积。
    """

    def __init__(self):
        self._last: Optional[Candidate] = None

    def observe(
        self,
        *,
        step_id: int,
        fold: int,
        valid_score: float,
        model: Any,
        feature_cols: List[str],
        meta: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        if self._last is not None:
            _safe_release_model(self._last.model)

        self._last = Candidate(
            step_id=step_id,
            fold=fold,
            valid_score=float(valid_score),
            model=model,
            feature_cols=list(feature_cols),
            meta=dict(meta),
            params=dict(params),
        )

    def select(self) -> Candidate:
        if self._last is None:
            raise RuntimeError("No candidate observed; cannot select model.")
        return self._last


class BestValidSelector(BaseSelector):
    """
    选择 valid_score 最好的模型（全局 best）。

    内存策略:
        - 若新候选更好：释放旧 best.model，然后替换。
        - 若新候选不够好：立即释放新候选的 model（因为不会保留它）。
    """

    def __init__(self):
        self._best: Optional[Candidate] = None

    def observe(
        self,
        *,
        step_id: int,
        fold: int,
        valid_score: float,
        model: Any,
        feature_cols: List[str],
        meta: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        cand = Candidate(
            step_id=step_id,
            fold=fold,
            valid_score=float(valid_score),
            model=model,
            feature_cols=list(feature_cols),
            meta=dict(meta),
            params=dict(params),
        )

        if self._best is None:
            self._best = cand
            return

        if cand.valid_score > self._best.valid_score:
            _safe_release_model(self._best.model)
            self._best = cand
        else:
            _safe_release_model(cand.model)

    def select(self) -> Candidate:
        if self._best is None:
            raise RuntimeError("No candidate observed; cannot select model.")
        return self._best


class BestValidLastNSelector(BaseSelector):
    """
    在最近 N 个 step 里选 valid_score 最好的模型。

    参数:
        last_n (int): 缓存最近多少个候选。

    内存策略:
        - buffer 超过 last_n 时，会把被挤出去的候选释放 model 引用。
        - 注意：该策略本意就是“最多保留 N 个模型引用”，last_n 越大越占内存。
    """

    def __init__(self, last_n: int = 5):
        self.last_n = int(last_n)
        self._buf: List[Candidate] = []

    def observe(
        self,
        *,
        step_id: int,
        fold: int,
        valid_score: float,
        model: Any,
        feature_cols: List[str],
        meta: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        self._buf.append(
            Candidate(
                step_id=step_id,
                fold=fold,
                valid_score=float(valid_score),
                model=model,
                feature_cols=list(feature_cols),
                meta=dict(meta),
                params=dict(params),
            )
        )

        if len(self._buf) > self.last_n:
            dropped = self._buf[:-self.last_n]
            for d in dropped:
                _safe_release_model(d.model)
            self._buf = self._buf[-self.last_n:]

    def select(self) -> Candidate:
        if not self._buf:
            raise RuntimeError("No candidate observed; cannot select model.")
        return max(self._buf, key=lambda c: c.valid_score)


def build_model_selector(cfg: Optional[Dict[str, Any]]) -> BaseSelector:
    """
    根据配置构建模型选择器（RollingTrainer 依赖此函数）。

    参数:
        cfg (Optional[Dict[str, Any]]):
            - None 或 {}：默认 strategy='last'
            - 例：
              {"strategy": "last"}
              {"strategy": "best_valid"}
              {"strategy": "best_valid_last_n", "last_n": 5}

    返回:
        BaseSelector: 选择器实例。
    """
    cfg = cfg or {}
    strategy = str(cfg.get("strategy", "last"))

    if strategy == "last":
        return LastStepSelector()
    if strategy == "best_valid":
        return BestValidSelector()
    if strategy == "best_valid_last_n":
        return BestValidLastNSelector(last_n=int(cfg.get("last_n", 5)))

    raise KeyError(
        "Unknown model_select.strategy "
        f"'{strategy}', available: ['last','best_valid','best_valid_last_n']"
    )
