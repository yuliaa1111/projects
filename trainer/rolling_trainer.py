"""
ret_pred/trainer/rolling_trainer.py
核心训练主循环在def run()里 

实现：滚动步进训练（walk-forward / rolling windows）+ 可插拔自动寻参（tuner）+ 可选保存预测（saver）
并支持：按策略选择并保存“最终模型”（model_save: last / best_valid / best_valid_last_n）

- 当valid 为空时，跳过 valid 的 predict/score/save/observe，valid_score 记为 None。
- tuner 也会在 valid 为空时被强制禁止触发（避免 objective_fn 依赖 valid）。

"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import copy
import logging
import gc

import numpy as np
import pandas as pd

from .registry import register_trainer
from .plugins import (
    History, METRIC_FN,
    build_schedule, build_tuner, build_gate, build_saver,
    _stable_hash,
)

from .model_select import build_model_selector
from .model_bundle import save_model_bundle

logger = logging.getLogger(__name__)

# NN 训练用（可选依赖）
try:
    import torch
except Exception:
    torch = None  # type: ignore


@register_trainer("rolling")
class RollingTrainer:
    """
    RollingTrainer：滚动步进训练器（walk-forward）

    核心职责：
    1) 接收 windows（一个可迭代对象），每个元素为：
        (train_payload, valid_payload, test_payload, meta)
       其中 payload 来自 cutting.py（对一个时间窗口切出来的数据子集）
    2) 每个 step：
       - 用 train_pl 训练模型（可选使用 valid_pl 作为 eval_set）
       - 对 train/valid/test 做预测与打分（valid 可为空则跳过）
       - 可按 schedule 触发 tuner，对候选参数进行搜索，并用 gate 决定是否更新参数
       - 可选保存预测结果到磁盘（saver）
       - 可选将模型交给 model_selector，用于最终保存策略（model_save）
    3) 输出：
       - history_df：包含每一步的 score、params_hash、是否调参、是否更新、保存路径等信息

    参数（__init__）：
        model (Dict[str, Any]):
            模型配置字典。通常包含：
              - name: 模型名（lgbm/xgb/catboost/nn_xxx）
              - family: "tree" 或 "nn"
              - params 或 model_config: 模型参数（会作为初始参数）
            注意：在训练过程中 current_model_params 可能被 tuner 更新。

        metric (str):
            训练过程打分指标名。必须在 METRIC_FN 中，例如 "rankic", "mse", "mae", "r2", "icir" 等。

        maximize (bool):
            指标方向。True 表示越大越好；False 表示越小越好（内部会取负号统一成“越大越好”）。

        task (str):
            任务类型：一般是 "regression" 或 "classification"（具体取决于你 evaluate/tasks 的实现）。

        loss (Optional[Dict[str, Any]]):
            Loss 配置，用于 tree objective 映射或 nn 的 criterion 构建。
            形如：
              { enabled: true, name: "mse"|"mae"|"huber"|"quantile", params: {...} }

        nn_fit (Optional[Dict[str, Any]]):
            NN 训练超参，例如 epochs/lr/batch_size。

        schedule (Optional[Dict[str, Any]]):
            调参触发策略配置。例如每 k 步调一次：every_k_steps。

        tuner (Optional[Dict[str, Any]]):
            寻参配置。enabled/name/params 等。enabled=false 则不会调参。

        update_gate (Optional[Dict[str, Any]]):
            参数更新门槛策略。决定 tuner 找到更好参数后是否更新 current_model_params。

        saver (Optional[Dict[str, Any]]):
            预测保存器配置。enabled=true 时会保存每 step 的 preds parquet。

        model_save (Optional[Dict[str, Any]]):
            最终模型保存策略。enabled=true 时会把每 step 的模型交给 selector，
            训练结束后 selector 决定保存哪一个模型（last/best_valid/...）

        run_id (str):
            当前训练 run 标识，用于日志与输出文件的 meta。

        date_col / stockid_col (str):
            keys DataFrame 中用于 RankIC 等指标计算的字段名。

        device (str):
            "cpu"|"cuda"|"mps"|"auto"
            - tree 模型一般忽略 device（除非模型本身支持 GPU 参数）
            - nn 模型会使用 device 放置 tensor/model
            - "auto" 会优先 cuda，其次 mps，否则 cpu

        seed (int):
            随机种子（主要用于 tuner/random_search 以及部分模型的随机性）
    """

    def __init__(
        self,
        model: Dict[str, Any],
        metric: str = "rankic",
        maximize: bool = True,
        task: str = "regression",
        loss: Optional[Dict[str, Any]] = None,
        nn_fit: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
        tuner: Optional[Dict[str, Any]] = None,
        update_gate: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        model_save: Optional[Dict[str, Any]] = None,
        run_id: str = "exp001",
        date_col: str = "date",
        stockid_col: str = "stockid",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model_cfg = model
        self.metric_name = metric
        self.maximize = bool(maximize)
        self.task = task
        self.loss_cfg = loss or {"name": "mse", "params": {}}
        self.nn_fit_cfg = nn_fit or {"epochs": 5, "lr": 1e-3, "batch_size": 256}

        # 插件体系：schedule/tuner/gate/saver
        self.schedule = build_schedule(schedule)
        self.tuner_enabled, self.tuner, self.tuner_params = build_tuner(tuner)
        self.gate = build_gate(update_gate)
        self.save_enabled, self.saver = build_saver(saver)

        # 最终模型选择+保存（可选）
        self.model_save_cfg = model_save or {"enabled": False}
        self.model_save_enabled = bool(self.model_save_cfg.get("enabled", False))
        self.model_selector = build_model_selector(self.model_save_cfg if self.model_save_enabled else None)

        self.run_id = run_id
        self.date_col = date_col
        self.stockid_col = stockid_col
        self.device = device
        self.seed = seed

        # device can be "cpu" | "cuda" | "mps" | "auto"
        # - tree models ignore this
        # - nn models use it to place tensors/models
        if isinstance(self.device, str) and self.device.lower() == "auto":
            try:
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    self.device = "cuda"
                elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"

        if self.metric_name not in METRIC_FN:
            raise KeyError(f"Unknown metric '{self.metric_name}', available: {list(METRIC_FN.keys())}")

        logger.info(
            "RollingTrainer init | run_id=%s | model=%s | family=%s | metric=%s | maximize=%s | device=%s | seed=%s | "
            "tuner=%s | saver=%s | model_save=%s",
            self.run_id,
            str(self.model_cfg.get("name", "")),
            str(self.model_cfg.get("family", "")),
            self.metric_name,
            self.maximize,
            self.device,
            self.seed,
            bool(self.tuner_enabled),
            bool(self.save_enabled),
            bool(self.model_save_enabled),
        )

        # =========================================================
        # Sweep / external orchestration support
        # ---------------------------------------------------------
        # 为了让外层（如 SweepRollingTrainer）能“自己决定如何保存模型”，
        # RollingTrainer 会在每个 step 结束时记录“最后一个窗口训练出的模型候选信息”。
        #
        # 注意：
        # - 这里只记录 *最后一步* 的模型引用（self._last_model）。外层如果要长期保存，
        #   应当在 rolling.run() 返回后立即调用 get_last_candidate() 并落盘。
        # - 如果你同时开启了 model_save（selector 策略），RollingTrainer 仍会按 selector 逻辑保存。
        #   但 sweep 场景建议关闭 model_save，由 sweep_trainer 统一保存 last。
        # =========================================================
        self._last_model = None
        self._last_feature_cols: Optional[list[str]] = None
        self._last_params: Optional[Dict[str, Any]] = None
        self._last_meta: Optional[Dict[str, Any]] = None
        self._last_step_id: Optional[int] = None
        self._last_fold: Optional[int] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(
        self,
        windows: Iterable[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        执行滚动训练核心主循环。
        对每个 step/window 做一次训练-预测-打分-（可选）保存-（可选）调参-记录 history。

        参数:
            windows:
                可迭代对象，每次 yield:
                    (train_pl, valid_pl, test_pl, meta)
                - train_pl / test_pl 必须存在并非空
                - valid_pl 允许为 None 或空 payload（valid_ratio=0 的场景）

        返回:
            pd.DataFrame:
                history_df，每一行对应一个 step（一个窗口一次训练评估），后面 evaluator/画诊断图都靠它。
        """
        hist = History()

        # current_model_params 是“会随时间演化”的参数（tuner 可能更新它）
        current_model_params = copy.deepcopy(
            self.model_cfg.get("model_config", {}) or self.model_cfg.get("params", {})
        )

        logger.info(
            "RollingTrainer run start | init_params_hash=%s | n_params=%d",
            _stable_hash(current_model_params),
            len(current_model_params),
        )
        logger.debug("init_params=%s", current_model_params)

        # 从外部传进来的 windows 迭代器里解包出来的，包含 train_pl / valid_pl / test_pl / meta 四个部分。
        for step_id, (train_pl, valid_pl, test_pl, meta) in enumerate(windows):
            model = None
            try:
                meta = dict(meta or {})
                meta.setdefault("step_id", step_id)
                meta.setdefault("fold", meta.get("fold", 0))
                meta.setdefault("run_id", self.run_id)

                fold = int(meta.get("fold", 0))

                # —— sizes (safe) —— #
                n_train = len(train_pl.get("y", [])) if isinstance(train_pl, dict) else -1
                n_valid = len(valid_pl.get("y", [])) if isinstance(valid_pl, dict) and valid_pl is not None else 0
                n_test = len(test_pl.get("y", [])) if isinstance(test_pl, dict) else -1

                logger.info(
                    "step start | step_id=%d | fold=%d | n_train=%d | n_valid=%d | n_test=%d | params_hash=%s",
                    step_id,
                    fold,
                    n_train,
                    n_valid,
                    n_test,
                    _stable_hash(current_model_params),
                )

                # 训练
                model, fit_info = self._fit_one_window(train_pl, valid_pl, current_model_params)

                # 预测：valid 可能为空，需跳过
                train_pred = self._predict(model, train_pl)

                valid_pred = None
                if not self._is_empty_payload(valid_pl):
                    valid_pred = self._predict(model, valid_pl)  # type: ignore[arg-type]

                test_pred = self._predict(model, test_pl)

                # 打分：valid 为空则 valid_score=None
                train_score = self._score(train_pl, train_pred)

                valid_score = None
                if valid_pred is not None:
                    valid_score = self._score(valid_pl, valid_pred)  # type: ignore[arg-type]

                test_score = self._score(test_pl, test_pred)

                logger.info(
                    "step scores | step_id=%d | fold=%d | train=%.6f | valid=%s | test=%.6f",
                    step_id,
                    fold,
                    float(train_score),
                    "nan" if valid_score is None else f"{float(valid_score):.6f}",
                    float(test_score),
                )

                # 保存预测：valid 为空则不保存
                saved_train, saved_valid, saved_test = None, None, None
                if self.save_enabled and self.saver is not None:
                    saved_train = self._save_preds(
                        train_pl, train_pred, meta, part="train", params=current_model_params
                    )

                    if valid_pred is not None:
                        saved_valid = self._save_preds(
                            valid_pl, valid_pred, meta, part="valid", params=current_model_params  # type: ignore[arg-type]
                        )

                    saved_test = self._save_preds(
                        test_pl, test_pred, meta, part="test", params=current_model_params
                    )

                    logger.info(
                        "preds saved | step_id=%d | fold=%d | train=%s | valid=%s | test=%s",
                        step_id,
                        fold,
                        str(saved_train),
                        str(saved_valid),
                        str(saved_test),
                    )

                row = dict(
                    step_id=step_id,
                    fold=fold,
                    model_name=self.model_cfg.get("name"),
                    params=current_model_params,
                    params_hash=_stable_hash(current_model_params),
                    train_loss=fit_info.get("train_loss"),
                    train_score=train_score,
                    valid_score=valid_score,
                    test_score=test_score,
                    tuned=False,
                    best_score=None,
                    updated=False,
                    tune_meta=None,
                    saved_train=saved_train,
                    saved_valid=saved_valid,
                    saved_test=saved_test,
                    meta=meta,
                )

                # tuner 调参（必须有 valid 才有意义）
                current_score_ref = float(valid_score) if valid_score is not None else None
                do_tune = self.tuner_enabled and (not self._is_empty_payload(valid_pl)) and self.schedule.should_tune(step_id, hist, meta)

                logger.info(
                    "tune decision | step_id=%d | fold=%d | tuner_enabled=%s | do_tune=%s",
                    step_id,
                    fold,
                    bool(self.tuner_enabled),
                    bool(do_tune),
                )

                if do_tune:

                    def objective_fn(delta_params: Dict[str, Any]) -> float:
                        """
                        tuner 单次 trial 的目标函数：
                        - 用 trial_params 重新 fit 模型
                        - 在 valid 上 predict 并 score
                        - 返回 score（float）
                        """
                        trial_params = copy.deepcopy(current_model_params)
                        trial_params.update(delta_params)

                        m, _ = self._fit_one_window(train_pl, valid_pl, trial_params)
                        vp = self._predict(m, valid_pl)  # type: ignore[arg-type]
                        s = float(self._score(valid_pl, vp))  # type: ignore[arg-type]

                        # cleanup per trial
                        if torch is not None and self.device != "cpu":
                            try:
                                del m
                            except Exception:
                                pass
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass

                        return s

                    tune_params = self.tuner_params or {}
                    model_name = str(self.model_cfg.get("name", ""))

                    cand_by_model = tune_params.get("candidates_by_model")
                    if isinstance(cand_by_model, dict):
                        candidates = cand_by_model.get(model_name)
                    else:
                        candidates = tune_params.get("candidates")

                    search_space = tune_params.get("search_space")
                    ss_by_model = tune_params.get("search_space_by_model")
                    if isinstance(ss_by_model, dict):
                        search_space = ss_by_model.get(model_name, search_space)

                    tuner_name = str(getattr(self.tuner, "name", ""))

                    if tuner_name == "grid_search":
                        if not candidates:
                            raise ValueError(
                                f"grid_search requires candidates for model '{model_name}'. "
                                f"Please provide tuner.params.candidates_by_model.{model_name} or tuner.params.candidates."
                            )
                    else:
                        if not search_space:
                            raise ValueError(
                                f"random_search requires search_space for model '{model_name}'. "
                                f"Please provide 'search_space' or 'search_space_by_model.{model_name}'."
                            )

                    logger.info(
                        "tuning start | step_id=%d | fold=%d | tuner=%s | n_trials=%s",
                        step_id,
                        fold,
                        tuner_name,
                        str(tune_params.get("n_trials", 20)),
                    )

                    best_params, best_score, tune_meta = self.tuner.search(
                        objective_fn=objective_fn,
                        candidates=candidates,
                        search_space=search_space,
                        n_trials=tune_params.get("n_trials", 20),
                        seed=tune_params.get("seed", self.seed),
                    )

                    row["tuned"] = True
                    row["best_score"] = best_score
                    row["tune_meta"] = tune_meta

                    updated = False
                    if current_score_ref is not None and self.gate.allow_update(current_score_ref, best_score):
                        current_model_params.update(best_params)
                        updated = True

                    row["updated"] = updated

                    if updated:
                        row["params"] = copy.deepcopy(current_model_params)
                        row["params_hash"] = _stable_hash(current_model_params)

                hist.append(**row)

                # -------------------------
                # record "last model" candidate for external orchestrators (e.g., sweep)
                # -------------------------
                try:
                    feature_cols = (
                        train_pl.get("feature_cols")
                        or (valid_pl.get("feature_cols") if isinstance(valid_pl, dict) else None)
                        or test_pl.get("feature_cols")
                    )
                    if feature_cols is None:
                        X0 = train_pl.get("X")
                        if isinstance(X0, pd.DataFrame):
                            feature_cols = list(X0.columns)

                    self._last_model = model
                    self._last_feature_cols = list(feature_cols) if feature_cols is not None else None
                    self._last_params = copy.deepcopy(current_model_params)
                    self._last_meta = copy.deepcopy(meta)
                    self._last_step_id = int(step_id)
                    self._last_fold = int(fold)
                except Exception:
                    # 记录失败不应影响训练主流程
                    pass

                # model_save observe：依赖 valid_score（valid 空就跳过）
                if self.model_save_enabled and (valid_score is not None):
                    feature_cols = (
                        train_pl.get("feature_cols")
                        or (valid_pl.get("feature_cols") if isinstance(valid_pl, dict) else None)
                        or test_pl.get("feature_cols")
                    )
                    if feature_cols is None:
                        X = train_pl.get("X")
                        if isinstance(X, pd.DataFrame):
                            feature_cols = list(X.columns)
                        else:
                            raise ValueError(
                                "model_save enabled but cannot determine feature_cols "
                                "(payload has no feature_cols and X is not DataFrame)."
                            )

                    self.model_selector.observe(
                        step_id=step_id,
                        fold=int(meta.get("fold", 0)),
                        valid_score=float(valid_score),
                        model=model,
                        feature_cols=list(feature_cols),
                        meta=meta,
                        params=copy.deepcopy(current_model_params),
                    )

                logger.info("step done | step_id=%d | fold=%d", step_id, fold)

            except Exception:
                logger.exception("step failed | step_id=%d | meta=%s", step_id, str(meta))
                raise

            finally:
                # per-window cleanup
                try:
                    del model
                except Exception:
                    pass

                for _obj in ("train_pl", "valid_pl", "test_pl"):
                    if _obj in locals():
                        try:
                            del locals()[_obj]
                        except Exception:
                            pass

                gc.collect()

                if torch is not None and self.device != "cpu":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        history_df = hist.to_frame()
        logger.info("RollingTrainer run end | history_shape=%s", history_df.shape)

        # 最终模型保存：selector 可能在 valid 为空时无候选
        # 但如果 strategy=last，我们就用 self.get_last_candidate() 兜底保存
        if self.model_save_enabled:
            strategy = str(self.model_save_cfg.get("strategy", "last")).lower()

            chosen = None
            if self.model_selector is not None:
                try:
                    chosen = self.model_selector.select()
                except Exception as e:
                    logger.warning("model_save selector.select() failed (%s)", str(e))
                    chosen = None

            if chosen is None and strategy == "last":
                last = self.get_last_candidate()
                if last is None:
                    logger.warning("model_save skipped: no last candidate to save (no successful step).")
                    return history_df

                # 组一个“伪 chosen”，沿用统一保存逻辑
                class _TmpChosen:
                    def __init__(self, d):
                        self.step_id = d.get("step_id")
                        self.fold = d.get("fold")
                        self.valid_score = None
                        self.model = d.get("model")
                        self.feature_cols = d.get("feature_cols", [])
                        self.meta = d.get("meta", {})
                        self.params = d.get("params", {})

                chosen = _TmpChosen(last)
                logger.info(
                    "model_save fallback: use LAST candidate | step_id=%s fold=%s",
                    str(chosen.step_id),
                    str(chosen.fold),
                )

            if chosen is None:
                logger.warning("model_save skipped: selector has no candidate and strategy != last.")
                return history_df

            out_dir = str(self.model_save_cfg.get("out_dir", f"./runs/{self.run_id}/model"))
            runs_root = str(self.model_save_cfg.get("runs_root", ""))

            if "{run_id}" in out_dir:
                out_dir = out_dir.replace("{run_id}", self.run_id)
            if runs_root and "{runs_root}" in out_dir:
                out_dir = out_dir.replace("{runs_root}", runs_root)

            filename = str(self.model_save_cfg.get("filename", "model.pkl"))

            bundle_meta = {
                "run_id": self.run_id,
                "selected_strategy": str(self.model_save_cfg.get("strategy", "last")),
                "selected_step_id": chosen.step_id,
                "selected_fold": chosen.fold,
                "selected_valid_score": getattr(chosen, "valid_score", None),
                "model_name": str(self.model_cfg.get("name", "")),
                "params": chosen.params,
                "params_hash": _stable_hash(chosen.params),
                "window_meta": chosen.meta,
            }

            preprocess_state = chosen.meta.get("preprocess_state", {}) if isinstance(chosen.meta, dict) else {}

            save_model_bundle(
                out_dir,
                model=chosen.model,
                feature_cols=chosen.feature_cols,
                preprocess_state=preprocess_state,
                bundle_meta=bundle_meta,
                filename=filename,
            )


        return history_df

    # ---------------------------------------------------------------------
    # External-orchestrator helper
    # ---------------------------------------------------------------------
    def get_last_candidate(self) -> Optional[Dict[str, Any]]:
        """拿到“最后一个成功 step”训练出来的模型候选信息（model / feature_cols / params / meta / step_id / fold）。
        给 sweep 外层用，避免依赖 valid selection。

        Returns:
            None if no step has been successfully finished; otherwise a dict with:
              - model: trained model object
              - feature_cols: list[str]
              - params: dict
              - meta: dict
              - step_id: int
              - fold: int
        """
        if self._last_model is None:
            return None
        return {
            "model": self._last_model,
            "feature_cols": self._last_feature_cols or [],
            "params": self._last_params or {},
            "meta": self._last_meta or {},
            "step_id": self._last_step_id,
            "fold": self._last_fold,
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _is_empty_payload(self, pl: Optional[Dict[str, Any]]) -> bool:
        """
        判断 payload 是否为空（用于 valid_ratio=0 的场景）。

        规则：
        - pl is None -> empty
        - y 不存在 / len(y)==0 -> empty
        - 如果存在 X，则 X.shape[0]==0 -> empty
        - 如果存在 X_seq，则 X_seq.shape[0]==0 -> empty
        """
        if pl is None:
            return True
        if not isinstance(pl, dict):
            return True

        y = pl.get("y", None)
        if y is None:
            return True
        try:
            if len(y) == 0:
                return True
        except Exception:
            return True

        if "X" in pl:
            X = pl.get("X", None)
            if X is None:
                return True
            try:
                return getattr(X, "shape", (0,))[0] == 0
            except Exception:
                return True

        if "X_seq" in pl:
            Xs = pl.get("X_seq", None)
            if Xs is None:
                return True
            try:
                return getattr(Xs, "shape", (0,))[0] == 0
            except Exception:
                return True

        # 走到这里说明：y 非空，且 (X 或 X_seq) 如果存在也非空。
        return False

    def _is_nn_payload(self, payload: Dict[str, Any]) -> bool:
        """nn payload 必须包含 'X_seq'。"""
        return "X_seq" in payload

    def _fit_one_window(
        self,
        train_pl: Dict[str, Any],
        valid_pl: Optional[Dict[str, Any]],
        model_params: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """在一个 rolling window 上训练一次模型（tree 或 nn）。"""
        model_family = self.model_cfg.get("family", None)
        if model_family is None:
            model_family = "nn" if self._is_nn_payload(train_pl) else "tree"

        if model_family == "nn":
            return self._fit_nn(train_pl, valid_pl, model_params)
        if model_family == "tree":
            return self._fit_tree(train_pl, valid_pl, model_params)
        raise ValueError(f"Unknown model family '{model_family}', expected 'nn' or 'tree'.")

    def _fit_tree(
        self,
        train_pl: Dict[str, Any],
        valid_pl: Optional[Dict[str, Any]],
        model_params: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        训练 tree 模型（LGBM/XGB/CatBoost）。

        - valid 非空：尽量用 eval_set（如果 wrapper 支持）
        - valid 为空：不传 eval_set
        """
        from ret_pred.tree_models.builder import build_tree_model
        from ret_pred.losses.objective_map import apply_tree_objective_from_loss

        cfg = copy.deepcopy(self.model_cfg)
        cfg["model_config"] = copy.deepcopy(model_params)
        cfg["params"] = copy.deepcopy(model_params)

        user_loss_cfg = self.loss_cfg if isinstance(self.loss_cfg, dict) else None
        if user_loss_cfg and user_loss_cfg.get("enabled", False):
            model_name = str(cfg.get("name", "")).lower()
            mapped = apply_tree_objective_from_loss(
                model_name=model_name,
                model_params=cfg.get("model_config", {}) or {},
                loss_cfg=user_loss_cfg,
            )
            cfg["model_config"] = mapped
            cfg["params"] = copy.deepcopy(mapped)

        model = build_tree_model(cfg)

        X_train = train_pl["X"]
        y_train = train_pl["y"]

        has_valid = not self._is_empty_payload(valid_pl)
        fit_info: Dict[str, Any] = {"used_eval_set": False}

        if has_valid:
            X_valid = valid_pl["X"]  # type: ignore[index]
            y_valid = valid_pl["y"]  # type: ignore[index]
            try:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
                fit_info["used_eval_set"] = True
            except TypeError:
                model.fit(X_train, y_train)
                fit_info["used_eval_set"] = False
        else:
            model.fit(X_train, y_train)

        return model, fit_info

    def _fit_nn(
        self,
        train_pl: Dict[str, Any],
        valid_pl: Optional[Dict[str, Any]],
        model_params: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """训练 NN 模型（PyTorch），当前仅用 train_pl。"""
        if torch is None:
            raise ImportError("PyTorch not available, cannot train nn models.")

        from ret_pred.nn_models.builder import build_nn_model
        from ret_pred.losses.builder import build_loss

        model_name = self.model_cfg["name"]
        model = build_nn_model({"name": model_name, "params": model_params})
        model.to(self.device)
        model.train()

        loss_cfg = self.loss_cfg or {"name": "mse", "params": {}}
        if isinstance(loss_cfg, dict) and ("enabled" not in loss_cfg or loss_cfg.get("enabled", True)):
            criterion = build_loss(loss_cfg["name"], **(loss_cfg.get("params", {}) or {}))
        else:
            criterion = build_loss("mse")

        lr = float(self.nn_fit_cfg.get("lr", 1e-3))
        epochs = int(self.nn_fit_cfg.get("epochs", 5))
        batch_size = int(self.nn_fit_cfg.get("batch_size", 256))
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        X = self._to_tensor(train_pl["X_seq"])
        y = self._to_tensor(train_pl["y"])

        n = X.shape[0]
        train_losses = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=X.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx]
                yb = y[idx]
                pred = model(xb)
                loss = criterion(pred.view_as(yb), yb)

                optim.zero_grad()
                loss.backward()
                optim.step()
                train_losses.append(float(loss.detach().cpu().item()))

        fit_info = {"train_loss": float(np.mean(train_losses)) if train_losses else None}
        return model, fit_info

    def _predict(self, model: Any, payload: Dict[str, Any]) -> np.ndarray:
        """统一预测入口：nn -> _predict_nn；tree -> _predict_tree。"""
        if self._is_nn_payload(payload):
            return self._predict_nn(model, payload)
        return self._predict_tree(model, payload)

    def _predict_tree(self, model: Any, pl: Dict[str, Any]) -> np.ndarray:
        """tree 模型预测（payload['X'] 必须是非空 2D）。"""
        return model.predict(pl["X"])

    def _predict_nn(self, model: Any, payload: Dict[str, Any]) -> np.ndarray:
        """nn 模型预测（torch no_grad），返回 (N,) numpy。"""
        if torch is None:
            raise ImportError("PyTorch not available, cannot predict nn models.")
        model.eval()
        with torch.no_grad():
            X = self._to_tensor(payload["X_seq"])
            pred = model(X).detach().cpu().numpy().reshape(-1)
        return pred

    def _score(self, payload: Dict[str, Any], y_pred: np.ndarray) -> float:
        """
        作用：算当前 payload 的 metric，并统一成“越大越好”的方向。
        从 payload 里拿 y_true = payload["y"]
        keys = payload.get("keys")（rankic 需要它且需要 date_col）
        如果 metric_name == rankic：传 date_col=self.date_col
        否则就是 mse/mae 这种
        最后：maximize=False → 取负号
        """
        y_true = np.asarray(payload["y"]).reshape(-1)
        keys = payload.get("keys", None)

        metric_fn = METRIC_FN[self.metric_name]
        if self.metric_name == "rankic":
            s = metric_fn(y_true, y_pred, keys, date_col=self.date_col)
        else:
            s = metric_fn(y_true, y_pred)

        s = float(s)
        return s if self.maximize else -s

    def _save_preds(
        self,
        payload: Dict[str, Any],
        y_pred: np.ndarray,
        meta: Dict[str, Any],
        part: str,
        params: Dict[str, Any],
    ):
        """把每一步预测结果保存为 parquet（通过 saver）。
        输出 df 列包含：
        keys（如 date/stockid）如果有
        y_true / y_pred
        part / step_id / fold / run_id / model_name
        并在 meta2 里放：
        part
        params_hash（用于追踪本次预测对应的参数）
        返回值是 saver.save 返回的路径或 None。
        """
        if self.saver is None:
            return None

        keys = payload.get("keys", None)
        if keys is None:
            df = pd.DataFrame({
                "y_true": np.asarray(payload["y"]).reshape(-1),
                "y_pred": np.asarray(y_pred).reshape(-1),
            })
        else:
            df = keys.copy()
            df["y_true"] = np.asarray(payload["y"]).reshape(-1)
            df["y_pred"] = np.asarray(y_pred).reshape(-1)

        meta2 = dict(meta)
        meta2["part"] = part
        meta2["params_hash"] = _stable_hash(params)

        df["part"] = part
        df["step_id"] = meta.get("step_id", 0)
        df["fold"] = meta.get("fold", 0)
        df["run_id"] = meta.get("run_id", self.run_id)
        df["model_name"] = self.model_cfg.get("name", "")

        return self.saver.save(df, meta2)

    def _to_tensor(self, x):
        """将输入转为 torch.Tensor 并放到 self.device。"""
        if torch is None:
            raise ImportError("PyTorch not available.")
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(np.asarray(x), dtype=torch.float32)
        return t.to(self.device)
