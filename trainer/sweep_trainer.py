# ret_pred/trainer/sweep_trainer.py
"""
核心训练存储在def run()里

SweepRollingTrainer：参数组 sweep + walk-forward rolling 训练评估

不做自动调参（tuner），而是“轮询多组固定参数”，
每一组参数都从头到尾跑完整个 rolling windows（walk-forward）训练流程，并产出：
- 每组参数的 rolling history（每 step 的 train/valid/test score）
- 每组参数的预测文件（可选 saver）
- 每组参数的模型保存（可选 model_save）
- 每组参数的 evaluator 评估图与 metrics（可选）
- 最终汇总表 summary_df：方便横向对比参数效果

使用场景：
- 你已经手动准备好了 10~50 组参数（比如不同的 num_leaves/max_depth/min_data_in_leaf...）
- 想在同一段数据、同一套 rolling 规则下，对比每一组参数的整体表现
- 不想引入 CV/自动寻参，而是对比不同风格参数的效果

重要设计点：
- sweep.param_sets 的每一项可能是“结构化对象”，例如：
  {id, objective, model:{family,name,params:{...}}}
  但 RollingTrainer / build_tree_model 期望的是“扁平的模型参数 dict”：
  {learning_rate:..., num_leaves:..., ...}
- 因此这里必须做 **normalize/flatten**，把 param_set 归一化成扁平参数 dict，再写回 model_cfg["model_config"] / model_cfg["params"]。

依赖：
- RollingTrainer：负责单组参数的 rolling 核心逻辑
- build_streaming_windows：每组参数都要重新构建 windows generator（否则 generator 被消费完）
- Evaluator（可选）：每组参数跑完后对 pred_dir 做统一评估/画图

维护建议：
- RollingTrainer 只专注“单次 rolling 引擎”
- SweepRollingTrainer 只负责“多组参数 orchestration”
- 两者职责分离，便于后续扩展（比如新增 “grid_sweep”、“ablation_sweep” 等）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .plugins import _stable_hash
from .rolling_trainer import RollingTrainer

logger = logging.getLogger(__name__)


@dataclass
class SweepOneResult:
    """
    单组参数 sweep 的产出（内部使用）。

    字段:
        sweep_id:   参数组 ID（如 lgbm_01）
        params:     扁平化后的模型参数 dict（真正喂给模型的那份）
        params_hash: params 的稳定 hash（便于追踪、去重）
        run_id:     该参数组对应的 run_id（通常是 base_run_id__sweep_id）
        out_dir:    该参数组输出目录（preds/model/eval/history 等都在这里）
        history_path: rolling history parquet 路径
        metrics:    evaluator 输出的 metrics（dict），若未跑 evaluator 则为空
    """
    sweep_id: str
    params: Dict[str, Any]
    params_hash: str
    run_id: str
    out_dir: str
    history_path: str
    metrics: Dict[str, Any]


class SweepRollingTrainer:
    """
    多参数组 sweep 的 rolling trainer。

    参数（核心）：
        model:
            基础模型配置（family/name 等）。本类会在每个 param_set 上覆盖其 model_config/params。
        sweep:
            sweep 配置字典。至少应包含：
              - param_sets: List[dict]   # 多组参数
            可选：
              - id_prefix: "lgbm"        # 生成 lgbm_01, lgbm_02...
              - merge_with_base: true    # param_set 是“增量”时，是否叠加到 base_params 上（推荐 true）
        metric / maximize / task / loss / nn_fit / saver / model_save:
            会透传给 RollingTrainer（但 tuner/schedule/update_gate 在 sweep 中会强制关闭）
        run_id:
            sweep 的 base run_id（每组参数会变成 run_id__{sweep_id}）
        preprocess_path / folds / model_family / label_col / datacutting_cfg / runs_root:
            用于 build_streaming_windows 的参数（每组参数都要重建 generator）
        evaluate:
            Evaluator 的 params（dict）。若提供，则每组参数跑完会调用 evaluator 输出图/metrics。
        out_dir:
            sweep 总输出目录（每组参数会在 out_dir/{sweep_id}/ 下输出）
        summary_name:
            sweep 汇总 parquet 文件名（保存到 out_dir/summary_name）

    输出：
        run() 返回 summary_df，并会落盘到 out_dir/summary_name
    """

    def __init__(
        self,
        *,
        model: Dict[str, Any],
        sweep: Dict[str, Any],
        metric: str = "rankic",
        maximize: bool = True,
        task: str = "regression",
        loss: Optional[Dict[str, Any]] = None,
        nn_fit: Optional[Dict[str, Any]] = None,
        saver: Optional[Dict[str, Any]] = None,
        model_save: Optional[Dict[str, Any]] = None,
        sample_weighting: Optional[Dict[str, Any]] = None,
        run_id: str = "exp001",
        date_col: str = "date",
        stockid_col: str = "stockid",
        device: str = "cpu",
        seed: int = 42,
        # --- windows build args ---
        preprocess_path: str = "",
        folds: Optional[List[Dict[str, Any]]] = None,
        model_family: str = "tree",
        label_col: str = "y",
        datacutting_cfg: Optional[Dict[str, Any]] = None,
        runs_root: str = "",
        # --- evaluator args ---
        evaluate: Optional[Dict[str, Any]] = None,
        # --- output ---
        out_dir: str = "./runs/exp001/sweeps",
        summary_name: str = "sweep_summary.parquet",
    ):
        self.base_model_cfg = copy.deepcopy(model)
        self.sweep_cfg = copy.deepcopy(sweep)

        self.metric = metric
        self.maximize = bool(maximize)
        self.task = task
        self.loss = copy.deepcopy(loss) if isinstance(loss, dict) else loss
        self.nn_fit = copy.deepcopy(nn_fit) if isinstance(nn_fit, dict) else nn_fit
        self.saver = copy.deepcopy(saver) if isinstance(saver, dict) else saver
        self.model_save = copy.deepcopy(model_save) if isinstance(model_save, dict) else model_save
        self.sample_weighting = copy.deepcopy(sample_weighting) if isinstance(sample_weighting, dict) else sample_weighting

        self.base_run_id = str(run_id)
        self.date_col = date_col
        self.stockid_col = stockid_col
        self.device = device
        self.seed = int(seed)

        self.preprocess_path = str(preprocess_path)
        self.folds = folds or []
        self.model_family = str(model_family)
        self.label_col = str(label_col)
        self.datacutting_cfg = copy.deepcopy(datacutting_cfg) if isinstance(datacutting_cfg, dict) else (datacutting_cfg or {})
        self.runs_root = str(runs_root)

        self.evaluate_cfg = copy.deepcopy(evaluate) if isinstance(evaluate, dict) else None

        self.out_dir = str(out_dir)
        self.summary_name = str(summary_name)

        # sweep.param_sets 必须是 list
        param_sets = self.sweep_cfg.get("param_sets") or []
        if not isinstance(param_sets, list) or len(param_sets) == 0:
            raise ValueError("SweepRollingTrainer requires sweep.param_sets as a non-empty list.")

        # 一些 sweep 控制项
        self.id_prefix = str(self.sweep_cfg.get("id_prefix") or self.base_model_cfg.get("name") or "sweep")
        self.merge_with_base = bool(self.sweep_cfg.get("merge_with_base", True))

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        执行 sweep。

        返回:
            summary_df:
                每行对应一组 param_set 的汇总信息，包含：
                - sweep_id / params_hash / params
                - history_path
                - 一些 rolling 聚合指标（mean_test_score / last_test_score 等）
                - evaluator 的核心 metrics（若启用）
        """
        out_root = Path(self.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        param_sets: List[Dict[str, Any]] = self.sweep_cfg.get("param_sets") or []
        n = len(param_sets)

        logger.info(
            "sweep start | n_param_sets=%d | preprocess=%s | out_dir=%s",
            n, Path(self.preprocess_path).name, str(out_root)
        )

        results: List[SweepOneResult] = []
        summary_rows: List[Dict[str, Any]] = []

        for i, raw_set in enumerate(param_sets, start=1):
            sweep_id = self._make_sweep_id(i, raw_set, n)
            one_dir = out_root / sweep_id
            one_dir.mkdir(parents=True, exist_ok=True)

            # 1) normalize param_set -> flat params
            flat_params = self._normalize_param_set(raw_set)

            # 2) optionally merge with base params (recommended)
            base_params = copy.deepcopy(
                self.base_model_cfg.get("model_config", {}) or self.base_model_cfg.get("params", {}) or {}
            )
            if self.merge_with_base:
                base_params.update(flat_params)
                flat_params = base_params

            params_hash = _stable_hash(flat_params)

            logger.info(
                "sweep param start | %s (%d/%d) | params_hash=%s",
                sweep_id, i, n, params_hash
            )

            # 3) per-param model_cfg override
            model_cfg = copy.deepcopy(self.base_model_cfg)
            model_cfg["model_config"] = copy.deepcopy(flat_params)
            model_cfg["params"] = copy.deepcopy(flat_params)

            # 4) per-param run_id / dirs
            run_id = f"{self.base_run_id}__{sweep_id}"
            preds_dir = one_dir / "preds"
            model_dir = one_dir / "model"
            eval_dir = one_dir / "eval"
            preds_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            # 5) build windows generator (must rebuild every param set)
            windows = self._build_windows()

            # 6) run rolling (force tuner off; sweep only)
            saver_cfg = self._override_saver_dir(self.saver, str(preds_dir))

            # 你可以选择每组参数都保存模型：
            # - 若 model_save.enabled=true，RollingTrainer 会在最后保存 bundle 到 model_dir
            model_save_cfg = self._override_model_save_dir(self.model_save, str(model_dir))

            rolling = RollingTrainer(
                model=model_cfg,
                metric=self.metric,
                maximize=self.maximize,
                task=self.task,
                loss=self.loss,
                nn_fit=self.nn_fit,
                schedule=None,          # sweep 不用 schedule
                tuner={"enabled": False},
                update_gate=None,
                saver=saver_cfg,
                model_save=model_save_cfg,
                sample_weighting=self.sample_weighting,
                run_id=run_id,
                date_col=self.date_col,
                stockid_col=self.stockid_col,
                device=self.device,
                seed=self.seed,
            )

            hist_df = rolling.run(windows)

            history_path = one_dir / "train_history.parquet"
            hist_df.to_parquet(history_path, index=False)

            # 7) aggregate rolling scores
            agg = self._aggregate_history(hist_df)

            # 8) optional evaluator per param set
            metrics: Dict[str, Any] = {}
            if self.evaluate_cfg is not None:
                metrics = self._run_evaluator(
                    pred_dir=str(preds_dir),
                    out_dir=str(eval_dir),
                )

            # 9) collect
            results.append(
                SweepOneResult(
                    sweep_id=sweep_id,
                    params=copy.deepcopy(flat_params),
                    params_hash=params_hash,
                    run_id=run_id,
                    out_dir=str(one_dir),
                    history_path=str(history_path),
                    metrics=copy.deepcopy(metrics),
                )
            )

            row = {
                "sweep_id": sweep_id,
                "run_id": run_id,
                "params_hash": params_hash,
                "params": flat_params,
                "history_path": str(history_path),
                "out_dir": str(one_dir),
                "pred_dir": str(preds_dir),
                "model_dir": str(model_dir),
                "eval_dir": str(eval_dir),
                **agg,
                "eval_metrics": metrics,
            }
            summary_rows.append(row)

            logger.info(
                "sweep param done | %s | mean_test=%.6f | last_test=%.6f",
                sweep_id,
                float(row.get("mean_test_score", np.nan)),
                float(row.get("last_test_score", np.nan)),
            )

        summary_df = pd.DataFrame(summary_rows)

        # 保存汇总
        summary_path = out_root / self.summary_name
        summary_df.to_parquet(summary_path, index=False)
        logger.info("sweep done | summary_shape=%s | saved=%s", summary_df.shape, str(summary_path))

        # 同时保存一个更易读的 json（可选）
        try:
            json_path = out_root / (Path(self.summary_name).stem + ".json")
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "sweep_id": r.sweep_id,
                            "params_hash": r.params_hash,
                            "history_path": r.history_path,
                            "metrics": r.metrics,
                        }
                        for r in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("failed to save sweep summary json (non-critical)")

        return summary_df

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _make_sweep_id(self, i: int, raw_set: Dict[str, Any], n: int) -> str:
        """
        生成 sweep_id。

        优先级：
        1) raw_set["id"] 若存在
        2) {id_prefix}_{i:02d}
        """
        if isinstance(raw_set, dict) and isinstance(raw_set.get("id"), str) and raw_set["id"].strip():
            return raw_set["id"].strip()
        width = max(2, len(str(n)))
        return f"{self.id_prefix}_{i:0{width}d}"

    def _normalize_param_set(self, param_set: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 sweep 的单个 param_set 归一化为「扁平的模型参数 dict」。

        兼容三种写法：
        1) 直接给扁平参数：
           {"learning_rate": 0.05, "n_estimators": 800, ...}

        2) 结构化对象（你现在日志里这种）：
           {
             "id": "lgbm_01",
             "objective": "regression",
             "model": {"family":"tree","name":"lgbm","params": {...}}
           }
           -> 提取 model.params

        3) 结构化对象但用 params/model_config 字段：
           {"params": {...}} 或 {"model_config": {...}}

        返回:
            flat_params (dict): 只包含模型可接受的扁平 key->标量。
        """
        if not isinstance(param_set, dict):
            raise TypeError(f"param_set must be dict, got {type(param_set)}")

        # case 2: {"model": {...}}
        if "model" in param_set and isinstance(param_set["model"], dict):
            m = param_set["model"]
            if isinstance(m.get("params"), dict):
                return dict(m["params"])
            if isinstance(m.get("model_config"), dict):
                return dict(m["model_config"])

        # case 3: {"params": {...}} / {"model_config": {...}}
        if isinstance(param_set.get("params"), dict):
            return dict(param_set["params"])
        if isinstance(param_set.get("model_config"), dict):
            return dict(param_set["model_config"])

        # case 1: assume already flat
        blacklist = {"id", "model", "objective", "name", "family"}
        flat = {k: v for k, v in param_set.items() if k not in blacklist}

        # protect: no dict/list values
        for k, v in list(flat.items()):
            if isinstance(v, (dict, list)):
                raise TypeError(
                    f"param_set contains non-scalar param '{k}'={type(v).__name__}. "
                    f"Please flatten it. Got param_set keys={list(param_set.keys())}"
                )

        return flat

    def _build_windows(self):
        """
        每组参数都要重建 windows generator（generator 一旦消费完就不能复用）。

        这里直接调用你的 build_streaming_windows，参数来自 main.py 传进来的那一套。
        """
        from ret_pred.windows import build_streaming_windows

        return build_streaming_windows(
            preprocess_path=str(self.preprocess_path),
            folds=self.folds,
            model_family=self.model_family,
            label_col=self.label_col,
            datacutting_cfg=self.datacutting_cfg,
            date_col=self.date_col,
            runs_root=str(self.runs_root),
        )

    def _override_saver_dir(self, saver_cfg: Optional[Dict[str, Any]], pred_dir: str) -> Optional[Dict[str, Any]]:
        """
        将 saver 输出目录覆盖到每组参数的 preds_dir 下。

        这样每组参数的预测文件都隔离开，避免互相覆盖。
        """
        if not isinstance(saver_cfg, dict):
            return saver_cfg

        cfg = copy.deepcopy(saver_cfg)
        cfg.setdefault("enabled", True)
        params = cfg.get("params", {}) or {}
        if not isinstance(params, dict):
            params = {}
        params["dir"] = pred_dir
        cfg["params"] = params
        return cfg

    def _override_model_save_dir(self, model_save_cfg: Optional[Dict[str, Any]], model_dir: str) -> Optional[Dict[str, Any]]:
        """
        将 model_save 输出目录覆盖到每组参数的 model_dir 下。

        注意：
        - 如果你 valid_ratio=0，那么 RollingTrainer 的 model_save 里 “best_valid” 策略没有意义。
          此时推荐 strategy="last"（每折/每 step 最后那个）或直接 disable。
        - 这里不强制改 strategy，只改 out_dir，避免覆盖。
        """
        if not isinstance(model_save_cfg, dict):
            return model_save_cfg

        cfg = copy.deepcopy(model_save_cfg)
        cfg.setdefault("enabled", False)

        # 强制隔离目录
        cfg["out_dir"] = model_dir
        return cfg

    def _aggregate_history(self, hist_df: pd.DataFrame) -> Dict[str, Any]:
        """
        从 rolling history 中聚合出一些方便横向对比的 summary 指标。

        说明：
        - valid_score 可能为 None（valid_ratio=0 场景），因此聚合时要自动跳过。
        - train/test 一般都有。

        返回:
            dict: 聚合结果（mean/last 等）
        """
        out: Dict[str, Any] = {}

        def _mean_col(col: str) -> float:
            if col not in hist_df.columns:
                return float("nan")
            s = pd.to_numeric(hist_df[col], errors="coerce")
            return float(np.nanmean(s.values)) if len(s) else float("nan")

        def _last_col(col: str) -> float:
            if col not in hist_df.columns or len(hist_df) == 0:
                return float("nan")
            s = pd.to_numeric(hist_df[col], errors="coerce")
            return float(s.values[-1])

        out["n_steps"] = int(len(hist_df))
        out["mean_train_score"] = _mean_col("train_score")
        out["mean_valid_score"] = _mean_col("valid_score")
        out["mean_test_score"] = _mean_col("test_score")
        out["last_train_score"] = _last_col("train_score")
        out["last_valid_score"] = _last_col("valid_score")
        out["last_test_score"] = _last_col("test_score")
        return out

    def _run_evaluator(self, *, pred_dir: str, out_dir: str) -> Dict[str, Any]:
        """
        对单组参数的 pred_dir 执行 Evaluator。

        参数:
            pred_dir: RollingTrainer saver 输出的 parquet 目录
            out_dir: evaluator 输出图和 metrics 的目录

        返回:
            metrics dict（尽量可 JSON 化）
        """
        from ret_pred.evaluate.evaluator import Evaluator

        ev_params = copy.deepcopy(self.evaluate_cfg) if isinstance(self.evaluate_cfg, dict) else {}
        ev_params["pred_dir"] = pred_dir
        ev_params["out_dir"] = out_dir

        evaluator = Evaluator(**ev_params)
        res = evaluator.run()

        metrics: Dict[str, Any] = {}
        try:
            if res is not None and hasattr(res, "metrics_df") and isinstance(res.metrics_df, pd.DataFrame):
                # metrics_df 通常是 “指标 x part” 或类似结构；这里尽量转成 dict
                metrics = {"metrics_df": res.metrics_df.to_dict(orient="list")}
            else:
                metrics = {}
        except Exception:
            logger.exception("failed to serialize evaluator metrics (non-critical)")
            metrics = {}

        return metrics
