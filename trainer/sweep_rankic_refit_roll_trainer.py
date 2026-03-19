"""
ret_pred/trainer/sweep_rankic_refit_roll_trainer.py

Sweep trainer for rankic_refit_roll strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import copy
import json
import logging

import numpy as np
import pandas as pd

from .plugins import _stable_hash
from .rankic_refit_roll_trainer import RankICRefitRollTrainer
from ret_pred.evaluate.evaluator import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class SweepRankICOneResult:
    sweep_id: str
    run_id: str
    params_hash: str
    out_dir: str
    history_path: str
    metrics: Dict[str, Any]


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out


class SweepRankICRefitRollTrainer:
    def __init__(
        self,
        *,
        model: Dict[str, Any],
        sweep: Dict[str, Any],
        trainer_params: Dict[str, Any],
        preprocess_path: str,
        datacutting_cfg: Dict[str, Any],
        metric: str = "rankic",
        maximize: bool = True,
        task: str = "regression",
        loss: Optional[Dict[str, Any]] = None,
        nn_fit: Optional[Dict[str, Any]] = None,
        run_id: str = "exp001",
        date_col: str = "date",
        stockid_col: str = "stockid",
        label_col: str = "y",
        device: str = "cpu",
        seed: int = 42,
        evaluate: Optional[Dict[str, Any]] = None,
        out_dir: str = "./runs/exp001/sweeps_rankic",
        summary_name: str = "sweep_rankic_summary.parquet",
    ):
        self.base_model_cfg = copy.deepcopy(model)
        self.sweep_cfg = copy.deepcopy(sweep or {})
        self.base_trainer_params = copy.deepcopy(trainer_params or {})
        self.preprocess_path = str(preprocess_path)
        self.datacutting_cfg = copy.deepcopy(datacutting_cfg or {})

        self.metric = str(metric)
        self.maximize = bool(maximize)
        self.task = str(task)
        self.loss = copy.deepcopy(loss) if isinstance(loss, dict) else loss
        self.nn_fit = copy.deepcopy(nn_fit) if isinstance(nn_fit, dict) else nn_fit

        self.base_run_id = str(run_id)
        self.date_col = str(date_col)
        self.stockid_col = str(stockid_col)
        self.label_col = str(label_col)
        self.device = str(device)
        self.seed = int(seed)

        self.evaluate_cfg = copy.deepcopy(evaluate) if isinstance(evaluate, dict) else None
        self.out_dir = str(out_dir)
        self.summary_name = str(summary_name)

        self.id_prefix = str(self.sweep_cfg.get("id_prefix") or self.base_model_cfg.get("name") or "rankic")
        self.merge_with_base = bool(self.sweep_cfg.get("merge_with_base", True))

        param_sets = self.sweep_cfg.get("param_sets") or []
        if not isinstance(param_sets, list) or len(param_sets) == 0:
            raise ValueError("SweepRankICRefitRollTrainer requires sweep.param_sets as a non-empty list.")

    def run(self, dates: Sequence[Any]) -> pd.DataFrame:
        out_root = Path(self.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        param_sets: List[Dict[str, Any]] = self.sweep_cfg.get("param_sets") or []
        n = len(param_sets)
        logger.info("sweep_rankic start | n_param_sets=%d | out_dir=%s", n, str(out_root))

        rows: List[Dict[str, Any]] = []
        results: List[SweepRankICOneResult] = []

        for i, raw_set in enumerate(param_sets, start=1):
            sweep_id = self._make_sweep_id(i, raw_set, n)
            one_dir = out_root / sweep_id
            one_dir.mkdir(parents=True, exist_ok=True)

            model_override = self._extract_model_override(raw_set)
            model_cfg = self._merge_model_params(model_override)
            params_hash = _stable_hash(model_cfg.get("model_config", {}) or model_cfg.get("params", {}) or {})

            one_tr_params = self._merge_trainer_params(raw_set)
            one_run_id = f"{self.base_run_id}__{sweep_id}"

            saver_cfg = dict(one_tr_params.get("saver", {}) or {})
            if bool(saver_cfg.get("enabled", False)):
                sp = dict(saver_cfg.get("params", {}) or {})
                sp["dir"] = str(one_dir / "preds")
                saver_cfg["params"] = sp
                one_tr_params["saver"] = saver_cfg

            one_tr_params["artifacts_dir"] = str(one_dir)

            trainer = RankICRefitRollTrainer(
                model=model_cfg,
                preprocess_path=self.preprocess_path,
                datacutting_cfg=self.datacutting_cfg,
                metric=str(one_tr_params.get("metric", self.metric)),
                maximize=bool(one_tr_params.get("maximize", self.maximize)),
                task=self.task,
                loss=one_tr_params.get("loss", self.loss),
                nn_fit=one_tr_params.get("nn_fit", self.nn_fit),
                run_id=one_run_id,
                date_col=self.date_col,
                stockid_col=self.stockid_col,
                label_col=self.label_col,
                device=str(one_tr_params.get("device", self.device)),
                seed=self.seed,
                params=one_tr_params,
            )

            hist_df = trainer.run(dates)
            hist_path = one_dir / "train_history.parquet"
            hist_df.to_parquet(hist_path, index=False)

            agg = self._aggregate_history(hist_df)
            metrics: Dict[str, Any] = {}
            if self.evaluate_cfg is not None:
                metrics = self._run_evaluator(pred_dir=str(one_dir / "preds"), out_dir=str(one_dir / "eval"))

            row = {
                "sweep_id": sweep_id,
                "run_id": one_run_id,
                "params_hash": params_hash,
                "out_dir": str(one_dir),
                "pred_dir": str(one_dir / "preds"),
                "eval_dir": str(one_dir / "eval"),
                "history_path": str(hist_path),
                "params": model_cfg.get("model_config", {}) or model_cfg.get("params", {}) or {},
                "trainer_params_override": self._extract_trainer_override(raw_set),
                **agg,
                "eval_metrics": metrics,
            }
            rows.append(row)
            results.append(
                SweepRankICOneResult(
                    sweep_id=sweep_id,
                    run_id=one_run_id,
                    params_hash=params_hash,
                    out_dir=str(one_dir),
                    history_path=str(hist_path),
                    metrics=metrics,
                )
            )

            logger.info(
                "sweep_rankic param done | %s | mean_test=%.6f | mean_future=%.6f",
                sweep_id,
                float(row.get("mean_test_score", np.nan)),
                float(row.get("mean_future_score", np.nan)),
            )

        summary_df = pd.DataFrame(rows)
        summary_path = out_root / self.summary_name
        summary_df.to_parquet(summary_path, index=False)
        logger.info("sweep_rankic done | summary_shape=%s | saved=%s", summary_df.shape, str(summary_path))

        try:
            js = out_root / (Path(self.summary_name).stem + ".json")
            js.write_text(
                json.dumps(
                    [
                        {
                            "sweep_id": r.sweep_id,
                            "run_id": r.run_id,
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
            logger.exception("failed to save sweep_rankic summary json (non-critical)")

        return summary_df

    def _make_sweep_id(self, i: int, raw_set: Dict[str, Any], n: int) -> str:
        if isinstance(raw_set, dict) and isinstance(raw_set.get("id"), str) and raw_set["id"].strip():
            return raw_set["id"].strip()
        width = max(2, len(str(n)))
        return f"{self.id_prefix}_{i:0{width}d}"

    def _extract_model_override(self, raw_set: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_set, dict):
            raise TypeError(f"sweep param_set must be dict, got {type(raw_set)}")
        if isinstance(raw_set.get("model"), dict):
            m = raw_set["model"]
            if isinstance(m.get("params"), dict):
                return dict(m["params"])
            if isinstance(m.get("model_config"), dict):
                return dict(m["model_config"])
        if isinstance(raw_set.get("params"), dict):
            return dict(raw_set["params"])
        if isinstance(raw_set.get("model_config"), dict):
            return dict(raw_set["model_config"])
        blacklist = {"id", "trainer", "trainer_params", "factor_selection", "windowing"}
        flat = {k: v for k, v in raw_set.items() if k not in blacklist and not isinstance(v, (dict, list))}
        return flat

    def _merge_model_params(self, override: Dict[str, Any]) -> Dict[str, Any]:
        model_cfg = copy.deepcopy(self.base_model_cfg)
        base_params = copy.deepcopy(
            model_cfg.get("model_config", {}) or model_cfg.get("params", {}) or {}
        )
        if self.merge_with_base:
            base_params.update(override or {})
            out_params = base_params
        else:
            out_params = dict(override or {})
        model_cfg["model_config"] = copy.deepcopy(out_params)
        model_cfg["params"] = copy.deepcopy(out_params)
        return model_cfg

    def _extract_trainer_override(self, raw_set: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_set, dict):
            return {}
        if isinstance(raw_set.get("trainer_params"), dict):
            return dict(raw_set["trainer_params"])
        if isinstance(raw_set.get("trainer"), dict):
            tr = raw_set["trainer"]
            if isinstance(tr.get("params"), dict):
                return dict(tr["params"])
            return dict(tr)
        out: Dict[str, Any] = {}
        for k in ("windowing", "factor_selection", "loss", "sample_weighting", "saver"):
            if isinstance(raw_set.get(k), dict):
                out[k] = copy.deepcopy(raw_set[k])
        return out

    def _merge_trainer_params(self, raw_set: Dict[str, Any]) -> Dict[str, Any]:
        tr = copy.deepcopy(self.base_trainer_params)
        over = self._extract_trainer_override(raw_set)
        return _deep_merge_dict(tr, over)

    def _aggregate_history(self, hist_df: pd.DataFrame) -> Dict[str, Any]:
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

        out["n_windows"] = int(len(hist_df))
        out["mean_test_score"] = _mean_col("test_score")
        out["mean_future_score"] = _mean_col("future_score")
        out["last_test_score"] = _last_col("test_score")
        out["last_future_score"] = _last_col("future_score")
        return out

    def _run_evaluator(self, *, pred_dir: str, out_dir: str) -> Dict[str, Any]:
        if self.evaluate_cfg is None:
            return {}
        ev = copy.deepcopy(self.evaluate_cfg)
        ev["pred_dir"] = pred_dir
        ev["out_dir"] = out_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator(**ev)
        res = evaluator.run()
        metrics = {}
        if hasattr(res, "metrics_df") and isinstance(res.metrics_df, pd.DataFrame):
            metrics["rows"] = int(len(res.metrics_df))
            metrics["parts"] = res.metrics_df.get("part", pd.Series(dtype=object)).astype(str).tolist()
        return metrics

