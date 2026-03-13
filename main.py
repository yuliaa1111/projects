"""
ret_pred/main.py

项目主入口（thin main）：负责
- 读取配置（YAML）
- 解析命令行覆盖参数（run_id / runs_root）
- resolve_paths：把 runs_root/run_id 等路径占位符渲染为绝对路径
- 初始化日志（只在 main 里初始化一次）
- 设置随机种子
- 根据 task.mode 调度训练或预测流程

用法:
    # 训练
    python -m ret_pred.main --config ret_pred/config_train.yaml
    python -m ret_pred.main --config ret_pred/config_train_sweep.yaml

    # 预测/推理
    python -m ret_pred.main --config ret_pred/config_pred.yaml

运行模式:
- task.mode=train:
    load_cfg
    resolve_paths
    setup_logging
    seed_all
    run_train:
        A) load_long          : dataloader 读取全区间 long 数据
        B) preprocess_long    : 清洗/标准化/可选落 parquet，并返回 preprocess_state（FIT）
        C) datasplit_long     : 只基于 dates 生成 folds（train/valid/test 日期列表）
        D) build_streaming_windows:
             从 preprocess parquet “流式”读取每个 fold 的窗口数据，切分 train/valid/test，并 yield 给 trainer
        E) trainer.run        : 滚动训练 / sweep 训练，保存模型/预测/历史
        F) post_train_predict : （可选）训练结束后立即做一次预测落盘
        G) Evaluator.run      : （可选）对 pred_dir 做评估并输出图表/指标

- task.mode=predict (或 infer):
    load_cfg
    resolve_paths
    setup_logging
    seed_all
    run_predict_only:
        调用 predictor.run_predict 加载 bundle 推理，并保存预测 parquet

设计原则:
- main 尽量薄：不在顶层导入大量模块，避免启动慢/循环依赖；训练细节下放到 run_train。
- 统一路径管理：先 resolve_paths 再 setup_logging，保证日志文件路径正确。
- 可复现：seed_all 固定随机性（Python random + numpy）。
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import copy
import logging

from ret_pred.paths import resolve_paths


# ======================================================
# Helpers
# ======================================================
def load_cfg(config_path: Path) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    参数:
        config_path (Path): 配置文件路径。

    返回:
        Dict[str, Any]: 解析后的配置 dict。

    异常:
        FileNotFoundError: 配置文件不存在。
        yaml.YAMLError: YAML 格式错误。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    return yaml.safe_load(config_path.read_text())


def seed_all(seed: int) -> None:
    """
    设置随机种子，保证实验可复现（尽量）。

    目前覆盖：
    - Python random
    - numpy random

    说明：
    - 如果引入 torch / lightgbm 的随机性，建议在各自模块里也补充 seed 设置。
    - 这里先保持 main 的职责：只做全局最基础的 seed。

    参数:
        seed (int): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)


# ======================================================
# Train / Predict pipeline
# ======================================================
def run_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练模式总流程调度。

    该函数会根据 trainer.name 分派：
    - rolling：单次滚动训练（RollingTrainer）
    - sweep_rolling：参数组 sweep（SweepRollingTrainer），每组参数都走完整 rolling

    参数:
        cfg (Dict[str, Any]): 完整配置（已 resolve_paths，已初始化 logging）。

    返回:
        Dict[str, Any]: 关键产物路径汇总（history、model_dir、pred_dir、eval_dir 等）。

    异常:
        任何阶段失败会抛出异常给上层 main 统一捕获并打印 traceback。
    """
    logger = logging.getLogger(__name__)

    # 延迟 import：避免 main 启动时就加载一堆依赖
    from ret_pred.dataloader import DLRequest, DLCfg, load_long
    from ret_pred.preprocess import preprocess_long
    from ret_pred.split import datasplit_long
    from ret_pred.windows import build_streaming_windows
    from ret_pred.evaluate.evaluator import Evaluator
    from ret_pred.predictor.rolling_predictor import run_predict

    task_cfg = cfg.get("task", {})
    paths = cfg.get("paths", {})
    dl = cfg["dataloader"]

    # -------------------------
    # A) dataloader
    # -------------------------
    # label_col 优先从 task_cfg 指定；否则从 dataloader.label_name 取；最后 fallback 到 "y"
    label_col = task_cfg.get("label_col", dl.get("label_name", "y"))

    # DLRequest：描述本次要取的数据区间与字段
    dl_req = DLRequest(
        date_start=dl["date_start"],
        date_end=dl["date_end"],
        fields=dl.get("fields", []) or [],
        label_name=dl.get("label_name", label_col),
    )

    # DLCfg：描述数据源与字段约定
    dl_cfg = DLCfg(
        source=dl.get("source", "parquet_dir"),
        parquet_dir=dl.get("parquet_dir", "./data"),
        long_filename=dl.get("long_filename"),
        long_path=dl.get("long_path"),
        date_col=dl.get("date_col", "date"),
        stockid_col=dl.get("stockid_col", "stockid"),
        check_unique_key=dl.get("check_unique_key", True),
        feature_mask_csv=dl.get("feature_mask_csv"),
        feature_col_name=dl.get("feature_col_name", "feature"),
        feature_use_col=dl.get("feature_use_col", "use"),
        # label 可选：由 dataloader 构造（比如 open_to_open / close_to_close）
        build_label=bool(dl.get("build_label", False)),
        label_price_col=dl.get("label_price_col", "open_1d"),
        label_method=dl.get("label_method", "open_to_open"),
        label_log_return=bool(dl.get("label_log_return", False)),
    )

    # 读取全区间 long 数据
    long_df, meta = load_long(dl_req, dl_cfg)
    logger.info("dataloader done | shape=%s", long_df.shape)

    # -------------------------
    # B) preprocess (FIT)
    # -------------------------
    # preprocess_long 通常会：清洗缺失/inf，winsorize，zscore，填充策略等
    # 并返回：
    # - clean_df：清洗后的 long df（通常仍在内存里）
    # - pp_state：训练期(FIT)统计量与配置回显，可用于 transform
    pp_cfg = dict(cfg.get("preprocess", {}))
    pp_cfg["label_col"] = label_col
    pp_cfg.setdefault("save_parquet", True)  # 训练默认落盘，以便后续 streaming windows

    clean_df, pp_state = preprocess_long(long_df, pp_cfg, meta=meta)

    preprocess_path = pp_state.get("saved_path") or pp_state.get("save_path")
    if not preprocess_path:
        raise ValueError("preprocess did not return saved_path; please ensure preprocess.save_parquet=True")

    logger.info("preprocess done | shape=%s", clean_df.shape)
    logger.info("preprocess parquet | path=%s", preprocess_path)

    # -------------------------
    # C) datasplit（dates only）
    # -------------------------
    # 注意：split 只需要 date 序列，不需要全特征列
    # 这样做可以减少在 split 阶段传递/复制大 df 的成本，并且让 split 逻辑更纯粹。
    sp_cfg = dict(cfg.get("datasplit", {}))
    sp_cfg["return_dfs"] = False
    sp_cfg["save_parquet"] = False

    dates_df = clean_df[[dl_cfg.date_col]]
    meta2 = dict(meta, preprocess_state=pp_state, run_id=paths.get("run_id"))

    folds, _ = datasplit_long(dates_df, sp_cfg, meta=meta2)
    logger.info("split done | folds=%d", len(folds))

    # -------------------------
    # D) windows (streaming read from preprocess parquet)
    # -------------------------
    model_cfg = dict(cfg.get("model", {}))
    cut_cfg = dict(cfg.get("datacutting", {}))

    # candidates 机制：如果 config 里把不同模型/不同参数写在 candidates 下，
    # 这里根据 model.name 自动挑一组作为当前参数（同时写入 params/model_config）
    if isinstance(model_cfg.get("candidates"), dict):
        name = model_cfg.get("name")
        if name in model_cfg["candidates"]:
            base_params = dict(model_cfg["candidates"][name] or {})
            model_cfg["model_config"] = copy.deepcopy(base_params)
            model_cfg["params"] = copy.deepcopy(base_params)
            logger.info("model candidates resolved | name=%s | n_params=%d", name, len(base_params))

    # -------------------------
    # D2) choose trainer
    # -------------------------
    tr_cfg = cfg.get("trainer", {})
    tr_name = str(tr_cfg.get("name", "rolling"))
    tr_params = dict(tr_cfg.get("params", {}))

    # =====================================================
    # Trainer: rolling
    # =====================================================
    if tr_name == "rolling":
        from ret_pred.trainer.rolling_trainer import RollingTrainer

        # windows 是 generator：每次 yield 一个 fold 的 train/valid/test payload
        # streaming 的关键：只在需要时读 preprocess parquet 的窗口范围
        windows = build_streaming_windows(
            preprocess_path=str(preprocess_path),
            folds=folds,
            model_family=model_cfg.get("family", "tree"),
            label_col=label_col,
            datacutting_cfg=cut_cfg,
            date_col=dl_cfg.date_col,
            runs_root=str(paths.get("runs_root")),
        )
        logger.info("windows generator built | preprocess_path=%s", preprocess_path)

        trainer = RollingTrainer(
            model=model_cfg,
            metric=tr_params.get("metric", "rankic"),
            maximize=tr_params.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params.get("loss"),
            nn_fit=tr_params.get("nn_fit"),
            schedule=tr_params.get("schedule"),
            tuner=tr_params.get("tuner"),
            update_gate=tr_params.get("update_gate"),
            saver=tr_params.get("saver"),
            model_save=tr_params.get("model_save"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            device=tr_params.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
        )

        # 训练主循环：逐 fold 训练、预测、记录 history
        hist_df = trainer.run(windows)

        # 保存训练历史（用于复盘/画图/调参比较）
        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        hist_df.to_parquet(hist_path, index=False)
        logger.info("train done | history_shape=%s", hist_df.shape)
        logger.info("train_history saved | path=%s", str(hist_path))

        # -------------------------
        # F) post-train predict（可选）
        # -------------------------
        ptp_out = None
        ptp = cfg.get("post_train_predict", {})
        if ptp.get("enabled", False):
            # 复用整份 cfg，但覆盖 predict 段落
            ptp_cfg = dict(cfg)
            ptp_cfg["predict"] = dict(ptp)

            target_date = str(ptp.get("target_date", ""))
            asof_date = str(ptp.get("asof_date", ""))

            # 支持 out_path 模板替换（避免你每次手动写日期）
            if "out_path" in ptp_cfg["predict"]:
                ptp_cfg["predict"]["out_path"] = (
                    ptp_cfg["predict"]["out_path"]
                    .replace("{target_date}", target_date)
                    .replace("{asof_date}", asof_date)
                )

            ptp_cfg["paths"] = cfg["paths"]
            ptp_out = run_predict(ptp_cfg)
            logger.info("post-train predict saved | out=%s", ptp_out)

        # -------------------------
        # G) evaluate（可选/默认）
        # -------------------------
        # evaluator 会从 pred_dir 读取预测 parquet，计算指标并输出图表
        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})
        ev_params.setdefault("pred_dir", cfg["paths"]["pred_dir"])
        ev_params.setdefault("out_dir", cfg["paths"]["eval_dir"])

        evaluator = Evaluator(**ev_params)
        res = evaluator.run()

        logger.info("evaluate done")
        logger.info("metrics:\n%s", res.metrics_df)

        return {
            "train_history": str(hist_path),
            "post_train_pred": ptp_out,
            "eval_dir": cfg["paths"]["eval_dir"],
            "model_dir": cfg["paths"]["model_dir"],
            "pred_dir": cfg["paths"]["pred_dir"],
        }

    # =====================================================
    # Trainer: sweep_rolling
    # =====================================================
    elif tr_name == "sweep_rolling":
        from ret_pred.trainer.sweep_trainer import SweepRollingTrainer

        sweep_cfg = dict(cfg.get("sweep", {}) or {})
        param_sets = sweep_cfg.get("param_sets") or []
        if not isinstance(param_sets, list) or len(param_sets) == 0:
            raise ValueError("trainer.name=sweep_rolling but cfg.sweep.param_sets is empty")

        # evaluate cfg 传给 sweep trainer，让它对每组参数分别评估
        ev = dict(cfg.get("evaluate", {}) or {})
        ev_params = dict(ev.get("params", {}) or {})

        # SweepRollingTrainer 内部会对每组 param_set 重新 build windows（必须！）
        # 因为 windows 是 generator，一旦被消费完就没了。
        trainer = SweepRollingTrainer(
            model=model_cfg,
            sweep=sweep_cfg,
            metric=tr_params.get("metric", "rankic"),
            maximize=tr_params.get("maximize", True),
            task=task_cfg.get("type", "regression"),
            loss=tr_params.get("loss"),
            nn_fit=tr_params.get("nn_fit"),
            saver=tr_params.get("saver"),
            model_save=tr_params.get("model_save"),
            run_id=str(paths.get("run_id")),
            date_col=dl_cfg.date_col,
            stockid_col=dl_cfg.stockid_col,
            device=tr_params.get("device", "cpu"),
            seed=int(task_cfg.get("seed", 42)),
            # windows build args
            preprocess_path=str(preprocess_path),
            folds=folds,
            model_family=model_cfg.get("family", "tree"),
            label_col=label_col,
            datacutting_cfg=cut_cfg,
            runs_root=str(paths.get("runs_root")),
            # evaluate args
            evaluate=ev_params,
            # output
            out_dir=str(Path(paths["run_dir"]) / "sweeps"),
            summary_name=str(tr_params.get("summary_name", "sweep_summary.parquet")),
        )

        summary_df = trainer.run()

        hist_path = Path(paths["run_dir"]) / "train_history.parquet"
        summary_df.to_parquet(hist_path, index=False)
        logger.info("sweep done | summary_shape=%s", summary_df.shape)
        logger.info("sweep summary saved | path=%s", str(hist_path))

        # sweep_compare：对 sweep 结果做对比汇总/画图（可选）
        if cfg.get("sweep_compare", {}).get("enabled", False):
            from ret_pred.evaluate.sweep_compare import run_sweep_compare

            run_sweep_compare(cfg)
            logger.info("sweep_compare done | out_dir=%s", cfg["sweep_compare"]["out_dir"])

        return {
            "train_history": str(hist_path),
            "sweep_dir": str(Path(paths["run_dir"]) / "sweeps"),
            "sweep_compare_dir": str(cfg.get("sweep_compare", {}).get("out_dir", "")),
        }

    else:
        raise KeyError(f"Unknown trainer.name='{tr_name}', expected 'rolling' or 'sweep_rolling'")

    # NOTE: rolling / sweep_rolling 分支内已 return


def run_predict_only(cfg: Dict[str, Any]) -> str:
    """
    只跑预测（task.mode=predict/infer）时的入口。

    逻辑：
    - 填充 predict.bundle_dir 默认指向 cfg["paths"]["model_dir"]
    - 填充 predict.out_path 默认落到 pred_dir 下
    - 调用 predictor.run_predict 执行推理并保存 parquet

    参数:
        cfg (Dict[str, Any]): 完整配置（已 resolve_paths，已初始化 logging）。

    返回:
        str: 预测输出 parquet 路径。
    """
    logger = logging.getLogger(__name__)
    from ret_pred.predictor.rolling_predictor import run_predict

    pr = dict(cfg.get("predict", {}) or {})
    pr.setdefault("bundle_dir", cfg["paths"]["model_dir"])
    pr.setdefault("out_path", str(Path(cfg["paths"]["pred_dir"]) / "pred_{target_date}.parquet"))
    cfg["predict"] = pr

    out_path = run_predict(cfg)
    logger.info("predict saved | out=%s", out_path)
    return str(out_path)


# ======================================================
# main
# ======================================================
def main() -> int:
    """
    CLI 入口。

    命令行参数：
    - --config   : YAML 配置路径
    - --run-id   : 覆盖 paths.run_id（常用于快速起不同实验）
    - --runs-root: 覆盖 paths.runs_root（常用于切换输出根目录）

    返回:
        int: 进程退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(description="ret_pred runner (thin main)")
    parser.add_argument("--config", type=str, default="ret_pred/config.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    args = parser.parse_args()

    # 1) load config
    cfg = load_cfg(Path(args.config))

    # 2) apply CLI overrides before resolve_paths
    #    因为日志路径/产物路径通常依赖 run_id / runs_root
    cfg.setdefault("paths", {})
    if args.run_id is not None:
        cfg["paths"]["run_id"] = str(args.run_id)
    if args.runs_root is not None:
        cfg["paths"]["runs_root"] = str(args.runs_root)

    # 3) resolve all paths placeholders
    cfg = resolve_paths(cfg)

    # 4) init logging (ONLY ONCE)
    #    main 负责初始化 logging，其它模块只需要 logging.getLogger(__name__)
    from ret_pred.utils.logger import setup_logging

    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("config loaded | path=%s", args.config)
    logger.info("CWD=%s", os.getcwd())
    logger.info("runs_root=%s", cfg["paths"]["runs_root"])
    logger.info("run_id=%s", cfg["paths"]["run_id"])
    logger.info("run_dir=%s", cfg["paths"]["run_dir"])

    # 5) seed
    seed = int(cfg.get("task", {}).get("seed", 42))
    seed_all(seed)
    logger.info("seed=%d", seed)

    # 6) mode dispatch
    mode = str(cfg.get("task", {}).get("mode", "train")).lower()
    if mode == "infer":
        mode = "predict"

    try:
        if mode == "predict":
            run_predict_only(cfg)
            return 0

        run_train(cfg)
        return 0

    except Exception:
        logger.exception("Unhandled exception in main")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
