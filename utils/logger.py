# ret_pred/utils/logger.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict
import os


def _render(s: str, mapping: Dict[str, str]) -> str:
    out = s
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def setup_logging(cfg: Dict[str, Any]) -> str:
    log_cfg = dict(cfg.get("logging", {}) or {})
    paths = dict(cfg.get("paths", {}) or {})

    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = str(log_cfg.get("fmt", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    datefmt = str(log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"))
    console = bool(log_cfg.get("console", True))

    max_bytes = int(log_cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(log_cfg.get("backup_count", 20))

    runs_root = str(paths.get("runs_root", "runs"))
    run_id = str(paths.get("run_id", "exp000"))
    run_dir = str(paths.get("run_dir", str(Path(runs_root) / run_id)))

    mapping = {
        "runs_root": runs_root,
        "run_id": run_id,
        "run_dir": run_dir,
    }

    log_dir_tpl = str(log_cfg.get("log_dir", "{run_dir}/logs"))
    log_dir = _render(log_dir_tpl, mapping)
    filename = str(log_cfg.get("filename", "run.log"))
    log_file = str(Path(log_dir) / filename)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # 关键：避免重复加 handler（你多次运行/Notebook 会重复打印）
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # file handler
    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # 可选：屏蔽 noisy 第三方库
    noisy_loggers = [
        "PIL", "PIL.PngImagePlugin",
        "graphviz", "graphviz._tools",
        "matplotlib",
        "urllib3",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # LightGBM python 包有时也会啰嗦（不一定生效，取决于它的输出通道）
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


    logging.getLogger(__name__).info(
        "logging initialized | level=%s | file=%s",
        level_name,
        log_file,
    )
    return log_file
