# 统一存储路径

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os


def _render(s: str, mapping: Dict[str, str]) -> str:
    out = s
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _to_abs(p: str, base: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((base / pp).resolve())


def resolve_paths(cfg: Dict[str, Any], *, project_root: str | None = None) -> Dict[str, Any]:
    cfg = dict(cfg)
    paths = dict(cfg.get("paths", {}))

    root = Path(project_root or os.getcwd()).resolve()

    runs_root = Path(paths.get("runs_root", "./runs"))
    runs_root = (runs_root if runs_root.is_absolute() else (root / runs_root)).resolve()

    run_id = str(paths.get("run_id", "exp001"))
    run_dir = (runs_root / run_id).resolve()

    dirs = {
        "runs_root": str(runs_root),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_dir": str((run_dir / "model").resolve()),
        "pred_dir": str((run_dir / "preds").resolve()),
        "pred_infer_dir": str((run_dir / "preds_infer").resolve()),  # ✅ NEW
        "eval_dir": str((run_dir / "eval").resolve()),
        "preprocess_dir": str((run_dir / "preprocess").resolve()),
    }

    for k in [
        "run_dir",
        "model_dir",
        "pred_dir",
        "pred_infer_dir",   # ✅ NEW
        "eval_dir",
        "preprocess_dir",
    ]:
        Path(dirs[k]).mkdir(parents=True, exist_ok=True)

    paths.update(dirs)
    cfg["paths"] = paths


    mapping: Dict[str, str] = dict(dirs)

    dl0 = dict(cfg.get("dataloader", {}) or {})
    mapping.setdefault("date_start", str(dl0.get("date_start", "")))
    mapping.setdefault("date_end", str(dl0.get("date_end", "")))

    task0 = dict(cfg.get("task", {}) or {})
    mapping.setdefault("label", str(task0.get("label_col", "y")))
    mapping.setdefault("label_col", str(task0.get("label_col", "y")))

    pr0 = dict(cfg.get("predict", {}) or {})
    mapping.setdefault("asof_date", str(pr0.get("asof_date", "")))
    mapping.setdefault("target_date", str(pr0.get("target_date", "")))

    ptp0 = dict(cfg.get("post_train_predict", {}) or {})
    if not mapping.get("asof_date"):
        mapping["asof_date"] = str(ptp0.get("asof_date", ""))
    if not mapping.get("target_date"):
        mapping["target_date"] = str(ptp0.get("target_date", ""))

    def render_inplace(d: Dict[str, Any], key: str):
        if key in d and isinstance(d[key], str):
            d[key] = _render(d[key], mapping)

    # preprocess
    pp = dict(cfg.get("preprocess", {}) or {})
    render_inplace(pp, "save_path")
    render_inplace(pp, "state_path")
    cfg["preprocess"] = pp

    # evaluate
    ev = dict(cfg.get("evaluate", {}) or {})
    evp = dict(ev.get("params", {}) or {})
    render_inplace(evp, "pred_dir")
    render_inplace(evp, "out_dir")
    ev["params"] = evp
    cfg["evaluate"] = ev

    # predict
    pr = dict(cfg.get("predict", {}) or {})
    render_inplace(pr, "bundle_dir")
    render_inplace(pr, "out_path")
    cfg["predict"] = pr

    # post_train_predict
    ptp = dict(cfg.get("post_train_predict", {}) or {})
    render_inplace(ptp, "bundle_dir")
    render_inplace(ptp, "out_path")
    cfg["post_train_predict"] = ptp

    # trainer.model_save
    tr = dict(cfg.get("trainer", {}) or {})
    trp = dict(tr.get("params", {}) or {})
    ms = trp.get("model_save")
    if isinstance(ms, dict):
        ms = dict(ms)
        render_inplace(ms, "out_dir")
        trp["model_save"] = ms

    # saver.dir
    sv = trp.get("saver")
    if isinstance(sv, dict):
        sv = dict(sv)
        svp = dict(sv.get("params", {}) or {})
        render_inplace(svp, "dir")
        sv["params"] = svp
        trp["saver"] = sv

    tr["params"] = trp
    cfg["trainer"] = tr

    # dataloader
    dl = dict(cfg.get("dataloader", {}) or {})
    for key in ["parquet_dir", "long_path", "long_filename"]:
        if key in dl and isinstance(dl[key], str):
            dl[key] = _to_abs(dl[key], root)
    if "long_paths" in dl and isinstance(dl["long_paths"], list):
        dl["long_paths"] = [_to_abs(p, root) if isinstance(p, str) else p for p in dl["long_paths"]]
    cfg["dataloader"] = dl

    return cfg
