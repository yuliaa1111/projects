# ret_pred/evaluate/sweep_compare.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import logging

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _render(s: str, mapping: Dict[str, str]) -> str:
    out = s
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _expand_path(p: str, cfg: Dict[str, Any]) -> str:
    paths = dict(cfg.get("paths", {}) or {})
    mapping = {k: str(v) for k, v in paths.items()}
    return _render(p, mapping)


def _make_grid(
    img_paths: List[Path],
    titles: List[str],
    out_path: Path,
    *,
    ncols: int = 4,
    suptitle: Optional[str] = None,
    dpi: int = 200,
) -> Optional[str]:
    if not img_paths:
        return None

    n = len(img_paths)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 3.4))
    if nrows * ncols == 1:
        axes = [axes]  # type: ignore
    else:
        axes = axes.flatten()  # type: ignore

    for ax in axes:
        ax.axis("off")

    for i, (p, t) in enumerate(zip(img_paths, titles)):
        try:
            img = plt.imread(str(p))
        except Exception:
            continue
        axes[i].imshow(img)
        axes[i].set_title(t, fontsize=10)
        axes[i].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
    return str(out_path)


def run_sweep_compare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sc = dict(cfg.get("sweep_compare", {}) or {})
    if not sc.get("enabled", False):
        return {}

    out_dir = Path(_expand_path(sc.get("out_dir", "{eval_dir}/sweep_compare"), cfg))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_spec = sc.get("summary_path") or "{run_dir}/sweeps/sweep_summary.parquet"
    summary_path = Path(_expand_path(str(summary_spec), cfg))
    if not summary_path.exists():
        raise FileNotFoundError(f"sweep_summary not found: {summary_path}")

    df = pd.read_parquet(summary_path)
    if df.empty:
        logger.warning("sweep_summary empty | path=%s", str(summary_path))
        return {"summary_path": str(summary_path), "out_dir": str(out_dir), "outputs": []}

    # ============ 关键：panel -> 单图文件名映射 ============
    # 你 evaluator 里 stitch 的 order 就是这 8 张图
    panel_to_file = {
        0: "quantile_cum_return_bar.png",
        1: "quantile_mean_return.png",
        2: "rankic_distribution.png",
        3: "scatter_true_vs_pred.png",
        4: "residual_hist.png",
        5: "residual_time_curve.png",
        6: "time_curve_true_vs_pred_by_part.png",
        7: "time_curve_true_vs_pred.png",
    }
    panel_title = {
        0: "Quantile cumulative return (bar)",
        1: "Mean return by quantile (line)",
        2: "RankIC distribution",
        3: "Scatter true vs pred",
        4: "Residual distribution",
        5: "Residual time curve",
        6: "Daily true vs pred by part",
        7: "Daily true vs pred (all parts)",
    }

    panel_idxs = list(sc.get("panel_idxs", [1, 2, 3, 6]))
    ncols = int(sc.get("ncols", 4))
    sort_by = str(sc.get("sort_by", "rankic"))  # 用于排序显示（可选）

    # 排序：优先用 summary 里的 last_test_score / mean_test_score
    # 你 config 写 sort_by=rankic，但 summary 里是 mean_test_score/last_test_score 更稳定
    if sort_by.lower() in ("rankic", "ic", "test_rankic"):
        key = "last_test_score" if "last_test_score" in df.columns else "mean_test_score"
    else:
        key = "last_test_score" if "last_test_score" in df.columns else "mean_test_score"

    if key in df.columns:
        df2 = df.sort_values(key, ascending=False).reset_index(drop=True)
    else:
        df2 = df.copy()

    outputs: List[str] = []

    for pidx in panel_idxs:
        fname = panel_to_file.get(int(pidx))
        if not fname:
            logger.warning("unknown panel_idx=%s, skip", str(pidx))
            continue

        img_paths: List[Path] = []
        titles: List[str] = []

        for _, r in df2.iterrows():
            eval_dir = Path(str(r["eval_dir"]))
            sweep_id = str(r.get("sweep_id", r.get("run_id", "unknown")))
            img = eval_dir / fname

            # 如果你未来把单图放到子目录（比如 eval/panels/），这里也兼容一下
            if not img.exists():
                alt = eval_dir / "panels" / fname
                if alt.exists():
                    img = alt

            if img.exists():
                img_paths.append(img)
                titles.append(sweep_id)

        out_path = out_dir / f"compare_panel_{int(pidx):02d}__{fname}"
        stitched = _make_grid(
            img_paths,
            titles,
            out_path,
            ncols=ncols,
            suptitle=f"SWEEP COMPARE | panel {pidx} | {panel_title.get(int(pidx), fname)}",
            dpi=int(sc.get("dpi", 200)),
        )
        if stitched:
            outputs.append(stitched)
            logger.info("sweep_compare panel done | panel=%s | n=%d | out=%s", str(pidx), len(img_paths), stitched)
        else:
            logger.warning("sweep_compare panel empty | panel=%s | file=%s | (did you keep individual plots?)", str(pidx), fname)

    logger.info("sweep_compare done | out_dir=%s | n_outputs=%d", str(out_dir), len(outputs))
    return {"summary_path": str(summary_path), "out_dir": str(out_dir), "outputs": outputs}
