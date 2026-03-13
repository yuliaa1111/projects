# ret_pred/evaluate/evaluator.py
# 从 pred_dir 里把一堆 parquet 预测文件读出来，并过滤掉不合格的，比如 infer 文件
# 按 part(train/valid/test) 计算指标（回归 or 分类）
# 生成图（regression 8 张；classification 3 张）
# 回归任务下，把 8 张图按拼成一张长图（并且可选删除单图）

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .tasks import (
    RegressionCfg, ClassificationCfg,
    evaluate_regression, evaluate_classification,
    rankic_series_by_date,
    quantile_mean_realized_return,
    quantile_cumulative_return,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    metrics_df: pd.DataFrame
    fig_paths: List[str]
    pred_df: Optional[pd.DataFrame] = None


class Evaluator:

    def __init__(
        self,
        task: str = "regression",
        pred_dir: str = "./runs/exp001/preds/exp001",
        pred_glob: str = "**/*.parquet",
        out_dir: str = "./runs/exp001/eval",
        save_fig: bool = True,
        parts: Optional[List[str]] = None,
        folds: Optional[List[int]] = None,
        step_range: Optional[List[int]] = None,
        date_col: str = "date",
        daily_agg: str = "mean",  # "mean" or "median"
        regression: Optional[Dict[str, Any]] = None,
        classification: Optional[Dict[str, Any]] = None,
        max_files: Optional[int] = None,
        return_pred_df: bool = False,
        q_bins: int = 10,

        stitch_regression: bool = True,
        stitch_keep_individual: bool = False,
        stitch_filename: str = "regression_all_plots_long.png",
    ):
        self.task = task
        self.pred_dir = pred_dir
        self.pred_glob = pred_glob
        self.out_dir = out_dir
        self.save_fig = bool(save_fig)

        self.parts = parts or ["train", "valid", "test"]
        self.folds = folds
        self.step_range = step_range
        self.date_col = date_col
        self.daily_agg = daily_agg
        self.max_files = max_files
        self.return_pred_df = return_pred_df
        self.q_bins = int(q_bins)

        self.stitch_regression = bool(stitch_regression)
        self.stitch_keep_individual = bool(stitch_keep_individual)
        self.stitch_filename = str(stitch_filename)

        regression = regression or {"metrics": ["rankic", "mse", "mae"], "ic_method": "spearman"}
        classification = classification or {"metrics": ["auc", "f1", "acc"], "threshold": 0.5}

        self.reg_cfg = RegressionCfg(
            metrics=list(regression.get("metrics", ["rankic"])),
            ic_method=str(regression.get("ic_method", "spearman")),
        )
        self.cls_cfg = ClassificationCfg(
            metrics=list(classification.get("metrics", ["auc"])),
            threshold=float(classification.get("threshold", 0.5)),
        )

        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------------
    # Public
    # -------------------------
    def run(self) -> EvalResult:
        logger.info(
            "Evaluator run start | task=%s | pred_dir=%s | pred_glob=%s | out_dir=%s | save_fig=%s | parts=%s | folds=%s | step_range=%s | date_col=%s | q_bins=%d | stitch=%s",
            self.task, self.pred_dir, self.pred_glob, self.out_dir, self.save_fig,
            self.parts, str(self.folds), str(self.step_range), self.date_col, self.q_bins, self.stitch_regression
        )

        try:
            pred_df = self._load_preds()
            logger.info("pred_df loaded | shape=%s | cols=%d", pred_df.shape, pred_df.shape[1])

            # part counts
            try:
                part_counts = pred_df["part"].value_counts().to_dict()
                logger.info("pred_df part counts | %s", part_counts)
            except Exception:
                pass

            metrics_df = self._compute_metrics(pred_df)
            metrics_path = os.path.join(self.out_dir, "metrics.csv")
            metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
            logger.info("metrics saved | path=%s | shape=%s", metrics_path, metrics_df.shape)

            fig_paths: List[str] = []
            if self.save_fig:
                raw_paths = self._make_plots(pred_df)
                logger.info("plots generated | n=%d", len(raw_paths))

                if self.task == "regression" and self.stitch_regression:
                    long_path = os.path.join(self.out_dir, self.stitch_filename)
                    stitched = self._stitch_regression_plots(raw_paths, long_path)

                    if stitched:
                        logger.info("plots stitched | out=%s", stitched)
                    else:
                        logger.warning("plots stitch failed or missing required images | fallback to raw paths")

                    # remove individual pngs if requested
                    if stitched and (not self.stitch_keep_individual):
                        removed = 0
                        for p in raw_paths:
                            try:
                                if p and os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(stitched):
                                    os.remove(p)
                                    removed += 1
                            except Exception:
                                pass
                        logger.info("individual plots removed | removed=%d | keep_individual=%s", removed, self.stitch_keep_individual)

                    fig_paths = [stitched] if stitched else raw_paths
                else:
                    fig_paths = raw_paths

            logger.info("Evaluator run end | figs=%d", len(fig_paths))

            return EvalResult(
                metrics_df=metrics_df,
                fig_paths=fig_paths,
                pred_df=pred_df if self.return_pred_df else None,
            )

        except Exception:
            logger.exception("Evaluator run failed | pred_dir=%s | out_dir=%s", self.pred_dir, self.out_dir)
            raise

    # -------------------------
    # Load
    # -------------------------
    def _load_preds(self) -> pd.DataFrame:
        pattern = os.path.join(self.pred_dir, self.pred_glob)
        files = sorted(glob.glob(pattern, recursive=True))
        if self.max_files is not None:
            files = files[: int(self.max_files)]

        logger.info("load preds | pattern=%s | found_files=%d | max_files=%s", pattern, len(files), str(self.max_files))

        if not files:
            raise FileNotFoundError(f"No parquet files found under: {pattern}")

        dfs = []
        skipped = 0
        for fp in files:
            df = pd.read_parquet(fp)

            # 🔥 关键：跳过没有 y_true/y_pred 的文件（例如 infer 文件）
            if not {"y_true", "y_pred"}.issubset(df.columns):
                logger.info("skip file without y_true/y_pred | file=%s | cols=%s", fp, list(df.columns))
                skipped += 1
                continue

            df["_source_file"] = os.path.basename(fp)
            dfs.append(df)

        logger.info("files loaded | kept=%d | skipped=%d", len(dfs), skipped)

        if not dfs:
            raise ValueError("No valid prediction files with y_true/y_pred found.")

        pred_df = pd.concat(dfs, ignore_index=True)

        if "part" not in pred_df.columns:
            pred_df["part"] = "test"
            logger.warning("pred_df missing 'part' column, defaulting all to 'test'")

        before = len(pred_df)
        pred_df = pred_df[pred_df["part"].isin(self.parts)]
        logger.info("filter by parts | before=%d | after=%d | parts=%s", before, len(pred_df), self.parts)

        if self.folds is not None and "fold" in pred_df.columns:
            before = len(pred_df)
            pred_df = pred_df[pred_df["fold"].isin(self.folds)]
            logger.info("filter by folds | before=%d | after=%d | folds=%s", before, len(pred_df), self.folds)

        if self.step_range is not None and "step_id" in pred_df.columns:
            lo, hi = int(self.step_range[0]), int(self.step_range[1])
            before = len(pred_df)
            pred_df = pred_df[(pred_df["step_id"] >= lo) & (pred_df["step_id"] <= hi)]
            logger.info("filter by step_range | before=%d | after=%d | range=[%d,%d]", before, len(pred_df), lo, hi)

        if self.date_col in pred_df.columns:
            pred_df[self.date_col] = pd.to_datetime(pred_df[self.date_col])
            logger.info("date_col parsed | date_col=%s | min=%s | max=%s",
                        self.date_col, str(pred_df[self.date_col].min()), str(pred_df[self.date_col].max()))
        else:
            logger.warning("date_col not found in pred_df | date_col=%s | available_cols=%s", self.date_col, list(pred_df.columns))

        if len(pred_df) == 0:
            raise ValueError("pred_df becomes empty after filtering (parts/folds/step_range).")

        return pred_df

    # -------------------------
    # Metrics
    # -------------------------
    def _compute_metrics(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for part, g in pred_df.groupby("part"):
            r: Dict[str, Any] = {"part": part, "n": int(len(g))}
            logger.info("compute metrics | part=%s | n=%d", part, int(len(g)))

            if self.task == "regression":
                r.update(evaluate_regression(g, date_col=self.date_col, cfg=self.reg_cfg))
            elif self.task == "classification":
                r.update(evaluate_classification(g, cfg=self.cls_cfg))
            else:
                raise ValueError(f"Unknown task '{self.task}', expected 'regression' or 'classification'.")
            rows.append(r)

        order = {p: i for i, p in enumerate(["train", "valid", "test"])}
        df = pd.DataFrame(rows)
        df["__ord"] = df["part"].map(order).fillna(999).astype(int)
        df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
        return df

    # -------------------------
    # Plots
    # -------------------------
    def _make_plots(self, pred_df: pd.DataFrame) -> List[str]:
        logger.info(
            "make plots | task=%s | date_col_present=%s | parts=%s",
            self.task, (self.date_col in pred_df.columns), sorted(pred_df["part"].unique().tolist()) if "part" in pred_df.columns else None
        )

        paths: List[str] = []

        if self.task == "regression":
            # 8 figures (regression)
            if self.date_col in pred_df.columns:
                paths.append(self._plot_time_curve_all(pred_df))          # time_curve_true_vs_pred.png
                paths.append(self._plot_time_curve_by_part(pred_df))      # time_curve_true_vs_pred_by_part.png
                paths.append(self._plot_residual_time_curve(pred_df))     # residual_time_curve.png
                paths.append(self._plot_quantile_mean_line(pred_df))      # quantile_mean_return.png
                paths.append(self._plot_quantile_cum_bar(pred_df))        # quantile_cum_return_bar.png
                paths.append(self._plot_rankic_distribution(pred_df))     # rankic_distribution.png
            paths.append(self._plot_scatter(pred_df))                     # scatter_true_vs_pred.png
            paths.append(self._plot_residual_hist(pred_df))               # residual_hist.png
        else:
            # classification plots (3)
            paths.append(self._plot_roc(pred_df))
            paths.append(self._plot_pr(pred_df))
            paths.append(self._plot_confusion(pred_df))

        paths2 = [p for p in paths if p]
        logger.info("make plots done | n=%d", len(paths2))
        return paths2

    # -------------------------
    # NEW: stitch helper
    # -------------------------
    def _stitch_regression_plots(self, raw_paths: List[str], out_path: str) -> Optional[str]:
        """
        Stitch the 8 regression pngs into ONE long png, top-to-bottom, in user's specified order.
        It relies on filenames (basenames) to find the right plots.
        """
        order = [
            "quantile_cum_return_bar.png",
            "quantile_mean_return.png",
            "rankic_distribution.png",
            "residual_hist.png",
            "residual_time_curve.png",
            "scatter_true_vs_pred.png",
            "time_curve_true_vs_pred_by_part.png",
            "time_curve_true_vs_pred.png",
        ]

        mp: Dict[str, str] = {}
        for p in raw_paths:
            if not p:
                continue
            mp[os.path.basename(p)] = p

        paths = []
        for name in order:
            p = mp.get(name)
            if p and os.path.isfile(p):
                paths.append(p)

        if not paths:
            logger.warning("stitch skipped | no required images found from raw_paths | raw_n=%d", len(raw_paths))
            return None

        try:
            from PIL import Image  # type: ignore

            imgs = [Image.open(p).convert("RGB") for p in paths]
            widths = [im.size[0] for im in imgs]
            heights = [im.size[1] for im in imgs]

            max_w = int(max(widths))
            total_h = int(sum(heights))

            canvas = Image.new("RGB", (max_w, total_h), color=(255, 255, 255))

            y = 0
            for im in imgs:
                x = (max_w - im.size[0]) // 2
                canvas.paste(im, (x, y))
                y += im.size[1]

            canvas.save(out_path, format="PNG")

            for im in imgs:
                try:
                    im.close()
                except Exception:
                    pass

            logger.info("stitch success | out_path=%s | n_imgs=%d", out_path, len(paths))
            return out_path

        except Exception:
            try:
                import matplotlib.image as mpimg

                arrays = []
                widths = []
                for p in paths:
                    arr = mpimg.imread(p)
                    if arr.ndim == 3 and arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                    arrays.append(arr)
                    widths.append(arr.shape[1])

                max_w = int(max(widths))

                padded = []
                for arr in arrays:
                    h, w = arr.shape[0], arr.shape[1]
                    if w < max_w:
                        pad_w = max_w - w
                        pad = np.ones((h, pad_w, arr.shape[2]), dtype=arr.dtype)
                        arr2 = np.concatenate([arr, pad], axis=1)
                    else:
                        arr2 = arr
                    padded.append(arr2)

                stitched = np.concatenate(padded, axis=0)

                fig = plt.figure(figsize=(max_w / 150.0, stitched.shape[0] / 150.0))
                ax = plt.gca()
                ax.imshow(stitched)
                ax.axis("off")
                fig.tight_layout(pad=0)
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                logger.info("stitch success (mpl fallback) | out_path=%s | n_imgs=%d", out_path, len(paths))
                return out_path
            except Exception:
                logger.exception("stitch failed | out_path=%s", out_path)
                return None

    # -------- regression: (1) True vs Pred (daily aggregated) all in one
    def _plot_time_curve_all(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "time_curve_true_vs_pred.png")
        fig = plt.figure(figsize=(12, 7))
        ax = plt.gca()

        agg_fn = np.mean if self.daily_agg == "mean" else np.median

        for part, g in pred_df.groupby("part"):
            if self.date_col not in g.columns:
                continue
            daily = g.groupby(self.date_col).apply(
                lambda x: pd.Series({
                    "y_true": float(agg_fn(x["y_true"].values)),
                    "y_pred": float(agg_fn(x["y_pred"].values)),
                }),
                include_groups=False,
            ).reset_index().sort_values(self.date_col)

            ax.plot(daily[self.date_col], daily["y_true"], label=f"{part}-true")
            ax.plot(daily[self.date_col], daily["y_pred"], label=f"{part}-pred")

        ax.set_title("True vs Pred (daily aggregated)")
        ax.set_xlabel("date")
        ax.set_ylabel("value")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (2) True vs Pred (daily aggregated) by part (subplots)
    def _plot_time_curve_by_part(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "time_curve_true_vs_pred_by_part.png")

        parts_order = [p for p in ["train", "valid", "test"] if p in pred_df["part"].unique()]
        if not parts_order:
            parts_order = sorted(pred_df["part"].unique())

        fig, axes = plt.subplots(len(parts_order), 1, figsize=(12, 10), sharex=True)
        if len(parts_order) == 1:
            axes = [axes]

        agg_fn = np.mean if self.daily_agg == "mean" else np.median

        for ax, part in zip(axes, parts_order):
            g = pred_df[pred_df["part"] == part]
            if self.date_col not in g.columns:
                continue
            daily = g.groupby(self.date_col).apply(
                lambda x: pd.Series({
                    "y_true": float(agg_fn(x["y_true"].values)),
                    "y_pred": float(agg_fn(x["y_pred"].values)),
                }),
                include_groups=False,
            ).reset_index().sort_values(self.date_col)

            ax.plot(daily[self.date_col], daily["y_true"], label="true")
            ax.plot(daily[self.date_col], daily["y_pred"], label="pred")
            ax.set_title(part)
            ax.set_ylabel("value")
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("date")
        fig.suptitle("True vs Pred (daily aggregated) by part", y=0.995)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (3) Residual (daily aggregated) time curve
    def _plot_residual_time_curve(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "residual_time_curve.png")

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()

        agg_fn = np.mean if self.daily_agg == "mean" else np.median

        for part, g in pred_df.groupby("part"):
            if self.date_col not in g.columns:
                continue
            daily = g.groupby(self.date_col).apply(
                lambda x: float(agg_fn((x["y_true"].values - x["y_pred"].values))),
                include_groups=False,
            ).reset_index(name="resid").sort_values(self.date_col)

            ax.plot(daily[self.date_col], daily["resid"], label=f"{part}-resid")

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title("Residual (daily aggregated): y_true - y_pred")
        ax.set_xlabel("date")
        ax.set_ylabel("residual")
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (4) Mean realized return by predicted quantile (line)
    def _plot_quantile_mean_line(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "quantile_mean_return.png")
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

        for part, g in pred_df.groupby("part"):
            if self.date_col not in g.columns:
                continue
            qdf = quantile_mean_realized_return(
                g,
                date_col=self.date_col,
                true_col="y_true",
                pred_col="y_pred",
                q_bins=self.q_bins,
            )
            ax.plot(qdf["quantile"], qdf["mean_ret"], marker="o", label=part)

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title("Mean realized return by predicted quantile (higher q = higher score)")
        ax.set_xlabel(f"quantile (0=lowest ... {self.q_bins-1}=highest)")
        ax.set_ylabel("mean y_true")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (5) Cumulative returns by predicted quantile (bars, 3 panels)
    def _plot_quantile_cum_bar(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "quantile_cum_return_bar.png")

        parts_order = [p for p in ["train", "valid", "test"] if p in pred_df["part"].unique()]
        if not parts_order:
            parts_order = sorted(pred_df["part"].unique())

        fig, axes = plt.subplots(len(parts_order), 1, figsize=(14, 10), sharex=True)
        if len(parts_order) == 1:
            axes = [axes]

        groups = [f"Group_{i}" for i in range(1, self.q_bins + 1)]

        for ax, part in zip(axes, parts_order):
            g = pred_df[pred_df["part"] == part]
            if self.date_col not in g.columns:
                continue

            cdf = quantile_cumulative_return(
                g,
                date_col=self.date_col,
                true_col="y_true",
                pred_col="y_pred",
                q_bins=self.q_bins,
            )

            vals = cdf["cum_return"].to_numpy()
            x = np.arange(self.q_bins)

            colors = ["#E15759" if v < 0 else "#59A14F" for v in vals]
            bars = ax.bar(x, vals, color=colors, alpha=0.9)

            ax.axhline(0.0, linewidth=1)
            ax.set_ylabel("Cumulative Return")
            ax.set_title(f"{part} — Quantile Portfolio Cumulative Returns (Low → High Score)")

            for rect, v in zip(bars, vals):
                y = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y,
                    f"{v*100:.2f}%",
                    ha="center",
                    va="bottom" if y >= 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        axes[-1].set_xticks(np.arange(self.q_bins))
        axes[-1].set_xticklabels(groups, rotation=0)
        axes[-1].set_xlabel("Quantile Group")

        fig.suptitle("Cumulative Returns by Predicted Quantile (Train / Valid / Test)", y=0.995, fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (6) RankIC distribution
    def _plot_rankic_distribution(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "rankic_distribution.png")
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

        if self.date_col not in pred_df.columns:
            ax.set_title("RankIC distribution (no date column found)")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        for part, g in pred_df.groupby("part"):
            s = rankic_series_by_date(g, date_col=self.date_col, method=self.reg_cfg.ic_method)
            vals = s.to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, density=True, alpha=0.35, label=part)

        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title("RankIC distribution")
        ax.set_xlabel("RankIC")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (7) Scatter y_true vs y_pred
    def _plot_scatter(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "scatter_true_vs_pred.png")
        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()

        df = pred_df
        if len(df) > 200_000:
            df = df.sample(200_000, random_state=42)

        for part, g in df.groupby("part"):
            ax.scatter(g["y_true"], g["y_pred"], s=4, alpha=0.25, label=part)

        lo = float(min(df["y_true"].min(), df["y_pred"].min()))
        hi = float(max(df["y_true"].max(), df["y_pred"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

        ax.set_title("Scatter: y_true vs y_pred")
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # -------- regression: (8) Residual distribution
    def _plot_residual_hist(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "residual_hist.png")
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for part, g in pred_df.groupby("part"):
            resid = (g["y_true"].values - g["y_pred"].values)
            if len(resid) > 200_000:
                resid = np.random.RandomState(42).choice(resid, size=200_000, replace=False)
            ax.hist(resid, bins=80, alpha=0.35, density=True, label=part)

        ax.set_title("Residual distribution (y_true - y_pred)")
        ax.set_xlabel("residual")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---- Classification plots (binary) ----
    def _get_prob(self, df: pd.DataFrame) -> np.ndarray:
        if "y_prob" in df.columns:
            p = df["y_prob"].to_numpy().reshape(-1)
            return np.clip(p, 0.0, 1.0)
        p = df["y_pred"].to_numpy().reshape(-1)
        if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
            p = 1.0 / (1.0 + np.exp(-np.clip(p, -50, 50)))
        return np.clip(p, 0.0, 1.0)

    def _roc_points(self, y_true: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yt = y_true.astype(int).reshape(-1)
        pr = prob.reshape(-1)
        idx = np.argsort(pr)[::-1]
        yt = yt[idx]
        tp = np.cumsum(yt == 1)
        fp = np.cumsum(yt == 0)
        n_pos = max(int(np.sum(yt == 1)), 1)
        n_neg = max(int(np.sum(yt == 0)), 1)
        tpr = tp / n_pos
        fpr = fp / n_neg
        return fpr, tpr

    def _pr_points(self, y_true: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yt = y_true.astype(int).reshape(-1)
        pr = prob.reshape(-1)
        idx = np.argsort(pr)[::-1]
        yt = yt[idx]
        tp = np.cumsum(yt == 1)
        fp = np.cumsum(yt == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(int(np.sum(yt == 1)), 1)
        return recall, precision

    def _plot_roc(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "roc_curve.png")
        fig = plt.figure()
        ax = plt.gca()

        for part, g in pred_df.groupby("part"):
            yt = g["y_true"].to_numpy().reshape(-1).astype(int)
            prob = self._get_prob(g)
            if len(np.unique(yt)) < 2:
                continue
            fpr, tpr = self._roc_points(yt, prob)
            ax.plot(fpr, tpr, label=part)

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title("ROC Curve")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def _plot_pr(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "pr_curve.png")
        fig = plt.figure()
        ax = plt.gca()

        for part, g in pred_df.groupby("part"):
            yt = g["y_true"].to_numpy().reshape(-1).astype(int)
            prob = self._get_prob(g)
            if len(np.unique(yt)) < 2:
                continue
            recall, prec = self._pr_points(yt, prob)
            ax.plot(recall, prec, label=part)

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def _plot_confusion(self, pred_df: pd.DataFrame) -> str:
        out_path = os.path.join(self.out_dir, "confusion_matrix_test.png")
        fig = plt.figure()
        ax = plt.gca()

        df = pred_df[pred_df["part"] == "test"] if "test" in pred_df["part"].unique() else pred_df
        yt = df["y_true"].to_numpy().reshape(-1).astype(int)
        prob = self._get_prob(df)
        thr = float(self.cls_cfg.threshold)
        yh = (prob >= thr).astype(int)

        tp = int(np.sum((yt == 1) & (yh == 1)))
        tn = int(np.sum((yt == 0) & (yh == 0)))
        fp = int(np.sum((yt == 0) & (yh == 1)))
        fn = int(np.sum((yt == 1) & (yh == 0)))

        mat = np.array([[tn, fp], [fn, tp]], dtype=int)

        ax.imshow(mat)
        ax.set_title(f"Confusion Matrix (threshold={thr})")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])

        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
