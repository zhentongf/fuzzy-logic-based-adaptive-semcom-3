"""Create binned PSNR comparison plots from an evaluation CSV."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TRANSMISSION_MODES = ("adaptive", "all_direct", "all_semantic")
MODE_LABELS = {
    "adaptive": "Adaptive",
    "all_direct": "All direct",
    "all_semantic": "All semantic",
}
MODE_COLORS = {
    "adaptive": "#1f77b4",
    "all_direct": "#2ca02c",
    "all_semantic": "#d62728",
}
# 圆圈、方块、五角星
MODE_MARKERS = {
    "adaptive": "o",
    "all_direct": "s",
    "all_semantic": "*",
}
# 五角星看起来比其他标记小，适当放大
MODE_MARKERSIZES = {
    "adaptive": 5,
    "all_direct": 5,
    "all_semantic": 9,
}
CONFIDENCE_LEVEL = 0.95


def _validate_results(frame: pd.DataFrame) -> None:
    required = {"snr_db", "distance_m", "bandwidth_bps", "mean_psnr_db", "transmission_mode"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "Evaluation CSV must contain "
            f"{sorted(required)}; got {list(frame.columns)}. "
            "Regenerate results with the current eval.py."
        )


def _confidence_halfwidth(values: pd.Series, confidence: float = CONFIDENCE_LEVEL) -> float:
    """Half-width of the t-based confidence interval of the mean."""
    values = values.dropna()
    n = len(values)
    if n < 2:
        return np.nan
    sem = values.std(ddof=1) / np.sqrt(n)
    return float(stats.t.ppf((1 + confidence) / 2, n - 1) * sem)


def binned_mode_means(
    frame: pd.DataFrame, x_column: str, bin_edges: list[float]
) -> tuple[list[float], dict[str, pd.Series], dict[str, pd.Series]]:
    """Return per-bin mean PSNR and confidence-interval half-width for each mode.

    Bins are left-inclusive, except that the final bin also includes its upper
    endpoint. Thus, for example, SNR bins are [0, 2), ..., [18, 20].
    """
    labels = list(range(len(bin_edges) - 1))
    bins = pd.cut(frame[x_column], bins=bin_edges, labels=labels, right=False, include_lowest=True)
    # ``right=False`` excludes a value exactly equal to the final upper edge.
    bins = bins.astype("object")
    bins.loc[frame[x_column] == bin_edges[-1]] = labels[-1]
    centers = [(low + high) / 2 for low, high in zip(bin_edges[:-1], bin_edges[1:])]
    means: dict[str, pd.Series] = {}
    ci_halfwidths: dict[str, pd.Series] = {}
    for mode in TRANSMISSION_MODES:
        selected = frame.loc[frame["transmission_mode"] == mode].copy()
        selected["_bin"] = bins.loc[selected.index]
        grouped = selected.groupby("_bin", observed=False)["mean_psnr_db"]
        means[mode] = grouped.mean().reindex(labels)
        ci_halfwidths[mode] = grouped.apply(_confidence_halfwidth).reindex(labels)
    return centers, means, ci_halfwidths


def draw_binned_modes(
    ax: plt.Axes, frame: pd.DataFrame, x_column: str, bin_edges: list[float], title: str, x_label: str
) -> None:
    """Draw one ten-point PSNR curve (with 95% CI band) for each transmission mode."""
    centers, mode_means, mode_ci = binned_mode_means(frame, x_column, bin_edges)
    for mode in TRANSMISSION_MODES:
        mean = mode_means[mode].to_numpy(dtype=float)
        ci = mode_ci[mode].to_numpy(dtype=float)
        color = MODE_COLORS[mode]
        ax.plot(centers, mean, linestyle="-", linewidth=1.8,
                marker=MODE_MARKERS[mode], markersize=MODE_MARKERSIZES[mode],
                color=color, label=MODE_LABELS[mode])
        # 样本数不足的 bin 的 ci 为 NaN，fill_between 会在该处自动断开
        ax.fill_between(centers, mean - ci, mean + ci, color=color, alpha=0.2, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average PSNR (dB)")
    ax.set_xticks(centers)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, title=f"{int(CONFIDENCE_LEVEL * 100)}% CI shaded", title_fontsize=7)


def plot_evaluation_csv(config: dict) -> Path:
    """Save PSNR-vs-SNR, distance, and bandwidth binned comparison plots."""
    plot_cfg = config["eval_plot"]
    if not plot_cfg.get("enabled", False):
        print("Plotting is disabled by eval_plot.enabled.")
        return Path()
    csv_path = Path(plot_cfg["eval_results"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    results = pd.read_csv(csv_path)
    _validate_results(results)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    draw_binned_modes(axes[0], results, "snr_db", list(range(0, 21, 2)), "PSNR vs SNR", "SNR (dB)")
    draw_binned_modes(axes[1], results, "distance_m", list(range(0, 401, 40)),
                      "PSNR vs Distance", "Distance (m)")
    draw_binned_modes(axes[2], results, "bandwidth_bps", [10, *range(100, 1001, 100)],
                      "PSNR vs Bandwidth", "Bandwidth (bps)")
    output_dir = Path(plot_cfg.get("directory", "graphs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"adaptive_psnr_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved graph: {image_path}")
    return image_path
