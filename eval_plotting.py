"""Create PSNR plots from a saved adaptive-evaluation CSV file."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _within(frame: pd.DataFrame, column: str, value_range: list[float]) -> pd.DataFrame:
    low, high = value_range
    return frame.loc[frame[column].between(low, high)].sort_values(column)


def _draw(ax: plt.Axes, frame: pd.DataFrame, x_column: str, title: str, x_label: str) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "No conditions satisfy this filter", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(frame[x_column], frame["mean_psnr_db"], "o-", linewidth=1.5, markersize=4,
                color="#1f77b4", label="Adaptive transmission")
        for method, color in (("semantic", "#d62728"), ("direct", "#2ca02c")):
            selected = frame.loc[frame["method"] == method]
            if not selected.empty:
                ax.scatter(selected[x_column], selected["mean_psnr_db"], s=28, color=color,
                           label=method.title(), zorder=3)
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average PSNR (dB)")
    ax.grid(True, alpha=0.3)


def plot_evaluation_csv(config: dict) -> Path:
    """Read YAML-selected evaluation CSV and save the required three-panel plot."""
    plot_cfg = config["eval_plot"]
    if not plot_cfg.get("enabled", False):
        print("Plotting is disabled by eval_plot.enabled.")
        return Path()
    csv_path = Path(plot_cfg["eval_results"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    results = pd.read_csv(csv_path)
    required = {"snr_db", "distance_m", "bandwidth_bps", "mean_psnr_db", "method"}
    if not required.issubset(results.columns):
        raise ValueError(f"Evaluation CSV must contain {sorted(required)}; got {list(results.columns)}")
    snr_range = plot_cfg["snr_db_range"]
    distance_range = plot_cfg["distance_m_range"]
    bandwidth_range = plot_cfg["bandwidth_bps_range"]
    snr_frame = _within(_within(results, "distance_m", distance_range), "bandwidth_bps", bandwidth_range)
    distance_frame = _within(_within(results, "snr_db", snr_range), "bandwidth_bps", bandwidth_range)
    bandwidth_frame = _within(_within(results, "snr_db", snr_range), "distance_m", distance_range)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    _draw(axes[0], snr_frame, "snr_db", f"PSNR vs SNR\nd={distance_range}, BW={bandwidth_range}", "SNR (dB)")
    _draw(axes[1], distance_frame, "distance_m", f"PSNR vs Distance\nSNR={snr_range}, BW={bandwidth_range}", "Distance (m)")
    _draw(axes[2], bandwidth_frame, "bandwidth_bps", f"PSNR vs Bandwidth\nSNR={snr_range}, d={distance_range}", "Bandwidth (bps)")
    output_dir = Path(plot_cfg.get("directory", "graphs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"adaptive_psnr_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved graph: {image_path}")
    return image_path
