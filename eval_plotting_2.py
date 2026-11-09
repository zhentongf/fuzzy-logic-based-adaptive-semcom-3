"""Compare PSNR-vs-distance curves across four SNR ranges."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eval_plotting import _validate_results, draw_binned_modes


SNR_RANGES = ((0, 5), (5, 10), (10, 15), (15, 20))
DISTANCE_BINS = list(range(0, 401, 40))


def plot_distance_by_snr_csv(config: dict) -> Path:
    """Save four PSNR-vs-distance plots, one for each five-dB SNR interval."""
    plot_cfg = config["eval_plot"]
    if not plot_cfg.get("enabled", False):
        print("Plotting is disabled by eval_plot.enabled.")
        return Path()
    csv_path = Path(plot_cfg["eval_results"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    results = pd.read_csv(csv_path)
    _validate_results(results)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (low, high) in zip(axes.flat, SNR_RANGES):
        # The final interval includes 20 dB; other intervals are [low, high).
        if high == 20:
            subset = results.loc[results["snr_db"].between(low, high)]
        else:
            subset = results.loc[(results["snr_db"] >= low) & (results["snr_db"] < high)]
        draw_binned_modes(ax, subset, "distance_m", DISTANCE_BINS,
                          f"PSNR vs Distance (SNR {low}\N{EN DASH}{high} dB)", "Distance (m)")
    output_dir = Path(plot_cfg.get("directory", "graphs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"distance_psnr_under_different_snr_conditions_{timestamp}.png"
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved graph: {image_path}")
    return image_path
