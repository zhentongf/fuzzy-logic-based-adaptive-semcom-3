"""Visualize the membership functions and response of the fuzzy controller."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fuzzy_controller import FuzzyController, MFS, SCORE_MFS


LEVELS = (0.1, 0.5, 0.9)
TERM_COLORS = {
    "low": "#d95f5f",
    "near": "#4c9f70",
    "medium": "#e6a23c",
    "high": "#4c78a8",
    "far": "#8e63b0",
}


def _output_directory(config: dict) -> Path:
    """Return the configured graph directory and create it when necessary."""
    plot_config = config.get("eval_plot", {})
    output_dir = Path(plot_config.get("directory", "graphs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_membership_functions(config: dict) -> Path:
    """Save a 2x2 overview of all normalized membership functions."""
    x = np.linspace(0.0, 1.0, 1001)
    groups = (
        ("SNR", MFS["snr"]),
        ("Distance", MFS["distance"]),
        ("Bandwidth", MFS["bandwidth"]),
        ("Fuzzy score", SCORE_MFS),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (title, membership_functions) in zip(axes.flat, groups):
        for term, function in membership_functions.items():
            ax.plot(
                x,
                function(x),
                label=term.title(),
                color=TERM_COLORS.get(term),
                linewidth=2.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Membership degree")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Fuzzy Membership Functions", fontsize=16)
    fig.tight_layout()
    image_path = _output_directory(config) / "fuzzy_membership_functions.png"
    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fuzzy membership functions to {image_path}")
    return image_path


def plot_fuzzy_score_grid(config: dict) -> Path:
    """Save three 3x3 score grids for fixed normalized SNR levels.

    Rows vary normalized distance and columns vary normalized bandwidth.  Each
    colored square is the controller output for one of the nine combinations.
    """
    controller = FuzzyController()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    last_image = None

    for ax, snr in zip(axes, reversed(LEVELS)):
        scores = np.array(
            [
                [controller.compute(snr=snr, distance=distance, bandwidth=bandwidth)
                 for bandwidth in LEVELS]
                for distance in LEVELS
            ]
        )
        last_image = ax.imshow(scores, cmap="RdYlGn", vmin=0.0, vmax=1.0, origin="lower")
        for row in range(len(LEVELS)):
            for column in range(len(LEVELS)):
                score = scores[row, column]
                text_color = "white" if score < 0.28 or score > 0.78 else "black"
                ax.text(column, row, f"{score:.2f}", ha="center", va="center",
                        color=text_color, fontsize=12, fontweight="bold")

        ax.set_title(f"SNR = {snr:.1f}")
        ax.set_xticks(range(len(LEVELS)), [f"{value:.1f}" for value in LEVELS])
        ax.set_yticks(range(len(LEVELS)), [f"{value:.1f}" for value in LEVELS])
        ax.set_xlabel("Normalized bandwidth")
        ax.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    axes[0].set_ylabel("Normalized distance")
    fig.suptitle("Fuzzy Scores for Normalized Link Conditions", fontsize=16)
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Fuzzy score")
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.14, top=0.82, wspace=0.18)
    image_path = _output_directory(config) / "fuzzy_score_grids.png"
    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fuzzy score grids to {image_path}")
    return image_path


def plot_fuzzy_visualizations(config: dict) -> tuple[Path, Path]:
    """Generate both fuzzy-controller figures."""
    return plot_membership_functions(config), plot_fuzzy_score_grid(config)

