"""Standalone graph entry point for saved adaptive-evaluation results."""
from __future__ import annotations

import argparse
import yaml

from eval_plotting import plot_evaluation_csv
from eval_plotting_2 import plot_distance_by_snr_csv


def load_graph_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if "eval_plot" not in config:
        raise ValueError("Missing YAML section: eval_plot")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot adaptive-evaluation CSV results")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    args = parser.parse_args()
    config = load_graph_config(args.config)
    plot_evaluation_csv(config)
    plot_distance_by_snr_csv(config)


if __name__ == "__main__":
    main()
