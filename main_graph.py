"""Standalone graph entry point for saved adaptive-evaluation results."""
from __future__ import annotations

import argparse
import yaml

from eval_plotting import plot_evaluation_csv


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
    plot_evaluation_csv(load_graph_config(args.config))


if __name__ == "__main__":
    main()
