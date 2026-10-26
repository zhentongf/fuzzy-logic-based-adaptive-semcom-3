"""YAML-driven entry point for the adaptive vehicular semantic communication experiment."""
from __future__ import annotations

import argparse
import yaml

from eval import evaluate
from sim_data_generator import generate_conditions
from train import train_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for name in ("run", "data", "model", "train", "channel", "fuzzy", "output"):
        if name not in config:
            raise ValueError(f"Missing YAML section: {name}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive fuzzy-logic vehicular semantic communication experiment")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--generate-csv", action="store_true", help="Generate the configured condition CSV then exit")
    parser.add_argument("--csv-count", type=int, default=100, help="Rows generated with --generate-csv")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.generate_csv:
        path = generate_conditions(config["data"]["eval_csv"], args.csv_count, config["run"]["seed"])
        print(f"Generated condition CSV: {path}")
        return
    if config["run"].get("train", False):
        train_model(config)
    if config["run"].get("evaluate", False):
        evaluate(config)
    if not config["run"].get("train", False) and not config["run"].get("evaluate", False):
        print("Nothing selected: set run.train and/or run.evaluate to true.")


if __name__ == "__main__":
    main()
