"""Generate uniformly random vehicle-link conditions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_conditions(output_csv: str | Path, count: int = 100, seed: int = 2026) -> Path:
    rng = np.random.default_rng(seed)
    # Independent uniform draws give every value in each specified range the
    # same probability of appearing in the simulated link-condition dataset.
    def values(low: float, high: float) -> np.ndarray:
        return rng.uniform(low, high, size=count)
    frame = pd.DataFrame({"sample_id": np.arange(count), "snr_db": values(0, 20),
                          "distance_m": values(0, 400), "bandwidth_bps": values(10, 1000)})
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.6f")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sim_conditions.csv")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(f"Saved {args.count} conditions to {generate_conditions(args.output, args.count, args.seed)}")
