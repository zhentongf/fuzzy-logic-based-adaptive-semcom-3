"""Mamdani fuzzy controller for adaptive semantic/direct transmission."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


def normalize(value: float, lower: float, upper: float) -> float:
    """Map a physical input to ``[0, 1]``, clamping values outside the range."""
    if upper <= lower:
        raise ValueError("normalization upper bound must exceed lower bound")
    return float(np.clip((float(value) - lower) / (upper - lower), 0.0, 1.0))


# ---------------------------------------------------------------
# Membership function
# ---------------------------------------------------------------
def tri(x: float | np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership: 0 at a, peak 1 at b, 0 at c."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b != a else (x >= b).astype(float)
    right = (c - x) / (c - b) if c != b else (x <= b).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


# ---------------------------------------------------------------
# Membership functions (edit the shapes here)
# ---------------------------------------------------------------
MFS: dict[str, dict[str, Callable[[float | np.ndarray], np.ndarray]]] = {
    "snr": {
        "low": lambda x: tri(x, -0.5, 0.0, 0.5),
        "medium": lambda x: tri(x, 0.0, 0.5, 1.0),
        "high": lambda x: tri(x, 0.5, 1.0, 1.5),
    },
    "distance": {
        "near": lambda x: tri(x, -0.5, 0.0, 0.5),
        "medium": lambda x: tri(x, 0.0, 0.5, 1.0),
        "far": lambda x: tri(x, 0.5, 1.0, 1.5),
    },
    "bandwidth": {
        "low": lambda x: tri(x, -0.5, 0.0, 0.5),
        "medium": lambda x: tri(x, 0.0, 0.5, 1.0),
        "high": lambda x: tri(x, 0.5, 1.0, 1.5),
    },
}

SCORE_MFS: dict[str, Callable[[float | np.ndarray], np.ndarray]] = {
    "low": lambda x: tri(x, -0.5, 0.0, 0.5),
    "medium": lambda x: tri(x, 0.3, 0.5, 0.7),
    "high": lambda x: tri(x, 0.5, 1.0, 1.5),
}


# ---------------------------------------------------------------
# RULES  (edit / add rules here)
#
# Format: (operator, [(input, term), ...], output_term)
#   operator: "OR"  -> max of the conditions
#             "AND" -> min of the conditions
#
# Rules that share the same output term are combined with max.
# ---------------------------------------------------------------
RULES = [
    # IF snr is low OR distance is far OR bandwidth is low THEN score is low
    ("OR", [("snr", "low"), ("distance", "far"), ("bandwidth", "low")], "low"),
    # IF snr is medium THEN score is medium
    ("OR", [("snr", "medium")], "medium"),
    # IF snr is high OR distance is near OR bandwidth is high THEN score is high
    ("OR", [("snr", "high"), ("distance", "near"), ("bandwidth", "high")], "high"),
]


# ---------------------------------------------------------------
# Controller
# ---------------------------------------------------------------
class FuzzyController:
    """Evaluate normalized link inputs using the configured Mamdani rule base."""

    def __init__(
        self,
        mfs: Mapping[str, Mapping[str, Callable[[float | np.ndarray], np.ndarray]]] = MFS,
        score_mfs: Mapping[str, Callable[[float | np.ndarray], np.ndarray]] = SCORE_MFS,
        rules: list[tuple[str, list[tuple[str, str]], str]] = RULES,
        resolution: int = 1001,
    ) -> None:
        if resolution < 2:
            raise ValueError("resolution must be at least 2")
        self.mfs = mfs
        self.score_mfs = score_mfs
        self.rules = rules
        self.universe = np.linspace(0.0, 1.0, resolution)

    def fuzzify(self, **inputs: float) -> dict[str, dict[str, float]]:
        """Return membership degrees, e.g. ``{'snr': {'low': 0.2}}``."""
        missing = set(self.mfs) - set(inputs)
        if missing:
            raise ValueError(f"missing fuzzy inputs: {sorted(missing)}")
        return {
            name: {term: float(mf(inputs[name])) for term, mf in terms.items()}
            for name, terms in self.mfs.items()
        }

    def evaluate_rules(self, degrees: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Return the maximum firing strength for each output term."""
        strengths = {term: 0.0 for term in self.score_mfs}
        for op, conditions, out_term in self.rules:
            values = [degrees[var][term] for var, term in conditions]
            fire = max(values) if op.upper() == "OR" else min(values)
            strengths[out_term] = max(strengths[out_term], fire)
        return strengths

    def defuzzify(self, strengths: Mapping[str, float]) -> float:
        """Clip output memberships, aggregate with max, then take the centroid."""
        aggregated = np.zeros_like(self.universe)
        for term, strength in strengths.items():
            clipped = np.minimum(self.score_mfs[term](self.universe), strength)
            aggregated = np.maximum(aggregated, clipped)

        area = aggregated.sum()
        if area == 0:
            return 0.5  # no rule fired; neutral fallback
        return float((self.universe * aggregated).sum() / area)

    def compute(self, snr: float, distance: float, bandwidth: float) -> float:
        """Return a normalized quality score: low is poor and high is good."""
        degrees = self.fuzzify(snr=snr, distance=distance, bandwidth=bandwidth)
        return self.defuzzify(self.evaluate_rules(degrees))


class FuzzyTransmissionController:
    """Adapt physical link conditions to semantic or direct transmission.

    ``semantic_threshold`` is the quality-score cutoff: scores below it use
    semantic transmission, while scores at or above it use direct transmission.
    """

    def __init__(self, semantic_threshold: float = 0.5) -> None:
        self.semantic_threshold = float(semantic_threshold)
        self.controller = FuzzyController()

    def score_normalized(self, snr: float, distance: float, bandwidth: float) -> float:
        """Return a quality score for normalized inputs."""
        return self.controller.compute(snr=snr, distance=distance, bandwidth=bandwidth)

    def decide(self, snr_db: float, distance_m: float, bandwidth_bps: float) -> dict[str, float | str]:
        snr = normalize(snr_db, 0.0, 20.0)
        distance = normalize(distance_m, 0.0, 400.0)
        bandwidth = normalize(bandwidth_bps, 10.0, 1000.0)
        quality_score = self.score_normalized(snr, distance, bandwidth)
        method = "semantic" if quality_score < self.semantic_threshold else "direct"
        return {
            "method": method,
            "quality_score": quality_score,
            "snr_normalized": snr,
            "distance_normalized": distance,
            "bandwidth_normalized": bandwidth,
        }


if __name__ == "__main__":
    fc = FuzzyController()
    print(fc.compute(snr=0.9, distance=0.1, bandwidth=0.8))  # good link -> high
    print(fc.compute(snr=0.5, distance=0.5, bandwidth=0.5))  # middling
    print(fc.compute(snr=0.1, distance=0.9, bandwidth=0.2))  # poor link -> low
