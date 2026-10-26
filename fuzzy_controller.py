"""Mamdani fuzzy controller for adaptive semantic/direct transmission."""
from __future__ import annotations

import numpy as np


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("normalization upper bound must exceed lower bound")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return float(x == b and a == b == c)
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def _shoulder_low(x: float) -> float:
    return 1.0 if x <= 0.0 else max(0.0, (0.5 - x) / 0.5)


def _shoulder_high(x: float) -> float:
    return 1.0 if x >= 1.0 else max(0.0, (x - 0.5) / 0.5)


class FuzzyTransmissionController:
    """Three normalized inputs -> channel-quality score in [0, 1].

    Inputs are normalized SNR, normalized distance and normalized bandwidth.
    A low score denotes poor conditions and selects semantic transmission.
    Rules implement: good SNR + near distance + large bandwidth => direct;
    poor SNR + far distance + small bandwidth => semantic.  Intermediate
    combinations are handled by an additional quality rule and centroid
    defuzzification.
    """
    def __init__(self, semantic_threshold: float = 0.5):
        self.semantic_threshold = semantic_threshold
        self.output_grid = np.linspace(0.0, 1.0, 1001)

    @staticmethod
    def _memberships(x: float) -> tuple[float, float, float]:
        return _shoulder_low(x), _tri(x, 0.25, 0.5, 0.75), _shoulder_high(x)

    def score_normalized(self, snr: float, distance: float, bandwidth: float) -> float:
        snr_low, snr_mid, snr_high = self._memberships(snr)
        dist_near, dist_mid, dist_far = self._memberships(distance)
        bw_low, bw_mid, bw_high = self._memberships(bandwidth)
        semantic_strength = max(min(snr_low, dist_far, bw_low),
                                min(snr_low, dist_far), min(snr_low, bw_low), min(dist_far, bw_low))
        direct_strength = min(snr_high, dist_near, bw_high)
        medium_strength = max(min(snr_mid, dist_mid), min(snr_mid, bw_mid), min(dist_mid, bw_mid),
                              min(snr_high, dist_near), min(snr_high, bw_high), min(dist_near, bw_high))
        poor = np.maximum(0.0, (0.5 - self.output_grid) / 0.5)
        medium = np.maximum(0.0, 1.0 - np.abs(self.output_grid - 0.5) / 0.25)
        good = np.maximum(0.0, (self.output_grid - 0.5) / 0.5)
        aggregate = np.maximum.reduce([np.minimum(semantic_strength, poor),
                                       np.minimum(medium_strength, medium),
                                       np.minimum(direct_strength, good)])
        denominator = aggregate.sum()
        return float((aggregate * self.output_grid).sum() / denominator) if denominator else 0.5

    def decide(self, snr_db: float, distance_m: float, bandwidth_bps: float) -> dict[str, float | str]:
        snr = normalize(snr_db, 0.0, 20.0)
        distance = normalize(distance_m, 0.0, 400.0)
        bandwidth = normalize(bandwidth_bps, 10.0, 1000.0)
        score = self.score_normalized(snr, distance, bandwidth)
        method = "semantic" if score < self.semantic_threshold else "direct"
        return {"method": method, "quality_score": score, "snr_normalized": snr,
                "distance_normalized": distance, "bandwidth_normalized": bandwidth}
