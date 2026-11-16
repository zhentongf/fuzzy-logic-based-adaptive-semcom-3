"""Generate two-vehicle trajectory data along an OpenStreetMap road route."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np


# Berlin road coordinates in (latitude, longitude) order.  OSRM uses the same
# OpenStreetMap road data but expects coordinates in (longitude, latitude).
START = (52.51628, 13.37771)
END = (52.51628, 13.50000)
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

SPEED_A_MPS = 20.0
SPEED_B_MPS = 22.0
DURATION_SECONDS = 400
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(point_1: tuple[float, float], point_2: tuple[float, float]) -> float:
    """Calculate great-circle distance between two (latitude, longitude) points."""
    lat_1, lon_1 = map(math.radians, point_1)
    lat_2, lon_2 = map(math.radians, point_2)
    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1
    value = (math.sin(delta_lat / 2.0) ** 2
             + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def fetch_osm_route(start: tuple[float, float], end: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Fetch a driving route based on OpenStreetMap data from the public OSRM API."""
    coordinates = f"{start[1]:.6f},{start[0]:.6f};{end[1]:.6f},{end[0]:.6f}"
    query = urlencode({"overview": "full", "geometries": "geojson", "steps": "false"})
    request = Request(
        f"{OSRM_URL}/{coordinates}?{query}",
        headers={"User-Agent": "adaptive-semcom-route-simulation/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)

    if result.get("code") != "Ok" or not result.get("routes"):
        raise RuntimeError(f"OSRM could not find a route: {result.get('message', result.get('code'))}")

    route = result["routes"][0]
    # GeoJSON coordinates are [longitude, latitude]; internally use [latitude, longitude].
    points = np.asarray(route["geometry"]["coordinates"], dtype=float)[:, ::-1]
    return points, float(route["distance"])


def cumulative_route_distances(points: np.ndarray) -> np.ndarray:
    """Return the cumulative geodesic distance at each route point."""
    segment_lengths = [
        haversine_m(tuple(points[index - 1]), tuple(points[index]))
        for index in range(1, len(points))
    ]
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def position_at_distance(points: np.ndarray, cumulative: np.ndarray, distance_m: float) -> np.ndarray:
    """Linearly interpolate a latitude/longitude position along the route."""
    target = float(np.clip(distance_m, 0.0, cumulative[-1]))
    upper = int(np.searchsorted(cumulative, target, side="right"))
    if upper == 0:
        return points[0].copy()
    if upper >= len(points):
        return points[-1].copy()

    lower = upper - 1
    segment_length = cumulative[upper] - cumulative[lower]
    fraction = 0.0 if segment_length == 0.0 else (target - cumulative[lower]) / segment_length
    return points[lower] + fraction * (points[upper] - points[lower])


def simulate_vehicles(points: np.ndarray) -> list[dict[str, float | int]]:
    """Simulate both vehicles at one-second intervals for 400 seconds."""
    cumulative = cumulative_route_distances(points)
    required_distance = SPEED_B_MPS * DURATION_SECONDS
    if cumulative[-1] < required_distance:
        raise ValueError(
            f"The route is only {cumulative[-1]:.1f} m; vehicle B needs at least "
            f"{required_distance:.1f} m for the 400-second simulation."
        )

    records: list[dict[str, float | int]] = []
    for second in range(DURATION_SECONDS + 1):
        position_a = position_at_distance(points, cumulative, SPEED_A_MPS * second)
        position_b = position_at_distance(points, cumulative, SPEED_B_MPS * second)
        records.append({
            "time_s": second,
            "vehicle_a_latitude": float(position_a[0]),
            "vehicle_a_longitude": float(position_a[1]),
            "vehicle_b_latitude": float(position_b[0]),
            "vehicle_b_longitude": float(position_b[1]),
            "distance_between_vehicles_m": haversine_m(tuple(position_a), tuple(position_b)),
        })
    return records


def save_csv(records: list[dict[str, float | int]], path: Path) -> None:
    """Save trajectory records with stable decimal precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: value if key == "time_s" else f"{float(value):.8f}"
                for key, value in record.items()
            })


def save_plot(points: np.ndarray, records: list[dict[str, float | int]], path: Path) -> None:
    """Save the route map and vehicle-separation time series as two subplots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a_lat = np.array([row["vehicle_a_latitude"] for row in records], dtype=float)
    a_lon = np.array([row["vehicle_a_longitude"] for row in records], dtype=float)
    b_lat = np.array([row["vehicle_b_latitude"] for row in records], dtype=float)
    b_lon = np.array([row["vehicle_b_longitude"] for row in records], dtype=float)
    times = np.array([row["time_s"] for row in records], dtype=float)
    separations = np.array([row["distance_between_vehicles_m"] for row in records], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(points[:, 1], points[:, 0], color="#888888", linewidth=3, label="OSM route")
    axes[0].plot(a_lon, a_lat, color="#2878b5", linewidth=1.8, label="Vehicle A (20 m/s)")
    axes[0].plot(b_lon, b_lat, color="#d95319", linewidth=1.8, label="Vehicle B (22 m/s)")
    axes[0].scatter(START[1], START[0], color="green", marker="o", s=70, label="Start", zorder=5)
    axes[0].scatter(END[1], END[0], color="black", marker="*", s=120, label="End", zorder=5)
    axes[0].set_title("Route and vehicle trajectories")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(times, separations, color="#7a3e9d", linewidth=2)
    axes[1].set_title("Distance between vehicles over time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Great-circle distance (m)")
    axes[1].set_xlim(0, DURATION_SECONDS)
    axes[1].grid(alpha=0.3)

    fig.suptitle("OpenStreetMap route simulation", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points, osrm_distance = fetch_osm_route(START, END)
    records = simulate_vehicles(points)
    csv_path = Path("data/sim_track_from_openstreetmap.csv")
    plot_path = Path("graphs/track_and_distance_from_openstreetmap.png")
    save_csv(records, csv_path)
    save_plot(points, records, plot_path)
    print(f"Start (lat, lon): {START}")
    print(f"End   (lat, lon): {END}")
    print(f"OSRM route length: {osrm_distance:.1f} m")
    print(f"Saved {len(records)} records to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
