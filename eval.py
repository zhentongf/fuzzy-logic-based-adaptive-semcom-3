"""Adaptive evaluation: fuzzy selection between Deep-JSCC and direct image transmission."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from channel import NakagamiChannel
from fuzzy_controller import FuzzyTransmissionController
from model import DeepJSCC
from train import get_device, set_seed


DISTANCE_BAND_METERS = 5.0
DISTANCE_BAND_COUNT = 80
PATH_LOSS_EXPONENTS_CSV = Path(__file__).with_name("list_e.csv")


def load_path_loss_exponents(path: Path = PATH_LOSS_EXPONENTS_CSV) -> tuple[list[float], list[float]]:
    """Load the 80 direct and semantic path-loss exponents from ``list_e.csv``."""
    try:
        exponent_frame = pd.read_csv(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Path-loss exponent CSV not found: {path}") from error

    required_columns = {"e_dir", "e_sem"}
    if not required_columns.issubset(exponent_frame.columns):
        raise ValueError(
            f"{path} must contain columns {sorted(required_columns)}; "
            f"got {list(exponent_frame.columns)}"
        )
    if len(exponent_frame) != DISTANCE_BAND_COUNT:
        raise ValueError(
            f"{path} must contain {DISTANCE_BAND_COUNT} rows for "
            f"{DISTANCE_BAND_METERS:g} m distance bands; got {len(exponent_frame)}."
        )

    direct = pd.to_numeric(exponent_frame["e_dir"], errors="raise").to_numpy(dtype=float)
    semantic = pd.to_numeric(exponent_frame["e_sem"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(direct).all() or not np.isfinite(semantic).all() or (direct <= 0.0).any() or (semantic <= 0.0).any():
        raise ValueError("All path-loss exponents in list_e.csv must be finite and positive.")
    return direct.tolist(), semantic.tolist()


def path_loss_exponent_for_distance(exponents: list[float], distance_m: float) -> tuple[float, int]:
    """Select a path-loss exponent for one of the 5 m distance bands.

    The supplied 80 values cover 0--400 m. Values outside that interval use
    the nearest endpoint band, so evaluation remains defined for all inputs.
    """
    if len(exponents) != DISTANCE_BAND_COUNT:
        raise ValueError(
            f"path-loss exponent lists must contain {DISTANCE_BAND_COUNT} values "
            f"for {DISTANCE_BAND_METERS:g} m bands; got {len(exponents)}."
        )
    if any(float(exponent) <= 0.0 for exponent in exponents):
        raise ValueError("path-loss exponents must all be positive.")
    band_index = min(max(int(float(distance_m) // DISTANCE_BAND_METERS), 0), DISTANCE_BAND_COUNT - 1)
    return float(exponents[band_index]), band_index


def psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    mse = torch.mean((reference - reconstruction).square()).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def evaluate(config: dict) -> pd.DataFrame:
    set_seed(int(config["run"]["seed"]))
    device = get_device(config["run"].get("device", "auto"))
    data_cfg, model_cfg, channel_cfg = config["data"], config["model"], config["channel"]
    conditions = pd.read_csv(data_cfg["eval_csv"])
    required = {"snr_db", "distance_m", "bandwidth_bps"}
    if not required.issubset(conditions.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}; got {list(conditions.columns)}")
    limit = int(data_cfg.get("eval_samples", 100))
    test_set = datasets.CIFAR10(data_cfg["root"], train=False, download=True, transform=transforms.ToTensor())
    images, _ = next(iter(DataLoader(Subset(test_set, range(limit)), batch_size=limit, shuffle=False)))
    images = images.to(device)

    checkpoint_path = Path(model_cfg["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Set run.train=true or provide a trained model.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepJSCC(int(checkpoint["channel_uses"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    controller = FuzzyTransmissionController(config["fuzzy"].get("semantic_threshold", 0.5))
    direct_exponents, semantic_exponents = load_path_loss_exponents()
    # Keep adaptive selection and add fixed baselines for every link condition.
    # This makes the three transmission strategies directly comparable in one CSV.
    transmission_modes = ("adaptive", "all_direct", "all_semantic")
    results: list[dict] = []
    with torch.no_grad():
        for index, condition in enumerate(conditions.to_dict("records")):
            decision = controller.decide(float(condition["snr_db"]), float(condition["distance_m"]),
                                         float(condition["bandwidth_bps"]))
            for transmission_mode in transmission_modes:
                method = (decision["method"] if transmission_mode == "adaptive"
                          else "direct" if transmission_mode == "all_direct" else "semantic")
                path_loss_exponent, distance_band = path_loss_exponent_for_distance(
                    semantic_exponents if method == "semantic" else direct_exponents,
                    float(condition["distance_m"]),
                )
                channel = NakagamiChannel(float(condition["snr_db"]), channel_cfg["nakagami_m"],
                                          path_loss_exponent,
                                          channel_cfg["reference_distance_m"]).to(device)
                if method == "semantic":
                    received = model(images, channel, float(condition["distance_m"]))
                else:
                    received = channel(images, float(condition["distance_m"])).clamp(0.0, 1.0)
                image_mse = (images - received).square().mean(dim=(1, 2, 3))
                image_psnr = 10.0 * torch.log10(1.0 / image_mse.clamp_min(1e-12))
                row = {"sample_id": int(condition.get("sample_id", index)), **condition, **decision,
                       "transmission_mode": transmission_mode, "method": method,
                       "distance_band": distance_band,
                       "path_loss_exponent": path_loss_exponent,
                       "effective_snr_db": channel.effective_snr_db(float(condition["distance_m"])),
                       "transmitted_images": limit, "mean_mse": image_mse.mean().item(),
                       "mean_psnr_db": image_psnr.mean().item()}
                results.append(row)
    result_frame = pd.DataFrame(results)
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"adaptive_evaluation_{timestamp}.csv"
    result_frame.to_csv(result_path, index=False, float_format="%.6f")
    print(f"Evaluated {len(conditions)} link conditions × {len(transmission_modes)} transmission modes × {limit} images.")
    print(f"Mean condition PSNR: {result_frame.mean_psnr_db.mean():.4f} dB")
    print(f"Saved per-condition results: {result_path}")
    return result_frame
