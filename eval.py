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
    results: list[dict] = []
    with torch.no_grad():
        for index, condition in enumerate(conditions.to_dict("records")):
            decision = controller.decide(float(condition["snr_db"]), float(condition["distance_m"]),
                                         float(condition["bandwidth_bps"]))
            channel = NakagamiChannel(float(condition["snr_db"]), channel_cfg["nakagami_m"],
                                      channel_cfg["path_loss_exponent"], channel_cfg["reference_distance_m"]).to(device)
            if decision["method"] == "semantic":
                received = model(images, channel, float(condition["distance_m"]))
            else:
                received = channel(images, float(condition["distance_m"])).clamp(0.0, 1.0)
            image_mse = (images - received).square().mean(dim=(1, 2, 3))
            image_psnr = 10.0 * torch.log10(1.0 / image_mse.clamp_min(1e-12))
            row = {"sample_id": int(condition.get("sample_id", index)), **condition, **decision,
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
    print(f"Evaluated {len(result_frame)} link conditions × {limit} images.")
    print(f"Mean condition PSNR: {result_frame.mean_psnr_db.mean():.4f} dB")
    print(f"Saved per-condition results: {result_path}")
    return result_frame
