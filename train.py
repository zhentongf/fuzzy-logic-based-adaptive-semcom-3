"""Training utilities for the Deep-JSCC semantic transmission model."""
from __future__ import annotations

from pathlib import Path
import random
from datetime import datetime
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from channel import NakagamiChannel
from model import DeepJSCC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _checkpoint_path(base_checkpoint: Path, channel_uses: int) -> Path:
    """Add the channel-use count to a checkpoint name to avoid overwrites."""
    return base_checkpoint.with_name(
        f"{base_checkpoint.stem}_c{channel_uses}{base_checkpoint.suffix}"
    )


def train_model(config: dict) -> list[Path]:
    """Train and save one model for each configured channel-use count."""
    set_seed(int(config["run"]["seed"]))
    device = get_device(config["run"].get("device", "auto"))
    data_cfg, train_cfg, model_cfg, channel_cfg = (config["data"], config["train"], config["model"], config["channel"])
    dataset = datasets.CIFAR10(data_cfg["root"], train=True, download=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                        num_workers=train_cfg.get("num_workers", 0), pin_memory=device.type == "cuda")
    training_path_loss_exponent = float(channel_cfg["training_path_loss_exponent"])
    # Cycle through the configured SNR values, changing the channel once per epoch.
    # ``snr_db`` remains a fallback for older configuration files.
    training_snrs = train_cfg.get("dynamic_snr_db")
    if training_snrs is None:
        training_snrs = [train_cfg["snr_db"]]
    if not isinstance(training_snrs, list) or not training_snrs:
        raise ValueError("train.dynamic_snr_db must be a non-empty list of SNR values.")
    training_snrs = [float(snr) for snr in training_snrs]
    channel_uses_values = model_cfg["channel_uses"]
    if not isinstance(channel_uses_values, list) or not channel_uses_values:
        raise ValueError("model.channel_uses must be a non-empty list of positive integers.")
    channel_uses_values = [int(value) for value in channel_uses_values]
    if any(value <= 0 for value in channel_uses_values):
        raise ValueError("model.channel_uses must contain only positive integers.")
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_checkpoint = Path(model_cfg["checkpoint"])
    saved_checkpoints = []

    for channel_uses in channel_uses_values:
        # Start every variant from the same random seed for reproducibility.
        set_seed(int(config["run"]["seed"]))
        model = DeepJSCC(channel_uses).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
        criterion = nn.MSELoss()
        log_path = output_dir / f"train_log_c{channel_uses}_{timestamp}.csv"
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(["epoch", "snr_db", "mse"])

        print(f"\nTraining model with channel_uses={channel_uses}")
        model.train()
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            epoch_snr_db = training_snrs[(epoch - 1) % len(training_snrs)]
            channel = NakagamiChannel(epoch_snr_db, channel_cfg["nakagami_m"],
                                      training_path_loss_exponent,
                                      channel_cfg["reference_distance_m"]).to(device)
            total_loss = 0.0
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                reconstructed = model(images, channel, distance_m=channel_cfg["reference_distance_m"])
                loss = criterion(reconstructed, images)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
            epoch_mse = total_loss / len(dataset)
            with log_path.open("a", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow([epoch, f"{epoch_snr_db:.6f}", f"{epoch_mse:.10f}"])
            print(f"channel_uses={channel_uses} | Epoch {epoch:03d}/{train_cfg['epochs']} | "
                  f"SNR: {epoch_snr_db:g} dB | MSE: {epoch_mse:.6f}")

        checkpoint = _checkpoint_path(base_checkpoint, channel_uses)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "channel_uses": channel_uses,
                    "training_snr_db": training_snrs, "channel": channel_cfg}, checkpoint)
        saved_checkpoints.append(checkpoint)
        print(f"Saved semantic model: {checkpoint}")
        print(f"Saved training log: {log_path}")

    return saved_checkpoints
