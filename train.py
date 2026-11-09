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


def train_model(config: dict) -> Path:
    """Train on all 50,000 CIFAR-10 training images and save a checkpoint."""
    set_seed(int(config["run"]["seed"]))
    device = get_device(config["run"].get("device", "auto"))
    data_cfg, train_cfg, model_cfg, channel_cfg = (config["data"], config["train"], config["model"], config["channel"])
    dataset = datasets.CIFAR10(data_cfg["root"], train=True, download=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                        num_workers=train_cfg.get("num_workers", 0), pin_memory=device.type == "cuda")
    model = DeepJSCC(model_cfg["channel_uses"]).to(device)
    training_path_loss_exponent = float(channel_cfg["training_path_loss_exponent"])
    # Cycle through the configured SNR values, changing the channel once per epoch.
    # ``snr_db`` remains a fallback for older configuration files.
    training_snrs = train_cfg.get("dynamic_snr_db")
    if training_snrs is None:
        training_snrs = [train_cfg["snr_db"]]
    if not isinstance(training_snrs, list) or not training_snrs:
        raise ValueError("train.dynamic_snr_db must be a non-empty list of SNR values.")
    training_snrs = [float(snr) for snr in training_snrs]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.MSELoss()
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"train_log_{timestamp}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(["epoch", "snr_db", "mse"])
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
        print(f"Epoch {epoch:03d}/{train_cfg['epochs']} | SNR: {epoch_snr_db:g} dB | MSE: {epoch_mse:.6f}")
    checkpoint = Path(model_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "channel_uses": model_cfg["channel_uses"],
                "training_snr_db": training_snrs, "channel": channel_cfg}, checkpoint)
    print(f"Saved semantic model: {checkpoint}")
    print(f"Saved training log: {log_path}")
    return checkpoint
