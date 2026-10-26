"""Distance-aware Nakagami-m channel used by both transmission schemes."""
from __future__ import annotations

import math
import torch
from torch import nn


class NakagamiChannel(nn.Module):
    """Flat-fading real-valued channel with distance-dependent path loss.

    The Nakagami power gain g follows Gamma(m, 1/m), hence E[g] = 1.  The
    received tensor is sqrt(path_gain * g) * x + n.  Noise is calibrated to
    the requested pre-channel SNR, so distance and fading additionally reduce
    the received SNR in a physically interpretable way.
    """

    def __init__(self, snr_db: float, m: float = 2.0,
                 path_loss_exponent: float = 2.2,
                 reference_distance_m: float = 1.0):
        super().__init__()
        if m <= 0 or path_loss_exponent <= 0 or reference_distance_m <= 0:
            raise ValueError("Nakagami m, path-loss exponent and reference distance must be positive.")
        self.snr_db = float(snr_db)
        self.m = float(m)
        self.path_loss_exponent = float(path_loss_exponent)
        self.reference_distance_m = float(reference_distance_m)

    def path_gain(self, distance_m: float | torch.Tensor) -> torch.Tensor:
        distance = torch.as_tensor(distance_m, dtype=torch.float32)
        safe_distance = torch.clamp(distance, min=self.reference_distance_m)
        return (self.reference_distance_m / safe_distance).pow(self.path_loss_exponent)

    def forward(self, x: torch.Tensor, distance_m: float | torch.Tensor = 1.0) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Channel input must have shape [batch, channels, height, width].")
        batch = x.shape[0]
        distance = torch.as_tensor(distance_m, dtype=x.dtype, device=x.device).reshape(-1)
        if distance.numel() == 1:
            distance = distance.expand(batch)
        if distance.numel() != batch:
            raise ValueError("distance_m must be scalar or have one value per batch item.")

        path_gain = self.path_gain(distance).to(device=x.device, dtype=x.dtype).view(batch, 1, 1, 1)
        gamma = torch.distributions.Gamma(self.m, self.m)
        fading_power = gamma.sample((batch,)).to(device=x.device, dtype=x.dtype).view(batch, 1, 1, 1)
        amplitude = torch.sqrt(path_gain * fading_power)

        signal_power = x.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return amplitude * x + noise

    def effective_snr_db(self, distance_m: float) -> float:
        """Mean received SNR before random fading (for result interpretation)."""
        gain = float(self.path_gain(distance_m))
        return self.snr_db + 10.0 * math.log10(max(gain, 1e-12))
