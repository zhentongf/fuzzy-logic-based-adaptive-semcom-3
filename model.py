"""Deep-JSCC network adapted from chunbaobao/Deep-JSCC-PyTorch.

The encoder/decoder block layout follows that repository; only the channel is
injected externally so the same distance-aware Nakagami channel can be used in
both the semantic and direct-transmission baselines.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvPReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2), nn.PReLU())
        nn.init.kaiming_normal_(self.block[0].weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransConvPReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 output_padding: int = 0, final: bool = False):
        super().__init__()
        activation: nn.Module = nn.Sigmoid() if final else nn.PReLU()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 5, stride,
                                                       padding=2, output_padding=output_padding), activation)
        if final:
            nn.init.xavier_normal_(self.block[0].weight)
        else:
            nn.init.kaiming_normal_(self.block[0].weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.layers = nn.Sequential(ConvPReLU(3, 16, 2), ConvPReLU(16, 32, 2),
                                    ConvPReLU(32, 32), ConvPReLU(32, 32), ConvPReLU(32, 2 * c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layers(x)
        energy = z.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return z / torch.sqrt(energy)  # unit average power per codeword


class Decoder(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.layers = nn.Sequential(TransConvPReLU(2 * c, 32), TransConvPReLU(32, 32),
                                    TransConvPReLU(32, 32), TransConvPReLU(32, 16, 2, 1),
                                    TransConvPReLU(16, 3, 2, 1, final=True))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class DeepJSCC(nn.Module):
    def __init__(self, channel_uses: int):
        super().__init__()
        self.encoder = Encoder(channel_uses)
        self.decoder = Decoder(channel_uses)

    def forward(self, x: torch.Tensor, channel: nn.Module | None = None,
                distance_m: float | torch.Tensor = 1.0) -> torch.Tensor:
        z = self.encoder(x)
        if channel is not None:
            z = channel(z, distance_m)
        return self.decoder(z)
