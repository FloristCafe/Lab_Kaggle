"""Causal-history unit-centric U-Net with global bottleneck context."""

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Network(nn.Module):
    """[K,35,24,24] -> raw logits [K,1,24,24]; no sigmoid/softmax.

    24 -> 12 -> 6 -> 12 -> 24. Distribution normalization belongs to the loss.

    Coordinates occupy channels 33/34, anchor 32 (x/y, indexing='ij'). Global mean
    context is broadcast at the bottleneck so all output cells can receive
    information from the full map, including opposite-corner unit anchors.
    GroupNorm avoids replay-dependent running statistics in small batches.
    """

    def __init__(self, in_channels=35):
        super().__init__()
        self.in_channels = in_channels
        self.encoder24 = DoubleConv(in_channels, 32)
        self.down12 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.encoder12 = DoubleConv(64, 64)
        self.down6 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bottleneck = DoubleConv(128, 128)
        self.global_context = nn.Sequential(nn.Conv2d(128, 128, 1), nn.SiLU())
        self.up12 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder12 = DoubleConv(64 + 64, 64)
        self.up24 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder24 = DoubleConv(32 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (self.in_channels, 24, 24):
            raise ValueError(f'expected features [K,{self.in_channels},24,24]')
        skip24 = self.encoder24(features)
        skip12 = self.encoder12(self.down12(skip24))
        deep = self.bottleneck(self.down6(skip12))
        deep = deep + self.global_context(deep.mean(dim=(2, 3), keepdim=True))
        up12 = self.up12(deep)
        up12 = self.decoder12(torch.cat((up12, skip12), dim=1))
        up24 = self.up24(up12)
        return self.head(self.decoder24(torch.cat((up24, skip24), dim=1)))
