"""Resolution-preserving destination logits conditioned on one unit anchor."""

from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activation(features + self.layers(features))


class Network(nn.Module):
    """[B, 26, 24, 24] -> [B, 1, 24, 24], without normalization at the head.

    Shared convolutions avoid unit-slot-specific heads. Finite-map zero padding
    still introduces boundary effects; this is not absolute shift equivariance.
    """

    def __init__(self, blocks=4):
        super().__init__()
        if blocks not in (3, 4):
            raise ValueError("blocks must be 3 or 4")
        self.stem = nn.Sequential(
            nn.Conv2d(26, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*(ResidualBlock() for _ in range(blocks)))
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (26, 24, 24):
            raise ValueError("expected features [B, 26, 24, 24]")
        return self.head(self.backbone(self.stem(features)))
