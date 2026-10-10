"""Resolution-preserving macro-target logits; unit channels use replay slot IDs."""

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
    """[B, 21, 24, 24] -> [B, 16, 24, 24], without normalization at the head."""

    def __init__(self, blocks=4):
        super().__init__()
        if blocks not in (3, 4):
            raise ValueError("blocks must be 3 or 4")
        self.stem = nn.Sequential(
            nn.Conv2d(21, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*(ResidualBlock() for _ in range(blocks)))
        self.head = nn.Conv2d(64, 16, 1)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (21, 24, 24):
            raise ValueError("expected features [B, 21, 24, 24]")
        return self.head(self.backbone(self.stem(features)))
