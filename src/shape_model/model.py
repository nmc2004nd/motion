"""CNN backbones for shape classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    out_dim = 256

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(in_channels, 32, kernel=5),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 256, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.features(x)).flatten(1)


class _ResNet18Backbone(nn.Module):
    out_dim = 512

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShapeClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = "small_cnn",
        pretrained: bool = False,
        hidden_head: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes phải >= 1")

        if backbone == "small_cnn":
            self.backbone = SmallCNN(in_channels=3)
        elif backbone == "resnet18":
            self.backbone = _ResNet18Backbone(pretrained=pretrained)
        else:
            raise ValueError(f"backbone không hỗ trợ: {backbone}")

        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_head),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
