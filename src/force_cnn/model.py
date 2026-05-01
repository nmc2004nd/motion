"""ForceCNN: end-to-end CNN scalar force regressor.

Backbone configurable:
- "resnet18" (default): ~11M params, có option pretrain ImageNet với conv1 inflate.
- "small_cnn": ~110K params, debug nhanh trên CPU/laptop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------- #
#  Small CNN (lightweight backbone)                                             #
# ---------------------------------------------------------------------------- #


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
    """4-block conv + GAP → (B, 128)."""

    out_dim = 128

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(in_channels, 16, kernel=5),
            _ConvBlock(16, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.features(x)).flatten(1)


# ---------------------------------------------------------------------------- #
#  ResNet18 backbone                                                            #
# ---------------------------------------------------------------------------- #


def _inflate_conv1_from_pretrained(
    pretrained_conv1_weight: torch.Tensor, in_channels: int
) -> torch.Tensor:
    """Inflate ImageNet conv1 (3,64,7,7) sang (in_channels,64,7,7).

    Trick: lấy mean qua chiều RGB → (1,64,7,7), tile in_channels lần. Giữ được
    edge/blob filter mà ImageNet đã học, ngay cả khi input không phải RGB.
    """
    mean_w = pretrained_conv1_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    return mean_w.repeat(1, in_channels, 1, 1)


class _ResNet18Backbone(nn.Module):
    out_dim = 512

    def __init__(self, in_channels: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        old_conv1_w = net.conv1.weight.detach().clone() if pretrained else None

        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained and old_conv1_w is not None:
            with torch.no_grad():
                net.conv1.weight.copy_(
                    _inflate_conv1_from_pretrained(old_conv1_w, in_channels)
                )

        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------- #
#  ForceCNN                                                                     #
# ---------------------------------------------------------------------------- #


class ForceCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_head: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone == "small_cnn":
            self.backbone = SmallCNN(in_channels=in_channels)
        elif backbone == "resnet18":
            self.backbone = _ResNet18Backbone(
                in_channels=in_channels, pretrained=pretrained,
            )
        else:
            raise ValueError(f"backbone không hỗ trợ: {backbone}")

        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_head),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
