"""ForceNet: PointNet-style scalar force regressor over per-marker features.

Input feature mỗi marker: [x_ref_norm, y_ref_norm, dx_norm, dy_norm].
Mask-aware max+mean pool để robust với marker bị LK lost (mask=False).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SharedMLP(nn.Module):
    """Per-point MLP qua Conv1d (kernel=1) — share weights across markers."""

    def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Conv1d(prev, h, kernel_size=1), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim, N) → (B, out_dim, N)
        return self.net(x)


def _masked_max_mean_pool(
    feats: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mask-aware pool.

    Args:
        feats: (B, C, N) — per-point features.
        mask:  (B, N) bool — True = valid.

    Returns:
        (B, 2C) — concat của max-pool và mean-pool.
    """
    # Broadcast mask to (B, 1, N)
    m = mask.unsqueeze(1).to(feats.dtype)

    # Max pool: set invalid to very negative
    very_neg = torch.finfo(feats.dtype).min
    feats_for_max = feats.masked_fill(~mask.unsqueeze(1), very_neg)
    max_pool = feats_for_max.max(dim=2).values  # (B, C)

    # Mean pool: sum valid / count valid (clamp 1 để tránh div 0)
    sum_pool = (feats * m).sum(dim=2)
    count = m.sum(dim=2).clamp(min=1.0)
    mean_pool = sum_pool / count

    # Nếu batch nào không có valid marker (count=0 trước clamp), set max_pool=0
    no_valid = (mask.sum(dim=1) == 0).unsqueeze(1)  # (B, 1)
    max_pool = max_pool.masked_fill(no_valid, 0.0)

    return torch.cat([max_pool, mean_pool], dim=1)  # (B, 2C)


class ForceNet(nn.Module):
    """PointNet-style scalar force regressor.

    Args:
        in_dim: số feature mỗi point (mặc định 4).
        hidden_per_point: kích thước hidden cho shared per-point MLP.
        hidden_head: kích thước hidden cho head MLP.
        dropout: dropout ở head.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_per_point: tuple[int, ...] = (64, 128),
        hidden_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.point_mlp = _SharedMLP(in_dim, list(hidden_per_point))
        feat_dim = self.point_mlp.out_dim * 2  # max + mean
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_head),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) feature.
            mask: (B, N) bool.

        Returns:
            (B,) predicted force.
        """
        # Conv1d expects (B, C, N)
        x = x.transpose(1, 2).contiguous()        # (B, in_dim, N)
        feats = self.point_mlp(x)                 # (B, C, N)
        pooled = _masked_max_mean_pool(feats, mask)  # (B, 2C)
        out = self.head(pooled).squeeze(-1)       # (B,)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
