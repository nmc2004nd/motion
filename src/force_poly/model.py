"""PolynomialRegressor: standardize → poly_expand(degree) → linear/MLP head.

Polynomial expansion sinh tất cả tích monomial bậc ≤ d của D feature đầu vào,
bao gồm cả bias 1. Số chiều output = C(D+d, d). Ví dụ D=9, d=3 → 220 monomial.

Standardize được học từ dữ liệu train (running mean/std) qua method `fit_scaler`
gọi 1 lần trước khi train, sau đó freeze. Phép standardize đặt trước expansion
để hệ số bậc cao không blow-up vì feature có scale lớn.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------- #
#  Polynomial expansion                                                         #
# ---------------------------------------------------------------------------- #


def _monomial_indices(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """Sinh danh sách tổ hợp index cho monomial từ bậc 0 đến `degree`.

    Mỗi tuple mã hoá tích các feature theo index, có lặp. Bậc 0 là tuple rỗng
    (= 1, bias). Tổng số phần tử = C(n_features + degree, degree).
    """
    out: list[tuple[int, ...]] = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            out.append(combo)
    return out


def polynomial_expand(
    x: torch.Tensor,
    indices: list[tuple[int, ...]],
) -> torch.Tensor:
    """Expand (B, D) → (B, P) theo danh sách monomial indices.

    Bậc 0 → cột hằng 1. Bậc cao → tích các cột tương ứng.
    """
    B = x.shape[0]
    cols: list[torch.Tensor] = []
    for combo in indices:
        if len(combo) == 0:
            cols.append(torch.ones(B, device=x.device, dtype=x.dtype))
        else:
            term = x[:, combo[0]]
            for i in combo[1:]:
                term = term * x[:, i]
            cols.append(term)
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------- #
#  Model                                                                        #
# ---------------------------------------------------------------------------- #


class PolynomialRegressor(nn.Module):
    """Standardize → polynomial expand → linear hoặc MLP head → scalar.

    Args:
        n_features: số scalar feature đầu vào (D).
        degree: bậc đa thức tối đa.
        head: "linear" (= polynomial regression thuần) | "mlp_small" (poly
            expansion làm input cho MLP nhỏ, vẫn giữ "polynomial nature").
        hidden_head: hidden dim cho MLP (chỉ dùng nếu head="mlp_small").
        dropout: dropout cho MLP head.
    """

    def __init__(
        self,
        n_features: int,
        degree: int,
        head: str = "linear",
        hidden_head: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.degree = degree
        self.head_kind = head

        indices = _monomial_indices(n_features, degree)
        # Lưu indices vào buffer để serialize với state_dict không cần state riêng.
        # combinations là Python list-of-tuple; flatten sang 1D + offset để buffer hoá.
        flat: list[int] = []
        offsets: list[int] = [0]
        for combo in indices:
            flat.extend(combo)
            offsets.append(len(flat))
        self.register_buffer(
            "_mono_flat", torch.tensor(flat, dtype=torch.long), persistent=True,
        )
        self.register_buffer(
            "_mono_offsets", torch.tensor(offsets, dtype=torch.long), persistent=True,
        )
        self._mono_indices: list[tuple[int, ...]] = indices  # cache cho forward
        self.n_terms = len(indices)

        # Standardize: mean/std học từ train data (fit_scaler), không phải param.
        self.register_buffer("feat_mean", torch.zeros(n_features), persistent=True)
        self.register_buffer("feat_std", torch.ones(n_features), persistent=True)

        if head == "linear":
            self.head = nn.Linear(self.n_terms, 1, bias=False)  # bias đã có trong term bậc 0
        elif head == "mlp_small":
            self.head = nn.Sequential(
                nn.Linear(self.n_terms, hidden_head),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_head, 1),
            )
        else:
            raise ValueError(f"head không hỗ trợ: {head}")

    def fit_scaler(self, x_train: np.ndarray, eps: float = 1e-6) -> None:
        """Cập nhật feat_mean, feat_std từ feature matrix train (numpy).

        Gọi 1 lần trước khi train. Sau đó scaler bị freeze (lưu vào ckpt).
        """
        if x_train.ndim != 2 or x_train.shape[1] != self.n_features:
            raise ValueError(
                f"x_train shape {x_train.shape} không khớp n_features={self.n_features}"
            )
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std < eps, 1.0, std)  # tránh chia 0 với feature constant
        self.feat_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.feat_std.copy_(torch.from_numpy(std.astype(np.float32)))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize + polynomial expand. (B, D) → (B, P)."""
        x_std = (x - self.feat_mean) / self.feat_std
        return polynomial_expand(x_std, self._mono_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B,)."""
        z = self.expand(x)
        return self.head(z).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
