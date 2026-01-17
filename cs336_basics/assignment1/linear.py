from __future__ import annotations
import torch
from torch import nn
import math


from jaxtyping import Float
from torch import Tensor

class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        std = math.sqrt(2 / (in_features + out_features))
        a = -3 * std
        b = 3 * std
        weight_tensor = torch.zeros(
            [out_features, in_features], device=device, dtype=dtype
        )
        nn.init.trunc_normal_(
            weight_tensor,
            mean=0,
            std=std,
            a=a,
            b=b,
            generator=torch.Generator(device=device),
        )
        self.W = nn.Parameter(weight_tensor)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return torch.einsum("...i, oi -> ...o", x, self.W)
