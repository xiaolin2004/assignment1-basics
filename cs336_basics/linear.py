import torch
from torch import nn


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_tensor = torch.zeros(
            [out_features, in_features], device=device, dtype=dtype
        )
        nn.init.trunc_normal_(weight_tensor, generator=torch.Generator(device=device))
        self.W = nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
