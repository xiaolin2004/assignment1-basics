from __future__ import annotations
import torch
from torch import nn, Tensor
from jaxtyping import Float


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            cal_d_ff = int(8 / 3 * d_model)
            self.d_ff = cal_d_ff
        else:
            self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        w1_out = torch.einsum("...i, oi -> ...o", x, self.w1.weight)
        w3_out = torch.einsum("...i, oi -> ...o", x, self.w3.weight)
        SiLU = self.SiLU(w1_out)
        return torch.einsum("...i, oi -> ...o", SiLU * w3_out, self.w2.weight)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
