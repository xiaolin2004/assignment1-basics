import torch
from torch import nn


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
        weight_1 = torch.empty([self.d_ff, d_model], **factory_kwargs)
        weight_2 = torch.empty([d_model, self.d_ff], **factory_kwargs)
        weight_3 = torch.empty([self.d_ff, d_model], **factory_kwargs)
        self.W1 = nn.Parameter(weight_1)
        self.W2 = nn.Parameter(weight_2)
        self.W3 = nn.Parameter(weight_3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SiLU = self.SiLU(x @ self.W1.T)
        return (SiLU * (x @ self.W3.T)) @ self.W2.T

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
