import torch
from torch import nn
from einops import reduce


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        g = torch.ones(d_model,device=device,dtype=dtype)
        self.g = nn.Parameter(g)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #You should upcast your input to torch.float32 to prevent overflow when you square the input
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 运行速度显著下降
        # rms = torch.sqrt(reduce(x.pow(2),"... d_model -> ... 1","mean") + self.eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result =  (x / rms) * self.g
        return result.to(in_dtype)
        