from __future__ import annotations
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int


class RoPE(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k should be even"

        # 创建频率
        freqs = 1.0 / torch.pow(torch.tensor(theta), (torch.arange(0, d_k, 2).float() / d_k))

        # 创建位置编码
        t = torch.arange(0,max_seq_len,1, device=device, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, freqs)  # 外积

        # 创建复数形式的旋转
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer("freqs_complex", freqs_complex)

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d_k"]:
        # x: [batch, seq_len, d_k]
        # 将实数转换为复数
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        # 应用旋转
        freqs = self.freqs_complex[token_positions]
        x_rotated = torch.einsum("...d,...d->...d", x_complex, freqs)

        # 转回实数
        return torch.view_as_real(x_rotated).flatten(-2)
