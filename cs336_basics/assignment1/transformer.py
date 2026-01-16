from cs336_basics.assignment1 import multihead_self_attention_rope, rms_norm, swi_glu
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int
    ) -> None:
        super().__init__()
        self.attn = multihead_self_attention_rope.MultiHeadSelfAttentionRoPE(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta
        )
        self.ln1 = rms_norm.RMSNorm(d_model=d_model)
        self.ffn = swi_glu.SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln2 = rms_norm.RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        output = h + self.ffn(self.ln2(h))
        return output
