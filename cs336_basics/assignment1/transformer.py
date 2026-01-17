from __future__ import annotations
from cs336_basics.assignment1 import multihead_self_attention_rope, rms_norm, swi_glu
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int


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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        h = x + self.attn(self.ln1(x))
        output = h + self.ffn(self.ln2(h))
        return output

class Transformer_lm(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int, 
        theta: float, 
        max_seq_len: int,
        vocab_size:int,
        context_length:int,
        num_layers:int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(Transformer(d_model, num_heads, d_ff, theta, max_seq_len))
        self.layers = layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.ln_final = rms_norm.RMSNorm(d_model=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x