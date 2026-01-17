from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(
    inputs: Float[Tensor, "*batch vocab_size"],
    targets: Int[Tensor, "*batch"],
) -> Float[Tensor, ""]:
    # inputs: [..., V], targets: [...]
    V = inputs.shape[-1]

    # 1) subtract max for numerical stability
    # keepdim 实现了广播机制，生成了shape为[..., 1]的张量
    m = inputs.max(dim=-1, keepdim=True).values              # [..., 1]
    z = inputs - m                                           # [..., V]

    # 2) compute logsumexp(z) without overflow
    log_denom = torch.log(torch.exp(z).sum(dim=-1))          # [...]

    # 3) pick z at target index: z_y = z[..., y]
    z_y = z.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [...]

    # 4) per-example loss: -log softmax = -z_y + logsumexp(z)
    # 形状为[...]
    loss = -z_y + log_denom                                  # [...]

    # 5) average across all batch-like dims
    return loss.mean()
