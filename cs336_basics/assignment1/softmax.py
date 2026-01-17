from __future__ import annotations
import torch
from torch import Tensor
from jaxtyping import Float


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:

    max_item = torch.max(x, dim=dim, keepdim=True).values

    x_stable = x - max_item

    exp = torch.exp(x_stable)

    sum_exp = exp.sum(dim=dim, keepdim=True)

    return exp / sum_exp
