from __future__ import annotations
import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.assignment1 import softmax
import math


def scaled_dot_product_attention(
    query: Float[Tensor, "... qd"],
    key: Float[Tensor, "... kd"],
    value: Float[Tensor, "... kv"],
    mask: Float[Tensor, "..."] | None,
) -> Float[Tensor, "... qv"]:
    d_k = key.shape[-1]

    score = torch.einsum("... qd,... kd->...qk", query, key)
    score = score / math.sqrt(d_k)
    if mask is not None:
        # we can do this by taking the pre-softmax values and adding a −∞ in any entry of the mask matrix that is False
        score = score.masked_fill(mask=mask == 0, value=-1e9)
    softmax_qtk = softmax.softmax(score, -1)

    return torch.einsum("...qk,...kv->...qv", softmax_qtk, value)
