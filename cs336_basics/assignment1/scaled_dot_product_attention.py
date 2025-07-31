import torch
from torch import nn
from cs336_basics.assignment1 import softmax
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    d_k = key.shape[-1]

    qtk = torch.einsum("... qd,... kd->...qk", query, key)
    score = qtk / math.sqrt(d_k)
    if mask is not None:
        #we can do this by taking the pre-softmax values and adding a −∞ in any entry of the mask matrix that is False
        score = score.masked_fill(mask=mask == 0, value=-1e9)
    softmax_qtk = softmax.softmax(score, -1)

    return torch.einsum("...qk,...kv->...qv", softmax_qtk, value)
