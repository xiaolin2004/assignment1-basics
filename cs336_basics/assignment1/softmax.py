import torch


def softmax(x: torch.Tensor, i: int):
    assert x.ndim == 2, f"输入张量必须是 2D 的 (N, C)，但实际维度是 {x.ndim}"
    
    max_item = torch.max(x, dim=i, keepdim=True).values

    x_stable = x - max_item

    exp = torch.exp(x_stable)

    sum_exp = exp.sum(dim=1, keepdim=True)

    return exp / sum_exp
