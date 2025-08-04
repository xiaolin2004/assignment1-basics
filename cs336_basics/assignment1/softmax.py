import torch


def softmax(x: torch.Tensor, dim: int):

    max_item = torch.max(x, dim=dim, keepdim=True).values

    x_stable = x - max_item

    exp = torch.exp(x_stable)

    sum_exp = exp.sum(dim=dim, keepdim=True)

    return exp / sum_exp
