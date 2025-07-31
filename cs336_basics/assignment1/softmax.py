import torch


def softmax(x: torch.Tensor, i: int):

    max_item = torch.max(x, dim=i, keepdim=True).values

    x_stable = x - max_item

    exp = torch.exp(x_stable)

    sum_exp = exp.sum(dim=i, keepdim=True)

    return exp / sum_exp
