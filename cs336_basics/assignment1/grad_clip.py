from __future__ import annotations
from collections.abc import Iterable
import torch

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    
    max_l2_norm = float(max_l2_norm)
    eps = 1e-6
    
    total_sq_norm = torch.tensor(0.0)
    for param in parameters:
        if param.grad is not None:
            total_sq_norm += param.grad.pow(2).sum()
    
    total_norm = total_sq_norm.sqrt()
    clip_coef = max_l2_norm / (total_norm + eps)
    
    if clip_coef < 1:
        with torch.no_grad():
            for param in parameters:
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
