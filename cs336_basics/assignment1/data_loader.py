from __future__ import annotations
import numpy.typing as npt
import torch

def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # x: shape (n,), dtype int
    n = dataset.shape[0]
    # 起点 i 的最大值要保证 i+m 以及 i+m+1 不越界
    max_i = n - context_length - 1
    idx = torch.randint(0, max_i + 1, (batch_size,))

    # 用 numpy 高级索引 + 广播拿到 (B, m) 的切片
    offsets = torch.arange(context_length)  # (m,)
    inp = dataset[idx[:, None] + offsets[None, :]]          # (B, m)
    tgt = dataset[idx[:, None] + offsets[None, :] + 1]      # (B, m)

    inp = torch.from_numpy(inp).to(device).long()
    tgt = torch.from_numpy(tgt).to(device).long()
    return inp, tgt