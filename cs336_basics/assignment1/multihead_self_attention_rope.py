from __future__ import annotations
import torch
from torch import nn, Tensor
import math
from torch.nn import functional as F
from cs336_basics.assignment1.rope import RoPE
from jaxtyping import Float, Int


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        h = num_heads
        assert d_model % h == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = h
        self.d_k = d_k = d_model // h

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device)

        self.rope = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len)

    def forward(
        self, x: Float[Tensor, "batch seq d_model"], token_positions: Int[Tensor, "batch seq"] | None = None
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        Args:
            x (torch.Tensor): Input tensor, shape: (batch_size, seq_len, d_model)
            token_positions (torch.Tensor | None): Optional tensor with token positions.
                                                   If None, defaults to [0, 1, ..., seq_len-1].
        """
        b, s, d = x.shape
        h, d_k = self.num_heads, self.d_k

        # ✨ 新增逻辑: 处理 token_positions 为 None 的情况
        if token_positions is None:
            token_positions = torch.arange(s, device=x.device)

        # 1. 线性投影
        q = torch.einsum("...i, oi -> ...o", x, self.q_proj.weight)
        k = torch.einsum("...i, oi -> ...o", x, self.k_proj.weight)
        v = torch.einsum("...i, oi -> ...o", x, self.v_proj.weight)

        # 2. 拆分多头
        q = q.view(b, s, h, d_k).permute(0, 2, 1, 3)
        k = k.view(b, s, h, d_k).permute(0, 2, 1, 3)
        v = v.view(b, s, h, d_k).permute(0, 2, 1, 3)

        # 3. 对每个头的Q和K应用RoPE
        q = self.rope(q.contiguous(), token_positions)
        k = self.rope(k.contiguous(), token_positions)

        # 4. 计算注意力分数
        scores = torch.einsum("bhik, bhjk -> bhij", q, k) / math.sqrt(d_k)

        # 5. 应用因果掩码
        mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -torch.inf)

        # 6. Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 7. 计算上下文向量
        context = torch.einsum("bhij, bhjv -> bhiv", attention_weights, v)

        # 8. 合并多头
        context = context.permute(0, 2, 1, 3).contiguous().view(b, s, d)

        # 9. 最终投影
        output = torch.einsum("...i, oi -> ...o", context, self.output_proj.weight)

        return output
