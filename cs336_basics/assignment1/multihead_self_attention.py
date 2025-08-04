import torch
from torch import nn
import math
from cs336_basics.assignment1.softmax import softmax


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        h = num_heads
        assert d_model % h == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = h
        self.d_k = d_k = d_model // h

        # 1. CORRECT WEIGHT DEFINITIONS
        # These are all 2D matrices, just like in a standard nn.Linear layer.
        # The test harness will be able to load its weights into these parameters.
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))

        # Standard initialization
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.Wo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, shape: (batch_size, seq_len, d_model)
        """
        b, s, d = x.shape  # batch_size, seq_len, d_model
        h, dk = self.num_heads, self.d_k  # num_heads, head_dim

        # 1. PROJECT Q, K, V USING 2D WEIGHTS
        # We use einsum for the batched matrix multiplication.
        # Wq: (d, d), x: (b, s, d) -> q_proj: (b, s, d)
        q_proj = torch.einsum("dm, bsm -> bsd", self.Wq, x)
        k_proj = torch.einsum("dm, bsm -> bsd", self.Wk, x)
        v_proj = torch.einsum("dm, bsm -> bsd", self.Wv, x)

        # 2. SPLIT INTO HEADS
        # This part cannot be done by einsum; it requires reshaping.
        # (b, s, d) -> (b, s, h, k) -> (b, h, s, k)
        q = q_proj.view(b, s, h, dk).permute(0, 2, 1, 3)
        k = k_proj.view(b, s, h, dk).permute(0, 2, 1, 3)
        v = v_proj.view(b, s, h, dk).permute(0, 2, 1, 3)

        # 3. CALCULATE ATTENTION SCORES
        # This part is the same as our previous correction.
        # (b,h,s,k) @ (b,h,k,s) -> (b,h,s,s)
        scores = torch.einsum("bhik, bhjk -> bhij", q, k) / math.sqrt(dk)

        # 4. APPLY CAUSAL MASK
        mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -torch.inf)

        # 5. SOFTMAX
        attention_weights = softmax(scores, dim=-1)

        # 6. COMPUTE CONTEXT VECTOR
        # (b,h,s,s) @ (b,h,s,k) -> (b,h,s,k)
        context = torch.einsum("bhij, bhjv -> bhiv", attention_weights, v)

        # 7. COMBINE HEADS
        # This is the reverse of step 2.
        # (b, h, s, k) -> (b, s, h, k) -> (b, s, d)
        context = context.permute(0, 2, 1, 3).contiguous().view(b, s, d)

        # 8. FINAL PROJECTION
        # Wo: (d, d), context: (b, s, d) -> output: (b, s, d)
        output = torch.einsum("dm, bsm -> bsd", self.Wo, context)

        return output
