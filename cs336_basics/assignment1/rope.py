import torch
from torch import nn

class RoPE(nn.Module):
    
    def __init__(self, 
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k should be even"
        
        # 创建频率
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        
        # 创建位置编码
        t = torch.arange(max_seq_len, device=device,dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, freqs)  # 外积
        
        # 创建复数形式的旋转
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        self.register_buffer("freqs_complex", freqs_complex)
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_k]
        # 将实数转换为复数
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        # 应用旋转
        freqs = self.freqs_complex[token_position]
        x_rotated = torch.einsum('...d,...d->...d', x_complex, freqs)
        
        # 转回实数
        return torch.view_as_real(x_rotated).flatten(-2)