from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.nn.attention import MultiHeadAttention
from models.nn.norm import RMSNorm

class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        # https://github.com/deepseek-ai/DeepSeek-V3/blob/b5d872ead062c94b852d75ce41ae0b10fcfa1c86/inference/model.py#L529
        return self.l3(self.act(self.l1(z)) * self.l2(z))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
    ):
        super(TransformerBlock, self).__init__()
       
        ffn = ParallelGatedMLP()
        
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = RMSNorm(embed_dim)
        self.norm_attn = RMSNorm(embed_dim)
        self.ffn = ffn

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        # normal transformer structure
        h = x + self.attention(self.norm_attn(x), mask)
        
        h = h + self.ffn(self.norm_ffn(h))
        
        return h
