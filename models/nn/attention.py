import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional

def scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple (exact) Scaled Dot-Product Attention in RL4CO without customized kernels (i.e. no Flash Attention)."""

    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)

class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            # (B,S) -> (B,1,1,S)  あるいは (B,S,S) -> (B,1,S,S)
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
 
        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))
