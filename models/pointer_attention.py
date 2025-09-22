import itertools
import math
import warnings

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

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


class PointerAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134).

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        **kwargs,
    ):
        super(PointerAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
                
        self.sdpa_fn = scaled_dot_product_attention

    def forward(self, query, key, value, logit_key,pair_bias=None,attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = self._project_out(heads, attn_mask)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))
        if pair_bias is not None:
            logits = logits 
        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        return self.project_out(out)
