from typing import Callable

from einops import rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .helpers import default, exists

class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Attention(nn.Module):
    "Multi-headed attention"

    def __init__(self, dim: int, *, dim_head: int = 64, heads: int = 8, context_dim: int = None):
        """
        :param dim: Input dimensionality.
        :param dim_head: Dimensionality for each att. head.
        :param heads: Number of attention heads.
        :param: context_dim: Context dimensionality.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        
        self.norm = LayerNorm(dim=dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_head, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 2 * dim_head)) if exists(context_dim) else None
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))

        def forward(self, x: torch.tensor, context: torch.tensor=None, mask: torch.tensor=None, attn_bias: torch.tensor=None) -> torch.tensor:
            b, n, device = *x.shape[:2], x.device
            x = self.norm(x)
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            q = q * self.scale

            # add null key / value for classifier free guidance in prior net
            nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

            # add text conditioning, if present
            if exists(context):
                assert exists(self.to_context)
                ck, cv = self.to_context(context).chunk(2, dim=-1)
                k = torch.cat((ck, k), dim=-2)
                v = torch.cat((cv, v), dim=-2)

            # calculate query / key similarities
            sim = einsum('b h i d, b j d -> b h i j', q, k)

            # relative positional encoding (T5 style)
            if exists(attn_bias):
                sim = sim + attn_bias

            # masking
            max_neg_value = -torch.finfo(sim.dtype).max
            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, max_neg_value)

            # attention
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # aggregate values
            out = einsum('b h i j, b j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

    