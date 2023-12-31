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

class Identity(nn.Module):
    """
    Identity module: forward pass returns input
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        return x

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

class Block(nn.Module):
    """
    Sub-block for ResnetBlock
    """
    def __init__(self, dim: int, dim_out: int, groups: int = 8, norm: bool = True):
        super().__init__()
        # Apply group normalization https://arxiv.org/abs/1803.08494
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.projection = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x: torch.tensor, scale_shift: tuple[torch.tensor, torch.tensor] = None) -> torch.tensor:
        """
        :param x: Input images
        :param scale_shift: Tensors to use for scale-shift
        """
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = F.silu(x)
        return self.projection(x)
    
class ChanLayerNorm(nn.Module):
    """
    LayernNorm (channel-wise)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # b, i, c
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean)/(var+self.eps).sqrt() * self.g

def ChanFeedForward(dim: int, mult: int = 2) -> torch.nn.Sequential:
    hidden_dim = int(dim * mult)
    return nn.Sequential(ChanLayerNorm(dim=dim),
                         nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
                         nn.GELU(),
                         ChanLayerNorm(hidden_dim),
                         nn.Conv2d(hidden_dim, dim, 1, bias=False))

class CrossAttention(nn.Module):
    """
    Multi-headed cross attention
    """
    def __init__(self, dim:int, *, context_dim: int=None, dim_head: int= 64, heads: int =8, norm_context: bool = False):
        super().__init__()
        self.scale = dim ** -0,5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_head, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 2 * dim_head)) if exists(context_dim) else None
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))

    def forward(self, x: torch.tensor, context: torch.tensor=None, mask: torch.tensor=None, attn_bias: torch.tensor=None) -> torch.tensor:
            b, n, device = *x.shape[:2], x.device
            x = self.norm(x)
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

            # add null key / value for classifier free guidance in prior net
            nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

            q = q * self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            max_neg_value = -torch.finfo(sim.dtype).max

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, max_neg_value)

            attn = sim.softmax(dim=-1, dtype=torch.float32)

            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

class CrossEmbedLayer(nn.Module):
    """
    Module performing cross-embedding on an input image, like an inception module, and keeping channel depth.
    """
    def __init__(self, dim_in: int, kernel_sizes: tuple[int, ...], dim_out: int = None, stride: int = 2):
        super().__init__()
        assert all([*map(lambda t: (t%2)== (stride%2), kernel_sizes)]) #ensuring stride and kernels are either all odd or all even

        dim_out = default(dim_out, dim_in)
        kernel_sizes = sorted(kernel_sizes) # sort kernels by size
        num_scales = len(kernel_sizes) # number of kernels
        
        # check number of filters for each kernel. We sum them to dim_out and be descending with kernel size
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # createthe convolution objects
        self.convs - nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Perform each convolution and concatenate the outputs along the channel dim
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)
    
def Downsample(dim: int, dim_out: int = None) -> torch.nn.Conv2d:
    """
    Return a convolutional layer that cuts the spatial dimensions of an image in half and potentially modifies the number of channels
    """    
    dim_out = default(dim_out, dim)
    return nn.Conv2d(dim, dim_out, kernel_size=4, stride=2, padding=1)

def Upsample(dim: int, dim_out: int = None) -> torch.nn.Sequential:
    """
    Returns sequential module upsampling twice the spatial width with an nn.Upsample
    """
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1)
        )

class Parallel(nn.Module):
    """
    Passes input through parallel functions and then sums the result.
    """
    def __init__(self, *fns: tuple[Callable, ...]):
        super().__init__()
        self.fns = nn.ModuleList(fns)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)
    
class Residual(nn.Module):
    """
    Passes input through a function and then adds the output to input
    """
    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn
    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return self.fn(x, **kwargs) + x