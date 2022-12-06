"""
gated cross-attention layer.
adapted from flamingo-pytorch.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn, einsum, tanh
from einops import rearrange, repeat
from einops_exts import rearrange_many

from .utils import FeedForward


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        n_visual=64
    ):
        """
        :param dim:      d_token, d_visual  dimensionality of language- and visual tokens
        :param dim_head: dimensionality of the q, k, v vectors inside one attention head
        :param heads:   number of attention heads
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.n_visual = n_visual
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        y: torch.Tensor,
        media_locations: torch.Tensor,
        visual_features: torch.Tensor,
        previous_kv=None,
        output_kv=False
    ):
        """This has the same inputs as the GatedCrossAttentionBlock

        Args:
            y (FloatTensor):
                language features (n_batch, n_token, d_token)
            visual_features (FloatTensor, optional):
                visual features   (n_batch, n_images, n_queries, d_visual). Defaults to None.
            media_locations (BoolTensor, optional):
                boolean matrix of shape: (n_batch, n_token). Defaults to None.
                Indicates the positions of <image> tokens in the input sequence y (more precisely, the position of '<')
            previous_kv (Tuple, optional):
                tuple of previous keys and values. Passed when caching is used during text generation.
                Defaults to None.
            output_kv (bool, optional):
                whether to return the keys and values. Defaults to False.

        Returns:
            FloatTensor: Tensor (n_batch, n_token, d_token)
        """
        n_batch, n_media = visual_features.shape[:2]
        n_batch_y, n_token, d_token = y.shape
        n_heads = self.heads

        # LayerNorm
        y = self.norm(y)

        # 2. compute the queries from the text tokens:
        q = self.to_q(y)
        q = q * self.scale

        # 3. compute the keys and values from the visual tokens:
        if previous_kv is None:
            # flatten, so t is #images, n is #visual features per image.
            # Now there is only one set of visual features per # sentence.
            visual_features = rearrange(visual_features, 'b t n d -> b (t n) d')

            k, v = self.to_kv(visual_features).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=n_heads)
        else:
            # visual_features can be ignored, k, v already computed
            k, v = previous_kv
            n_media = k.size(2) // self.n_visual
            q = rearrange(q, 'b n (h d) -> b h n d', h=n_heads)

        # 5. compute the attention scores from the queries and keys:
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        text_time = media_locations.cumsum(dim=-1) # at each boolean of True, increment the time counter (relative to media time)
        
        # >> David
        # this needs to be adjusted if caching is used.
        # text_time has shape (n_batch, n_token)
        if previous_kv is not None:
            text_time = text_time[:, -n_token:]
            assert text_time.shape == y.shape[:2]
        
        media_time = torch.arange(n_media, device=y.device) + 1
        # >> David:
        # side note: here, text tokens attend to ALL previous visual tokens. If We only want to attend to the
        # one image coming before in the text (like in the flamingo paper),
        # we need to change >= to == at the line where 'text_to_media_mask' is created.
        text_to_media_mask = rearrange(text_time, 'b i -> b 1 i 1') == repeat(media_time, 'j -> 1 1 1 (j m)', m=self.n_visual)
        sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)
        
        # a bug and an update from lucidrains: 
        # any text without a preceding media needs to have attention zeroed out
        text_without_media_mask = text_time == 0
        text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
        alphas = alphas.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        conditioned_tokens = self.to_out(out)

        if output_kv:
            return conditioned_tokens, (k, v)
        else:
            return conditioned_tokens, None
            


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        act='gelu',
        n_visual=64
    ):
        """
        :param dim:      d_token, d_visual
        :param dim_head: dimensionality of q, k, v inside the attention head
        :param heads:    number of attention heads
        :param ff_mult:  factor for the number of inner neurons in the ffw layer
        """
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual, dim_head=dim_head, heads=heads, n_visual=n_visual)
        self.alpha_attn = nn.Parameter(torch.tensor([0.]))

        self.ffw = FeedForward(dim, mult=ff_mult, act=act)
        self.alpha_ffw = nn.Parameter(torch.tensor([0.]))

    def forward(
        self, 
        y: torch.Tensor, 
        visual_features: torch.Tensor, 
        media_locations: torch.Tensor, 
        previous_kv=None, 
        output_kv=False
    ):
        """
        :param y:           (n_batch, n_tokens, d_token) - language features from previous LM layer
        :param media:       (n_batch, n_media, n_queries, dim) - visual features, encoded by perceiver resample
        :param media_locations:  (n_batch, n_tokens) - boolean matrix, indicating start positions of <image> tokens
        :return:
        """
        if previous_kv is None:
            assert visual_features.ndim == 4
        shape_before = y.shape
        
        # kv will be None if output_kv=False
        attn_out, kv = self.attn(y, media_locations, visual_features, previous_kv=previous_kv, output_kv=output_kv)
        y = y + tanh(self.alpha_attn) * attn_out
        assert y.shape == shape_before        
        y = y + tanh(self.alpha_ffw) * self.ffw(y)
        assert y.shape == shape_before        
        return y, kv


class ModifiedLMBlock(nn.Module):
    """
    A block that wraps a gated cross-attention layer, followed by a LM layer.
    We replace the original layers in the LM with these at a certain frequency
    to introduce the xattn layer. This layer mimics the functionality and behavior 
    of the underlying LM block. This way, the LM can be used in the same way as before,
    and we can do the conditioning without altering the LM implementation.
    
    One drawback of this approach is that we cannot pass the visual features to forward()
    directly, but instead we need to pass them before the actual forward pass, via a 
    side-channel, which is the condition() method. In addition, when use_cache is used,
    the cached keys and values for the xattn layers need to be retrieved separately from
    the kv_output property.
    
    (!) This implementation works with GPT-2 and OPT layers, but hasn't been tested with other LMs yet.
    """
    
    def __init__(self, lm_block, **kwargs):
        super().__init__()
        
        self.xattn_block = GatedCrossAttentionBlock(**kwargs)
        self.lm_block = lm_block
        self.visual_features = None
        self.media_locations = None
        self.xattn_layer_past = None
        self.kv_output = None
        
    def condition(self, visual_features: torch.Tensor, media_locations: torch.Tensor, xattn_layer_past=None) -> None:
        """
        conditioning. Called from outside of the LM before passing the text input to the LM.
        This way, the gated cross-attention layers get informed about the visual input
        without the need to pipe the visual input through the LM forward() function.
        
        xattn_layer_past can contain the cached cross-attention keys and values (computed
        from the visual input). Passing them is useful to speed up the autoregressive text
        generation where the keys and values will be the same for every word, since the 
        visual input doesn't change.
        If both visual_features and xattn_layer past are passed, visual_features will be 
        ignored in the xattn layers.
        """
        self.visual_features = visual_features 
        self.media_locations = media_locations
        self.xattn_layer_past = xattn_layer_past
        
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor] | None,
        use_cache: bool | None = False,
        **kwargs
    ):
        """
        This forward function mimics forward() of GPT2Block, so it has the same input and output.
        """
        
        # pass through xattn
        hidden_states, kv = self.xattn_block(
            y=hidden_states, 
            visual_features=self.visual_features, 
            media_locations=self.media_locations,
            previous_kv=self.xattn_layer_past,
            output_kv=use_cache
        )
        self.kv_output = kv
        
        # pass through original LM layer
        return self.lm_block(hidden_states, use_cache=use_cache, **kwargs)
        
    
