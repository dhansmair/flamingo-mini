import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

from .utils import FeedForward


class PerceiverAttentionLayer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents):
        """
        Latent vectors are cross-attending to the visual features x.
        :param x:       Tensor (n_batch, n_features, dim)
                        visual features
        :param latents: Tensor (n_batch, n_latents, dim)
                        latent learnt vectors from which the queries are computed.
                        Actually the same, just replicated in n_batch and n_frames dimension.
        :return:        Tensor (n_batch, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # layer normalization, as usual
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # queries
        # compute the queries from the latents, for all attention heads simultaneously.
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # keys and values for all attention heads
        # kv_input = torch.cat((x, latents), dim=-2)
        # k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # latents_ = repeat(latents, 'q d -> b q d', b=n_batch)
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries

        # keys, values
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # batch, features, (heads, dim)

        # split so we have an extra dimension for the heads
        # q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)
        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size([n_batch, n_heads, n_features_latents, self.dim_head])

        # scale queries?
        q = q * self.scale

        # attention

        # attention scores
        # sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = einsum('b h q d, b h f d -> b h q f', q, k)

        # Is this for numerical stability? Does not affect the result of the softmax operation
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        # out = einsum('... i j, ... j d -> ... i d', alphas, v)
        out = einsum('b h q f, b h f v -> b h q v', alphas, v)

        # out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        out = rearrange(out, 'b h q v -> b q (h v)')
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_time_embeds=4,
        ff_mult=4,
        act='gelu'
    ):
        """
        :param dim:             length of the visual features and of the queries (-> thus also length of the keys)
        :param depth:           number of attention layers
        :param dim_head:        inner dimensionality of the q, k, v vectors per attention head
        :param heads:           number of attention heads
        :param num_latents:     number of queries, default 64 as in the flamingo paper
        :param num_time_embeds: TODO what is this? maximum number of frames per video clip? Then 4 seems not enough
        :param ff_mult:         factor for the number of inner neurons in the feedforward layer
        """
        super().__init__()

        self.dim = dim
        self.n_queries = num_latents

        # latents are not the queries themselves, but latent vectors from which the queries are computed.
        # (same dimension as the length of the features
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # the time positional embeddings are learnable parameters as well?
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult, act=act)
            ]))

        # layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_f):
        """
        TODO does the output dimensionality of visual features does need to match the input dimensionality? (d_visual)
        d_visual must be equal to the dimensionality of the latents, since they are concatenated in the xattn layer.
        :param x_f: Tensor (n_batch, n_features, d_visual) or (n_batch, n_frames, n_features, d_visual)
        :return:    Tensor (n_batch, T, n_queries, d_visual)
        """
        if x_f.ndim == 3:
            # if batch is just a tensor of images, extend frame dimension -> images are like videos with just one frame
            x_f = rearrange(x_f, 'b n d -> b 1 n d')

        assert x_f.ndim == 4

        n_batches = x_f.shape[0]
        n_frames = x_f.shape[1]
        n_visual_features = x_f.shape[2]
        dim = x_f.shape[3]

        assert dim == self.dim

        # add time embeddings
        # for each frame. Times is the number of frames
        # time embedding is added to every visual feature of one frame.
        x_f = x_f + self.time_pos_emb[:n_frames]

        # >> David added
        # flatten frames
        # not sure if it makes a difference, but lucidrains did not flatten the features from the frames.
        # It makes a difference in the output shape, if we are processing video clips.
        x_f = rearrange(x_f, 'b T n d -> b (T n) d')

        # >> David modified
        # copy the latents for every element in the batch.
        # lucidrains did extend the latents to b T q d, however makes no sense if frames are flattened before
        # q = number of queries
        # d = dimension of queries
        x = repeat(self.latents, 'q d -> b q d', b=n_batches)

        for attn, ffw in self.layers:  # type: ignore
            x = x + attn(x_f, x)
            x = x + ffw(x)

        assert x.shape == torch.Size([n_batches, self.n_queries, self.dim])

        norm = self.norm(x)
        return norm


