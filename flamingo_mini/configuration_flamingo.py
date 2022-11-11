from transformers.configuration_utils import PretrainedConfig

AVAILABLE_ACTIVATION_FUNCTIONS = ('sqrelu', 'gelu')


class FlamingoConfig(PretrainedConfig):
    """ Configuration file for Flamingo.
    
    The parameters `xattn` and `resampler` are legacy params. Previously, all xattn_* and resampler_* params
    were passed in separate dictionaries. So the two are kept for backwards compatibility.
    """
    
    def __init__(
        self,
        lm: str = 'gpt2',
        clip_model_type: str = 'openai/clip-vit-base-patch32',
        dim: int = 1024,
        dim_visual: int = 768,
        xattn_every: int = 1,
        xattn_dim_head: int = 64,
        xattn_heads: int = 8,
        xattn_ff_mult: int = 4,
        xattn_act: str = 'gelu',
        resampler_depth: int = 6,
        resampler_dim_head: int = 64,
        resampler_heads: int = 8 ,
        resampler_num_latents: int = 64,
        resampler_num_time_embeds: int = 4,
        resampler_ff_mult: int = 4,
        resampler_act: str = 'gelu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lm = lm
        self.clip_model_type = clip_model_type
        self.dim = dim
        self.dim_visual = dim_visual
        self.xattn_every = xattn_every
        self.xattn_dim_head = xattn_dim_head
        self.xattn_heads = xattn_heads
        self.xattn_ff_mult = xattn_ff_mult
        self.xattn_act = xattn_act
        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head 
        self.resampler_heads = resampler_heads 
        self.resampler_num_latents = resampler_num_latents
        self.resampler_num_time_embeds = resampler_num_time_embeds
        self.resampler_ff_mult = resampler_ff_mult
        self.resampler_act = resampler_act