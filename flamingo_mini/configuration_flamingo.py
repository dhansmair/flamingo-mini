from typing import Optional
from transformers.configuration_utils import PretrainedConfig


# AVAILABLE_LANGUAGE_MODELS = ('gpt2', 'facebook/opt')
AVAILABLE_ACTIVATION_FUNCTIONS = ('sqrelu', 'gelu')


# def is_lm_supported(lm_name: str) -> bool: 
#     return any(lm_name.startswith(pref) for pref in AVAILABLE_LANGUAGE_MODELS)


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
        xattn: Optional[dict] = None,
        resampler_depth: int = 6,
        resampler_dim_head: int = 64,
        resampler_heads: int = 8 ,
        resampler_num_latents: int = 64,
        resampler_num_time_embeds: int = 4,
        resampler_ff_mult: int = 4,
        resampler_act: str = 'gelu',
        resampler: Optional[dict] = None,
        **kwargs
    ):
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

        if xattn is not None:
            xattn_keys_to_ignore = ('dim', 'dim_visual')
            for key, value in xattn.items():
                if key not in xattn_keys_to_ignore:
                    setattr(self, f'xattn_{key}', value)

        if resampler is not None:
            resampler_keys_to_ignore = ('dim')
            for key, value in resampler.items():
                if key not in resampler_keys_to_ignore:
                    setattr(self, f'resampler_{key}', value)
                
        # TODO move to the top? Check if there is a side effect
        super().__init__(**kwargs)
            