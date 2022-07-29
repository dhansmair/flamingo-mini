from dataclasses import dataclass, field
from transformers.configuration_utils import PretrainedConfig


AVAILABLE_LANGUAGE_MODELS = ('gpt2', 'facebook/opt')
AVAILABLE_ACTIVATION_FUNCTIONS = ('sqrelu', 'gelu')


def is_lm_supported(lm_name: str) -> bool: 
    return any(lm_name.startswith(pref) for pref in AVAILABLE_LANGUAGE_MODELS)


# class GatedXAttnConfig:
    
#     def __init__(self,
#             # (!) dim depends on the used language model
#             dim: int = 1024,
#             # (!) dim_visual depends on the used vision encoder
#             dim_visual: int = 768,
#             dim_head: int = 64,
#             heads: int = 8,
#             ff_mult: int = 4,
#             act: str = 'gelu'
#     ):
#         self.dim = dim
#         self.dim_visual = dim_visual
#         self.dim_head = dim_head
#         self.heads = heads
#         self.ff_mult = ff_mult
#         self.act = act
        
    # def __dict__(self):
    #     return dict(
    #         dim = self.dim,
    #         dim_visual = self.dim_visual,
    #         dim_head = self.dim_head,
    #         heads = self.heads, 
    #         ff_mult = self.ff_mult,
    #         act = self.act
        # )
        
    
# @dataclass
# class ResamplerConfig:
    
#     def __init__(self,
#         # (!) dim depends on the used vision encoder
#         # must be equal to dim_visual from GatedXAttnConfig
#         dim: int = 768,
#         depth: int = 6,
#         dim_head: int = 64,
#         heads: int = 8 ,
#         num_latents: int = 64,
#         num_time_embeds: int = 4,
#         ff_mult: int = 4,
#         act: str = 'gelu'):
        
#         self.dim = dim
#         self.depth = depth
#         self.dim_head = dim_head
#         self.heads = heads
#         self.num_latents = num_latents
#         self.num_time_embeds = num_time_embeds
#         self.ff_mult = ff_mult
#         self.act = act
        
    # def __dict__(self):
    #     return dict(
    #         dim = self.dim,
    #         depth = self.depth,
    #         dim_head = self.dim_head,
    #         heads = self.heads,
    #         num_latents = self.num_latents,
    #         num_time_embeds = self.num_time_embeds,
    #         ff_mult = self.ff_mult,
    #         act = self.act,
    #     )
        
    

# class FlamingoConfig(PretrainedConfig):
    
#     def __init__(self, 
#                  lm: str = 'gpt2',
#                  xattn_every: int = 1, 
#                  xattn: GatedXAttnConfig = None, 
#                  resampler: ResamplerConfig = None, 
#                  **kwargs):

#         self.lm: str = lm
#         self.xattn_every: int = xattn_every
#         self.xattn: GatedXAttnConfig = xattn if xattn is not None else GatedXAttnConfig()
#         self.resampler: ResamplerConfig = resampler if resampler is not None else ResamplerConfig()

#         super().__init__(**kwargs)
    
#     @staticmethod  
#     def from_dict(d: dict):
#         d = d.copy()

#         mappings = {
#             'resampler': ResamplerConfig,
#             'xattn': GatedXAttnConfig
#         }
        
#         for param, constructor in mappings.items():
#             if param in d:
#                 d[param] = constructor(**d[param])
        
#         return FlamingoConfig(**d)


class FlamingoConfig(PretrainedConfig):
    
    def __init__(self,
                 lm: str = 'gpt2',
                 clip_model_type: str = 'openai/clip-vit-base-patch32',
                 xattn_every: int = 1,
                 
                 dim: int = 1024,
                 dim_visual: int = 768,

                 xattn_dim_head: int = 64,
                 xattn_heads: int = 8,
                 xattn_ff_mult: int = 4,
                 xattn_act: str = 'gelu',
                 xattn = None,
                 
                 resampler_depth: int = 6,
                 resampler_dim_head: int = 64,
                 resampler_heads: int = 8 ,
                 resampler_num_latents: int = 64,
                 resampler_num_time_embeds: int = 4,
                 resampler_ff_mult: int = 4,
                 resampler_act: str = 'gelu',
                 resampler = None,
                 **kwargs):
        
        
        self.lm = lm
        self.clip_model_type = clip_model_type
        self.xattn_every = xattn_every
        self.dim = dim
        self.dim_visual = dim_visual

        self.xattn_dim_head = xattn_dim_head
        self.xattn_heads = xattn_heads
        self.xattn_ff_mult = xattn_ff_mult
        self.xattn_act = xattn_act

        if xattn is not None:
            xattn_keys_to_ignore = ('dim', 'dim_visual')
            # if isinstance(xattn, GatedXAttnConfig):
            #     xattn = vars(xattn)
            for key, value in xattn.items():
                if key not in xattn_keys_to_ignore:
                    setattr(self, f'xattn_{key}', value)

        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head 
        self.resampler_heads = resampler_heads 
        self.resampler_num_latents = resampler_num_latents
        self.resampler_num_time_embeds = resampler_num_time_embeds
        self.resampler_ff_mult = resampler_ff_mult
        self.resampler_act = resampler_act

        if resampler is not None:
            resampler_keys_to_ignore = ('dim')
            # if isinstance(resampler, ResamplerConfig):
            #     resampler = vars(resampler)
            for key, value in resampler.items():
                if key not in resampler_keys_to_ignore:
                    setattr(self, f'resampler_{key}', value)
                
        super().__init__(**kwargs)
            
    # @property
    # def xattn(self):
    #     print('deprecated: xattn property')

    #     return GatedXAttnConfig(
    #         dim=self.dim,
    #         dim_visual=self.dim_visual,
    #         dim_head = self.xattn_dim_head,
    #         heads = self.xattn_heads,
    #         ff_mult = self.xattn_ff_mult,
    #         act = self.xattn_act
    #     )
        
    # @property
    # def resampler(self):
    #     print('deprecated: resampler property')

    #     return ResamplerConfig(
    #         dim = self.dim_visual,
    #         depth = self.resampler_depth,
    #         dim_head = self.resampler_dim_head,
    #         heads = self.resampler_heads,
    #         num_latents=self.resampler_num_latents,
    #         num_time_embeds=self.resampler_num_time_embeds,
    #         ff_mult = self.resampler_ff_mult,
    #         act = self.resampler_act
    #     )
