from transformers.configuration_utils import PretrainedConfig


class FlamingoConfig(PretrainedConfig):
    
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
        freeze_language_model: bool = True,
        freeze_vision_model: bool = True,
        **kwargs
    ):
        """ Flamingo Configuration Class
        
        Args:
            lm (str): huggingface identifier of the language model. supported are 'gpt2' variations and 'facebook/opt-*'
            clip_model_type (str): huggingface identifier of the vision encoder.
            dim (int): LM embedding size
            dim_visual (int): Vision encoder embedding size
            xattn_every (int): frequency of interleaved gated xattn layers.
            xattn_dim_head (int): inner dim of xattn heads
            xattn_heads (int): number of attention heads in the xattn layers
            xattn_ff_mult (int): ?
            xattn_act (str): activation function to use in the xattn layers. Flamingo used 'sqrelu' in their paper.
            resampler_depth (int): number of attention layers in the perceiver resampler.
            resampler_dim_head: inner dim of resampler attention heads
            resampler_heads (int): number of attention heads in the resampler
            resampler_num_latents (int): number of learnable queries in the resampler
            resampler_num_time_embeds (int): ?
            resampler_ff_mult (int): ?
            resampler_act (str): activation function of the resampler. Flamingo used 'sqrelu' in their paper.
            freeze_language_model (bool): whether to freeze the language model or not.
            freeze_vision_model (bool): whether to freeze the vision model or not.
        """
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
        self.freeze_language_model = freeze_language_model
        self.freeze_vision_model = freeze_vision_model