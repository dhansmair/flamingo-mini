from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union
from PIL import Image

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import GPT2LMHeadModel, GPT2Model, OPTForCausalLM, OPTModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from .configuration_flamingo import FlamingoConfig, is_lm_supported
from .flamingo_processor import FlamingoProcessor
from .perceiver_resampler import PerceiverResampler
from .gated_cross_attention import ModifiedLMBlock


_TRAINABLE_STATE_DICT_KEYWORDS = ('resampler', 'xattn_block', 'lm_head')


class FlamingoBaseModel(PreTrainedModel):
    """ 
    abstract class, which is inherited by FlamingoGPT2 and FlamingoOPT.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.
    """
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)
        self.config: FlamingoConfig = self.config
        
        self.lm: PreTrainedModel = None
        self.lm_head: nn.Linear = None
        self.resampler: PerceiverResampler = PerceiverResampler(
                dim=config.dim_visual,
                depth=config.resampler_depth,
                dim_head = config.resampler_dim_head,
                heads = config.resampler_heads,
                num_latents = config.resampler_num_latents,
                num_time_embeds=config.resampler_num_time_embeds,
                ff_mult=config.resampler_ff_mult,
                act=config.resampler_act,)

        self.modified_layers: List[ModifiedLMBlock] = []
    
    def _init_layers(self, lm_layers: nn.ModuleList):
        """ 
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        modified_layers: List[ModifiedLMBlock] = []
        
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every == 0:
                modified_layer = ModifiedLMBlock(lm_layer,
                                                 dim=self.config.dim,
                                                 dim_visual = self.config.dim_visual,
                                                 dim_head=self.config.xattn_dim_head,
                                                 heads=self.config.xattn_heads,
                                                 ff_mult=self.config.xattn_ff_mult,
                                                 act=self.config.xattn_act)
                modified_layers.append(modified_layer)
                lm_layers[i] = modified_layer
                
        self.modified_layers = modified_layers
    
    def freeze_lm(self):
        """
        set requires_grad = False for all components of the LM.
        (!) in our implementation, lm_head is kept trainable and only 
        initialized from lm_head of GPT2LMHeadModel.
        """
        for param in self.lm.parameters():
            param.requires_grad = False
            
        for param in self.lm_head.parameters():
            param.requires_grad = True

        for param in self.resampler.parameters():
            param.requires_grad = True

        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True
                
    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True
    
    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """
        not precisely right, wte consists of 50257 non-trainable and 1 trainable embedding (for <EOC> token)
        wte.weight will contain both. Not a problem as long as this function is only used to store the trainable
        model components
        """
        dict_trainable = {}
        
        for k, v in self.state_dict().items():
            if any(keyword in k for keyword in _TRAINABLE_STATE_DICT_KEYWORDS):
                dict_trainable[k] = v
            
        return dict_trainable

    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.LongTensor] = None,
                media_locations: Optional[torch.BoolTensor] = None,
                visual_features: Optional[torch.FloatTensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[tuple] = None,
                return_dict: bool = True,
                **kwargs) -> CausalLMOutputWithPast:
        """ Flamingo forward pass 
        
        :param token_ids:       LongTensor (n_batch, n_tokens)
        :param attention_mask:  LongTensor (n_batch, n_tokens)
        :param media_locations: BoolTensor (n_batch, n_tokens) 
            every 'True' value marks the beginning of a new media location, 
            i.e. marks the '<'-token of '<image>' in the original sequence.

        :param visual_features: FloatTensor (n_batch, n_images, n_frames, n_features, dim_feature)
            if resampled_visual_features are given, visual_features are ignored. This is useful for text generation, because during autoregressive generation 
            The output of the perceiver resampler will be the same while a sequence is generated.
        :param resampled_visual_features: Tensor (n_batch, n_image, n_queries, dim_feature)
        """
        assert return_dict == True

        if past_key_values is None:
            xattn_past_key_values, lm_past_key_values = None, None
        else: 
            xattn_past_key_values, lm_past_key_values = past_key_values 

        # perceiver resampler
        # (only need to do if kv of the xattn layers were not calculated yet.)
        # resample visual features (b N T v d) -> (b N T q d)
        if xattn_past_key_values is None and visual_features is not None:
            n_batch = visual_features.shape[0]
            visual_features = rearrange(visual_features, 'b N T v d -> (b N) T v d')
            visual_features = self.resampler(visual_features)
            visual_features = rearrange(visual_features, '(b N) q d -> b N q d', b=n_batch)
            assert visual_features.ndim == 4
            
        # condition xattn layers with the visual features
        # TODO if layer_past is not None, I can ignore the visual features
        for i, xattn in enumerate(self.modified_layers):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(visual_features, media_locations, layer_past)
            
        # pass through LM
        out: BaseModelOutputWithPast = self.lm(input_ids=input_ids, 
                                               attention_mask=attention_mask,
                                               use_cache=use_cache,
                                               past_key_values=lm_past_key_values, 
                                               return_dict=True,
                                               **kwargs)
        
        logits = self.lm_head(out.last_hidden_state)
        
        # collect the past_key_values from the xattn layers
        if use_cache:
            xattn_past_key_values = [] 
            for modified_layer in self.modified_layers:
                xattn_past_key_values.append(modified_layer.kv_output)
                
        # TODO implement loss here
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=(tuple(xattn_past_key_values), out.past_key_values) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
        


class FlamingoGPT2(FlamingoBaseModel):
    """
    new implementation of the flamingo class.
    The implementation of the underlying language model does not need to be altered. 
    Instead, a wrapped LM instance is slighly modified:
    - it's token embeddings are extended with an additional trainable embedding
      for the <EOC> token
    - LM layers are replaced with wrappers to the original LM layers, which also
      contain the gated cross-attention layers.
    """
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert config.lm.startswith('gpt')
        super().__init__(config)

        base_lm = GPT2LMHeadModel.from_pretrained(config.lm)
        self.config.dim = base_lm.config.n_embd        
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: GPT2Model = base_lm.transformer
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.h)
        
    
class FlamingoOPT(FlamingoBaseModel):
    """
    new implementation of the flamingo class.
    The implementation of the underlying language model does not need to be altered. 
    Instead, a wrapped LM instance is slighly modified:
    - it's token embeddings are extended with an additional trainable embedding
      for the <EOC> token
    - LM layers are replaced with wrappers to the original LM layers, which also
      contain the gated cross-attention layers.
    
    This way, the modified LM can be used the same way as the original.
    Plus, it should be easier to adopt the Flamingo model to other LMs like OPT.
    """
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert config.lm.startswith('facebook/opt')
        super().__init__(config)
        
        base_lm = OPTForCausalLM.from_pretrained(config.lm)
        self.config.dim = base_lm.config.hidden_size
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.decoder.layers)
        

class FlamingoModel(PreTrainedModel):
    """
    wrapper class for a FlamingoBase decending model (FlamingoGPT2 or FlamingoOPT)
    
    composition > inheritance
    """
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert isinstance(config, FlamingoConfig)
        assert is_lm_supported(config.lm)
        super().__init__(config)
        self.config: FlamingoConfig = self.config
        
        self.flamingo: FlamingoBaseModel = None
        
        if config.lm.startswith('gpt2'):
            self.flamingo = FlamingoGPT2(config)
        else:
            self.flamingo = FlamingoOPT(config)

    def parameters_trainabale(self):
        """ call freeze_fixed_components() first! """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def freeze_lm(self):
        self.flamingo.freeze_lm()
        
    def unfreeze_lm(self):
        self.flamingo.unfreeze_lm() 
        
    def num_params(self, only_trainable=True):
        if only_trainable:
            self.freeze_lm()
            return sum(p.numel() for p in self.parameters() if p.requires_grad) 
        else:
            return sum(p.numel() for p in self.parameters()) 
        
    def state_dict_trainable(self):
        return self.flamingo.state_dict_trainable()
            
    def forward(self, *args, **kwargs):
        return self.flamingo(*args, **kwargs)

    def prepare_inputs_for_generation(self, 
                                      input_ids: torch.LongTensor, 
                                      visual_features: torch.FloatTensor = None, 
                                      media_locations: torch.LongTensor = None, 
                                      past=None, 
                                      **kwargs) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.
        
        for beam search, input_ids is replicated times the number of beams. 
        I.e., batch_size' = batch_size * num_beams. 
        This function replicates also the visual_features and media_locations accordingly.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        """ 
        
        if visual_features is not None:
            n_inputs = input_ids.shape[0]
            n_visual = visual_features.shape[0]
            
            if n_inputs != n_visual:
                assert n_inputs % n_visual == 0
                visual_features = repeat(visual_features, 'n ... -> (n m) ...', m=n_inputs // n_visual)
                media_locations = repeat(media_locations, 'n ... -> (n m) ...', m=n_inputs // n_visual)
                
        if past is not None:
            input_ids = input_ids[:, -1:]
        
        return dict(
            input_ids=input_ids,
            past_key_values=past,
            visual_features=visual_features,
            media_locations=media_locations,
            **kwargs
        )
    
    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.
        this is required for beam search in combination with use_cache.

        past is a tuple of past_key_values of the xattn layers, and of the LM layers.
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in xattn_past
        )
        
        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in lm_past
        )
        
        return xattn_past_beam, lm_past_beam
    
    @torch.no_grad()
    def generate_captions(self, 
                          processor: FlamingoProcessor, 
                          visual_features: torch.FloatTensor= None, 
                          images: Union[Image.Image, List[Image.Image]] = None,
                          prompt="<image>", max_length=150, num_beams=1,
                          ):
        """
        helper utility for image captioning.
        prompt is replicated for all batches.
        """
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
                
            assert visual_features is None, "you can only pass either images or visual features to generate_captions()!"
            visual_features = processor.extract_features(images)

        assert visual_features is not None, "you must pass either images or visual features to generate_captions()!"
        
        if visual_features.ndim == 2:
            visual_features = rearrange(visual_features, 'f d -> 1 1 1 f d') 
            
        elif visual_features.ndim == 3:
            visual_features = rearrange(visual_features, 'b f d -> b 1 1 f d')
            
        batch_size = visual_features.shape[0]
        device = visual_features.device
        input_ids, media_locations, attention_mask = processor.encode_text(prompt, device)
        input_ids = repeat(input_ids[0], 'l -> n l', n=batch_size)
        media_locations = repeat(media_locations[0], 'l -> n l', n=batch_size)
        attention_mask = repeat(attention_mask[0], 'l -> n l', n=batch_size)
            
        out_ids = self.generate(inputs=input_ids,
                              visual_features=visual_features,
                              media_locations=media_locations,
                              attention_mask=attention_mask,
                              num_beams=num_beams,
                              early_stopping=True,
                              use_cache=True,
                              bos_token_id=self.flamingo.lm.config.bos_token_id,
                              eos_token_id=self.flamingo.lm.config.eos_token_id,
                              pad_token_id=self.flamingo.lm.config.eos_token_id,
                              max_length=max_length)
        
        captions = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        captions = [processor.remove_tags(t) for t in captions]
        return captions
    
