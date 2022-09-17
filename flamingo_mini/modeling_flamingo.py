from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from einops import rearrange, repeat
from transformers import GPT2LMHeadModel, GPT2Model, OPTForCausalLM, OPTModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from .configuration_flamingo import FlamingoConfig
from .flamingo_processor import FlamingoProcessor
from .perceiver_resampler import PerceiverResampler
from .gated_cross_attention import ModifiedLMBlock


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
        self.config: FlamingoConfig = self.config           # just to enable type hints on self.config
        
        self.lm: PreTrainedModel = None                     # set in child class
        # self.lm_head: nn.Linear = None                      # set in child class
        self.resampler: PerceiverResampler = PerceiverResampler(
            dim=config.dim_visual,
            depth=config.resampler_depth,
            dim_head=config.resampler_dim_head,
            heads=config.resampler_heads,
            num_latents=config.resampler_num_latents,
            num_time_embeds=config.resampler_num_time_embeds,
            ff_mult=config.resampler_ff_mult,
            act=config.resampler_act
        )

        # a list of direct references to the augmented lm layers
        self.modified_layers: List[ModifiedLMBlock] = []
    
    def _init_layers(self, lm_layers: nn.ModuleList):
        """ 
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        self.modified_layers: List[ModifiedLMBlock] = []
        
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every != 0: 
                continue

            modified_layer = ModifiedLMBlock(
                lm_layer,
                dim=self.config.dim,
                dim_visual=self.config.dim_visual,
                dim_head=self.config.xattn_dim_head,
                heads=self.config.xattn_heads,
                ff_mult=self.config.xattn_ff_mult,
                act=self.config.xattn_act,
                n_visual=self.config.resampler_num_latents
            )
            self.modified_layers.append(modified_layer)
            lm_layers[i] = modified_layer
    
    def freeze_lm(self):
        """
        set requires_grad = False for all components of the LM.
        Must be implemented in child classes
        """
        raise ValueError("implement freeze_lm() in a child class!")
                
    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [w for w, t in self.named_parameters() if t.requires_grad]
        return {k:v for k, v in self.state_dict().items() if k in trainable_param_names}
    
    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        media_locations: Optional[torch.BoolTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[tuple] = None,
        return_dict: bool = True,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Flamingo forward pass
        
        Most of the parameters are inspired by huggingface language model implementations, so this doc may be informative:
        https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.forward
        
        Args:
            input_ids (LongTensor):         shape (n_batch, n_tokens). the tokenized input text
            attention_mask (LongTensor):    shape (n_batch, n_tokens). 
                Mask as produced by the tokenizer. Required when a batch of input strings are tokenized and thus padded at the end.
                Then this will indicate the locations of 'real' tokens vs. the location of 'pad' tokens.
                TODO why is this a LongTensor and not a BoolTensor?
            media_locations (BoolTensor):   shape (n_batch, n_tokens).
                indicates the locations of the starts of the <image> tags beginning, i.e. the location of the token representing '<'
            visual_features (FloatTensor):  shape (n_batch, n_images, n_frames, n_features, dim_feature).
            use_cache (bool): whether to return the inner keys and values. Used to speed up text generation at inference. defaults to False
            past_key_values (tuple): tuple of past_key_values of (1) the xattn layers (2) the language model
            return_dict (bool): Whether to return a dictionary. Right now, only dicts are supported, so this must be set to True. Defaults to True.
            labels (LongTensor): 
                It is possible to pass the exact value as input_ids also as labels. If present, the output will contain a CE loss of the next token prediction.
                optional, defaults to None
            **kwargs
        
        Returns:
            (CausalLMOutputWithPast): an object containing all the useful stuff. Refer to hf documentation.
        
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
            
        # condition xattn layers
        for i, xattn in enumerate(self.modified_layers):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(visual_features, media_locations, layer_past)
            
        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=lm_past_key_values, 
            return_dict=True,
            **kwargs
        )
        
        logits: torch.FloatTensor = self.lm_head(out.last_hidden_state)
        
        # collect the past_key_values from the xattn layers
        if use_cache:
            xattn_past_key_values = [] 
            for modified_layer in self.modified_layers:
                xattn_past_key_values.append(modified_layer.kv_output)
                
        loss = None
        if labels is not None:
            # loss function calculation, inspired by hf implementations
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()     # logits shape (batch, seq_length, #words)
            shift_labels = labels[..., 1:].contiguous()         # labels shape (batch, seq_length)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(tuple(xattn_past_key_values), out.past_key_values) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


class FlamingoGPT2(FlamingoBaseModel):
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert config.lm.startswith('gpt')
        super().__init__(config)

        base_lm = GPT2LMHeadModel.from_pretrained(config.lm)
        self.config.dim = base_lm.config.n_embd        
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: GPT2Model = base_lm.transformer
        
        # copy the linear layer over to this class. With the previous line self.lm_head = base_lm.lm_head 
        # the lm_head was for some reason not included in model.parameters()
        self.lm_head = nn.Linear(base_lm.lm_head.in_features, base_lm.lm_head.out_features, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(base_lm.lm_head.weight)
        
        self._init_layers(self.lm.h)
        
    def freeze_lm(self):
        """ freeze weights of the language model.
        
        (!) does not freeze token embedding matrix and gated xattn layers
        """
        
        for param in self.lm.parameters():
            param.requires_grad = False
            
        self.lm.wte.weight.requires_grad = True
            
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True
                
    
class FlamingoOPT(FlamingoBaseModel):
    config_class = FlamingoConfig
    
    def __init__(self, config: FlamingoConfig):
        assert config.lm.startswith('facebook/opt')
        super().__init__(config)
        
        base_lm = OPTForCausalLM.from_pretrained(config.lm)
        self.config.dim = base_lm.config.hidden_size
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: OPTModel = base_lm.model

        # copy the linear layer over to this class. With the previous line self.lm_head = base_lm.lm_head 
        # the lm_head was for some reason not included in model.parameters()
        self.lm_head = nn.Linear(base_lm.lm_head.in_features, base_lm.lm_head.out_features, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(base_lm.lm_head.weight)

        self._init_layers(self.lm.decoder.layers)
        
    def freeze_lm(self):
        """ freeze weights of the language model.
        
        (!) does not freeze token embedding matrix and gated xattn layers
        """
        
        for param in self.lm.parameters():
            param.requires_grad = False
            
        self.lm.decoder.embed_tokens.weight.requires_grad = True
            
        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True
        

class FlamingoModel(PreTrainedModel):
    """wrapper class for a FlamingoBase decending model (FlamingoGPT2 or FlamingoOPT)
    
    A generic flamingo interface that is independent of the underlying LM. Most of the methods are just forwarding to the actual model.
    This class implements prepare_inputs_for_generation() and reorder_cache(), which are required to utilize hf text generation methods.
    It also has a generate_captions() utility that can be used to create a caption for an image.
    """
    config_class = FlamingoConfig
    
    # key = prefix of an existing pretrained huggingface transformer language model
    # value = Flamingo class for the respective language model
    _LANGUAGE_MODEL_VERSIONS = {
        'gpt2': FlamingoGPT2,
        'facebook/opt': FlamingoOPT
    }
    
    def __init__(self, config: FlamingoConfig):
        super().__init__(config)
        
        flamingo_class = self._find_flamingo_class(config.lm)
        self.flamingo: FlamingoBaseModel = flamingo_class(config)
        
    @classmethod
    def is_lm_supported(cls, lm_id: str) -> bool:
        return any(lm_id.startswith(prefix) for prefix in cls._LANGUAGE_MODEL_VERSIONS.keys())
            
    @classmethod
    def _find_flamingo_class(cls, language_model_id: str):
        for prefix, flamingo_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return flamingo_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return self.flamingo.parameters_trainable()
    
    def freeze_lm(self):
        self.flamingo.freeze_lm()
        
    def unfreeze_lm(self):
        self.flamingo.unfreeze_lm() 
        
    def state_dict_trainable(self):
        return self.flamingo.state_dict_trainable()
            
    def forward(self, *args, **kwargs) -> CausalLMOutputWithPast:
        return self.flamingo(*args, **kwargs)

    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor, 
        visual_features: torch.FloatTensor = None, 
        media_locations: torch.LongTensor = None, 
        past=None, 
        **kwargs
    ) -> Dict[str, Any]:
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
        
        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
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
    def generate_captions(
        self, 
        processor: FlamingoProcessor, 
        visual_features: Optional[torch.FloatTensor] = None, 
        images: Union[Image.Image, List[Image.Image]] = None,
        prompt: str = "<image>", 
        max_length: int = 150, 
        num_beams: int = 1
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
            
        out_ids = self.generate(
            inputs=input_ids,
            visual_features=visual_features,
            media_locations=media_locations,
            attention_mask=attention_mask,
            num_beams=num_beams,
            early_stopping=True,
            use_cache=True,
            bos_token_id=self.flamingo.lm.config.bos_token_id,
            eos_token_id=self.flamingo.lm.config.eos_token_id,
            pad_token_id=self.flamingo.lm.config.eos_token_id,
            max_length=max_length
        )
        
        captions = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        captions = [processor.remove_tags(t) for t in captions]
        return captions
    