from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import repeat_many
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from .configuration_flamingo import FlamingoConfig
from .flamingo_processor import FlamingoProcessor
from .gated_cross_attention import ModifiedLMBlock
from .perceiver_resampler import PerceiverResampler
from .utils import get_common_prefix_length



@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield




class FlamingoBaseModel(ABC, PreTrainedModel):
    """ 
    abstract class, which is inherited by FlamingoGPT2 and FlamingoOPT.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.
    """
    lm: PreTrainedModel
    lm_head: nn.Linear
    resampler: PerceiverResampler
    modified_layers: List[ModifiedLMBlock]
    vision_encoder: CLIPVisionModel

    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig, suppress_warnings=True):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)
        
        with suppress_model_loading_warnings(suppress_warnings):
            self.vision_encoder = CLIPVisionModel.from_pretrained(config.clip_model_type)

        self.resampler = PerceiverResampler(
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
        self.modified_layers = []

    def _init_layers(self, lm_layers: nn.ModuleList):
        """ 
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        self.modified_layers = []

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
            
    def freeze_vm(self):
        """freeze vision model """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        """ freeze weights of the language model.

        (!) does not freeze token embedding matrix and gated xattn layers
        """

        for param in self.lm.parameters():
            param.requires_grad = False

        # lm_head shares weights with the embeddings so no need to unfreeze that as well
        self.lm.get_input_embeddings().weight.requires_grad = True

        for xattn in self.modified_layers:
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [
            w for w, t in self.named_parameters() if t.requires_grad]
        return {k: v for k, v in self.state_dict().items() if k in trainable_param_names}

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        #visual_features: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
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
            media_locations (BoolTensor):   shape (n_batch, n_tokens).
                indicates the locations of the starts of the <image> tags beginning, i.e. the location of the token representing '<'
            #visual_features (FloatTensor):  shape (n_batch, n_images, n_frames, n_features, dim_feature).
            
            pixel_values (torch.Tensor | None):    shape [b N T c h w]. Optional.


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

        assert return_dict
        batch_size = input_ids.size(0)

        if past_key_values is None:
            xattn_past_key_values, lm_past_key_values = None, None
        else:
            xattn_past_key_values, lm_past_key_values = past_key_values
            
        visual_features = None

        if xattn_past_key_values is None and pixel_values is not None:

            assert pixel_values.size(0) == batch_size, \
                "visual_features must have the same batch size as the textual input!"
                
            if pixel_values.ndim == 4:              # (b c h w)
                b, N, T = pixel_values.size(0), 1, 1
            
            elif pixel_values.ndim == 5:              # (b N c h w)
                b, N, T = *pixel_values.shape[:1], 1
                pixel_values = rearrange(pixel_values, 'b N c h w -> (b N) c h w')

            elif pixel_values.ndim == 6:            # (b N T c h w) -> (b N T v d)
                b, N, T = pixel_values.shape[:2]
                pixel_values = rearrange(pixel_values, 'b N T c h w -> (b N T) c h w')
            else:
                raise ValueError('pixel_values must have ndim 5 or 6!')

            visual_features = self.vision_encoder(pixel_values).last_hidden_state         # (b N T) v d
            # visual_features = rearrange(visual_features, '(b N T) v d -> b N T v d', b=b, N=N, T=T)

            # perceiver resampler
            # (only need to do if kv of the xattn layers were not calculated yet.)
            # resample visual features ((b N T) v d) -> (b N T q d)
            visual_features = rearrange(visual_features, '(b N T) v d -> (b N) T v d', b=b, N=N, T=T)
            visual_features = self.resampler(visual_features)

            # T is gone at this point
            visual_features = rearrange(visual_features, '(b N) q d -> b N q d', b=b, N=N)

        if visual_features is None:
            # use dummy visual features.
            # This should not have an effect on the outcome of the model, unless media_locations is set incorrectly.
            visual_features = torch.zeros((batch_size, 1, self.config.resampler_num_latents, self.config.dim_visual),
                                          dtype=torch.float32,
                                          device=input_ids.device)

        if media_locations is None:
            media_locations = torch.zeros_like(input_ids)

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
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)

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
        from transformers import GPT2LMHeadModel, GPT2Model
        assert config.lm.startswith('gpt')
        super().__init__(config)

        base_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(config.lm)  # type: ignore
        
        assert self.config.dim == base_lm.config.n_embd, \
            f"specified {self.config.dim=} in FlamingoConfig, but {config.lm} has hidden size={base_lm.config.n_embd}"

        
        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: GPT2Model = base_lm.transformer
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.h)


class FlamingoOPT(FlamingoBaseModel):
    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig):
        from transformers import OPTForCausalLM, OPTModel
        assert config.lm.startswith('facebook/opt')
        super().__init__(config)

        base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(config.lm)  # type: ignore

        assert self.config.dim == base_lm.config.hidden_size, \
            f"specified {self.config.dim=} in FlamingoConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        base_lm.resize_token_embeddings(base_lm.config.vocab_size + 1)
        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self._init_layers(self.lm.decoder.layers)


class FlamingoModel(PreTrainedModel):
    """wrapper class for a FlamingoBase decending model (FlamingoGPT2 or FlamingoOPT)

    A generic flamingo interface that is independent of the underlying LM. Most of the methods are just forwarding to the actual model.
    This class implements prepare_inputs_for_generation() and reorder_cache(), which are required to utilize hf text generation methods.
    It also has a generate_captions() utility that can be used to create a caption for an image.
    """
    config: FlamingoConfig
    config_class = FlamingoConfig

    # key = prefix of an existing pretrained huggingface transformer language model
    # value = Flamingo class for the respective language model
    _LANGUAGE_MODEL_VERSIONS = {
        'gpt2': FlamingoGPT2,
        'facebook/opt': FlamingoOPT
    }
    
    _keys_to_ignore_on_load_missing = [r"flamingo.vision_encoder"]

    def __init__(self, config: FlamingoConfig, model_class: type | None = None):
        """constructor.

        Args:
            config (FlamingoConfig): 
                config for the flamingo model.
            model_class (Optional[type], optional): 
                optionally use a custom class that inherits FlamingoBaseModel. 
                If none, it will choose FlamingoGPT2 or FlamingoOPT based on the FlamingoConfig. Defaults to None.
        """
        super().__init__(config)

        if model_class is None:
            model_class = self._find_flamingo_class(config.lm)
        self.flamingo: FlamingoBaseModel = model_class(config)

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
        input_ids: torch.Tensor,
        #visual_features: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        past=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        for beam search, input_ids is replicated times the number of beams. 
        I.e., batch_size' = batch_size * num_beams. 
        This function replicates also the visual_features and media_locations accordingly.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        """

        # if visual_features is not None:
        #     n_inputs = input_ids.shape[0]
        #     n_visual = visual_features.shape[0]

        #     if n_inputs != n_visual:
        #         assert n_inputs % n_visual == 0
        #         visual_features = repeat(
        #             visual_features, 'n ... -> (n m) ...', m=n_inputs // n_visual)
        if pixel_values is not None:
            n_inputs = input_ids.shape[0]
            n_visual = pixel_values.shape[0]

            if n_inputs != n_visual:
                assert n_inputs % n_visual == 0
                pixel_values = repeat(
                    pixel_values, 'n ... -> (n m) ...', m=n_inputs // n_visual)
            

        if media_locations is not None:
            n_inputs = input_ids.shape[0]
            n_inputs_media = media_locations.shape[0]

            if n_inputs != n_inputs_media:
                assert n_inputs % n_inputs_media == 0
                media_locations = repeat(
                    media_locations, 'n ... -> (n m) ...', m=n_inputs // n_inputs_media)

        if past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            past_key_values=past,
            #visual_features=visual_features,
            media_locations=media_locations,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
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
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    @torch.no_grad()
    def generate_captions(
        self,
        processor: FlamingoProcessor,
        # visual_features: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        images: Image.Image | List[Image.Image] | None = None,
        prompt: str = "<image>",
        max_length: int = 150,
        num_beams: int = 1,
        device: torch.device | None = None
    ):
        """
        helper utility for image captioning.
        prompt is replicated for all batches.
        """
        if images is not None:
            assert pixel_values is None, "you can only pass either images or visual features to generate_captions()!"

            if isinstance(images, Image.Image):
                images = [images]

            pixel_values = processor(images=images, device=device)['pixel_values']

        assert pixel_values is not None, "you must pass either images or visual features to generate_captions()!"

        batch_size = pixel_values.size(0)
        input_ids, media_locations, attention_mask = processor.encode_text(
            prompt, device)

        input_ids = repeat(input_ids[0], 'l -> n l', n=batch_size)
        media_locations = repeat(media_locations[0], 'l -> n l', n=batch_size)
        attention_mask = repeat(attention_mask[0], 'l -> n l', n=batch_size)

        out_ids = self.generate(
            inputs=input_ids,
            # visual_features=visual_features,
            media_locations=media_locations,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            num_beams=num_beams,
            early_stopping=True,
            use_cache=True,
            bos_token_id=self.flamingo.lm.config.bos_token_id,
            eos_token_id=self.flamingo.lm.config.eos_token_id,
            pad_token_id=self.flamingo.lm.config.eos_token_id,
            max_length=max_length
        )

        captions = processor.tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        captions = [processor.remove_tags(t) for t in captions]
        return captions

    @torch.no_grad()
    def score_sequences(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        media_locations: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int = 100000,
    ) -> torch.Tensor:
        """

        EXPERIMENTAL

        This method can be used for zero-shot classification.
        Given a batch of tokenized sentences, it computes the log-prob over each sample.

        inspired by ALBEF:
            https://github.com/salesforce/ALBEF/blob/b9727e43c3040491774d1b22cc27718aa7772fac/models/model_vqa.py#L149

        To improve efficiency, the implementation works like this:
        (1) find the longest common prefix over all sequences.
        (2) pass the prefix once and obtain the attention keys and values for LM and xattn layers
        (3) based on the likelihood for the next token, filter the top-k sequences for the next steps.
        (4) repeat keys and values for all top-k sequences
        (5) pass the top-k sequences to the model. use cached kv
        (6) compute the log-prob from the remaining parts and use as a score.
            For all sequences that didn't make it to the top-k, set the score to -inf

        TODO method fails when all sequences are equal

        Args:
            visual_features (torch.FloatTensor):    [N 1 q d]
                (!) the visual features are treated as the same for the complete batch of sentences
            input_ids (torch.Tensor):           [b L]
            media_locations (torch.Tensor):     [b L]
            attention_mask (torch.Tensor):      [b L]

        Returns:
            torch.Tensor: log-probs for the batch of input sequences
                Tensor of shape [b], dtype torch.float 
        """

        assert visual_features.ndim == 4, f"visual features must have shape [N 1 q d], but has {visual_features.ndim} dimensions!"

        n_choices = input_ids.size(0)
        n_reuse = get_common_prefix_length(input_ids)
        k = min(k, n_choices)

        # first, pass the complete prefix and compute the hidden states.
        out = self.flamingo(
            input_ids=input_ids[:1, :n_reuse],
            media_locations=media_locations[:1, :n_reuse],
            attention_mask=attention_mask[:1, :n_reuse],
            # add outermost dimension [N 1 q d] -> [1 N 1 q d]
            visual_features=visual_features.unsqueeze(0),
            use_cache=True,
        )

        next_tokens = input_ids[:, n_reuse]
        next_token_logits = out.logits[0, -1, :].index_select(0, next_tokens)
        topk_indices = next_token_logits.topk(k).indices

        # extend past_key_values to all sequences
        xattn_past_key_values = [
            tuple(repeat_many(kv, "1 ... -> b ...", b=k))
            for kv in out.past_key_values[0]
        ]
        lm_past_key_values = [
            (
                repeat(keys, "1 ... -> b ...", b=k)[:, :, :-1, :],
                repeat(vals, "1 ... -> b ...", b=k)[:, :, :-1, :],
            )
            for keys, vals in out.past_key_values[1]
        ]

        past_key_values = (xattn_past_key_values, lm_past_key_values)

        # then pass all choice sequences individually.
        choice_input_ids = input_ids[topk_indices, n_reuse - 1:]
        choice_media_locations = media_locations[topk_indices]
        choice_attention_mask = attention_mask[topk_indices]

        # at this point, we don't need the visual features anymore, since they have already been passed through
        # the perceiver resampler and the keys and values for them have been precomputed in the xattn layers.
        out2 = self.flamingo(
            input_ids=choice_input_ids,
            media_locations=choice_media_locations,
            attention_mask=choice_attention_mask,
            visual_features=None,
            past_key_values=past_key_values,
            labels=choice_input_ids,
            loss_reduction="none",
        )

        losses = out2.loss.reshape((k, -1)).sum(dim=1)

        # copy the losses over to another vector
        scores = torch.full(
            [n_choices], torch.finfo(torch.float).min, device=losses.device
        )
        scores[topk_indices] = -losses
        return scores.detach()
