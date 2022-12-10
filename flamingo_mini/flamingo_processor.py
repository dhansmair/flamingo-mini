from __future__ import annotations
import contextlib
from typing import List, Tuple
import logging
from PIL import Image

import torch
from einops import rearrange
from transformers.models.clip.feature_extraction_clip import \
    CLIPFeatureExtractor

from .configuration_flamingo import FlamingoConfig


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


class FlamingoProcessor:
    """ 
    FlamingoProcessor offers functions to preprocess the raw data (images and text).
    Wrapper around a transformer GPT-2 tokenizer and a clip processor.
    """

    def __init__(
        self,
        config: FlamingoConfig,
        device: torch.device | None = None,
        load_tokenizer: bool = True,
        load_vision_model: bool = False,
        use_fast: bool = True,
        suppress_warnings: bool = True,
        eoc_token: str = '<EOC>'
    ):
        """
        Args:
            config (FlamingoConfig): pass the same FlamingoConfig as used to initialize the FlamingoModel.
            device (torch.device | None): if passed, vision_model will be directly loaded onto the device.
            load_tokenizer (bool): whether to load the tokenizer or not.
            load_vision_model (bool): whether to load the vision_model or not. In some cases we only need 
                the tokenizer, then not loading the vision_model will save time.
            use_fast (bool): whether to use the fast tokenizer implementations from huggingface.
            suppress_warnings (bool): when loading only the CLIPVisionModel from the checkpoint,
                from_pretrained() will log a warning that some weights have not been used. We can ignore this.
        """
        self.config = config
        self.device = device
        self.eoc_token = eoc_token
        self.vision_processor = CLIPFeatureExtractor.from_pretrained(config.clip_model_type)
        
        if load_vision_model:
            from transformers.models.clip.modeling_clip import CLIPVisionModel

            with suppress_model_loading_warnings(suppress_warnings):
                self.vision_model = CLIPVisionModel.from_pretrained(config.clip_model_type)
                self.vision_model.to(device)
        else:
            self.vision_model = None
        
        if load_tokenizer:
            if config.lm.startswith('gpt2'):
                if use_fast:
                    from transformers import GPT2TokenizerFast

                    self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                else:
                    from transformers import GPT2Tokenizer

                    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            elif config.lm.startswith('facebook/opt'):
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b', use_fast=use_fast)
            
            self.tokenizer.add_bos_token = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_tokens(self.eoc_token)

            # find the start token for "<image>". " <" is 1279, "<" is 27
            # the encoded "<" token-id is different if there is a preceding whitespace.
            #        with ws    without
            # gpt-2:  1279         27
            # opt:   28696      51552
            self.leq_ids = [
                self.tokenizer.encode("<")[-1],
                self.tokenizer.encode(" <")[-1]
            ]
            
    def to(self, device: torch.device | None):
        self.device = device
        self.dummy_output = self.dummy_output.to(device)
        
        if self.vision_model is not None:
            self.vision_model.to(device)

    def encode_text(
        self,
        text: str | List[str],
        device: torch.device | None = None,
        max_length=None,
        length=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if length is not None:
            result = self.tokenizer(
                text,
                return_tensors='pt',
                return_attention_mask=True,
                padding='max_length',
                truncation=True,
                max_length=length)
        elif max_length is None:
            result = self.tokenizer(
                text,
                return_tensors='pt', 
                padding=True)
        else:
            result = self.tokenizer(
                text,
                return_tensors='pt',
                return_attention_mask=True,
                padding=True,
                truncation=True,
                max_length=max_length)
            
            
        media_locs = self.get_media_locations(result.input_ids)

        return result.input_ids.to(device), media_locs.to(device), result.attention_mask.to(device)
    
    def prepare_caption(self, caption: str) -> str:
        # <BOS> token is added automatically by the tokenizer.
        # <EOS> token is not.
        return "<image>" + caption + self.eoc_token + self.tokenizer.eos_token
            
    def prepare_captions(self, captions: List[str]) -> List[str]:
        """preparation function for the conceptual captions dataset. """
        return [self.prepare_caption(c) for c in captions]
        
    def _remove_tags(self, text: str) -> str:
        for s in ('<image>', self.tokenizer.eos_token, self.eoc_token, self.tokenizer.pad_token):
            text = text.replace(s, '')
        return text.strip()
    
    def remove_tags(self, text: str | List[str]) -> str | List[str]:
        if isinstance(text, str):
            return self._remove_tags(text)
        else:
            return [self._remove_tags(t) for t in text]
    
    def get_media_locations(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([(input_ids == leq_id) for leq_id in self.leq_ids]).sum(0)
    
    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        :param images: a list of PIL image instances
        :return: Tensor of shape [n_images, width, height, depth]
        """
        return self.vision_processor(images=images, return_tensors="pt", padding=True)

    def extract_features(
            self,
            images: Image.Image | torch.Tensor | List[Image.Image] | List[torch.Tensor],
            to_device: bool = True
        ) -> torch.Tensor:
        
        if self.vision_model is None:
            raise ValueError("flamingo processor not initialized with vision processor")
        
        if isinstance(images, Image.Image):
            images = [images]
        
        pixels = self.vision_processor(images=images, return_tensors="pt", padding=True)
        pixels = pixels['pixel_values']

        if to_device:
            pixels = pixels.to(self.device)

        return self.vision_model(pixels).last_hidden_state
    
    def extract_features_from_preprocessed(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        only does the extraction step.
        Assuming pixels have been extracted from image(s) by a CLIPFeatureExtractor

        Args:
            pixels (torch.Tensor) expected shape [b N c=3 h w]
                b = batch size
                N = #images
                c = channels have to be 3
                h = height
                w = width
            
        Returns:
            (torch.Tensor) shape [b N T=1 v d]
            where
                b = batch size
                N = #images
                T = #frames for video, but video is not actively supported by flamingo_mini
                v = visual features
                d = dimensionality of the visual features
        """
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, 'b N c h w -> (b N) c h w')
        visual_features = self.vision_model(pixels).last_hidden_state
        return rearrange(visual_features, '(b N) v d -> b N 1 v d', b=batch_size)
