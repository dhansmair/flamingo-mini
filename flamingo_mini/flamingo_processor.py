from __future__ import annotations
from typing import List, Tuple
from PIL import Image

import torch
from transformers import CLIPImageProcessor

from .configuration_flamingo import FlamingoConfig


class FlamingoProcessor:
    """ 
    FlamingoProcessor offers functions to preprocess the raw data (images and text).
    Wrapper around a transformer GPT-2 tokenizer and a clip processor.
    """
    
    vision_processor: CLIPImageProcessor

    def __init__(
        self,
        config: FlamingoConfig,
        use_fast: bool = True,
        eoc_token: str = '<EOC>'
    ):
        """
        Args:
            config (FlamingoConfig): pass the same FlamingoConfig as used to initialize the FlamingoModel.
            use_fast (bool): whether to use the fast tokenizer implementations from huggingface.
            eoc_token (str): string representation of the token to add.
        """
        self.config = config
        self.eoc_token = eoc_token
        self.vision_processor = CLIPImageProcessor.from_pretrained(config.clip_model_type) #type: ignore
        
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

    def encode_text(
        self,
        text: str | List[str],
        device: torch.device | None = None,
        max_length=None,
        length=None,
        return_tensors='pt',
        return_attention_mask=True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if length is not None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                padding='max_length',
                truncation=True,
                max_length=length)
        elif max_length is None:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors, 
                padding=True)
        else:
            result = self.tokenizer(
                text,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
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

    def __call__(
        self, 
        images: Image.Image | List[Image.Image] | torch.Tensor | List[torch.Tensor] | None = None, 
        text: str | List[str] | None = None, 
        device: torch.device | None = None
    ):
        result = {}
        
        if images is not None:
            result['pixel_values'] = self.vision_processor(images=images, return_tensors='pt', padding=True)['pixel_values'].to(device)
            
        if text is not None:
            input_ids, media_locations, attention_mask = self.encode_text(text, device=device)
            result['input_ids'] = input_ids
            result['media_locations'] = media_locations
            result['attention_mask'] = attention_mask

        return result
