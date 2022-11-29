import os
from random import choice
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image

import torch
from torchvision.datasets import CocoCaptions


class MyCocoCaptions(CocoCaptions):
    
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        eoc_token: str = "<EOC>",
        eos_token: str = "</s>",
    ):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.prefixes = ['<image>', ' <image>']
        self.suffix = eoc_token + eos_token
        
    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, str]:
        """
        There is 5 captions per image in COCO. For training, we choose one at random.
        """
        pixels, captions = super().__getitem__(index)
        sentence = ''.join([choice(self.prefixes), choice(captions), self.suffix])
        return self.ids[index], pixels, sentence