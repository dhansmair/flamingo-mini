import os
import pickle
from random import choice
from typing import Callable, List, Optional, Tuple
from PIL import Image
import logging

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ConceptualCaptions(Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        eoc_token: str = "<EOC>",
        eos_token: str = "</s>",
        output_dims = (1, 3, 224, 224),
    ):
        assert split in ('train', 'val'), 'split must be \'train\' or \'val\'.'
        assert os.path.isdir(root), 'CC3M root is not an exisiting directory!'
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else ToTensor()
        self.prefixes = ['<image>', ' <image>']
        self.data = self._load_data()
        self.suffix = eoc_token + eos_token
        self.output_dims = output_dims

    def _load_data(self) -> List[Tuple[str, str]]: 
        with open(f"{self.root}/cleaned_index_{self.split}.pkl", 'rb') as f:
            return pickle.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, str]:
        image_id, caption = self.data[index]
        sentence = ''.join([choice(self.prefixes), caption, self.suffix])
        
        try:
            image = Image.open(f"{self.root}/{self.split}/{image_id}.jpg").convert('RGB')
            pixels = self.transform(image)
            del image
        except Exception as e:
            logging.warning(e)
            pixels = torch.zeros(self.output_dims, dtype=torch.float)

        return int(image_id), pixels, sentence
        