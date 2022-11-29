"""
Dataset builder class
This is something like a factory
"""

from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T 
from transformers.models.clip.feature_extraction_clip import CLIPFeatureExtractor

from flamingo_mini import FlamingoConfig

from .. import paths

from .cc3m import ConceptualCaptions
from .coco import MyCocoCaptions
from .randaugment import RandomAugment


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPFeatureExtractor.from_pretrained(clip_model_type)

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(
            images=image,
            return_tensors="pt",
            padding=True
        )['pixel_values']


class DatasetBuilder:

    def __init__(
        self,
        config: FlamingoConfig,
        tokenize_in_dataset=False,
        seq_length: Optional[int] = None
    ):
        self.config = config
        self.tokenize_in_dataset = tokenize_in_dataset
        self.seq_length = seq_length

    def build_coco_train(self, **kwargs) -> Dataset:
        transform = T.Compose([
            T.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            T.ToTensor(),
            CLIPImageTransform(self.config.clip_model_type)
        ])

        return MyCocoCaptions(
            paths.coco_img_dir, 
            paths.coco_ann_train, 
            transform=transform,
            **kwargs
        )
        
    def build_coco_val(self) -> Dataset:
        transform = CLIPImageTransform(self.config.clip_model_type)

        return MyCocoCaptions(
            paths.coco_img_dir, 
            paths.coco_ann_val, 
            transform=transform,
        )

    def build_cc3m_train(self, **kwargs) -> Dataset:
        transform = T.Compose([
            T.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']), 
            T.ToTensor(),
            CLIPImageTransform(self.config.clip_model_type)
        ])
        
        return ConceptualCaptions(
            paths.cc3m_root,
            split='train',
            transform=transform,
            **kwargs
        )
        
    def build_cc3m_val(self) -> Dataset:
        transform = CLIPImageTransform(self.config.clip_model_type)

        return ConceptualCaptions(
            paths.cc3m_root,
            split='val',
            transform=transform,
        )
        
    def build_train_sets(self, dataset_names: List[str]) -> List[Dataset]:
        """builds a list or train sets for training on natural language."""
        datasets = []
        
        name_to_build = {
            'coco': self.build_coco_train,
            'cc3m': self.build_cc3m_train,
        }
        
        for dataset_name in dataset_names:
            f = name_to_build[dataset_name]
            datasets.append(f())
            
        return datasets
