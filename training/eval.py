from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap

from flamingo_mini import FlamingoModel, FlamingoProcessor


class MyDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_id = self.dataset.ids[index]
        return image_id, image


@torch.no_grad()
def evaluate_image_captioning(
    dataset: CocoCaptions,
    model: FlamingoModel, 
    *,
    prefix: str = "<image>",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    results: List[dict] = []
    
    wrapper = MyDatasetWrapper(dataset)
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers)

    for image_ids, pixels in tqdm(loader, disable=not verbose):
        captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix
        )
        
        for image_id, caption in zip(image_ids.tolist(), captions):
            results.append(dict(image_id=image_id, caption=caption))

    coco_result = dataset.coco.loadRes(results)
    coco_eval = COCOEvalCap(dataset.coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    return coco_eval.eval

