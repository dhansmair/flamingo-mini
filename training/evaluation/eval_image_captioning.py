#!/usr/bin/env python
"""
evaluate flamingo on MS COCO captions and Conceptual Captions
"""
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from einops import repeat
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from flamingo_mini import FlamingoModel, FlamingoProcessor

from .. import paths, utils
from ..datasets import DatasetBuilder


def compute_metrics(annotation_file: str, results_file: str) -> Dict[str, float]:
    """compute the common image captioning metrics (CIDEr, SPICE, BLEU) using COCOEvalCap.
    
    Provided files need to be in COCO format.
    TODO is there also a way to directly pass the result dictionary?

    Args:
        annotation_file (str): ground truth caption annotations
        results_file (str): your predicted captions
    """

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
        
    return coco_eval.eval


@torch.no_grad()
def generate_result_file(
    model: FlamingoModel, 
    processor: FlamingoProcessor, 
    val_loader: DataLoader, 
    device: Optional[torch.device], 
    num_beams=1, 
    max_length=150,
    num_images: Optional[int] = None,
    filename: str = 'dummy.json',
    prefix: str = '<image>',
    verbose: bool = False
):
    """generate captions on the validation set and store them in a file.

    Args:
        model (FlamingoModel): _description_
        processor (FlamingoProcessor): _description_
        val_loader (_type_): _description_
        device (torch.device): _description_
        num_beams (int, optional): _description_. Defaults to 1.
        max_length (int, optional): _description_. Defaults to 150.

    Returns:
        _type_: _description_
    """
    
    tick = datetime.now()
    logging.info('generating captions...')
    
    result = []
    
    # prepare text input
    input_ids, media_locations, attention_mask = processor.encode_text(prefix, device=device)
    input_ids = repeat(input_ids[0], 'l -> n l', n=val_loader.batch_size)
    media_locations = repeat(media_locations[0], 'l -> n l', n=val_loader.batch_size)
    attention_mask = repeat(attention_mask[0], 'l -> n l', n=val_loader.batch_size)
    
    for i, batch in tqdm(enumerate(val_loader), disable=not verbose):
        image_ids, pixels, _ = batch
        if isinstance(image_ids, torch.Tensor):
            image_ids = image_ids.tolist()
        batch_size = len(image_ids)

        # extract visual features
        with torch.no_grad():
            visual_features = processor.extract_features_from_preprocessed(pixels.to(device))
            del pixels

        # generate the captions
        out_ids = model.generate(
            inputs=input_ids[:batch_size],
            visual_features=visual_features,
            media_locations=media_locations[:batch_size],
            attention_mask=attention_mask[:batch_size],
            num_beams=num_beams,
            early_stopping=True,
            use_cache=True,
            bos_token_id=model.flamingo.lm.config.bos_token_id,
            eos_token_id=model.flamingo.lm.config.eos_token_id,
            pad_token_id=model.flamingo.lm.config.eos_token_id,
            max_length=max_length
        )
        
        del visual_features
        
        for image_id, tok in zip(image_ids, out_ids):

            try:
                caption = processor.tokenizer.decode(tok, skip_special_tokens=True)
                caption = processor.remove_tags(caption)
                
                result.append({
                    'image_id': image_id,
                    'caption': caption
                })

            except Exception as e:
                logging.warning("Exception during decoding of the generated caption:")
                logging.warning(e)
                logging.warning(f"{image_id=} \n {tok=}")

            if num_images is not None and len(result) >= num_images:
                break
            
        if num_images is not None and len(result) >= num_images:
            break
        
    with open(filename, 'w') as f:
        json.dump(result, f)
        
    logging.info(f"caption predictions stored. (took {utils.time_since(tick)})")

        
def eval_image_captioning(
    cfg_id: str,
    run: int,
    dataset: str,
    device: Optional[torch.device],
    batch_size: int,
    num_workers: int,
    num_beams: int,
    num_images: Optional[int] = None, 
    prefix: str = '<image>',
    model: Optional[FlamingoModel] = None,
    processor: Optional[FlamingoProcessor] = None,
    verbose: bool = False
) -> dict:
    """main function
    
    TODO support nocaps

    Args:
        cfg_id (str): _description_
        run (int): _description_
        dataset (str): _description_
        device (torch.device): _description_
        model (FlamingoModel, optional): can also use a model that is already loaded. Defaults to None.

    Returns:
        dict: _description_
    """
    assert dataset in ('coco', 'cc3m')
    
    if model is None:
        model = utils.load_model_from_checkpoint(cfg_id, run)

    model.to(device)
    model.eval()
        
    if processor is None:
        processor = FlamingoProcessor(model.config, load_vision_model=True, device=device)
        
    builder = DatasetBuilder(model.config)
    
    if dataset == 'coco':
        ds = builder.build_coco_val()
    elif dataset == 'cc3m':
        ds = builder.build_cc3m_val()
    else:
        raise ValueError('unsupported dataset ' + dataset)
        
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )
    
    filename = f"{paths.evaluation_results_dir}/{dataset}_captions_{cfg_id}_{run}.json"

    generate_result_file(model, processor, loader, device, filename=filename, num_beams=num_beams, num_images=num_images, prefix=prefix, verbose=verbose)
    
    if dataset == 'coco':
        metrics = compute_metrics(paths.coco_ann_val, filename)
        
    else:
        metrics = compute_metrics(paths.cc3m_ann_val, filename)
    
    return {
        **metrics,
        'num_images': num_images,
        'num_beams': num_beams
    }
