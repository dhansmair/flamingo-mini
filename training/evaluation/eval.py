#!/usr/bin/env python
""" 

Overall evaluation script

"""
import logging
from typing import Optional
import json
import os

import torch
from flamingo_mini import FlamingoModel, FlamingoProcessor

from .eval_image_captioning import eval_image_captioning
from training import paths, utils


def save_to(filepath: str, result_dict: dict):
    """save result_dict containing metrics into a json file.
    if the file already exists, append the metrics.
    """
    
    try:
        # do not overwrite existing result file, just append the result
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            data = {}
                
        for k, v in result_dict.items():
            data[k] = v

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        logging.info('result file saved.')
        
    except Exception as e:
        logging.warning('exception while tried to save result file.')
        logging.warning(e)


@torch.no_grad()
def eval_all_metrics(
        cfg_id: str,
        run: int,
        device: Optional[torch.device],
        batch_size: int,
        num_workers: int,
        num_beams: int,
        cc3m_num_images: int = 2000,
        coco_num_images: int = 2000, 
        model: Optional[FlamingoModel] = None,
        processor: Optional[FlamingoProcessor] = None,
        metrics=('coco', 'cc3m'),
        verbose: bool = False,
        save_results: bool = True,
    ):

    if model is None:
        model = utils.load_model_from_checkpoint(cfg_id, run)
        model.to(device)
        
    model.eval()
    
    if processor is None:
        processor = FlamingoProcessor(model.config, load_vision_model=True, device=device)
        
    results = {}
    
    if not os.path.isdir(paths.evaluation_summary_dir):
        os.mkdir(paths.evaluation_summary_dir)

    summary_file_path = f"{paths.evaluation_summary_dir}/metrics_{cfg_id}_{run}.json"

    
    if 'coco' in metrics:
        logging.info('evaluating Image Captioning on COCO...')
        metrics_coco = eval_image_captioning(
            cfg_id,
            run,
            dataset='coco',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            num_beams=num_beams,
            num_images=coco_num_images,
            model=model,
            processor=processor,
            prefix='<image> a picture of',
            verbose=verbose
        )
        logging.info(metrics_coco)
        results['coco'] = metrics_coco
        
        if save_results:
            save_to(summary_file_path, {'coco':metrics_coco})

    if 'cc3m' in metrics:
        logging.info('evaluating Image Captioning on CC3M...')
        metrics_cc3m = eval_image_captioning(
            cfg_id,
            run,
            dataset='cc3m',
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            num_beams=num_beams,
            num_images=cc3m_num_images,
            model=model,
            processor=processor,
            prefix='<image>',
            verbose=verbose
        )
        logging.info(metrics_cc3m)
        results['cc3m'] = metrics_cc3m

        if save_results:
            save_to(summary_file_path, {'cc3m':metrics_cc3m})

    return results
    
