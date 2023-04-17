"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import random

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from torchvision import transforms as T
from torchvision.datasets import CocoCaptions

import transformers
from transformers import HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup

from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

from eval import evaluate_image_captioning  # don't ask me why this import works


logger = logging.getLogger(__name__)


# get images and annotations from https://cocodataset.org/#download
COCO_ROOT      = '/kaggle/input'
COCO_ANN_TRAIN = '/kaggle/working/caption_2m_train_v1.json'
COCO_ANN_VAL   = '/kaggle/working/caption_2m_val_v1.json'


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_type) # type: ignore

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values']

        
def prepare_training_dataset(config: FlamingoConfig):
    """ prepare a CocoCaptions training dataset """
    transform = T.Compose([
        T.RandomHorizontalFlip(),                       # add your favorite transforms
        CLIPImageTransform(config.clip_model_type)
    ])

    def target_transform(captions):
        # seems that the following return will only return the only caption if there is only one caption
        # print('training caption: ', captions)
        # print('returning: ', f"{random.choice(['', ' '])}<image>{random.choice(captions)}<EOC></s>")
        return f"{random.choice(['', ' '])}<image>{random.choice(captions)}<EOC></s>"
    
        # training caption:  ['A portrait photo of a kangaroo wearing an orange hoodie and blue \
        # sunglasses standing on the grass in front of the Sydney Opera House holding a sign on \
        # the chest that says Welcome Friends, subject: kangaroo, subject detail: wearing orange \
        # hoodie, wearing blue sunglasses, subject location: sydney opera house, subject action: \
        # holding sign.']

        # returning:   <image>A portrait photo of a kangaroo wearing an orange hoodie and blue \
        # sunglasses standing on the grass in front of the Sydney Opera House holding a sign on \
        # the chest that says Welcome Friends, subject: kangaroo, subject detail: wearing orange \
        # hoodie, wearing blue sunglasses, subject location: sydney opera house, subject action: \
        # holding sign.<EOC></s>

    return CocoCaptions(
        COCO_ROOT, 
        COCO_ANN_TRAIN, 
        transform=transform,
        target_transform=target_transform
    )
    

def prepare_evaluation_dataset(config: FlamingoConfig):
    return CocoCaptions(COCO_ROOT, COCO_ANN_VAL, 
        transform=CLIPImageTransform(config.clip_model_type))


class DataCollator:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        pixel_values, sentences = zip(*batch)
        inputs = self.processor(text=sentences)
        pixel_values = torch.stack(pixel_values)
        
        return dict(
            pixel_values=pixel_values,
            labels=inputs['input_ids'],
            **inputs
        )


@dataclass
class FlamingoTrainingArguments(TrainingArguments):
    """ custom arguments """
    eval_coco_captioning_prefix: str = field(default="<image>")
    # eval_coco_captioning_prefix: str = field(default="<image>A picture of")         # It's a common thing to do for COCO image captioning
    eval_coco_captioning_start: int = field(default=0)
    eval_coco_captioning_end: int = field(default=1000)
    

class FlamingoTrainer(Trainer):

    args: FlamingoTrainingArguments
    model: FlamingoModel
    processor: FlamingoProcessor
    eval_dataset: CocoCaptions
    
    def evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """ override evaluation method to inject custom behavior. 
        TODO this only runs on one GPU, how to do distributed evaluation?
        """
        metrics = evaluate_image_captioning(self.eval_dataset, self.model, 
            prefix=self.args.eval_coco_captioning_prefix,
            start=self.args.eval_coco_captioning_start,
            end=self.args.eval_coco_captioning_end,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers
        )
        metrics = {f"{metric_key_prefix}_{k}" : v for k, v in metrics.items()}

        # HF trainer stuff from overridden method
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(FlamingoTrainingArguments)
    training_args: FlamingoTrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s', 
        datefmt='%H:%M:%S',
        # force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
    config = FlamingoConfig(
        clip_model_type='openai/clip-vit-large-patch14',
        lm='facebook/opt-125m',
        dim=768,
        dim_visual=1024,
        xattn_act='sqrelu',
        resampler_act='sqrelu'
    )
    model = FlamingoModel(config)
    model.train()

    #################################################################
    # datasets
    #################################################################
    logger.info('loading datasets...')
    train_dataset = prepare_training_dataset(config)
    eval_dataset = prepare_evaluation_dataset(config)
    
    #################################################################
    # optimizer, scheduler, trainer
    #################################################################
    # optimizer = AdamW(model.parameters_trainable(), training_args.learning_rate)
    # scheduler = get_constant_schedule_with_warmup(optimizer, training_args.warmup_steps)

    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(config),
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    if training_args.resume_from_checkpoint is not None:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()
