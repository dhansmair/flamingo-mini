#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers.optimization import get_constant_schedule_with_warmup

import wandb

from training.evaluation import eval_all_metrics
from training import paths, utils
from training.datasets import DatasetBuilder


ALL_DATASETS = ('coco', 'cc3m')
ALL_METRICS = ('coco', 'cc3m')


def parse_args(input_str=None):
    if isinstance(input_str, str):
        input_str = input_str.split()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='flamingo-tiny-vitL')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', type=int, default=None,
                       help='pass a run id for which you want to resume training')
    group.add_argument('--finetune', type=int, default=None,
                       help='pass a run id from which you want to load the model for a new training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=3_000_000,
                        help='maximum number of training samples to process')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--evaluation-batch-size', type=int, default=32)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--gradient-update-freq', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--log-freq', type=int, default=5000,
                        help='frequency of logging to wandb and stdout')
    parser.add_argument('--val-freq', type=int, default=100_000,
                        help='frequency of validation or evaluation during training')
    parser.add_argument('--save-freq', type=int,
                        default=100_000, help='frequency of checkpointing')
    parser.add_argument('--entity', type=str,
                        default='ma_flamingo', help='entity wandb project')
    parser.add_argument('--project', type=str,
                        default='image captioning', help='name of wandb project')
    parser.add_argument('--seq-length', type=int, default=64,
                        help='maximum token sequence length')
    parser.add_argument('--offline', action='store_true',
                        help='whether to connect to wandb or not.')
    parser.add_argument('--evaluate', default=['coco', 'cc3m'], choices=ALL_METRICS, nargs='+')
    parser.add_argument('--datasets', type=str,
                        default=['coco', 'cc3m'], choices=ALL_DATASETS, nargs='+')

    args = parser.parse_args(input_str)
    if args.resume is not None:
        args_path = f"{paths.checkpoint_dir}/{args.id}/{args.resume}/args.json"

        assert os.path.isfile(
            args_path), f"passed --resume {args.resume}, but the run directory {args_path} does not exist."

        with open(args_path, 'r') as f:
            json_params = json.load(f)
            parser.set_defaults(**json_params)

    return parser.parse_args(input_str)


def build_model(args) -> FlamingoModel:
    with open(f"{paths.config_dir}/{args.id}.json", 'r') as f:
        config_dict = json.load(f)

    config = FlamingoConfig(**config_dict)
    model = FlamingoModel(config)
    model.freeze_lm()
    model.train()
    return model


def build_logger(args):
    logging.basicConfig(
        format=f'%(asctime)s {args.id} run={args.run_id} %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(utils.get_logfile_path(args.id, args.run_id))
        ]
    )


def build_train_loader(args, datasets: List[Dataset]) -> DataLoader:

    if len(datasets) == 1:
        return DataLoader(
            datasets[0],
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            timeout=120
        )

    else:
        batch_sampler = utils.RoundRobinBatchSampler(
            [len(ds) for ds in datasets],
            batch_size=args.batch_size,
            drop_last=True
        )

        return DataLoader(
            ConcatDataset(datasets),
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            timeout=120
        )


def train(args):
    """set up the training

    Initialize model, processor, optimizer

    Args:
        args (_type_): _description_
    """
    cfg_id = args.id
    device = torch.device(args.device)

    if args.resume is None:
        # first, create a new run id
        run_id = utils.init_new_run(args)
        args.run_id = run_id
    else:
        run_id = args.resume
        args.run_id = args.resume

    build_logger(args)

    logging.info('loading model...')
    # build model
    model = build_model(args)
    model.to(device)
    processor = FlamingoProcessor(model.config, device, load_vision_model=True)
    logging.info('model loaded.')
    
    logging.info('loading training datasets...')

    builder = DatasetBuilder(model.config)
    train_sets = builder.build_train_sets(args.datasets)
    train_loader = build_train_loader(args, train_sets)
    
    logging.info('datasets loaded.')

    optimizer = AdamW(model.parameters_trainable(), args.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    start_epoch = 0
    step = 0

    if args.resume is not None:
        data = utils.load_checkpoint(cfg_id, run_id)
        step = data['step']
        start_epoch = data['epoch']
        model.flamingo.load_state_dict(data['model'], strict=False)
        optimizer.load_state_dict(data['optimizer'])

        if 'scheduler' in data:
            scheduler.load_state_dict(data['scheduler'])

        if not args.offline:
            wandb.init(project=args.project, entity=args.entity, resume='must',
                       config=vars(args), id=data['wandb_id'],
                       settings=wandb.Settings(start_method='thread'))
        logging.info(f'checkpoint {cfg_id=} {run_id=} restored.')

    elif args.finetune is not None:
        data = utils.load_checkpoint(cfg_id, args.finetune)
        model.flamingo.load_state_dict(data['model'], strict=False)
        if not args.offline:
            wandb.init(project=args.project, entity=args.entity, resume='never',
                       config=vars(args),
                       settings=wandb.Settings(start_method='thread'))
        logging.info(f'starting new finetune training for {cfg_id=} {run_id=}')

    else:
        if not args.offline:
            wandb.init(project=args.project, entity='ma_flamingo', resume='never',
                       config=vars(args),
                       settings=wandb.Settings(start_method='thread'))
        logging.info(f'starting new training for {cfg_id=} {run_id=}')

    if not args.offline:
        wandb.run.name = f"{cfg_id}_{run_id}"
        wandb.watch(model)

    for epoch in range(start_epoch, args.epochs):
        if step >= args.steps:
            break
        logging.info(f">>> epoch {epoch}")
        step = train_epoch(model, processor, optimizer, scheduler,
                           train_loader, args, epoch=epoch, step=step, device=device)

    logging.info('training finished.')


def train_epoch(
    model: FlamingoModel,
    processor: FlamingoProcessor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loader: DataLoader,
    args,
    epoch: int = 0,
    step: int = 0,
    device: Optional[torch.device] = None,
):
    """train a flamingo model for one epoch on the train_loader.

    a batch should be a tuple of pixels and captions.
    pixels are a tensor of shape [batch_size, height, width, depth]
    captions are a list of strings of length batch_size.

    Args:
        model (FlamingoModel): _description_
        processor (FlamingoProcessor): _description_
        optimizer (torch.optim.Optimizer): _description_
        train_loader (DataLoader): _description_
        args (_type_): _description_
        epoch (int, optional): _description_. Defaults to 0.
        step (int, optional): _description_. Defaults to 0.
        device (torch.device, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    model.train()
    total_loss = utils.RunningLoss()
    step_start = step
    tick = datetime.now()

    parameters_trainable = list(model.parameters_trainable())

    for i, batch in enumerate(train_loader):
        loss = process_batch(model, processor, batch,
                             device, max_length=args.seq_length)
        loss = loss / args.gradient_update_freq
        loss.backward()

        if i % args.gradient_update_freq == 0:

            if args.gradient_clip is not None and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters_trainable, args.gradient_clip)

            optimizer.step()
            model.zero_grad(set_to_none=True)
            scheduler.step()

        step += args.batch_size
        total_loss.add(loss.item() * args.batch_size, count=args.batch_size)

        del loss

        # logging
        if step % args.log_freq < args.batch_size:
            # compute the est. steps per hour
            time_delta = (datetime.now() - tick).seconds / \
                3600             # in hours
            step_delta = step - step_start
            step_per_hour = int(
                step_delta / time_delta) if time_delta > 0 else 0

            if not args.offline:
                wandb.log({"loss_train": total_loss.get(),
                          "sph": step_per_hour}, step=step)
            logging.info(
                f'loss_train={total_loss.get():.2f} {epoch=} batch={i} {step=} (steps/h={step_per_hour})')
            total_loss.reset()
            tick = datetime.now()
            step_start = step

        # checkpointing
        if step % args.save_freq < args.batch_size:
            # store a checkpoint of the model
            utils.save_checkpoint(args.id, args.run_id, {
                'model': model.state_dict_trainable(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'batch': i,
                'step': step,
                'args': args,
                'wandb_id': None if args.offline else wandb.run.id
            })

            logging.info(f'checkpoint saved. {epoch=} batch={i}')

        # evaluation
        if len(args.evaluate) > 0 and step % args.val_freq < args.batch_size:
            metrics = eval_all_metrics(
                cfg_id=args.id,
                run=args.run_id,
                device=device,
                batch_size=args.evaluation_batch_size,
                num_workers=args.num_workers,
                num_beams=1,
                model=model,
                processor=processor,
                metrics=args.evaluate,
                verbose=False
            )
            model.train()
            if not args.offline:
                wandb.log(metrics, step=step)

            # since evaluation takes some time, reset the counter to get more accurate speed metrics
            tick = datetime.now()
            step_start = step

        if step >= args.steps:
            break

    return step


def process_batch(
    model: FlamingoModel,
    processor: FlamingoProcessor,
    batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
    device: Optional[torch.device],
    max_length: Optional[int] = None
) -> torch.Tensor:
    """process one batch of pixel-caption pairs. returns the loss.

    Args:
        model (FlamingoModel): _description_
        processor (FlamingoProcessor): _description_
        batch (Tuple): batch of training samples:
            image_ids (Tensor shape [b]),
            pixels (Tensor shape [b N c h w]),
            captions (List[str] of length b)
        device (Optional[torch.device]):
        max_length (Optional[int]): maximum token input sequence length

    Returns:
        torch.Tensor: training loss
    """

    _, pixels, sentences = batch
    pixels = pixels.to(device, non_blocking=True)
    input_ids, media_locations, attention_mask = processor.encode_text(
        sentences, device=device, max_length=max_length)

    # extract visual features
    with torch.no_grad():
        visual_features = processor.extract_features_from_preprocessed(pixels)
        del pixels

    # forward pass
    out = model(
        input_ids=input_ids,
        visual_features=visual_features,
        media_locations=media_locations,
        attention_mask=attention_mask,
        labels=input_ids
    )
    del input_ids, media_locations, attention_mask, visual_features
    loss = out.loss

    if not math.isfinite(loss.item()):
        logging.critical("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)

    return loss


if __name__ == '__main__':
    args = parse_args()

    print(args)
    train(args)
