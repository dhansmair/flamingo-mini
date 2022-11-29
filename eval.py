import argparse
import torch

from flamingo_mini import FlamingoModel
from training.evaluation import eval_all_metrics


ALL_METRICS = ('coco', 'cc3m')


def parse_args(input_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--cc3m-num-images', type=int, default=None)
    parser.add_argument('--coco-num-images', type=int, default=None)
    parser.add_argument('--metrics', default=ALL_METRICS, choices=ALL_METRICS, nargs='+')
    parser.add_argument('--verbose', action='store_true')

    if isinstance(input_str, str):
        input_str = input_str.split()
    return parser.parse_args(input_str)
    

if __name__ == '__main__':
    args = parse_args()
    
    model: FlamingoModel = FlamingoModel.from_pretrained('dhansmair/flamingo-mini')  # type: ignore
    model.to(torch.device(args.device))

    eval_all_metrics(
        args.id,
        args.run,
        torch.device(args.device),
        args.batch_size,
        args.num_workers,
        args.num_beams, 
        metrics=args.metrics,
        verbose=args.verbose,
        model=model
    )