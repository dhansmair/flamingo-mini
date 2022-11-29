import json
import os

import torch
from flamingo_mini import FlamingoConfig, FlamingoModel

from .. import paths
    
    
def get_checkpoint_path(cfg_id:str, run:int) -> str:
    return f"{paths.checkpoint_dir}/{cfg_id}/{run}/checkpoint.pth"


def get_logfile_path(cfg_id: str, run: int) -> str:
    return f"{paths.checkpoint_dir}/{cfg_id}/{run}/log"


def get_args_path(cfg_id:str, run:int) -> str:
    return f"{paths.checkpoint_dir}/{cfg_id}/{run}/args.json"


def save_checkpoint(cfg_id:str, run:int, data:dict):
    path = get_checkpoint_path(cfg_id, run)
    
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    
    with open(path, 'wb') as f:
        torch.save(data, f)
        

def load_model_cfg(cfg_id:str) -> FlamingoConfig:
    with open(f'{paths.config_dir}/{cfg_id}.json', 'r') as f:
        return FlamingoConfig(**json.load(f))

        
def load_checkpoint(cfg_id:str, run:int) -> dict:
    path = get_checkpoint_path(cfg_id, run)
    with open(path, 'rb') as f:
        return torch.load(f, map_location='cpu')
    
    
def load_model_from_checkpoint(cfg_id: str, run:int) -> FlamingoModel:
    config = load_model_cfg(cfg_id)
    data = load_checkpoint(cfg_id, run)
    model = FlamingoModel(config)
    model.flamingo.load_state_dict(data['model'], strict=False)
    return model
    
    
def init_new_run(args) -> int:
    """determine next run_id, set up the folder structure and save args

    Args:
        args (_type_): arguments as given by the ArgumentParser

    Returns:
        int: next run_id
    """
    cfg_id = args.id
    path =  f'{paths.checkpoint_dir}/{cfg_id}'
    if not os.path.isdir(path):
        os.mkdir(path)
    run_id = find_next_run(cfg_id)
    path = f'{path}/{run_id}'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    # store args into a file
    with open(get_args_path(cfg_id, run_id), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return run_id
    
    
def find_next_run(cfg_id:str) -> int:
    path = f'{paths.checkpoint_dir}/{cfg_id}'

    if not os.path.isdir(path):
        return 0

    prev_runs = os.listdir(path)
    prev_runs = [s for s in prev_runs if s.isnumeric()]
    
    if len(prev_runs) == 0:
        return 0 

    return max([int(s) for s in prev_runs]) + 1
        