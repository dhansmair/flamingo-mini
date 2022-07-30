"""
utils
"""
from PIL import Image
import requests
import torch
from torch import nn


def load_url(url: str):
    return Image.open(requests.get(url, stream=True).raw)


def load_image(path: str):
    return Image.open(path)


def unzip(l):
    return list(zip(*l))


class SquaredReLU(nn.Module):
    """ squared ReLU activation function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def FeedForward(dim, mult=4, act='gelu'):
    """
    lucidrains implementation, slightly modified with the act parameter.
    """
    
    acts = dict(
        gelu=nn.GELU,
        sqrelu=SquaredReLU,
        relu=nn.ReLU
    )
    
    assert act in acts, f"act. can only be one of {acts.keys()}"
    
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        acts[act](),
        nn.Linear(inner_dim, dim, bias=False)
    )
