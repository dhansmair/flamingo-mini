"""
utils
"""
from PIL import Image
import requests
import torch
from torch import nn, Tensor, cat


def exists(val):
    return val is not None


def load_url(url: str):
    return Image.open(requests.get(url, stream=True).raw)


def load_image(path: str):
    return Image.open(path)


def add_column(matrix: Tensor, vec: Tensor) -> Tensor:
    rows, cols = matrix.shape
    out = cat([matrix, vec.reshape((rows, 1))], dim=-1)
    return out
    

def unzip(l):
    return list(zip(*l))


def num_params(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class SquaredReLU(nn.Module):
    """ squared ReLU activation function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def FeedForward(dim, mult=4, act='gelu'):
    """
    lucidrains implementation.
    TODO check if the architecture matches the one described in the flamingo paper.
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
        # nn.GELU(),
        acts[act](),
        nn.Linear(inner_dim, dim, bias=False)
    )
