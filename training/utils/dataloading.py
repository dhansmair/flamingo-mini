from typing import List 

import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
        

class RoundRobinBatchSampler:
    
    def __init__(self, sizes: List[int], batch_size: int, drop_last:bool):
        self.sizes = torch.tensor(sizes)
        self.batch_size = batch_size            
        self.drop_last = drop_last
        
        starts = self.sizes.cumsum(0) - self.sizes
        self._samplers = [BatchSampler(
            SubsetRandomSampler(range(start, start+size)), 
            batch_size=batch_size,
            drop_last=drop_last
        ) for start, size in zip(starts, self.sizes)]
        
    def __len__(self):
        return sum(self.sizes)
        
    def __iter__(self):
        iterators = [iter(s) for s in self._samplers]
        
        while True:
            try:
                for it in iterators:
                    yield next(it)
            except StopIteration:
                break