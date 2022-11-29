from datetime import datetime
import math
import sys


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def time_since(start_time) -> str:
    secs = (datetime.now()-start_time).total_seconds()

    if secs > 60:
        mins = math.floor(secs / 60)
        secs = secs % 60
        
        if mins > 59:
            hours = math.floor(mins / 60)
            mins = mins % 60
            return f"{hours}h {mins}m {secs:.2f}s"
        
        else:
            return f"{mins}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"


class RunningLoss:
    """ simple class to keep track of the mean of a training loss. """
    def __init__(self):
        self.loss_sum: float = 0.0
        self.count: int = 0
        
    def reset(self):
        self.loss_sum = 0.0
        self.count = 0
        
    def add(self, loss: float, count: int = 1):
        self.loss_sum += loss
        self.count += count 
        
    def get(self) -> float:
        return float(self.loss_sum / self.count)
    

def round_dict(el, decimals=3):
    """
    given a container of results (dict, list, ...)
    return a version where all floats are rounded to `decimals`.
    Careful, this will do in-place modification
    """

    if isinstance(el, float):
        return round(el, decimals)
        
    elif isinstance(el, dict):
        for k in el.keys():
            el[k] = round_dict(el[k], decimals)
        return el
        
    elif isinstance(el, list):
        return [round_dict(el2) for el2 in el]
    
    else:
        return el

            