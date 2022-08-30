import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmpty(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        
    def forward(self, x):
        return x

        