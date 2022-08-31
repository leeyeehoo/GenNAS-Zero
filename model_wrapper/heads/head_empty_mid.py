import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmptyMid(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(head_config.out_channel),\
                                  nn.Conv2d(head_config.out_channel,head_config.out_channel,3,padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(head_config.out_channel),\
                                  nn.Conv2d(head_config.out_channel,head_config.out_channel,3,padding=1),
                                  nn.ReLU()
                    )
    def forward(self, x):
        self.model(x)
        return x

        