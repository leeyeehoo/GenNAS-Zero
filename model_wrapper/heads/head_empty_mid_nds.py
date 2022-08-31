import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmptyMidNDS(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.model = nn.Sequential(
                                  nn.Conv2d(head_config.in_channel,head_config.out_channel,3,padding=1),
                                  nn.ReLU(),\
                                    nn.BatchNorm2d(head_config.out_channel),)
    def forward(self, x):
        x = self.model(x)
        return x

        