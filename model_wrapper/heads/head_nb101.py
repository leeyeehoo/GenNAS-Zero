import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadNB101(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.loss_type = head_config.loss_type
        if head_config.loss_type == 'celoss':
            self.last_output = 2
        else:
            self.last_output = 1
        self.model = nn.Sequential(\
            nn.BatchNorm2d(head_config.out_channel),\
                nn.Conv2d(head_config.out_channel,\
                    head_config.last_channel * self.last_output,1))
    def forward(self, x):
        x = self.model(x)
        if self.loss_type == 'celoss':
            x = rearrange(x, 'b (d c) h w -> b d c h w', d = self.last_output)
        return x