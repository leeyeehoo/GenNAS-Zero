import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmptyMid(nn.Module):
    def __init__(self, head_config):
        super().__init__()

        self.model = nn.ModuleList()
        self.model_len = len(head_config.out_channel)
        for i in range(self.model_len): #in_ch, o_ch in zip(head_config.in_channel, head_config.out_channel):
            self.model.append(nn.Sequential(
                            nn.Conv2d(head_config.in_channel[i],head_config.out_channel[i],1,padding=0),
                            nn.ReLU(),
                            nn.BatchNorm2d(head_config.out_channel[i]),
                            ))

        # self.model_len = len(self.model)
        # head_config.out_channel = head_config.in_channel
        
    def forward(self, x):
        xs = []
        for i in range(self.model_len):
            # xs.append(x[i])
            xs.append(self.model[i](x[i]))
        return xs

        