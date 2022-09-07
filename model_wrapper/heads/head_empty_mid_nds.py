import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmptyMidNDS(nn.Module):
    def __init__(self, head_config):
        super().__init__()

        self.model = nn.ModuleList()
        self.model_len = len(head_config.out_channel)
        for i in range(self.model_len): #in_ch, o_ch in zip(head_config.in_channel, head_config.out_channel):
            self.model.append(nn.Sequential(
                            nn.Conv2d(head_config.in_channel[i],head_config.out_channel[i],3,padding=1),
                            nn.ReLU(),\
                            nn.BatchNorm2d(head_config.out_channel[i]),
                            ))

        self.model_len = len(self.model)

        
    def forward(self, x):
        xs = []
        for i in range(self.model_len):
            xs.append(self.model[i](x[i]))
        return xs


        