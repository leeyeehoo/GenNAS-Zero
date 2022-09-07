from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBarrier(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.model_len = len(head_config.last_channel)
        self.model = nn.ModuleList()
        for i in range(self.model_len):
            self.model.append(nn.Conv2d(head_config.out_channel[i],head_config.last_channel[i],3,padding=1))
    def forward(self, x):
        xs = []
        for i in range(self.model_len):
            xs.append(self.model[i](x[i]))
        return xs
