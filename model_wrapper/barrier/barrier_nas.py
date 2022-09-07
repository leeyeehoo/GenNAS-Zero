from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class BarrierNAS(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.model_len = len(head_config.last_channel)
        self.model = nn.ModuleList()
        self.index = head_config.barrier_index
        self.kernel_size_list = []
        # print(f'barrier_index {self.index}')
        # print(f'last_channel: {head_config.last_channel}')
        for i in range(self.model_len):
            if self.index[i] == 0:
                kernel_size = 1
            elif self.index[i] == 1:
                kernel_size = 3
            elif self.index[i] == 2:
                kernel_size = 7
            self.kernel_size_list.append(kernel_size)
        for i in range(self.model_len):
            self.model.append(nn.Conv2d(head_config.out_channel[i],head_config.last_channel[i],self.kernel_size_list[i],padding=(self.kernel_size_list[i] - 1)//2))
    def forward(self, x):
        xs = []
        for i in range(self.model_len):
            xs.append(self.model[i](x[i]))
        return xs
