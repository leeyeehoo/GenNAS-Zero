import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HeadEmptyNAS(nn.Module):
    def __init__(self, head_config):
        super().__init__()

        self.model = nn.ModuleList()
        self.model_len = len(head_config.out_channel)
        self.index = head_config.index
        self.kernel_size_list = []
        # print(f'head_index {self.index}')
        # print(f'out_channel {head_config.out_channel}')
        for i in range(self.model_len):
            if self.index[i] == 0:
                kernel_size = 1
            elif self.index[i] == 1:
                kernel_size = 3
            elif self.index[i] == 2:
                kernel_size = 7
            self.kernel_size_list.append(kernel_size)
        for i in range(self.model_len): #in_ch, o_ch in zip(head_config.in_channel, head_config.out_channel):
            self.model.append(nn.Sequential(
                            nn.Conv2d(head_config.in_channel[i],head_config.out_channel[i],self.kernel_size_list[i],padding=(self.kernel_size_list[i] - 1)//2),
                            nn.ReLU(),
                            nn.BatchNorm2d(head_config.out_channel[i]),
                            ))

        
    def forward(self, x):
        xs = []
        for i in range(self.model_len):
            xs.append(self.model[i](x[i]))
        return xs

        