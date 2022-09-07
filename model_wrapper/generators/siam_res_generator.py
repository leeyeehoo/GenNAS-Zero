import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules.resnet import ResNet18
class SiamResGenerator(nn.Module):
    def __init__(self, head_config):
        super(SiamResGenerator, self).__init__()
        self.loss_type = head_config.loss_type
        if head_config.loss_type == 'celoss':
            self.last_output = 2
        else:
            self.last_output = 1
        self.last_channel = head_config.last_channel
        self.res = ResNet18()
        self.conv6 = nn.Conv2d(512,self.last_channel,3,padding = 1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2,2)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.ff = nn.Linear(1024, 10)
        self.another_branch = nn.Sequential(nn.BatchNorm2d(head_config.out_channel),\
                                  nn.Conv2d(head_config.out_channel,head_config.last_channel * self.last_output,3,padding=1)
                    )
        self.random_noise = nn.Parameter(torch.rand(16, head_config.last_channel * self.last_output, 8,8),requires_grad = False)
    def forward(self,x):
        x = self.res(x)
        x = self.conv6(x) #+ self.random_noise
        return x
    def forward_another_branch(self,x):
        x = self.another_branch(x)
        if self.loss_type == 'celoss':
            x = rearrange(x, 'b (d c) h w -> b d c h w', d = self.last_output)
        return x
