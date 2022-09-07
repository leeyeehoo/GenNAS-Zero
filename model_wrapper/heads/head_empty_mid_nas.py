from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1)//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class TinyResBlock(nn.Module):
    def __init__(self, inplane, kernel_size):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(inplane,inplane,kernel_size,padding=(kernel_size - 1)//2),
                            nn.ReLU(),
                            nn.BatchNorm2d(inplane),\
                    )
    def forward(self, x):
        return x + self.block(x)

def block(index, inplane):
    if index == 0:
        return torch.nn.Identity()
    elif index == 1:
        return InvertedResidual(inplane, inplane,1, 3, 3)
    elif index == 2:
        return InvertedResidual(inplane, inplane,1, 3, 5)
    elif index == 3:
        return InvertedResidual(inplane, inplane,1, 3, 7)
    elif index == 4:
        return TinyResBlock(inplane, 3)
    elif index == 5:
        return TinyResBlock(inplane, 5)
    elif index == 6:
        return TinyResBlock(inplane, 7)
    elif index == 7:
        return nn.Sequential(
                                  nn.Conv2d(inplane,inplane,1,padding=0),
                                  nn.ReLU(),\
                                    nn.BatchNorm2d(inplane))
    elif index == 8:
        return nn.Sequential(
                                  nn.Conv2d(inplane,inplane,3,padding=1),
                                  nn.ReLU(),\
                                    nn.BatchNorm2d(inplane))
    else:
        raise NotImplementedError

class HeadEmptyMidNAS(nn.Module):
    def __init__(self, head_config):
        super().__init__()
        self.inplane = head_config.out_channel
        self.index = head_config.index

        self.model = nn.ModuleList()
        for index in self.index:
            self.model.append(block(index, self.inplane))
            
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x

        