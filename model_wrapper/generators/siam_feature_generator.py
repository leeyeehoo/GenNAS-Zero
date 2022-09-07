import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/visualization.ipynb
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

    

class SiamFeatureGenerator(nn.Module):
    def __init__(self, head_config):
        super(SiamFeatureGenerator, self).__init__()
        self.loss_type = head_config.loss_type
        assert self.loss_type == 'mseloss'
        self.last_channel = head_config.last_channel
        self.last_output = 1
        self.conv1 = nn.Conv2d(32,head_config.last_channel,1,padding = 0)
        self.bn1 = nn.BatchNorm2d(head_config.last_channel)
        self.conv2 = nn.Conv2d(head_config.last_channel,head_config.last_channel,1,padding = 0)
        self.features = nn.Parameter(positionalencoding2d(32 * head_config.batch_size, head_config.width, head_config.height).reshape(32,head_config.batch_size, head_config.width, head_config.height).transpose(0,1), requires_grad=False)
        self.relu = nn.ReLU()
        self.scale = head_config.batch_size
        self.another_branch = nn.Sequential(nn.BatchNorm2d(head_config.out_channel),\
                                  nn.Conv2d(head_config.out_channel,head_config.last_channel * self.last_output,3,padding=1)
                    )
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(self.features)))
        x = self.conv2(x)
        
        # if self.loss_type == 'celoss':
        #     x = rearrange(x, 'b (d c) h w -> b d c h w', d = self.last_output)
        return x
    def forward_another_branch(self,x):
        x = self.another_branch(x)
        return x

