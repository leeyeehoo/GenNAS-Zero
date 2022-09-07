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

class SiamGeneratorNAS(nn.Module):
    def __init__(self, head_config):
        super(SiamGeneratorNAS, self).__init__()
        self.loss_type = head_config.loss_type
        self.last_channel = head_config.last_channel
        self.last_output = 1
        self.combo = head_config.combo
        if 'net' in self.combo:
            self.conv1 = nn.Conv2d(3,64,3,padding = 1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64,128,3,padding = 1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128,256,3,padding = 1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256,512,3,padding = 1)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(512,1024,3,padding = 1)
            self.bn5 = nn.BatchNorm2d(1024)
            self.conv6 = nn.Conv2d(1024,self.last_channel,3,padding = 1)
            self.relu = nn.ReLU()
            self.maxpooling = nn.MaxPool2d(2,2)
        if 'sin' in self.combo:
            self.sin = nn.Parameter(positionalencoding2d(head_config.last_channel * head_config.batch_size, head_config.width, head_config.height).reshape(head_config.last_channel,head_config.batch_size, head_config.width, head_config.height).transpose(0,1), requires_grad=False)
        if 'dot' in self.combo:
            self.dot = nn.Parameter(torch.randint(-1, 1, [head_config.batch_size, head_config.last_channel, head_config.width, head_config.height]).float() * 2 + 1)

        

        self.another_branch = nn.Sequential(nn.BatchNorm2d(head_config.out_channel),\
                                  nn.Conv2d(head_config.out_channel,head_config.last_channel * self.last_output,3,padding=1)
                    )
        # self.random_noise = nn.Parameter(torch.rand(16, head_config.last_channel * self.last_output, 8,8),requires_grad = False)
    def forward(self,x):
        xs = []
        
        if 'net' in self.combo:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpooling(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.maxpooling(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.relu(self.bn5(self.conv5(x)))

            x = self.conv6(x) #+ self.random_noise
            xs.append(x/x.abs().max())
        if 'sin' in self.combo:
            x = self.sin
            xs.append(x)
        if 'dot' in self.combo:
            x = self.dot
            xs.append(x)
        
        
        output = torch.stack(xs).sum(0)
        output = output/output.abs().max()

        return output

    def forward_another_branch(self,x):
        x = self.another_branch(x)
        return x
