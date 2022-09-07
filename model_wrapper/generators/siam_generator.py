import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SiamGenerator(nn.Module):
    def __init__(self, head_config):
        super(SiamGenerator, self).__init__()
        self.loss_type = head_config.loss_type
        if head_config.loss_type == 'celoss':
            self.last_output = 2
        else:
            self.last_output = 1
        self.last_channel = head_config.last_channel
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
        # self.conv6 = nn.Conv2d(1024,self.last_channel,3,padding = 1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2,2)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        # self.ff = nn.Linear(1024, 10)


        # self.another_branch = nn.Sequential(
        #                           nn.Conv2d(head_config.out_channel,head_config.last_channel * self.last_output,3,padding=1)
        #             )

        self.conv6 = nn.ModuleList()
        self.another_branch = nn.ModuleList()

        self.model_len = len(head_config.last_channel)

        for i in range(self.model_len):
            self.conv6.append(nn.Conv2d(1024,self.last_channel[i],3,padding = 1))
            self.another_branch.append(nn.Conv2d(head_config.out_channel[i],head_config.last_channel[i],3,padding=1))

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpooling(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        xs = []
        for i in range(self.model_len):
            xs.append( self.conv6[i](x)) 
        return xs

    def forward_another_branch(self,x):
        xs = []
        for i in range(self.model_len):
            xs.append(self.another_branch[i](x[i]))
        return xs
