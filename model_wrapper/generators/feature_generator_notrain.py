import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math



class FeatureGeneratorNotrain(nn.Module):
    def __init__(self, head_config):
        super(FeatureGeneratorNotrain, self).__init__()
        self.loss_type = head_config.loss_type
        self.last_channel = head_config.last_channel
        self.feature_list = head_config.feature_list
        self.width = 8
        self.height = 8
        self.batch_size = 16
        self.features = nn.ParameterList()
        self.level = head_config.levels
        try:
            self.level = head_config.levels
        except:
            pass

        for i, lc in enumerate(self.last_channel):
            feat_list = self.feature_list[i]
            feature = torch.zeros(self.batch_size, head_config.last_channel[i], self.width, self.height)
            for feat in feat_list:
                sub_feature = torch.load(f'./data/features/{feat}.pth')
                feature += sub_feature[:self.batch_size, :head_config.last_channel[i], :self.width, :self.height]
            if type(self.level) == list:
                feature = feature * self.level[i]
            else:
                feature = feature * self.level
            self.features.append(nn.Parameter(feature, requires_grad= False))
            
    def forward(self,x):
        xs = []
        for f in self.features:
            xs.append(f)
        return xs

