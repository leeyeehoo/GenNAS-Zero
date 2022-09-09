from .model_ncp.supernet import MultiResolutionNet
import torch
import torch.nn as nn
import torch.nn.functional as F

def resize(xs, width, height):
    ys = []
    for x in xs:
        ys.append(F.interpolate(x, size = (width, height), mode = 'bilinear'))
    return torch.cat(ys, dim = 1)

class BackboneNCP(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.arch = backbone_config.arch

        self.model = MultiResolutionNet(input_channel = self.arch['input_channel'], \
                               network_setting = self.arch['inverted_residual_setting'],\
                              last_channel = self.arch['last_channel'], task= 'test')


        self.out_channel = [self.arch['last_channel'],\
        sum(self.arch['inverted_residual_setting'][-2][-1]),\
        sum(self.arch['inverted_residual_setting'][-3][-1])]

        self.width = 8
        self.height = 8
        try: 
            self.width = backbone_config.width 
            self.height = backbone_config.height
        except:
            pass

    def forward(self, x):
        x = self.model.downsamples(x)
        x = [x]
        L = len(self.model.features)
        for i, feature in enumerate(self.model.features):
            x = feature(x)
            if L - i == 5:
                x1 = resize(x, self.width, self.height)
            if L - i == 3:
                x2 = resize(x, self.width, self.height)
            if L - i == 1:
                x3 = self.model._transform_inputs(x)
        return [x3, x2, x1]
