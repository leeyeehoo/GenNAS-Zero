from .model_ncp.supernet import MultiResolutionNet
import torch
import torch.nn as nn
import torch.nn.functional as F



class BackboneNCP(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.arch = backbone_config.arch

        self.model = MultiResolutionNet(input_channel = self.arch['input_channel'], \
                               network_setting = self.arch['inverted_residual_setting'],\
                              last_channel = self.arch['last_channel'])


        self.out_channel = self.arch['inverted_residual_setting'][-1][-1]


        self.width = 8
        self.height = 8
        try: 
            self.width = backbone_config.width 
            self.height = backbone_config.height
        except:
            pass

    def forward(self, x):
        x = self.model.downsamples(x)
        x = self.model.features([x])
        xs = []
        for sub in x:
            xs.append(F.interpolate(sub,(self.width, self.height)))
        return xs
