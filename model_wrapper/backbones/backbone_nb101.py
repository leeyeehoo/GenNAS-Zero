from .model_nb101.model_nb101 import Network
from .model_nb101 import model_spec
import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNB101(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        matrix, ops = backbone_config.arch
        spec = model_spec._ToModelSpec(matrix, ops)
        self.model = Network(spec,\
             backbone_config.stem_out_channels, \
                backbone_config.num_stacks,\
                     backbone_config.num_modules_per_stack,\
                         backbone_config.num_labels)
        self.width = 8
        self.height = 8
        try: 
            self.width = backbone_config.width 
            self.height = backbone_config.height
        except:
            pass
        self.out_channel = []
        for stack_num in range(backbone_config.num_stacks):
            self.out_channel.append(backbone_config.stem_out_channels * 2 **stack_num)
        self.out_channel.reverse()
    def forward(self, x):
        count = 0
        for _, layer in enumerate(self.model.layers):
            if isinstance(layer,nn.MaxPool2d) and count == 0:
                x0 = F.interpolate(x,(self.width,self.height))
                count += 1
            elif isinstance(layer,nn.MaxPool2d) and count == 1:
                x1 = F.interpolate(x,(self.width,self.height))
                count += 1
            x = layer(x)
        xs = [x, x1, x0]
        return xs
