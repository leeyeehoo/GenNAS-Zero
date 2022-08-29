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
        
    def forward(self, x):
        for _, layer in enumerate(self.model.layers):
            x = layer(x)
        return x
