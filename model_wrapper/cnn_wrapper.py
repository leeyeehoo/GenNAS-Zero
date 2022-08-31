import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_dict import BACKBONE_CONFIGS, HEAD_CONFIGS

class CNNWrapper(nn.Module):
    def __init__(self, backbone_config, head_config):
        super().__init__()
        self.backbone = BACKBONE_CONFIGS[backbone_config.model](backbone_config)
        try: 
            head_config.in_channel = self.backbone.out_channel
        except:
            pass
        self.head = HEAD_CONFIGS[head_config.model](head_config)
        self.backbone_config = backbone_config
        self.head_config = head_config
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

        
