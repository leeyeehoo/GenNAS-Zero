from .model_trans.net_macro import MacroNet
import torch
import torch.nn as nn
import torch.nn.functional as F



class BackboneTrans(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.init_channels = backbone_config.stem_out_channels
        arch = backbone_config.arch.replace('41414','41412')
        self.model = MacroNet(arch, structure='backbone')

        self.out_channel = self.model.out_channel
        self.out_channel.reverse()
        self.out_channel = self.out_channel[:3]



        self.width = 8
        self.height = 8
        try: 
            self.width = backbone_config.width 
            self.height = backbone_config.height
        except:
            pass

    def forward(self, x):
        x = self.model.stem(x)
        xs = []
        for i, layer_name in enumerate(self.model.layers):
            res_layer = getattr(self.model, layer_name)
            x = res_layer(x)
            xs.append(F.interpolate(x, (self.width, self.height)))
        xs.reverse()
        return xs[:3]
