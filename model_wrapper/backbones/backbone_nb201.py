from .model_nb201 import nasbench2
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_arch(structure):
    NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    strings = []
    for i in range(3):
        
        string = '|'.join([NAS_BENCH_201[structure[i][k]]+'~{:}'.format(k) for k in range(i+1)])
        string = '|{:}|'.format(string)
        strings.append( string )
    return '+'.join(strings)

class BackboneNB201(nn.Module):
    def __init__(self, backbone_config):
        self.init_channels = backbone_config.stem_out_channels
        super().__init__()
        arch = backbone_config.arch
        if isinstance(arch,list):
            arch = generate_arch(arch)
        self.model = nasbench2.get_model_from_arch_str(arch, 10, init_channels = self.init_channels)
        
        self.out_channel = []

        for stack_num in range(backbone_config.num_stacks):
            self.out_channel.append(backbone_config.stem_out_channels * 2 **stack_num)
        self.out_channel.reverse()
        self.width = 8
        self.height = 8
        try: 
            self.width = backbone_config.width 
            self.height = backbone_config.height
        except:
            pass

    def forward(self, x):
        x = self.model.stem(x)        
        x = self.model.stack_cell1(x)
        x0 = (F.interpolate((x),(self.width,self.height)))
        x = self.model.reduction1(x)
        x = self.model.stack_cell2(x)
        x1 = (F.interpolate((x),(self.width,self.height)))
        x = self.model.reduction2(x)
        x = self.model.stack_cell3(x)
        x2 = (x)
        return [x2,x1,x0]
