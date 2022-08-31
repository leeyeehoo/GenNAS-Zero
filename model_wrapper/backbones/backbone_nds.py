
import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNDS(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.NDS = backbone_config.NDS
        self.search_space = backbone_config.search_space
        self.id = backbone_config.arch
        model = self.NDS.get_network(self.id)
        model_config = self.NDS.get_network_config(self.id)
        if self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in']:
            concat_len = len(model_config['genotype']['reduce_concat'])
            out_channel = model.classifier.classifier.in_features
        elif self.search_space in ['ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
            width = model_config['stem_w']
            ws = model_config['ws']
            depth = model_config['ds']
            out_channel = ws[3]
        self.out_channel = out_channel
        self.model = model
        
    def forward(self, x):
        if self.search_space in ['DARTS','DARTS_in','DARTS_fix-w-d','DARTS_fix-w-d_in',\
                                     'ENAS','ENAS_in','ENAS_fix-w-d',\
                                     'PNAS','PNAS_in','PNAS_fix-w-d',\
                                     'Amoeba','Amoeba_in',\
                                     'NASNet','NASNet_in']:
            s0 = s1 = self.model.stem(x)
            for i, cell in enumerate(self.model.cells):
                s0, s1 = s1, cell(s0, s1, self.model.drop_path_prob)
            x = s1
        elif self.search_space in ['ResNet','ResNeXt-A','ResNeXt-A_in','ResNeXt-B','ResNeXt-B_in']:
            x = self.model.stem(x)
            x1 = self.model.s1(x)
            x2 = self.model.s2((x1))
            x3 = self.model.s3((x2))
            x4 = self.model.s4((x3))
            x = x4
        return x
