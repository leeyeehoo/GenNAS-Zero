from .backbones.backbone_nb101 import BackboneNB101 as BackboneNB101

from .heads.head_nb101 import HeadNB101

BACKBONE_CONFIGS = {'BackboneNB101': BackboneNB101}

HEAD_CONFIGS = {'HeadNB101': HeadNB101}