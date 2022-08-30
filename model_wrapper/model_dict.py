from .backbones.backbone_nb101 import BackboneNB101 as BackboneNB101

from .heads.head_nb101 import HeadNB101
from .heads.head_empty import HeadEmpty

from .generators.simple_generator import SimpleGenerator
from .generators.siam_generator import SiamGenerator
from .generators.siam_generator_ws import SiamGeneratorWS
from .generators.feature_generator import FeatureGenerator

BACKBONE_CONFIGS = {'BackboneNB101': BackboneNB101}

HEAD_CONFIGS = {'HeadNB101': HeadNB101, 'HeadEmpty': HeadEmpty}

GENERATOR_CONFIGS = {'SimpleGenerator': SimpleGenerator, 'SiamGenerator': SiamGenerator, 'SiamGeneratorWS': SiamGeneratorWS,
'FeatureGenerator': FeatureGenerator}