from .backbones.backbone_nb101 import BackboneNB101 as BackboneNB101
from .backbones.backbone_nds import BackboneNDS as BackboneNDS

from .heads.head_nb101 import HeadNB101
from .heads.head_nb101_2l import HeadNB1012L
from .heads.head_empty import HeadEmpty
from .heads.head_empty_mid import HeadEmptyMid
from .heads.head_empty_mid_nas import HeadEmptyMidNAS
from .heads.head_empty_mid_nds import HeadEmptyMidNDS

from .generators.simple_generator import SimpleGenerator
from .generators.siam_generator import SiamGenerator
from .generators.siam_generator_fc import SiamGeneratorFC
from .generators.siam_generator_ws import SiamGeneratorWS
from .generators.siam_generator_ws_arch2 import SiamGeneratorWSArch2
from .generators.feature_generator import FeatureGenerator
from .generators.feature_generator_notrain import FeatureGeneratorNotrain

BACKBONE_CONFIGS = {'BackboneNB101': BackboneNB101,
'BackboneNDS': BackboneNDS}

HEAD_CONFIGS = {'HeadNB101': HeadNB101,
 'HeadNB1012L': HeadNB1012L,
'HeadEmpty': HeadEmpty,
'HeadEmptyMid': HeadEmptyMid,
'HeadEmptyMidNAS': HeadEmptyMidNAS,
'HeadEmptyMidNDS': HeadEmptyMidNDS
}

GENERATOR_CONFIGS = {'SimpleGenerator': SimpleGenerator, 
'SiamGenerator': SiamGenerator, 
'SiamGeneratorWS': SiamGeneratorWS,
'FeatureGenerator': FeatureGenerator, 
'FeatureGeneratorNotrain': FeatureGeneratorNotrain, 
'SiamGeneratorWSArch2': SiamGeneratorWSArch2,
'SiamGeneratorFC': SiamGeneratorFC}