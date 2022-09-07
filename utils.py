
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

def load_one_batch_image(dataset_config = None):
  batch_size = dataset_config.batch_size
  # transform = transforms.Compose(
  #   [transforms.ToTensor(),
  #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  transform = transforms.Compose(
  [transforms.RandomCrop(32, padding=4),
  transforms.Resize(32),
  transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4913997551666284, 0.48215855929893703, 0.4465309133731618], [0.24703225141799082, 0.24348516474564, 0.26158783926049628])])
  trainset = torchvision.datasets.CIFAR10(root='./data/dataset', train=True,
                                      download=True, transform=transform)

  test_queue = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
  
  test_queue = iter(test_queue)
  data,_ = test_queue.next()
  return data

# https://github.com/quark0/darts/tree/master/cnn
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

# https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object
class yaml_parser(object):
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [yaml_parser(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, yaml_parser(v) if isinstance(v, dict) else v)

def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


def init_net(net, w_type):
    if w_type == 'xavier_uniform':
        net.apply(init_weights_xavier_uniform)
    elif w_type == 'xavier_normal':
        net.apply(init_weights_xavier_normal)
    elif w_type == 'kaiming_uniform':
        net.apply(init_weights_kaiming_uniform)
    elif w_type == 'kaiming_normal':
        net.apply(init_weights_kaiming_normal)
    
    else:
        raise NotImplementedError(f'init_type={w_type} is not supported.')


def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def init_weights_xavier_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def init_weights_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def init_weights_kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_macs_lut(search_space):
    lut = {'nb101': 7218069504.0,\
        'nb101imgnet': 7218069504,\
        'nb201': 187166720,\
        'nb201c100': 187166720,\
        'nb201tinyimg': 187166720,\
        'amoeba': 1853440000,\
        'amoebain': 877355008,\
        'darts': 515047424,\
        'dartsfixwd': 178651136,\
        'dartsfixwdin': 584392704,\
        'dartsin': 912556032,\
        'enas': 600064000,\
        'enasfixwd': 233881600,\
        'enasin': 842526720,\
        'nasnet': 1458520064,\
        'nasnetin': 908869632,\
        'ncp_3ddet': 12419122944,\
        'ncp_cls-10-1000': 13727907840,\
        'ncp_cls-10c': 13727907840,\
        'ncp_cls-50-100': 13727907840,\
        'ncp_cls-50-1000': 13727907840,\
        'ncp_seg': 13727907840,\
        'ncp_seg-4x': 13727907840,\
        'ncp_video': 13727907840,\
        'ncp_video-p': 13727907840,\
        'pnas': 1622589440,\
        'pnasfixwd': 540258304,\
        'pnasin': 898228224,\
        'resnet': 896475136,\
        'resnexta': 611136512,\
        'resnextain': 1226891264,\
        'resnextb': 898228224,\
        'resnextbin': 2751660032,\
        'transmicroautoencoder': 1505384448,\
        'transmicroclassobject': 1505384448,\
        'transmicroclassscene': 1505384448,\
        'transmicrojigsaw': 1505384448,\
        'transmicronormal': 1505384448, \
        'transmicroroomlayout': 1505384448, \
        'transmicrosegmentsemantic': 1505384448
        }
    if search_space not in lut:
        raise NotImplementedError
    return lut[search_space]