
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import shutil


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


def init_net(net, w_type, b_type):
    if w_type == 'none':
        pass
    elif w_type == 'xavier':
        net.apply(init_weights_vs)
    elif w_type == 'kaiming':
        net.apply(init_weights_he)
    elif w_type == 'zero':
        net.apply(init_weights_zero)
    else:
        raise NotImplementedError(f'init_type={w_type} is not supported.')

    if b_type == 'none':
        pass
    elif b_type == 'xavier':
        net.apply(init_bias_vs)
    elif b_type == 'kaiming':
        net.apply(init_bias_he)
    elif b_type == 'zero':
        net.apply(init_bias_zero)
    else:
        raise NotImplementedError(f'init_type={b_type} is not supported.')

def init_weights_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

def init_bias_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.xavier_normal_(m.bias)

def init_weights_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)

def init_bias_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.kaiming_normal_(m.bias)

def init_weights_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(.0)

def init_bias_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            m.bias.data.fill_(.0)
