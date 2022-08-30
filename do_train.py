import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import time
import glob
import logging
import sys
import os
import torch.backends.cudnn as cudnn
import numpy as np
import random
import yaml
import json
from scipy import stats
from model_wrapper import model_dict
from model_wrapper.cnn_wrapper import CNNWrapper

def get_rank_corr(accs, gts, metric = 'spearmanr'):
    if metric == 'spearmanr':
        corr, p = stats.spearmanr(accs, gts)
    elif metric == 'kendalltau':
        corr, p = stats.kendalltau(accs, gts)
    else:
        raise NotImplementedError
    return corr, p

def get_optimizer(optim_config, model):
    if optim_config.optim == 'sgd':
        optimizer = torch.optim.SGD(utils.get_parameters(model),
                                lr=optim_config.lr,
                                momentum=optim_config.momentum,
                                weight_decay=optim_config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
    

def get_archs_acc(arch_config):
    with open(arch_config.train_archs_accs, 'r') as t:
        train_archs_accs = json.load(t)
    with open(arch_config.eval_archs_accs, 'r') as t:
        eval_archs_accs = json.load(t)
    train_acc_rank = {}
    accs = []
    for _,acc in train_archs_accs:
        accs.append(acc)
    orders = np.argsort(accs) # np.flip(np.argsort(accs))
    for indx,order in enumerate(orders):
        train_acc_rank[accs[order]] = indx + 1
    return train_archs_accs, eval_archs_accs, train_acc_rank

def evaluation(archs_accs, config, data, generator, device):
    loss_matrix = []
    accs = []
    final_test_loss = []

    with torch.no_grad():
        generator.train()
        output_g = generator(data)
        if config.head_config.loss_type == 'celoss':
            output_g = torch.sigmoid(output_g).data.round().long()
        elif config.head_config.loss_type == 'mseloss':
            output_g = output_g.data

    for iter,(arch,acc) in enumerate(archs_accs):
        accs.append(acc)
        config.backbone_config.arch = arch
        model = CNNWrapper(config.backbone_config, config.head_config).to(device)
        utils.init_net(model, 'none', 'none')
        model.train()
        losses = []
        optimizer = get_optimizer(config.optim_a_config, model)
        for a_iter in range(config.train_config.a_iters):
            output_m_cache = model(data)
            if config.head_config.model == 'HeadEmpty':
                output_m = generator.forward_another_branch(output_m_cache)
            else:
                output_m = output_m_cache
            if config.head_config.loss_type == 'celoss':
                loss_m = F.cross_entropy(output_m,output_g)
            elif config.head_config.loss_type == 'mseloss':
                loss_m = F.mse_loss(output_m,output_g)
            optimizer.zero_grad()
            loss_m.backward()
            optimizer.step()
            losses.append(float(loss_m))
        loss_matrix.append(losses)
        final_test_loss.append(losses[-1])
        if (iter + 1) % config.train_config.eval_show_metric_interval == 0:
            logging.info(f'eval iter: {iter}; ranking: {get_rank_corr(final_test_loss, accs)[0]:.2f}')

    loss_matrix = np.asarray(loss_matrix).reshape(-1, config.train_config.a_iters)
    corr, _ = get_rank_corr(loss_matrix[:,-1], accs)
    if config.train_config.rank_reverse == True:
        corr = corr * -1
    return loss_matrix, corr
            


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help='config yaml', default='nb101_siam_mse')
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--log', type=str, default='./log')
    argparser.add_argument('--note', type=str, default='nb101')
    argparser.add_argument('--seed', type=int,default='111')
    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = "cuda:"+ "0" if torch.cuda.is_available() else "cpu" 

    torch.manual_seed(args.seed) 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True #reproducibility
    torch.backends.cudnn.benchmark = False

    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.save = os.path.join(args.log,'exp-{}-{}-{}-{}'.format(args.note, args.config, args.seed, timestamp))

    scripts_to_save = []
    for files in ['*.py','./model_wrapper/**/*.py']:
        scripts_to_save.extend(glob.glob(files, recursive = True))
    utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(f"device is --------------{args.device}")
    # load config
    with open(os.path.join('./configs',args.config + '.yaml'), 'r') as f:
        config = utils.yaml_parser(yaml.unsafe_load(f))
    # load data and generator
    data = utils.load_one_batch_image(config.dataset_config)
    generator = model_dict.GENERATOR_CONFIGS[config.generator_config.model](config.head_config)
    data = data.to(args.device)
    generator = generator.to(args.device)
    optimizer_g = get_optimizer(config.optim_g_config, generator)
    train_archs_accs, eval_archs_accs, train_acc_rank = get_archs_acc(config.arch_config)
    
    # train
    best_rank = 0.
    loss_matrix, rank = evaluation(train_archs_accs, config, data, generator, args.device)
    if rank > best_rank:
        best_rank = rank
        torch.save(generator.state_dict(), os.path.join(args.save, 'best.pth'))
    for pair in range(config.train_config.total_pairs):
        with torch.no_grad():
            output_g = generator(data)
            if config.head_config.loss_type == 'celoss':
                output_g = torch.sigmoid(output_g).data.round().long()
            else:
                output_g = output_g

        a,b = random.sample(train_archs_accs,2)
        arch_a,acc_a = a 
        arch_b,acc_b = b 
        config.backbone_config.arch = arch_a
        model_a = CNNWrapper(config.backbone_config, config.head_config).to(args.device)
        config.backbone_config.arch = arch_b
        model_b = CNNWrapper(config.backbone_config, config.head_config).to(args.device)
        optimizer_m_a = get_optimizer(config.optim_a_config, model_a)
        optimizer_m_b = get_optimizer(config.optim_a_config, model_b)
        for a_iter in range(config.train_config.a_iters):
            output_m_a_cache = model_a(data)
            output_m_b_cache = model_b(data)

            if config.head_config.model == 'HeadEmpty':
                output_m_a = generator.forward_another_branch(output_m_a_cache)
                output_m_b = generator.forward_another_branch(output_m_b_cache)
            else:
                output_m_a = output_m_a_cache
                output_m_b = output_m_b_cache
            if config.head_config.loss_type == 'celoss':
                loss_m_a = F.cross_entropy(output_m_a,output_g)
            elif config.head_config.loss_type == 'mseloss':
                loss_m_a = F.mse_loss(output_m_a,output_g)
            optimizer_m_a.zero_grad()
            loss_m_a.backward()
            optimizer_m_a.step()
            if config.head_config.loss_type == 'celoss':
                loss_m_b = F.cross_entropy(output_m_b,output_g)
            elif config.head_config.loss_type == 'mseloss':
                loss_m_b = F.mse_loss(output_m_b,output_g)
            optimizer_m_b.zero_grad()
            loss_m_b.backward()
            optimizer_m_b.step()
        

        if config.head_config.loss_type == 'celoss':
            output_m_a = torch.max(output_m_a.data, 1)[1]
            output_m_b = torch.max(output_m_b.data, 1)[1]
        elif config.head_config.loss_type == 'mseloss':
            output_m_a = output_m_a.data
            output_m_b = output_m_b.data

        for g_iter in range(config.train_config.g_iters):
            if config.head_config.model == 'HeadEmpty': #override
                if config.head_config.loss_type == 'celoss':
                    output_m_a = torch.max(generator.forward_another_branch(output_m_a_cache.data), 1)[1]
                    output_m_b = torch.max(generator.forward_another_branch(output_m_b_cache.data), 1)[1]
                elif config.head_config.loss_type == 'mseloss':
                    output_m_a = generator.forward_another_branch(output_m_a_cache.data)
                    output_m_b = generator.forward_another_branch(output_m_b_cache.data)

            output_g = generator(data)
            try: 
                if config.train_config.mask == True:
                    mask = torch.rand_like(output_g) > 0.5
                    output_g = output_g * mask
            except:
                pass
            dis_a = ((output_g - output_m_a)**2).flatten() #loss1
            dis_b = ((output_g - output_m_b)**2).flatten() #loss2
            if train_acc_rank[acc_a] < train_acc_rank[acc_b]:
                a_gt_b = 1
            else:
                a_gt_b = -1
            a_gt_b = torch.ones(dis_a.shape).to(args.device) * a_gt_b
            loss_g = F.margin_ranking_loss(dis_a,dis_b,a_gt_b) # + loss_function(output_d,output_rs.data) #+ loss_function(output_d,torch.sign(output_d).to(args.device))
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
        logging.info(f'arch a loss/acc: {float(loss_m_a):.2f} {acc_a:.4f},arch b loss/acc: {float(loss_m_b):.2f} {acc_b:.4f}; generator loss: {float(loss_g):.4f}')
        if (pair + 1) % config.train_config.eval_interval == 0:
            loss_matrix, rank = evaluation(train_archs_accs, config, data, generator, args.device)
            if rank > best_rank:
                best_rank = rank
                torch.save(generator.state_dict(), os.path.join(args.save, 'best.pth'))
    if best_rank > 0:
        generator.load_state_dict(torch.load(os.path.join(args.save, 'best.pth')))
    loss_matrix, rank = evaluation(eval_archs_accs, config, data, generator, args.device)








    
    
