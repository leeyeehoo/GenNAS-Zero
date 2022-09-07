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
from model_wrapper.backbones.model_nds.pycls.models.nas.fbnas import NDS
import copy
from ptflops import get_model_complexity_info

def group_loss(accs, gts, train = True, decay = 'exp'):
    losses = []
    if train:
        for acc,gt in zip(accs, gts):
            losses.append(F.mse_loss(acc, gt))
        return sum(losses)
    else:
        if decay == 'exp':
            for i, (acc,gt) in enumerate(zip(accs, gts)):
                losses.append(F.mse_loss(acc, gt) / 2**i )
        elif decay == 'constant':
            for i, (acc,gt) in enumerate(zip(accs, gts)):
                losses.append(F.mse_loss(acc, gt))
        return sum(losses)

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
    elif optim_config.optim == 'adam':
        optimizer = torch.optim.Adam(utils.get_parameters(model),
                                lr=optim_config.lr
                                )
    elif optim_config.optim == 'adamw':
        optimizer = torch.optim.AdamW(utils.get_parameters(model),
                                lr=optim_config.lr
                                )
    elif optim_config.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(utils.get_parameters(model),
                                lr=optim_config.lr)
    else:
        raise NotImplementedError
    return optimizer
def config_by_config_dict(config, config_dict):
    config.head_config.out_channel = config_dict['channel']
    config.head_config.last_channel = config_dict['channel']
    config.head_config.combo = config_dict['combo']
    config.optim_a_config.lr = config_dict['lr']
    config.head_config.index = config_dict['index']

def generate_nas_config(config):
    nas_config_dict = {}

    total_features = len(config.nas_config.features)
    total_branch_number = len(config.nas_config.network.branch_number)
    total_head_ops = len(config.nas_config.network.indexes)
    total_barrier_ops = len(config.nas_config.network.barrier_indexes)
    total_channel = len(config.nas_config.network.channels)
    total_level = len(config.nas_config.levels)

    nas_config_dict['feature_indx'] = np.random.random((total_branch_number, total_features)) > 0.8
    nas_config_dict['head_indx'] = (np.random.random(total_branch_number) * total_head_ops).astype(int)
    nas_config_dict['barrier_indx'] = (np.random.random(total_branch_number) * total_barrier_ops).astype(int)
    nas_config_dict['out_channel_indx'] = (np.random.random(total_branch_number) * total_channel).astype(int)
    nas_config_dict['last_channel_indx'] = (np.random.random(total_branch_number) * total_channel).astype(int)
    nas_config_dict['level_indx'] = (np.random.random(total_branch_number) * total_level).astype(int) 
    nas_config_dict['lr'] = random.choice(config.nas_config.lrs)
    nas_config_dict['alpha'] = random.choice(config.nas_config.alphas)
    nas_config_dict['branch_number'] = random.choice(config.nas_config.network.branch_number)
    nas_config_dict['init_net'] = random.choice(config.nas_config.cnn_inits)
    nas_config_dict['init_barrier'] = random.choice(config.nas_config.barrier_inits)
    return nas_config_dict

def set_nas_config(config, nas_config_dict):
    branch_number = nas_config_dict['branch_number'] 
    config.optim_a_config.lr = nas_config_dict['lr']
    config.train_config.init_net = nas_config_dict['init_net']
    config.train_config.init_barrier = nas_config_dict['init_barrier']
    config.head_config.levels = []
    config.head_config.alpha = nas_config_dict['alpha']
    config.head_config.feature_list = []
    config.head_config.out_channel = []
    config.head_config.last_channel = []
    config.head_config.index = []
    config.head_config.barrier_index = []
    for i in range(branch_number):
        features = []
        for k, flag in enumerate(nas_config_dict['feature_indx'][i]):
            if flag == 1:
                features.append(config.nas_config.features[k])
        

        config.head_config.feature_list.append(features)
        config.head_config.out_channel.append(config.nas_config.network.channels[nas_config_dict['out_channel_indx'][i]])
        config.head_config.last_channel.append(config.nas_config.network.channels[nas_config_dict['last_channel_indx'][i]])
        config.head_config.index.append(config.nas_config.network.indexes[nas_config_dict['head_indx'][i]])
        config.head_config.barrier_index.append(config.nas_config.network.indexes[nas_config_dict['barrier_indx'][i]])
        config.head_config.levels.append(config.nas_config.levels[nas_config_dict['level_indx'][i]])

class Explorer():
    def __init__(self, config):
        self.config = config
        self.mutate_ratio = config.ea_config.mutate_ratio
        self.p_size = config.ea_config.p_size
        self.e_size = config.ea_config.e_size
        assert self.p_size > self.e_size
        self.counter = 0
        self.history = []
        self.accs = []
        for _ in range(self.p_size):
            self.history.append([0, generate_nas_config(config)])
    def update(self, acc):
        self.history[self.counter][0] = acc
        self.counter += 1
        if self.counter >= self.p_size:
            p = self.history[-self.p_size:]
            sample = random.sample(p, self.e_size)
            best_config_dict = sorted(sample, key=lambda i:i[0])[-1][1]
            new_config_dict = self.mutate(best_config_dict)
            self.history.append([0, new_config_dict])

    def mutate(self, config_dict):
        new_config_dict = copy.deepcopy(config_dict)

        total_features = len(self.config.nas_config.features)
        total_branch_number = len(self.config.nas_config.network.branch_number)
        total_head_ops = len(self.config.nas_config.network.indexes)
        total_barrier_ops = len(self.config.nas_config.network.barrier_indexes)
        total_channel = len(self.config.nas_config.network.channels)
        total_level = len(config.nas_config.levels)
        if random.random() > self.mutate_ratio:
            new_config_dict['lr'] = random.choice(self.config.nas_config.lrs)
        if random.random() > self.mutate_ratio:
            new_config_dict['alpha'] = random.choice(self.config.nas_config.alphas)
        if random.random() > self.mutate_ratio:
            new_config_dict['init_net'] = random.choice(config.nas_config.cnn_inits)
        if random.random() > self.mutate_ratio:
            new_config_dict['init_barrier'] = random.choice(config.nas_config.barrier_inits)
        if random.random() > self.mutate_ratio:
            new_config_dict['branch_number'] = random.choice(self.config.nas_config.network.branch_number)
        for i in range(total_branch_number):
            for j in range(total_features):
                if random.random() > self.mutate_ratio:
                    new_config_dict['feature_indx'][i][j] = 1 - new_config_dict['feature_indx'][i][j]
        for i in range(total_branch_number):
            if random.random() > self.mutate_ratio:
                new_config_dict['head_indx'][i] = int(random.random() * total_head_ops)
            if random.random() > self.mutate_ratio:
                new_config_dict['barrier_indx'][i] = int(random.random() * total_barrier_ops)
            if  random.random() > self.mutate_ratio:
                new_config_dict['out_channel_indx'][i] = int(random.random() * total_channel)
            if  random.random() > self.mutate_ratio:
                new_config_dict['last_channel_indx'][i] = int(random.random() * total_channel)
            if  random.random() > self.mutate_ratio:
                new_config_dict['level_indx'][i] = int(random.random() * total_level)
        return new_config_dict



def get_archs_acc(config):
    try:
        if 'NDS' in config.backbone_config.model:
            config.backbone_config.NDS = NDS(config.backbone_config.search_space,config.backbone_config.datapath)
            total_archs = list(np.arange(len(config.backbone_config.NDS )))
            random.shuffle(total_archs)
            train_eval_archs = total_archs
            train_archs = train_eval_archs[:20]
            eval_archs = train_eval_archs[20:]
            try:
                total_archs = total_archs[:(config.train_config.max_eval_sample + config.train_config.train_sample)]
                train_archs = total_archs[:config.train_config.train_sample]
                eval_archs = total_archs[config.train_config.train_sample:]
            except:
                pass
            train_archs_accs = []
            eval_archs_accs = []
            for arch in train_archs:
                acc = config.backbone_config.NDS.get_final_accuracy(arch,None,None)
                train_archs_accs.append([arch,acc])
            for arch in eval_archs:
                acc = config.backbone_config.NDS.get_final_accuracy(arch,None,None)
                eval_archs_accs.append([arch,acc])
            train_acc_rank = {}
            accs = []
            for _,acc in train_archs_accs:
                accs.append(acc)
            orders = np.argsort(accs) # np.flip(np.argsort(accs))
            for indx,order in enumerate(orders):
                train_acc_rank[accs[order]] = indx + 1
            return train_archs_accs, eval_archs_accs, train_acc_rank
    except Exception as e: 
        print(e)
    with open(config.arch_config.train_archs_accs, 'r') as t:
        train_archs_accs = json.load(t)
    with open(config.arch_config.eval_archs_accs, 'r') as t:
        eval_archs_accs = json.load(t)
    random.shuffle(train_archs_accs)
    random.shuffle(eval_archs_accs)
    try:
        train_archs_accs = train_archs_accs[:config.train_config.train_sample]
        eval_archs_accs = eval_archs_accs[:config.train_config.max_eval_sample]
    except Exception as e: 
        print(e)

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
    decay = 'exp'
    try:
        decay = config.train_config.decay
    except:
        pass
    with torch.no_grad():
        generator.train()
        output_g = generator(data)
        if config.head_config.loss_type == 'celoss':
            output_g = torch.sigmoid(output_g).data.round().long()
        elif config.head_config.loss_type == 'mseloss':
            if type(output_g) == list:
                for i in range(len(output_g)):
                    output_g[i] = output_g[i].data
            else: output_g = output_g.data
    
    mac_mode = 'loss'
    alpha = 0
    try:
        mac_mode = config.train_config.mac_mode
        alpha = config.head_config.alpha
    except:
        pass
    
    for iter,(arch,acc) in enumerate(archs_accs):
        accs.append(acc)
        config.backbone_config.arch = arch
        model = CNNWrapper(config.backbone_config, config.head_config).to(device)
        
        if 'macs' in mac_mode:
            try:
                max_macs = utils.get_macs_lut(config.head_config.search_space)
            except:
                pass
            macs, params = get_model_complexity_info(model.backbone, (3, 32, 32), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        else:
            macs = 0
        
        if 'loss' in mac_mode:
            try: 
                utils.init_net(model, config.train_config.init_net)
            except:
                pass
            if 'HeadEmpty' in config.head_config.model:
                barrier = model_dict.BARRIER_CONFIGS[config.barrier_config.model](config.head_config).to(args.device)
            else:
                barrier = None
            try: 
                utils.init_net(barrier, config.train_config.init_barrier)
            except:
                pass
            model.train()
            losses = []
            optimizer = get_optimizer(config.optim_a_config, model)

            for a_iter in range(config.train_config.a_iters):
                
                output_m_cache = model(data)
                if 'HeadEmpty' in config.head_config.model:
                    output_m = barrier(output_m_cache)
                else:
                    output_m = output_m_cache

                if config.head_config.loss_type == 'celoss':
                    loss_m = F.cross_entropy(output_m,output_g)
                elif config.head_config.loss_type == 'mseloss':
                    loss_m = group_loss(output_m,output_g, train = True, decay = 'constant')

                optimizer.zero_grad()
                loss_m.backward()
                grad_flag = False
                try: grad_flag = config.train_config.grad_metric
                except: pass
                if grad_flag == 'gradmean':
                    grad = []
                    for p in model.parameters():
                        if p.grad is not None:
                            grad.append(p.grad.detach().flatten())
                    grad_mean = torch.cat(grad).mean()
                    loss_m = grad_mean
                    # print(grad_mean)
                # else:
                #     raise NotImplementedError
                optimizer.step()

                with torch.no_grad():
                    loss_e = group_loss(output_m,output_g, train = False, decay = decay)
                
                if 'macs' in mac_mode:
                    losses.append(float(loss_e * (1 + alpha * (- macs/max_macs))))
                else:
                    losses.append(float(loss_e))

            # print(float(loss_e), 1 + (alpha * (- macs/max_macs)))
        else:
            losses = [-macs]
        loss_matrix.append(losses)
        final_test_loss.append(losses[-1])
        if (iter + 1) % config.train_config.eval_show_metric_interval == 0:
            logging.info(f'eval iter: {iter}; ranking: {get_rank_corr(final_test_loss, accs)[0]:.2f}')
    if 'loss' in mac_mode: 
        loss_matrix = np.asarray(loss_matrix).reshape(-1, config.train_config.a_iters)
    else:
        loss_matrix = np.asarray(loss_matrix).reshape(-1, 1)
    corr, _ = get_rank_corr(loss_matrix[:,-1], accs)
    if config.train_config.rank_reverse == True:
        corr = corr * -1
    return loss_matrix, corr
            


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help='config yaml', default='nb101_siam_mse')
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--log', type=str, default='./log')
    argparser.add_argument('--note', type=str, default='benchmark')
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
    

    try:
        generator.load_state_dict(torch.load(config.generator_config.load_from_save,map_location='cpu')[config.generator_config.state_dict_name])
        logging.info('load from pretrained')
    except Exception as e: 
        print(e)

    data = data.to(args.device)
    generator = generator.to(args.device)
    optimizer_g = get_optimizer(config.optim_g_config, generator)
    
    train_archs_accs, eval_archs_accs, train_acc_rank = get_archs_acc(config)
    np.save(os.path.join(args.save, 'benchmark.npy'),np.asarray([train_archs_accs, eval_archs_accs]))
    # train
    nas_flag = False
    try:
        if config.nas_config:
            nas_flag = True
    except Exception as e: 
        print(e)
    if nas_flag == True:
        ea_flag = False
        try:
            if config.ea_config:
                ea_flag = True
        except Exception as e: 
            print(e)
        if ea_flag:
            ea_engine = Explorer(config)

        config_dicts_ranks = []
        best_rank = 0.
        best_pair = -1
        for pair in range(config.train_config.total_pairs):
            if ea_flag:
                config_dict = ea_engine.history[pair][1]
                set_nas_config(config, config_dict)
            else:
                config_dict = generate_nas_config(config)
                set_nas_config(config, config_dict)
            generator = model_dict.GENERATOR_CONFIGS[config.generator_config.model](config.head_config).to(args.device)
            loss_matrix, rank = evaluation(train_archs_accs, config, data, generator, args.device)
            if ea_flag:
                ea_engine.update(rank)

            if np.isnan(rank):
                rank = -1
            if rank > best_rank:
                best_rank = rank
                best_pair = pair
                torch.save(generator.state_dict(), os.path.join(args.save, 'generator.pth'))
            config_dicts_ranks.append([config_dict, rank])
            logging.info(f'pair: {pair}, nas config: {config_dict}, rank: {rank}')
        set_nas_config(config, config_dicts_ranks[best_pair][0])
        np.save(os.path.join(args.save, 'loss_matrix.npy'), config_dicts_ranks)
        generator = model_dict.GENERATOR_CONFIGS[config.generator_config.model](config.head_config).to(args.device)
        try:
            generator.load_state_dict(torch.load(os.path.join(args.save, 'generator.pth')))
        except Exception as e: 
            print(e)

    # train_flag = False
    # try:
    #     if config.train_config.train == True:
    #         train_flag = True
    # except:
    #     pass
    # if train_flag == True:
    #     best_rank = 0.
    #     loss_matrix, rank = evaluation(eval_archs_accs, config, data, generator, args.device)
    #     if rank > best_rank:
    #         best_rank = rank
    #         torch.save(generator.state_dict(), os.path.join(args.save, 'best.pth'))
    #     for pair in range(config.train_config.total_pairs):
    #         with torch.no_grad():
    #             output_g = generator(data)
    #             if config.head_config.loss_type == 'celoss':
    #                 output_g = torch.sigmoid(output_g).data.round().long()
    #             else:
    #                 output_g = output_g

    #         a,b = random.sample(train_archs_accs,2)
    #         arch_a,acc_a = a 
    #         arch_b,acc_b = b 
    #         config.backbone_config.arch = arch_a
    #         model_a = CNNWrapper(config.backbone_config, config.head_config).to(args.device)
    #         config.backbone_config.arch = arch_b
    #         model_b = CNNWrapper(config.backbone_config, config.head_config).to(args.device)
    #         optimizer_m_a = get_optimizer(config.optim_a_config, model_a)
    #         optimizer_m_b = get_optimizer(config.optim_a_config, model_b)
    #         for a_iter in range(config.train_config.a_iters):
    #             output_m_a_cache = model_a(data)
    #             output_m_b_cache = model_b(data)

    #             if 'HeadEmpty' in config.head_config.model:
    #                 output_m_a = generator.forward_another_branch(output_m_a_cache)
    #                 output_m_b = generator.forward_another_branch(output_m_b_cache)
    #             else:
    #                 output_m_a = output_m_a_cache
    #                 output_m_b = output_m_b_cache
    #             if config.head_config.loss_type == 'celoss':
    #                 loss_m_a = F.cross_entropy(output_m_a,output_g)
    #             elif config.head_config.loss_type == 'mseloss':
    #                 loss_m_a = F.mse_loss(output_m_a,output_g)
    #             optimizer_m_a.zero_grad()
    #             loss_m_a.backward()
    #             optimizer_m_a.step()
    #             if config.head_config.loss_type == 'celoss':
    #                 loss_m_b = F.cross_entropy(output_m_b,output_g)
    #             elif config.head_config.loss_type == 'mseloss':
    #                 loss_m_b = F.mse_loss(output_m_b,output_g)
    #             optimizer_m_b.zero_grad()
    #             loss_m_b.backward()
    #             optimizer_m_b.step()
            

    #         if config.head_config.loss_type == 'celoss':
    #             output_m_a = torch.max(output_m_a.data, 1)[1]
    #             output_m_b = torch.max(output_m_b.data, 1)[1]
    #         elif config.head_config.loss_type == 'mseloss':
    #             output_m_a = output_m_a.data
    #             output_m_b = output_m_b.data

    #         for g_iter in range(config.train_config.g_iters):
    #             if 'HeadEmpty' in config.head_config.model: #override
    #                 if config.head_config.loss_type == 'celoss':
    #                     output_m_a = torch.max(generator.forward_another_branch(output_m_a_cache.data), 1)[1]
    #                     output_m_b = torch.max(generator.forward_another_branch(output_m_b_cache.data), 1)[1]
    #                 elif config.head_config.loss_type == 'mseloss':
    #                     output_m_a = generator.forward_another_branch(output_m_a_cache.data)
    #                     output_m_b = generator.forward_another_branch(output_m_b_cache.data)

    #             output_g = generator(data)
    #             try: 
    #                 if config.train_config.mask == True:
    #                     mask = torch.rand_like(output_g) > 0.5
    #                     output_g = output_g * mask
    #             except:
    #                 pass
    #             dis_a = ((output_g - output_m_a)**2).flatten() #loss1
    #             dis_b = ((output_g - output_m_b)**2).flatten() #loss2
    #             if train_acc_rank[acc_a] < train_acc_rank[acc_b]:
    #                 a_gt_b = 1
    #             else:
    #                 a_gt_b = -1
    #             a_gt_b = torch.ones(dis_a.shape).to(args.device) * a_gt_b
    #             loss_g = F.margin_ranking_loss(dis_a,dis_b,a_gt_b) # + loss_function(output_d,output_rs.data) #+ loss_function(output_d,torch.sign(output_d).to(args.device))
    #             optimizer_g.zero_grad()
    #             loss_g.backward()
    #             optimizer_g.step()
    #         logging.info(f'arch a loss/acc: {float(loss_m_a):.2f} {acc_a:.4f},arch b loss/acc: {float(loss_m_b):.2f} {acc_b:.4f}; generator loss: {float(loss_g):.4f}')
    #         if (pair + 1) % config.train_config.eval_interval == 0:
    #             loss_matrix, rank = evaluation(train_archs_accs, config, data, generator, args.device)
    #             if rank > best_rank:
    #                 best_rank = rank
    #                 torch.save(generator.state_dict(), os.path.join(args.save, 'best.pth'))
    # if best_rank > 0:
    #     generator.load_state_dict(torch.load(os.path.join(args.save, 'best.pth')))
    start = time.time()
    # for i in range(5):

    loss_matrix, rank = evaluation(eval_archs_accs, config, data, generator, args.device)
    end = time.time()
    logging.info(f'evaluation time: {end - start}')
    np.save(os.path.join(args.save, 'loss_matrix.npy'), loss_matrix)
    logging.info(f'saved evaluation loss matrix')







    
    
