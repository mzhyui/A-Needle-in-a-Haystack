#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import datetime
import gc
import math
import os
import pickle
import re
import time
import atexit

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import torch.multiprocessing as mp
# from torch.multiprocessing import Pool

from models.test import Evaluator
from utils.attacker import Attacker
from utils.channelLipz import CL
from utils.defense import Defender
from utils.options import args_parser
from utils.dataUtils import getDataWithDistribution
from utils.sampling import plotDataDistribution
from utils.trainUtils import getModel, getWglob, getWglobTSSWeight, parallelTrainingIntegrated
from utils.trainUtils import train_user_attack, train_user_normal
from utils.logger import myLogger

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    def getDefenseMethod():
        if args.krum or args.mkrum:
            return "Krum"
        elif args.clipping:
            return "Clipping"
        elif args.tracer:
            return "Tracer"
        elif args.rlr:
            return "RLR"
        elif args.tss_statistical_ban < 999 or args.tss_statistical_reject < 999 or args.tss_statistical_reject_kmeans < 999 or args.tss_hard_reject < 999 or args.tss_statistical_layerwise < 999:
            return "TSS"
        elif args.tss_soft_reject < 999:
            return "TSSSoftReject"
        elif args.grad_mask_only < 999:
            return "GradReject"
        elif args.flame:
            return "FLAME"
        else:
            return 'None'

    if args.noniid_metric == 'iid':
        distribution_metric = 0
    elif args.noniid_metric == 'shard':
        distribution_metric = args.shard_per_user
    elif args.noniid_metric == 'dirichlet':
        distribution_metric = args.dirichlet_alpha
    elif args.noniid_metric == 'unbalanced':
        distribution_metric = args.ub_label
    else:
        raise ValueError("noniid_metric not found")
    base_dir = os.path.join(args.results_save, 'debug' if (args.debug) else 'release', args.dataset, args.noniid_metric+str(distribution_metric), f'{args.model}_num-{args.num_users}_C-{args.frac}',
                            args.attack_type, f'lr{args.lr}ep{args.local_ep}', f"{getDefenseMethod()}_{datetime.datetime.now().strftime('%m-%d--%H-%M-%S')}")


    net_glob = getModel(model_name=args.model, dataset=args.dataset,
                            num_channels=args.num_channels, num_classes=args.num_classes, input_size=args.input_size, device=args.device)
    last_global_dict = net_glob.state_dict()
    net_glob.train()

    dataset_train, dataset_test, dict_users_train, dict_users_test = getDataWithDistribution(
            args)
    lr = args.lr

    attacker_default = Attacker(args)
    attacker_default.setAttackClients([0, 1])
    server_defender = Defender(args=args)
    evaluator = Evaluator(args=args)

    idxs_weight_dict = dict(
        list(zip(list(range(0,20)), np.linspace(100, 100, args.num_users, dtype=int))))

    for iter_ in range(20):
        w_local_list = []
        m_trigger=copy.deepcopy(attacker_default.mirage_trigger_set)
        for idx in range(1, 2):
            idx, w_local, _, loss = train_user_attack(
                iter_, idx, args, attacker_default, server_defender, list(range(0,20)), idxs_weight_dict, net_glob, last_global_dict, dataset_train, dict_users_train, lr, False, base_dir)
            w_local_list.append([idx, w_local, idxs_weight_dict[idx]])

        for idx in range(1,2):
            idx, w_local, _, loss = train_user_attack(
                iter_, idx, args, attacker_default, server_defender, list(range(0,20)), idxs_weight_dict, net_glob, last_global_dict, dataset_train, dict_users_train, lr, False, base_dir)
            w_local_list.append([idx, w_local, idxs_weight_dict[idx]])

        for idx in range(2,20):
            idx, w_local, idxs_weight_dict[idx], loss = train_user_normal(
                iter_, idx, args, server_defender, idxs_weight_dict, net_glob, dataset_train, dict_users_train, lr, False, base_dir, [])
            w_local_list.append([idx, w_local, idxs_weight_dict[idx]])
    
        w_glob = getWglob(w_local_list)
        net_glob.load_state_dict(w_glob)
    
        net_glob_eval = copy.deepcopy(net_glob)
        net_glob_eval.eval()
        acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test(
            # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
            net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
        print('Round {:3d}, Total users: {}, Test acc: {:.2f}%, Backdoor acc: {:.2f}%, Test loss: {:.4f}'.format(
            iter_, args.num_users, acc_test, attack_prediction, loss_test))
        
        
        net_glob_eval.load_state_dict(w_local_list[0][1])
        net_glob_eval.eval()
        acc_test_mal, loss_test_mal, correct_prediction_mal, attack_prediction_mal = evaluator.test(
            net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
        print('Round {:3d}, Malicious user: {}, Test acc: {:.2f}%, Backdoor acc: {:.2f}%, Test loss: {:.4f}'.format(
            iter_, 0, acc_test_mal, attack_prediction_mal, loss_test_mal))
        
        net_glob_eval.load_state_dict(w_local_list[1][1])
        net_glob_eval.eval()
        acc_test_mal, loss_test_mal, correct_prediction_mal, attack_prediction_mal = evaluator.test(
            net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=1)
        print('Round {:3d}, Malicious user: {}, Test acc: {:.2f}%, Backdoor acc: {:.2f}%, Test loss: {:.4f}'.format(
            iter_, 1, acc_test_mal, attack_prediction_mal, loss_test_mal))

        # print(m_trigger == attacker_default.mirage_trigger_set)
        print([torch.equal(m_t, m_t_new) for m_t, m_t_new in zip(m_trigger, attacker_default.mirage_trigger_set)])