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
import atexit

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm
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
from utils.assess import Assessor, reverse_engineer, realTriggerImportance
from utils.logger import myLogger


TSS_DEFENSE_OPTIONS = (
    "tss_statistical_ban",
    "tss_statistical_reject",
    "tss_statistical_reject_kmeans",
    "tss_statistical_layerwise",
    "tss_statistical_reject_trigger",
    "tss_statistical_reject_noise",
    "tss_hard_reject",
)


def get_defense_method(args):
    if args.krum or args.mkrum:
        return "Krum"
    if args.clipping:
        return "Clipping"
    if args.tracer:
        return "Tracer"
    if args.rlr:
        return "RLR"
    if any(getattr(args, option) < 999 for option in TSS_DEFENSE_OPTIONS):
        return "TSS"
    if args.tss_soft_reject < 999:
        return "TSSSoftReject"
    if args.grad_mask_only < 999:
        return "GradReject"
    if args.gss < 999:
        return "GSS"
    return "FLAME" if args.flame else "None"


def get_distribution_metric(args):
    metrics = {
        "iid": 0,
        "shard": args.shard_per_user,
        "dirichlet": args.dirichlet_alpha,
        "unbalanced": args.ub_label,
    }
    try:
        return metrics[args.noniid_metric]
    except KeyError:
        raise ValueError(f"Unknown non-IID metric: {args.noniid_metric}")


def get_run_directory(args, timestamp):
    if args.base_dir and args.load_fed:
        return args.base_dir

    distribution = f"{args.noniid_metric}{get_distribution_metric(args)}"
    run_name = f"{get_defense_method(args)}_{timestamp:%m-%d--%H-%M-%S}"
    return os.path.join(
        args.results_save,
        "debug" if args.debug else "release",
        args.dataset,
        distribution,
        f"{args.model}_num-{args.num_users}_C-{args.frac}",
        args.attack_type,
        f"lr{args.lr}ep{args.local_ep}",
        run_name,
    )


def create_run_directories(base_dir):
    for directory in (
        "fed",
        "local_attack_save",
        "local_normal_save",
        "visual",
        "assessor",
    ):
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)


def select_attack_clients(all_clients, attacker_portion, configured_attackers):
    if configured_attackers:
        return np.array(configured_attackers)
    attacker_count = math.ceil(len(all_clients) * attacker_portion)
    return np.random.choice(all_clients, size=attacker_count, replace=False)


def select_normal_save_clients(round_clients, normal_clients, attack_clients):
    round_normal_clients = np.intersect1d(round_clients, normal_clients)
    round_attack_clients = np.intersect1d(round_clients, attack_clients)
    save_count = min(len(round_normal_clients), len(round_attack_clients))
    return np.random.choice(round_normal_clients, save_count, replace=False)


def exit_handler(logger: myLogger):
    logger.info("Exiting...")
    logger.info(datetime.datetime.now().strftime("%m-%d--%H-%M-%S"))
    logger.print("Exit handler called")
    logger.announceDir()


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )

    spawn_ctx = mp.get_context('spawn')
    now = datetime.datetime.now()
    base_dir = get_run_directory(args, now)

    logger = myLogger(
        name=str(os.getpid()),
        log_dir=base_dir,
        log_filename=f"{os.path.basename(__file__)}.log",
        debug=args.debug,
        verbose=args.verbose,
        propagate=args.propagate,
        arg_dict=args.__dict__,
    )
    logger.print("Base_dir: ", base_dir)
    args.base_dir = base_dir
    create_run_directories(base_dir)

    logger.registerDir(base_dir)
    atexit.register(exit_handler, logger)

    data_parallel_devices = args.gpu_list if torch.cuda.device_count() else []
    logger.print("Working on device:", data_parallel_devices or args.device)
    net_glob = getModel(
        model_name=args.model,
        dataset=args.dataset,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        input_size=args.input_size,
        device=args.device,
        device_list=data_parallel_devices,
    )
    last_global_dict = net_glob.state_dict()
    net_glob.train()

    logger("Preparing Defender")
    server_defender = Defender(args=args)
    logger("Preparing Attacker")
    attacker_default = Attacker(args=args)
    logger("Preparing Evaluator")
    evaluator = Evaluator(args=args)
    logger("Preparing Assessor")
    assessor = Assessor(evaluator.dataset_test)

    logger.info(net_glob.state_dict().keys())

    results_save_path = os.path.join(base_dir, 'fed', 'results.csv')
    result_columns = [
        'epoch',
        'loss_avg',
        'loss_test',
        'acc_test',
        'best_acc',
        'correct_prediction',
        'attack_prediction',
    ]
    pd.DataFrame(columns=result_columns).to_csv(
        results_save_path, mode='a', index=False, header=True
    )

    best_acc = -1
    best_epoch = None
    lr = args.lr
    attack_portion = args.portion
    with_local_save = not args.no_local_save

    with open(os.path.join(base_dir, 'settings.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(vars(args))

    logger.info("begin")
    logger.info(datetime.datetime.now().strftime("%m-%d--%H-%M-%S"))

    all_users = np.arange(args.num_users)
    client_weights = dict(zip(all_users, np.full(args.num_users, 100)))
    attack_clients = select_attack_clients(
        all_users, attack_portion, args.attackers
    )
    normal_clients = np.setdiff1d(all_users, attack_clients)
    logger.info(f"attack_clients: {attack_clients}")
    attack_client_ids = attack_clients.tolist()
    attacker_default.setAttackClients(attack_client_ids)
    attacker_dict = {
        client_id: Attacker(args=args, attack_clients=tuple(attack_client_ids))
        for client_id in attack_client_ids
    }

    if (args.load_fed != ''):
        net_glob.load_state_dict(torch.load(args.load_fed, weights_only=True))
        logger.info('model weight loaded')

    # %% init training data
    if args.load_user_dict != '':
        with open(args.load_user_dict, 'rb') as handle:
            dict_users_train, dict_users_test = pickle.load(handle)
        dataset_train, dataset_test, _, _ = getDataWithDistribution(
            args)
        logger.info('user dict loaded')
    else:
        dataset_train, dataset_test, dict_users_train, dict_users_test = getDataWithDistribution(
            args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as dict_save_path_handle:
        pickle.dump((dict_users_train, dict_users_test), dict_save_path_handle)
    logger.info([len(user_indices) for user_indices in dict_users_train.values()])

    plotDataDistribution(
        dict_users=dict_users_train,
        dataset=dataset_train,
        num_classes=args.num_classes,
        fig_path=os.path.join(base_dir, 'visual', 'train_data_distribution.png'),
    )
    plotDataDistribution(
        dict_users=dict_users_test,
        dataset=dataset_test,
        num_classes=args.num_classes,
        fig_path=os.path.join(base_dir, 'visual', 'test_data_distribution.png'),
    )

    assert args.load_begin_epoch <= args.epochs
    # %% training
    stats = ""
    pbar = tqdm(range(args.load_begin_epoch, args.epochs + 1), ncols=200)
    for iter_ in pbar:
        # %% round init
        net_glob.train()
        server_defender.robust_list = [0] * len(all_users)
        if args.debug:
            logger.info(
                f"current average weight: {np.mean(list(client_weights.values()))}")

        w_local_list = []
        loss_locals = []
        if args.dynamic_frac and iter_ == args.dynamic_frac[0]:
            args.frac = args.dynamic_frac[1]
            args.dynamic_frac = args.dynamic_frac[2:]
        selected_client_count = max(int(args.frac * args.num_users), 1)
        selected_clients = np.sort(
            np.random.choice(all_users, selected_client_count, replace=False)
        )
        normal_save_clients = select_normal_save_clients(
            selected_clients, normal_clients, attack_clients
        )
        pbar.set_description(f"Round {iter_}, lr: {lr:.6f}")

        normal_training_clients = np.intersect1d(selected_clients, normal_clients)
        attack_training_clients = np.intersect1d(selected_clients, attack_clients)
        training_clients = np.concatenate(
            (normal_training_clients, attack_training_clients)
        )
        np.random.shuffle(training_clients)
        task_slices = []
        tasks = []
        if args.parallel:
            worker_count = min(args.threads, len(training_clients))
            if worker_count:
                with spawn_ctx.Pool(processes=worker_count) as pool:
                    tasks = [
                        (
                            iter_, client_id, args,
                            client_id in attack_training_clients,
                            attacker_default if client_id not in attack_client_ids
                            else attacker_dict[client_id],
                            server_defender, selected_clients, client_weights,
                            net_glob, last_global_dict, dataset_train,
                            dict_users_train, lr,
                            with_local_save, base_dir, normal_save_clients,
                        )
                        for client_id in training_clients
                    ]
                    task_slices = [
                        tasks[index::worker_count] for index in range(worker_count)
                    ]
                    results = pool.map(parallelTrainingIntegrated, task_slices)
                    for task_results in results:
                        for client_id, local_weights, client_weight, loss in task_results:
                            client_weights[client_id] = client_weight
                            loss_locals.append(loss)
                            w_local_list.append([client_id, local_weights, client_weight])
            else:
                raise ValueError(
                    "No training clients were selected; check --threads and --frac."
                )
        else:
            for client_id in training_clients:
                if client_id in attack_training_clients:
                    client_id, local_weights, client_weight, loss = train_user_attack(
                        iter_, client_id, args,
                        attacker_default if client_id not in attack_client_ids
                        else attacker_dict[client_id],
                        server_defender, selected_clients, client_weights,
                        net_glob, last_global_dict, dataset_train,
                        dict_users_train, lr,
                        with_local_save, base_dir,
                    )
                else:
                    client_id, local_weights, client_weight, loss = train_user_normal(
                        iter_, client_id, args, server_defender, client_weights,
                        net_glob, dataset_train, dict_users_train, lr,
                        with_local_save, base_dir, normal_save_clients,
                    )
                client_weights[client_id] = client_weight
                loss_locals.append(loss)
                w_local_list.append([client_id, local_weights, client_weight])

        # %% aggregate and defend
        lr *= args.lr_decay
        if server_defender.enabled and args.krum:
            logger("KRUM update")
            w_glob = server_defender.getWglobKrum(
                w_local_list, mclients=args.krum_k)
            net_glob.load_state_dict(w_glob)
        elif server_defender.enabled and args.mkrum:
            logger("mKRUM update")
            # w_glob = server_defender.getWglobKrum(w_local_list, mclients=args.krum_k)
            benign_candidates = server_defender.getWglobKrum_multi(
                [w_local_i[1] for w_local_i in w_local_list], last_global_dict)
            logger("benign_candidates", benign_candidates)
            exclued_indices = np.setdiff1d(all_users, benign_candidates)
            w_glob = getWglob(w_local_list, exclued_indices)
            net_glob.load_state_dict(w_glob)
        elif server_defender.enabled and args.clipping:
            logger("clipping")
            w_local_list = [(w_local_list[i][0], server_defender.norm_clipping_per_layer(
                w_local_list[i][1], net_glob), w_local_list[i][2]) for i in range(len(w_local_list))]
            w_glob = getWglob(w_local_list)
            net_glob.load_state_dict(w_glob)
        elif args.batch_gen != -1 and iter_ >= args.batch_gen:
            logger("skip merge")
            w_glob = net_glob.state_dict()

        # %% tracer
        elif server_defender.enabled and args.tracer:
            top_k_indices = server_defender.defense_tracer_topK(np.concatenate(server_defender.defense_tracer(
                [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list])))
            logger("Tracer", top_k_indices)
            w_glob = getWglob(w_local_list, top_k_indices)
            net_glob.load_state_dict(w_glob)
        # %% RLR
        elif server_defender.enabled and args.rlr:
            logger("RLR")
            w_glob = server_defender.defense_rlr(net_glob, [
                                                 w_local_i[1] for w_local_i in w_local_list], last_global_dict, threshold=args.rlr_threshold, device=args.device)
            net_glob.load_state_dict(w_glob)

        elif server_defender.enabled and args.flame:
            logger("FLAME")
            w_glob, benign_clients = server_defender.getWglobFlame(
                [w_local_i[1] for w_local_i in w_local_list], last_global_dict)
            logger("FLAME", benign_clients)
            net_glob.load_state_dict(w_glob)

        # TSS
        # %% hard reject
        elif server_defender.enabled and (iter_ % args.tss_hard_reject == 0):
            # set metamask
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())
            top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), top_k=args.tss_hard_k)
            logger("Top K", top_k_indices)
            w_glob = getWglob(w_local_list, top_k_indices)
            net_glob.load_state_dict(w_glob)

        # TSS
        # %% hard reject
        elif server_defender.enabled and (iter_ % args.grad_mask_only == 0):
            # set metamask
            mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseGradCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                   threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())
            top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), top_k=args.tss_hard_k)
            logger("Top K", top_k_indices)
            w_glob = getWglob(w_local_list, top_k_indices)
            net_glob.load_state_dict(w_glob)

        # %% statistical reject
        elif server_defender.enabled and (iter_ % args.tss_statistical_reject == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())

            top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("TSS top_k_indices", top_k_indices)
            Defender.historical_excluded_list.append(indicator)
            probability = server_defender.banClientProbe(
                all_users, limit=args.tss_statistical_lookback)
            logger.debug_("TSS Probability", probability.tolist())
            banned_indices = np.where(
                probability >= 1/len(all_users)*args.tss_statistical_penalty)[0]
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% statistical reject
        elif server_defender.enabled and (iter_ % args.tss_statistical_reject_noise == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)
            info = [mask, trigger, args.blend_alpha, args.attack_type, args.label,
                    args.pos_choice, args.num_classes, args.num_channels, args.input_size]
            pair_wise_cos_mat = server_defender.defenseTssCompare_noise(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                        threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, info=info, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())

            top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("TSS top_k_indices", top_k_indices)
            Defender.historical_excluded_list.append(indicator)
            probability = server_defender.banClientProbe(
                all_users, limit=args.tss_statistical_lookback)
            logger.debug_("TSS Probability", probability.tolist())
            banned_indices = np.where(
                probability >= 1/len(all_users)*args.tss_statistical_penalty)[0]
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% statistical reject
        elif server_defender.enabled and (iter_ % args.tss_statistical_reject_trigger == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)
            info = [mask, trigger, args.blend_alpha, args.attack_type, args.label,
                    args.pos_choice, args.num_classes, args.num_channels, args.input_size]
            pair_wise_cos_mat = server_defender.defenseTssCompare_trigger(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                          threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, info=info, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())

            top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("TSS top_k_indices", top_k_indices)
            Defender.historical_excluded_list.append(indicator)
            probability = server_defender.banClientProbe(
                all_users, limit=args.tss_statistical_lookback)
            logger.debug_("TSS Probability", probability.tolist())
            banned_indices = np.where(
                probability >= 1/len(all_users)*args.tss_statistical_penalty)[0]
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% tss + kmeans
        elif server_defender.enabled and (iter_ % args.tss_statistical_reject_kmeans == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())

            top_k_indices = server_defender.defense_TSS_kmeans(pair_wise_cos_mat, layer_names=tuple(
                layer_list[top_k_layers_index] if args.tss_layer_list == [] else args.tss_layer_list), nclasses=2)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("TSS top_k_indices", top_k_indices)
            Defender.historical_excluded_list.append(indicator)
            probability = server_defender.banClientProbe(
                all_users, limit=args.tss_statistical_lookback)
            logger.debug_("TSS Probability", probability.tolist())
            banned_indices = np.where(
                probability >= 1/len(all_users)*args.tss_statistical_penalty)[0]
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% tss layerwise
        elif server_defender.enabled and (iter_ % args.tss_statistical_layerwise == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug))
            top_k_layers_index = server_defender.extract_top_k_matrices(
                pair_wise_cos_mat, layer_importance_score, k=args.tss_layer_count)
            layer_list = np.array(list(pair_wise_cos_mat.keys()))
            logger.debug_("Selected layers Statistical",
                          layer_list[top_k_layers_index].reshape(-1).tolist())

            top_k_indices = server_defender.defense_TSSGetTopK_layerwise(
                pair_wise_cos_mat, layer_names=(), top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("TSS indicator labels", indicator)
            Defender.historical_excluded_list.append(indicator)
            probability = server_defender.banClientProbe(all_users)
            logger.debug_("TSS Probability", probability.tolist())
            banned_indices = np.where(probability >= 1/len(all_users)*5)[0]
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% tss ban
        elif server_defender.enabled and (iter_ % args.tss_statistical_ban == 0):
            mask_dict = None
            if args.tss_metamask:
                mask_dict, portion_of_ones = server_defender.analyseImportantNeurons(
                    last_global_dict, [w_local_i[1] for w_local_i in w_local_list], percentile=args.tss_important_percentile)

            layer_importance_score = {x[0].replace('.layer.weight', ''): x[1] for x in server_defender.analyseImportantLayers(
                last_global_dict, [w_local_i[1] for w_local_i in w_local_list])}
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac, meta_mask=mask_dict,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, plot=(args.tss_plot and args.debug))
            cluster_labels, silhouette_avg = server_defender.clusterByDistance(
                pair_wise_cos_mat)
            logger("Cluster labels", cluster_labels)
            Defender.historical_excluded_list.append(cluster_labels)
            banned_indices = server_defender.banClient(all_users)
            logger("Banned indices", banned_indices)
            w_glob = getWglob(w_local_list, banned_indices)
            net_glob.load_state_dict(w_glob)

        # %% soft reject
        elif server_defender.enabled and (iter_ % args.tss_soft_reject == 0):
            pair_wise_cos_mat = server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], iter_=iter_, frac=args.tss_eval_frac,
                                                                  threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, plot=(args.tss_plot and args.debug))
            tss_weight = server_defender.defense_TSSLayerMinMax(
                pair_wise_cos_mat)
            logger("Soft weight", tss_weight)
            w_glob = getWglobTSSWeight(w_local_list, tss_weight)
            net_glob.load_state_dict(w_glob)

        # %% gss
        elif server_defender.enabled and (iter_ % args.gss == 0):
            pattern = attacker_default.getPattern(0)
            result = server_defender.defenseGSSCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], pattern, args.pos_choice, iter_=iter_,
                                                       perform_normalize=args.tss_norm, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug), 
                                                       img_channel=args.num_channels, img_size=args.input_size, num_class=args.num_classes, attacking_label=args.label,
                                                       layer_names=args.gss_layer_list,
                                                       p=args.gss_threshold,
                                                       device=args.device)
            logger.debug_("GSS top_k_indices", result)
            logger("Banned indices", result)
            w_glob = getWglob(w_local_list, result)
            net_glob.load_state_dict(w_glob)

        # %% mae
        elif server_defender.enabled and (iter_ % args.mae == 0) and iter_ in range(args.robust_range[0], args.robust_range[1]):
            # net_glob = defense_mae(net_glob, base_dir, iter_, collect_rounds=10, mae_train_epoch=args.mae_train_epoch, script_path=args.script_path)
            w_glob = getWglob(w_local_list)
            net_mae = copy.deepcopy(net_glob)
            net_mae.load_state_dict(w_glob)
            net_glob.load_state_dict(
                server_defender.defense_mae(net_mae, base_dir, iter_))
            logger.info("mae recovered")

        else:
            logger("default fedavg")
            w_glob = getWglob(w_local_list)
            net_glob.load_state_dict(w_glob)

        # copy weight to net_glob
        # last_global_dict = copy.deepcopy(net_glob)
        last_global_dict = net_glob.state_dict()
        # net_glob.load_state_dict(w_glob)

        # %% analysis
        if args.cl and iter_ % args.cl == 0:
            for idx, w_local, _ in w_local_list:
                # net_cl = type(net_glob)()
                net_cl = copy.deepcopy(net_glob)
                net_cl.load_state_dict(w_local)
                if idx in attack_training_clients:
                    CL(net_cl, 'attack'+str(iter_))
                elif idx in normal_training_clients:
                    CL(net_cl, 'normal'+str(iter_))
                del net_cl
            CL(net_glob, 'global'+str(iter_))

        if args.tss_cosine and iter_ % args.tss_cosine == 0:
            server_defender.defenseTssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list],
                                              threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, iter_=iter_, frac=0.5)

        if (args.trigger_inversion and iter_ % args.trigger_inversion == 0):
            param = {
                "dataset": args.dataset,
                "Epochs": 100,
                "batch_size": 64,
                "lamda": 0.01,
                "num_classes": args.num_classes,
                "image_size": (args.input_size, args.input_size),
                "CWH": (args.num_channels, args.input_size, args.input_size),
                "target_label": [args.label],
            }
            net_el = copy.deepcopy(net_glob)
            trigger_list, mask_list = reverse_engineer(
                dataset=evaluator.dataset_test, model=net_el, param=param, device=args.device)

            if (args.inversion_importance):
                mask, trigger = torch.tensor(mask_list[0]).to(
                    'cuda'), torch.tensor(trigger_list[0]).to('cuda')
                importance_alignment_tp, importance_alignment_tn, importance_alignment_fp = realTriggerImportance(
                    net_el, w_local_list, evaluator.dataset_test, attack_clients[0], mask, trigger, args.label, args.pos_choice, args.num_classes, args.num_channels, args.input_size, device=args.device)
                # print("importance_alignment", importance_alignment)
                for layer_name, imp in importance_alignment_tp.items():
                    logger.info(
                        f"Trigger importance alignment tp {layer_name}: {imp.sum()/imp.numel():.4f}")
                for layer_name, imp in importance_alignment_tn.items():
                    logger.info(
                        f"Trigger importance alignment tn {layer_name}: {imp.sum()/imp.numel():.4f}")
                for layer_name, imp in importance_alignment_fp.items():
                    logger.info(
                        f"Trigger importance alignment fp {layer_name}: {imp.sum()/imp.numel():.4f}")

                for layer_name, imp in importance_alignment_tp.items():
                    logger.info(
                        f"Trigger importance alignment sensitivity (tp / (tp + fn)) {layer_name}: {imp.sum()/ (imp.sum() + (1-importance_alignment_tn[layer_name] - importance_alignment_tp[layer_name] - importance_alignment_fp[layer_name]).sum() + 1e-6):.4f}")
                    logger.info(
                        f"Trigger importance alignment precision (tp / (tp + fp)) {layer_name}: {imp.sum()/ (imp.sum() + importance_alignment_fp[layer_name].sum() + 1e-6):.4f}")

            del net_el

        if args.tri_importance and iter_ % args.tri_importance == 0:
            net_el = copy.deepcopy(net_glob)
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, attack_clients[0])

            assessor.updateTriggerDataset(evaluator.dataset_test, mask, trigger, args.blend_alpha, args.label,
                                          args.pos_choice, args.num_classes, args.num_channels, args.input_size, args.attack_type)
            impt_dict = assessor.triImportance(
                net_el, w_local_list, last_global_dict,
                trigger_model_idx=attack_clients[0],
                # trigger_model_idx=normal_clients[0],
                p=args.tss_threshold,
                device=args.device,
                visual_path=os.path.join(args.base_dir, 'visual'),
                assessor_path=os.path.join(
                    args.base_dir, 'assessor') if args.save_assessor else args.assessor_path,
                assessing_layer=args.assessing_layer,
                save=args.save_assessor
            )

            # Extract results from dictionary
            trigger_gt, importance_alignment_tp_nt, importance_alignment_tn_nt, importance_alignment_fp_nt = impt_dict[
                'noise']
            impt_eval, importance_alignment_ne_tp, importance_alignment_ne_tn, importance_alignment_ne_fp = impt_dict[
                'ne']
            impt_eval, importance_alignment_tp, importance_alignment_tn, importance_alignment_fp = impt_dict[
                'eval']
            impt_eval, importance_intersection_tp, importance_intersection_tn, importance_intersection_fp = impt_dict[
                'intersection']
            impt_eval, importance_union_tp, importance_union_tn, importance_union_fp = impt_dict[
                'union']

            def calculate_and_log_metrics(tp_dict, tn_dict, fp_dict, reference_dict, title, logger):
                """Calculate and log alignment metrics for a set of importance scores."""
                logger.info(f"=== {title} ===")

                for layer_name in tp_dict.keys():
                    tp, tn, fp = tp_dict[layer_name], tn_dict[layer_name], fp_dict[layer_name]
                    ref = reference_dict[layer_name]

                    # Calculate element counts
                    pos_elements = (ref > 0).sum()
                    neg_elements = (ref <= 0).sum()

                    # Calculate basic metrics
                    tp_sum, tn_sum, fp_sum = tp.sum(), tn.sum(), fp.sum()
                    fn = (1 - tn - tp - fp).sum()

                    # Calculate rates
                    tp_rate = tp_sum / pos_elements if pos_elements > 0 else 0
                    tn_rate = tn_sum / neg_elements if neg_elements > 0 else 0
                    fp_rate = fp_sum / pos_elements if pos_elements > 0 else 0

                    # Calculate metrics with epsilon for numerical stability
                    eps = 1e-6
                    sensitivity = tp_sum / (tp_sum + fn + eps)
                    precision = tp_sum / (tp_sum + fp_sum + eps)
                    contamination_rate = fp_sum / (tp_sum + fp_sum + eps)

                    # Log results
                    logger.info(
                        f"{title} importance alignment tp {layer_name}: {tp_rate:.4f} with {pos_elements} elements")
                    logger.info(
                        f"{title} importance alignment tn {layer_name}: {tn_rate:.4f} with {neg_elements} elements")
                    logger.info(
                        f"{title} importance alignment fp {layer_name}: {fp_rate:.4f} with {pos_elements} elements")
                    logger.info(
                        f"{title} importance alignment sensitivity (tp / (tp + fn)) {layer_name}: {sensitivity:.4f}")
                    logger.info(
                        f"{title} importance alignment precision (tp / (tp + fp)) {layer_name}: {precision:.4f}")
                    logger.info(
                        f"{title.split()[0]} contamination rate {layer_name}: {contamination_rate:.4f}")

            # Log all comparisons using the helper function
            comparisons = [
                (importance_alignment_tp_nt, importance_alignment_tn_nt,
                 importance_alignment_fp_nt, trigger_gt, "Noise-Trigger Comparison"),
                (importance_alignment_tp, importance_alignment_tn, 
                 importance_alignment_fp, trigger_gt, "Eval-Trigger Comparison"),
                (importance_alignment_ne_tp, importance_alignment_ne_tn,
                 importance_alignment_ne_fp, impt_eval, "Noise-Eval Comparison"),
                (importance_intersection_tp, importance_intersection_tn,
                 importance_intersection_fp, impt_eval, "Noise-Eval Intersection"),
                (importance_union_tp, importance_union_tn,
                 importance_union_fp, impt_eval, "Noise-Eval Union")
            ]

            for tp_dict, tn_dict, fp_dict, ref_dict, title in comparisons:
                calculate_and_log_metrics(
                    tp_dict, tn_dict, fp_dict, ref_dict, title, logger)

            del net_el

        if (args.trigger_tss_eval and iter_ % args.trigger_tss_eval == 0):
            pattern = attacker_default.getPattern(0)
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)

            filtered_elements = list(
                filter(lambda x: x[0] == attack_clients[0], w_local_list))
            net_el = copy.deepcopy(net_glob)
            net_el.load_state_dict(filtered_elements[0][1])
            acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test_void(
                # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
                net_el, last_global_dict=last_global_dict, attacker=attacker_default if len(attack_clients) == 0 else list(attacker_dict.values())[0], idx=0)

            # logger(f'trigger_tss_eval Round {iter_:3d}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')

            acc = evaluator.triggerEval(net_el, args.label, args.pos_choice,
                                        trigger, args.num_classes, args.num_channels, args.input_size, mask)
            logger(f'trigger_tss_eval: acc: {acc}')

            # pair_wise_cos_mat = server_defender.eval_tssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], pattern, args.pos_choice, iter_=iter_, meta_mask=None,
            #                                                       threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug), img_channel=args.num_channels, img_size=args.input_size, num_class=args.num_classes, attacking_label=args.label)

            del net_el

        if (args.trigger_tss_eval_v2 and iter_ % args.trigger_tss_eval_v2 == 0):
            pattern = attacker_default.getPattern(0)
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)

            filtered_elements = list(
                filter(lambda x: x[0] == attack_clients[0], w_local_list))
            net_el = copy.deepcopy(net_glob)
            net_el.load_state_dict(filtered_elements[0][1])
            acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test_void(
                # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
                net_el, last_global_dict=last_global_dict, attacker=attacker_default if len(attack_clients) == 0 else list(attacker_dict.values())[0], idx=0)

            # logger(f'trigger_tss_eval Round {iter_:3d}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')

            acc = evaluator.triggerEval(net_el, args.label, args.pos_choice,
                                        trigger, args.num_classes, args.num_channels, args.input_size, mask)
            logger(f'trigger_tss_eval_v2: acc: {acc}')

            pair_wise_cos_mat = server_defender.eval_tssCompare_v2(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], pattern, args.pos_choice, iter_=iter_, meta_mask=None,
                                                                   threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug), img_channel=args.num_channels, img_size=args.input_size, num_class=args.num_classes, attacking_label=args.label)

            top_k_indices = server_defender.defense_TSSGetTopK(
                pair_wise_cos_mat, layer_names=args.tss_layer_list, top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("Trigger_tss_eval_v2 top_k_indices", top_k_indices)
            logger.debug_("Trigger_tss_eval_v2 indicator", indicator)

            del net_el

        if (args.trigger_gss_eval and iter_ % args.trigger_gss_eval == 0):
            pattern = attacker_default.getPattern(0)
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)

            filtered_elements = list(
                filter(lambda x: x[0] == attack_clients[0], w_local_list))
            net_el = copy.deepcopy(net_glob)
            net_el.load_state_dict(filtered_elements[0][1])
            acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test_void(
                # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
                net_el, last_global_dict=last_global_dict, attacker=attacker_default if len(attack_clients) == 0 else list(attacker_dict.values())[0], idx=0)

            # logger(f'trigger_tss_eval Round {iter_:3d}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')

            acc = evaluator.triggerEval(net_el, args.label, args.pos_choice,
                                        trigger, args.num_classes, args.num_channels, args.input_size, mask)
            logger(f'trigger_tss_eval_v2: acc: {acc}')

            pair_wise_cos_mat = server_defender.eval_gssCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], pattern, args.pos_choice, iter_=iter_, meta_mask=None,
                                                                threshold=args.tss_threshold, threshold_common=args.tss_threshold_common, perform_normalize=args.tss_norm, synthesize=args.tss_synthesize, bn_only=args.tss_bn_only, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug), img_channel=args.num_channels, img_size=args.input_size, num_class=args.num_classes, attacking_label=args.label)

            top_k_indices = server_defender.defense_TSSGetTopK(
                pair_wise_cos_mat, layer_names=args.tss_layer_list, top_k=args.tss_hard_k)
            indicator = np.zeros(len(all_users), dtype=int)
            indicator[top_k_indices] = 1
            logger.debug_("Trigger_tss_eval_v2 top_k_indices", top_k_indices)
            logger.debug_("Trigger_tss_eval_v2 indicator", indicator)

            del net_el

        if (args.trigger_gss_eval_v2 and iter_ % args.trigger_gss_eval_v2 == 0):
            pattern = attacker_default.getPattern(0)
            mask, trigger = attacker_default.getTrigger(
                evaluator.dataset_test, 0)

            filtered_elements = list(
                filter(lambda x: x[0] == attack_clients[0], w_local_list))
            net_el = copy.deepcopy(net_glob)
            net_el.load_state_dict(filtered_elements[0][1])
            acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test_void(
                # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
                net_el, last_global_dict=last_global_dict, attacker=attacker_default if len(attack_clients) == 0 else list(attacker_dict.values())[0], idx=0)

            # logger(f'trigger_tss_eval Round {iter_:3d}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')

            acc = evaluator.triggerEval(net_el, args.label, args.pos_choice,
                                        trigger, args.num_classes, args.num_channels, args.input_size, mask)
            logger(f'trigger_tss_eval_v2: acc: {acc}')

            result = server_defender.defenseGSSCompare(net_glob, [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list], pattern, args.pos_choice, iter_=iter_,
                                                       perform_normalize=args.tss_norm, sorted_mat=args.tss_sort, plot=(args.tss_plot and args.debug), 
                                                       img_channel=args.num_channels, img_size=args.input_size, num_class=args.num_classes, attacking_label=args.label,
                                                       layer_names=args.gss_layer_list,
                                                       device=args.device)

            # top_k_indices = server_defender.defense_TSSGetTopK(pair_wise_cos_mat, layer_names=args.tss_layer_list, top_k=args.tss_hard_k)
            # indicator = np.zeros(len(all_users), dtype=int)
            # indicator[top_k_indices] = 1
            # logger.debug_("Trigger_tss_eval_v2 top_k_indices", top_k_indices)
            # logger.debug_("Trigger_tss_eval_v2 indicator", indicator)

            del net_el

        # %% evaluate
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        if (iter_) % args.test_freq == 0:
            net_glob_eval = copy.deepcopy(net_glob)
            net_glob_eval.eval()
            # acc_test, loss_test, correct_prediction, attack_prediction = test_img_attack_eval(
            #     net_glob, dataset_test, args, attacker=attacker)
            acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test(
                # net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default, idx=0)
                net_glob_eval, last_global_dict=last_global_dict, attacker=attacker_default if len(attack_clients) == 0 else list(attacker_dict.values())[0], idx=0)

            logger(f'Round {iter_:3d}, Average loss {loss_avg:.3f}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')

            if acc_test > best_acc:
                best_acc = acc_test
                best_epoch = iter_
                best_save_path = os.path.join(
                    base_dir, 'fed', f'attack_portion{attack_portion}_best.pt'
                )
                torch.save(net_glob_eval.state_dict(), best_save_path)
                logger(f'Best updated, iter_: {iter_}, acc: {best_acc}')

            current_results = np.array([iter_, loss_avg, loss_test, acc_test,
                                       best_acc, correct_prediction, attack_prediction]).reshape(1, -1)
            current_results_pd = pd.DataFrame(current_results, columns=[
                'epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc', 'correct_prediction', 'attack_prediction'])
            current_results_pd.to_csv(
                results_save_path, mode='a', index=False, header=False)
            stats = f'TestLs {loss_test:.3f}, TestAcc: {acc_test:.2f}, ASR: {attack_prediction:.2f}'
            pbar.set_postfix_str(stats)
            del net_glob_eval

        if (iter_) % args.global_saving_interval == 0 and iter_ >= args.global_saving_start:
            model_save_path = os.path.join(
                base_dir, 'fed', f'attack_portion{attack_portion}_model_{iter_}.pt')
            torch.save(net_glob.state_dict(), model_save_path)
            # TODO 2025-03-07 git.V.87012: save after defense

        if (args.eval_local and iter_ % args.eval_local == 0):
            net_el = copy.deepcopy(net_glob)
            for idx, w_local, _ in w_local_list:
                net_el.load_state_dict(w_local)
                net_el.eval()
                acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test(
                    # net_el, last_global_dict=last_global_dict, attacker=attacker_default, idx=idx)
                    net_el, last_global_dict=last_global_dict, attacker=attacker_default if (idx not in attack_clients.tolist()) else attacker_dict[idx], idx=idx)
                logger(f'Round {iter_:3d}, Local {idx}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')
            del net_el

        elif (iter_ % 5 == 0 and args.debug):
            # quick test with all attackers and <5 normal clients
            # random pick 5 or max_normal_clients idx from normal_clients
            idx_eval = np.random.choice(normal_clients, min(
                len(normal_clients), 5), replace=False)
            idx_eval = np.concatenate((idx_eval, attack_clients))
            net_el = copy.deepcopy(net_glob)
            for idx, w_local, _ in w_local_list:
                if idx not in idx_eval:
                    continue
                net_el.load_state_dict(w_local)
                net_el.eval()
                acc_test, loss_test, correct_prediction, attack_prediction = evaluator.test(
                    # net_el, last_global_dict=last_global_dict, attacker=attacker_default, idx=idx)
                    net_el, last_global_dict=last_global_dict, attacker=attacker_default if (idx not in attack_clients.tolist())else attacker_dict[idx], idx=idx)
                logger(f'Round {iter_:3d}, Local {idx}, Test loss {loss_test:.3f}, Test accuracy: {acc_test:.2f}, Backdoor base acc: {correct_prediction:.2f}, Backdoor target acc: {attack_prediction:.2f}')
            del net_el

        w_local_list.clear()
        if args.parallel:
            task_slices.clear()
            del tasks
        gc.collect()
        torch.cuda.empty_cache()
        summary = torch.cuda.memory_summary(device=args.device)
        logger.debug_(re.search(r"^\| Allocated memory\s+\|.*$",
                      summary, re.MULTILINE).group(0))  # type: ignore

    logger(f'Best model at round: {best_epoch}, acc: {best_acc:.2f}')
    model_save_path = os.path.join(
        base_dir, 'fed', f'attack_portion{attack_portion}_model_{args.epochs}.pt')
    torch.save(net_glob.state_dict(), model_save_path)

    logger.print(base_dir)
