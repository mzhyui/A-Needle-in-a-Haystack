import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn

from models.Nets import LeNet, lenetsm, vgg16, vgg16sm, resnet20, resnet20sm, vit_tiny_cifar10, vitsm
from models.smFunction import set_compute_mask_impt, set_ft_task
from models.Update import LocalUpdater

from utils.clip import clipping
from utils.krum import KrumDefense


def configure_softmask_model(model):
    set_compute_mask_impt(model, False)
    set_ft_task(model, 0)


def getModel(
    model_name: str,
    dataset: str,
    num_channels: int = 3,
    num_classes: int = 10,
    input_size=32,
    device: str | torch.device = 'cuda',
    device_list=None,
    rand_init: int = 0,
    option: str = 'B',
):
    if model_name == 'vgg' and dataset == 'cifar10':
        net_glob = vgg16(num_channels, num_classes).to(device)
    elif model_name == 'vgg' and dataset == 'GTSRB':
        # net_glob = VGG16softmasking(num_channels, num_classes, input_size=input_size).to(device)
        net_glob = vgg16(num_channels, num_classes,
                           input_size=input_size).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'vggsm' and dataset == 'cifar10':
        # net_glob = VGG16softmasking(num_channels, num_classes).to(device)
        net_glob = vgg16sm(num_channels, num_classes,
                           input_size=input_size).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'vggsm' and dataset == 'GTSRB':
        # net_glob = VGG16softmasking(num_channels, num_classes, input_size=input_size).to(device)
        net_glob = vgg16sm(num_channels, num_classes,
                           input_size=input_size).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'vggsm' and dataset == 'fmnist':
        # net_glob = VGG16softmasking(num_channels, num_classes, input_size=input_size).to(device)
        net_glob = vgg16sm(num_channels, num_classes,
                           input_size=input_size).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'lenetsm' and dataset == 'cifar10':
        net_glob = lenetsm(num_channels, num_classes).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'lenetsm' and dataset == 'GTSRB':
        net_glob = lenetsm(num_channels, num_classes,
                           input_size=input_size).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'lenetsm' and dataset == 'mnist':
        net_glob = lenetsm(
            num_channels, num_classes, input_size=28).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'lenetsm' and dataset == 'fmnist':
        net_glob = lenetsm(
            num_channels, num_classes, input_size=28).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'lenet' and dataset == 'mnist':
        net_glob = LeNet(num_channels=num_channels, input_size=28).to(device)

    elif model_name == 'resnet20' and dataset == 'cifar10':
        net_glob = resnet20(
            num_channels, num_classes).to(device)
    elif model_name == 'resnet20' and dataset == 'GTSRB':
        net_glob = resnet20(
            num_channels, num_classes).to(device)

    elif model_name == 'resnet20sm' and dataset == 'cifar10':
        net_glob = resnet20sm(
            num_channels, num_classes).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'resnet20sm' and dataset == 'cifar100':
        net_glob = resnet20sm(
            num_channels, num_classes).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'resnet20sm' and dataset == 'GTSRB':
        net_glob = resnet20sm(
            num_channels, num_classes).to(device)
        configure_softmask_model(net_glob)

    elif model_name == 'vit' and dataset == 'cifar10':
        net_glob = vit_tiny_cifar10(
            num_channels, num_classes, input_size=input_size).to(device)
    elif model_name == 'vitsm' and dataset == 'cifar10':
        net_glob = vitsm(num_channels, num_classes,
                         input_size=input_size).to(device)
        configure_softmask_model(net_glob)
    elif model_name == 'vitsm' and dataset == 'cifar100':
        net_glob = vitsm(num_channels, num_classes,
                         input_size=input_size).to(device)
        configure_softmask_model(net_glob)

    # elif model_name == 'resnet20' and dataset == 'cifar10':
    #     net_glob = resnet20(option=option, rand_init=rand_init).to(device)
    # elif model_name == 'lenet' and dataset == 'mnist':
    #     net_glob = lenet().to(device)
    # elif model_name == 'lenetMini' and dataset == 'mnist':
    #     net_glob = lenetMini().to(device)
    # elif model_name == 'cnn' and dataset == 'mnist':
    #     net_glob = CNNMnist(num_channels, num_classes).to(device)
    # elif model_name == 'mlp' and dataset == 'mnist':
    #     net_glob = MLP(dim_in=784, dim_hidden=256,
    #                    dim_out=num_classes).to(device)
    # elif model_name == 'lenet' and dataset == 'fmnist':
    #     net_glob = lenet().to(device)
    # elif model_name == 'stn' and dataset == 'GTSRB':
    #     net_glob = STNet(num_channels, num_classes).to(device)
    # # elif model_name == 'resnet20' and dataset == 'GTSRB':
    # #     net_glob = resnet20(option=option, rand_init=rand_init).to(device)
    else:
        raise ValueError('Error: unrecognized model')

    if device_list:
        return nn.DataParallel(net_glob, device_list)
    return net_glob


def loadModel(model_name: str, dataset: str, num_channels: int, num_classes: int,
              pre_trained: str = '',
              rand_init: int = 0,
              device: str = 'cuda',
              option='B',
              ):
    model = getModel(model_name=model_name, dataset=dataset, num_classes=num_classes,
                     num_channels=num_channels, device=device, rand_init=rand_init, option=option)
    status_loaded = model.load_state_dict(torch.load(
        pre_trained, weights_only=True)) if pre_trained else None

    return model


def compare_weights(w1, w2):
    if w1.keys() != w2.keys():
        return False
    for k in w1.keys():
        if not torch.equal(w1[k], w2[k]):
            return False
    return True


def get_local_training_params(args):
    return {
        'data_augmentation_local': args.data_augmentation_local,
        'model': args.model,
        'dataset': args.dataset,
        'num_classes': args.num_classes,
        'data_portion': args.data_portion,
        'max_workers': args.max_workers,
        'local_bs': args.local_bs,
        'local_ep': args.local_ep,
        'local_ep_times': args.local_ep_times,
        'input_size': args.input_size,
        'blend_alpha': args.blend_alpha,
        'local_ep_pretrain': args.local_ep_pretrain,
        'attack_type': args.attack_type,
        'pattern_choice': args.pattern_choice,
        'pos_choice': args.pos_choice,
        'label': args.label,
        'device': args.device,
    }


def scale_model_update(local_weights, reference_model, round_clients, device):
    client_count = len(round_clients)
    reference_weights = reference_model.to(device).state_dict()
    for layer_name in local_weights:
        local_weights[layer_name] = (
            client_count * local_weights[layer_name]
            - (client_count - 1) * reference_weights[layer_name]
        )
    return local_weights


def clear_training_cache():
    gc.collect()
    torch.cuda.empty_cache()


def getWglob(w_glob_list: list, exclude=None):
    assert w_glob_list
    excluded_clients = set() if exclude is None else set(exclude)
    w_glob_list = [
        (client_id, local_weights, client_weight)
        for client_id, local_weights, client_weight in w_glob_list
        if client_id not in excluded_clients
    ]
    if not w_glob_list:
        raise ValueError("No client updates remain after exclusions.")
    
    # If only one client remains, return its weights directly
    if len(w_glob_list) == 1:
        return copy.deepcopy(w_glob_list[0][1])
    
    # Initialize aggregated weights with zeros (same structure as first client's weights)
    aggregated_weights = copy.deepcopy(w_glob_list[0][1])
    for layer_name in aggregated_weights:
        aggregated_weights[layer_name] = torch.zeros_like(aggregated_weights[layer_name])
    
    # Initialize total weight
    total_weight = 0
    
    # Aggregate weights from all clients
    for _, local_weights, client_weight in w_glob_list:
        # Skip clients with small weights (should be a numeric comparison)
        if isinstance(client_weight, (int, float)) and client_weight < 10:
            continue
        total_weight += client_weight
        for layer_name in aggregated_weights:
            aggregated_weights[layer_name] += local_weights[layer_name] * client_weight
    
    # Normalize by total weight
    if total_weight > 0:  # Avoid division by zero
        for layer_name in aggregated_weights:
            aggregated_weights[layer_name] = torch.div(
                aggregated_weights[layer_name], total_weight
            )

    return aggregated_weights


def getWglobTSSWeight(w_glob_list: list, tss_weight):

    layer_names = tss_weight.keys()
    # TODO 2025-03-13 git.V.c0dfb: do not use weight on conv1
    # tss_weight['conv1'] = [1 for _ in tss_weight['conv1']]

    w = copy.deepcopy(w_glob_list[0][1])
    if (len(w_glob_list) == 1):
        return w

    user_idx = w_glob_list[0][0]
    for k in w.keys():
        # print(w_glob_list[0][2] * (tss_weight[k.split('.')[0]][user_idx] if k.split('.')[0] in layer_names else 1))
        # print(tss_weight[k.split('.')[0]][user_idx] if k.split('.')[0] in layer_names else 1)
        w[k] = w[k] * int(w_glob_list[0][2] * (tss_weight[k.split('.')[0]]
                          [user_idx] if k.split('.')[0] in layer_names else 1))

    accumulated_weight = {k: int(w_glob_list[0][2] * (tss_weight[k.split(
        '.')[0]][user_idx] if k.split('.')[0] in layer_names else 1)) for k in w.keys()}

    for idx, w_local, idxs_weight in w_glob_list[1:]:
        if idxs_weight < 10:
            continue
        # w += w_local * idxs_weight
        # user_weight += base_weight * tss_weight[k.strip('.')[0]][idx]
        for k in w.keys():
            w[k] += w_local[k] * int(idxs_weight * (tss_weight[k.split('.')[0]]
                                     [user_idx] if k.split('.')[0] in layer_names else 1))
            accumulated_weight[k] += int(idxs_weight * (tss_weight[k.split(
                '.')[0]][user_idx] if k.split('.')[0] in layer_names else 1))

    for k in w.keys():
        w[k] = torch.div(w[k], accumulated_weight[k])

    return w


def getWglobKrum(w_glob_list: list, krumClients=70, mclients=3):
    kd = KrumDefense(mclients, krumClients)
    clients = []
    for idx, w_local, idxs_weight in w_glob_list:
        clients.append(tuple([idxs_weight, w_local]))
    clients = kd.defend_before_aggregation(clients)

    print(len(clients))

    w = clients[0][1]
    user_weight = clients[0][0]
    for k in w.keys():
        w[k] *= w_glob_list[0][0]

    for idxs_weight, w_local in clients[1:]:
        # w += w_local * idxs_weight
        user_weight += idxs_weight
        for k in w.keys():
            w[k] += w_local[k] * idxs_weight

    for k in w.keys():
        w[k] = torch.div(w[k], user_weight)

    return w


# def parallelTrainingNormal(tasks):
#     results = []
#     for task in tasks:
#         local_updater = LocalUpdater()
#         iter_, idx, args, server_defender, idxs_weight_dict, net_glob, dataset_train, dict_users_train, lr, with_local_save, base_dir, normal_save_candidates = task
#         local_updater.update_params(
#             args=args, dataset=dataset_train, idxs=dict_users_train[idx])
#         net_local = copy.deepcopy(net_glob)
#         if args.fedsam:
#             w_local, loss = local_updater.train_sam(
#                 net=net_local.to(args.device), lr=lr)
#         else:
#             w_local, loss = local_updater.train(
#                 net=net_local.to(args.device), lr=lr)

#         if args.clipping:
#             w_local = clipping(w_local, net_local)

#         if (iter_) % args.local_saving_interval == 0 and idx in normal_save_candidates and iter_ >= args.local_saving_start and with_local_save:
#             torch.save(w_local, os.path.join(
#                 base_dir, 'local_normal_save', 'iter_{}_normal_{}.pt'.format(iter_, idx)))

#         if args.subnet and (iter_) % args.subnet == 0 and iter_ >= args.subnet:
#             net_tss = server_defender.defense_subnetmasking(net_local, w_local)
#             np.save(os.path.join(base_dir, 'visual',
#                     'iter_{}_normal_{}.npy'.format(iter_, idx)), net_tss)

#         results.append((idx, w_local, idxs_weight_dict[idx], loss))

#         del net_local
#         del local_updater
#         gc.collect()
#         torch.cuda.empty_cache()
#     return results


def train_user_normal(
    iter_, idx, args, server_defender, idxs_weight_dict, net_glob,
    dataset_train, dict_users_train, lr, with_local_save, base_dir,
    normal_save_candidates,
):
    local_updater = LocalUpdater()
    local_updater.update_params(
        dataset=dataset_train,
        idxs=dict_users_train[idx],
        params=get_local_training_params(args),
    )
    net_local = copy.deepcopy(net_glob)
    if args.fedsam:
        w_local, loss = local_updater.train_sam(
            net=net_local.to(args.device), lr=lr)
    else:
        w_local, loss = local_updater.train(
            net=net_local.to(args.device), lr=lr)

    if args.clipping:
        w_local = clipping(w_local, net_local)

    should_save = (
        iter_ % args.local_saving_interval == 0
        and idx in normal_save_candidates
        and iter_ >= args.local_saving_start
        and with_local_save
    )
    if should_save:
        torch.save(
            w_local,
            os.path.join(base_dir, 'local_normal_save', f'iter_{iter_}_normal_{idx}.pt'),
        )

    if args.subnet and iter_ % args.subnet == 0 and iter_ >= args.subnet:
        net_tss = server_defender.defense_subnetmasking(net_local, w_local)
        torch.save(
            os.path.join(base_dir, 'visual', f'iter_{iter_}_normal_{idx}.pth'),
            net_tss,
        )

    del net_local
    del local_updater
    clear_training_cache()

    return idx, w_local, idxs_weight_dict[idx], loss


# def parallelTrainingAttack(tasks: list):
#     # iter_, idx, args, server_defender, current_round_users_indices, idxs_weight_dict, net_glob, dataset_train, dict_users_train, lr, with_local_save, base_dir = task
#     # return train_user_attack(iter_, idx, args, server_defender, current_round_users_indices, idxs_weight_dict, net_glob, dataset_train, dict_users_train, lr, with_local_save, base_dir)

#     results = []
#     for task in tasks:
#         local_updater = LocalUpdater()
#         iter_, idx, args, attacker, server_defender, current_round_users_indices, idxs_weight_dict, net_glob, last_global_dict, dataset_train, dict_users_train, lr, with_local_save, base_dir = task
#         local_updater.update_params(
#             args=args, dataset=dataset_train, idxs=dict_users_train[idx])
#         net_local = copy.deepcopy(net_glob)
#         if args.attack_on_attack != []:
#             if idx in args.attack_on_attack:
#                 w_local, loss = local_updater.train_attack_dynamic(
#                     model_local=net_local.to(args.device), attacker=attacker, last_global_dict=last_global_dict, lr=lr, idx=idx)
#             else:
#                 w_local, loss = local_updater.train(
#                     net=net_local.to(args.device), lr=lr)
#         elif args.attack_type != "peace" and iter_ >= args.start_attack:
#             w_local, loss = local_updater.train_attack_dynamic(
#                 model_local=net_local.to(args.device), attacker=attacker, last_global_dict=last_global_dict, lr=lr, idx=idx)
#         else:
#             w_local, loss = local_updater.train(
#                 net=net_local.to(args.device), lr=lr)

#         if args.clipping:
#             w_local = clipping(w_local, net_local)

#         if args.scale:
#             for k in w_local.keys():
#                 w_local[k] = len(current_round_users_indices)*w_local[k] - (
#                     len(current_round_users_indices)-1)*net_local.to(args.device).state_dict()[k]

#         if (iter_) % args.local_saving_interval == 0 and iter_ >= args.local_saving_start and with_local_save:
#             torch.save(w_local, os.path.join(
#                 base_dir, 'local_attack_save', 'iter_{}_attack_{}.pt'.format(iter_, idx)))

#         if args.subnet and (iter_) % args.subnet == 0 and iter_ >= args.subnet:
#             net_tss = server_defender.defense_subnetmasking(net_local, w_local)
#             np.save(os.path.join(base_dir, 'visual',
#                     'iter_{}_attack_{}.npy'.format(iter_, idx)), net_tss)

#         if not args.no_attack_on_attack:
#             results.append((idx, w_local, idxs_weight_dict[idx], loss))
#         else:
#             results.append((idx, net_glob.state_dict(),
#                            idxs_weight_dict[idx], loss))

#         del net_local
#         del local_updater
#         gc.collect()
#         torch.cuda.empty_cache()
#     return results


def train_user_attack(
    iter_, idx, args, attacker, server_defender, current_round_users_indices,
    idxs_weight_dict, net_glob, last_global_dict, dataset_train,
    dict_users_train, lr, with_local_save, base_dir,
):
    local_updater = LocalUpdater()
    local_updater.update_params(
        dataset=dataset_train,
        idxs=dict_users_train[idx],
        params=get_local_training_params(args),
    )
    net_local = copy.deepcopy(net_glob)
    if args.attack_type != "peace" and iter_ >= args.start_attack:
        w_local, loss = local_updater.train_attack_dynamic(
            model_local=net_local.to(args.device), attacker=attacker, last_global_dict=last_global_dict, lr=lr, idx=idx)
    else:
        w_local, loss = local_updater.train(
            net=net_local.to(args.device), lr=lr)

    if args.clipping:
        w_local = clipping(w_local, net_local)

    if args.scale:
        w_local = scale_model_update(
            w_local, net_local, current_round_users_indices, args.device
        )

    should_save = (
        iter_ % args.local_saving_interval == 0
        and iter_ >= args.local_saving_start
        and with_local_save
    )
    if should_save:
        torch.save(
            w_local,
            os.path.join(base_dir, 'local_attack_save', f'iter_{iter_}_attack_{idx}.pt'),
        )

    if args.subnet and iter_ % args.subnet == 0 and iter_ >= args.subnet:
        net_tss = server_defender.defense_subnetmasking(net_local, w_local)
        np.save(
            os.path.join(base_dir, 'visual', f'iter_{iter_}_attack_{idx}.npy'),
            net_tss,
        )

    del net_local
    del local_updater
    clear_training_cache()
    return idx, w_local, idxs_weight_dict[idx], loss


def parallelTrainingIntegrated(tasks: list):
    results = []
    for task in tasks:
        local_updater = LocalUpdater()
        (
            iter_, idx, args, is_attacker, attacker, server_defender,
            current_round_users_indices, idxs_weight_dict, net_glob,
            last_global_dict, dataset_train, dict_users_train, lr,
            with_local_save, base_dir, normal_save_candidates,
        ) = task
        local_updater.update_params(
            dataset=dataset_train,
            idxs=dict_users_train[idx],
            params=get_local_training_params(args),
        )
        net_local = copy.deepcopy(net_glob)
        if args.attack_on_attack != []:
            if idx in args.attack_on_attack:
                w_local, loss = local_updater.train_attack_dynamic(
                    model_local=net_local.to(args.device), attacker=attacker, last_global_dict=last_global_dict, lr=lr, idx=idx)
            else:
                w_local, loss = local_updater.train(
                    net=net_local.to(args.device), lr=lr)
        elif is_attacker and args.attack_type != "peace" and iter_ >= args.start_attack:
            w_local, loss = local_updater.train_attack_dynamic(
                model_local=net_local.to(args.device), attacker=attacker, last_global_dict=last_global_dict, lr=lr, idx=idx)
        else:
            w_local, loss = local_updater.train(
                net=net_local.to(args.device), lr=lr)

        if args.scale:
            w_local = scale_model_update(
                w_local, net_local, current_round_users_indices, args.device
            )

        should_save = (
            iter_ % args.local_saving_interval == 0
            and iter_ >= args.local_saving_start
            and with_local_save
        )
        if should_save and is_attacker:
            torch.save(
                w_local,
                os.path.join(base_dir, 'local_attack_save', f'iter_{iter_}_attack_{idx}.pt'),
            )
        if should_save and idx in normal_save_candidates:
            torch.save(
                w_local,
                os.path.join(base_dir, 'local_normal_save', f'iter_{iter_}_normal_{idx}.pt'),
            )

        if args.subnet and iter_ % args.subnet == 0 and iter_ >= args.subnet:
            net_tss = server_defender.defense_subnetmasking(net_local, w_local)
            np.save(
                os.path.join(
                    base_dir,
                    'visual',
                    f'iter_{iter_}_{"attack" if is_attacker else "normal"}_{idx}.npy',
                ),
                net_tss,
            )

        result_weights = net_glob.state_dict() if args.no_attack_on_attack else w_local
        results.append((idx, result_weights, idxs_weight_dict[idx], loss))

        del net_local
        del local_updater
        clear_training_cache()
    return results
