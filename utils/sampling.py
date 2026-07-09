#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
from itertools import permutations
import numpy as np
import torch
import pdb

def fair_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fairness dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fair_noniid(train_data, num_users, num_shards=200, num_imgs=300, train=True, rand_set_all=[]):
    """
    Sample non-I.I.D client data from fairness dataset
    :param dataset:
    :param num_users:
    :return:
    """
    assert num_shards % num_users == 0
    shard_per_user = int(num_shards / num_users)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    #import pdb; pdb.set_trace()

    labels = train_data[1].numpy().reshape(len(train_data[0]),)
    assert num_shards * num_imgs == len(labels)
    #import pdb; pdb.set_trace()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if len(rand_set_all) == 0:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
            for rand in rand_set:
                rand_set_all.append(rand)

            idx_shard = list(set(idx_shard) - rand_set) # remove shards from possible choices for other users
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else: # this only works if the train and test set have the same distribution of labels
        for i in range(num_users):
            rand_set = rand_set_all[i*shard_per_user: (i+1)*shard_per_user]
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users, rand_set_all

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index (consistent with dirichlet_noniid_distribute format)
    """
    num_items = int(len(dataset)/num_users)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = [i for i in range(len(dataset))]
    
    for i in range(num_users):
        selected_idxs = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = selected_idxs.astype('int64')
        all_idxs = list(set(all_idxs) - set(selected_idxs))
    
    # Handle remaining samples (if any due to integer division)
    if all_idxs:
        for i, remaining_idx in enumerate(all_idxs):
            client_id = i % num_users
            dict_users[client_id] = np.concatenate((dict_users[client_id], [remaining_idx]))
    
    return dict_users

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all

def dirichlet_noniid(dataset, num_users, alpha=0.5):
    """
    Sample non-IID client data from dataset using Dirichlet distribution
    Args:
        dataset: dataset to split
        num_users: number of clients
        alpha: parameter for Dirichlet distribution
               (smaller alpha -> more non-iid)
    Returns:
        dict_users: dictionary containing data indices for each client
    """
    num_classes = len(np.unique(dataset.targets))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # 获取每个类别的索引
    label_indices = {}
    for k in range(num_classes):
        label_indices[k] = np.where(np.array(dataset.targets) == k)[0]
    
    # 为每个类别生成dirichlet分布
    distributions = np.random.dirichlet([alpha] * num_users, num_classes)
    
    # 分配数据
    for k in range(num_classes):
        # 获取当前类别的索引
        class_indices = label_indices[k]
        # 根据dirichlet分布计算每个客户端应获得的样本数量
        proportions = distributions[k]
        # 计算实际分配的数量
        amounts = np.random.multinomial(len(class_indices), proportions)
        
        # 分配给每个客户端
        start_idx = 0
        for client_id in range(num_users):
            amount = amounts[client_id]
            if amount > 0:
                selected_indices = class_indices[start_idx:start_idx + amount]
                dict_users[client_id] = np.concatenate((dict_users[client_id], selected_indices))
                start_idx += amount
    
    # 验证
    all_indices = []
    for k in range(num_users):
        all_indices.extend(dict_users[k])
    all_indices = np.array(all_indices)
    
    # 确保没有重复
    assert len(np.unique(all_indices)) == len(all_indices)
    # 确保所有数据都被分配
    assert len(all_indices) == len(dataset)
    
    return dict_users

def dirichlet_noniid_distribute(dataset, num_users, data_augmentation=0, alpha=0.5, distributions=None):
    """
    Sample non-IID client data from dataset using Dirichlet distribution
    Args:
        dataset: dataset to split
        num_users: number of clients
        alpha: parameter for Dirichlet distribution
               (smaller alpha -> more non-iid)
        distributions: Dirichlet distribution parameters from training dataset
        data_augmentation: by default it is the number of users, which means each user gets at least one sample from each class.
    Returns:
        dict_users: dictionary containing data indices for each client
        distributions: Dirichlet distribution parameters
    """
    data_augmentation = data_augmentation or num_users
    assert data_augmentation != 0
    num_classes = len(np.unique(dataset.targets))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    # 获取每个类别的索引
    label_indices = {}
    for k in range(num_classes):
        label_indices[k] = np.where(np.array(dataset.targets) == k)[0]
    
    # 为每个类别生成dirichlet分布
    distributions = np.random.dirichlet([alpha] * num_users, num_classes) if distributions is None else distributions
    
    # 分配数据
    for k in range(num_classes):
        # 获取当前类别的索引
        class_indices = label_indices[k]
        # 根据dirichlet分布计算每个客户端应获得的样本数量
        proportions = distributions[k]
        # 计算实际分配的数量
        amounts = np.random.multinomial(len(class_indices), proportions)
        
        # 分配给每个客户端
        start_idx = 0
        for client_id in range(num_users):
            amount = amounts[client_id]
            if amount > 0:
                selected_indices = class_indices[start_idx:start_idx + amount]
                dict_users[client_id] = np.concatenate((dict_users[client_id], selected_indices))
                start_idx += amount
    
    # 验证
    all_indices = []
    for k in range(num_users):
        all_indices.extend(dict_users[k])
    all_indices = np.array(all_indices)
    
    # 确保没有重复
    assert len(np.unique(all_indices)) == len(all_indices)
    # 确保所有数据都被分配
    assert len(all_indices) == len(dataset)
    
    return dict_users, distributions


def plotDataDistribution(dict_users, dataset, num_classes, fig_path):
    import matplotlib.pyplot as plt
    """
    Plots the data distribution for n clients and m classes as an n x m matrix.
    Args:
        dict_users: dict, keys are client ids, values are arrays of data indices
        dataset: dataset object with .targets attribute
        num_classes: int, number of classes
    """
    num_clients = len(dict_users)
    data_matrix = np.zeros((num_clients, num_classes), dtype=int)
    targets = np.array(dataset.targets)

    for client_id, indices in dict_users.items():
        labels = targets[indices]
        for cls in range(num_classes):
            data_matrix[client_id, cls] = np.sum(labels == cls)

    plt.figure(figsize=(num_classes, num_clients))
    plt.imshow(data_matrix, aspect='auto', cmap='Blues')
    plt.colorbar(label='Number of samples')
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title('Client-Class Data Distribution')
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_clients))
    plt.savefig(fig_path)
    plt.close('all')
    plt.clf()
    plt.cla()


            
def noniid_unbalanced(dataset, num_users, shard_per_user, rand_set_all=[], ub_at=0, ub_users_percent=0.3):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    print("unbalanced")
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    for i in range(0, int(num_users*ub_users_percent)):
        # for change_i in range(int(len(rand_set_all[i]) * 1)):
        #     j = np.random.choice(len(rand_set_all[i]), replace=False)
        #     if rand_set_all[i][j] != ub_at:
        #         q = new_func(num_users, rand_set_all, ub_at, ub_users_percent, rand_set_all[i][j])
        #     rand_set_all[i][j] = q
        # if ub_at not in rand_set_all[i]:
        for j in range(int(len(rand_set_all[i]) * 0.3)):
            # new_func(num_users, rand_set_all, ub_at, ub_users_percent, rand_set_all[i][0])
            find = 0
            for k in range(int(num_users*ub_users_percent), num_users):
                if find: break
                for idx, q in enumerate(rand_set_all[k]):
                    if q == ub_at:
                        rand_set_all[k][idx] = rand_set_all[i][j]
                        find = 1
                        break
            if find:
                rand_set_all[i][j] = ub_at

    print("test")
    for i in range(0, int(num_users)):
        print(i, rand_set_all[i])

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all


def noniid_replace(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    imgs_per_shard = int(len(dataset) / (num_users * shard_per_user))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    if len(rand_set_all) == 0:
        for i in range(num_users):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            pdb.set_trace()
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        dict_users[i] = np.concatenate(rand_set)

    for key, value in dict_users.items():
        assert(len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dict_users, rand_set_all