#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import pickle
import random
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb

import torch.utils.data.dataloader
import torchvision

from utils.attacker import Attacker
from utils.dataUtils import getDataWithDistribution, getDataGlobal

from .backdoorpattern import pattern_tensor_dba, pattern_tensor_normal, pattern_tensor_empty
from utils.attackUtils import get_fl_update, ref_f
from utils.dataUtils import normalizers, CorruptedDataset, VoidDataset, FilteredDataset, ShrinkedDataset



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset_train, self.dataset_test, _, _ = getDataWithDistribution(args)
        
        if self.args.attack_type == 'edge':
            _, self.edge_dataset_test = getDataGlobal(self.args.dataset)

            with open('assets/southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)

            sampled_targets_array_test = self.args.label * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int)

            self.edge_dataset_test.data = saved_southwest_dataset_test
            self.edge_dataset_test.targets = sampled_targets_array_test

    def test(self, model_t0, last_global_dict, attacker: Attacker, alpha=1, idx=0):
        # TODO 2025-09-03 git.V.300b8: set idx == -1 when evaluating global model
        model_t0.eval()
        running_dataloader = DataLoader(self.dataset_test, batch_size=self.args.bs, shuffle=False, num_workers=4, drop_last=True)

        # prepare pattern
        if self.args.attack_type in ['peace', 'adam']:
            pattern_tensor = pattern_tensor_empty
        elif self.args.attack_type == 'static':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == 'blend':
            pattern_tensor = attacker.blend_trigger
            alpha = self.args.blend_alpha
            self.args.pos_choice = [0, 0]
        elif self.args.attack_type == 'dynamic':
            # TODO 2025-01-19 git.V.29c9f: fix dynamic attack
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif self.args.attack_type == 'dark':
            # TODO 2025-01-19 git.V.29c9f: fix dark
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == '3dfed':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            global_update = get_fl_update(model_t0.state_dict(), last_global_dict)
            attacker.tdFedReadIndicator(global_update)
            attacker.tdAdaptiveTuning()
        elif self.args.attack_type == 'invisible':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == 'dba':
            pattern_tensor = pattern_tensor_dba[0] + pattern_tensor_dba[1] # TODO 2025-03-19 git.V.0d567: check
        elif self.args.attack_type == 'groupattack':
            pattern_tensor = pattern_tensor_normal[0]
        elif self.args.attack_type == 'edge':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            # no actual pattern is used in edge attack
        elif self.args.attack_type == 'mirage':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        else:
            raise ValueError("Invalid attack type")

        iterator = iter(running_dataloader)
        inputs, _ = next(iterator)
        full_image = torch.zeros(inputs[0].shape)

        mask_value = -10
        full_image.fill_(mask_value)
        x_top = self.args.pos_choice[0]
        y_top = self.args.pos_choice[1]
        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        mask = 1 * (full_image != mask_value)
        # pattern = normalize_cifar(full_image) if datatype == "c" else normalize_mnist(full_image)
        # if datatype == "c":
        #     pattern = normalize_cifar(full_image)
        # elif datatype == "g":
        #     pattern = normalize_gtsrb(full_image)
        # else:
        #     pattern = normalize_mnist(full_image)
        normalizer = normalizers[self.args.dataset]
        pattern = normalizer(full_image)
        attackportion = 1.0


        model_t0.eval()

        correct = 0
        total = 0
        loss_list_base = []
        for batch_idx, (images, labels) in enumerate(running_dataloader):
            images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
            log_probs = model_t0(images)
            loss = self.loss_func(log_probs, labels)
            loss_list_base.append(loss.item())
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy_base = 100 * correct / total

        #%% backdoored to correct class
        correct_original = 0
        total = 0
        total_altered = 0
        correct_backdoor = 0
        loss_list_attack = []
        for batch_idx, (images, labels) in enumerate(running_dataloader):
            # if self.args.attack_type not in ['peace', 'adam']:
            for j in range(int(len(labels)*1.0)):
                images[j] = (1 - mask) * images[j] + mask * (
                    (1 - alpha) * images[j] + alpha * pattern
                )
                # labels[j] = self.args.label
            labels_backdoor = torch.ones_like(labels) * self.args.label
            images, labels_backdoor, labels = images.to(
                    self.args.device), labels_backdoor.to(self.args.device), labels.to(self.args.device)
            log_probs = model_t0(images)
            loss = self.loss_func(log_probs, labels)
            loss_bd = self.loss_func(log_probs, labels_backdoor)
            loss_list_attack.append(loss_bd.item())
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            total_altered += (labels != labels_backdoor).sum().item()
            correct_original += ((predicted == labels) & (labels != labels_backdoor)).sum().item()
            correct_backdoor += ((predicted == labels_backdoor) & (labels != labels_backdoor)).sum().item()
        accuracy_original = 100 * correct_original / total
        accuracy_target = 100 * correct_backdoor / total_altered


        if self.args.attack_type == 'invisible':
            dim_f = attacker.dim_f
            attacker.invs_feature_r = ref_f(model_t0, running_dataloader, self.args.num_classes, self.args.local_bs, dim_f=dim_f[self.args.model], device=self.args.device)
            ac_list_target, Loss, L1, LF = attacker.invisibleEval_withFreqAnalysis(model_t0, running_dataloader, device=self.args.device)
            accuracy_target = np.mean(ac_list_target)

        elif self.args.attack_type == 'edge':
            accuracy_target, test_loss = self.edgeEval(model_t0, device=self.args.device)

        elif self.args.attack_type == 'mirage':
            m_trigger=attacker.mirage_trigger_set[idx]
            m_mask=attacker.mirage_mask_set[idx]
            m_label_swap=attacker.mirage_params['poison_label_swap'][idx]
            accuracy_target, test_loss = attacker.mirage_test_model(model_t0, running_dataloader, is_poisoned=True, trigger=m_trigger, mask=m_mask, label_swap=m_label_swap, client_id=idx)

        del model_t0
        return accuracy_base, np.mean(loss_list_base), accuracy_original, accuracy_target

        # original_labels = 0
        # bd_labels = 0
        # for batch_idx, (images, labels) in enumerate(running_dataloader):
        #     labels_altered = labels.clone()
        #     for j in range(int(len(labels)*attackportion)):
        #         images[j] = (1 - mask) * images[j] + mask * (
        #             (1 - alpha) * images[j] + alpha * pattern
        #         )
        #         # labels[j] = self.args.label
        #         labels_altered[j] = self.args.label
        #     labels_backdoor = torch.ones_like(labels) * self.args.label
        #     images, labels, labels_backdoor = images.to(
        #             self.args.device), labels.to(self.args.device), labels_backdoor.to(self.args.device)
        #     log_probs = model_t0(images)
        #     loss = self.loss_func(log_probs, labels)
        #     loss_list_attack.append(loss.item())
        #     _, predicted = torch.max(log_probs.data, 1)
        #     total += labels.size(0)
        #     correct_original += (predicted == labels).sum().item()
        #     correct_backdoor += ((predicted == labels_backdoor) & (labels != labels_backdoor)).sum().item()
        #     original_labels += (labels != labels_backdoor).sum().item()
        #     bd_labels += (labels_altered == labels_backdoor).sum().item()
        
        # accuracy_target = 100 * correct_backdoor / bd_labels
        # accuracy_original = 100 * correct_original / original_labels

    def edgeEval(self, net, device='cuda'):
        model_t0 = copy.deepcopy(net).to(device)
        model_t0.eval()
        running_dataloader = DataLoader(self.edge_dataset_test, batch_size=self.args.bs, shuffle=False, num_workers=4, drop_last=True)
        
        correct = 0
        total = 0
        test_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(running_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = self.loss_func(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            c = (predicted == targets).squeeze()

            #print(targets)


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total
        return accuracy, test_loss / len(running_dataloader)

    def triggerEval(self, model_t0, target_label, pos_choice, trigger, num_class, img_channel, img_size, mask):
        dataset = CorruptedDataset(VoidDataset(self.dataset_test), 
                [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask)
        # dataset = VoidDataset(self.dataset_test)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

        model_t0.eval()
        correct = 0
        total = 0
        loss_list_base = []
        for batch_idx, (images, labels) in enumerate(dataloader):
            # images = torch.zeros_like(images)
            # for j in range(int(len(labels)*1.0)):
            #     images[j] = (1 - mask) * images[j] + mask * trigger
            images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
            log_probs = model_t0(images)
            loss = self.loss_func(log_probs, labels)
            loss_list_base.append(loss.item())
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy_base = 100 * correct / total

        return accuracy_base

    def test_void(self, model_t0, last_global_dict, attacker: Attacker, alpha=1, idx=0):
        # TODO 2025-09-03 git.V.300b8: set idx == -1 when evaluating global model
        model_t0.eval()
        running_dataloader = DataLoader(self.dataset_test, batch_size=self.args.bs, shuffle=False, num_workers=4, drop_last=True)

        # prepare pattern
        if self.args.attack_type in ['peace', 'adam']:
            pattern_tensor = pattern_tensor_empty
        elif self.args.attack_type == 'static':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == 'blend':
            pattern_tensor = attacker.blend_trigger
            alpha = self.args.blend_alpha
            self.args.pos_choice = [0, 0]
        elif self.args.attack_type == 'dynamic':
            # TODO 2025-01-19 git.V.29c9f: fix dynamic attack
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif self.args.attack_type == 'dark':
            # TODO 2025-01-19 git.V.29c9f: fix dark
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == '3dfed':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            global_update = get_fl_update(model_t0.state_dict(), last_global_dict)
            attacker.tdFedReadIndicator(global_update)
            attacker.tdAdaptiveTuning()
        elif self.args.attack_type == 'invisible':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.args.attack_type == 'dba':
            pattern_tensor = pattern_tensor_dba[0] + pattern_tensor_dba[1] # TODO 2025-03-19 git.V.0d567: check
        elif self.args.attack_type == 'groupattack':
            pattern_tensor = pattern_tensor_normal[0]
        elif self.args.attack_type == 'edge':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            # no actual pattern is used in edge attack
        elif self.args.attack_type == 'mirage':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        else:
            raise ValueError("Invalid attack type")

        iterator = iter(running_dataloader)
        inputs, _ = next(iterator)
        full_image = torch.zeros(inputs[0].shape)

        mask_value = -10
        full_image.fill_(mask_value)
        x_top = self.args.pos_choice[0]
        y_top = self.args.pos_choice[1]
        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
        mask = 1 * (full_image != mask_value)
        # pattern = normalize_cifar(full_image) if datatype == "c" else normalize_mnist(full_image)
        # if datatype == "c":
        #     pattern = normalize_cifar(full_image)
        # elif datatype == "g":
        #     pattern = normalize_gtsrb(full_image)
        # else:
        #     pattern = normalize_mnist(full_image)
        normalizer = normalizers[self.args.dataset]
        pattern = normalizer(full_image)
        attackportion = 1.0


        model_t0.eval()

        correct = 0
        total = 0
        loss_list_base = []
        for batch_idx, (images, labels) in enumerate(running_dataloader):
            images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
            log_probs = model_t0(images)
            loss = self.loss_func(log_probs, labels)
            loss_list_base.append(loss.item())
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy_base = 100 * correct / total

        #%% backdoored to correct class
        correct_original = 0
        total = 0
        total_altered = 0
        correct_backdoor = 0
        loss_list_attack = []
        for batch_idx, (images, labels) in enumerate(running_dataloader):
            # if self.args.attack_type not in ['peace', 'adam']:
            for j in range(int(len(labels)*1.0)):
                images[j] = torch.zeros_like(images[j])
                images[j] = (1 - mask) * images[j] + mask * (
                    (1 - alpha) * images[j] + alpha * pattern
                )
                # labels[j] = self.args.label
            labels_backdoor = torch.ones_like(labels) * self.args.label
            images, labels_backdoor, labels = images.to(
                    self.args.device), labels_backdoor.to(self.args.device), labels.to(self.args.device)
            log_probs = model_t0(images)
            loss = self.loss_func(log_probs, labels)
            loss_bd = self.loss_func(log_probs, labels_backdoor)
            loss_list_attack.append(loss_bd.item())
            _, predicted = torch.max(log_probs.data, 1)
            total += labels.size(0)
            total_altered += (labels != labels_backdoor).sum().item()
            correct_original += ((predicted == labels) & (labels != labels_backdoor)).sum().item()
            correct_backdoor += ((predicted == labels_backdoor) & (labels != labels_backdoor)).sum().item()
        accuracy_original = 100 * correct_original / total
        accuracy_target = 100 * correct_backdoor / total_altered


        if self.args.attack_type == 'invisible':
            dim_f = attacker.dim_f
            attacker.invs_feature_r = ref_f(model_t0, running_dataloader, self.args.num_classes, self.args.local_bs, dim_f=dim_f[self.args.model], device=self.args.device)
            ac_list_target, Loss, L1, LF = attacker.invisibleEval(model_t0, running_dataloader, device=self.args.device)
            accuracy_target = np.mean(ac_list_target)

        elif self.args.attack_type == 'edge':
            accuracy_target, test_loss = self.edgeEval(model_t0, device=self.args.device)

        elif self.args.attack_type == 'mirage':
            m_trigger=attacker.mirage_trigger_set[idx]
            m_mask=attacker.mirage_mask_set[idx]
            m_label_swap=attacker.mirage_params['poison_label_swap'][idx]
            accuracy_target, test_loss = attacker.mirage_test_model(model_t0, running_dataloader, is_poisoned=True, trigger=m_trigger, mask=m_mask, label_swap=m_label_swap, client_id=idx)

        del model_t0
        return accuracy_base, np.mean(loss_list_base), accuracy_original, accuracy_target