#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from collections import defaultdict
import copy

import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import math

import torchvision
import random

from models.backdoorpattern import pattern_tensor_dba, pattern_tensor_normal, pattern_tensor_empty, pattern_tensor_blend
from utils.attackUtils import compute_cos_sim_loss_1, compute_euclidean_loss, get_fl_update, predict_the_global_model, preprocessDistilledData, ref_f, update_the_Ss
from utils.attacker import Attacker
from utils.tdFedUtils import get_update_norm, scale_update
from utils.dataUtils import TensorLabelsDataset, normalizers
from utils.dataUtils import AugmentedDataset, DatasetSplit, ConcatDataset


class SAM():
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAMOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        # Wrap a base optimizer (e.g., SGD, Adam)
        self.base_optimizer = base_optimizer
        self.rho = rho
        super(SAMOptimizer, self).__init__(
            params, {'lr': base_optimizer.defaults['lr']})

    # @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "A closure is required for SAM"
        # First pass: evaluate and get the loss
        loss = closure()
        loss.backward()

        # Compute sharpness-aware perturbation
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_norm = torch.norm(p.grad)
                    if grad_norm != 0:
                        perturbation = self.rho * p.grad / grad_norm
                        p.add_(perturbation)  # Perturb weights

        # Second pass: compute gradient at perturbed weights
        loss = closure()
        loss.backward()

        # Restore original weights and apply base optimizer step
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_norm = torch.norm(p.grad)
                    if grad_norm != 0:
                        perturbation = self.rho * p.grad / grad_norm
                        p.sub_(perturbation)  # Restore original weights

        # Apply base optimizer step
        self.base_optimizer.step()

        return loss


class LocalUpdater(object):
    def __init__(self):
        # self.params = {}
        # self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        # if isinstance(idxs, ndarray):
        #     self.ldr_train = DataLoader(DatasetSplit(
        #         dataset, idxs), batch_size=args.local_bs, shuffle=True, num_workers=args.max_workers)
        # self.pretrain = pretrain
        pass

    def update_params(self, dataset, idxs, params):
        # self.params = {
        #     'data_augmentation_local': data_augmentation_local,
        #     'model': model_str,
        #     'dataset': dataset_str,
        #     'num_classes': num_classes,
        #     'data_portion': data_portion,
        #     'max_workers': max_workers,
        #     'local_bs': local_bs,
        #     'local_ep': local_ep,
        #     'local_ep_times': local_ep_times,
        #     'input_size': input_size,
        #     'blend_alpha': blend_alpha,

        #     'local_ep_pretrain': local_ep_pretrain,
        #     'attack_type': attack_type,
        #     'pattern_choice': pattern_choice,
        #     'pos_choice': pos_choice,
        #     'label': label,

        #     'device': device,
        #     }
        self.params = params
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if self.params['data_augmentation_local'] > 1:
            augmented_dataset_list_train = [
                DatasetSplit(dataset, idxs) for _ in range(self.params['data_augmentation_local'])]
            dataset_train = ConcatDataset(augmented_dataset_list_train)
            self.ldr_train = DataLoader(dataset_train, batch_size=self.params['local_bs'], shuffle=True, num_workers=self.params['max_workers'])
        else:
            self.ldr_train = DataLoader(DatasetSplit(
                dataset, idxs), batch_size=self.params['local_bs'], shuffle=True, num_workers=self.params['max_workers'])
        

    def train(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.params['local_ep_pretrain']:
            local_eps = self.params['local_ep_pretrain'] + self.params['local_ep']
        else:
            local_eps = self.params['local_ep']
        for _ in range(local_eps):
            batch_loss = []
            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.params['device']), labels.to(self.params['device'])
                # print(len(images), len(labels))
                net.zero_grad()
                log_probs = net(images)

                # print(log_probs.shape, labels.shape)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_sam(self, net, idx=-1, lr=0.1):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
        minimizer = SAM(optimizer, net, rho=0.05, eta=0.01)

        epoch_loss = []
        if self.params['local_ep_pretrain']:
            local_eps = self.params['local_ep_pretrain'] + self.params['local_ep']
        else:
            local_eps = self.params['local_ep']
        for _ in range(local_eps):
            running_loss = []
            for _, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.params['device'])
                labels = labels.to(self.params['device'])

                # Ascent Step
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                minimizer.ascent_step()

                # Descent Step
                self.loss_func(net(images), labels).backward()
                minimizer.descent_step()

                with torch.no_grad():
                    running_loss.append(loss.item())
            epoch_loss.append(sum(running_loss) / len(running_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_attack_dynamic(self, model_local, attacker: Attacker, last_global_dict=None, idx=-1, lr=0.1, alpha=1):
        # TODO 2025-03-03 git.V.32546: attack should happen after epoch 1
        # TODO 2025-03-02 git.V.15c84: label counts correct. check pattern
        # if args.dba:
        #     pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        # elif args.groupattack:
        #     pattern_tensor = pattern_tensor_normal[idx % len(pattern_tensor_normal)]
        # else:
        #     pattern_tensor = pattern_tensor_normal[args.pattern_choice-1]
        model_t0 = copy.deepcopy(model_local)
        attack_type = self.params['attack_type']
        pattern_choice = self.params['pattern_choice']
        if attack_type == 'dark':
            # TODO 2025-02-27 git.V.36afa: fix dataset
            running_dataset = preprocessDistilledData(model_local, self.ldr_train, batch_size=self.params['local_bs'], dataset_str=self.params['dataset'], device=self.params['device'])
            running_dataloader = attacker.getDataloader(TensorLabelsDataset(torch.utils.data.ConcatDataset([running_dataset, self.ldr_train.dataset])))
            # running_dataloader = attacker.getDataloader(self.ldr_train.dataset)
            # running_dataloader = self.ldr_train
        elif attack_type == 'invisible':
            running_dataloader = attacker.getDataloader(self.ldr_train.dataset)
        elif attack_type == 'edge':
            # running_dataset = attacker.edge_dataset
            # running_dataloader = attacker.getDataloader(attacker.edge_dataset)
            running_dataloader = attacker.getDataloader(TensorLabelsDataset(torch.utils.data.ConcatDataset([attacker.edge_dataset, self.ldr_train.dataset])))
        else:
            running_dataloader = self.ldr_train
            
        # prepare pattern
        if attack_type in ['peace', 'adam']:
            pattern_tensor = pattern_tensor_empty
        elif attack_type == 'static':
            pattern_tensor = pattern_tensor_normal[pattern_choice % len(pattern_tensor_normal)]
        elif attack_type == 'blend':
            pattern_tensor = attacker.blend_trigger[:self.params['input_size'], :self.params['input_size']]
            alpha = self.params['blend_alpha']
            self.params['pos_choice'] = [0, 0]
        elif attack_type == 'dynamic':
            # TODO 2025-01-19 git.V.29c9f: fix dynamic attack
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif attack_type == 'dark':
            # TODO 2025-01-19 git.V.29c9f: fix dark
            pattern_tensor = pattern_tensor_normal[pattern_choice % len(pattern_tensor_normal)]
        elif attack_type == '3dfed':
            pattern_tensor = pattern_tensor_normal[pattern_choice % len(pattern_tensor_normal)]
            global_update = get_fl_update(model_local.state_dict(), last_global_dict) # TODO 2025-03-04 git.V.32546: requires global model
            attacker.tdFedReadIndicator(global_update)
            attacker.tdAdaptiveTuning()
        elif attack_type == 'invisible':
            pattern_tensor = pattern_tensor_empty
        elif attack_type == 'dba':
            pattern_tensor = pattern_tensor_dba[np.where(attacker.attack_clients == idx)[0][0] % len(pattern_tensor_dba)]
        elif attack_type == 'groupattack':
            pattern_tensor = pattern_tensor_normal[idx % len(
                pattern_tensor_normal)]
        elif attack_type == 'edge':
            pattern_tensor = pattern_tensor_normal[pattern_choice % len(pattern_tensor_normal)]
            # pattern is not used in edge attack
        elif attack_type == 'mirage':
            pattern_tensor = pattern_tensor_empty
        else:
            raise ValueError("Invalid attack type")

        iterator = iter(running_dataloader)
        inputs, _ = next(iterator)
        full_image = torch.zeros(inputs[0].shape)

        # TODO 2025-03-10 git.V.83183: extract function
        mask_value = -10
        full_image.fill_(mask_value)
        x_top = self.params['pos_choice'][0]
        y_top = self.params['pos_choice'][1]
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
        normalizer = normalizers[self.params['dataset']]
        pattern = normalizer(full_image)
        attackportion = self.params['data_portion']

        if attack_type == 'dark':
            s1 = model_local.state_dict()
            if (last_global_dict == None):
                s2 = model_local.state_dict()
            else:
                s1, s2 = update_the_Ss(s1, last_global_dict,
                                    model_t0.state_dict(), alpha=0.8)

            new_state_dict = predict_the_global_model(s1, s2, alpha=0.8)
            predicted_model = copy.deepcopy(model_local)
            predicted_model.load_state_dict(new_state_dict)
            # Scale-up
            benign_update_dict = get_fl_update(new_state_dict, model_t0.state_dict())
            benign_norm = get_update_norm(benign_update_dict)

        
        if attack_type == '3dfed':
            net_benign = copy.deepcopy(model_local)
            net_benign.train()
            optimizer_3dfed = torch.optim.SGD(net_benign.parameters(), lr=lr, momentum=0.5)
            for _ in range(self.params['local_ep'] * self.params['local_ep_times']):
                for batch_idx, (images, labels) in enumerate(running_dataloader):
                    images, labels = images.to(
                        self.params['device']), labels.to(self.params['device'])
                    net_benign.zero_grad()
                    log_probs = net_benign(images)

                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer_3dfed.step()
            benign_update_dict = get_fl_update(net_benign.state_dict(), model_t0.state_dict())
            benign_norm = get_update_norm(benign_update_dict)

        
        if attack_type == 'edge':
            edge_model = copy.deepcopy(model_t0)
            attacker.edgeTrain(edge_model, running_dataloader, self.params['local_ep'] * self.params['local_ep_times'], device=self.params['device'])

        model_local.train()
        # train and update
        if attack_type == 'adam':
            optimizer = torch.optim.Adam(model_local.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model_local.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []

        if attack_type in ['peace', 'adam', 'static', 'blend', 'dynamic', 'dark', 'dba', 'groupattack']:
            if self.params['local_ep_pretrain']:
                local_eps = self.params['local_ep_pretrain'] + self.params['local_ep']
            else:
                local_eps = self.params['local_ep']
            for _ in range(local_eps * self.params['local_ep_times']):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(running_dataloader):
                    if attack_type not in ['peace', 'adam']:
                        for j in range(int(len(labels)*attackportion)):
                            # images[j] = (1 - mask) * images[j] + mask * pattern
                            images[j] = (1 - mask) * images[j] + mask * (
                                (1 - alpha) * images[j] + alpha * pattern
                            )
                            labels[j] = self.params['label']
                    images, labels = images.to(
                        self.params['device']), labels.to(self.params['device'])
                    model_local.zero_grad()
                    log_probs = model_local(images)

                    loss = self.loss_func(log_probs, labels)

                    if attack_type == 'dark':
                        eu_loss = 0
                        eu_loss = compute_euclidean_loss(
                            model_local, model_t0, self.params['device'])
                        cos_loss = 0
                        cos_loss = compute_cos_sim_loss_1(
                            model_local, self.params['dataset'], attacker.shared_models, predicted_model, device=self.params['device'])
                        loss = loss + 0.5*eu_loss + 0.5*cos_loss
                    loss.backward()
                    optimizer.step()

                    batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if attack_type == 'invisible':
            dim_f = attacker.dim_f
            invi_model = copy.deepcopy(model_local)
            attacker.invs_feature_r = ref_f(invi_model, running_dataloader, self.params['num_classes'], self.params['local_bs'], dim_f=dim_f[self.params['model']], device=self.params['device'])
            invi_model.train()
            attacker.invisibleTrain(running_dataloader, invi_model, lr=lr, epochs=self.params['local_ep'] * self.params['local_ep_times'], device=self.params['device'])
        

        if attack_type == 'mirage':
            mirage_model = copy.deepcopy(model_local)
            mirage_model = attacker.mirage_train(
                mirage_model, running_dataloader, client_id=idx, epochs=self.params['local_ep'] * self.params['local_ep_times'])
            model_local = mirage_model
        
        if attack_type == '3dfed':
            backdoor_norm = get_update_norm(model_local.state_dict())
            scale_f = min((benign_norm / backdoor_norm), 1.1)
            scale_update(model_local.state_dict(), max(scale_f, 1))
            attacker.tdDesignIndicators(copy.deepcopy(model_t0), copy.deepcopy(model_local), running_dataloader, mask, pattern, self.loss_func, device=self.params['device'])
            # TODO 2025-03-03 git.V.32546: add noise
            attacker.tdMask(copy.deepcopy(model_t0), copy.deepcopy(model_local), device=self.params['device'])

        # if self.args.attack_type == 'dark':
        #     # Scale-up
        #     backdoor_norm = get_update_norm(net.state_dict())
        #     scale_f = min((benign_norm / backdoor_norm), 1.1)
        #     scale_update(net.state_dict(), max(scale_f, 1))

        if attack_type == 'invisible':
            model_local = invi_model
        
        elif attack_type == 'edge':
            model_local = edge_model

        attacker.shared_models.append(copy.deepcopy(model_local))

        return model_local.state_dict(), sum(epoch_loss) / (len(epoch_loss) or 1)


class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep

        for _ in range(local_eps):
            batch_loss = []
            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone()

                W_local = [net.state_dict(keep_vars=True)[
                    key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                k = 4000
                for i in range(W.shape[0] // k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
