from argparse import Namespace
import copy
import math
import pickle
# import seaborn as sns
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from models.Nets import Unet, EmbeddingNet
from models.backdoorpattern import pattern_tensor_blend, pattern_tensor_dba, pattern_tensor_normal, pattern_tensor_empty, pattern_tensor_blend
from utils.attackUtils import compute_noise_loss
from utils.dataUtils import getDataGlobal, normalizers
from utils.logger import myLogger
from utils.tdFedUtils import dual_ascent, getAnalogUpdate, getIndex, read_indicator


class Attacker(object):
    """
    A class to perform adversarial attacks (collusion)
    """

    def __init__(self, args: Namespace, attack_clients=(), logger: myLogger | None = None, **kwargs):
        self.args = args

        self.logger = logger
        self.shared_models:list[nn.Module] = []
        self.attack_clients = [] if attack_clients is None else list(
            attack_clients)

        self.attack_type = args.attack_type

        # self.td_ind_layer = 'conv2.layer.weight' if 'lenetsm' in self.args.model else 'conv12.layer.weight'
        # self.td_layer_name = 'fc3.layer' if 'lenetsm' in self.args.model else 'fc' # TODO 2025-03-04 git.V.32546: fix
        if self.attack_type == '3dfed':
            if self.args.model not in ['lenetsm', 'vggsm', 'resnet20sm', 'vitsm']:
                raise ValueError(
                    f"Model {self.args.model} not supported for 3DFed. Supported models are: lenetsm, vggsm, resnet20sm, vitsm.")
            td_ind_layer_dict = {'lenetsm': 'conv1.layer.weight',
                                 'vggsm': 'conv1.layer.weight',
                                 'resnet20sm': 'conv1.layer.weight',
                                 'vitsm': 'embedding.patch_embeddings.projection.layer.weight'}
            td_layer_name_dict = {'lenetsm': 'fc3.layer',
                                  'vggsm': 'fc14.layer',
                                  'resnet20sm': 'linear.layer',
                                  'vitsm': 'classifier.layer.weight'}
            self.td_layer_name = td_layer_name_dict[self.args.model]
            self.td_ind_layer = td_ind_layer_dict[self.args.model]
            self.td_accept:list[str] = []
            self.td_weakDP = True
            self.td_indicators:list[list] = []
            self.td_k = 0
            self.td_alpha = []

        if args.attack_type == 'invisible':
            # self.invs_Target_labels = torch.stack([i*torch.ones(1) for i in range(self.args.num_classes)]).expand(
            #     self.args.num_classes, self.args.local_bs).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device=args.device)
            self.invs_Target_labels = torch.full(
                (self.args.num_classes * self.args.local_bs,),
                # (self.args.local_bs,),
                args.label,
                dtype=torch.long,
                device=args.device
            )
            self.invs_feature_r = None
            self.invs_EmbeddingNet = EmbeddingNet().to(args.device)
            self.invs_EmbeddingNet.share_memory()
            self.invs_TriggerNet = Unet(
                num_class=args.num_classes, img_ch=args.num_channels, input_size=args.input_size).to(args.device)
            self.invs_TriggerNet.share_memory()
            self.dim_f = {'resnet20sm': self.args.num_classes,
                          'vggsm': 128,
                          'vgg': 128,
                          'vitsm': 192,
                          'vit': 192,
                          }

        if args.attack_type == 'edge':
            self.edge_dataset, _ = getDataGlobal(args.dataset)
            with open('assets/southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            sampled_targets_array_train = args.label * \
                np.ones((saved_southwest_dataset_train.shape[0],), dtype=int)

            self.edge_dataset.data = np.append(
                self.edge_dataset.data, saved_southwest_dataset_train, axis=0)
            self.edge_dataset.targets = np.append(
                self.edge_dataset.targets, sampled_targets_array_train, axis=0)

        if args.attack_type == 'blend':
            self.blend_trigger = pattern_tensor_blend[0]

        if args.attack_type == 'mirage':
            self.mirage_dataset_train, _ = getDataGlobal(args.dataset)
            test_dataloader = DataLoader(
                self.mirage_dataset_train, batch_size=args.local_bs, shuffle=False, drop_last=True)
            self.mirage_params = {
                'model_type': args.model,
                # 'classifier_name': 'fc5',
                'discriminator_lr': 0.01,
                'discriminator_momentum': 0.9,
                'discriminator_weight_decay': 5e-4,
                'discriminator_train_samples_pre_class': 100,
                'discriminator_batch_size': 64,
                'discriminator_train_no_times': args.mirage_disc_train_ep,
                'poisoned_pattern_choose': 1,
                'malicious_train_algo': 'Mirage',
                'poisoned_len': 8,
                'poisoned_lr': 0.01,
                'poisoned_momentum': 0.9,
                'poisoned_weight_decay': 5e-4,
                'poisoned_retrain_no_times': args.local_ep * args.local_ep_times,
                'poison_label_swap': [args.label for _ in range(args.num_users)],
                'run_device': args.device,
                'no_of_adversaries': math.ceil(args.num_users * args.portion),
                'no_of_participants': args.num_users,
                'class_num': args.num_classes,
                'trigger_size': 3,
                'trigger_search_no_times': 10,
                'trigger_lr': 0.01,
                'trigger_search_batch_size': 64,
                'trigger_search_epochs': 10,
            }
            if args.model == 'lenetsm':
                self.mirage_params['classifier_name'] = 'fc5'
            elif args.model == 'vggsm':
                self.mirage_params['classifier_name'] = 'fc14'
            elif args.model == 'resnet20sm' or args.model == 'resnet20':
                self.mirage_params['classifier_name'] = 'linear'
            else:
                raise ValueError(
                    f"Model {args.model} not supported for Mirage attack. Supported models are: lenetsm, vggsm, resnet20.")
            self.mirage_trigger_set, self.mirage_mask_set = self.mirage_init_trigger_mask(
                test_dataloader)

        if args.attack_type == '3dfed':
            # self.setAttackClients(self.attack_clients)
            self.attack_clients = attack_clients
            self.td_alpha = [0.1 for _ in range(len(self.attack_clients))]

    def setAttackClients(self, attack_clients):
        self.attack_clients = attack_clients
        # self.td_alpha = [0.1 for _ in range(len(self.attack_clients))]

    def tdFedReadIndicator(self, global_update):
        self.td_accept, self.td_weakDP = read_indicator(len(
            self.attack_clients), self.args.dataset, global_update, self.td_indicators, self.td_ind_layer, self.td_weakDP)


    def getTrigger(self, dataset, idx):
        # prepare pattern
        if self.attack_type in ['peace', 'adam']:
            pattern_tensor = pattern_tensor_empty
        elif self.attack_type == 'static':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == 'blend':
            pattern_tensor = self.blend_trigger[:self.args.input_size-5, :self.args.input_size-5]
            self.args.pos_choice = [0, 0]
        elif self.attack_type == 'dynamic':
            # TODO 2025-01-19 git.V.29c9f: fix dynamic attack
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif self.attack_type == 'dark':
            # TODO 2025-01-19 git.V.29c9f: fix dark
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == '3dfed':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == 'invisible':
            pattern_tensor = pattern_tensor_empty
        elif self.attack_type == 'dba':
            pattern_tensor = pattern_tensor_dba[np.where(self.attack_clients == idx)[0][0] % len(pattern_tensor_dba)]
        elif self.attack_type == 'groupattack':
            pattern_tensor = pattern_tensor_normal[idx % len(
                pattern_tensor_normal)]
        elif self.attack_type == 'edge':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            # pattern is not used in edge attack
        elif self.attack_type == 'mirage':
            # pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            pattern_tensor = pattern_tensor_empty
        else:
            raise ValueError("Invalid attack type")

        running_dataloader = self.getDataloader(dataset)
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
        return mask, pattern

    def getPattern(self, idx):
        # prepare pattern
        if self.attack_type in ['peace', 'adam']:
            pattern_tensor = pattern_tensor_empty
        elif self.attack_type == 'static':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == 'blend':
            pattern_tensor = self.blend_trigger[:self.args.input_size-5, :self.args.input_size-5]
            self.args.pos_choice = [0, 0]
        elif self.attack_type == 'dynamic':
            # TODO 2025-01-19 git.V.29c9f: fix dynamic attack
            pattern_tensor = pattern_tensor_dba[random.randint(0, 1)]
        elif self.attack_type == 'dark':
            # TODO 2025-01-19 git.V.29c9f: fix dark
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == '3dfed':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
        elif self.attack_type == 'invisible':
            pattern_tensor = pattern_tensor_empty
        elif self.attack_type == 'dba':
            pattern_tensor = pattern_tensor_dba[np.where(self.attack_clients == idx)[0][0] % len(pattern_tensor_dba)]
        elif self.attack_type == 'groupattack':
            pattern_tensor = pattern_tensor_normal[idx % len(
                pattern_tensor_normal)]
        elif self.attack_type == 'edge':
            pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            # pattern is not used in edge attack
        elif self.attack_type == 'mirage':
            # pattern_tensor = pattern_tensor_normal[self.args.pattern_choice % len(pattern_tensor_normal)]
            pattern_tensor = pattern_tensor_empty
        else:
            raise ValueError("Invalid attack type")
        
        return pattern_tensor

    # def tdDesignIndicators(self, global_model, backdoor_update, benign_update,
    #                        criterion, train_loader, poisoning_proportion, backdoor_dynamic_position,
    #                        resize_scale, pattern_tensor, input_shape, mask_value):
    # TODO 2025-03-03 git.V.32546: naive implementation
    def tdDesignIndicators(self, global_model, backdoor_update, trainloader, mask, pattern, loss_func, device='cuda'):
        # self.td_indicators = design_indicator(
        #     len(self.attack_clients), self.args.dataset, self.td_k, global_model, backdoor_update, benign_update,
        #     criterion, train_loader, poisoning_proportion, backdoor_dynamic_position,
        #     resize_scale, pattern_tensor, input_shape, mask_value, self.args.label)
        benign_update = copy.deepcopy(global_model)
        global_model.train()
        benign_update.train()
        backdoor_update.train()
        optimizer = torch.optim.SGD(
            benign_update.parameters(), lr=1e-3, momentum=0.5)
        for iter_ in range(10):
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                outputs = benign_update(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        analog_update, gradient, curvature, no_layer, num_candidate = getAnalogUpdate(
            backdoor_update.state_dict(), benign_update.state_dict(), self.args.model)

        for batch_idx, (images, labels) in enumerate(trainloader):
            # Compute gradient and curvature for normal loss
            images_cuda, labels_cuda = images.to(device), labels.to(device)

            outputs = global_model(images_cuda)
            loss = loss_func(outputs, labels_cuda)
            grad = torch.autograd.grad(loss.mean(),
                                       [x for x in global_model.parameters() if
                                        x.requires_grad],
                                       retain_graph=True,
                                       create_graph=True,
                                       allow_unused=True
                                       )[no_layer]
            grad.requires_grad_()
            grad_sum = torch.sum(grad)
            curv = torch.autograd.grad(grad_sum,
                                       [x for x in global_model.parameters() if
                                        x.requires_grad],
                                       retain_graph=True,
                                       allow_unused=True
                                       )[no_layer]
            gradient += grad.detach().cpu().numpy()
            curvature += curv.detach().cpu().numpy()

            # Compute gradient and curvature for backdoor loss
            for j in range(int(len(labels)*0.1)):
                images[j] = (1 - mask) * images[j] + mask * pattern
                labels[j] = self.args.label
            images_cuda, labels_cuda = images.to(device), labels.to(device)
            outputs = global_model(images_cuda)
            loss = loss_func(outputs, labels_cuda)
            grad = torch.autograd.grad(loss.mean(),
                                       [x for x in global_model.parameters() if
                                        x.requires_grad],
                                       create_graph=True,
                                       retain_graph=True,
                                       allow_unused=True
                                       )[no_layer]
            grad.requires_grad_()
            grad_sum = torch.sum(grad)
            curv = torch.autograd.grad(grad_sum,
                                       [x for x in global_model.parameters() if
                                        x.requires_grad],
                                       retain_graph=True,
                                       allow_unused=True
                                       )[no_layer]
            gradient += grad.detach().cpu().numpy()
            curvature += curv.detach().cpu().numpy()

        update_val = []
        idx_candidate = []
        for i, grad in enumerate(analog_update):
            if len(idx_candidate) < num_candidate * (self.td_k + len(self.attack_clients)):
                update_val.append(grad)
                idx_candidate.append(i)
            elif grad < max(update_val):
                temp = update_val.index(max(update_val))
                update_val[temp] = grad
                idx_candidate[temp] = i

        index = []
        curv_val = []
        curvature = np.abs(curvature.flatten()).tolist()
        # # print(len(curvature))
        # # print(idx_candidate)
        for idx in idx_candidate:
            if len(index) < len(self.attack_clients):
                curv_val.append(curvature[idx % len(curvature)])
                index.append(idx % len(curvature))
            elif curvature[idx % len(curvature)] == 0:
                # The index having max curvature
                temp = curv_val.index(max(curv_val))
                if analog_update[idx % len(curvature)] < analog_update[index[temp]]:
                    curv_val[temp] = curvature[idx % len(curvature)]
                    index[temp] = idx % len(curvature)
            elif curvature[idx % len(curvature)] < max(curv_val):
                temp = curv_val.index(max(curv_val))
                curv_val[temp] = curvature[idx % len(curvature)]
                index[temp] = idx % len(curvature)

        self.td_indicators = getIndex(
            curvature, index, model_name=self.args.model)

    def tdAdaptiveTuning(self):
        if self.td_weakDP:
            # logger.warning("3DFed: disable adaptive tuning")
            for i in range(len(self.td_alpha)):
                self.td_alpha[i] = 0.1
            return self.td_alpha, self.td_k

    def tdMask(self, global_model, backdoor_update, fl_num_neurons=8, fl_adv_group_size=1, device='cuda'):
        noise_masks = []
        random_neurons = []
        for gp_id in range(math.ceil(len(self.attack_clients)/fl_adv_group_size)):
            # Select low-update neurons for this group
            temp = list(range(200)) if 'Imagenet' in self.args.dataset else list(
                range(10))
            temp.remove(self.args.label)
            np.random.shuffle(temp)
            random_neurons.append(temp[:fl_num_neurons])

            # Initialize noise masks with random number
            noise_lists = []
            for i in range(fl_adv_group_size):
                noised_model = copy.deepcopy(global_model)
                for name, data in noised_model.state_dict().items():
                    if self.td_layer_name in name:
                        noised_layer = torch.FloatTensor(data.shape).fill_(0)
                        noised_layer = noised_layer.to(device)
                        if 'Imagenet' in self.args.dataset:
                            noised_layer.normal_(
                                mean=0, std=0.0005)  # std=0.01)
                        elif 'mnist' in self.args.dataset:
                            if self.td_weakDP:
                                noised_layer.normal_(mean=0, std=0.01)
                            else:
                                noised_layer.normal_(mean=0, std=0.05)
                        else:
                            noised_layer.normal_(mean=0, std=0.01)
                        data.add_(noised_layer-data)
                noise_lists.append(noised_model)

            # Centralize the noise mask
            avg_params = copy.deepcopy(backdoor_update).state_dict()
            for _, data in avg_params.items():
                if self.td_layer_name in name:  # TODO 2025-03-03 git.V.32546: name is not defined?
                    data.fill_(0)
            for i in range(fl_adv_group_size):
                for name, data in noise_lists[i].state_dict().items():
                    if self.td_layer_name in name:
                        avg_params[name].add_(data / fl_adv_group_size)
            for i in range(fl_adv_group_size):
                for name, data in noise_lists[i].state_dict().items():
                    if self.td_layer_name in name:
                        data.add_(- avg_params[name])

            # Start optimization
            optimizer_lists = []
            for i in range(fl_adv_group_size):
                optimizer_lists.append(optim.SGD(noise_lists[i].parameters(),
                                                 lr=0.1,
                                                 weight_decay=0.00005,
                                                 momentum=0.9))  # TODO 2025-03-03 git.V.32546: decay = 0.00005, momentum = 0.9
            lagrange_mul = 1
            # print(gp_id, self.td_alpha)
            for _ in range(30):
                for i in range(fl_adv_group_size):
                    noise_lists[i].zero_grad()
                    noise_lists[i] = noise_lists[i].to(device)
                losses = compute_noise_loss(backdoor_update, noise_lists,
                                            self.td_alpha[gp_id], random_neurons[gp_id], lagrange_mul, self.args.model, device=device)
                for i in range(fl_adv_group_size):
                    losses[i].backward(retain_graph=True)
                    optimizer_lists[i].step()
                constrain, lagrange_mul = dual_ascent(0.1, noise_lists,
                                                      random_neurons[gp_id], lagrange_mul, self.td_layer_name, device=device)  # TODO 2025-03-03 git.V.32546: lagrange_step = 0.1
            # logger.info("Lagrange duality loss: {0} | Lagrange mul: {1}".format(constrain,
            #                                                                     lagrange_mul))
            for temp in noise_lists:
                noise_masks.append(temp)
        # logger.info("3DFed: Finish optimizing noise masks")

        # Shuffle the adversaries
        shuffled_adv = list(range(len(self.attack_clients)))
        np.random.shuffle(shuffled_adv)

        # Adding noise masks and implant indicators
        for nm_id, i in enumerate(shuffled_adv):
            saved_update = copy.deepcopy(backdoor_update).state_dict()
            gp_id = int(nm_id / fl_adv_group_size)
            for name, data in noise_masks[nm_id].state_dict().items():
                if self.td_layer_name in name:
                    sum_var = torch.FloatTensor(data.shape).fill_(0)
                    sum_var = sum_var.to(device)
                    for j in range(sum_var.shape[0]):
                        if j in random_neurons[gp_id]:
                            # sum_var.index_add_(0, torch.tensor(
                            #     [j], device=sum_var.device), data[j].view(-1))
                            sum_var[j].add_(data[j])
                    saved_update[name].add_(sum_var)

            # Implant the indicator
            # print(saved_update[self.td_ind_layer].shape)
            I = self.td_indicators[i]
            bounded_I = [min(I[i], saved_update[self.td_ind_layer].shape[i] - 1)
                         for i in range(len(I))]
            # print(bounded_I)
            saved_update[self.td_ind_layer][bounded_I[0]
                                            ][bounded_I[1]][bounded_I[2]][bounded_I[3]].mul_(1e5)
            # Avoid zero value
            if saved_update[self.td_ind_layer][bounded_I[0]][bounded_I[1]][bounded_I[2]][bounded_I[3]] == 0:
                if 'mnist' in self.args.dataset:
                    saved_update[self.td_ind_layer][bounded_I[0]
                                                    ][bounded_I[1]][bounded_I[2]][bounded_I[3]].add_(1e-2)
                else:
                    saved_update[self.td_ind_layer][bounded_I[0]
                                                    ][bounded_I[1]][bounded_I[2]][bounded_I[3]].add_(1e-3)
            self.td_indicators[i] = [I, saved_update[self.td_ind_layer]
                                     [bounded_I[0]][bounded_I[1]][bounded_I[2]][bounded_I[3]].item()]

            # save_name = '{0}/saved_updates/update_{1}.pth'.format(
            #     params.folder_path, i)
            # torch.save(saved_update, save_name)

    def _prepare_batch(self, fs, labels, device):
        """Prepare batch data and move to device"""
        fs = fs.to(dtype=torch.float, device=device, non_blocking=True)
        labels = labels.to(dtype=torch.long, device=device,
                           non_blocking=True).squeeze()
        return fs, labels

    def _generate_triggers(self, fs):
        """Generate and process triggers"""
        triggers = self.invs_TriggerNet(fs)
        triggers_l2norm = torch.mean(torch.abs(triggers))

        num_channel = fs.size(1)
        num_classes = self.args.num_classes

        # Split triggers into two parts for embedding
        triggers_part1 = triggers[:, :num_channel * num_classes]
        triggers_part2 = triggers[:, num_channel *
                                  num_classes:2 * num_channel * num_classes]

        triggers = self.invs_EmbeddingNet(triggers_part1, triggers_part2)
        triggers = triggers / 255.0  # Normalize

        return triggers.reshape(-1, 3, self.args.input_size, self.args.input_size), triggers_l2norm

    def _create_poisoned_samples(self, fs, triggers):
        """Create poisoned samples by adding triggers"""
        batch_size = fs.shape[0]
        num_classes = self.args.num_classes
        input_size = self.args.input_size

        # Expand fs to match trigger dimensions
        fs_expanded = fs.unsqueeze(1).expand(
            batch_size, num_classes, 3, input_size, input_size)
        fs_reshaped = fs_expanded.reshape(-1, 3, input_size, input_size)

        return fs_reshaped + triggers

    def _compute_losses(self, out, f, fs_batch_size, labels, triggers_l2norm, a, b):
        """Compute all losses"""
        # Original classification loss
        loss_ori = F.cross_entropy(out[:fs_batch_size], labels)

        # Poisoned samples loss
        loss_p = F.cross_entropy(out[fs_batch_size:], self.invs_Target_labels)

        # Feature loss
        loss_f = F.l1_loss(f[fs_batch_size:], self.invs_feature_r)

        return loss_ori, loss_p, loss_f, triggers_l2norm

    def invisibleTrain(self, dataloader, model, lr=1e-3, epochs=10, a=0.3, b=0.1, c=0.3, device='cuda'):
        model.train()
        optimizer_net = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.5)
        optimizer_map = torch.optim.Adam(
            self.invs_TriggerNet.parameters(), lr=1e-3)
        self.invs_TriggerNet.train()

        # main task train

        # for epoch in range(epochs):
        #     for batch_idx, (fs, labels) in enumerate(dataloader):
        #         fs = fs.to(dtype=torch.float).to(device)
        #         fs_copy = copy.deepcopy(fs)
        #         num_channel = fs.size(1)
        #         # # Reshape to group every 8 samples and take mean
        #         # reduced_batch_size = fs.size(0)
        #         # fs_reshaped = fs[:reduced_batch_size * 8].view(reduced_batch_size, 8, num_channel,
        #         #                                             self.args.input_size, self.args.input_size)
        #         # fs_reduced = fs_reshaped.mean(dim=1)  # Take mean across the 8 samples
        #         # Triggers = self.invs_TriggerNet(fs_reduced)
        #         Triggers = self.invs_TriggerNet(fs)
        #         # print(f"Triggers shape: {Triggers.shape}")
        #         Triggersl2norm = torch.mean(torch.abs(Triggers))
        #         Triggers = self.invs_EmbeddingNet(Triggers[:, 0:num_channel*self.args.num_classes, :, :],
        #                                           Triggers[:, num_channel*self.args.num_classes:2*num_channel*self.args.num_classes, :, :])
        #         Triggers = (Triggers)/255
        #         Triggers = Triggers.reshape(-1, 3,
        #                                     self.args.input_size, self.args.input_size)
        #         # print(f"Triggers embedding shape: {Triggers.shape}")
        #         fs = fs.unsqueeze(1).expand(fs.shape[0], self.args.num_classes, 3, self.args.input_size,
        #                                     self.args.input_size).reshape(-1, 3, self.args.input_size, self.args.input_size)
        #         fs_poison = fs + Triggers
        #         # fsp_reshaped = fs_poison.view(reduced_batch_size, -1, num_channel,
        #         #                                             self.args.input_size, self.args.input_size)
        #         # fsp_reduced = fsp_reshaped.mean(dim=1)  # Take mean across the 8 samples
        #         labels = torch.as_tensor(
        #             labels, dtype=torch.long, device=self.args.device).squeeze()
        #         imgs_input = torch.cat((fs_copy, fs_poison), 0)
        #         optimizer_net.zero_grad()
        #         optimizer_map.zero_grad()
        #         out, f = model.forward_feature(imgs_input)
        #         # print(f"imgs_input shape: {imgs_input.shape}")
        #         # print(f"out shape: {out.shape}, f shape: {f.shape}")

        #         # print(f"fs_copy shape: {fs_copy.shape}")
        #         # print(f"self.invs_feature_r shape: {self.invs_feature_r.shape}")

        #         loss_f = torch.nn.L1Loss()(
        #             f[fs_copy.shape[0]::, :], self.invs_feature_r)
        #         loss_ori = torch.nn.CrossEntropyLoss()(
        #             out[0:labels.shape[0], :], labels)
        #         loss_p = torch.nn.CrossEntropyLoss()(
        #             out[labels.shape[0]::], self.invs_Target_labels)
        #         loss = loss_ori + (loss_p + loss_f * a + Triggersl2norm * b) * c
        #         loss.backward()
        #         optimizer_net.step()
        #         optimizer_map.step()

        # Optimized training loop
        for epoch in range(epochs):
            for batch_idx, (fs, labels) in enumerate(dataloader):
                # Prepare batch
                fs, labels = self._prepare_batch(fs, labels, device)
                fs_copy = fs.clone()  # More efficient than deepcopy

                # Generate triggers
                triggers, triggers_l2norm = self._generate_triggers(fs)

                # Create poisoned samples
                fs_poison = self._create_poisoned_samples(fs, triggers)

                # Combine original and poisoned samples
                imgs_input = torch.cat((fs_copy, fs_poison), dim=0)

                # Zero gradients
                optimizer_net.zero_grad()
                optimizer_map.zero_grad()

                # Forward pass
                out, f = model.forward_feature(imgs_input)

                # Compute losses
                loss_ori, loss_p, loss_f, triggers_l2norm = self._compute_losses(
                    out, f, fs_copy.shape[0], labels, triggers_l2norm, a, b
                )

                # Combined loss
                loss = loss_ori + (loss_p + loss_f * a +
                                   triggers_l2norm * b) * c

                # Backward pass and optimization
                loss.backward()
                optimizer_net.step()
                optimizer_map.step()

    def invisibleTrain_with_freq_eval(self, dataloader, model, lr=1e-3, epochs=10, a=0.3, b=0.1, c=0.3, device='cuda'):
        from utils.assess import plot_frequency_comparison
        model.train() 
        optimizer_net = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.5)
        optimizer_map = torch.optim.Adam(
            self.invs_TriggerNet.parameters(), lr=1e-3)
        self.invs_TriggerNet.train()

        # Optimized training loop
        for epoch in range(epochs):
            for batch_idx, (fs, labels) in enumerate(dataloader):
                # Prepare batch
                fs, labels = self._prepare_batch(fs, labels, device)
                fs_copy = fs.clone()  # More efficient than deepcopy

                # Generate triggers
                triggers, triggers_l2norm = self._generate_triggers(fs)

                # Create poisoned samples
                fs_poison = self._create_poisoned_samples(fs, triggers)


                # Combine original and poisoned samples
                imgs_input = torch.cat((fs_copy, fs_poison), dim=0)

                # Zero gradients
                optimizer_net.zero_grad()
                optimizer_map.zero_grad()

                # Forward pass
                out, f = model.forward_feature(imgs_input)

                # Compute losses
                loss_ori, loss_p, loss_f, triggers_l2norm = self._compute_losses(
                    out, f, fs_copy.shape[0], labels, triggers_l2norm, a, b
                )

                # Combined loss
                loss = loss_ori + (loss_p + loss_f * a +
                                   triggers_l2norm * b) * c

                # Backward pass and optimization
                loss.backward()
                optimizer_net.step()
                optimizer_map.step()

                    
        detailed_metrics = plot_frequency_comparison(fs_copy[0].cpu().detach().permute(1, 2, 0).numpy() * 255, fs_poison[0].cpu().detach().permute(1, 2, 0).numpy() * 255)

    def invisibleEval(self, model, dataloader, device='cuda'):
        model.eval()
        self.invs_TriggerNet.eval()
        Correct = 0
        Loss = 0
        Tot = 0
        L1 = 0
        LF = 0
        for batch_idx, (fs, labels) in enumerate(dataloader):
            fs = fs.to(dtype=torch.float).to(device)
            reduced_batch_size = fs.size(0)
            num_channel = fs.size(1)
            Triggers = self.invs_TriggerNet(fs)
            Triggers = self.invs_EmbeddingNet(Triggers[:, 0:num_channel*self.args.num_classes, :, :],
                                              Triggers[:, num_channel*self.args.num_classes:num_channel*2*self.args.num_classes, :, :])
            Triggers = torch.round(Triggers)/255
            fs = fs.unsqueeze(1).expand(fs.shape[0], self.args.num_classes, num_channel, self.args.input_size,
                                        self.args.input_size).reshape(-1, num_channel, self.args.input_size, self.args.input_size)
            Triggers = Triggers.reshape(-1, num_channel,
                                        self.args.input_size, self.args.input_size)
            fs = fs + Triggers
            fs = torch.clip(fs, min=0, max=1)
            # fsp_reshaped = fs.view(reduced_batch_size, -1, num_channel,
            #                                                 self.args.input_size, self.args.input_size)
            # fsp_reduced = fsp_reshaped.mean(dim=1)  # Take mean across the 8 samples
            # fsp_reduced = torch.clip(fsp_reduced, min=0, max=1)
            out, feat = model.forward_feature(fs)
            loss_f = torch.nn.L1Loss()(feat, self.invs_feature_r)
            loss = torch.nn.CrossEntropyLoss()(out, self.invs_Target_labels)
            _, predicts = out.max(1)
            Correct += predicts.eq(self.invs_Target_labels).sum().item()
            Loss += loss.item()
            Tot += fs.shape[0]
            L1 += torch.sum(torch.abs(Triggers*255)).item()
            LF += loss_f.item()
        Acc = 100*Correct/Tot
        LF = LF / len(dataloader)
        return Acc, Loss, L1, LF
    
    def invisibleEval_withFreqAnalysis(self, model, dataloader, device='cuda', img_path='./'):
        from utils.assess import plot_frequency_comparison
        from torchvision import transforms


        def reverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            """Reverse the normalization transform"""
            mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
            std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
            
            # Reverse: x_original = x_normalized * std + mean
            return tensor * std + mean
        
        model.eval()
        self.invs_TriggerNet.eval()
        Correct = 0
        Loss = 0
        Tot = 0
        L1 = 0
        LF = 0
        for batch_idx, (fs, labels) in enumerate(dataloader):
            fs = fs.to(dtype=torch.float).to(device)

            reduced_batch_size = fs.size(0)
            num_channel = fs.size(1)
            Triggers = self.invs_TriggerNet(fs)
            Triggers = self.invs_EmbeddingNet(Triggers[:, 0:num_channel*self.args.num_classes, :, :],
                                            Triggers[:, num_channel*self.args.num_classes:num_channel*2*self.args.num_classes, :, :])
            Triggers = torch.round(Triggers)/255
            fs = fs.unsqueeze(1).expand(fs.shape[0], self.args.num_classes, num_channel, self.args.input_size,
                                        self.args.input_size).reshape(-1, num_channel, self.args.input_size, self.args.input_size)
            Triggers = Triggers.reshape(-1, num_channel,
                                        self.args.input_size, self.args.input_size)
            fs_copy = fs.clone()
            fs_copy = torch.clip(fs_copy, min=0, max=1)
            fs = fs + Triggers
            fs = torch.clip(fs, min=0, max=1)
            # fsp_reshaped = fs.view(reduced_batch_size, -1, num_channel,
            #                                                 self.args.input_size, self.args.input_size)
            # fsp_reduced = fsp_reshaped.mean(dim=1)  # Take mean across the 8 samples
            # fsp_reduced = torch.clip(fsp_reduced, min=0, max=1)
            out, feat = model.forward_feature(fs)
            loss_f = torch.nn.L1Loss()(feat, self.invs_feature_r)
            loss = torch.nn.CrossEntropyLoss()(out, self.invs_Target_labels)
            _, predicts = out.max(1)
            Correct += predicts.eq(self.invs_Target_labels).sum().item()
            Loss += loss.item()
            Tot += fs.shape[0]
            L1 += torch.sum(torch.abs(Triggers*255)).item()
            LF += loss_f.item()
        Acc = 100*Correct/Tot
        LF = LF / len(dataloader)
        detailed_metrics = plot_frequency_comparison(fs_copy[0].cpu().detach().permute(1, 2, 0).numpy() * 255, fs[0].cpu().detach().permute(1, 2, 0).numpy() * 255, img_path=img_path)
        return Acc, Loss, L1, LF

    def getDataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    # def threeDfed(self, dataset_str, fl_number_of_adversaries, fl_adv_group_size, noise_mask_alpha):
    #     ind_layer = 'conv2.weight' if 'MNIST' in dataset_str else 'layer4.1.conv2.weight'
    #     layer_name = 'fc2' if 'MNIST' in dataset_str else 'fc'
    #     group_size = math.ceil(
    #         fl_number_of_adversaries / fl_adv_group_size)
    #     alpha = [random.uniform(
    #         noise_mask_alpha, 1.) for _ in range(group_size)]

    #     global_update = get_fl_update(
    #         global_model, last_global_model)

    #     return

    def colludeMask(self, model: nn.Module, num: int):
        """
        generate n random mask that sums up equal to 0
        """
        mask_candidates = {i: [] for i in range(num)}
        for name, params in model.named_parameters():
            if 'weight' in name:
                masks = [torch.randn_like(params) for _ in range(num)]
                sum_masks = sum(masks)
                mean_masks = sum_masks / num
                zero_sum_masks = [mask - mean_masks for mask in masks]
                # Normalize to reduce variance
                zero_sum_masks = [
                    mask / torch.std(mask) for mask in zero_sum_masks]
                for i, mask in enumerate(zero_sum_masks):
                    mask_candidates[i].append(mask)

        return mask_candidates

    def edgeTrain(self, model, running_dataloader, epochs=10, device='cuda'):
        """
        Train the edge dataset with the given model and dataloader.
        """
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        model.train()

        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(running_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def mirage_train(self, model, train_loader, client_id, epochs=10):
        '''
        poisoning training process

        :param iteration:
        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''
        cache_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(cache_model.parameters(), lr=self.mirage_params['poisoned_lr'],
                                    momentum=self.mirage_params['poisoned_momentum'],
                                    weight_decay=self.mirage_params['poisoned_weight_decay'])

        trigger_ = self.mirage_search_trigger(
            cache_model, train_loader, client_id)
        self.mirage_trigger_set[client_id] = trigger_

        model.train()
        cache_model.train()

        for epoch in range(epochs):

            cache_model.train()
            total_loss = 0.
            counter = 0.
            correct = 0
            total = 0
            for batch_idx, batch in enumerate(train_loader):
                counter += 1

                inputs, labels = batch
                inputs, labels = inputs.to(self.mirage_params["run_device"]), labels.to(self.mirage_params["run_device"])
                outputs = cache_model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                inputs, labels = self.mirage_poisoned_batch_injection(batch, trigger=self.mirage_trigger_set[client_id],
                                                          mask=self.mirage_mask_set[client_id], is_eval=False,
                                                          label_swap=self.mirage_params["poison_label_swap"][client_id], client_id=client_id)

                inputs, labels = inputs.to(self.mirage_params["run_device"]), labels.to(
                    self.mirage_params["run_device"])
                outputs = cache_model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            print(f"Client {client_id}, Epoch {epoch}: Loss {total_loss/counter}, Acc {acc}")
        return cache_model

    def mirage_test_model(self, model, test_dataloader, is_poisoned=False, trigger=None, mask=None,
                        label_swap=0, client_id=0):
        '''
        test model
        :param iteration: current iterations
        :param test_dataloader:  test dataloader
        :param is_poisoned: is poison
        :param trigger: trigger for testing, shape, (channel, height, width)
        :param mask: trigger mask, shape (channel, height, width)
        :param label_swap: labels
        :return: results, acc, loss
        '''
        model.eval()
        with torch.no_grad():
            total_loss = 0.
            total_correct = 0.
            total_num = 0.

            criterion = nn.CrossEntropyLoss(reduction='sum')
            for i, batch in enumerate(test_dataloader):
                if is_poisoned:
                    # 如果需要测试ASR，则需要对batch进行投毒
                    sample_indices = ~(batch[1] == label_swap)
                    samples = batch[0][sample_indices]
                    labels = batch[1][sample_indices]
                    batch = self.mirage_poisoned_batch_injection((samples, labels), trigger, mask, is_eval=True,
                                                     label_swap=label_swap, client_id=client_id)
                data, target = batch
                data, target = data.to(self.mirage_params["run_device"]), target.to(
                    self.mirage_params["run_device"])
                output = model(data)
                loss = criterion(output, target)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_loss += loss.item()
                total_num += data.size(0)
        acc = 100 * total_correct / total_num
        loss = total_loss / total_num
        return acc, loss

    def mirage_search_trigger(self, model, train_loader, client_id):
        '''
        optimize trigger
        :param model:
        :param train_loader:
        :param client_id:
        :return:
        '''
        model.eval()
        local_train_loader = copy.deepcopy(train_loader)
        trigger_ = copy.deepcopy(self.mirage_trigger_set[client_id])
        mask_ = copy.deepcopy(self.mirage_mask_set[client_id])
        ce_loss = nn.functional.cross_entropy
        cos_loss = nn.CosineSimilarity(dim=1, eps=1e-08)
        feature_extractor = copy.deepcopy(model)
        feature_extractor.linear = torch.nn.Sequential()
        t = copy.deepcopy(trigger_)
        t.requires_grad_(True)

        for iters in range(self.mirage_params["trigger_search_no_times"]):
            dataloader_discriminator = self.mirage_generate_discriminator_dataloader(
                model, local_train_loader, t, mask_, client_id)
            total_loss = 0.
            trigger_optim = torch.optim.Adam(
                [t], lr=self.mirage_params["trigger_lr"], weight_decay=5e-4)
            counter = 0
            loss_adv = 0.
            loss_acc = 0.
            model_discriminator = self.mirage_get_discriminator(
                model, dataloader_discriminator)

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.mirage_params["run_device"]), targets.to(
                    self.mirage_params["run_device"])
                batch_clean_indices = targets == self.mirage_params["poison_label_swap"][client_id]
                if batch_clean_indices.sum() == 0:
                    continue
                counter += 1
                batch_backdoor_indices = ~batch_clean_indices
                backdoor_inputs = inputs[batch_backdoor_indices]
                backdoor_targets = targets[batch_backdoor_indices]

                # Create backdoor inputs with current trigger
                backdoor_inputs, backdoor_targets = self.mirage_poisoned_batch_injection(
                    (backdoor_inputs, backdoor_targets),
                    trigger=t, mask=mask_, is_eval=False,
                    label_swap=self.mirage_params["poison_label_swap"][client_id])
                backdoor_inputs = backdoor_inputs.to(
                    self.mirage_params["run_device"])

                # Forward pass
                backdoor_pred_disc = model_discriminator(backdoor_inputs)
                loss_discriminator = ce_loss(
                    backdoor_pred_disc,
                    torch.zeros(len(backdoor_pred_disc), device=self.mirage_params["run_device"]).long())

                backdoor_pred = model(backdoor_inputs)
                loss_asr = ce_loss(backdoor_pred, backdoor_targets)
                loss_sim = cos_loss(backdoor_pred, model(
                    inputs[batch_backdoor_indices])).mean()

                loss = loss_discriminator + loss_asr + loss_sim
                total_loss += loss.item()

                if loss is not None and loss.item() != 0.:
                    trigger_optim.zero_grad()
                    loss.backward()  # Remove retain_graph=True unless necessary
                    trigger_optim.step()

                    # Apply constraints without in-place operations
                    with torch.no_grad():
                        # This is okay in no_grad context
                        t.clamp_(min=-2, max=2)

        
        return t.detach()  # Remove underscore - detach_() doesn't exist


    def mirage_init_trigger_mask(self, train_dataloader):
        sample_data = next(iter(train_dataloader))[0][0]
        n_adversaries = self.mirage_params["no_of_participants"]
        # 固定位置触发器，pixel (3,3) , fixed trigger
        if self.mirage_params["poisoned_pattern_choose"] == 1:
            trigger_set = []
            for i in range(n_adversaries):
                if self.mirage_params["malicious_train_algo"] == "A3FL" or self.mirage_params["malicious_train_algo"] == "Mirage":
                    trigger = torch.ones(
                        sample_data.shape, device=self.mirage_params["run_device"]) * 0.5
                else:
                    trigger = (torch.rand(sample_data.shape) - 0.5) * 2
                trigger_set.append(trigger.to(self.mirage_params["run_device"]))
            
            # Generate n_adversaries masks for different positions
            mask_set = []
            trigger_size = self.mirage_params["trigger_size"]
            
            # Define possible trigger positions
            positions = [
                (1, 1),  # top-left
                (-1 - trigger_size, 1),  # bottom-left
                (-1 - trigger_size, -1 - trigger_size),  # bottom-right
                (1, -1 - trigger_size),  # top-right
                (13, 13),  # center (or any other position you prefer)
            ]
            
            for i in range(n_adversaries):
                mask = torch.zeros_like(sample_data).to(self.mirage_params["run_device"])
                
                # Cycle through positions if n_adversaries > len(positions)
                pos_idx = i % len(positions)
                start_h, start_w = positions[pos_idx]
                
                # Handle negative indexing for end positions
                if start_h < 0:
                    end_h = start_h
                    start_h = sample_data.shape[1] + start_h
                else:
                    end_h = start_h + trigger_size
                    
                if start_w < 0:
                    end_w = start_w
                    start_w = sample_data.shape[2] + start_w
                else:
                    end_w = start_w + trigger_size
                
                mask[:, start_h:end_h, start_w:end_w] = 1
                mask_set.append(mask)
            
            # Generate test_trigger_mask for all adversaries
            test_trigger_mask = [trigger_set[i] * mask_set[i] for i in range(n_adversaries)]
            
        elif self.mirage_params["poisoned_pattern_choose"] == 2:
            trigger_set = []
            mask_set = []
            dataloader = train_dataloader
            sample_data = next(iter(dataloader))[0][0]
            n_adversaries = self.mirage_params["no_of_adversaries"]
            for i in range(n_adversaries):
                if self.mirage_params["malicious_train_algo"] == "A3FL" or self.mirage_params["malicious_train_algo"] == "Mirage":
                    trigger = torch.ones(
                        sample_data.shape, device=self.mirage_params["run_device"]) * 0.5
                else:
                    trigger = (torch.rand(
                        sample_data.shape, device=self.mirage_params["run_device"]) - 0.5) * 2
                mask = torch.ones(sample_data.shape).to(
                    self.mirage_params["run_device"])
                # visualize(trigger, f"trigger_{i}")
                trigger_set.append(trigger)
                mask_set.append(mask)

        # 固定位置触发器，pixel (3,3) , fixed trigger
        elif self.mirage_params["poisoned_pattern_choose"] == 3:
            trigger_set = []
            for i in range(n_adversaries):
                trigger = (torch.rand(sample_data.shape) - 0.5) * 2
                trigger_set.append(trigger.to(
                    self.mirage_params["run_device"]))

            mask1 = torch.zeros_like(sample_data).to(
                self.mirage_params["run_device"])
            mask2 = torch.zeros_like(sample_data).to(
                self.mirage_params["run_device"])
            mask3 = torch.zeros_like(sample_data).to(
                self.mirage_params["run_device"])
            mask4 = torch.zeros_like(sample_data).to(
                self.mirage_params["run_device"])
            mask5 = torch.zeros_like(sample_data).to(
                self.mirage_params["run_device"])
            trigger_size = self.mirage_params["trigger_size"]
            mask1[:, 1:1 + trigger_size, 1:1 + trigger_size] = 1
            mask2[:, -1 - trigger_size:-1, 1:1 + trigger_size] = 1
            mask3[:, -1 - trigger_size:-1, -1 - trigger_size:-1] = 1
            mask4[:, 1:1 + trigger_size, -1 - trigger_size:-1] = 1
            mask5[:, 13:13 + trigger_size, 13:13 + trigger_size] = 1
            mask_set = [mask1, mask2, mask3, mask4, mask5]
            # 设置torch输出全部内容

        return trigger_set, mask_set

    def mirage_init_test_sample_cache(self, test_dataloader):
        samples_per_class_cache = {i: [torch.tensor([]), torch.tensor(
            [])] for i in range(self.mirage_params["class_num"])}
        for batch in test_dataloader:
            inputs, labels = batch
            for i in range(self.mirage_params["class_num"]):
                indices_of_class_i = labels == i
                if indices_of_class_i.sum() > 0 and len(samples_per_class_cache[i][0]) < 200:
                    samples_per_class_cache[i][0] = torch.cat(
                        (samples_per_class_cache[i][0],
                         inputs[indices_of_class_i]),
                        dim=0)
                    samples_per_class_cache[i][1] = torch.cat(
                        (samples_per_class_cache[i][1],
                         labels[indices_of_class_i]),
                        dim=0)
        print(
            {f"class_{i}: {len(samples_per_class_cache[i][0])}" for i in range(self.mirage_params["class_num"])})
        return samples_per_class_cache

    def mirage_generate_discriminator_dataloader(self, model, train_loader, trigger_, mask_, client_id):
        '''
        Generate discriminator trainset where clean samples have label 0 and poisoned samples have label 1
        :param model: The model to use for sample selection
        :param train_loader: Training data loader
        :param trigger_: Trigger pattern for poisoning
        :param mask_: Mask for trigger application
        :param client_id: Client identifier
        :return: DataLoader for discriminator training
        '''
        class_num = self.mirage_params["class_num"]
        
        # Initialize with empty lists instead of tensors to avoid device issues
        samples_per_class = {i: [] for i in range(class_num)}
        
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.mirage_params["run_device"])
        label_list = [0 for _ in range(class_num)]
        
        # Put model in evaluation mode for consistent behavior
        model.eval()
        
        with torch.no_grad():  # No gradients needed for sample collection
            for index, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.mirage_params["run_device"]), labels.to(self.mirage_params["run_device"])
                
                for class_ind in range(class_num):
                    indices = labels == class_ind
                    label_list[class_ind] += indices.sum().item()  # Use .item() for proper counting
                    if indices.any():  # Only process if there are samples of this class
                        class_samples = inputs[indices]
                        if len(samples_per_class[class_ind]) == 0:
                            samples_per_class[class_ind] = class_samples
                        else:
                            samples_per_class[class_ind] = torch.cat(
                                (samples_per_class[class_ind], class_samples), dim=0)
        
        target_class = self.mirage_params["poison_label_swap"][client_id]
        
        # Select representative samples based on loss
        with torch.no_grad():
            for i in range(class_num):
                if len(samples_per_class[i]) == 0:
                    continue
                    
                sample = samples_per_class[i]
                outputs = model(sample)
                tmp_label = torch.full(
                    (len(outputs),), i, dtype=torch.long, device=self.mirage_params["run_device"])
                loss_sort_by_samples = criterion(outputs, tmp_label)
                
                # Determine number of samples to select
                max_samples = self.mirage_params.get("discriminator_train_samples_pre_class", len(outputs))
                samples_selected_len = min(max_samples, len(outputs))
                
                if i == target_class:
                    samples_selected_len = len(outputs)  # Use all target class samples
                
                # Select samples with lowest loss (most confident predictions)
                _, indices = torch.topk(loss_sort_by_samples, samples_selected_len, largest=False)
                representative_samples = sample[indices]
                samples_per_class[i] = representative_samples
        
        # Initialize result tensors
        samples_discriminator_dataloader = []
        labels_discriminator_dataloader = []
        
        # Add poisoned samples (label 1) - all non-target classes
        for i in range(class_num):
            if i == target_class or len(samples_per_class[i]) == 0:
                continue
                
            samples = samples_per_class[i]
            # Create dummy labels for poisoning function
            dummy_labels = torch.full((len(samples),), i, dtype=torch.long, device=self.mirage_params["run_device"])
            
            # Apply poisoning
            poisoned_sample, _ = self.mirage_poisoned_batch_injection(
                (samples, dummy_labels), 
                trigger=trigger_, 
                mask=mask_, 
                is_eval=True,
                label_swap=target_class
            )
            
            # Label poisoned samples as 1
            poison_labels = torch.ones(len(poisoned_sample), dtype=torch.long, device=self.mirage_params["run_device"])
            
            samples_discriminator_dataloader.append(poisoned_sample)
            labels_discriminator_dataloader.append(poison_labels)
        
        # Add clean target class samples (label 0)
        if len(samples_per_class[target_class]) > 0:
            clean_labels = torch.zeros(
                len(samples_per_class[target_class]), 
                dtype=torch.long, 
                device=self.mirage_params["run_device"]
            )
            samples_discriminator_dataloader.append(samples_per_class[target_class])
            labels_discriminator_dataloader.append(clean_labels)
        
        # Concatenate all samples and labels
        if samples_discriminator_dataloader:  # Check if we have any data
            all_samples = torch.cat(samples_discriminator_dataloader, dim=0)
            all_labels = torch.cat(labels_discriminator_dataloader, dim=0)
            
            discriminator_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(all_samples, all_labels),
                batch_size=self.mirage_params["discriminator_batch_size"], 
                shuffle=True
            )
        else:
            # Return empty dataloader if no data
            empty_tensor = torch.empty(0, device=self.mirage_params["run_device"])
            empty_labels = torch.empty(0, dtype=torch.long, device=self.mirage_params["run_device"])
            discriminator_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(empty_tensor, empty_labels),
                batch_size=self.mirage_params["discriminator_batch_size"], 
                shuffle=True
            )
        
        return discriminator_dataloader

    def mirage_get_discriminator(self, model, discriminator_dataloader):
        discriminator_ = copy.deepcopy(model)
        if "lenet" in self.mirage_params["model_type"].lower():
            discriminator_.fc5 = torch.nn.Sequential(
                torch.nn.Linear(84, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "resnet" in self.mirage_params["model_type"].lower():
            discriminator_.linear = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.linear.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "vgg" in self.mirage_params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "mobilenet" in self.mirage_params["model_type"].lower():
            discriminator_.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator_.classifier[1].in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )

        for name, param in discriminator_.named_parameters():
            if self.mirage_params["classifier_name"] not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        discriminator_optimizer = torch.optim.SGD(discriminator_.parameters(), lr=self.mirage_params["discriminator_lr"],
                                                  momentum=self.mirage_params['discriminator_momentum'],
                                                  weight_decay=self.mirage_params['discriminator_weight_decay'])

        discriminator_criterion = nn.CrossEntropyLoss().to(
            self.mirage_params["run_device"])

        discriminator_ = discriminator_.to(self.mirage_params["run_device"])

        for iter in range(self.mirage_params["discriminator_train_no_times"]):
            total_loss = 0.
            for batch in discriminator_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.mirage_params["run_device"]), labels.to(
                    self.mirage_params["run_device"])
                outputs = discriminator_(inputs)
                loss = discriminator_criterion(outputs, labels)
                discriminator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                total_loss += loss.item()
                discriminator_optimizer.step()
        discriminator_.eval()

        return discriminator_

    def mirage_poisoned_batch_injection(self, batch, trigger, mask, is_eval=False, client_id=0, label_swap=None, mode="all"):
        '''
        对batch数据进行投毒，并返回新的batch数据

        :param batch: 需要投毒的batch
        :param trigger: trigger tensor, shape为(channel, height, width)
        :param mask: mask tensor, shape为(channel, height, width)
        :param is_eval: 是否为eval模式
        :param client_id: 客户端id (for attacker)
        :param label_swap: target label (for attacker)
        :param mode: 模式，是否在注入时排除干净目标类, value: "all"/"escape_clean"
        :return: poisoned batch
        '''
        if label_swap is None:
            label_swap = self.mirage_params["poison_label_swap"][client_id]
        data, label = copy.deepcopy(batch)
        
        if mode == "all" and is_eval == False:
            # FIX: Limit to actual batch size
            actual_poison_len = min(self.mirage_params["poisoned_len"], len(data))
            poison_indices = list(range(actual_poison_len))
        elif mode == "escape_clean" and is_eval == False:
            clean_indices = np.nonzero(label != label_swap)[0]
            # FIX: Limit to available clean samples and batch size
            actual_poison_len = min(self.mirage_params["poisoned_len"], len(clean_indices), len(data))
            poison_indices = clean_indices[:actual_poison_len].tolist()
        elif is_eval == True:
            poison_indices = list(range(len(label)))
        else:
            raise ValueError("mode should be 'all' or 'escape_clean'")
        
        # Now these assertions should pass
        assert len(poison_indices) <= len(data)
        assert len(poison_indices) == 0 or max(poison_indices) < len(data)
        assert max(poison_indices) < len(data)

        poisoned_len = self.mirage_params["poisoned_len"] if not is_eval else len(
            label)
        data = data.to(self.mirage_params["run_device"])
        trigger = trigger.to(self.mirage_params["run_device"])
        mask = mask.to(self.mirage_params["run_device"])
        if self.mirage_params["poisoned_pattern_choose"] == 1:  # 1 -> pixel block
            data[poison_indices] = trigger * mask + \
                (1 - mask) * data[poison_indices]
        elif self.mirage_params["poisoned_pattern_choose"] == 2:  # 1 -> blend trigger
            data[poison_indices] = trigger * self.mirage_params["blend_alpha"] + (1 - self.mirage_params["blend_alpha"]) * data[
                poison_indices]
        label[poison_indices] = label_swap
        return data, label


def setBlendTrigger(input_size):
    height, width = input_size, input_size
    num_elements = height * width
    num_neg_tens = num_elements // 2  # 50% of the values

    # Flattened mask
    mask = torch.empty(num_elements)

    # Indices to set as -10
    neg_indices = torch.randperm(num_elements)[:num_neg_tens]
    mask[neg_indices] = -10.0

    # Remaining indices
    all_indices = torch.arange(num_elements)
    pos_indices = all_indices[~torch.isin(all_indices, neg_indices)]

    # Assign random values in [0, 0.1]
    mask[pos_indices] = torch.rand(len(pos_indices)) * 0.1

    mask = mask.view(height, width)

    return mask
