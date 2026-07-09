
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd


class GetSubnet(autograd.Function):
    '''Subnetwork forward from hidden networks'''
    @staticmethod
    def forward(ctx, scores):
        return (scores >= 0).float() # Use 0 as threshold. this is related to the signed_constant initialization

    @staticmethod
    def backward(ctx, g):
        # Send the gradient g straight-through on the backward pass. so that it is trainable
        return g

class SoftMaskedLayer(nn.Module):
    def __init__(self, layer_type, *args, num_tasks=2, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks

        if layer_type == "linear":
            self.layer = NNSubnetworkSoftmask(
                *args, num_tasks=num_tasks, **kwargs)
            self.in_features = args[0]
        elif layer_type == "conv2d":
            self.layer = NNSubnetworkSoftmaskConv2d(
                *args, num_tasks=num_tasks, **kwargs)
        elif layer_type == "bn":
            self.layer = NNSubnetworkSoftmaskBN(
                *args, num_tasks=num_tasks, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    # def copy_score(self, ft_task, target_task=-1):
    #     self.layer.copy_score(ft_task, target_task)
    def copy_score(self, source_task, target_task=-1):
        target_task = (source_task+1) % self.num_tasks if target_task == -1 else target_task % self.num_tasks
        with torch.no_grad():
            # self.scores[target_task].copy_(self.scores[source_task].clone())
            self.layer.scores[target_task].copy_(self.layer.scores[source_task].clone())

    def unfreeze_weight(self, status=False):
        self.layer.weight.requires_grad = status

    def unfreeze_scores(self, status=False):
        self.layer.scores.requires_grad = status

    def forward(self, x):
        return self.layer(x)

    def __repr__(self):
        return f"SoftMaskedLayer({self.layer.__repr__()})"


class NNSubnetworkSoftmask(nn.Linear):
    
    """
    impt_mask: to store the importance mask (temproal) for each task.
                With compute_mask_impt as true, the alphas is updated by forward propogation
                then the tss_impt_dict is updated by the tss grad.
                With tss_impt_dict, the score weight is updated in the next training.
                However, the impt_mask is not actually used in the forward propogation.
    """
    def __init__(self, *args, num_tasks=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.ft_task = 0
        self.scores = nn.ParameterList(
            [
                nn.Parameter(self.mask_init())
                for _ in range(num_tasks)
            ]
        )
        self.impt_mask = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.weight.size())
                             ).requires_grad_(False)
                for _ in range(self.num_tasks)
            ]
        )
        self.compute_mask_impt = False

        # Alphas are used later when we compute the importance of the scores.
        self.alphas = nn.Parameter(torch.ones(self.weight.size()))

        # Keep weights untrained
        # self.weight.requires_grad = False
        self.signed_constant()

    # def copy_score(self, ft_task, target_task=-1):
    #     target_task = ft_task+1 if target_task == -1 else target_task
    #     with torch.no_grad():
    #         self.scores[target_task].copy_(self.scores[ft_task].clone())

    def mask_init(self):
        scores = torch.Tensor(self.weight.size())
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
        return scores

    def signed_constant(self):
        fan = nn.init._calculate_correct_fan(self.weight, 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * std

    def forward(self, x):
        if self.compute_mask_impt:  # Whether it is to compute the importance
            selected_mask = self.scores[self.ft_task]

            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet * self.alphas
            x = F.linear(x, w, self.bias)

        else:
            selected_mask = self.scores[self.ft_task]
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet
            x = F.linear(x, w, self.bias)

        return x

    def __repr__(self):
        return f"NNSubnetworkSoftmask({self.weight.size(0)}, {self.weight.size(1)})"


class NNSubnetworkSoftmaskConv2d(nn.Conv2d):
    def __init__(self, *args, num_tasks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.ft_task = 0

        # Initialize scores and importance masks for each task
        self.scores = nn.ParameterList(
            [
                nn.Parameter(self.mask_init())
                for _ in range(num_tasks)
            ]
        )
        self.impt_mask = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(self.weight)
                             ).requires_grad_(False)
                for _ in range(num_tasks)
            ]
        )
        self.compute_mask_impt = False

        # Alphas are used later to compute the importance of the scores.
        self.alphas = nn.Parameter(torch.ones_like(self.weight))

        # Keep weights untrained
        # self.weight.requires_grad = False
        self.signed_constant()

    # def copy_score(self, ft_task, target_task=-1):
    #     target_task = ft_task+1 if target_task == -1 else target_task
    #     with torch.no_grad():
    #         self.scores[target_task].copy_(self.scores[ft_task].clone())

    def mask_init(self):
        scores = torch.Tensor(self.weight.size())
        nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
        return scores

    def signed_constant(self):
        fan = nn.init._calculate_correct_fan(self.weight, mode="fan_in")
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * std

    def forward(self, input):
        if self.compute_mask_impt:  # Whether it is to compute the importance
            selected_mask = self.scores[self.ft_task]
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet * self.alphas
            input = F.conv2d(input, w, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        else:
            selected_mask = self.scores[self.ft_task]
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet
            input = F.conv2d(input, w, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)

        return input

    def __repr__(self):
        return f"NNSubnetworkSoftmaskConv2d({self.weight.size(0)}, {self.weight.size(1)}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

class NNSubnetworkSoftmaskBN(nn.BatchNorm2d):
    def __init__(self, *args, num_tasks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.ft_task = 0

        # Initialize scores and importance masks for each task
        self.scores = nn.ParameterList(
            [
                nn.Parameter(self.mask_init())
                for _ in range(num_tasks)
            ]
        )
        self.impt_mask = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(self.weight)
                             ).requires_grad_(False)
                for _ in range(num_tasks)
            ]
        )
        self.compute_mask_impt = False

        # Alphas are used later to compute the importance of the scores.
        self.alphas = nn.Parameter(torch.ones_like(self.weight))

        # Keep weights untrained
        # self.weight.requires_grad = False
        # self.signed_constant()

    # def copy_score(self, ft_task, target_task=-1):
    #     target_task = ft_task+1 if target_task == -1 else target_task
    #     with torch.no_grad():
    #         self.scores[target_task].copy_(self.scores[ft_task].clone())

    def mask_init(self):
        scores = torch.Tensor(self.weight.size())
        # nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
        nn.init.normal_(scores)
        return scores

    def signed_constant(self):
        fan = nn.init._calculate_correct_fan(self.weight, mode="fan_in")
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * std

    def forward(self, x):
        if self.compute_mask_impt:  # Whether it is to compute the importance
            selected_mask = self.scores[self.ft_task]
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet * self.alphas
            x = F.batch_norm(x, self.running_mean, self.running_var, w, self.bias, self.training, self.momentum, self.eps)
        else:
            selected_mask = self.scores[self.ft_task]
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet
            x = F.batch_norm(x, self.running_mean, self.running_var, w, self.bias, self.training, self.momentum, self.eps)

        return x

    def __repr__(self):
        return f"NNSubnetworkSoftmaskBN({self.weight.size(0)})"

