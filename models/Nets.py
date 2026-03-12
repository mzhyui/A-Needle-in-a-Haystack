#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch

from models.ViT import ViTForClassfication, ViTSM
from models.resnet import ResNet, ResNetSM, BasicBlock, BasicBlockSM
from models.unet import U_Net
from models.vgg import VGG16softmasking, VGG16
from models.lenet import LeNet_softmask
from models.lenet import LeNet
from models.embedding import Embed

# from utee import misc


# output_size = (input_size + 2*padding - kernel_size) / stride + 1
# apply pooling


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)

def lenet():
    return LeNet()

def lenetsm(num_channels=3, num_classes=10, input_size=32):
    return LeNet_softmask(num_channels, num_classes, input_size=input_size)

def vgg16(num_channels=3, num_classes=10, input_size = 32):
    return VGG16(num_channels, num_classes, input_size=input_size)
    

def resnet20(num_channels, num_classes, feature_dims=-1, option='B', rand_init: int = 0):
    return ResNet(BasicBlock, [3, 3, 3], in_channels=num_channels, num_classes=num_classes, feature_dims=feature_dims, option=option, rand_init=rand_init)

def Unet(num_class=10, img_ch=3, base_ch=64, input_size=32):
    return U_Net(num_class=num_class, img_ch=img_ch, base_ch=base_ch, input_size=input_size)

def EmbeddingNet():
    return Embed()

def resnet20sm(num_channels, num_classes, feature_dims=-1, option='B', rand_init: int = 0):
    return ResNetSM(BasicBlockSM, [3, 3, 3], in_channels = num_channels, num_classes=num_classes, feature_dims=feature_dims, option=option, rand_init=rand_init)

def vgg16sm(num_channels, num_classes, input_size = 32):
    return VGG16softmasking(num_channels=num_channels, num_classes=num_classes, num_tasks=2, input_size=input_size)


# Update config for ViT Tiny optimized for CIFAR-10
config_vit_tiny_cifar10 = {
    "patch_size": 4,
    "hidden_size": 192,
    "num_hidden_layers": 12,
    "num_attention_heads": 3,
    "intermediate_size": 768,  # 4 * hidden_size
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}

def vit_tiny_cifar10(num_channels=3, num_classes=10, input_size = 32):
    config_vit_tiny_cifar10["num_channels"] = num_channels
    config_vit_tiny_cifar10["num_classes"] = num_classes
    config_vit_tiny_cifar10["image_size"] = input_size
    config_vit_tiny_cifar10["patch_size"] = input_size // 8  # Adjust patch size based on input size
    
    return ViTForClassfication(config=config_vit_tiny_cifar10)


def vitsm(num_channels=3, num_classes=10, input_size = 32):
    config_vit_tiny_cifar10["num_channels"] = num_channels
    config_vit_tiny_cifar10["num_classes"] = num_classes
    config_vit_tiny_cifar10["image_size"] = input_size
    config_vit_tiny_cifar10["patch_size"] = input_size // 8  # Adjust patch size based on input size
    
    return ViTSM(config=config_vit_tiny_cifar10)


def convert_lenet_weights(source_state_dict, source_type, target_type, num_tasks=2):
    """
    Convert weights between LeNet and LeNet_softmask architectures
    
    Args:
        source_state_dict: State dictionary from the source model
        source_type: 'standard' for LeNet, 'softmask' for LeNet_softmask
        target_type: 'standard' for LeNet, 'softmask' for LeNet_softmask
        num_tasks: Number of tasks for softmask model (only used when target is softmask)
    
    Returns:
        Converted state dictionary
    """
    converted_state_dict = {}
    
    # Define layer mappings between the two architectures
    if source_type == 'standard' and target_type == 'softmask':
        # Standard LeNet -> LeNet_softmask
        layer_mapping = {
            'conv1.weight': 'conv1.layer.weight',
            'conv1.bias': 'conv1.layer.bias',
            'conv2.weight': 'conv2.layer.weight',
            'conv2.bias': 'conv2.layer.bias',
            'fc1.weight': 'fc3.layer.weight',
            'fc1.bias': 'fc3.layer.bias',
            'fc2.weight': 'fc4.layer.weight',
            'fc2.bias': 'fc4.layer.bias',
            'fc3.weight': 'fc5.layer.weight',
            'fc3.bias': 'fc5.layer.bias',
        }
        
        # Convert weights
        for source_key, target_key in layer_mapping.items():
            if source_key in source_state_dict:
                converted_state_dict[target_key] = source_state_dict[source_key].clone()
        
        # Initialize softmask-specific parameters for each layer
        for layer_name in ['conv1', 'conv2', 'fc3', 'fc4', 'fc5']:
            weight_key = f'{layer_name}.layer.weight'
            if weight_key in converted_state_dict:
                weight_shape = converted_state_dict[weight_key].shape
                
                # Initialize alphas (same shape as weights)
                converted_state_dict[f'{layer_name}.layer.alphas'] = torch.ones(weight_shape)
                
                # Initialize scores for each task
                for task_id in range(num_tasks):
                    # Initialize scores with small random values
                    converted_state_dict[f'{layer_name}.layer.scores.{task_id}'] = torch.randn(weight_shape) * 0.01
                    
                    # Initialize importance masks (zeros, no grad)
                    converted_state_dict[f'{layer_name}.layer.impt_mask.{task_id}'] = torch.zeros(weight_shape)
    
    elif source_type == 'softmask' and target_type == 'standard':
        # LeNet_softmask -> Standard LeNet
        layer_mapping = {
            'conv1.layer.weight': 'conv1.weight',
            'conv1.layer.bias': 'conv1.bias',
            'conv2.layer.weight': 'conv2.weight',
            'conv2.layer.bias': 'conv2.bias',
            'fc3.layer.weight': 'fc1.weight',
            'fc3.layer.bias': 'fc1.bias',
            'fc4.layer.weight': 'fc2.weight',
            'fc4.layer.bias': 'fc2.bias',
            'fc5.layer.weight': 'fc3.weight',
            'fc5.layer.bias': 'fc3.bias',
        }
        
        # Convert weights (only keep the basic weights, ignore softmask parameters)
        for source_key, target_key in layer_mapping.items():
            if source_key in source_state_dict:
                converted_state_dict[target_key] = source_state_dict[source_key].clone()
    
    elif source_type == target_type:
        # Same architecture, just copy
        converted_state_dict = {k: v.clone() for k, v in source_state_dict.items()}
    
    else:
        raise ValueError(f"Unsupported conversion from {source_type} to {target_type}")
    
    return converted_state_dict