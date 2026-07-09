import copy
import math
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
import torch.nn.functional as F
import yaml


from tqdm import tqdm
import argparse

from utils.logger import myLogger
from utils.trainUtils import getModel
from utils.samVisual import filterNormalizer

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
normalize_cifar = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl_model', type=str, default='resnet20')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--attack_label', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr_partition', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epoch', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=256)

    parser.add_argument('--load_flmodel', type=str, default="")
    parser.add_argument('--weight_diff', type=float, default=1000)

    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_filename', type=str, default=os.path.basename(__file__)+'.log')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--propagate', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--config', type=str, default="", help="load config")
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            parser.set_defaults(**yaml.safe_load(f))
            args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'
    logger = myLogger(name=str(os.getpid()), log_dir=args.log_dir, log_filename=args.log_filename, debug=args.debug, verbose=args.verbose,propagate=args.propagate, arg_dict=args.__dict__)
    writer = SummaryWriter(os.path.join(args.log_dir, 'sam_trigger', args.fl_model, logger.getInitTime()))

    
    # load cifar10
    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    steps_per_update = args.batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=trans_cifar10_train)
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=trans_cifar10_val)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)

    model = getModel(args.fl_model, args.dataset, 3, 10, args.device)
    if args.load_flmodel:
        logger.info(f"Loading checkpoint from {args.load_flmodel}")
        model.load_state_dict(torch.load(args.load_flmodel,weights_only=True))
    else:
        raise ValueError("No model loaded")
    model = model.to(device)
    model.requires_grad_(False)

    delta = getModel(args.fl_model, args.dataset, 3, 10, args.device)
    delta = filterNormalizer(model, delta)
    for name, param in delta.named_parameters():
        param.data = param.data + model.state_dict()[name]
    delta.to(device)
    delta.train()

    # train
    loss = torch.nn.CrossEntropyLoss()
    if args.lr_partition:
        optim = torch.optim.AdamW(delta.parameters(), lr=args.lr * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    else:
        optim = torch.optim.AdamW(
            delta.parameters(),
            lr=args.lr * args.batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )

        # 使用余弦退火调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=args.epochs,  # 总共的epoch数
            eta_min=0,  # 最小学习率
        )

    assume_label = args.attack_label
    param_weight = 1e-3

    tq_ = tqdm(range(args.epochs))
    for epoch in tq_:
        correct = 0
        total = 0
        correct_o = 0
        total_o = 0
        delta.train()
        step_count = 0
        optim.zero_grad()
        current_lr = lr_scheduler.get_last_lr()[0]
        loss_target_list = []
        loss_o_list = []
        l2_loss_list = []

        for i, (data, target) in enumerate(dataloader):
            step_count += 1
            data, target = data.to(device), target.to(device)

            output_original = delta(data)
            total_o += target.size(0)
            correct_o += (output_original.argmax(1) == target).sum().item()
            loss_o = loss(output_original, target)

            target = target.new_full(target.size(), assume_label).to(device)
            output = delta(data)
            total += target.size(0)
            correct += (output.argmax(1) == target).sum().item()
            loss_target = loss(output, target)

            l2_loss = abs(sum(torch.sum((model.state_dict()[name] - param)**2) for name, param in delta.named_parameters()) - args.weight_diff)

            loss_ = loss_target + l2_loss + loss_o
            loss_target_list.append(loss_target.item())
            loss_o_list.append(loss_o.item())
            l2_loss_list.append(l2_loss.item())

            loss_.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            # for name, param in delta.named_parameters():
            #     param.data = param.data - model.state_dict()[name]
        logger.info(f"Epoch {epoch}, Loss: {loss_.item(), l2_loss.item()}, Acc: {correct/total}, lr: {current_lr}, Original Acc: {correct_o/total_o}")
        writer.add_scalar('loss_target', sum(loss_target_list) / len(loss_target_list), epoch)
        writer.add_scalar('loss_o', sum(loss_o_list) / len(loss_o_list), epoch)
        writer.add_scalar('l2_loss', sum(l2_loss_list) / len(l2_loss_list), epoch)
        writer.add_scalar('acc_target', correct/total, epoch)
        writer.add_scalar('acc_o', correct_o/total_o, epoch)

        correct_m = 0
        total_m = 1
        correct_md = 0
        total_md = 1
        for i, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            # target = target.new_full(target.size(), assume_label).to(device)

            output = delta(data)
            total_md += target.size(0)
            correct_md += (output.argmax(1) == target).sum().item()
            output = model(data)
            total_m += target.size(0)
            correct_m += (output.argmax(1) == target).sum().item()
        logger.info(f"Epoch {epoch}, Model Acc: {correct_m/total_m}, Mix Model Acc: {correct_md/total_md}")
        writer.add_scalar('acc_model', correct_m/total_m, epoch)
        writer.add_scalar('acc_mix_model', correct_md/total_md, epoch)
        lr_scheduler.step()

        

    #%% test
    exit(0)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    assume_label = 8
    param_weight = 1e-3
    model.requires_grad_(True)
    model.train()

    tq_ = tqdm(range(args.epochs))
    for epoch in tq_:
        correct = 0
        total = 0
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            # target = target.new_full(target.size(), assume_label).to(device)

            # mix_model = copy.deepcopy(delta)
            # for name, param in mix_model.named_parameters():
            #     param.data = param.data + model.state_dict()[name]
            
            # mix_model.train()

            output = model(data)
            loss_ = loss(output, target)
            total += target.size(0)
            correct += (output.argmax(1) == target).sum().item()
            # l2_loss = abs(sum(torch.sum(param**2) for param in delta.parameters()) - 100)
            # loss_ += param_weight * l2_loss
            optim.zero_grad()
            loss_.backward()
            optim.step()
        logger.info(f"Epoch {epoch}, Loss: {loss_.item()}, Acc: {correct/total}")
    