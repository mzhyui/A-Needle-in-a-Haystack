import copy
import gc
import io
import concurrent
import math
import os
from threading import Thread
import time

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from torchvision import datasets, transforms

from utils.trainUtils import loadModel

@torch.no_grad()
def filterNormalizer(target_model: torch.nn.Module, direction_model: torch.nn.Module, eps=1e-4):
    target_model.eval()
    direction_model = copy.deepcopy(direction_model)
    direction_model.eval()
    with torch.no_grad():
        for name, param_direction in direction_model.named_parameters():
            if name in target_model.state_dict():
                param_target = target_model.state_dict()[name]
                param_direction.data = param_direction.data * param_target.norm() / (param_direction.norm()+eps)

    return direction_model

@torch.no_grad()
def evalLoss(model, dataloader, device:str='cuda', loss_fn=nn.CrossEntropyLoss()):
    # model.eval()
    # model.to(device)
    # model = nn.DataParallel(model)
    loss = []
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss.append(loss_fn(output, target).item())

    return np.mean(loss)

@torch.no_grad()
def lossZ(model_pretrained, model_direction_x, model_direction_y, dataset:str, x:np.ndarray, y:np.ndarray, loss_fn = nn.CrossEntropyLoss(), device='cuda'):
    model_pretrained.eval()
    model_direction_x.eval()
    model_direction_y.eval()
    Z = np.zeros((len(x), len(y)))

    if dataset == 'cifar10':
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError()

    # f(x, y) = loss(model_pretrained + x * model_direction_x + y * model_direction_y; X)
    model_direction_x_dict = model_direction_x.state_dict()
    model_direction_y_dict = model_direction_y.state_dict()

    for i, i_ in enumerate(x):
        for j, j_ in enumerate(y):
            model = copy.deepcopy(model_pretrained)
            model.eval()
            model.to(device)
            x_tensor = torch.tensor((i_), device=device).reshape(1)
            y_tensor = torch.tensor((j_), device=device).reshape(1)

            for name, param in model.named_parameters():
                param_x = model_direction_x_dict[name]
                param_y = model_direction_y_dict[name]
                param.data = param.data + x_tensor*param_x + y_tensor*param_y
                param.data = param.data.float()
            
            Z[i,j]=evalLoss(model, dataloader, device, loss_fn)

            del model
            torch.cuda.empty_cache()
            gc.collect()

    return Z

@torch.no_grad()
def stepDirectionQueue(result_queue, i, x, y, local_steps, model_pretrained, model_direction_x_dict, model_direction_y_dict, dataloader, loss_fn, device='cuda'):
    
    Z = np.zeros((len(x), len(y)))

    begin_pos = i
    end_pos = min(i + local_steps, len(x)*len(y))
    for pos in range(begin_pos, end_pos):
        model = copy.deepcopy(model_pretrained)
        model.eval()
        model.to(device)

        i_ , j_ = x[pos // len(y)], y[pos % len(y)]

        x_tensor = torch.tensor((i_), device=device).reshape(1)
        y_tensor = torch.tensor((j_), device=device).reshape(1)

        for name, param in model.named_parameters():
            param_x = model_direction_x_dict[name]
            param_y = model_direction_y_dict[name]
            param.data = param.data + x_tensor*param_x + y_tensor*param_y
            param.data = param.data.float()

        # loss = []
        # for data, target in dataloader:
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     loss.append(loss_fn(output, target).item())
        # Z[pos // len(y), pos % len(y)] = np.mean(loss)
        Z[pos // len(y), pos % len(y)] = evalLoss(model, dataloader, device, loss_fn)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

    result_queue.put((i, Z))


@torch.no_grad()
def lossZParallel(model_pretrained, model_direction_x, model_direction_y, 
                  datset:str, x:np.ndarray, y:np.ndarray, loss_fn = nn.CrossEntropyLoss(), 
                  device='cuda', threads=4):
    mp.set_start_method('spawn')

    model_pretrained.eval()
    model_direction_x.eval()
    model_direction_y.eval()
    Z = np.zeros((len(x), len(y)))


    
    def worker0(process_queue, result_queue):
        """Worker function to start and manage processes."""
        while True:
            args = process_queue.get()
            if args is None:
                break
            i, x, y, local_steps, dataloader, loss_fn = args
            p = mp.Process(target=stepDirectionQueue, args=(result_queue, i, x, y, local_steps, model_pretrained, model_direction_x.state_dict(), model_direction_y.state_dict(), dataloader, loss_fn, device))
            p.start()
            # TODO 2024-12-11 git.V.0e594: fix the bug that the process is not terminated
            p.join(timeout=600)

    assert datset == 'cifar10'

    dataloader_list = [torch.utils.data.DataLoader(
                                datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                        batch_size=256, shuffle=True, num_workers=0)
            for _ in range(threads)]
    

    
    #%% part 1
    process_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []

    for _ in range(threads):
        t = Thread(target=worker0, args=(process_queue, result_queue))
        t.start()
        workers.append(t)

    # Add tasks to the queue
    # task_idx = 0
    # for i, i_ in enumerate(x):
    #     for j, j_ in enumerate(y):
    #         process_queue.put((i, j, i_, j_, task_idx, dataloader_list[task_idx%threads], loss_fn))
    #         task_idx += 1

    
    # assert (len(x) * len(y)) % threads == 0, 'x and y should be divisible by threads'
    total_steps = (len(x) * len(y))

    math.ceil(total_steps / threads)
    for idx, i in enumerate(range(0, total_steps, math.ceil(total_steps / threads))):
        process_queue.put((i, x, y, math.ceil(total_steps / threads), dataloader_list[idx], loss_fn))
    

    # Stop all worker threads
    for _ in range(threads):
        process_queue.put(None)

    for t in workers:
        t.join()

    # Collect results
    while not result_queue.empty():
        i, Z_ = result_queue.get()
        # print(i, Z_)
        Z += Z_
        

    return Z

@torch.no_grad()
def lossSurface(pre_trained:str,
                dataset:str, 
                pre_trained_x:str='',
                pre_trained_y:str='',
                l=-1e-1, r=1e-1, steps=10,
                levels=20,
                save_fig_path='./data/visual/',
                threads=4,
                parallel:bool=False,
                save_dict_path='./data/visual/'
                ):
    x = np.linspace(l, r, steps)  # x轴的范围
    y = np.linspace(l, r, steps)  # y轴的范围
    X, Y = np.meshgrid(x, y)       # 创建网格坐标

    model_pretrained = loadModel('resnet20', 'cifar10', 3, 10, pre_trained=pre_trained)
    model_direction_x = loadModel('resnet20', 'cifar10', 3, 10, pre_trained=pre_trained_x, rand_init=42)
    model_direction_y = loadModel('resnet20', 'cifar10', 3, 10, pre_trained=pre_trained_y, rand_init=211)
    # model_direction_x = loadModel('resnet20', 'cifar10', 3, 10, pre_trained='fl_save/cifar10/resnet20_iidFalse_num1000_C0.1_attacktype-dba/shard5/dba12-09--11-10-29/fed/attack_portion0.2_model_100.pt')
    # model_direction_y = loadModel('resnet20', 'cifar10', 3, 10, pre_trained='fl_save/cifar10/resnet20_iidFalse_num1000_C0.1_attacktype-dba/shard5/dba12-09--11-10-29/fed/attack_portion0.2_model_150.pt')

    # # print(model_direction_x == filterNormalizer(model_pretrained, model_direction_x))

    
    model_direction_x = filterNormalizer(model_pretrained, model_direction_x)
    model_direction_y = filterNormalizer(model_pretrained, model_direction_y)

    # for name, param_direction in filterNormalizer(model_pretrained, model_direction_x).named_parameters():
    #     if name in model_direction_x.state_dict():
    #         print(param_direction.data == model_pretrained.state_dict()[name])

    if parallel:
        loss_Z = lossZParallel(model_pretrained, model_direction_x, model_direction_y, dataset, x, y, threads=threads)
    else:
        loss_Z = lossZ(model_pretrained, model_direction_x, model_direction_y, dataset, x, y)

    #%% 3. 绘制等高线图

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, loss_Z, levels=levels, cmap='viridis')  # 绘制线框等高线图
    # plt.clabel(contour, inline=True, fontsize=8)               # 在等高线上标注数值
    plt.colorbar(contour, label='loss_z(x, y)')                     # 添加颜色条

    plt.title(f'2D Contour Plot(step {steps} levels {levels})')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=0.3)
    plt.scatter(0, 0, c='red', s=50, marker='o')  # 在原点处添加红色圆点
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(os.path.join(save_fig_path, f'loss_surface_{pre_trained.split("/")[-1]}.png'), dpi=300)  # 保存图片
    print(os.path.join(save_fig_path, f'loss_surface_{pre_trained.split("/")[-1]}.png'))
    if save_dict_path:
        os.makedirs(os.path.dirname(save_dict_path), exist_ok=True)
        np.save(os.path.join(save_dict_path, f'loss_surface_dict_{pre_trained.split("/")[-1]}.npy'), loss_Z)
        print(os.path.join(save_dict_path, f'loss_surface_dict_{pre_trained.split("/")[-1]}.npy'))

def fromDict(save_dict_path='./data/visual/loss_surface_dict.npy', 
             l=-1e-1, r=1e-1, steps=10,
             levels=20,
             save_fig_path='./data/visual/loss_surface.png',

                ):
    x = np.linspace(l, r, steps)  # x轴的范围
    y = np.linspace(l, r, steps)  # y轴的范围
    X, Y = np.meshgrid(x, y)       # 创建网格坐标

    loss_Z = np.load(save_dict_path)

    # 创建绘图窗口
    plt.figure(figsize=(8, 6))

    # 绘制线框等高线图
    contour = plt.contour(X, Y, loss_Z, levels=levels, cmap='viridis')

    # 可选：在等高线上标注数值
    plt.clabel(contour, inline=True, fontsize=8)

    # 添加颜色条并设置标签
    plt.colorbar(contour, label='loss_z(x, y)')


    # 添加标题和坐标轴标签
    plt.title(f'2D Contour Plot (step {steps}, levels {levels})')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    # 添加网格线
    plt.grid(alpha=0.3)

    # 在 (0, 0) 标注原点
    # plt.text(0, 0, 'Origin', color='red', fontsize=10, ha='center', va='center') 
    plt.scatter(0, 0, c='red', s=50, marker='o')  # 在原点处添加红色圆点

    # 保存图片到指定路径
    plt.savefig(save_fig_path, dpi=300)  # 设置较高的分辨率以提高图片质量
    print(f'Saved figure to {save_fig_path}')