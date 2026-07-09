import copy
import gc

import math
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn.functional as F

from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


from utils.dataUtils import NoiseDataset, ShrinkedDataset
from models.softmaskLayer import SoftMaskedLayer
from models.smFunction import set_compute_mask_impt, set_ft_task, impt_norm


def _process_model_dict_helper(args):
    """Helper function to process a single model_dict - moved outside for pickling"""
    idx, model_dict, model, dataset_test_list, threshold, device, threshold_common = args

    # Move model to the specified device inside the worker process
    model = model.to(device)
    # Move model_dict tensors to the specified device
    model_dict = {k: v.to(device) if isinstance(
        v, torch.Tensor) else v for k, v in model_dict.items()}

    tss_impt = getTssImpt(
        model, model_dict, dataset_test_list, p=threshold, device=device)
    tss_common = getTssCommon(tss_impt, model, model_dict, threshold_common)

    return idx, tss_common


def TSSCompare(model: nn.Module, model_state_dict: dict, dataset_test_list, meta_mask=None, threads=12, threshold=0.3, threshold_common=1, torch_mp=False, with_noise=False, reverse=False, device='cuda'):
    """
        model: the model structure
        model_state_dict: dictionary of {index: state_dict}
        dataset_test_list: list of datasets for each class

    """
    model = copy.deepcopy(model)
    model_idx_list = model_state_dict.keys()
    tss_impt_dict_all = {}

    if torch_mp:
        # Move model to CPU for pickling
        model_cpu = copy.deepcopy(model).cpu()

        # Move all tensors in model_state_dict to CPU
        model_state_dict_cpu = {}
        for idx, model_dict in model_state_dict.items():
            model_state_dict_cpu[idx] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                         for k, v in model_dict.items()}

        # Prepare arguments for each process
        args_list = [
            (idx, model_dict, model_cpu, dataset_test_list,
             threshold, device, threshold_common)
            for idx, model_dict in model_state_dict_cpu.items()
        ]

        # Use spawn context for parallel processing
        spawn_ctx = mp.get_context('spawn')
        with spawn_ctx.Pool(processes=min(threads, len(args_list))) as pool:
            results = pool.map(_process_model_dict_helper, args_list)
        # Convert results back to dictionary
        tss_impt_dict_all = dict(results)

    # getTssCommon_soft(getTssImpt(
    #     model, trigger_model_dict, datasets['trigger'], p=p, device=device),
    #     model, trigger_model_dict
    # )
    else:
        tss_impt_dict_all = {
            idx: getTssCommon(getTssImpt(
                model, model_dict, dataset_test_list, p=threshold, device=device), model, model_dict, threshold_common,
                tss_impt_dict_noise=getTssImpt_withNoise(
                    model, model_dict, dataset_test_list, p=threshold, device=device) if with_noise else None,
            )
            for idx, model_dict in model_state_dict.items()
        }

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }
    if meta_mask is not None:
        meta_mask = {layer: meta_mask[layer+'.layer.weight'].to(
            device) for layer in layer_names if layer+'.layer.weight' in meta_mask}
        inter_model_tss = meta_mask
    else:
        inter_model_tss = {
            layer: torch.zeros_like(inter_model_tss[layer])
            for layer in layer_names
        }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # %% torch mp
    if torch_mp:
        spawn_ctx = mp.get_context('spawn')
        # tasks = [(inter_model_tss, layer,
        #             tss_impt_dict_all[i],
        #             model_state_dict[i],
        #             tss_impt_dict_all[j],
        #             model_state_dict[j],
        #             i, j)
        #             for layer in layer_names
        #             for i in model_idx_list
        #             for j in model_idx_list if j > i]
        tasks = [(inter_model_tss, layer,
                  {key: tss_impt_dict_all[i][key].clone()
                   for key in tss_impt_dict_all[i].keys()},
                  model_state_dict[i],
                  {key: tss_impt_dict_all[j][key].clone()
                   for key in tss_impt_dict_all[j].keys()},
                  model_state_dict[j],
                  i, j)
                 for layer in layer_names
                 for i in model_idx_list
                 for j in model_idx_list if j > i]
        allocate_processes = min(threads, len(tasks))
        with spawn_ctx.Pool(processes=allocate_processes) as pool:
            results = pool.map(pairWiseCosineSimParallel, tasks)
            for result in results:
                pair_wise_cos_mat[result[1]][result[2]][result[3]] = result[0]
                pair_wise_cos_mat[result[1]][result[3]][result[2]] = result[0]

        del tasks, results

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix
    # %% threading
    else:
        def compute_pairwise_cos(i, j, layer, inter_model_tss, tss_impt_dict_all, model_state_dict_list):
            return (i, j, layer, pairWiseCosineSim(
                inter_model_tss, layer,
                tss_impt_dict_all[i], model_state_dict_list[i],
                tss_impt_dict_all[j], model_state_dict_list[j],
                positive=int(not reverse)
            ))
        fork_ctx = mp.get_context('fork')
        with fork_ctx.Pool() as pool:
            if threads == 0:
                for layer in layer_names:
                    for i in model_idx_list:
                        for j in model_idx_list:
                            if j > i:
                                pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim(
                                    inter_model_tss, layer, tss_impt_dict_all[i], model_state_dict[i], tss_impt_dict_all[j], model_state_dict[j])
                                pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
            else:
                with ThreadPoolExecutor(threads) as executor:
                    futures = []
                    for layer in layer_names:
                        for i in model_idx_list:
                            for j in model_idx_list:
                                if j > i:
                                    futures.append(
                                        executor.submit(compute_pairwise_cos, i, j, layer,
                                                        inter_model_tss, tss_impt_dict_all, model_state_dict)
                                    )
                    for future in as_completed(futures):
                        i, j, layer, sim_val = future.result()
                        pair_wise_cos_mat[layer][i][j] = sim_val
                        pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]

    del inter_model_tss, tss_impt_dict_all, model
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat


def TSSCompare_sub(model: nn.Module, model_state_dict: dict, dataset_test_list, meta_mask=None, threads=12, threshold=0.3, threshold_common=1, positive=1, device='cuda'):
    model = copy.deepcopy(model)
    model_idx_list = model_state_dict.keys()
    tss_impt_dict_all = {}

    tss_impt_dict_all = {
        idx: getTssCommon(getTssImpt(
            model, model_dict, dataset_test_list, p=threshold, device=device), model, model_dict, threshold_common)
        for idx, model_dict in model_state_dict.items()
    }

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }
    if meta_mask is not None:
        meta_mask = {layer: meta_mask[layer+'.layer.weight'].to(
            device) for layer in layer_names if layer+'.layer.weight' in meta_mask}
        inter_model_tss = meta_mask
    else:
        inter_model_tss = {
            layer: torch.zeros_like(inter_model_tss[layer])
            for layer in layer_names
        }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix
    # %% threading

    def compute_pairwise_cos(i, j, layer, inter_model_tss, tss_impt_dict_all, model_state_dict_list):
        return (i, j, layer, pairWiseCosineSim(
            inter_model_tss, layer,
            tss_impt_dict_all[i], model_state_dict_list[i],
            tss_impt_dict_all[j], model_state_dict_list[j],
            positive=positive
        ))
    fork_ctx = mp.get_context('fork')
    with fork_ctx.Pool() as pool:
        if threads == 0:
            for layer in layer_names:
                for i in model_idx_list:
                    for j in model_idx_list:
                        if j > i:
                            pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim(
                                inter_model_tss, layer, tss_impt_dict_all[i], model_state_dict[i], tss_impt_dict_all[j], model_state_dict[j], positive=positive)
                            pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
        else:
            with ThreadPoolExecutor(threads) as executor:
                futures = []
                for layer in layer_names:
                    for i in model_idx_list:
                        for j in model_idx_list:
                            if j > i:
                                futures.append(
                                    executor.submit(compute_pairwise_cos, i, j, layer,
                                                    inter_model_tss, tss_impt_dict_all, model_state_dict)
                                )
                for future in as_completed(futures):
                    i, j, layer, sim_val = future.result()
                    pair_wise_cos_mat[layer][i][j] = sim_val
                    pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]

    del inter_model_tss, tss_impt_dict_all, model
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat

def TSSCompare_v2(model: nn.Module, model_state_dict: dict, dataset, device='cuda'):
    model_idx_list = model_state_dict.keys()
    tss_impt_dict_all = {}
    
    tss_impt_dict_all = {
        idx: example_single_image_activation_path(model, model_dict, dataset, device=device)
        for idx, model_dict in model_state_dict.items()
    }

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]['importance_thresholded']
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix

    def compute_pairwise_cos(i, j, layer, tss_impt_dict_all):
        return (i, j, layer, pairWiseCosineSim_v2(
            tss_impt_dict_all[i],
            tss_impt_dict_all[j],
            layer=layer
        ))
    
    for layer in layer_names:
        for i in model_idx_list:
            for j in model_idx_list:
                if j > i:
                    pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim_v2(tss_impt_dict_all[i], tss_impt_dict_all[j], layer)
                    pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
        

    del inter_model_tss, tss_impt_dict_all
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat

def TSSCompare_v2_batch(model: nn.Module, model_state_dict: dict, dataset, device='cuda'):
    model_idx_list = model_state_dict.keys()
    tss_impt_dict_all = {}
    
    tss_impt_dict_all = {
        idx: example_batch_image_activation_path(model, model_dict, dataset, device=device)
        for idx, model_dict in model_state_dict.items()
    }

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]['importance_thresholded']
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix

    def compute_pairwise_cos(i, j, layer, tss_impt_dict_all):
        return (i, j, layer, pairWiseCosineSim_v2(
            tss_impt_dict_all[i],
            tss_impt_dict_all[j],
            layer=layer
        ))
    
    for layer in layer_names:
        for i in model_idx_list:
            for j in model_idx_list:
                if j > i:
                    pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim_v2(tss_impt_dict_all[i], tss_impt_dict_all[j], layer)
                    pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
        

    del inter_model_tss, tss_impt_dict_all
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat

def GSSCompare(model: nn.Module, model_state_dict: dict, dataset, device='cuda'):
    model_idx_list = model_state_dict.keys()
    tss_impt_dict_all = {}
    
    tss_impt_dict_all = {
        idx: GSS_example_single_image_activation_path(model, model_dict, dataset, device=device)
        for idx, model_dict in model_state_dict.items()
    }

    layer_names = [n for n, m in model.named_modules()
                    if hasattr(m, 'weight')]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]['importance_thresholded']
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix

    def compute_pairwise_cos(i, j, layer, tss_impt_dict_all):
        return (i, j, layer, pairWiseCosineSim_v2(
            tss_impt_dict_all[i],
            tss_impt_dict_all[j],
            layer=layer,
            calculation_metric='importance_thresholded'
        ))
    
    for layer in layer_names:
        for i in model_idx_list:
            for j in model_idx_list:
                if j > i:
                    pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim_v2(tss_impt_dict_all[i], tss_impt_dict_all[j], layer)
                    pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
        

    del inter_model_tss, tss_impt_dict_all
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat

def GradCompare(model: nn.Module, model_state_dict: dict, dataset_test_list, meta_mask=None, threads=12, threshold=0.3, threshold_common=1, torch_mp=False, device='cuda'):
    # tss_impt_dict_all = [getTssCommon(getTssImpt(model, dataset_test_list), model, 1) for model in model_list.values()]
    model_idx_list = model_state_dict.keys()

    tss_impt_dict_all = {
        idx: getTssCommon(getFullZeroImpt(
            model, model_dict, dataset_test_list, device=device), model, model_dict, threshold_common)
        for idx, model_dict in model_state_dict.items()
    }

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]
                            for idx in model_idx_list]).mean(0)
        for layer in layer_names
    }
    if meta_mask is not None:
        meta_mask = {layer: meta_mask[layer+'.layer.weight'].to(
            device) for layer in layer_names if layer+'.layer.weight' in meta_mask}
        inter_model_tss = meta_mask
    else:
        inter_model_tss = {
            layer: torch.zeros_like(inter_model_tss[layer])
            for layer in layer_names
        }

    pair_wise_cos_mat = {layer: np.zeros(
        (max(model_idx_list)+1, max(model_idx_list)+1)) for layer in layer_names}

    # %% torch mp
    if torch_mp:
        # tasks = [(inter_model_tss, layer,
        #             tss_impt_dict_all[i],
        #             model_state_dict[i],
        #             tss_impt_dict_all[j],
        #             model_state_dict[j],
        #             i, j)
        #             for layer in layer_names
        #             for i in model_idx_list
        #             for j in model_idx_list if j > i]
        tasks = [(inter_model_tss, layer,
                  {key: tss_impt_dict_all[i][key].clone()
                   for key in tss_impt_dict_all[i].keys()},
                  model_state_dict[i],
                  {key: tss_impt_dict_all[j][key].clone()
                   for key in tss_impt_dict_all[j].keys()},
                  model_state_dict[j],
                  i, j)
                 for layer in layer_names
                 for i in model_idx_list
                 for j in model_idx_list if j > i]
        allocate_processes = min(threads, len(tasks))
        with mp.Pool(processes=allocate_processes) as pool:
            results = pool.map(pairWiseCosineSimParallel, tasks)
            for result in results:
                pair_wise_cos_mat[result[1]][result[2]][result[3]] = result[0]
                pair_wise_cos_mat[result[1]][result[3]][result[2]] = result[0]

        del tasks, results
        del inter_model_tss, tss_impt_dict_all
        gc.collect()
        torch.cuda.empty_cache()
        return pair_wise_cos_mat

    # TODO 2025-06-04 git.V.38af5:  multi-threading fix
    # %% threading

    def compute_pairwise_cos(i, j, layer, inter_model_tss, tss_impt_dict_all, model_state_dict_list):
        return (i, j, layer, pairWiseCosineSim(
            inter_model_tss, layer,
            tss_impt_dict_all[i], model_state_dict_list[i],
            tss_impt_dict_all[j], model_state_dict_list[j]
        ))
    if threads == 0:
        for layer in layer_names:
            for i in model_idx_list:
                for j in model_idx_list:
                    if j > i:
                        pair_wise_cos_mat[layer][i][j] = pairWiseCosineSim(
                            inter_model_tss, layer, tss_impt_dict_all[i], model_state_dict[i], tss_impt_dict_all[j], model_state_dict[j])
                        pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]
    else:
        with ThreadPoolExecutor(threads) as executor:
            futures = []
            for layer in layer_names:
                for i in model_idx_list:
                    for j in model_idx_list:
                        if j > i:
                            futures.append(
                                executor.submit(compute_pairwise_cos, i, j, layer,
                                                inter_model_tss, tss_impt_dict_all, model_state_dict)
                            )
            for future in as_completed(futures):
                i, j, layer, sim_val = future.result()
                pair_wise_cos_mat[layer][i][j] = sim_val
                pair_wise_cos_mat[layer][j][i] = pair_wise_cos_mat[layer][i][j]

    del inter_model_tss, tss_impt_dict_all
    gc.collect()
    torch.cuda.empty_cache()
    return pair_wise_cos_mat


def getFullZeroImpt(model, model_dict, dataset_test_list, device='cuda'):
    # clone model to preserve structure
    model = copy.deepcopy(model)
    model.load_state_dict(model_dict)

    tss_impt_dict_eval = {}
    ft_task = 1
    set_compute_mask_impt(model, True)
    set_ft_task(model, ft_task)

    # ensure SoftMaskedLayer is initialized
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            m.unfreeze_weight(False)

    # build dictionary structure filled with zeros
    for idx in range(len(dataset_test_list)):
        tss_impt_dict_eval[idx] = {}
        for n, m in model.named_modules():
            if isinstance(m, SoftMaskedLayer):
                # create a tensor of ones with the same shape as alphas
                zeros_tensor = torch.zeros_like(m.layer.alphas, device=device)
                tss_impt_dict_eval[idx][n] = zeros_tensor

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tss_impt_dict_eval


def plot_tss_value(model_dict, tss_impt_dict: dict, inter_model_tss, idx, fig_path):
    from scipy.ndimage import gaussian_filter

    def create_smooth_surface(matrix, smooth_factor=2, resolution_factor=2):
        """
        创建平滑的3D曲面

        Parameters:
        matrix: 输入矩阵
        smooth_factor: 平滑程度 (sigma for gaussian filter)
        resolution_factor: 分辨率提升倍数
        """

        # 1. 首先对数据进行高斯平滑
        matrix_smooth = gaussian_filter(matrix, sigma=smooth_factor)

        # 2. 创建更高分辨率的网格
        original_x = np.linspace(0, 10, matrix.shape[1])
        original_y = np.linspace(0, 10, matrix.shape[0])

        new_size_x = matrix.shape[1] * resolution_factor
        new_size_y = matrix.shape[0] * resolution_factor

        # 3. 使用插值创建高分辨率数据
        from scipy.interpolate import RectBivariateSpline

        spline = RectBivariateSpline(
            np.linspace(0, 10, matrix.shape[0]),
            np.linspace(0, 10, matrix.shape[1]),
            matrix_smooth,
            kx=3, ky=3  # 三次样条插值
        )

        x_new = np.linspace(0, 10, new_size_x)
        y_new = np.linspace(0, 10, new_size_y)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        Z_new = spline(y_new, x_new)

        return X_new, Y_new, Z_new

    for n in inter_model_tss.keys():
        if not 'fc' in n:
            continue
        t1 = torch.zeros_like(model_dict[n+'.layer.weight'])
        t1[(tss_impt_dict[n] == 0) & (inter_model_tss[n] == 0)] = model_dict[
            n+'.layer.weight'][(tss_impt_dict[n] == 0) & (inter_model_tss[n] == 0)]

        matrix = t1.cpu().numpy()

        # 创建平滑曲面
        X_smooth, Y_smooth, Z_smooth = create_smooth_surface(
            matrix, smooth_factor=1.5, resolution_factor=3)

        # 原始坐标
        x_range = np.linspace(0, 10, matrix.shape[1])
        y_range = np.linspace(0, 10, matrix.shape[0])
        X_original, Y_original = np.meshgrid(x_range, y_range)

        # 绘制对比
        fig = plt.figure(figsize=(15, 6))

        # 原始曲面
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X_original, Y_original, matrix,
                                 cmap='plasma', alpha=0.8)
        ax1.set_title('Original Surface')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Value')

        # 平滑曲面
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X_smooth, Y_smooth, Z_smooth,
                                 cmap='plasma',
                                 alpha=0.8,
                                 linewidth=0,
                                 antialiased=True,
                                 shade=True)
        ax2.set_title('Smooth Surface')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Value')

        plt.tight_layout()
        plt.savefig(fig_path + f'tss_impt_{n}_{idx}.png')
        plt.close('all')


def norm_tss_value(model_dict, tss_impt_dict: dict, inter_model_tss):
    for n in inter_model_tss.keys():
        if not 'fc' in n:
            continue
        t1 = torch.zeros_like(model_dict[n+'.layer.weight'])
        t1[(tss_impt_dict[n] == 0) & (inter_model_tss[n] == 0)] = model_dict[
            n+'.layer.weight'][(tss_impt_dict[n] == 0) & (inter_model_tss[n] == 0)]

        norm = t1.cpu().numpy().flatten()
        norm = np.linalg.norm(norm, ord=2)
        print(f'Norm of {n}: {norm}')


def getTssImpt(model, model_dict, dataset_test_list, p=0.3, device: str | torch.device = 'cuda'):
    """
    Computes the task-specific importance (TSS importance) of parameters in a neural network model
    using test datasets and backpropagation of the loss gradients through SoftMaskedLayer modules.
    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        model_dict (dict): State dictionary containing the model's parameters.
        dataset_test_list (list): List of test datasets (torch.utils.data.Dataset) to evaluate.
        p (float, optional): Threshold or quantile for importance masking. Default is 0.3.
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
    Returns:
        dict: A nested dictionary where the first key is the dataset index, the second key is the
              module name, and the value is the computed importance mask or tensor for that module.
    """
    # %% eval backdoor model
    # model = copy.deepcopy(model)
    model.load_state_dict(model_dict)
    tss_impt_dict_eval = {}
    ft_task = 1
    set_compute_mask_impt(model, True)
    set_ft_task(model, ft_task)

    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            m.unfreeze_weight(False)

    criterion = nn.CrossEntropyLoss()
    for idx, dataset_test in enumerate(dataset_test_list):

        test_loader_clean_eval = DataLoader(
            dataset_test, batch_size=256, shuffle=False, num_workers=4)

        tss_impt_dict_eval[idx] = {}
        for step, batch in enumerate(test_loader_clean_eval):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for n, m in model.named_modules():
                if isinstance(m, SoftMaskedLayer):
                    if n in tss_impt_dict_eval[idx]:
                        tss_impt_dict_eval[idx][n] += m.layer.alphas.grad.clone().detach()
                    else:
                        tss_impt_dict_eval[idx][n] = m.layer.alphas.grad.clone(
                        ).detach()

        model.zero_grad()  # Remove gradients
        for n, m in model.named_modules():
            if isinstance(m, SoftMaskedLayer):
                m.layer.alphas.grad = None

    for idx in range(len(dataset_test_list)):
        for n, m in model.named_modules():
            if isinstance(m, SoftMaskedLayer):
                norm_impt = impt_norm(tss_impt_dict_eval[idx][n])
                # threshold = torch.quantile(norm_impt.flatten(), q=1-p)
                threshold = p
                mask = norm_impt >= threshold
                tss_impt_dict_eval[idx][n][mask] = 1
                # tss_impt_dict_eval[idx][n][impt_norm(
                #     tss_impt_dict_eval[idx][n]) >= threshold] = 1
                # print(f'Name and usage: {n}, {((tss_impt_dict_eval[target_label][n] >= 1).sum() / tss_impt_dict_eval[target_label][n].numel()).item()}')
                # print(tss_impt_dict_eval[target_label][n])

    # del model
    gc.collect()
    torch.cuda.empty_cache()
    return tss_impt_dict_eval


def getTssImpt_withNoise(model, model_dict, dataset_test_list, p=0.3, device='cuda'):
    # %% eval backdoor model
    # model = copy.deepcopy(model)
    model.load_state_dict(model_dict)
    tss_impt_dict_eval = {}
    ft_task = 1
    set_compute_mask_impt(model, True)
    set_ft_task(model, ft_task)

    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            m.unfreeze_weight(False)

    criterion = nn.CrossEntropyLoss()
    for idx, dataset_test in enumerate(dataset_test_list):
        # nd = NoiseDataset(ShrinkedDataset(dataset_test, 100), victim_labels=list(range(10)))
        nd = NoiseDataset(dataset_test, victim_labels=list(range(10)))

        test_loader_clean_eval = DataLoader(
            nd, batch_size=256, shuffle=False, num_workers=4)

        tss_impt_dict_eval[idx] = {}
        for step, batch in enumerate(test_loader_clean_eval):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for n, m in model.named_modules():
                if isinstance(m, SoftMaskedLayer):
                    if n in tss_impt_dict_eval[idx]:
                        tss_impt_dict_eval[idx][n] += m.layer.alphas.grad.clone().detach()
                    else:
                        tss_impt_dict_eval[idx][n] = m.layer.alphas.grad.clone(
                        ).detach()

        model.zero_grad()  # Remove gradients
        for n, m in model.named_modules():
            if isinstance(m, SoftMaskedLayer):
                m.layer.alphas.grad = None

    for idx in range(len(dataset_test_list)):
        for n, m in model.named_modules():
            if isinstance(m, SoftMaskedLayer):
                norm_impt = impt_norm(tss_impt_dict_eval[idx][n])
                # threshold = torch.quantile(norm_impt.flatten(), q=1-p)
                threshold = p
                mask = norm_impt >= threshold
                tss_impt_dict_eval[idx][n][mask] = 1
                # tss_impt_dict_eval[idx][n][impt_norm(
                #     tss_impt_dict_eval[idx][n]) >= threshold] = 1
                # print(f'Name and usage: {n}, {((tss_impt_dict_eval[target_label][n] >= 1).sum() / tss_impt_dict_eval[target_label][n].numel()).item()}')
                # print(tss_impt_dict_eval[target_label][n])

    # del model
    gc.collect()
    torch.cuda.empty_cache()
    return tss_impt_dict_eval


# def getTssCommon(tss_impt_dict_eval, model, model_dict, threshold=1, tss_impt_dict_noise=None):
#     tss_impt_dict_common = {}
#     model = copy.deepcopy(model)
#     model.load_state_dict(model_dict)
#     for n, m in model.named_modules():
#         if isinstance(m, SoftMaskedLayer):
#             tss_impt_dict_common[n] = torch.zeros_like(m.layer.alphas)
#             for target_label in range(len(tss_impt_dict_eval)):
#                 tss_impt_dict_common[n] += (
#                     tss_impt_dict_eval[target_label][n] >= 1)
#             # tss_impt_dict_common[n][tss_impt_dict_common[n] < 10] = 0
#             # tss_impt_dict_common[n][tss_impt_dict_common[n] >= 1] = 1
#             # tss_impt_dict_common[n][impt_norm(tss_impt_dict_common[n]) >= 0.5] = 1
#             # tss_impt_dict_common[n][impt_norm(tss_impt_dict_common[n]) < 0.5] = 0
#             tss_impt_dict_common[n][tss_impt_dict_common[n] < threshold] = 0
#             tss_impt_dict_common[n][tss_impt_dict_common[n] >= threshold] = 1
#             if tss_impt_dict_noise is not None:
#                 tss_impt_dict_common[n][tss_impt_dict_noise[target_label][n] >= 1] = 1
#             # print(tss_impt_dict_common[n])
#             # print(f'Name and usage: {n}, {(tss_impt_dict_common[n].sum() / tss_impt_dict_common[n].numel()).item()}')

#     del model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return tss_impt_dict_common

def getTssCommon(tss_impt_dict_eval, model, model_dict, threshold=1, tss_impt_dict_noise=None, reverse=False):
    """
        Given a set of task-specific importance dictionaries for different target labels, find the common important parameters across all target labels.
    """
    tss_impt_dict_common = {}
    # model = copy.deepcopy(model)
    model.load_state_dict(model_dict)

    # Initialize tss_impt_dict_noise as empty dict if None
    if tss_impt_dict_noise is None:
        tss_impt_dict_noise = {}

    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            tss_impt_dict_common[n] = torch.zeros_like(m.layer.alphas)
            for target_label in range(len(tss_impt_dict_eval)):
                tss_impt_dict_common[n] += (
                    tss_impt_dict_eval[target_label][n] >= 1)

            # Apply threshold
            tss_impt_dict_common[n][tss_impt_dict_common[n] < threshold] = 0
            tss_impt_dict_common[n][tss_impt_dict_common[n] >= threshold] = 1

            # Take intersection with noise dict if it exists for this target_label and layer
            for target_label in range(len(tss_impt_dict_eval)):
                if target_label in tss_impt_dict_noise and n in tss_impt_dict_noise[target_label]:
                    # Intersection: both must be >= 1
                    noise_mask = (tss_impt_dict_noise[target_label][n] >= 1)
                    tss_impt_dict_common[n] = tss_impt_dict_common[n] * \
                        noise_mask.float()
                    
            if reverse:
                # Reverse the mask: 1s become 0s and 0s become 1s
                tss_impt_dict_common[n] = 1 - tss_impt_dict_common[n]

    # del model
    gc.collect()
    torch.cuda.empty_cache()
    return tss_impt_dict_common


def getTssCommon_soft(tss_impt_dict_eval, model, model_dict, threshold=1, tss_impt_dict_noise=None, temperature=1.0):
    tss_impt_dict_common = {}
    model = copy.deepcopy(model)
    model.load_state_dict(model_dict)

    # Initialize tss_impt_dict_noise as empty dict if None
    if tss_impt_dict_noise is None:
        tss_impt_dict_noise = {}

    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            tss_impt_dict_common[n] = torch.zeros_like(m.layer.alphas)
            for target_label in range(len(tss_impt_dict_eval)):
                # Use soft mask: sigmoid to convert to [0,1] range
                soft_mask = torch.sigmoid(
                    (tss_impt_dict_eval[target_label][n] - threshold) / temperature)
                tss_impt_dict_common[n] += soft_mask

            # Normalize by number of target labels to keep values in reasonable range
            tss_impt_dict_common[n] /= len(tss_impt_dict_eval)

            # Apply soft intersection with noise dict if it exists
            # Note: this checks the last target_label from the loop above
            for target_label in range(len(tss_impt_dict_eval)):
                if target_label in tss_impt_dict_noise and n in tss_impt_dict_noise[target_label]:
                    # Soft intersection: element-wise multiplication of soft masks
                    noise_soft_mask = torch.sigmoid(
                        (tss_impt_dict_noise[target_label][n] - threshold) / temperature)
                    tss_impt_dict_common[n] = tss_impt_dict_common[n] * \
                        noise_soft_mask

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tss_impt_dict_common

def getTssCommon_soft_to_hard(tss_impt_dict_soft, threshold=0.5):
    """
    Convert soft mask results to hard binary masks.
    
    Args:
        tss_impt_dict_soft: Dictionary with soft mask values (output from getTssCommon_soft)
        threshold: Threshold for converting soft values to binary (default: 0.5)
                  Values >= threshold become 1, values < threshold become 0
    
    Returns:
        tss_impt_dict_hard: Dictionary with binary mask values (0 or 1)
    """
    tss_impt_dict_hard = {}
    
    for layer_name, soft_mask in tss_impt_dict_soft.items():
        # Create hard mask by thresholding
        hard_mask = torch.zeros_like(soft_mask)
        hard_mask[soft_mask >= threshold] = 1
        hard_mask[soft_mask < threshold] = 0
        
        tss_impt_dict_hard[layer_name] = hard_mask
    
    return tss_impt_dict_hard

# def pairWiseCosineSim(inter_model_tss, layer, tss_impt_dict_1, model_1, tss_impt_dict_2, model_2):
#     vec_0 = []
#     vec_1 = []
#     with torch.no_grad():
#         for n, m in model_1.named_modules():
#             if n in inter_model_tss and layer in n:
#                 # print(model_clean_copy.state_dict()[n+'.layer.weight'][model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
#                 # print(model_test.state_dict()[n+'.layer.weight'][model_test.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
#                 # print(nn.CosineSimilarity(dim=0)(model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1), model_test.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1)))
#                 t2 = torch.zeros_like(model_2.state_dict()[n+'.layer.weight'])
#                 t2[(tss_impt_dict_2[n] == 0)] = model_2.state_dict()[
#                     n+'.layer.weight'][(tss_impt_dict_2[n] == 0)]
#                 t1 = torch.zeros_like(model_1.state_dict()[n+'.layer.weight'])
#                 t1[(tss_impt_dict_1[n] == 0)] = model_1.state_dict()[
#                     n+'.layer.weight'][(tss_impt_dict_1[n] == 0)]

#                 t0_norm = ((t2 - t2.mean())/t2.std()).view(-1)
#                 t1_norm = ((t1 - t1.mean())/t1.std()).view(-1)
#                 # print(t0_norm.shape, t0_norm.sum())
#                 vec_0.append(t0_norm)
#                 vec_1.append(t1_norm)
#                 # print(torch.acos(nn.CosineSimilarity(dim=0)(t0_norm, t1_norm))* 180 / 3.1415926)
#                 # print(torch.dist(t0_norm, t1_norm))
#                 # print(t1_norm)
#     vec_0 = torch.cat(vec_0)
#     vec_1 = torch.cat(vec_1)
#     return (torch.acos(nn.CosineSimilarity(dim=0)(vec_0, vec_1)) * 180 / math.pi)

def pairWiseCosineSim(inter_model_tss, layer, tss_impt_dict_1, model_dict_1, tss_impt_dict_2, model_dict_2, eps=1e-5, positive=0):
    """
    Compute the pairwise cosine similarity between two models' parameters,
    Args:
        inter_model_tss (dict): Dictionary containing the intersection of task-specific importance scores. Type of: {layer_name: tensor}.
        layer (str): The specific layer name to compute similarity for.
        tss_impt_dict_1 (dict): Task-specific importance dictionary for model 1.
        model_dict_1 (dict): State dictionary for model 1.
        tss_impt_dict_2 (dict): Task-specific importance dictionary for model 2.
        model_dict_2 (dict): State dictionary for model 2.
        positive (int, optional): Value indicating positive importance in the TSS importance dictionaries. Default is 0. If 1 is passed, only parameters marked as important (1) are considered.
    Returns:
        float: The minimum angle (in radians) between the normalized parameter vectors of the two models
    """
    vec_0 = []
    vec_1 = []
    with torch.no_grad():
        for n in inter_model_tss:
            if layer in n:
                # print(model_clean_copy.state_dict()[n+'.layer.weight'][model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
                # print(model_test.state_dict()[n+'.layer.weight'][model_test.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
                # print(nn.CosineSimilarity(dim=0)(model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1), model_test.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1)))
                t2 = torch.zeros_like(model_dict_2[n+'.layer.weight'])
                t2[(tss_impt_dict_2[n] == positive) & (inter_model_tss[n] == 0)] = model_dict_2[
                    n+'.layer.weight'][(tss_impt_dict_2[n] == positive) & (inter_model_tss[n] == 0)]
                t1 = torch.zeros_like(model_dict_1[n+'.layer.weight'])
                t1[(tss_impt_dict_1[n] == positive) & (inter_model_tss[n] == 0)] = model_dict_1[
                    n+'.layer.weight'][(tss_impt_dict_1[n] == positive) & (inter_model_tss[n] == 0)]

                t0_norm = ((t2 - t2.mean())/(t2.std()+eps)).view(-1)
                t1_norm = ((t1 - t1.mean())/(t1.std()+eps)).view(-1)
                # print(t0_norm.shape, t0_norm.sum())
                vec_0.append(t0_norm)
                vec_1.append(t1_norm)
                # print(torch.acos(nn.CosineSimilarity(dim=0)(t0_norm, t1_norm))* 180 / 3.1415926)
                # print(torch.dist(t0_norm, t1_norm))
                # print(t1_norm)
    vec_0 = torch.cat(vec_0)
    vec_1 = torch.cat(vec_1)
    return min(torch.acos(nn.CosineSimilarity(dim=0)(vec_0, vec_1)), math.pi)


def pairWiseCosineSimParallel(task, eps=1e-5):
    inter_model_tss, layer, tss_impt_dict_1, model_dict_1, tss_impt_dict_2, model_dict_2, i, j = task
    vec_0 = []
    vec_1 = []
    with torch.no_grad():
        for n in inter_model_tss:
            if layer in n:
                # print(model_clean_copy.state_dict()[n+'.layer.weight'][model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
                # print(model_test.state_dict()[n+'.layer.weight'][model_test.state_dict()[n+'.layer.scores.'+str(ft_task)] >= 0])
                # print(nn.CosineSimilarity(dim=0)(model_clean_copy.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1), model_test.state_dict()[n+'.layer.scores.'+str(ft_task)].view(-1)))
                t2 = torch.zeros_like(model_dict_2[n+'.layer.weight'])
                t2[(tss_impt_dict_2[n] == 0) & (inter_model_tss[n] == 0)] = model_dict_2[
                    n+'.layer.weight'][(tss_impt_dict_2[n] == 0) & (inter_model_tss[n] == 0)]
                t1 = torch.zeros_like(model_dict_1[n+'.layer.weight'])
                t1[(tss_impt_dict_1[n] == 0) & (inter_model_tss[n] == 0)] = model_dict_1[
                    n+'.layer.weight'][(tss_impt_dict_1[n] == 0) & (inter_model_tss[n] == 0)]

                t0_norm = ((t2 - t2.mean())/(t2.std()+eps)).view(-1)
                t1_norm = ((t1 - t1.mean())/(t1.std()+eps)).view(-1)
                # print(t0_norm.shape, t0_norm.sum())
                vec_0.append(t0_norm)
                vec_1.append(t1_norm)
                # print(torch.acos(nn.CosineSimilarity(dim=0)(t0_norm, t1_norm))* 180 / 3.1415926)
                # print(torch.dist(t0_norm, t1_norm))
                # print(t1_norm)
    vec_0 = torch.cat(vec_0)
    vec_1 = torch.cat(vec_1)
    return min(torch.acos(nn.CosineSimilarity(dim=0)(vec_0, vec_1)), math.pi), layer, i, j

def pairWiseCosineSim_v2(activation_path_1, activation_path_2, layer=None, calculation_metric='importance', eps=1e-8):
    """
    Compute the pairwise cosine similarity between two models' importance scores from activation paths.
    
    Args:
        activation_path_1 (dict): Activation path dictionary for model 1 (from get_activation_path_single_image)
        activation_path_2 (dict): Activation path dictionary for model 2 (from get_activation_path_single_image)
        layer (str, optional): Specific layer name to compute similarity for. If None, computes for all common layers.
        eps (float): Small epsilon value to avoid division by zero
        
    Returns:
        dict or float: If layer is None, returns dict with similarities for all layers. 
                      If layer is specified, returns single similarity value.
    """
    
    def compute_layer_similarity(imp1, imp2, eps=eps):
        """Compute cosine similarity between two importance vectors"""
        # Flatten the importance tensors
        vec1 = imp1.view(-1)
        vec2 = imp2.view(-1)
        
        # Check if vectors have the same shape
        if vec1.shape != vec2.shape:
            raise ValueError(f"Importance tensors have different shapes: {vec1.shape} vs {vec2.shape}")
        
        # Normalize vectors (z-score normalization)
        vec1_norm = (vec1 - vec1.mean()) / (vec1.std() + eps)
        vec2_norm = (vec2 - vec2.mean()) / (vec2.std() + eps)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(vec1_norm.unsqueeze(0), vec2_norm.unsqueeze(0))
        
        # Convert to angle (minimum angle between vectors)
        angle = torch.acos(torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps))
        
        return min(angle.item(), math.pi)
    
    # Find common layers between the two activation paths
    common_layers = set(activation_path_1.keys()).intersection(set(activation_path_2.keys()))
    
    if not common_layers:
        raise ValueError("No common layers found between the two activation paths")
    
    # If specific layer is requested
    if layer is not None:
        matching_layers = [l for l in common_layers if layer == l]
        if not matching_layers:
            raise ValueError(f"Layer '{layer}' not found in common layers")
        
        if len(matching_layers) > 1:
            print(f"Warning: Multiple layers match '{layer}': {matching_layers}")
            print(f"Using first match: {matching_layers[0]}")
        
        target_layer = matching_layers[0]
        imp1 = activation_path_1[target_layer][calculation_metric]
        imp2 = activation_path_2[target_layer][calculation_metric]
        
        return compute_layer_similarity(imp1, imp2, eps)
    
    # Compute similarity for all common layers
    similarities = {}
    
    with torch.no_grad():
        for layer_name in sorted(common_layers):
            try:
                imp1 = activation_path_1[layer_name][calculation_metric]
                imp2 = activation_path_2[layer_name][calculation_metric]
                
                similarity = compute_layer_similarity(imp1, imp2, eps)
                similarities[layer_name] = similarity
                
            except Exception as e:
                print(f"Warning: Could not compute similarity for layer {layer_name}: {e}")
                continue
    
    return similarities

def parameter_variability(models, layer_name):
    params = [model.state_dict()[layer_name] for model in models]
    variability = 0.0
    for i in range(len(params)):
        for j in range(i+1, len(params)):
            variability += torch.norm(params[i] - params[j], p='fro').item()
    return variability / (len(params)*(len(params)-1)/2)


def gradient_sensitivity(model, data_loader, device='cuda'):
    model.eval()
    gradients = {}
    for batch in data_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.mean(torch.abs(param.grad)).item()
                if name not in gradients:
                    gradients[name] = []
                gradients[name].append(grad_norm)
        model.zero_grad()

    # 平均梯度敏感度
    sensitivity = {name: np.mean(values) for name, values in gradients.items()}
    return sensitivity


def computeLayerImportance(last_global_dict, params_list):
    '''
    假设有 K 个预训练模型，params_list 是它们的参数列表
    '''

    def normalize_dict_values(d):
        values = list(d.values())
        min_v = min(values)
        max_v = max(values)

        if max_v == min_v:
            # 防止除零，所有值设为0.0（也可以根据需求设为其他）
            return {k: 0.0 for k in d}

        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

    # 初始化元参数 theta（与预训练模型同结构）
    # theta = torch.randn_like(params_list[0])
    # theta = {key: torch.randn_like(params_list[0][key]) for key in params_list[0].keys() if 'weight' in key}
    theta = {key: val for key, val in last_global_dict.items()
             if 'weight' in key}

    # 计算参数差异的方差
    importance = {}
    for key in theta.keys():
        if 'weight' in key:
            diffs = [theta[key] - model[key] for model in params_list]
            importance[key] = torch.var(torch.stack(
                diffs), dim=0).mean().cpu().numpy()

    importance = normalize_dict_values(importance)

    # 输出重要性排名
    sorted_layers = sorted(
        importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_layers


def computeLayerImportance_smooth(last_global_dict, params_list):
    '''
    AI generated
    '''
    import numpy as np
    from scipy.stats import skew, kurtosis

    def normalize_dict_values(d):
        values = list(d.values())
        min_v = min(values)
        max_v = max(values)

        if max_v == min_v:
            # 防止除零，所有值设为0.0
            return {k: 0.0 for k in d}

        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

    # 初始化参数差异字典
    theta = {key: val for key, val in last_global_dict.items()
             if 'weight' in key}
    importance = {}

    for key in theta.keys():
        if 'weight' in key:
            # 计算每个预训练模型与元参数的差异
            diffs = [theta[key] - model[key] for model in params_list]

            # 将张量转换为一维数组用于分析
            flattened_diffs = torch.stack(diffs).flatten().cpu().numpy()

            # 如果数据点太少，使用简单方差
            if len(flattened_diffs) < 30:
                importance[key] = torch.var(torch.stack(
                    diffs), dim=0).mean().cpu().numpy()
                continue

            # 排序差异值用于分析分布特性
            sorted_diffs = np.sort(flattened_diffs)

            # 基础统计量
            mean_diff = np.mean(flattened_diffs)
            std_diff = np.std(flattened_diffs)

            # 防止除零
            if std_diff == 0:
                importance[key] = 0.0
                continue

            # 计算中间区域平滑度 (25%-75%部分)
            middle_start_idx = int(len(sorted_diffs) * 0.25)
            middle_end_idx = int(len(sorted_diffs) * 0.75)
            middle_values = sorted_diffs[middle_start_idx:middle_end_idx]
            middle_smoothness = 1.0 / \
                (np.std(middle_values) + 1e-10)  # 值越高表示越平滑

            # 计算尾部极值程度
            left_tail = sorted_diffs[:middle_start_idx]
            right_tail = sorted_diffs[middle_end_idx:]
            left_tail_gap = abs(mean_diff - np.mean(left_tail)
                                ) if len(left_tail) > 0 else 0
            right_tail_gap = abs(np.mean(right_tail) -
                                 mean_diff) if len(right_tail) > 0 else 0

            # 极值突出性
            max_deviation = np.abs(
                flattened_diffs.max() - mean_diff) / std_diff
            min_deviation = np.abs(
                flattened_diffs.min() - mean_diff) / std_diff
            extremeness_score = max(max_deviation, min_deviation)

            # 偏度和峰度可以帮助识别分布的形状
            skewness = skew(flattened_diffs)
            kurt = kurtosis(flattened_diffs)

            # 综合指标：中间平滑度 * 尾部极值程度 * 原始方差
            # 这样既保留了原始方差的信息，又强调了中间平滑且两端极值突出的层
            base_importance = torch.var(torch.stack(
                diffs), dim=0).mean().cpu().numpy()
            smooth_center_extreme_tails = middle_smoothness * \
                (left_tail_gap + right_tail_gap) / (2 * std_diff + 1e-10)

            # 使用组合指标作为重要性
            importance[key] = base_importance * \
                (1 + smooth_center_extreme_tails)

    # 规范化重要性值
    importance = normalize_dict_values(importance)

    # 按重要性排序
    sorted_layers = sorted(
        importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_layers


def computeNeuronImportance(last_global_dict, params_list, layer_names=None):
    '''
    Neuron-wise importance: each weight element gets its own importance score
    '''

    # Initialize theta using the last global model
    if layer_names is None:
        theta = {key: val for key, val in last_global_dict.items()
                if 'weight' in key}
    else:
        theta = {key: val for key, val in last_global_dict.items()
                 if key in layer_names}

    # Compute variance across models for each parameter element
    importance = {}
    for key in theta.keys():
        if 'weight' in key:
            diffs = [theta[key] - model[key] for model in params_list]
            stacked_diffs = torch.stack(diffs, dim=0)  # Shape: (K, ...)
            # Shape: same as weight tensor
            importance[key] = torch.var(stacked_diffs, dim=0)

    # Dictionary: {layer_name: tensor of same shape as layer weight}
    return importance

def computeNeuronImportance_v2(model, last_global_dict, params_list):
    importance = {}
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_name = n + '.layer.weight'
            diffs = [last_global_dict[layer_name] - model_dict[layer_name] for model_dict in params_list]
            stacked_diffs = torch.stack(diffs, dim=0)  # Shape: (K, ...)
            importance[n] = torch.var(stacked_diffs, dim=0)
    return importance

def computeNeuronImportance_v3(model, last_global_dict, params_list, hard=False):
    importance = {}
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_name = n + '.layer.weight'
            diffs = [last_global_dict[layer_name] - model_dict[layer_name] for model_dict in params_list]
            stacked_diffs = torch.stack(diffs, dim=0)  # Shape: (K, ...)
            variance = torch.var(stacked_diffs, dim=0)
            if hard:
                # Normalize to 0-1 range
                avg_val = torch.mean(variance)
                importance[n] = variance > avg_val
            else:
                importance[n] = variance
    return importance


def importance_to_mask(importance_dict, percentile=80, quant=False):
    """
    Convert importance tensors to binary masks using a global percentile threshold.
    Returns:
        mask_dict: dict with same keys as importance_dict, values are 0/1 masks
        portion_of_ones: float, ratio of 1s to total elements
    """
    # Step 1: Flatten all importance values to compute global threshold
    all_scores = torch.cat([v.flatten() for v in importance_dict.values()])
    if quant:
        threshold = torch.quantile(all_scores, percentile / 100.0)
    else:
        min_score = torch.min(all_scores)
        max_score = torch.max(all_scores)

        # Calculate threshold using min-max scaling
        threshold = min_score + (percentile / 100.0) * (max_score - min_score)

    # Step 2: Generate binary masks and count 1s
    mask_dict = {}
    total_elements = 0
    total_ones = 0

    for key, score_tensor in importance_dict.items():
        mask = (score_tensor > threshold).to(torch.uint8)
        mask_dict[key] = mask
        total_ones += mask.sum().item()
        total_elements += mask.numel()

    portion_of_ones = total_ones / total_elements

    return mask_dict, portion_of_ones


# def is_right_edge_concentrated(matrix: np.ndarray, k: int = 2, threshold: float = 0.45, max_right_var: float = 0.01):
#     """
#     AI generated

#     Args:
#         matrix (np.ndarray): 输入矩阵 (n x n)
#         k (int): 右侧参与计算的列数
#         threshold (float): RECI阈值
#         max_right_var (float): 右侧列均值的方差上限（控制不均匀性）

#     Returns:
#         bool: 是否为符合图1特征的矩阵
#     """
#     col_means = matrix.mean(axis=0)
#     total_mean_sum = col_means.sum()

#     right_means = col_means[-k:]
#     right_sum = right_means.sum()

#     reci = right_sum / total_mean_sum
#     right_var = np.var(right_means)

#     return reci > threshold and right_var < max_right_var

def compute_reci_score(matrix: np.ndarray, k: int = 2, use_penalty: bool = True, penalty_weight: float = 0.1):
    '''
    AI generated
    Args:
        matrix (np.ndarray): 输入矩阵 (n x n)
        k (int): 右侧参与计算的列数
        use_penalty (bool): 是否使用方差惩罚
        penalty_weight (float): 惩罚权重
    '''
    col_means = matrix.mean(axis=0)
    total = col_means.sum()
    right_cols = col_means[-k:]
    score = right_cols.sum() / total
    middle_cols = col_means[k:-k]

    if use_penalty:
        variance_penalty = np.var(right_cols)
        middle_var = np.var(middle_cols)
        score -= penalty_weight * (variance_penalty + middle_var)

    return score


def select_top_matrices(matrices: dict, top_k: int = 5, threshold: float = None,
                        k: int = 2, use_penalty=True):
    '''
    AI generated
    '''
    scored = []
    matrices = sort_mat(matrices)
    for layer, mat in matrices.items():
        score = compute_reci_score(mat, k=k, use_penalty=use_penalty)
        scored.append((layer, score))

    if threshold is not None:
        selected = [layer for layer, score in scored if score >= threshold]
    else:
        # Sort by score descending and take top_k
        selected = [layer for layer, score in sorted(
            scored, key=lambda x: x[1], reverse=True)[:top_k]]

    # return argsort format with indices
    selected = [i for i, layer in enumerate(scored) if layer[0] in selected]

    return selected, scored


def sort_mat(pair_wise_cos_mat: dict[str, np.ndarray]):
    """
    Sort the matrix by the sum of each row and column.
    """
    pair_wise_cos_mat_plot = {}
    for layer, mat in pair_wise_cos_mat.items():
        # Calculate column means
        col_means = np.mean(mat, axis=0)
        # Get sorted indices based on column means (ascending order)
        sorted_indices = np.argsort(col_means)
        # Reorder the matrix rows and columns according to the sorted indices
        pair_wise_cos_mat_plot[layer] = mat[sorted_indices,
                                            :][:, sorted_indices]

    return pair_wise_cos_mat_plot


def normalize_state_dicts_layerwise(model_state_dict_list, alpha=0.2):
    '''
        !!AI generated
        model_state_dict_list: List of (index, state_dict) tuples
        alpha: normalization strength (0 = no normalization, 1 = full normalization)

        Returns:
            normalized_state_dict_list: List of (index, normalized_state_dict) tuples
    '''

    # 第一步：收集所有参数，按层名分组
    layerwise_params = {}  # key: layer_name, value: list of tensors

    for idx, state_dict in model_state_dict_list:
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if key not in layerwise_params:
                    layerwise_params[key] = []
                if not torch.is_floating_point(param):
                    param = param.float()
                layerwise_params[key].append(param)

    # 第二步：计算每层的 global mean 和 std，并归一化
    layerwise_norm_stats = {}  # key: layer_name, value: (mean, std)
    for key, tensors in layerwise_params.items():
        stacked = torch.stack(tensors)  # shape: [num_models, ...]
        mean = stacked.mean()
        std = stacked.std()
        layerwise_norm_stats[key] = (mean, std)

    # normalized_state_dict_list = {}
    # for idx, state_dict in model_state_dict_list:
    #     normalized_state_dict = {}
    #     for key, param in state_dict.items():
    #         if isinstance(param, torch.Tensor):
    #             original_dtype = param.dtype
    #             if not torch.is_floating_point(param):
    #                 param = param.float()
    #             mean, std = layerwise_norm_stats[key]
    #             # 软归一化：alpha控制归一化强度
    #             normalized_param = param + alpha * ((param - mean) / (std + 1e-5) - param)
    #             # 或者更简单的形式：
    #             # normalized_param = (1 - alpha) * param + alpha * (param - mean) / (std + 1e-5)

    #             if original_dtype != torch.float32:
    #                 normalized_param = normalized_param.to(original_dtype)
    #             normalized_state_dict[key] = normalized_param
    #         else:
    #             normalized_state_dict[key] = param
    #     normalized_state_dict_list[idx] = normalized_state_dict
    # return normalized_state_dict_list

    all_stds = []
    layerwise_norm_stats = {}
    for key, tensors in layerwise_params.items():
        stacked = torch.stack(tensors)
        mean = stacked.mean()
        std = stacked.std()
        layerwise_norm_stats[key] = (mean, std)
        all_stds.append(std.item())

    # 计算自适应权重
    global_std_median = torch.tensor(all_stds).median()

    normalized_state_dict_list = {}
    for idx, state_dict in model_state_dict_list:
        normalized_state_dict = {}
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                original_dtype = param.dtype
                if not torch.is_floating_point(param):
                    param = param.float()
                mean, std = layerwise_norm_stats[key]

                # 自适应权重：方差越大，归一化强度越小
                adaptive_weight = torch.clamp(
                    global_std_median / (std + 1e-5), 0.1, alpha)

                normalized_param = param + adaptive_weight * \
                    ((param - mean) / (std + 1e-5) - param)

                if original_dtype != torch.float32:
                    normalized_param = normalized_param.to(original_dtype)
                normalized_state_dict[key] = normalized_param
            else:
                normalized_state_dict[key] = param
        normalized_state_dict_list[idx] = normalized_state_dict

    return normalized_state_dict_list


def get_activation_path_single_image(model, single_input, target_label, ft_task=0, device='cuda'):
    """
    Get the activation path (back propagation) for a single input image

    Args:
        model: The neural network model with SoftMaskedLayer modules
        single_input: Single input image tensor (should be on same device as model)
        target_label: Target label for the input (can be ground truth or predicted)
        ft_task: Fine-tuning task index (default: 0)

    Returns:
        activation_path_dict: Dictionary containing gradient information for each SoftMaskedLayer
    """
    criterion = nn.CrossEntropyLoss()

    # Set model to appropriate mode for getting activation path
    model.train()  # or model.train() depending on your needs
    set_compute_mask_impt(model, True)
    set_ft_task(model, ft_task)

    # Ensure input is on correct device and has batch dimension
    if single_input.dim() == 3:  # Add batch dimension if not present
        single_input = single_input.unsqueeze(0)

    single_input = single_input.to(device)
        # Fix target label formatting
    if isinstance(target_label, int):
        target_label = torch.tensor([target_label], device=device)
    elif target_label.dim() == 0:  # scalar tensor
        target_label = target_label.unsqueeze(0).to(device)
    else:
        target_label = target_label.to(device)
        if target_label.dim() > 1:
            target_label = target_label.squeeze()
        if target_label.dim() == 0:
            target_label = target_label.unsqueeze(0)

    # Clear any existing gradients
    model.zero_grad()

    # Forward pass
    with torch.enable_grad():
        single_input.requires_grad_(True)  # Enable gradients for input if needed
        output = model(single_input)
        
        # Compute loss
        loss = criterion(output, target_label)
        
        # Backward pass to compute gradients
        loss.backward()

    # Collect activation path information
    activation_path_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, SoftMaskedLayer):
            # Get gradient information (activation path)
            if hasattr(module.layer, 'alphas') and module.layer.alphas.grad is not None:
                grad = module.layer.alphas.grad.clone().detach()
                importance = impt_norm(grad)
                thresh_mask = (importance >= 0.99)
                activation_path_dict[name] = {
                    'gradient': grad,
                    'importance': importance,
                    'importance_thresholded': importance * thresh_mask.float(),
                    'active_neurons': int(thresh_mask.sum().item()),
                    'total_neurons': grad.numel()
                }
                
                # Calculate activation percentage
                activation_percentage = activation_path_dict[name]['active_neurons'] / activation_path_dict[name]['total_neurons']
                activation_path_dict[name]['activation_percentage'] = activation_percentage
                
                print(f'Layer: {name}')
                print(f'  Active neurons: {activation_path_dict[name]["active_neurons"]}/{activation_path_dict[name]["total_neurons"]} ({activation_percentage:.2%})')

    # Clear gradients after collection
    model.zero_grad()

    return activation_path_dict

def GSS_get_activation_path_single_image(model, single_input, target_label, ft_task=0, device='cuda'):
    """
    Get the activation path (back propagation) for a single input image

    Args:
        model: The neural network model with SoftMaskedLayer modules
        single_input: Single input image tensor (should be on same device as model)
        target_label: Target label for the input (can be ground truth or predicted)
        ft_task: Fine-tuning task index (default: 0)

    Returns:
        activation_path_dict: Dictionary containing gradient information for each SoftMaskedLayer
    """
    criterion = nn.CrossEntropyLoss()

    # Set model to appropriate mode for getting activation path
    model.train()  # or model.train() depending on your needs

    # Ensure input is on correct device and has batch dimension
    if single_input.dim() == 3:  # Add batch dimension if not present
        single_input = single_input.unsqueeze(0)

    single_input = single_input.to(device)
        # Fix target label formatting
    if isinstance(target_label, int):
        target_label = torch.tensor([target_label], device=device)
    elif target_label.dim() == 0:  # scalar tensor
        target_label = target_label.unsqueeze(0).to(device)
    else:
        target_label = target_label.to(device)
        if target_label.dim() > 1:
            target_label = target_label.squeeze()
        if target_label.dim() == 0:
            target_label = target_label.unsqueeze(0)

    # Clear any existing gradients
    model.zero_grad()

    # Forward pass
    with torch.enable_grad():
        single_input.requires_grad_(True)  # Enable gradients for input if needed
        output = model(single_input)
        
        # Compute loss
        loss = criterion(output, target_label)
        
        # Backward pass to compute gradients
        loss.backward()

    # Collect activation path information
    activation_path_dict = {}

    for name, module in model.named_modules():
        # Get gradient information (activation path)
        if hasattr(module, 'weight') and module.weight.grad is not None:
            grad = module.weight.grad.clone().detach()
            importance = impt_norm(grad)
            thresh_mask = (importance >= 0.9)
            activation_path_dict[name] = {
                'gradient': grad,
                'importance': importance,
                'importance_thresholded': importance * thresh_mask.float(),
                'active_neurons': int(thresh_mask.sum().item()),
                'total_neurons': grad.numel()
            }
            
            # Calculate activation percentage
            activation_percentage = activation_path_dict[name]['active_neurons'] / activation_path_dict[name]['total_neurons']
            activation_path_dict[name]['activation_percentage'] = activation_percentage
            
            print(f'Layer: {name}')
            print(f'  Active neurons: {activation_path_dict[name]["active_neurons"]}/{activation_path_dict[name]["total_neurons"]} ({activation_percentage:.2%})')

    # Clear gradients after collection
    model.zero_grad()

    return activation_path_dict


# Example usage:
def example_single_image_activation_path(model, model_dict, dataset, device):
    # Assuming you have your model, and a single image
    # single_image should be a tensor of shape [C, H, W] or [1, C, H, W]
    
    # # Get a single image from your test loader (example)
    # for data in test_loader:
    #     inputs, labels = data
    #     single_image = inputs[0]  # Take first image from batch
    #     true_label = labels[0]    # Take corresponding label
    #     break

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.load_state_dict(model_dict)
    inputs, labels = next(iter(test_loader))
    single_image = inputs[0]
    true_label = labels[0]
    
    # Get activation path
    activation_path = get_activation_path_single_image(
        model=model,
        single_input=single_image,
        target_label=true_label,
        ft_task=0,
        device=device
    )
    
    return activation_path

def example_batch_image_activation_path(model, model_dict, dataset, ft_task=1, device='cuda'):
    # Assuming you have your model, and a single image
    # single_image should be a tensor of shape [C, H, W] or [1, C, H, W]
    
    # # Get a single image from your test loader (example)
    # for data in test_loader:
    #     inputs, labels = data
    #     single_image = inputs[0]  # Take first image from batch
    #     true_label = labels[0]    # Take corresponding label
    #     break

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.load_state_dict(model_dict)
    for batch_idx, (batch_input, batch_labels) in enumerate(test_loader):
            
        print(f"Batch shape: {batch_input.shape}")
        
        # Get activation path for this batch
        batch_activation_path = get_activation_path_batch(
            model=model,
            batch_input=batch_input,
            batch_labels=batch_labels,
            ft_task=ft_task,
            device=device,
            aggregation='mean'  # or 'sum', 'individual', 'mean'
        )
        
        activation_paths = batch_activation_path
        break
    
    return activation_paths

def get_activation_path_batch(model, batch_input, batch_labels, ft_task=0, device='cuda', aggregation='individual'):
    """
    Get the activation path (back propagation) for a batch of input images
    
    Args:
        model: The neural network model with SoftMaskedLayer modules
        batch_input: Batch of input images tensor from dataloader
        batch_labels: Batch of target labels from dataloader
        ft_task: Fine-tuning task index (default: 0)
        device: Device to run computation on
        aggregation: How to aggregate gradients across batch ('mean', 'sum', 'individual')
    
    Returns:
        activation_path_dict: Dictionary containing gradient information for each SoftMaskedLayer
    """
    criterion = nn.CrossEntropyLoss()
    
    # Set model to appropriate mode for getting activation path
    model.eval()  # or model.train() depending on your needs
    set_compute_mask_impt(model, True)
    set_ft_task(model, ft_task)
    
    # Ensure inputs are on correct device
    batch_input = batch_input.to(device)
    batch_labels = batch_labels.to(device)
    
    # Handle different label formats
    if batch_labels.dim() > 1:
        batch_labels = batch_labels.squeeze()
    
    batch_size = batch_input.size(0)
    
    # Clear any existing gradients
    model.zero_grad()
    
    if aggregation == 'individual':
        # Store individual activation paths for each sample
        batch_activation_paths = []
        
        for i in range(batch_size):
            model.zero_grad()
            single_input = batch_input[i:i+1]  # Keep batch dimension
            single_label = batch_labels[i:i+1]
            
            with torch.enable_grad():
                single_input.requires_grad_(True)
                output = model(single_input)
                loss = criterion(output, single_label)
                loss.backward()
            
            # Collect activation path for this sample
            sample_activation_path = {}
            for name, module in model.named_modules():
                if isinstance(module, SoftMaskedLayer):
                    if hasattr(module.layer, 'alphas') and module.layer.alphas.grad is not None:
                        grad = module.layer.alphas.grad.clone().detach()
                        importance = impt_norm(grad)
                        thresh_mask = (importance >= 0.9)
                        sample_activation_path[name] = {
                            'gradient': module.layer.alphas.grad.clone().detach(),
                            'importance': impt_norm(module.layer.alphas.grad.clone().detach()),
                            'importance_thresholded': importance * thresh_mask.float(),
                        }
            
            batch_activation_paths.append(sample_activation_path)
        
        return batch_activation_paths
    
    else:
        # Aggregate gradients across the entire batch
        with torch.enable_grad():
            batch_input.requires_grad_(True)
            output = model(batch_input)
            
            # Compute loss for the entire batch
            loss = criterion(output, batch_labels)
            
            # Backward pass to compute gradients
            loss.backward()
        
        # Collect activation path information
        activation_path_dict = {}
        
        for name, module in model.named_modules():
            if isinstance(module, SoftMaskedLayer):
                if hasattr(module.layer, 'alphas') and module.layer.alphas.grad is not None:
                    grad = module.layer.alphas.grad.clone().detach()
                    importance = impt_norm(grad)
                    thresh_mask = (importance >= 0.9)
                    
                    # Apply aggregation if specified
                    if aggregation == 'sum':
                        # Gradients are already summed across batch by default
                        aggregated_grad = grad
                    elif aggregation == 'mean':
                        # Average the gradients across batch
                        aggregated_grad = grad / batch_size
                    else:
                        aggregated_grad = grad
                    
                    importance = impt_norm(aggregated_grad)
                    active_neurons = (importance >= 0.5).sum().item()
                    total_neurons = aggregated_grad.numel()
                    
                    activation_path_dict[name] = {
                        'gradient': aggregated_grad,
                        'importance': importance,
                        'importance_thresholded': importance * thresh_mask.float(),
                        'active_neurons': active_neurons,
                        'total_neurons': total_neurons,
                        'activation_percentage': active_neurons / total_neurons,
                        'batch_size': batch_size
                    }
                    
                    print(f'Layer: {name}')
                    print(f'  Batch size: {batch_size}')
                    print(f'  Active neurons: {active_neurons}/{total_neurons} ({activation_path_dict[name]["activation_percentage"]:.2%})')
        
        # Clear gradients after collection
        model.zero_grad()
        
        return activation_path_dict

def get_neuron_wise_activation_analysis(activation_clean, activation_corrupt, layer_name=None, threshold=0.5):
    """
    Analyze neuron-wise activation differences between clean and corrupted inputs
    
    Args:
        activation_clean: Dictionary containing activation info for clean input
        activation_corrupt: Dictionary containing activation info for corrupted input
        layer_name: Specific layer to analyze (if None, analyzes all layers)
        threshold: Threshold for considering a neuron as "active"
    
    Returns:
        neuron_analysis_dict: Dictionary containing detailed neuron-wise analysis
    """
    
    neuron_analysis_dict = {}
    
    # Select layers to analyze
    layers_to_analyze = [layer_name] if layer_name else list(activation_clean.keys())
    
    for layer in layers_to_analyze:
        if layer not in activation_clean or layer not in activation_corrupt:
            print(f"Warning: Layer {layer} not found in both activation dictionaries")
            continue
            
        # Get importance scores (normalized gradients)
        clean_importance = activation_clean[layer]['importance']
        corrupt_importance = activation_corrupt[layer]['importance']
        
        # Convert to binary activation maps based on threshold
        clean_active = (clean_importance >= threshold)
        corrupt_active = (corrupt_importance >= threshold)
        
        # Calculate differences
        importance_diff = corrupt_importance - clean_importance
        activation_diff = corrupt_active.float() - clean_active.float()
        
        # Categorize neurons based on activation changes
        newly_activated = (corrupt_active & ~clean_active)  # Activated in corrupt but not clean
        newly_deactivated = (~corrupt_active & clean_active)  # Deactivated in corrupt
        consistently_active = (corrupt_active & clean_active)  # Active in both
        consistently_inactive = (~corrupt_active & ~clean_active)  # Inactive in both
        
        # Store analysis results
        neuron_analysis_dict[layer] = {
            'clean_importance': clean_importance,
            'corrupt_importance': corrupt_importance,
            'importance_diff': importance_diff,
            'clean_active': clean_active,
            'corrupt_active': corrupt_active,
            'activation_diff': activation_diff,
            'newly_activated': newly_activated,
            'newly_deactivated': newly_deactivated,
            'consistently_active': consistently_active,
            'consistently_inactive': consistently_inactive,
            'stats': {
                'total_neurons': clean_importance.numel(),
                'newly_activated_count': newly_activated.sum().item(),
                'newly_deactivated_count': newly_deactivated.sum().item(),
                'consistently_active_count': consistently_active.sum().item(),
                'consistently_inactive_count': consistently_inactive.sum().item(),
                'clean_active_count': clean_active.sum().item(),
                'corrupt_active_count': corrupt_active.sum().item(),
                'max_importance_increase': importance_diff.max().item(),
                'max_importance_decrease': importance_diff.min().item(),
                'mean_importance_change': importance_diff.mean().item(),
                'std_importance_change': importance_diff.std().item()
            }
        }
    
    return neuron_analysis_dict

def find_most_affected_neurons(neuron_analysis_dict, layer_name, top_k=10):
    """
    Find and analyze the most affected neurons in a specific layer
    
    Args:
        neuron_analysis_dict: Output from get_neuron_wise_activation_analysis
        layer_name: Name of the layer to analyze
        top_k: Number of top neurons to return
    
    Returns:
        Dictionary containing indices and information about most affected neurons
    """
    
    if layer_name not in neuron_analysis_dict:
        print(f"Layer {layer_name} not found in analysis dictionary")
        return None
    
    data = neuron_analysis_dict[layer_name]
    importance_diff = data['importance_diff'].flatten()
    
    # Find neurons with highest absolute change
    abs_diff = torch.abs(importance_diff)
    top_indices = torch.argsort(abs_diff, descending=True)[:top_k]
    
    results = {
        'layer_name': layer_name,
        'most_affected_neurons': []
    }
    
    print(f"\nTop {top_k} Most Affected Neurons in {layer_name}:")
    print("=" * 60)
    
    for i, idx in enumerate(top_indices):
        idx_item = idx.item()
        clean_imp = data['clean_importance'].flatten()[idx].item()
        corrupt_imp = data['corrupt_importance'].flatten()[idx].item()
        diff = importance_diff[idx].item()
        
        neuron_info = {
            'rank': i + 1,
            'neuron_index': idx_item,
            'clean_importance': clean_imp,
            'corrupt_importance': corrupt_imp,
            'importance_change': diff,
            'activation_status': 'Newly Activated' if data['newly_activated'].flatten()[idx] else 
                              'Newly Deactivated' if data['newly_deactivated'].flatten()[idx] else
                              'Changed Activity'
        }
        
        results['most_affected_neurons'].append(neuron_info)
        
        print(f"Rank {i+1:2d}: Neuron {idx_item:6d} | "
              f"Clean: {clean_imp:.4f} | Corrupt: {corrupt_imp:.4f} | "
              f"Change: {diff:+.4f} | Status: {neuron_info['activation_status']}")
    
    return results


def GSS_example_single_image_activation_path(model, model_dict, dataset, device):
    # Assuming you have your model, and a single image
    # single_image should be a tensor of shape [C, H, W] or [1, C, H, W]
    
    # # Get a single image from your test loader (example)
    # for data in test_loader:
    #     inputs, labels = data
    #     single_image = inputs[0]  # Take first image from batch
    #     true_label = labels[0]    # Take corresponding label
    #     break

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.load_state_dict(model_dict)
    inputs, labels = next(iter(test_loader))
    single_image = inputs[0]
    true_label = labels[0]
    
    # Get activation path
    activation_path = GSS_get_activation_path_single_image(
        model=model,
        single_input=single_image,
        target_label=true_label,
        ft_task=0,
        device=device
    )
    
    return activation_path