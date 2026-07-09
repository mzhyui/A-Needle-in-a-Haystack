from argparse import Namespace
import copy
import gc
import math
import os
from collections import Counter


import subprocess
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.gradEval import analyze_model_outliers, calculate_model_correlations_from_neuron_analysis, find_top_k_outliers, visualize_model_correlations, find_top_k_clustered_outliers, visualize_clustered_outliers
from utils.logger import myLogger
from utils.dataUtils import SynthesizeDataset, getDataWithDistribution
from models.softmaskLayer import SoftMaskedLayer
# from models.resnet import BasicBlockSM
# from models.ViT import EncoderSM, BlockSM, MLPSM, AttentionHeadSM, EmbeddingsSM, PatchEmbeddingsSM
from models.smFunction import set_compute_mask_impt, set_ft_task, impt_norm
from utils.dataUtils import FilteredDataset, ShrinkedDataset, NoiseDataset, CorruptedDataset, VoidDataset
from utils.tracer import detecting_expert_1_2, detecting_expert_3, processing_updates
from utils.krum import KrumDefense
from utils.subnetworkUtils import GSSCompare, TSSCompare, GradCompare, getTssImpt, getTssImpt_withNoise, getTssCommon, normalize_state_dicts_layerwise, plot_tss_value, pairWiseCosineSim, norm_tss_value, sort_mat, select_top_matrices, TSSCompare_sub, TSSCompare_v2, TSSCompare_v2_batch
from utils.subnetworkUtils import computeLayerImportance_smooth, computeNeuronImportance, importance_to_mask
from utils.assess import _prepare_datasets_trigger, _prepare_datasets_noise


class Defender(object):
    """
    config for fl defense
    """
    historical_excluded_list = []

    def __init__(self, args: Namespace, logger: myLogger | None = None, **kwargs):
        # TODO 2025-02-26 git.V.894ba: logger cannot be broadcasted to parallel processes
        self.args = args
        self.basedir = args.base_dir
        self.logger = logger

        self.enabled = args.enable_rb

        # self.robust_strategy:str = args.robust_strategy
        self.robust_range: list = args.rb_range
        self.robust_list: list = []

        self.robust_wait: bool = args.rb_wait  # wait for outer scripts done

        # self.robust_rate:int = args.rb_rate # the penalty possibility
        self.penalty_rate: float = args.penalty  # the penalty rate, 'w *= p'
        # self.robust_rootpth = args.rb_rootpth # to load the rb list from file
        self.dataset_train, self.dataset_test, _, _ = getDataWithDistribution(
            args)

        self.visual_path = os.path.join(self.basedir, 'visual')

    def guess(self, **kwargs) -> list:
        """
        guess the true or false of a client being attacker
        """
        self.robust_list = [0.5] * len(self.robust_list)
        return self.robust_list

    def defense_mae(self, model: torch.nn.Module, fl_base_path: str, current_round: int,
                    # attack_portion: float = 0.2, collect_rounds: int = 5,
                    # mae_train_epoch: int = 50, recover_range: tuple = (0,70), script_path: str = '/home/mzhyui/git/Signal_and_nn/',
                    # ratio: float = 1.0,
                    # blocks: int = 3, image_size: int = 48, crop_pos: int = 0, patch_size: int = 4,
                    ) -> dict:
        # 执行另一个 Python 脚本，并传递参数
        # %% make dataset
        # for loading saved files, not for defense strategy
        attack_portion: float = self.args.portion
        collect_rounds: int = self.args.mae_collect_rounds
        mae_train_epoch: int = self.args.mae_train_epoch
        recover_range: tuple = self.args.mae_recover_range
        script_path: str = self.args.script_path
        ratio: float = self.args.mae_train_ratio
        blocks: int = self.args.blocks
        image_size: int = self.args.image_size
        crop_pos: int = self.args.crop_pos
        patch_size: int = self.args.patch_size

        '''
        python makePkl.py --data_path ../FL_box_dev/fl_save/cifar10/resnet20_iidFalse_num1000_C1_le2_DBATrue/shard5/dba11-30--00-20-25/ --round_range 41 51
        '''
        try:
            print('making dataset with client weights')
            subprocess.run(['python', os.path.join(script_path, 'makePkl.py'),
                            '--data_path', fl_base_path,
                            '--save_path', os.path.join(script_path,
                                                        './datasets/nn_weights'),
                            '--round_range', str(current_round -
                                                 collect_rounds), str(current_round),
                            ], check=False)
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            exit(0)
        # %% train mae
        '''
        python NN_train_mae.py --rounds 10 --blocks 3 --image_size 48 --crop_size 48 --batch_size 64 --lr_partition --total_epoch 100 --warmup_epoch 5 --use_x_loss --save_path ./checkpoints/ --ratio 0.2
        '''
        try:
            print('training mae model')
            subprocess.run(['python', os.path.join(script_path, 'NN_train_mae.py'),
                            # '--config', os.path.join(script_path, './conf/cratemaetrain.yaml'),
                            '--rounds', str(collect_rounds),
                            ' --blocks', str(blocks),
                            '--chunks', '1',
                            '--dataset_path', os.path.join(
                                script_path, './datasets/nn_weights'),
                            '--ratio', str(ratio),
                            '--image_size', str(image_size),
                            '--crop_size', str(image_size),
                            '--crop_pos', str(crop_pos),
                            '--patch_size', str(patch_size),
                            '--total_epoch', str(mae_train_epoch),
                            '--warmup_epoch', '5',
                            '--batch_size', '128',
                            '--max_device_batch_size', '128',
                            '--latest',
                            '--save_path', os.path.join(script_path,
                                                        './checkpoints/recover/'),
                            ], check=False)
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            exit(0)

        # %% make weight
        '''
        python makeWeight.py --blocks 3 --image_size 48 --crop_size 48 --patch_size 4 --load_checkpoint ./checkpoints/nncrate_mae_nn_2024-12-08_17-34-59/nncrate_mae_100.pth \
        --load_flmodel ../FL_box_dev/fl_save/cifar10/resnet20_iidFalse_num1000_C1_le2_DBATrue/shard5/dba11-30--00-20-25/fed/attack_portion0.2_model_50.pt
        '''
        try:
            print('making recovered global model with trained mae')
            subprocess.run(['python', os.path.join(script_path, 'makeWeight.py'),
                            #  '--config', os.path.join(script_path, './conf/nnmakeweight.yaml'),
                            '--blocks', str(blocks),
                            '--image_size', str(image_size),
                            '--crop_size', str(image_size),
                            '--crop_pos', str(crop_pos),
                            '--patch_size', str(patch_size),
                            '--load_checkpoint', os.path.join(
                                script_path, f'./checkpoints/recover/nncrate_mae_nn_latest/nncrate_mae_{mae_train_epoch}.pth'),
                            '--load_flmodel', os.path.join(
                                fl_base_path, 'fed', f'attack_portion{attack_portion}_model_{current_round}.pt'),
                            '--save_path', os.path.join(script_path, './data'),
                            ], check=False)
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            exit(0)

        # %% recover
        '''
        python resnet_eval.py --blocks 3 --load_flmodel ../FL_box_dev/fl_save/cifar10/resnet20_iidFalse_num1000_C1_le2_DBATrue/shard5/dba11-30--00-20-25/fed/attack_portion0.2_model_50.pt --load_matrix ./data/flmodel_recovered.npy
        '''
        try:
            print('eval and recover the predicted global model')
            subprocess.run(['python', os.path.join(script_path, 'resnet_eval.py'),
                            '--load_flmodel', os.path.join(
                                fl_base_path, 'fed', f'attack_portion{attack_portion}_model_{current_round}.pt'),
                            '--load_matrix', os.path.join(
                                script_path, './data/normal.npy'),
                            '--blocks', str(blocks),
                            '--recover_selection', str(
                                recover_range[0]), str(recover_range[1]),
                            '--save_recovery', os.path.join(
                                script_path, './data/recovery.pth'),
                            ], check=False)
        except Exception as e:
            if self.logger:
                self.logger.error(e)
            exit(0)

        state_dict = torch.load(os.path.join(
            script_path, './data/recovery.pth'))
        return state_dict

    def defenseSubnetmasking(self, model, model_state_dict, idx=-1, device='cuda'):
        model = copy.deepcopy(model)
        model.load_state_dict(model_state_dict)
        model.eval()
        ft_task = 0
        set_compute_mask_impt(model, True)
        set_ft_task(model, ft_task)

        criterion = torch.nn.CrossEntropyLoss()
        tss_impt_dict = {}

        dataloader = DataLoader(
            self.dataset_test, batch_size=128, shuffle=False, num_workers=self.args.max_workers)

        for step, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for n, m in model.named_modules():
                if isinstance(m, SoftMaskedLayer):
                    if n in tss_impt_dict:
                        tss_impt_dict[n] += m.layer.alphas.grad.clone().detach()
                    else:
                        tss_impt_dict[n] = m.layer.alphas.grad.clone().detach()

        return tss_impt_dict

    def defenseTssCompare(self, model, model_state_dict_list, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, frac=0.5, synthesize=0, bn_only=0, perform_normalize=False, sorted_mat=False, plot=True):
        """
            model: the model structure
            model_state_dict_list: list of (id, state_dict)

        """
        # %% normalize the state dicts layerwise
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        # %% prepare dataset
        if synthesize:
            if synthesize == 1:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test), [label]) for label in range(num_classes)]
            elif synthesize == 2:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test, use_interclass_distribution=True), [label]) for label in range(num_classes)]
            else:
                raise ValueError('synthesize should be 0, 1 or 2')
        else:
            dataset_test_list = [FilteredDataset(
                ShrinkedDataset(self.dataset_test, frac=frac), [label]) for label in range(num_classes)]

        # noise_dataset = [FilteredDataset(
        #         NoiseDataset(ShrinkedDataset(self.dataset_test, frac=10.0), victim_labels=list(range(num_classes))),
        #         [label]
        #     ) for label in range(num_classes)]

        # %% get pairwise cosine similarity matrix
        pair_wise_cos_mat = TSSCompare(
            model, model_state_dicts, dataset_test_list, meta_mask=meta_mask, threshold=threshold, threshold_common=threshold_common, torch_mp=self.args.tss_mp, device=self.args.device, with_noise=False)
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}
        
        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat

    def defenseTssCompare_trigger(self, model, model_state_dict_list, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, frac=0.5, info=[], bn_only=False, perform_normalize=False, sorted_mat=False, plot=True):
        """
            model: the model structure
            model_state_dict_list: list of (id, state_dict)

        """
        # %% normalize the state dicts layerwise
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        # %% prepare dataset
        mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size = info
        corrupt_ds = _prepare_datasets_trigger(
            self.dataset_test, mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size)

        dataset_test_list = corrupt_ds

        # noise_dataset = [FilteredDataset(
        #         NoiseDataset(ShrinkedDataset(self.dataset_test, frac=10.0), victim_labels=list(range(num_classes))),
        #         [label]
        #     ) for label in range(num_classes)]

        # %% get pairwise cosine similarity matrix
        pair_wise_cos_mat = TSSCompare(
            model, model_state_dicts, dataset_test_list, meta_mask=meta_mask, threshold=threshold, threshold_common=threshold_common, torch_mp=self.args.tss_mp, device=self.args.device,
            with_noise=False,
            reverse=True
        )
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}

        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat

    def defenseTssCompare_noise(self, model, model_state_dict_list, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, frac=0.5, info=[], bn_only=False, perform_normalize=False, sorted_mat=False, plot=True):
        """
            model: the model structure
            model_state_dict_list: list of (id, state_dict)

        """
        # %% normalize the state dicts layerwise
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        # %% prepare dataset
        mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size = info
        corrupt_ds = _prepare_datasets_noise(
            self.dataset_test, num_class)

        dataset_test_list = corrupt_ds

        # noise_dataset = [FilteredDataset(
        #         NoiseDataset(ShrinkedDataset(self.dataset_test, frac=10.0), victim_labels=list(range(num_classes))),
        #         [label]
        #     ) for label in range(num_classes)]

        # %% get pairwise cosine similarity matrix
        pair_wise_cos_mat = TSSCompare(
            model, model_state_dicts, dataset_test_list, meta_mask=meta_mask, threshold=threshold, threshold_common=threshold_common, torch_mp=self.args.tss_mp, device=self.args.device,
            with_noise=self.args.tss_with_noise,
            reverse=True
        )
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}

        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat

    def eval_tssCompare(self, model, model_state_dict_list, trigger, pos_choice, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, synthesize=0, bn_only=False, perform_normalize=False, sorted_mat=False, plot=True, img_channel=3, img_size=32, num_class=-1, attacking_label=-1):
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        if synthesize:
            if synthesize == 1:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test), [label]) for label in range(num_classes)]
            elif synthesize == 2:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test, use_interclass_distribution=True), [label]) for label in range(num_classes)]
            else:
                raise ValueError('synthesize should be 0, 1 or 2')
        else:
            dataset_test_list = [CorruptedDataset(VoidDataset(FilteredDataset(
                ShrinkedDataset(self.dataset_test, frac=0.1), [label])),
                [attacking_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size)
                for label in range(num_classes)]

        # noise_dataset = [FilteredDataset(
        #         NoiseDataset(ShrinkedDataset(self.dataset_test, frac=10.0), victim_labels=list(range(num_classes))),
        #         [label]
        #     ) for label in range(num_classes)]

        pair_wise_cos_mat = TSSCompare_sub(
            model, model_state_dicts, dataset_test_list, meta_mask=meta_mask, threshold=threshold, threshold_common=threshold_common, device=self.args.device, positive=1)
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}
        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat

    def eval_tssCompare_v2(self, model, model_state_dict_list, trigger, pos_choice, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, synthesize=0, bn_only=False, perform_normalize=False, sorted_mat=False, plot=True, img_channel=3, img_size=32, num_class=-1, attacking_label=-1):
        model = copy.deepcopy(model)
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        dataset = CorruptedDataset(VoidDataset(FilteredDataset(ShrinkedDataset(self.dataset_test, frac=0.1), [attacking_label])), [
                                   attacking_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size)

        pair_wise_cos_mat = TSSCompare_v2(
            model, model_state_dicts, dataset, device=self.args.device)
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}
        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat
    
    def eval_gssCompare(self, model, model_state_dict_list, trigger, pos_choice, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, synthesize=0, bn_only=False, perform_normalize=False, sorted_mat=False, plot=True, img_channel=3, img_size=32, num_class=-1, attacking_label=-1):
        model = copy.deepcopy(model)
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        dataset = CorruptedDataset(VoidDataset(FilteredDataset(ShrinkedDataset(self.dataset_test, frac=0.1), [attacking_label])), [
                                   attacking_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size)

        pair_wise_cos_mat = GSSCompare(
            model, model_state_dicts, dataset, device=self.args.device)
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}
        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat

    def defenseGradCompare(self, model, model_state_dict_list, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, frac=0.5, synthesize=0, bn_only=False, perform_normalize=False, sorted_mat=False, plot=True):
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        if synthesize:
            if synthesize == 1:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test), [label]) for label in range(num_classes)]
            elif synthesize == 2:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test, use_interclass_distribution=True), [label]) for label in range(num_classes)]
            else:
                raise ValueError('synthesize should be 0, 1 or 2')
        else:
            dataset_test_list = [FilteredDataset(
                ShrinkedDataset(self.dataset_test, frac=frac), [label]) for label in range(num_classes)]

        pair_wise_cos_mat = GradCompare(
            model, model_state_dicts, dataset_test_list, meta_mask=meta_mask, threshold=threshold, threshold_common=threshold_common, device=self.args.device)
        if bn_only == 1:
            pair_wise_cos_mat = {layer: mat for layer,
                                 mat in pair_wise_cos_mat.items() if 'bn' in layer}
        elif bn_only == 0:
            pass
        elif bn_only == 2:
            pair_wise_cos_mat = {
                layer: mat for layer, mat in pair_wise_cos_mat.items() if 'bn' not in layer}
        if plot:
            if sorted_mat:
                pair_wise_cos_mat_plot = sort_mat(pair_wise_cos_mat)
            else:
                pair_wise_cos_mat_plot = pair_wise_cos_mat

            assert iter_ != -1, iter_
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            for layer, mat in pair_wise_cos_mat_plot.items():
                plt.figure(figsize=(12*len(model_state_dict_list) /
                           20, 8*len(model_state_dict_list)/20))
                sns.heatmap(mat, annot=True, fmt=".2f")
                plt.savefig(fig_path+f'_{layer}.png')
                plt.close('all')
                plt.clf()
                plt.cla()

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

        return pair_wise_cos_mat
    
    def defenseGSSCompare(self, model, model_state_dict_list, trigger, pos_choice, img_channel=3, img_size=32, num_class=-1, attacking_label=-1, layer_names=[], p=0.9, perform_normalize=False, iter_=-1, sorted_mat=False, plot=True, device:str|torch.device='cuda'):
        """
        Compare models using GSS (Gradient Space Similarity) defense mechanism
        
        Args:
           model_state_dict_list: like [(w_local_i[0], w_local_i[1]) for w_local_i in w_local_list]
        
        Returns:
            distance_matrix: NxN matrix where entry (i,j) represents distance between model i and j
        """
        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(sorted(model_state_dict_list))

        dataset_clean = FilteredDataset(ShrinkedDataset(self.dataset_test, frac=0.1), [attacking_label])
        dataset_void = VoidDataset(dataset_clean)
        dataset_corrput = CorruptedDataset(dataset_void, [
                                   attacking_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size)
        
        test_dataloader_clean = DataLoader(dataset_clean, batch_size=256, shuffle=False, num_workers=4)
        test_dataloader_corrput = DataLoader(dataset_corrput, batch_size=256, shuffle=False, num_workers=4)

        clean_samples = (next(iter(test_dataloader_clean))[0], next(iter(test_dataloader_clean))[1])
        corrupt_samples = (next(iter(test_dataloader_corrput))[0], next(iter(test_dataloader_corrput))[1])
        
        # print(f"Loaded {len(model_state_dict_list)} models for comparison")
        model = copy.deepcopy(model)
        model.to(device)
        
        correlation_results = calculate_model_correlations_from_neuron_analysis(
            model=model,
            model_state_dicts=model_state_dicts,
            clean_samples=clean_samples,
            corrupt_samples=corrupt_samples,
            # layer_names=['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10'],  # Analyze all common layers
            layer_names=layer_names,
            correlation_metrics=['importance_diff', 'activation_diff'],
            aggregation_method='mean',
            p=p,
            device=device
        )
        
        del model
        
        # top_k_result = find_top_k_outliers(correlation_results)
        # outlier_result = analyze_model_outliers(correlation_results)

        top_k_result = find_top_k_clustered_outliers(correlation_results, k=1, methods=['density_separation'])
        # print(f"\nFound {len(top_k_result['top_k_clustered_groups'])} top clustered groups:")
        # print("="*60)
        
        # for i, (group_name, analysis) in enumerate(top_k_result['detailed_analysis'].items()):
        #     print(f"\n{group_name.upper()}:")
        #     print(f"  Models: {analysis['models']}")
        #     print(f"  Size: {analysis['size']}")
        #     print(f"  Separation Score: {analysis['separation_score']:.4f}")
        #     print(f"  Avg Internal Similarity: {analysis['avg_internal_similarity']:.4f}")
        #     print(f"  Avg External Similarity: {analysis['avg_external_similarity']:.4f}")
        #     print(f"  Found by methods: {analysis['found_by_methods']}")
    
        
        if plot:
            fig_path = os.path.join(self.visual_path, 'tss_cosine'+str(iter_))
            visualize_model_correlations(correlation_results, save_path=fig_path+f'.png')


        return [int(x) for x in list(top_k_result['detailed_analysis'].values())[0]['models']]

    def TSSCurvePlot(self, model, model_state_dict_list, meta_mask=None, threshold=0.3, threshold_common=1, iter_=-1, frac=0.5, synthesize=0, perform_normalize=False, attackers=()):
        '''
        plot:
        attackers' mean
        non-attackers' mean
        attackers - non-attackers diff
        attackers' diff
        non-attackers' diff
        '''

        if perform_normalize:
            model_state_dicts = normalize_state_dicts_layerwise(
                model_state_dict_list)
        else:
            model_state_dicts = dict(model_state_dict_list)

        num_classes = max(self.dataset_test.targets) + 1

        if synthesize:
            if synthesize == 1:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test), [label]) for label in range(num_classes)]
            elif synthesize == 2:
                dataset_test_list = [FilteredDataset(
                    SynthesizeDataset(self.dataset_test, use_interclass_distribution=True), [label]) for label in range(num_classes)]
            else:
                raise ValueError('synthesize should be 0, 1 or 2')
        else:
            dataset_test_list = [FilteredDataset(
                ShrinkedDataset(self.dataset_test, frac=frac), [label]) for label in range(num_classes)]

        model_idx_list = model_state_dicts.keys()
        tss_impt_dict_all = {
            idx: getTssCommon(getTssImpt(
                model, model_dict, dataset_test_list, p=threshold, device=self.args.device), model, model_dict, threshold_common)
            for idx, model_dict in model_state_dicts.items()
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
                self.args.device) for layer in layer_names if layer+'.layer.weight' in meta_mask}
            inter_model_tss = meta_mask
        else:
            inter_model_tss = {
                layer: torch.zeros_like(inter_model_tss[layer])
                for layer in layer_names
            }

        fig_path = os.path.join(self.visual_path, 'tss_curve')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig_path = os.path.join(
            self.visual_path, 'tss_curve', f'round{str(iter_)}_')
        # for i in model_idx_list:
        #     plot_tss_value(model_state_dicts[i], tss_impt_dict_all[i], inter_model_tss, i, fig_path)

        # model_state_dict attackers mean
        attackers_dict_mean = {
            layer: torch.stack([model_state_dicts[i][layer] for i in attackers]).mean(0) for layer in model.state_dict().keys() if layer in model_state_dicts[0].keys()}
        non_attackers_dict_mean = {
            layer: torch.stack([model_state_dicts[i][layer] for i in model_idx_list if i not in attackers]).mean(0) for layer in model.state_dict().keys() if layer in model_state_dicts[0].keys()}
        attackers_impt_mean = {
            layer: (torch.stack([tss_impt_dict_all[i][layer] for i in attackers]).sum(0) > 0) for layer in tss_impt_dict_all[0].keys()}
        non_attackers_impt_mean = {
            layer: (torch.stack([tss_impt_dict_all[i][layer] for i in model_idx_list if i not in attackers]).sum(0) > 0) for layer in tss_impt_dict_all[0].keys()}

        diff_dict = {layer: attackers_dict_mean[layer] - non_attackers_dict_mean[layer]
                     for layer in attackers_dict_mean.keys()}
        diff_impt = {layer: attackers_impt_mean[layer] + non_attackers_impt_mean[layer]
                     for layer in attackers_impt_mean.keys()}

        plot_tss_value(attackers_dict_mean, attackers_impt_mean,
                       inter_model_tss, 'attackers_mean', fig_path)
        plot_tss_value(non_attackers_dict_mean, non_attackers_impt_mean,
                       inter_model_tss, 'non_attackers_mean', fig_path)
        plot_tss_value(diff_dict, diff_impt, inter_model_tss, 'diff', fig_path)

        print('cos diff', pairWiseCosineSim(inter_model_tss, layer='fc3', tss_impt_dict_1=attackers_impt_mean,
              model_dict_1=attackers_dict_mean, tss_impt_dict_2=non_attackers_impt_mean, model_dict_2=non_attackers_dict_mean))
        norm_tss_value(diff_dict, diff_impt, inter_model_tss)

        del dataset_test_list
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def defense_TSSGetTopK(self, pair_wise_cos_mat, top_k=2, layer_names: tuple = ('conv2')):
        """
        Selects the top K identities that have the most large average pairwise distance.

        Args:

        Returns:
            list: A list of top K identity indices sorted by largest average distance.
        """
        # Get the list of layers
        layers = list(pair_wise_cos_mat.keys())

        # Compute the average pairwise distance for each identity
        num_identities = pair_wise_cos_mat[layers[0]].shape[0]
        avg_distances = np.zeros(num_identities)

        for layer in layers:
            if (layer in layer_names) or (len(layer_names) == 0):
                # Sum over layers
                avg_distances += pair_wise_cos_mat[layer].mean(axis=1)

        avg_distances /= len(layers)  # Normalize by the number of layers

        # Select top K indices with the highest average distances
        # Sort in descending order
        top_k_indices = np.argsort(avg_distances)[-top_k:][::-1]

        return top_k_indices.tolist()

    @torch.no_grad()
    def defense_TSSGetTopK_layerwise(self, pair_wise_cos_mat, top_k=2, layer_names: tuple = ('conv2')):
        """
        Selects the top K identities that have the most large average pairwise distance.
        Each layer gives top k.
        Count max occurrence.

        Args:

        Returns:
            list: A list of top K identity indices sorted by largest average distance.
        """
        # Get the list of layers
        layers = list(pair_wise_cos_mat.keys())

        # Compute the average pairwise distance for each identity
        num_identities = pair_wise_cos_mat[layers[0]].shape[0]

        total_occurrence = {}

        for layer in layers:
            if (layer in layer_names) or (len(layer_names) == 0):
                # Sum over layers
                layer_distance = pair_wise_cos_mat[layer].mean(axis=1)
                # total_occurrence[np.argsort(layer_distance)[-top_k:][::-1]] += 1
                selected = str(sorted(np.argsort(layer_distance)
                               [-top_k:][::-1].tolist()))
                if selected in total_occurrence:
                    total_occurrence[selected] += 1
                else:
                    total_occurrence[selected] = 1
                print(selected)

        print(total_occurrence)
        # return np.argsort(total_occurrence)[-top_k:][::-1]
        selected_str = sorted(total_occurrence.items(),
                              key=lambda x: x[1], reverse=True)[0]
        # selected = [int(x[0][1:-1].split(',')[0]) for x in selected_str]
        selected = [int(x) for x in selected_str[0][1:-1].split(',')]
        return selected

    @torch.no_grad()
    def defense_TSS_kmeans(self, pair_wise_cos_mat, nclasses=2, layer_names: tuple = ('conv2')):
        """
        Selects the top K identities that have the most large average pairwise distance.

        Args:

        Returns:
            list: A list of top K identity indices sorted by largest average distance.
        """
        # Get the list of layers
        layers = list(pair_wise_cos_mat.keys())

        # Compute the average pairwise distance for each identity
        num_identities = pair_wise_cos_mat[layers[0]].shape[0]
        # avg_distances = np.zeros(num_identities)
        client_features = np.zeros((num_identities, len(layers)))

        for idx, layer in enumerate(layers):
            if (layer in layer_names) or (len(layer_names) == 0):
                # set the client features of position idx with the value in the layer of corresponding pairwise cosine matrix
                client_features[:, idx] = pair_wise_cos_mat[layer].mean(axis=1)

        # Select top K indices with the highest average distances
        # Sort in descending order
        kmeans = KMeans(n_clusters=nclasses, random_state=0)
        # Fit the KMeans model to the features
        kmeans.fit(client_features)
        # Get the cluster labels
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        if counts[1] > counts[0]:
            # If label 0 is majority, flip labels
            labels = 1 - labels
        print(f"Labels: {labels}")

        # Return the indices of the minority class
        return np.where(labels == 0)[0].tolist()

    def defense_TSSLayerMinMax(self, pair_wise_cos_mat, eps=1e-5, k=0.5):
        """
        Args:

        Returns:
            dict: Layer-wise min-max weight
        """
        # TODO 2025-03-13 git.V.96ca0: when frac != 1
        # Get the list of layers
        layers = list(pair_wise_cos_mat.keys())

        # client_vector = {i:[] for i in range(cos_sim_layer.shape[0])}
        client_vector = {layer: [] for layer in layers}

        for idx, layer in enumerate(layers):
            cos_sim_layer = pair_wise_cos_mat[layer]
            for i in range(cos_sim_layer.shape[0]):
                client_vector[layer].append(np.sum(cos_sim_layer[i, :]))

            client_vector[layer] = np.nan_to_num(
                client_vector[layer], nan=math.pi)
            # print(client_vector[layer])
            min_val = min(client_vector[layer])
            max_val = max(client_vector[layer])
            avg_val = np.mean(client_vector[layer])
            # client_vector[layer] = [1 - (val - min_val) / (max_val - min_val + eps) for val in client_vector[layer]]
            # client_vector[layer] = [1 / (1 + math.exp(k * (val - min_val) / (max_val - min_val + eps))) - 0.05 for val in client_vector[layer]]
            client_vector[layer] = [
                (1 - (val - min_val) / ((max_val - min_val) or 1)) ** (1/k) for val in client_vector[layer]]
            # client_vector[layer] = [2 * (1- (val / max_val)) ** 2 - 1 for val in client_vector[layer]]
            # client_vector[layer] = [1 + 2 * (min_val - val) / (max_val - min_val + eps) for val in client_vector[layer]]
            # client_vector[layer] = [1 / (1 + math.exp(k * (val - avg_val))) - 0.4 for val in client_vector[layer]]

        return client_vector

    def analyseImportantLayers(self, last_global_dict, model_state_dicts):
        # return computeLayerImportance(last_global_dict, model_state_dicts)
        return computeLayerImportance_smooth(last_global_dict, model_state_dicts)

    def analyseImportantNeurons(self, last_global_dict, model_state_dicts, percentile=80):
        importance = computeNeuronImportance(
            last_global_dict, model_state_dicts)
        return importance_to_mask(importance, percentile=percentile)

    def defense_tracer(self, model_state_dict_list, detecting_method=None):
        user_ids = [x[0] for x in model_state_dict_list]
        local_updated_weights = {}
        for user_id, model_state_dict in model_state_dict_list:
            local_updated_weights[user_id] = model_state_dict
        client_weights_flattern = processing_updates(
            local_updated_weights, user_ids)
        return detecting_expert_1_2(client_weights_flattern, user_ids, threshold=0.05, detecting_method=detecting_method), detecting_expert_1_2(client_weights_flattern, user_ids, threshold=0.1, detecting_method='sign'), detecting_expert_3(local_updated_weights, user_ids)

    def defense_tracer_topK(self, array, k=2):
        counter = Counter(array.tolist())
        most_common = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        result = [item[0] for item in most_common[:k]]

        return result

    def distanceToFeatures_layerwise(self, distance_matrices, layer_importance_score):
        features = []

        for layer, D in distance_matrices.items():
            values = D[np.triu_indices_from(D, k=1)]
            mean = np.mean(values)
            std = np.std(values)
            skewness = skew(values)
            kurt = kurtosis(values)
            value_range = np.max(values) - np.min(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            layer_ip = layer_importance_score[layer]
            if std == 0:
                extremeness_score = 0
            else:
                max_deviation = np.abs(values.max() - mean) / std
                min_deviation = np.abs(values.min() - mean) / std
                extremeness_score = max(max_deviation, min_deviation)

            features.append([mean, std, skewness, kurt,
                            value_range, iqr, extremeness_score, layer_ip])

        features = np.array(features)
        return features

    def distanceToFeatures_layerwise_smooth(self, distance_matrices, layer_importance_score):
        '''
        AI generated
        '''
        features = []
        for layer, D in distance_matrices.items():
            values = D[np.triu_indices_from(D, k=1)]
            # Sort the values
            sorted_values = np.sort(values)

            # Basic statistics (keeping your original features)
            mean = np.mean(values)
            std = np.std(values)
            skewness = skew(values)
            kurt = kurtosis(values)
            value_range = np.max(values) - np.min(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            layer_ip = layer_importance_score[layer]

            # Extreme value detection (improved)
            if std == 0:
                extremeness_score = 0
            else:
                max_deviation = np.abs(values.max() - mean) / std
                min_deviation = np.abs(values.min() - mean) / std
                extremeness_score = max(max_deviation, min_deviation)

            # New features for your specific requirement
            # 1. Calculate middle region smoothness
            middle_start_idx = int(len(sorted_values) * 0.25)
            middle_end_idx = int(len(sorted_values) * 0.75)
            middle_values = sorted_values[middle_start_idx:middle_end_idx]
            # Higher value means smoother
            middle_smoothness = 1.0 / (np.std(middle_values) + 1e-10)

            # 2. Calculate tail extremeness
            left_tail = sorted_values[:middle_start_idx]
            right_tail = sorted_values[middle_end_idx:]
            left_tail_gap = mean - \
                np.mean(left_tail) if len(left_tail) > 0 else 0
            right_tail_gap = np.mean(right_tail) - \
                mean if len(right_tail) > 0 else 0

            # 3. Combined metric: high when middle is smooth AND tails are extreme
            tail_extremeness = (
                left_tail_gap + right_tail_gap) / (2 * std + 1e-10)
            smooth_center_extreme_tails = middle_smoothness * tail_extremeness

            features.append([
                mean, std, skewness, kurt, value_range, iqr, extremeness_score, layer_ip,
                middle_smoothness, left_tail_gap, right_tail_gap, smooth_center_extreme_tails
            ])

        features = np.array(features)
        return features

    def distance_to_features_clientwise(self, distance_matrices):
        """
        Represent a client with a vector from the distance matrix.

        Args:
            distance_matrices (dict): Dictionary of distance matrices, where the key is the layer name, 
                                      and the value is the distance matrix.

        Returns:
            np.ndarray: List of features for each client, where each feature is a list of 8 values.
        """
        features = []
        for client_idx in range(distance_matrices[next(iter(distance_matrices))].shape[0]):
            client_features = []
            for layer, D in distance_matrices.items():
                values = D[client_idx, :]
                mean = np.mean(values)
                std = np.std(values)
                skewness = skew(values)
                kurt = kurtosis(values)
                value_range = np.max(values) - np.min(values)
                iqr = np.percentile(values, 75) - np.percentile(values, 25)
                if std == 0:
                    extremeness_score = 0
                else:
                    max_deviation = np.abs(values.max() - mean) / std
                    min_deviation = np.abs(values.min() - mean) / std
                    extremeness_score = max(max_deviation, min_deviation)

                client_features.extend(
                    [mean, std, skewness, kurt, value_range, iqr, extremeness_score])

            features.append(client_features)

        features = np.array(features)
        return features

    def extract_top_k_matrices(self, distance_matrices, layer_importance_score, k=5):
        # TODO 2025-05-15 git.V.6beb3: combine statistical with MAML feature
        features = self.distanceToFeatures_layerwise_smooth(
            distance_matrices, layer_importance_score)
        center = np.mean(features, axis=0)

        distances = np.linalg.norm(features - center, axis=1)
        noise = math.ceil(len(distances) * 0.2)
        top_k_indices = np.argsort(-distances)[noise:noise+k]
        # top_k_indices = np.argsort(-distances)[-k:]

        selected, scored = select_top_matrices(distance_matrices, top_k=k, k=2)
        # print(f"Selected matrices: {selected}")
        # print(f"Scores: {scored}")

        # return top_k_indices.reshape(-1)
        return np.array(selected).reshape(-1)

    def clusterByDistance(self, distance_matrices, k=2):
        """
        Clustering the features by distance
        """
        features = self.distance_to_features_clientwise(distance_matrices)
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(features)

        unique, counts = np.unique(labels, return_counts=True)
        if counts[1] > counts[0]:
            # If label 0 is majority, flip labels
            labels = 1 - labels

        # Calculate silhouette score
        silhouette_avg = silhouette_score(features, labels)

        return labels, silhouette_avg

    def banClient(self, all_clients, limit=5):
        """
        """
        total_counts = np.zeros(len(all_clients))
        for L in Defender.historical_excluded_list[-10:]:
            for clientid, stat in enumerate(L):
                total_counts[clientid] += stat

        total_counts -= np.min(total_counts)
        # return total_counts >= limit
        banned_indices = np.where(total_counts >= limit)[0]
        return banned_indices.tolist()

    def banClientProbe(self, all_clients, limit=5):
        """
        """
        total_counts = np.zeros(len(all_clients))
        for L in Defender.historical_excluded_list[-10:]:
            for clientid, stat in enumerate(L):
                total_counts[clientid] += stat

        probability = total_counts / total_counts.sum()
        return probability
        # banned_indices = np.where(total_counts > 1/len(all_clients)*limit)[0]
        # return banned_indices.tolist()

    def defense_rlr(self, net_glob: nn.Module, w_local_list, w_glob, lr=1, threshold=6, device: str | torch.device = 'cuda'):
        # agent_updates_list = []
        # model = copy.deepcopy(net_glob)
        # g_param_v = nn.utils.parameters_to_vector(net_glob.parameters())
        # for i, w_local in enumerate(w_local_list):
        #     model.load_state_dict(w_local)
        #     param = model.parameters()
        #     agent_updates_list.append(nn.utils.parameters_to_vector(param) - g_param_v)

        # agent_updates_sign = [torch.sign(update) for update in agent_updates_list]
        # sm_of_signs = torch.abs(sum(agent_updates_sign))
        # sm_of_signs[sm_of_signs < threshold] = -lr
        # sm_of_signs[sm_of_signs >= threshold] = lr

        # model.load_state_dict(w_glob)
        # new_global_params_v =  (torch.nn.utils.parameters_to_vector(net_glob.parameters()) + sm_of_signs.to(device)*(torch.nn.utils.parameters_to_vector(model.parameters()))).float()
        # torch.nn.utils.vector_to_parameters(new_global_params_v, model.parameters())
        # return model.state_dict()

        agent_updates_list = []
        model = copy.deepcopy(net_glob)
        for i, w_local in enumerate(w_local_list):
            model.load_state_dict(w_local)
            param = model.state_dict()
            agent_updates_list.append(parameters_dict_to_vector_rlr(
                param) - parameters_dict_to_vector_rlr(w_glob))

        grad_list = []
        for i in agent_updates_list:
            grad_list.append(i)

        aggregated_updates = 0.0
        for update in grad_list:
            # print(update.shape)  # torch.Size([1199882])
            aggregated_updates += update
        aggregated_updates /= len(grad_list)

        lr_vector = compute_robustLR(grad_list, lr, threshold, device)
        cur_global_params = parameters_dict_to_vector_rlr(w_glob)
        new_global_params = (cur_global_params +
                             lr_vector*aggregated_updates).float()
        global_w = vector_to_parameters_dict(new_global_params, w_glob)
        # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))

        del model, agent_updates_list, grad_list
        torch.cuda.empty_cache()
        return global_w

    def getWglobKrum(self, w_glob_list: list, krumClients=2, mclients=2):
        kd = KrumDefense(mclients, krumClients)
        clients = []

        # Prepare clients list for Krum defense
        for idx, w_local, idxs_weight in w_glob_list:
            clients.append(tuple([idxs_weight, w_local]))

        # Apply Krum defense
        clients_flatten_weight = kd.defend_before_aggregation(clients)

        # Initialize aggregated weights with zeros (same structure as first client's weights)
        w = copy.deepcopy(clients_flatten_weight[0][1])
        for k in w.keys():
            w[k] = torch.zeros_like(w[k])

        # Initialize total weight
        total_weight = 0

        # Aggregate weights from all selected clients
        for idxs_weight, w_local in clients_flatten_weight:
            # Add weighted contribution from this client
            total_weight += idxs_weight

            for k in w.keys():
                w[k] += w_local[k] * idxs_weight

        # Normalize by total weight
        if total_weight > 0:  # Avoid division by zero
            for k in w.keys():
                w[k] = torch.div(w[k], total_weight)

        return w

    def getWglobFlame(self, w_local_list, w_glob):
        """
        Get the global model weights using FLAME defense.

        Args:
            w_local_list (list): List of tuples containing user ID and local model weights.
            w_glob (dict): Global model weights.

        Returns:
            dict: Aggregated global model weights after applying FLAME defense.
        """
        flame_glob, benign_clients = flame(w_local_list, w_glob)
        return flame_glob, benign_clients

    def getWglobKrum_multi(self, w_local_list, w_glob, num_users=2, participation=1.0, n_attackers=2):
        krum_distance = []

        gradients = [get_update(w_local, w_glob) for w_local in w_local_list]

        grads = flatten_grads_gpu(gradients)

        candidates = []
        candidate_indices = []
        remaining_updates = grads
        all_indices = np.arange(len(grads))

        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            scores = None
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(
                    distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(
                distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            # print(scores)
            krum_distance.append(scores)
            indices = torch.argsort(scores)[:len(
                remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(
                candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat(
                (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

        # aggregate = torch.mean(candidates, dim=0)

        # return aggregate, np.array(candidate_indices)
        num_clients = max(int(participation * num_users), 1)
        num_malicious_clients = n_attackers
        num_benign_clients = num_clients - num_malicious_clients

        return np.array(candidate_indices).tolist()

    def clipping(self, w_local: dict, net_global: torch.nn.Module, threshold=0.1):
        d_w = copy.deepcopy(w_local)
        for k in w_local.keys():
            d_w[k] = w_local[k] - net_global.state_dict()[k]
        d_n = copy.deepcopy(w_local)
        for k in w_local.keys():
            d_n[k] = torch.nn.functional.normalize(
                d_w[k].float(), dim=0)
        for k in w_local.keys():
            w_local[k] = w_local[k] - (threshold*torch.nn.functional.normalize(
                d_n[k].float(), dim=0)).long()
        return w_local

    def norm_clipping(self, w_local: dict, net_global: torch.nn.Module, max_norm: float = 1.0):
        """
        Perform norm clipping on the weight updates.

        Args:
            w_local: Local model weights
            net_global: Global model
            max_norm: Maximum allowed norm for the weight updates

        Returns:
            Clipped local weights
        """
        # Calculate the weight updates (difference between local and global)
        d_w = {}
        for k in w_local.keys():
            d_w[k] = w_local[k] - net_global.state_dict()[k]

        # Calculate the total norm of all weight updates
        total_norm = 0.0
        for k in d_w.keys():
            total_norm += torch.norm(d_w[k].float(), p=2).item() ** 2
        total_norm = total_norm ** 0.5

        # Calculate the clipping factor
        clip_factor = min(1.0, max_norm / (total_norm + 1e-10))

        # Apply clipping to the weight updates
        w_clipped = copy.deepcopy(w_local)
        for k in w_local.keys():
            # Clip the update and apply it back to get the clipped weights
            clipped_update = d_w[k] * clip_factor
            w_clipped[k] = net_global.state_dict()[k] + clipped_update

        return w_clipped

    def norm_clipping_per_layer(self, w_local: dict, net_global: torch.nn.Module, max_norm: float = 1.0):
        """
        Perform norm clipping on the weight updates (per layer).

        Args:
            w_local: Local model weights
            net_global: Global model
            max_norm: Maximum allowed norm for the weight updates per layer

        Returns:
            Clipped local weights
        """
        w_clipped = copy.deepcopy(w_local)

        for k in w_local.keys():
            # Calculate the weight update for this layer
            d_w = w_local[k] - net_global.state_dict()[k]

            # Calculate the norm of this layer's update
            layer_norm = torch.norm(d_w.float(), p=2).item()

            # Calculate the clipping factor for this layer
            clip_factor = min(1.0, max_norm / (layer_norm + 1e-10))

            # Apply clipping to this layer's weights
            w_clipped[k] = net_global.state_dict()[k] + d_w * clip_factor

        return w_clipped


def flatten_grads_gpu(gradients):
    param_order = gradients[0].keys()
    flat_epochs = []

    for _, grads in enumerate(gradients):
        user_tensors = []
        for param in param_order:
            # Keep on GPU and flatten
            user_tensors.append(grads[param].flatten())
        # Concatenate all parameters into a single tensor on GPU
        flat_epochs.append(torch.cat(user_tensors))

    # Stack all epochs into a single tensor on GPU
    flat_epochs = torch.stack(flat_epochs)
    return flat_epochs


def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for _, grads in enumerate(gradients):
        user_arr = []
        for param in param_order:
            user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs


def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> dict:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict


def compute_robustLR(params, lr, threshold, device='cuda'):
    agent_updates_sign = [torch.sign(update) for update in params]
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    # print(f"sm_of_signs: {sm_of_signs.shape}, threshold: {threshold}, lr: {lr}")
    # print(f"mean sm_of_signs: {sm_of_signs.mean()}, max: {sm_of_signs.max()}, min: {sm_of_signs.min()}")
    sm_of_signs[sm_of_signs < threshold] = -lr
    sm_of_signs[sm_of_signs >= threshold] = lr
    return sm_of_signs.to(device)


def get_update(model_dict_updated: dict, model_dict_t0: dict, filte_keys=True):
    '''get the update weight'''
    update = {}
    for key, var in model_dict_updated.items():
        if filte_keys and (not 'alphas' in key and not 'scores' in key and not 'impt_mask' in key):
            update[key] = model_dict_updated[key] - model_dict_t0[key]
        elif not filte_keys:
            update[key] = model_dict_updated[key] - model_dict_t0[key]
    return update


def multi_krum(gradients, n_attackers, args, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn += 1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1

    print(candidate_indices)

    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]

    return np.array(candidate_indices)


def flame(local_model_list, global_model, noise=1e-3):
    update_params = [get_update(w_local, global_model, filte_keys=False)
                     for w_local in local_model_list]

    def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
        vec = []
        for key, param in net_dict.items():
            # print(key, torch.max(param))
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    def parameters_dict_to_vector(net_dict) -> torch.Tensor:
        r"""Convert parameters to one vector

        Args:
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.

        Returns:
            The parameters represented by a single vector
        """
        vec = []
        for key, param in net_dict.items():
            if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    def no_defence_balance(params, global_parameters):
        total_num = len(params)
        sum_parameters = None
        for i in range(total_num):
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in params[i].items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + params[i][var]
        for key, var in global_parameters.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                global_parameters[key] = params[0][key]
                continue
            global_parameters[key] += (sum_parameters[key] / total_num)

        return global_parameters

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in local_model_list:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    # num_clients = max(int(frac * num_users), 1)
    # num_malicious_clients = int(args.malicious * num_clients)
    # num_benign_clients = num_clients - num_malicious_clients
    num_clients = len(local_model_list)
    clusterer = HDBSCAN(min_cluster_size=num_clients//2 + 1,
                        min_samples=1, allow_single_cluster=True).fit(cos_list)
    # print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model_list)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(
                parameters_dict_to_vector(update_params[i]), p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(
                    clusterer.labels_[clusterer.labels_ == index_cluster])
        for idx, lb in enumerate(clusterer.labels_):
            if lb == max_cluster_index:
                benign_client.append(idx)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list, torch.norm(
            parameters_dict_to_vector(update_params[i]), p=2).item())  # no consider BN

    # print(benign_client)
    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         args.wrong_mal+=1
    #     else:
    #         #  minus per benign in cluster
    #         args.right_ben += 1
    # args.turn+=1
    # print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance(
        [update_params[i] for i in benign_client], global_model)
    # add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0, std=noise*clip_value)
        var += temp
    return global_model, benign_client


def fltrust(params, central_param, global_parameters, args):
    def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
        vec = []
        for key, param in net_dict.items():
            # print(key, torch.max(param))
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                # 乘得分 再乘裁剪值
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        # 除以所以客户端的信任得分总和
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters
