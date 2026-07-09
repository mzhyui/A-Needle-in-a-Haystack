#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import yaml
import os

def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="debug mode")

    
    # train arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[], help="data parallel gpu list")
    parser.add_argument('--parallel', type=int, default=1, help="1 for parallel")
    parser.add_argument('--max_workers', type=int, default=0, help="max_workers for dataloader")
    # parser.add_argument('--dl_workers', type=int, default=4, help="max_workers for dataloader")
    parser.add_argument('--threads', type=int, default=9, help="threads for parallel training ")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=5, help='how often to test on val set')


    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--input_size', type=int, default=32, help="input size of images")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--dataset_path', type=str, default='data/', help='dataset loading path')
    parser.add_argument('--data_augmentation', type=int, default=1, help="use data augmentation")
    parser.add_argument('--data_augmentation_local', type=int, default=1, help="user's local data augmentation")
    parser.add_argument('--data_user_adjust', type=int, default=0, help="adjust diri alpha according to user number")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_metric', type=str, default='shard', help='non-iid metric: iid, (fair_iid), shard, unbalanced, dirichlet')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='dirichlet alpha')
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=5, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.8, help="the fraction of clients per round: C")
    parser.add_argument('--dynamic_frac', type=float, nargs='+', default=[], help="the fraction of clients changes at round, eg: [50, 0.5, 100, 0.1]")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=0.99, help="learning rate decay per round, 'lr *= args.lr_decay'")
    parser.add_argument('--fedsam', action='store_true', help="use FedSAM")


    # loading
    parser.add_argument('--base_dir', type=str, default='', help="continue from path, only with load_fed, overwrite orignal")
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--load_user_dict', type=str, default='', help='load saved user dict')
    parser.add_argument('--load_begin_epoch', type=int, default=1, help='define epochs finished of the loaded fed model, must < epochs')

    # attack
    # TODO 2025-04-24 git.V.ea4c0: no_attack_on_attack cause averaged weight computing error
    # under development
    parser.add_argument('--attackers', type=int, default=[], nargs='+', help="dedicate attacker number")
    parser.add_argument('--no_attack_on_attack', action='store_true', help="do not upload attack weights by attackers")
    parser.add_argument('--portion', type=float, default=0.3, help="the fraction of attackers")
    parser.add_argument('--data_portion', type=float, default=0.7, help="the fraction of poison in a batch of data")
    parser.add_argument('--start_attack', type=int, default=1, help="attack beginning epoch")
    parser.add_argument('--label', type=int, default=8, help="attack label")
    parser.add_argument('--pattern_choice', type=int, default=1, help="choose a pattern")
    parser.add_argument('--pos_choice', type=int, default=[1,1], help="choose a position")
    parser.add_argument('--local_ep_times', type=int, default=3, help="multiply local ep")
    parser.add_argument('--scale', type=int, default=0, help="do scale attack")
    parser.add_argument('--attack_on_attack', type=int, default=[], nargs='+', help="all users are attackers, while selected ones performs attack")
    # parser.add_argument('--groupattack', action='store_true', help="multiply local ep")
    # parser.add_argument('--dba', action='store_true', help="dba attack")
    parser.add_argument('--attack_type', type=str, default="peace", help="the attack strategy: static, dynamic, dba, groupattack, peace(empty pattern)")
    parser.add_argument('--blend_alpha', type=float, default=0.3, help="noise pattern transparency")
    parser.add_argument('--mirage_disc_train_ep', type=int, default=2, help="mirage discriminator training ep per round")

    # unbalanced
    parser.add_argument('--ub_label', type=int, default=-1, help="unbalanced at target label, -1 not to ub")
    parser.add_argument('--ub_users_percent', type=float, default=0.3, help="unbalanced user percentage")

    # defense
    parser.add_argument('--enable_rb', type=int, default=0, help='whether detect attack or not')
    parser.add_argument('--rb_rate', type=float, default=0, help="the penalty possibility")
    parser.add_argument('--rb_rootpth', type=str, default="rb_root", help="the rb weight path")
    parser.add_argument('--rb_wait', type=int, default=0, help='wait input')
    parser.add_argument('--rb_range', type=int, nargs='+', default=[0,20], help="robust range like [0,20]")
    parser.add_argument('--penalty', type=float, default=0.3, help="the penalty rate, 'w *= p'")
    # MAE
    parser.add_argument('--mae', type=int, default=999, help="do mae defense every n rounds (0 for no defense)")
    parser.add_argument('--script_path', type=str, default='/home/mzhyui/git/Signal_and_nn/', help="the script path")
    parser.add_argument('--mae_train_epoch', type=int, default=20)
    parser.add_argument('--mae_train_ratio', type=float, default=1)
    parser.add_argument('--mae_collect_rounds', type=int, default=10)
    parser.add_argument('--mae_recover_range', type=int, nargs='+', default=[40,70])
    # TSS
    parser.add_argument('--tss_eval_frac', type=float, default=0.3, help="")
    parser.add_argument('--tss_with_noise', type=int, default=0, help="use noise dataset for tss eval")
    parser.add_argument('--tss_synthesize', type=int, default=0, help="synthesize dataset; 1 = use normalized gaussian noise; 2 = use original sampled noise")
    parser.add_argument('--tss_statistical_ban', type=int, default=999, help="do tss statistical reject")
    parser.add_argument('--tss_statistical_reject', type=int, default=999, help="do tss reject based on historical behavior")
    parser.add_argument('--tss_statistical_reject_trigger', type=int, default=999, help="do tss reject based on historical behavior (with trigger knowledge)")
    parser.add_argument('--tss_statistical_reject_noise', type=int, default=999, help="do tss reject based on historical behavior (with all noise input)")
    parser.add_argument('--tss_statistical_reject_kmeans', type=int, default=999, help="do tss reject based on kmeans clustering")
    parser.add_argument('--tss_statistical_layerwise', type=int, default=999, help="do tss reject based on historical behavior")
    parser.add_argument('--tss_statistical_penalty', type=int, default=5, help="round accumulated penalty for tss statistical reject")
    parser.add_argument('--tss_statistical_lookback', type=int, default=10, help="round lookbacks for tss statistical reject")
    parser.add_argument('--tss_hard_reject', type=int, default=999, help="do tss selection")
    parser.add_argument('--tss_hard_k', type=int, default=2, help="tss exclude clients")
    parser.add_argument('--tss_soft_reject', type=int, default=999, help="do tss aggregation")
    parser.add_argument('--tss_layer_list', type=str, nargs='+', default=[], help="tss cosine calculate layers. eg ['conv2']")
    parser.add_argument('--tss_bn_only', type=int, default=2, help="analyse bn layers only")
    parser.add_argument('--tss_layer_count', type=int, default=3, help="tss cosine calculate layers")
    parser.add_argument('--tss_threshold', type=float, default=0.3, help="")
    parser.add_argument('--tss_threshold_common', type=int, default=1, help="")
    parser.add_argument('--tss_norm', type=int, default=0, help="perform normalization on model dict")
    parser.add_argument('--tss_metamask', type=int, default=0, help="use meta mask")
    parser.add_argument('--tss_important_percentile', type=int, default=20, help="meta mask percentile")
    parser.add_argument('--tss_sort', type=int, default=0, help="plot tss heatmap sorted by col mean")
    parser.add_argument('--tss_plot', type=int, default=0, help="plot cosine similarity heatmap")
    parser.add_argument('--grad_mask_only', type=int, default=999, help="do grad compare without tss")
    parser.add_argument('--tss_mp', type=int, default=0, help="use torch.multiprocessing with tss")
    # GSS
    parser.add_argument('--gss', type=int, default=999, help="do gss reject")
    parser.add_argument('--gss_layer_list', type=str, nargs='+', default=[], help="gss correlation calculating layers. eg ['conv2']")
    parser.add_argument('--gss_threshold', type=float, default=0.5, help="")
    # FL TRACER
    # TODO 2025-04-24 git.V.ea4c0: attacker updater cause tracer true-positive rate high
    parser.add_argument('--tracer', action='store_true', help="execute trace")
    # RLR
    parser.add_argument('--rlr', action='store_true', help="robust learning rate")
    parser.add_argument('--rlr_threshold', type=float, default=4, help="robust lr threshold")
    # KRUM
    parser.add_argument('--krum', action='store_true', help="do krumming defense")
    parser.add_argument('--mkrum', action='store_true', help="do multi krumming defense")
    parser.add_argument('--krum_k', type=int, default=2, help="krum k value")
    # CLIPPING
    parser.add_argument('--clipping', action='store_true', help="do weight clipping")
    # FLAME
    parser.add_argument('--flame', action='store_true', help="FLAME defense")



    # saving
    parser.add_argument('--results_save', type=str, default='./fl_base_save', help='define fed results save folder')
    parser.add_argument('--local_saving_start', type=int, default=0, help='when to start saving local models')
    # TODO 2024-12-13 git.V.9ea50: save by using np.random.choice, add setting for saving more normal clients than attackers
    parser.add_argument('--local_saving_interval', type=int, default=1, help='save at ROUND % r')
    parser.add_argument('--normal_clients_save_interval', type=int, default=5, help="save by idx % r. -1 for (1-portion) / portion")
    parser.add_argument('--global_saving_start', type=int, default=10, help='when to start saving global models')
    parser.add_argument('--global_saving_interval', type=int, default=10, help='save at round % r')
    parser.add_argument('--no_local_save', type=int, default=0, help='do not save local models')
    parser.add_argument('--batch_gen', type=int, default=-1, help='dont merge and repeat training after epoch > batch_gen')

    # analysis
    parser.add_argument('--cl', type=int, default=0, help='perform channel lipschitz distance recording, 0 == not performing')
    parser.add_argument('--visualize', type=int, default=0, help='perform channel lipschitz distance recording, 0 == not performing')
    parser.add_argument('--subnet', type=int, default=0, help='perform subnet analysis, 0 == not performing')
    parser.add_argument('--tss_cosine', type=int, default=0, help='perform cosine similarity analysis, 0 == not performing')
    parser.add_argument('--eval_local', type=int, default=0, help='evaluate local model')
    parser.add_argument('--trigger_inversion', type=int, default=0, help='perform trigger inversion every n rounds, 0 == not performing')
    parser.add_argument('--inversion_importance', type=int, default=0, help='eval the importance score mask with inversed trigger')
    parser.add_argument('--noise_importance', type=int, default=0, help='eval the importance score mask with noise sample')
    parser.add_argument('--trigger_importance', type=int, default=0, help='eval the importance score mask with triggered sample')
    parser.add_argument('--tri_importance', type=int, default=0, help='eval the importance score mask with triple sample')
    parser.add_argument('--trigger_tss_eval', type=int, default=0, help='calculate the accuracy with trigger dataset tss')
    parser.add_argument('--trigger_tss_eval_v2', type=int, default=0, help='calculate the accuracy with trigger dataset tss, evaluate the model correlation based on tss')
    parser.add_argument('--trigger_gss_eval', type=int, default=0, help='evaluate the model correlation based on gss')
    parser.add_argument('--trigger_gss_eval_v2', type=int, default=0, help='evaluate the model correlation based on gss')
    parser.add_argument('--save_assessor', type=int, default=1, help='save the svm/rf model')
    parser.add_argument('--assessor_path', type=str, default='', help='load the saved svm/rf checkpoint')
    parser.add_argument('--assessing_layer', type=str, default='fc4', help='assess which fl model layer')

    # logs
    parser.add_argument('--comment', type=str, default="none", help="leave a comment")
    parser.add_argument('--config', type=str, default="./conf/vgg_dba_subnetmasking_nonparallel.yaml", help="load config")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_filename', type=str, default=os.path.basename(__file__)+'.log')
    parser.add_argument('--propagate', type=int, default=0, help="propagate the info to stdout")
    parser.add_argument('--verbose', type=int, default=0, help="verbose")

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
            args = parser.parse_args()
    else:
        raise Exception("No config file provided!")

    return args
