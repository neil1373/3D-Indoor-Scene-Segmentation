# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import argparse


def str2opt(arg):
    assert arg in ['SGD', 'Adagrad', 'Adam', 'RMSProp', 'Rprop', 'SGDLars']
    return arg


def str2scheduler(arg):
    assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR']
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


def str2list(l):
    return [int(i) for i in l.split(',')]


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument(
    '--model', type=str, default='ResUNet14', help='Model name')
net_arg.add_argument(
    '--conv1_kernel_size', type=int, default=5, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default='None',
                     help='Saved weights to load')
net_arg.add_argument(
    '--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')
net_arg.add_argument('--nonlinearity', default='ReLU', type=str)
net_arg.add_argument(
    '--weights_for_inner_model',
    type=str2bool,
    default=False,
    help='Weights for model inside a wrapper')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--lr', type=float, default=1e-2)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1,
                     help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='StepLR')
opt_arg.add_argument('--max_iter', type=int, default=6e4)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--category_weights', type=str, default='feature_data/scannet200_category_weights.pkl',
                     help='A dictionary containing normalized weights based on the validation point frequencies/category')
dir_arg.add_argument('--category_frequencies_path', type=str, default='feature_data/dataset_frequencies.pkl',
                     help='Loading teh weights (log histogram of cat frequencies) for all categories')
dir_arg.add_argument('--instance_sampling_weights', type=str, default='feature_data/tail_split_inst_sampling_weights.pkl',
                     help='A dictionary containing probability weights to pick a category for additional sampling')
dir_arg.add_argument('--bounding_boxes_path', type=str, default='feature_data/full_train_bbs_with_rels.pkl',
                     help='A precomputed dictionary containing bounding boxes of al instances')
dir_arg.add_argument('--num_instances_to_add', type=int, default=5,
                     help='How many new instances we want to sample to every train scene')
dir_arg.add_argument('--max_instance_placing_iterations', type=int, default=50,
                     help='If we cant find a place for the new instance we skip the placement')
dir_arg.add_argument('--sampled_features', type=str2bool, default=False,
                     help='If we want to save sample intermediate features for evaluation')

dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')
dir_arg.add_argument('--pred_dir', type=str, default='sample_submission')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str,
                      default='Scannet200Voxelization2cmDataset')
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--pre_point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--val_batch_size', type=int, default=1)
data_arg.add_argument('--test_batch_size', type=int, default=1)
data_arg.add_argument('--cache_data', type=str2bool, default=False)
data_arg.add_argument('--threads', type=int, default=1,
                      help='num threads for train/test dataloader')
data_arg.add_argument('--val_threads', type=int, default=1,
                      help='num threads for val dataloader')
data_arg.add_argument('--ignore_label', type=int, default=255)
data_arg.add_argument('--train_elastic_distortion',
                      type=str2bool, default=True)
data_arg.add_argument('--test_elastic_distortion',
                      type=str2bool, default=False)
data_arg.add_argument('--return_transformation', type=str2bool, default=False)
data_arg.add_argument('--ignore_duplicate_class', type=str2bool, default=False)
data_arg.add_argument('--partial_crop', type=float, default=0.)
data_arg.add_argument('--train_limit_numpoints', type=int, default=0)
data_arg.add_argument('--merge', type=str2bool, default=False)

# Point Cloud Dataset
data_arg.add_argument(
    '--scannet_path',
    type=str,
    default='../student_data',
    help='Scannet online voxelization dataset root dir')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int,
                       default=40, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int,
                       default=100, help='print frequency')
train_arg.add_argument('--save_freq', type=int,
                       default=1000, help='save frequency')
train_arg.add_argument('--start_save_iter', type=int,
                        default=35000, help='start saving individual ckpt')
train_arg.add_argument('--val_freq', type=int,
                       default=1000, help='validation frequency')
train_arg.add_argument('--empty_cache_freq', type=int, default=1,
                       help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str,
                       default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str,
                       default='val', help='Dataset for validation')
train_arg.add_argument(
    '--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument(
    '--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument(
    '--resume_optimizer',
    default=True,
    type=str2bool,
    help='Use checkpoint optimizer states when resume training')
train_arg.add_argument('--eval_upsample', type=str2bool, default=False)

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument(
    '--use_feat_aug', type=str2bool, default=True, help='Simple feat augmentation')
data_aug_arg.add_argument(
    '--data_aug_color_trans_ratio', type=float, default=0.10, help='Color translation range')
data_aug_arg.add_argument(
    '--data_aug_color_jitter_std', type=float, default=0.05, help='STD of color jitter')
data_aug_arg.add_argument(
    '--data_aug_height_trans_std', type=float, default=1, help='STD of height translation')
data_aug_arg.add_argument(
    '--data_aug_height_jitter_std', type=float, default=0.1, help='STD of height jitter')
data_aug_arg.add_argument(
    '--data_aug_normal_jitter_std', type=float, default=0.01, help='STD of normal jitter')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)
data_aug_arg.add_argument('--data_aug_scale_min', type=float, default=0.8)
data_aug_arg.add_argument('--data_aug_scale_max', type=float, default=1.2)
data_aug_arg.add_argument(
    '--data_aug_hue_max', type=float, default=0.5, help='Hue translation range. [0, 1]')
data_aug_arg.add_argument(
    '--data_aug_saturation_max', type=float, default=0.20, help='Saturation translation range, [0, 1]')
data_aug_arg.add_argument('--temporal_rand_dilation',
                          type=str2bool, default=False)
data_aug_arg.add_argument('--temporal_rand_numseq',
                          type=str2bool, default=False)
data_aug_arg.add_argument('--sample_tail_instances', type=str2bool, default=False,
                     help='Adding new instances on the fly for more balanced sampling')
data_aug_arg.add_argument('--instance_augmentation', type=str, default=None, help='For applying targeted augmentation of less frequent categories. [raw, latent] for [pointcloud, latent space] augmentation')
data_aug_arg.add_argument('--instance_augmentation_color_aug_prob', type=float, default=0.1, help='The probability to apply targeted hue shift on the instances')
data_aug_arg.add_argument('--instance_augmentation_scale_aug_prob', type=float, default=0.1, help='The probability to apply targeted scaling on the instances')
data_aug_arg.add_argument('--data_aug_3d', type=str2bool, default=False)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument(
    '--test_config', default=None, type=str, help='path to the json config file for testing.')
test_arg.add_argument('--test_phase', type=str,
                      default='test', help='Dataset for test')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument(
    '--test_original_pointcloud',
    type=str2bool,
    default=False,
    help='Test on the original pointcloud space as given by the dataset using kd-tree.')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str,
                      default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--num_gpu', type=str2bool, default=1)
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument(
    '--debug', type=str2bool, default=True, help='print out detailed results for debugging')
data_aug_arg.add_argument(
    '--lenient_weight_loading',
    type=str2bool,
    default=False,
    help='Weights with the same size will be loaded')


def get_config():
    config = parser.parse_args()
    return config  # Training settings
