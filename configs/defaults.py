from yacs.config import CfgNode as CN
import os

default_conf = CN()

# basic
default_conf.seed = 5
default_conf.dataset = 'cub200'
# default_conf.arch = 'resnet50_frozen_normalize'
default_conf.arch = 'googlenet'
default_conf.full_dim = 512
default_conf.n_epochs = 150
default_conf.kernels = 0
default_conf.BS = 112
default_conf.pretrained = True
default_conf.data_path = ''

default_conf.device = 'cuda'

# group
default_conf.k = 3
default_conf.reduction = 8
default_conf.group_name = 'LearnQ_fen'

# train 
default_conf.decay = 0.003
default_conf.key_val_decay = 0.003
default_conf.que_decay = 0.001

default_conf.ft_params_lr = 1e-4
default_conf.new_params_lr = 1e-4

default_conf.optim = 'adam'
default_conf.scheduler = 'step'
default_conf.tau = [55, 1000]
default_conf.gamma = 0.3


# eval
default_conf.eval_begin = -1
default_conf.eval_every = 1
default_conf.eval_group = False
default_conf.eval_train = False
default_conf.att_vis = False

# loss
default_conf.loss_emb_name = 'Margin'
default_conf.loss_div_name = 'DivBD'
default_conf.lam_div = 0.01
default_conf.batchminer_name = 'distance'
default_conf.cn = 25
default_conf.loss_margin_beta_lr = 1e-4

# log
default_conf.save_path = ''
default_conf.savename = 'test'

# dataset-related
default_conf.use_tv_split = False
default_conf.tv_split_by_samples = False
default_conf.tv_split_perc = 0.8
default_conf.augmentation = 'base'


# evaluation
default_conf.evaluate_on_gpu = False
default_conf.evaluation_metrics = ['e_recall@1']
default_conf.storage_metrics = ['e_recall@1']
default_conf.evaltypes = ['discriminative']
default_conf.k_vals = [1]

# online logging
default_conf.log_online = False
default_conf.online_backend = 'wandb'

# loss
default_conf.loss_contrastive_pos_margin = 0.0
default_conf.loss_contrastive_neg_margin = 1.0

default_conf.loss_margin_margin = 0.2
default_conf.loss_margin_beta = 1.2
default_conf.loss_margin_nu = 0
default_conf.loss_margin_beta_constant = False

# batch miner
default_conf.miner_distance_lower_cutoff = 0.5
default_conf.miner_distance_upper_cutoff = 1.4
default_conf.miner_rho_distance_lower_cutoff = 0.5
default_conf.miner_rho_distance_upper_cutoff = 1.4
default_conf.miner_rho_distance_cp = 0.2

# batch creation
default_conf.data_sampler = 'class_random'
default_conf.samples_per_class = 2











