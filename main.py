import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,6"
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
import matplotlib

matplotlib.use('agg')

import torch, torch.nn as nn
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler as dsamplers
import datasets
import criteria
import metrics
from utilities import misc
from utilities import logger
import outline
import configs
import sys
from termcolor import cprint


def main(args):
    full_training_start_time = time.time()
    outline.set_random_seed(args.seed)

    dataset = datasets.select(args)
    train_data_sampler = dsamplers.select(args, dataset['training'].image_dict, dataset['training'].image_list)
    dataloaders = datasets.dataloader(args, dataset, train_data_sampler)

    model = archs.select(args.arch, args)

    args.num_model_weights = int(misc.gimme_params(model))

    model = nn.DataParallel(model)
    model.to(args.device)

    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model, args.device)

    to_optim = outline.set_lr(model, args)

    outline.print_setting_summary(args)

    sub_loggers = ['Train', 'Test', 'Model Grad']
    if args.use_tv_split:
        sub_loggers.append('Val')

    LOG = logger.LOGGER(args, sub_loggers=sub_loggers, start_new=True, log_online=args.log_online)

    loss_emb, to_optim = criteria.select(args.loss_emb_name, args, to_optim)
    loss_div, to_optim = criteria.select(args.loss_div_name, args, to_optim)

    optimizer = torch.optim.Adam(to_optim)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.tau, gamma=args.gamma)

    metric_computer = metrics.MetricComputer(args.evaluation_metrics, args)
    for epoch in range(1, args.n_epochs + 1):
        args.epoch = epoch

        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

        outline.train(args, model, loss_emb, loss_div, optimizer, dataloaders, LOG)
        if epoch > args.eval_begin and epoch % args.eval_every == 0:
            outline.test(args, model, dataloaders, metric_computer, LOG)
        LOG.update(all=True)

        scheduler.step()

    # CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time() - full_training_start_time
    summary_text += 'Training Time: {} hours.\n'.format(np.round(full_training_time / 60 / 60, 2))

    summary_text += '---------------\n'
    for sub_logger in LOG.sub_loggers:
        results = LOG.graph_writer[sub_logger].ov_title
        summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), results)

    with open(args.save_path + '/training_summary.txt', 'w') as summary_file:
        summary_file.write(summary_text)


if __name__ == '__main__':
    conf = configs.defaults.default_conf
    conf.merge_from_list(sys.argv[1:])
    conf.merge_from_file(f'./configs/{conf.dataset}.yaml')
    conf.merge_from_list(sys.argv[1:])

    if not os.path.exists(conf.data_path):
        conf.data_path = conf.data_path.replace('/dive', '/data')

    conf.save_path = os.getcwd() + '/Training_Results/' + conf.dataset

    conf.dim = conf.full_dim // conf.k
    cprint(conf, 'red')
    main(conf)
