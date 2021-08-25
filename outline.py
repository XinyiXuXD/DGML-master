import argparse
import parameters as par
import torch
import numpy as np
import random
from tqdm import tqdm
import evaluation    as eval
import time


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_lr(model, args):
    ft_params, key_val_params, que_params = [], [], []
    for name, params in model.named_parameters():
        if 'key' in name or 'val' in name:
            key_val_params.append(params)
        elif 'que' in name:
            que_params.append(params)
        else:
            ft_params.append(params)
        print(name, params.requires_grad)

    to_optim = [{'params': ft_params, 'lr': args.ft_params_lr, 'weight_decay': args.decay},
                {'params': key_val_params, 'lr': args.new_params_lr, 'weight_decay': args.key_val_decay},
                {'params': que_params, 'lr': args.new_params_lr, 'weight_decay': args.que_decay}]

    return to_optim


def print_setting_summary(args):
    data_text = 'Dataset:\t {}'.format(args.dataset.upper())
    setup_text = 'Loss_emb:\t {}'.format(args.loss_emb_name.upper())
    miner_text = 'Batchminer:\t {}'.format(args.batchminer_name)
    arch_text = 'Backbone:\t {} (#weights: {})'.format(args.arch.upper(), args.num_model_weights)
    summary = data_text + '\n' + setup_text + '\n' + miner_text + '\n' + arch_text

    group_text = 'Group:\t {} (#groups: {})'.format(args.group_name.upper(), args.k)
    div_text = 'Loss_div:\t {}, {}'.format(args.loss_div_name.upper(), args.lam_div)
    summary += '\n' + group_text + '\n' + div_text
    print(summary)

    print('\n-----\n')


def train(args, model, loss_emb, loss_div, optimizer, dataloaders, LOG):
    start = time.time()
    model.train()
    loss_collect, loss_emb_collect, loss_div_collect = [], [], []
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(args.epoch))
    for i, out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        input = input.to(args.device)
        model_args = {'x': input.to(args.device)}
        # Needed for MixManifold settings.
        if 'mix' in args.arch: model_args['labels'] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple): embeds, embs_aux = embeds

        loss_args = {'batch': embeds, 'labels': class_labels}
        l_emb, l_div = loss_emb(**loss_args), loss_div(**loss_args)
        l_all = l_emb + args.lam_div * l_div

        loss_collect.append(l_all.item())
        loss_emb_collect.append(l_emb.item())
        loss_div_collect.append(l_div.item())

        optimizer.zero_grad()
        l_all.backward()

        # Compute Model Gradients and log them!
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

        optimizer.step()

    print('Loss {:.3f}, Loss_emb {:.3f}, Loss_div {:.3f}'.
          format(np.mean(loss_collect), np.mean(loss_emb_collect), np.mean(loss_div_collect)))

    LOG.progress_saver['Train'].log('epochs', args.epoch)
    LOG.progress_saver['Train'].log('loss', np.mean(loss_collect))
    LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))


def test(args, model, dataloaders, metric_computer, LOG):
    model.eval()
    print('\nComputing Testing Metrics...')
    if args.dataset in ['cub200', 'cars196', 'online_products']:
        eval.evaluate(LOG, metric_computer, [dataloaders['testing']], model, args, args.evaltypes,
                      args.device, log_key='Test')
    elif 'in_shop' in args.dataset:
        eval.evaluate_query_and_gallery_dataset(LOG, dataloaders['testing_query'], dataloaders['testing_gallery'],
                                                model, args)

    if args.use_tv_split:
        print('\nComputing Validation Metrics...')
        eval.evaluate(LOG, metric_computer, [dataloaders['validation']], model, args, args.evaltypes,
                      args.device, log_key='Val')

    if args.eval_train:
        print('\nComputing Training Metrics...')
        eval.evaluate(LOG, metric_computer, [dataloaders['evaluation']], model, args, args.evaltypes,
                      args.device,
                      log_key='Train')


