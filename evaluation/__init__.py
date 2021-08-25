import faiss, matplotlib.pyplot as plt, os, numpy as np, torch
from PIL import Image
from . import auxiliaries as aux
from termcolor import cprint


def evaluate(LOG, metric_computer, dataloaders, model, opt, evaltypes, device,
             aux_store=None, make_recall_plot=False, store_checkpoints=True, log_key='Test'):
    """
    Parent-Function to compute evaluation metrics, print summary string and store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos, fea, labels = metric_computer.compute_standard(
        opt, model, dataloaders[0], evaltypes, device)

    if opt.eval_group and opt.group_name != '':
        evaluate_group(np.vstack(fea[evaltypes[0]]), np.reshape(labels, [-1, ]), opt)

    numeric_metrics = {}
    histogr_metrics = {}
    for main_key in computed_metrics.keys():
        for name, value in computed_metrics[main_key].items():
            if isinstance(value, np.ndarray):
                if main_key not in histogr_metrics: histogr_metrics[main_key] = {}
                histogr_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics: numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

    ###
    full_result_str = ''
    for evaltype in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
        for i, (metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i > 0 else '', metricname, metricval)
        full_result_str += '\n'

    print(full_result_str)

    ###
    for evaltype in evaltypes:
        for storage_metric in opt.storage_metrics:
            parent_metric = evaltype + '_{}'.format(storage_metric.split('@')[0])
            if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
                    numeric_metrics[evaltype][storage_metric] > np.max(
                LOG.progress_saver[log_key].groups[parent_metric][storage_metric]['content']):
                print('Saved weights for best {}: {}\n'.format(log_key, parent_metric))
                opt.best_epoch = opt.epoch
                set_checkpoint(model, opt, LOG.progress_saver,
                               LOG.prop.save_path + '/checkpoint_{}_{}_{}.pth.tar'.
                               format(log_key, evaltype, storage_metric), aux=aux_store)

    ###
    if opt.log_online:
        for evaltype in histogr_metrics.keys():
            for eval_metric, hist in histogr_metrics[evaltype].items():
                import wandb, numpy
                wandb.log({log_key + ': ' + evaltype + '_{}'.format(eval_metric): wandb.Histogram(
                    np_histogram=(list(hist), list(np.arange(len(hist) + 1))))}, step=opt.epoch)
                wandb.log({log_key + ': ' + evaltype + '_LOG-{}'.format(eval_metric): wandb.Histogram(
                    np_histogram=(list(np.log(hist) + 20), list(np.arange(len(hist) + 1))))}, step=opt.epoch)

    ###
    for evaltype in numeric_metrics.keys():
        for eval_metric in numeric_metrics[evaltype].keys():
            parent_metric = evaltype + '_{}'.format(eval_metric.split('@')[0])
            LOG.progress_saver[log_key].log(eval_metric, numeric_metrics[evaltype][eval_metric], group=parent_metric)

        ###
        if make_recall_plot:
            recover_closest_standard(extra_infos[evaltype]['features'],
                                     extra_infos[evaltype]['image_paths'],
                                     LOG.prop.save_path + '/sample_recoveries.png')


def evaluate_query_and_gallery_dataset(LOG, query_dataloader, gallery_dataloader, model, opt,
                                       give_return=False, log_key='Test'):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for In-Shop Clothes.
    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        query_dataloader:    PyTorch Dataloader, Query-testdata to be evaluated.
        gallery_dataloader:  PyTorch Dataloader, Gallery-testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    with torch.no_grad():
        # Compute Metrics.
        F1, NMI, recall_at_ks, query_feature_matrix_all, gallery_feature_matrix_all = \
            aux.eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader,
                                                       device=opt.device, k_vals=opt.k_vals, opt=opt)
        # Generate printable summary string.
        result_str = ', '.join('@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(opt.epoch, NMI, F1,
                                                                                            result_str)

        parent_metric = 'e_recall'
        eval_metric = 'e_recall@1'
        if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
                recall_at_ks[0] > np.max(LOG.progress_saver[log_key].groups[parent_metric]['e_recall@1']['content']):
            print('Saved weights for best {}: {}\n'.format(log_key, parent_metric))
            opt.best_epoch = opt.epoch
            set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path + '/checkpoint.pth.tar')
            # recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths,
            #                        gallery_image_paths, LOG.prop.save_path + '/sample_recoveries.png')

            LOG.progress_saver[log_key].log(eval_metric, recall_at_ks[0], group=parent_metric)

    cprint(result_str, 'red')
    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None


def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key: item for key, item in vars(opt).items() if key != 'experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    torch.save({'state_dict': model.state_dict(), 'opt': save_opt, 'progress': progress_saver, 'aux': aux}, savepath)


def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger


def recover_closest_standard(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest + 1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f, axes = plt.subplots(n_image_samples, n_closest + 1)
    for i, (ax, plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % (n_closest + 1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10, 20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


def recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths,
                           save_path, n_image_samples=10, n_closest=3):
    """
    Provide sample recoveries.

    Args:
        query_feature_matrix_all:   np.ndarray [n_query_samples x embed_dim], full data embedding of query samples.
        gallery_feature_matrix_all: np.ndarray [n_gallery_samples x embed_dim], full data embedding of gallery samples.
        query_image_paths:          list [n_samples], list of datapaths corresponding to <query_feature_matrix_all>
        gallery_image_paths:        list [n_samples], list of datapaths corresponding to <gallery_feature_matrix_all>
        save_path:          str, where to store sample image.
        n_image_samples:    Number of sample recoveries.
        n_closest:          Number of closest recoveries to show.
    Returns:
        Nothing!
    """
    query_image_paths, gallery_image_paths = np.array(query_image_paths), np.array(gallery_image_paths)
    sample_idxs = np.random.choice(np.arange(len(query_feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(gallery_feature_matrix_all.shape[-1])
    faiss_search_index.add(gallery_feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(query_feature_matrix_all, n_closest)

    image_paths = gallery_image_paths[closest_feature_idxs]
    image_paths = np.concatenate([query_image_paths.reshape(-1, 1), image_paths], axis=-1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f, axes = plt.subplots(n_image_samples, n_closest + 1)
    for i, (ax, plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % (n_closest + 1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10, 20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


def evaluate2(fvecs, labels, tag='hid'):
    """
    Evaluation of a single embedding.

    Args:
        fvecs: numpy array of feature vectors
        labels: labels
    Returns:
        The recall @1
    """
    fvecs = fvecs.astype(np.float32)
    D = fvecs.dot(fvecs.T)
    # Remove the diagonal for evalution! This is the same sample as the query.
    I = np.eye(D.shape[0]) * abs(D).max() * 10.0
    D -= I
    predictions = D.argmax(axis=1)
    pred_labels = labels[predictions]

    recall = (pred_labels == labels).sum() / float(len(labels))
    print('R@1 ({}): {}'.format(tag, round(recall, 4)))
    return recall


def evaluate_group(fea, labels, opt):
    """
    Parent-Function to compute evaluation metrics, print summary string and store checkpoint files/plot sample recall plots.
    """
    fvecs, lbaels = np.vstack(fea), np.vstack(labels)
    labels = np.reshape(labels, [-1, ])
    embedding_sizes = [opt.full_dim // opt.k] * opt.k
    embedding_scales = [float(e) / sum(embedding_sizes) for e in embedding_sizes]
    start_idx = 0
    for e, s in zip(embedding_sizes, embedding_scales):
        stop_idx = start_idx + e
        fvecs[:, start_idx:stop_idx] *= s
        _ = evaluate2(np.array(fvecs[:, start_idx:stop_idx].copy()), labels, tag='Embedding-{}'.format(e))
        start_idx = stop_idx
