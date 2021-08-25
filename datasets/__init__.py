from . import cub200
from . import cars196
from . import stanford_online_products
from . import in_shop
import torch


def select(args):
    if 'cub200' in args.dataset:
        return cub200.Give(args, args.data_path)

    if 'cars196' in args.dataset:
        return cars196.Give(args, args.data_path)

    if 'online_products' in args.dataset:
        return stanford_online_products.Give(args, args.data_path)
    if 'in_shop' in args.dataset:
        return in_shop.Give(args, args.data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & online_products!'.format(args.dataset))


def dataloader(args, dataset, train_data_sampler):

    dataloaders = {}

    dataloaders['training'] = torch.utils.data.DataLoader(dataset['training'], num_workers=args.kernels,
                                                          batch_sampler=train_data_sampler)
    dataloaders['evaluation'] = torch.utils.data.DataLoader(dataset['evaluation'], num_workers=args.kernels,
                                                            batch_size=args.BS, shuffle=False)
    if args.use_tv_split:
        dataloaders['validation'] = torch.utils.data.DataLoader(dataset['validation'], num_workers=args.kernels,
                                                                batch_size=args.BS, shuffle=False)

    if args.dataset in ['cub200', 'cars196', 'online_products']:
        dataloaders['testing'] = torch.utils.data.DataLoader(dataset['testing'], num_workers=args.kernels,
                                                             batch_size=args.BS,
                                                             shuffle=False)
    elif 'in_shop' in args.dataset:
        dataloaders['testing_query'] = torch.utils.data.DataLoader(dataset['testing_query'],
                                                                   num_workers=args.kernels,
                                                                   batch_size=args.BS,
                                                                   shuffle=False)

        dataloaders['testing_gallery'] = torch.utils.data.DataLoader(dataset['testing_gallery'],
                                                                     num_workers=args.kernels,
                                                                     batch_size=args.BS,
                                                                     shuffle=False)

    args.n_classes = len(dataloaders['training'].dataset.avail_classes)
    return dataloaders
