from . import random_sampler
from . import class_random_sampler


def select(args, image_dict, image_list=None, **kwargs):
    if 'random' in args.data_sampler:
        if 'class' in args.data_sampler:
            sampler_lib = class_random_sampler
        elif 'full' in args.data_sampler:
            sampler_lib = random_sampler
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(args.data_sampler))
    sampler = sampler_lib.Sampler(args, image_dict=image_dict, image_list=image_list)
    return sampler
