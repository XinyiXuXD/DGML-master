import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list, **kwargs):
        self.pars = opt

        #####
        self.image_dict         = image_dict
        self.image_list         = image_list

        #####
        self.classes        = list(self.image_dict.keys())

        ####
        self.batch_size         = opt.BS
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.BS
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'class_random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            draws = self.batch_size//self.samples_per_class
            class_keys = np.random.choice(self.classes, draws, replace=False)
            for ck in class_keys:
                this_class_image = self.image_dict[ck]
                idx = np.random.choice(np.arange(len(this_class_image)), self.samples_per_class)
                class_ix_list = [this_class_image[i][-1] for i in idx]
                subset.extend(class_ix_list)
            yield subset

    def __len__(self):
        return self.sampler_length
