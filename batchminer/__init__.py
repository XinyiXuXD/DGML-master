from batchminer import distance, semihard, random, parametric, npair
from batchminer import random_distance
from batchminer import lifted

BATCHMINING_METHODS = {'random':random,
                       'semihard':semihard,
                       'distance':distance,
                       'npair':npair,
                       'parametric':parametric,
                       'lifted':lifted,
                       'random_distance': random_distance}


def select(opt):
    #####
    batchminername = opt.batchminer_name
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
