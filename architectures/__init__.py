from . import resnet50
from. import googlenet
from . import bninception


def select(arch, opt):
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)

