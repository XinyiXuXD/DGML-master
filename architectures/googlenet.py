"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import torchvision.models as mod
from architectures.group_layers import LinearEmb, MulLinear, Channel, Gate, LearnQ_fen, LearnQ_he
emb_layer = {'LearnQ_fen': LearnQ_fen,
             'LearnQ_he': LearnQ_he,
             'MulLinear': MulLinear,
             'LinearEmb': LinearEmb,
             'Gate': Gate,
             'Channel': Channel}


class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        gmodel = mod.googlenet(pretrained=True)
        backbone = nn.Sequential(*list(gmodel.children())[:-3])

        self.arch = opt.arch
        self.group_name = opt.group_name
        self.in_channel = 1024
        self.full_dim = opt.full_dim
        self.reduction = opt.reduction
        self.k, self.dim_per = opt.k, self.full_dim // opt.k

        assert self.group_name in emb_layer.keys()

        emb_layer_args = {'in_channel': self.in_channel, 'out_channel': self.full_dim,
                          'k': self.k, 'dim_per': self.dim_per, 'reduction': self.reduction}

        self.model = nn.Sequential(backbone, emb_layer[self.group_name](**emb_layer_args))

    def forward(self, x):
        x = self.model(x)
        aux_output = None
        if isinstance(x, tuple):
            x, aux_output = x
        if 'normalize' in self.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x, aux_output
