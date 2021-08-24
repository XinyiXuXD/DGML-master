import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.googlenet import Inception


class LinearEmb(torch.nn.Module):
    def __init__(self, **kwargs):
        super(LinearEmb, self).__init__()
        out_channel = kwargs['out_channel']
        in_channel = kwargs['in_channel']

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear_emb = torch.nn.Linear(in_channel, out_channel, bias=False)

    def forward(self, x):
        x = self.pool(x).squeeze()
        x = self.linear_emb(x)
        return x


class MulLinear(nn.Module):
    def __init__(self, **kwargs):
        super(MulLinear, self).__init__()
        self.k, self.dim_per = kwargs['k'], kwargs['dim_per']
        self.in_channel = kwargs['in_channel']

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.mul_linear = nn.Linear(self.in_channel, self.dim_per * self.k, bias=False)

    def forward(self, x):
        x = self.pool(x).squeeze()
        embs = self.mul_linear(x).view(-1, self.k, self.dim_per)
        embs_stop = self.mul_linear(x.detach()).view(-1, self.k, self.dim_per)
        return embs, embs_stop


class Gate(nn.Module):
    def __init__(self, **kwargs):
        super(Gate, self).__init__()
        self.k, self.dim_per = kwargs['k'], kwargs['dim_per']
        self.in_channel = kwargs['in_channel']
        self.reduction = kwargs['reduction']

        self.linear = nn.Linear(self.in_channel, self.dim_per, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_conv = nn.Conv2d(self.in_channel, self.in_channel // self.reduction, kernel_size=1)

        self.indep_cov = nn.ModuleList()
        for i in range(self.k):
            self.indep_cov.append(nn.Conv2d(self.in_channel // self.reduction, self.in_channel, kernel_size=1))

    def forward(self, x):
        embs, masks = [], []
        x_inter = self.share_conv(x)
        for i in range(self.k):
            mask = self.indep_cov[i](x_inter)
            y = self.avg_pool(x*mask)
            y = self.linear(y.squeeze())
            embs.append(y)
            masks.append(mask.unsqueeze(1))
        embs = torch.cat(embs, 1).view(-1, self.k, self.dim_per)
        masks = torch.cat(masks, dim=1)
        return embs, masks


class Channel(nn.Module):
    def __init__(self, **kwargs):
        super(Channel, self).__init__()
        self.k, self.dim_per = kwargs['k'], kwargs['dim_per']
        self.in_channel = kwargs['in_channel']

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.SE_ops = nn.ModuleList()
        for i in range(self.k):
            self.SE_ops.append(SqueezeExcitation(self.in_channel))

        self.linear = nn.Linear(self.in_channel, self.dim_per, bias=True)

    def forward(self, x):
        embs, att_wei = [], []
        for i in range(self.k):
            y, w_c = self.SE_ops[i](x)
            y = self.linear(self.avg_pool(y).squeeze())
            embs.append(y)
            att_wei.append(w_c.unsqueeze(1))
        embs = torch.cat(embs, 1).view(-1, self.k, self.dim_per)
        att_wei = torch.cat(att_wei, dim=1)
        return embs, att_wei


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=5):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).squeeze()
        w_c = self.fc(y)
        masked_fp = x * w_c.unsqueeze(2).unsqueeze(3)
        return masked_fp, w_c


class LearnQ_he(nn.Module):
    def __init__(self, **kwargs):
        super(LearnQ_he, self).__init__()
        print('************he*********')

        self.k, self.dim_per = kwargs['k'], kwargs['dim_per']
        self.in_channel = kwargs['in_channel']
        self.reduction = kwargs['reduction']

        c_k = self.in_channel // self.reduction
        self.key_conv = nn.Conv2d(self.in_channel, c_k, kernel_size=1, stride=1, padding=0, bias=False)
        self.val_conv = nn.Conv2d(self.in_channel, self.dim_per, kernel_size=1, stride=1, padding=0, bias=False)
        self.que_embedding = nn.ModuleList()
        for i in range(self.k):
            self.que_embedding.append(nn.Conv2d(c_k, 1, kernel_size=1, stride=1, padding=0, bias=False))
        # self.ph = nn.Sequential(nn.Linear(in_features=self.dim_per, out_features=self.dim_per),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=self.dim_per, out_features=self.dim_per))
        self.ph = nn.Linear(in_features=self.dim_per, out_features=self.dim_per, bias=False)

    def forward(self, x):
        n, _, h, w = list(x.size())
        key = self.key_conv(x)  # n*c_k*h*w
        value = self.val_conv(x)  # n*dim_per*h*w
        out = []
        out_att_wei = []
        for i in range(self.k):
            att_weight = self.que_embedding[i](key)  # n*1*h*w
            att_weight_s = F.softmax(att_weight.reshape([n, -1, h * w]), dim=-1)  # n*1*(h*w)
            att_out = torch.matmul(att_weight_s,
                                   value.reshape(n, -1, h * w).permute([0, 2, 1]))  # n*1*c_v
            att_out = self.ph(att_out)
            out.append(att_out)
            out_att_wei.append(att_weight_s.reshape(n, 1, h, w))
        out = torch.cat(out, dim=1)
        out_att_wei = torch.cat(out_att_wei, dim=1)
        return out, out_att_wei


class LearnQ_fen(nn.Module):
    def __init__(self, **kwargs):
        super(LearnQ_fen, self).__init__()
        print('************fen*********')

        self.k, self.dim_per = kwargs['k'], kwargs['dim_per']
        self.in_channel = kwargs['in_channel']
        self.reduction = kwargs['reduction']

        c_k = self.in_channel // self.reduction
        self.key_conv = nn.Conv2d(self.in_channel, c_k, kernel_size=1, stride=1, padding=0, bias=False)
        self.val_conv = nn.Conv2d(self.in_channel, self.dim_per, kernel_size=1, stride=1, padding=0, bias=False)
        self.que_embedding = nn.ModuleList()
        for i in range(self.k):
            self.que_embedding.append(nn.Conv2d(c_k, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        n, _, h, w = list(x.size())
        key = self.key_conv(x)  # n*c_k*h*w
        value = self.val_conv(x)  # n*dim_per*h*w
        out = []
        out_att_wei = []
        for i in range(self.k):
            att_weight = self.que_embedding[i](key)  # n*1*h*w
            att_weight_s = F.softmax(att_weight.reshape([n, -1, h*w]), dim=-1)  # n*1*(h*w)
            att_out = torch.matmul(att_weight_s,
                                   value.reshape(n, -1, h*w).permute([0, 2, 1]))  # n*1*c_v
            out.append(att_out)
            out_att_wei.append(att_weight_s.reshape(n, 1, h, w))
        out = torch.cat(out, dim=1)
        out_att_wei = torch.cat(out_att_wei, dim=1)
        return out, out_att_wei
