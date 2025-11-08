import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    if isinstance(kernel_size, int):
        k = kernel_size
        padding = (k // 2, k // 2, k // 2)
    else:
        padding = tuple(k // 2 for k in kernel_size)
    return nn.Conv3d(in_channels, out_channels, kernel_size,
                     padding=padding, bias=bias)

class CALayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB3D(nn.Module):
    def __init__(self, conv3d, n_feat, kernel_size, reduction,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        body = []
        for i in range(2):
            body.append(conv3d(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                body.append(nn.BatchNorm3d(n_feat))
            if i == 0:
                body.append(act)
        body.append(CALayer3D(n_feat, reduction))
        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        
        res = res + x
        return res

class ResidualGroup3D(nn.Module):
    def __init__(self, conv3d, n_feat, kernel_size, reduction, act, n_resblocks):
        super().__init__()
        blocks = [
            RCAB3D(conv3d, n_feat, kernel_size, reduction,
                   bias=True, bn=False, act=act, res_scale=1)
            for _ in range(n_resblocks)
        ]
        blocks.append(conv3d(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

class RCAN4D(nn.Module):

    def __init__(self, n_resgroups=5, n_resblocks=10, n_feats=64, reduction=16,
                 in_channels=10, len_Z=128, out_channels=64, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels 
        self.n_feats = n_feats
        self.len_Z = len_Z

        act = nn.ReLU(True)
        conv3d = default_conv3d

        self.head = nn.Sequential(conv3d(in_channels, n_feats, kernel_size))

        body = [
            ResidualGroup3D(conv3d, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        body.append(conv3d(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        self.tail1 = nn.Sequential(
            conv3d(n_feats, len_Z, kernel_size=1)  
        )

        self.tail2 = nn.Sequential(
            nn.Linear(len_Z, len_Z),
            nn.ReLU(inplace=True),
            nn.Linear(len_Z, out_channels)
        )

    def forward(self, x):

        assert x.dim() == 5, f'Expected 5D input (B, X, Y, Z, N), got {x.shape}'
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x_head = self.head(x)
        res = self.body(x_head)
        res = res + x_head
        out0 = self.tail1(res)
        out0 = out0.permute(0, 2, 3, 4, 1).contiguous()


        out0 = math.pi * torch.tanh(out0)

        B, X, Y, Z, C = out0.shape
        out0_flat = out0.view(-1, C)
        out_flat = self.tail2(out0_flat)
        out = out_flat.view(B, X, Y, Z, -1)

        
        
        return out, out0

def make_rcan4d(n_resgroups=5, n_resblocks=10, n_feats=64, reduction=16,
                len_Z=128,in_channels=10, out_channels=64):
    return RCAN4D(n_resgroups=n_resgroups,
                  n_resblocks=n_resblocks,
                  n_feats=n_feats,
                  reduction=reduction,
                  len_Z=len_Z,
                  in_channels=in_channels,
                  out_channels=out_channels)
