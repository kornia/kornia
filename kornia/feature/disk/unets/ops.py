import torch, functools
import torch.nn as nn
import torch.nn.functional as F

from .utils import size_is_pow2


class AttentionGate(nn.Module):
    def __init__(self, n_features):
        super(AttentionGate, self).__init__()
        self.n_features = n_features

        self.seq = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        g = self.seq(inp)

        return g * inp


class TrivialUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialUpsample, self).__init__()

    def forward(self, x):
        r = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False
        )
        return r


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return F.avg_pool2d(x, 2)


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class UGroupNorm(nn.GroupNorm):
    def __init__(self, in_channels, group_size):
        group_size = max(1, min(group_size, in_channels))

        if in_channels % group_size != 0:
            for upper in range(group_size+1, in_channels + 1):
                if in_channels % upper == 0:
                    break

            for lower in range(group_size-1, 0, -1):
                if in_channels % lower == 0:
                    break

            if upper - group_size < group_size - lower:
                group_size = upper
            else:
                group_size = lower

        assert in_channels % group_size == 0
        num_groups = in_channels // group_size

        super(UGroupNorm, self).__init__(num_groups, in_channels)


def u_group_norm(group_size):
    return functools.partial(UGroupNorm, group_size=group_size)
