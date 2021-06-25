import warnings

import torch

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.depth import depth_warp as _depth_warp
from kornia.geometry.depth import DepthWarper as _DepthWarper

__all__ = ["depth_warp", "DepthWarper"]


class DepthWarper(_DepthWarper):
    __doc__ = _DepthWarper.__doc__

    def __init__(
        self,
        pinhole_dst: PinholeCamera,
        height: int,
        width: int,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
    ):
        super(DepthWarper, self).__init__(
            pinhole_dst=pinhole_dst,
            height=height,
            width=width,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        warnings.warn(
            "`DepthWarper` is deprecated and will be removed > 0.6.0. "
            "Please use `kornia.geometry.DepthWarper instead.`",
            DeprecationWarning,
            stacklevel=2,
        )


def depth_warp(
    pinhole_dst: PinholeCamera,
    pinhole_src: PinholeCamera,
    depth_src: torch.Tensor,
    patch_dst: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool = True,
):
    __doc__ = _depth_warp.__doc__
    warnings.warn(
        "`depth_warp` is deprecated and will be removed > 0.6.0. Please use `kornia.geometry.depth_warp instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _depth_warp(pinhole_dst, pinhole_src, depth_src, patch_dst, height, width, align_corners)
