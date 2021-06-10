import warnings
from typing import Optional, Tuple

import torch

import kornia.geometry.transform.homography_warper as HMW

__all__ = [
    "HomographyWarper",
    "homography_warp",
    "homography_warp3d",
    "warp_grid",
    "warp_grid3d",
    "normalize_homography",
    "normalize_homography3d",
    "normal_transform_pixel",
    "normal_transform_pixel3d",
]


def warp_grid(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    __doc__ = HMW.warp_grid.__doc__
    warnings.warn(
        "`warp_grid` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.transform.warp_grid instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.warp_grid(grid, src_homo_dst)


def warp_grid3d(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    __doc__ = HMW.warp_grid3d.__doc__
    warnings.warn(
        "`warp_grid3d` is deprecated. Please use `kornia.geometry.transform.warp_grid3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.warp_grid3d(grid, src_homo_dst)


# functional api
def homography_warp(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    __doc__ = HMW.homography_warp.__doc__
    warnings.warn(
        "`homography_warp` is deprecated and will be removed > 0.6.0."
        "Please use `kornia.geometry.transform.homography_warp instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.homography_warp(
        patch_src, src_homo_dst, dsize, mode, padding_mode, align_corners, normalized_coordinates
    )


def homography_warp3d(
    patch_src: torch.Tensor,
    src_homo_dst: torch.Tensor,
    dsize: Tuple[int, int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    __doc__ = HMW.homography_warp3d.__doc__
    warnings.warn(
        "`homography_warp3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.transform.homography_warp3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.homography_warp3d(
        patch_src, src_homo_dst, dsize, mode, padding_mode, align_corners, normalized_coordinates
    )


class HomographyWarper(HMW.HomographyWarper):
    __doc__ = HMW.HomographyWarper.__doc__

    def __init__(
        self,
        height: int,
        width: int,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        normalized_coordinates: bool = True,
        align_corners: bool = False,
    ) -> None:
        super(HomographyWarper, self).__init__(height, width, mode, padding_mode, normalized_coordinates, align_corners)
        warnings.warn(
            "`HomographyWarper` is deprecated and will be removed > 0.6.0. "
            "Please use `kornia.geometry.transform.HomographyWarper instead.`",
            DeprecationWarning,
            stacklevel=2,
        )


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    __doc__ = HMW.normal_transform_pixel.__doc__
    warnings.warn(
        "`normal_transform_pixel` is deprecated and will be removed > 0.6.0."
        "Please use `kornia.geometry.transform.normal_transform_pixel instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.normal_transform_pixel(height, width, eps, device, dtype)


def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    __doc__ = HMW.normal_transform_pixel3d.__doc__
    warnings.warn(
        "`normal_transform_pixel3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.transform.normal_transform_pixel3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.normal_transform_pixel3d(depth, height, width, eps, device=device, dtype=dtype)


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    __doc__ = HMW.normalize_homography.__doc__
    warnings.warn(
        "`normalize_homography` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.transform.normalize_homography instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst)


def normalize_homography3d(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int, int], dsize_dst: Tuple[int, int, int]
) -> torch.Tensor:
    __doc__ = HMW.normalize_homography3d.__doc__
    warnings.warn(
        "`normalize_homography3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.transform.normalize_homography3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return HMW.normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst)
