import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.linalg import transform_points
from kornia.testing import check_is_tensor
from kornia.utils import create_meshgrid, create_meshgrid3d
from kornia.utils.helpers import _torch_inverse_cast

__all__ = [
    "warp_perspective",
    "HomographyWarper",
    "homography_warp",
    "homography_warp3d",
    "warp_grid",
    "warp_grid3d",
    "normalize_homography",
    "denormalize_homography",
    "normalize_homography3d",
    "normal_transform_pixel",
    "normal_transform_pixel3d",
]


def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
    normalized_homography: bool = False,
    normalized_coordinates: bool = True,
) -> torch.Tensor:
    r"""Applies a perspective transformation to an image.

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Args:
        src (torch.Tensor): input image with shape :math:`(B, C, H, W)`.
        M (torch.Tensor): transformation matrix with shape :math:`(B, 3, 3)`.
        dsize (tuple): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool, optional): interpolation flag. Default: None.
        normalized_homography(bool): Warp image patchs or tensors by normalized 2D homographies.
                                      See :class:`~kornia.geometry.warp.HomographyWarper` for details.
        normalized_coordinates (bool): Whether the homography assumes [-1, 1] normalized
                                      coordinates or not.

    Returns:
        torch.Tensor: the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjuntion with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia.readthedocs.io/en/latest/
        tutorials/warp_perspective.html>`_.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("Input src type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}".format(M.shape))

    if not normalized_homography:
        # TODO: remove the statement below in kornia v0.6
        if align_corners is None:
            message: str = (
                "The align_corners default value has been changed. By default now is set True "
                "in order to match cv2.warpPerspective. In case you want to keep your previous "
                "behaviour set it to False. This warning will disappear in kornia > v0.6."
            )
            warnings.warn(message)
            # set default value for align corners
            align_corners = True

        B, C, H, W = src.size()
        h_out, w_out = dsize

        # we normalize the 3x3 transformation matrix and convert to 3x4
        dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3

        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

        # this piece of code substitutes F.affine_grid since it does not support 3x3
        grid = (
            create_meshgrid(h_out, w_out, normalized_coordinates=normalized_coordinates, device=src.device)
            .to(src.dtype)
            .repeat(B, 1, 1, 1)
        )
        grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    else:
        if align_corners is None:
            align_corners = False

        if not M.device == src.device:
            raise TypeError(
                "Patch and homography must be on the same device. \
                            Got patch.device: {} M.device: {}.".format(
                    src.device, M.device
                )
            )

        height, width = dsize
        grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)
        grid = warp_grid(grid, M)

    return F.grid_sample(src, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def warp_grid(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the grid to warp the coordinates grid by the homography/ies.

    Args:
        grid: Unwrapped grid of the shape :math:`(1, N, W, 2)`.
        src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.

    Returns:
        torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
    """
    batch_size: int = src_homo_dst.size(0)
    _, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxHxWx2
    return flow.view(batch_size, height, width, 2)  # NxHxWx2


def warp_grid3d(grid: torch.Tensor, src_homo_dst: torch.Tensor) -> torch.Tensor:
    r"""Compute the grid to warp the coordinates grid by the homography/ies.

    Args:
        grid: Unwrapped grid of the shape :math:`(1, D, H, W, 3)`.
        src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
          transform all points in the grid. Shape of the homography
          has to be :math:`(1, 4, 4)` or :math:`(N, 1, 4, 4)`.

    Returns:
        torch.Tensor: the transformed grid of shape :math:`(N, H, W, 3)`.
    """
    batch_size: int = src_homo_dst.size(0)
    _, depth, height, width, _ = grid.size()
    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1, -1)  # NxDxHxWx3
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 4, 4)  # Nx1x3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow: torch.Tensor = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxDxHxWx3
    return flow.view(batch_size, depth, height, width, 3)  # NxDxHxWx3


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
    r"""Warp image patchs or tensors by normalized 2D homographies.

    See :class:`~kornia.geometry.warp.HomographyWarper` for details.

    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, H, W)`.
        src_homo_dst (torch.Tensor): The homography or stack of homographies
                                     from destination to source of shape
                                     :math:`(N, 3, 3)`.
        dsize (Tuple[int, int]): The height and width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool): interpolation flag. Default: False. See
          https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
        normalized_coordinates (bool): Whether the homography assumes [-1, 1] normalized
                                       coordinates or not.

    Return:
        torch.Tensor: Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))
    """

    warnings.warn(
        "`homography_warp` is deprecated and will be removed > 0.6.0."
        "Please use `kornia.geometry.transform.warp_perspective instead.`",
        DeprecationWarning,
        stacklevel=2,
    )

    return warp_perspective(
        patch_src,
        src_homo_dst,
        dsize,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        normalized_homography=True,
        normalized_coordinates=normalized_coordinates,
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
    r"""Warp image patchs or tensors by normalized 3D homographies.

    Args:
        patch_src (torch.Tensor): The image or tensor to warp. Should be from
                                  source of shape :math:`(N, C, D, H, W)`.
        src_homo_dst (torch.Tensor): The homography or stack of homographies
                                     from destination to source of shape
                                     :math:`(N, 4, 4)`.
        dsize (Tuple[int, int, int]): The height and width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners(bool): interpolation flag. Default: False. See
            https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details.
        normalized_coordinates (bool): Whether the homography assumes [-1, 1] normalized
            coordinates or not.

    Return:
        torch.Tensor: Patch sampled at locations from source to destination.

    Example:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> homography = torch.eye(3).view(1, 3, 3)
        >>> output = homography_warp(input, homography, (32, 32))
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError(
            "Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(
                patch_src.device, src_homo_dst.device
            )
        )

    depth, height, width = dsize
    grid = create_meshgrid3d(
        depth, height, width, normalized_coordinates=normalized_coordinates, device=patch_src.device
    )
    warped_grid = warp_grid3d(grid, src_homo_dst)

    return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


# layer api
class HomographyWarper(nn.Module):
    r"""Warp tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height (int): The height of the destination tensor.
        width (int): The width of the destination tensor.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        normalized_coordinates (bool): wether to use a grid with
          normalized coordinates.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail
    """
    _warped_grid: Optional[torch.Tensor]

    def __init__(
        self,
        height: int,
        width: int,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        normalized_coordinates: bool = True,
        align_corners: bool = False,
    ) -> None:
        super(HomographyWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.normalized_coordinates: bool = normalized_coordinates
        self.align_corners: bool = align_corners
        # create base grid to compute the flow
        self.grid: torch.Tensor = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)

        # initialice the warped destination grid
        self._warped_grid = None

    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) -> None:
        r"""Compute and store internaly the transformations of the points.

        Useful when the same homography/homographies are reused.

        Args:
            src_homo_dst (torch.Tensor): Homography or homographies (stacked) to
              transform all points in the grid. Shape of the homography
              has to be :math:`(1, 3, 3)` or :math:`(N, 1, 3, 3)`.
              The homography assumes normalized coordinates [-1, 1] if
              normalized_coordinates is True.
        """
        self._warped_grid = warp_grid(self.grid, src_homo_dst)

    def forward(self, patch_src: torch.Tensor, src_homo_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Warp a tensor from source into reference frame.

        Args:
            patch_src (torch.Tensor): The tensor to warp.
            src_homo_dst (torch.Tensor, optional): The homography or stack of
              homographies from destination to source. The homography assumes
              normalized coordinates [-1, 1] if normalized_coordinates is True.
              Default: None.

        Return:
            torch.Tensor: Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = HomographyWarper(32, 32)
            >>> # without precomputing the warp
            >>> output = warper(input, homography)  # NxCxHxW
            >>> # precomputing the warp
            >>> warper.precompute_warp_grid(homography)
            >>> output = warper(input)  # NxCxHxW
        """
        _warped_grid = self._warped_grid
        if src_homo_dst is not None:
            warped_patch = warp_perspective(
                patch_src,
                src_homo_dst,
                (self.height, self.width),
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                normalized_homography=True,
                normalized_coordinates=self.normalized_coordinates,
            )
        elif _warped_grid is not None:
            if not _warped_grid.device == patch_src.device:
                raise TypeError(
                    "Patch and warped grid must be on the same device. \
                                 Got patch.device: {} warped_grid.device: {}. Whether \
                                 recall precompute_warp_grid() with the correct device \
                                 for the homograhy or change the patch device.".format(
                        patch_src.device, _warped_grid.device
                    )
                )
            warped_patch = F.grid_sample(
                patch_src,
                _warped_grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
        else:
            raise RuntimeError(
                "Unknown warping. If homographies are not provided \
                                they must be preset using the method: \
                                precompute_warp_grid()."
            )

        return warped_patch


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height (int): image height.
        width (int): image width.
        eps (float): epsilon to prevent divide-by-zero errors

    Returns:
        torch.Tensor: normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normal_transform_pixel3d(
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        depth (int): image depth.
        height (int): image height.
        width (int): image width.
        eps (float): epsilon to prevent divide-by-zero errors

    Returns:
        Tensor: normalized transform with shape :math:`(1, 4, 4)`.
    """
    tr_mat = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 4x4

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    depth_denom: float = eps if depth == 1 else depth - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / depth_denom

    return tr_mat.unsqueeze(0)  # 1x4x4


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).

    Returns:
        torch.Tensor: the normalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(dst_pix_trans_src_pix.shape)
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def denormalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> torch.Tensor:
    r"""De-normalize a given homography in pixels from [-1, 1] to actual height and width.

    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destination to be
          denormalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).

    Returns:
        torch.Tensor: the denormalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(dst_pix_trans_src_pix.shape)
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)
    dst_denorm_trans_dst_pix = _torch_inverse_cast(dst_norm_trans_dst_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_denorm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_norm_trans_src_pix)
    return dst_norm_trans_src_norm


def normalize_homography3d(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int, int], dsize_dst: Tuple[int, int, int]
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (torch.Tensor): homography/ies from source to destination to be
          normalized. :math:`(B, 4, 4)`
        dsize_src (tuple): size of the source image (depth, height, width).
        dsize_src (tuple): size of the destination image (depth, height, width).

    Returns:
        Tensor: the normalized homography.

    Shape:
        Output: :math:`(B, 4, 4)`
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (4, 4)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(dst_pix_trans_src_pix.shape)
        )

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel3d(src_d, src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel3d(dst_d, dst_h, dst_w).to(dst_pix_trans_src_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm
